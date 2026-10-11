"""Habits projection (H1): signals, the daily picture and streaks.

Camera and connector events become `habits.signal.recorded`; each local day is
closed into one `habits.day.closed` per habit. A day closes on event time: when
an event for a later local day arrives, or when the ticker calls `flush(now)`.

State lives in memory and rebuilds from the log: subscribe every `on_*` handler
with `replay=True`. Each source event is counted once (deduped on event id),
and events for a day that is already closed are ignored.

A habit that had nothing to judge on a day (no deadline near, nothing thrown
away, or no camera had reported yet) closes as met with its streak unchanged.
Only camera events say the room was watched: a connector sync says nothing
about the desk, so it never starts the clock for sleep, meals, drinks, focus
or the leaving check.

Sleep and focus are durations, so they count only time the desk camera was
reporting (see coverage.py): not while it was paused, and not across a gap in
its observations. When too little of the night or the day was watched they
have nothing to judge either, and the row says why (`coverage`, `reason`).

Sleep from a watch (K9): a `health` item from the connectors is a night the watch measured. Its hours asleep
are the sleep value for the day the night ended on, in place of the empty-desk estimate, and that day is
judged whatever the desk camera saw of it (`source` on the row: "watch", otherwise "camera"). It counts only
while that day is still open: a closed day is in the log and never changes. A night the watch was not worn
has no item, and is judged from the desk camera exactly as before.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from contexts.connectors.domain.health import night_day, parse_summary  # pure: reads a health item's summary line
from contexts.habits.domain.config import Habit, HabitsConfig, word_pattern
from contexts.habits.domain.coverage import Coverage
from contracts.connectors import ItemSynced
from contracts.events import Event
from contracts.habits import DayClosed, NudgeRaised, SignalRecorded
from contracts.impact import ItemDisposed
from contracts.perception import SceneDescribed, ZoneEvent

ONE_DAY = timedelta(days=1)
MIN_AWAY_H = 1.0  # shorter gaps in the sleep window are not worth a signal


@dataclass
class Day:
    """What one local day has added up to so far."""

    meals: int = 0
    drinks: int = 0
    desk_min: float = 0.0
    away_h: float = 0.0  # longest empty-desk stretch in the night that ended this morning
    forgot: int = 0
    disposed: int = 0
    correct: int = 0
    due_flagged: set[str] = field(default_factory=set)  # assignments already signalled as due soon


@dataclass
class Item:
    """An assignment or calendar event pulled by a connector."""

    title: str
    when: datetime
    url: str | None
    known_at: datetime
    desk_min: float = 0.0  # assignment: desk time since it was first synced
    started_at: datetime | None = None  # assignment: when that passed `min_minutes`
    flagged: bool = False  # calendar event: a forgotten item was already counted against it


@dataclass
class Closed:
    day: date
    value: float
    met: bool
    streak: int
    applies: bool
    coverage: float | None = None  # sleep, focus: share of the window the desk camera reported for, when measured
    reason: str | None = None  # why it was not judged, when that share was too small
    source: str | None = None  # sleep: "watch" when a watch measured the night, "camera" for the empty-desk estimate


def _hours(delta: timedelta) -> float:
    return delta.total_seconds() / 3600


def _h(delta: timedelta) -> str:
    return f"{round(_hours(delta), 1):g} h"


class HabitTracker:
    def __init__(self, cfg: HabitsConfig, cover: Coverage | None = None) -> None:
        self.cfg = cfg
        self.sched = cfg.schedule
        self.cover = cover or Coverage(cfg.desk.camera, timedelta(minutes=cfg.coverage.gap_minutes),
                                       timedelta(minutes=cfg.coverage.steady_minutes))
        self.today: date | None = None  # the earliest day still open
        self.now: datetime | None = None  # latest event time seen
        self.watching_since: datetime | None = None  # first camera event: nothing is known about the room before this
        self.days: dict[date, Day] = {}
        self.history: dict[str, list[Closed]] = {h.id: [] for h in cfg.habits}
        self.assignments: dict[str, Item] = {}
        self.calendar: dict[str, Item] = {}
        self.measured_sleep: dict[date, float] = {}  # open day -> hours asleep in the night that ended on it, from a watch
        self.active_until: datetime | None = None  # the desk is accounted for up to here
        self.desk_frame: str | None = None
        self.on_desk: dict[str, str | None] = {}  # watch item -> frame it was last seen in
        self.forgotten: dict | None = None  # the latest "left without it", for the nudge text
        self.latest_nudge: dict[str, NudgeRaised] = {}
        self._since: dict[str, datetime] = {}  # person at the desk -> start of time not yet counted
        self._session_min = 0.0
        self._sighted: dict[str, tuple[datetime, int]] = {}  # habit -> last sighting, how many
        self._seen: dict[date, set[str]] = {}
        self._words = {h.id: h.pattern() for h in cfg.habits if h.keywords}
        forgot = cfg.by_signal("forgot")
        self._watch = {w: word_pattern([w]) for w in (forgot.watch_items if forgot else [])}
        # Owner recognition (P9): while this set of habits is not the one being told about people (`pause`),
        # its camera habits are not judged. Never set unless contexts.habits.domain.members sets it.
        self.unjudged_since: datetime | None = None
        self.unjudged: set[date] = set()  # open days that were paused for some part

    # --- handlers (replay=True) ---

    def on_zone_event(self, e: ZoneEvent) -> list[Event]:
        """A person in the desk zone is focus time; the gaps between are the away window."""
        desk, door = self.cfg.desk.zone, self.cfg.door_zone
        self._watching(e)
        if e.label != "person" or e.zone_id not in (desk, door) or not self._admit(e):
            return []
        out = self._desk(e) if e.zone_id == desk else []  # before the roll: minutes before midnight stay on their day
        out += self._roll(self.sched.local_day(e.ts), e.ts)
        if e.zone_id == door and e.kind == "exit":
            out += self._leaving(e)
        return out

    def on_scene(self, e: SceneDescribed) -> list[Event]:
        """Desk captions and object lists: meals, energy drinks, and where the watch items are."""
        desk = self.cfg.desk
        self._watching(e)
        at_desk = e.zone_id == desk.zone or (e.zone_id is None and e.camera_id == desk.camera)
        if not at_desk or not self._admit(e):
            return []
        out = self._roll(self.sched.local_day(e.ts), e.ts)
        labels = [o.label for o in e.objects]
        self._spot_watch_items(e, labels)
        day = self._day(self.sched.local_day(e.ts))
        for kind in ("meal", "drink"):
            h = self.cfg.by_signal(kind)
            n = self._new_sightings(h, e, labels) if h else 0
            if not n:
                continue
            if kind == "meal":
                day.meals += n
            else:
                day.drinks += n
            out.append(self._signal(kind, n, e))
        return out

    def on_observation(self, e: Event) -> list[Event]:
        """About one a second per camera, whatever is in view: the camera was reporting. Nothing is counted
        from it, so it is not deduped and does not move the day on."""
        self._watching(e)
        if e.camera_id == self.cfg.desk.camera:
            self.cover.beat(e.ts)
        return []

    def on_policy(self, e: Event) -> list[Event]:
        """`vault.policy.changed`: a paused desk camera is not watching, from then until it is switched back on."""
        self.cover.policy(self.cfg.desk.camera in (getattr(e, "paused_cameras", None) or []), e.ts)
        return []

    def on_item(self, e: ItemSynced) -> list[Event]:
        """Assignments (due dates), calendar events (start times) and nights a watch measured, from the connectors."""
        if e.kind == "health":
            return self._measured(e)
        if e.when is None or e.kind not in ("assignment", "calendar_event") or not self._admit(e):
            return []
        if e.external_id.startswith("nearby:"):  # an optional event nobody said yes to: nothing to be ready for
            return []
        if e.kind == "calendar_event" and self.cfg.is_listing is not None and self.cfg.is_listing(e):  # a campus listing: the same
            return []
        out = self._roll(self.sched.local_day(e.ts), e.ts)
        when = e.when if e.when.tzinfo else e.when.replace(tzinfo=self.sched.tz)
        book = self.assignments if e.kind == "assignment" else self.calendar
        key = f"{e.connector}:{e.external_id}"
        item = book.setdefault(key, Item(e.title, when, e.url, known_at=e.ts))
        item.title, item.when, item.url = e.title, when, e.url  # a re-sync updates, never duplicates
        if e.kind == "assignment":
            out += self._due_soon(key, item, e)
        return out

    def _measured(self, e: ItemSynced) -> list[Event]:
        """A night from the watch. The latest reading of a night wins (a later upload has more of it); one for
        a day already closed changes nothing. It says nothing about the desk, so it starts no camera clock."""
        reading, day = parse_summary(e.summary), night_day(e.external_id)
        if reading is None or day is None or not self._admit(e):
            return []
        out = self._roll(self.sched.local_day(e.ts), e.ts)
        if day >= self.today:
            self.measured_sleep[day] = reading.asleep_h
        return out

    def on_disposed(self, e: ItemDisposed) -> list[Event]:
        if not self._admit(e):
            return []
        out = self._roll(self.sched.local_day(e.ts), e.ts)
        day = self._day(self.sched.local_day(e.ts))
        day.disposed += 1
        day.correct += e.correct
        return out + [self._signal("disposal", float(e.correct), e)]

    def on_day_closed(self, e: DayClosed) -> None:
        """A close already in the log (from an earlier flush) closes the day here too, once."""
        self.now = max(self.now or e.ts, e.ts)  # the tick that closed it is a time this tracker has seen, on replay too
        if self.today is None:
            self.today = e.day + ONE_DAY
        elif e.day >= self.today:
            self._roll(e.day + ONE_DAY, e.ts)

    def on_nudge(self, e: NudgeRaised) -> None:
        self.latest_nudge[e.habit_id] = e

    def flush(self, now: datetime) -> list[DayClosed]:
        """Ticker: close every local day that ended more than the grace period before `now`."""
        if self.today is None:
            return []
        self.now = max(self.now or now, now)
        return self._roll(self.sched.local_day(now - timedelta(minutes=self.cfg.close_grace_minutes)), now)

    # --- owner recognition (P9): used by MemberHabits only ---

    def clock(self, e: Event) -> list[DayClosed]:
        """A camera event that is not counted here: the room was watched, and the day moves on."""
        self._watching(e)
        return self._roll(self.sched.local_day(e.ts), e.ts) if self._admit(e) else []

    def keep_up(self, today: date | None, now: datetime | None, ts: datetime) -> list[DayClosed]:
        """Close the same days as the tracker the bus feeds, at the same moment."""
        if now is not None:
            self.now = max(self.now or now, now)
        if today is None:
            return []
        if self.today is None:
            self.today = today
        return self._roll(today, ts)

    def pause(self, ts: datetime) -> None:
        """From `ts` on, people are counted somewhere else: whoever was at the desk is no longer followed here."""
        if self.unjudged_since is None:
            self.unjudged_since = ts
            self._since.clear()
            self._session_min = 0.0

    def resume(self, ts: datetime) -> None:
        if self.unjudged_since is None:
            return
        day = self.sched.local_day(self.unjudged_since)
        while day <= self.sched.local_day(ts):
            self.unjudged.add(day)
            day += ONE_DAY
        self.unjudged_since = None
        self.active_until = max(self.active_until or ts, ts)  # the pause is not an empty desk

    def left(self, camera_id: str, track_id: int | None = None) -> None:
        """A track stopped being this person's (its tag ended) before it left the desk: stop the clock on it."""
        gone = [k for k in self._since if k == f"{camera_id}:{track_id}" or (track_id is None and k.startswith(f"{camera_id}:"))]
        for key in gone:
            del self._since[key]

    @property
    def at_desk(self) -> bool:
        return bool(self._since)

    def _judged(self, day: date, end: datetime) -> bool:
        return day not in self.unjudged and (self.unjudged_since is None or self.unjudged_since >= end)

    # --- days ---

    def _admit(self, e: Event) -> bool:
        """False when the event was already counted, or belongs to a day that is closed."""
        day = self.sched.local_day(e.ts)
        if (self.today is not None and day < self.today) or e.event_id in self._seen.get(day, ()):
            return False
        self._seen.setdefault(day, set()).add(e.event_id)
        self.today = self.today or day
        self.now = max(self.now or e.ts, e.ts)
        return True

    def _watching(self, e: Event) -> None:
        """Any camera event, whatever it shows, means the room was being watched from then on."""
        self.watching_since = self.watching_since or e.ts

    def _day(self, day: date) -> Day:
        return self.days.setdefault(day, Day())

    def _roll(self, day: date, ts: datetime) -> list[DayClosed]:
        out = []
        while self.today < day:
            out += self._close(self.today, ts)
            self.today += ONE_DAY
        return out

    def _close(self, day: date, ts: datetime) -> list[DayClosed]:
        end = self.sched.day_start(day + ONE_DAY)
        out = []
        for h in self.cfg.habits:
            value, applies = self.measure(h, day, end)
            met = h.met(value) or not applies
            streak = self.streak(h.id) if not applies else self.streak(h.id) + 1 if met else 0
            self.history[h.id].append(Closed(day, value, met, streak, applies, *self.watched(h, day, end), self.source(h, day)))
            out.append(DayClosed(event_id=f"habits-day-{h.id}-{day}", ts=ts, habit_id=h.id, day=day,
                                 met=met, value=value, streak=streak))
        self.days.pop(day, None)
        self.measured_sleep = {d: hours for d, hours in self.measured_sleep.items() if d > day}
        self._seen = {d: ids for d, ids in self._seen.items() if d > day}
        self.assignments = {k: a for k, a in self.assignments.items() if a.when >= end}
        self.calendar = {k: c for k, c in self.calendar.items() if c.when >= end}
        self._since = {k: t for k, t in self._since.items() if end - t < ONE_DAY}  # tracks that never exited
        self.unjudged = {d for d in self.unjudged if d > day}
        self.cover.forget(self.sched.day_start(day) - ONE_DAY)  # the next night began after this
        return out

    def streak(self, habit_id: str) -> int:
        closed = self.history[habit_id]
        return closed[-1].streak if closed else 0

    def measure(self, h: Habit, day: date, now: datetime) -> tuple[float, bool]:
        """(value, applies) for a habit on an open day, as of `now`."""
        d = self.days.get(day) or Day()
        start, end = self.sched.day_start(day), self.sched.day_start(day + ONE_DAY)
        since = self.watching_since
        judged = self._judged(day, end)  # False only for a set of habits paused by owner recognition (P9)
        watched = judged and since is not None and since < end  # a camera had reported by the end of this day
        enough = self.watched(h, day, now)[1] is None  # sleep, focus: the desk camera saw enough of the window
        match h.signal:
            case "away_window" if day in self.measured_sleep:  # the watch measured it: judged whatever the cameras saw
                return self.measured_sleep[day], True
            case "away_window":  # a night that ended before any camera, or has not begun by `now`, is not judged
                night_start, night_end = self.sched.sleep_window(day)
                observed = judged and since is not None and since < night_end and night_start < now
                return round(max(d.away_h, self._open_gap_h(day, now)), 2), observed and enough
            case "meal":
                return d.meals, watched
            case "drink":
                return d.drinks, watched
            case "desk_time":
                near = end + timedelta(hours=h.deadline_hours)
                return round(d.desk_min / 60, 2), watched and enough and any(start <= a.when < near for a in self.assignments.values())
            case "start_lead":  # only assignments an early start on could have been both known and seen
                due = [a for a in self.assignments.values() if watched and start <= a.when < end
                       and _hours(a.when - max(a.known_at, since)) >= h.target]
                leads = [max(0.0, _hours(a.when - a.started_at)) if a.started_at else 0.0 for a in due]
                return round(min(leads, default=0.0), 1), bool(due)
            case "forgot":
                return d.forgot, watched
            case "disposal":
                return (round(100 * d.correct / d.disposed) if d.disposed else 0), d.disposed > 0

    def watched(self, h: Habit, day: date, now: datetime) -> tuple[float | None, str | None]:
        """(coverage, reason) for sleep and focus as of `now`: the share of the night, or of the waking day, the
        desk camera was reporting for, and why the habit is not judged when that is too little. (None, None)
        for the other habits, before the window has begun, while the camera's coverage is not measured, and
        for a night a watch measured (the camera is not what that figure rests on)."""
        if h.signal == "away_window" and day in self.measured_sleep:
            return None, None
        if h.signal == "away_window":
            (start, end), name = self.sched.sleep_window(day), "night"
        elif h.signal == "desk_time":  # wake to bedtime: a camera that is off overnight says nothing about focus
            start, end, name = self.sched.sleep_window(day)[1], self.sched.sleep_window(day + ONE_DAY)[0], "day"
        else:
            return None, None
        upto = min(now, end)
        if not self.cover.measured or upto <= start:
            return None, None
        known = max(self.now or now, now)
        total = upto - start
        seen = sum((y - x for x, y in self.cover.seen(start, upto, known)), timedelta(0))
        if seen / total >= self.cfg.coverage.min_fraction:
            return round(seen / total, 2), None
        window = f"the {_h(total)} {name}" if upto == end else f"{_h(total)} of the {name} so far"
        paused = self.cover.paused_in(start, upto, known)
        if paused and paused >= (total - seen) / 2:
            return round(seen / total, 2), f"cameras were paused for {_h(paused)} of {window}"
        return round(seen / total, 2), f"cameras saw {_h(seen)} of {window}"

    def source(self, h: Habit, day: date) -> str | None:
        """Where an open day's sleep figure comes from: "watch" or "camera". None for every other habit."""
        if h.signal != "away_window":
            return None
        return "watch" if day in self.measured_sleep else "camera"

    # --- the desk: focus time and the away window ---

    def _desk(self, e: ZoneEvent) -> list[Event]:
        """Each event accounts for a stretch of presence ending at `e.ts`; overlapping people count once."""
        key = f"{e.camera_id}:{e.track_id}"
        if e.kind == "enter":
            start = e.ts
        elif e.kind == "dwell":
            start = e.ts - timedelta(seconds=e.dwell_s or 0)
        else:
            start = self._since.pop(key, e.ts)
        if e.kind != "exit":
            self._since[key] = e.ts
        self.desk_frame = e.frame_ref or self.desk_frame
        out = self._away(start, e) + self._credit(start, e)
        if e.kind == "exit" and self._session_min > 0:
            out.append(self._signal("desk_time", round(self._session_min, 1), e))
            self._session_min = 0.0
        return out

    def _away(self, start: datetime, e: ZoneEvent) -> list[Event]:
        """The desk was empty from the last activity until `start`: longest such gap per sleep window."""
        prev = self.active_until or self.watching_since
        out = []
        day = self.sched.local_day(prev)
        while start > prev and day <= self.sched.local_day(start) + ONE_DAY:
            night_start, night_end = self.sched.sleep_window(day)
            hours = self._empty_h(max(prev, night_start), min(start, night_end), e.ts)
            if day >= self.today and hours > self._day(day).away_h:
                self._day(day).away_h = hours
                if hours >= MIN_AWAY_H:
                    out.append(self._signal("away_window", round(hours, 2), e, f"-{day}"))
            day += ONE_DAY
        return out

    def _open_gap_h(self, day: date, now: datetime) -> float:
        """Hours of this night's window the desk has been empty since the last activity."""
        prev = self.active_until or self.watching_since
        known = max(self.now or now, now)
        if prev is None or any(self.cover.unbroken(t, known) for t in self._since.values()):  # someone is at the desk
            return 0.0
        start, end = self.sched.sleep_window(day)
        return self._empty_h(max(prev, start), min(now, end), known)

    def _empty_h(self, a: datetime, b: datetime, known: datetime) -> float:
        """The longest stretch of [a, b] the desk camera was reporting for: an empty desk never runs across a gap."""
        return max((_hours(y - x) for x, y in self.cover.seen(a, b, known)), default=0.0)

    def _credit(self, start: datetime, e: ZoneEvent) -> list[Event]:
        """Add the part of [start, e.ts] not yet counted to desk time, split at local midnight."""
        start = max(start, self.active_until or start)
        self.active_until = max(self.active_until or e.ts, e.ts)
        out = []
        while start < e.ts:
            day = self.sched.local_day(start)
            stop = min(e.ts, self.sched.day_start(day + ONE_DAY))
            minutes = sum((y - x).total_seconds() for x, y in self.cover.seen(start, stop, e.ts)) / 60  # watched time only
            if day >= self.today and minutes:
                self._day(day).desk_min += minutes
                self._session_min += minutes
                out += self._work(minutes, stop, e)
            start = stop
        return out

    def _work(self, minutes: float, at: datetime, e: ZoneEvent) -> list[Event]:
        """Desk time counts towards every open assignment; enough of it means it was started."""
        h = self.cfg.by_signal("start_lead")
        out = []
        for key, a in self.assignments.items() if h else []:
            if a.started_at or a.when <= at or a.known_at > at:
                continue
            a.desk_min += minutes
            if a.desk_min >= h.min_minutes:
                a.started_at = at
                out.append(self._signal("start_lead", round(_hours(a.when - at), 1), e, f"-{key}"))
        return out

    # --- connectors: deadlines and the calendar ---

    def _due_soon(self, key: str, a: Item, e: ItemSynced) -> list[Event]:
        h = self.cfg.by_signal("desk_time")
        day = self._day(self.sched.local_day(e.ts))
        hours = _hours(a.when - e.ts)
        if not h or not 0 < hours <= h.deadline_hours or key in day.due_flagged:
            return []
        day.due_flagged.add(key)
        return [self._signal("due_soon", round(hours, 1), e)]

    def next_deadline(self, now: datetime, within_hours: float) -> Item | None:
        soon = [a for a in self.assignments.values() if timedelta(0) < a.when - now <= timedelta(hours=within_hours)]
        return min(soon, key=lambda a: a.when, default=None)

    def unstarted(self, now: datetime, h: Habit) -> Item | None:
        """An assignment that can still be started early (target hours ahead) and has not been, as far as a camera can tell."""
        late = [a for a in self.assignments.values()
                if self.watching_since is not None and self.unjudged_since is None and not a.started_at
                and _hours(a.when - a.known_at) >= h.target
                and h.target < _hours(a.when - now) <= h.deadline_hours]
        return min(late, key=lambda a: a.when, default=None)

    # --- scenes: meals, drinks, watch items ---

    def _new_sightings(self, h: Habit, e: SceneDescribed, labels: list[str]) -> int:
        """How many meals or cans in this frame were not there at the last sighting."""
        words = self._words.get(h.id)
        if words is None:
            return 0
        hits = sum(1 for label in labels if words.search(label))
        n = (hits if h.count_each else min(hits, 1)) or int(bool(words.search(e.caption)))
        if not n:
            return 0
        last_ts, last_n = self._sighted.get(h.id, (None, 0))
        fresh = last_ts is None or e.ts - last_ts > timedelta(minutes=h.gap_minutes)
        self._sighted[h.id] = (e.ts, n if fresh else max(n, last_n))
        return n if fresh else max(0, n - last_n)

    def _spot_watch_items(self, e: SceneDescribed, labels: list[str]) -> None:
        found = {w: e.frame_ref for w, pat in self._watch.items() if any(pat.search(label) for label in labels)}
        # A desk crop is the full answer; a whole-camera caption can only add what it happened to list.
        self.on_desk = found if e.zone_id == self.cfg.desk.zone else self.on_desk | found

    def _leaving(self, e: ZoneEvent) -> list[Event]:
        """Out the door with a watch item still on the desk and something on the calendar soon."""
        h = self.cfg.by_signal("forgot")
        if not h or not self.on_desk:
            return []
        soon = [c for c in self.calendar.values()
                if not c.flagged and timedelta(0) <= c.when - e.ts <= timedelta(minutes=h.event_minutes)]
        if not soon:
            return []
        event = min(soon, key=lambda c: c.when)
        event.flagged = True  # one miss per calendar event, however often they pace through the door
        items = sorted(self.on_desk)
        self._day(self.sched.local_day(e.ts)).forgot += len(items)
        self.forgotten = {"ts": e.ts, "items": items, "title": event.title, "url": event.url, "starts": event.when,
                          "frame_ref": self.on_desk[items[0]]}
        return [self._signal("forgot", len(items), e)]

    def _signal(self, kind: str, value: float, e: Event, tag: str = "") -> SignalRecorded:
        # The id follows from the source event, so a repeat can never be logged twice.
        return SignalRecorded(event_id=f"habits-{kind}-{e.event_id}{tag}", ts=e.ts, kind=kind, value=value,
                              source_event_id=e.event_id)

    # --- views for the phone ---

    def digest(self, day: date | None = None, now: datetime | None = None) -> dict:
        """One day's habits: value, target, met so far, streak and the latest nudge. Today by default."""
        now = now or self.now
        day = day or (self.sched.local_day(now) if now else self.today)
        rows = [self._row(h, day, now) for h in self.cfg.habits] if day else []
        nudges = [r["nudge"] for r in rows if r and r["nudge"]]
        return {"day": day.isoformat() if day else None, "habits": [r for r in rows if r],
                "latest_nudge": max(nudges, key=lambda n: n["ts"], default=None)}

    def history_for(self, habit_id: str, days: int = 7) -> list[dict]:
        """The last `days` days of one habit, oldest first, ending with today so far."""
        h = self.cfg.by_id(habit_id)
        closed = [self._view(h, c, closed=True) for c in self.history[habit_id]]
        today = self._row(h, self.sched.local_day(self.now), self.now) if self.now else None
        if today and not today["closed"]:
            closed.append(today)
        return closed[-days:] if days > 0 else []

    def _row(self, h: Habit, day: date, now: datetime | None) -> dict | None:
        done = next((c for c in self.history[h.id] if c.day == day), None)
        if done:
            return self._view(h, done, closed=True)
        if self.today is not None and day < self.today:
            return None  # before Atlas was watching
        now = now or self.sched.day_start(day + ONE_DAY)
        if self.now is not None:
            now = min(now, self.now)  # nothing is known past the latest event or tick: a later `now` is not an empty desk
        value, applies = self.measure(h, day, now)
        return self._view(h, Closed(day, value, h.met(value) or not applies, self.streak(h.id), applies,
                                    *self.watched(h, day, now), self.source(h, day)), closed=False)

    def _view(self, h: Habit, c: Closed, closed: bool) -> dict:
        n = self.latest_nudge.get(h.id)
        nudge = {"message": n.message, "tone": n.tone, "ts": n.ts.isoformat()} if n and self.sched.local_day(n.ts) == c.day else None
        row = {"id": h.id, "name": h.name, "unit": h.unit, "direction": h.direction, "target": h.target,
               "day": c.day.isoformat(), "value": c.value, "met": c.met, "applies": c.applies, "streak": c.streak,
               "closed": closed, "nudge": nudge}
        if c.source:  # sleep: measured by a watch, or estimated from the empty desk
            row["source"] = c.source
        # Sleep and focus, once the desk camera's coverage is measured: how much of the window it saw.
        return row if c.coverage is None else row | {"coverage": c.coverage, "reason": c.reason}
