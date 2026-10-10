"""Habits per enrolled person (P9): who the desk time, the meals and the door belong to.

    habits = MemberHabits(HabitTracker(cfg), nudger_for=lambda tracker: Nudger(cfg, tracker))
    bus.subscribe("perception.identity", habits.on_identity, replay=True)   # recognition on/off, who is enrolled
    bus.subscribe("perception.person", habits.on_person, replay=True)       # track N on camera C is member M
    ... and every subscription the plain tracker has, on `habits` instead

It stands where the HabitTracker stood and answers everything the tracker answers.

Recognition off, or on with nobody enrolled (the default): every event goes to the one unnamed tracker and
comes back unchanged, exactly as without this class.

Recognition on with someone enrolled: there is one more HabitTracker per member. A person's zone events
(desk time, the door) count only for the member their track is tagged as, and a desk scene (meals, drinks,
what is on the desk) counts for the members who were at the desk when its frame was taken. A track nobody
is tagged on counts for no one. The unnamed tracker still keeps the calendar: it sees every camera event as
a tick of the clock, closes the days, and its `habits.day.closed` events in the log are what make every
tracker here close the same days at the same moments on replay. Its own camera habits are not judged for
those days; things thrown away (nobody's track) and connector items stay with it, and the items are also
given to each member.

The members' own signals and closed days are NOT published: the habits events have no field to say whose
they are (that is a contract change), so a member's numbers exist only as this projection, rebuilt from the
log. Their nudges are raised as ordinary `habits.nudge.raised` (no name, no member id).

`perception.person.*` and `perception.identity.changed` are read by field name, so they work both as the
classes in contracts/perception.py (what the log gives back) and as plain events with the same fields.
"""

from collections.abc import Callable
from copy import copy
from datetime import date, datetime, timedelta
from typing import Any

from contexts.habits.domain.tracker import HabitTracker
from contracts.connectors import ItemSynced
from contracts.events import Event
from contracts.habits import DayClosed, NudgeRaised
from contracts.impact import ItemDisposed
from contracts.perception import SceneDescribed, ZoneEvent


class MemberHabits:
    def __init__(self, base: HabitTracker, nudger_for: Callable[[HabitTracker], Any] | None = None) -> None:
        self.base, self.cfg = base, base.cfg
        self.attributing = False  # recognition is on and somebody is enrolled
        self.members: dict[str, HabitTracker] = {}
        self.nudgers: dict[str, Any] = {}
        self.tags: dict[tuple[str, int], str] = {}  # (camera, track) -> member
        self._stays: dict[str, tuple[datetime, datetime | None]] = {}  # member -> their latest time at the desk
        self._seated: dict[tuple[str, int], datetime] = {}  # person track in the desk zone -> since when
        self._nudger_for = nudger_for

    def __getattr__(self, name: str) -> Any:
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)  # sched, now, today, measure, history, ...: the unnamed tracker's

    # --- recognition (replay=True) ---

    def on_identity(self, e: Event) -> list[Event]:
        ids = [str(m) for m in (getattr(e, "members", None) or [])]
        on = bool(getattr(e, "enabled", False)) and bool(ids)
        for gone in [m for m in self.members if m not in ids]:  # a deleted member's habits go with the template
            del self.members[gone]
            self.nudgers.pop(gone, None)
            self._stays.pop(gone, None)
        self.tags = {k: m for k, m in self.tags.items() if m in self.members}
        if on != self.attributing:
            self.attributing = on
            for t in self.members.values():
                self._switch(t, on, e.ts)
            if on:
                self.base.pause(e.ts)
            else:
                self.base.resume(e.ts)
                self.tags.clear()
                self._stays.clear()
        for m in ids:
            if m not in self.members:
                self.members[m] = t = HabitTracker(self.cfg, self.base.cover)  # one room: the same cameras watched it
                t.keep_up(self.base.today, self.base.now, e.ts)
                t.assignments = {k: copy(a) for k, a in self.base.assignments.items()}  # deadlines already synced
                t.calendar = {k: copy(c) for k, c in self.base.calendar.items()}
                for item in [*t.assignments.values(), *t.calendar.values()]:
                    item.desk_min, item.started_at, item.flagged = 0.0, None, False
                self._switch(t, on, e.ts)
                if self._nudger_for:
                    self.nudgers[m] = self._nudger_for(t)
        return []

    @staticmethod
    def _switch(t: HabitTracker, on: bool, ts: datetime) -> None:
        if on:
            t.resume(ts)
            t.watching_since = t.watching_since or ts  # nothing is known about this person before now
        else:
            t.pause(ts)

    def on_person(self, e: Event) -> list[Event]:
        camera, track = e.camera_id, getattr(e, "track_id", None)
        if e.type.endswith(".identified"):
            if not (self.attributing and e.member_id in self.members and track is not None):
                return []
            key = (camera, int(track))
            self.tags[key] = e.member_id
            if key in self._seated:  # already sitting there when the face was matched: theirs since they sat down
                sat = ZoneEvent(event_id=f"{e.event_id}-seated", ts=e.ts, camera_id=camera, track_id=key[1], label="person",
                                zone_id=self.cfg.desk.zone, kind="dwell", dwell_s=(e.ts - self._seated[key]).total_seconds())
                return self._member_zone(e.member_id, sat) + self._sync(e.ts)
        elif e.type.endswith(".lost"):
            for key in [k for k in self.tags if k[0] == camera and (track is None or k[1] == track)]:
                member = self.tags.pop(key)
                t = self.members[member]
                t.left(*key)
                self._presence(member, t, e.ts)
        return []

    # --- the tracker's handlers (replay=True) ---

    def on_zone_event(self, e: ZoneEvent) -> list[Event]:
        key = (e.camera_id, e.track_id)
        if e.label == "person" and e.zone_id == self.cfg.desk.zone:  # which tracks sit at the desk: ids and times only
            if e.kind == "exit":
                self._seated.pop(key, None)
            else:
                self._seated.setdefault(key, e.ts - timedelta(seconds=e.dwell_s or 0))
        if not self.attributing:
            return self.base.on_zone_event(e) + self._sync(e.ts)
        member = self.tags.get(key) if e.label == "person" else None
        out = self._member_zone(member, e) if member is not None else []  # before the clock: minutes before midnight stay on their day
        return self.base.clock(e) + out + self._sync(e.ts)

    def _member_zone(self, member: str, e: ZoneEvent) -> list[Event]:
        t = self.members[member]
        at_desk = e.zone_id == self.cfg.desk.zone
        if at_desk and e.kind != "exit" and not t.at_desk:
            self._stays[member] = (e.ts - timedelta(seconds=e.dwell_s or 0), None)
        out = self._run(member, t.on_zone_event(e))
        if at_desk:
            self._presence(member, t, e.ts)
        return out

    def on_scene(self, e: SceneDescribed) -> list[Event]:
        if not self.attributing:
            return self.base.on_scene(e) + self._sync(e.ts)
        out: list[Event] = []
        for member, (start, end) in list(self._stays.items()):
            if start <= e.ts and (end is None or e.ts <= end):  # the frame was taken while they were at the desk
                out += self._run(member, self.members[member].on_scene(e))
        return self.base.clock(e) + out + self._sync(e.ts)

    def on_item(self, e: ItemSynced) -> list[Event]:
        out = self.base.on_item(e)
        for member, t in self.members.items():
            out += self._run(member, t.on_item(e))
        return out + self._sync(e.ts)

    def on_disposed(self, e: ItemDisposed) -> list[Event]:
        return self.base.on_disposed(e) + self._sync(e.ts)

    def on_day_closed(self, e: DayClosed) -> list[Event]:
        self.base.on_day_closed(e)
        return self._sync(e.ts)

    def on_nudge(self, e: NudgeRaised) -> None:
        self.base.on_nudge(e)
        sole = self._sole()
        if sole is not None:  # with one person enrolled it can only be theirs
            sole.on_nudge(e)

    def flush(self, now: datetime) -> list[Event]:
        """Ticker: close finished days everywhere, and (live only) the members' own nudges."""
        out = self.base.flush(now) + self._sync(now)
        if self.attributing:
            for nudger in self.nudgers.values():
                out += nudger.check(now)
        return out

    # --- keeping the members in step ---

    def _presence(self, member: str, t: HabitTracker, ts: datetime) -> None:
        stay = self._stays.get(member)
        if stay and stay[1] is None and not t.at_desk:
            self._stays[member] = (stay[0], ts)

    def _sync(self, ts: datetime) -> list[Event]:
        out: list[Event] = []
        for member, t in self.members.items():
            out += self._run(member, t.keep_up(self.base.today, self.base.now, ts))
        return out

    def _run(self, member: str, events: list[Event]) -> list[Event]:
        """What a member's tracker said goes to their nudger and no further."""
        nudger = self.nudgers.get(member)
        out: list[Event] = []
        for e in events if nudger else []:
            if e.type == "habits.signal.recorded":
                out += nudger.on_signal(e)
            elif e.type == "habits.day.closed":
                out += nudger.on_day_closed(e)
        return out

    # --- views for the phone ---

    def _sole(self) -> HabitTracker | None:
        return next(iter(self.members.values())) if self.attributing and len(self.members) == 1 else None

    def view(self, member: str | None = None) -> HabitTracker:
        """One member's tracker (KeyError if not enrolled). Without a member: the only enrolled person's while
        recognition is on, otherwise the unnamed one."""
        return self.members[member] if member is not None else self._sole() or self.base

    def digest(self, day: date | None = None, now: datetime | None = None, member: str | None = None) -> dict:
        return self.view(member).digest(day, now)

    def history_for(self, habit_id: str, days: int = 7, member: str | None = None) -> list[dict]:
        return self.view(member).history_for(habit_id, days)
