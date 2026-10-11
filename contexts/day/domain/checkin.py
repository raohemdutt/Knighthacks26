"""The check-in (D4): Atlas looks back across its signals and speaks up unasked. Pure: no clock, no model, no I/O.

    found = signals(reading)                 # the signals that are there right now, each in plain words
    if agree(found) and held(now, gate) is None:
        past = history(reading, found)       # earlier days that looked like this one, or None
        ...                                  # contexts/day/application/checkin.py publishes `day.checkin.raised`

Every time is event time. The rule, in plain words:

- Six things can be noticed, each from something the box really recorded, each left out when it is not there:
  - `sleep`: last night (the watch's night, from the health connector) was SLEEP_SHORT_MIN or more under this
    person's own median night;
  - `resting_hr`: today's resting heart rate is above this person's own median by RHR_MARGIN bpm or RHR_SPREADS
    times their own spread (median absolute deviation), whichever is more;
  - `expression`: EXPR_SHARE or more of the counted looks in the last EXPR_WINDOW_MIN minutes read as strained
    (angry, disgusted, fearful or sad), with EXPR_MIN_SAMPLES looks or more spread over most of that window;
  - `sitting`: at the desk for SIT_MIN minutes without a break of SIT_BREAK_S or more;
  - `drink`: no drink seen picked up for DRINK_GAP_MIN minutes of this sitting, while a bottle, cup, mug, can or
    glass has been in the desk camera's view (see `DrinkWatch` for what "picked up" is and how far to trust it);
  - `history`: never a reason to speak, only a clause once the others agreed (see `history`).
- A baseline is the person's own: the median of their other measured nights within BASELINE_DAYS days, and
  nothing is said until there are MIN_BASELINE of them. No population norm appears anywhere.
- It speaks when two independent groups agree and at least one of them is about right now. The groups: `sleep`,
  `resting_hr` and `expression` each count as one; `sitting` and `drink` count as one together, because both
  only say "a long time at the desk". So two watch signals alone never fire in the morning (nothing is
  happening now), a long sitting with no drink never fires by itself, and no single signal ever does.
- It stays quiet (`held`): outside waking hours, with nobody at the desk, during a fixed block (a class), in
  the LEAVE_GUARD_MIN minutes before a time to leave or between that time and the event, and for COOLDOWN_MIN
  minutes after the last check-in.
- No diagnosis: "looked strained" and "resting heart rate is up" are what was measured. The words in
  contexts/day/domain/narration.py DIAGNOSIS never appear (`diagnosis_in`), in the template or in a model's rewrite.
"""

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, tzinfo
from statistics import median

from contexts.day.domain.coach import STRAINED_LOOKS, Span, stretches
from contexts.day.domain.narration import DIAGNOSIS, clock, hhmm, meal_name
from contexts.play.dayrun.models import Move
from contracts.day import CheckinHistory, CheckinPastDay, CheckinSignal, CheckinSuggestion, PlanMove

BASELINE_DAYS = 14  # a baseline is the other measured nights within this many days
MIN_BASELINE = 5  # ... and nothing is said with fewer of them
SLEEP_SHORT_MIN = 60  # a night this far under the person's own median is short
RHR_MARGIN = 3.0  # bpm over the person's own median before "up" is said
RHR_SPREADS = 1.5  # ... or this many of their own spreads (median absolute deviation), whichever is more
EXPR_WINDOW_MIN = 20
EXPR_MIN_SAMPLES = 20  # counted looks in the window
EXPR_SHARE = 0.5  # strained for half the looks or more
EXPR_COVER_MIN = 15  # the looks have to reach back this far, and
EXPR_FRESH_MIN = 5  # ... the latest has to be this recent, for "the last twenty minutes" to be true
SIT_MIN = 60
SIT_BREAK_S = 180  # standing up for less than this does not end a sitting
AT_DESK_WITHIN = timedelta(minutes=3)  # seen at the desk this recently: there is somebody to speak to
DRINK_GAP_MIN = 90
CONTAINER_WITHIN = timedelta(hours=3)  # a drink container was in view this recently, or nothing is said about drinks
COOLDOWN_MIN = 90
LEAVE_GUARD_MIN = 30
EVERY_MIN = 3  # the rule is looked at this often at most
HISTORY_MIN = 3  # days behind "the last three times a day looked like this"
HISTORY_DAYS = 60  # how far back those days may lie
FELL_SHARE = 0.5  # a block with less than this share of its planned minutes at the desk fell apart
MOVE_WITHIN_MIN = 120  # "move the block" is offered for a block that starts this soon (or is running)

BODY = ("sleep", "resting_hr", "expression")  # each an independent reading
DESK = ("sitting", "drink")  # one group: both only say "a long time at the desk"
PRESENT = ("expression", "sitting", "drink")  # about right now
WATCH = ("sleep", "resting_hr")  # what makes a past day "look like this one"

DRINK_LABELS = ("bottle", "water bottle", "cup", "mug", "can", "glass", "wine glass")
DRINK_VERBS = re.compile(r"\b(?:drinking|drinks from|sipping|sips|tak(?:es|ing) a (?:sip|drink))\b", re.I)
UPPER_SHARE = 0.4  # the top of a person's box: head and shoulders
LIFT_RISE = 0.10  # the container has to be this far (of the picture's height) above where it rests
STILL = 0.03  # ... and "rests" is three looks in a row within this much
LIFT_WITHIN = timedelta(seconds=6)  # two raised looks this close together are a lift; one is a detector blink
SAME_DRINK = timedelta(seconds=60)

LIMITS = (
    "A baseline is your own: the median of your other measured nights in the last two weeks. No population norm is used.",
    "\"Picked up a drink\" is a bottle, cup, mug, can or glass the desk camera saw raised to head height. It cannot "
    "tell water from coffee, and it does not see a drink taken out of its view.",
    "\"Looked strained\" is how a face looked to a small model over many looks: a noisy reading, not an emotion "
    "and nothing about health.",
    "Minutes at the desk are what the camera saw, not minutes of focus.",
    "This is not medical advice and nothing here is a diagnosis.",
)


# ---------- the watch: this person's own baseline ----------


@dataclass(frozen=True)
class Night:
    """One night from the health connector's item, with the day's resting heart rate."""

    day: date  # the local day it ended on
    asleep_min: int | None = None
    resting_hr: float | None = None
    ref: str = ""  # "health:sleep:2026-10-11", the item's id
    in_bed_min: int | None = None
    stages: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Baseline:
    median: float
    spread: float  # median absolute deviation
    n: int


def baseline(values: list, min_n: int = MIN_BASELINE) -> Baseline | None:
    xs = [float(v) for v in values if v is not None]
    if len(xs) < min_n:
        return None
    mid = median(xs)
    return Baseline(mid, median(abs(x - mid) for x in xs), len(xs))


def own(nights: dict[date, Night], day: date, what: str) -> Baseline | None:
    """This person's baseline for `day`: their other measured nights within BASELINE_DAYS days of it."""
    return baseline([getattr(n, what) for d, n in nights.items() if d != day and abs((d - day).days) <= BASELINE_DAYS])


def short_night(nights: dict[date, Night], day: date) -> bool:
    n, b = nights.get(day), own(nights, day, "asleep_min")
    return bool(n and b and n.asleep_min is not None and n.asleep_min <= b.median - SLEEP_SHORT_MIN)


def resting_up(nights: dict[date, Night], day: date) -> bool:
    n, b = nights.get(day), own(nights, day, "resting_hr")
    return bool(n and b and n.resting_hr is not None and n.resting_hr - b.median >= max(RHR_MARGIN, RHR_SPREADS * b.spread))


# ---------- how the person looked ----------


@dataclass(frozen=True)
class Look:
    """One `perception.expression.observed` window of the person at the desk."""

    start: datetime
    end: datetime
    samples: int
    strained: float  # how many of the counted looks read as strained
    ref: str = ""  # the event's id


def strained_looks(shares: dict[str, float], samples: int) -> float:
    return sum(share for label, share in shares.items() if label in STRAINED_LOOKS) * samples


def looked_strained(looks: list[Look], now: datetime) -> tuple[float, list[Look]] | None:
    """(the strained share, the windows behind it) when the last EXPR_WINDOW_MIN minutes looked strained."""
    recent = sorted((x for x in looks if now - timedelta(minutes=EXPR_WINDOW_MIN) < x.end <= now and x.samples > 0), key=lambda x: x.end)
    total = sum(x.samples for x in recent)
    if total < EXPR_MIN_SAMPLES:
        return None
    if recent[0].start > now - timedelta(minutes=EXPR_COVER_MIN) or recent[-1].end < now - timedelta(minutes=EXPR_FRESH_MIN):
        return None  # too short a stretch, or an old one
    share = sum(x.strained for x in recent) / total
    return (share, recent) if share >= EXPR_SHARE else None


# ---------- at the desk ----------


def sitting(spans: list[Span], now: datetime) -> Span | None:
    """The unbroken stretch at the desk that reaches `now`, or None when nobody is there."""
    runs = stretches([s for s in spans if s[0] <= now], SIT_BREAK_S)
    if not runs or now - runs[-1][1] > AT_DESK_WITHIN:
        return None
    return runs[-1][0], now


# ---------- a drink ----------


class DrinkWatch:
    """A drink seen picked up, from what the desk camera's detector tracks. A projection: feed it observations in time order.

    The detector tracks bottles, cups and cans as boxes about once a second. A container "rests" where it was seen
    three looks in a row without moving. It was picked up when its centre is in the top of a person's box (head and
    shoulders), at least LIFT_RISE above where it rests, on two looks within a few seconds. A caption of the desk
    frame that says someone is drinking counts as well.

    How far to trust it: this is a proxy for drinking, not drinking. It misses a drink the detector does not
    track (a small or hidden one), one taken out of the camera's view, and a sip too quick for two looks; it
    cannot tell water from coffee; and a container waved in front of a face counts. So the only claim ever made
    from it is "I haven't seen you pick up a drink since ...", and only while a container is in view."""

    def __init__(self) -> None:
        self.moments: list[tuple[datetime, str]] = []  # (when, frame ref): a drink seen picked up
        self.container: tuple[datetime, str, str] | None = None  # (when, label, frame ref): the last container in view
        self._rest: dict[str, float] = {}  # label -> the height where it rests
        self._still: dict[str, tuple[float, int]] = {}  # label -> (height, looks in a row there)
        self._raised: dict[str, datetime] = {}  # label -> the last look that had it raised

    def see(self, ts: datetime, tracks: list, frame_ref: str | None = None) -> bool:
        """One observation of the desk camera. True when it completed a pick-up."""
        people = [t.box for t in tracks if t.label == "person"]
        lifted = False
        for t in tracks:
            label = t.label.lower()
            if label not in DRINK_LABELS:
                continue
            self.container = (ts, label, frame_ref or "")
            x, y = (t.box.x1 + t.box.x2) / 2, (t.box.y1 + t.box.y2) / 2
            rest = self._rest.get(label)
            up = any(p.x1 <= x <= p.x2 and p.y1 <= y <= p.y1 + UPPER_SHARE * (p.y2 - p.y1) for p in people)
            if up and rest is not None and rest - y >= LIFT_RISE:
                last = self._raised.get(label)
                self._raised[label] = ts
                if last is not None and timedelta(0) < ts - last <= LIFT_WITHIN:
                    lifted = self.add(ts, frame_ref) or lifted
                continue
            at, n = self._still.get(label, (y, 0))
            at, n = (at, n + 1) if abs(y - at) < STILL else (y, 1)
            self._still[label] = (at, n)
            if n >= 3:
                self._rest[label] = at
        return lifted

    def caption(self, ts: datetime, text: str, frame_ref: str | None = None) -> bool:
        return bool(DRINK_VERBS.search(text or "")) and self.add(ts, frame_ref)

    def listed(self, ts: datetime, labels: list[str], frame_ref: str | None = None) -> None:
        """The desk zone's object list: a container on it is a container in view."""
        hit = next((label.lower() for label in labels if label.lower() in DRINK_LABELS), None)
        if hit and (self.container is None or ts >= self.container[0]):
            self.container = (ts, hit, frame_ref or "")

    def add(self, ts: datetime, frame_ref: str | None) -> bool:
        if self.moments and ts - self.moments[-1][0] < SAME_DRINK:
            return False
        self.moments.append((ts, frame_ref or ""))
        return True

    def last(self, now: datetime, since: datetime) -> tuple[datetime, str] | None:
        return next((m for m in reversed(self.moments) if since <= m[0] <= now), None)

    def prune(self, before: datetime) -> None:
        self.moments = [m for m in self.moments if m[0] >= before]


# ---------- everything the rule looks at, and what it finds ----------


@dataclass
class Reading:
    now: datetime
    tz: tzinfo
    day: date
    nights: dict[date, Night] = field(default_factory=dict)
    looks: list[Look] = field(default_factory=list)
    desk: list[Span] = field(default_factory=list)  # when a person was at the desk, lately
    drinks: DrinkWatch | None = None
    sessions: dict[date, list] = field(default_factory=dict)  # earlier days -> their measured blocks (PastSession)
    activity: str = "study"
    workplace: str = "desk"


def signals(r: Reading) -> list[CheckinSignal]:
    """What is there right now, in the order it is said. Empty fields mean the signal is absent, never zero."""
    out: list[CheckinSignal] = []
    night = r.nights.get(r.day)
    if night and short_night(r.nights, r.day):
        minutes = int(round(night.asleep_min / 30) * 30)
        out.append(CheckinSignal(kind="sleep", summary="Sleep: short", minutes=minutes, evidence_refs=[night.ref] if night.ref else [],
                                 detail=f"About {_hours(minutes)} asleep last night, at least an hour under your own usual night."))
    if night and resting_up(r.nights, r.day):
        out.append(CheckinSignal(kind="resting_hr", summary="Heart rate: up", evidence_refs=[night.ref] if night.ref else [],
                                 detail="Today's resting heart rate is above your own recent baseline."))
    strained = looked_strained(r.looks, r.now)
    if strained:
        out.append(CheckinSignal(kind="expression", summary=f"Looked strained {EXPR_WINDOW_MIN} min", minutes=EXPR_WINDOW_MIN,
                                 since=r.now - timedelta(minutes=EXPR_WINDOW_MIN), evidence_refs=[x.ref for x in strained[1] if x.ref][-8:],
                                 detail=f"Most of the looks the desk camera counted in the last {EXPR_WINDOW_MIN} minutes read as strained."))
    run = sitting(r.desk, r.now)
    if run:
        minutes = int((r.now - run[0]).total_seconds() // 60)
        if minutes >= SIT_MIN:
            shown = minutes // 5 * 5
            out.append(CheckinSignal(kind="sitting", summary=f"At the {r.workplace} {shown} min", minutes=shown, since=run[0],
                                     detail=f"At the {r.workplace} since {clock(hhmm(run[0], r.tz))} without a break of three minutes or more."))
        drink = no_drink(r.drinks, run[0], r.now, r.tz)
        if drink:
            out.append(drink)
    return out


def no_drink(drinks: DrinkWatch | None, sat_at: datetime, now: datetime, tz: tzinfo) -> CheckinSignal | None:
    if drinks is None or drinks.container is None or not timedelta(0) <= now - drinks.container[0] <= CONTAINER_WITHIN:
        return None  # no container in view: nothing can be said about drinks
    seen_at, label, frame = drinks.container
    last = drinks.last(now, sat_at)
    since = last[0] if last else sat_at
    if now - since < timedelta(minutes=DRINK_GAP_MIN):
        return None
    refs = [ref for ref in ((last[1] if last else ""), frame) if ref]
    if last:
        return CheckinSignal(kind="drink", summary=f"No drink since {clock(hhmm(since, tz))}", since=since, evidence_refs=refs,
                             detail=f"The last time the desk camera saw a drink picked up was {clock(hhmm(since, tz))}; "
                                    f"the {label} was in view at {clock(hhmm(seen_at, tz))}.")
    return CheckinSignal(kind="drink", summary="No drink this sitting", evidence_refs=refs,
                         detail=f"The desk camera has not seen a drink picked up since you sat down at {clock(hhmm(since, tz))}; "
                                f"the {label} was in view at {clock(hhmm(seen_at, tz))}.")


def kinds(found: list[CheckinSignal]) -> set[str]:
    return {s.kind for s in found}


def agree(found: list[CheckinSignal]) -> bool:
    """Two independent groups, at least one of them about right now. Never one signal, never the desk alone."""
    have = kinds(found)
    groups = sum(1 for k in BODY if k in have) + (1 if any(k in have for k in DESK) else 0)
    return groups >= 2 and any(k in have for k in PRESENT)


@dataclass(frozen=True)
class Gate:
    """What keeps a check-in from being said even when the signals agree."""

    awake: bool = True  # between the household's wake and bedtime
    at_desk: bool = True  # somebody is there to speak to
    fixed: str | None = None  # the fixed block that is running (a class)
    leave_by: datetime | None = None  # when to leave for the next fixed event
    event_start: datetime | None = None  # ... and when it starts
    last: datetime | None = None  # the last check-in


def held(now: datetime, gate: Gate) -> str | None:
    """Why nothing is said right now, in a few words; None when it may be."""
    # Either side of the last one: an event that arrives late, with an older time, is inside the cooldown too.
    if gate.last is not None and abs(now - gate.last) < timedelta(minutes=COOLDOWN_MIN):
        return "a check-in was raised less than ninety minutes ago"
    if not gate.awake:
        return "outside waking hours"
    if not gate.at_desk:
        return "nobody at the desk"
    if gate.fixed:
        return f"during {gate.fixed}"
    if gate.leave_by is not None and gate.leave_by - timedelta(minutes=LEAVE_GUARD_MIN) <= now < max(gate.leave_by, gate.event_start or gate.leave_by):
        return "close to a time to leave"
    return None


# ---------- earlier days that looked like this one ----------


@dataclass(frozen=True)
class PastSession:
    """One measured block of the main activity on an earlier day (from `day.coach.noted` / `day.reviewed`)."""

    start: datetime
    planned: int
    desk: int
    ref: str = ""  # the event it was read from


def fell_apart(sessions: list[PastSession]) -> PastSession | None:
    """The block from which the day fell apart: under FELL_SHARE of its planned minutes at the desk, and no later
    block did better. None when the day's last block held."""
    first = None
    for s in sorted(sessions, key=lambda s: s.start):
        if s.planned > 0 and s.desk < FELL_SHARE * s.planned:
            first = first or s
        else:
            first = None
    return first


def history(r: Reading, found: list[CheckinSignal]) -> tuple[CheckinHistory, CheckinSignal] | None:
    """"The last three times a day looked like this, your study blocks fell apart by four", or None.

    A past day looked like this one when every watch signal that is true today (a short night, resting heart rate
    up; at least one of them) was true of it too, each judged against the person's own baseline around that day.
    Only days with a measured block count. The claim is made when the HISTORY_MIN most recent such days all fell
    apart (`fell_apart`); "by" is the latest of those times, rounded up to the half hour. Anything less: nothing."""
    today = [k for k in WATCH if k in kinds(found)]
    if not today:
        return None
    test = {"sleep": short_night, "resting_hr": resting_up}
    alike = sorted((d for d in r.sessions if d < r.day and (r.day - d).days <= HISTORY_DAYS and r.sessions[d]
                    and all(test[k](r.nights, d) for k in today)), reverse=True)[:HISTORY_MIN]
    if len(alike) < HISTORY_MIN:
        return None
    falls = [fell_apart(r.sessions[d]) for d in alike]
    if not all(falls):
        return None
    latest = max(f.start.astimezone(r.tz).hour * 60 + f.start.astimezone(r.tz).minute for f in falls)
    latest = min(-(-latest // 30) * 30, 23 * 60 + 30)
    by = f"{latest // 60:02d}:{latest % 60:02d}"
    days = []
    for d, f in zip(alike, falls):
        later = [s for s in r.sessions[d] if s.start >= f.start]
        days.append(CheckinPastDay(day=d, text=f"{sum(s.desk for s in later)} of {sum(s.planned for s in later)} planned minutes at the "
                                                f"{r.workplace} from {clock(hhmm(f.start, r.tz))} on"))
    outcome = f"your {r.activity} blocks fell apart by {clock(by)}"
    refs = list(dict.fromkeys(f.ref for f in falls if f.ref))
    like = " and ".join({"sleep": "a short night", "resting_hr": "resting heart rate up"}[k] for k in today)
    signal = CheckinSignal(kind="history", summary=f"{len(alike)} days like this before", evidence_refs=refs,
                           detail=f"The last {len(alike)} days with {like}: each time, {outcome}.")
    return CheckinHistory(matches=len(alike), outcome=outcome, by=by, days=days), signal


# ---------- what to do about it ----------


def suggestions(found: list[CheckinSignal], move: str | None = None) -> list[CheckinSuggestion]:
    """Plain nudges, each tied to a signal that is there; `move` is the offer to move a block, when one can be made."""
    have = kinds(found)
    out = []
    if "drink" in have:
        out.append(CheckinSuggestion(kind="drink", text="Drink some water"))
    if "sitting" in have:
        out.append(CheckinSuggestion(kind="walk", text="Take ten minutes outside"))
    else:
        out.append(CheckinSuggestion(kind="break", text="Take ten minutes away from the desk"))
    if move:
        out.append(CheckinSuggestion(kind="move_block", text=move))
    return out


def after_meal(moves: list[Move], now: datetime, tz: tzinfo) -> tuple[list[Move], list[int], int] | None:
    """A re-plan's moves with today's blocks of the main activity that come before the next meal moved to straight
    after it: (the moves, where the moved blocks now are, where the meal is). None when there is nothing to move:
    no such block starting within MOVE_WITHIN_MIN, or no meal after it today."""
    today = now.astimezone(tz).date()

    def on_today(m: Move) -> bool:
        return m.start is not None and m.start.astimezone(tz).date() == today

    first = next((i for i, m in enumerate(moves) if m.kind == "study" and on_today(m)), None)
    if first is None or moves[first].start - now > timedelta(minutes=MOVE_WITHIN_MIN):
        return None
    meal = next((i for i, m in enumerate(moves) if i > first and m.kind == "eat" and on_today(m) and m.slot == moves[first].slot), None)
    if meal is None:
        return None
    early = [i for i in range(first, meal) if moves[i].kind == "study"]
    after = moves[meal].start + timedelta(minutes=(moves[meal].minutes or 30) + 1)
    moved = [moves[i].model_copy(update={"start": after if n == 0 else None}) for n, i in enumerate(early)]
    kept = [m for i, m in enumerate(moves[:meal + 1]) if i not in early]
    out = kept + moved + moves[meal + 1:]
    return out, list(range(len(kept), len(kept) + len(moved))), len(kept) - 1


def moved_after(blocks: list[PlanMove], now: datetime, tz: tzinfo) -> str | None:
    """The meal ("dinner") every one of today's remaining blocks of the main activity now comes after, in a
    re-plan's blocks; None when one of them still starts before it. The offer is only ever made from this."""
    today = now.astimezone(tz).date()
    rest = sorted((b for b in blocks if b.start.astimezone(tz).date() == today and b.end > now), key=lambda b: b.start)
    study = [b for b in rest if b.kind == "study" and not b.fixed]
    meals = [b for b in rest if b.kind == "eat"]
    if not study or not meals:
        return None
    meal = max((m for m in meals if m.end <= study[0].start), key=lambda m: m.start, default=None)
    return meal_name(hhmm(meal.start, tz)) if meal else None


def move_text(meal: str, n: int, activity: str) -> str:
    return f"Move the {activity} block to after {meal}" if n == 1 else f"Move your {activity} blocks to after {meal}"


# ---------- the facts the narrator says (contexts/day/domain/narration.py renders them) ----------


def facts(signals_: list[CheckinSignal], suggestions_: list[CheckinSuggestion], past: CheckinHistory | None, tz: tzinfo,
          callback=None) -> dict:
    """The check-in as the narrator's facts: only what is there. No heart number is ever among them. `callback`
    (contracts.day.CheckinCallback): an earlier look this one is like, from the look memory, and what followed it."""
    f: dict = {"moment": "checkin"}
    for s in signals_:
        if s.kind == "sleep" and s.minutes:
            f["sleep_hours"] = round(s.minutes / 60, 1)
        elif s.kind == "resting_hr":
            f["resting_hr"] = "up"
        elif s.kind == "expression" and s.minutes:
            f["strained_min"] = s.minutes
        elif s.kind == "sitting" and s.minutes:
            f["sitting_min"] = s.minutes
        elif s.kind == "drink":
            f["drink"] = {"since": hhmm(s.since, tz)} if s.since else {"this_sitting": True}
    if past:
        f["history"] = {"matches": past.matches, "outcome": past.outcome} | ({"by": past.by} if past.by else {})
    if callback is not None and callback.said:
        f["callback"] = {"said": callback.said} | ({"learned": callback.learned} if callback.learned else {})
    if suggestions_:
        f["suggest"] = [{"kind": s.kind, "text": s.text} for s in suggestions_]
    return f


def diagnosis_in(text: str) -> list[str]:
    """The diagnosis words in `text`: there must be none in anything a check-in says."""
    return [w for w in re.findall(r"[a-z]+", (text or "").lower()) if w in DIAGNOSIS]


def _hours(minutes: int) -> str:
    whole, half = divmod(minutes, 60)
    return f"{whole}{'.5' if half else ''} h"
