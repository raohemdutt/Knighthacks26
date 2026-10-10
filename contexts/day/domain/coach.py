"""The study coach's arithmetic (D2): one planned block against what the desk camera saw. Pure functions.

    spans   = presence.intervals(block.start, block.end)       # when a person was in the desk zone
    session = review_session(block, spans, signals, desk_place, words, tz)
    advice  = suggest(sessions, words)

The rules, in plain words (every time is event time; nothing here reads a clock or calls a model):

- At the desk: a person track is in the desk zone from its `enter` to its `exit` (the zone engine exits a
  track 3 s after it was last seen). Several people count once. An absence shorter than MIN_BREAK_S is not
  an absence: reaching for a bag or a detector blink does not split a stretch.
- A break: an absence of MIN_BREAK_S or more *between two stretches* at the desk inside the block. The
  time before the first sit-down is the late start, and the time after the last stretch is "ended early":
  neither is a break.
- The trigger of a break: what the box recorded in the TRIGGER_WINDOW_S before the desk exit. Every such
  signal is kept as evidence; the trigger is the first kind present in this order: a chat message
  arriving, the phone coming into view, someone entering through the door, a drink or food showing up.
  With no signal the trigger is "none": no visible reason. It is the most likely reason, not a proven one.
- Not measured: a block planned somewhere else than the box (a library) in which the desk camera saw
  nobody for AWAY_MIN_S or more. Its numbers are None, never zero. A block planned at the desk with
  nobody seen is measured as 0 minutes, and the text says the camera may have been off.
- What nobody can know from here: the camera sees the desk, not the screen, and it does not know who sits there.
"""

import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, tzinfo

from contracts.day import CoachBreak, CoachEvidence, PlanMove, SessionReview

MIN_BREAK_S = 60  # an absence shorter than this does not split a stretch
TRIGGER_WINDOW_S = 120  # "the minute or two before" a break
AWAY_MIN_S = 300  # desk time that makes a block planned elsewhere count as done at the desk
LATE_START_MIN = 5  # a first sit-down this late is worth a sentence
MIN_BLOCK, MAX_BLOCK, BLOCK_STEP = 25, 120, 5  # suggested block lengths, minutes
PRIORITY = ("chat", "phone", "door", "food")
LABELS = {"chat": "a message in the group chat", "phone": "the phone was in view",
          "door": "someone came in through the door", "food": "a drink or food showed up"}
AFTER = {"chat": "right after a message arrived in the group chat", "phone": "right after the phone came into view",
         "door": "right after someone came in through the door", "food": "right after a drink or food showed up",
         "none": "with no visible reason"}
SHORT = {"chat": "right after the group chat", "phone": "right after the phone came out",
         "door": "right after someone came in", "food": "right after a drink or snack showed up"}
PHONE_WORDS = ("phone", "smartphone", "cellphone", "cell phone", "mobile")
FOOD_WORDS = ("energy drink", "drink", "drinks", "coffee", "soda", "tea", "eat", "eats", "eating", "snack", "pizza",
              "bagel", "bagels", "burrito", "sandwich", "food", "meal", "lunch", "dinner", "breakfast")
FOOD_LABELS = ("can", "cup", "bottle", "mug", "pizza", "pizza box", "takeout bowl", "food container", "snack", "plate")
LIMITS = ("The camera sees the {workplace}, not the screen: minutes at the {workplace} are not minutes of focus.",
          "It does not know who is at the {workplace}.",
          "A trigger is what happened in the two minutes before a break: the most likely reason, not a proven one.",
          "A block away from the box is not measured.")

Span = tuple[datetime, datetime]


@dataclass(frozen=True)
class Signal:
    """Something the box recorded that can pull a person away from the desk."""

    kind: str  # chat, phone, door, food
    at: datetime
    id: str  # connector item ("groupme:friends:17601"), or a Memory episode id


@dataclass(frozen=True)
class Words:
    """The profile's words (contexts/profiles Vocabulary): "study" and "desk" for a student."""

    activity: str = "study"
    workplace: str = "desk"


@dataclass(frozen=True)
class Advice:
    suggestions: list[str]
    block_minutes: int | None  # the block length to plan next, when the day showed shorter stretches than planned
    longest_stretch: int | None


def mentions(text: str, words: tuple[str, ...]) -> bool:
    """Whole words only: "phone" is not in "headphones"."""
    return any(re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", text.lower()) for w in words)


class DeskPresence:
    """When a person was in the desk zone, from its zone events in time order. No I/O: a projection of the log."""

    def __init__(self) -> None:
        self.open: dict[object, tuple[datetime, datetime]] = {}  # track -> (entered, last seen)
        self.closed: list[Span] = []

    def add(self, ts: datetime, track: object, kind: str, dwell_s: float | None = None) -> None:
        if kind == "exit":
            stay = self.open.pop(track, None)
            if stay:
                self.closed.append((stay[0], max(ts, stay[0])))
        elif track in self.open:
            self.open[track] = (self.open[track][0], max(ts, self.open[track][1]))
        else:  # an enter, or a dwell whose enter was before the log this box has
            self.open[track] = (ts - timedelta(seconds=dwell_s or 0) if kind == "dwell" else ts, ts)

    def intervals(self, start: datetime, end: datetime) -> list[Span]:
        """The stays that touch [start, end), clipped to it, overlapping ones joined. A stay still open ends when it was last seen."""
        spans = sorted((max(a, start), min(b, end)) for a, b in [*self.closed, *self.open.values()] if a < end and b > start)
        out: list[Span] = []
        for a, b in spans:
            if out and a <= out[-1][1]:
                out[-1] = (out[-1][0], max(out[-1][1], b))
            else:
                out.append((a, b))
        return out

    def prune(self, before: datetime) -> None:
        self.closed = [s for s in self.closed if s[1] >= before]


def stretches(spans: list[Span], min_break_s: int = MIN_BREAK_S) -> list[Span]:
    """Presence with the short absences closed up: what is left between two stretches is a break."""
    out: list[Span] = []
    for a, b in sorted(spans):
        if out and (a - out[-1][1]).total_seconds() < min_break_s:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _minutes(a: datetime, b: datetime) -> int:
    return int(round((b - a).total_seconds() / 60))


def _mins(n: int) -> str:
    return f"{n} minute{'' if n == 1 else 's'}"


def hm(t: datetime, tz: tzinfo | None = None) -> str:
    t = t.astimezone(tz) if tz else t
    return f"{t.hour % 12 or 12}:{t:%M} {t:%p}"


def trigger_of(at: datetime, signals: list[Signal], window_s: int = TRIGGER_WINDOW_S) -> tuple[str, list[CoachEvidence]]:
    """The most likely reason for leaving the desk at `at`, and everything recorded just before it."""
    near = sorted((s for s in signals if timedelta(0) <= at - s.at <= timedelta(seconds=window_s)), key=lambda s: (s.at, s.kind))
    seen: set[tuple[str, str]] = set()
    evidence = []
    for s in near:
        if (s.kind, s.id) not in seen:
            seen.add((s.kind, s.id))
            evidence.append(CoachEvidence(kind=s.kind, id=s.id, at=s.at, label=LABELS[s.kind]))
    kinds = {e.kind for e in evidence}
    return next((k for k in PRIORITY if k in kinds), "none"), evidence


def review_session(block: PlanMove, at_desk: list[Span], signals: list[Signal], desk_place: str | None,
                   words: Words = Words(), tz: tzinfo | None = None) -> SessionReview:
    """One planned block against the desk camera. `at_desk` is DeskPresence.intervals for the block."""
    base = dict(block_id=block.id, title=block.title, start=block.start, end=block.end, place=block.place,
                place_name=block.place_name, planned_minutes=block.minutes)
    spans = stretches([(max(a, block.start), min(b, block.end)) for a, b in at_desk if a < block.end and b > block.start])
    seconds = sum((b - a).total_seconds() for a, b in spans)
    at_the_box = desk_place is not None and block.place == desk_place
    if not at_the_box and seconds < AWAY_MIN_S:
        where = block.place_name or "somewhere else"
        why = f"planned at {where}, away from the box"
        return SessionReview(**base, measured=False, not_measured=why,
                             text=f"{block.title}, {hm(block.start, tz)} to {hm(block.end, tz)}: not measured ({why}).")
    if not spans:
        return SessionReview(**base, measured=True, desk_minutes=0, breaks=[],
                             text=(f"{block.title}, {hm(block.start, tz)} to {hm(block.end, tz)}: you planned {_mins(block.minutes)} and "
                                   f"nobody was seen at the {words.workplace} (or the camera was off)."))
    breaks = []
    for (_, left), (back, _) in zip(spans, spans[1:], strict=False):
        trigger, evidence = trigger_of(left, signals)
        breaks.append(CoachBreak(start=left, end=back, minutes=max(1, _minutes(left, back)), trigger=trigger, evidence=evidence))
    longest = max(spans, key=lambda s: (s[1] - s[0], -s[0].timestamp()))
    s = SessionReview(**base, measured=True, desk_minutes=int(round(seconds / 60)),
                      late_start_minutes=_minutes(block.start, spans[0][0]), ended_early_minutes=_minutes(spans[-1][1], block.end),
                      longest_stretch_minutes=_minutes(*longest), longest_stretch_start=longest[0], breaks=breaks)
    s.text = session_text(s, words, tz, moved=not at_the_box)
    return s


def moments(s: SessionReview) -> list[tuple[int, datetime, str, CoachBreak | None]]:
    """What cost the block minutes, the most expensive first: (minutes, when, kind, the break)."""
    out: list[tuple[int, datetime, str, CoachBreak | None]] = [(b.minutes, b.start, "break", b) for b in s.breaks]
    if (s.late_start_minutes or 0) >= LATE_START_MIN:
        out.append((s.late_start_minutes, s.start, "late", None))
    if (s.ended_early_minutes or 0) >= LATE_START_MIN:
        out.append((s.ended_early_minutes, s.end - timedelta(minutes=s.ended_early_minutes), "early", None))
    return sorted(out, key=lambda m: (-m[0], m[1]))


def _reason(b: CoachBreak) -> str:
    also = sorted({e.kind for e in b.evidence} - {b.trigger}, key=PRIORITY.index)
    extra = {"phone": "the phone was in view", "door": "someone came in", "food": "a drink or food showed up", "chat": "a message arrived"}
    return AFTER[b.trigger] + (f" (and {', '.join(extra[k] for k in also)})" if also else "")


def session_text(s: SessionReview, words: Words = Words(), tz: tzinfo | None = None, moved: bool = False) -> str:
    """Planned against done, the two or three moments that cost the most, then what worked. Kind and exact."""
    w = words.workplace
    parts = [f"{s.title}, {hm(s.start, tz)} to {hm(s.end, tz)}: you planned {_mins(s.planned_minutes)} and were at the {w} for {s.desk_minutes}."]
    if moved:
        parts.append(f"It was planned at {s.place_name or 'another place'}; the camera saw the {w} in use, so it is measured there.")
    for minutes, when, kind, b in sorted(moments(s)[:3], key=lambda m: m[1]):
        if kind == "late":
            parts.append(f"You sat down {_mins(minutes)} after the block started.")
        elif kind == "early":
            parts.append(f"You left the {w} {_mins(minutes)} before the end.")
        else:
            parts.append(f"At {hm(when, tz)} you were away for {_mins(minutes)}, {_reason(b)}.")
    if not s.breaks and s.desk_minutes >= 0.9 * s.planned_minutes:
        parts.append("What worked: you stayed for the whole block.")
    elif s.longest_stretch_minutes:
        first = s.longest_stretch_start is not None and (s.longest_stretch_start - s.start) < timedelta(seconds=MIN_BREAK_S)
        where = f"the first {_mins(s.longest_stretch_minutes)}" if first else f"{_mins(s.longest_stretch_minutes)}, from {hm(s.longest_stretch_start, tz)}"
        parts.append(f"What worked: your longest unbroken stretch was {where}.")
    return " ".join(parts)


NUMBER = ("No", "One", "Two", "Three", "Four", "Five")


def note_text(s: SessionReview, next_block: PlanMove | None = None, words: Words = Words()) -> str:
    """One sentence or three short ones, for the phone or a voice."""
    parts = [f"You did {s.desk_minutes} of {_mins(s.planned_minutes)}."]
    n = len(s.breaks)
    if not s.desk_minutes:  # said as what the camera saw, not as what the person did
        parts = [f"I did not see you at the {words.workplace} for the {_mins(s.planned_minutes)[:-1] if s.planned_minutes != 1 else '1 minute'} "
                 f"{words.activity} block. If you worked somewhere else, ignore me."]
    elif n == 0:
        late = s.late_start_minutes or 0
        parts.append(f"No breaks; you sat down {_mins(late)} in." if late >= LATE_START_MIN else "No breaks.")
    else:
        count = f"{NUMBER[n] if n < len(NUMBER) else n} break{'' if n == 1 else 's'}"
        long = max(s.breaks, key=lambda b: (b.minutes, -b.start.timestamp()))
        why = SHORT.get(long.trigger)
        if n == 1:
            parts.append(f"{count}, {why}." if why else f"{count}, for {_mins(long.minutes)}.")
        else:
            parts.append(f"{count}; the long one came {why}." if why else f"{count}; nothing I can see explains the long one.")
    if next_block is not None:
        parts.append(f"Next {words.activity} block is {_mins(next_block.minutes)}.")
    return " ".join(parts)


def block_size(longest: int) -> int:
    """A block the length of the longest unbroken stretch, rounded up to five minutes, inside sane limits."""
    return max(MIN_BLOCK, min(MAX_BLOCK, math.ceil(longest / BLOCK_STEP) * BLOCK_STEP))


def suggest(sessions: list[SessionReview], words: Words = Words()) -> Advice:
    """One or two suggestions drawn from the pattern of the measured sessions, never more."""
    measured = [s for s in sessions if s.measured and s.desk_minutes]
    if not measured:
        return Advice([], None, None)
    w = words.workplace
    breaks = [b for s in measured for b in s.breaks]
    lost: dict[str, int] = {}
    for b in breaks:
        if b.trigger != "none":
            lost[b.trigger] = lost.get(b.trigger, 0) + b.minutes
    out: list[str] = []
    if lost:
        top = max(lost, key=lambda k: (lost[k], -PRIORITY.index(k)))
        n = sum(1 for b in breaks if b.trigger == top)
        tail = f"{n} of your {len(breaks)} break{'' if len(breaks) == 1 else 's'} ({_mins(lost[top])}) began {SHORT[top]}."
        out.append({"chat": f"Hold chat notifications until the break: {tail}",
                    "phone": f"Keep the phone face down or outside the {w} zone during a block: {tail}",
                    "door": f"Let people know when a block is on, or close the door: {tail}",
                    "food": f"Bring the drink or snack to the {w} before the block starts: {tail}"}[top])
    longest = max(s.longest_stretch_minutes or 0 for s in measured)
    cut_short = [s for s in measured if s.breaks or (s.ended_early_minutes or 0) >= LATE_START_MIN]
    size = None
    if cut_short and longest and block_size(longest) < max(s.planned_minutes for s in cut_short):
        size = block_size(longest)
        floor = " (the shortest I plan)" if size > longest + BLOCK_STEP else ""
        out.append(f"Plan shorter blocks: your longest unbroken stretch was {_mins(longest)}, so I'm planning {size}-minute blocks{floor}.")
    late = max(s.late_start_minutes or 0 for s in measured)
    if late >= 2 * LATE_START_MIN:
        out.append(f"Start with the hardest thing the moment the block opens: the first sit-down came {_mins(late)} in.")
    if not out:
        out.append("Keep this shape: nothing pulled you away for long.")
    return Advice(out[:2], size, longest or None)


def day_text(sessions: list[SessionReview], words: Words = Words(), habits_desk_minutes: int | None = None,
             tz: tzinfo | None = None) -> str:
    """The day's blocks of the main activity in a few plain sentences."""
    if not sessions:
        return ""
    a, w = words.activity, words.workplace
    measured = [s for s in sessions if s.measured]
    away = [s for s in sessions if not s.measured]
    parts = []
    if measured:
        planned, done = sum(s.planned_minutes for s in measured), sum(s.desk_minutes or 0 for s in measured)
        n = len(measured)
        parts.append(f"You planned {_mins(planned)} of {a} in {n} block{'' if n == 1 else 's'} the camera could see, and were at the {w} for {done}.")
        costly = sorted(((m, s) for s in measured for m in moments(s)), key=lambda x: (-x[0][0], x[0][1]))[:3]
        for (minutes, when, kind, b), s in sorted(costly, key=lambda x: x[0][1]):
            if kind == "late":
                parts.append(f"{s.title} started {_mins(minutes)} late.")
            elif kind == "early":
                parts.append(f"{s.title} stopped {_mins(minutes)} early.")
            else:
                parts.append(f"{_mins(minutes).capitalize()} away from {hm(when, tz)}, {_reason(b)}.")
        best = max(measured, key=lambda s: s.longest_stretch_minutes or 0)
        if best.longest_stretch_minutes:
            parts.append(f"What worked: an unbroken {_mins(best.longest_stretch_minutes)} in {best.title}.")
    if away:
        parts.append(f"Not measured: {', '.join(s.title for s in away)} (away from the box).")
    if habits_desk_minutes is not None:
        parts.append(f"Habits counted {_mins(habits_desk_minutes)} at the {w} over the whole day.")
    return " ".join(parts)
