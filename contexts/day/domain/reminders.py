"""When a planned block starts, Atlas says so once: which block, and when (pure: no clock, bus or I/O).

The rule (contexts/day/application/reminders.py feeds it, the arbiter decides whether the line goes out):

- Which blocks: the chosen (non-fixed) blocks of the committed plan, and accepted optional events (origin
  "optional") in any plan. Fixed events have their own leave-now line; so does an accepted optional event with a
  walk of 3 minutes or more, which therefore gets no start line here.
- When: at the block's start, or at `start - walk` when the person is (or the plan has them) somewhere else and
  the walk is 3 minutes or more. Said while `trigger <= now < trigger + 5 min`.
- A lead time that falls inside a fixed event is not lost and is never said during the event: the reminder waits
  for the event to end when that is before the block starts, and for the block's start otherwise. Unless the
  cameras saw the person since the event began, the plan has them at the event, and the walk is counted from there.
- Already there: seen at the block's place, the line ends "You're at the desk already." Continuing (the block
  before ended within 5 minutes, same kind, same place, and the person is there): nothing is said.
- Batching: consecutive blocks of one kind, each 30 minutes or less, with rests of 30 minutes or less between
  them, are one line ("three short study blocks from two"); every block of the run, rests included, counts as said.
- Quiet: 10 minutes between two reminders; nothing outside the wake/bed window; nothing while a fixed block runs.
- Once: `ref = "<plan_id>:<block ids joined by +>@<HH:MM>"`. A block id or a local start time already said is not
  said again, so a re-plan landing mid-window does not announce the same start twice.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, tzinfo

from contracts.day import PlanMove

WINDOW = timedelta(minutes=5)
WALK_MIN = 3
SHORT_MIN = 30
CONTINUING = timedelta(minutes=5)
GAP = timedelta(minutes=10)
ORIGIN = "optional"


@dataclass
class Said:
    """What was already said, rebuilt from `day.narrated` moment "start" (see `parse`)."""

    blocks: set[str]  # "<plan_id>:<block id>"
    starts: set[str]  # "<day>@<HH:MM>"
    last: datetime | None = None

    def note(self, ref: str, ts: datetime, day) -> None:
        parsed = parse(ref)
        if parsed is None:
            return
        plan_id, ids, hm = parsed
        self.blocks.update(f"{plan_id}:{i}" for i in ids)
        self.starts.add(f"{day}@{hm}")
        if self.last is None or ts > self.last:
            self.last = ts


@dataclass(frozen=True)
class Due:
    """A start to announce: the first block of the run, the run's blocks (rests included), and the facts' raw parts."""

    first: PlanMove
    run: tuple[PlanMove, ...]
    count: int  # blocks of the first's kind in the run
    until: datetime  # the end of the run's last block of that kind
    walk: int  # minutes, 0 under WALK_MIN
    there: bool
    ref: str


def ref_of(plan_id: str, run, at: datetime, tz: tzinfo) -> str:
    return f"{plan_id}:{'+'.join(m.id for m in run)}@{at.astimezone(tz).strftime('%H:%M')}"


def parse(ref: str | None) -> tuple[str, list[str], str] | None:
    """`ref_of` read back: (plan id, block ids, "HH:MM"), or None for a ref of another shape."""
    if not ref or "@" not in ref or ":" not in ref:
        return None
    head, hm = ref.rsplit("@", 1)
    plan_id, _, ids = head.partition(":")
    return (plan_id, ids.split("+"), hm) if ids else None


def eligible(m: PlanMove, committed: bool) -> bool:
    return m.origin == ORIGIN or (committed and not m.fixed)


def run_from(moves: list[PlanMove], i: int) -> tuple[PlanMove, ...]:
    """The batch that starts at moves[i]: itself, or the short blocks of its kind with short rests between them."""
    first = moves[i]
    if first.fixed or first.kind == "rest" or first.minutes > SHORT_MIN:
        return (first,)
    run, pending = [first], []
    for m in moves[i + 1:]:
        if m.fixed:
            break
        if m.kind == "rest" and m.minutes <= SHORT_MIN:
            pending.append(m)
            continue
        if m.kind == first.kind and m.minutes <= SHORT_MIN and pending:
            run += pending + [m]
            pending = []
            continue
        if m.kind == first.kind and m.minutes <= SHORT_MIN and m.start <= run[-1].end + timedelta(minutes=1):
            run.append(m)
            continue
        break
    return tuple(run)


def fixed_at(blocks: list[PlanMove], t: datetime, but: PlanMove) -> PlanMove | None:
    """The fixed block that is running at `t`, other than `but` itself."""
    return next((b for b in blocks if b.fixed and b is not but and b.start <= t < b.end), None)


def due(moves: list[PlanMove], plan_id: str, committed: bool, now: datetime, seen: str | None, walk, said: Said,
        tz: tzinfo, awake: bool = True, seen_at: datetime | None = None) -> Due | None:
    """The start to say at `now`, or None. `seen` is the place the person was last seen ("pack.place") and `seen_at`
    when (None: not known), `walk(a, b)` the walk in minutes."""
    if not awake or (said.last is not None and timedelta(0) <= now - said.last < GAP):
        return None
    blocks = sorted(moves, key=lambda m: m.start)
    if any(m.fixed and m.start <= now < m.end and not (m.origin == ORIGIN and m.start == now) for m in blocks):
        return None  # a fixed event is running (an accepted optional event starting this minute is the one to say)
    for i, m in enumerate(blocks):
        if not eligible(m, committed) or f"{plan_id}:{m.id}" in said.blocks:
            continue
        if f"{m.start.astimezone(tz).date()}@{m.start.astimezone(tz).strftime('%H:%M')}" in said.starts:
            continue
        before = [b for b in blocks[:i] if b.place and b.end <= m.start]
        origin = seen or (before[-1].place if before else None)
        minutes = walk(origin, m.place) if origin and m.place and origin != m.place else 0
        minutes = minutes if minutes >= WALK_MIN else 0
        trigger = m.start - timedelta(minutes=minutes)
        event = fixed_at(blocks, trigger, m)
        away = False
        if event is not None:  # the lead time falls inside a fixed event: not during it, and not lost
            away = bool(event.place) and not (seen_at is not None and seen_at >= event.start)
            if away and m.place:  # not seen since it began: the plan has the person there, so the walk is from there
                minutes = walk(event.place, m.place) if event.place != m.place else 0
                minutes = minutes if minutes >= WALK_MIN else 0
            trigger = max(m.start - timedelta(minutes=minutes), event.end)
            while trigger < m.start and (after := fixed_at(blocks, trigger, m)) is not None:
                trigger = after.end  # another fixed event straight behind it
            trigger = min(trigger, m.start)  # an event that runs up to the block: at the block's start
        if m.origin == ORIGIN and minutes:
            continue  # the tracker's leave-now line is its reminder
        if not trigger <= now < trigger + WINDOW:
            continue
        there = bool(seen) and seen == m.place and not away
        prev = blocks[i - 1] if i else None
        if there and prev is not None and prev.kind == m.kind and prev.place == m.place and m.start - prev.end <= CONTINUING:
            continue  # carrying on with the same thing: nothing to say
        run = run_from(blocks, i)
        same = [b for b in run if b.kind == m.kind]
        return Due(first=m, run=run, count=len(same), until=same[-1].end, walk=minutes, there=there,
                   ref=ref_of(plan_id, run, m.start, tz))
    return None
