"""Optional events ("Near you"): what one is, whether it fits the plan, and when Atlas may bring it up.

    offer = parse(item, locate, name, tz)                  # an Offer from a `nearby:` calendar item, or None
    f = fit(offer, plan.moves, now, walk, seen_place)      # Fit: yes / nearly / no, what it displaces, the walk, leave-by
    why_not = suggest_held(now, offer, f, state, gate)     # None when Atlas may say "want me to add it?"

An optional event is a calendar item whose `external_id` starts with `nearby:` (contexts/demo/nearby.py). It is
never a commitment by itself: Day Run's board, the leaving checklist, the Circle's free/busy answer and Habits
skip that prefix. A yes makes it one through the plan: the day's re-plan carries it as a fixed block whose id
starts with `optional:` (BLOCK_PREFIX) and whose `origin` is "optional", and it counts once that plan's approval
card is approved.

Everything here is pure: no clock, no memory, no I/O. Friends' names stay in the Offer (`going`, `host`): they are
said in the suggestion on the owner's own phone and are never copied into a plan block.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, tzinfo

from contracts.day import PlanMove

PREFIX = "nearby:"  # the mark of an optional event on a calendar item
BLOCK_PREFIX = "optional:"  # ... and of the plan block an accepted one becomes
ORIGIN = "optional"
UNTIL = re.compile(r"until\s+(\d{1,2}):(\d{2})\s*([AP]M)", re.I)
DEFAULT_MINUTES = 60

# The suggestion ("Want me to add it?"): config/agent.yaml `day:` keys in brackets.
SUGGEST_FROM_MIN = 90  # [suggest_from_min] not earlier than this before the event starts
SUGGEST_UNTIL_MIN = 30  # [suggest_until_min] ... and not later than this before it
SUGGEST_COOLDOWN_MIN = 45  # [suggest_cooldown_min] between two suggestions
LEAVE_GUARD_MIN = 30  # not this close to a time to leave for a fixed event
WALK_MIN = 3  # a walk shorter than this is not worth a leave-by
EARLY_MAX_MIN = 15  # "nearly fits" may mean leaving this much early for the next fixed event; more is "no"

State = str  # suggested | accepted | declined | passed


@dataclass(frozen=True)
class Offer:
    """One optional event as its calendar item describes it."""

    id: str  # the item's external id: "nearby:<slug>:<start ISO>"
    title: str
    start: datetime
    end: datetime
    place: str | None = None  # "pack.place" when the place pack knows it
    place_name: str | None = None
    tags: tuple[str, ...] = ()  # the summary's tags as written: "with Riley, Sam", "free", "optional", "via GroupMe"
    going: tuple[str, ...] = ()  # the friends a "with ..." tag names
    host: str | None = None  # "hosted by ..."
    source: str | None = None  # "GroupMe", "UCF Events"

    @property
    def slug(self) -> str:
        return self.id[len(PREFIX):].split(":", 1)[0]

    @property
    def minutes(self) -> int:
        return round((self.end - self.start).total_seconds() / 60)

    @property
    def people(self) -> tuple[str, ...]:
        """Everybody the item names: the host first."""
        return ((self.host,) if self.host else ()) + tuple(n for n in self.going if n != self.host)


def block_id(offer_id: str) -> str:
    """The id of the plan block an accepted optional event becomes: never one the `nearby:` guards skip."""
    return BLOCK_PREFIX + offer_id[len(PREFIX):]


def offer_id(block: str) -> str | None:
    return PREFIX + block[len(BLOCK_PREFIX):] if block.startswith(BLOCK_PREFIX) else None


def names(text: str) -> tuple[str, ...]:
    return tuple(n.strip() for n in re.split(r",|\band\b", text) if n.strip())


def parse(item, locate: Callable[[str], str | None], name: Callable[[str], str], tz: tzinfo) -> Offer | None:
    """An Offer from a calendar item (`external_id`, `title`, `when`, `summary` "until <end> · <place> · <tag> · ...");
    None for anything that is not an optional event with a start."""
    if not item.external_id.startswith(PREFIX) or item.when is None:
        return None
    start = item.when.astimezone(tz) if item.when.tzinfo else item.when.replace(tzinfo=tz)
    parts = [p.strip() for p in (item.summary or "").split("·") if p.strip()]
    end = start + timedelta(minutes=DEFAULT_MINUTES)
    place = place_name = host = source = None
    going: tuple[str, ...] = ()
    tags: list[str] = []
    for part in parts:
        if m := UNTIL.search(part):
            end = start.replace(hour=int(m[1]) % 12 + (12 if m[3].upper() == "PM" else 0), minute=int(m[2]))
            if end <= start:
                end += timedelta(days=1)
            continue
        low = part.lower()
        if low.startswith("with "):
            going = names(part[5:])
        elif low.startswith("hosted by "):
            host = part[10:].strip() or None
        elif low.startswith("via "):
            source = part[4:].strip() or None
        elif place_name is None and not tags and low not in ("free", "optional"):
            place_name, place = part, locate(part)  # the first thing after the end that is not a tag is where it is
            continue
        tags.append(part)
    if place:
        place_name = re.sub(r"\s*\([^)]*\)", "", name(place)).strip() or place_name
    return Offer(id=item.external_id, title=item.title, start=start, end=end, place=place, place_name=place_name, tags=tuple(tags),
                 going=going, host=host, source=source)


# ---------- does it fit the plan? ----------


@dataclass(frozen=True)
class Touched:
    """A block of the plan the event would run into."""

    id: str
    title: str
    kind: str
    start: datetime
    end: datetime
    fixed: bool
    overlap_minutes: int


@dataclass(frozen=True)
class Fit:
    fits: str  # "yes": nothing in the way; "nearly": only suggested blocks move; "no": a fixed event is in the way
    walk_minutes: int = 0  # from where the plan (or the cameras) has the person just before it
    leave_by: datetime | None = None
    from_place: str | None = None
    displaces: tuple[Touched, ...] = ()  # suggested blocks that would shift or shrink
    conflicts: tuple[Touched, ...] = ()  # fixed events it overlaps: these never move
    leave_early_minutes: int = 0  # to make the next fixed event on foot
    before: PlanMove | None = None  # the plan's next block after it, for "it fits before your study block"
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.fits != "no"


def _minutes(a: datetime, b: datetime) -> int:
    return max(0, round((b - a).total_seconds() / 60))


def fit(offer: Offer, moves: list[PlanMove], now: datetime, walk: Callable[[str | None, str | None], int],
        seen_place: str | None = None) -> Fit:
    """`offer` against the plan's blocks. A fixed block never moves: overlapping one is a "no". Suggested blocks
    (the ones Atlas chose) may shift or shrink: that is "nearly". The walk is from the last place the plan has the
    person before the event, or from where the cameras saw them when nothing is planned in between."""
    mine = block_id(offer.id)
    blocks = sorted((m for m in moves if m.id != mine and m.end > now), key=lambda m: m.start)
    earlier = [m for m in blocks if m.place and m.start < offer.start]
    origin = earlier[-1].place if earlier else seen_place
    there = walk(origin, offer.place) if origin and offer.place else 0
    depart = offer.start - timedelta(minutes=there)
    touched = [Touched(m.id, m.title, m.kind, m.start, m.end, m.fixed, _minutes(max(m.start, depart), min(m.end, offer.end)))
               for m in blocks if m.start < offer.end and m.end > depart]
    conflicts = tuple(t for t in touched if t.fixed and t.start < offer.end and t.end > offer.start)
    displaces = tuple(t for t in touched if not t.fixed)
    after = next((m for m in blocks if m.start >= offer.start and m.id not in {t.id for t in conflicts}), None)
    nxt = next((m for m in blocks if m.fixed and m.place and m.start >= offer.end), None)
    early = 0
    if nxt is not None and offer.place:
        early = max(0, _minutes(nxt.start, offer.end + timedelta(minutes=walk(offer.place, nxt.place))))
    leave_by = depart if there >= WALK_MIN else None
    base = {"walk_minutes": there, "leave_by": leave_by, "from_place": origin, "displaces": displaces, "conflicts": conflicts,
            "leave_early_minutes": early, "before": after}
    if conflicts:
        return Fit("no", reason=f"it overlaps {conflicts[0].title}, which is fixed", **base)
    if depart < now:
        return Fit("no", reason="it is too late to walk there", **base)
    if early > min(EARLY_MAX_MIN, offer.minutes // 2):
        return Fit("no", reason=f"you would have to leave {early} minutes early for {nxt.title}", **base)
    if displaces or early:
        what = displaces[0].title if displaces else nxt.title
        return Fit("nearly", reason=f"{what} would move" if displaces else f"you would leave {early} minutes early for {what}", **base)
    return Fit("yes", **base)


# ---------- when Atlas may bring it up ----------


@dataclass(frozen=True)
class Gate:
    """What keeps a suggestion from being said."""

    awake: bool = True  # between the household's wake and bedtime
    fixed: str | None = None  # the fixed block that is running (a class)
    leave_by: datetime | None = None  # when to leave for the next fixed event
    event_start: datetime | None = None  # ... and when it starts
    last: datetime | None = None  # the last suggestion, of any event
    said: bool = False  # this event was already suggested
    enabled: bool = True
    from_min: int = SUGGEST_FROM_MIN
    until_min: int = SUGGEST_UNTIL_MIN
    cooldown_min: int = SUGGEST_COOLDOWN_MIN


def state_of(offer: Offer, now: datetime, answer: str | None, accepted: bool) -> State:
    """accepted: said yes and its plan was approved (until it ends). declined: said no. passed: it started without a
    yes that was approved. suggested: still on offer (a yes whose card was never approved is still on offer)."""
    if accepted:
        return "accepted" if now < offer.end else "passed"
    if answer == "no":
        return "declined"
    return "passed" if now >= offer.start else "suggested"


def suggest_held(now: datetime, offer: Offer, f: Fit, state: State, gate: Gate) -> str | None:
    """Why this event is not suggested right now, in a few words; None when it may be."""
    if not gate.enabled:
        return "suggestions are switched off"
    if state != "suggested":
        return state
    if gate.said:
        return "already suggested"
    lead = (offer.start - now).total_seconds() / 60
    if not gate.until_min <= lead <= gate.from_min:
        return "not within the suggestion window"
    if not f.ok:
        return f"does not fit: {f.reason}"
    if not offer.people:
        return "nobody is named"
    if not gate.awake:
        return "outside waking hours"
    if gate.fixed:
        return f"during {gate.fixed}"
    if gate.leave_by is not None and gate.leave_by - timedelta(minutes=LEAVE_GUARD_MIN) <= now < max(gate.leave_by, gate.event_start or gate.leave_by):
        return "close to a time to leave"
    if gate.last is not None and abs(now - gate.last) < timedelta(minutes=gate.cooldown_min):
        return "something was suggested less than forty-five minutes ago"
    return None


# ---------- what a yes would move ----------


def changes(old: list[PlanMove], new: list[PlanMove], now: datetime, tz: tzinfo) -> list[dict]:
    """What a re-plan does to the suggested blocks of the plan it replaces, from `now` on: each one kept as it is is
    left out; the rest are `moved` (a new start), `shortened`/`lengthened`, or `dropped`. Fixed blocks are listed
    only if one changed, which a yes must never cause (the caller refuses the re-plan then)."""

    def key(m: PlanMove) -> tuple:
        return (m.kind, m.place, m.deadline, m.title) if not m.fixed else ("fixed", m.id)

    def hm(t: datetime) -> str:
        return t.astimezone(tz).strftime("%H:%M")

    left = [m for m in sorted(new, key=lambda m: m.start) if m.end > now]
    out: list[dict] = []
    for m in sorted((m for m in old if m.end > now), key=lambda m: m.start):
        same = [n for n in left if key(n) == key(m)]
        if not same:
            out.append({"id": m.id, "title": m.title, "kind": m.kind, "fixed": m.fixed, "change": "dropped", "start": hm(m.start), "end": hm(m.end)})
            continue
        n = min(same, key=lambda n: abs((n.start - m.start).total_seconds()))
        left.remove(n)
        if n.start == m.start and n.end == m.end:
            continue
        if m.fixed and n.start <= max(m.start, now) and n.end == m.end:
            continue  # a fixed block already running is cut to "from now" by every re-plan: not a change
        change = "moved" if n.start != m.start else "shortened" if n.minutes < m.minutes else "lengthened"
        out.append({"id": m.id, "title": m.title, "kind": m.kind, "fixed": m.fixed, "change": change, "start": hm(m.start), "end": hm(m.end),
                    "new_start": hm(n.start), "new_end": hm(n.end)})
    return out
