"""Choosing a day: a small deterministic search over Day Run moves, scored by the engine itself.

    moves, result = choose(cfg, walks, board)

The rule, in plain words: for each open slot in time order, lay down the things the day needs at their
usual times (meals near 8:30, 12:30 and 18:30; the night's sleep in the usual sleep window; at most one
optional event), fill the gaps with study at the nearest good place (or rest, or nothing), simulate the
whole day for each way of doing that, and keep the one with the most points; then go round once more with
every other slot already chosen, so a late slot can change an early one. Points are the engine's own
(deadline risk, sleep, meals, energy), plus a bonus for a winning plan and its stars. Nothing random: the
same board always gives the same plan.
"""

from dataclasses import dataclass
from datetime import datetime, time, timedelta

from contexts.play.dayrun.board import minutes_between
from contexts.play.dayrun.config import DayRunConfig
from contexts.play.dayrun.engine import simulate
from contexts.play.dayrun.models import Board, Move, OpenSlot, SimResult
from contexts.play.dayrun.routes import Walks

MEAL_TIMES = (time(8, 30), time(12, 30), time(18, 30))
MEAL_MINUTES = 30
BREAK_AFTER = 120  # study this long, then
BREAK_MINUTES = 20
STUDY_PLACES = 2  # the nearest study places tried per slot
EVENT_MAX_MINUTES = 150  # an optional event longer than this is not worth a candidate
WIN_BONUS, STAR_BONUS, BIAS_BONUS, REJECT_PENALTY, CRASH_PENALTY = 100, 5, 3, 2, 0.5
FILLERS = ("study", "study_breaks", "rest", "none")


@dataclass(frozen=True)
class Anchor:
    kind: str  # eat, sleep, event
    place: str
    start: datetime
    minutes: int
    event: str | None = None


def score(cfg: DayRunConfig, result: SimResult, bias: frozenset[str], slot: OpenSlot, moves: list[Move]) -> float:
    s = result.totals.points + (WIN_BONUS if result.outcome.won else 0) + STAR_BONUS * result.stars
    s -= REJECT_PENALTY * len(result.rejected)
    s -= CRASH_PENALTY * max(0.0, cfg.stars.energy_floor - result.totals.energy_min)  # rest before energy runs out
    if "earlier_study" in bias:
        first = next((m for m in moves if m.slot == slot.id and m.kind == "study"), None)
        if first and first.start and minutes_between(slot.start, first.start) <= 30:
            s += BIAS_BONUS
    return s


def _places(cfg: DayRunConfig, w: Walks, slot: OpenSlot, kind: str) -> list[tuple[str, list[str]]]:
    """Feasible places of `kind` in this slot, nearest first, each with the friends sharing it."""
    out = [(o.place, o.friends) for o in slot.moves if o.kind == kind and o.feasible and o.place]
    return out[:STUDY_PLACES] if kind == "study" else out[:1]


def anchors_for(cfg: DayRunConfig, w: Walks, board: Board, slot: OpenSlot, meals: bool, event_id: str | None,
                bias: frozenset[str]) -> list[Anchor]:
    out: list[Anchor] = []
    eat = _places(cfg, w, slot, "eat")
    if meals and eat:
        day = slot.start.date()
        for n in range(3):
            for t in MEAL_TIMES:
                at = datetime.combine(day + timedelta(days=n), t, slot.start.tzinfo)
                if slot.start <= at and at + timedelta(minutes=MEAL_MINUTES) <= slot.end:
                    out.append(Anchor("eat", eat[0][0], at, MEAL_MINUTES))
    if board.sleep_window:
        bed, wake = board.sleep_window
        if "earlier_sleep" in bias:
            bed -= timedelta(minutes=30)
        a, b = max(bed, slot.start), min(wake, slot.end)
        if minutes_between(a, b) >= cfg.sleep.min_block_minutes:
            out.append(Anchor("sleep", board.home, a, minutes_between(a, b)))
    if event_id:
        e = next(e for e in board.events if e.id == event_id)
        a, b = max(e.start, slot.start), min(e.end, slot.end)
        if e.place and minutes_between(a, b) >= cfg.min_move_minutes:
            out.append(Anchor("event", e.place, a, minutes_between(a, b), event=e.id))
    # an anchor that overlaps an earlier one is dropped: eat before sleep, not during it
    kept: list[Anchor] = []
    for a in sorted(out, key=lambda a: (a.start, a.kind)):
        if not kept or a.start >= kept[-1].start + timedelta(minutes=kept[-1].minutes):
            kept.append(a)
    return kept


def pieces(cfg: DayRunConfig, w: Walks, board: Board, slot: OpenSlot, anchors: list[Anchor], filler: str,
           study: tuple[str, list[str]] | None) -> list[Move]:
    """The slot's moves: fillers between the anchors, every move with its departure time."""
    moves: list[Move] = []
    cursor, here = slot.start, slot.from_place

    def fill(until: datetime | None, last: bool) -> None:
        nonlocal cursor, here
        if filler == "none":
            return
        place = (study[0] if study else here) if filler.startswith("study") else here
        friend = study[1][0] if study and study[1] and filler.startswith("study") else None
        kind = "study" if filler.startswith("study") else "rest"
        there = w.walk(here, place)[0]
        begin = cursor + timedelta(minutes=there)
        end = until if until is not None else slot.end - timedelta(minutes=w.walk(place, slot.to_place)[0])
        room = minutes_between(begin, end)
        if room < cfg.min_move_minutes:
            return
        if filler == "study_breaks" and room > BREAK_AFTER + BREAK_MINUTES + cfg.min_move_minutes:
            t, depart = begin, cursor  # the first move leaves at `cursor` and walks; the rest start where it is
            while minutes_between(t, end) >= cfg.min_move_minutes:
                chunk = min(BREAK_AFTER, minutes_between(t, end))
                moves.append(Move(slot=slot.id, kind="study", place=place, minutes=chunk, start=depart, with_friend=friend))
                t = depart = t + timedelta(minutes=chunk)
                if minutes_between(t, end) < BREAK_MINUTES + cfg.min_move_minutes:
                    break
                moves.append(Move(slot=slot.id, kind="rest", place=place, minutes=BREAK_MINUTES, start=t))
                t = depart = t + timedelta(minutes=BREAK_MINUTES)
            cursor, here = t, place
            return
        moves.append(Move(slot=slot.id, kind=kind, place=place, minutes=None if last else room, start=cursor, with_friend=friend))
        cursor, here = end, place

    for a in anchors:
        filler_place = (study[0] if study else here) if filler.startswith("study") else here
        fill(a.start - timedelta(minutes=w.walk(filler_place, a.place)[0]), last=False)
        depart = max(cursor, a.start - timedelta(minutes=w.walk(here, a.place)[0]))
        moves.append(Move(slot=slot.id, kind=a.kind, place=a.place, minutes=a.minutes, start=depart, event=a.event))
        cursor = depart + timedelta(minutes=w.walk(here, a.place)[0] + a.minutes)
        here = a.place
    fill(None, last=True)
    return moves


def candidates(cfg: DayRunConfig, w: Walks, board: Board, slot: OpenSlot, bias: frozenset[str]) -> list[list[Move]]:
    """Every way this slot is tried, in a fixed order; the first of two equal scores wins."""
    studies = _places(cfg, w, slot, "study") or [None]
    events = [e.id for e in board.events if e.place and e.start < slot.end and e.end > slot.start
              and minutes_between(max(e.start, slot.start), min(e.end, slot.end)) <= EVENT_MAX_MINUTES][:1]
    out: list[list[Move]] = [[]]
    seen = {()}
    for meals in (True, False):
        for event_id in [*events, None]:
            anchors = anchors_for(cfg, w, board, slot, meals, event_id, bias)
            for filler in FILLERS:
                for study in (studies if filler.startswith("study") else [None]):
                    moves = pieces(cfg, w, board, slot, anchors, filler, study)
                    key = tuple((m.kind, m.place, m.minutes, m.start, m.with_friend, m.event) for m in moves)
                    if key not in seen:
                        seen.add(key)
                        out.append(moves)
    return out


def choose(cfg: DayRunConfig, w: Walks, board: Board, bias: frozenset[str] = frozenset()) -> tuple[list[Move], SimResult]:
    """The best plan the search finds for this board, and its simulation."""
    chosen: dict[str, list[Move]] = {s.id: [] for s in board.slots}
    options = {s.id: candidates(cfg, w, board, s, bias) for s in board.slots}

    def plan() -> list[Move]:
        return [m for s in board.slots for m in chosen[s.id]]

    for _ in range(2):  # greedy, then one pass with everything else in place
        for slot in board.slots:
            best: tuple[float, int] | None = None
            for n, cand in enumerate(options[slot.id]):
                chosen[slot.id] = cand
                result = simulate(cfg, w, board, plan())
                s = score(cfg, result, bias, slot, cand)
                if best is None or s > best[0]:
                    best = (s, n)
            chosen[slot.id] = options[slot.id][best[1]] if best else []
    moves = plan()
    return moves, simulate(cfg, w, board, moves)
