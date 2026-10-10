"""Day Run's other modes (S6), all on the same engine: hard mode, campaign, and the replay baseline.

Hard mode's disruptions are picked with `random.Random(seed)`: the same seed, board and plan
always give the same disruptions.
"""

import random
from collections.abc import Callable
from datetime import datetime, timedelta

from contexts.play.dayrun.board import hm, lay_out, minutes_between
from contexts.play.dayrun.config import DayRunConfig
from contexts.play.dayrun.engine import simulate
from contexts.play.dayrun.models import (
    Board,
    CampaignDay,
    CampaignResult,
    Change,
    Disruption,
    HardResult,
    InitialOverride,
    Measured,
    Move,
)
from contexts.play.dayrun.routes import Walks

MOVE_TO_KINDS = ["class", "library", "gym", "outdoors", "dining", "venue"]


# --- hard mode ---


def pick_disruptions(cfg: DayRunConfig, w: Walks, board: Board, moves: list[Move], seed: int, count: int) -> list[Disruption]:
    rng = random.Random(seed)
    makers: dict[str, Callable[[], Disruption]] = {}

    movable = sorted(b.id for b in board.blocks if b.place and b.activity != "sleep")
    if movable:
        def block_moved() -> Disruption:
            chosen = rng.choice(movable)
            block = next(b for b in board.blocks if b.id == chosen)
            return Disruption(kind="block_moved", block=block.id,
                              to_place=rng.choice(sorted(p for p in w.of_kind(MOVE_TO_KINDS, [board.pack]) if p != block.place)),
                              shift_minutes=rng.choice([0, -15, -30]))
        makers["block_moved"] = block_moved

    friends = sorted({m.with_friend for m in moves if m.with_friend} & {f.member for f in board.friends}) or sorted({f.member for f in board.friends})
    if friends:
        makers["friend_cancelled"] = lambda: Disruption(kind="friend_cancelled", friend=rng.choice(friends))

    eat_at = sorted({w.ref(m.place) for m in moves if m.kind == "eat" and m.place} & set(w.of_kind(cfg.kinds["eat"], [board.pack])))
    eat_at = eat_at or sorted(w.of_kind(cfg.kinds["eat"], [board.pack]))
    if eat_at:
        makers["place_closed"] = lambda: Disruption(kind="place_closed", place=rng.choice(eat_at))

    kinds = sorted(makers)
    rng.shuffle(kinds)
    return [makers[k]() for k in kinds[: max(0, count)]]


def apply_disruptions(cfg: DayRunConfig, w: Walks, board: Board, disruptions: list[Disruption]) -> tuple[Board, list[Disruption]]:
    """The board after the disruptions (slots and routes rebuilt), and the disruptions with their text filled in."""
    board = board.model_copy(deep=True)
    told: list[Disruption] = []
    for d in disruptions:
        d = d.model_copy()
        if d.kind == "block_moved":
            block = next((b for b in board.blocks if b.id == d.block), None)
            if block is None:
                raise ValueError(f"no fixed block {d.block!r} on this board")
            was_at, was_start = block.place_name or "nowhere in particular", block.start
            if d.to_place:
                block.place = w.ref(d.to_place)
                block.place_name = w.name(block.place)
            shift = timedelta(minutes=d.shift_minutes)
            block.start, block.end = max(board.start, block.start + shift), block.end + shift
            parts = ([f"moves from {was_at} to {block.place_name}"] if d.to_place else []) + (
                [f"now starts at {hm(block.start)} instead of {hm(was_start)}"] if d.shift_minutes else [])
            d.text = d.text or f"{block.title} {' and '.join(parts) or 'is unchanged'}"
        elif d.kind == "friend_cancelled":
            board.friends = [f for f in board.friends if f.member.lower() != (d.friend or "").lower()]
            d.text = d.text or f"{d.friend} cancels: not coming after all"
        else:
            place = w.ref(d.place or "")
            if place not in board.closed:
                board.closed.append(place)
            d.text = d.text or f"{w.name(place)} is closed today"
        told.append(d)
    return lay_out(cfg, w, board), told


def hard(cfg: DayRunConfig, w: Walks, board: Board, moves: list[Move], seed: int = 0, count: int = 1,
         disruptions: list[Disruption] | None = None) -> HardResult:
    chosen = disruptions if disruptions is not None else pick_disruptions(cfg, w, board, moves, seed, count)
    before = simulate(cfg, w, board, moves)
    shaken, told = apply_disruptions(cfg, w, board, chosen)
    after = simulate(cfg, w, shaken, moves, repair=True)  # re-validate every move and re-route
    changes = list(after.changes)
    for d in told:
        if d.kind != "block_moved":
            continue
        title = next(b.title for b in shaken.blocks if b.id == d.block)
        was = next((r for r in before.routes if r.purpose == f"to {title}"), None)
        now = next((r for r in after.routes if r.purpose == f"to {title}"), None)

        def say(r) -> str:
            return f"{' -> '.join(w.name(p) for p in r.places)} ({r.minutes:g} min, leave {hm(r.depart)})" if r else "no walk needed"

        if say(was) != say(now):
            changes.append(Change(kind="route_changed", text=f"Route to {title} was {say(was)}; now {say(now)}"))
    return HardResult(seed=seed, disruptions=told, before=before, after=after, changes=changes)


# --- campaign ---


def campaign(cfg: DayRunConfig, w: Walks, board_for: Callable[[datetime, InitialOverride | None], Board], start: datetime,
             days: list[list[Move]], initial: InitialOverride | None = None) -> CampaignResult:
    """Consecutive days: each starts with the energy, sleep debt, hunger and work the day before left."""
    over = initial
    out: list[CampaignDay] = []
    for n, moves in enumerate(days):
        board = board_for(start + timedelta(days=n), over)
        result = simulate(cfg, w, board, moves)
        out.append(CampaignDay(day=board.start.date(), carry_in=board.initial, result=result))
        c = result.carry
        over = InitialOverride(energy=c.energy, focus=c.focus, sleep_debt=c.sleep_debt, hungry_minutes=c.hungry_minutes,
                               start_place=c.place, work_left=c.work_left, sleep_hours=result.totals.sleep_hours, meals=0)
    return CampaignResult(days=out, won=bool(out) and all(d.result.outcome.won for d in out),
                          stars=sum(d.result.stars for d in out),
                          points=round(sum(d.result.totals.points for d in out) / len(out)) if out else 0)


# --- replay: what actually happened, as a plan ---


def baseline_moves(cfg: DayRunConfig, w: Walks, board: Board, actual: Measured) -> list[Move]:
    """The measured day as moves the engine can run: that much sleep, that many meals, that much desk time.

    Habits knows totals, not times of day, so the times are a reconstruction: sleep ends when
    the board does, meals sit near 12:30 and 18:30 (then 08:30), and desk time fills the earliest
    free time. Everything happens at home, where the box can see it.
    """
    desk = w.ref(cfg.desk)
    kitchens = sorted(w.of_kind(cfg.kinds["eat"], [desk.split(".")[0], board.pack]), key=lambda p: (w.walk(desk, p)[0], p))
    free: dict[str, list[tuple[datetime, datetime]]] = {}
    for s in board.slots:  # what is left of each slot once you have walked home and back out
        a = s.start + timedelta(minutes=w.walk(s.from_place, desk)[0] + 2)
        b = s.end - timedelta(minutes=w.walk(desk, s.to_place)[0] + 2)
        if minutes_between(a, b) >= cfg.min_move_minutes:
            free[s.id] = [(a, b)]
    placed: list[Move] = []

    def take(slot: str, a: datetime, b: datetime, kind: str, place: str) -> None:
        pad = timedelta(minutes=3)  # room to walk between rooms
        free[slot] = [(x, y) for lo, hi in free[slot] for x, y in ((lo, min(hi, a - pad)), (max(lo, b + pad), hi))
                      if minutes_between(x, y) >= cfg.min_move_minutes]
        placed.append(Move(slot=slot, kind=kind, place=place, minutes=minutes_between(a, b), start=a))

    sleep = round((actual.sleep_hours or 0) * 60)
    last = board.slots[-1] if board.slots and board.slots[-1].next_block is None else None
    if sleep >= cfg.min_move_minutes and last and last.id in free:
        lo, hi = free[last.id][-1]
        take(last.id, max(lo, hi - timedelta(minutes=sleep)), hi, "sleep", board.home)

    day = board.start.replace(hour=0, minute=0)
    targets = [t if t >= board.start else t + timedelta(days=1)  # lunch first, then dinner, then breakfast
               for t in (day + timedelta(hours=12, minutes=30), day + timedelta(hours=18, minutes=30), day + timedelta(hours=8, minutes=30))]
    wanted = sorted(targets[: int(actual.meals or 0)]) if kitchens else []
    for target in wanted:
        spots = [(slot, max(lo, target)) for slot, spans in free.items() for lo, hi in spans
                 if minutes_between(max(lo, target), hi) >= 25]
        if spots:
            slot, at = min(spots, key=lambda s: s[1])
            take(slot, at, at + timedelta(minutes=25), "eat", kitchens[0])

    left = round(actual.focus_minutes or 0)
    while left >= cfg.min_move_minutes:
        spans = sorted((lo, hi, slot) for slot, spans in free.items() for lo, hi in spans)
        if not spans:
            break
        lo, hi, slot = spans[0]
        chunk = min(left, 120, minutes_between(lo, hi))
        take(slot, lo, lo + timedelta(minutes=chunk), "study", desk)
        left -= chunk
    return sorted(placed, key=lambda m: m.start)
