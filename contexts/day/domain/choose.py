"""Choosing a day: what the win rules need first, then points, checked by Day Run's own engine.

    moves, result = choose(cfg, walks, board)

The rule, in plain words. Read what the day has to hold from the rules and the board: the night's sleep, the
meals still to eat, the focused minutes each deadline still wants, the energy floor. Then:

1. Night. Sleep goes in the usual sleep window, a little earlier after a short night, and runs on past the
   usual wake time when the night started late and nothing is waiting.
2. Meals. Each meal still to eat goes where it fits nearest its usual time (8:00, 12:30, 18:30), with the
   walk there and on to what comes next: a fixed block in the way moves the meal earlier or later, it does
   not drop it. If the day is still short of the meals a win needs, one more goes wherever one fits.
3. Study. Only as much as the deadlines need, earliest due first, in blocks no longer than `block` minutes
   with a break after each, each block in the earliest place where the engine says energy stays above the
   floor. A block that would cross the floor gets a meal, a nap or an earlier night before the low point,
   or is cut in half; it is taken anyway only when the deadline cannot stay on track without it.
4. Repair. Simulate; while the plan does not win, read the failing checks and the rejected moves, try the
   matching repairs (another meal, more sleep, more study, a later bedtime, a shorter or dropped move) and
   keep the one the engine scores best; stop when it wins or nothing helps.
5. Improve. Cut study the deadlines turned out not to need, rest before an energy low point, add one
   optional event that costs nothing, and put a break between two study blocks.

What is left open stays open: free time is not filled with study or rest. When no plan wins, the best
attempt comes back with `outcome.summary` and `outcome.reasons` saying what cannot fit and why.

A re-plan "from here" is the same function on a board that starts now, where you are, with what already
happened (meals eaten, work left, energy). Nothing random and no clock: the same board gives the same plan.
"""

from dataclasses import dataclass, replace
from datetime import datetime, time, timedelta

from contexts.play.dayrun.board import hm, minutes_between
from contexts.play.dayrun.config import DayRunConfig
from contexts.play.dayrun.engine import simulate
from contexts.play.dayrun.models import Board, Deadline, Move, Rejected, SimResult
from contexts.play.dayrun.routes import Walks

MEAL_TIMES = (time(8, 0), time(12, 30), time(18, 30))  # the usual breakfast, lunch and dinner
MEAL_MINUTES = 30
MEAL_REACH = 150  # a meal moves at most this far from its usual time, unless a win needs it
MEAL_GRACE = 120  # a usual meal time this recently passed can still be had now
MEALS_A_DAY = 3  # an extra meal for energy never takes a day past this
MEAL_APART = 150  # meals are laid this far apart when they can be (the engine counts one from its own, shorter gap)
MAX_BLOCK = 90  # the longest study block, unless the caller knows the person's own
BREAK_MINUTES = 15  # between two study blocks (five more after a short night)
LONG_BREAK_MINUTES = 45  # the longest break tried between two blocks
MIN_STUDY = 25  # a shorter study block is not worth the walk
STUDY_MARGIN = 10  # study minutes planned beyond the estimate of what a deadline needs
WALK_WEIGHT = 0.5  # study minutes a minute of detour is worth, when picking where to study
NAP_MINUTES = (90, 60, 30)
NAP_BEFORE_BED = 240  # a nap ends at least this long before the night
WAKE_UP_MINUTES = 30  # no study straight out of bed
EARLIER_NIGHT = 30  # to bed this much earlier after a short night, and again when yesterday's review says so
EVENT_MAX_MINUTES = 150  # a longer optional event is not offered
REST_MAX_MINUTES = 90
REST_MIN_MINUTES = 30  # a shorter rest is only ever the break between two study blocks
MAX_STUDY_STEPS = 24
STUDY_RATE = 0.5  # focused minutes a study minute is first taken to give; the surplus is cut afterwards
MIN_GAIN = 0.2  # a study block has to turn at least this share of its minutes into work
MAX_REPAIRS = 12
ENERGY_STEPS = 5
HELPED_SPOTS = 3  # study blocks tried with help for energy, per step
MAX_RESTS = 4  # rest blocks laid down for energy or focus

Plan = dict  # slot id -> the slot's pieces in time order (a tuple)


@dataclass(frozen=True)
class Piece:
    """One chosen activity: `start` is when it begins, after the walk to `place`."""

    kind: str  # the move kind: study, eat, sleep, rest, event
    place: str
    start: datetime
    minutes: int
    role: str = ""  # night, nap, meal, study, break, rest, event
    friend: str | None = None
    deadline: str | None = None
    event: str | None = None

    @property
    def end(self) -> datetime:
        return self.start + timedelta(minutes=self.minutes)


@dataclass(frozen=True)
class Gap:
    """Open time in a slot: after `t0` at `p0` (what ended there was `before`), until `t1` at `p1`."""

    slot: str
    t0: datetime
    p0: str
    before: str
    t1: datetime
    p1: str | None
    after: str


def _ceil5(x: float) -> int:
    return int(-(-x // 5) * 5)


class _Day:
    def __init__(self, cfg: DayRunConfig, w: Walks, board: Board, bias: frozenset, block: int | None) -> None:
        self.cfg, self.w, self.board, self.bias = cfg, w, board, bias
        self.block = max(cfg.min_move_minutes, int(block or MAX_BLOCK))
        self.slots = {s.id: s for s in board.slots}
        self.slot_before = {s.next_block: s for s in board.slots}  # fixed block id (None: the board's end) -> the slot before it
        self.floor = cfg.stars.energy_floor
        self.short_night = board.initial.sleep_hours < cfg.win.sleep_hours
        self.pause = BREAK_MINUTES + (5 if self.short_night else 0)
        drain, gain = abs(cfg.focus_per_hour.get("study", 0.0)), cfg.focus_per_hour.get("rest", 0.0)
        # the break that gives back the focus a full block takes, by the engine's own rates
        self.long_pause = max(self.pause, min(LONG_BREAK_MINUTES, _ceil5(self.block * drain / gain) if gain > 0 else self.pause))
        self.sims = 0
        self._walks: dict[tuple, int] = {}
        self._places: dict[str, list[str]] = {}
        self._results: dict[tuple, SimResult] = {}
        self._traces: dict[tuple, tuple[bool, dict[str, str]]] = {}
        self.base_late: dict[str, int] = {}
        self.base_late = self._trace(self.empty(), measure=True)
        self.base_bad = self.bad(self.sim(self.empty()))

    # --- plans ---

    def empty(self) -> Plan:
        return {s.id: () for s in self.board.slots}

    def frozen(self, plan: Plan) -> tuple:
        return tuple(plan[s.id] for s in self.board.slots)

    def walk(self, a: str | None, b: str | None) -> int:
        if (a, b) not in self._walks:
            self._walks[a, b] = self.w.walk(a, b)[0]
        return self._walks[a, b]

    def _trace(self, plan: Plan, measure: bool = False):
        """Walk the board the way the engine does. Is every piece reached on time, in order, and every fixed
        block reached no later than an empty plan reaches it; and where does each slot really start from (a
        fixed block with no place leaves you where you were)."""
        late: dict[str, int] = {}
        origins: dict[str, str] = {}
        ok = True
        clock, place = self.board.start, self.board.start_place
        for nxt in [*self.board.blocks, None]:
            slot = self.slot_before.get(nxt.id if nxt else None)
            if slot:
                origins[slot.id] = place
                for p in plan[slot.id]:
                    if p.minutes < 1 or p.start - timedelta(minutes=self.walk(place, p.place)) < clock:
                        ok = False
                    clock, place = max(clock, p.end), p.place
                if plan[slot.id] and clock + timedelta(minutes=self.walk(place, slot.to_place)) > slot.end:
                    ok = False  # (an empty slot too short for its own walk is late whatever is chosen)
            if nxt is None:
                break
            there = self.walk(place, nxt.place)
            arrive = max(clock, nxt.start - timedelta(minutes=there)) + timedelta(minutes=there)
            late[nxt.id] = max(0, minutes_between(nxt.start, arrive))
            if late[nxt.id] > self.base_late.get(nxt.id, 0):
                ok = False
            clock, place = max(arrive, nxt.end), nxt.place or place
        return late if measure else (ok, origins)

    def trace(self, plan: Plan) -> tuple[bool, dict[str, str]]:
        key = self.frozen(plan)
        if key not in self._traces:
            self._traces[key] = self._trace(plan)
        return self._traces[key]

    def put(self, plan: Plan, slot: str, piece: Piece, instead_of: Piece | None = None) -> Plan | None:
        """The plan with `piece` in `slot` (replacing `instead_of`), or None when it cannot be walked."""
        pieces = [p for p in plan[slot] if p != instead_of] + [piece]
        new = {**plan, slot: tuple(sorted(pieces, key=lambda p: (p.start, p.kind)))}
        return new if self.trace(new)[0] else None

    def drop(self, plan: Plan, slot: str, piece: Piece) -> Plan:
        return {**plan, slot: tuple(p for p in plan[slot] if p != piece)}

    def pieces(self, plan: Plan) -> list[tuple[str, Piece]]:
        """Every piece with its slot, in the order the engine runs them (the index of its move)."""
        return [(s.id, p) for s in self.board.slots for p in plan[s.id]]

    def moves(self, plan: Plan) -> list[Move]:
        origins = self.trace(plan)[1]
        out: list[Move] = []
        for s in self.board.slots:
            here = origins.get(s.id, s.from_place)
            for p in plan[s.id]:
                depart = p.start - timedelta(minutes=self.walk(here, p.place))
                out.append(Move(slot=s.id, kind=p.kind, place=p.place, minutes=p.minutes, start=depart, with_friend=p.friend,
                                deadline=p.deadline, event=p.event))
                here = p.place
        return out

    def sim(self, plan: Plan) -> SimResult:
        key = self.frozen(plan)
        if key not in self._results:
            self.sims += 1
            self._results[key] = simulate(self.cfg, self.w, self.board, self.moves(plan))
        return self._results[key]

    @staticmethod
    def bad(res: SimResult) -> int:
        """Moves the engine refused, and fixed blocks reached late."""
        return len(res.rejected) + sum(c.kind == "late" for c in res.changes)

    def under(self, res: SimResult) -> float:
        """How far and how long energy is under the floor (0: never)."""
        return round(sum(max(0.0, self.floor - t.energy) for t in res.timeline), 1)

    def key(self, res: SimResult, plan: Plan) -> tuple:
        """How good the engine says a plan is, most important first: nothing refused or late, the win checks
        (and how far the failing ones are off), energy above the floor, stars, the work left, points; then
        fewer pieces. So a block a win depends on may cross the floor; one that only finishes the work may not."""
        short = 0.0
        work = {d.id: d.work_left for d in self.board.deadlines}
        for c in res.outcome.checks:
            if c.ok:
                continue
            if c.status:  # a deadline
                short += 1 + res.carry.work_left.get(c.id, 0.0) / max(work.get(c.id, 1.0), 1.0)
            else:
                short += (c.target - c.value) / max(c.target, 1.0)
        return (-self.bad(res), -sum(not c.ok for c in res.outcome.checks), -round(short, 3), -self.under(res), res.stars,
                -self.left(res), res.totals.points, -len(self.pieces(plan)))

    @staticmethod
    def left(res: SimResult) -> float:
        return round(sum(res.carry.work_left.values()), 1)

    def score(self, plan: Plan) -> tuple:
        return self.key(self.sim(plan), plan)

    def no_worse(self, plan: Plan, than: Plan) -> bool:
        """As good by the engine's score, the number of pieces aside."""
        return self.score(plan)[:-1] >= self.score(than)[:-1]

    # --- the open time ---

    def gaps(self, plan: Plan) -> list[Gap]:
        origins = self.trace(plan)[1]
        blocks = self.board.blocks
        out: list[Gap] = []
        for s in self.board.slots:
            n = next((i for i, b in enumerate(blocks) if b.id == s.next_block), len(blocks))
            t, here = s.start, origins.get(s.id, s.from_place)
            was = blocks[n - 1].activity if n and blocks[n - 1].end >= s.start else ""
            for p in [*plan[s.id], None]:
                if p is None:
                    out.append(Gap(s.id, t, here, was, s.end, s.to_place, blocks[n].activity if n < len(blocks) else ""))
                else:
                    out.append(Gap(s.id, t, here, was, p.start, p.place, p.kind))
                    t, here, was = p.end, p.place, p.kind
        return [g for g in out if g.t1 > g.t0]

    def room(self, g: Gap, place: str) -> tuple[datetime, datetime]:
        """The earliest start and the latest end of something at `place` in this gap, walks taken out."""
        return g.t0 + timedelta(minutes=self.walk(g.p0, place)), g.t1 - timedelta(minutes=self.walk(place, g.p1))

    def detour(self, g: Gap, place: str) -> int:
        return self.walk(g.p0, place) + self.walk(place, g.p1) - self.walk(g.p0, g.p1)

    def places(self, kind: str) -> list[str]:
        """Where the board offers `kind`, in the order its slots list them. (A slot's own "feasible" is measured
        from where the calendar leaves you, which is not always where the plan does: the room is worked out here.)"""
        if kind not in self._places:
            offered = [o.place for s in self.board.slots for o in s.moves if o.kind == kind and o.place]
            self._places[kind] = [p for p in dict.fromkeys(offered) if p not in self.board.closed]
        return self._places[kind]

    # --- the night ---

    def night(self, plan: Plan, early: int | None = None) -> Plan:
        """Sleep in the usual window; `early` minutes before the usual bedtime (None: what the night before asks)."""
        if not self.board.sleep_window:
            return plan
        bed, wake = self.board.sleep_window
        if early is None:
            early = EARLIER_NIGHT * (("earlier_sleep" in self.bias) + self.short_night)
        need, got = round(self.cfg.sleep.need_hours * 60), 0
        home = self.board.home
        for g in self.gaps(plan):
            a, b = self.room(g, home)
            start, end = max(a, bed - timedelta(minutes=early)), min(b, wake)
            if start > bed:  # in bed late: sleep on past the usual wake time, as far as what comes next allows
                end = min(b, max(end, start + timedelta(minutes=need - got)))
            n = minutes_between(start, end)
            if n >= self.cfg.sleep.min_block_minutes:
                new = self.put(plan, g.slot, Piece("sleep", home, start, n, "night"))
                if new:
                    plan, got = new, got + n
        return plan

    def nights(self, plan: Plan) -> list[tuple[str, Piece]]:
        return [(s, p) for s, p in self.pieces(plan) if p.role == "night"]

    def bedtimes(self, plan: Plan, minutes: tuple[int, ...]) -> list[Plan]:
        """The plan with each night starting that much later (negative: earlier), waking when it does now."""
        out = []
        for s, p in self.nights(plan):
            for by in minutes:
                if p.minutes - by >= self.cfg.sleep.min_block_minutes:
                    new = self.put(plan, s, replace(p, start=p.start + timedelta(minutes=by), minutes=p.minutes - by), instead_of=p)
                    if new:
                        out.append(new)
        return out

    def more_sleep(self, plan: Plan) -> list[Plan]:
        """Ways to sleep longer: to bed earlier, up later, or the usual night laid down again where it is missing."""
        out = self.bedtimes(plan, (-30, -60, -120))
        for s, p in self.nights(plan):
            for by in (30, 60, 120):
                new = self.put(plan, s, replace(p, minutes=p.minutes + by), instead_of=p)
                if new:
                    out.append(new)
        if not self.nights(plan):
            out.append(self.night(plan))
            for g in self.gaps(plan):  # no usual night fits: the longest stretch at home that does
                a, b = self.room(g, self.board.home)
                n = min(minutes_between(a, b), round(self.cfg.sleep.need_hours * 60))
                new = self.put(plan, g.slot, Piece("sleep", self.board.home, a, n, "night")) if n >= self.cfg.sleep.min_block_minutes else None
                if new:
                    out.append(new)
        return out

    # --- meals ---

    def meal_starts(self, plan: Plan) -> list[datetime]:
        fixed = [b.start for b in self.board.blocks if b.activity == "eat"]
        return sorted(fixed + [p.start for _, p in self.pieces(plan) if p.kind == "eat"])

    def meal_count(self, plan: Plan) -> int:
        return self.board.initial.meals + len(self.meal_starts(plan))

    def meals_on(self, plan: Plan, when: datetime) -> int:
        eaten = self.board.initial.meals if when.date() == self.board.start.date() else 0
        return eaten + sum(m.date() == when.date() for m in self.meal_starts(plan))

    def meal(self, plan: Plan, ideal: datetime, reach: int | None = None, before: datetime | None = None,
             apart: int | None = None, role: str = "meal") -> Plan | None:
        """The plan with one more meal as near `ideal` as one fits (within `reach` minutes, ending by `before`),
        far enough from every other meal to count as one. None when nowhere can hold it."""
        apart = max(apart or 0, self.cfg.meal_gap_minutes)
        others = self.meal_starts(plan)
        wanted = [ideal] + [m + timedelta(minutes=d) for m in others for d in (apart, -apart)]
        lengths = list(dict.fromkeys([MEAL_MINUTES, max(self.cfg.meal_min_minutes, self.cfg.min_move_minutes)]))
        found: list[tuple] = []
        for g in self.gaps(plan):
            for n, place in enumerate(self.places("eat")):
                a, b = self.room(g, place)
                b = min(b, before) if before else b
                for minutes in lengths:
                    if minutes_between(a, b) < minutes:
                        continue
                    for want in wanted:
                        start = min(max(want, a), b - timedelta(minutes=minutes))
                        off = abs(minutes_between(ideal, start))
                        if any(abs(minutes_between(m, start)) < apart for m in others) or (reach is not None and off > reach):
                            continue
                        found.append((off + MEAL_MINUTES - minutes, self.detour(g, place), start, n, g.slot, place, minutes))
        for _, _, start, _, slot, place, minutes in sorted(found):
            new = self.put(plan, slot, Piece("eat", place, start, minutes, role))
            if new:
                return new
        return None

    def meal_wishes(self) -> list[tuple[datetime, int | None]]:
        """When the meals still to eat would usually be, each with how far it may move: the rest of today's
        (what was eaten stands for the usual meals already past, and any more for those coming up soon), a
        late one as soon as it fits if today would otherwise end short of a win, and tomorrow's that fall on
        this board. A meal already on the calendar stands for the usual one nearest it."""
        start, end = self.board.start, self.board.end
        eaten = self.board.initial.meals
        today = [datetime.combine(start.date(), t, start.tzinfo) for t in MEAL_TIMES]
        past = [t for t in today if t < start]
        spare = max(0, eaten - len(past))  # eaten beyond the usual meals so far: the next usual ones, if near
        open_past = past[eaten:]
        ahead = []
        for t in today[len(past):]:
            if spare and minutes_between(start, t) <= MEAL_REACH:
                spare -= 1
            else:
                ahead.append(t)
        recent = [t for t in open_past if t >= start - timedelta(minutes=MEAL_GRACE)]
        late = max(0, min(1, len(open_past) - len(recent), self.cfg.win.meals - eaten - len(recent) - len(ahead)))
        wishes: list[tuple[datetime, int | None]] = [(start, None)] * late + [(start, MEAL_REACH) for _ in recent]
        wishes += [(t, MEAL_REACH) for t in ahead]
        day = start.date() + timedelta(days=1)
        while day <= end.date():
            wishes += [(datetime.combine(day, t, start.tzinfo), MEAL_REACH) for t in MEAL_TIMES]
            day += timedelta(days=1)
        wishes = [(t, reach) for t, reach in wishes if t + timedelta(minutes=MEAL_MINUTES) <= end]
        for b in self.board.blocks:
            if b.activity == "eat" and wishes:
                near = min(wishes, key=lambda x: abs(minutes_between(x[0], b.start)))
                if abs(minutes_between(near[0], b.start)) <= MEAL_REACH:
                    wishes.remove(near)
        return wishes

    def lay_meals(self, plan: Plan, wishes: list[tuple[datetime, int | None]]) -> tuple[Plan, list[datetime]]:
        missed: list[datetime] = []
        for ideal, reach in wishes:
            new = self.meal(plan, ideal, reach=reach, apart=MEAL_APART) or self.meal(plan, ideal, reach=reach)
            if new:
                plan = new
            else:
                missed.append(ideal)
        return plan, missed

    def meals(self, plan: Plan) -> Plan:
        """Every meal still to eat, each near its usual time. If that leaves today short of the meals a win
        needs, today's are laid again as near as any slot can hold them (a fixed block in the way moves a
        meal, it does not drop it); if the board is still short, one more goes wherever one fits."""
        start, need = self.board.start, self.cfg.win.meals
        wishes = self.meal_wishes()
        first, missed = self.lay_meals(plan, wishes)
        if self.meals_on(first, start) < need:
            again, missed_again = self.lay_meals(plan, [(t, None if t.date() == start.date() else reach) for t, reach in wishes])
            if self.meals_on(again, start) > self.meals_on(first, start):
                first, missed = again, missed_again
        plan = first
        for ideal in [*missed, start]:
            if self.meal_count(plan) >= need:
                break
            plan = self.meal(plan, ideal) or plan
        return plan

    # --- study ---

    def needs(self, res: SimResult) -> list[tuple[Deadline, str, float]]:
        """The deadlines that still want work on this board, earliest due first: how badly ("must": a win
        depends on it; "should": the third star does; "could": it just finishes it) and the focused minutes."""
        cfg, out = self.cfg, []
        for d in sorted(self.board.deadlines, key=lambda d: (d.due, d.id)):
            left = res.carry.work_left.get(d.id, 0.0)
            must = should = 0.0
            if d.due > self.board.end:
                capacity = (d.due - self.board.end).total_seconds() / 3600 * cfg.capacity_minutes_per_day / 24
                must, should = cfg.on_track_risk * capacity, cfg.stars.risk * capacity
            if left > (0.01 if d.due <= self.board.end else 0.5):  # due on this board: every minute, or it is missed
                out.append((d, "must" if left > must else "should" if left > should else "could", left))
        return out

    def study_piece(self, g: Gap, d: Deadline, want: int, pause: int) -> Piece | None:
        """The best study block for this gap: where the minutes are worth most, `pause` after other study."""
        wait = pause if g.before == "study" else WAKE_UP_MINUTES if g.before == "sleep" else 0
        lo = g.t0 + timedelta(minutes=wait)
        hi = min(g.t1 - timedelta(minutes=self.pause if g.after == "study" else 0), d.due)
        best = None
        for place in self.places("study"):
            a, b = self.room(g, place)
            a, b = max(a, lo), min(b, hi)
            minutes = min(want, self.block, minutes_between(a, b))
            if minutes < min(MIN_STUDY, want):
                continue
            end = a + timedelta(minutes=minutes)
            friend = next((f.member for f in sorted(self.board.friends, key=lambda f: f.member)
                           if f.place == place and f.start < end and f.end > a), None)
            factor = self.cfg.study_place_factor.get(self.w.kind(place), self.cfg.study_place_factor.get("default", 0.7))
            worth = minutes * factor * (self.cfg.friend_boost if friend else 1.0) - WALK_WEIGHT * self.detour(g, place)
            if best is None or worth > best[0]:
                best = (worth, Piece("study", place, a, minutes, "study", friend, d.id))
        return best[1] if best else None

    def study_spots(self, plan: Plan, d: Deadline, todo: float) -> list[tuple[Piece, Plan]]:
        """One block per gap, as early as it fits; after other study, once after the short break and once
        after the long one, the break itself as rest when it is spent in the same place."""
        want = max(MIN_STUDY, _ceil5(todo / STUDY_RATE + STUDY_MARGIN))
        out = []
        for g in self.gaps(plan):
            for pause in ((self.pause, self.long_pause) if g.before == "study" else (0,)):
                piece = self.study_piece(g, d, want, pause)
                new = self.put(plan, g.slot, piece) if piece else None
                if new and g.before == "study" and piece.place == g.p0 and minutes_between(g.t0, piece.start) >= 10:
                    new = self.put(new, g.slot, Piece("rest", g.p0, g.t0, minutes_between(g.t0, piece.start), "break")) or new
                if new:
                    out.append((piece, new))
        return out

    def rests(self, plan: Plan) -> list[tuple[Piece, Plan]]:
        """Rest in place of free time, up against what comes next: the engine's way to get focus back."""
        out = []
        if sum(p.role == "rest" for _, p in self.pieces(plan)) >= MAX_RESTS:
            return out
        for g in self.gaps(plan):
            end = g.t1 - timedelta(minutes=self.walk(g.p0, g.p1))
            minutes = min(REST_MAX_MINUTES, minutes_between(g.t0, end))
            if minutes >= REST_MIN_MINUTES and g.before != "rest" and g.after != "rest":
                piece = Piece("rest", g.p0, end - timedelta(minutes=minutes), minutes, "rest")
                new = self.put(plan, g.slot, piece)
                if new:
                    out.append((piece, new))
        return out

    def study(self, plan: Plan) -> Plan:
        """Study until the deadlines have what they need or nothing more helps. Each step takes the one block
        (for the earliest-due deadline that has room) or rest the engine scores best: a block that turns too
        little of its time into work is passed over, and one that would cross the energy floor is tried with
        help before the low point, whole and at half length."""
        for _ in range(MAX_STUDY_STEPS):
            res = self.sim(plan)
            needs = self.needs(res)
            if not needs:
                break
            now, left = self.key(res, plan), self.left(res)
            spots: list[tuple[Piece, Plan]] = []
            share = MIN_GAIN
            for d, urgency, todo in needs:
                spots = self.study_spots(plan, d, todo)
                if spots:
                    share = 0.0 if urgency == "must" else MIN_GAIN  # a win depends on it: any work counts
                    break
            cands = []  # (score, order, plan)
            for n, (piece, cand) in enumerate([*spots, *self.rests(plan)]):
                gain = left - self.left(self.sim(cand))
                if gain >= (max(1.0, share * piece.minutes) if piece.kind == "study" else 1.0):
                    cands.append((self.score(cand), -n, cand, piece))
            best = max(cands, key=lambda c: c[:2], default=None)
            if best is None or best[0] <= now:  # the useful blocks cross the floor: help them
                helped = []
                for _, n, cand, piece in sorted(cands, key=lambda c: c[:2], reverse=True)[:HELPED_SPOTS]:
                    if piece.kind != "study":
                        continue
                    slot = next(s for s, p in self.pieces(cand) if p == piece)
                    half = max(MIN_STUDY, _ceil5(piece.minutes / 2))
                    shorter = self.put(plan, slot, replace(piece, minutes=half)) if half < piece.minutes else None
                    for t in (cand, shorter):
                        lifted = self.energize(t, rest=True) if t else None
                        if lifted and self.left(self.sim(lifted)) <= left - 1:
                            helped.append((self.score(lifted), n, lifted, piece))
                best = max(helped, key=lambda c: c[:2], default=None)
            if best is None or best[0] <= now:
                break
            plan = best[2]
        return plan

    def resized(self, plan: Plan, slot: str, p: Piece, minutes: int) -> Plan | None:
        """The plan with study block `p` cut to `minutes` (0: gone); the break after it starts when it now ends."""
        after = next((q for q in plan[slot] if q.role == "break" and q.start == p.end), None)
        new = self.drop(plan, slot, p)
        if after:
            new = self.drop(new, slot, after)
        if minutes:
            new = self.put(new, slot, replace(p, minutes=minutes))
            if new and after:
                new = self.put(new, slot, replace(after, start=after.start - timedelta(minutes=p.minutes - minutes),
                                                  minutes=after.minutes + p.minutes - minutes)) or new
        return new if new and self.trace(new)[0] else None

    def trim(self, plan: Plan) -> Plan:
        """Cut the study, and the rest, naps and extra meals, the day turned out not to need, the latest first."""
        for slot, p in reversed([(s, p) for s, p in self.pieces(plan) if p.role in ("study", "rest", "nap", "extra")]):
            without = self.resized(plan, slot, p, 0)
            if without and self.no_worse(without, plan):
                plan = without
                continue
            if p.role != "study":
                continue
            lo, hi = MIN_STUDY // 5, p.minutes // 5  # the shortest length (in fives) that costs nothing
            while lo < hi:
                mid = (lo + hi) // 2
                new = self.resized(plan, slot, p, mid * 5)
                if new and self.no_worse(new, plan):
                    hi = mid
                else:
                    lo = mid + 1
            new = self.resized(plan, slot, p, lo * 5) if lo * 5 < p.minutes else None
            if new and self.no_worse(new, plan):
                plan = new
        for slot, p in self.pieces(plan):  # a break with no study on one side of it is not a break
            if p.role == "break":
                i = plan[slot].index(p)
                before = plan[slot][i - 1].kind if i else next((g.before for g in self.gaps(plan) if g.slot == slot), "")
                after = plan[slot][i + 1].kind if i + 1 < len(plan[slot]) else ""
                if not (before == after == "study"):
                    plan = self.drop(plan, slot, p)
        return plan

    # --- energy ---

    def low_points(self, res: SimResult) -> list[datetime]:
        """When energy crosses the floor (the first few times)."""
        ticks = res.timeline
        out = [b.t for a, b in zip(ticks, ticks[1:]) if b.energy < self.floor <= a.energy]
        return out[:3]

    def lifts(self, plan: Plan, low: datetime, rest: bool) -> list[Plan]:
        """Ways to have more energy at `low`: a meal before it, a nap at home, the night started earlier,
        and (when asked) rest in place of free time."""
        out: list[Plan] = []
        if self.meals_on(plan, low) < MEALS_A_DAY:
            new = self.meal(plan, low - timedelta(minutes=MEAL_MINUTES), before=low, apart=MEAL_APART, role="extra")
            if new:
                out.append(new)
        home = self.board.home
        nights = self.nights(plan)
        bed = min((p.start for _, p in nights if p.start > low), default=self.board.end)
        if not any(p.role == "nap" for _, p in self.pieces(plan)) and minutes_between(low, bed) > 120:
            for g in self.gaps(plan):
                if g.t0 >= low:
                    break
                a, b = self.room(g, home)
                b = min(b, low, bed - timedelta(minutes=NAP_BEFORE_BED))
                minutes = next((n for n in NAP_MINUTES if minutes_between(a, b) >= n), None)
                new = self.put(plan, g.slot, Piece("sleep", home, a, minutes, "nap")) if minutes else None
                if new:
                    out.append(new)
        for s, p in nights:
            by = min(120, _ceil5(minutes_between(low, p.start)) + 15)
            if 0 < minutes_between(low, p.start) <= 120:
                new = self.put(plan, s, replace(p, start=p.start - timedelta(minutes=by), minutes=p.minutes + by), instead_of=p)
                if new:
                    out.append(new)
        if rest and sum(p.role == "rest" for _, p in self.pieces(plan)) < MAX_RESTS:
            free = sorted((g for g in self.gaps(plan) if g.t0 < low and "rest" not in (g.before, g.after)),
                          key=lambda g: (-minutes_between(g.t0, min(g.t1, low)), g.t0))
            for g in free[:2]:
                end = min(g.t1 - timedelta(minutes=self.walk(g.p0, g.p1)), low)
                minutes = min(REST_MAX_MINUTES, minutes_between(g.t0, end))
                if minutes >= REST_MIN_MINUTES:
                    new = self.put(plan, g.slot, Piece("rest", g.p0, end - timedelta(minutes=minutes), minutes, "rest"))
                    if new:
                        out.append(new)
        return out

    def energize(self, plan: Plan, rest: bool = False) -> Plan:
        """While energy crosses the floor, do before a low point whatever the engine credits most. Rest in place
        of free time is kept only when it gets the day all the way above the floor: it is not padding."""
        start = plan
        for _ in range(ENERGY_STEPS):
            lows = self.low_points(self.sim(plan))
            now = self.score(plan)[:-1]
            cands = [c for low in lows for c in self.lifts(plan, low, rest)]
            best = max(((self.score(c)[:-1], -n) for n, c in enumerate(cands)), default=None)
            if best is None or best[0] <= now:
                break
            plan = cands[-best[1]]
        if rest and self.under(self.sim(plan)) > 0:
            return self.energize(start)
        return plan

    # --- repair ---

    def unreject(self, plan: Plan, r: Rejected) -> list[Plan]:
        """A refused move: shorter or earlier by the minutes it runs over, or gone."""
        pieces = self.pieces(plan)
        if r.index >= len(pieces):
            return []
        slot, p = pieces[r.index]
        out = [self.drop(plan, slot, p)]
        if r.minutes and p.minutes - r.minutes >= self.cfg.min_move_minutes:
            out.append({**plan, slot: tuple(replace(p, minutes=p.minutes - r.minutes) if q == p else q for q in plan[slot])})
        if r.minutes:
            out.append({**plan, slot: tuple(replace(p, start=p.start - timedelta(minutes=r.minutes)) if q == p else q for q in plan[slot])})
        return out

    def repair(self, plan: Plan) -> Plan:
        """While the plan does not win: the repairs that match what the engine says is wrong, best one kept."""
        deadlines = {d.id: d for d in self.board.deadlines}
        for _ in range(MAX_REPAIRS):
            res = self.sim(plan)
            if res.outcome.won and self.bad(res) <= self.base_bad:
                break
            cands: list[Plan] = []
            for r in res.rejected:
                cands += self.unreject(plan, r)
            for c in res.outcome.checks:
                if c.ok:
                    continue
                if c.id == "meals":
                    cands += [m for m in (self.meal(plan, t) for t in [self.board.start, *(t for t, _ in self.meal_wishes())]) if m]
                elif c.id == "sleep":
                    cands += self.more_sleep(plan)
                elif c.id in deadlines:
                    todo = res.carry.work_left.get(c.id, 0.0)
                    cands += [new for _, new in self.study_spots(plan, deadlines[c.id], todo)]
                    cands += self.bedtimes(plan, (30, 60, 120))  # a later night makes room; the sleep check still has to hold
            k = self.key(res, plan)
            scored = [(self.key(self.sim(c), c), n) for n, c in enumerate(cands)]
            best = max(scored, key=lambda x: (x[0], -x[1]), default=None)
            if best is None or best[0] <= k:
                break
            plan = cands[best[1]]
        return plan

    # --- improve ---

    def with_event(self, plan: Plan) -> Plan:
        """One optional event the board offers, when it fits in open time and the engine likes the day no less."""
        for e in self.board.events:
            if not e.place or e.place in self.board.closed or minutes_between(e.start, e.end) > EVENT_MAX_MINUTES:
                continue
            for g in self.gaps(plan):
                a, b = self.room(g, e.place)
                a, b = max(a, e.start), min(b, e.end)
                minutes = minutes_between(a, b)
                if minutes < max(30, minutes_between(e.start, e.end) // 2):
                    continue
                new = self.put(plan, g.slot, Piece("event", e.place, a, minutes, "event", event=e.id))
                if new and self.no_worse(new, plan):
                    return new
        return plan

    def with_breaks(self, plan: Plan) -> Plan:
        """A break shown between two study blocks at the same place."""
        for s in self.board.slots:
            for p, q in zip(plan[s.id], plan[s.id][1:]):
                gap = minutes_between(p.end, q.start)
                if p.kind == q.kind == "study" and p.place == q.place and cfg_min(self.cfg) <= gap <= 2 * self.pause:
                    new = self.put(plan, s.id, Piece("rest", p.place, p.end, gap, "break"))
                    if new and self.no_worse(new, plan):
                        plan = new
        return plan

    def clean(self, plan: Plan) -> Plan:
        """Nothing the engine refuses is handed back."""
        for _ in range(MAX_REPAIRS):
            res = self.sim(plan)
            pieces = self.pieces(plan)
            refused = [pieces[r.index] for r in res.rejected if r.index < len(pieces)]
            if not refused:
                break
            for slot, p in refused:
                plan = self.drop(plan, slot, p)
        return plan

    # --- when it cannot win ---

    def explain(self, res: SimResult) -> list[str]:
        """Why each failing check cannot be met, from the board itself: what the open time can hold at most."""
        cfg, board = self.cfg, self.board
        deadlines = {d.id: d for d in board.deadlines}
        out: list[str] = []
        for c in res.outcome.checks:
            if c.ok:
                continue
            if c.id == "meals":
                plan, fit = self.empty(), 0
                while fit < cfg.win.meals:
                    plan = self.meal(plan, board.start)
                    if plan is None:
                        break
                    fit += 1
                eaten = board.initial.meals + sum(b.activity == "eat" for b in board.blocks)
                if eaten + fit < cfg.win.meals:
                    room = ("no open slot can still hold a meal" if fit == 0 else
                            f"only {fit} more fit{'s' if fit == 1 else ''} in the open time before {hm(board.end)}")
                    out.append(f"cannot reach {cfg.win.meals} meals: {eaten} eaten so far and {room}")
                else:
                    out.append(f"could not fit {cfg.win.meals} meals around the rest of this day ({c.value:g} planned)")
            elif c.id == "sleep":
                most = sum(p.minutes for _, p in self.nights(self.night(self.empty(), early=120))) / 60
                if most < cfg.win.sleep_hours:
                    out.append(f"cannot reach {cfg.win.sleep_hours:g} h of sleep: the night has room for {most:.1f} h between the fixed blocks")
                else:
                    out.append(f"could not fit {cfg.win.sleep_hours:g} h of sleep around the rest of this day ({c.value:.1f} h planned)")
            elif c.id in deadlines:
                d = deadlines[c.id]
                left = res.carry.work_left.get(d.id, 0.0)
                until = min(d.due, board.end)
                open_minutes = 0
                for g in self.gaps(self.empty()):
                    rooms = [self.room(g, p) for p in self.places("study")]
                    open_minutes += max([minutes_between(a, min(b, until)) for a, b in rooms] + [0])
                open_minutes += sum(minutes_between(b.start, min(b.end, until)) for b in board.blocks if b.activity == "study" and b.start < until)
                if d.due <= board.end:
                    out.append(f"cannot finish {d.title} by {d.due:%a} {hm(d.due)}: it needs {d.work_left:.0f} focused minutes and "
                               f"{open_minutes} open minutes are left before it is due ({left:.0f} would still be undone)")
                else:
                    out.append(f"cannot get {d.title} on track today: it needs {d.work_left:.0f} focused minutes, {open_minutes} open "
                               f"minutes are left on this board and {left:.0f} would still be undone (due {d.due:%a} {hm(d.due)})")
        return out

    def honest(self, res: SimResult) -> SimResult:
        """A plan that does not win says so, with why: the summary and the reasons the planner shows."""
        why = self.explain(res)
        summary = (f"No plan from {hm(self.board.start)} wins; this is the best attempt. "
                   + " ".join(f"{w[0].upper()}{w[1:]}." for w in why))
        outcome = res.outcome.model_copy(update={"summary": summary.strip(), "reasons": [*why, *res.outcome.reasons]})
        return res.model_copy(update={"outcome": outcome})


def cfg_min(cfg: DayRunConfig) -> int:
    return max(5, min(cfg.min_move_minutes, BREAK_MINUTES))


def choose(cfg: DayRunConfig, w: Walks, board: Board, bias: frozenset[str] = frozenset(),
           preferred_block_minutes: int | None = None) -> tuple[list[Move], SimResult]:
    """The plan for this board and its simulation: a winning one whenever the search finds one, else the best
    attempt with `outcome.summary` and `outcome.reasons` saying what cannot fit.

    `bias` carries yesterday's review ("earlier_sleep": to bed half an hour earlier; "earlier_study": study
    already goes in the earliest block that holds). `preferred_block_minutes` is the longest study block
    (default 90): the person's own longest focus stretch, when something has measured it."""
    day = _Day(cfg, w, board, bias, preferred_block_minutes)
    plan = day.meals(day.night(day.empty()))
    plan = day.energize(plan)
    plan = day.study(plan)
    plan = day.repair(plan)
    plan = day.trim(plan)
    plan = day.energize(plan, rest=day.sim(plan).outcome.won)
    plan = day.with_breaks(day.with_event(plan))
    plan = day.clean(plan)
    result = day.sim(plan)
    return day.moves(plan), result if result.outcome.won else day.honest(result)
