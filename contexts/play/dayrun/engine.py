"""The Day Run simulation (S4): run a plan through a model of the day and see what happens.

    result = simulate(cfg, walks, board, moves)

Two passes, both pure (no clock, no randomness, no I/O):

1. Schedule. Walk the board in time order. Each move is placed after the walk
   to its place (A* over the place pack); a move that would make you late for
   the next fixed block is rejected with the minutes ("you miss the 12:00 ...
   by 6 minutes") and its time stays free. Walking is real time taken out of
   the slot.
2. Dynamics. Step the meters in 15-minute ticks over that schedule: energy,
   focus and the work left on each deadline. Rates are in config/dayrun.yaml.

`repair=True` (hard mode) lets the engine keep a broken plan standing: a move
at a place that closed is sent to the nearest open one, and a move that now
runs late is cut short instead of dropped.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

from contexts.play.dayrun.board import hm, local, minutes_between, risk_of
from contexts.play.dayrun.config import DayRunConfig
from contexts.play.dayrun.models import (
    Board,
    Carry,
    Change,
    Check,
    FixedBlock,
    Move,
    Outcome,
    PlannedBlock,
    Rejected,
    RouteLeg,
    RouteTaken,
    SimResult,
    Tick,
    Totals,
)
from contexts.play.dayrun.routes import Trip, Walks


@dataclass
class Seg:
    """A stretch of one activity at one place."""

    start: datetime
    end: datetime
    activity: str
    place: str
    deadline: str | None = None
    friend: str | None = None
    factor: float = 1.0  # study: how good the place is for work
    boost: float = 0.0  # eat: energy per minute from the food
    meal: bool = False
    counts: bool = False  # sleep: long enough to count as sleep


class _Schedule:
    def __init__(self, cfg: DayRunConfig, w: Walks, board: Board, repair: bool) -> None:
        self.cfg, self.w, self.board, self.repair = cfg, w, board, repair
        self.segs: list[Seg] = []
        self.routes: list[RouteTaken] = []
        self.legs: list[RouteLeg] = []
        self.rejected: list[Rejected] = []
        self.changes: list[Change] = []
        self.notes: list[str] = []
        self.blocks: list[PlannedBlock] = []
        self.lates = 0
        self.place = board.start_place
        self.clock = board.start
        self.last_meal: datetime | None = None

    # --- building blocks ---

    def emit(self, start: datetime, end: datetime, activity: str, place: str, **kw) -> Seg | None:
        start, end = max(start, self.clock), min(end, self.board.end)
        if self.clock < start:
            self.segs.append(Seg(self.clock, start, "idle", self.place))
        if end <= start:
            self.clock = max(self.clock, start)
            return None
        seg = Seg(start, end, activity, place, **kw)
        self.segs.append(seg)
        self.clock = end
        return seg

    def go(self, trip: Trip | None, minutes: int, depart: datetime, purpose: str, move: int | None) -> datetime:
        """Walk `trip` leaving at `depart`; returns the arrival time."""
        if trip is None:
            return depart
        arrive = depart + timedelta(minutes=minutes)
        self.emit(depart, arrive, "walk", trip.places[-1])
        index = len(self.routes)
        self.routes.append(RouteTaken(index=index, purpose=purpose, move=move, start=trip.places[0], end=trip.places[-1],
                                      depart=depart, arrive=arrive, minutes=trip.minutes, places=trip.places))
        self.legs += [RouteLeg(**leg.model_dump(), route=index, depart=depart) for leg in trip.legs]
        self.place = trip.places[-1]
        return arrive

    def eat(self, start: datetime, end: datetime, place: str) -> None:
        cfg = self.cfg
        minutes = minutes_between(start, end)
        spaced = self.last_meal is None or minutes_between(self.last_meal, start) >= cfg.meal_gap_minutes
        meal = minutes >= cfg.meal_min_minutes and spaced
        boost = cfg.meal_energy * (1 if meal else cfg.snack_factor) / max(minutes, 1)
        if self.emit(start, end, "eat", place, boost=boost, meal=meal) and meal:
            self.last_meal = start

    def study(self, start: datetime, end: datetime, place: str, deadline: str | None, friend: str | None, move: int | None) -> None:
        factor = self.cfg.study_place_factor.get(self.w.kind(place), self.cfg.study_place_factor.get("default", 0.7))
        beside = sorted((max(f.start, start), min(f.end, end)) for f in self.board.friends
                        if friend and f.member.lower() == friend.lower() and f.place == place and f.start < end and f.end > start)
        if friend and not beside:
            text = (f"{friend} is not at {self.w.name(place)} between {hm(start)} and {hm(end)}: no boost from "
                    f"{self.cfg.vocabulary.activity}ing together")
            self.notes.append(text)
            if self.repair:
                self.changes.append(Change(move=move, kind="friend_gone", text=text))
        t = start
        for a, b in beside:
            self.emit(t, a, "study", place, deadline=deadline, factor=factor)
            self.emit(a, b, "study", place, deadline=deadline, factor=factor, friend=friend)
            t = max(t, b)
        self.emit(t, end, "study", place, deadline=deadline, factor=factor)

    # --- the day ---

    def run(self, moves: list[Move]) -> None:
        board = self.board
        slot_ids = {s.id for s in board.slots}
        for i, m in enumerate(moves):
            if m.slot not in slot_ids:
                self.rejected.append(Rejected(index=i, move=m, reason=f"there is no open slot {m.slot!r} on this board"))
        for nxt in [*board.blocks, None]:
            slot = next((s for s in board.slots if s.next_block == (nxt.id if nxt else None)), None)
            if slot:
                for i, m in enumerate(moves):
                    if m.slot == slot.id:
                        self.move(i, m, slot.end, slot.to_place, nxt)
            if nxt is None:
                break
            minutes, trip = self.w.walk(self.place, nxt.place)
            if trip:
                arrive = self.go(trip, minutes, max(self.clock, nxt.start - timedelta(minutes=minutes)), f"to {nxt.title}", None)
                if arrive > nxt.start:
                    late = minutes_between(nxt.start, arrive)
                    text = f"you reach the {hm(nxt.start)} {nxt.title} {late} minutes late: the walk from {self.w.name(trip.places[0])} takes {minutes}"
                    self.lates += 1
                    self.notes.append(text)
                    self.changes.append(Change(kind="late", text=text))
            self.fixed(nxt)
        self.emit(board.end, board.end, "idle", self.place)

    def fixed(self, b: FixedBlock) -> None:
        start, place = max(b.start, self.clock), b.place or self.place
        if start >= b.end:
            return
        if b.activity == "eat":
            self.eat(start, b.end, place)
        elif b.activity == "study":
            self.study(start, b.end, place, b.deadline, None, None)
        else:
            counts = minutes_between(start, b.end) >= self.cfg.sleep.min_block_minutes
            self.emit(start, b.end, b.activity, place, counts=counts)
        self.place = place
        self.blocks.append(PlannedBlock(id=b.id, kind=b.activity, title=b.title, start=start, end=b.end,
                                        minutes=minutes_between(start, b.end), place=place, place_name=self.w.name(place), fixed=True))

    def move(self, i: int, m: Move, slot_end: datetime, to_place: str | None, nxt: FixedBlock | None) -> None:
        cfg, w, board = self.cfg, self.w, self.board

        def reject(reason: str, minutes: int | None = None) -> None:
            self.rejected.append(Rejected(index=i, move=m, reason=reason, minutes=minutes))
            if self.repair:
                self.changes.append(Change(move=i, kind="rejected", text=reason))

        event = next((e for e in board.events if e.id == m.event), None) if m.kind == "event" else None
        if m.kind == "event" and event is None:
            return reject(f"there is no event {m.event!r} on this board")
        try:
            target = w.ref(m.place) if m.place else board.home if m.kind == "sleep" else (event.place if event else None) or self.place
        except ValueError as exc:
            return reject(str(exc))
        if m.kind == "eat" and w.kind(target) not in cfg.kinds["eat"]:
            return reject(f"{w.name(target)} is not a place to eat")
        if target in board.closed and m.kind in ("eat", "study"):
            kinds = cfg.kinds[m.kind]
            packs = list(dict.fromkeys([board.pack, board.home.split(".")[0]]))
            open_ones = [p for p in w.of_kind(kinds, packs) if p not in board.closed]
            if not self.repair or not open_ones:
                return reject(f"{w.name(target)} is closed today")
            best = min(open_ones, key=lambda p: (w.walk(self.place, p)[0] + w.walk(p, to_place)[0], p))
            there, back = w.walk(self.place, best)[0], w.walk(best, to_place)[0]
            onward = f"on to the {hm(nxt.start)} {nxt.title}" if nxt and back else ""
            walk = (f", {there} minutes from {w.name(self.place)}" + (f" and {back} {onward}" if onward else "") if there
                    else " (you are already there)" + (f", {back} minutes {onward}" if onward else ""))  # never "0 minutes from" itself
            self.changes.append(Change(move=i, kind="rerouted", text=f"{w.name(target)} is closed: re-routed to {w.name(best)}{walk}"))
            target = best

        depart = max(self.clock, local(cfg, m.start)) if m.start else self.clock
        there, trip = w.walk(self.place, target)
        if event:
            depart = max(depart, event.start - timedelta(minutes=there))
        if depart >= slot_end:
            return reject(f"it would start at {hm(depart)}, after the slot ends at {hm(slot_end)}")
        begin = depart + timedelta(minutes=there)
        back, _ = w.walk(target, to_place)
        latest = slot_end - timedelta(minutes=back)  # when you have to leave for what comes next
        if event:
            begin = max(begin, event.start)
            end = min(event.end, begin + timedelta(minutes=m.minutes)) if m.minutes else min(event.end, latest)
            if begin >= event.end:
                return reject(f"{event.title} is over by the time you get there ({hm(begin)})")
        else:
            end = begin + timedelta(minutes=m.minutes) if m.minutes else latest
        late = minutes_between(latest, end)
        if late > 0 or end <= begin:
            late = max(late, minutes_between(latest, begin))
            if late <= 0:
                reason = f"there is no time left in the slot by {hm(begin)}"
            elif nxt:
                reason = f"you miss the {hm(nxt.start)} {nxt.title} by {late} minutes"
                if there or back:  # say where the minutes went
                    reason += f" (walking: {there} there, {back} on to {w.name(to_place) if to_place else 'it'})"
            else:
                reason = f"it runs {late} minutes past the end of this board ({hm(board.end)})"
            room = minutes_between(begin, latest)
            if not self.repair or room < cfg.min_move_minutes:
                return reject(reason, late)
            self.changes.append(Change(move=i, kind="shortened", text=f"cut from {minutes_between(begin, end)} to {room} minutes: otherwise {reason}"))
            end = latest

        self.go(trip, there, depart, f"to {w.name(target)} ({m.kind})", i)
        friend = None
        if m.kind == "study":
            self.study(begin, end, target, m.deadline, m.with_friend, i)
            friend = m.with_friend if any(s.friend for s in self.segs if s.start >= begin and s.end <= end) else None
            title = f"{cfg.vocabulary.activity.capitalize()} at {w.name(target)}" + (f" with {friend}" if friend else "")
        elif m.kind == "eat":
            self.eat(begin, end, target)
            title = f"Eat at {w.name(target)}"
        elif m.kind == "sleep":
            self.emit(begin, end, "sleep", target, counts=minutes_between(begin, end) >= cfg.sleep.min_block_minutes)
            title = "Sleep"
        elif m.kind == "event":
            self.emit(begin, end, "event", target)
            title = event.title
        else:
            self.emit(begin, end, "rest", target)
            title = f"Rest at {w.name(target)}"
        self.place = target
        self.blocks.append(PlannedBlock(id=f"m{i}", kind=m.kind, title=title, start=begin, end=end, minutes=minutes_between(begin, end),
                                        place=target, place_name=w.name(target), move=i, with_friend=friend))


def simulate(cfg: DayRunConfig, w: Walks, board: Board, moves: list[Move], repair: bool = False) -> SimResult:
    plan = _Schedule(cfg, w, board, repair)
    plan.run(moves)
    segs = [s for s in plan.segs if s.end > s.start]

    # --- dynamics, in ticks ---
    init = board.initial
    energy, fresh, hungry, debt = init.energy, init.focus, init.hungry_minutes, init.sleep_debt
    work = {d.id: d.work_left for d in board.deadlines}
    due = {d.id: d.due for d in board.deadlines}
    missed: set[str] = set()
    sleep_min = focused = walked = 0.0
    study_min = 0
    # Meals already eaten today (Habits, on a board that starts mid-day) count toward the day's meal target;
    # a board that starts at wake has none. `eaten` is the meals inside this board's window.
    before = max(0, init.meals)
    eaten = 0
    energy_min = energy

    def factor() -> float:  # how much of your freshness low energy lets you use
        return max(cfg.focus_floor, min(1.0, energy / cfg.energy_full_focus))

    def risks(t: datetime) -> dict[str, float]:
        return {d: 1.0 if d in missed else risk_of(cfg, left, t, due[d]) for d, left in work.items()}

    def row(t: datetime, activity: str, place: str) -> Tick:
        return Tick(t=t, energy=round(energy, 1), focus=round(fresh * factor(), 1), deadline_risk=max(risks(t).values(), default=0.0),
                    work_left={d: round(v, 1) for d, v in work.items()}, activity=activity, place=place)

    timeline = [row(board.start, "idle", board.start_place)]
    step = timedelta(minutes=cfg.tick_minutes)
    t, n = board.start, 0
    while t < board.end:
        tick_end = min(t + step, board.end)
        spent: dict[tuple[str, str], float] = {}
        while n < len(segs) and segs[n].end <= t:
            n += 1
        k = n
        while k < len(segs) and segs[k].start < tick_end:
            seg = segs[k]
            a, b = max(seg.start, t), min(seg.end, tick_end)
            m = (b - a).total_seconds() / 60
            act = seg.activity
            if act == "study":
                open_ones = [d for d in work if work[d] > 0 and d not in missed and due[d] > a]
                target = seg.deadline if seg.deadline in open_ones else max(open_ones, key=lambda d: (risks(a)[d], d), default=None)
                if target:
                    done = min(work[target], m * fresh * factor() / 100 * seg.factor * (cfg.friend_boost if seg.friend else 1.0))
                    work[target] -= done
                    focused += done
                study_min += round(m)
            rate = cfg.energy_per_hour.get(act, cfg.energy_per_hour["idle"])
            if act not in ("sleep", "eat") and hungry >= cfg.hunger_after_minutes:
                rate += cfg.hunger_energy_per_hour
            if rate < 0:
                rate *= 1 + cfg.debt_drain_per_hour * debt
            energy = max(0.0, min(100.0, energy + rate * m / 60 + seg.boost * m))
            drift = cfg.focus_per_hour.get(act, 0.0) * (cfg.friend_decay if act == "study" and seg.friend else 1.0)
            fresh = max(0.0, min(100.0, fresh + drift * m / 60))
            if act == "sleep":
                sleep_min += m if seg.counts else 0
            else:
                hungry += m
            if act == "walk":
                walked += m
            if seg.meal and b == seg.end:
                eaten += 1
                hungry = 0.0
            energy_min = min(energy_min, energy)
            spent[act, seg.place] = spent.get((act, seg.place), 0.0) + m
            k += 1
        t = tick_end
        missed |= {d for d in work if work[d] > 0 and due[d] <= t}
        (activity, place), _ = max(spent.items(), key=lambda kv: kv[1], default=(("idle", plan.place), 0))
        timeline.append(row(t, activity, place))

    # --- outcome ---
    sleep_h = round(sleep_min / 60, 2)
    final = risks(board.end)
    checks: list[Check] = []
    for d in board.deadlines:
        left, risk = round(work[d.id], 1), final[d.id]
        status = "done" if left <= 0 else "missed" if d.id in missed else "on_track" if risk <= cfg.on_track_risk else "at_risk"
        hours = max(0, round((d.due - board.end).total_seconds() / 3600))
        text = {"done": f"{d.title}: done",
                "missed": f"{d.title}: due at {hm(d.due)} with {left:.0f} focused minutes still to do",
                "on_track": f"{d.title}: on track, {left:.0f} focused minutes left with {hours} h to go (risk {risk:.2f})",
                "at_risk": f"{d.title}: at risk, {left:.0f} focused minutes left with {hours} h to go (risk {risk:.2f}, needs {cfg.on_track_risk:.2f} or less)"}[status]
        checks.append(Check(id=d.id, ok=status in ("done", "on_track"), value=risk, target=cfg.on_track_risk, text=text, status=status))
    checks.append(Check(id="sleep", ok=sleep_h >= cfg.win.sleep_hours, value=sleep_h, target=cfg.win.sleep_hours,
                        text=f"Sleep: {sleep_h:.1f} h ({cfg.win.sleep_hours:g} h needed)"))
    meals = before + eaten
    so_far = f", {before} of them already eaten before this board starts" if before else ""
    checks.append(Check(id="meals", ok=meals >= cfg.win.meals, value=meals, target=cfg.win.meals,
                        text=f"Meals: {meals} ({cfg.win.meals} needed{so_far})"))
    won = all(c.ok for c in checks)
    worst = max(final.values(), default=0.0)
    stars = 0
    if won:
        stars = 1 + (energy_min >= cfg.stars.energy_floor and plan.lates == 0) + (sleep_h >= cfg.stars.sleep_hours and worst <= cfg.stars.risk)
    reasons = [c.text for c in checks if not c.ok]
    points = round(40 * (1 - worst) + 25 * min(1.0, sleep_h / cfg.sleep.need_hours) + 15 * min(1.0, meals / max(cfg.win.meals, 1))
                   + 20 * energy / 100)
    summary = (f"Win, {stars} star{'s' if stars != 1 else ''}: every deadline on track, {sleep_h:.1f} h of sleep, {meals} meals."
               if won else "The plan fails. " + " ".join(f"{r}." for r in reasons))
    debt_end = max(0.0, min(cfg.max_sleep_debt_hours, debt + cfg.sleep.need_hours - sleep_h))
    return SimResult(
        start=board.start, end=board.end, tick_minutes=cfg.tick_minutes,
        outcome=Outcome(won=won, stars=stars, reasons=reasons, checks=checks, summary=summary), stars=stars,
        timeline=timeline, legs=plan.legs, routes=plan.routes, rejected=sorted(plan.rejected, key=lambda r: r.index),
        blocks=sorted(plan.blocks, key=lambda b: b.start), changes=plan.changes, notes=plan.notes,
        totals=Totals(study_minutes=study_min, focused_minutes=round(focused, 1), walk_minutes=round(walked, 1), sleep_hours=sleep_h,
                      meals=meals, meals_before=before, energy_end=round(energy, 1), energy_min=round(energy_min, 1), points=points),
        carry=Carry(energy=round(energy, 1), focus=round(fresh, 1), sleep_debt=round(debt_end, 2), hungry_minutes=round(hungry, 1),
                    place=plan.place, work_left={d: round(v, 1) for d, v in work.items()}),
    )
