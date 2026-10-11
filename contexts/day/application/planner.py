"""The morning plan and the evening review, unasked (D1).

    planner = DayPlanner(dayrun, clock, habits=habits, memory=memory, presence=presence, hub=hub)

Once per local day, at wake time (or when the box comes up after it with no plan for today), the planner
builds today's board from what the box already knows, through Day Run's service (calendar and campus
events and Canvas deadlines from Memory's items, friends from the Circle, last night's sleep and today's
meals from Habits), chooses a plan with `contexts.day.domain.choose`, and publishes `day.plan.proposed`
with one approval card for the calendar blocks. Every input is in `sources`, so the phone can say why.

At bedtime (or when Habits closes the day) it runs Day Run's reality review against the plan and publishes
`day.reviewed`; the next morning's choice is biased by it (a study slip: start earlier; a sleep slip: to
bed earlier). With the study coach (contexts/day/application/coach.py) the review carries the day's
sessions, and the next plan reads the coach's advice back as a `review` source and sizes its blocks of the
main activity to the longest unbroken stretch.

State (which day has a plan, which card belongs to which plan, what was committed or reviewed) rebuilds
from the `day.*` and workspace events in the log: those handlers replay. Deciding and proposing is live only.
"""

import logging
from datetime import date, datetime, timedelta
from uuid import uuid4

from contexts.day.domain.choose import choose
from contexts.day.domain.clock import DayClock
from contexts.day.domain.optional import BLOCK_PREFIX as OPTIONAL_BLOCK
from contexts.day.domain.optional import ORIGIN as OPTIONAL_ORIGIN
from contexts.day.domain.optional import block_id as optional_block
from contexts.play.dayrun.board import local
from contexts.play.dayrun.engine import simulate
from contexts.play.dayrun.models import Board, InitialOverride, Measured, Move, PlannedBlock, SimResult, Totals
from contexts.play.dayrun.review import reality
from contexts.play.dayrun.service import DayRun
from contracts.agent import ActionProposed
from contracts.day import (
    DayPlanCommitted,
    DayPlanProposed,
    DayReplanProposed,
    DayReviewed,
    MoveIn,
    PlanMove,
    PlanOutcome,
    PlanSource,
)
from contracts.events import Event
from contracts.experience import ApprovalDecided
from contracts.habits import DayClosed
from contracts.memory import Item
from contracts.workspace import ActionCompleted, ApprovalRequested

log = logging.getLogger("atlas.day")
CALENDAR_WRITE = "calendar.create_event"
MAX_BECAUSE = 4  # sources named in the plan's text


def plan_moves(result: SimResult, moves: list[Move]) -> list[PlanMove]:
    return [PlanMove(id=b.id, kind=b.kind, title=b.title, start=b.start, end=b.end, minutes=b.minutes, place=b.place,
                     place_name=b.place_name, fixed=b.fixed,
                     deadline=moves[b.move].deadline if b.move is not None else None, with_friend=b.with_friend,
                     origin=OPTIONAL_ORIGIN if b.id.startswith(OPTIONAL_BLOCK) else None)  # an optional event the person said yes to
            for b in result.blocks]


def plan_outcome(result: SimResult) -> PlanOutcome:
    t = result.totals
    worst = max((c.value for c in result.outcome.checks if c.status), default=0.0)
    return PlanOutcome(won=result.outcome.won, stars=result.stars, points=t.points, summary=result.outcome.summary,
                       sleep_hours=t.sleep_hours, meals=t.meals, study_minutes=t.study_minutes, energy_min=t.energy_min,
                       deadline_risk=round(worst, 3))


def hm(t: datetime) -> str:
    t = t.astimezone(t.tzinfo)
    return f"{t.hour % 12 or 12}:{t:%M} {t:%p}".replace(":00 ", " ")


def card_events(moves: list[PlanMove]) -> list[dict]:
    return [{"title": f"Day Run: {m.title}", "start": m.start.isoformat(), "end": m.end.isoformat(), "location": m.place_name}
            for m in moves if not m.fixed]


class DayPlanner:
    def __init__(self, dayrun: DayRun, clock: DayClock, habits=None, memory=None, presence=None, hub=None) -> None:
        self.dayrun, self.clock = dayrun, clock
        self.habits, self.memory, self.presence, self.hub = habits, memory, presence, hub
        self.coach = None  # the study coach (DayLoop sets it): sessions for the review, advice for the next plan
        # A port the composition root may fill: (day, now) -> one clause about the day's sorting, or None (Waste Watch's
        # story, contexts/impact). The review's text and the evening narration carry it; this context never imports that one.
        self.waste_clause = None
        # --- state, rebuilt from the log ---
        self.plans: dict[date, DayPlanProposed] = {}  # the morning plan of each day
        self.current: dict[date, DayPlanProposed | DayReplanProposed] = {}  # what the day is measured against
        self.pending: dict[date, DayReplanProposed] = {}  # a re-plan waiting for the phone
        self.earlier: dict[date, list[DayPlanProposed | DayReplanProposed]] = {}  # the plans `current` replaced, oldest first
        self.cards: dict[str, tuple[date, str]] = {}  # action id -> (day, plan id)
        self.committed: set[str] = set()
        self.reviews: dict[date, DayReviewed] = {}
        self.expires: dict[str, datetime] = {}
        self.card_status: dict[str, str] = {}
        self.last_replan: dict[date, datetime] = {}
        self.denied_at: dict[date, datetime] = {}
        self.skipped: set[date] = set()  # days the box came up too late to plan
        self.ids: set[str] = set()
        self.replans_for: dict[tuple[date, str], int] = {}
        self.last_event: datetime | None = None  # the latest live event this planner saw (not replayed: a tick needs a sign of life)
        self._edited = False

    # --- state handlers (replay=True) ---

    def fresh_id(self, plan_id: str) -> str:
        n = 1
        while plan_id in self.ids:
            n += 1
            plan_id = f"{plan_id.split('#')[0]}#{n}"
        return plan_id

    def on_plan_proposed(self, e: DayPlanProposed) -> None:
        self.ids.add(e.plan_id)
        self.plans[e.day] = e
        self._replace(e.day, e)
        self.pending.pop(e.day, None)  # a new plan supersedes a re-plan waiting on the old one
        if e.action_id:
            self.cards[e.action_id] = (e.day, e.plan_id)

    def on_replan_proposed(self, e: DayReplanProposed) -> None:
        self.ids.add(e.plan_id)
        self.pending[e.day] = e
        self.last_replan[e.day] = e.ts
        if e.for_event:
            self.replans_for[e.day, e.for_event] = self.replans_for.get((e.day, e.for_event), 0) + 1
        if e.action_id:
            self.cards[e.action_id] = (e.day, e.plan_id)

    def on_plan_committed(self, e: DayPlanCommitted) -> None:
        self.committed.add(e.plan_id)
        replan = self.pending.get(e.day)
        if replan and replan.plan_id == e.plan_id:
            self._replace(e.day, self.pending.pop(e.day))

    def _replace(self, day: date, plan: DayPlanProposed | DayReplanProposed) -> None:
        """`plan` is now what the day is measured against; the one before it is kept, because the hours it
        covered were already lived (the coach still judges them: contexts/day/application/coach.py)."""
        old = self.current.get(day)
        if old is not None and old.plan_id != plan.plan_id:
            self.earlier.setdefault(day, []).append(old)
        self.current[day] = plan

    def on_reviewed(self, e: DayReviewed) -> None:
        self.reviews[e.day] = e

    def on_approval_requested(self, e: ApprovalRequested) -> None:
        if e.action_id in self.cards:
            self.expires[e.action_id] = e.expires_at

    def on_action_completed(self, e: ActionCompleted) -> None:
        if e.action_id not in self.cards:
            return
        self.card_status[e.action_id] = e.status
        day, plan_id = self.cards[e.action_id]
        if e.status in ("denied", "expired") and self.pending.get(day) and self.pending[day].plan_id == plan_id:
            self.pending.pop(day)
            if e.status == "denied":
                self.denied_at[day] = e.ts

    # --- live handlers ---

    def on_decided(self, e: ApprovalDecided) -> list[Event]:
        """The phone said yes to a plan's card: the plan is committed (the calendar write is Workspace's business)."""
        if e.decision != "approve" or e.action_id not in self.cards:
            return []
        if (expires := self.expires.get(e.action_id)) and e.ts >= expires:
            return []
        day, plan_id = self.cards[e.action_id]
        if plan_id in self.committed:
            return []
        return [DayPlanCommitted(event_id=f"day-committed-{plan_id}", ts=e.ts, day=day, plan_id=plan_id, action_id=e.action_id)]

    def on_event(self, e: Event) -> list[Event]:
        """Any live event after wake with no plan for its day: plan now, at the event's time (and review a day
        whose bedtime has passed)."""
        self.last_event = max(self.last_event or e.ts, e.ts)
        return self.maybe_plan(e.ts) + self.maybe_review(e.ts)

    def on_day_closed(self, e: DayClosed) -> list[Event]:
        return self.maybe_review(e.ts, closing=e.day)

    def tick(self, now: datetime) -> list[Event]:
        """The ticker: a box that comes up after wake plans once something has happened today (a camera or a
        connector event, or Habits' event time): an idle box with an empty day has nothing to plan around."""
        return (self.maybe_plan(now) if self.alive(now) else []) + self.maybe_review(now)

    def alive(self, now: datetime) -> bool:
        day = self.clock.day_of(now)
        seen = [t for t in (self.last_event, getattr(self.habits, "now", None)) if t is not None]
        return any(self.clock.day_of(t) == day for t in seen)

    # --- the morning plan ---

    def maybe_plan(self, now: datetime) -> list[Event]:
        day = self.clock.day_of(now)
        if day in self.plans or day in self.skipped or now < self.clock.wake_at(day):
            return []
        if now >= self.clock.bed_at(day):  # the day is over before the box saw it: nothing to plan
            self.skipped.add(day)
            return []
        try:
            return self.plan(now)
        except Exception:  # a broken source must not stop the box; try again on the next tick
            log.exception("the morning plan failed")
            return []

    def plan(self, now: datetime, remake: bool = False, moves: list[Move] | None = None) -> list[Event]:
        """Today's plan, from the real board at `now`: `day.plan.proposed` and its approval card.

        `moves` are the person's own (edited in Day Run): they replace the chosen ones, simulated as given. The
        plan is a starting point, never a decision: it can be remade or replaced any time."""
        day = self.clock.day_of(now)
        board = self.dayrun.board(now)
        yesterday = self.reviews.get(day - timedelta(days=1))
        advice = self.advice(day)
        if moves is None:
            moves, result = choose(self.dayrun.cfg, self.dayrun.walks, board, frozenset(biases(yesterday)),
                                   preferred_block_minutes=advice[0].block_minutes if advice else None)
        else:
            result = simulate(self.dayrun.cfg, self.dayrun.walks, board, moves)
        sources, missing = self.sources(board, now, yesterday)
        if advice:  # the coach's advice, read back: what it said and that the blocks follow it
            sources.append(PlanSource(kind="review", id=f"coach:{advice[1]}", text=advice[2], origin="coach"))
        plan_id = self.fresh_id(f"plan-{day}-{local(self.dayrun.cfg, now):%H%M}")
        blocks = plan_moves(result, moves)
        outcome = plan_outcome(result)
        by = "person" if self._edited else "atlas"
        card = self.card(blocks, day, result, now, replace=remake or day in self.plans)
        text = self.text(blocks, outcome, sources, card is not None, by)
        proposed = DayPlanProposed(event_id=f"day-plan-{plan_id}", ts=now, day=day, plan_id=plan_id, start=board.start,
                                   end=board.end, moves=blocks, dayrun_moves=[MoveIn(**m.model_dump()) for m in moves],
                                   outcome=outcome, sources=sources, missing=missing, text=text,
                                   action_id=card.action_id if card else None, by=by)
        return [proposed] + ([card] if card else [])

    def advice(self, day: date):
        """(advice, plan id, one line) from the coach about the day before, or None."""
        if self.coach is None:
            return None
        try:
            return self.coach.advice(day)
        except Exception:
            log.exception("the coach's advice failed")
            return None

    def edited(self, now: datetime, moves: list[Move]) -> list[Event]:
        """The person's own version of today's plan, from Day Run's editor."""
        self._edited = True
        try:
            return self.plan(now, remake=True, moves=moves)
        finally:
            self._edited = False

    def card(self, blocks: list[PlanMove], day: date, result: SimResult, now: datetime, replace: bool = False,
             why: str | None = None, introduce: str | None = None) -> ActionProposed | None:
        """One approval card for the plan's calendar blocks. Nothing is written until the phone says yes."""
        events = card_events(blocks)
        # `introduce`: the fixed block of an optional event this plan brings in. It is written too, under its own title,
        # once: by the card of the re-plan that adds it, and by no later plan's card.
        events = [{"title": m.title, "start": m.start.isoformat(), "end": m.end.isoformat(), "location": m.place_name}
                  for m in blocks if introduce is not None and m.id == introduce] + events
        if not events:
            return None
        verb = "Replace today's Day Run blocks with" if replace else "Add"
        reason = f"{verb} {len(events)} block(s) for {day:%a %b} {day.day}: " + ", ".join(
            f"{m.title} {hm(m.start)}" for m in blocks if not m.fixed or (introduce is not None and m.id == introduce))
        if why:
            reason = f"{why} {reason}"
        reason += f" ({result.outcome.summary.rstrip('.')})"
        if self.hub is not None and self.hub.is_write(CALENDAR_WRITE):
            return self.hub.propose_write(CALENDAR_WRITE, {"events": events}, reason, now)
        reason += ". No calendar connector on this box: approving keeps this as today's plan, nothing is written"
        return ActionProposed(ts=now, action_id=uuid4().hex, kind="connector_write", target=CALENDAR_WRITE, reason=reason,
                              params={"events": events})

    def sources(self, board: Board, now: datetime, yesterday: DayReviewed | None) -> tuple[list[PlanSource], list[str]]:
        """Every input the board was built from, by id, and what the box did not have."""
        out: list[PlanSource] = []
        missing: list[str] = []
        items: dict[str, Item] = {}
        if self.memory is not None:
            try:
                lo = board.start - timedelta(days=1)
                hi = board.start + timedelta(hours=self.dayrun.cfg.window_hours + self.dayrun.cfg.deadline_horizon_hours)
                items = {i.external_id: i for i in self.memory.items(lo, hi)}
            except Exception as exc:
                missing.append(f"memory: {exc}")
        else:
            missing.append("memory: not configured")
        for b in board.blocks:
            if b.activity == "sleep":
                continue
            item = items.get(b.id)
            where = f" at {b.place_name}" if b.place_name else ""
            origin = item.connector if item else "circle" if b.id.startswith("circle:") else "calendar"  # agreed with friends
            out.append(PlanSource(kind="item", id=b.id, text=f"{b.title}, {hm(b.start)}{where}", origin=origin))
        for d in board.deadlines:
            item = items.get(d.id)
            out.append(PlanSource(kind="item", id=d.id, text=f"{d.title} due {d.due:%a} {hm(d.due)}, about {d.work_left:g} focused minutes left",
                                  origin=item.connector if item else "canvas"))
        for e in board.events:
            item = items.get(e.id)
            out.append(PlanSource(kind="item", id=e.id, text=f"{e.title}, {hm(e.start)}{' at ' + e.place_name if e.place_name else ''} (optional)",
                                  origin=item.connector if item else "campus"))
        if not board.blocks and not board.deadlines:
            missing.append("calendar: nothing on it for today")

        rows: dict[str, dict] = {}
        if self.habits is not None:
            try:
                rows = {r["id"]: r for r in self.habits.digest(now=now).get("habits", [])}
            except Exception as exc:
                missing.append(f"habits: {exc}")
        else:
            missing.append("habits: not configured")
        sleep = rows.get("sleep")
        if sleep and sleep.get("applies", True):
            origin = "watch" if sleep.get("source") == "watch" else "desk camera"
            out.append(PlanSource(kind="habit", id="sleep", text=f"slept {sleep['value']:g} h last night", origin=origin))
            if origin != "watch":
                missing.append("watch: no night measured, sleep is the empty-desk estimate")
        elif rows:
            missing.append("habits: last night's sleep not measured" + (f" ({sleep['reason']})" if sleep and sleep.get("reason") else ""))
        meals = rows.get("meals")
        if meals and meals.get("applies", True):
            out.append(PlanSource(kind="habit", id="meals", text=f"{int(meals['value'])} meal(s) seen today", origin="desk camera"))
        focus = rows.get("focus")
        if focus and focus.get("applies", True) and focus["value"]:
            out.append(PlanSource(kind="habit", id="focus", text=f"{focus['value']:g} h at the desk today", origin="desk camera"))
        out.append(PlanSource(kind="config", id="initial", text=f"starting energy {board.initial.energy:g} from {board.initial.source}"))

        if self.presence is None:
            missing.append("circle: not configured")
        elif not board.friends:
            missing.append("circle: no friend is sharing a place")
        for f in board.friends:
            out.append(PlanSource(kind="member", id=f.member, text=f"{f.member} at {f.place_name} until {hm(f.end)}", origin="circle"))
        out.append(PlanSource(kind="place", id=board.pack, text=f"walking times from the {board.pack} place pack"))
        if yesterday:
            out.append(PlanSource(kind="review", id=yesterday.plan_id, text=f"yesterday: {yesterday.change_tomorrow}", origin="review"))
        return out, missing

    def text(self, blocks: list[PlanMove], outcome: PlanOutcome, sources: list[PlanSource], card: bool, by: str = "atlas") -> str:
        chosen = [m for m in blocks if not m.fixed]
        steps = ", ".join(f"{m.title} {hm(m.start)}" for m in chosen[:6]) + (", ..." if len(chosen) > 6 else "")
        because = ""
        for s in sources:
            if s.kind in ("item", "habit", "member", "review") and s.origin != "coach" and len(because) + len(s.text) < 400:
                because += ("; " if because else "") + s.text
        lead = "Your edited day" if by == "person" else "Here's your day"
        parts = [f"{lead}: {steps}." if chosen else "Nothing to add: the calendar fills the day.",
                 f"Simulated: {outcome.summary}"]
        if because:
            parts.append(f"Because: {because}.")
        parts += [f"Coach: {s.text.removeprefix('coach: ')}" for s in sources if s.origin == "coach"]  # read back in full
        if card:
            parts.append("Approve to put it on your calendar.")
        return " ".join(parts)

    # --- a re-plan from where you are ---

    def replan(self, now: datetime, start_place: str | None, why: str, asked: bool = False, for_event: str | None = None,
               adjust=None, draft: bool = False, for_optional: str | None = None) -> list[Event]:
        """A new plan for the rest of the day from `start_place` (where the box thinks you are), with one card.
        Drift triggers it (contexts/day/application/tracker.py); the person can ask for one at any time.

        `adjust(moves) -> moves | None` changes the chosen moves before they are simulated (the check-in's "move the
        block to after dinner"); None, or a move the engine then rejects, means no re-plan at all. `draft` returns
        the re-plan without its card, for looking at what it would be: nothing is proposed. `for_optional` is the
        optional event ("nearby:...") a yes asked to add or a no asked to take out (contexts/day/application/optional.py
        puts it on, or takes it off, the board this is built from); the card of the re-plan that adds it writes it too."""
        day = self.clock.day_of(now)
        current = self.current.get(day)
        if current is None:
            return []
        board = self.dayrun.board(now, InitialOverride(start_place=start_place) if start_place else None)
        advice = self.advice(day)
        moves, result = choose(self.dayrun.cfg, self.dayrun.walks, board, preferred_block_minutes=advice[0].block_minutes if advice else None)
        if adjust is not None:
            moves = adjust(moves)
            if moves is None:
                return []
            result = simulate(self.dayrun.cfg, self.dayrun.walks, board, moves)
            if result.rejected:
                return []
        plan_id = self.fresh_id(f"plan-{day}-{local(self.dayrun.cfg, now):%H%M}r")
        blocks = plan_moves(result, moves)
        card = None if draft else self.card(blocks, day, result, now, replace=True, why=why,
                                            introduce=optional_block(for_optional) if for_optional else None)
        replan = DayReplanProposed(event_id=f"day-replan-{plan_id}", ts=now, day=day, plan_id=plan_id, replaces=current.plan_id,
                                   why=why, start=board.start, moves=blocks, dayrun_moves=[MoveIn(**m.model_dump()) for m in moves],
                                   outcome=plan_outcome(result), action_id=card.action_id if card else None, asked=asked,
                                   for_event=for_event, for_optional=for_optional)
        return [replan] + ([card] if card else [])

    def can_replan(self, now: datetime, gap: timedelta, for_event: str | None = None, at_most: int = 2) -> bool:
        """Drift may ask for a re-plan when none is waiting, the last one (or a denial) is `gap` ago, and the
        same fixed event has not already had `at_most` of them (a card lives five minutes; the person may let it go)."""
        day = self.clock.day_of(now)
        if day in self.pending or day not in self.current:
            return False
        if for_event and self.replans_for.get((day, for_event), 0) >= at_most:
            return False
        last = max((t for t in (self.last_replan.get(day), self.denied_at.get(day)) if t), default=None)
        return last is None or now - last >= gap

    # --- the evening review ---

    def maybe_review(self, now: datetime, closing: date | None = None) -> list[Event]:
        out: list[Event] = []
        for day, plan in sorted(self.current.items()):
            if day in self.reviews or now < plan.start:
                continue
            if day == closing or now >= self.clock.bed_at(day):
                try:
                    out.append(self.review(day, now))
                except Exception:
                    log.exception("the evening review of %s failed", day)
                    self.reviews.setdefault(day, None)  # type: ignore[arg-type]
        return out

    def sorting(self, day: date, now: datetime) -> str | None:
        """The day's sorting in one clause ("3 of 4 things you threw away went in the right bin; ...") when Waste Watch
        is wired in and something was thrown away; None otherwise, and the review reads as it always did."""
        if self.waste_clause is None:
            return None
        try:
            clause = self.waste_clause(day, now)
        except Exception:
            log.exception("the day's sorting clause failed")
            return None
        return clause.strip().rstrip(".") or None if clause else None

    def review(self, day: date, now: datetime) -> DayReviewed:
        plan = self.current[day]
        blocks = [PlannedBlock(id=m.id, kind=m.kind, title=m.title, start=m.start, end=m.end, minutes=m.minutes,
                               place=m.place or "", place_name=m.place_name or "", fixed=m.fixed) for m in plan.moves]
        o = plan.outcome
        totals = Totals(study_minutes=o.study_minutes, sleep_hours=o.sleep_hours, meals=o.meals, energy_min=o.energy_min, points=o.points)
        try:
            measured = self.dayrun.measured(day)
        except Exception:
            measured = Measured()
        rv = reality(self.dayrun.cfg, self.dayrun.walks.ref(self.dayrun.cfg.desk), day, blocks, totals, measured)
        followed = [r.title for r in rv.blocks if r.followed is not None and r.followed >= 0.8]
        slipped = [r.title for r in rv.blocks if r.followed is not None and r.followed < 0.5]
        gap = rv.biggest_gap
        if gap is None:
            change = "Keep the same shape tomorrow." if rv.score is not None else "Nothing could be measured; the desk camera has to see the day."
        elif gap.kind == "study":
            change = "Start the first study block earlier tomorrow, before anything can push it."
        elif gap.kind == "sleep":
            change = "Head to bed half an hour earlier tomorrow."
        else:
            change = "Put the meal on the calendar with a reminder tomorrow."
        # The game's numbers for the real day: Day Run's own replay runs what was measured as a baseline plan
        # through the same engine, so a lived day earns stars and points by the rules a played one does.
        stars, points, plan_stars = 0, 0, o.stars
        try:
            rp = self.dayrun.replay(day, [Move(**m.model_dump()) for m in plan.dayrun_moves], actual=measured)
            stars, points = rp.baseline.stars, rp.baseline.totals.points
        except Exception:
            log.exception("the day's stars could not be scored")
        before = self.reviews.get(day - timedelta(days=1))
        streak = (before.streak if before else 0) + 1 if stars > 0 else 0
        text = rv.text + (f" The real day scores {points} points and {stars} star{'s' if stars != 1 else ''} by Day Run's rules"
                          f" (the plan would have earned {plan_stars}); streak {streak}.")
        waste = self.sorting(day, now)
        if waste:
            text += f" Waste Watch: {waste}."
        coach: dict = {}
        if self.coach is not None:  # the day's blocks of the main activity, against the desk camera
            try:
                coach = self.coach.day_review(day, now)
            except Exception:
                log.exception("the coach's part of the review failed")
        return DayReviewed(event_id=f"day-review-{plan.plan_id}", ts=now, day=day, plan_id=plan.plan_id, score=rv.score,
                           followed=followed, slipped=slipped, text=text, change_tomorrow=change, stars=stars, points=points,
                           plan_stars=plan_stars, streak=streak, **coach)


def biases(review: DayReviewed | None) -> list[str]:
    if review is None:
        return []
    text = review.change_tomorrow.lower()
    return ["earlier_study"] if "study" in text else ["earlier_sleep"] if "bed" in text else []
