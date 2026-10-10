"""The Day loop (D1): the morning plan, the live status, re-plans and the evening review, unasked.

Every payload is small and typed: ids, times, places and minutes. No frames, no raw signals, and the only
health figure that ever appears is hours asleep (in a plan's `sources`, as text).
"""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel

from contracts.events import TypedEvent, register

SourceKind = Literal["item", "habit", "member", "place", "review", "config"]
Risk = Literal["ok", "leave_now", "late", "infeasible"]


class PlanMove(BaseModel):
    """One block of the day as the plan simulates it: a chosen move, or a fixed block from the calendar."""

    id: str  # the fixed block's id, or "m<index>" for a chosen move
    kind: str  # study, eat, sleep, rest, event, class
    title: str
    start: datetime
    end: datetime
    minutes: int
    place: str | None = None  # "pack.place"; None for an online block (a deadline)
    place_name: str | None = None
    fixed: bool = False
    deadline: str | None = None  # study: the deadline it works on
    with_friend: str | None = None


class PlanSource(BaseModel):
    """Why the plan is what it is: one input the box already had, with where it came from."""

    kind: SourceKind
    id: str  # item external id, habit id, member id, pack name
    text: str  # "Quiz 5 - Greedy Algorithms due Wed 5:00 PM", "slept 5.4 h"
    origin: str | None = None  # "canvas", "desk camera", "watch", "circle"


class PlanOutcome(BaseModel):
    """The simulated outcome of the plan (contexts/play/dayrun engine), summarised."""

    won: bool
    stars: int
    points: int  # 0 to 100
    summary: str
    sleep_hours: float
    meals: int
    study_minutes: int
    energy_min: float
    deadline_risk: float  # the worst deadline at the end of the plan, 0 to 1


class DayMeters(BaseModel):
    """The day so far, from what the other contexts measured."""

    meals: int = 0
    desk_minutes: int = 0
    recycled: int = 0
    sleep_hours: float | None = None  # last night, when measured
    sleep_source: str | None = None  # "watch" or "camera"


@register
class DayPlanProposed(TypedEvent):
    """The morning plan: chosen by the box from what it already knows, proposed to the phone with one card."""

    type: Literal["day.plan.proposed"] = "day.plan.proposed"
    day: date
    plan_id: str
    start: datetime
    end: datetime
    moves: list[PlanMove] = []  # fixed blocks and chosen moves, in time order
    outcome: PlanOutcome
    sources: list[PlanSource] = []
    missing: list[str] = []  # inputs the box did not have ("circle: not configured")
    text: str  # one paragraph for the phone
    action_id: str | None = None  # the approval card for the calendar blocks


@register
class DayPlanCommitted(TypedEvent):
    """The phone approved the plan's card: the plan (or a re-plan) is now the day's plan."""

    type: Literal["day.plan.committed"] = "day.plan.committed"
    day: date
    plan_id: str
    action_id: str


class NextEvent(BaseModel):
    id: str
    title: str
    start: datetime
    end: datetime
    place: str | None = None
    place_name: str | None = None
    walk_minutes: int = 0
    leave_by: datetime | None = None


@register
class DayStatusChanged(TypedEvent):
    """The live day: the plan against what the box sees. Published when something material changes."""

    type: Literal["day.status.changed"] = "day.status.changed"
    day: date
    plan_id: str | None = None
    now: datetime
    block_id: str | None = None  # the plan block that covers `now`
    block_title: str | None = None
    block_end: datetime | None = None
    should_be: str | None = None  # "pack.place" the plan has you at
    should_be_name: str | None = None
    seen_at: str | None = None  # where the box thinks you are, from the cameras or your own shared place
    seen_at_name: str | None = None
    seen_how: str | None = None  # "desk camera", "door camera", "shared place"
    seen_ts: datetime | None = None
    drift_minutes: int = 0  # positive: behind the plan; negative: ahead
    next_event: NextEvent | None = None
    risk: Risk = "ok"
    deadline_risk: float = 0.0
    meters: DayMeters = DayMeters()
    reason: str = ""  # what changed, in a few words


@register
class DayReplanProposed(TypedEvent):
    """Drift made the next fixed event infeasible: a new plan from where you are, with one card."""

    type: Literal["day.replan.proposed"] = "day.replan.proposed"
    day: date
    plan_id: str  # the re-plan's id
    replaces: str  # the plan it would replace
    why: str
    start: datetime
    moves: list[PlanMove] = []
    outcome: PlanOutcome
    action_id: str | None = None


@register
class DayReviewed(TypedEvent):
    """The evening review: the plan against what Habits measured."""

    type: Literal["day.reviewed"] = "day.reviewed"
    day: date
    plan_id: str
    score: int | None = None  # 0 to 100; None when nothing could be measured
    followed: list[str] = []  # block titles that happened
    slipped: list[str] = []  # block titles that did not
    text: str
    change_tomorrow: str  # one sentence
