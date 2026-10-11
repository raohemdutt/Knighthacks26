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
    origin: str | None = None  # "optional": an optional event the person said yes to (contexts/day/application/optional.py)


class MoveIn(BaseModel):
    """A chosen move in the shape Day Run's editor consumes (contexts.play.dayrun.models.Move): the plan is a
    starting point, and the person can change it in Day Run and send it back through POST /api/day/plan."""

    slot: str
    kind: str
    place: str | None = None
    minutes: int | None = None
    start: datetime | None = None
    with_friend: str | None = None
    deadline: str | None = None
    event: str | None = None


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
    dayrun_moves: list[MoveIn] = []  # the chosen moves as Day Run's editor takes them
    outcome: PlanOutcome
    sources: list[PlanSource] = []
    missing: list[str] = []  # inputs the box did not have ("circle: not configured")
    text: str  # one paragraph for the phone
    action_id: str | None = None  # the approval card for the calendar blocks
    by: Literal["atlas", "person"] = "atlas"  # who chose the moves: the box, or the person editing them


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
    next_block: NextEvent | None = None  # the plan's next block, fixed or chosen
    next_event: NextEvent | None = None  # the next fixed event with a place, and when to leave for it
    risk: Risk = "ok"
    deadline_risk: float = 0.0
    meters: DayMeters = DayMeters()
    reason: str = ""  # what changed, in a few words


# --- the study coach (D2): a planned block against what the desk camera saw ---

Trigger = Literal["chat", "phone", "door", "food", "none"]


class CoachEvidence(BaseModel):
    """One thing that happened just before a break, by id, so the screen can cite it. Ids and times only:
    the API looks the picture and the words up in Memory when the phone asks (never in the event)."""

    kind: Literal["chat", "phone", "door", "food"]
    id: str  # a connector item's "connector:external_id", or a Memory episode id
    at: datetime
    label: str  # "a message in the group chat", "the phone was in view"


class CoachBreak(BaseModel):
    """Away from the desk in the middle of a block, and the most likely reason from the two minutes before."""

    start: datetime
    end: datetime
    minutes: int
    trigger: Trigger = "none"  # "none": nothing the box records explains it
    evidence: list[CoachEvidence] = []


class SessionExpression(BaseModel):
    """How the person at the desk looked across a block, added up from `perception.expression.observed` (only for an
    enrolled member who switched "read my expressions" on). Apparent expression is a noisy reading of a face's shape:
    "looked strained" is not an emotion and nothing about health."""

    samples: int  # counted looks inside the block
    strained_share: float  # the fraction of them that looked angry, disgusted, fearful or sad
    rose_after_minutes: int | None = None  # minutes into the block after which they looked more strained, when that showed
    before_share: float | None = None  # the strained fraction before that mark
    after_share: float | None = None  # ... and after it
    text: str = ""  # "You looked more strained after the 35-minute mark; a break around then next time."


class SessionReview(BaseModel):
    """One planned block of the main activity, measured. `measured` False means the box could not see it
    (a block away from the box): every number is then None, never zero."""

    block_id: str
    title: str
    start: datetime
    end: datetime
    place: str | None = None
    place_name: str | None = None
    planned_minutes: int
    measured: bool = False
    not_measured: str | None = None  # why, in words, when `measured` is False
    desk_minutes: int | None = None
    late_start_minutes: int | None = None  # block start to the first sit-down
    ended_early_minutes: int | None = None  # last time at the desk to the block's end
    longest_stretch_minutes: int | None = None
    longest_stretch_start: datetime | None = None
    breaks: list[CoachBreak] = []
    text: str = ""  # planned against done, the moments that cost the most, what worked
    expression: SessionExpression | None = None  # only when enough looks were counted in the block; None otherwise


class CoachCalibration(BaseModel):
    """Day Run's forecast calibration, fed with every measured block: planned minutes are the estimate."""

    score: int  # 0 to 100
    ratio: float  # planned / measured: 1.7 means plans run 1.7 times what happens
    blocks: int  # measured blocks behind the figures
    text: str


@register
class DayCoachNoted(TypedEvent):
    """A planned block of the main activity ended and the desk camera measured it: one sentence for the
    phone (and for a voice), with the session behind it."""

    type: Literal["day.coach.noted"] = "day.coach.noted"
    day: date
    plan_id: str
    block_id: str
    text: str  # "You did 70 of 120 minutes. Two breaks; the long one came right after the group chat. ..."
    session: SessionReview
    suggestions: list[str] = []
    calibration: CoachCalibration | None = None
    next_block: NextEvent | None = None  # the plan's next block of the main activity


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
    dayrun_moves: list[MoveIn] = []
    outcome: PlanOutcome
    action_id: str | None = None
    asked: bool = False  # True when the person asked for it (POST /api/day/plan from_here), not drift
    for_event: str | None = None  # the fixed block the drift was about
    for_optional: str | None = None  # the optional event ("nearby:...") a yes asked to add, or a no asked to take out


@register
class DayReviewed(TypedEvent):
    """The evening review: the plan against what Habits measured (Day Run's reality score), and the real day
    scored by the same engine as a simulated one, so a lived day earns stars and points the way a played
    one does (Day Run's scoring is untouched: the measured day is run through it as a baseline plan)."""

    type: Literal["day.reviewed"] = "day.reviewed"
    day: date
    plan_id: str
    score: int | None = None  # 0 to 100; None when nothing could be measured
    followed: list[str] = []  # block titles that happened
    slipped: list[str] = []  # block titles that did not
    text: str
    change_tomorrow: str  # one sentence
    stars: int = 0  # the real day's stars, by Day Run's rules
    points: int = 0  # the real day's points, 0 to 100
    plan_stars: int = 0  # what the plan would have earned
    streak: int = 0  # consecutive reviewed days with at least one star
    # the study coach (D2): the day's blocks of the main activity that had ended by the review
    sessions: list[SessionReview] = []
    coach_text: str = ""  # the day in the coach's words; "" when there was no such block
    suggestions: list[str] = []  # one or two, drawn from the day's pattern
    calibration: CoachCalibration | None = None
    block_minutes: int | None = None  # the block length the coach suggests for tomorrow


# "suggest": an optional event coming up that fits ("Want me to add it?"); "start": a planned block is starting.
Moment = Literal["morning", "now", "replan", "coach", "evening", "approval", "checkin", "suggest", "start"]


@register
class DayNarrated(TypedEvent):
    """One moment of the day in words (D3), written once so the phone's screen, its voice and the push all say
    the same thing and a restart replays it without calling a model again. `spoken` is for the ear (a few
    sentences, no lists, times as people say them); `display` is a short paragraph that may keep exact times."""

    type: Literal["day.narrated"] = "day.narrated"
    day: date
    moment: Moment
    plan_id: str | None = None
    ref: str | None = None  # what it narrates: the source event's id, or the action id of an approval ask
    spoken: str
    display: str
    engine: Literal["template", "model"] = "template"
    facts_hash: str
    urgent: bool = False  # a "now" worth saying aloud unasked: leave now, late, you will miss it
    # One speaker (contexts/day/domain/arbiter.py): "box" when the box's own speaker says this line, so the phone's
    # voice stays silent for it; None (the default, as before) when the phone or browser may say it.
    spoken_by: Literal["box"] | None = None
    # Said inside another narration (that one's event id): the screen may show it, nobody says or pushes it again.
    merged_into: str | None = None


# --- the coach's feedback loop (contexts/day/application/coach_feedback.py): additive ---

CoachAnswer = Literal["chat", "phone", "door", "food", "bathroom", "done", "other", "none"]


@register
class DayCoachCorrected(TypedEvent):
    """The person answered "Was that right?" about one break. `right` True confirms the coach's guess;
    False carries what it really was. The latest answer for a break wins. Ids and times only: the picture
    behind a break is looked up in Memory when a teach task is proposed, never stored here."""

    type: Literal["day.coach.corrected"] = "day.coach.corrected"
    day: date
    block_id: str
    break_index: int  # position in the session's breaks, 0 first
    break_start: datetime  # when the person left the desk: what identifies the break if the list shifts
    guess: Trigger  # what the coach said
    right: bool
    answer: CoachAnswer  # the person's reason; equals `guess` when `right`
    note: str | None = None  # free text, for "something else"
    plan_id: str | None = None


# --- the check-in (contexts/day/application/checkin.py): additive, and box-only ---

# Events that never leave the box. A check-in is built from a watch's night, how a face looked and what the desk
# camera saw: the owner's own phone may show it, and nothing else may carry it. Not the Circle (no handler there
# takes a `day.` event: contexts/circle/tests/test_health_privacy.py), not the training export (it reads Memory's
# camera episodes, never the event log: contexts/learning/tests/test_export.py), and the push that announces one
# carries the suggestions only (contexts/day/application/loop.py). Every field below is box-only; none is safe
# to forward, the plain-word summaries included ("Heart rate: up" is still a statement about a person's body).
# `day.optional.answered` is here too: which invitations a person took or turned down is nobody else's business.
# `day.look.forgotten` (the look memory, contexts/day/application/looks.py): that a remembered look was forgotten.
BOX_ONLY_EVENTS = frozenset({"day.checkin.raised", "day.checkin.answered", "day.optional.answered", "day.look.forgotten"})

CheckinSignalKind = Literal["sleep", "resting_hr", "expression", "sitting", "drink", "history"]
CheckinSuggestionKind = Literal["drink", "break", "move_block", "walk"]


class CheckinSignal(BaseModel):
    """One thing the box noticed, in plain words. Never a raw health number: sleep is said to the half hour, as the
    owner's own screen already shows it, and a heart rate is only ever "up" against the person's own baseline."""

    kind: CheckinSignalKind
    summary: str  # the chip: "Sleep: short", "Heart rate: up", "Looked strained 20 min", "No drink since 2:05 PM"
    detail: str = ""  # one plain sentence behind the chip, with what it cannot know
    since: datetime | None = None  # when it began: the last drink seen, the start of the sitting stretch
    minutes: int | None = None  # how long it covers: the night to the half hour, the strained stretch, the sitting
    evidence_refs: list[str] = []  # event ids, "connector:external_id" of an item, or frame refs: looked up on the box


class CheckinSuggestion(BaseModel):
    kind: CheckinSuggestionKind
    text: str  # "Drink some water", "Take ten minutes outside", "Move the study block to after dinner"


class CheckinPastDay(BaseModel):
    day: date
    text: str  # what the desk camera measured: "20 of 90 planned minutes at the desk from 3:30 PM on"


class CheckinHistory(BaseModel):
    """Earlier days that looked like this one, and what the desk camera measured on them."""

    matches: int  # the days behind the claim: at least three, or there is no history at all
    outcome: str  # "your study blocks fell apart by 4 PM"
    by: str | None = None  # that time as local "HH:MM"
    days: list[CheckinPastDay] = []


class CheckinCallback(BaseModel):
    """An earlier look this one is like, from the look memory (contexts/day/domain/looks.py): "That's the same look you
    had Tuesday afternoon, when you'd gone two hours without moving." Words, a day and a label only: no picture, no
    frame ref, no health value. Box-only with the check-in that carries it."""

    moment_id: str  # the remembered moment ("look-<member>-<yyyymmdd>-<hhmm>"), looked up on the box
    day: date  # the earlier local day
    look: str = "tired"  # the kind of look, in the People screen's words: "tired" or "tense". Never in the spoken sentence.
    said: str  # the callback sentence, for the ear and the screen
    learned: str = ""  # what followed last time, when that is known: "Ten minutes away from the desk helped then."
    lead: CheckinSuggestionKind | None = None  # the advice that was followed by the look easing: it is said first
    chip: str = ""  # "Looked tired · like Tue"
    lines: list[str] = []  # the remembered moment in one or two plain lines, for the chip


@register
class DayCheckinRaised(TypedEvent):
    """Atlas speaking up unasked: enough independent signals agreed (the rule is contexts/day/domain/checkin.py).
    Box-only (BOX_ONLY_EVENTS). A clause exists only for a signal that was there; nothing here is a diagnosis."""

    type: Literal["day.checkin.raised"] = "day.checkin.raised"
    day: date
    at: datetime
    signals: list[CheckinSignal] = []
    suggestions: list[CheckinSuggestion] = []
    history: CheckinHistory | None = None
    proposes_replan: bool = False  # "Move the block" can build a re-plan; the plan changes only through its approval card
    plan_id: str | None = None
    text: str = ""  # the whole check-in as one paragraph for the screen (the template's words, before any rewrite)
    callback: CheckinCallback | None = None  # an earlier look like this one, when the look memory has one similar enough


@register
class DayCheckinAnswered(TypedEvent):
    """The person answered a check-in on their phone: "Done", or "Move the block" (which raised a re-plan card)."""

    type: Literal["day.checkin.answered"] = "day.checkin.answered"
    day: date
    checkin_id: str  # the `day.checkin.raised` event's id
    answer: Literal["done", "move"]
    action_id: str | None = None  # the re-plan's approval card, when the answer was "move"


# --- the look memory (contexts/day/application/looks.py): additive, and box-only ---


@register
class DayLookForgotten(TypedEvent):
    """The person asked Atlas to forget a remembered look, or all of them. The memory itself is a projection of the
    log (the looks, the desk, the plan); this event is what keeps a forgotten one forgotten on replay. Ids only."""

    type: Literal["day.look.forgotten"] = "day.look.forgotten"
    moment_id: str | None = None  # the remembered moment; None: every moment up to `ts`
    member_id: str | None = None  # with no `moment_id`: whose are forgotten; None: everyone's


# --- optional events (contexts/day/application/optional.py): additive, and box-only ---


@register
class DayOptionalAnswered(TypedEvent):
    """The person said yes or no to an optional event ("Near you": a `connector.item.synced` calendar item whose
    `external_id` starts with `nearby:`). A no is all there is to it: the event is not suggested again. A yes never
    changes the plan by itself: it raises a re-plan (`day.replan.proposed`, `for_optional`) with its approval card,
    and the event is a block of the day only once that card is approved (`day.plan.committed`). Box-only."""

    type: Literal["day.optional.answered"] = "day.optional.answered"
    day: date
    id: str  # the optional event's `external_id`
    answer: Literal["yes", "no"]
    at: datetime
    plan_id: str | None = None  # the re-plan the answer raised, when it raised one
    action_id: str | None = None  # ... and its approval card
    until: datetime | None = None  # a yes that means leaving early for the next fixed event: when its block ends in the plan
