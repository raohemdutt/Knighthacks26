"""Day Run shapes: what the phone sends and gets back. Kept here, not in contracts/, until they settle."""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

from contexts.play.dayrun.routes import Leg

MAX_CAMPAIGN_DAYS = 31  # each day is a board and a simulation, run on the event loop
MoveKind = Literal["study", "eat", "sleep", "rest", "event"]
Activity = Literal["idle", "study", "class", "walk", "rest", "event", "eat", "sleep"]


# --- the board ---


class FixedBlock(BaseModel):
    """Something already on the calendar: you have to be there."""

    id: str
    title: str
    start: datetime
    end: datetime
    place: str | None = None  # "pack.place"; None = wherever you are (an online deadline)
    place_name: str | None = None
    activity: Activity = "class"
    deadline: str | None = None  # a study block's deadline id
    url: str | None = None


class Deadline(BaseModel):
    id: str
    title: str
    due: datetime
    work_left: float  # focused minutes still needed
    in_window: bool  # due before this board ends
    risk: float = 0.0  # at the start of the board
    url: str | None = None
    # Where `work_left` came from (additive): "kind" (config's estimate for this kind of work, `estimate_word` is the
    # word in the title that matched), "default" (no kind matched: the box's one-size guess) or "yours" (the request set it).
    estimate_from: Literal["kind", "default", "yours"] | None = None
    estimate_word: str | None = None
    estimate_minutes: float | None = None  # the box's own guess, also when the request overrode it


class FriendAt(BaseModel):
    """A friend who shared a place (opt-in, coarse) for a stretch of time."""

    member: str
    place: str
    place_name: str = ""
    start: datetime
    end: datetime


class OptionalEvent(BaseModel):
    id: str
    title: str
    start: datetime
    end: datetime
    place: str | None = None
    place_name: str | None = None
    url: str | None = None
    # A "Near you" suggestion (Board.suggestions) also says which open slot it falls in and the walk to it.
    slot: str | None = None
    walk_minutes: float | None = None
    leave_by: datetime | None = None
    note: str | None = None  # the rest of the item's line: "with Riley, Sam · optional · via GroupMe"


class MoveOption(BaseModel):
    """One concrete thing an open slot can hold."""

    kind: MoveKind
    place: str | None = None
    place_name: str | None = None
    title: str
    walk_minutes: float = 0  # from where the slot starts, by the route
    back_minutes: float = 0  # from there to the next fixed block
    max_minutes: int = 0  # what is left of the slot after both walks
    feasible: bool = True
    reason: str | None = None  # why not, when not feasible
    friends: list[str] = []  # who shared this place during the slot
    event: str | None = None
    route: list[str] = []  # the places walked through, for a preview on the map


class OpenSlot(BaseModel):
    id: str
    start: datetime
    end: datetime
    minutes: int
    from_place: str  # where the previous block leaves you
    to_place: str | None = None  # where the next block is
    next_block: str | None = None
    moves: list[MoveOption] = []


class Initial(BaseModel):
    """Where the meters start. `source` says what supplied the numbers: "habits", "scenario" (a scripted day),
    "defaults", "request", or, when they differ, each with its values: "defaults (sleep), habits (meals)"."""

    energy: float
    focus: float
    sleep_hours: float  # last night
    meals: int  # so far today
    sleep_debt: float
    hungry_minutes: float
    source: str = "defaults"
    sleep_from: str | None = None  # additive: "watch" or "camera" when Habits measured last night, else None


class Rules(BaseModel):
    """The settings behind the numbers a person might question (config/dayrun.yaml), for one plain sentence each."""

    capacity_minutes_per_day: float  # focused minutes a day the box counts on for one deadline
    on_track_risk: float  # a deadline is on track when the work left takes at most this share of that capacity
    win_sleep_hours: float
    win_meals: int
    need_sleep_hours: float
    walk_meters_per_minute: float | None = None  # the pace walks are timed at, when the pack has a path network


class Board(BaseModel):
    start: datetime
    end: datetime
    tick_minutes: int
    pack: str
    start_place: str
    home: str
    initial: Initial
    blocks: list[FixedBlock] = []
    deadlines: list[Deadline] = []
    slots: list[OpenSlot] = []
    friends: list[FriendAt] = []
    events: list[OptionalEvent] = []
    closed: list[str] = []  # places shut for this board (hard mode)
    sleep_window: tuple[datetime, datetime] | None = None  # the usual night, as a hint
    # Additive. "Near you" events (the `nearby:` items the board otherwise ignores) that fall in an open slot: offered,
    # never placed. A move of kind "event" may name one by id.
    suggestions: list[OptionalEvent] = []
    rules: Rules | None = None  # the numbers the outcome is judged by, so a screen can say where a figure comes from


# --- a plan and its simulation ---


class Move(BaseModel):
    """One chosen move. Moves in a slot run in the order given, each after the walk to its place."""

    slot: str
    kind: MoveKind
    place: str | None = None  # default: bed for sleep, where you are for rest and study
    minutes: int | None = None  # None: fill what is left of the slot
    start: datetime | None = None  # None: as soon as the previous move ends
    with_friend: str | None = None
    deadline: str | None = None  # study: which deadline; None = the most urgent
    event: str | None = None  # event: which optional event


class InitialOverride(BaseModel):
    """Numbers the caller knows better than Habits does (also how a campaign carries a day over)."""

    energy: float | None = None
    focus: float | None = None
    sleep_hours: float | None = None
    meals: int | None = None
    sleep_debt: float | None = None
    hungry_minutes: float | None = None
    start_place: str | None = None
    work_left: dict[str, float] = {}


class Tick(BaseModel):
    """One row of the meter timeline: the state at `t`, after the 15 minutes before it."""

    t: datetime
    energy: float
    focus: float
    deadline_risk: float  # the worst deadline
    work_left: dict[str, float]
    activity: Activity
    place: str


class RouteLeg(Leg):
    route: int  # index into `routes`
    depart: datetime


class RouteTaken(BaseModel):
    index: int
    purpose: str  # "to John C. Hitt Library (study)"
    move: int | None = None
    start: str
    end: str
    depart: datetime
    arrive: datetime
    minutes: float
    places: list[str]
    meters: float | None = None  # the length walked, when the pack has a path network


class Rejected(BaseModel):
    index: int
    move: Move
    reason: str
    minutes: int | None = None  # how late it would have made you


class Check(BaseModel):
    id: str  # "sleep", "meals", or a deadline id
    ok: bool
    value: float
    target: float
    text: str
    status: str | None = None  # deadlines: done, on_track, at_risk, missed
    # Additive, deadlines only: what the risk was worked out from.
    left_minutes: float | None = None  # focused minutes still to do when the plan ends
    done_minutes: float | None = None  # focused minutes this plan puts into it
    hours_left: float | None = None  # from the end of the plan to the due time
    due: datetime | None = None


class Outcome(BaseModel):
    won: bool
    stars: int
    reasons: list[str]  # why it failed; empty on a win
    checks: list[Check]
    summary: str


class PlannedBlock(BaseModel):
    """One block of the day as simulated: a fixed block or an accepted move."""

    id: str  # the fixed block's id, or "m<index>" for a move
    kind: str
    title: str
    start: datetime
    end: datetime
    minutes: int
    place: str
    place_name: str
    fixed: bool = False
    move: int | None = None
    with_friend: str | None = None


class Change(BaseModel):
    """Something the engine did to keep a plan standing (hard mode): a new place, a shorter block."""

    move: int | None = None
    kind: Literal["rerouted", "shortened", "friend_gone", "late", "route_changed", "rejected"]
    text: str


class Totals(BaseModel):
    study_minutes: int = 0
    focused_minutes: float = 0  # work actually done
    walk_minutes: float = 0
    sleep_hours: float = 0
    meals: int = 0  # the day's meals: those already eaten when the board starts, plus those in its window
    meals_before: int = 0  # of `meals`, how many were eaten before the board starts (Initial.meals)
    energy_end: float = 0
    energy_min: float = 0
    points: int = 0  # 0 to 100, for comparing plans
    points_parts: dict[str, int] = {}  # additive: how `points` adds up (deadlines of 40, sleep of 25, meals of 15, energy of 20)


class Fix(BaseModel):
    """One thing wrong with a plan, in plain words, and the edit that would mend it (POST /api/dayrun/fixes)."""

    id: str  # a check id ("sleep", "meals", a deadline id) or "move:<index>" for a move that did not fit
    problem: str  # "You'd miss judging by 11 minutes"
    label: str | None = None  # the edit: "Shorten lunch to 25 minutes"; None when the box found none
    moves: list[Move] | None = None  # the whole plan with the edit applied
    changed: int | None = None  # index in `moves` of the move that was added or changed (None: one was removed)
    mends: bool = False  # True when the edit makes this problem go away
    # No edit of its own yet (`label` is None), but there is one once another problem of the list is mended: that
    # problem's id, and the line to show in place of a button ("Fix the one above first"). Both None otherwise.
    after: str | None = None
    hint: str | None = None


class Carry(BaseModel):
    """What the end of one day hands to the next (campaign)."""

    energy: float
    focus: float
    sleep_debt: float
    hungry_minutes: float
    place: str
    work_left: dict[str, float]


class SimResult(BaseModel):
    start: datetime
    end: datetime
    tick_minutes: int
    outcome: Outcome
    stars: int
    timeline: list[Tick]
    legs: list[RouteLeg]
    routes: list[RouteTaken]
    rejected: list[Rejected]
    blocks: list[PlannedBlock]
    changes: list[Change] = []
    notes: list[str] = []
    totals: Totals
    carry: Carry


# --- requests ---


class PlanBody(BaseModel):
    start: datetime | None = None  # default: now
    moves: list[Move] = []
    initial: InitialOverride | None = None
    scenario: str | None = None  # a scripted scenario in fixtures/play/, instead of live sources


class Disruption(BaseModel):
    kind: Literal["block_moved", "friend_cancelled", "place_closed"]
    block: str | None = None
    to_place: str | None = None
    shift_minutes: int = 0  # negative: the block starts earlier
    friend: str | None = None
    place: str | None = None
    text: str = ""


class HardBody(PlanBody):
    seed: int = 0
    count: int = 1
    disruptions: list[Disruption] | None = None  # None: pick `count` from the seed


class HardResult(BaseModel):
    seed: int
    disruptions: list[Disruption]
    before: SimResult
    after: SimResult
    changes: list[Change]
    requested: int | None = None  # how many seeded disruptions were asked for (None: the request listed its own)
    note: str = ""  # set when fewer came back than asked for: there is one per kind, and not every kind fits every day


class CampaignBody(BaseModel):
    start: datetime | None = None
    days: list[list[Move]] = Field(max_length=MAX_CAMPAIGN_DAYS)
    initial: InitialOverride | None = None
    scenario: str | None = None


class CampaignDay(BaseModel):
    day: date
    carry_in: Initial
    result: SimResult


class CampaignResult(BaseModel):
    days: list[CampaignDay]
    won: bool
    stars: int
    points: int


class Measured(BaseModel):
    """What really happened, as far as Atlas can measure it."""

    focus_minutes: float | None = None  # desk time from Habits (stands in for face presence)
    meals: int | None = None
    sleep_hours: float | None = None
    blocks: dict[str, float] = {}  # block id -> measured minutes, when something can tell


class BlockScore(BaseModel):
    id: str
    title: str
    kind: str
    planned_minutes: float
    measured_minutes: float | None = None
    followed: float | None = None  # 0 to 1; None when nothing can measure this block
    note: str = ""


class Gap(BaseModel):
    block: str | None = None
    kind: str
    planned: float
    measured: float
    text: str


class Review(BaseModel):
    day: date
    score: int | None  # 0 to 100; None when nothing in the plan could be measured
    blocks: list[BlockScore]
    planned: dict[str, float]  # study minutes, meals, sleep hours as simulated
    measured: Measured
    biggest_gap: Gap | None
    text: str
    measured_by: str  # what stood in for the face-presence check


class ReplayBody(BaseModel):
    day: date
    moves: list[Move] = []
    baseline: list[Move] | None = None  # what happened, as moves; None: rebuild it from `actual` or Habits
    actual: Measured | None = None
    scenario: str | None = None


class ReplayResult(BaseModel):
    day: date
    board: Board
    actual: Measured
    baseline_moves: list[Move]
    baseline: SimResult
    plan: SimResult
    points_gained: int
    stars_gained: int
    text: str
    unknown: list[str] = []  # "sleep": that night is not measured yet, so both sides were given the plan's night


class Estimate(BaseModel):
    block: str
    minutes: float = Field(ge=0)


class ForecastBody(BaseModel):
    estimates: list[Estimate]
    measured: dict[str, float] | None = None  # block id -> minutes; None: the review of `day`
    day: date | None = None
    scenario: str | None = None


class ForecastRow(BaseModel):
    block: str
    estimated: float
    measured: float
    error: float  # 0 = spot on, 1 = off by the whole thing
    off_by: float  # estimated - measured, minutes


class ForecastResult(BaseModel):
    score: int  # 0 to 100 calibration
    bias: float  # (estimated - measured) / measured over all blocks; negative = you underestimate
    multiplier: float  # multiply your estimates by this next time
    rows: list[ForecastRow]
    unmeasured: list[str]
    text: str


class FixesResult(BaseModel):
    fixes: list[Fix]  # what needs fixing, worst first; empty for a plan that works


class EditBody(PlanBody):
    """A change asked for in plain words: "move lunch to one", "drop the gym", "study two hours before judging"."""

    text: str = Field(max_length=300)


class EditOption(BaseModel):
    label: str
    moves: list[Move]
    changed: int | None = None


class EditResult(BaseModel):
    ok: bool  # False: nothing was changed; `understood` says what was heard and `options` offers the nearest edits
    understood: str
    moves: list[Move]
    changed: int | None = None
    options: list[EditOption] = []
    warning: str | None = None  # the edit was made but does not fit: "you'd miss judging by 11 minutes"
    # nothing in the words could be read ("banana"): two or three things that can be said about this very plan, and
    # no `options` ("move breakfast to 10 AM", "drop the second study block", "add a break after lunch")
    examples: list[str] = []


class CommitResult(BaseModel):
    action_id: str
    day: date
    events: list[dict]
    stars: int
    reason: str

