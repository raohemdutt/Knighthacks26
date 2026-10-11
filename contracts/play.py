from datetime import date
from typing import Literal

from pydantic import BaseModel

from contracts.events import TypedEvent, register
from contracts.types import CameraId, ZoneId


class Cell(BaseModel):
    row: int
    col: int


@register
class RoundStarted(TypedEvent):
    type: Literal["play.round.started"] = "play.round.started"
    round_id: str
    session_id: str  # rounds in one sitting, for round 1 vs round 5 accuracy
    players: int
    item: str  # label the player must sort
    target_bin: ZoneId
    path: list[Cell] = []  # A* route on the zone grid, for the phone to draw


@register
class RoundScored(TypedEvent):
    type: Literal["play.round.scored"] = "play.round.scored"
    round_id: str
    session_id: str
    players: int
    item: str
    correct: bool
    seconds: float
    path_len: int
    score: float


@register
class FacesCounted(TypedEvent):
    """Face presence from Perception (YuNet). Counts only; no identity, faces stay blurred."""

    type: Literal["perception.faces.counted"] = "perception.faces.counted"
    camera_id: CameraId
    count: int
    facing: int  # how many are facing the camera


class FloorPoint(BaseModel):
    """A point of the room's floor plan, in metres (config/rooms/<name>.yaml: x east, y south)."""

    x: float
    y: float


class PlannedRoute(BaseModel):
    player: int  # 1, 2: the order the players are drawn in
    track: str | None = None  # "<camera>:<track id>" of the player the route starts from, when a camera sees them
    points: list[FloorPoint] = []  # the walk to the bin, already smoothed; empty when no way exists
    distance_m: float = 0.0
    par_s: float = 0.0  # expected seconds at walking speed
    reachable: bool = True


@register
class RoutePlanned(TypedEvent):
    """Sort It's route in room space (the floor plan), when a round starts and whenever it is planned again."""

    type: Literal["play.route.planned"] = "play.route.planned"
    round_id: str
    item: str
    target_bin: ZoneId
    room: str  # the floor plan the points are in
    reason: Literal["round_started", "replan"] = "round_started"
    routes: list[PlannedRoute] = []  # one per player


# --- The League (contexts/play/league): the friends' weekly game. Only daily summaries, standings and
# stakes travel in these; never raw signals, frames or health numbers beyond the summary itself.

StakeKind = Literal["pays_for_food", "picks_the_event", "custom"]
SummaryKind = Literal["habit_summary", "impact_count"]


class Stake(BaseModel):
    kind: StakeKind = "pays_for_food"
    text: str | None = None  # for custom


class Standing(BaseModel):
    member_id: str
    display_name: str
    rank: int
    score: float | None  # 0..100, None when the member has nothing judged this week
    met: int  # judged target-days hit
    judged: int
    reasons: list[str] = []  # "missed sleep 4 of 7 nights"


class ChallengeResult(BaseModel):
    member_id: str
    display_name: str
    met_days: int
    days: int
    done: bool


@register
class LeagueSummaryReceived(TypedEvent):
    """A friend's shared daily summary, as the league sees it: a habit_summaries or impact_counters row."""

    type: Literal["play.league.summary.received"] = "play.league.summary.received"
    circle_id: str
    member_id: str
    kind: SummaryKind
    payload: dict  # habit_summary: habit_id, day, met, value, streak; impact_count: the CountUpdated fields
    display_name: str | None = None  # the members table's name, when the box has not got it from config


@register
class LeagueStakeSet(TypedEvent):
    """The stake for a week, after its approval card was approved."""

    type: Literal["play.league.stake.set"] = "play.league.stake.set"
    circle_id: str
    week_start: date
    stake: Stake
    set_by: str


@register
class LeagueWeekClosed(TypedEvent):
    """The first event after the week boundary (event time) closes the week with its standings."""

    type: Literal["play.league.week.closed"] = "play.league.week.closed"
    circle_id: str
    week_start: date
    week_end: date  # the last day of the week (inclusive)
    standings: list[Standing]
    worst: str | None = None  # member id
    best: str | None = None
    stake: Stake
    summary: str  # "Maya had the worst week: 2 of 7 sleep targets. Dinner is on Maya. Hemdutt picks the place."


@register
class LeagueChallengeStarted(TypedEvent):
    type: Literal["play.league.challenge.started"] = "play.league.challenge.started"
    circle_id: str
    challenge_id: str
    title: str
    habit_id: str
    direction: Literal["at_least", "at_most"]
    target: float
    starts: date
    ends: date  # inclusive
    members: list[str]
    started_by: str


@register
class LeagueChallengeClosed(TypedEvent):
    type: Literal["play.league.challenge.closed"] = "play.league.challenge.closed"
    circle_id: str
    challenge_id: str
    title: str
    results: list[ChallengeResult]
    winners: list[str]  # member ids who met it every day
    summary: str
