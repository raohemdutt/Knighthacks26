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
