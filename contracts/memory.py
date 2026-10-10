from datetime import datetime
from typing import Literal

from contracts.events import TypedEvent, register
from contracts.types import CameraId, FrameRef


@register
class EpisodeOpened(TypedEvent):
    type: Literal["memory.episode.opened"] = "memory.episode.opened"
    episode_id: str
    camera_id: CameraId


@register
class EpisodeClosed(TypedEvent):
    type: Literal["memory.episode.closed"] = "memory.episode.closed"
    episode_id: str
    camera_id: CameraId
    start: datetime
    end: datetime
    labels: list[str]
    captions: list[str]
    summary: str | None = None
    keyframe_refs: list[FrameRef] = []
