from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from contracts.events import TypedEvent, register
from contracts.types import CameraId, FrameRef, ZoneId

# --- events ---


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


# --- read API (GET /memory/...), used by Agent and Experience ---


class Episode(BaseModel):
    """Row of memory_episodes. GET /memory/episodes?from&to&camera&label"""

    id: str
    camera_id: CameraId
    start: datetime
    end: datetime | None = None  # None while still open
    labels: list[str] = []
    captions: list[str] = []
    summary: str | None = None
    keyframe_refs: list[FrameRef] = []


class Sighting(BaseModel):
    """Last known place of a watch item. GET /memory/objects/{label}/last-seen"""

    label: str
    camera_id: CameraId
    zone_id: ZoneId | None = None
    ts: datetime
    frame_ref: FrameRef
    position: str | None = None  # "left of the laptop"


class SearchHit(BaseModel):
    """GET /memory/search?q= returns list[SearchHit], best first."""

    episode_id: str
    camera_id: CameraId
    ts: datetime
    text: str  # matching caption or summary
    score: float
    frame_ref: FrameRef | None = None


class HourBucket(BaseModel):
    hour: datetime  # start of the hour
    episode_ids: list[str]
    captions: list[str]  # deduped, capped so prompts stay small
    keyframe_refs: list[FrameRef] = []


class EpisodeWindow(BaseModel):
    """Episodes in [start, end) grouped by hour, for "what happened overnight"."""

    start: datetime
    end: datetime
    hours: list[HourBucket]


class MemoryStats(BaseModel):
    """GET /memory/stats"""

    hours_recorded: float
    episodes: int
    captions: int
    first_event: datetime | None = None


class Item(BaseModel):
    """Connector item projected into memory_items. GET /memory/items?from&to&kind"""

    connector: str
    kind: str  # contracts.connectors.ItemKind
    external_id: str
    title: str
    when: datetime | None = None
    url: str | None = None
    summary: str | None = None
