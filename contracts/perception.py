from typing import Literal

from contracts.events import TypedEvent, register
from contracts.types import CameraId, FrameRef, ObjectHit, Track, ZoneId


@register
class ObservationRecorded(TypedEvent):
    """Sampled ~1/s per camera."""

    type: Literal["perception.observation.recorded"] = "perception.observation.recorded"
    camera_id: CameraId
    tracks: list[Track]
    frame_ref: FrameRef | None = None


@register
class ZoneEvent(TypedEvent):
    type: Literal["perception.zone.event"] = "perception.zone.event"
    camera_id: CameraId
    track_id: int
    label: str
    zone_id: ZoneId
    kind: Literal["enter", "exit", "dwell"]
    dwell_s: float | None = None  # set when kind == "dwell"
    frame_ref: FrameRef | None = None


@register
class SceneDescribed(TypedEvent):
    type: Literal["perception.scene.described"] = "perception.scene.described"
    camera_id: CameraId
    caption: str
    objects: list[ObjectHit] = []
    zone_id: ZoneId | None = None  # set when the frame is a zone crop (watch-item finder)
    frame_ref: FrameRef
    model_version: str
