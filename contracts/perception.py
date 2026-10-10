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


# --- owner recognition (P9, opt-in): ids only, never a name, a picture or a face vector ---


@register
class PersonIdentified(TypedEvent):
    """Track `track_id` on this camera is the enrolled member `member_id`."""

    type: Literal["perception.person.identified"] = "perception.person.identified"
    camera_id: CameraId
    track_id: int
    member_id: str


@register
class PersonLost(TypedEvent):
    """That track's tag is over."""

    type: Literal["perception.person.lost"] = "perception.person.lost"
    camera_id: CameraId
    track_id: int | None = None  # None: every track on this camera (the pipeline restarted, or recognition went off)


@register
class IdentityChanged(TypedEvent):
    """Recognition was switched on or off, or the set of enrolled members changed."""

    type: Literal["perception.identity.changed"] = "perception.identity.changed"
    enabled: bool
    members: list[str] = []  # enrolled member ids
