from typing import Literal

from contracts.events import TypedEvent, register
from contracts.types import CameraId


@register
class FramesCaptured(TypedEvent):
    """In-process only: the frame itself is passed by reference beside the event."""

    type: Literal["capture.frames.captured"] = "capture.frames.captured"
    camera_id: CameraId
    seq: int
    width: int
    height: int
    motion_score: float


@register
class CameraHealth(TypedEvent):
    type: Literal["capture.camera.health"] = "capture.camera.health"
    camera_id: CameraId
    status: Literal["ok", "degraded", "down", "paused"]
    fps_actual: float
    dropped: int = 0
