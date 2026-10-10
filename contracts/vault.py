from typing import Literal

from contracts.events import TypedEvent, register
from contracts.types import CameraId


@register
class PolicyChanged(TypedEvent):
    type: Literal["vault.policy.changed"] = "vault.policy.changed"
    blur_faces: bool
    paused_cameras: list[CameraId] = []
    offline: bool = False


@register
class RetentionChanged(TypedEvent):
    type: Literal["vault.retention.changed"] = "vault.retention.changed"
    keep_days: int | None  # None = keep forever


@register
class WipeRequested(TypedEvent):
    type: Literal["vault.wipe.requested"] = "vault.wipe.requested"
    scope: Literal["all", "keyframes", "memory"]
