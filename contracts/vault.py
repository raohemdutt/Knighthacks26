from typing import Literal

from contracts.events import TypedEvent, register
from contracts.types import CameraId


@register
class PolicyChanged(TypedEvent):
    type: Literal["vault.policy.changed"] = "vault.policy.changed"
    blur_faces: bool
    paused_cameras: list[CameraId] = []
    offline: bool = False
    # The live picture on the owner's own signed-in screen: "clear" shows faces there, and only there. None is an
    # event from before this field existed and means what the box did then: blurred. Saved pictures are `blur_faces`.
    live_faces: Literal["blurred", "clear"] | None = None


@register
class RetentionChanged(TypedEvent):
    type: Literal["vault.retention.changed"] = "vault.retention.changed"
    keep_days: int | None  # None = keep forever


@register
class WipeRequested(TypedEvent):
    type: Literal["vault.wipe.requested"] = "vault.wipe.requested"
    scope: Literal["all", "keyframes", "memory"]
