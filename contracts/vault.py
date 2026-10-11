from typing import Literal

from contracts.events import TypedEvent, register
from contracts.types import CameraId


@register
class PolicyChanged(TypedEvent):
    type: Literal["vault.policy.changed"] = "vault.policy.changed"
    blur_faces: bool
    paused_cameras: list[CameraId] = []
    offline: bool = False
    # The live picture on the owner's own signed-in screen: "clear" shows faces there, and only there; "owner" shows
    # only the scanned owner's own face there and blurs every other face; "blurred" blurs everyone. None is an event
    # from before this field existed: nothing recorded (contexts/vault/application/policy.py says what that means).
    # Saved pictures are `blur_faces`.
    live_faces: Literal["blurred", "clear", "owner"] | None = None
    # A training export the owner approves: "owner_clear" lets the scanned owner's own face stay unblurred in it (every
    # other face is still blurred, and nothing else that leaves the box changes). None or "blurred": everyone blurred.
    train_faces: Literal["blurred", "owner_clear"] | None = None


@register
class RetentionChanged(TypedEvent):
    type: Literal["vault.retention.changed"] = "vault.retention.changed"
    keep_days: int | None  # None = keep forever


@register
class WipeRequested(TypedEvent):
    type: Literal["vault.wipe.requested"] = "vault.wipe.requested"
    scope: Literal["all", "keyframes", "memory"]
