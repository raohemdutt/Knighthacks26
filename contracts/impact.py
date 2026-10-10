from typing import Literal

from contracts.events import TypedEvent, register
from contracts.types import CameraId, FrameRef, ZoneId

Bin = Literal["recycling", "trash", "compost"]


@register
class ItemSeen(TypedEvent):
    type: Literal["impact.item.seen"] = "impact.item.seen"
    item_id: str  # one per distinct item (track id + camera, or a caption object)
    label: str  # "can", "bottle", "cup"
    camera_id: CameraId
    recyclable: bool  # from config/sorting.yaml
    frame_ref: FrameRef | None = None


@register
class ItemDisposed(TypedEvent):
    type: Literal["impact.item.disposed"] = "impact.item.disposed"
    item_id: str
    label: str
    zone: ZoneId  # bin zone it ended in
    bin: Bin
    correct: bool  # per config/sorting.yaml
    frame_ref: FrameRef | None = None


@register
class CountUpdated(TypedEvent):
    type: Literal["impact.count.updated"] = "impact.count.updated"
    items_seen: int
    recyclable_seen: int
    recycled: int  # recyclables that ended in recycling
    contamination: int  # non-recyclables that ended in recycling
    diversion_rate: float  # recycled / recyclable_seen
