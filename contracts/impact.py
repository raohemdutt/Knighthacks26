from datetime import date
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


# --- the Impact screen (contexts/impact/application/story.py): additive, nothing above changed ---


@register
class MilestoneReached(TypedEvent):
    """The table passed a round number this week: a sentence the phone and the voice can celebrate.
    Published once per (week, kind, threshold); the event id is built from those, so it is never said twice."""

    type: Literal["impact.milestone.reached"] = "impact.milestone.reached"
    week_start: date  # the Monday of the local week
    kind: Literal["kept_out", "clean_streak"]  # recyclables that reached recycling this week; clean days in a row
    threshold: int  # the round number passed: 10 items, 3 days
    value: int  # the count when it was passed
    text: str  # "That's 10 items kept out of landfill this week."
    member_id: str | None = None  # None: the whole table


@register
class PracticeAdded(TypedEvent):
    """Someone tapped "teach me" on a wrong disposal: that item joins the Sort It practice set for the week."""

    type: Literal["impact.practice.added"] = "impact.practice.added"
    label: str  # the item as the rules file names it: "cup"
    item_id: str | None = None  # the disposal that prompted it
    member_id: str | None = None  # whose practice set, when owner recognition is on
