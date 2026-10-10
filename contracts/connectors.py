from datetime import datetime
from typing import Literal

from contracts.events import TypedEvent, register

ItemKind = Literal["assignment", "calendar_event", "message", "post", "page_change", "search_result", "place", "health"]


@register
class ItemSynced(TypedEvent):
    """One external item pulled by a scheduled read. Stored in Memory beside camera episodes."""

    type: Literal["connector.item.synced"] = "connector.item.synced"
    connector: str  # "canvas", "calendar", "site", "groupme", "search"
    kind: ItemKind
    external_id: str  # stable id at the source, so re-syncs update instead of duplicating
    title: str
    when: datetime | None = None  # due date, event start, post time
    url: str | None = None
    summary: str | None = None  # one line; never secrets or tokens


@register
class SyncFailed(TypedEvent):
    type: Literal["connector.sync.failed"] = "connector.sync.failed"
    connector: str
    error: str
    offline: bool = False  # True when the box is offline on purpose, so the UI shows "paused"
