"""Event envelope shared by every context (F2 will add the concrete event types).

Events carry references (frame_ref), never pixels. Type names follow
`<context>.<entity>.<verb>`, e.g. `perception.zone.event`.
"""

from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class Event(BaseModel):
    model_config = ConfigDict(extra="allow")

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    type: str
    version: int = 1
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
