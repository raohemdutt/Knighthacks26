"""Event envelope shared by every context.

Events carry references (frame_ref), never pixels. Type names follow
`<context>.<entity>.<verb>`, e.g. `perception.zone.event`. `ts` is when the
thing happened, not when it was logged.

Concrete events subclass `Event`, pin `type` with a Literal default, and
register with `@register` so the log can be replayed into typed objects.
"""

from datetime import UTC, datetime
from typing import ClassVar, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class Event(BaseModel):
    model_config = ConfigDict(extra="allow")

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    type: str
    version: int = 1
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TypedEvent(Event):
    """Base for concrete events: unknown fields are a bug, not data."""

    model_config = ConfigDict(extra="forbid")
    TYPE: ClassVar[str]


REGISTRY: dict[str, type[TypedEvent]] = {}

E = TypeVar("E", bound=type[TypedEvent])


def register(cls: E) -> E:
    type_name = cls.model_fields["type"].default
    if type_name in REGISTRY:
        raise ValueError(f"duplicate event type {type_name}")
    cls.TYPE = type_name
    REGISTRY[type_name] = cls
    return cls


def parse_event(raw: str | bytes | dict) -> Event:
    """Parse one log line (or dict) into its registered class, else a bare Event."""
    data = Event.model_validate_json(raw) if isinstance(raw, str | bytes) else Event.model_validate(raw)
    cls = REGISTRY.get(data.type)
    return cls.model_validate(data.model_dump()) if cls else data
