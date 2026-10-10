"""Contracts v1. Importing this package registers every event type."""

from contracts import (  # noqa: F401
    agent,
    capture,
    experience,
    learning,
    memory,
    perception,
    runtime,
    vault,
    workspace,
)
from contracts.events import REGISTRY, Event, TypedEvent, parse_event, register

__all__ = ["REGISTRY", "Event", "TypedEvent", "parse_event", "register"]
