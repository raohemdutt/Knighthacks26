"""Contracts v1. Importing this package registers every event type."""

from contracts import (  # noqa: F401
    agent,
    capture,
    circle,
    connectors,
    day,
    experience,
    habits,
    impact,
    learning,
    memory,
    perception,
    play,
    runtime,
    telemetry,
    vault,
    workspace,
)
from contracts.events import REGISTRY, Event, TypedEvent, parse_event, register

__all__ = ["REGISTRY", "Event", "TypedEvent", "parse_event", "register"]
