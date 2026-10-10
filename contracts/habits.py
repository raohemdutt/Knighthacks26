from datetime import date
from typing import Literal

from contracts.events import TypedEvent, register
from contracts.types import Citation


@register
class SignalRecorded(TypedEvent):
    """One observed fact: a meal at the desk, an energy drink, the desk empty 2:10 to 8:40."""

    type: Literal["habits.signal.recorded"] = "habits.signal.recorded"
    kind: str  # "away_window", "desk_time", "drink", "meal", "due_soon"
    value: float  # minutes, count, or hours depending on kind
    source_event_id: str | None = None


@register
class DayClosed(TypedEvent):
    type: Literal["habits.day.closed"] = "habits.day.closed"
    habit_id: str  # from config/habits.yaml
    day: date
    met: bool
    value: float
    streak: int


@register
class NudgeRaised(TypedEvent):
    type: Literal["habits.nudge.raised"] = "habits.nudge.raised"
    nudge_id: str
    habit_id: str
    message: str
    tone: Literal["warn", "celebrate"] = "warn"
    evidence: list[Citation] = []
