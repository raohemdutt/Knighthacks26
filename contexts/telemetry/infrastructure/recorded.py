"""Recorded telemetry: a JSONL file with one event per line, in the same format as the event log.

    events = load(DEFAULT)      # fixtures/telemetry/synthetic_minute.jsonl

Every line must parse into a registered `telemetry.*` event. A bad line is an error, never a
skip: a fixture that silently dropped lines would replay less than it claims.
"""

from pathlib import Path

from contracts import parse_event
from contracts.events import Event, TypedEvent

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "telemetry"
DEFAULT = FIXTURES / "synthetic_minute.jsonl"


def load(path: str | Path = DEFAULT) -> list[Event]:
    """The events in a recorded file, in file order."""
    path = Path(path)
    out: list[Event] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = parse_event(line)
        except ValueError as exc:
            raise ValueError(f"{path.name}:{n}: not an event: {exc}") from exc
        if not event.type.startswith("telemetry.") or not isinstance(event, TypedEvent):
            raise ValueError(f"{path.name}:{n}: {event.type} is not a registered telemetry event")
        out.append(event)
    if not out:
        raise ValueError(f"{path.name}: no events")
    return out
