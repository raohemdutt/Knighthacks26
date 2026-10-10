from datetime import UTC, datetime

from contexts.memory.infrastructure.store import parse


def test_stored_timestamps_with_a_trailing_z_parse_on_python_3_10_too():
    """The Jetson runs Python 3.10, where datetime.fromisoformat() rejects 'Z'."""
    assert parse("2026-10-10T11:17:20.528259Z") == datetime(2026, 10, 10, 11, 17, 20, 528259, tzinfo=UTC)
    assert parse(None) is None
