import pytest
from pydantic import ValidationError

from contracts import REGISTRY, Event, parse_event
from contracts.memory import EpisodeClosed
from contracts.perception import ZoneEvent


def test_event_round_trips_through_json():
    e = Event(type="something.new.happened", zone="door")
    assert parse_event(e.model_dump_json()) == e


def test_typed_event_parses_back_to_its_class():
    e = ZoneEvent(camera_id="cam-door", track_id=7, label="package", zone_id="door", kind="dwell", dwell_s=5)
    back = parse_event(e.model_dump_json())
    assert isinstance(back, ZoneEvent)
    assert back == e


def test_typed_event_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        ZoneEvent(camera_id="c", track_id=1, label="x", zone_id="z", kind="enter", oops=1)


def test_every_context_registered():
    contexts = {t.split(".")[0] for t in REGISTRY}
    assert contexts == {
        "agent", "capture", "circle", "connector", "day", "experience", "habits", "impact", "learning", "memory",
        "perception", "play", "runtime", "telemetry", "vault", "workspace",
    }
    assert EpisodeClosed.TYPE == "memory.episode.closed"
