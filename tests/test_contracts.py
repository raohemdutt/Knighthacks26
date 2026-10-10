from contracts.events import Event


def test_event_round_trips_through_json():
    e = Event(type="perception.zone.event", zone="door", frame_ref="kf/0001.jpg")
    back = Event.model_validate_json(e.model_dump_json())
    assert back == e
