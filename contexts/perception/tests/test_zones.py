from datetime import UTC, datetime, timedelta

from contexts.perception.domain.zones import ZoneEngine, anchor, load_zones, point_in_polygon
from contracts.perception import ObservationRecorded
from contracts.types import Box, Track

T0 = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)
DOOR = (0.75, 0.9)  # bottom-center points per config/zones.yaml
RECYCLING = (0.15, 0.8)
NOWHERE = (0.5, 0.3)


def track(track_id, label, at):
    x, bottom = at
    return Track(track_id=track_id, label=label, confidence=0.9, box=Box(x1=x - 0.05, y1=bottom - 0.2, x2=x + 0.05, y2=bottom))


def obs(at, *tracks, camera="cam-door"):
    return ObservationRecorded(ts=T0 + timedelta(seconds=at), camera_id=camera, tracks=list(tracks),
                               frame_ref=f"frames:{camera}:{at}")


def engine(**kwargs):
    return ZoneEngine.load("config/zones.yaml", **kwargs)


def run(eng, observations):
    return [e for o in observations for e in eng.on_observation(o)]


def brief(events):
    return [(int((e.ts - T0).total_seconds()), e.track_id, e.zone_id, e.kind, e.dwell_s) for e in events]


def test_loads_every_camera_and_uses_the_bottom_center():
    zones = load_zones("config/zones.yaml")
    assert set(zones) == {"cam-door", "cam-desk"} and set(zones["cam-door"]) == {"door", "recycling", "trash"}
    tall = Box(x1=0.7, y1=0.1, x2=0.8, y2=0.9)  # head above the zone, feet in it
    assert anchor(tall) == (0.75, 0.9)
    assert point_in_polygon(*anchor(tall), zones["cam-door"]["door"])
    assert not point_in_polygon(0.75, 0.5, zones["cam-door"]["door"])


def test_enter_then_repeated_dwell_then_exit():
    eng = engine()
    stay = [obs(s, track(7, "package", DOOR)) for s in range(10)]
    events = run(eng, stay + [obs(10, track(7, "package", NOWHERE))])
    assert brief(events) == [(0, 7, "door", "enter", None), (3, 7, "door", "dwell", 3.0), (6, 7, "door", "dwell", 6.0),
                             (9, 7, "door", "dwell", 9.0), (10, 7, "door", "exit", None)]
    assert events[2].label == "package" and events[2].camera_id == "cam-door"
    assert [e.frame_ref for e in events] == [f"frames:cam-door:{s}" for s in (0, 3, 6, 9, 10)]
    assert eng.stays == {}


def test_dwell_period_is_configurable_and_survives_missed_samples():
    eng = engine(dwell_every_s=2)
    events = run(eng, [obs(s, track(1, "person", DOOR)) for s in (0, 1, 2, 3, 4)])
    assert [e.dwell_s for e in events if e.kind == "dwell"] == [2.0, 4.0]
    eng = engine()  # samples at 3 s and 6 s were dropped: one dwell each, with the real time
    events = run(eng, [obs(s, track(1, "person", DOOR)) for s in (0, 2, 4, 5, 7)])
    assert brief(events)[1:] == [(4, 1, "door", "dwell", 4.0), (7, 1, "door", "dwell", 7.0)]


def test_vanished_track_exits_on_flush_after_the_grace_period():
    eng = engine()
    run(eng, [obs(0, track(7, "package", DOOR)), obs(1, track(7, "package", DOOR))])
    assert eng.flush(T0 + timedelta(seconds=3)) == []  # last seen at 1 s: still within 3 s
    [gone] = eng.flush(T0 + timedelta(seconds=60))
    assert brief([gone]) == [(4, 7, "door", "exit", None)]  # last seen + grace, not the ticker's clock
    assert gone.label == "package" and gone.frame_ref == "frames:cam-door:1"
    assert eng.flush(T0 + timedelta(seconds=120)) == []


def test_vanished_track_exits_on_a_later_observation_with_the_same_event():
    by_flush, by_obs = engine(), engine()
    seen = [obs(0, track(7, "package", DOOR)), obs(1, track(7, "package", DOOR))]
    run(by_flush, seen)
    run(by_obs, seen)
    assert run(by_obs, [obs(2), obs(3)]) == []
    assert by_obs.on_observation(obs(4)) == by_flush.flush(T0 + timedelta(seconds=30))  # same id: the bus keeps one


def test_a_blink_shorter_than_the_grace_period_keeps_the_stay():
    seen = track(7, "package", DOOR)
    events = run(engine(), [obs(0, seen), obs(1), obs(2), obs(2.5, seen), obs(3, seen)])
    assert brief(events) == [(0, 7, "door", "enter", None), (3, 7, "door", "dwell", 3.0)]


def test_track_outside_every_zone_emits_nothing():
    eng = engine()
    assert run(eng, [obs(s, track(1, "person", NOWHERE)) for s in range(10)]) == []
    assert run(eng, [obs(0, track(1, "person", DOOR), camera="cam-unknown")]) == []
    assert eng.flush(T0 + timedelta(minutes=5)) == []


def test_two_tracks_in_two_zones_are_independent():
    eng = engine()
    events = run(eng, [
        obs(0, track(1, "package", DOOR)),
        obs(1, track(1, "package", DOOR), track(2, "can", RECYCLING)),
        obs(2, track(1, "package", DOOR), track(2, "can", RECYCLING)),
        obs(3, track(1, "package", DOOR), track(2, "can", NOWHERE)),
        obs(4, track(1, "package", DOOR)),
    ])
    assert brief(events) == [(0, 1, "door", "enter", None), (1, 2, "recycling", "enter", None),
                             (3, 1, "door", "dwell", 3.0), (3, 2, "recycling", "exit", None)]
    assert list(eng.stays) == [("cam-door", 1, "door")]


def test_same_track_id_on_two_cameras_is_two_stays():
    eng = engine()
    events = run(eng, [obs(0, track(1, "person", DOOR)), obs(0, track(1, "person", (0.5, 0.9)), camera="cam-desk")])
    assert [(e.camera_id, e.zone_id, e.kind) for e in events] == [("cam-door", "door", "enter"), ("cam-desk", "desk", "enter")]


def test_moving_between_zones_exits_one_and_enters_the_other():
    events = run(engine(), [obs(0, track(4, "can", (0.38, 0.8))), obs(1, track(4, "can", RECYCLING))])
    assert brief(events) == [(0, 4, "trash", "enter", None), (1, 4, "recycling", "enter", None), (1, 4, "trash", "exit", None)]


def test_replay_gives_identical_events():
    observations = [obs(s, track(7, "package", DOOR), track(2, "can", RECYCLING)) for s in range(8)] + [obs(s) for s in range(8, 14)]
    first, second = run(engine(), observations), run(engine(), observations)
    assert first == second and len(first) == 8  # 2 enters, 4 dwells, 2 exits; ids included
    assert len({e.event_id for e in first}) == len(first)
