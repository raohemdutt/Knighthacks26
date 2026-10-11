"""The room twin live: tracks on the floor, the Sort It route event and endpoint, the "which bin?" hint."""

import asyncio
import math
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from bus import EventBus
from contexts.demo import fakes
from contexts.gateway.app import INPUT_EVENTS, create_app
from contexts.gateway.bootstrap import Settings
from contexts.perception.fakes import room_scene, script
from contexts.play.domain.game import SortIt
from contexts.play.domain.pathfinding import SortingRules, ZoneMap
from contexts.play.room.api import room_router
from contexts.play.room.live import RoomLive, project_tracks, side_of
from contexts.play.room.plan import load_room
from contracts.perception import ObservationRecorded, PersonIdentified, PersonLost
from contracts.play import FacesCounted, RoundScored, RoundStarted, RoutePlanned
from contracts.types import Box, Track

T0 = datetime(2026, 10, 11, 15, tzinfo=UTC)


def at(s: float) -> datetime:
    return T0 + timedelta(seconds=s)


def track(track_id, label, x, bottom, w=0.2, h=0.6):
    return Track(track_id=track_id, label=label, confidence=0.9, box=Box(x1=x - w / 2, y1=bottom - h, x2=x + w / 2, y2=bottom))


def obs(s, camera, *tracks):
    return ObservationRecorded(ts=at(s), camera_id=camera, tracks=list(tracks))


def live(room="dorm"):
    return RoomLive(load_room(f"config/rooms/{room}.yaml"), SortingRules.load("config/sorting.yaml"))


def started(s, item="can", players=1, round_id="r1"):
    return RoundStarted(ts=at(s), round_id=round_id, session_id="s1", players=players, item=item, target_bin="recycling")


def polyline_clear_of(points, x, y, radius):
    for a, b in zip(points, points[1:], strict=False):
        for i in range(21):
            px, py = a.x + (b.x - a.x) * i / 20, a.y + (b.y - a.y) * i / 20
            if math.hypot(px - x, py - y) < radius:
                return False
    return True


# --- tracks on the floor ---


def test_a_seated_person_keys_on_the_desk_and_a_package_on_the_floor():
    plan = load_room("config/rooms/dorm.yaml")
    desk = {t.label: t for t in project_tracks(plan, "cam-desk", [track(1, "person", 0.5, 0.9, 0.3), track(3, "keys", 0.3, 0.7, 0.06, 0.04)])}
    assert desk["person"].seated and desk["person"].on == "chair" and desk["person"].z == 0
    assert (desk["keys"].on, desk["keys"].z, desk["keys"].held_by) == ("desk", 0.75, None)  # on the desk, not in the sitter's hand
    [package] = project_tracks(plan, "cam-door", [track(2, "package", 0.8, 0.9, 0.16, 0.14)])
    assert package.z == 0 and package.on is None and abs(package.x - 2.82) < 0.01 and abs(package.y - 1.4) < 0.01
    assert project_tracks(plan, "cam-garage", [track(1, "person", 0.5, 0.9)]) == []  # no calibration: not placed


def test_a_carried_can_is_in_the_hand_and_a_dropped_one_is_in_the_bin():
    plan = load_room("config/rooms/dorm.yaml")
    sortable = lambda label: label in ("can", "cup")  # noqa: E731
    person, can = project_tracks(plan, "cam-door", [track(5, "person", 0.6, 0.99), track(6, "can", 0.65, 0.53, 0.04, 0.07)], sortable)
    assert can.held_by == person.key == "cam-door:5" and (can.x, can.y, can.z) == (person.x, person.y, 1.0)
    person, can = project_tracks(plan, "cam-door", [track(5, "person", 0.224, 0.7375), track(6, "can", 0.15, 0.8, 0.04, 0.07)], sortable)
    assert can.held_by is None and can.in_bin == "recycling" and (can.x, can.y) == plan.bins()["recycling"].at
    [alone] = project_tracks(plan, "cam-door", [track(4, "can", 0.6, 0.5, 0.04, 0.07)], sortable)
    assert alone.z == 1.0 and alone.held_by is None  # above the floor line with nobody in view: carried, not lying by the wall


def test_tracks_go_stale_and_owner_recognition_tags_the_figure():
    room = live()
    room.on_identified(PersonIdentified(ts=at(0), camera_id="cam-desk", track_id=1, member_id="m-ana"))
    room.on_observation(obs(1, "cam-desk", track(1, "person", 0.5, 0.9, 0.3)))
    room.on_observation(obs(1, "cam-door", track(2, "package", 0.8, 0.9, 0.16, 0.14)))
    state = room.state(at(2))
    assert {t["key"]: t["member_id"] for t in state["tracks"]} == {"cam-desk:1": "m-ana", "cam-door:2": None}
    assert state["seen"]["package"]["camera"] == "cam-door" and "person" not in state["seen"]
    room.on_lost(PersonLost(ts=at(2), camera_id="cam-desk", track_id=1))
    assert room.state(at(2))["tracks"][0]["member_id"] is None
    room.on_observation(obs(9, "cam-door"))  # the desk camera has been quiet for 8 s
    assert room.state(at(9))["tracks"] == []
    assert room.state(at(60))["seen"]["package"]["x"] == 2.82  # where it was last seen is kept for the pin
    assert room.state(at(60))["tracks"] == []  # nothing reported for a minute: nobody is drawn


# --- the route event ---


def test_round_start_plans_a_route_from_the_player_and_publishes_it():
    room = live()
    room.on_observation(obs(0, "cam-door", track(5, "person", 0.6, 0.99), track(6, "can", 0.65, 0.53, 0.04, 0.07),
                            track(2, "package", 0.8, 0.9, 0.16, 0.14)))
    assert room.held_item() == "can"
    [event] = room.on_round_started(started(0))
    assert isinstance(event, RoutePlanned) and (event.reason, event.room, event.item, event.target_bin) == ("round_started", "dorm", "can", "recycling")
    [route] = event.routes
    assert route.track == "cam-door:5" and route.reachable and route.player == 1
    assert (route.points[0].x, route.points[0].y) == (2.24, 1.76)
    end = route.points[-1]
    recycling, trash = room.plan.bins()["recycling"], room.plan.bins()["trash"]
    assert math.hypot(end.x - recycling.at[0], end.y - recycling.at[1]) < 0.62 < math.hypot(end.x - trash.at[0], end.y - trash.at[1])
    assert abs(route.par_s - (route.distance_m / 1.2 + 2.5)) < 0.1 and 1.0 < route.distance_m < 2.5
    state = room.state(at(0))
    assert state["round"]["item"] == "can" and state["round"]["routes"][0]["track"] == "cam-door:5" and state["assist"] is None


def test_route_is_planned_again_when_someone_steps_into_it_and_only_then():
    room = live()
    me, can = track(5, "person", 0.6, 0.99), track(6, "can", 0.65, 0.53, 0.04, 0.07)
    room.on_observation(obs(0, "cam-door", me, can))
    [first] = room.on_round_started(started(0))
    assert room.on_observation(obs(1, "cam-door", me, can)) == []  # nothing changed: no new event
    friend = track(8, "person", 0.448, 0.875)  # steps onto the straight line to the bin
    fx, fy = room.plan.to_floor("cam-door", 0.448, 0.875)
    assert not polyline_clear_of(first.routes[0].points, fx, fy, 0.5)
    [again] = room.on_observation(obs(2, "cam-door", me, can, friend))
    assert again.reason == "replan" and again.round_id == "r1"
    assert polyline_clear_of(again.routes[0].points, fx, fy, 0.5)
    assert again.routes[0].distance_m > first.routes[0].distance_m and again.routes[0].par_s > first.routes[0].par_s
    assert room.on_observation(obs(3, "cam-door", me, can, friend)) == []  # still the same answer: silent
    assert room.on_observation(obs(3.2, "cam-door", me, can)) == []  # at most once a second


def test_two_players_get_two_routes_that_keep_apart():
    room = live("table")
    room.on_observation(obs(0, "cam-door", track(5, "person", 0.9, 0.99), track(6, "can", 0.95, 0.53, 0.04, 0.07), track(8, "person", 0.9, 0.7)))
    [event] = room.on_round_started(started(0, players=2))
    one, two = event.routes
    assert (one.player, one.track, two.player, two.track) == (1, "cam-door:5", 2, "cam-door:8")  # the holder is player 1
    assert one.reachable and two.reachable
    # away from their starts and from the bin they share, the two walks stay a body apart
    bin_at = room.plan.bins()["recycling"].at
    mids = [p for p in two.points[1:-1] if math.hypot(p.x - bin_at[0], p.y - bin_at[1]) > 1.0]
    assert all(polyline_clear_of(one.points[1:], p.x, p.y, 0.3) for p in mids)


def test_no_camera_sees_a_player_so_the_route_starts_at_the_rooms_start_point():
    room = live()
    [event] = room.on_round_started(started(0, item="bottle"))
    [route] = event.routes
    assert route.track is None and (route.points[0].x, route.points[0].y) == room.plan.start and route.reachable


def test_scored_round_reports_seconds_against_par_and_clears_the_round():
    room = live()
    room.on_observation(obs(0, "cam-door", track(5, "person", 0.6, 0.99), track(6, "can", 0.65, 0.53, 0.04, 0.07)))
    [event] = room.on_round_started(started(0))
    room.on_round_scored(RoundScored(ts=at(5), round_id="r1", session_id="s1", players=1, item="can", correct=True,
                                     seconds=5.0, path_len=9, score=150))
    state = room.state(at(5))
    assert state["round"] is None
    assert state["result"] == {"round_id": "r1", "item": "can", "correct": True, "seconds": 5.0, "score": 150, "players": 1,
                               "par_s": event.routes[0].par_s, "bin": "recycling", "ts": at(5).isoformat(), "right": 1, "rounds": 1}


def test_a_round_the_game_dropped_without_an_event_is_forgotten():
    active = ["r1"]
    room = RoomLive(load_room("config/rooms/dorm.yaml"), SortingRules.load("config/sorting.yaml"), active_round=lambda: active[0])
    room.on_round_started(started(0))
    room.on_observation(obs(1, "cam-door"))
    assert room.round is not None
    assert room.state(at(60))["round"] is not None and room.state(at(100))["round"] is None  # the cameras went quiet: not shown for ever
    active[0] = None  # nobody faced the camera for 10 s: the game abandoned it
    room.on_observation(obs(12, "cam-door"))
    assert room.round is None


# --- which bin? ---


def test_assist_says_which_bin_and_on_which_side():
    room = live()
    person = track(5, "person", 0.345, 0.925)
    room.on_observation(obs(0, "cam-door", person, track(7, "cup", 0.395, 0.5, 0.05, 0.07)))
    assist = room.state(at(0))["assist"]
    assert assist["text"] == "That cup goes in the trash, the grey bin on your right."
    assert (assist["item"], assist["bin"], assist["bin_id"], assist["track"], assist["person"]) == ("cup", "trash", "trash-bin", "cam-door:7", "cam-door:5")
    room.on_observation(obs(1, "cam-door", person, track(7, "soda can", 0.395, 0.5, 0.05, 0.07)))  # an alias sorts like its item
    assert room.state(at(1))["assist"]["text"] == "That soda can goes in recycling, the blue bin on your left."


def test_assist_needs_a_sortable_thing_near_the_bins_with_a_person_and_no_round():
    room = live()
    person, cup = track(5, "person", 0.345, 0.925), track(7, "cup", 0.395, 0.5, 0.05, 0.07)
    room.on_observation(obs(0, "cam-door", cup))  # carried by nobody the camera sees
    assert room.assist is None
    room.on_observation(obs(1, "cam-door", person, track(2, "package", 0.4, 0.9, 0.16, 0.14)))  # not something to sort
    assert room.assist is None
    room.on_observation(obs(2, "cam-door", track(5, "person", 0.95, 0.99), track(7, "cup", 1.0, 0.53, 0.05, 0.07)))  # by the door, far from the bins
    assert room.assist is None
    room.on_observation(obs(3, "cam-door", person, track(7, "cup", 0.38, 0.8, 0.05, 0.07)))  # already in the bin
    assert room.assist is None
    room.on_observation(obs(4, "cam-door", person, cup))
    assert room.assist is not None
    room.on_round_started(started(5, item="cup"))
    room.on_observation(obs(6, "cam-door", person, cup))
    assert room.assist is None and room.state(at(6))["assist"] is None  # the round's own route says it


def test_side_of():
    assert side_of((2, 2), (2, 0), (3, 0)) == "on your right"  # looking north, east is to the right
    assert side_of((2, 2), (2, 0), (1, 0)) == "on your left"
    assert side_of((2, 2), (2, 0), (2.05, 0.5)) == "in front of you"
    assert side_of((2, 2), (2, 4), (3, 4)) == "on your left"  # looking south


# --- the endpoints ---


def room_client(tmp_path: Path, token: str | None = None):
    config = tmp_path / "config"
    (config / "rooms").mkdir(parents=True)
    shutil.copy("config/rooms/dorm.yaml", config / "rooms" / "dorm.yaml")
    shutil.copy("config/zones.yaml", config / "zones.yaml")
    room = RoomLive(load_room(config / "rooms" / "dorm.yaml"), SortingRules.load("config/sorting.yaml"))
    app = FastAPI()

    def guard(authorization: str | None = None):
        from fastapi import HTTPException
        if token and authorization != f"Bearer {token}":
            raise HTTPException(401, "missing or wrong token")

    from fastapi import Header

    def authorized(authorization: str | None = Header(default=None)):
        guard(authorization)

    app.include_router(room_router(lambda: room, config, guard=authorized))
    return TestClient(app), room, config


def test_get_room_returns_the_plan_calibration_and_zones_on_the_floor(tmp_path):
    c, _, _ = room_client(tmp_path)
    body = c.get("/api/room").json()
    assert body["name"] == "dorm" and body["size"] == [4.2, 4.4] and body["rooms"] == ["dorm"]
    assert {f["type"] for f in body["furniture"]} >= {"bed", "desk", "bin"}
    cal = body["calibration"]["cam-door"]
    assert len(cal["image"]) == len(cal["floor"]) == 4 and len(cal["homography"]) == 3
    # the points the room file ships with are a placeholder, and the API says so
    assert {cam: c["source"] for cam, c in body["calibration"].items()} == {"cam-door": "default", "cam-desk": "default"}
    assert set(body["zones"]["cam-door"]) == {"door", "recycling", "trash"} and body["uncalibrated"] == []
    assert body["play"] == {"walk_speed": 1.2, "body_radius": 0.25, "start": [1.3, 3.7]}


def test_put_calibration_saves_four_taps_and_moves_the_tracks(tmp_path):
    c, room, config = room_client(tmp_path, token="s3cret")
    taps = {"image": [[0, 0.55], [1, 0.55], [1, 1], [0, 1]], "floor": [[0.5, 0], [3.4, 0], [3.4, 1.0], [0.5, 1.0]]}
    assert c.put("/api/room/calibration/cam-door", json=taps).status_code == 401
    auth = {"Authorization": "Bearer s3cret"}
    r = c.put("/api/room/calibration/cam-door", json=taps, headers=auth)
    assert r.status_code == 200 and r.json()["calibration"]["cam-door"]["floor"][2] == [3.4, 1.0]
    assert load_room(config / "rooms" / "dorm.yaml").calibration["cam-door"].floor[2] == (3.4, 1.0)  # on disk
    # only the camera that was given its four points is calibrated, and a restart (the file) still knows which
    assert {cam: c["source"] for cam, c in r.json()["calibration"].items()} == {"cam-door": "set", "cam-desk": "default"}
    assert {cam: c.source for cam, c in load_room(config / "rooms" / "dorm.yaml").calibration.items()} == {"cam-door": "set", "cam-desk": "default"}
    room.on_observation(obs(0, "cam-door", track(2, "package", 0.8, 0.9, 0.16, 0.14)))
    assert abs(room.tracks()[0].y - 0.778) < 0.01  # was 1.4 m from the wall with the old calibration
    line = {"image": [[0, 0.5], [0.5, 0.5], [1, 0.5], [0, 1]], "floor": taps["floor"]}
    assert c.put("/api/room/calibration/cam-door", json=line, headers=auth).status_code == 422
    assert c.put("/api/room/calibration/cam-door", json={**taps, "image": [[0, 0], [2, 0], [1, 1], [0, 1]]}, headers=auth).status_code == 422
    assert c.put("/api/room/calibration/cam-door", json={**taps, "floor": [[0, 0], [40, 0], [40, 9], [0, 9]]}, headers=auth).status_code == 422
    new = c.put("/api/room/calibration/cam-hall", json=taps, headers=auth)  # a camera the room did not have yet
    assert new.status_code == 200 and "cam-hall" in new.json()["calibration"]


def test_room_route_endpoint(tmp_path):
    c, room, _ = room_client(tmp_path)
    r = c.get("/api/play/room-route", params={"item": "can"}).json()
    assert (r["bin"], r["zone"], r["room"], r["reachable"]) == ("recycling", "recycling", "dorm", True)
    assert r["from"] == {"x": 1.3, "y": 3.7, "track": None} and r["points"][0] == [1.3, 3.7]
    assert r["distance_m"] > 3 and abs(r["par_s"] - (r["distance_m"] / 1.2 + 2.5)) < 0.06
    assert c.get("/api/play/room-route", params={"item": "soda can"}).json()["bin"] == "recycling"
    assert c.get("/api/play/room-route", params={"item": "cup", "x": 3.0, "y": 1.0}).json()["from"] == {"x": 3.0, "y": 1.0, "track": None}
    assert c.get("/api/play/room-route", params={"item": "unicorn"}).status_code == 404
    assert c.get("/api/play/room-route", params={"item": "can", "from_track": "cam-door:5"}).status_code == 404
    room.on_observation(ObservationRecorded(ts=datetime.now(UTC), camera_id="cam-door", tracks=[track(5, "person", 0.6, 0.99)]))
    seen = c.get("/api/play/room-route", params={"item": "can", "from_track": "cam-door:5"}).json()
    assert seen["from"] == {"x": 2.24, "y": 1.76, "track": "cam-door:5"}
    assert c.get("/api/play/room-route", params={"item": "can"}).json()["from"]["track"] == "cam-door:5"  # the person a camera sees
    live_state = c.get("/api/room/live").json()
    assert [t["key"] for t in live_state["tracks"]] == ["cam-door:5"] and live_state["round"] is None


def test_no_room_file_means_404_not_a_crash():
    app = FastAPI()
    app.include_router(room_router(lambda: None, Path("config")))
    c = TestClient(app)
    assert c.get("/api/room").status_code == c.get("/api/room/live").status_code == 404
    assert c.get("/api/play/room-route", params={"item": "can"}).status_code == 404


# --- the whole scene, through the real game and the gateway ---


def test_sort_it_takes_the_held_item_when_the_room_sees_one():
    game = SortIt(ZoneMap.load("cam-door"), SortingRules.load(), items=["bottle", "cup"])
    game.on_faces(FacesCounted(ts=at(0), camera_id="cam-door", count=1, facing=1))
    game.pick_item = lambda: "can"
    [start] = game.on_faces(FacesCounted(ts=at(2), camera_id="cam-door", count=1, facing=1))
    assert (start.item, start.target_bin) == ("can", "recycling")
    game.round = None
    game.pick_item = lambda: None  # nothing in view: the rota, as before
    [start] = game.on_faces(FacesCounted(ts=at(5), camera_id="cam-door", count=1, facing=1))
    assert start.item == "cup"  # the second of the rota: every round takes a turn


def test_the_room_scene_keeps_the_original_half_minute():
    old, new = script(T0), room_scene(T0)
    assert len(new) == 126 and new[-1].ts == at(62)
    assert sum(1 for o in new for t in o.tracks if t.track_id == 5) == 24  # the player is in view from 36 s to 59 s
    assert [o.tracks for o in new[:62]] == [o.tracks for o in old]  # the first 30 s are script(), track for track


def test_the_fake_scene_plays_a_whole_round_through_the_gateway(tmp_path):
    """python -m contexts.demo fake perception --gateway ...: a face starts the round on the can in hand, a friend
    steps into the way (the route is planned again), the can lands in recycling (scored), then "which bin?" for a cup."""
    start = datetime.now(UTC).replace(microsecond=0)
    events = asyncio.run(fakes.publish("perception", tmp_path / "scratch", start))
    pushed = sorted((e for e in events if e.type.startswith(INPUT_EVENTS)), key=lambda e: e.ts)
    settings = Settings(data_dir=tmp_path / "data", connectors=False, ntfy_url=None, llm_url=None)
    with TestClient(create_app(settings)) as c:
        atlas = c.app.state.atlas
        seen: list = []
        assists: list = []

        def watch(e):
            seen.append(e)
            if atlas.room.assist:
                assists.append(((e.ts - start).total_seconds(), atlas.room.assist["text"]))

        atlas.bus.subscribe("", watch)
        body = [e.model_dump(mode="json", exclude={"event_id"}) for e in pushed]
        assert c.post("/api/events", json=body).status_code == 200
        when = lambda e: int((e.ts - start).total_seconds())  # noqa: E731
        rounds = [(when(e), e.type, e.item) for e in seen if e.type.startswith("play.round.")]
        assert rounds == [(39, "play.round.started", "can"), (45, "play.round.scored", "can")]
        scored = next(e for e in seen if e.type == "play.round.scored")
        assert scored.correct and scored.seconds == 6.0
        routes = [e for e in seen if isinstance(e, RoutePlanned)]
        assert [(when(e), e.reason) for e in routes] == [(39, "round_started"), (40, "replan")]
        assert all(e.routes[0].track == "cam-door:5" and e.routes[0].reachable for e in routes)
        assert routes[1].routes[0].distance_m > routes[0].routes[0].distance_m  # around the friend
        assert {text for s, text in assists if 52 <= s <= 56} == {"That cup goes in the trash, the grey bin on your right."}
        assert not [a for a in assists if 39 <= a[0] <= 45]  # never during the round
        state = c.get("/api/room/live").json()
        assert state["result"]["correct"] and state["result"]["par_s"] == routes[1].routes[0].par_s and state["round"] is None
        assert {"keys", "package", "can", "cup"} <= set(state["seen"])
        assert (state["seen"]["keys"]["on"], state["seen"]["can"]["on"], state["seen"]["cup"]["on"]) == ("desk", "recycling-bin", "trash-bin")
        assert c.get("/api/play/accuracy").json()["rounds"] == 1
        assert c.get("/api/impact").json()["total"]["recycled"] == 2  # the two cans; the cup went to the trash


def test_route_events_are_not_planned_again_on_replay(tmp_path):
    bus = EventBus(tmp_path)
    room = live()
    bus.subscribe("perception.observation", room.on_observation, replay=True)
    bus.subscribe("play.round.started", room.on_round_started)
    out: list = []
    bus.subscribe("play.route", out.append, replay=True)

    async def run():
        await bus.publish(obs(0, "cam-door", track(5, "person", 0.6, 0.99), track(6, "can", 0.65, 0.53, 0.04, 0.07)))
        await bus.publish(started(0))

    asyncio.run(run())
    assert len(out) == 1
    again = EventBus(tmp_path)
    fresh = live()
    again.subscribe("perception.observation", fresh.on_observation, replay=True)
    again.subscribe("play.round.started", fresh.on_round_started)
    replayed: list = []
    again.subscribe("play.route", replayed.append, replay=True)
    asyncio.run(again.replay_into())
    assert replayed == out and fresh.round is None and fresh.seen["can"]["camera"] == "cam-door"
