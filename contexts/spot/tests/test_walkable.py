"""The walkable floor traced from the kit's plan picture, checked against the kit: these fail if the trace is wrong."""

import json
import math
from functools import lru_cache
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from contexts.gateway.app import create_app
from contexts.gateway.bootstrap import Settings
from contexts.spot.application.guide import TRACED_NOTE, Guide
from contexts.spot.application.service import Spot
from contexts.spot.domain.directions import along_words, directions, side_words, turn_words
from contexts.spot.domain.route import WalkNet
from contexts.spot.domain.walkable import Walkable
from contexts.spot.infrastructure.kit import load_kit

TRACE = Path("fixtures/spot/floor_walkable.json")


@lru_cache(maxsize=1)
def real() -> tuple[Guide, Walkable, WalkNet]:
    spot = Spot(load_kit(Path("fixtures/spot/kit")))
    return Guide(spot), spot.walkable, WalkNet.of(spot.kit)


def trace() -> dict:
    return json.loads(TRACE.read_text(encoding="utf-8"))


def samples(points: list[list[float]], w: int, h: int, every: float = 1.0):
    for (ax, ay), (bx, by) in zip(points, points[1:]):
        n = max(1, int(math.hypot((bx - ax) * w, (by - ay) * h) / every))
        for k in range(n + 1):
            yield ax + (bx - ax) * k / n, ay + (by - ay) * k / n


# --- the trace file itself ---


def test_the_trace_says_where_it_came_from_and_is_well_formed():
    t = trace()
    assert "by eye" in t["source"] and "picture" in t["source"]
    assert t["image"] == "fictional_floor_plan.png" and (t["width"], t["height"]) == (1095, 595)
    panels = json.loads(Path("fixtures/spot/waypoint_pixels.json").read_text())["panels"]
    for floor, fl in t["floors"].items():
        p = panels[floor]
        for item in [*fl["rooms"], *fl["doors"], *fl["obstacles"]]:
            x0, y0, x1, y1 = item["rect"]
            assert x0 < x1 and y0 < y1, item
            assert p["x0"] <= x0 and x1 <= p["x1"] and p["y0"] <= y0 and y1 <= p["y1"], f"{item} is outside floor {floor}'s part of the picture"
        assert all("seen" in d and d["name"].startswith("the ") for d in fl["doors"])
        assert all(r["name"] and r["kind"] in {"room", "corridor", "steps", "yard"} for r in fl["rooms"])
    # the same stairs on both floors, and nothing else joins them
    assert {c["id"] for c in t["floors"]["1"]["connectors"]} == {c["id"] for c in t["floors"]["2"]["connectors"]} != set()


def test_the_grid_is_about_the_size_asked_for_and_mostly_wall():
    _, wk, _ = real()
    f1, f2 = wk.floors[1].grid, wk.floors[2].grid
    assert 130 <= f1.cols <= 170 and 90 <= f1.rows <= 125
    assert 55 <= f2.cols <= 80 and 90 <= f2.rows <= 125  # floor 2 is a narrow panel: same cell size, fewer columns
    for g in (f1, f2):
        free = g.cols * g.rows - len(g.static)
        assert 0.15 < free / (g.cols * g.rows) < 0.75


def test_every_connector_stands_on_walkable_floor_on_both_floors():
    _, wk, _ = real()
    for n, f in wk.floors.items():
        for c in f.connectors:
            assert wk.free(n, c["x"], c["y"]), f"{c['id']} on floor {n} is inside something"


# --- against the kit ---


def test_every_waypoint_stands_on_free_floor_or_snaps_within_the_stated_radius():
    _, wk, _ = real()
    moved = {}
    for wp in sorted(wk.pins):
        got = wk.stand(wp)
        assert got is not None, f"{wp}: no walkable floor within {wk.snap_px} px of its pin"
        assert got[2] <= wk.snap_px and wk.free(int(wk.pins[wp]["floor"]), got[0], got[1])
        moved[wp] = got[2]
    assert len(moved) == 15
    # twelve pins are on (or a cell from) free floor; three sit on the machine their area is named after and stand beside it
    assert sorted(wp for wp, d in moved.items() if d > wk.cell) == ["WP06", "WP08", "WP09"]
    assert max(moved.values()) < 26


def test_every_leg_spot_walked_is_reachable_on_the_grid():
    _, wk, net = real()
    assert len(net.walked) >= 80
    same = cross = 0
    for (a, b) in net.walked:
        leg = wk.leg(a, b)
        assert leg is not None, f"SPOT walked {a} to {b} but the trace has no way between them"
        if net.floor[a] == net.floor[b]:
            same += 1
            assert leg["connector"] is None and len(leg["floors"]) == 1 and leg["floors"][0]["floor"] == net.floor[a]
        else:
            cross += 1
            assert leg["connector"] in {"centre-stairs", "north-stairs"} and [fl["floor"] for fl in leg["floors"]] == [net.floor[a], net.floor[b]]
    assert same > 0 and cross > 0


def test_every_pair_of_waypoints_is_joined_and_no_route_touches_an_obstacle():
    _, wk, _ = real()
    t = trace()
    ids = sorted(wk.pins)
    checked = 0
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            leg = wk.leg(a, b)
            assert leg is not None, f"no way from {a} to {b}"
            for part in leg["floors"]:
                fl = t["floors"][str(part["floor"])]
                open_rects = [x["rect"] for x in [*fl["rooms"], *fl["doors"]]]
                pad_x, pad_y = wk.cell / wk.width, wk.cell / wk.height  # a path runs through cell centres: it may sit a cell outside a rectangle's edge
                for x, y in samples(part["points"], wk.width, wk.height):
                    assert wk.free(part["floor"], x, y), f"{a}>{b} crosses a blocked cell at {x:.3f},{y:.3f}"
                    assert not any(r[0] < x < r[2] and r[1] < y < r[3] for r in (o["rect"] for o in fl["obstacles"])), f"{a}>{b} goes through equipment"
                    assert any(r[0] - pad_x <= x <= r[2] + pad_x and r[1] - pad_y <= y <= r[3] + pad_y for r in open_rects), f"{a}>{b} leaves the floor"
                    checked += 1
    assert checked > 20000


def test_every_piece_of_equipment_can_be_reached_from_its_waypoint_and_from_the_entrance():
    guide, wk, _ = real()
    kit = guide.spot.kit
    for asset in kit.assets:
        home = guide.route(asset.location_id, asset.asset_id)  # the mark is drawn at its waypoint: standing there is being there
        assert home["found"] and home["say"] == [f"You are already at {kit.location[asset.location_id].location_name}."]
        far = guide.route("WP01", asset.asset_id)
        assert far["found"] and far["routed"]["end"]["waypoint"] == asset.location_id
        assert far["routed"]["floors"] or asset.location_id == "WP01"
        assert far["precision"] == "about" and far["precision_reason"] == "traced from the picture"


# --- a small made-up floor: the route goes through the door, not the wall ---


def two_rooms(door: bool = True) -> Walkable:
    t = {"width": 200, "height": 100, "cell_px": 5, "clearance_px": 3, "snap_px": 20, "source": "test",
         "floors": {"1": {"rooms": [{"id": "a", "name": "the left room", "kind": "room", "rect": [0.05, 0.1, 0.45, 0.9]},
                                    {"id": "b", "name": "the right room", "kind": "room", "rect": [0.55, 0.1, 0.95, 0.9]}],
                          "doors": [{"id": "d", "name": "the middle door", "seen": True, "rect": [0.45, 0.7, 0.55, 0.85]}] if door else [],
                          "obstacles": [{"kind": "tank", "label": "the tank", "wp": "B", "rect": [0.7, 0.3, 0.8, 0.6]}], "connectors": []}}}
    return Walkable.of(t, {"A": {"floor": 1, "x": 0.2, "y": 0.2}, "B": {"floor": 1, "x": 0.75, "y": 0.45}})


def test_a_route_goes_through_the_door_and_round_the_tank():
    wk = two_rooms()
    assert wk.stand("A")[2] == 0 and 0 < wk.stand("B")[2] <= 20  # B's pin is on the tank: it stands beside it
    leg = wk.leg("A", "B")
    pts = leg["floors"][0]["points"]
    assert any(0.45 <= x <= 0.55 and 0.7 <= y <= 0.85 for x, y in samples(pts, 200, 100)), "it must pass the door"
    assert not any(0.7 < x < 0.8 and 0.3 < y < 0.6 for x, y in samples(pts, 200, 100))
    assert len(pts) <= 6  # string-pulled: a handful of corners, not a cell-by-cell staircase
    assert wk.leg("B", "A")["floors"][0]["points"] == pts[::-1]
    assert two_rooms(door=False).leg("A", "B") is None  # a wall with no door is a wall
    assert wk.snap(1, 0.5, 0.3) is not None and wk.snap(1, 0.5, 0.3, limit_px=2) is None  # in the wall: only a generous snap reaches a room


def test_turns_sides_and_shares_are_named_from_the_walkers_point_of_view():
    assert [turn_words(d) for d in (0, 10, 40, -40, 90, -90, 170, -170)] == [
        "Keep going", "Keep going", "Bear right", "Bear left", "Turn right", "Turn left", "Turn back to the right", "Turn back to the left"]
    assert [side_words(d) for d in (0, 80, -80, 179)] == ["straight ahead", "on your right", "on your left", "just behind you"]
    assert [along_words(s) for s in (0.1, 0.33, 0.5, 0.7, 0.95)] == [
        "", "about a third of the way along", "about halfway along", "about two thirds of the way along", "to the end"]


def test_words_come_from_the_geometry():
    wk = two_rooms()
    leg = wk.leg("A", "B")
    steps = directions(wk, leg, "the bench", "Pump 1", "B")
    says = [s["say"] for s in steps]
    assert says[0].startswith("From the bench,") and any("go through the middle door" in s for s in says)
    # heading east and down through the door, then up to the tank: that is a left turn for the walker
    assert any(s.startswith(("Turn left", "Bear left")) for s in says[1:-1])
    assert says[-1].startswith("Pump 1 is ") and "by the tank" in says[-1] and steps[-1]["arrive"] is True
    ups = [s["upto"] for s in steps]
    assert ups == sorted(ups) and ups[-1] == 1.0 and all(s["floor"] == 1 for s in steps)
    assert directions(wk, {"found": False, "floors": []}, "a", "b") == []


def test_the_real_route_in_words_and_never_in_metres():
    guide, _, _ = real()
    out = guide.route("WP01", "HX-202")
    assert out["found"] and out["walked"] is True and out["times_walked"] >= 1
    assert out["say"][0] == "From Main Entrance, go along the entrance path."
    assert any("the entrance steps" in s for s in out["say"]) and out["say"][-1].startswith("Heat exchanger 2 is ")
    assert out["note"] == TRACED_NOTE and out["via_walked_legs"]["found"] and "paths SPOT has walked" in out["via_walked_legs"]["note"]
    assert out["path"] == out["via_walked_legs"]["path"]  # the old keys still carry the walked-legs route
    up = guide.route("WP06", "HX-210")
    assert [p["floor"] for p in up["routed"]["floors"]] == [1, 2] and up["routed"]["connector"] in {"north-stairs", "centre-stairs"}
    assert any(s.get("stairs") and "up to floor 2" in s["say"] for s in up["steps"])
    assert any(s["say"].startswith("On floor 2,") for s in up["steps"])
    never = guide.route("WP02", "WP10")
    assert never["found"] and never["walked"] is False and "has not walked this leg" in never["summary"]
    assert all(p["walked"] is False for p in never["routed"]["floors"])
    for a, b in [("WP01", "HX-202"), ("WP06", "HX-210"), ("WP02", "WP10"), ("WP15", "WP01"), ("WP11", "WP14"), ("WP07", "WP03")]:
        got = guide.route(a, b)
        text = " ".join(got["say"] + [got["summary"], got["note"]]).lower()
        assert " metre" not in text and " meter" not in text and " m." not in text and "feet" not in text
        assert all(s["say"].endswith(".") and "  " not in s["say"] and " and and " not in s["say"] for s in got["steps"])
        assert len(got["steps"]) <= 10


def test_a_tapped_start_snaps_to_the_floor_or_is_refused():
    guide, wk, _ = real()
    out = guide.route("xy:0.30,0.57", "PANEL-212")
    assert out["found"] and out["from"] is None and out["via_walked_legs"] is None and out["walked"] is False
    assert out["say"][0].startswith("From where you are,") and out["routed"]["start"]["snapped_px"] == 0
    on_tank = guide.route("xy:0.36,0.58", "WP09")  # on the blue tank: stand beside it
    assert on_tank["found"] and 0 < on_tank["routed"]["start"]["snapped_px"] <= 60
    assert wk.free(1, on_tank["routed"]["start"]["x"], on_tank["routed"]["start"]["y"])
    for bad in ("xy:0.02,0.02", "xy:0.686,0.5", "xy:banana"):
        refused = guide.route(bad, "WP05")
        assert refused["found"] is False and refused["routed"] is None and "not on the floor I traced" in refused["say"][0]


def test_without_a_plan_the_answer_is_the_walked_legs_route_as_before():
    from contexts.spot.tests.test_route import line_kit

    out = Guide(Spot(line_kit())).route("WP01", "P-1")
    assert out["say"] == ["From Dock, go right on the plan to Hall.", "Then down on the plan to Pump Bay."] and "routed" not in out
    assert Guide(Spot(line_kit())).walkable() == {"configured": True, "available": False, "floors": {}, "waypoints": {}, "caption": ""}
    assert Guide(Spot(None)).walkable_legs() == {"configured": False, "available": False, "legs": {}}


def test_the_endpoints(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path, connectors=False, memory_fixture=""))) as client:
        got = client.get("/api/spot/route", params={"from": "WP06", "to": "asset:HX-210"}).json()
        assert got["found"] and got["precision"] == "about" and got["routed"]["floors"] and got["steps"] and got["via_walked_legs"]["legs"]
        assert got["routed"]["end"]["pin"] and got["routed"]["start"]["waypoint"] == "WP06"
        tap = client.get("/api/spot/route", params={"from": "xy:0.30,0.57", "to": "WP09"}).json()
        assert tap["found"] and tap["routed"]["start"]["name"] == "where you are"
        w = client.get("/api/spot/walkable").json()
        assert w["available"] and w["caption"] == TRACED_NOTE and set(w["floors"]) == {"1", "2"} and len(w["waypoints"]) == 15
        f1 = w["floors"]["1"]
        assert f1["rooms"] and f1["doors"] and f1["obstacles"] and f1["connectors"] and f1["grid"]["runs"]
        assert sum(c1 - c0 + 1 for _, c0, c1 in f1["grid"]["runs"]) == f1["grid"]["free"]
        legs = client.get("/api/spot/walkable/legs").json()
        assert len(legs["legs"]) == 105 and legs["legs"]["WP01>WP05"]["walked"] is True and legs["legs"]["WP02>WP10"]["walked"] is False


@pytest.mark.parametrize("ref", ["xy:0.3,0.5", "XY:0.30,0.57", "xy:1,0"])
def test_a_tap_is_read_as_a_point(ref):
    guide, _, _ = real()
    point = guide._point(ref)
    assert point is None or (point[0] in (1, 2) and 0 <= point[1] <= 1)
