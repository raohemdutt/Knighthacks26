"""Walking routes over the legs SPOT has walked: the network comes from the rounds, never from the picture."""

from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from contexts.gateway.app import create_app
from contexts.gateway.bootstrap import Settings
from contexts.spot.application.guide import Guide
from contexts.spot.application.service import Spot
from contexts.spot.domain.models import Action, Asset, Kit, Location
from contexts.spot.domain.route import FLOOR_CHANGE_COST, WalkNet, heading
from contexts.spot.infrastructure.kit import load_kit
from contexts.spot.tests.helpers import mission


def line_kit() -> Kit:
    """Five waypoints. One round walks A - B - C - D (D is upstairs); E is never visited. A to C direct was never walked."""
    locs = [Location(location_id=i, location_name=n, zone_type="ACCESS", floor=f, x_coord=x, y_coord=y, environment="NORMAL")
            for i, n, f, x, y in [("WP01", "Dock", 1, 0, 0), ("WP02", "Hall", 1, 30, 0), ("WP03", "Pump Bay", 1, 30, 40), ("WP04", "Loft", 2, 30, 50),
                                  ("WP05", "Annex", 1, 90, 90)]]
    m = mission(1)
    acts = [Action(action_id=f"A{n:04d}", mission_id=m.mission_id, sequence=n, timestamp=m.start_time + timedelta(minutes=n), location_id=loc, asset_id="P-1",
                   action_type="VISUAL_INSPECTION", battery_pct=80, status="SUCCESS", failure_type="", duration_sec=30, scenario=m.scenario,
                   configuration_version=m.configuration_version)
            for n, loc in enumerate(["WP01", "WP01", "WP02", "WP03", "WP03", "WP04", "WP03"], 1)]
    return Kit(missions=[m], actions=acts, readings=[], anomalies=[], locations=locs,
               assets=[Asset(asset_id="P-1", asset_name="Pump 1", asset_type="Pump", location_id="WP03", criticality="LOW")], replay=[], validation=None, path="(test)")


def test_the_network_is_exactly_the_legs_spot_walked():
    net = WalkNet.of(line_kit())
    assert net.walked == {("WP01", "WP02"): 1, ("WP02", "WP03"): 1, ("WP03", "WP04"): 2}  # there and back counts twice; staying put is no leg
    assert net.neighbours("WP03") == ["WP02", "WP04"] and net.neighbours("WP05") == []


def test_a_route_goes_round_by_the_walked_legs_not_straight_across():
    net = WalkNet.of(line_kit())
    out = net.route("WP01", "WP03")
    assert out["path"] == ["WP01", "WP02", "WP03"]  # 30 + 40 = 70, though the straight line is 50: nobody walked that
    assert out["length"] == 70.0 and out["floor_changes"] == 0
    assert [leg["times_walked"] for leg in out["legs"]] == [1, 1]
    up = net.route("WP01", "WP04")
    assert up["path"] == ["WP01", "WP02", "WP03", "WP04"] and up["floor_changes"] == 1 and up["legs"][-1]["floor_change"] is True
    assert net.route("WP04", "WP01")["path"] == ["WP04", "WP03", "WP02", "WP01"]  # walkable both ways
    assert net.route("WP01", "WP05") is None and net.route("WP01", "WP99") is None  # never joined; unknown
    assert FLOOR_CHANGE_COST > 0


def test_heading_words_follow_the_plan_as_drawn():
    assert heading((0, 0), (30, 2)) == "right" and heading((30, 0), (0, 0)) == "left"
    assert heading((0, 0), (1, 40)) == "down" and heading((0, 40), (0, 0)) == "up"
    assert heading((0, 0), (30, 30)) == "down and to the right" and heading((30, 30), (0, 0)) == "up and to the left"


def test_the_route_in_words():
    g = Guide(Spot(line_kit()))
    out = g.route("WP01", "P-1")  # a piece of equipment is reached at its waypoint
    assert out["found"] and out["path"] == ["WP01", "WP02", "WP03"] and out["to"] == "WP03"
    assert out["say"] == ["From Dock, go right on the plan to Hall.", "Then down on the plan to Pump Bay."]
    assert out["summary"] == "Dock to Pump Bay: 2 legs, all walked by SPOT before."
    assert "paths SPOT has walked" in out["note"] and "not metres" in out["note"]
    assert g.route(None, "asset:P-1")["from"] == "WP01"  # nobody said where from: the first stop of the latest round
    assert g.route("WP01", "WP04")["say"][-1] == "Then to floor 2, to Loft."
    assert g.route("WP03", "WP03")["say"] == ["You are already at Pump Bay."]
    never = g.route("WP01", "WP05")
    assert never["found"] is False and never["path"] == [] and "never walked a path" in never["say"][0]
    assert g.route("WP01", "nowhere")["found"] is False
    assert [p["walked"] for p in never["places"]] == [True, True, True, True, False]


def test_the_real_kit_and_the_endpoint(tmp_path):
    kit = load_kit(Path("fixtures/spot/kit"))
    net = WalkNet.of(kit)
    assert all(a in kit.location and b in kit.location and n >= 1 for (a, b), n in net.walked.items())
    out = Guide(Spot(kit)).route(None, "HX-202")
    assert out["found"] and out["path"][0] == "WP01" and out["path"][-1] == "WP05"
    assert all((min(leg["from"], leg["to"]), max(leg["from"], leg["to"])) in net.walked for leg in out["legs"])
    assert Guide(Spot(None)).route(None, "HX-202")["configured"] is False
    with TestClient(create_app(Settings(data_dir=tmp_path, connectors=False, memory_fixture=""))) as client:
        got = client.get("/api/spot/route", params={"from": "WP06", "to": "asset:HX-210"}).json()
        assert got["found"] and got["from"] == "WP06" and got["to"] == "WP11" and got["legs"] and got["say"]
        assert client.get("/api/spot/route").status_code == 422


@pytest.mark.parametrize("ref, place", [("WP05", "WP05"), ("wp05", "WP05"), ("HX-202", "WP05"), ("asset:HX-202", "WP05"), ("place:WP11", "WP11"), ("x", None)])
def test_a_place_can_be_named_three_ways(ref, place):
    assert Guide(Spot(load_kit(Path("fixtures/spot/kit"))))._place_of(ref) == place
