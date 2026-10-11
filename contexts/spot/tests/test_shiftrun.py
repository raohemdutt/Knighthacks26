"""Shift Run (domain/shiftrun.py, application/shiftrun.py): a toy model from the kit's recorded actions. Pure and deterministic:
the same plan gives the same result, the score is an expected value, and the obvious things hold (urgent first beats urgent last,
a lower turn-back level only adds worse checks)."""

import asyncio
import inspect
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bus import EventBus
from contexts.gateway.app import create_app
from contexts.gateway.bootstrap import Settings
from contexts.spot.application import shiftrun as app_shiftrun
from contexts.spot.application.service import Spot
from contexts.spot.application.shiftrun import ShiftRun
from contexts.spot.domain import shiftrun as sr
from contexts.spot.infrastructure.kit import load_kit
from contexts.spot.tests.test_plain import readable
from contexts.spot.wiring import build_spot
from contexts.workspace.domain.approvals import Approvals
from contracts.experience import ApprovalDecided
from runtime import FakeRuntime


@pytest.fixture(scope="module")
def spot():
    return Spot(load_kit(Path("fixtures/spot/kit")))


@pytest.fixture(scope="module")
def run(spot):
    return ShiftRun(spot)


def test_the_model_is_measured_from_the_kit(spot, run):
    m = run.model
    assert (m["base_k"], m["base_n"], m["rounds"]) == (49, 270, 27) and m["start_battery"] == 79.0
    assert 4 < m["drop"]["same"] < 5.5 and 2 < m["seconds"]["same"] / 60 < 3.5  # the kit: about 4.6 points and 2.7 minutes a step
    assert m["by_type"]["ACOUSTIC_INSPECTION"]["ratio"] > 1 > m["by_type"]["VISUAL_INSPECTION"]["ratio"]
    assert m["by_band"][(0, 20)]["ratio"] == sr.OR_MAX and m["by_band"][(20, 40)]["ratio"] > 1 > m["by_band"][(60, 80)]["ratio"]
    assert all(sr.OR_MIN <= v["ratio"] <= sr.OR_MAX for table in ("by_type", "by_location", "by_band", "late") for v in m[table].values())
    assert m["worst_location"] == "WP09" and set(m["assets"]) == {a.asset_id for a in spot.kit.assets}
    top = max(m["assets"].values(), key=lambda a: a["value"])
    assert top["asset_id"] == "HX-202" and top["value"] == round(sum(top["value_parts"].values()), 1) and top["value_parts"]["triage"] == 72.3
    fresh, stale = m["assets"]["VALVE-207"], m["assets"]["GAUGE-209"]  # both HIGH importance: one checked 20 minutes ago, one 2 days ago
    assert stale["value"] > fresh["value"] and stale["value_parts"]["stale"] == sr.STALE_VALUE_CAP


def test_a_thin_group_says_nothing_and_no_probability_runs_away(run):
    assert sr._ratio(3, 4, 0.18) == 1.0  # under MIN_N rows: no conclusion
    assert sr._ratio(0, 40, 0.18) == sr.OR_MIN and sr._ratio(40, 40, 0.18) == sr.OR_MAX
    worst = sr.fail_probability(run.model, "ACOUSTIC_INSPECTION", "WP09", 5, 12)
    best = sr.fail_probability(run.model, "VISUAL_INSPECTION", "WP13", 90, 1)
    assert sr.P_MIN <= best["p"] < worst["p"] <= sr.P_MAX
    # battery and position are one effect in the kit: the larger ratio is used, never their product
    assert worst["late_low"] == max(run.model["by_band"][(0, 20)]["ratio"], run.model["late"][True]["ratio"])


def test_same_plan_same_result_and_the_seed_only_changes_the_play(run):
    route = ["HX-202", "HX-204", "PANEL-212", "VALVE-207", "HX-213"]
    a, b = run.simulate(route, 40, seed=7), run.simulate(route, 40, seed=7)
    assert a == b and json.dumps(a)
    plays = {tuple(s["played"] for s in run.simulate(route, 40, seed=n)["steps"]) for n in range(40)}
    scores = {run.simulate(route, 40, seed=n)["score"] for n in range(40)}
    assert len(plays) > 1 and len(scores) == 1  # outcomes are drawn for the animation; the score is an expected value


def test_urgent_first_beats_urgent_last(run):
    others = ["VALVE-207", "HX-216", "MOTOR-206", "HX-201", "PANEL-215", "COMP-214", "PUMP-205", "HX-208", "PANEL-211"]
    first = run.simulate(["HX-202", *others], 25)
    last = run.simulate([*others, "HX-202"], 25)
    assert first["score"] > last["score"]
    assert first["steps"][0]["p_fail"] < last["steps"][-1]["p_fail"] and last["steps"][-1]["low"]
    assert any("You left Heat exchanger 2 (HX-202) for stop 10" in r["text"] for r in last["reasons"])
    assert any(r["tone"] == "good" and "Heat exchanger 2 (HX-202) is stop 1" in r["text"] for r in first["reasons"])


def test_a_lower_turn_back_level_only_adds_checks_that_are_no_better(run):
    route = ["HX-216"] * sr.MAX_STOPS  # one kind of check at one place, so only battery and position change
    results = {level: sr.simulate(run.model, route, level) for level in (60, 45, 30, 15)}
    reached = {level: [s for s in r["steps"] if s["outcome"] == "checked"] for level, r in results.items()}
    for high, low in ((60, 45), (45, 30), (30, 15)):
        assert len(reached[low]) >= len(reached[high])
        shared = len(reached[high])
        assert [s["p_fail"] for s in reached[low][:shared]] == [s["p_fail"] for s in reached[high]]  # the early stops are untouched
        extra = reached[low][shared:]
        if extra and shared:
            assert min(s["p_fail"] for s in extra) >= max(s["p_fail"] for s in reached[high])  # what is unlocked is never more likely to work
    probabilities = [s["p_fail"] for s in reached[15]]
    assert probabilities == sorted(probabilities) and probabilities[-1] > probabilities[0]
    assert results[60]["not_reached"] > results[15]["not_reached"] and results[60]["stopped"] == "turn_back"
    assert results[15]["end_battery_pct"] < sr.DANGER and results[60]["end_battery_pct"] > sr.LOW


def test_atlas_plan_is_at_least_as_good_as_ours_and_starts_with_what_matters(run):
    best = run.recommended()
    assert best["route"][0] == "HX-202" and best["route"][1] == "HX-204" and best["stars"] == 3
    assert len(set(best["route"])) == len(best["route"]) <= sr.MAX_STOPS and best["end_battery_pct"] >= sr.DANGER
    for route, level in ((run.model["last_route"], 40), (list(reversed(best["route"])), 25), (["VALVE-207", "MOTOR-206"], 50), ([], 40)):
        mine = run.simulate(route, level)
        assert mine["score"] <= best["score"] and 0 <= mine["stars"] <= 3 and mine["best_score"] == best["score"]
        assert mine["recommended"]["route"] == best["route"] and 1 <= len(mine["differences"]) <= 3
    same = run.simulate(best["route"], best["turn_back_at"])
    assert same["stars"] == 3 and same["differences"] == ["Your plan is the same as Atlas's."]
    assert run.simulate(["VALVE-207", "MOTOR-206"], 50)["stars"] == 0
    assert run.recommended() == run.recommended()


def test_running_into_the_danger_band_costs_and_is_said(run):
    long = [a["asset_id"] for a in run.board()["candidates"]][: sr.MAX_STOPS]
    careless, careful = run.simulate(long, 5), run.simulate(long, 40)
    assert careless["end_battery_pct"] < sr.DANGER <= careful["end_battery_pct"] and careful["not_reached"] > 0
    assert any("too close" in r["text"] for r in careless["reasons"]) and any("Turning back at 40%" in r["text"] for r in careful["reasons"])
    # a round that starts on a half-empty battery reaches the danger band while still checking: each such check costs, and the plan says so
    tired = {**run.model, "start_battery": 45.0}
    deep, shallow = sr.simulate(tired, long, 5), sr.simulate(tired, long, 25)
    assert any(s.get("danger") for s in deep["steps"]) and not any(s.get("danger") for s in shallow["steps"])
    assert deep["score"] < shallow["score"] and deep["end_battery_pct"] < shallow["end_battery_pct"]
    assert sr.simulate({**run.model, "start_battery": 30.0}, long, 5)["stopped"] == "stranded"


def test_hard_mode_blocks_the_weakest_waypoint(run):
    route = ["PANEL-212", "PANEL-211", "HX-202"]  # the two panels live at WP09
    easy, hard = run.simulate(route, 40), run.simulate(route, 40, hard=True)
    assert [s["outcome"] for s in hard["steps"]] == ["blocked", "blocked", "checked"] and hard["blocked"] == "WP09"
    assert hard["score"] < easy["score"] and any("blocked" in r["text"] for r in hard["reasons"])
    assert "p. 43" in run.board()["hard"]["words"]


def test_every_sentence_is_plain(run):
    board = run.board()
    assert len(board["candidates"]) == 20 and board["candidates"][0]["asset_id"] == "HX-202" and "training aid, not a prediction" in board["footnote"]
    assert board["footnote"].startswith("Simulated from 27 recorded rounds") and board["last_route"]
    for c in board["candidates"]:
        readable(f"{c['label']} is {c['why']}.")
    for route, level, hard in ((board["last_route"], 40, False), ([c["asset_id"] for c in board["candidates"]][:12], 5, False), (["PANEL-212"], 40, True), ([], 40, False)):
        r = run.simulate(route, level, hard=hard)
        for text in [r["summary"], r["verdict"], *[x["text"] for x in r["reasons"]], *r["differences"], r["recommended"]["summary"]]:
            readable(text)
        for s in r["steps"]:
            readable(s["sentence"] + ".")
            assert s["played_words"] in ("worked", "failed", "skipped", "battery ran out", "not reached")


def test_no_model_is_called_and_nothing_is_written():
    for module in (sr, app_shiftrun):
        source = inspect.getsource(module)
        assert "import runtime" not in source and "from runtime" not in source and ".chat(" not in source  # no model
        assert "open(" not in source and "write_text" not in source and "datetime" not in source  # no file, no clock
    assert ShiftRun(Spot(None)).board()["configured"] is False and ShiftRun(Spot(None)).simulate(["HX-202"], 40)["configured"] is False


def test_the_api_and_sending_a_plan_as_an_approval_card(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path, connectors=False, memory_fixture=""))) as c:
        board = c.get("/api/spot/shiftrun/board").json()
        assert board["configured"] is True and board["max_stops"] == sr.MAX_STOPS and board["turn_back"]["default"] == 40
        best = c.get("/api/spot/shiftrun/recommended").json()
        assert best["route"][0] == "HX-202" and c.get("/api/spot/shiftrun/recommended?hard=true").json()["blocked"] == "WP09"
        body = {"route": ["hx-202", "HX-204", "NOPE"], "turn_back_at": 45}
        r = c.post("/api/spot/shiftrun/simulate", json=body).json()
        assert r["route"] == ["HX-202", "HX-204"] and r["stars"] in (0, 1, 2, 3) and r == c.post("/api/spot/shiftrun/simulate", json=body).json()
        assert c.post("/api/spot/shiftrun/simulate", json={"route": [], "turn_back_at": 200}).status_code == 422
        assert c.get("/api/approvals").json() == [] and c.get("/api/spot/work_orders").json()["work_orders"] == []  # simulating wrote nothing

        card = c.post("/api/spot/actions/plan", json={"route": ["HX-202", "HX-204"], "turn_back_at": 45}).json()
        pending = c.get("/api/approvals").json()
        assert [p["id"] for p in pending] == [card["action_id"]] and pending[0]["target"] == "spot.mission_request"
        assert "SPOT is not commanded" in card["reason"] and c.get("/api/spot/work_orders").json()["work_orders"] == []
        assert c.post(f"/api/approvals/{card['action_id']}", json={"decision": "approve"}).status_code == 200
        orders = []
        for _ in range(50):
            orders = c.get("/api/spot/work_orders").json()["work_orders"]
            if orders:
                break
        record = json.loads(Path(orders[0]["path"]).read_text())
        assert orders[0]["kind"] == "mission_request" and record["route"] == "HX-202, HX-204" and record["turn_back_at"] == 45
        assert c.post("/api/spot/actions/plan", json={"route": ["NOPE"], "turn_back_at": 40}).status_code == 404
        assert c.post("/api/spot/actions/plan", json={"route": [], "turn_back_at": 40}).status_code == 404


def test_the_two_older_cards_still_write_their_own_kinds(spot, tmp_path):
    async def go():
        bus = EventBus(tmp_path / "events")
        approvals = Approvals({"connector_write": lambda req: "the hub ran it"})
        bus.subscribe("agent.action.proposed", approvals.on_proposed, replay=True)
        bus.subscribe("experience.approval.decided", approvals.on_decided)
        ctx = build_spot(tmp_path, bus, FakeRuntime(), approvals, kit_folder="fixtures/spot/kit")
        kinds = []
        for card in (ctx.orders.propose_reinspect("HX-202"), ctx.orders.propose_work_order("AN014"), ctx.orders.propose_plan(["HX-202"], 40)):
            await bus.publish(card)
            await bus.publish(ApprovalDecided(action_id=card.action_id, decision="approve"))
            await asyncio.sleep(0.01)
        for f in sorted((tmp_path / "spot" / "work_orders").glob("WO-*.json")):
            kinds.append(json.loads(f.read_text())["kind"])
        assert sorted(kinds) == ["mission_request", "reinspect", "work_order"]

    asyncio.run(go())
