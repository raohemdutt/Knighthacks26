"""Ask (intents, citations, the grounding check), the approval-gated work orders, and the API with and without a kit."""

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bus import EventBus
from contexts.gateway.app import create_app
from contexts.gateway.bootstrap import Settings
from contexts.spot.application.assistant import REPHRASE_MAX_DRAFT_CHARS, REPHRASE_TIMEOUT_S, Assistant, grounded
from contexts.spot.application.service import Spot
from contexts.spot.application.workorders import WorkOrders
from contexts.spot.infrastructure.kit import load_kit
from contexts.spot.wiring import build_spot
from contexts.workspace.domain.approvals import Approvals
from contracts.experience import ApprovalDecided
from contracts.types import ChatResult
from runtime import FakeRuntime


@pytest.fixture(scope="module")
def spot():
    return Spot(load_kit(Path("fixtures/spot/kit")))


# --- ask ---


@pytest.mark.parametrize("question, intent, must_cite, must_say", [
    ("What should I check first this shift?", "investigate_next", ("assets", "HX-202"), "Re-inspect HX-202"),
    ("what needs attention now", "triage", ("assets", "HX-202"), "score 72.3"),
    ("Show hidden failures", "hidden_failures", ("missions", "M001"), "21 of 26"),
    ("Is V2 better than V1?", "version_comparison", ("actions", "A0001"), "91.1% (82/90)"),
    ("Which inspection types are least reliable?", "reliability_by_type", ("actions", "A0016"), "acoustic inspection, 13 of 43"),
    ("What happened at WP05?", "waypoint", ("locations", "WP05"), "11 of 43"),
    ("what happened at wp 9", "waypoint", ("locations", "WP09"), "8 of 22"),
    ("Does battery affect reliability?", "battery", ("actions", "A0009"), "25 of 71"),
    ("Status of HX-202", "asset", ("anomalies", "AN005"), "88 h"),
    ("how is heat exchanger 04 doing", "asset", ("assets", "HX-204"), "AN014"),
    ("what about M017", "mission", ("missions", "M017"), "ended FAILED"),
    ("Summarize the last mission", "mission", ("missions", "M027"), "1 of 10 actions failed"),
    ("give me the shift handoff", "handoff", ("assets", "HX-202"), "SPOT shift handoff"),
    ("Are anomalies hidden in successful missions?", "anomalies", ("anomalies", "AN011"), "38 of 40"),
    ("where are failures concentrated", "where", ("locations", "WP09"), "WP09 Turbine Hall South"),
    ("which assets are overdue for inspection", "coverage", ("assets", "GAUGE-209"), "48 h"),
    ("what can I get from the replay stream", "replay", ("missions", "M024"), "500 events"),
    ("what assumptions did you make", "assumptions", None, "newest timestamp"),
])
def test_intents_answer_from_the_analysis_with_citations(spot, question, intent, must_cite, must_say):
    d = Assistant(spot, FakeRuntime()).draft(question)
    assert d["intent"] == intent
    assert must_say in d["text"]
    if must_cite:
        assert must_cite[1] in d["citations"][must_cite[0]]
    kit = spot.kit
    for a in d["citations"]["actions"]:
        assert a in kit.action
    for m in d["citations"]["missions"]:
        assert m in kit.mission


def test_an_unknown_question_says_what_it_can_answer(spot):
    d = Assistant(spot, FakeRuntime()).draft("tell me a joke")
    assert d["intent"] == "overview" and "81.9%" in d["text"] and "Try:" in d["text"]


def test_no_kit_no_answer():
    out = asyncio.run(Assistant(Spot(None), FakeRuntime()).ask("what needs attention?"))
    assert out["intent"] == "not_configured" and not out["rephrased"]


def test_grounding_check_accepts_rewording_and_rejects_new_numbers_or_ids():
    draft = "HX-202 failed 2 of 16 actions (12.5%). AN005 is HIGH. Last good inspection A0156 in M016, 88 h ago."
    assert grounded("Two of sixteen? No: 2 of 16 actions on HX-202 failed, 12.5%. AN005 (HIGH); last good A0156, M016, 88 h back.", draft) == (True, "ok")
    assert not grounded("HX-202 failed 3 of 16 actions.", draft)[0]
    assert "HX-204" in grounded("HX-204 failed 2 of 16 actions.", draft)[1]
    assert "AN006" in grounded("AN006 is HIGH on HX-202.", draft)[1]
    assert not grounded("About 13% of actions on HX-202 failed.", draft)[0]
    assert not grounded("", draft)[0] and not grounded("x" * 5000, draft)[0]
    assert grounded("WP05 failed 7 of 25.", "Navigation to WP05 failed 7 of 25 times.")[0]  # the 05 in WP05 is not a number


SHORT = "Summarize the last mission"  # a draft under the rephrase budget
LONG = "Is V2 better than V1?"  # the version comparison: long and number-dense


def reworded(draft: str) -> str:
    return "In short: " + draft


def test_fake_runtime_never_rephrases(spot):
    rt = FakeRuntime()
    out = asyncio.run(Assistant(spot, rt).ask(SHORT, rephrase=True))
    assert not out["rephrased"] and out["text"] == out["draft"] and rt.calls == [] and out["can_rephrase"] is False


def test_the_default_ask_returns_the_draft_without_calling_the_model(spot):
    rt = FakeRuntime(chat_replies=[ChatResult(content="never read", model_version="atlas-v3")])
    a = Assistant(spot, rt, rephrase=True)
    out = asyncio.run(a.ask(SHORT))
    assert rt.calls == []
    assert out["text"] == out["draft"] == a.draft(SHORT)["text"] and out["rephrased"] is False
    assert out["model"] == "none (built from the tables by code)" and out["check"] is None
    assert out["can_rephrase"] is True  # the screen may come back for the wording


def test_long_or_number_dense_drafts_are_never_sent_to_the_model(spot):
    rt = FakeRuntime(chat_replies=[ChatResult(content="never read", model_version="atlas-v3")])
    a = Assistant(spot, rt, rephrase=True)
    for q in (LONG, "What should I check first this shift?", "What needs attention now?", "give me the shift handoff"):
        out = asyncio.run(a.ask(q, rephrase=True))
        assert out["can_rephrase"] is False and not out["rephrased"] and out["text"] == out["draft"], q
    assert rt.calls == []
    assert REPHRASE_MAX_DRAFT_CHARS == 450 and REPHRASE_TIMEOUT_S == 8
    assert len(a.draft(SHORT)["text"]) <= REPHRASE_MAX_DRAFT_CHARS < len(a.draft(LONG)["text"])
    # the version comparison stays out even if it were ever short
    assert a.can_rephrase({"intent": "version_comparison", "text": "V2 91.1% (82/90)."}) is False


def test_a_timeout_leaves_the_draft(spot, monkeypatch):
    import time

    from contexts.spot.application import assistant

    class Slow(FakeRuntime):
        def chat(self, *a, **k):
            time.sleep(0.3)
            return ChatResult(content="too late", model_version="atlas-v3")

    monkeypatch.setattr(assistant, "REPHRASE_TIMEOUT_S", 0.05)
    draft = Assistant(spot, FakeRuntime()).draft(SHORT)["text"]
    out = asyncio.run(Assistant(spot, Slow(), rephrase=True).ask(SHORT, rephrase=True))
    assert not out["rephrased"] and out["text"] == draft and "TimeoutError" in out["check"]


def test_a_real_model_may_only_rephrase_and_is_dropped_on_any_mismatch(spot):
    draft = Assistant(spot, FakeRuntime()).draft(SHORT)["text"]
    good = reworded(draft)
    rt = FakeRuntime(chat_replies=[ChatResult(content=good, model_version="atlas-v3")])
    out = asyncio.run(Assistant(spot, rt, rephrase=True).ask(SHORT, rephrase=True))
    assert out["rephrased"] and out["text"] == good and out["draft"] == draft and out["model"] == "atlas-v3" and out["check"] == "ok"
    assert rt.calls[0][0] == "chat" and rt.calls[0][1][-1].content == draft  # the model is given the draft, nothing else

    bad = FakeRuntime(chat_replies=[ChatResult(content=good + " Check M999 too.", model_version="atlas-v3")])
    out = asyncio.run(Assistant(spot, bad, rephrase=True).ask(SHORT, rephrase=True))
    assert not out["rephrased"] and out["text"] == draft and "M999" in out["check"]

    class Down(FakeRuntime):
        def chat(self, *a, **k):
            raise ConnectionError("llama-server is not running")

    out = asyncio.run(Assistant(spot, Down(), rephrase=True).ask(SHORT, rephrase=True))
    assert not out["rephrased"] and out["text"] == draft and "ConnectionError" in out["check"]


def test_the_rewrite_is_asked_for_at_the_non_urgent_priority(spot):
    seen = []

    class Watching(FakeRuntime):
        def chat(self, messages, tools=None, priority="chat", **k):
            seen.append(priority)
            return ChatResult(content=reworded(messages[-1].content), model_version="atlas-v3")

    out = asyncio.run(Assistant(spot, Watching(), rephrase=True).ask(SHORT, rephrase=True))
    assert out["rephrased"] and seen == ["summary"]  # it never jumps ahead of someone's question


def test_handoff_is_five_to_eight_lines_each_with_evidence(spot):
    h = spot.handoff()
    assert 5 <= len(h["lines"]) <= 8
    assert all(ln["text"] and ln["target"] for ln in h["lines"])
    assert sum(1 for ln in h["lines"] if sum(len(v) for v in ln["evidence"].values()) > 0) >= 6
    assert h["text"].count("\n") == len(h["lines"])


# --- work orders behind approvals ---


def test_a_card_is_never_approved_by_itself_and_the_file_is_written_only_after_approval(spot, tmp_path):
    async def run():
        bus = EventBus(tmp_path / "events")
        approvals = Approvals({"connector_write": lambda req: "the hub ran it"})
        bus.subscribe("agent.action.proposed", approvals.on_proposed, replay=True)
        bus.subscribe("experience.approval.decided", approvals.on_decided)
        ctx = build_spot(tmp_path, bus, FakeRuntime(), approvals, kit_folder="fixtures/spot/kit")
        created = []
        bus.subscribe("spot.work_order.created", created.append)
        folder = tmp_path / "spot" / "work_orders"

        card = ctx.orders.propose_reinspect("HX-202")
        assert card.kind == "connector_write" and card.target == "spot.reinspect" and "HX-202" in card.reason and "nothing is sent" in card.reason
        await bus.publish(card)
        await asyncio.sleep(0.01)
        assert [r.status for r in approvals.pending()] == ["proposed"] and not folder.exists() and created == []

        denied = ctx.orders.propose_work_order("AN014")
        await bus.publish(denied)
        await bus.publish(ApprovalDecided(action_id=denied.action_id, decision="deny"))
        await asyncio.sleep(0.01)
        assert approvals.requests[denied.action_id].status == "denied" and not folder.exists()

        await bus.publish(ApprovalDecided(action_id=card.action_id, decision="approve"))
        await asyncio.sleep(0.01)
        req = approvals.requests[card.action_id]
        assert req.status == "done" and "Nothing left this box" in req.result
        files = list(folder.glob("WO-*.json"))
        assert len(files) == 1
        record = json.loads(files[0].read_text())
        assert record["kind"] == "reinspect" and record["asset_id"] == "HX-202" and record["approved_action_id"] == card.action_id
        assert len(created) == 1 and created[0].asset_id == "HX-202" and created[0].path == str(files[0])
        assert ctx.orders.view()[0]["work_order_id"] == record["work_order_id"]

        # the connector hub still gets everything that is not ours
        other = card.model_copy(update={"action_id": "other", "target": "calendar.create_event", "event_id": "e-other"})
        await bus.publish(other)
        await bus.publish(ApprovalDecided(action_id="other", decision="approve"))
        await asyncio.sleep(0.01)
        assert approvals.requests["other"].result == "the hub ran it" and len(list(folder.glob("*.json"))) == 1

    asyncio.run(run())


def test_unknown_asset_or_anomaly_is_refused(spot, tmp_path):
    orders = WorkOrders(spot, tmp_path)
    with pytest.raises(ValueError):
        orders.propose_reinspect("NOPE-1")
    with pytest.raises(ValueError):
        orders.propose_work_order("AN999")
    card = orders.propose_work_order("AN014")
    assert "from a successful inspection" in card.reason and card.params["anomaly_id"] == "AN014"


# --- the API ---


def client(tmp_path) -> TestClient:
    return TestClient(create_app(Settings(data_dir=tmp_path, connectors=False, memory_fixture="")))


GETS = ["overview", "findings", "triage", "missions", "locations", "assets", "handoff", "assumptions", "validation", "replay/state", "work_orders"]


def test_api_with_the_kit(tmp_path):
    with client(tmp_path) as c:
        for path in GETS:
            r = c.get(f"/api/spot/{path}")
            assert r.status_code == 200 and r.json().get("configured") is True, path
        o = c.get("/api/spot/overview").json()
        assert o["counts"]["actions"] == 270 and len(o["next"]) == 3 and o["kpis"][1]["n"] == 270 and o["model"]["real"] is False
        assert o["validation"]["agree"] == 12 and o["index"]["action_mission"]["A0265"] == "M027"
        assert len(c.get("/api/spot/findings").json()["findings"]) >= 15
        t = c.get("/api/spot/triage").json()
        assert t["items"][0]["id"] == "HX-202" and t["uninspected"][0]["asset_id"] == "HX-202"
        m = c.get("/api/spot/missions/m017").json()
        assert m["mission_status"] == "FAILED" and len(m["timeline"]) == 10 and m["timeline"][8]["anomalies"][0]["anomaly_id"] == "AN030"
        a = c.get("/api/spot/assets/HX-202").json()
        assert a["inspection"]["last_good_action"] == "A0156" and a["triage"]["rank"] == 1 and a["series"] and a["samples"]
        assert c.get("/api/spot/missions/M999").status_code == 404 and c.get("/api/spot/assets/NOPE").status_code == 404
        loc = c.get("/api/spot/locations").json()
        wp09 = next(x for x in loc["locations"] if x["location_id"] == "WP09")
        assert wp09["health"] == "bad" and (wp09["k"], wp09["n"]) == (8, 22) and wp09["zone_text"].startswith("Pumps")
        assert loc["plan"]["image"] == "fictional_floor_plan.png" and len(loc["plan"]["waypoints"]) == 15 and loc["plan"]["fit"]["follows_csv"] is False
        # images: only what the loader listed
        assert c.get("/api/spot/kit/assets/sample_inspection_images/gauge_pressure_01.jpg").status_code == 200
        assert c.get("/api/spot/plan/image").headers["content-type"] == "image/jpeg"
        assert c.get("/api/spot/kit/assets/README.md").status_code == 404
        assert c.get("/api/spot/kit/assets/..%2F02_DATA%2Factions.csv").status_code == 404

        # replay control
        s = c.post("/api/spot/replay/seek", json={"mission_id": "M017"}).json()
        assert s["index"] == 16 * 22 and s["playing"] is False
        s = c.post("/api/spot/replay/seek", json={"index": 22}).json()
        assert s["mission"]["status"] == "COMPLETED" and s["mission"]["failed"] == 2 and len(s["marks"]) == 27
        assert c.post("/api/spot/replay/speed", json={"speed": 1200}).json()["speed"] == 1200
        assert c.post("/api/spot/replay/start").json()["playing"] is True
        assert c.post("/api/spot/replay/pause").json()["playing"] is False
        assert c.post("/api/spot/replay/seek", json={}).status_code == 422

        ans = c.post("/api/spot/ask", json={"text": "what should I check first this shift?"}).json()
        assert ans["intent"] == "investigate_next" and "HX-202" in ans["citations"]["assets"] and ans["rephrased"] is False
        assert ans["text"] == ans["draft"] and ans["model"] == "none (built from the tables by code)"
        again = c.post("/api/spot/ask", json={"text": "summarize the last mission", "rephrase": True}).json()
        assert again["rephrased"] is False and again["can_rephrase"] is False  # no model on this box: the draft stands

        # an approval card, approved over the same door the phone uses
        card = c.post("/api/spot/actions/reinspect", json={"asset_id": "hx-202"}).json()
        pending = c.get("/api/approvals").json()
        assert [p["id"] for p in pending] == [card["action_id"]] and pending[0]["target"] == "spot.reinspect"
        assert c.get("/api/spot/work_orders").json()["work_orders"] == []
        assert c.post(f"/api/approvals/{card['action_id']}", json={"decision": "approve"}).status_code == 200
        for _ in range(50):
            orders = c.get("/api/spot/work_orders").json()["work_orders"]
            if orders:
                break
        assert orders[0]["asset_id"] == "HX-202" and Path(orders[0]["path"]).is_file()
        assert c.post("/api/spot/actions/work_order", json={"anomaly_id": "AN999"}).status_code == 404


def test_api_without_a_kit_answers_200_and_says_how_to_add_one(tmp_path, monkeypatch):
    monkeypatch.setenv("ATLAS_SPOT_KIT", str(tmp_path / "no-kit-here"))
    with client(tmp_path) as c:
        for path in GETS:
            r = c.get(f"/api/spot/{path}")
            assert r.status_code == 200, path
            assert r.json()["configured"] is False, path
        assert "ATLAS_SPOT_KIT" in c.get("/api/spot/overview").json()["note"]
        assert c.get("/api/spot/missions/M001").json()["configured"] is False
        assert c.post("/api/spot/replay/start").json() == {**c.get("/api/spot/replay/state").json()}
        assert c.post("/api/spot/ask", json={"text": "what needs attention?"}).json()["intent"] == "not_configured"
        assert c.post("/api/spot/actions/reinspect", json={"asset_id": "HX-202"}).status_code == 404
        assert c.get("/api/spot/plan/image").status_code == 404
        assert c.get("/health").status_code == 200
