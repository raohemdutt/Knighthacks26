"""The look guide: a question about where something is, resolved by code to a place on the Inspection screen."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from contexts.gateway.app import create_app
from contexts.gateway.bootstrap import Settings
from contexts.spot.application.guide import CONFIDENT, Guide, similarity, words
from contexts.spot.application.service import Spot
from contexts.spot.infrastructure.kit import load_kit


@pytest.fixture(scope="module")
def guide():
    return Guide(Spot(load_kit(Path("fixtures/spot/kit"))))


@pytest.mark.parametrize("question, target", [
    # equipment, by tag and by name
    ("Where do I look for HX-202?", "asset:HX-202"),
    ("where can I find heat exchanger 4", "asset:HX-204"),
    ("show me hx 204", "asset:HX-204"),
    ("show me HX204", "asset:HX-204"),
    ("where is the heat exchnager 2", "asset:HX-202"),
    ("point me to pump 5", "asset:PUMP-205"),
    ("where is motor 6", "asset:MOTOR-206"),
    ("show me exchanger 13", "asset:HX-213"),
    ("where is electrical panel 11", "asset:PANEL-211"),
    ("show me valve seven", "asset:VALVE-207"),
    ("find the pressure gauge", "asset:GAUGE-209"),
    ("where's Heat Exchanger 02?", "asset:HX-202"),
    # places
    ("show me the pump room", "place:WP04"),
    ("where is the battery room", "place:WP11"),
    ("where's the pipe gallery", "place:WP05"),
    ("where is wp05", "place:WP05"),
    ("show me waypoint 9", "place:WP09"),
    ("where is turbine hall south", "place:WP09"),
    ("Hey Atlas, where can I find the turbine hall north please", "place:WP08"),
    ("where is the compressor area", "place:WP06"),
    ("point me to stop 12", "place:WP12"),
    # the cards of the first screen
    ("show me what to do first", "card:do-first"),
    ("where do I start?", "card:do-first"),
    ("where's the first thing to do", "do-first:1"),
    ("show me the second job", "do-first:2"),
    ("show me the watch out card", "card:watch"),
    ("where do I see what looks fine", "card:fine"),
    ("show me all equipment", "all-equipment"),
    ("where is trust", "trust"),
    ("where can i find failed checks", "trust"),
    ("where is the map", "map"),
    ("find the floor plan", "map"),
    ("show me floor 2", "map:floor:2"),
    ("where do I ask a question", "ask"),
    ("where do I write the handoff", "handoff"),
    ("point me to the shift handoff", "handoff"),
    # rounds
    ("which round", "round"),
    ("show me round 27", "round:27"),
    ("where is M017", "round:17"),
    ("where is the play button", "round:play"),
    ("where is spot's battery level", "round:battery"),
    ("where can I see where spot went", "round:map"),
    # tabs, Shift Run, the analysis and its explainers
    ("where can I plan the next round", "tab:shift-run"),
    ("point me to the turn back slider", "shiftrun:turn-back"),
    ("show me the analysis", "tab:analysis"),
    ("where do I look for the battery", "explainer:battery"),
    ("where do i look for the batery", "explainer:battery"),
    ("is V2 better", "explainer:version"),
    ("where are the alarms", "explainer:alarms"),
    ("where is the weakest place", "explainer:weak"),
    ("where can I find the assumptions", "analysis:assumptions"),
    ("show me the data check", "analysis:data-check"),
    ("show me the tables", "analysis:tables"),
    ("where is the uninspected risk", "analysis:tables"),
    # a guide id, as a link carries it
    ("asset:HX-202", "asset:HX-202"),
    ("explainer:battery", "explainer:battery"),
])
def test_a_question_resolves_to_its_place_on_the_screen(guide, question, target):
    out = guide.resolve(question)
    assert out["target"] == target, (out["target"], out["alternatives"])
    assert out["confidence"] >= CONFIDENT
    assert out["label"] and out["steps"]
    assert all(s["say"].endswith((".", "?")) and len(s["say"]) < 140 for s in out["steps"])
    assert "reveal" not in out["steps"][-1]  # the last sentence is the one said on arrival


def test_the_walk_through_names_the_floor_the_mark_and_the_equipment(guide):
    out = guide.resolve("where is heat exchanger 4")
    assert [s.get("reveal") for s in out["steps"]] == ["tab:shift", "map:floor:1", None, None]
    assert out["steps"][2]["say"] == "Look at the plan of the plant floor, floor 1."
    assert out["steps"][3]["say"] == "One of the 4 marks at Pipe Gallery, near the middle, is Heat exchanger 4. Needs you."
    alone = guide.resolve("where is HX-210")["steps"][-1]["say"]
    assert alone.startswith("The ") and "at Battery Room" in alone and "is Heat exchanger 10." in alone
    battery = guide.resolve("where do I look for the battery")
    assert battery["steps"][0] == {"reveal": "tab:analysis", "say": "I am opening the Analysis tab."}
    assert guide.resolve("show me round 17")["steps"][1]["reveal"] == "round:pick:M017"


@pytest.mark.parametrize("question", ["where are the findings", "show me every finding", "finding:battery", "analysis:findings", "where is listening checks fail most",
                                      "where is the trained scorer"])
def test_what_the_analysis_tab_no_longer_lists_is_never_pointed_at(guide, question):
    out = guide.resolve(question)
    ids = [out["target"] or ""] + [a["target"] for a in out["alternatives"]]
    assert not any(i.startswith("finding:") or i == "analysis:findings" for i in ids), ids
    assert out["target"] or out["alternatives"]  # the nearest thing still on the screen, or "did you mean"
    assert not any(s.get("reveal") == "open:analysis:findings" for s in out["steps"])


@pytest.mark.parametrize("question", ["where is the coffee machine", "show me the weather", "where is heat exchanger 99", "where is the motor or the valve"])
def test_nothing_is_guessed_when_the_match_is_poor(guide, question):
    out = guide.resolve(question)
    assert out["target"] is None
    assert 2 <= len(out["alternatives"]) <= 3
    assert out["steps"][0]["say"].startswith("I am not sure what to point at. Did you mean ")
    assert all(a["target"] and a["label"] for a in out["alternatives"])


def test_every_target_id_is_unique_and_resolves_to_itself(guide):
    ids = [t.id for t in guide.targets()]
    assert len(ids) == len(set(ids)) > 80
    assert all(guide.resolve(i)["target"] == i for i in ids)


def test_words_and_similarity():
    assert words("Heat Exchanger 02") == ["heat", "exchanger", "2"]
    assert words("the Pump Bay") == ["pump", "room"]
    assert words("HX202") == ["hx", "202"] and words("v2") == ["version"]
    assert similarity(words("pump room"), words("Pump Bay")) == 1.0
    assert similarity(words("heat exchanger 4"), words("Heat exchanger 2")) < CONFIDENT  # a wrong number never matches
    assert similarity(words("room"), words("Battery Room")) == 0.0  # a light word alone names nothing
    assert Guide.thing("Where do I look for the battery chart?") == "the battery chart"
    assert Guide.thing("Atlas, show me HX-202 please") == "hx-202"


def test_without_a_kit_the_guide_says_so():
    out = Guide(Spot(None)).resolve("where is HX-202")
    assert out["configured"] is False and out["target"] is None and "No inspection data" in out["steps"][0]["say"]


def test_the_endpoint(tmp_path):
    app = create_app(Settings(data_dir=tmp_path, connectors=False, memory_fixture=""))
    with TestClient(app) as client:
        out = client.get("/api/spot/guide", params={"q": "where do I look for HX-202?"}).json()
        assert out["target"] == "asset:HX-202" and out["label"] == "Heat exchanger 2 (HX-202)" and out["confidence"] == 1.0
        assert set(out) >= {"configured", "query", "target", "label", "steps", "confidence", "alternatives"}
        assert client.get("/api/spot/guide", params={"q": "where is the coffee machine"}).json()["target"] is None
        assert client.get("/api/spot/guide").status_code == 422
