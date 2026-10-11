"""Every shipped profile boots the gateway on an empty data dir and shows its own words."""

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from contexts.gateway.app import create_app
from contexts.gateway.bootstrap import Settings
from contexts.profiles import available

CONFIG = Path(__file__).resolve().parents[3] / "config"


def client(tmp_path: Path, profile: str | None) -> TestClient:
    settings = Settings(data_dir=tmp_path, config_dir=CONFIG, connectors=False, ntfy_url=None, llm_url=None, profile=profile)
    return TestClient(create_app(settings))


def test_the_default_box_reports_the_student_profile_with_no_overrides(tmp_path, monkeypatch):
    monkeypatch.delenv("ATLAS_PROFILE", raising=False)
    with client(tmp_path, None) as c:
        got = c.get("/api/profile").json()
        assert got["name"] == "student" and got["overrides"] == {} and got["vocabulary"]["event"] == "class"
        assert got["words"]["Activity"] == "Study" and got["words"]["event_plural"] == "classes"
        assert got["suggestions"] == ["Where is my phone?", "Where is my water bottle?", "Where did I leave my laptop?", "What is due this week?", "What happened at our table overnight?",
                                      "What should I eat?", "Where should I go tonight?", "How's your day?", "Tell me a joke",
                                      "I'm stressed about my exam"]  # the Ask screen's chips: the demo desk's things first
        board = c.get("/api/dayrun/board", params={"scenario": "demo"}).json()
        titles = {o["title"] for s in board["slots"] for o in s["moves"]}
        assert any(t.startswith("Study at ") for t in titles)
        script = c.get("/api/dayrun/scenarios/demo").json()
        assert script["plans"]["good"][0] == {"slot": "s1", "kind": "eat", "place": "ucf.dining_63", "minutes": 30}
        assert len(script["campaign"]) == 2 and script["disruptions"][0]["kind"] == "block_moved"


@pytest.mark.parametrize("name", ["shift-worker", "family-home"])
def test_a_profile_boots_on_an_empty_data_dir_and_its_words_show(tmp_path, name):
    with client(tmp_path, name) as c:
        got = c.get("/api/profile").json()
        assert got["name"] == name and got["description"] and got["words"]["Activity"] != "Study"
        assert {"habits.yaml", "dayrun.yaml", "agent.yaml", "places/", "scenarios/"} <= set(got["overrides"])
        assert got["overrides"]["habits.yaml"] == f"profiles/{name}/habits.yaml"

        today = c.get("/api/habits/today").json()
        assert {h["id"] for h in today["habits"]} == set(got["meters"]["habits"])

        packs = {p["name"] for p in c.get("/api/places").json()}
        assert {"ucf", "apartment"} <= packs  # base packs stay; the profile's are added

        live = c.get("/api/dayrun/board")
        assert live.status_code == 200, live.text
        board = c.get("/api/dayrun/board", params={"scenario": got["scenario"]})
        assert board.status_code == 200, board.text
        board = board.json()
        verb = got["words"]["Activity"]
        titles = {o["title"] for s in board["slots"] for o in s["moves"]}
        assert any(t.startswith(f"{verb} at ") for t in titles) and not any(t.startswith("Study at ") for t in titles)
        assert board["deadlines"], "the scenario has something due"

        # the scenario's good plan wins and its bad plan does not
        sc = yaml.safe_load((CONFIG / "profiles" / name / "scenarios" / f"{got['scenario']}.yaml").read_text())
        script = c.get(f"/api/dayrun/scenarios/{got['scenario']}").json()  # the phone's one-tap presets
        assert script["plans"].keys() == sc["plans"].keys() and len(script["disruptions"]) == len(sc["disruptions"])
        assert c.get("/api/dayrun/scenarios/nope").status_code == 404
        for plan, expect in (("good", True), ("bad", False)):
            res = c.post("/api/dayrun/simulate", json={"moves": sc["plans"][plan], "scenario": got["scenario"]})
            assert res.status_code == 200, res.text
            assert res.json()["outcome"]["won"] is expect, (plan, res.json()["outcome"])


def test_an_unknown_profile_refuses_to_boot_with_a_clear_message(tmp_path):
    with pytest.raises(ValueError, match=r"no profile 'nurse'.*have: " + ", ".join(available(CONFIG))):
        client(tmp_path, "nurse")


def test_shift_worker_night_follows_the_shift(tmp_path):
    with client(tmp_path, "shift-worker") as c:
        got = c.get("/api/profile").json()
        assert got["shift"]["start"] == "22:00:00"
        atlas = c.app.state.atlas
        sched = atlas.habits.sched
        assert (sched.bedtime.strftime("%H:%M"), sched.wake.strftime("%H:%M")) == ("07:30", "14:30")
        assert (atlas.dayrun.cfg.sleep.bedtime.strftime("%H:%M"), atlas.dayrun.cfg.sleep.wake.strftime("%H:%M")) == ("07:30", "14:30")
        assert atlas.habits.cfg.by_id("leave_ready").watch_items == ["keys", "badge", "phone", "charger"]


@pytest.mark.parametrize(("name", "theirs"), [("shift-worker", "handover"), ("family-home", "chores"), ("inspection", "work orders")])
def test_the_suggested_questions_follow_the_profile(tmp_path, name, theirs):
    with client(tmp_path, name) as c:
        chips = c.get("/api/profile").json()["suggestions"]
        assert 4 <= len(chips) <= 8 and len(set(chips)) == len(chips)
        assert any(theirs in q for q in chips)  # the profile's own word for something due
        assert not any(word in q.lower() for q in chips for word in ("exam", "assignment", "class"))  # none of the student's
