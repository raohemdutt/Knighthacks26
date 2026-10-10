"""The one-box Circle demo: seeded through the real app, read back from the log alone."""

import asyncio
import shutil
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
import yaml
from fastapi.testclient import TestClient

from contexts.demo import circle
from contexts.demo.__main__ import main
from contexts.demo.seed import build_app, seed, settings
from contexts.demo.story import ROOT
from contexts.gateway.app import create_app
from contexts.gateway.bootstrap import build

TZ = ZoneInfo("America/New_York")
NOW = datetime(2026, 10, 10, 15, 20, tzinfo=TZ)  # a Saturday afternoon: the next mealtime seven hours on is breakfast


@pytest.fixture(scope="module")
def seeded(tmp_path_factory):
    root = tmp_path_factory.mktemp("circle-demo")
    cfg = circle.config_dir(root / "config")
    out = asyncio.run(seed(root / "data", now=NOW, config_dir=cfg, extra=circle.circle_events))
    out.app.projection.store.close()
    return root, cfg, out


@pytest.fixture(scope="module")
def phone(seeded, tmp_path_factory):
    """The gateway on a copy of the seeded log: every projection rebuilt by replay, as after a restart."""
    root, cfg, _ = seeded
    data = tmp_path_factory.mktemp("circle-replayed")
    shutil.copytree(root / "data" / "events", data / "events")
    s = settings(data, cfg)
    s.replay = True
    atlas = build(s)
    from contexts.circle import api

    real = api.circle_router

    with pytest.MonkeyPatch.context() as mp:  # the screen at the story's `now`, not the wall clock
        mp.setattr("contexts.gateway.app.circle_router", lambda bus, **kw: real(bus, clock=lambda: NOW, **kw))
        app = create_app(atlas=atlas)
    atlas.dayrun.clock = lambda: NOW
    with TestClient(app) as c:
        yield c
    atlas.projection.store.close()


def test_the_demo_config_joins_the_circle_and_the_committed_one_does_not(seeded):
    _, cfg, _ = seeded
    committed = yaml.safe_load((ROOT / "config" / "agent.yaml").read_text())["circle"]
    assert not committed.get("member_id") and not committed.get("circle_id")
    demo = yaml.safe_load((cfg / "agent.yaml").read_text())["circle"]
    assert (demo["member_id"], demo["circle_id"], demo["names"]) == ("hemdutt", "demo-circle", circle.NAMES)
    assert (cfg / "league.yaml").read_text() == (ROOT / "config" / "league.yaml").read_text()  # nothing else changed


def test_the_circle_story_has_fixed_ids_and_moves_with_now():
    events = circle.circle_events(NOW)
    assert [e.model_dump_json() for e in circle.circle_events(NOW)] == [e.model_dump_json() for e in events]
    assert all(e.event_id.startswith("demo-circle-") for e in events) and len({e.event_id for e in events}) == len(events)
    assert [e.ts for e in events] == sorted(e.ts for e in events) and events[-1].ts <= NOW
    moved = circle.circle_events(NOW + timedelta(days=2))
    assert [e.ts - timedelta(days=2) for e in moved] == [e.ts for e in events]


def test_the_conversations_come_out_of_the_log_alone(phone):
    body = phone.get("/api/circle/threads").json()
    assert (body["configured"], body["circle_id"], body["member_id"]) == (True, "demo-circle", "hemdutt")
    assert [m["display_name"] for m in body["members"]] == ["Hemdutt", "Maya", "Dominick"]
    threads = {t["title"]: t for t in body["threads"]}
    assert [(t["title"], t["state"]) for t in body["threads"]] == [
        ("Gym", "waiting"), ("COP4533 exam review", "agreed"), ("Breakfast", "agreed")]
    assert [w["display_name"] for w in threads["Gym"]["waiting_on"]] == ["Maya"]

    def lines(title: str) -> list[str]:
        return [m["line"] for m in phone.get(f"/api/circle/threads/{threads[title]['thread_id']}").json()["messages"]]

    assert lines("COP4533 exam review") == [
        "Hemdutt's Atlas: Can everyone do COP4533 exam review, tomorrow 2:00 to 4:00 PM at the Library?",
        "Dominick's Atlas: Free.",
        "Maya's Atlas: Busy then. Suggests 3:00 PM.",
        "Hemdutt's Atlas: Can everyone do tomorrow 3:00 to 5:00 PM instead?",
        "Dominick's Atlas: Free.",
        "Maya's Atlas: Free.",
        "Agreed: tomorrow 3:00 PM, the Library, 3 of 3 in.",
    ]
    assert lines("Breakfast") == [
        "Maya's Atlas: Can everyone do Breakfast, tomorrow 8:30 to 9:30 AM at the Student Union?",
        "Dominick's Atlas: Free.",
        "Hemdutt's Atlas: Free.",
        "Agreed: tomorrow 8:30 AM, the Student Union, 3 of 3 in.",
    ]
    assert lines("Gym") == [
        "Hemdutt's Atlas: Can everyone do Gym, Mon Oct 12 10:00 to 11:00 AM at the Gym?",
        "Dominick's Atlas: Free.",
    ]


def test_people_places_calendar_and_league_are_all_there(phone):
    members = {m["member_id"]: m for m in phone.get("/api/circle/members").json()["members"]}
    assert set(members) == {"hemdutt", "maya", "dominick"}
    assert (members["maya"]["place"]["place"], members["dominick"]["place"]["place"]) == ("Library", "Student Union")
    assert all(m["has_box"] for m in members.values())
    events = phone.get("/api/circle/events").json()
    assert [e["title"] for e in events["events"]] == ["Breakfast", "COP4533 exam review"] and events["source"] == "local"
    league = phone.get("/api/league").json()
    assert league["local"] is False and [s["display_name"] for s in league["standings"]] == ["Hemdutt", "Maya", "Dominick"]


def test_the_agreed_plans_are_fixed_blocks_of_the_day(phone):
    blocks = {b["id"]: b for b in phone.get("/api/dayrun/board").json()["blocks"]}
    meal, study = blocks["circle:demo-circle-meal"], blocks["circle:demo-circle-study-2"]
    assert (meal["title"], meal["place"], meal["activity"]) == ("Breakfast", "ucf.union", "eat")
    assert (study["title"], study["place_name"], study["activity"]) == ("COP4533 exam review", "John C. Hitt Library", "study")
    assert study["start"].startswith("2026-10-11T15:00")


def test_the_app_wrote_its_own_part_and_seeding_twice_adds_nothing(seeded):
    root, cfg, out = seeded
    assert out.written["workspace.action.completed"] >= 3  # the three approved proposals were sent by the real executor
    assert sum("Everyone's in" in a.message for a in out.alerts) == 2  # the delivery told the phone, twice
    before = sum(1 for f in (root / "data" / "events").glob("*.jsonl") for _ in f.open())
    again = asyncio.run(seed(root / "data", now=NOW, force=True, config_dir=cfg, extra=circle.circle_events))
    again.app.projection.store.close()
    assert sum(again.written.values()) == 0
    assert sum(1 for f in (root / "data" / "events").glob("*.jsonl") for _ in f.open()) == before


def test_the_default_seed_has_no_circle(tmp_path):
    app = build_app(tmp_path)
    assert app.circle is None and app.league.local
    app.projection.store.close()


def test_the_command_prints_how_to_run_the_app_on_it(tmp_path, capsys):
    assert main(["circle", "--data-dir", str(tmp_path / "d"), "--end", NOW.isoformat()]) == 0
    out = capsys.readouterr().out
    assert f"ATLAS_DATA_DIR={tmp_path / 'd'} ATLAS_CONFIG_DIR={tmp_path / 'd' / 'config'} ATLAS_MEMORY_FIXTURE= ATLAS_CONNECTORS=0" in out
    assert "python -m contexts.gateway" in out and "Circle demo-circle: Hemdutt, Maya, Dominick" in out
    assert main(["circle", "--data-dir", str(tmp_path / "d"), "--end", NOW.isoformat()]) == 1  # never over an existing log
