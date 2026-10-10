"""The demo night, seeded through the real app and read back from the real projections."""

import asyncio
import shutil
from datetime import timedelta

import pytest

from contexts.demo.__main__ import main
from contexts.demo.seed import PACKAGE_RULE, SeedRefused, build_app, seed
from contexts.demo.story import load, story
from contexts.memory.fixture import load as load_memory_fixture

SOURCE_TYPES = {"perception.observation.recorded", "perception.scene.described", "perception.faces.counted",
                "connector.item.synced"}
NOW = load().now
SATURDAY = NOW.date() - timedelta(days=1)  # the story's last full day


@pytest.fixture(scope="module")
def seeded_dir(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("seeded")
    seeded = asyncio.run(seed(data_dir))
    seeded.app.projection.store.close()
    return data_dir, seeded


def replayed(log_dir, data_dir):
    """A freshly built app on a copy of the log, with every replaying projection rebuilt from it."""
    shutil.copytree(log_dir, data_dir / "events")
    app = build_app(data_dir)
    asyncio.run(app.bus.replay_into())
    return app


def picture(app) -> dict:
    """Everything the phone reads that rebuilds from the log."""
    return {
        "memory": app.projection.store.dump(),
        "habits": [app.habits.digest(day=day, now=NOW) for day in (SATURDAY, NOW.date())],
        "history": {h.id: app.habits.history_for(h.id) for h in app.habits.cfg.habits},
        "waste": app.waste.metrics(),
        "disposed": sorted((d.item_id, d.bin, d.correct) for d in app.waste.disposed.values()),
    }


def test_the_story_is_only_source_events_in_time_order_and_deterministic():
    events = story()
    assert {e.type for e in events} == SOURCE_TYPES
    assert [e.ts for e in events] == sorted(e.ts for e in events)
    assert events[-1].ts <= NOW
    assert len({e.event_id for e in events}) == len(events) < 3000
    assert [e.model_dump_json() for e in story()] == [e.model_dump_json() for e in events]


def test_the_story_moves_with_now():
    later = NOW + timedelta(days=3, hours=5)
    moved = story(later)
    assert [e.ts - timedelta(days=3, hours=5) for e in moved] == [e.ts for e in story()]
    due = next(e for e in moved if e.type == "connector.item.synced" and "Milestone1" in e.title)
    assert due.title == "Project - Milestone1 Fall2026 [COP4533]"
    assert due.when == load_memory_fixture().items[0].when + timedelta(days=3, hours=5)


def test_seed_fills_every_projection_of_a_fresh_app_replaying_the_log(seeded_dir, tmp_path):
    data_dir, seeded = seeded_dir
    assert set(seeded.written) > SOURCE_TYPES  # the app derived the rest
    assert sum(seeded.written.values()) < 5000

    app = replayed(data_dir / "events", tmp_path)

    # Memory: about one episode per beat, all closed, at the times the canned memory fixture has them.
    episodes = app.memory.episodes(NOW - timedelta(days=3), NOW)
    assert 40 <= len(episodes) <= 60 and all(e.end for e in episodes)
    for canned in load_memory_fixture().episodes:
        assert any(e.camera_id == canned.camera_id and abs(e.start - canned.start) < timedelta(minutes=1) for e in episodes), canned.id
    keys = app.memory.last_seen("keys")
    assert keys.zone_id == "desk" and keys.camera_id == "cam-desk"
    for label in ("laptop", "phone"):  # the detector tracked them all night; the last desk crop says where they are
        seen = app.memory.last_seen(label)
        assert seen and seen.zone_id == "desk" and seen.position, label
    can = app.memory.last_seen("can")  # the detector saw the last can go in after the caption: a zone, no text position
    assert (can.camera_id, can.zone_id, can.position) == ("cam-door", "trash", None)
    assert {i.connector for i in app.memory.items(NOW - timedelta(days=3), NOW + timedelta(days=7))} == {"canvas", "calendar", "groupme"}

    # Habits: Friday had a connector sync and no camera, so nothing a camera measures was judged that day.
    friday = {h.id: app.habits.history_for(h.id)[0] for h in app.habits.cfg.habits}
    assert all(r["day"] == (SATURDAY - timedelta(days=1)).isoformat() and r["closed"] for r in friday.values())
    assert not any(friday[i]["applies"] or friday[i]["streak"] for i in ("sleep", "meals", "energy_drinks", "focus", "leave_ready"))
    # Saturday is closed, with the meals, drinks and desk time the script wrote. The cameras came on at
    # noon, so the night before it was never watched: no sleep value is claimed for it.
    saturday = {r["id"]: r for r in app.habits.digest(day=SATURDAY, now=NOW)["habits"]}
    assert all(r["closed"] for r in saturday.values())
    assert saturday["meals"]["value"] >= 2
    assert saturday["energy_drinks"]["value"] >= 3 and not saturday["energy_drinks"]["met"]
    assert saturday["focus"]["value"] > 0
    assert not saturday["sleep"]["applies"] and saturday["sleep"]["value"] == 0 and saturday["sleep"]["streak"] == 0
    assert saturday["leave_ready"]["applies"] and saturday["leave_ready"]["streak"] == 1  # its first watched day
    sunday = {r["id"]: r for r in app.habits.digest(now=NOW)["habits"]}
    assert sunday["sleep"]["applies"] and sunday["sleep"]["value"] >= 6  # the desk was empty from half past midnight to a quarter to seven
    assert sunday["leave_ready"]["value"] == 1  # out the door at 9:05 with the keys on the desk
    assert sunday["focus"]["applies"] and sunday["focus"]["value"] > 0  # the milestone is due within 48 hours of the day

    # Waste Watch: right and wrong bins both happened.
    waste = app.waste.metrics()
    assert waste["items_seen"] >= 5 and waste["contamination"] >= 1
    assert 0 < waste["diversion_rate"] < 1
    assert any(d.correct for d in app.waste.disposed.values()) and any(not d.correct for d in app.waste.disposed.values())

    # The package alert is in the log the app replayed.
    assert any(e.type == "agent.alert.raised" and e.rule_id == PACKAGE_RULE for e in app.bus.read_log())


def test_the_handlers_raised_the_alert_the_nudges_the_checklist_and_a_game_round(seeded_dir):
    _, seeded = seeded_dir
    assert any(a.rule_id == PACKAGE_RULE for a in seeded.alerts)
    assert {n.habit_id for n in seeded.nudges} >= {"energy_drinks", "leave_ready", "sort_right"}
    leaving = [p for p in seeded.plans if p.subject == "Leaving"]
    assert any("mentor check-in" in p.text and "Keys: last seen in the desk zone" in p.text for p in leaving)
    assert all(not p.text.startswith("(fake)") for p in seeded.plans)  # no plan was written by the fake model
    assert [r.correct for r in seeded.rounds] == [True]


def test_replaying_the_same_log_twice_gives_identical_projections(seeded_dir, tmp_path):
    data_dir, seeded = seeded_dir
    first = picture(replayed(data_dir / "events", tmp_path / "a"))
    second = picture(replayed(data_dir / "events", tmp_path / "b"))
    assert first == second
    assert first["memory"]["memory_episodes"] and first["waste"]["items_seen"]
    # ... and the same habits and counts the app had when it handled the events live.
    live = seeded.app
    assert first["habits"] == [live.habits.digest(day=day, now=NOW) for day in (SATURDAY, NOW.date())]
    assert first["waste"] == live.waste.metrics()


def test_seed_refuses_a_directory_with_a_log_and_force_adds_nothing_twice(seeded_dir, tmp_path, capsys):
    data_dir, seeded = seeded_dir
    copy = tmp_path / "data"
    shutil.copytree(data_dir, copy)
    before = sorted((p.name, p.read_bytes()) for p in (copy / "events").iterdir())

    with pytest.raises(SeedRefused):
        asyncio.run(seed(copy))
    assert main(["seed", "--data-dir", str(copy)]) == 1
    assert "already has" in capsys.readouterr().err
    assert sorted((p.name, p.read_bytes()) for p in (copy / "events").iterdir()) == before

    again = asyncio.run(seed(copy, force=True))  # the same story on top of itself: every event is a repeat
    again.app.projection.store.close()
    assert not again.written
    assert sorted((p.name, p.read_bytes()) for p in (copy / "events").iterdir()) == before


def test_seed_command_prints_the_summary(tmp_path, capsys):
    assert main(["seed", "--data-dir", str(tmp_path / "data")]) == 0
    out = capsys.readouterr().out
    for expected in ("Events written, by type:", "perception.zone.event", "Memory: ", "Habits, last full day",
                     "Waste Watch: ", "Alerts raised", "Package left at the door"):
        assert expected in out
