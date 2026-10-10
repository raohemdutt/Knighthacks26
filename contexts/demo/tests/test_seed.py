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


def test_item_text_is_worded_from_the_moved_time():
    """O5: the story moved `when` but left "Due Tue Oct 13, 11:59 PM" in the summary under it."""
    def items(now=None):
        return {e.external_id: e for e in story(now) if e.type == "connector.item.synced"}

    milestone, opening, chat = "cop4533:assignment:88213", "outlook:kh-opening:2026-10-10T11:00:00-04:00", "class:1760100001"
    plain = items()
    assert plain[milestone].summary == "Due Tue Oct 13, 11:59 PM. Dynamic programming: code plus a runtime analysis writeup. 100 points."
    assert plain[opening].summary == "until 11:45 AM · Student Union ballroom"
    assert plain[chat].title == "Maya: anyone started the COP4533 milestone 1? due tuesday" and plain[chat].summary.endswith("due tuesday")

    moved = items(NOW - timedelta(days=1, minutes=30))
    assert moved[milestone].summary.startswith("Due Mon Oct 12, 11:29 PM. Dynamic programming")
    assert moved[opening].summary == "until 11:15 AM · Student Union ballroom"  # Day Run reads a block's end from this
    assert moved[chat].title.endswith("due monday") and moved[chat].summary.endswith("due monday")
    assert moved["class:1760100004"].summary.endswith("study session saturday afternoon at the library? 2pm?")
    for e in moved.values():
        assert "{" not in e.title + (e.summary or ""), e.external_id
        if e.kind == "assignment":
            local = e.when.astimezone(NOW.tzinfo)
            assert e.summary.startswith(f"Due {local:%a %b} {local.day}, {local.hour % 12 or 12}:{local:%M %p}."), e.external_id


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


def test_the_study_coach_reviews_saturday_nights_block_with_a_real_distraction(seeded_dir, tmp_path):
    data_dir, seeded = seeded_dir
    app = seeded.app
    (note,) = [e for e in app.bus.read_log() if e.type == "day.coach.noted"]
    # the "Next study block is N minutes." tail depends on what the day's plan still holds, so only the review is pinned
    assert note.day == SATURDAY and note.text.startswith("You did 15 of 50 minutes. Two breaks; the long one came right after the group chat.")
    s = note.session
    assert s.block_id.startswith("outlook:study-block:") and s.title == "Study block: milestone 1 writeup"  # the one on the calendar
    assert (s.planned_minutes, s.desk_minutes, s.late_start_minutes, s.longest_stretch_minutes) == (50, 15, 5, 10)
    long, short = s.breaks
    assert (long.minutes, long.trigger, short.minutes, short.trigger) == (17, "chat", 10, "none")
    chat, phone = long.evidence
    assert chat.id == "groupme:friends:1760100009" and phone.kind == "phone"
    assert "Hold chat notifications until the break" in note.suggestions[0] and "25-minute blocks" in note.suggestions[1]
    # Sunday's plan reads it back and sizes its own blocks to it
    sunday = app.day.planner.current[NOW.date()]
    assert any(src.origin == "coach" and "25-minute blocks" in src.text for src in sunday.sources)
    chosen = [m.minutes for m in sunday.moves if m.kind == "study" and not m.fixed]
    assert chosen and max(chosen) < 25 + 15
    # and a box restarted on the same log says the same, without saying it again
    again = replayed(data_dir / "events", tmp_path)
    assert again.day.coach.latest == note and again.day.coach.sessions(SATURDAY, NOW) == [s]
    assert again.day.coach.tick(NOW) == []
    episode = next(e for e in again.memory.episodes(NOW - timedelta(days=2), NOW) if e.id == phone.id)  # a real Memory episode
    assert any("picks up their phone" in c for c in episode.captions)
    again.projection.store.close()


def test_the_coach_finds_the_same_distraction_whatever_time_the_story_is_moved_to(tmp_path):
    """`seed --now` moves the story by any amount; the calendar block moves with it, so the session does too."""
    by = timedelta(hours=-20, minutes=-7)  # an afternoon run: the block lands in the small hours, in the day before's plan
    seeded = asyncio.run(seed(tmp_path, now=NOW + by))
    notes = [e for e in seeded.app.bus.read_log() if e.type == "day.coach.noted"]
    seeded.app.projection.store.close()
    (note,) = [n for n in notes if n.block_id.startswith("outlook:study-block:")]
    assert note.text.startswith("You did 15 of 50 minutes. Two breaks; the long one came right after the group chat.")
    assert note.session.start == load().beats[0].at.replace(hour=22, minute=25) + by
    assert [b.trigger for b in note.session.breaks] == ["chat", "none"] and note is max(notes, key=lambda n: n.ts)  # the Now screen's card


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
