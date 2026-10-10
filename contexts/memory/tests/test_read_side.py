from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from contexts.agent.application.memory_port import MemoryReader
from contexts.agent.eval import GOLDEN
from contexts.memory.__main__ import main
from contexts.memory.api import build_router
from contexts.memory.application.reader import SqliteMemory
from contexts.memory.fixture import Fixture, fake_memory, fill_store, load
from contexts.memory.infrastructure.store import MemoryStore
from contracts.memory import Episode, EpisodeWindow, Item, MemoryStats, SearchHit, Sighting

NY = ZoneInfo("America/New_York")
SAT_NOON = datetime(2026, 10, 10, 12, tzinfo=NY)


@pytest.fixture(scope="module")
def fx() -> Fixture:
    return load()


@pytest.fixture
def memory(fx, tmp_path) -> SqliteMemory:
    store = MemoryStore(tmp_path / "memory.sqlite")
    fill_store(store, fx)
    return SqliteMemory(store)


def test_fixture_tells_the_demo_story(fx):
    assert 45 <= len(fx.episodes) <= 55
    assert {e.camera_id for e in fx.episodes} == {"cam-desk", "cam-door"}
    assert {s.zone_id for s in fx.sightings} == {"desk", "door", "recycling"}
    assert fx.episodes[0].start.astimezone(NY).hour == 12 and fx.episodes[-1].start < fx.now
    assert all(e.end and e.end > e.start and e.captions and e.keyframe_refs for e in fx.episodes)
    assert len({e.id for e in fx.episodes}) == len(fx.episodes)
    overnight = [e for e in fx.episodes if 1 <= e.start.astimezone(NY).hour < 6]
    assert 3 <= len(overnight) <= 8  # long empty stretches, a few visitors
    milestone = next(i for i in fx.items if i.title == "Project - Milestone1 Fall2026 [COP4533]")
    assert (milestone.connector, milestone.kind, milestone.when.astimezone(NY).date().isoformat()) == ("canvas", "assignment", "2026-10-13")
    assert sum(i.kind == "calendar_event" for i in fx.items) >= 3
    assert sum((i.connector, i.kind) == ("groupme", "message") for i in fx.items) >= 3


def test_sqlite_memory_answers_like_fake_memory(fx, memory):
    fake: MemoryReader = fake_memory(fx)
    start, end = SAT_NOON, fx.now
    assert memory.episodes(start, end) == fake.episodes(start, end) and len(memory.episodes(start, end)) == len(fx.episodes)
    night = (datetime(2026, 10, 10, 20, tzinfo=NY), datetime(2026, 10, 11, 8, tzinfo=NY))
    assert memory.episodes(*night, camera="cam-door", label="package") == fake.episodes(*night, camera="cam-door", label="package") != []
    assert memory.window(*night) == fake.window(*night)
    assert memory.items(start, end + timedelta(days=7)) == fake.items(start, end + timedelta(days=7))
    assert memory.items(start, end, kind="message") == fake.items(start, end, kind="message") != []
    for label in ("keys", "package", "can", "umbrella"):
        assert memory.last_seen(label) == fake.last_seen(label)
    assert memory.episodes(end, end + timedelta(days=1)) == []


def test_the_agent_passes_its_golden_questions_on_sqlite(tmp_path):
    """Drop-in check: the agent's own eval, with the golden fixture served from the SQLite store."""
    import asyncio

    from contexts.agent.application.qa import QA, SleepSchedule
    from contexts.agent.eval import load_fixture
    from runtime import FakeRuntime

    _, now, questions = load_fixture()
    store = MemoryStore(tmp_path / "golden.sqlite")
    fill_store(store, load(GOLDEN / "memory.yaml"))
    memory = SqliteMemory(store)
    assert memory.last_seen("keys").frame_ref == "frames:cam-desk:k1941"
    assert memory.search("courier cardboard box")[0].episode_id == "e08"

    qa = QA(FakeRuntime(), memory, clock=lambda: now, sleep=SleepSchedule.load())
    for g in questions:
        answer = asyncio.run(qa.ask(g.q))
        cited = {ref for c in answer.citations for ref in (c.episode_id, c.frame_ref, c.url) if ref}
        assert not cited if g.no_evidence else set(g.cite) <= cited, g.q


def test_last_seen_keys_is_the_morning_sighting_on_the_desk(memory):
    keys = memory.last_seen("keys")
    assert (keys.camera_id, keys.zone_id) == ("cam-desk", "desk")
    assert keys.ts == datetime(2026, 10, 11, 9, 12, 45, tzinfo=NY)
    assert keys.frame_ref == f"frames:cam-desk:{int(keys.ts.timestamp() * 1000)}"
    assert keys.position == "left of the laptop, behind the coffee mug"


def test_window_groups_by_hour(memory):
    window = memory.window(datetime(2026, 10, 10, 13, tzinfo=NY), datetime(2026, 10, 10, 16, tzinfo=NY))
    assert [b.hour for b in window.hours] == [datetime(2026, 10, 10, h, tzinfo=NY) for h in (13, 14, 15)]
    lunch = window.hours[0]
    assert lunch.episode_ids == ["e04", "e05", "e06"]
    assert lunch.captions[-1] == "Lunch: three people eat pizza at the table."  # a summary stands in for its captions
    assert len(lunch.keyframe_refs) == 3

    overnight = memory.window(datetime(2026, 10, 11, 1, tzinfo=NY), datetime(2026, 10, 11, 6, tzinfo=NY))
    assert [b.hour.astimezone(NY).hour for b in overnight.hours] == [1, 2, 3, 4, 5]
    assert all(len(b.captions) <= 6 for b in memory.window(SAT_NOON, SAT_NOON + timedelta(days=1)).hours)


def test_search_returns_episodes_and_items_together(memory):
    assert memory.search("keys coffee mug")[0].episode_id == "e18"
    hits = memory.search("milestone", limit=10)
    assert "item:canvas:cop4533:assignment:88213" in [h.episode_id for h in hits]
    drinks = memory.search("energy drinks", limit=10)
    assert {h.camera_id for h in drinks} >= {"cam-desk", "groupme"}
    assert all(h.frame_ref for h in drinks if h.camera_id.startswith("cam-"))


def test_search_returns_each_episode_and_item_once_with_its_best_text(tmp_path):
    """O4: one episode came back three times, for its summary, its object caption and its first caption."""
    store = MemoryStore(tmp_path / "memory.sqlite")
    start = datetime(2026, 10, 10, 19, 40, tzinfo=NY)
    for n, captions in enumerate((["A person puts a set of keys next to the coffee mug.", "On the desk: keys (next to the coffee mug)."],
                                  ["The keys are still on the desk.", "Keys, keys and more keys: three sets of keys."],
                                  ["Nobody is at the table."])):
        ep = Episode(id=f"ep{n}", camera_id="cam-desk", start=start + timedelta(hours=n), end=start + timedelta(hours=n, minutes=2),
                     captions=captions)
        store.put_episode(ep, ep.end)
        for caption in captions:
            store.index("caption", ep.id, caption)
    store.set_summary("ep0", "Someone left their keys on the desk.")
    for n in (1, 2):
        store.put_item(Item(connector="groupme", kind="message", external_id=f"m{n}", title="Sam: has anyone seen my keys",
                            when=start, summary="Sam in Friends: has anyone seen my keys"), synced_at=start)
    memory = SqliteMemory(store)

    hits = memory.search("keys", limit=20)
    ids = [h.episode_id for h in hits]
    assert sorted(ids) == ["ep0", "ep1", "item:groupme:m1", "item:groupme:m2"]  # was 7 rows: ep0 three times, ep1 twice
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert next(h for h in hits if h.episode_id == "ep1").text == "Keys, keys and more keys: three sets of keys."  # its best text
    # `limit` counts distinct results, and the cut is still best first
    assert [h.episode_id for h in memory.search("keys", limit=3)] == ids[:3]
    assert len(memory.search("keys", limit=1)) == 1


def test_stats(memory, fx):
    stats = memory.stats()
    assert stats.episodes == len(fx.episodes)
    assert stats.captions == sum(len(e.captions) for e in fx.episodes)
    assert stats.first_event == fx.episodes[0].start
    assert 21 < stats.hours_recorded < 22
    assert SqliteMemory(MemoryStore(":memory:")).stats() == MemoryStats(hours_recorded=0, episodes=0, captions=0)


def test_api_returns_the_contract_shapes(memory, tmp_path):
    frames = tmp_path / "frames"
    (frames / "cam-desk").mkdir(parents=True)
    (frames / "cam-desk" / "1791724365000.jpg").write_bytes(b"\xff\xd8jpeg")
    app = FastAPI()
    app.include_router(build_router(memory, frames))
    c = TestClient(app)

    night = {"from": "2026-10-10T20:00:00-04:00", "to": "2026-10-11T08:00:00-04:00"}
    eps = [Episode.model_validate(e) for e in c.get("/memory/episodes", params={**night, "camera": "cam-door", "label": "package"}).json()]
    assert eps == memory.episodes(datetime(2026, 10, 10, 20, tzinfo=NY), datetime(2026, 10, 11, 8, tzinfo=NY), camera="cam-door", label="package")
    assert len(c.get("/memory/episodes").json()) == 50

    window = EpisodeWindow.model_validate(c.get("/memory/window", params=night).json())
    assert window.start == datetime(2026, 10, 10, 20, tzinfo=NY) and len(window.hours) >= 8

    keys = Sighting.model_validate(c.get("/memory/objects/keys/last-seen").json())
    assert keys.zone_id == "desk"
    assert c.get("/memory/objects/umbrella/last-seen").status_code == 404

    hits = [SearchHit.model_validate(h) for h in c.get("/memory/search", params={"q": "package door", "limit": 3}).json()]
    assert len(hits) == 3 and "package" in hits[0].text
    assert c.get("/memory/search").status_code == 422

    items = [Item.model_validate(i) for i in c.get("/memory/items", params={"from": "2026-10-11T09:30:00-04:00", "kind": "assignment"}).json()]
    assert [i.title for i in items] == ["Project - Milestone1 Fall2026 [COP4533]", "Quiz 5 - Greedy Algorithms [COP4533]"]

    jpeg = c.get(f"/memory/keyframes/{keys.frame_ref}")
    assert jpeg.status_code == 200 and jpeg.headers["content-type"] == "image/jpeg" and jpeg.content == b"\xff\xd8jpeg"
    for missing in ("frames:cam-desk:1", "frames:..:1791724365000", "cam-desk", "frames:cam-desk:..%2F..%2Fmemory"):
        assert c.get(f"/memory/keyframes/{missing}").status_code == 404

    assert MemoryStats.model_validate(c.get("/memory/stats").json()).episodes == 50


def test_keyframes_never_follow_a_link_out_of_the_frames_folder(memory, tmp_path):
    frames = tmp_path / "frames"
    (frames / "cam-desk").mkdir(parents=True)
    (frames / "cam-desk" / "1.jpg").write_bytes(b"\xff\xd8jpeg")
    secret = tmp_path / "secret.jpg"
    secret.write_bytes(b"SECRET")
    (frames / "cam-desk" / "2.jpg").symlink_to(secret)  # a linked file
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "3.jpg").write_bytes(b"OUTSIDE")
    (frames / "cam-link").symlink_to(tmp_path / "elsewhere")  # a linked camera folder
    app = FastAPI()
    app.include_router(build_router(memory, frames))
    c = TestClient(app)
    assert c.get("/memory/keyframes/frames:cam-desk:1").content == b"\xff\xd8jpeg"
    assert c.get("/memory/keyframes/frames:cam-desk:2").status_code == 404
    assert c.get("/memory/keyframes/frames:cam-link:3").status_code == 404


def test_seed_command_fills_a_database(tmp_path, capsys):
    assert main(["seed", "--data-dir", str(tmp_path)]) == 0
    assert "50 episodes" in capsys.readouterr().out
    assert SqliteMemory(MemoryStore(tmp_path / "memory.sqlite")).last_seen("keys").zone_id == "desk"


def test_a_fixture_can_be_moved_to_end_now():
    from datetime import UTC, datetime

    from contexts.memory.fixture import as_of

    fx = load()
    now = datetime(2027, 3, 1, 9, 30, tzinfo=UTC)
    moved = as_of(fx, now)
    by = now - fx.now
    assert moved.now == now and len(moved.episodes) == len(fx.episodes)
    assert moved.episodes[0].start == fx.episodes[0].start + by
    assert moved.sightings[0].ts == fx.sightings[0].ts + by
    assert [i.when for i in moved.items if i.when] == [i.when + by for i in fx.items if i.when]
    assert moved.episodes[0].keyframe_refs == fx.episodes[0].keyframe_refs  # refs are names, not times
