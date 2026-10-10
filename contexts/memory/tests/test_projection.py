import asyncio
from datetime import UTC, datetime, timedelta

from bus import EventBus
from contexts.memory.__main__ import rebuild
from contexts.memory.application.projection import MemoryProjection
from contexts.memory.application.reader import SqliteMemory
from contexts.memory.infrastructure.store import MemoryStore
from contracts.connectors import ItemSynced
from contracts.memory import EpisodeClosed, EpisodeOpened
from contracts.perception import ObservationRecorded, SceneDescribed, ZoneEvent
from contracts.types import Box, ObjectHit, Track, frame_ref

T0 = datetime(2026, 10, 10, 23, 40, tzinfo=UTC)
BOX = Box(x1=0.1, y1=0.1, x2=0.2, y2=0.2)


def at(s: float) -> datetime:
    return T0 + timedelta(seconds=s)


def ref(camera: str, s: float) -> str:
    return frame_ref(camera, int(at(s).timestamp() * 1000))


def seen(s: float, *labels: str, camera: str = "cam-desk") -> ObservationRecorded:
    tracks = [Track(label=label, confidence=0.9, box=BOX, track_id=i) for i, label in enumerate(labels)]
    return ObservationRecorded(ts=at(s), camera_id=camera, tracks=tracks, frame_ref=ref(camera, s))


def scene(s: float, caption: str, *objects: tuple[str, str], camera: str = "cam-desk", zone: str | None = "desk") -> SceneDescribed:
    return SceneDescribed(ts=at(s), camera_id=camera, caption=caption, zone_id=zone, frame_ref=ref(camera, s), model_version="fake-0",
                          objects=[ObjectHit(label=label, position=pos) for label, pos in objects])


def script() -> list:
    """Someone leaves keys on the desk, a package dwells at the door, then the desk has a second visit."""
    return [
        seen(0, "person", "laptop"),
        seen(1, "person", "laptop"),
        scene(5, "A person puts a set of keys next to the coffee mug.", ("keys", "next to the coffee mug")),
        seen(20, "person"),
        ZoneEvent(ts=at(25), camera_id="cam-door", track_id=7, label="package", zone_id="door", kind="enter", frame_ref=ref("cam-door", 25)),
        ZoneEvent(ts=at(50), camera_id="cam-door", track_id=7, label="package", zone_id="door", kind="dwell", dwell_s=25),
        seen(60, camera="cam-desk"),  # empty frame: quiet, closes the desk episode (last activity at 20 s)
        seen(100, "person"),
        scene(110, "The table is empty; the keys are left of the closed laptop.", ("keys", "left of the closed laptop")),
        ItemSynced(ts=at(120), connector="canvas", kind="assignment", external_id="88213", title="Project - Milestone1 Fall2026 [COP4533]",
                   when=datetime(2026, 10, 14, 3, 59, tzinfo=UTC), summary="Due Tuesday night."),
    ]


def run(tmp_path, events, db: str = "memory.sqlite", log: str = "events") -> tuple[MemoryStore, MemoryProjection, list]:
    """Publish through a real bus, as the app does. Returns what the projection emitted too."""
    store = MemoryStore(tmp_path / db)
    projection = MemoryProjection(store)
    bus = EventBus(tmp_path / log)
    emitted: list = []
    bus.subscribe("memory.", emitted.append)
    projection.subscribe(bus)
    asyncio.run(bus.publish_all(events))
    return store, projection, emitted


def test_episodes_open_and_close_on_a_scripted_sequence(tmp_path):
    store, projection, emitted = run(tmp_path, script())
    memory = SqliteMemory(store)
    first, door, second = memory.episodes(at(-1), at(1000))

    assert (first.camera_id, first.start, first.end) == ("cam-desk", at(0), at(20))
    assert first.labels == ["person", "laptop"]
    assert first.captions == ["A person puts a set of keys next to the coffee mug."]
    assert first.keyframe_refs == [ref("cam-desk", 0), ref("cam-desk", 5)]  # first frame, then the captioned one
    assert (door.camera_id, door.start, door.end, door.labels) == ("cam-door", at(25), at(50), ["package"])
    assert (second.start, second.end) == (at(100), None)  # still open: nothing has gone quiet after it yet

    opened = [e for e in emitted if isinstance(e, EpisodeOpened)]
    closed = [e for e in emitted if isinstance(e, EpisodeClosed)]
    assert [e.episode_id for e in opened] == [first.id, door.id, second.id]
    assert [(e.episode_id, e.start, e.end) for e in closed] == [(first.id, at(0), at(20)), (door.id, at(25), at(50))]
    assert closed[0].ts == at(50)  # event time: 30 s after the last activity
    assert closed[0].labels == first.labels and closed[0].captions == first.captions

    # The ticker closes what is left once time has passed, and only once.
    assert projection.flush(at(125)) == []
    last = projection.flush(at(141))
    assert [(e.episode_id, e.end) for e in last] == [(second.id, at(110))]
    assert projection.flush(at(500)) == []
    assert memory.episodes(at(90), at(1000))[0].end == at(110)


def test_a_long_busy_stretch_is_cut_into_bounded_episodes(tmp_path):
    store, _, emitted = run(tmp_path, [seen(s, "laptop") for s in range(0, 1300, 20)])
    eps = SqliteMemory(store).episodes(at(-1), at(2000))
    assert [(e.start, e.end) for e in eps] == [(at(0), at(580)), (at(600), at(1180)), (at(1200), None)]
    assert len([e for e in emitted if isinstance(e, EpisodeClosed)]) == 2


def test_replaying_the_same_log_gives_identical_tables(tmp_path):
    live, projection, _ = run(tmp_path, script())
    ticker = EventBus(tmp_path / "events")
    projection.subscribe(ticker)
    asyncio.run(ticker.publish_all(projection.flush(at(300))))  # a close made by the ticker: logged, but no event caused it
    expected = live.dump()
    assert expected["memory_episodes"] and expected["memory_sightings"] and expected["memory_items"] and expected["memory_fts"]

    # Replay into the live file (an app restart): nothing is inserted twice.
    bus = EventBus(tmp_path / "events")
    MemoryProjection(live).subscribe(bus)
    assert asyncio.run(bus.replay_into()) > len(script())
    assert live.dump() == expected

    # Rebuild into an empty file, twice: the same rows every time.
    for name in ("rebuilt.sqlite", "rebuilt.sqlite", "again.sqlite"):
        rebuild(tmp_path / "events", tmp_path / name)
        rebuilt = MemoryStore(tmp_path / name)
        assert rebuilt.dump() == expected
        rebuilt.close()


def test_last_seen_keys_gives_the_latest_frame_and_zone(tmp_path):
    store, _, _ = run(tmp_path, script())
    memory = SqliteMemory(store)
    keys = memory.last_seen("keys")
    assert (keys.camera_id, keys.zone_id, keys.ts) == ("cam-desk", "desk", at(110))
    assert keys.frame_ref == ref("cam-desk", 110)
    assert keys.position == "left of the closed laptop"
    assert memory.last_seen("wallet") is None


def test_search_finds_a_caption_and_a_canvas_item(tmp_path):
    store, _, _ = run(tmp_path, script())
    memory = SqliteMemory(store)

    hit = memory.search("Where did I leave my keys next to the mug?")[0]
    first = memory.episodes(at(-1), at(1000))[0]
    assert (hit.episode_id, hit.camera_id, hit.ts) == (first.id, "cam-desk", at(0))
    assert hit.text == "A person puts a set of keys next to the coffee mug."
    assert hit.frame_ref == first.keyframe_refs[0] and hit.score > 0

    canvas = memory.search("when is the COP4533 milestone due?")[0]
    assert canvas.episode_id == "item:canvas:88213" and canvas.camera_id == "canvas"
    assert canvas.text.startswith("Project - Milestone1 Fall2026 [COP4533]")
    assert canvas.ts == datetime(2026, 10, 14, 3, 59, tzinfo=UTC)

    both = memory.search("keys milestone", limit=10)
    assert {h.camera_id for h in both} == {"cam-desk", "canvas"}
    assert [h.score for h in both] == sorted((h.score for h in both), reverse=True)
    assert memory.search("is it?") == [] and memory.search('"unbalanced AND (') == []


def test_item_upserts_when_the_same_item_syncs_again(tmp_path):
    def synced(s: float, title: str, due_day: int) -> ItemSynced:
        return ItemSynced(ts=at(s), connector="canvas", kind="assignment", external_id="88213", title=title,
                          when=datetime(2026, 10, due_day, 3, 59, tzinfo=UTC), url="https://canvas.example.edu/a/88213")

    old = synced(0, "Project - Milestone1 Fall2026 [COP4533]", 14)
    new = synced(600, "Project - Milestone1 (extended) [COP4533]", 16)
    other = ItemSynced(ts=at(5), connector="groupme", kind="message", external_id="88213", title="Maya: anyone started the milestone?", when=at(5))
    store, projection, _ = run(tmp_path, [old, other, new])
    memory = SqliteMemory(store)

    assignments = memory.items(at(0), at(0) + timedelta(days=30), kind="assignment")
    assert [(i.title, i.when.day, i.url) for i in assignments] == [("Project - Milestone1 (extended) [COP4533]", 16, "https://canvas.example.edu/a/88213")]
    assert len(memory.items(at(0), at(0) + timedelta(days=30))) == 2  # same external_id on another connector is another item
    assert [h.text for h in memory.search("extended")] == ["Project - Milestone1 (extended) [COP4533]"]
    assert memory.search("Fall2026") == []  # the old title is gone from the index

    projection.on_item(old)  # a replayed event must not bring the old version back
    assert memory.items(at(0), at(0) + timedelta(days=30), kind="assignment")[0].when.day == 16
