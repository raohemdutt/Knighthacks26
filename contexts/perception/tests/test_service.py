import asyncio
from datetime import UTC, datetime, timedelta

from bus import EventBus
from contexts.memory.application.projection import MemoryProjection
from contexts.memory.application.reader import SqliteMemory
from contexts.memory.infrastructure.store import MemoryStore
from contexts.perception.application.service import PerceptionService
from contexts.perception.fakes import script
from contracts.perception import SceneDescribed, ZoneEvent
from contracts.types import ObjectHit, frame_ref
from runtime import FakeRuntime

T0 = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)
END = T0 + timedelta(seconds=40)


def feed() -> list:
    """The scripted scene, with the keyframe a real camera would have saved beside each observation."""
    return [o.model_copy(update={"frame_ref": frame_ref(o.camera_id, int(o.ts.timestamp() * 1000))}) for o in script(T0)]


def runtime() -> FakeRuntime:
    return FakeRuntime(objects=[ObjectHit(label="keys", position="left of the laptop"), ObjectHit(label="stapler", position="far right")])


def service(rt: FakeRuntime, **settings) -> PerceptionService:
    return PerceptionService.load("config/zones.yaml", rt, lambda ref: ref.encode(), **settings)


async def settle() -> None:
    """Let the background model calls finish and publish, as they would between two camera samples."""
    while pending := [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]:
        await asyncio.gather(*pending)


async def live(bus: EventBus, perception: PerceptionService) -> None:
    for o in feed():
        await bus.publish(o)
        await settle()
    await bus.publish_all(perception.flush(END))  # the ticker: whoever is left has gone
    await settle()


def test_feed_becomes_zone_events_captions_and_a_desk_object_list_memory_can_answer_from(tmp_path):
    rt = runtime()
    bus = EventBus(tmp_path / "events")
    perception = service(rt)
    perception.subscribe(bus)
    store = MemoryStore(tmp_path / "memory.sqlite")
    MemoryProjection(store).subscribe(bus)
    out: list = []
    bus.subscribe("perception.zone", out.append)
    bus.subscribe("perception.scene", out.append)
    asyncio.run(live(bus, perception))

    zone_events = [e for e in out if isinstance(e, ZoneEvent)]
    story = [(int((e.ts - T0).total_seconds()), e.label, e.zone_id, e.kind) for e in zone_events if e.kind != "dwell"]
    assert story[:5] == [(0, "person", "door", "enter"), (3, "person", "door", "exit"), (5, "person", "desk", "enter"),
                         (8, "keys", "desk", "enter"), (10, "package", "door", "enter")]
    assert (33, "person", "desk", "exit") in story

    scenes = [e for e in out if isinstance(e, SceneDescribed)]
    captions = [e for e in scenes if e.zone_id is None]
    assert {e.camera_id for e in captions} == {"cam-door", "cam-desk"}
    assert all(e.caption == rt.caption_text and e.model_version == "fake-0" for e in captions)
    assert len({e.event_id for e in scenes}) == len(scenes)
    # Enters and exits, plus the 15 s cadence (desk at 23). An exit shows the last frame the thing was in.
    frames_captioned = {cam: sorted(int((e.ts - T0).total_seconds()) for e in captions if e.camera_id == cam) for cam in ("cam-door", "cam-desk")}
    assert frames_captioned == {"cam-door": [0, 3, 10, 24, 25, 30], "cam-desk": [5, 8, 23, 30]}

    desk = [e for e in scenes if e.zone_id == "desk"]
    assert desk and all(e.camera_id == "cam-desk" and [o.label for o in e.objects] == ["keys"] for e in desk)
    assert desk[-1].ts == T0 + timedelta(seconds=30)  # the last frame with the person in it, looked at when they left

    keys = SqliteMemory(store).last_seen("keys")
    assert (keys.camera_id, keys.zone_id, keys.position, keys.ts) == ("cam-desk", "desk", "left of the laptop", desk[-1].ts)
    assert keys.frame_ref == desk[-1].frame_ref
    episodes = SqliteMemory(store).episodes(T0, END)
    assert {ep.camera_id for ep in episodes if rt.caption_text in ep.captions} == {"cam-door", "cam-desk"}
    store.close()


def test_replay_rebuilds_zone_state_without_calling_the_model(tmp_path):
    first = service(runtime())
    bus = EventBus(tmp_path / "events")
    first.subscribe(bus)

    async def cut_off() -> None:  # the app stops mid-scene: person and keys at the desk, package at the door
        for o in feed():
            await bus.publish(o)
            await settle()

    asyncio.run(cut_off())
    logged = sum(1 for _ in bus.read_log())

    rt = runtime()
    again = service(rt)
    bus2 = EventBus(tmp_path / "events")
    again.subscribe(bus2)

    async def restart() -> None:
        await bus2.replay_into()
        await settle()

    asyncio.run(restart())
    assert rt.calls == []
    assert again.engine.stays.keys() == first.engine.stays.keys() and again.engine.stays
    assert sum(1 for _ in bus2.read_log()) == logged


def test_settings_reach_the_captioner_and_the_finder():
    perception = service(FakeRuntime(), watch_items=["keys", "mug"], caption_every_s=5, objects_every_s=60)
    assert perception.finder.watch_items == ["keys", "mug"]
    assert perception.captioner.every == timedelta(seconds=5) and perception.finder.every == timedelta(seconds=60)
    assert list(perception.finder.boxes) == ["cam-desk"]
