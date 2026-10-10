import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from bus import EventBus
from contexts.memory.application.projection import MemoryProjection
from contexts.memory.application.reader import SqliteMemory
from contexts.memory.infrastructure.store import MemoryStore
from contexts.perception.application.service import PerceptionService
from contexts.perception.fakes import script
from contexts.perception.fakes.script import can_recycled, keys_left, observations, package_left, person_arrives
from contexts.perception.tests.images import jpeg_size, make_jpeg
from contracts.agent import AnswerGenerated
from contracts.experience import QuestionAsked
from contracts.perception import ObservationRecorded, SceneDescribed, ZoneEvent
from contracts.types import ObjectHit, frame_ref
from runtime import FakeRuntime

T0 = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)
END = T0 + timedelta(seconds=40)
# What perception did before it had a budget: full-size images, a caption every 15 s whatever the scene, no pacing,
# no waiting for questions.
OLD = {"caption_max_side": 0, "objects_max_side": 0, "caption_idle_max_s": 15, "min_gap_s": 0, "yield_timeout_s": 0}


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
    for o in observations(END, [], 8):  # the cameras keep reporting an empty room: the last calls take their turns
        await bus.publish(o)
        await settle()


@pytest.mark.parametrize("settings", [{}, OLD], ids=["defaults", "old behaviour"])
def test_feed_becomes_zone_events_captions_and_a_desk_object_list_memory_can_answer_from(tmp_path, settings):
    rt = runtime()
    bus = EventBus(tmp_path / "events")
    perception = service(rt, **settings)
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
    # The scene keeps changing, so the budget drops nothing here: it only makes the calls take turns.
    frames_captioned = {cam: sorted(int((e.ts - T0).total_seconds()) for e in captions if e.camera_id == cam) for cam in ("cam-door", "cam-desk")}
    assert frames_captioned == {"cam-door": [0, 3, 10, 24, 25, 30], "cam-desk": [5, 8, 23, 30]}

    assert [name for name, _ in rt.calls].count("caption") == 10 == perception.stats()["captions_sent"]
    assert perception.stats()["model_calls"] == len(rt.calls) == 12 and perception.stats()["queued"] == 0

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
    # The budget's defaults live in code ...
    assert (perception.captioner.max_side, perception.finder.max_side) == (640, 960)
    assert perception.captioner.idle_max == timedelta(seconds=60) and perception.captioner.coalesce == timedelta(seconds=2)
    assert perception.queue.min_gap == timedelta(seconds=2) and perception.queue.yield_after == timedelta(seconds=3)
    assert perception.queue.yield_timeout == timedelta(seconds=30)
    assert perception.captioner.queue is perception.finder.queue is perception.queue  # one queue between the two
    # ... and a `perception:` block overrides them.
    tuned = service(FakeRuntime(), caption_max_side=512, objects_max_side=0, caption_idle_max_s=120, caption_coalesce_s=1,
                    min_gap_s=0.5, yield_after_question_s=1, yield_timeout_s=10)
    assert (tuned.captioner.max_side, tuned.finder.max_side) == (512, 0)
    assert tuned.captioner.idle_max == timedelta(seconds=120) and tuned.captioner.coalesce == timedelta(seconds=1)
    assert (tuned.queue.min_gap, tuned.queue.yield_after, tuned.queue.yield_timeout) == (
        timedelta(seconds=0.5), timedelta(seconds=1), timedelta(seconds=10))
    assert service(FakeRuntime(), caption_every_s=20, caption_idle_max_s=5).captioner.idle_max == timedelta(seconds=20)


# --- the budget ---


class Watched(FakeRuntime):
    """Notes when (in event time) each image went to the model, and how big it was."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sent: list[tuple[str, float, int, bytes]] = []  # (kind, seconds from T0, bytes, the image)
        self.clock = lambda: T0

    def caption(self, jpeg, prompt="Describe the scene in one sentence."):
        self.sent.append(("caption", (self.clock() - T0).total_seconds(), len(jpeg), jpeg))
        return super().caption(jpeg, prompt)

    def find_objects(self, jpeg, items=None):
        self.sent.append(("objects", (self.clock() - T0).total_seconds(), len(jpeg), jpeg))
        return super().find_objects(jpeg, items)


def keyframed(events: list[ObservationRecorded]) -> list[ObservationRecorded]:
    """Keyframes as the tracking pipeline stores them: when the track ids change, every 5 s with something in
    view, every 30 s regardless. The other observations carry no frame."""
    ids: dict[str, frozenset] = {}
    last: dict[str, datetime] = {}
    out = []
    for o in events:
        now_ids = frozenset(t.track_id for t in o.tracks)
        since = (o.ts - last[o.camera_id]).total_seconds() if o.camera_id in last else float("inf")
        if now_ids != ids.get(o.camera_id) or (o.tracks and since >= 5) or since >= 30:
            ids[o.camera_id], last[o.camera_id] = now_ids, o.ts
            o = o.model_copy(update={"frame_ref": frame_ref(o.camera_id, int(o.ts.timestamp() * 1000))})
        out.append(o)
    return out


def ten_minutes() -> list:
    """Someone comes in, works at the desk for six and a half minutes with their keys beside them and a package by
    the door, asks three questions (the last answer is lost), and leaves. The keys and the package stay."""
    steps = person_arrives(until=400) + keys_left(until=600) + package_left(until=600) + can_recycled()
    asks = [QuestionAsked(ts=T0 + timedelta(seconds=s), question_id=f"q{s}", text="Where are my keys?") for s in (120, 300, 500)]
    answers = [AnswerGenerated(ts=T0 + timedelta(seconds=s + 8), question_id=f"q{s}", answer="On the desk.", model_version="fake-0")
               for s in (120, 300)]
    return sorted([*keyframed(observations(T0, steps, 600)), *asks, *answers], key=lambda e: e.ts)


def play(tmp_path, events: list, keyframe: bytes | None = None, **settings) -> tuple[Watched, PerceptionService]:
    rt = Watched()
    perception = PerceptionService.load("config/zones.yaml", rt, lambda ref: keyframe or ref.encode(), **settings)
    rt.clock = lambda: perception.queue.now
    bus = EventBus(tmp_path / "events")
    perception.subscribe(bus)

    async def go() -> None:
        for e in events:
            await bus.publish(e)
            await settle()

    asyncio.run(go())
    return rt, perception


def test_ten_minutes_cost_the_model_less_than_half_of_what_they_did(tmp_path):
    keyframe = make_jpeg(1280, 720)  # None on a box with neither Pillow nor OpenCV: the counts still hold, the bytes do not
    old_rt, old = play(tmp_path / "old", ten_minutes(), keyframe, **OLD)
    new_rt, new = play(tmp_path / "new", ten_minutes(), keyframe)
    was, now = old.stats(), new.stats()

    assert (was["captions_sent"], was["objects_sent"], was["model_calls"]) == (82, 20, 102) and len(old_rt.sent) == 102
    assert (now["captions_sent"], now["objects_sent"], now["model_calls"]) == (29, 20, 49) and len(new_rt.sent) == 49
    assert was["captions_skipped_redundant"] == 0 and now["captions_skipped_redundant"] == 51
    for stats in (was, now):  # every trigger is accounted for
        assert stats["captions_requested"] == (stats["captions_sent"] + stats["captions_skipped_redundant"]
                                               + stats["captions_coalesced"] + stats["captions_without_frame"])
    assert now["captions_coalesced"] >= 1 and now["queued"] == 0

    times = sorted(t for _, t, _, _ in new_rt.sent)
    assert all(b - a >= 2 for a, b in zip(times, times[1:]))  # min_gap_s: the model is free between two calls
    assert not [t for t in times if 120 <= t < 131 or 300 <= t < 311]  # asked .. answered 8 s later + 3 s of grace
    assert not [t for t in times if 500 <= t < 530] and [t for t in times if 530 <= t < 540]  # lost answer: 30 s, no more
    assert [t for _, t, _, _ in old_rt.sent if 120 <= t < 131]  # before, captions ran straight through a question

    if keyframe is None:
        return
    captions = [(jpeg_size(image), size) for kind, _, size, image in new_rt.sent if kind == "caption"]
    assert {size for size, _ in captions} == {(640, 360)}
    assert {jpeg_size(image) for kind, _, _, image in new_rt.sent if kind == "objects"} == {(960, 371)}
    assert {jpeg_size(image) for kind, _, _, image in old_rt.sent} == {(1280, 720), (1024, 396)}  # frame, and desk crop
    sent_before, sent_now = sum(size for _, _, size, _ in old_rt.sent), sum(size for _, _, size, _ in new_rt.sent)
    assert sent_now == now["bytes_sent"] and sent_before == was["bytes_sent"] == was["bytes_before_scaling"]
    assert now["avg_bytes_sent"] < now["avg_bytes_before_scaling"] and sent_now < sent_before / 3
    assert max(size for _, size in captions) < len(keyframe) / 2


def test_perception_waits_while_a_question_is_answered_and_a_lost_answer_does_not_stall_it(tmp_path):
    desk = [o for o in keyframed(observations(T0, person_arrives(at=-5, until=80), 80)) if o.camera_id == "cam-desk"]
    ask = QuestionAsked(ts=T0 - timedelta(seconds=1), question_id="q1", text="What did I miss?")
    answer = AnswerGenerated(ts=T0 + timedelta(seconds=10.5), question_id="q1", answer="Nothing.", model_version="fake-0")
    lost = QuestionAsked(ts=T0 + timedelta(seconds=20.5), question_id="q2", text="Anyone home?")
    rt, perception = play(tmp_path, sorted([ask, *desk, answer, lost], key=lambda e: e.ts))
    times = [t for _, t, _, _ in rt.sent]
    # Nothing starts before the answer at 10.5 s plus 3 s of grace; the enter and the first desk look then take turns.
    assert times[:2] == [14, 16] and [kind for kind, _, _, _ in rt.sent[:2]] == ["caption", "objects"]
    # The second question is never answered: perception holds off for 30 s and then carries on.
    assert not [t for t in times if 20.5 <= t < 50.5] and [t for t in times if t >= 51]
    assert perception.queue.questions == {} and perception.stats()["queued"] == 0

    never, _ = play(tmp_path / "off", sorted([ask, *desk, answer, lost], key=lambda e: e.ts), yield_timeout_s=0)
    assert [t for _, t, _, _ in never.sent][:2] == [0, 2]  # yield_timeout_s: 0 turns the yielding off
