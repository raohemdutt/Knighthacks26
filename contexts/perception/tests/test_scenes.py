import asyncio
import sys
import time
from datetime import UTC, datetime, timedelta

import pytest

from contexts.perception.application.captioner import Captioner
from contexts.perception.application.keyframes import ModelQueue
from contexts.perception.application.object_finder import ObjectFinder
from contexts.perception.domain.scenes import bounding_box, frame_time, objects_caption, scene
from contexts.perception.domain.zones import load_zones
from contexts.perception.infrastructure import frames
from contexts.perception.tests.images import jpeg_size, make_jpeg
from contracts.perception import ObservationRecorded, ZoneEvent
from contracts.types import Box, ObjectHit, Track, frame_ref
from runtime import FakeRuntime

T0 = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)
ZONES = load_zones("config/zones.yaml")
PERSON = Track(track_id=1, label="person", confidence=0.9, box=Box(x1=0.4, y1=0.3, x2=0.6, y2=0.9))
KEYS = ObjectHit(label="keys", position="left of the laptop")


def at(s: float) -> datetime:
    return T0 + timedelta(seconds=s)


def ref(camera: str, s: float) -> str:
    return frame_ref(camera, int(at(s).timestamp() * 1000))


def jpeg_of(frame: str) -> bytes:
    return frame.encode()  # stands in for the pixels: tests can tell which frame reached the model


def no_frames(frame: str) -> bytes:
    raise FileNotFoundError(frame)


def seen(s: float, *tracks: Track, camera: str = "cam-desk", with_frame: bool = True) -> ObservationRecorded:
    return ObservationRecorded(ts=at(s), camera_id=camera, tracks=list(tracks), frame_ref=ref(camera, s) if with_frame else None)


def zone(s: float, kind: str, zone_id: str = "desk", camera: str = "cam-desk", label: str = "person") -> ZoneEvent:
    return ZoneEvent(ts=at(s), camera_id=camera, track_id=1, label=label, zone_id=zone_id, kind=kind,
                     dwell_s=3.0 if kind == "dwell" else None, frame_ref=ref(camera, s))


def run(handler, events) -> list:
    """Every scene event the handler returned, in order. A handler returns those of the model calls it started."""
    async def go():
        return [out for e in events for out in await handler(e)]

    return asyncio.run(go())


# --- scene events ---


def test_scene_takes_its_time_and_id_from_the_frame():
    a = scene("cam-desk", ref("cam-desk", 4), at(9), "A desk.", "fake-0")
    assert a.ts == at(4) and frame_time(a.frame_ref) == at(4)
    assert a.event_id == scene("cam-desk", ref("cam-desk", 4), at(60), "A desk, again.", "fake-1").event_id
    assert a.event_id != scene("cam-desk", ref("cam-desk", 4), at(9), "On the desk: keys.", "fake-0", zone_id="desk").event_id
    fixture = scene("cam-desk", "frames:cam-desk:k1941", at(9), "A desk.", "fake-0")  # no time in the ref
    assert fixture.ts == at(9)


def test_objects_caption_lists_what_was_found():
    assert objects_caption("desk", [KEYS, ObjectHit(label="phone", position="")]) == "On the desk: keys (left of the laptop), phone."
    assert objects_caption("desk", []) == "No watch items on the desk."


def test_bounding_box_of_a_zone_polygon():
    assert bounding_box([(0.2, 0.5), (0.9, 0.4), (0.7, 0.95), (0.1, 0.8)]) == Box(x1=0.1, y1=0.4, x2=0.9, y2=0.95)


# --- captions (P5) ---


def test_caption_on_enter_and_exit_but_not_dwell():
    rt = FakeRuntime(caption_text=" Someone walks in. ")
    out = run(Captioner(rt, jpeg_of).on_zone_event, [zone(0, "enter"), zone(3, "dwell"), zone(6, "dwell"), zone(9, "exit")])
    assert [(e.ts, e.frame_ref) for e in out] == [(at(0), ref("cam-desk", 0)), (at(9), ref("cam-desk", 9))]
    assert all(e.caption == "Someone walks in." and e.model_version == "fake-0" and e.zone_id is None and not e.objects for e in out)


def test_caption_every_15_s_per_camera_while_something_is_in_view():
    rt = FakeRuntime()
    events = [seen(s, PERSON, camera=cam) for s in range(40) for cam in ("cam-desk", "cam-door")]
    out = run(Captioner(rt, jpeg_of, idle_max_s=15).on_observation, events)  # the fixed cadence: no backoff
    assert [(e.camera_id, e.ts) for e in out] == [(cam, at(s)) for s in (0, 15, 30) for cam in ("cam-desk", "cam-door")]
    assert run(Captioner(rt, jpeg_of).on_observation, [seen(s) for s in range(40)]) == []  # empty room: no captions


def test_a_still_scene_is_captioned_less_and_less_often_until_something_changes():
    c = Captioner(FakeRuntime(), jpeg_of)  # 15 s, doubling to 60 s while nothing changes
    still = run(c.on_observation, [seen(s, PERSON) for s in range(200)])
    assert [e.ts for e in still] == [at(s) for s in (0, 15, 45, 105, 165)]
    assert c.stats.captions_sent == 5 and c.stats.captions_requested == 14  # the 14 the fixed cadence would have sent
    assert c.stats.captions_skipped_redundant == 9 and c.period("cam-desk") == timedelta(seconds=60)
    # Keys appear at 200 s: a change, so the next caption is due 15 s after the last one, which is now.
    keys = Track(track_id=3, label="keys", confidence=0.9, box=Box(x1=0.2, y1=0.6, x2=0.3, y2=0.7))
    changed = run(c.on_observation, [seen(s, PERSON, keys) for s in range(200, 250)])
    assert [e.ts for e in changed] == [at(s) for s in (200, 215, 245)]  # 15 s, then 30 s: still again
    # An exit is never held back by the backoff, and puts the cadence back to 15 s.
    assert [e.ts for e in run(c.on_zone_event, [zone(250, "exit", label="keys")])] == [at(250)]
    assert [e.ts for e in run(c.on_observation, [seen(s, PERSON, keys) for s in range(251, 270)])] == [at(265)]


def test_enters_and_exits_close_together_are_one_caption_and_a_waiting_one_takes_the_newest_frame():
    c = Captioner(FakeRuntime(), jpeg_of)
    out = run(c.on_zone_event, [zone(0, "enter"), zone(0, "enter", label="keys"), zone(1.5, "exit", label="keys"), zone(2, "exit")])
    assert [e.ts for e in out] == [at(0), at(2)]
    assert (c.stats.captions_requested, c.stats.captions_sent, c.stats.captions_coalesced) == (4, 2, 2)

    c = Captioner(FakeRuntime(), jpeg_of, queue=ModelQueue(jpeg_of, min_gap_s=5))  # the model may not be used again until 5 s
    out = run(c.on_zone_event, [zone(0, "enter"), zone(3, "exit"), zone(4, "enter"), zone(5, "dwell")])
    assert [e.ts for e in out] == [at(0), at(4)]  # the exit at 3 s was still waiting when the enter at 4 s replaced it
    assert c.stats.captions_coalesced == 1


def test_zone_caption_resets_the_cadence_and_same_moment_events_are_one_caption():
    rt = FakeRuntime()
    c = Captioner(rt, jpeg_of)
    events = [seen(0, PERSON), zone(10, "enter"), zone(10, "enter", label="keys"), zone(11, "exit", label="keys"), *[seen(s, PERSON) for s in range(11, 30)]]

    async def go():
        return [out for e in events for out in await (c.on_zone_event(e) if isinstance(e, ZoneEvent) else c.on_observation(e))]

    assert [e.ts for e in asyncio.run(go())] == [at(0), at(10), at(25)]


def test_missing_frame_is_skipped_quietly_and_does_not_use_up_the_cadence():
    rt = FakeRuntime()
    frames_on_disk = {ref("cam-desk", 2)}

    def load(frame: str) -> bytes:
        if frame not in frames_on_disk:
            raise FileNotFoundError(frame)
        return jpeg_of(frame)

    out = run(Captioner(rt, load).on_observation, [seen(0, PERSON, with_frame=False), seen(1, PERSON), seen(2, PERSON), seen(3, PERSON)])
    assert [e.ts for e in out] == [at(2)]
    assert [name for name, _ in rt.calls].count("caption") == 1


def test_no_caption_event_when_the_model_says_nothing():
    assert run(Captioner(FakeRuntime(caption_text="  "), jpeg_of).on_zone_event, [zone(0, "enter")]) == []


def test_one_model_call_at_a_time_and_one_waiting_per_camera():
    running, most = 0, 0

    class Slow(FakeRuntime):
        def caption(self, jpeg, prompt="Describe the scene in one sentence."):
            nonlocal running, most
            running += 1
            most = max(most, running)
            time.sleep(0.02)
            running -= 1
            return super().caption(jpeg, prompt)

    rt = Slow()
    c = Captioner(rt, jpeg_of)

    async def go():
        return await asyncio.gather(c.on_zone_event(zone(0, "enter")), c.on_zone_event(zone(5, "exit")),
                                    c.on_zone_event(zone(8, "enter")), c.on_zone_event(zone(8, "enter", "door", "cam-door")))

    out = [e for events in asyncio.run(go()) for e in events]
    # While the first caption runs, the desk's exit at 5 s is replaced by its enter at 8 s; the door waits its turn.
    assert sorted((e.camera_id, e.ts) for e in out) == [("cam-desk", at(0)), ("cam-desk", at(8)), ("cam-door", at(8))]
    assert most == 1 and len(rt.calls) == 3 and c.stats.captions_coalesced == 1


# --- watch items on the desk (P4) ---


def finder(rt: FakeRuntime, load=jpeg_of, **kwargs) -> ObjectFinder:
    return ObjectFinder(rt, load, ZONES, **kwargs)


def test_desk_objects_every_30_s_only_on_cameras_with_a_desk():
    rt = FakeRuntime(objects=[KEYS, ObjectHit(label="stapler", position="far right")])
    events = [seen(s, PERSON, camera=cam) for s in range(65) for cam in ("cam-desk", "cam-door")]
    out = run(finder(rt).on_observation, events)
    assert [(e.camera_id, e.zone_id, e.ts) for e in out] == [("cam-desk", "desk", at(s)) for s in (0, 30, 60)]
    assert out[0].objects == [KEYS] and out[0].caption == "On the desk: keys (left of the laptop)."  # stapler is not a watch item
    assert out[0].frame_ref == ref("cam-desk", 0) and out[0].model_version == "fake-0"
    assert rt.calls[0] == ("find_objects", ["keys", "wallet", "charger", "laptop", "phone", "can", "bottle", "cup"])


def test_desk_objects_when_a_person_leaves_the_desk():
    rt = FakeRuntime(objects=[KEYS])
    f = finder(rt)
    assert [e.ts for e in run(f.on_observation, [seen(0, PERSON)])] == [at(0)]
    ignored = [zone(4, "enter"), zone(5, "exit", label="keys"), zone(6, "exit", "door", "cam-door"), zone(7, "dwell")]
    assert run(f.on_zone_event, ignored) == []
    assert [e.ts for e in run(f.on_zone_event, [zone(8, "exit")])] == [at(8)]  # 30 s have not passed: looked anyway
    assert [e.ts for e in run(f.on_observation, [seen(s, PERSON) for s in range(9, 40)])] == [at(38)]


def test_empty_desk_is_published_too():
    out = run(finder(FakeRuntime()).on_zone_event, [zone(8, "exit")])
    assert [(e.objects, e.caption) for e in out] == [([], "No watch items on the desk.")]


def test_watch_items_come_from_config_and_the_model_sees_the_desk_crop():
    rt = FakeRuntime(objects=[KEYS, ObjectHit(label="mug", position="by the lamp")])
    crops = []

    def crop(jpeg: bytes, box: Box) -> bytes:
        crops.append((jpeg, box))
        return jpeg

    out = run(finder(rt, watch_items=["mug"], crop=crop).on_observation, [seen(0, PERSON)])
    assert [o.label for o in out[0].objects] == ["mug"] and rt.calls == [("find_objects", ["mug"])]
    assert crops == [(jpeg_of(ref("cam-desk", 0)), Box(x1=0.10, y1=0.40, x2=0.90, y2=0.95))]


def test_finder_skips_frames_it_cannot_load():
    rt = FakeRuntime(objects=[KEYS])
    assert run(finder(rt, load=no_frames).on_observation, [seen(0, PERSON), seen(1, PERSON, with_frame=False)]) == []
    assert rt.calls == []


# --- keyframes on disk ---


def test_keyframes_are_found_oldest_first_and_loaded_by_ref(tmp_path):
    for camera, name in (("cam-desk", "2000"), ("cam-door", "1000"), ("cam-desk", "3000"), ("cam-desk", "k1941")):
        (tmp_path / "frames" / camera).mkdir(parents=True, exist_ok=True)
        (tmp_path / "frames" / camera / f"{name}.jpg").write_bytes(f"{camera}/{name}".encode())
    assert frames.stored_frames(tmp_path) == [("cam-door", "frames:cam-door:1000"), ("cam-desk", "frames:cam-desk:2000"),
                                              ("cam-desk", "frames:cam-desk:3000")]
    load = frames.keyframe_loader(tmp_path)
    assert load("frames:cam-desk:2000") == b"cam-desk/2000"
    with pytest.raises(FileNotFoundError):
        load("frames:cam-desk:9999")
    assert frames.stored_frames(tmp_path / "nowhere") == []


def test_crop_without_pillow_passes_the_whole_frame(monkeypatch):
    monkeypatch.setattr(frames, "Image", None)
    assert frames.crop_jpeg(b"not really a jpeg", Box(x1=0.1, y1=0.4, x2=0.9, y2=0.95)) == b"not really a jpeg"


def test_crop_with_pillow_cuts_the_box_at_full_resolution():
    image = pytest.importorskip("PIL.Image")
    import io

    buf = io.BytesIO()
    image.new("RGB", (200, 100), "white").save(buf, "JPEG")
    out = image.open(io.BytesIO(frames.crop_jpeg(buf.getvalue(), Box(x1=0.1, y1=0.4, x2=0.9, y2=0.95))))
    assert out.size == (160, 55)


def test_a_frame_that_cannot_be_read_is_passed_whole():
    from contexts.perception.infrastructure.frames import crop_jpeg

    assert crop_jpeg(b"not a jpeg", Box(x1=0.1, y1=0.4, x2=0.9, y2=0.95)) == b"not a jpeg"


# --- smaller copies for the model ---


def test_scaled_size_keeps_the_aspect_ratio_and_never_scales_up():
    assert frames.scaled_size(1280, 720, 640) == (640, 360)
    assert frames.scaled_size(720, 1280, 640) == (360, 640)  # the longest side, whichever it is
    assert frames.scaled_size(1024, 396, 960) == (960, 371)
    assert frames.scaled_size(640, 360, 640) is None and frames.scaled_size(320, 240, 640) is None  # fits already
    assert frames.scaled_size(1280, 720, 0) is None  # 0 = unchanged
    assert frames.scaled_size(4000, 2, 640) == (640, 1)


def check_scaling() -> None:
    """With whichever library `frames` is using: a real 1280x720 JPEG comes out 640x360 and much smaller."""
    big = make_jpeg(1280, 720)
    small = frames.scale_jpeg(big, 640)
    assert jpeg_size(small) == (640, 360) and len(small) < len(big) / 2
    assert jpeg_size(frames.scale_jpeg(make_jpeg(360, 640), 320)) == (180, 320)
    assert frames.scale_jpeg(big, 0) is big and frames.scale_jpeg(big, 1280) is big and frames.scale_jpeg(big, 4000) is big
    assert frames.scale_jpeg(small, 640) is small  # scaling twice changes nothing: not re-encoded when it fits
    assert frames.scale_jpeg(b"not a jpeg", 640) == b"not a jpeg" and frames.scale_jpeg(big[:500], 640) == big[:500]
    desk = frames.scale_jpeg(frames.crop_jpeg(big, Box(x1=0.1, y1=0.4, x2=0.9, y2=0.95)), 960)  # crop first, then scale
    assert jpeg_size(desk) == (960, 371)


def test_scale_with_pillow():
    pytest.importorskip("PIL.Image")
    check_scaling()


def test_scale_and_crop_with_opencv_when_there_is_no_pillow(monkeypatch):
    pytest.importorskip("cv2")
    monkeypatch.setattr(frames, "Image", None)
    check_scaling()
    assert jpeg_size(frames.crop_jpeg(make_jpeg(200, 100), Box(x1=0.1, y1=0.4, x2=0.9, y2=0.95))) == (160, 55)


def test_without_an_image_library_the_frame_goes_out_unchanged(monkeypatch):
    monkeypatch.setattr(frames, "Image", None)
    monkeypatch.setitem(sys.modules, "cv2", None)  # importing it now fails, as on a box without OpenCV
    jpeg = make_jpeg(1280, 720) or b"\xff\xd8 some camera frame"
    assert frames.scale_jpeg(jpeg, 640) is jpeg
    assert frames.crop_jpeg(jpeg, Box(x1=0.1, y1=0.4, x2=0.9, y2=0.95)) is jpeg


def test_the_model_gets_the_scaled_copy_and_the_stored_keyframe_is_untouched(tmp_path):
    jpeg = make_jpeg(1280, 720)
    if jpeg is None:
        pytest.skip("no image library")
    path = tmp_path / "frames" / "cam-desk" / f"{int(at(0).timestamp() * 1000)}.jpg"
    path.parent.mkdir(parents=True)
    path.write_bytes(jpeg)
    sent = []

    class Looking(FakeRuntime):
        def caption(self, image, prompt="Describe the scene in one sentence."):
            sent.append(("caption", jpeg_size(image), len(image)))
            return super().caption(image, prompt)

        def find_objects(self, image, items=None):
            sent.append(("objects", jpeg_size(image), len(image)))
            return super().find_objects(image, items)

    load = frames.keyframe_loader(tmp_path)
    c = Captioner(Looking(), load)
    run(c.on_observation, [seen(0, PERSON)])
    run(ObjectFinder(Looking(), load, ZONES).on_observation, [seen(0, PERSON)])
    run(Captioner(Looking(), load, max_side=0).on_observation, [seen(0, PERSON)])
    run(ObjectFinder(Looking(), load, ZONES, max_side=0).on_observation, [seen(0, PERSON)])
    assert [(kind, size) for kind, size, _ in sent] == [("caption", (640, 360)), ("objects", (960, 371)),
                                                        ("caption", (1280, 720)), ("objects", (1024, 396))]
    assert sent[2][2] == len(jpeg) and sent[0][2] < len(jpeg) / 2  # 0 = the stored bytes, as before
    assert path.read_bytes() == jpeg
    assert (c.stats.images_sent, c.stats.bytes_before_scaling, c.stats.bytes_sent) == (1, len(jpeg), sent[0][2])
