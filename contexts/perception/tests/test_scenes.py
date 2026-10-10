import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from contexts.perception.application.captioner import Captioner
from contexts.perception.application.object_finder import ObjectFinder
from contexts.perception.domain.scenes import bounding_box, frame_time, objects_caption, scene
from contexts.perception.domain.zones import load_zones
from contexts.perception.infrastructure import frames
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
    async def go():
        return [out for e in events if (out := await handler(e)) is not None]

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
    out = run(Captioner(rt, jpeg_of).on_observation, events)
    assert [(e.camera_id, e.ts) for e in out] == [(cam, at(s)) for s in (0, 15, 30) for cam in ("cam-desk", "cam-door")]
    assert run(Captioner(rt, jpeg_of).on_observation, [seen(s) for s in range(40)]) == []  # empty room: no captions


def test_zone_caption_resets_the_cadence_and_same_moment_events_are_one_caption():
    rt = FakeRuntime()
    c = Captioner(rt, jpeg_of)
    events = [seen(0, PERSON), zone(10, "enter"), zone(10, "enter", label="keys"), zone(11, "exit", label="keys"), *[seen(s, PERSON) for s in range(11, 30)]]

    async def go():
        return [out for e in events if (out := await (c.on_zone_event(e) if isinstance(e, ZoneEvent) else c.on_observation(e)))]

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


def test_camera_is_skipped_while_its_last_caption_is_still_running():
    c = Captioner(FakeRuntime(), jpeg_of)

    async def go():
        return await asyncio.gather(c.on_zone_event(zone(0, "enter")), c.on_zone_event(zone(5, "exit")),
                                    c.on_zone_event(zone(5, "enter", "door", "cam-door")))

    first, second, other_camera = asyncio.run(go())
    assert first and second is None and other_camera


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
