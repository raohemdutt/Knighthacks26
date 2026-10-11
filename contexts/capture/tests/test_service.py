import asyncio

import pytest

pytest.importorskip("cv2")
import cv2  # noqa: E402
import httpx  # noqa: E402
import numpy as np  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from datetime import UTC, datetime  # noqa: E402

from bus import EventBus  # noqa: E402
from contexts.capture.api import capture_router  # noqa: E402
from contexts.capture.application.service import CaptureService  # noqa: E402
from contracts.types import Box, Track  # noqa: E402


class Policy:
    def __init__(self, blur=True, paused=()):
        self.blur_faces, self.paused = blur, set(paused)

    def is_paused(self, camera_id):
        return camera_id in self.paused


def person(frame):
    return [Track(label="person", confidence=0.9, track_id=1, box=Box(x1=0.1, y1=0.1, x2=0.4, y2=0.9))]


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.avi"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for i in range(10):
        out.write(np.full((48, 64, 3), 20 * i, np.uint8))
    out.release()
    return str(path)


def run(clip, tmp_path, policy, seconds=2.0):
    seen = []

    async def main():
        bus = EventBus(persist=False)

        async def on(event):
            seen.append(event)

        bus.subscribe("perception.observation", on)
        cameras = {"cam-desk": {"source": clip, "fps": 15, "width": 64, "height": 48, "observe_every_s": 0.3}}
        service = CaptureService(cameras, bus, lambda: person, tmp_path / "data", policy)
        service.start(asyncio.get_running_loop())
        await asyncio.sleep(seconds)
        stats = service.stats()
        service.stop()
        await asyncio.sleep(0.05)
        return service, stats

    service, stats = asyncio.run(main())
    return service, stats, seen


def test_a_camera_becomes_observations_and_a_stored_keyframe(clip, tmp_path):
    service, stats, seen = run(clip, tmp_path, Policy())
    assert len(seen) >= 3 and seen[0].camera_id == "cam-desk" and seen[0].tracks[0].label == "person"
    ref = seen[0].frame_ref
    assert ref and (tmp_path / "data" / "frames" / "cam-desk" / f"{ref.split(':')[2]}.jpg").exists()
    assert stats["cam-desk"]["status"] == "ok" and stats["cam-desk"]["detected"] > 5 and stats["cam-desk"]["errors"] == 0


def test_a_paused_camera_is_not_looked_at(clip, tmp_path):
    service, stats, seen = run(clip, tmp_path, Policy(paused={"cam-desk"}), seconds=1.0)
    assert seen == [] and stats["cam-desk"]["detected"] == 0


def test_blur_is_applied_unless_the_policy_turns_it_off(clip, tmp_path, monkeypatch):
    calls = []

    class Spy:
        def __init__(self):
            pass

        def __call__(self, frame):
            calls.append(1)
            return frame

    monkeypatch.setattr("contexts.capture.application.service.FaceBlur", Spy)
    run(clip, tmp_path / "a", Policy(blur=True), seconds=1.0)
    assert calls  # keyframes were blurred
    calls.clear()
    run(clip, tmp_path / "b", Policy(blur=False), seconds=1.0)
    assert not calls


def test_nothing_starts_without_cameras_or_a_detector(tmp_path, monkeypatch):
    bus = EventBus(persist=False)
    models = tmp_path / "models.yaml"
    models.write_text("detector: {backend: ultralytics, weights: " + (tmp_path / "missing.engine").as_posix() + "}\n")
    cams = tmp_path / "cameras.yaml"
    assert CaptureService.load(cams, bus, models_yaml=models) is None  # no cameras file
    cams.write_text("cameras: {cam-desk: {source: /dev/video0}}\n")
    assert CaptureService.load(cams, bus, models_yaml=models) is None  # cameras, but no engine here
    (tmp_path / "x.engine").write_bytes(b"x")
    models.write_text("detector: {backend: ultralytics, weights: " + (tmp_path / "x.engine").as_posix() + "}\n")
    monkeypatch.setenv("ATLAS_CAPTURE", "0")
    assert CaptureService.load(cams, bus, models_yaml=models) is None  # switched off


def test_status_endpoint(clip, tmp_path):
    service, _, _ = run(clip, tmp_path, Policy(), seconds=0.8)
    app = FastAPI()
    app.include_router(capture_router(service))
    body = TestClient(app).get("/capture/status").json()
    assert set(body["cameras"]["cam-desk"]) == {"status", "fps_actual", "dropped", "detected", "errors"}


def test_the_detector_is_only_built_when_a_frame_arrives(tmp_path):
    from contexts.capture.application.service import LazyDetector

    built = []

    def factory():
        built.append(1)
        return person

    lazy = LazyDetector(factory)
    assert built == []  # nothing yet: a camera that is unplugged costs no memory
    assert lazy("frame")[0].track_id == 1 and lazy("frame") and built == [1]  # built once, on the first frame


def test_the_live_screen_gets_a_blurred_snapshot_and_the_newest_tracks(clip, tmp_path, monkeypatch):
    calls = []

    class Spy:
        def __call__(self, frame):
            calls.append(1)
            return frame

    monkeypatch.setattr("contexts.capture.application.service.FaceBlur", Spy)
    cameras = {"cam-desk": {"source": clip, "fps": 15, "width": 64, "height": 48, "faces": False}}
    holder = {}

    async def main():
        bus = EventBus(persist=False)
        service = CaptureService(cameras, bus, lambda: person, tmp_path / "data", Policy())
        holder["service"] = service
        service.start(asyncio.get_running_loop())
        await asyncio.sleep(1.0)
        holder["jpeg"] = service.snapshot("cam-desk")
        holder["again"] = service.snapshot("cam-desk")
        holder["tracks"] = service.tracks("cam-desk")
        app = FastAPI()
        app.include_router(capture_router(service))
        holder["http"] = TestClient(app)
        holder["r"] = holder["http"].get("/capture/cam-desk/snapshot.jpg")
        holder["missing"] = holder["http"].get("/capture/cam-nope/snapshot.jpg")
        service.policy.paused.add("cam-desk")
        holder["paused"] = service.snapshot("cam-desk")
        service.stop()

    asyncio.run(main())
    assert holder["jpeg"][:2] == b"\xff\xd8" and calls  # a JPEG, and the blur ran on it
    assert holder["tracks"]["tracks"][0]["track_id"] == 1 and holder["tracks"]["ts_ms"] > 0
    assert holder["r"].status_code == 200 and holder["r"].headers["content-type"] == "image/jpeg"
    assert holder["r"].headers["cache-control"] == "no-store"
    assert holder["missing"].status_code == 404 and holder["paused"] is None


def test_health_is_published_on_change_and_as_a_five_minute_heartbeat(clip, tmp_path):
    from datetime import timedelta

    async def main():
        bus = EventBus(persist=False)
        policy = Policy()
        service = CaptureService({"cam-desk": {"source": clip, "fps": 15, "width": 64, "height": 48, "faces": False}}, bus,
                                 lambda: person, tmp_path / "data", policy)
        service.start(asyncio.get_running_loop())
        await asyncio.sleep(0.8)
        t0 = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)
        first = service.health(t0)
        again = service.health(t0 + timedelta(seconds=30))
        policy.paused.add("cam-desk")
        paused = service.health(t0 + timedelta(seconds=60))
        policy.paused.clear()
        beat = service.health(t0 + timedelta(minutes=7))
        service.stop()
        return first, again, paused, beat

    first, again, paused, beat = asyncio.run(main())
    assert [(e.camera_id, e.status) for e in first] == [("cam-desk", "ok")] and first[0].fps_actual > 5
    assert again == []  # nothing changed, and not yet five minutes
    assert [e.status for e in paused] == ["paused"]  # a change publishes at once
    assert [e.status for e in beat] == ["ok"]  # back to ok after the pause (a change, and past the heartbeat)


def test_the_mjpeg_stream_sends_boundaries_with_jpegs(clip, tmp_path):
    # The whole test runs on one event loop. The camera reader publishes onto the loop it was started on; once that
    # loop is closed the reader thread dies on its first publish and the newest frame never changes again, and a
    # stream asked for two frames waits for the second one for ever.
    async def main():
        bus = EventBus(persist=False)
        service = CaptureService({"cam-desk": {"source": clip, "fps": 15, "width": 64, "height": 48, "faces": False}}, bus,
                                 lambda: person, tmp_path / "data", Policy())
        service.start(asyncio.get_running_loop())
        try:
            deadline = asyncio.get_running_loop().time() + 30.0  # the first frame: a bound, not a fixed sleep
            while service.snapshot("cam-desk") is None:
                assert asyncio.get_running_loop().time() < deadline, "the camera delivered no frame"
                await asyncio.sleep(0.02)
            app = FastAPI()
            app.include_router(capture_router(service))
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://box") as http:
                assert (await http.get("/capture/live/cam-nope")).status_code == 404
                async with http.stream("GET", "/capture/live/cam-desk?fps=15&max=2") as r:
                    assert r.status_code == 200
                    assert r.headers["content-type"].startswith("multipart/x-mixed-replace; boundary=frame")
                    return b"".join([part async for part in r.aiter_bytes()])
        finally:
            service.stop()

    chunk = asyncio.run(asyncio.wait_for(main(), 60.0))
    assert chunk.startswith(b"--frame\r\nContent-Type: image/jpeg") and b"\xff\xd8" in chunk
    assert chunk.count(b"--frame\r\n") == 2  # max=2: two frames, then the stream ends
