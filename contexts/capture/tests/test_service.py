import asyncio

import pytest

pytest.importorskip("cv2")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

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
