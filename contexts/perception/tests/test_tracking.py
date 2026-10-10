import asyncio
import time

import pytest

pytest.importorskip("cv2")
import numpy as np  # noqa: E402

from bus import EventBus  # noqa: E402
from contexts.perception.application.tracking import TrackingPipeline  # noqa: E402
from contexts.perception.infrastructure.frames import keyframe_loader  # noqa: E402
from contexts.perception.infrastructure.keyframe_store import KeyframeStore  # noqa: E402
from contracts.types import Box, Track  # noqa: E402


class FakeReader:
    """Serves a new frame every 50 ms, like a 20 fps camera."""

    def __init__(self):
        self.t0 = time.monotonic()

    def latest(self):
        n = int((time.monotonic() - self.t0) / 0.05) + 1
        return n, 1_700_000_000_000 + n * 50, np.full((48, 64, 3), n % 250, np.uint8)


def person(track_id):
    return Track(label="person", confidence=0.9, track_id=track_id, box=Box(x1=0.1, y1=0.1, x2=0.4, y2=0.9))


def drive(detector, seconds, store=None, **kw):
    seen, calls = [], []

    async def main():
        bus = EventBus(persist=False)

        async def on(event):
            seen.append(event)

        bus.subscribe("perception.observation", on)

        def counted(frame):
            calls.append(time.monotonic())
            return detector(frame)

        pipe = TrackingPipeline("cam-desk", FakeReader(), counted, bus, asyncio.get_running_loop(), store=store, **kw)
        pipe.start()
        await asyncio.sleep(seconds)
        pipe.stop()
        await asyncio.sleep(0.05)
        return pipe

    return asyncio.run(main()), seen, calls


def test_detects_at_its_own_rate_and_observes_once_per_period():
    pipe, seen, calls = drive(lambda frame: [person(1)], 1.6, detect_fps=10, observe_every_s=0.5)
    assert 12 <= len(calls) <= 17  # about 10 per second although the camera gives 20
    assert 3 <= len(seen) <= 4  # one every half second
    assert all(e.camera_id == "cam-desk" and [t.track_id for t in e.tracks] == [1] for e in seen)
    assert pipe.errors == 0


def test_a_failing_detector_does_not_stop_the_camera():
    state = {"n": 0}

    def flaky(frame):
        state["n"] += 1
        if state["n"] % 2:
            raise RuntimeError("engine hiccup")
        return [person(1)]

    pipe, seen, _ = drive(flaky, 1.2, detect_fps=20, observe_every_s=0.3)
    assert pipe.errors >= 3 and seen


def test_keyframes_are_stored_when_who_is_in_view_changes_and_named_in_the_observation(tmp_path):
    ids = {"now": 1}
    store = KeyframeStore(tmp_path)
    pipe, seen, _ = drive(lambda frame: [person(ids["now"])], 1.5, store=store, detect_fps=10, observe_every_s=0.3,
                          keyframe_every_s=60, keepalive_s=60)
    assert seen[0].frame_ref and all(e.frame_ref is None for e in seen[1:])  # same person: no new keyframe
    jpeg = keyframe_loader(tmp_path)(seen[0].frame_ref)
    assert jpeg[:2] == b"\xff\xd8"
    camera_dir = tmp_path / "frames" / "cam-desk"
    assert len(list(camera_dir.glob("*.jpg"))) == 1 and not list(camera_dir.glob("*.tmp"))


def test_a_new_track_id_stores_another_keyframe(tmp_path):
    state = {"t0": time.monotonic()}

    def detector(frame):
        return [person(1 if time.monotonic() - state["t0"] < 0.8 else 2)]

    pipe, seen, _ = drive(detector, 1.8, store=KeyframeStore(tmp_path), detect_fps=10, observe_every_s=0.3,
                          keyframe_every_s=60, keepalive_s=60)
    assert sum(e.frame_ref is not None for e in seen) == 2
