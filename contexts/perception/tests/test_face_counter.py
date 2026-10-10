import asyncio
import time

import pytest

pytest.importorskip("cv2")

from bus import EventBus  # noqa: E402
from contexts.perception.application.tracking import TrackingPipeline  # noqa: E402
from contexts.perception.infrastructure.face_counter import FaceCounter, is_facing  # noqa: E402


def row(w=80, eye_dist=40, nose_dx=0, score=0.9):
    """A YuNet row: a face w wide, eyes eye_dist apart centred at x=100, the nose nose_dx off centre."""
    return [80, 60, w, 100, 100 - eye_dist / 2, 90, 100 + eye_dist / 2, 90, 100 + nose_dx, 110, 90, 130, 110, 130, score]


def test_a_frontal_face_is_facing_and_a_profile_or_a_doubtful_one_is_not():
    assert is_facing(row())  # eyes half the face width apart, nose between them
    assert not is_facing(row(eye_dist=20))  # eyes squeezed together: seen from the side
    assert not is_facing(row(nose_dx=25))  # nose well off to one side: turned away
    assert not is_facing(row(score=0.55))  # YuNet itself is unsure


def test_the_counter_returns_faces_and_how_many_face_the_camera():
    class Finder:
        def faces(self, frame):
            return [row(), row(eye_dist=15), row(score=0.5)]

    assert FaceCounter(Finder())(object()) == (3, 1)
    assert FaceCounter(type("F", (), {"faces": lambda self, f: []})())(object()) == (0, 0)


class Reader:
    def __init__(self):
        self.t0 = time.monotonic()

    def latest(self):
        import numpy as np

        n = int((time.monotonic() - self.t0) / 0.05) + 1
        return n, 1_700_000_000_000 + n * 50, np.zeros((48, 64, 3), np.uint8)


def test_the_pipeline_publishes_counts_when_they_change_and_as_a_heartbeat():
    seen = []
    state = {"t0": time.monotonic()}

    def faces(frame):
        return (1, 1) if time.monotonic() - state["t0"] < 1.0 else (2, 1)

    async def main():
        bus = EventBus(persist=False)

        async def on(event):
            seen.append(event)

        bus.subscribe("perception.faces", on)
        pipe = TrackingPipeline("cam-desk", Reader(), lambda f: [], bus, asyncio.get_running_loop(), detect_fps=20,
                                observe_every_s=0.25, faces=faces, faces_every_s=0.8)
        pipe.start()
        await asyncio.sleep(2.4)
        pipe.stop()
        await asyncio.sleep(0.05)

    asyncio.run(main())
    counts = [(e.count, e.facing) for e in seen]
    assert counts[0] == (1, 1) and (2, 1) in counts  # the first reading, then the change
    assert 3 <= len(counts) <= 6  # unchanged readings only as a heartbeat every 0.8 s, not on every observation
    assert all(e.camera_id == "cam-desk" for e in seen)
