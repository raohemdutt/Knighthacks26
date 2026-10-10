import asyncio

import pytest

pytest.importorskip("cv2")
import numpy as np  # noqa: E402

from bus import EventBus  # noqa: E402
from contexts.capture.infrastructure.camera import CameraReader  # noqa: E402


class FakeCapture:
    """A 'camera' that serves alternating black and white 320x240 frames, as fast as it is asked."""

    def __init__(self, limit=None):
        self.opened, self.n, self.limit = True, 0, limit

    def isOpened(self):
        return self.opened

    def set(self, *_):
        return True

    def grab(self):
        return self.read()[0]

    def read(self):
        if self.limit is not None and self.n >= self.limit:
            return False, None
        self.n += 1
        return True, np.full((240, 320, 3), 255 * (self.n % 2), np.uint8)

    def release(self):
        self.opened = False


def drive(source, open_capture, seconds, **kw):
    """Run a reader for a while on a real event loop; returns (reader, events seen on the bus)."""
    events = []

    async def main():
        bus = EventBus(persist=False)

        async def handler(event):
            events.append(event)

        bus.subscribe("capture.frames", handler)
        reader = CameraReader("cam-table", source, bus=bus, loop=asyncio.get_running_loop(),
                              open_capture=open_capture, **kw)
        reader.start()
        await asyncio.sleep(seconds)
        stats = reader.stats()
        reader.stop()
        await asyncio.sleep(0.05)
        return reader, stats

    reader, stats = asyncio.run(main())
    return reader, stats, events


def test_a_clip_is_replayed_in_real_time_at_the_target_rate():
    reader, stats, events = drive("clip.mp4", lambda s: FakeCapture(), 1.5, fps=20, width=160, height=90, retry_s=0.05)
    assert 20 <= len(events) <= 36  # about 20 per second, not the file's full speed
    assert [e.seq for e in events] == list(range(1, len(events) + 1))
    assert (events[0].camera_id, events[0].width, events[0].height) == ("cam-table", 160, 90)
    assert events[0].motion_score == 0.0 and events[1].motion_score > 0.9  # black -> white
    seq, ts_ms, frame = reader.latest()
    assert seq == events[-1].seq and frame.shape == (90, 160, 3) and ts_ms > 1_700_000_000_000
    assert stats["status"] == "ok" and 15 <= stats["fps_actual"] <= 25


def test_a_missing_camera_is_down_and_recovers_when_it_appears():
    opened = []

    def flaky(source):
        opened.append(source)
        cap = FakeCapture()
        cap.opened = len(opened) > 2  # the first two attempts fail
        return cap

    reader, stats, events = drive(0, flaky, 1.0, fps=20, width=160, height=90, retry_s=0.1)
    assert len(opened) >= 3 and events
    assert stats["status"] == "ok" and stats["dropped"] >= 2


def test_no_camera_at_all_reports_down():
    def never(source):
        cap = FakeCapture()
        cap.opened = False
        return cap

    reader, stats, events = drive(0, never, 0.4, fps=20, retry_s=0.05)
    assert events == [] and reader.latest() is None
    assert stats["status"] == "down" and stats["fps_actual"] == 0.0


def test_a_real_video_file_loops(tmp_path):
    import cv2

    clip = tmp_path / "clip.avi"
    out = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for i in range(5):
        out.write(np.full((48, 64, 3), 40 * i, np.uint8))
    out.release()

    reader, stats, events = drive(str(clip), None, 1.2, fps=20, width=64, height=48, retry_s=0.05)
    assert len(events) > 5  # the 5-frame clip came round again
    assert reader.latest()[2].shape == (48, 64, 3)


def test_a_device_that_is_not_plugged_in_is_not_opened_at_all():
    reader = CameraReader("cam-table", "/dev/video99", bus=None, loop=None)
    assert reader._open_capture("/dev/video99") is None  # no OpenCV warning on every retry
