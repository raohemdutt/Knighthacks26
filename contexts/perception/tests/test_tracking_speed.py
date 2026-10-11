"""TrackingPipeline times its detector, so GET /capture/status and /runtime/stats can say how fast it is (R5)."""

import asyncio
import time

from contexts.perception.application.tracking import TrackingPipeline


class Reader:
    def __init__(self):
        self.seq = 0

    def latest(self):
        self.seq += 1
        return self.seq, self.seq * 100, b"frame"


class Bus:
    async def publish(self, event):
        return None


def slow_detector(frame):
    time.sleep(0.01)
    return []


def test_the_detector_speed_is_the_mean_of_its_own_calls():
    loop = asyncio.new_event_loop()
    p = TrackingPipeline("cam-desk", Reader(), slow_detector, Bus(), loop, detect_fps=200, observe_every_s=60)
    assert p.detector_speed() is None  # nothing timed yet
    p.start()
    deadline = time.time() + 3
    while p.detected < 5 and time.time() < deadline:
        time.sleep(0.01)
    p.stop()
    loop.close()
    speed = p.detector_speed()
    assert speed["samples"] == p.detected >= 5
    assert speed["ms"] >= 9.5 and speed["fps"] <= 105  # a 10 ms detector can do at most about 100 frames a second
    assert speed["fps"] == round(1000 / speed["ms"], 1) or abs(speed["fps"] - 1000 / speed["ms"]) < 0.2
    assert speed["ts"]
