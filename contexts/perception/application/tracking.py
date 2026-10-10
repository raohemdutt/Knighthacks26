"""P2: frames in, tracked objects out. One pipeline per camera.

    camera reader (C1) -> detector with ByteTrack (P1) -> perception.observation.recorded

The tracker needs frames close together to keep ids stable, so the detector runs at `detect_fps`
(default 10) on the newest frame the reader has. Once per `observe_every_s` the latest tracks are
published as an observation (and, if `faces` is given, the face counts as `perception.faces.counted`: when they change
and every `faces_every_s`). A keyframe is stored with it (and named in `frame_ref`) when the set of
track ids changed, every `keyframe_every_s` while something is in view, and every `keepalive_s`
regardless, so a still room does not fill the disk. The captioner and object finder read these keyframes.

Each camera needs its own `YoloDetector(track=True)`: ByteTrack's state lives in the model.
"""

import asyncio
import threading
import time
from collections.abc import Callable
from typing import Any

from contracts.perception import ObservationRecorded
from contracts.play import FacesCounted
from contracts.types import CameraId, FrameRef, Track


class TrackingPipeline:
    def __init__(self, camera_id: CameraId, reader: Any, detector: Callable[[Any], list[Track]], bus: Any,
                 loop: asyncio.AbstractEventLoop, *, store: Callable[[CameraId, int, Any], FrameRef] | None = None,
                 detect_fps: float = 10, observe_every_s: float = 1.0, keyframe_every_s: float = 5.0,
                 keepalive_s: float = 30.0, paused: Callable[[], bool] | None = None,
                 faces: Callable[[Any], tuple[int, int]] | None = None, faces_every_s: float = 5.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.camera_id, self.reader, self.detector, self.store = camera_id, reader, detector, store
        self._bus, self._loop, self._clock = bus, loop, clock
        self._paused = paused or (lambda: False)  # a camera the Vault policy has paused is not looked at
        self.detect_period, self.observe_every_s = 1.0 / detect_fps, observe_every_s
        self.keyframe_every_s, self.keepalive_s = keyframe_every_s, keepalive_s
        self.faces, self.faces_every_s = faces, faces_every_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.detected = 0
        self.errors = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=f"tracking-{self.camera_id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        last_seq, tracks, key_ids = 0, [], None
        next_detect = next_observe = 0.0
        last_key = float("-inf")
        faces_seen: tuple[int, int] | None = None
        last_faces = float("-inf")
        item = None
        while not self._stop.is_set():
            now = self._clock()
            item = self.reader.latest()
            if item is None or item[0] == last_seq or now < next_detect or self._paused():
                self._stop.wait(0.005)
                continue
            seq, ts_ms, frame = item
            last_seq, next_detect = seq, now + self.detect_period
            try:
                tracks = self.detector(frame)
            except Exception:  # one bad frame must not stop the camera
                self.errors += 1
                continue
            self.detected += 1
            if now < next_observe:
                continue
            next_observe = now + self.observe_every_s
            ids = frozenset(t.track_id for t in tracks)
            ref = None
            if self.store is not None and (ids != key_ids or (tracks and now - last_key >= self.keyframe_every_s)
                                           or now - last_key >= self.keepalive_s):
                try:
                    ref = self.store(self.camera_id, ts_ms, frame)
                    key_ids, last_key = ids, now
                except Exception:
                    self.errors += 1
            event = ObservationRecorded(camera_id=self.camera_id, tracks=tracks, frame_ref=ref)
            asyncio.run_coroutine_threadsafe(self._bus.publish(event), self._loop)
            if self.faces is not None:
                try:
                    seen = self.faces(frame)
                except Exception:
                    self.errors += 1
                    continue
                if seen != faces_seen or now - last_faces >= self.faces_every_s:  # when it changes, and as a heartbeat
                    faces_seen, last_faces = seen, now
                    counted = FacesCounted(camera_id=self.camera_id, count=seen[0], facing=seen[1])
                    asyncio.run_coroutine_threadsafe(self._bus.publish(counted), self._loop)
