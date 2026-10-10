"""C1: the camera reader. One thread per camera reads frames with OpenCV and, at the target rate,
keeps the newest one in memory and publishes `capture.frames.captured` for it.

    reader = CameraReader("cam-table", "/dev/video0", bus=bus, loop=loop, fps=15)
    reader.start()
    seq, ts_ms, frame = reader.latest()      # BGR array; frame_ref(camera_id, ts_ms) names it on disk (C2)

The event carries no pixels (it is in-process only and not logged); whoever needs the picture asks
`latest()`. With `motion_threshold` set, `capture.frames.captured` is only published for frames that moved at least that much
(motion score 0..1) plus one every `keepalive_s` (C3); `latest()` and `stats()` still see every frame.
`source` is a device index or path (USB webcam, V4L2 on the Jetson) or a video file, which
is replayed in real time and loops, so everything downstream can be tested without a camera.
If the camera drops off, the reader keeps retrying; `stats()` says whether it is ok, degraded or down.
"""

import asyncio
import os
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from contracts.capture import FramesCaptured
from contracts.types import CameraId

_MOTION_SIZE = (64, 36)


class CameraReader:
    def __init__(self, camera_id: CameraId, source: int | str, *, bus: Any, loop: asyncio.AbstractEventLoop,
                 fps: float = 15, width: int | None = 1280, height: int | None = 720,
                 open_capture: Callable[[int | str], Any] | None = None, retry_s: float = 2.0,
                 motion_threshold: float | None = None, keepalive_s: float = 5.0) -> None:
        self.camera_id, self.source = camera_id, source
        self.fps, self.size = fps, (width, height)
        self._bus, self._loop = bus, loop
        self._open = open_capture or self._open_capture
        self._retry_s = retry_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest: tuple[int, int, Any] | None = None
        self._times: deque[float] = deque(maxlen=int(fps * 5) + 1)
        self._failed_reads = 0
        self._last_ok = 0.0
        self._seq = 0
        self.motion_threshold, self.keepalive_s = motion_threshold, keepalive_s  # None: an event for every frame
        self._last_event = float("-inf")

    # --- public ---------------------------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=f"camera-{self.camera_id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def latest(self) -> tuple[int, int, Any] | None:
        """(seq, ts_ms, BGR frame) of the newest frame, or None before the first one."""
        with self._lock:
            return self._latest

    def stats(self) -> dict:
        with self._lock:
            times = list(self._times)
            failed = self._failed_reads
        now = time.monotonic()
        fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else 0.0
        if now - self._last_ok > max(3.0, 3 / self.fps) or not times:
            status = "down"
        elif fps < 0.7 * self.fps:
            status = "degraded"
        else:
            status = "ok"
        return {"status": status, "fps_actual": round(fps, 2), "dropped": failed}

    # --- the thread -----------------------------------------------------------------------------

    def _open_capture(self, source: int | str) -> Any:
        import cv2

        if isinstance(source, str) and source.startswith("/dev/") and not os.path.exists(source):
            return None  # nothing plugged in: do not let OpenCV print a warning on every retry
        linux_device = sys.platform.startswith("linux") and (isinstance(source, int) or str(source).startswith("/dev/video"))
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2) if linux_device else cv2.VideoCapture(source)
        if linux_device or isinstance(source, int):  # a live camera: ask for MJPG, which USB webcams do at full rate
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            if self.size[0]:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _run(self) -> None:
        is_file = isinstance(self.source, str) and not self.source.startswith("/dev/")
        period, due, prev_small = 1.0 / self.fps, 0.0, None
        cap = None
        while not self._stop.is_set():
            if cap is None or not cap.isOpened():
                cap = self._open(self.source)
                if cap is None or not cap.isOpened():
                    self._failed_reads += 1
                    self._stop.wait(self._retry_s)
                    continue
            now = time.monotonic()
            if is_file and now < due:  # a file would otherwise play at full speed
                self._stop.wait(due - now)
                continue
            if not is_file and now < due:
                if not cap.grab():  # drop the frames we do not need without decoding them
                    self._failed_reads += 1
                    cap.release()
                    cap = None
                continue
            ok, frame = cap.read()
            if not ok:
                if is_file:
                    cap.set(1, 0)  # cv2.CAP_PROP_POS_FRAMES: loop the clip
                    continue
                self._failed_reads += 1
                cap.release()
                cap = None
                self._stop.wait(self._retry_s)
                continue
            due = max(due + period, now - period) if due else now + period
            frame = self._fit(frame)
            prev_small = self._publish(frame, prev_small)
        if cap is not None:
            cap.release()

    def _fit(self, frame: Any) -> Any:
        import cv2

        w, h = self.size
        if w and (frame.shape[1], frame.shape[0]) != (w, h):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        return frame

    def _publish(self, frame: Any, prev_small: Any) -> Any:
        import cv2

        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), _MOTION_SIZE, interpolation=cv2.INTER_AREA)
        motion = float(cv2.absdiff(small, prev_small).mean() / 255) if prev_small is not None else 0.0
        self._seq += 1
        ts_ms, mono = int(time.time() * 1000), time.monotonic()
        with self._lock:
            self._latest = (self._seq, ts_ms, frame)
            self._times.append(mono)
        self._last_ok = mono
        if self.motion_threshold is not None and motion < self.motion_threshold and mono - self._last_event < self.keepalive_s:
            return small  # a still scene: the frame is kept (latest()) but no event, except one every keepalive_s
        self._last_event = mono
        event = FramesCaptured(camera_id=self.camera_id, seq=self._seq, width=frame.shape[1], height=frame.shape[0],
                               motion_score=round(motion, 4))
        asyncio.run_coroutine_threadsafe(self._bus.publish(event), self._loop)
        return small
