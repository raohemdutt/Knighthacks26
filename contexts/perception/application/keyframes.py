"""One model call per camera at a time, on a cadence in event time.

Model calls take seconds on the Jetson. While one is running for a camera, new
triggers for that camera are dropped instead of queued, so a slow model falls
behind by skipping frames, never by building a backlog.
"""

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import TypeVar

from contracts.types import FrameRef

log = logging.getLogger("atlas.perception")
T = TypeVar("T")


class KeyframeGate:
    def __init__(self, load_keyframe: Callable[[FrameRef], bytes]) -> None:
        self.load_keyframe = load_keyframe
        self.last: dict[str, datetime] = {}  # camera -> event time of the last frame handed to the model
        self.busy: set[str] = set()

    def due(self, camera_id: str, ts: datetime, gap: timedelta) -> bool:
        last = self.last.get(camera_id)
        return camera_id not in self.busy and (last is None or ts - last >= gap)

    async def run(self, camera_id: str, ref: FrameRef | None, ts: datetime, work: Callable[[bytes], T]) -> T | None:
        """`work(jpeg)` for one keyframe, off the event loop. None when the camera is busy or the frame is missing."""
        if ref is None or camera_id in self.busy:
            return None
        self.busy.add(camera_id)
        try:
            jpeg = await asyncio.to_thread(self._load, ref)
            if jpeg is None:
                return None  # the cadence is not used up: the next frame gets its turn
            self.last[camera_id] = max(ts, self.last.get(camera_id, ts))
            return await asyncio.to_thread(work, jpeg)
        finally:
            self.busy.discard(camera_id)

    def _load(self, ref: FrameRef) -> bytes | None:
        try:
            return self.load_keyframe(ref)
        except Exception as err:  # pruned, never written, or a ref from a fixture
            log.debug("no keyframe for %s: %s", ref, err)
            return None
