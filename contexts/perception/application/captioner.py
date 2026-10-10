"""Scene captions (P5): keyframe in, `perception.scene.described` out.

A camera is captioned when something enters or leaves one of its zones, and
otherwise every `every_s` while it has tracks in view. Live only: the handlers
call the model, so they are wired through `bus.in_background` without replay.
"""

from collections.abc import Callable
from datetime import datetime, timedelta

from contexts.perception.application.keyframes import KeyframeGate
from contexts.perception.domain.scenes import scene
from contracts.perception import ObservationRecorded, SceneDescribed, ZoneEvent
from contracts.types import FrameRef
from runtime.interface import Runtime

CAPTION_EVERY_S = 15.0
MIN_GAP_S = 2.0  # several tracks crossing a zone edge together are one scene, not one caption each


def describe(runtime: Runtime, jpeg: bytes, camera_id: str, ref: FrameRef, seen_at: datetime) -> SceneDescribed | None:
    """Caption one frame. None when the model had nothing to say."""
    caption = runtime.caption(jpeg).strip()
    return scene(camera_id, ref, seen_at, caption, runtime.model_info().version) if caption else None


class Captioner:
    def __init__(self, runtime: Runtime, load_keyframe: Callable[[FrameRef], bytes], every_s: float = CAPTION_EVERY_S,
                 min_gap_s: float = MIN_GAP_S) -> None:
        self.runtime = runtime
        self.gate = KeyframeGate(load_keyframe)
        self.every = timedelta(seconds=every_s)
        self.min_gap = timedelta(seconds=min_gap_s)

    async def on_zone_event(self, e: ZoneEvent) -> SceneDescribed | None:
        if e.kind == "dwell":
            return None
        return await self._caption(e.camera_id, e.frame_ref, e.ts, self.min_gap)

    async def on_observation(self, e: ObservationRecorded) -> SceneDescribed | None:
        if not e.tracks:  # nothing in view: nothing to say
            return None
        return await self._caption(e.camera_id, e.frame_ref, e.ts, self.every)

    async def _caption(self, camera_id: str, ref: FrameRef | None, ts: datetime, gap: timedelta) -> SceneDescribed | None:
        if not self.gate.due(camera_id, ts, gap):
            return None
        return await self.gate.run(camera_id, ref, ts, lambda jpeg: describe(self.runtime, jpeg, camera_id, ref, ts))
