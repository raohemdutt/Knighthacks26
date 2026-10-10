"""Small-object finder (P4): where the watch items are on the desk.

Every `every_s` while a desk camera has tracks in view, and whenever a person
leaves the desk zone, the desk's bounding box is cropped from the keyframe and
the model lists the watch items in it. An empty list is published too: it is
how consumers learn that the keys are no longer there. Live only, like captions.
"""

from collections.abc import Callable
from datetime import datetime, timedelta

from contexts.perception.application.keyframes import KeyframeGate
from contexts.perception.domain.scenes import DESK_ZONE, WATCH_ITEMS, bounding_box, objects_caption, scene
from contexts.perception.domain.zones import Polygon
from contexts.perception.infrastructure.frames import crop_jpeg
from contracts.perception import ObservationRecorded, SceneDescribed, ZoneEvent
from contracts.types import Box, FrameRef
from runtime.interface import Runtime

OBJECTS_EVERY_S = 30.0


class ObjectFinder:
    def __init__(self, runtime: Runtime, load_keyframe: Callable[[FrameRef], bytes], zones: dict[str, dict[str, Polygon]],
                 watch_items: list[str] | None = None, every_s: float = OBJECTS_EVERY_S, zone_id: str = DESK_ZONE,
                 crop: Callable[[bytes, Box], bytes] = crop_jpeg) -> None:
        self.runtime = runtime
        self.gate = KeyframeGate(load_keyframe)
        self.boxes = {camera: bounding_box(z[zone_id]) for camera, z in zones.items() if zone_id in z}
        self.watch_items = list(watch_items or WATCH_ITEMS)
        self.every = timedelta(seconds=every_s)
        self.zone_id = zone_id
        self.crop = crop

    async def on_observation(self, e: ObservationRecorded) -> SceneDescribed | None:
        if e.camera_id not in self.boxes or not e.tracks or not self.gate.due(e.camera_id, e.ts, self.every):
            return None
        return await self._find(e.camera_id, e.frame_ref, e.ts)

    async def on_zone_event(self, e: ZoneEvent) -> SceneDescribed | None:
        """Someone got up from the desk: look now for what they left behind, whatever the cadence says."""
        if e.kind != "exit" or e.label != "person" or e.zone_id != self.zone_id or e.camera_id not in self.boxes:
            return None
        return await self._find(e.camera_id, e.frame_ref, e.ts)

    async def _find(self, camera_id: str, ref: FrameRef | None, ts: datetime) -> SceneDescribed | None:
        def look(jpeg: bytes) -> SceneDescribed:
            hits = self.runtime.find_objects(self.crop(jpeg, self.boxes[camera_id]), items=self.watch_items)
            return scene(camera_id, ref, ts, objects_caption(self.zone_id, hits), self.runtime.model_info().version,
                         zone_id=self.zone_id, objects=hits)

        return await self.gate.run(camera_id, ref, ts, look)
