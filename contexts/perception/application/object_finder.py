"""Small-object finder (P4): where the watch items are on the desk.

Every `every_s` while a desk camera has tracks in view, and whenever a person
leaves the desk zone, the desk's bounding box is cropped from the keyframe and
the model lists the watch items in it. An empty list is published too: it is
how consumers learn that the keys are no longer there. Live only, like captions.

The crop is taken first, at full resolution, and then scaled to `max_side` pixels on its longest side
(0 = unscaled). Small things need more pixels than a caption does, so the default is larger. The calls
go through the same queue as the captions (keyframes.py): one model call at a time between the two.
"""

from collections.abc import Callable
from datetime import datetime, timedelta

from contexts.perception.application.keyframes import ModelQueue, PerceptionStats, Request
from contexts.perception.domain.scenes import DESK_ZONE, WATCH_ITEMS, bounding_box, objects_caption, scene
from contexts.perception.domain.zones import Polygon
from contexts.perception.infrastructure.frames import crop_jpeg, scale_jpeg
from contracts.perception import ObservationRecorded, SceneDescribed, ZoneEvent
from contracts.types import Box, FrameRef
from runtime.interface import Runtime

OBJECTS_EVERY_S = 30.0
OBJECTS_MAX_SIDE = 960  # px, of the desk crop


class ObjectFinder:
    def __init__(self, runtime: Runtime, load_keyframe: Callable[[FrameRef], bytes], zones: dict[str, dict[str, Polygon]],
                 watch_items: list[str] | None = None, every_s: float = OBJECTS_EVERY_S, zone_id: str = DESK_ZONE,
                 crop: Callable[[bytes, Box], bytes] = crop_jpeg, max_side: int = OBJECTS_MAX_SIDE,
                 queue: ModelQueue | None = None, stats: PerceptionStats | None = None,
                 scale: Callable[[bytes, int], bytes] = scale_jpeg) -> None:
        self.runtime = runtime
        self.queue = queue or ModelQueue(load_keyframe, min_gap_s=0)  # on its own: unpaced, like a lone Captioner
        self.stats = stats or PerceptionStats()
        self.boxes = {camera: bounding_box(z[zone_id]) for camera, z in zones.items() if zone_id in z}
        self.watch_items = list(watch_items or WATCH_ITEMS)
        self.every = timedelta(seconds=every_s)
        self.zone_id = zone_id
        self.crop = crop
        self.max_side = max_side
        self.scale = scale
        self.last: dict[str, datetime] = {}  # camera -> event time of the last frame handed to the model

    async def on_observation(self, e: ObservationRecorded) -> list[SceneDescribed]:
        self.queue.tick(e.ts)
        if e.camera_id in self.boxes and e.tracks and e.frame_ref is not None:
            waiting = self.queue.pending(("objects", e.camera_id))
            last = self.last.get(e.camera_id)
            if waiting is not None:
                if not waiting.urgent:  # still waiting its turn: let it look at the newest frame
                    self.queue.put(self._request(e.camera_id, e.frame_ref, e.ts, urgent=False))
            elif last is None or e.ts - last >= self.every:
                self.stats.objects_requested += 1
                self.queue.put(self._request(e.camera_id, e.frame_ref, e.ts, urgent=False))
        return await self.queue.drain()

    async def on_zone_event(self, e: ZoneEvent) -> list[SceneDescribed]:
        """Someone got up from the desk: look now for what they left behind, whatever the cadence says."""
        self.queue.tick(e.ts)
        if e.kind == "exit" and e.label == "person" and e.zone_id == self.zone_id and e.camera_id in self.boxes:
            self.stats.objects_requested += 1
            if e.frame_ref is None:
                self.stats.objects_without_frame += 1
            elif self.queue.put(self._request(e.camera_id, e.frame_ref, e.ts, urgent=True)) is not None:
                self.stats.objects_coalesced += 1  # one look for both triggers, at the newer frame
        return await self.queue.drain()

    def _request(self, camera_id: str, ref: FrameRef, ts: datetime, urgent: bool) -> Request:
        def look(jpeg: bytes) -> SceneDescribed:
            desk = self.crop(jpeg, self.boxes[camera_id])
            small = self.scale(desk, self.max_side)
            self.stats.image(len(desk), len(small))
            hits = self.runtime.find_objects(small, items=self.watch_items)
            return scene(camera_id, ref, ts, objects_caption(self.zone_id, hits), self.runtime.model_info().version,
                         zone_id=self.zone_id, objects=hits)

        return Request(("objects", camera_id), ref, ts, look, urgent=urgent, started=self._started, missing=self._missing)

    def _started(self, request: Request) -> None:
        camera_id = request.key[1]
        self.last[camera_id] = max(request.ts, self.last.get(camera_id, request.ts))
        self.stats.objects_sent += 1

    def _missing(self, request: Request) -> None:
        self.stats.objects_without_frame += 1
