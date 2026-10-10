"""Scene captions (P5): keyframe in, `perception.scene.described` out.

A camera is captioned when something enters or leaves one of its zones, and otherwise on a cadence
while it has tracks in view. Live only: the handlers call the model, so they are wired through
`bus.in_background` without replay.

The caption budget, all in event time:

- An enter or exit is captioned at once. Several within `coalesce_s` on one camera are one scene, not
  one caption each: the later ones are dropped, or replace the one still waiting in the queue.
- The periodic caption comes `every_s` after the last caption while the scene keeps changing. While
  nothing changes (the same track ids and labels, no enter or exit) each periodic caption doubles the
  wait for the next one, up to `idle_max_s`: 15 s, 30 s, 60 s, 60 s, ... Any change puts it back to `every_s`.
- The model gets a copy scaled to `max_side` pixels on its longest side (0 = the keyframe as stored).
  The keyframe on disk is not touched.

Handlers return the scene events of whatever model calls they started, which may be for an earlier
trigger that was waiting in the queue (see keyframes.py).
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from contexts.perception.application.keyframes import ModelQueue, PerceptionStats, Request
from contexts.perception.domain.scenes import scene
from contexts.perception.infrastructure.frames import scale_jpeg
from contracts.perception import ObservationRecorded, SceneDescribed, ZoneEvent
from contracts.types import FrameRef
from runtime.interface import Runtime

CAPTION_EVERY_S = 15.0
CAPTION_IDLE_MAX_S = 60.0
CAPTION_MAX_SIDE = 640  # px. 640 is about a quarter of the image tokens of a 1280 px keyframe
COALESCE_S = 2.0  # several tracks crossing a zone edge together are one scene, not one caption each


def describe(runtime: Runtime, jpeg: bytes, camera_id: str, ref: FrameRef, seen_at: datetime) -> SceneDescribed | None:
    """Caption one frame. None when the model had nothing to say."""
    caption = runtime.caption(jpeg).strip()
    return scene(camera_id, ref, seen_at, caption, runtime.model_info().version) if caption else None


@dataclass
class _Camera:
    last: datetime | None = None  # event time of the last frame handed to the model
    ref: FrameRef | None = None  # that frame
    slot: datetime | None = None  # the last caption or skipped periodic slot, for counting skips once each
    tracks: frozenset = frozenset()  # (track id, label) in the newest observation
    version: int = 0  # goes up whenever the scene changes
    captioned: int = -1  # `version` when the last caption was requested
    idle: int = 0  # periodic captions in a row that found nothing changed


class Captioner:
    def __init__(self, runtime: Runtime, load_keyframe: Callable[[FrameRef], bytes], every_s: float = CAPTION_EVERY_S,
                 coalesce_s: float = COALESCE_S, idle_max_s: float = CAPTION_IDLE_MAX_S, max_side: int = CAPTION_MAX_SIDE,
                 queue: ModelQueue | None = None, stats: PerceptionStats | None = None,
                 scale: Callable[[bytes, int], bytes] = scale_jpeg) -> None:
        self.runtime = runtime
        # On its own (tests, tools) a captioner is not paced against anything else: the service shares one paced queue.
        self.queue = queue or ModelQueue(load_keyframe, min_gap_s=0)
        self.stats = stats or PerceptionStats()
        self.every = timedelta(seconds=every_s)
        self.idle_max = timedelta(seconds=max(idle_max_s, every_s))
        self.coalesce = timedelta(seconds=coalesce_s)
        self.max_side = max_side
        self.scale = scale
        self.cameras: dict[str, _Camera] = {}

    async def on_zone_event(self, e: ZoneEvent) -> list[SceneDescribed]:
        self.queue.tick(e.ts)
        if e.kind != "dwell":
            cam = self._camera(e.camera_id)
            cam.version += 1
            self.stats.captions_requested += 1
            if e.frame_ref is None:
                self.stats.captions_without_frame += 1
            elif self._covered(e.camera_id, e.frame_ref, e.ts, cam):
                self.stats.captions_coalesced += 1
            else:
                old = self.queue.put(self._request(e.camera_id, e.frame_ref, e.ts, cam, urgent=True))
                if old is not None:  # one caption for both triggers, of the newer frame
                    self.stats.captions_coalesced += 1
        return await self.queue.drain()

    async def on_observation(self, e: ObservationRecorded) -> list[SceneDescribed]:
        self.queue.tick(e.ts)
        cam = self._camera(e.camera_id)
        tracks = frozenset((t.track_id, t.label) for t in e.tracks)
        if tracks != cam.tracks:
            cam.tracks = tracks
            cam.version += 1
        if e.tracks and e.frame_ref is not None:  # nothing in view: nothing to say
            self._periodic(e.camera_id, e.frame_ref, e.ts, cam)
        return await self.queue.drain()

    def _covered(self, camera_id: str, ref: FrameRef, ts: datetime, cam: _Camera) -> bool:
        """The caption just sent, or the one being written now, covers this moment. An exit shows the last frame the
        thing was in, which may be one already captioned: the same frame gives the same event, which the bus drops."""
        sent = [(cam.last, cam.ref)] if cam.last is not None else []
        running = self.queue.running
        if running is not None and running.key == ("caption", camera_id):
            sent.append((running.ts, running.ref))
        return any(ref == was or ts - at < self.coalesce for at, was in sent)

    def period(self, camera_id: str) -> timedelta:
        """How long after its last caption this camera gets a periodic one."""
        cam = self._camera(camera_id)
        if cam.version != cam.captioned:
            return self.every
        period = self.every
        for _ in range(cam.idle):
            if period >= self.idle_max:
                break
            period *= 2
        return min(period, self.idle_max)

    def _periodic(self, camera_id: str, ref: FrameRef, ts: datetime, cam: _Camera) -> None:
        waiting = self.queue.pending(("caption", camera_id))
        if waiting is not None:
            if not waiting.urgent:  # still waiting its turn: let it describe the newest frame
                self.queue.put(self._request(camera_id, ref, ts, cam, urgent=False))
            return
        if cam.last is not None and ts - cam.last < self.period(camera_id):
            if ts - cam.last >= self.every and ts - (cam.slot or cam.last) >= self.every:
                cam.slot = ts  # the fixed cadence would have captioned here
                self.stats.captions_requested += 1
                self.stats.captions_skipped_redundant += 1
            return
        self.stats.captions_requested += 1
        self.queue.put(self._request(camera_id, ref, ts, cam, urgent=False))

    def _request(self, camera_id: str, ref: FrameRef, ts: datetime, cam: _Camera, urgent: bool) -> Request:
        def work(jpeg: bytes) -> SceneDescribed | None:
            small = self.scale(jpeg, self.max_side)
            self.stats.image(len(jpeg), len(small))
            return describe(self.runtime, small, camera_id, ref, ts)

        return Request(("caption", camera_id), ref, ts, work, urgent=urgent, started=self._started, missing=self._missing,
                       tag=cam.version)

    def _started(self, request: Request) -> None:
        cam = self._camera(request.key[1])
        unchanged = not request.urgent and request.tag == cam.captioned
        cam.idle = cam.idle + 1 if unchanged else 0
        cam.captioned = request.tag
        cam.last = request.ts if cam.last is None else max(cam.last, request.ts)
        cam.slot, cam.ref = cam.last, request.ref
        self.stats.captions_sent += 1

    def _missing(self, request: Request) -> None:
        self.stats.captions_without_frame += 1

    def _camera(self, camera_id: str) -> _Camera:
        return self.cameras.setdefault(camera_id, _Camera())
