"""Perception end to end (P7): observations in, zone events and scene events out.

    perception.observation.recorded -> ZoneEngine    -> perception.zone.event
    observation / zone event        -> Captioner     -> perception.scene.described
    observation / desk exit         -> ObjectFinder  -> perception.scene.described (zone_id="desk", objects)

Only the zone engine's stays rebuild from the log. Captions and object lists
call the model, so they run on live events only and in the background; what
they said the first time replays from the log as events.
"""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from bus import EventBus
from contexts.perception.application.captioner import CAPTION_EVERY_S, Captioner
from contexts.perception.application.object_finder import OBJECTS_EVERY_S, ObjectFinder
from contexts.perception.domain.zones import Polygon, ZoneEngine, load_zones
from contracts.perception import ZoneEvent
from contracts.types import FrameRef
from runtime.interface import Runtime


class PerceptionService:
    def __init__(self, zones: dict[str, dict[str, Polygon]], runtime: Runtime, load_keyframe: Callable[[FrameRef], bytes],
                 watch_items: list[str] | None = None, caption_every_s: float = CAPTION_EVERY_S,
                 objects_every_s: float = OBJECTS_EVERY_S) -> None:
        self.engine = ZoneEngine(zones)
        self.captioner = Captioner(runtime, load_keyframe, every_s=caption_every_s)
        self.finder = ObjectFinder(runtime, load_keyframe, zones, watch_items=watch_items, every_s=objects_every_s)

    @classmethod
    def load(cls, zones_file: str | Path, runtime: Runtime, load_keyframe: Callable[[FrameRef], bytes],
             **settings) -> "PerceptionService":
        """`settings`: watch_items, caption_every_s, objects_every_s (e.g. a `perception:` block in agent.yaml)."""
        return cls(load_zones(zones_file), runtime, load_keyframe, **settings)

    def subscribe(self, bus: EventBus) -> None:
        bus.subscribe("perception.observation", self.engine.on_observation, replay=True)  # stays rebuild from the log
        # Live only: model calls.
        bus.subscribe("perception.observation", bus.in_background(self.captioner.on_observation))
        bus.subscribe("perception.observation", bus.in_background(self.finder.on_observation))
        bus.subscribe("perception.zone", bus.in_background(self.captioner.on_zone_event))
        bus.subscribe("perception.zone", bus.in_background(self.finder.on_zone_event))

    def flush(self, now: datetime) -> list[ZoneEvent]:
        """Exit tracks that vanished. The app's ticker publishes what this returns."""
        return self.engine.flush(now)
