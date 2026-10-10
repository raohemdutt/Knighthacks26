"""Perception end to end (P7): observations in, zone events and scene events out.

    perception.observation.recorded -> ZoneEngine    -> perception.zone.event
    observation / zone event        -> Captioner     -> perception.scene.described
    observation / desk exit         -> ObjectFinder  -> perception.scene.described (zone_id="desk", objects)

Only the zone engine's stays rebuild from the log. Captions and object lists
call the model, so they run on live events only and in the background; what
they said the first time replays from the log as events.

The model is shared with the people asking questions, so perception keeps its use of it small.
Settings (the optional `perception:` block in config/agent.yaml), with their defaults:

    watch_items: [keys, wallet, ...]  what the object finder looks for on the desk
    caption_every_s: 15         periodic caption per camera while the scene keeps changing
    caption_idle_max_s: 60      while nothing changes the period doubles up to this (15, 30, 60, 60, ...)
    caption_coalesce_s: 2       enters and exits this close together on one camera are one caption
    caption_max_side: 640       px, longest side of the copy sent for a caption; 0 = as stored
    objects_every_s: 30         periodic desk object list
    objects_max_side: 960       px, longest side of the desk crop sent to the model; 0 = as cropped
    min_gap_s: 2                event time between two perception model calls, whatever the camera
    yield_after_question_s: 3   no new call while a question is being answered, nor this long after
    yield_timeout_s: 30         an unanswered question stops holding perception back after this; 0 = never yield

Expected effect: a 640 px image is about a quarter of the image tokens of a 1280 px keyframe (about 940
tokens and 3.5 s of prompt processing each on the Jetson before), and a still room is captioned once a
minute instead of four times. `stats()` counts what was requested, sent, skipped and coalesced, and the
bytes per image before and after scaling.
The old behaviour is `caption_max_side: 0`, `objects_max_side: 0`, `caption_idle_max_s` equal to
`caption_every_s`, `min_gap_s: 0`, `yield_timeout_s: 0`.
"""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from bus import EventBus
from contexts.perception.application.captioner import (CAPTION_EVERY_S, CAPTION_IDLE_MAX_S, CAPTION_MAX_SIDE, COALESCE_S,
                                                       Captioner)
from contexts.perception.application.keyframes import (MIN_GAP_S, YIELD_AFTER_QUESTION_S, YIELD_TIMEOUT_S, ModelQueue,
                                                       PerceptionStats)
from contexts.perception.application.object_finder import OBJECTS_EVERY_S, OBJECTS_MAX_SIDE, ObjectFinder
from contexts.perception.domain.zones import Polygon, ZoneEngine, load_zones
from contracts.perception import ZoneEvent
from contracts.types import FrameRef
from runtime.interface import Runtime


class PerceptionService:
    def __init__(self, zones: dict[str, dict[str, Polygon]], runtime: Runtime, load_keyframe: Callable[[FrameRef], bytes],
                 watch_items: list[str] | None = None, caption_every_s: float = CAPTION_EVERY_S,
                 objects_every_s: float = OBJECTS_EVERY_S, caption_idle_max_s: float = CAPTION_IDLE_MAX_S,
                 caption_coalesce_s: float = COALESCE_S, caption_max_side: int = CAPTION_MAX_SIDE,
                 objects_max_side: int = OBJECTS_MAX_SIDE, min_gap_s: float = MIN_GAP_S,
                 yield_after_question_s: float = YIELD_AFTER_QUESTION_S, yield_timeout_s: float = YIELD_TIMEOUT_S) -> None:
        self.engine = ZoneEngine(zones)
        self.counters = PerceptionStats()
        self.queue = ModelQueue(load_keyframe, min_gap_s=min_gap_s, yield_after_question_s=yield_after_question_s,
                                yield_timeout_s=yield_timeout_s)  # captions and object lists take turns
        self.captioner = Captioner(runtime, load_keyframe, every_s=caption_every_s, coalesce_s=caption_coalesce_s,
                                   idle_max_s=caption_idle_max_s, max_side=caption_max_side, queue=self.queue,
                                   stats=self.counters)
        self.finder = ObjectFinder(runtime, load_keyframe, zones, watch_items=watch_items, every_s=objects_every_s,
                                   max_side=objects_max_side, queue=self.queue, stats=self.counters)

    @classmethod
    def load(cls, zones_file: str | Path, runtime: Runtime, load_keyframe: Callable[[FrameRef], bytes],
             **settings) -> "PerceptionService":
        """`settings`: the keys in the module docstring (e.g. a `perception:` block in agent.yaml)."""
        return cls(load_zones(zones_file), runtime, load_keyframe, **settings)

    def subscribe(self, bus: EventBus) -> None:
        bus.subscribe("perception.observation", self.engine.on_observation, replay=True)  # stays rebuild from the log
        # Live only: model calls, and who they give way to.
        bus.subscribe("experience.question.asked", self.queue.on_question)
        bus.subscribe("agent.answer.generated", self.queue.on_answer)
        bus.subscribe("perception.observation", bus.in_background(self.captioner.on_observation))
        bus.subscribe("perception.observation", bus.in_background(self.finder.on_observation))
        bus.subscribe("perception.zone", bus.in_background(self.captioner.on_zone_event))
        bus.subscribe("perception.zone", bus.in_background(self.finder.on_zone_event))

    def stats(self) -> dict[str, float]:
        """Model use since start: captions requested / sent / skipped as redundant / coalesced, and bytes per image."""
        return {**self.counters.as_dict(), "model_errors": self.queue.errors, "queued": len(self.queue.queued)}

    def flush(self, now: datetime) -> list[ZoneEvent]:
        """Exit tracks that vanished. The app's ticker publishes what this returns."""
        return self.engine.flush(now)
