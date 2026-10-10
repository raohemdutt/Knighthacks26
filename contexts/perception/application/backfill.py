"""Caption backfill (P8): stored keyframes the live path never got to.

Safe to run again and again: a frame counts as done once the log holds a
whole-frame `perception.scene.described` for it, and a frame captioned here
gets the same event id the live path would have given it.
"""

from collections.abc import Callable, Iterable

from bus import EventBus
from contexts.perception.application.captioner import describe
from contexts.perception.domain.scenes import frame_time
from contracts.perception import SceneDescribed
from contracts.types import FrameRef
from runtime.interface import Runtime


def captioned(bus: EventBus) -> set[FrameRef]:
    return {e.frame_ref for e in bus.read_log() if isinstance(e, SceneDescribed) and e.zone_id is None}


async def backfill(bus: EventBus, runtime: Runtime, frames: Iterable[tuple[str, FrameRef]],
                   load_keyframe: Callable[[FrameRef], bytes], limit: int | None = None) -> int:
    """Caption `frames` ((camera, ref), oldest first) that have no caption in the log. Returns how many were published."""
    done = captioned(bus)
    count = 0
    for camera_id, ref in frames:
        if limit is not None and count >= limit:
            break
        if ref in done or frame_time(ref) is None:
            continue
        try:
            jpeg = load_keyframe(ref)
        except OSError:  # pruned since the listing
            continue
        event = describe(runtime, jpeg, camera_id, ref, frame_time(ref))
        if event:
            await bus.publish(event)
            done.add(ref)
            count += 1
    return count
