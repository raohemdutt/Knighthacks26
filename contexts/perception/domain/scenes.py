"""Scene events (P5, P4): what the model said about one keyframe.

The event id comes from the camera, the zone and the frame, and the time is
the frame's own, so describing a frame twice (live, then a backfill) gives the
same event and the bus drops the second.
"""

from datetime import UTC, datetime
from uuid import NAMESPACE_URL, uuid5

from contracts.perception import SceneDescribed
from contracts.types import Box, FrameRef, ObjectHit

DESK_ZONE = "desk"
WATCH_ITEMS = ("keys", "wallet", "charger", "laptop", "phone", "can", "bottle", "cup")


def frame_time(ref: FrameRef) -> datetime | None:
    """When `frames:<camera>:<ts_ms>` was taken; None for refs that carry no time (fixtures)."""
    ts_ms = ref.rsplit(":", 1)[-1]
    return datetime.fromtimestamp(int(ts_ms) / 1000, UTC) if ts_ms.isdigit() else None


def bounding_box(polygon: list[tuple[float, float]]) -> Box:
    xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
    return Box(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys))


def objects_caption(zone_id: str, hits: list[ObjectHit]) -> str:
    """"On the desk: keys (left of the laptop), phone." Built from the hits: no second model call."""
    if not hits:
        return f"No watch items on the {zone_id}."
    return f"On the {zone_id}: " + ", ".join(f"{h.label} ({h.position})" if h.position else h.label for h in hits) + "."


def scene(camera_id: str, ref: FrameRef, seen_at: datetime, caption: str, model_version: str,
          zone_id: str | None = None, objects: list[ObjectHit] = ()) -> SceneDescribed:
    """`seen_at` is used only when the frame ref carries no time of its own."""
    event_id = uuid5(NAMESPACE_URL, f"atlas:scene:{camera_id}:{zone_id or ''}:{ref}").hex
    return SceneDescribed(event_id=event_id, ts=frame_time(ref) or seen_at, camera_id=camera_id, caption=caption,
                          objects=list(objects), zone_id=zone_id, frame_ref=ref, model_version=model_version)
