"""A scripted half minute of `perception.observation.recorded`, one per camera per second.

    0 s   a person walks in through the door zone (cam-door) ...
    5 s   ... and sits at the desk (cam-desk)
    8 s   keys are left on the desk
    10 s  a package is put down in the door zone and stays
    20 s  a can is carried across the room and dropped in the recycling zone

Deterministic: the same start time gives the same observations. Boxes match
config/zones.yaml. Track ids are stable per thing, like a tracker's would be.
"""

import asyncio
from datetime import datetime, timedelta

from bus import EventBus
from contracts.perception import ObservationRecorded
from contracts.types import Box, Track

Step = tuple[int, str, Track]  # second from the start, camera, what it sees
LENGTH_S = 30
CAMERAS = ("cam-door", "cam-desk")


def _track(track_id: int, label: str, x: float, bottom: float, w: float, h: float) -> Track:
    """A box `w` x `h` whose bottom-center is at (x, bottom)."""
    return Track(track_id=track_id, label=label, confidence=0.9,
                 box=Box(x1=round(x - w / 2, 3), y1=round(bottom - h, 3), x2=round(x + w / 2, 3), y2=bottom))


def person_arrives(at: int = 0, until: int = LENGTH_S) -> list[Step]:
    """In through the door zone, out of that camera's view, then seated at the desk."""
    walk = [(at + i, "cam-door", _track(1, "person", x, 0.9, 0.2, 0.6)) for i, x in enumerate((0.85, 0.75, 0.65, 0.51))]
    sit = [(s, "cam-desk", _track(1, "person", 0.5, 0.9, 0.3, 0.6)) for s in range(at + 5, until + 1)]
    return walk + sit


def keys_left(at: int = 8, until: int = LENGTH_S) -> list[Step]:
    return [(s, "cam-desk", _track(3, "keys", 0.3, 0.7, 0.06, 0.04)) for s in range(at, until + 1)]


def package_left(at: int = 10, until: int = LENGTH_S) -> list[Step]:
    return [(s, "cam-door", _track(2, "package", 0.8, 0.9, 0.16, 0.14)) for s in range(at, until + 1)]


def can_recycled(at: int = 20) -> list[Step]:
    """Carried at hand height (above the bin zones), lowered into recycling, then out of sight in the bin."""
    carried = [(at + i, "cam-door", _track(4, "can", x, 0.5, 0.04, 0.07)) for i, x in enumerate((0.8, 0.6, 0.4, 0.2))]
    dropped = [(at + 4 + i, "cam-door", _track(4, "can", 0.15, 0.8, 0.04, 0.07)) for i in range(2)]
    return carried + dropped


def observations(start: datetime, steps: list[Step], length_s: int | None = None) -> list[ObservationRecorded]:
    """One observation per camera per second (empty when nothing is in view), in event time from `start`."""
    seen: dict[tuple[int, str], list[Track]] = {}
    for second, camera, track in steps:
        seen.setdefault((second, camera), []).append(track)
    last = max(s for s, _ in seen) if length_s is None else length_s
    return [
        ObservationRecorded(event_id=f"fake-{camera}-{int((start + timedelta(seconds=s)).timestamp() * 1000)}",
                            ts=start + timedelta(seconds=s), camera_id=camera, tracks=seen.get((s, camera), []))
        for s in range(last + 1)
        for camera in CAMERAS
    ]


def script(start: datetime) -> list[ObservationRecorded]:
    """The whole scene. The last seconds are quiet so the can's exit shows up."""
    return observations(start, person_arrives() + keys_left() + package_left() + can_recycled(), LENGTH_S)


async def play(bus: EventBus, start: datetime, realtime: bool = False) -> int:
    """Publish the script into `bus`; `realtime` paces it one second per second. Returns observations sent."""
    events = script(start)
    for i, e in enumerate(events):
        if realtime and i and e.ts != events[i - 1].ts:
            await asyncio.sleep((e.ts - events[i - 1].ts).total_seconds())
        await bus.publish(e)
    return len(events)
