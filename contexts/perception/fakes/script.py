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


ROOM_LENGTH_S = 62
_WALK = ((0.60, 0.99), (0.60, 0.99), (0.60, 0.99), (0.60, 0.99), (0.60, 0.99),  # 36-40 s: stands holding the can; the round starts
         (0.638, 0.8875), (0.586, 0.75), (0.38, 0.74), (0.224, 0.7375))  # 41-44 s: around the friend, at a walk


def sort_round(at: int = 36) -> list[Step]:
    """A Sort It round on the bin camera: someone stands holding a can, a friend steps into the straight way to the
    bins, the player walks around them and drops the can in recycling. The can stays out of every zone until the drop."""
    walk = [(at + i, "cam-door", _track(5, "person", x, y, 0.2, 0.6)) for i, (x, y) in enumerate(_WALK)]
    stay = [(at + len(_WALK) + i, "cam-door", _track(5, "person", 0.224, 0.7375, 0.2, 0.6)) for i in range(5)]
    carried = [(at + i, "cam-door", _track(6, "can", x + 0.05, min(y - 0.42, 0.53), 0.04, 0.07)) for i, (x, y) in enumerate(_WALK)]
    dropped = [(at + len(_WALK) + i, "cam-door", _track(6, "can", 0.15, 0.8, 0.04, 0.07)) for i in range(2)]
    friend = [(s, "cam-door", _track(8, "person", 0.448, 0.875, 0.2, 0.6)) for s in range(at + 4, at + 13)]  # steps in at 40 s
    return walk + stay + carried + dropped + friend


def which_bin(at: int = 52) -> list[Step]:
    """Outside a round: the same person holds a cup near the bins (the "which bin?" moment), then drops it in the trash."""
    spots = [(0.345, 0.925)] * 5 + [(0.362, 0.825)] * 3
    person = [(at - 2 + i, "cam-door", _track(5, "person", x, y, 0.2, 0.6)) for i, (x, y) in enumerate([(0.26, 0.9), (0.3, 0.925)] + spots)]
    held = [(at + i, "cam-door", _track(7, "cup", 0.345 + 0.05, 0.5, 0.05, 0.07)) for i in range(5)]
    dropped = [(at + 5 + i, "cam-door", _track(7, "cup", 0.38, 0.8, 0.05, 0.07)) for i in range(2)]
    return person + held + dropped


def room_scene(start: datetime) -> list[ObservationRecorded]:
    """`script` and then a minute more for the room twin: the person leaves the desk, plays a Sort It round with a can
    (walking around a friend who steps into the way), then asks "which bin?" with a cup. The keys and the package stay."""
    steps = (person_arrives(until=33) + keys_left(until=ROOM_LENGTH_S) + package_left(until=ROOM_LENGTH_S) + can_recycled()
             + sort_round() + which_bin())
    return observations(start, steps, ROOM_LENGTH_S)


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


async def play(bus: EventBus, start: datetime, realtime: bool = False, scene=script) -> int:
    """Publish the script (or another `scene`) into `bus`; `realtime` paces it one second per second. Returns observations sent."""
    events = scene(start)
    for i, e in enumerate(events):
        if realtime and i and e.ts != events[i - 1].ts:
            await asyncio.sleep((e.ts - events[i - 1].ts).total_seconds())
        await bus.publish(e)
    return len(events)
