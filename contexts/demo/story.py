"""The demo night as source events: fixtures/demo/night.yaml -> a time-ordered list of contract events.

    events = story()                # ends at the script's own `now` (the same as fixtures/memory/memory.yaml)
    events = story(now=some_time)   # the same story, moved so it ends at `some_time`

Only what cameras and connectors publish comes out of here: `perception.observation.recorded`,
`perception.scene.described`, `connector.item.synced`, and `perception.faces.counted` for the one
Sort It round. Zone events, episodes, habits, Waste Watch counts, alerts, plans and game rounds are
left to the app's own handlers.

Deterministic: the same script and `now` give the same events with the same ids. Boxes are placed
from config/zones.yaml, in the style of contexts/perception/fakes/script.py (a `Step` is a second,
a camera and a track; a track stands where the bottom-center of its box is).

A night at one observation a second would be far too many events, so a beat is a short burst:
one a second at the door (dwell and alert timing), one every two seconds at the desk (under the
zone engine's 3 s exit, so a seated person is one unbroken stay), then two empty frames, 3 s and
34 s after the last activity, the way a camera reports an empty room: the first lets the zone
engine exit what left, the second lets Memory close the episode. Between beats there is nothing.
"""

from datetime import datetime, timedelta
from itertools import count
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

from contexts.perception.domain.scenes import bounding_box, objects_caption, scene
from contexts.perception.domain.zones import load_zones
from contexts.perception.fakes.script import Step
from contracts.connectors import ItemKind, ItemSynced
from contracts.events import Event
from contracts.perception import ObservationRecorded
from contracts.play import FacesCounted
from contracts.types import Box, ObjectHit, Track, frame_ref

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "fixtures" / "demo" / "night.yaml"
ZONES = ROOT / "config" / "zones.yaml"
MODEL = "demo-script"  # model_version on scripted captions: no model wrote them
DOOR, DESK = "door", "desk"
DESK_EVERY_S = 2  # under the zone engine's 3 s exit
EXIT_AFTER_S, QUIET_AFTER_S = 3, 34  # the two empty frames after a beat
_ORDER = {ItemSynced.TYPE: 0, ObservationRecorded.TYPE: 1}  # at the same instant: items, the frame, what was said about it


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a misspelled key in the script is a bug, not data


class Toss(_Strict):
    label: str
    bin: str  # zone id of the bin it goes in


class Beat(_Strict):
    at: datetime
    camera: str
    move: Literal["sit", "arrive", "leave", "pass", "bins", "none"] = "none"
    people: int = 1
    seconds: int = 0
    carrying: list[str] = []
    things: list[str] = []
    package: int = 0
    toss: list[Toss] = []
    faces: int = 0
    captions: list[str] = []
    zone: str | None = None
    objects: list[ObjectHit] = []


class Synced(_Strict):
    synced: datetime
    connector: str
    kind: ItemKind
    external_id: str
    title: str
    when: datetime | None = None
    url: str | None = None
    summary: str | None = None


class Script(_Strict):
    now: datetime
    items: list[Synced] = []
    beats: list[Beat] = []


def load(path: str | Path = SCRIPT) -> Script:
    return Script.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def _track(track_id: int, label: str, x: float, bottom: float, w: float, h: float) -> Track:
    """A box `w` x `h` whose bottom-center is at (x, bottom)."""
    return Track(track_id=track_id, label=label, confidence=0.9,
                 box=Box(x1=round(x - w / 2, 3), y1=round(bottom - h, 3), x2=round(x + w / 2, 3), y2=round(bottom, 3)))


class Stage:
    """Where things stand on each camera, worked out from the zone polygons."""

    def __init__(self, zones: dict) -> None:
        self.boxes = {cam: {z: bounding_box(poly) for z, poly in zs.items()} for cam, zs in zones.items()}

    def box(self, camera: str, zone: str) -> Box:
        return self.boxes[camera][zone]

    def hand_height(self, camera: str) -> float:
        """Above every zone on this camera: something carried is in no zone."""
        return round(min(b.y1 for b in self.boxes[camera].values()) - 0.05, 3)

    def steps(self, b: Beat, ids) -> list[Step]:
        cam = b.camera
        if b.move == "sit" or cam not in self.boxes or DOOR not in self.boxes[cam]:
            return self._desk(b, ids)
        door = self.box(cam, DOOR)
        floor, mid, hand = door.y2 - 0.08, (door.x1 + door.x2) / 2, self.hand_height(cam)
        out: list[Step] = []
        if b.move in ("arrive", "leave"):  # in from the far edge of the door zone and out of it, or the other way
            xs = [door.x2 - 0.1, mid, door.x1 + 0.1, door.x1 - 0.04]
            xs = xs[::-1] if b.move == "leave" else xs
            for p in range(b.people):
                pid, held = next(ids), [(next(ids), label) for label in (b.carrying if p == 0 else [])]
                for i, x in enumerate(xs):
                    out.append((2 * p + i, cam, _track(pid, "person", x, floor, 0.2, 0.6)))
                    out += [(i, cam, _track(hid, label, x + 0.04 * k, hand, 0.08, 0.1)) for k, (hid, label) in enumerate(held)]
        elif b.move == "pass":
            for p in range(b.people):
                pid = next(ids)
                out += [(s, cam, _track(pid, "person", mid - 0.08 * p, floor, 0.2, 0.6)) for s in range(b.seconds)]
        if b.package:
            pid = next(ids)
            out += [(s, cam, _track(pid, "package", door.x2 - 0.15, floor, 0.16, 0.14)) for s in range(b.package)]
        if b.move == "bins":  # carried at hand height across the room, lowered into the bin, then out of sight in it
            for k, t in enumerate(b.toss, start=b.faces):  # after the faces that start a Sort It round
                tid, target = next(ids), self.box(cam, t.bin)
                x_bin = (target.x1 + target.x2) / 2 + 0.03 * (k % 3 - 1)
                out += [(k + i, cam, _track(tid, t.label, mid + (x_bin - mid) * i / 3, hand, 0.04, 0.07)) for i in range(4)]
                out += [(k + 4 + i, cam, _track(tid, t.label, x_bin, (target.y1 + target.y2) / 2, 0.04, 0.07)) for i in range(2)]
            pid, last = next(ids), max(s for s, _, _ in out)
            # the person stands close to the camera, below every zone
            out += [(s, cam, _track(pid, "person", 0.5, 1.0, 0.3, 0.7)) for s in range(last + 1)]
        return out

    def _desk(self, b: Beat, ids) -> list[Step]:
        desk = self.box(b.camera, DESK)
        width = desk.x2 - desk.x1
        out: list[Step] = []
        if b.move == "sit":
            for p in range(b.people):
                pid, x = next(ids), desk.x1 + width * (p + 1) / (b.people + 1)
                out += [(s, b.camera, _track(pid, "person", x, desk.y2 - 0.05, 0.3, 0.5))
                        for s in range(0, b.seconds + 1, DESK_EVERY_S)]
        for k, label in enumerate(b.things):  # the detector picks these up as the scene starts; the caption carries on
            tid, x = next(ids), desk.x1 + width * (k + 1) / (len(b.things) + 1)
            out += [(s, b.camera, _track(tid, label, x, desk.y1 + 0.3, 0.08, 0.06)) for s in (0, DESK_EVERY_S)]
        return out


def _scenes(b: Beat, start: datetime, last_s: int) -> list[tuple[int, Event]]:
    """(second, scene): each caption on its own frame, the first a little into the beat, then the zone crop."""
    first = 4 if b.move == "sit" else 2
    out = []
    for i, caption in enumerate(b.captions):
        s = min(first + 30 * i, max(last_s, first))
        out.append((s, _scene(b.camera, start + timedelta(seconds=s), caption)))
    if b.objects:
        s, zone = first + 2, b.zone or DESK
        out.append((s, _scene(b.camera, start + timedelta(seconds=s), objects_caption(zone, b.objects), zone, b.objects)))
    return out


def _scene(camera: str, ts: datetime, caption: str, zone: str | None = None, objects: list[ObjectHit] = ()) -> Event:
    return scene(camera, _frame(camera, ts), ts, caption, MODEL, zone_id=zone, objects=objects)


def _frame(camera: str, ts: datetime) -> str:
    return frame_ref(camera, int(ts.timestamp() * 1000))


def story(now: datetime | None = None, script: Script | None = None, zones: str | Path = ZONES) -> list[Event]:
    """Every source event of the night, oldest first, moved so the script's `now` is `now`."""
    script = script or load()
    by = (now - script.now) if now else timedelta(0)
    stage, ids = Stage(load_zones(zones)), count(1)
    events: list[Event] = []
    frames: dict[tuple[datetime, str], list[Track]] = {}
    quiet: set[tuple[datetime, str]] = set()

    for item in script.items:
        ts = item.synced + by
        events.append(ItemSynced(
            event_id=f"demo-item-{item.connector}-{item.external_id}-{int(ts.timestamp())}", ts=ts,
            when=item.when + by if item.when else None, **item.model_dump(exclude={"synced", "when"})))

    for b in script.beats:
        start = b.at + by
        steps = stage.steps(b, ids)
        for second, camera, track in steps:
            frames.setdefault((start + timedelta(seconds=second), camera), []).append(track)
        last = max((s for s, _, _ in steps), default=0)
        scenes = _scenes(b, start, last)
        events += [e for _, e in scenes]
        last = max([last] + [s for s, _ in scenes])
        quiet |= {(start + timedelta(seconds=last + after), b.camera) for after in (EXIT_AFTER_S, QUIET_AFTER_S)}
        if b.faces:  # looking at the camera long enough to start a round, then gone once the beat is over
            looks = [(s, b.people) for s in range(b.faces + 1)] + [(last + EXIT_AFTER_S, 0)]
            events += [FacesCounted(event_id=f"demo-faces-{b.camera}-{int((start + timedelta(seconds=s)).timestamp() * 1000)}",
                                    ts=start + timedelta(seconds=s), camera_id=b.camera, count=n, facing=n) for s, n in looks]

    for ts, camera in quiet - set(frames):
        frames[(ts, camera)] = []
    for (ts, camera), tracks in frames.items():
        ms = int(ts.timestamp() * 1000)
        events.append(ObservationRecorded(event_id=f"demo-{camera}-{ms}", ts=ts, camera_id=camera, tracks=tracks,
                                          frame_ref=_frame(camera, ts) if tracks else None))
    return sorted(events, key=lambda e: (e.ts, _ORDER.get(e.type, 2), e.event_id))
