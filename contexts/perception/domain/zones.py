"""Zone engine (P3): tracks in, zone events out.

    observation (tracks per camera, ~1/s) -> enter / dwell / exit per (camera, track, zone)

A track is in a zone when the bottom-center of its box (where it touches the
floor or the desk) is inside the zone's polygon. `dwell` repeats every
`dwell_every_s` while it stays, with `dwell_s` counted from the enter. A track
that is not seen for `exit_after_s` exits at `last seen + exit_after_s`,
whether a later observation or the app's ticker (`flush`) notices first.

Pure state machine on event time, and event ids are derived from the content,
so replaying the log gives the same events and the bus drops any it already has.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import yaml

from contracts.perception import ObservationRecorded, ZoneEvent
from contracts.types import Box, FrameRef

Polygon = list[tuple[float, float]]
Key = tuple[str, int, str]  # camera, track, zone

DWELL_EVERY_S = 3.0
EXIT_AFTER_S = 3.0


def load_zones(path: str | Path = "config/zones.yaml") -> dict[str, dict[str, Polygon]]:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return {cam: {z: [tuple(p) for p in poly] for z, poly in zones.items()} for cam, zones in raw.items()}


def point_in_polygon(x: float, y: float, poly: Polygon) -> bool:
    inside = False
    j = len(poly) - 1
    for i, (xi, yi) in enumerate(poly):
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def anchor(box: Box) -> tuple[float, float]:
    """Bottom-center of the box: where the thing stands."""
    return (box.x1 + box.x2) / 2, box.y2


@dataclass
class Stay:
    """One track's current visit to one zone."""

    label: str
    entered: datetime
    last_seen: datetime
    frame_ref: FrameRef | None
    dwells: int = 0  # dwell periods already reported


class ZoneEngine:
    def __init__(self, zones: dict[str, dict[str, Polygon]], dwell_every_s: float = DWELL_EVERY_S,
                 exit_after_s: float = EXIT_AFTER_S) -> None:
        self.zones = zones
        self.dwell_every_s = dwell_every_s
        self.exit_after = timedelta(seconds=exit_after_s)
        self.stays: dict[Key, Stay] = {}

    @classmethod
    def load(cls, path: str | Path = "config/zones.yaml", **kwargs) -> "ZoneEngine":
        return cls(load_zones(path), **kwargs)

    def on_observation(self, e: ObservationRecorded) -> list[ZoneEvent]:
        out = self.flush(e.ts)
        for t in e.tracks:
            x, y = anchor(t.box)
            for zone, poly in self.zones.get(e.camera_id, {}).items():
                key = (e.camera_id, t.track_id, zone)
                if point_in_polygon(x, y, poly):
                    out += self._inside(key, t.label, e)
                elif key in self.stays:
                    self.stays.pop(key)
                    out.append(_event(key, t.label, "exit", e.ts, e.frame_ref))
        return out

    def _inside(self, key: Key, label: str, e: ObservationRecorded) -> list[ZoneEvent]:
        stay = self.stays.get(key)
        if stay is None:
            self.stays[key] = Stay(label, entered=e.ts, last_seen=e.ts, frame_ref=e.frame_ref)
            return [_event(key, label, "enter", e.ts, e.frame_ref)]
        stay.label, stay.last_seen, stay.frame_ref = label, e.ts, e.frame_ref
        dwell_s = (e.ts - stay.entered).total_seconds()
        periods = int(dwell_s // self.dwell_every_s)
        if periods <= stay.dwells:
            return []
        stay.dwells = periods  # one dwell even if samples were missed
        return [_event(key, label, "dwell", e.ts, e.frame_ref, dwell_s)]

    def flush(self, now: datetime) -> list[ZoneEvent]:
        """Exit every track not seen for `exit_after_s`. Call from the ticker: tracks vanish silently."""
        gone = [(s.last_seen + self.exit_after, key) for key, s in self.stays.items() if now - s.last_seen >= self.exit_after]
        out = []
        for ts, key in sorted(gone):
            stay = self.stays.pop(key)
            out.append(_event(key, stay.label, "exit", ts, stay.frame_ref))
        return out


def _event(key: Key, label: str, kind: str, ts: datetime, ref: FrameRef | None, dwell_s: float | None = None) -> ZoneEvent:
    camera, track, zone = key
    event_id = uuid5(NAMESPACE_URL, f"atlas:zone:{camera}:{track}:{zone}:{kind}:{ts.timestamp()}").hex
    return ZoneEvent(event_id=event_id, ts=ts, camera_id=camera, track_id=track, label=label, zone_id=zone,
                     kind=kind, dwell_s=dwell_s, frame_ref=ref)
