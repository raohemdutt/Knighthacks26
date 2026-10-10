"""Set the zones from a browser, live (the cold-room step of judging: the camera could not be preset).

    GET  /api/zones                  {camera: {zone: [[x, y], ...]}}, read fresh from config/zones.yaml
    PUT  /api/zones/{camera}         {zones: {zone: polygon}} -> rewrites config/zones.yaml and applies it to the running
                                     zone engine at once (enter / dwell / exit events use the new polygons, no restart)
    POST /api/zones/demo             apply config/zones.demo.yaml: big, forgiving zones for the same cameras and ids

Only zones that already exist for that camera can be changed: rules, Waste Watch and Habits refer to zone ids, and a new
id would not be known to them until a restart. Polygons are normalized 0..1, top left origin (like config/zones.yaml).
Sort It reads its bin zones at start, so after moving `recycling` or `trash` restart the gateway for the game's route.
"""

import logging
import os
import threading
from collections.abc import Callable
from pathlib import Path

import yaml
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from contexts.perception.domain.zones import Polygon, ZoneEngine, load_zones

log = logging.getLogger("atlas.zones")

MIN_AREA = 0.005  # of the frame: a stray click is not a zone
MAX_POINTS = 12


class ZonesBody(BaseModel):
    zones: dict[str, list[list[float]]]


def area(poly: Polygon) -> float:
    return abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1], strict=True))) / 2


def clean_polygon(raw: list[list[float]]) -> Polygon:
    if not 3 <= len(raw) <= MAX_POINTS or any(len(p) != 2 for p in raw):
        raise ValueError(f"a zone needs 3 to {MAX_POINTS} points of [x, y]")
    if any(not (0.0 <= v <= 1.0) for p in raw for v in p):
        raise ValueError("points are fractions of the picture, 0 to 1")
    poly = [(round(x, 4), round(y, 4)) for x, y in raw]
    if area(poly) < MIN_AREA:
        raise ValueError("that zone is too small to be useful")
    return poly


def render(zones: dict[str, dict[str, Polygon]], header: str) -> str:
    """config/zones.yaml in the shape the file has always had (one zone per line), keeping its comment header."""
    lines = [header.rstrip("\n")] if header.strip() else []
    for camera, named in zones.items():
        lines.append(f"{camera}:")
        for zone, poly in named.items():
            lines.append(f"  {zone}: [" + ", ".join(f"[{x:g}, {y:g}]" for x, y in poly) + "]")
    return "\n".join(lines) + "\n"


def _header(text: str) -> str:
    out = []
    for line in text.splitlines():
        if not line.startswith("#"):
            break
        out.append(line)
    return "\n".join(out)


def zones_router(config_dir: Path, engine: Callable[[], ZoneEngine | None], guard: Callable | None = None) -> APIRouter:
    router = APIRouter(prefix="/zones")
    deps = [Depends(guard)] if guard else []
    lock = threading.Lock()
    path = config_dir / "zones.yaml"

    def current() -> dict[str, dict[str, Polygon]]:
        return load_zones(path) if path.is_file() else {}

    def as_json(zones: dict[str, dict[str, Polygon]]) -> dict:
        return {cam: {z: [list(p) for p in poly] for z, poly in named.items()} for cam, named in zones.items()}

    def apply(changes: dict[str, dict[str, Polygon]]) -> dict:
        """Merge `changes` ({camera: {zone: polygon}}) into the file and the running engine."""
        with lock:
            text = path.read_text(encoding="utf-8") if path.is_file() else ""
            zones = current()
            for camera, named in changes.items():
                zones[camera].update(named)
            tmp = path.with_suffix(".yaml.tmp")
            tmp.write_text(render(zones, _header(text)), encoding="utf-8")
            os.replace(tmp, path)  # a reader never sees half a file
            live = engine()
            if live is not None:
                live.zones = {cam: dict(named) for cam, named in zones.items()}
                # a stay in an old polygon means nothing now; swap the dict whole, the engine's thread may be iterating it
                live.stays = {k: v for k, v in live.stays.items() if k[0] not in changes}
        log.info("zones set for %s", ", ".join(f"{c}: {', '.join(z)}" for c, z in changes.items()))
        return as_json(zones)

    @router.get("")
    def get_zones() -> dict:
        return as_json(current())

    @router.put("/{camera}", dependencies=deps)
    def put_zones(camera: str, body: ZonesBody) -> dict:
        known = current().get(camera)
        if known is None:
            raise HTTPException(404, f"no zones for camera {camera}; cameras with zones: {', '.join(current()) or 'none'}")
        unknown = [z for z in body.zones if z not in known]
        if unknown:
            raise HTTPException(404, f"{camera} has no zone {', '.join(unknown)}; its zones are {', '.join(known)}")
        try:
            polygons = {z: clean_polygon(p) for z, p in body.zones.items()}
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        return apply({camera: polygons})

    @router.post("/demo", dependencies=deps)
    def demo_zones() -> dict:
        demo_file = config_dir / "zones.demo.yaml"
        if not demo_file.is_file():
            raise HTTPException(404, "config/zones.demo.yaml is not on this box")
        have = current()
        changes = {cam: {z: clean_polygon([list(p) for p in poly]) for z, poly in named.items() if z in have[cam]}
                   for cam, named in load_zones(demo_file).items() if cam in have}
        changes = {cam: named for cam, named in changes.items() if named}
        if not changes:
            raise HTTPException(404, "the demo zones match no camera or zone in config/zones.yaml")
        return apply(changes)

    return router
