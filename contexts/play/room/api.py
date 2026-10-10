"""The room twin's endpoints.

    GET /api/room                         the floor plan (metres), each camera's calibration with its homography, and
                                          the zones of config/zones.yaml laid on the floor
    PUT /api/room/calibration/{camera}    {image: four [x, y] of the picture (0..1), floor: the four floor points (m)}
                                          -> saved into config/rooms/<name>.yaml and used at once
    GET /api/room/live                    tracks on the floor right now, where things were last seen, the Sort It
                                          round with its routes, the last result, and the "which bin?" hint
    GET /api/play/room-route              ?item=can[&from_track=cam-door:5 | &x=&y=] -> the walk to the right bin on
                                          the floor plan: points in metres, distance, par seconds

Calibrating in four taps: tap four spots on the floor in the camera's picture (the corners of a rug or of a taped
square work well), and give the same four spots in room metres, in the same order. Three of them must not be in a line.
"""

import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from contexts.perception.domain.zones import load_zones
from contexts.play.room.homography import Degenerate
from contexts.play.room.live import RoomLive
from contexts.play.room.plan import Calibration, RoomPlan, available, save_calibration


class CalibrationBody(BaseModel):
    image: list[list[float]]
    floor: list[list[float]]


def room_router(live: Callable[[], RoomLive | None], config_dir: Path, guard: Callable | None = None) -> APIRouter:
    router = APIRouter()
    deps = [Depends(guard)] if guard else []
    lock = threading.Lock()

    def room() -> RoomLive:
        r = live()
        if r is None:
            raise HTTPException(404, "no room plan on this box (config/rooms/<name>.yaml)")
        return r

    def describe(plan: RoomPlan) -> dict:
        zones_file = config_dir / "zones.yaml"
        zones = load_zones(zones_file) if zones_file.is_file() else {}  # read fresh: they can be redrawn live
        return {**plan.as_json(), "zones": plan.zones_on_floor(zones), "rooms": available(config_dir),
                "uncalibrated": sorted(c for c in zones if c not in plan.calibration)}

    @router.get("/api/room")
    def get_room() -> dict:
        return describe(room().plan)

    @router.put("/api/room/calibration/{camera}", dependencies=deps)
    def put_calibration(camera: str, body: CalibrationBody) -> dict:
        r = room()
        plan = r.plan
        if any(len(p) != 2 for p in body.image + body.floor) or len(body.image) != 4 or len(body.floor) != 4:
            raise HTTPException(422, "image and floor are four points of [x, y] each, in the same order")
        if any(not (0.0 <= v <= 1.0) for p in body.image for v in p):
            raise HTTPException(422, "picture points are fractions of the picture, 0 to 1")
        if any(not (-0.5 <= p[0] <= plan.width + 0.5 and -0.5 <= p[1] <= plan.depth + 0.5) for p in body.floor):
            raise HTTPException(422, f"floor points are metres inside the room ({plan.width:g} by {plan.depth:g})")
        try:
            cal = Calibration.of([[round(v, 4) for v in p] for p in body.image], [[round(v, 3) for v in p] for p in body.floor])
        except Degenerate as err:
            raise HTTPException(422, str(err)) from None
        with lock:
            plan.calibration = {**plan.calibration, camera: cal}
            save_calibration(plan)
            r.set_plan(plan)
        return describe(plan)

    @router.get("/api/room/live")
    def get_live() -> dict:
        return room().state(datetime.now(UTC))

    @router.get("/api/play/room-route")
    def room_route(item: str, from_track: str | None = None, x: float | None = Query(None, ge=0, le=100),
                   y: float | None = Query(None, ge=0, le=100)) -> dict:
        r = room()
        if r.rules.canonical(item) is None:
            raise HTTPException(404, f"unknown item {item!r}; sortable: {', '.join(sorted(r.rules.items))}")
        walker = None
        if from_track:
            walker = next((t for t in r.tracks() if t.key == from_track), None)
            if walker is None:
                raise HTTPException(404, f"no track {from_track} in the room right now")
        elif x is None or y is None:
            walker = next(iter(r.players_for(item, 1)), None)  # whoever a camera sees, else the plan's start point
        start = (walker.x, walker.y) if walker else ((x, y) if x is not None and y is not None else r.plan.start)
        target, route = r.route_for(item, start, walker)
        if target is None:
            raise HTTPException(404, f"this room has no {r.rules.bin_for(item)} bin")
        return {"item": item, "bin": target.bin, "bin_id": target.id, "zone": r.rules.zone_for(item), "room": r.plan.name,
                "from": {"x": round(start[0], 3), "y": round(start[1], 3), "track": walker.key if walker else None},
                "reachable": route is not None, **(route.as_json() if route else {"points": [], "distance_m": 0.0, "par_s": 0.0})}

    return router
