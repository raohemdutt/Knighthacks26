"""A room's floor plan and camera calibration (config/rooms/<name>.yaml), and the geometry that follows from them.

Plan coordinates are metres: x east along the north wall, y south, origin in the north-west corner. A piece of
furniture is a box: centre, [width, depth], a rotation (degrees clockwise seen from above; 0 = its front looks south)
and a height. A camera's calibration is four points of its picture and the four floor points they show; the
homography between them (homography.py) puts a track's foot point on the floor.
"""

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from contexts.play.room.homography import Degenerate, Matrix, Point, apply, homography, invert

FACING = {"south": 0.0, "west": 90.0, "north": 180.0, "east": 270.0}
WALLS = ("north", "east", "south", "west")
NOT_IN_THE_WAY = {"rug"}  # walked over, not around
DEFAULT_ROOM = "dorm"
_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")


class RoomError(ValueError):
    """The room file is missing something, or says something that cannot be drawn."""


@dataclass(frozen=True)
class Furniture:
    id: str
    type: str
    at: Point
    size: tuple[float, float]  # width (along its front), depth
    rot: float = 0.0  # degrees clockwise from above; 0 = front looks south (+y)
    height: float = 0.5
    bin: str | None = None  # for a bin: its name in config/sorting.yaml
    color: str | None = None
    label: str | None = None

    @property
    def blocks(self) -> bool:
        return self.type not in NOT_IN_THE_WAY

    def local(self, x: float, y: float) -> Point:
        """(x, y) in this box's own frame: its centre at the origin, its front along +y."""
        t = math.radians(self.rot)
        dx, dy = x - self.at[0], y - self.at[1]
        return dx * math.cos(t) + dy * math.sin(t), -dx * math.sin(t) + dy * math.cos(t)

    def contains(self, x: float, y: float, grow: float = 0.0) -> bool:
        lx, ly = self.local(x, y)
        return abs(lx) <= self.size[0] / 2 + grow and abs(ly) <= self.size[1] / 2 + grow

    def distance(self, x: float, y: float) -> float:
        """Metres from (x, y) to the box; 0 inside it."""
        lx, ly = self.local(x, y)
        return math.hypot(max(abs(lx) - self.size[0] / 2, 0.0), max(abs(ly) - self.size[1] / 2, 0.0))

    def corners(self) -> list[Point]:
        t = math.radians(self.rot)
        w, d = self.size[0] / 2, self.size[1] / 2
        return [(self.at[0] + lx * math.cos(t) - ly * math.sin(t), self.at[1] + lx * math.sin(t) + ly * math.cos(t))
                for lx, ly in ((-w, -d), (w, -d), (w, d), (-w, d))]

    def as_json(self) -> dict:
        return {"id": self.id, "type": self.type, "at": list(self.at), "size": list(self.size), "rot": self.rot,
                "height": self.height, "bin": self.bin, "color": self.color, "label": self.label}


@dataclass(frozen=True)
class Calibration:
    image: list[Point]
    floor: list[Point]
    h: Matrix  # picture -> floor
    h_inv: Matrix  # floor -> picture

    @classmethod
    def of(cls, image: list, floor: list) -> "Calibration":
        img = [(float(p[0]), float(p[1])) for p in image]
        flo = [(float(p[0]), float(p[1])) for p in floor]
        h = homography(img, flo)
        return cls(img, flo, h, invert(h))

    def as_json(self) -> dict:
        return {"image": [list(p) for p in self.image], "floor": [list(p) for p in self.floor], "homography": self.h}


@dataclass
class RoomPlan:
    name: str
    title: str
    width: float
    depth: float
    wall_height: float = 2.5
    doors: list[dict] = field(default_factory=list)
    windows: list[dict] = field(default_factory=list)
    posters: list[dict] = field(default_factory=list)
    furniture: list[Furniture] = field(default_factory=list)
    calibration: dict[str, Calibration] = field(default_factory=dict)
    walk_speed: float = 1.2
    body_radius: float = 0.25
    start: Point = (1.0, 1.0)
    path: Path | None = None

    # --- geometry ---

    def clamp(self, x: float, y: float, margin: float = 0.05) -> Point:
        """The nearest point inside the walls."""
        return min(max(x, margin), self.width - margin), min(max(y, margin), self.depth - margin)

    def to_floor(self, camera: str, x: float, y: float, clamp: bool = True) -> Point | None:
        """A point of `camera`'s picture (0..1) on the floor, in metres; None without a calibration for it."""
        cal = self.calibration.get(camera)
        p = apply(cal.h, x, y) if cal else None
        if p is None:
            return None
        return self.clamp(*p) if clamp else p

    def to_image(self, camera: str, x: float, y: float) -> Point | None:
        """The inverse: where a floor point is in `camera`'s picture."""
        cal = self.calibration.get(camera)
        return apply(cal.h_inv, x, y) if cal else None

    def zones_on_floor(self, zones: dict[str, dict[str, list]]) -> dict[str, dict[str, list[list[float]]]]:
        """Each camera's zone polygons (picture, 0..1) as floor polygons, kept inside the walls."""
        out: dict[str, dict[str, list[list[float]]]] = {}
        for camera, named in zones.items():
            if camera not in self.calibration:
                continue
            out[camera] = {}
            for zone, poly in named.items():
                points = [self.to_floor(camera, float(p[0]), float(p[1]), clamp=False) for p in poly]
                if all(p is not None for p in points):
                    out[camera][zone] = [[round(v, 3) for v in self.clamp(*p, margin=0.0)] for p in points]
        return out

    def furniture_at(self, x: float, y: float, grow: float = 0.0) -> Furniture | None:
        """The tallest piece of furniture this point is on (a rug does not count)."""
        under = [f for f in self.furniture if f.blocks and f.contains(x, y, grow)]
        return max(under, key=lambda f: f.height) if under else None

    def bins(self) -> dict[str, Furniture]:
        return {f.bin: f for f in self.furniture if f.type == "bin" and f.bin}

    # --- in and out ---

    def as_json(self) -> dict:
        return {"name": self.name, "title": self.title, "size": [self.width, self.depth],
                "outline": [[0, 0], [self.width, 0], [self.width, self.depth], [0, self.depth]],
                "wall_height": self.wall_height, "doors": self.doors, "windows": self.windows, "posters": self.posters,
                "furniture": [f.as_json() for f in self.furniture],
                "calibration": {cam: c.as_json() for cam, c in self.calibration.items()},
                "play": {"walk_speed": self.walk_speed, "body_radius": self.body_radius, "start": list(self.start)}}


def _opening(raw: dict, length: dict[str, float], what: str) -> dict:
    wall = raw.get("wall")
    if wall not in WALLS:
        raise RoomError(f"{what}: wall must be one of {', '.join(WALLS)}")
    at, width = float(raw.get("at", 0)), float(raw.get("width", 0))
    if width <= 0 or at - width / 2 < 0 or at + width / 2 > length[wall]:
        raise RoomError(f"{what} on the {wall} wall does not fit: at {at}, width {width}, wall {length[wall]} m")
    return {**raw, "wall": wall, "at": at, "width": width}


def parse_room(raw: dict, path: Path | None = None) -> RoomPlan:
    if not isinstance(raw, dict):
        raise RoomError("a room file is a mapping")
    outline = raw.get("outline") or []
    if len(outline) < 3:
        raise RoomError("outline: at least three [x, y] corners, in metres")
    xs, ys = [float(p[0]) for p in outline], [float(p[1]) for p in outline]
    if min(xs) != 0 or min(ys) != 0:
        raise RoomError("outline: the north-west corner is [0, 0]")
    width, depth = max(xs), max(ys)
    if width < 1 or depth < 1:
        raise RoomError("outline: the room is smaller than a metre")
    length = {"north": width, "south": width, "east": depth, "west": depth}
    furniture, seen = [], set()
    for i, f in enumerate(raw.get("furniture") or []):
        try:
            fid = str(f.get("id") or f"{f['type']}-{i}")
            rot = float(f["rot"]) if "rot" in f else FACING[f.get("facing", "south")]
            item = Furniture(id=fid, type=str(f["type"]), at=(float(f["at"][0]), float(f["at"][1])),
                             size=(float(f["size"][0]), float(f["size"][1])), rot=rot, height=float(f.get("height", 0.5)),
                             bin=f.get("bin"), color=f.get("color"), label=f.get("label"))
        except (KeyError, TypeError, IndexError, ValueError) as err:
            raise RoomError(f"furniture {i}: needs id, type, at [x, y], size [width, depth]; {err!r}") from None
        if fid in seen:
            raise RoomError(f"furniture id {fid} is used twice")
        if not (0 <= item.at[0] <= width and 0 <= item.at[1] <= depth):
            raise RoomError(f"furniture {fid} is outside the room")
        seen.add(fid)
        furniture.append(item)
    calibration = {}
    for camera, c in (raw.get("calibration") or {}).items():
        try:
            calibration[camera] = Calibration.of(c["image"], c["floor"])
        except (Degenerate, KeyError, TypeError, IndexError) as err:
            raise RoomError(f"calibration for {camera}: {err}") from None
    play = raw.get("play") or {}
    start = play.get("start") or [width / 2, depth / 2]
    return RoomPlan(
        name=str(raw.get("name") or (path.stem if path else "room")), title=str(raw.get("title") or raw.get("name") or "Room"),
        width=width, depth=depth, wall_height=float(raw.get("wall_height", 2.5)),
        doors=[_opening(d, length, "door") for d in raw.get("doors") or []],
        windows=[_opening(w, length, "window") for w in raw.get("windows") or []],
        posters=[_opening(p, length, "poster") for p in raw.get("posters") or []],
        furniture=furniture, calibration=calibration, walk_speed=float(play.get("walk_speed", 1.2)),
        body_radius=float(play.get("body_radius", 0.25)), start=(float(start[0]), float(start[1])), path=path)


def load_room(path: str | Path) -> RoomPlan:
    path = Path(path)
    return parse_room(yaml.safe_load(path.read_text(encoding="utf-8")), path)


def available(config_dir: str | Path) -> list[str]:
    return sorted(p.stem for p in (Path(config_dir) / "rooms").glob("*.yaml"))


def room_path(config_dir: str | Path, name: str | None = None) -> Path | None:
    """The room file to use: `name`, else ATLAS_ROOM, else dorm; None when there is no such file."""
    name = name or os.environ.get("ATLAS_ROOM") or DEFAULT_ROOM
    if not _NAME.match(name):
        return None
    path = Path(config_dir) / "rooms" / f"{name}.yaml"
    return path if path.is_file() else None


def render_calibration(calibration: dict[str, Calibration]) -> str:
    lines = ["calibration:"]
    for camera, c in calibration.items():
        lines.append(f"  {camera}:")
        for key, points in (("image", c.image), ("floor", c.floor)):
            lines.append(f"    {key}: [" + ", ".join(f"[{x:g}, {y:g}]" for x, y in points) + "]")
    return "\n".join(lines) + "\n"


def save_calibration(plan: RoomPlan) -> None:
    """Rewrite only the `calibration:` block of the room file (everything else, comments included, stays)."""
    if plan.path is None:
        raise RoomError("this room was not loaded from a file")
    lines = plan.path.read_text(encoding="utf-8").splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if line.startswith("calibration:")), None)
    if start is None:
        head, tail = lines + (["\n"] if lines and lines[-1].strip() else []), []
    else:
        end = next((i for i in range(start + 1, len(lines)) if lines[i][:1] not in (" ", "\n", "#", "")), len(lines))
        head, tail = lines[:start], lines[end:]
    tmp = plan.path.with_suffix(".yaml.tmp")
    tmp.write_text("".join(head) + render_calibration(plan.calibration) + "".join(tail), encoding="utf-8")
    os.replace(tmp, plan.path)  # a reader never sees half a file
