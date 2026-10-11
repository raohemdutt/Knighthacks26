"""The plant floor as somewhere to walk: an occupancy grid per floor, rasterised from a trace of the kit's plan picture.

    wk = Walkable.of(trace, pins)            # trace: fixtures/spot/floor_walkable.json; pins: waypoint_pixels.json
    wk.snap(1, 0.196, 0.173)                 -> (0.199, 0.168, 6.1)   the nearest place to stand, and how far (px) it moved
    wk.route((1, 0.10, 0.80), (2, 0.82, 0.82))
    -> {"found": True, "connector": "centre-stairs", "length_px": 912.4,
        "floors": [{"floor": 1, "points": [[x, y], ...]}, {"floor": 2, "points": [...]}]}

What this is and is not:
- The kit gives no walls, doors or walkways. The trace was made by eye from the picture (see the file's `source`), so
  every route from here is "about", never exact, and has no metres: lengths are pixels of that picture.
- A cell is walkable when its centre is inside a traced room, corridor, steps or door rectangle and at least
  `clearance_px` from every traced obstacle. Everything else is wall, roof or outdoors.
- The search is the room twin's A* (contexts/play/room/grid.py: 8 neighbours, no cutting a corner) and its string
  pulling, used as pure functions on a FloorGrid built here; nothing of the room model is involved.
- Floors are joined only at the traced connectors (stairs). A route between floors takes the connector with the
  shortest total walk.

Pure: no files, no clock. All coordinates in and out are normalised 0..1 on the whole picture.
"""

import math
from dataclasses import dataclass, field

from contexts.play.room.grid import FloorGrid, astar, smooth

Point = tuple[float, float]  # normalised on the picture
Rect = tuple[float, float, float, float]


class PlanGrid(FloorGrid):
    """The room twin's grid with a stricter straight-line test: a pulled string may not even clip the corner of a blocked cell.

    The line is sampled every 0.4 cell and each sample needs the little square of 0.2 cell around it free, so every point
    of the line is in a free cell (the room's own test samples the line only, which is right for people and too loose
    for a line that is drawn on a picture and checked against it).
    """

    def clear_line(self, a, b) -> bool:
        e = self.cell * 0.2
        steps = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) / (self.cell * 0.4)))
        for i in range(steps + 1):
            x, y = a[0] + (b[0] - a[0]) * i / steps, a[1] + (b[1] - a[1]) * i / steps
            for dx, dy in ((0, 0), (-e, -e), (e, -e), (-e, e), (e, e)):
                if not (0 <= x + dx < self.width and 0 <= y + dy < self.depth) or self.blocked(self.cell_of(x + dx, y + dy)):
                    return False
        return True


def _in(r: Rect, x: float, y: float) -> bool:
    return r[0] <= x <= r[2] and r[1] <= y <= r[3]


@dataclass
class Floor:
    number: int
    ox: float  # the grid's top-left corner on the picture, in pixels
    oy: float
    grid: FloorGrid
    rooms: list[dict]
    doors: list[dict]
    obstacles: list[dict]
    connectors: list[dict]
    pad: tuple[float, float] = (0.0, 0.0)

    def region(self, x: float, y: float) -> dict | None:
        """The door, else the smallest room, this point is in (give or take half a cell: a path runs through cell centres)."""
        px, py = self.pad

        def near(rect) -> bool:
            return rect[0] - px <= x <= rect[2] + px and rect[1] - py <= y <= rect[3] + py

        for d in self.doors:
            if near(d["rect"]):
                return d
        inside = [r for r in self.rooms if _in(tuple(r["rect"]), x, y)] or [r for r in self.rooms if near(r["rect"])]
        return min(inside, key=lambda r: (r["rect"][2] - r["rect"][0]) * (r["rect"][3] - r["rect"][1])) if inside else None


@dataclass
class Walkable:
    width: int
    height: int
    cell: float
    clearance: float
    snap_px: float
    source: str
    floors: dict[int, Floor]
    pins: dict[str, dict] = field(default_factory=dict)  # waypoint -> {"floor", "x", "y"}: the pin's tip
    _stand: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    _legs: dict[tuple[str, str], dict | None] = field(default_factory=dict)

    @classmethod
    def of(cls, trace: dict, pins: dict[str, dict] | None = None) -> "Walkable":
        w, h, cell, clearance = int(trace["width"]), int(trace["height"]), float(trace["cell_px"]), float(trace["clearance_px"])
        floors = {}
        for key, fl in trace["floors"].items():
            open_rects = [tuple(r["rect"]) for r in [*fl["rooms"], *fl["doors"]]]
            solid = [tuple(o["rect"]) for o in fl["obstacles"]]
            x0 = min(r[0] for r in open_rects) * w - 2 * cell
            y0 = min(r[1] for r in open_rects) * h - 2 * cell
            x1 = max(r[2] for r in open_rects) * w + 2 * cell
            y1 = max(r[3] for r in open_rects) * h + 2 * cell
            cols, rows = math.ceil((x1 - x0) / cell), math.ceil((y1 - y0) / cell)
            blocked = set()
            for row in range(rows):
                for col in range(cols):
                    px, py = x0 + (col + 0.5) * cell, y0 + (row + 0.5) * cell
                    nx, ny = px / w, py / h
                    if not any(_in(r, nx, ny) for r in open_rects) or any(_rect_distance(o, px, py, w, h) < clearance for o in solid):
                        blocked.add((row, col))
            grid = PlanGrid(cols * cell, rows * cell, cell, cols, rows, frozenset(blocked))
            floors[int(key)] = Floor(int(key), x0, y0, grid, fl["rooms"], fl["doors"], fl["obstacles"], fl.get("connectors", []), (0.6 * cell / w, 0.6 * cell / h))
        return cls(w, h, cell, clearance, float(trace.get("snap_px", 30)), trace.get("source", ""), floors, dict(pins or {}))

    # --- points ---

    def _local(self, f: Floor, x: float, y: float) -> tuple[float, float]:
        return x * self.width - f.ox, y * self.height - f.oy

    def _norm(self, f: Floor, p: tuple[float, float]) -> Point:
        return round((p[0] + f.ox) / self.width, 4), round((p[1] + f.oy) / self.height, 4)

    def floor_at(self, x: float, y: float) -> int | None:
        """Which floor's part of the picture a point is on (by the traced rooms' bounding box)."""
        for n, f in self.floors.items():
            lx, ly = self._local(f, x, y)
            if 0 <= lx <= f.grid.width and 0 <= ly <= f.grid.depth:
                return n
        return None

    def free(self, floor: int, x: float, y: float) -> bool:
        f = self.floors.get(floor)
        if f is None:
            return False
        lx, ly = self._local(f, x, y)
        return 0 <= lx < f.grid.width and 0 <= ly < f.grid.depth and not f.grid.blocked(f.grid.cell_of(lx, ly))

    def snap(self, floor: int, x: float, y: float, limit_px: float | None = None) -> tuple[float, float, float] | None:
        """The nearest place to stand to this point: (x, y, how far it moved in pixels), or None when none is within the limit."""
        f = self.floors.get(floor)
        if f is None:
            return None
        if self.free(floor, x, y):
            return x, y, 0.0
        limit = self.snap_px if limit_px is None else limit_px
        lx, ly = self._local(f, x, y)
        g = f.grid
        best, best_d = None, float("inf")
        n = int(limit / g.cell) + 2
        r0, c0 = int(ly // g.cell), int(lx // g.cell)
        for r in range(r0 - n, r0 + n + 1):
            for c in range(c0 - n, c0 + n + 1):
                if g.inside((r, c)) and not g.blocked((r, c)):
                    cx, cy = g.center((r, c))
                    d = math.hypot(cx - lx, cy - ly)
                    if d < best_d:
                        best, best_d = (cx, cy), d
        if best is None or best_d > limit:
            return None
        nx, ny = self._norm(f, best)
        return nx, ny, round(best_d, 1)

    def stand(self, waypoint: str) -> tuple[float, float, float] | None:
        """Where someone stands at a waypoint: its pin tip, or the nearest walkable cell within snap_px of it."""
        if waypoint not in self._stand:
            pin = self.pins.get(waypoint)
            got = self.snap(int(pin["floor"]), float(pin["x"]), float(pin["y"])) if pin else None
            if got is None:
                return None
            self._stand[waypoint] = got
        return self._stand[waypoint]

    # --- paths ---

    def path(self, floor: int, a: Point, b: Point) -> tuple[list[Point], float] | None:
        """A walk on one floor between two walkable points: a short polyline and its length in pixels of the picture."""
        f = self.floors.get(floor)
        if f is None or not self.free(floor, *a) or not self.free(floor, *b):
            return None
        g = f.grid
        la, lb = self._local(f, *a), self._local(f, *b)
        cells = astar(g, g.cell_of(*la), {g.cell_of(*lb)})
        if cells is None:
            return None
        pts = smooth(g, [la, *[g.center(c) for c in cells[1:-1]], lb] if len(cells) > 1 else [la, lb])
        length = sum(math.hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip(pts, pts[1:]))
        return [self._norm(f, p) for p in pts], round(length, 1)

    def route(self, start: tuple[int, float, float], goal: tuple[int, float, float]) -> dict:
        """A walk between two walkable points, on one floor or through the connector that makes it shortest."""
        (fa, ax, ay), (fb, bx, by) = start, goal
        none = {"found": False, "connector": None, "length_px": 0.0, "floors": []}
        if fa == fb:
            got = self.path(fa, (ax, ay), (bx, by))
            return {"found": True, "connector": None, "length_px": got[1], "floors": [{"floor": fa, "points": [list(p) for p in got[0]]}]} if got else none
        if fa not in self.floors or fb not in self.floors:
            return none
        best = None
        there = {c["id"]: c for c in self.floors[fb].connectors}
        for c in self.floors[fa].connectors:
            d = there.get(c["id"])
            if d is None:
                continue
            first, second = self.path(fa, (ax, ay), (c["x"], c["y"])), self.path(fb, (d["x"], d["y"]), (bx, by))
            if first and second and (best is None or first[1] + second[1] < best[0]):
                best = (first[1] + second[1], c, first, second)
        if best is None:
            return none
        total, c, first, second = best
        return {"found": True, "connector": c["id"], "connector_name": c["name"], "length_px": round(total, 1),
                "floors": [{"floor": fa, "points": [list(p) for p in first[0]]}, {"floor": fb, "points": [list(p) for p in second[0]]}]}

    def leg(self, a: str, b: str) -> dict | None:
        """The walk between two waypoints (kept: the same pairs are asked for again and again)."""
        if (a, b) not in self._legs:
            sa, sb = self.stand(a), self.stand(b)
            if sa is None or sb is None:
                self._legs[(a, b)] = None
            elif (b, a) in self._legs and self._legs[(b, a)]:
                back = self._legs[(b, a)]
                self._legs[(a, b)] = {**back, "floors": [{"floor": fl["floor"], "points": fl["points"][::-1]} for fl in back["floors"][::-1]]}
            else:
                got = self.route((int(self.pins[a]["floor"]), sa[0], sa[1]), (int(self.pins[b]["floor"]), sb[0], sb[1]))
                self._legs[(a, b)] = got if got["found"] else None
        return self._legs[(a, b)]

    # --- for drawing ---

    def geometry(self) -> dict:
        out = {}
        for n, f in sorted(self.floors.items()):
            g = f.grid
            runs = []  # walkable cells as horizontal runs [row, first col, last col]: small enough to draw as rectangles
            for r in range(g.rows):
                c = 0
                while c < g.cols:
                    if g.blocked((r, c)):
                        c += 1
                        continue
                    c0 = c
                    while c < g.cols and not g.blocked((r, c)):
                        c += 1
                    runs.append([r, c0, c - 1])
            out[str(n)] = {"rooms": f.rooms, "doors": f.doors, "obstacles": f.obstacles, "connectors": f.connectors,
                           "grid": {"x0": round(f.ox / self.width, 5), "y0": round(f.oy / self.height, 5), "cell_x": round(self.cell / self.width, 6),
                                    "cell_y": round(self.cell / self.height, 6), "cols": g.cols, "rows": g.rows, "free": g.cols * g.rows - len(g.static), "runs": runs}}
        return out


def _rect_distance(r: Rect, px: float, py: float, w: int, h: int) -> float:
    """How far a pixel point is from a normalised rectangle (0 inside it)."""
    dx = max(r[0] * w - px, 0.0, px - r[2] * w)
    dy = max(r[1] * h - py, 0.0, py - r[3] * h)
    return math.hypot(dx, dy)
