"""Real walking routes on a campus's footpath network (map data © OpenStreetMap contributors, ODbL).

A place pack can name a map file (`network: ucf.map.json`, written by
`scripts/build_campus_pack.py`). Its `walk` section is a graph: nodes where paths
meet, edges with the path's own shape and a kind (footpath, steps, crossing, or a
road shared with cars). This module loads it once and answers "how do I walk from
here to there": the polyline, the metres and the minutes.

    net = WalkNetwork.load("config/places/ucf.map.json")
    r = net.route((28.60191, -81.20055), (28.60195, -81.19846))
    r.meters, r.minutes, r.points      # 268.0, 3.4, ((lat, lon), ...)

Cost is time: length / walking speed, slower on steps, a short wait at each road
crossing, a little slower along a road. A* with the straight-line time as the
heuristic, which never overestimates because every factor is at least 1.
Points that are not on the network (a door, a lawn) join it at the nearest spot
on the nearest path, and that last straight stretch is counted too.
"""

import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path

ATTRIBUTION = "© OpenStreetMap contributors"
FOOT, STEPS, CROSSING, ROAD = 0, 1, 2, 3
DEFAULT_SPEED_MPS = 1.35
STEPS_FACTOR = 2.0  # stairs take about twice as long per metre
CROSSING_SECONDS = 10.0  # waiting for a gap at a road crossing
ROAD_FACTOR = 1.2  # along a service road or a campus street: no pavement, cars
MAX_SNAP_METERS = 250.0  # further than this from any path is not on this map

Point = tuple[float, float]  # (lat, lon)
_R = 6_371_000.0


@dataclass(frozen=True)
class WalkRoute:
    points: tuple[Point, ...]
    meters: float
    minutes: float
    steps: int = 0  # flights of steps on the way
    crossings: int = 0  # road crossings on the way

    def with_ends(self, a: Point | None, b: Point | None) -> "WalkRoute":
        """The same walk drawn from and to the places' own points (a door is not the middle of a building)."""
        pts = list(self.points)
        if a is not None and (not pts or pts[0] != a):
            pts.insert(0, a)
        if b is not None and pts[-1] != b:
            pts.append(b)
        return WalkRoute(tuple(pts), self.meters, self.minutes, self.steps, self.crossings)


@dataclass(frozen=True)
class _Snap:
    edge: int
    along: float  # metres from the edge's first node
    xy: tuple[float, float]
    offset: float  # metres from the point asked for to the path
    origin: Point


class WalkNetwork:
    def __init__(self, nodes: list[Point], edges: list[tuple[int, int, int, list[Point]]], *,
                 speed_mps: float = DEFAULT_SPEED_MPS, steps_factor: float = STEPS_FACTOR,
                 crossing_seconds: float = CROSSING_SECONDS, road_factor: float = ROAD_FACTOR,
                 doors: dict[str, list[Point]] | None = None, attribution: str = ATTRIBUTION) -> None:
        """`edges` are (node a, node b, kind, the points between them); kinds are FOOT, STEPS, CROSSING, ROAD."""
        if speed_mps <= 0:
            raise ValueError("walking speed must be positive")
        if min(steps_factor, road_factor) < 1 or crossing_seconds < 0:
            raise ValueError("penalties cannot make a walk faster than walking")
        self.speed = speed_mps
        self.doors = doors or {}
        self.attribution = attribution
        self.nodes = list(nodes)
        lats = [p[0] for p in nodes] or [0.0]
        self._lat0, self._lon0 = min(lats), min((p[1] for p in nodes), default=0.0)
        self._k = math.cos(math.radians((min(lats) + max(lats)) / 2))
        self._xy = [self._to_xy(p) for p in nodes]
        self._pts: list[list[tuple[float, float]]] = []  # each edge's shape in metres, both ends included
        self._cum: list[list[float]] = []  # metres along it at each point
        self._ends: list[tuple[int, int]] = []
        self._kind: list[int] = []
        self._rate: list[float] = []  # seconds per metre, penalties folded in
        self._adj: list[list[tuple[int, int]]] = [[] for _ in nodes]
        for a, b, kind, mid in edges:
            if not (0 <= a < len(nodes) and 0 <= b < len(nodes)):
                raise ValueError(f"edge {a}-{b} uses a node that does not exist")
            shape = [self._xy[a], *(self._to_xy(p) for p in mid), self._xy[b]]
            cum = [0.0]
            for p, q in zip(shape, shape[1:]):
                cum.append(cum[-1] + math.hypot(q[0] - p[0], q[1] - p[1]))
            if cum[-1] <= 0:
                continue
            factor = steps_factor if kind == STEPS else road_factor if kind == ROAD else 1.0
            seconds = cum[-1] / speed_mps * factor + (crossing_seconds if kind == CROSSING else 0.0)
            i = len(self._ends)
            self._pts.append(shape)
            self._cum.append(cum)
            self._ends.append((a, b))
            self._kind.append(kind)
            self._rate.append(seconds / cum[-1])
            self._adj[a].append((b, i))
            self._adj[b].append((a, i))
        self._snaps: dict[Point, _Snap | None] = {}
        self.path: Path | None = None  # the map file this came from, when it came from one

    # --- loading ---

    @classmethod
    def from_map(cls, data: dict, **kw) -> "WalkNetwork":
        """From a map file's content (see scripts/build_campus_pack.py for the format)."""
        walk = data.get("walk")
        if not isinstance(walk, dict):
            raise ValueError("this map file has no walk network")
        lat0, lon0 = data["origin"]
        unit = data.get("unit", 1e-6)

        def pairs(flat: list[int]) -> list[Point]:
            return [(lat0 + flat[i] * unit, lon0 + flat[i + 1] * unit) for i in range(0, len(flat) - 1, 2)]

        edges = [(e[0], e[1], e[2], pairs(e[3]) if len(e) > 3 else []) for e in walk["edges"]]
        doors = {pid: pairs(p["doors"]) for pid, p in (data.get("places") or {}).items() if p.get("doors")}
        return cls(pairs(walk["nodes"]), edges, doors=doors, attribution=data.get("attribution", ATTRIBUTION), **kw)

    @classmethod
    def load(cls, path: str | Path, **kw) -> "WalkNetwork":
        """Read a map file once; the same file with the same settings gives the same object back."""
        p = Path(path).resolve()
        key = (str(p), p.stat().st_mtime_ns, tuple(sorted(kw.items())))
        if key not in _LOADED:
            _LOADED[key] = cls.from_map(json.loads(p.read_text(encoding="utf-8")), **kw)
            _LOADED[key].path = p
        return _LOADED[key]

    # --- geometry ---

    def _to_xy(self, p: Point) -> tuple[float, float]:
        return math.radians(p[1] - self._lon0) * _R * self._k, math.radians(p[0] - self._lat0) * _R

    def _to_point(self, xy: tuple[float, float]) -> Point:
        return (round(self._lat0 + math.degrees(xy[1] / _R), 6), round(self._lon0 + math.degrees(xy[0] / (_R * self._k)), 6))

    def _at(self, edge: int, along: float) -> tuple[float, float]:
        cum, pts = self._cum[edge], self._pts[edge]
        for i in range(1, len(cum)):
            if along <= cum[i] or i == len(cum) - 1:
                seg = cum[i] - cum[i - 1]
                g = 0.0 if seg <= 0 else min(1.0, max(0.0, (along - cum[i - 1]) / seg))
                return pts[i - 1][0] + (pts[i][0] - pts[i - 1][0]) * g, pts[i - 1][1] + (pts[i][1] - pts[i - 1][1]) * g
        return pts[0]

    def _stretch(self, edge: int, d0: float, d1: float) -> list[tuple[float, float]]:
        """The edge's shape from d0 to d1 metres along it (either direction)."""
        lo, hi = min(d0, d1), max(d0, d1)
        cum, pts = self._cum[edge], self._pts[edge]
        out = [self._at(edge, lo), *(pts[i] for i in range(len(pts)) if lo < cum[i] < hi), self._at(edge, hi)]
        return out if d0 <= d1 else out[::-1]

    def snap(self, point: Point) -> _Snap | None:
        """The nearest spot on any path, or None when the point is not on this map."""
        key = (round(point[0], 6), round(point[1], 6))
        if key not in self._snaps:
            x, y = self._to_xy(point)
            best: tuple[float, int, float, tuple[float, float]] | None = None
            for e, pts in enumerate(self._pts):
                cum = self._cum[e]
                for i in range(1, len(pts)):
                    (ax, ay), (bx, by) = pts[i - 1], pts[i]
                    dx, dy = bx - ax, by - ay
                    seg2 = dx * dx + dy * dy
                    g = 0.0 if seg2 <= 0 else min(1.0, max(0.0, ((x - ax) * dx + (y - ay) * dy) / seg2))
                    px, py = ax + dx * g, ay + dy * g
                    d = math.hypot(x - px, y - py)
                    if best is None or d < best[0]:
                        best = (d, e, cum[i - 1] + (cum[i] - cum[i - 1]) * g, (px, py))
            self._snaps[key] = _Snap(best[1], best[2], best[3], best[0], point) if best and best[0] <= MAX_SNAP_METERS else None
        return self._snaps[key]

    # --- routing ---

    def route(self, src: Point | list[Point], dst: Point | list[Point]) -> WalkRoute:
        """The quickest walk. With several points on a side (a building's doors), the best door is used."""
        starts = [s for p in _as_list(src) if (s := self.snap(p)) is not None]
        goals = [s for p in _as_list(dst) if (s := self.snap(p)) is not None]
        if not starts or not goals:
            raise ValueError("that point is not near any path on this map")
        speed = self.speed
        goal_xy = [g.xy for g in goals]

        def h(xy: tuple[float, float]) -> float:
            return min(math.hypot(xy[0] - g[0], xy[1] - g[1]) for g in goal_xy) / speed

        # Each frontier entry: (estimate, cost so far, tie-break, node or -1 for "arrived", how we got here).
        frontier: list[tuple[float, float, int, int, tuple]] = []
        n = 0
        for si, s in enumerate(starts):
            a, b = self._ends[s.edge]
            base = s.offset / speed
            for node, dist in ((a, s.along), (b, self._cum[s.edge][-1] - s.along)):
                c = base + dist * self._rate[s.edge]
                heapq.heappush(frontier, (c + h(self._xy[node]), c, n, node, ("start", si)))
                n += 1
            for gi, g in enumerate(goals):  # both on the same stretch of path: walk straight along it
                if g.edge == s.edge:
                    c = base + abs(g.along - s.along) * self._rate[s.edge] + g.offset / speed
                    heapq.heappush(frontier, (c, c, n, -1, ("direct", si, gi)))
                    n += 1
        at_goal: dict[int, list[tuple[float, int]]] = {}
        for gi, g in enumerate(goals):
            a, b = self._ends[g.edge]
            tail = g.offset / speed
            at_goal.setdefault(a, []).append((g.along * self._rate[g.edge] + tail, gi))
            at_goal.setdefault(b, []).append(((self._cum[g.edge][-1] - g.along) * self._rate[g.edge] + tail, gi))

        done: dict[int, tuple] = {}
        while frontier:
            _, c, _, node, how = heapq.heappop(frontier)
            if node == -1:
                return self._build(starts, goals, done, how, c)
            if node in done:
                continue
            done[node] = how
            for extra, gi in at_goal.get(node, ()):
                heapq.heappush(frontier, (c + extra, c + extra, n, -1, ("node", node, gi)))
                n += 1
            for other, e in self._adj[node]:
                if other not in done:
                    c2 = c + self._cum[e][-1] * self._rate[e]
                    heapq.heappush(frontier, (c2 + h(self._xy[other]), c2, n, other, ("edge", node, e)))
                    n += 1
        raise ValueError("there is no walking path between those two points")

    def _build(self, starts: list[_Snap], goals: list[_Snap], done: dict[int, tuple], how: tuple, seconds: float) -> WalkRoute:
        steps = crossings = 0
        if how[0] == "direct":
            s, g = starts[how[1]], goals[how[2]]
            line = self._stretch(s.edge, s.along, g.along)
            kinds = [self._kind[s.edge]] if abs(s.along - g.along) > 0.5 else []
        else:
            g = goals[how[2]]
            node = how[1]
            chain: list[tuple[int, int]] = []  # (edge, the node it was entered from), goal end first
            while done[node][0] == "edge":
                _, prev, e = done[node]
                chain.append((e, prev))
                node = prev
            s = starts[done[node][1]]
            first = 0.0 if node == self._ends[s.edge][0] else self._cum[s.edge][-1]
            # A loop edge has the same node at both ends: take the nearer end.
            if self._ends[s.edge][0] == self._ends[s.edge][1]:
                first = 0.0 if s.along <= self._cum[s.edge][-1] - s.along else self._cum[s.edge][-1]
            line = self._stretch(s.edge, s.along, first)
            kinds = [self._kind[s.edge]] if abs(first - s.along) > 0.5 else []
            for e, prev in reversed(chain):
                full = self._cum[e][-1]
                line += self._stretch(e, 0.0, full) if self._ends[e][0] == prev else self._stretch(e, full, 0.0)
                kinds.append(self._kind[e])
            last = 0.0 if how[1] == self._ends[g.edge][0] else self._cum[g.edge][-1]
            if self._ends[g.edge][0] == self._ends[g.edge][1]:
                last = 0.0 if g.along <= self._cum[g.edge][-1] - g.along else self._cum[g.edge][-1]
            line += self._stretch(g.edge, last, g.along)
            if abs(last - g.along) > 0.5:
                kinds.append(self._kind[g.edge])
        for i, k in enumerate(kinds):
            if i == 0 or kinds[i - 1] != k:
                steps += k == STEPS
                crossings += k == CROSSING
        meters = s.offset + g.offset
        clean: list[tuple[float, float]] = []
        for p in line:
            if not clean or math.hypot(p[0] - clean[-1][0], p[1] - clean[-1][1]) > 0.05:
                if clean:
                    meters += math.hypot(p[0] - clean[-1][0], p[1] - clean[-1][1])
                clean.append(p)
        points = [self._to_point(p) for p in clean]
        for end, origin in ((0, s.origin), (-1, g.origin)):
            o = (round(origin[0], 6), round(origin[1], 6))
            if points[end] != o:
                points.insert(0, o) if end == 0 else points.append(o)
        return WalkRoute(tuple(points), round(meters, 1), round(seconds / 60, 2), int(steps), int(crossings))


_LOADED: dict[tuple, WalkNetwork] = {}


def _as_list(p: Point | list[Point]) -> list[Point]:
    return [p] if p and isinstance(p[0], (int, float)) else list(p)  # type: ignore[list-item]


def leave_sentence(depart_clock: str, minutes: float, to_name: str, meters: float | None) -> str:
    """"Leave by 2:41 PM: 9 minutes on foot to Engineering II, 720 m"."""
    whole = max(1, math.ceil(minutes - 1e-9))
    far = f", {_distance(meters)}" if meters else ""
    return f"Leave by {depart_clock}: {whole} minute{'' if whole == 1 else 's'} on foot to {to_name}{far}"


def _distance(meters: float) -> str:
    return f"{meters / 1000:.1f} km" if meters >= 1000 else f"{int(meters / 10 + 0.5) * 10 if meters >= 100 else int(meters + 0.5)} m"
