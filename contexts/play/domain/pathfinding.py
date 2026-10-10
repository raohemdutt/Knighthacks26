"""Sort It pathfinding (S2): A* on a grid built from the camera's zones.

The camera view is split into a grid. Bin zones are goals, tracked objects
are obstacles, and A* finds the shortest route from where the player stands
to the nearest bin that is correct for the item. The phone draws the cells.
"""

import heapq
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from contracts.play import Cell
from contracts.types import Box

Polygon = list[tuple[float, float]]


def point_in_polygon(x: float, y: float, poly: Polygon) -> bool:
    inside = False
    j = len(poly) - 1
    for i, (xi, yi) in enumerate(poly):
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


@dataclass
class ZoneMap:
    zones: dict[str, Polygon]
    rows: int = 24
    cols: int = 32
    blocked: set[tuple[int, int]] = field(default_factory=set)

    @classmethod
    def load(cls, camera: str, path: str | Path = "config/zones.yaml", rows: int = 24, cols: int = 32) -> "ZoneMap":
        raw = yaml.safe_load(Path(path).read_text())[camera]
        return cls({z: [tuple(p) for p in poly] for z, poly in raw.items()}, rows, cols)

    def center(self, r: int, c: int) -> tuple[float, float]:
        return (c + 0.5) / self.cols, (r + 0.5) / self.rows

    def cells_in(self, zone: str) -> list[tuple[int, int]]:
        poly = self.zones[zone]
        return [(r, c) for r in range(self.rows) for c in range(self.cols) if point_in_polygon(*self.center(r, c), poly)]

    def cell_of(self, x: float, y: float) -> tuple[int, int]:
        return min(int(y * self.rows), self.rows - 1), min(int(x * self.cols), self.cols - 1)

    def with_obstacles(self, boxes: list[Box]) -> "ZoneMap":
        """Tracked objects (chairs, bags, people) block the cells their boxes cover."""
        blocked = {
            (r, c)
            for b in boxes
            for r in range(self.rows)
            for c in range(self.cols)
            if b.x1 <= self.center(r, c)[0] <= b.x2 and b.y1 <= self.center(r, c)[1] <= b.y2
        }
        return ZoneMap(self.zones, self.rows, self.cols, blocked)


def astar(m: ZoneMap, start: tuple[int, int], goals: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
    """Shortest 8-connected path to any goal cell; diagonal steps cost sqrt(2)."""
    if not goals:
        return None

    def h(cell):  # octile distance to the nearest goal: admissible
        return min(max(abs(cell[0] - g[0]), abs(cell[1] - g[1])) + 0.414 * min(abs(cell[0] - g[0]), abs(cell[1] - g[1])) for g in goals)

    frontier = [(h(start), 0.0, start)]
    came: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    cost = {start: 0.0}
    while frontier:
        _, g, cell = heapq.heappop(frontier)
        if cell in goals:
            path = [cell]
            while came[path[-1]] is not None:
                path.append(came[path[-1]])
            return path[::-1]
        if g > cost[cell]:
            continue
        r, c = cell
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            nxt = (r + dr, c + dc)
            if not (0 <= nxt[0] < m.rows and 0 <= nxt[1] < m.cols) or (nxt in m.blocked and nxt not in goals):
                continue
            step = 1.414 if dr and dc else 1.0
            if g + step < cost.get(nxt, float("inf")):
                cost[nxt] = g + step
                came[nxt] = cell
                heapq.heappush(frontier, (g + step + h(nxt), g + step, nxt))
    return None


@dataclass
class SortingRules:
    bins: dict[str, str]  # bin -> zone id
    items: dict[str, str]  # item label -> bin
    aliases: dict[str, list[str]] = field(default_factory=dict)  # item -> other detector labels for the same thing

    @classmethod
    def load(cls, path: str | Path = "config/sorting.yaml") -> "SortingRules":
        raw = yaml.safe_load(Path(path).read_text())
        return cls(raw["bins"], raw["items"], raw.get("aliases") or {})

    def same_item(self, label: str, item: str) -> bool:
        """Is a tracked `label` the round's `item`? Exact, or one of the item's configured aliases."""
        return label == item or label in self.aliases.get(item, [])

    def canonical(self, label: str) -> str | None:
        """The listed item a label names: itself, or the item it is an alias of ("soda can" -> "can"); None if unknown."""
        if label in self.items:
            return label
        return next((item for item, names in self.aliases.items() if label in names and item in self.items), None)

    def bin_for(self, label: str) -> str:
        """Aliases sort like their item; anything not listed is trash (Waste Watch counts every drop)."""
        return self.items.get(self.canonical(label) or label, "trash")

    def zone_for(self, label: str) -> str:
        return self.bins[self.bin_for(label)]


def route(m: ZoneMap, rules: SortingRules, item: str, from_xy: tuple[float, float]) -> tuple[str, list[Cell]]:
    """Target bin zone and the cells to walk, from a normalized point (the player's feet).

    The other bins are obstacles: the route never crosses the wrong bin to reach the right one."""
    zone = rules.zone_for(item)
    goals = set(m.cells_in(zone))
    others = {cell for z in set(rules.bins.values()) - {zone} if z in m.zones for cell in m.cells_in(z)}
    m = ZoneMap(m.zones, m.rows, m.cols, (m.blocked | others) - goals)
    path = astar(m, m.cell_of(*from_xy), goals)
    return zone, [Cell(row=r, col=c) for r, c in (path or [])]
