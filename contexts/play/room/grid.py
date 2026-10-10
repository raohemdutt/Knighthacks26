"""Sort It pathfinding in the room: A* on a floor grid (cells of about 10 cm).

    furniture footprints, grown by a body radius   -> cells nobody can stand in
    other people, things left on the floor          -> discs that block for this plan only
    A* (8 neighbours, no cutting a corner)          -> cells from the walker to the bin's approach cells
    line-of-sight pulling                           -> a short polyline in metres, its length, and a par time

The existing image-space A* (contexts/play/domain/pathfinding.py) is untouched; this one works on the floor plan.
"""

import heapq
import math
from dataclasses import dataclass, field

from contexts.play.room.plan import RoomPlan

Cell = tuple[int, int]  # row (y), col (x)
Point = tuple[float, float]
Disc = tuple[float, float, float]  # x, y, radius in metres

CELL_M = 0.1
REACH_M = 0.35  # how close to a bin's edge a walker has to get to drop something in
DROP_S = 1.0  # par allows this long for the drop itself
SQRT2 = math.sqrt(2)


@dataclass
class FloorGrid:
    width: float
    depth: float
    cell: float
    cols: int
    rows: int
    static: frozenset[Cell]  # furniture and the strip along the walls, grown by the body radius
    dynamic: frozenset[Cell] = field(default_factory=frozenset)

    @classmethod
    def of(cls, plan: RoomPlan, cell: float = CELL_M, body_radius: float | None = None) -> "FloorGrid":
        r = plan.body_radius if body_radius is None else body_radius
        cols, rows = max(1, round(plan.width / cell)), max(1, round(plan.depth / cell))
        blocked = set()
        solid = [f for f in plan.furniture if f.blocks]
        for row in range(rows):
            for col in range(cols):
                x, y = (col + 0.5) * cell, (row + 0.5) * cell
                near_wall = min(x, y, plan.width - x, plan.depth - y) < r * 0.6  # shoulders do not go through walls
                if near_wall or any(f.distance(x, y) < r for f in solid):
                    blocked.add((row, col))
        return cls(plan.width, plan.depth, cell, cols, rows, frozenset(blocked))

    def cell_of(self, x: float, y: float) -> Cell:
        return min(max(int(y / self.cell), 0), self.rows - 1), min(max(int(x / self.cell), 0), self.cols - 1)

    def center(self, c: Cell) -> Point:
        return (c[1] + 0.5) * self.cell, (c[0] + 0.5) * self.cell

    def inside(self, c: Cell) -> bool:
        return 0 <= c[0] < self.rows and 0 <= c[1] < self.cols

    def blocked(self, c: Cell) -> bool:
        return c in self.static or c in self.dynamic

    def cells_within(self, x: float, y: float, radius: float) -> set[Cell]:
        r0, c0 = self.cell_of(x, y)
        n = int(radius / self.cell) + 1
        return {(r, c) for r in range(max(r0 - n, 0), min(r0 + n + 1, self.rows))
                for c in range(max(c0 - n, 0), min(c0 + n + 1, self.cols))
                if math.hypot((c + 0.5) * self.cell - x, (r + 0.5) * self.cell - y) <= radius}

    def with_obstacles(self, discs: list[Disc]) -> "FloorGrid":
        """The same room with these discs (people, a box on the floor) in the way as well."""
        cells: set[Cell] = set()
        for x, y, radius in discs:
            cells |= self.cells_within(x, y, radius)
        return FloorGrid(self.width, self.depth, self.cell, self.cols, self.rows, self.static, frozenset(cells))

    def nearest_free(self, c: Cell, limit: int = 12) -> Cell | None:
        """The closest cell someone can stand in (the walker may be sitting in a chair, or pressed to a wall)."""
        if not self.blocked(c):
            return c
        best, best_d = None, float("inf")
        for r in range(c[0] - limit, c[0] + limit + 1):
            for col in range(c[1] - limit, c[1] + limit + 1):
                if self.inside((r, col)) and not self.blocked((r, col)):
                    d = math.hypot(r - c[0], col - c[1])
                    if d < best_d:
                        best, best_d = (r, col), d
        return best

    def clear_line(self, a: Point, b: Point) -> bool:
        """Can someone walk straight from a to b?"""
        steps = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) / (self.cell * 0.4)))
        return not any(self.blocked(self.cell_of(a[0] + (b[0] - a[0]) * i / steps, a[1] + (b[1] - a[1]) * i / steps))
                       for i in range(steps + 1))


def astar(grid: FloorGrid, start: Cell, goals: set[Cell]) -> list[Cell] | None:
    """Shortest 8-connected path from `start` to any goal cell; a diagonal step needs both of its sides free."""
    if not goals or grid.blocked(start):
        return None
    gr = sum(g[0] for g in goals) / len(goals)
    gc = sum(g[1] for g in goals) / len(goals)
    spread = max(math.hypot(g[0] - gr, g[1] - gc) for g in goals)

    def h(c: Cell) -> float:  # straight-line distance to the goals' middle, less their spread: never an overestimate
        return max(0.0, math.hypot(c[0] - gr, c[1] - gc) - spread)

    frontier = [(h(start), 0.0, start)]
    came: dict[Cell, Cell | None] = {start: None}
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
            if not grid.inside(nxt) or grid.blocked(nxt):
                continue
            if dr and dc and (grid.blocked((r + dr, c)) or grid.blocked((r, c + dc))):
                continue  # no squeezing past a corner
            step = g + (SQRT2 if dr and dc else 1.0)
            if step < cost.get(nxt, float("inf")):
                cost[nxt] = step
                came[nxt] = cell
                heapq.heappush(frontier, (step + h(nxt), step, nxt))
    return None


def smooth(grid: FloorGrid, points: list[Point]) -> list[Point]:
    """Drop every point that can be walked past in a straight line (string pulling)."""
    if len(points) < 3:
        return points
    out, i = [points[0]], 0
    while i < len(points) - 1:
        j = len(points) - 1
        while j > i + 1 and not grid.clear_line(points[i], points[j]):
            j -= 1
        out.append(points[j])
        i = j
    return out


def length(points: list[Point]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:], strict=False))


@dataclass
class Route:
    points: list[Point]  # metres, from the walker to the approach point of the target
    distance_m: float
    par_s: float
    cells: list[Cell]  # the unsmoothed A* cells, for checking later whether something moved into the way

    def as_json(self) -> dict:
        return {"points": [[round(x, 3), round(y, 3)] for x, y in self.points], "distance_m": round(self.distance_m, 2),
                "par_s": round(self.par_s, 1)}


def approach_cells(grid: FloorGrid, target: Point, footprint_radius: float) -> set[Cell]:
    """Free cells from which the target (a bin) is within reach; the ring grows until there is somewhere to stand."""
    reach = footprint_radius + REACH_M
    while reach <= footprint_radius + 1.5:
        cells = {c for c in grid.cells_within(target[0], target[1], reach) if not grid.blocked(c)}
        if cells:
            return cells
        reach += grid.cell
    return set()


def plan_route(grid: FloorGrid, start: Point, target: Point, footprint_radius: float = 0.25,
               walk_speed: float = 1.2) -> Route | None:
    """The walk from `start` to within reach of `target`, or None when every way is blocked."""
    first = grid.nearest_free(grid.cell_of(*start))
    goals = approach_cells(grid, target, footprint_radius)
    cells = astar(grid, first, goals) if first is not None else None
    if cells is None:
        return None
    points = [grid.center(c) for c in cells]
    if first == grid.cell_of(*start):
        pulled = smooth(grid, [start] + points[1:])
    else:  # the walker's own spot is not free (in a chair, say): out to the first free cell, then the walk
        pulled = [start] + smooth(grid, points)
    d = length(pulled)
    return Route(pulled, d, d / walk_speed + DROP_S, cells)


def blocked_ahead(grid: FloorGrid, route: Route, at: Point, ignore_within: float = 0.45) -> bool:
    """Has something moved onto the part of the route still ahead of the walker at `at`?"""
    if not route.cells:
        return False
    nearest = min(range(len(route.cells)), key=lambda i: math.hypot(grid.center(route.cells[i])[0] - at[0],
                                                                    grid.center(route.cells[i])[1] - at[1]))
    for cell in route.cells[nearest:]:
        x, y = grid.center(cell)
        if math.hypot(x - at[0], y - at[1]) > ignore_within and cell in grid.dynamic:
            return True
    return False


def off_route(route: Route, at: Point) -> float:
    """Metres from `at` to the nearest point of the route."""
    best = float("inf")
    pts = route.points
    for (ax, ay), (bx, by) in zip(pts, pts[1:], strict=False):
        dx, dy = bx - ax, by - ay
        t = 0.0 if dx == dy == 0 else max(0.0, min(1.0, ((at[0] - ax) * dx + (at[1] - ay) * dy) / (dx * dx + dy * dy)))
        best = min(best, math.hypot(at[0] - ax - t * dx, at[1] - ay - t * dy))
    return best if pts[1:] else math.hypot(at[0] - pts[0][0], at[1] - pts[0][1])
