"""A walking route on the plant floor, over the only paths the data vouches for: the legs SPOT has walked.

    net = WalkNet.of(kit)
    net.route("WP01", "WP05") -> {"path": ["WP01", "WP04", "WP05"], "legs": [...], "length": 78.4, "floor_changes": 0}

What is known and what is not:
- The kit gives every waypoint a floor and an x/y "for visualization" (the data dictionary's words). It gives no walls,
  doors or corridors. So nothing here is read off the plan picture.
- Every round is a sequence of actions, each at a waypoint. When two actions in a row are at different waypoints, SPOT
  got from one to the other: that leg is walkable. The network is exactly those legs, taken as walkable both ways (a
  robot that walked A to B can be walked back), each with how many times it was walked.
- A leg between floors exists only where a round shows SPOT changing floor.
- Length is the straight distance between the two waypoints' plan coordinates: plan units, not metres. A change of
  floor adds a fixed cost so a route does not hop floors to save a few units.

Pure: no files, no clock. Dijkstra over at most a few dozen nodes.
"""

import heapq
import math
from dataclasses import dataclass, field

from contexts.spot.domain.models import Kit

FLOOR_CHANGE_COST = 40.0  # plan units; about half the width of a floor


@dataclass
class WalkNet:
    xy: dict[str, tuple[float, float]]
    floor: dict[str, int]
    walked: dict[tuple[str, str], int] = field(default_factory=dict)  # (a, b) with a < b -> times SPOT walked it, either way

    @classmethod
    def of(cls, kit: Kit) -> "WalkNet":
        net = cls(xy={x.location_id: (x.x_coord, x.y_coord) for x in kit.locations}, floor={x.location_id: x.floor for x in kit.locations})
        for mission in kit.missions:
            before = None
            for a in kit.actions_of.get(mission.mission_id, []):
                here = a.location_id
                if before and here != before and here in net.xy and before in net.xy:
                    key = (min(before, here), max(before, here))
                    net.walked[key] = net.walked.get(key, 0) + 1
                if here in net.xy:
                    before = here
        return net

    def length(self, a: str, b: str) -> float:
        (ax, ay), (bx, by) = self.xy[a], self.xy[b]
        return math.hypot(bx - ax, by - ay)

    def neighbours(self, a: str) -> list[str]:
        return sorted({y if x == a else x for (x, y) in self.walked if a in (x, y)})

    def route(self, start: str, goal: str) -> dict | None:
        """The shortest way over walked legs, or None when the two are not joined by any (or either is unknown)."""
        if start not in self.xy or goal not in self.xy:
            return None
        best = {start: 0.0}
        back: dict[str, str] = {}
        frontier = [(0.0, start)]
        while frontier:
            cost, here = heapq.heappop(frontier)
            if here == goal:
                break
            if cost > best.get(here, math.inf):
                continue
            for nxt in self.neighbours(here):
                step = cost + self.length(here, nxt) + (FLOOR_CHANGE_COST if self.floor[here] != self.floor[nxt] else 0.0)
                if step < best.get(nxt, math.inf):
                    best[nxt] = step
                    back[nxt] = here
                    heapq.heappush(frontier, (step, nxt))
        if goal not in best:
            return None
        path = [goal]
        while path[-1] != start:
            path.append(back[path[-1]])
        path.reverse()
        legs = [{"from": a, "to": b, "length": round(self.length(a, b), 1), "floor_change": self.floor[a] != self.floor[b],
                 "times_walked": self.walked[(min(a, b), max(a, b))]} for a, b in zip(path, path[1:])]
        return {"path": path, "legs": legs, "length": round(sum(leg["length"] for leg in legs), 1),
                "floor_changes": sum(leg["floor_change"] for leg in legs)}


def heading(a: tuple[float, float], b: tuple[float, float]) -> str:
    """Which way a leg runs on the plan as it is drawn (y grows downwards): 'right', 'up and to the left', ..."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    across = "right" if dx > 0 else "left"
    along = "down" if dy > 0 else "up"
    if abs(dx) >= 2.2 * abs(dy):
        return across
    if abs(dy) >= 2.2 * abs(dx):
        return along
    return f"{along} and to the {across}"
