from contexts.play.domain.pathfinding import SortingRules, ZoneMap, astar, point_in_polygon, route
from contracts.types import Box


def test_point_in_polygon():
    square = [(0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.1, 0.5)]
    assert point_in_polygon(0.3, 0.3, square) and not point_in_polygon(0.6, 0.3, square)


def test_route_goes_to_the_correct_bin_for_the_item():
    m = ZoneMap.load("cam-door")
    rules = SortingRules.load()
    zone, path = route(m, rules, "can", (0.75, 0.75))  # standing in the door zone
    assert zone == "recycling"
    end = path[-1]
    assert (end.row, end.col) in set(m.cells_in("recycling"))
    zone, _ = route(m, rules, "cup", (0.75, 0.75))
    assert zone == "trash"
    assert rules.bin_for("mystery") == "trash"


def test_obstacles_force_a_detour_and_paths_are_shortest():
    m = ZoneMap({"bin": [(0.0, 0.0), (0.1, 0.0), (0.1, 0.1), (0.0, 0.1)]}, rows=10, cols=10)
    straight = astar(m, (0, 9), set(m.cells_in("bin")))
    assert len(straight) == 10  # (0,9) -> (0,0) along the top row
    wall = m.with_obstacles([Box(x1=0.45, y1=0.0, x2=0.55, y2=0.85)])  # a wall with a gap at the bottom
    detour = astar(wall, (0, 9), set(wall.cells_in("bin")))
    assert detour and len(detour) > len(straight)
    assert not any(cell in wall.blocked for cell in detour)
    sealed = m.with_obstacles([Box(x1=0.45, y1=0.0, x2=0.55, y2=1.0)])
    assert astar(sealed, (0, 9), set(sealed.cells_in("bin"))) is None
