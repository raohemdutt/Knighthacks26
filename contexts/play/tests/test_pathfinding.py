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


def test_aliases_sort_like_their_item_and_unknown_items_are_not_routed():
    """Round 2 (T9): "soda can" and "water bottle" went to trash, and a unicorn got a route."""
    rules = SortingRules.load("config/sorting.yaml")
    assert rules.bin_for("soda can") == "recycling" and rules.bin_for("water bottle") == "recycling"
    assert rules.canonical("soda can") == "can" and rules.canonical("unicorn") is None
    assert rules.bin_for("unicorn") == "trash"  # Waste Watch still counts any drop


def test_route_never_crosses_the_other_bin():
    """Round 2 (T10): the route to recycling was drawn through the trash bin's zone."""
    zones = {"trash": [(0.4, 0.0), (0.6, 0.0), (0.6, 0.9), (0.4, 0.9)],  # a wall of trash bin between start and goal
             "recycling": [(0.0, 0.0), (0.2, 0.0), (0.2, 0.2), (0.0, 0.2)]}
    m = ZoneMap(zones, rows=10, cols=10)
    rules = SortingRules(bins={"recycling": "recycling", "trash": "trash"}, items={"can": "recycling"})
    zone, path = route(m, rules, "can", (0.85, 0.1))
    trash = set(m.cells_in("trash"))
    assert zone == "recycling" and path and not any((c.row, c.col) in trash for c in path)
    assert max(c.row for c in path) == 9  # it walks around the bottom of the trash zone
