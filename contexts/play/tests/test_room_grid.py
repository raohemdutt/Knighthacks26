"""Sort It pathfinding on the floor plan: around furniture, around people, re-planned when the way is blocked."""

import math

from contexts.play.room.grid import FloorGrid, astar, blocked_ahead, length, off_route, plan_route
from contexts.play.room.plan import load_room, parse_room


def dorm():
    return load_room("config/rooms/dorm.yaml")


def box_room(furniture=()):
    """Four metres square, with whatever is put in it."""
    return parse_room({"name": "box", "outline": [[0, 0], [4, 0], [4, 4], [0, 4]], "furniture": list(furniture)})


WALL = {"id": "wall", "type": "shelf", "at": [2.0, 1.4], "size": [0.3, 2.8], "height": 1.5}  # from the north wall down to y = 2.8
BIN = {"id": "bin", "type": "bin", "bin": "recycling", "at": [3.5, 0.5], "size": [0.36, 0.32], "height": 0.5}


def clear_of_furniture(plan, points, radius):
    """Every point of the walk (sampled every 5 cm) keeps the body radius from every piece of furniture."""
    for a, b in zip(points, points[1:], strict=False):
        steps = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) / 0.05))
        for i in range(steps + 1):
            x, y = a[0] + (b[0] - a[0]) * i / steps, a[1] + (b[1] - a[1]) * i / steps
            for f in plan.furniture:
                if f.blocks and f.distance(x, y) < radius - 0.08:  # a cell of slack: the grid is 10 cm
                    return False
    return True


def test_grid_blocks_furniture_grown_by_the_body_radius_but_not_the_rug():
    plan = dorm()
    grid = FloorGrid.of(plan)
    assert (grid.cols, grid.rows, grid.cell) == (42, 44, 0.1)
    assert grid.blocked(grid.cell_of(0.4, 2.0))  # the desk
    assert grid.blocked(grid.cell_of(0.9, 2.6))  # 0.18 m off the desk's edge: a body does not fit
    assert not grid.blocked(grid.cell_of(1.5, 2.9))  # open floor
    assert not grid.blocked(grid.cell_of(2.1, 2.45))  # the rug is walked over
    assert grid.blocked(grid.cell_of(0.05, 3.0))  # shoulders do not go through the wall


def test_route_goes_around_a_wall_of_furniture():
    plan = box_room([WALL, BIN])
    grid = FloorGrid.of(plan)
    route = plan_route(grid, (0.5, 0.5), (3.5, 0.5), 0.18, walk_speed=1.0)
    assert route is not None and clear_of_furniture(plan, route.points, plan.body_radius)
    assert max(y for _, y in route.points) > 2.8  # down past the end of the shelf and back up
    assert route.distance_m > 5.0 > math.hypot(3.0, 0.0)  # much longer than the straight line it could not take
    assert abs(route.par_s - (1.5 + route.distance_m / 1.0 + 1.0)) < 1e-9  # setting off, walking speed, the drop
    assert abs(length(route.points) - route.distance_m) < 1e-9
    assert len(route.points) <= 6  # smoothed: a few corners, not a staircase of cells
    assert route.points[0] == (0.5, 0.5)
    end = route.points[-1]
    assert math.hypot(end[0] - 3.5, end[1] - 0.5) <= 0.18 + 0.35 + 0.1  # within reach of the bin


def test_route_in_the_dorm_keeps_clear_of_the_chair_and_the_desk():
    plan = dorm()
    grid = FloorGrid.of(plan)
    target = plan.bins()["recycling"]
    route = plan_route(grid, plan.start, target.at, max(target.size) / 2, plan.walk_speed)
    assert route is not None and clear_of_furniture(plan, route.points, plan.body_radius)
    straight = math.hypot(plan.start[0] - target.at[0], plan.start[1] - target.at[1])
    assert route.distance_m > straight - 0.6  # ends a reach short of the bin
    assert max(x for x, _ in route.points) > 1.5  # east of the chair, which stands in the straight line


def test_a_walker_sitting_in_a_chair_starts_from_the_nearest_free_cell():
    plan = dorm()
    grid = FloorGrid.of(plan)
    chair = next(f for f in plan.furniture if f.id == "chair")
    assert grid.blocked(grid.cell_of(*chair.at))
    route = plan_route(grid, chair.at, plan.bins()["trash"].at, 0.18, plan.walk_speed)
    assert route is not None and route.points[0] == chair.at and len(route.points) >= 3


def test_unreachable_when_the_bin_is_walled_in():
    shut = {"id": "shut", "type": "wardrobe", "at": [2.0, 2.0], "size": [0.3, 4.0], "height": 2}  # wall to wall
    grid = FloorGrid.of(box_room([shut, BIN]))
    assert plan_route(grid, (0.5, 0.5), (3.5, 0.5), 0.18) is None
    assert astar(grid, grid.cell_of(0.5, 0.5), set()) is None


def test_a_person_stepping_into_the_way_blocks_the_route_and_the_new_one_goes_around():
    plan = box_room([BIN])
    grid = FloorGrid.of(plan)
    first = plan_route(grid, (0.5, 0.5), (3.5, 0.5), 0.18)
    assert first is not None and max(y for _, y in first.points) < 1.0  # nothing in the way: straight along the wall
    assert not blocked_ahead(grid, first, (0.5, 0.5))

    person = (2.0, 0.6, 0.55)  # someone stands on the line
    moved = grid.with_obstacles([person])
    assert blocked_ahead(moved, first, (0.5, 0.5))
    assert not blocked_ahead(moved, first, (3.0, 0.5))  # already past them: nothing ahead is blocked
    second = plan_route(moved, (0.5, 0.5), (3.5, 0.5), 0.18)
    assert second is not None and second.distance_m > first.distance_m
    assert all(math.hypot(x - 2.0, y - 0.6) >= 0.5 for x, y in second.points)
    assert max(y for _, y in second.points) > 1.1  # the wall is on the other side, so around the south of them

    assert grid.dynamic == frozenset()  # the room itself is unchanged: the next plan starts clean


def test_off_route_distance():
    grid = FloorGrid.of(box_room([BIN]))
    route = plan_route(grid, (0.5, 2.0), (3.5, 0.5), 0.18)
    assert off_route(route, (0.5, 2.0)) < 0.01
    assert off_route(route, (0.5, 3.5)) > 1.4
