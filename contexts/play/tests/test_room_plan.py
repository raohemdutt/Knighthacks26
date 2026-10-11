"""The room twin's maths and config: homography from four taps, projection and clamping, the room files."""

import shutil
from pathlib import Path

import pytest
import yaml

from contexts.perception.domain.zones import load_zones
from contexts.play.room.homography import Degenerate, apply, homography, invert
from contexts.play.room.plan import Calibration, RoomError, available, load_room, parse_room, room_path, save_calibration

SQUARE = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def close(a, b, tol=1e-9):
    return all(abs(x - y) <= tol for x, y in zip(a, b, strict=True))


def test_identity_homography():
    h = homography(SQUARE, SQUARE)
    assert all(abs(h[r][c] - (1.0 if r == c else 0.0)) < 1e-12 for r in range(3) for c in range(3))
    assert close(apply(h, 0.3, 0.8), (0.3, 0.8))


def test_a_known_perspective_case_maps_the_four_points_and_bends_the_middle():
    # a camera looking along a floor: the far edge of the picture shows four metres of wall, the near edge two
    floor = [(0.0, 0.0), (4.0, 0.0), (3.0, 2.0), (1.0, 2.0)]
    h = homography(SQUARE, floor)
    for src, dst in zip(SQUARE, floor, strict=True):
        assert close(apply(h, *src), dst, 1e-9)
    # solved by hand: x2 = (4x + 2y) / (1 + y), y2 = 4y / (1 + y); the centre of the picture is not the middle of the trapezoid
    assert close(apply(h, 0.5, 0.5), (2.0, 4 / 3), 1e-9)
    assert close(apply(h, 0.25, 0.5), (4 / 3, 4 / 3), 1e-9)
    assert abs(h[2][1] - 1.0) < 1e-9 and abs(h[2][0]) < 1e-9  # a true perspective term, not an affine map
    back = invert(h)
    assert close(apply(back, 2.0, 4 / 3), (0.5, 0.5), 1e-9)
    assert close(apply(back, *apply(h, 0.9, 0.1)), (0.9, 0.1), 1e-9)


def test_a_horizon_through_the_pictures_corner_still_solves():
    # floor seen in the lower half of the picture, twice as wide at the far edge: the horizon passes through (0, 0),
    # so the usual "h33 = 1" form does not exist. The solve is centred on the taps, and the map is still exact.
    image = [(0.0, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0)]
    floor = [(0.0, 0.0), (4.0, 0.0), (3.0, 2.0), (1.0, 2.0)]
    h = homography(image, floor)
    for src, dst in zip(image, floor, strict=True):
        assert close(apply(h, *src), dst, 1e-9)
    assert abs(h[2][2]) < 1e-9 and apply(h, 0.3, 0.0) is None


def test_degenerate_points_are_rejected():
    with pytest.raises(Degenerate, match="in a line"):
        homography([(0, 0), (0.5, 0), (1, 0), (0, 1)], SQUARE)  # three taps along one edge
    with pytest.raises(Degenerate, match="in a line"):
        homography(SQUARE, [(0, 0), (1, 1), (2, 2), (3, 0)])
    with pytest.raises(Degenerate, match="in a line"):
        homography([(0, 0), (0, 0), (1, 1), (0, 1)], SQUARE)  # the same spot tapped twice
    with pytest.raises(Degenerate, match="four points"):
        homography(SQUARE[:3], SQUARE[:3])


def test_a_point_on_the_horizon_has_no_floor_position():
    h = homography(SQUARE, [(0.0, 0.0), (4.0, 0.0), (3.0, 2.0), (1.0, 2.0)])
    assert apply(h, 0.5, -1.0) is None  # 1 + y = 0


def test_the_dorm_loads_with_its_furniture_bins_and_calibration():
    plan = load_room("config/rooms/dorm.yaml")
    assert (plan.name, plan.width, plan.depth) == ("dorm", 4.2, 4.4)
    types = {f.type for f in plan.furniture}
    assert {"bed", "desk", "chair", "wardrobe", "shelf", "fridge", "rug", "bin"} <= types
    assert set(plan.bins()) == {"recycling", "trash"} and plan.bins()["recycling"].color == "blue"
    assert set(plan.calibration) == {"cam-door", "cam-desk"}
    assert plan.doors[0]["wall"] == "north" and plan.windows[0]["wall"] == "west"
    desk = next(f for f in plan.furniture if f.id == "desk")
    assert desk.rot == 270 and desk.contains(0.1, 1.4) and desk.contains(0.7, 2.6) and not desk.contains(0.8, 2.0)  # 1.4 wide along the wall
    assert sorted(round(v, 3) for p in desk.corners() for v in p) == sorted([0.075, 1.3, 0.725, 1.3, 0.725, 2.7, 0.075, 2.7])
    assert plan.furniture_at(0.4, 2.0).id == "desk" and plan.furniture_at(2.1, 2.4) is None  # a rug is not furniture to stand on


def test_the_fake_scene_lands_where_the_room_says():
    plan = load_room("config/rooms/dorm.yaml")
    person = plan.to_floor("cam-desk", 0.5, 0.9)  # seated at the desk
    chair = next(f for f in plan.furniture if f.id == "chair")
    assert chair.contains(*person, grow=0.12)
    keys = plan.to_floor("cam-desk", 0.3, 0.7)
    assert plan.furniture_at(*keys).id == "desk"
    package = plan.to_floor("cam-door", 0.8, 0.9)
    door = plan.doors[0]
    assert abs(package[0] - door["at"]) < door["width"] / 2 and package[1] < 1.6  # inside the door, on the floor
    bins = plan.bins()
    assert abs(plan.to_floor("cam-door", 0.15, 0.8)[0] - bins["recycling"].at[0]) < 0.05
    assert abs(plan.to_floor("cam-door", 0.38, 0.8)[0] - bins["trash"].at[0]) < 0.05


def test_projection_is_clamped_to_the_room_and_inverts():
    plan = load_room("config/rooms/dorm.yaml")
    assert plan.to_floor("cam-door", 0.5, 0.2) == (plan.to_floor("cam-door", 0.5, 0.2)[0], 0.05)  # above the floor line: at the wall
    assert plan.to_floor("cam-door", 0.5, 0.2, clamp=False)[1] < 0
    assert close(plan.clamp(-3, 99), (0.05, 4.35), 1e-9)
    assert plan.to_floor("no-such-camera", 0.5, 0.5) is None
    x, y = plan.to_floor("cam-desk", 0.3, 0.7)
    assert close(plan.to_image("cam-desk", x, y), (0.3, 0.7), 1e-9)


def test_zones_are_laid_on_the_floor():
    plan = load_room("config/rooms/dorm.yaml")
    zones = plan.zones_on_floor(load_zones("config/zones.yaml"))
    assert set(zones) == {"cam-door", "cam-desk"} and set(zones["cam-door"]) == {"door", "recycling", "trash"}
    for name in ("recycling", "trash"):
        xs = [p[0] for p in zones["cam-door"][name]]
        assert min(xs) < plan.bins()[name].at[0] < max(xs)  # each bin stands in its own zone
    assert all(0 <= x <= plan.width and 0 <= y <= plan.depth for cam in zones.values() for poly in cam.values() for x, y in poly)
    assert plan.zones_on_floor({"cam-other": {"z": [[0, 0], [1, 0], [1, 1]]}}) == {}  # no calibration: not drawn


def test_the_table_room_is_a_second_layout_for_the_same_cameras():
    plan = load_room("config/rooms/table.yaml")
    assert {f.type for f in plan.furniture} == {"table", "chair", "bin"} and set(plan.bins()) == {"recycling", "trash"}
    assert plan.furniture_at(*plan.to_floor("cam-desk", 0.3, 0.7)).id == "table"
    assert available("config") == ["dorm", "table"]


def test_macos_metadata_files_are_not_rooms(tmp_path):
    (tmp_path / "rooms").mkdir()
    shutil.copy("config/rooms/dorm.yaml", tmp_path / "rooms" / "dorm.yaml")
    (tmp_path / "rooms" / "._dorm.yaml").write_bytes(b"\x00\x05\x16\x07")
    assert available(tmp_path) == ["dorm"]


def test_room_path_picks_the_room(monkeypatch):
    monkeypatch.delenv("ATLAS_ROOM", raising=False)
    assert room_path("config").name == "dorm.yaml"
    assert room_path("config", "table").name == "table.yaml"
    monkeypatch.setenv("ATLAS_ROOM", "table")
    assert room_path("config").name == "table.yaml"
    assert room_path("config", "nope") is None and room_path("config", "../agent") is None


def test_bad_room_files_say_what_is_wrong():
    base = yaml.safe_load(Path("config/rooms/dorm.yaml").read_text())
    with pytest.raises(RoomError, match="door on the north wall does not fit"):
        parse_room({**base, "doors": [{"wall": "north", "at": 4.0, "width": 0.9}]})
    with pytest.raises(RoomError, match="outside the room"):
        parse_room({**base, "furniture": [{"id": "x", "type": "bed", "at": [9, 9], "size": [1, 2]}]})
    with pytest.raises(RoomError, match="calibration for cam-door"):
        parse_room({**base, "calibration": {"cam-door": {"image": [[0, 0], [0.5, 0], [1, 0], [0, 1]], "floor": [[0, 0], [1, 0], [1, 1], [0, 1]]}}})
    with pytest.raises(RoomError, match="used twice"):
        parse_room({**base, "furniture": base["furniture"][:1] * 2})


def test_saving_a_calibration_rewrites_only_that_block(tmp_path):
    path = tmp_path / "dorm.yaml"
    shutil.copy("config/rooms/dorm.yaml", path)
    before = path.read_text()
    plan = load_room(path)
    plan.calibration = {**plan.calibration, "cam-new": Calibration.of([[0, 0.5], [1, 0.5], [1, 1], [0, 1]], [[0, 0], [4, 0], [3, 2], [1, 2]])}
    save_calibration(plan)
    after = path.read_text()
    head = before[:before.index("calibration:")]
    assert after.startswith(head) and "# The demo cameras." in after  # the plan and its comments are untouched
    again = load_room(path)
    assert set(again.calibration) == {"cam-door", "cam-desk", "cam-new"}
    assert again.calibration["cam-new"].floor == [(0, 0), (4, 0), (3, 2), (1, 2)]
    assert again.furniture == plan.furniture
