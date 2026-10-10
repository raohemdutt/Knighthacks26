"""Real walking routes: the graph built from OSM ways, A* on it, and the shipped UCF map."""

import importlib.util
import json
import math
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from contexts.play.domain.places import PlaceSet, parse_pack
from contexts.play.domain.walknet import CROSSING, FOOT, ROAD, STEPS, WalkNetwork, leave_sentence

NY = ZoneInfo("America/New_York")
LAT0, LON0 = 28.6, -81.2
M = 111_194.93  # metres per degree of latitude
K = math.cos(math.radians(LAT0))


def pt(x: float, y: float) -> tuple[float, float]:
    """A point x metres east and y metres north of the corner."""
    return (LAT0 + y / M, LON0 + x / (M * K))


def net(bc_kind: int = FOOT, **kw) -> WalkNetwork:
    """Two ways from A to C: straight through B (200 m), or round by D and E (300 m). F-G is an island.

        D ----------- E
        |             |
        A ---- B ---- C        F ---- G
    """
    nodes = [pt(0, 0), pt(100, 0), pt(200, 0), pt(0, 50), pt(200, 50), pt(1000, 0), pt(1100, 0)]
    edges = [(0, 1, FOOT, []), (1, 2, bc_kind, []), (0, 3, FOOT, []), (3, 4, FOOT, [pt(100, 50)]), (4, 2, FOOT, []), (5, 6, FOOT, [])]
    return WalkNetwork(nodes, edges, **kw)


A, B, C, E = pt(0, 0), pt(100, 0), pt(200, 0), pt(200, 50)


# --- A* ---


def test_the_shorter_of_two_ways_is_taken_with_its_line_metres_and_minutes():
    r = net().route(A, C)
    assert r.meters == pytest.approx(200, abs=0.5) and r.minutes == pytest.approx(200 / 1.35 / 60, abs=0.01)
    assert r.points[0] == pytest.approx(A, abs=1e-6) and r.points[-1] == pytest.approx(C, abs=1e-6)
    assert any(p == pytest.approx(B, abs=1e-5) for p in r.points)  # through B, not round by D and E
    assert (r.steps, r.crossings) == (0, 0)
    back = net().route(C, A)
    assert back.meters == r.meters and back.points[0] == pytest.approx(C, abs=1e-6)


def test_walking_speed_is_configurable():
    slow = net(speed_mps=1.0).route(A, C)
    assert slow.minutes == pytest.approx(200 / 60, abs=0.01) and slow.meters == pytest.approx(200, abs=0.5)
    with pytest.raises(ValueError):
        net(speed_mps=0)


def test_steps_are_slower_and_a_long_enough_flight_is_walked_around():
    mild = net(STEPS, steps_factor=1.2).route(A, C)  # 100 m + 100 m of steps at 1.2x: still quicker than 300 m
    assert mild.meters == pytest.approx(200, abs=0.5) and mild.steps == 1
    assert mild.minutes == pytest.approx((100 + 120) / 1.35 / 60, abs=0.01)
    steep = net(STEPS, steps_factor=3.0).route(A, C)  # 100 + 300 against 300: go round
    assert steep.meters == pytest.approx(300, abs=0.5) and steep.steps == 0
    assert any(p == pytest.approx(E, abs=1e-5) for p in steep.points)


def test_a_crossing_costs_a_wait_and_a_road_is_a_little_slower():
    cross = net(CROSSING, crossing_seconds=30).route(A, C)
    assert cross.crossings == 1 and cross.minutes == pytest.approx((200 / 1.35 + 30) / 60, abs=0.01)
    assert net(CROSSING, crossing_seconds=120).route(A, C).meters == pytest.approx(300, abs=0.5)  # not worth the wait
    road = net(ROAD, road_factor=1.2).route(A, C)
    assert road.minutes == pytest.approx((100 + 120) / 1.35 / 60, abs=0.01)


def test_unreachable_and_off_the_map_are_errors_not_guesses():
    with pytest.raises(ValueError, match="no walking path"):
        net().route(A, pt(1100, 0))  # the island
    with pytest.raises(ValueError, match="not near any path"):
        net().route(A, pt(0, 20_000))


def test_points_off_the_paths_join_at_the_nearest_spot_and_that_stretch_counts():
    r = net().route(pt(50, 10), pt(150, -5))  # 10 m to the path, 100 m along it, 5 m off it
    assert r.meters == pytest.approx(115, abs=0.5) and r.minutes == pytest.approx(115 / 1.35 / 60, abs=0.01)
    assert r.points[0] == pytest.approx(pt(50, 10), abs=1e-6) and r.points[-1] == pytest.approx(pt(150, -5), abs=1e-6)
    same = net().route(pt(20, 0), pt(60, 0))  # both on the same stretch
    assert same.meters == pytest.approx(40, abs=0.5)


def test_with_several_doors_the_best_one_is_used():
    r = net().route([A, pt(0, 50)], [pt(200, 50)])  # from the door at D straight along the top
    assert r.meters == pytest.approx(200, abs=0.5) and r.points[0] == pytest.approx(pt(0, 50), abs=1e-6)


def test_the_sentence_says_when_to_leave():
    assert leave_sentence("2:41 PM", 8.4, "Engineering II", 720) == "Leave by 2:41 PM: 9 minutes on foot to Engineering II, 720 m"
    assert leave_sentence("9:00 AM", 0.7, "Kitchen", None) == "Leave by 9:00 AM: 1 minute on foot to Kitchen"
    assert leave_sentence("9:00 AM", 17, "the gym", 1344).endswith("1.3 km")


# --- the map file and the graph built from OSM ways ---


def as_map() -> dict:
    def flat(points):
        return [v for p in points for v in (round((p[0] - LAT0) / 1e-6), round((p[1] - LON0) / 1e-6))]

    return {"origin": [LAT0, LON0], "unit": 1e-6, "attribution": "© OpenStreetMap contributors",
            "walk": {"nodes": flat([pt(0, 0), pt(100, 0), pt(200, 0)]), "edges": [[0, 1, FOOT], [1, 2, STEPS, flat([pt(150, 20)])]]},
            "places": {"hall": {"doors": flat([pt(0, 0)])}, "lab": {"doors": flat([pt(200, 0), pt(100, 0)])}}}


def test_a_map_file_loads_once_and_carries_doors(tmp_path):
    f = tmp_path / "tiny.map.json"
    f.write_text(json.dumps(as_map()), encoding="utf-8")
    a = WalkNetwork.load(f)
    assert WalkNetwork.load(f) is a and a.path == f.resolve()  # read once
    assert WalkNetwork.load(f, speed_mps=1.0) is not a
    assert len(a.doors["lab"]) == 2 and a.attribution == "© OpenStreetMap contributors"
    r = a.route(a.doors["hall"], a.doors["lab"])  # the nearer door of the lab: no need for the steps
    assert r.meters == pytest.approx(100, abs=0.5) and r.steps == 0
    with pytest.raises(ValueError, match="no walk network"):
        WalkNetwork.from_map({"origin": [0, 0]})


def builder():
    spec = importlib.util.spec_from_file_location("build_campus_pack", Path("scripts/build_campus_pack.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_graph_is_built_from_ways_with_nodes_where_they_meet():
    b = builder()

    def ll(x: float, y: float) -> tuple[float, float]:  # metres from the box's corner, as the builder counts them
        return (b.BBOX[0] + y / M, b.BBOX[1] + x / (M * b.K))

    def way(tags: dict, *nodes: tuple[int, float, float]) -> dict:
        return {"type": "way", "tags": tags, "nodes": [n for n, _, _ in nodes],
                "geometry": [dict(zip(("lat", "lon"), ll(x + 500, y + 500))) for _, x, y in nodes]}

    ways = [
        way({"highway": "footway"}, (1, 0, 0), (2, 50, 0), (3, 100, 0), (4, 200, 0)),  # 2 is only a bend; 3 is a junction
        way({"highway": "steps"}, (3, 100, 0), (5, 100, 40)),
        way({"highway": "footway", "footway": "crossing"}, (5, 100, 40), (6, 100, 50)),
        way({"highway": "service"}, (6, 100, 50), (7, 200, 50)),
        way({"highway": "footway"}, (7, 200, 50), (8, 250, 50), (9, 300, 50)),  # same kind does not merge across a kind change at 7
        way({"highway": "footway"}, (20, 900, 900), (21, 950, 900)),  # an island
        way({"highway": "primary"}, (4, 200, 0), (30, 200, -300)),  # not for walking
        way({"highway": "service", "access": "private"}, (4, 200, 0), (31, 300, 0)),
    ]
    nodes, edges, stats = b.build_graph(ways)
    assert stats["islands_dropped_edges"] == 1 and stats["pieces"] == 2
    kinds = sorted(k for _, _, k, _ in edges)
    assert kinds == [FOOT, FOOT, FOOT, STEPS, CROSSING, ROAD] and len(nodes) == 7  # 1, 3, 4, 5, 6, 7, 9: the bends 2 and 8 are not nodes
    net = WalkNetwork([ll(*p) for p in nodes], [(a, c, k, [ll(*p) for p in mid]) for a, c, k, mid in edges])
    r = net.route(ll(500, 500), ll(800, 550))  # node 1 to node 9, up the steps and over the crossing
    assert r.meters == pytest.approx(100 + 40 + 10 + 100 + 100, abs=1) and (r.steps, r.crossings) == (1, 1)

    # Douglas-Peucker keeps the corners and drops the points that say nothing
    line = [(0.0, 0.0), (10.0, 0.1), (20.0, 0.0), (30.0, 10.0), (40.0, 10.05), (50.0, 10.0)]
    assert b.simplify(line, 0.5) == [(0.0, 0.0), (20.0, 0.0), (30.0, 10.0), (50.0, 10.0)]


# --- the shipped UCF map and pack ---


def test_the_shipped_map_is_small_attributed_and_matches_the_pack():
    f = Path("config/places/ucf.map.json")
    assert f.stat().st_size < 600_000
    data = json.loads(f.read_text(encoding="utf-8"))
    assert data["attribution"] == "© OpenStreetMap contributors" and "ODbL" in data["license"]
    assert "OpenStreetMap contributors" in Path("config/places/ucf.yaml").read_text(encoding="utf-8")
    assert len(data["buildings"]) > 200 and len(data["walk"]["edges"]) > 2000 and data["areas"] and data["roads"]
    places = PlaceSet.load("config/places")
    pack = places.packs["ucf"]
    assert pack.network == "ucf.map.json" and pack.map_file == f.resolve()
    s, w, n, e = data["bbox"]
    for pid, place in pack.places.items():
        assert s < place.lat < n and w < place.lon < e, pid
        assert pid in data["places"], f"{pid} was not found in OpenStreetMap"  # every place snapped to something real
    # the pack's edges are the same real walks, so a box without the map file plans the same day
    for a, b, minutes in pack.edges:
        real = pack.walk(a, b)
        assert real is not None and minutes == pytest.approx(max(0.1, round(real.minutes, 1))), (a, b)
        crow = 2 * 6_371_000 * math.asin(math.sqrt(
            math.sin(math.radians(pack.places[b].lat - pack.places[a].lat) / 2) ** 2
            + math.cos(math.radians(pack.places[a].lat)) * math.cos(math.radians(pack.places[b].lat))
            * math.sin(math.radians(pack.places[b].lon - pack.places[a].lon) / 2) ** 2))
        assert 20 < real.meters < 3.0 * crow + 150, (a, b, real.meters, crow)  # a walk, not a teleport and not a tour


# --- place packs: with a network, and exactly as before without one ---


TINY = """
name: tiny
places:
  hall: {name: Hall, lat: %f, lon: %f}
  lab:  {name: Lab, lat: %f, lon: %f}
  shed: {name: Shed}
edges:
  - [hall, lab, 9]
  - [lab, shed, 1]
""" % (*pt(0, 0), *pt(200, 0))


def test_route_and_feasible_without_a_network_use_the_edges_as_before():
    p = PlaceSet({"tiny": parse_pack(TINY)})
    assert p.packs["tiny"].network is None and p.packs["tiny"].walk("hall", "lab") is None
    assert p.route("tiny.hall", "tiny.shed") == (["tiny.hall", "tiny.lab", "tiny.shed"], 10.0)
    f = p.feasible("tiny.hall", "tiny.lab", datetime(2026, 10, 11, 9, 50, tzinfo=NY), datetime(2026, 10, 11, 10, 0, tzinfo=NY))
    assert f["ok"] and f["slack_minutes"] == 1 and "polyline" not in f and "meters" not in f
    assert p.trace(["tiny.hall", "tiny.lab"]) == ([], None)


def test_route_and_feasible_with_a_network_follow_the_paths(tmp_path):
    (tmp_path / "tiny.map.json").write_text(json.dumps(as_map()), encoding="utf-8")
    (tmp_path / "tiny.yaml").write_text(TINY + "network: tiny.map.json\nwalk_speed_mps: 1.0\n", encoding="utf-8")
    p = PlaceSet.load(tmp_path)
    pack = p.packs["tiny"]
    assert pack.map_file == (tmp_path / "tiny.map.json").resolve()
    path, minutes = p.route("tiny.hall", "tiny.lab")
    assert path == ["tiny.hall", "tiny.lab"] and minutes == pytest.approx(100 / 60, abs=0.06)  # the lab's nearer door, at 1 m/s
    real = pack.walk("hall", "lab")
    assert real.points[0] == pytest.approx(pt(0, 0), abs=1e-6) and real.points[-1] == pytest.approx(pt(200, 0), abs=1e-6)  # drawn to the place itself
    f = p.feasible("tiny.hall", "tiny.lab", datetime(2026, 10, 11, 9, 50, tzinfo=NY), datetime(2026, 10, 11, 10, 0, tzinfo=NY))
    assert f["ok"] and f["slack_minutes"] == 8 and f["meters"] == pytest.approx(100, abs=0.5) and len(f["polyline"]) >= 2
    # a place with no coordinates is not on the network: its edge still routes
    assert p.route("tiny.lab", "tiny.shed") == (["tiny.lab", "tiny.shed"], 1.0)
    assert p.route("tiny.hall", "tiny.hall") == (["tiny.hall"], 0.0)
    # a pack that names a map file that is not there behaves as a pack without one
    (tmp_path / "tiny.map.json").unlink()
    assert PlaceSet.load(tmp_path).route("tiny.hall", "tiny.lab") == (["tiny.hall", "tiny.lab"], 9.0)


# --- every place id anyone uses still resolves ---

KNOWN = ["ucf.union", "ucf.hec", "ucf.eng2", "ucf.library", "ucf.dining_63", "ucf.rwc", "ucf.cb1", "ucf.towers", "ucf.memory_mall",
         "apartment.door", "apartment.kitchen", "apartment.bedroom", "apartment.desk", "apartment.bathroom"]
NOT_PLACES = {"edu", "yaml", "yml", "json", "map", "md", "nowhere", "py", "practice", "view"}  # ucf.edu, ucf.yaml, the "unknown place" test, a test variable named ucf (ucf.view(), ucf.practice())


def refs_in(files, packs: PlaceSet) -> set[str]:
    names = "|".join(re.escape(n) for n in packs.packs)
    found = set()
    for f in files:
        for pack, place in re.findall(rf"(?<![\w./@-])({names})\.([a-z][a-z0-9_]*)\b", f.read_text(encoding="utf-8", errors="ignore")):
            if place not in NOT_PLACES:
                found.add(f"{pack}.{place}")
    return found


def resolves(packs: PlaceSet, ref: str) -> bool:
    pack, place = ref.split(".", 1)
    return packs.packs[pack].find(place) is not None


def test_every_place_id_in_config_fixtures_tests_and_the_phone_still_resolves():
    packs = PlaceSet.load("config/places")
    files = [f for root in ("config", "fixtures", "contexts", "tests", "web/lib", "web/app", "web/components")
             for f in Path(root).rglob("*") if f.is_file() and f.suffix in (".yaml", ".yml", ".json", ".py", ".ts", ".tsx")
             and "profiles" not in f.parts and "node_modules" not in f.parts and f.name != "ucf.map.json"]
    used = refs_in(files, packs) | set(KNOWN)
    assert len(used) >= len(KNOWN)
    missing = sorted(r for r in used if not resolves(packs, r))
    assert missing == [], f"place ids that no longer resolve: {missing}"
    for a in KNOWN:  # and every one can be walked to from the demo's table
        assert packs.route("ucf.eng2", a)[1] >= 0


def test_every_profile_s_own_place_ids_resolve_in_its_own_packs():
    for profile in sorted(Path("config/profiles").glob("*/places")):
        packs = PlaceSet.load(profile)
        used = refs_in([f for f in profile.parent.rglob("*") if f.is_file() and f.suffix in (".yaml", ".yml")], packs)
        missing = sorted(r for r in used if not resolves(packs, r))
        assert missing == [], f"{profile.parent.name}: {missing}"
        assert all(p.network is None for p in packs.packs.values())  # hand-made packs: edges, as before
