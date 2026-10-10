from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from contexts.play.domain.places import PlaceSet, parse_pack

NY = ZoneInfo("America/New_York")


def packs() -> PlaceSet:
    return PlaceSet.load("config/places")


def test_shipped_packs_load_and_route_by_minutes():
    p = packs()
    assert {"ucf", "apartment"} <= set(p.packs)
    path, minutes = p.route("ucf.union", "ucf.eng2")
    assert path == ["ucf.union", "ucf.eng2"] and minutes == 2.6  # 185 m on the real footpaths (config/places/ucf.map.json)


def test_routes_cross_into_and_out_of_nested_packs():
    p = packs()
    path, minutes = p.route("ucf.library", "apartment.desk")
    assert path[0] == "ucf.library" and path[-1] == "apartment.desk"
    assert "ucf.towers" in path and "apartment.door" in path
    back, back_minutes = p.route("apartment.bedroom", "ucf.union")
    assert back[0] == "apartment.bedroom" and back[-1] == "ucf.union" and back_minutes > 9


def test_feasibility_says_how_late_you_are():
    p = packs()
    depart = datetime(2026, 10, 11, 9, 50, tzinfo=NY)
    late = p.feasible("ucf.rwc", "ucf.towers", depart, datetime(2026, 10, 11, 10, 0, tzinfo=NY))
    assert not late["ok"] and late["message"] == "you miss the 10:00 by 7 minutes"  # 1344 m, 17 minutes
    ok = p.feasible("ucf.union", "ucf.library", depart, datetime(2026, 10, 11, 10, 0, tzinfo=NY))
    assert ok["ok"] and ok["slack_minutes"] == 9


def test_friends_shared_places_resolve_to_pack_places():
    p = packs()
    assert p.find_shared("Library") == "ucf.library"
    assert p.find_shared("Student Union") == "ucf.union"
    assert p.find_shared("Mars") is None


@pytest.mark.parametrize("bad, problem", [
    ("name: x\nplaces: {a: {name: A}}\nedges: [[a, b, 2]]", "unknown place 'b'"),
    ("name: x\nplaces: {a: {name: A}, b: {name: B}, c: {name: C}}\nedges: [[a, b, 2]]", "not reachable from the rest: c"),
    ("name: x\nplaces: {a: {name: A}, b: {name: B}}\nedges: [[a, b, 0]]", "must be positive"),
])
def test_bad_packs_are_rejected_with_the_reason(bad, problem):
    with pytest.raises((ValueError, ValidationError), match=problem):
        parse_pack(bad)


def test_upload_checks_nesting_before_accepting():
    p = packs()
    lab = "name: lab\nplaces: {home: {name: Home, pack: tiny}}\n"
    with pytest.raises(ValueError, match="unknown pack 'tiny'"):
        p.add(parse_pack(lab))
    p.add(parse_pack('{"name": "tiny", "kind": "home", "places": {"a": {"name": "A"}, "b": {"name": "B"}}, "edges": [["a", "b", 1]]}'))
    with pytest.raises(ValueError, match="needs an `entry`"):
        p.add(parse_pack(lab))
    assert "lab" not in p.packs and "tiny" in p.packs
