"""GET /api/places/route ("when to leave") and GET /api/places/map/{pack} (the campus the phone draws)."""

from pathlib import Path

from fastapi.testclient import TestClient

from contexts.gateway.app import Settings, create_app


def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(Settings(data_dir=tmp_path, token=None, connectors=False, ntfy_url=None, llm_url=None)))


def test_when_to_leave_answers_for_an_arrival_time_and_for_a_departure(tmp_path):
    with client(tmp_path) as c:
        r = c.get("/api/places/route", params={"from": "ucf.rwc", "to": "ucf.eng2", "arrive_by": "2026-10-11T14:50:00-04:00"}).json()
        assert (r["from"], r["to"], r["mode"], r["path"]) == ("ucf.rwc", "ucf.eng2", "arrive_by", ["ucf.rwc", "ucf.eng2"])
        assert (r["minutes"], r["walk_minutes"], r["meters"]) == (9.6, 10, 768.8)  # the real footpaths, rounded up to whole minutes
        assert (r["depart"], r["arrive"]) == ("2026-10-11T14:40:00-04:00", "2026-10-11T14:50:00-04:00")
        assert r["sentence"] == "Leave by 2:40 PM: 10 minutes on foot to Engineering II, 770 m"
        assert len(r["polyline"]) > 10 and r["polyline"][0] == r["from_xy"] and r["polyline"][-1] == r["to_xy"]
        assert r["attribution"] == "© OpenStreetMap contributors"

        # names and aliases work as well as ids; with `depart` the answer is when you get there
        d = c.get("/api/places/route", params={"from": "Library", "to": "HEC", "depart": "2026-10-11T09:00:00-04:00"}).json()
        assert (d["from"], d["to"], d["mode"]) == ("ucf.library", "ucf.hec", "depart")
        assert (d["depart"], d["arrive"], d["walk_minutes"]) == ("2026-10-11T09:00:00-04:00", "2026-10-11T09:05:00-04:00", 5)
        assert d["sentence"] == "Leave at 9:00 AM: 5 minutes on foot to L3Harris Engineering Center, 340 m: there by 9:05 AM"
        named = c.get("/api/places/route", params={"from": "Student Union", "to": "Engineering II", "arrive_by": "2026-10-11T12:00:00-04:00"}).json()
        assert named["to"] == "ucf.eng2" and named["depart"] == "2026-10-11T11:57:00-04:00"


def test_routes_into_a_home_to_the_same_place_and_to_nowhere(tmp_path):
    with client(tmp_path) as c:
        # into the apartment: the rooms add their minutes, the line stops at the building
        home = c.get("/api/places/route", params={"from": "ucf.library", "to": "apartment.desk", "arrive_by": "2026-10-11T19:31:00-04:00"}).json()
        assert home["walk_minutes"] == 13 and home["path"][-1] == "apartment.desk" and home["to_xy"] is None and home["meters"] > 900
        same = c.get("/api/places/route", params={"from": "ucf.union", "to": "SU"}).json()
        assert same["walk_minutes"] == 0 and same["sentence"] == "You are already at Student Union"
        # with no `from`, the walk starts where the box last saw you (here: the day's start place)
        here = c.get("/api/places/route", params={"to": "ucf.library"})
        assert here.status_code == 200 and here.json()["mode"] == "depart" and here.json()["from"]

        assert c.get("/api/places/route", params={"from": "ucf.union", "to": "Narnia"}).status_code == 422
        assert c.get("/api/places/route", params={"from": "ucf.union", "to": "friend:nobody"}).status_code == 404
        assert c.get("/api/places/route", params={"from": "ucf.union"}).status_code == 422  # `to` is required


def test_the_map_is_served_with_its_credit_and_only_for_packs_that_have_one(tmp_path):
    with client(tmp_path) as c:
        packs = {p["name"]: p for p in c.get("/api/places").json()}
        assert packs["ucf"]["map"] == "/api/places/map/ucf" and "map" not in packs["apartment"]
        m = c.get(packs["ucf"]["map"])
        assert m.status_code == 200 and m.json()["attribution"] == "© OpenStreetMap contributors" and len(m.json()["buildings"]) > 200
        assert c.get("/api/places/map/apartment").status_code == 404 and c.get("/api/places/map/nope").status_code == 404
