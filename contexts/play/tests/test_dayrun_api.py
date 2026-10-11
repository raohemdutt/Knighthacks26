"""Day Run's endpoints through a FastAPI test client, on the scripted scenario."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

from contexts.connectors.application.hub import ConnectorHub
from contexts.connectors.domain.spec import ConnectorSpec
from contexts.play.dayrun.service import DayRun, Scenario
from contexts.play.dayrun_api import dayrun_router
from contexts.play.domain.places import PlaceSet

NOW = datetime(2026, 10, 11, 9, 30, tzinfo=ZoneInfo("America/New_York"))


def never(spec):
    raise AssertionError("Day Run started a connector")


@pytest.fixture
def published() -> list:
    return []


@pytest.fixture
def client(tmp_path, published) -> TestClient:
    hub = ConnectorHub([ConnectorSpec(name="calendar", module="x", writes=["create_event"])], factory=never)
    service = DayRun(PlaceSet.load("config/places"), hub=hub, data_dir=tmp_path, clock=lambda: NOW)

    async def publish(event) -> None:
        published.append(event)

    def guard(authorization: str | None = Header(default=None)) -> None:
        if authorization != "Bearer t":
            raise HTTPException(401, "missing or wrong token")

    app = FastAPI()
    app.include_router(dayrun_router(service, publish=publish, guard=guard))
    return TestClient(app)


@pytest.fixture
def plans(client) -> dict:
    s = Scenario.load("fixtures/play/demo.yaml")  # the scenario's plans, as the JSON a phone would send
    return {name: [m.model_dump(mode="json", exclude_none=True) for m in moves] for name, moves in s.plans.items()} | {
        "campaign": [[m.model_dump(mode="json", exclude_none=True) for m in day] for day in s.campaign],
        "disruptions": [d.model_dump(mode="json") for d in s.disruptions]}


def test_board(client):
    b = client.get("/api/dayrun/board", params={"scenario": "demo"}).json()
    assert b["start"] == "2026-10-11T09:30:00-04:00" and [s["id"] for s in b["slots"]] == ["s1", "s2", "s3"]
    assert b["initial"]["sleep_hours"] == 4 and b["deadlines"][0]["work_left"] == 300
    assert {"kind", "place", "place_name", "title", "walk_minutes", "back_minutes", "max_minutes", "feasible", "reason",
            "friends", "event", "route"} == set(b["slots"][0]["moves"][0])
    live = client.get("/api/dayrun/board", params={"start": "2026-10-12T08:00:00-04:00"}).json()  # no sources: an empty day
    assert live["blocks"] == [] and len(live["slots"]) == 1 and live["initial"]["source"] == "defaults"
    assert client.get("/api/dayrun/board", params={"scenario": "nope"}).status_code == 404
    assert client.get("/api/dayrun/board", params={"scenario": "../../config/agent"}).status_code == 404


def test_simulate(client, plans):
    r = client.post("/api/dayrun/simulate", json={"scenario": "demo", "moves": plans["good"]})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"start", "end", "tick_minutes", "outcome", "stars", "timeline", "legs", "routes", "rejected", "blocks",
                         "changes", "notes", "totals", "carry"}
    assert body["stars"] == 3 and body["outcome"]["won"] and len(body["timeline"]) == 97
    assert set(body["timeline"][0]) == {"t", "energy", "focus", "deadline_risk", "work_left", "activity", "place"}
    assert set(body["legs"][0]) == {"from", "to", "from_name", "to_name", "minutes", "from_xy", "to_xy", "path", "meters", "route", "depart"}
    assert body["legs"][0]["from"] == "ucf.eng2" and body["legs"][0]["to"] == "ucf.dining_63" and body["legs"][0]["minutes"] == 7.1
    assert body["legs"][0]["meters"] == body["routes"][0]["meters"] > 500 and len(body["legs"][0]["path"]) > 10  # the real line walked

    bad = client.post("/api/dayrun/simulate", json={"scenario": "demo", "moves": plans["bad"]}).json()
    assert not bad["outcome"]["won"] and bad["stars"] == 0 and len(bad["outcome"]["reasons"]) == 3
    assert bad["rejected"][0]["index"] == 1 and bad["rejected"][0]["minutes"] == 11
    assert bad["rejected"][0]["reason"].startswith("you miss the 12:00 Knight Hacks: judging and demos by 11 minutes")
    assert client.post("/api/dayrun/simulate", json={"moves": [{"slot": "s1", "kind": "fly"}]}).status_code == 422


def test_commit_goes_through_one_approval(client, plans, published, tmp_path):
    body = {"scenario": "demo", "moves": plans["good"]}
    assert client.post("/api/dayrun/commit", json=body).status_code == 401  # a write: needs the token
    auth = {"Authorization": "Bearer t"}
    lost = client.post("/api/dayrun/commit", json={"scenario": "demo", "moves": plans["bad"]}, headers=auth)
    assert lost.status_code == 422 and len(lost.json()["detail"]["reasons"]) == 3 and published == []

    r = client.post("/api/dayrun/commit", json=body, headers=auth)
    assert r.status_code == 200
    (proposal,) = published  # one card, not eight
    assert (proposal.type, proposal.kind, proposal.target) == ("agent.action.proposed", "connector_write", "calendar.create_event")
    assert r.json()["action_id"] == proposal.action_id and r.json()["events"] == proposal.params["events"]
    assert len(proposal.params["events"]) == 8 and r.json()["day"] == "2026-10-11"
    assert not (tmp_path / "invites").exists()  # nothing created: that is Workspace's job after a yes


def test_review(client, plans):
    assert client.get("/api/dayrun/review", params={"day": "2026-10-11", "scenario": "demo"}).status_code == 404  # nothing committed
    client.post("/api/dayrun/commit", json={"scenario": "demo", "moves": plans["good"]}, headers={"Authorization": "Bearer t"})
    r = client.get("/api/dayrun/review", params={"day": "2026-10-11", "scenario": "demo"}).json()
    assert r["measured"]["sleep_hours"] == 6.5 and 0 < r["score"] < 100
    # three meals planned, two seen; 6.5 of 8.5 hours slept. The missed meal weighs a little more.
    assert r["biggest_gap"]["kind"] == "eat" and "Biggest gap: Eat at Student Union at 16:56: no meal was seen" in r["text"]
    assert next(b for b in r["blocks"] if b["kind"] == "sleep")["measured_minutes"] == 390
    assert sum(1 for b in r["blocks"] if b["followed"] is None) == 2  # the two library sessions


def test_a_damaged_plan_file_is_a_422_not_a_500(client, tmp_path):
    plans_dir = tmp_path / "dayrun" / "plans"
    plans_dir.mkdir(parents=True)
    for text in ("", "not json", "[]", "{}", '{"blocks": 5, "totals": {}}', '{"blocks": [], "totals": null}'):
        (plans_dir / "2026-10-11.json").write_text(text)
        r = client.get("/api/dayrun/review", params={"day": "2026-10-11", "scenario": "demo"})
        assert r.status_code == 422 and "damaged" in r.json()["detail"], text
        r = client.post("/api/dayrun/forecast", json={"estimates": [{"block": "m0", "minutes": 5}], "day": "2026-10-11", "scenario": "demo"})
        assert r.status_code == 422, text


def test_replay(client, plans):
    r = client.post("/api/dayrun/replay", json={"day": "2026-10-11", "scenario": "demo", "moves": plans["good"],
                                                "actual": {"focus_minutes": 45, "meals": 1, "sleep_hours": 5}})
    assert r.status_code == 200
    body = r.json()
    assert body["board"]["start"] == "2026-10-11T07:30:00-04:00" and not body["baseline"]["outcome"]["won"]
    assert body["plan"]["outcome"]["won"] and body["points_gained"] > 0 and len(body["baseline_moves"]) == 3


def test_hard(client, plans):
    r = client.post("/api/dayrun/hard", json={"scenario": "demo", "moves": plans["good"], "disruptions": plans["disruptions"]})
    assert r.status_code == 200
    body = r.json()
    assert [c["kind"] for c in body["changes"]] == ["rerouted", "shortened", "rerouted", "route_changed"]
    assert body["after"]["totals"]["walk_minutes"] > body["before"]["totals"]["walk_minutes"]
    seeded = [client.post("/api/dayrun/hard", json={"scenario": "demo", "moves": plans["good"], "seed": 3, "count": 2}).json() for _ in range(2)]
    assert seeded[0] == seeded[1] and len(seeded[0]["disruptions"]) == 2 and seeded[0]["seed"] == 3
    missing = {"kind": "block_moved", "block": "nope", "to_place": "ucf.cb1"}
    assert client.post("/api/dayrun/hard", json={"scenario": "demo", "moves": [], "disruptions": [missing]}).status_code == 422


def test_campaign(client, plans):
    r = client.post("/api/dayrun/campaign", json={"scenario": "demo", "days": plans["campaign"]})
    assert r.status_code == 200
    body = r.json()
    one, two = body["days"]
    assert (one["day"], two["day"]) == ("2026-10-11", "2026-10-12")
    assert two["carry_in"]["energy"] == one["result"]["carry"]["energy"]
    assert two["result"]["timeline"][0]["work_left"] == one["result"]["carry"]["work_left"]
    assert body["stars"] == one["result"]["stars"] + two["result"]["stars"]


def test_campaign_length_is_bounded(client, plans):
    """Every day is simulated on the event loop: 400 days held the whole box for seconds, 3,000,000 for good."""
    assert client.post("/api/dayrun/campaign", json={"scenario": "demo", "days": [[]] * 31}).status_code == 200
    assert client.post("/api/dayrun/campaign", json={"scenario": "demo", "days": [[]] * 32}).status_code == 422
    assert client.post("/api/dayrun/campaign", json={"scenario": "demo"}).status_code == 422  # still required


def test_forecast(client, plans):
    r = client.post("/api/dayrun/forecast", json={"estimates": [{"block": "m3", "minutes": 90}, {"block": "m4", "minutes": 30}],
                                                  "measured": {"m3": 180, "m4": 30}})
    assert r.status_code == 200 and r.json()["score"] == 75 and r.json()["bias"] < 0
    assert client.post("/api/dayrun/forecast", json={"estimates": [{"block": "m3", "minutes": 90}]}).status_code == 422
    # a negative or infinite length is refused, not scored (it gave a score of -199999999900)
    assert client.post("/api/dayrun/forecast", json={"estimates": [{"block": "a", "minutes": -5}], "measured": {"a": 7}}).status_code == 422
    assert client.post("/api/dayrun/forecast", json={"estimates": [{"block": "a", "minutes": 5}], "measured": {"a": -7}}).status_code == 422
    inf = b'{"estimates": [{"block": "a", "minutes": 5}], "measured": {"a": 1e400}}'  # JSON that parses as infinity
    assert client.post("/api/dayrun/forecast", content=inf, headers={"content-type": "application/json"}).status_code == 422
    zero = client.post("/api/dayrun/forecast", json={"estimates": [{"block": "a", "minutes": 0}], "measured": {"a": 0}}).json()
    assert zero["score"] == 100
    # against a committed day's measured blocks
    client.post("/api/dayrun/commit", json={"scenario": "demo", "moves": plans["good"]}, headers={"Authorization": "Bearer t"})
    r = client.post("/api/dayrun/forecast", json={"estimates": [{"block": "m7", "minutes": 480}, {"block": "m3", "minutes": 120}],
                                                  "day": "2026-10-11", "scenario": "demo"}).json()
    assert r["rows"][0]["measured"] == 390 and r["unmeasured"] == ["m3"]


def test_dates_and_minutes_out_of_range_are_refused_not_a_crash(client):
    for start in ("0001-01-01", "0001-01-01T00:00:00+14:00", "9999-12-31", "9999-12-31T23:59:59-14:00"):
        assert client.get("/api/dayrun/board", params={"start": start}).status_code == 422
    assert client.post("/api/dayrun/replay", json={"day": "9999-12-31"}).status_code == 422
    move = {"slot": "s1", "kind": "study"}
    for bad in ({"minutes": 10**18}, {"start": "9999-12-31T23:59:59-14:00"}):
        r = client.post("/api/dayrun/simulate", json={"scenario": "demo", "moves": [move | bad]})
        assert r.status_code == 422 and "out of range" in r.json()["detail"]
    r = client.post("/api/dayrun/forecast", json={"estimates": [{"block": "a", "minutes": 1e308}], "measured": {"a": -1e308}})
    assert r.status_code == 422
