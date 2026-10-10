"""GET /api/league with and without a league or a Circle; the POSTs raise cards and nothing else."""

import asyncio
from datetime import timedelta

from fastapi import FastAPI
from fastapi.testclient import TestClient

from bus import EventBus
from contexts.play.league.api import league_router
from contexts.play.league.tests.helpers import MONDAY, at, league, week_of_sleep

NOW = at(MONDAY + timedelta(days=5), 9, 30)  # Saturday morning of the week


def app(tmp_path, lg=None, guard=None):
    bus = EventBus(tmp_path / "events")
    if lg is not None:
        lg.subscribe(bus)
    api = FastAPI()
    api.include_router(league_router(bus, league=lg, clock=lambda: NOW, guard=guard))
    return bus, TestClient(api)


def test_without_a_league_the_endpoint_answers_200_and_says_how_to_turn_it_on(tmp_path):
    _, c = app(tmp_path)
    body = c.get("/api/league").json()
    assert body["configured"] is False and body["standings"] == [] and "config/league.yaml" in body["note"]
    assert c.get("/api/league/me").status_code == 404
    assert c.post("/api/league/stake", json={"kind": "pays_for_food"}).status_code == 404
    assert c.post("/api/league/challenges", json={"kind": "sleep_six"}).status_code == 404


def test_without_a_circle_the_league_plays_alone_under_you_and_says_which_lines_turn_friends_on(tmp_path):
    lg = league(circle_id=None, me="me", names={})
    bus, c = app(tmp_path, lg)
    for e in week_of_sleep(lg, "me", 5)[:5]:
        asyncio.run(bus.publish(e))
    body = c.get("/api/league").json()
    assert body["configured"] and body["local"] and body["circle_id"] is None and body["member_id"] == "me"
    assert "circle.member_id and circle.circle_id" in body["note"]
    assert body["members"] == [{"member_id": "me", "display_name": "You"}]
    assert body["week"] == {"start": "2026-10-05", "end": "2026-10-11", "closes_at": "2026-10-12T03:00:00-04:00",
                            "closes_in_s": int((at(MONDAY + timedelta(days=7), 3) - NOW).total_seconds())}
    [you] = body["standings"]
    assert (you["display_name"], you["score"], you["met"], you["judged"]) == ("You", 100.0, 5, 5)
    assert body["best"]["member_id"] == "me" and body["worst"] is None
    assert body["summary"] == "Only You had a judged week (5 of 5 sleep targets). Nobody owes anything."
    assert body["stake"] == {"kind": "pays_for_food", "text": None, "set_for_week": None,
                             "text_for_week": "The worst week buys the next group meal; the best week picks the place."}
    assert body["team_waste"]["line"] == "team: nothing counted yet this week, target 80%"
    assert body["challenges"] == [] and {p["kind"] for p in body["presets"]} >= {"no_energy_drinks", "sleep_six", "three_meals"}
    assert body["last_closed"] is None
    me = c.get("/api/league/me").json()
    assert me["score"] == 100.0 and [b["target"] for b in me["breakdown"]] == ["sleep"]


def test_with_a_circle_the_standings_name_the_worst_and_best_with_reasons(tmp_path):
    lg = league()
    bus, c = app(tmp_path, lg)
    for e in week_of_sleep(lg, "hemdutt", 6) + week_of_sleep(lg, "maya", 2):
        asyncio.run(bus.publish(e))
    body = c.get("/api/league").json()
    assert body["circle_id"] == "team" and not body["local"] and body["note"] is None
    assert [(s["rank"], s["member_id"], s["score"]) for s in body["standings"]] == [(1, "hemdutt", 85.7), (2, "maya", 28.6), (3, "dom", None)]
    assert body["worst"] == {"member_id": "maya", "display_name": "Maya", "score": 28.6, "reason": "2 of 7 sleep targets"}
    assert body["best"]["member_id"] == "hemdutt"
    assert body["summary"] == "Maya had the worst week: 2 of 7 sleep targets. Dinner is on Maya. Hemdutt picks the place."
    assert body["standings"][1]["reasons"] == ["missed sleep 5 of 7 nights"]


def test_the_posts_raise_approval_cards_and_change_nothing_by_themselves(tmp_path):
    lg = league()
    bus, c = app(tmp_path, lg)
    r = c.post("/api/league/stake", json={"kind": "picks_the_event"})
    assert r.status_code == 200 and r.json()["action_id"]
    r2 = c.post("/api/league/challenges", json={"kind": "three_meals", "days": 5, "invitees": ["maya"]})
    assert r2.status_code == 200 and r2.json()["challenge_id"]
    assert c.post("/api/league/stake", json={"kind": "custom"}).status_code == 422
    assert c.post("/api/league/stake", json={"kind": "lottery"}).status_code == 422
    assert c.post("/api/league/challenges", json={"kind": "nope"}).status_code == 422
    logged = [e for e in bus.read_log()]
    assert [e.type for e in logged] == ["agent.action.proposed", "agent.action.proposed"]
    assert {e.params["kind"] for e in logged} == {"league_stake", "league_challenge"}
    body = c.get("/api/league").json()
    assert body["stake"]["kind"] == "pays_for_food" and body["challenges"] == []  # nothing until the cards are approved


def test_the_posts_need_the_token_when_the_box_has_one(tmp_path):
    from fastapi import HTTPException

    def guard():
        raise HTTPException(401, "missing or wrong token")

    _, c = app(tmp_path, league(), guard=guard)
    assert c.post("/api/league/stake", json={"kind": "pays_for_food"}).status_code == 401
    assert c.get("/api/league").status_code == 200
