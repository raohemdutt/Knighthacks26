from pathlib import Path

from fastapi.testclient import TestClient

from contexts.gateway.app import create_app
from contexts.gateway.bootstrap import Settings


def client(tmp_path: Path, token: str | None = None) -> TestClient:
    settings = Settings(data_dir=tmp_path, token=token, connectors=False, ntfy_url=None, llm_url=None)
    return TestClient(create_app(settings))


def test_ask_answers_from_memory_and_streams_events(tmp_path):
    with client(tmp_path) as c, c.websocket_connect("/ws") as ws:
        r = c.post("/api/ask", json={"text": "Where did I leave my keys?"})
        assert r.status_code == 200
        body = r.json()
        assert body["citations"][0]["frame_ref"] == "frames:cam-desk:k1941"
        types = []
        while "agent.answer.generated" not in types:
            types.append(ws.receive_json()["type"])
        types = [t for t in types if t != "agent.answer.delta"]  # streamed pieces (A11) arrive in between
        assert types == ["experience.question.asked", "agent.answer.generated"]
    assert len(list((tmp_path / "events").glob("*.jsonl"))) == 1


def test_a_memory_wipe_shows_an_empty_memory_at_once_without_a_restart(tmp_path):
    with client(tmp_path) as c:
        assert c.get("/memory/stats").json()["episodes"] > 0
        assert c.post("/api/ask", json={"text": "Where did I leave my keys?"}).json()["citations"]
        token = c.post("/vault/wipe/request", json={"scope": "memory"}).json()["token"]
        body = c.post("/vault/wipe", json={"token": token}).json()
        assert body["done"] and body["live_memory_cleared"] and body["removed"]["memory"] >= 1
        assert "the running Memory store (emptied)" in body["summary"]
        stats = c.get("/memory/stats").json()  # the same process, no restart
        assert (stats["episodes"], stats["captions"], stats["hours_recorded"]) == (0, 0, 0)
        assert c.get("/memory/search", params={"q": "keys"}).json() == []
        assert (tmp_path / "memory.sqlite").exists()  # a new empty file, so what is recorded next is kept


def test_teach_then_deny_through_the_api(tmp_path):
    with client(tmp_path) as c:
        r = c.post("/api/teach", json={"task": "desk-objects", "instruction": "List objects on the desk."})
        action_id = r.json()["action_id"]
        assert [a["id"] for a in c.get("/api/approvals").json()] == [action_id]
        r = c.post(f"/api/approvals/{action_id}", json={"decision": "deny"})
        assert r.status_code == 200 and r.json()["status"] == "denied"  # round 2: it said "proposed"
        assert c.get("/api/approvals").json() == []
        assert c.post("/api/approvals/nope", json={"decision": "deny"}).status_code == 404


def test_writes_need_the_token_and_state_survives_restart(tmp_path):
    with client(tmp_path, token="s3cret") as c:
        assert c.post("/api/teach", json={"task": "t", "instruction": "i"}).status_code == 401
        auth = {"Authorization": "Bearer s3cret"}
        action_id = c.post("/api/teach", json={"task": "t", "instruction": "i"}, headers=auth).json()["action_id"]
    with client(tmp_path, token="s3cret") as c:  # restart: pending card rebuilt from the log
        assert [a["id"] for a in c.get("/api/approvals").json()] == [action_id]


def test_plain_english_rule_through_the_api(tmp_path):
    from contracts.types import ChatResult

    with client(tmp_path) as c:
        rt = c.app.state.atlas.qa.runtime
        rt.chat_replies.append(ChatResult(model_version="fake-0", content='{"id": "late-visitor", "when": {"label": ["person"], "zone": "door", "kind": "enter"}, "notify": "Someone at the door", "cooldown_s": 60}'))
        r = c.post("/api/rules", json={"text": "tell me when someone comes to the door"})
        assert r.status_code == 200 and r.json()["id"] == "late-visitor"
        rt.chat_replies.append(ChatResult(model_version="fake-0", content="no idea"))
        assert c.post("/api/rules", json={"text": "???"}).status_code == 422


def test_play_route_and_a_round_over_the_bus(tmp_path):
    import asyncio
    from datetime import UTC, datetime, timedelta

    from contracts.perception import ZoneEvent
    from contracts.play import FacesCounted

    with client(tmp_path) as c:
        r = c.get("/api/play/route", params={"item": "can"}).json()
        assert r["bin"] == "recycling" and r["path"]
        atlas = c.app.state.atlas
        t0 = datetime.now(UTC)

        async def play():
            for s in (0, 2):
                await atlas.bus.publish(FacesCounted(ts=t0 + timedelta(seconds=s), camera_id="cam-door", count=1, facing=1))
            item = atlas.game.round.item  # the round's item is the one that has to be sorted
            zone = atlas.game.rules.zone_for(item)
            await atlas.bus.publish(ZoneEvent(ts=t0 + timedelta(seconds=4), camera_id="cam-door", track_id=1, label=item, zone_id=zone, kind="enter"))

        c.portal.call(play)
        assert c.get("/api/play/accuracy").json()["rounds"] == 1
        impact = c.get("/api/impact").json()["total"]
        assert impact["items_seen"] == 1  # the same item, counted by Waste Watch too


def test_sort_it_rounds_survive_a_restart(tmp_path):
    """The log had play.round.scored, but /api/play/accuracy said `rounds: 0` after every start."""
    from datetime import UTC, datetime, timedelta

    from contracts.perception import ZoneEvent
    from contracts.play import FacesCounted

    with client(tmp_path) as c:
        atlas = c.app.state.atlas
        t0 = datetime.now(UTC)

        async def play():
            for s in (0, 2):
                await atlas.bus.publish(FacesCounted(ts=t0 + timedelta(seconds=s), camera_id="cam-door", count=1, facing=1))
            item = atlas.game.round.item
            await atlas.bus.publish(ZoneEvent(ts=t0 + timedelta(seconds=4), camera_id="cam-door", track_id=1, label=item,
                                              zone_id=atlas.game.rules.zone_for(item), kind="enter"))

        c.portal.call(play)
        before = c.get("/api/play/accuracy").json()
        assert before["rounds"] == 1  # counted once, though the game also hears its own play.round.scored
    log = sorted((tmp_path / "events").glob("*.jsonl"))
    rounds_logged = sum(f.read_text().count('"play.round.') for f in log)
    for _ in range(2):  # restart twice: the round is rebuilt from the log, and replay adds nothing to it
        with client(tmp_path) as c:
            assert c.get("/api/play/accuracy").json() == before
            assert c.app.state.atlas.game.round is None
    assert sum(f.read_text().count('"play.round.') for f in log) == rounds_logged == 2  # started + scored


def test_invites_are_served_and_names_are_checked(tmp_path):
    (tmp_path / "invites").mkdir()
    (tmp_path / "invites" / "invite-20261010T120000-abc123.ics").write_text("BEGIN:VCALENDAR\nEND:VCALENDAR\n")
    with client(tmp_path) as c:
        [inv] = c.get("/api/invites").json()
        r = c.get(inv["url"])
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/calendar")
        assert c.get("/api/invites/..%2Fevents%2Fx.ics").status_code == 404
        assert c.get("/api/invites/secret.txt").status_code == 404


def test_place_packs_routes_feasibility_and_friend_routes(tmp_path):
    from datetime import UTC, datetime, timedelta

    from contracts.circle import PlaceShared

    with client(tmp_path) as c:
        assert {p["name"] for p in c.get("/api/places").json()} >= {"ucf", "apartment"}
        r = c.get("/api/play/campus-route", params={"from": "ucf.union", "to": "ucf.eng2"}).json()
        assert r == {"path": ["ucf.union", "ucf.hec", "ucf.eng2"], "minutes": 6}
        late = c.get("/api/play/campus-route", params={"from": "ucf.rwc", "to": "ucf.eng2",
                                                       "depart": "2026-10-11T09:50:00-04:00", "due": "2026-10-11T10:00:00-04:00"}).json()
        assert late["message"] == "you miss the 10:00 by 7 minutes"

        bad = c.post("/api/places", content="name: x\nplaces: {a: {name: A}}\nedges: [[a, b, 2]]")
        assert bad.status_code == 422 and "unknown place" in bad.json()["detail"]
        ok = c.post("/api/places", content='{"name": "lab", "places": {"a": {"name": "A"}, "b": {"name": "B"}}, "edges": [["a", "b", 2]]}')
        assert ok.status_code == 200 and (tmp_path / "places" / "lab.yaml").exists()
        assert c.get("/api/play/campus-route", params={"from": "lab.a", "to": "lab.b"}).json()["minutes"] == 2

        assert c.get("/api/play/campus-route", params={"from": "ucf.union", "to": "friend:hemdutt"}).status_code in (404,)


def test_route_for_an_unknown_item_is_a_404(tmp_path):
    with client(tmp_path) as c:
        r = c.get("/api/play/route", params={"item": "unicorn"})
        assert r.status_code == 404 and "unicorn" in r.json()["detail"]
        assert c.get("/api/play/route", params={"item": "soda can"}).json()["bin"] == "recycling"


def test_rules_list_toggle_and_alert_history(tmp_path):
    """Q10 (Xander, X6): the Rules screen lists rules, switches them, and shows recent alerts."""
    from datetime import UTC, datetime

    from contracts.perception import ZoneEvent

    door = dict(camera_id="cam-door", track_id=7, label="package", zone_id="door", kind="dwell", dwell_s=9)
    with client(tmp_path) as c:
        rules = {r["id"]: r for r in c.get("/api/rules").json()}
        assert rules["package-at-door"]["enabled"] and rules["package-at-door"]["source"] == "config"
        assert "door zone" in rules["package-at-door"]["description"]
        assert rules["odd-hours-activity"]["source"] == "built-in"
        atlas = c.app.state.atlas
        c.portal.call(atlas.bus.publish, ZoneEvent(ts=datetime(2026, 10, 11, 12, tzinfo=UTC), **door))
        [alert] = c.get("/api/alerts").json()
        assert alert["rule_id"] == "package-at-door" and alert["message"] == "Package left at the door"
        r = c.post("/api/rules/package-at-door/enabled", json={"enabled": False})
        assert r.status_code == 200 and r.json()["enabled"] is False
        c.portal.call(atlas.bus.publish, ZoneEvent(ts=datetime(2026, 10, 11, 13, tzinfo=UTC), **door))
        assert len(c.get("/api/alerts").json()) == 1  # switched off: no new alert
        assert c.post("/api/rules/nope/enabled", json={"enabled": False}).status_code == 404
    with client(tmp_path) as c:  # restart: the switch and the history come back from the log
        assert {r["id"]: r["enabled"] for r in c.get("/api/rules").json()}["package-at-door"] is False
        assert len(c.get("/api/alerts", params={"limit": 5}).json()) == 1


def test_presence_is_capped_can_stop_early_and_an_offline_leaderboard_says_503(tmp_path):
    """Q12: any `hours` was accepted, sharing could not be stopped, and Neon offline gave a bare 500."""
    from contexts.circle.application.presence import PresenceBoard
    from contexts.circle.application.sharing import CircleOffline

    class Offline:
        def __getattr__(self, name):
            def fail(*a, **k):
                raise CircleOffline("NEON_DATABASE_URL is not set")
            return fail

    class Me:
        circle_id = "team"

    with client(tmp_path) as c:
        atlas = c.app.state.atlas
        atlas.presence = PresenceBoard("team", "dom", ["Library"], opted_in=True)
        atlas.bus.subscribe("circle.place.shared", atlas.presence.on_shared)
        assert c.post("/api/presence", json={"place": "Library", "hours": 48}).status_code == 422
        assert c.post("/api/presence", json={"place": "Library", "hours": 2}).status_code == 200
        assert c.delete("/api/presence").json() == {"sharing": False}
        mine = atlas.presence.current["dom"]
        assert mine.place == "Library" and mine.expires_at <= mine.ts  # ended now; PresenceSync sends it to friends
        atlas.circle, atlas.board = Me(), Offline()
        r = c.get("/api/circle/leaderboard")
        assert r.status_code == 503 and "offline" in r.json()["detail"]


def test_camera_input_can_be_pushed_into_a_running_box_and_nothing_else(tmp_path):
    """T7: a camera feed could not be faked into a running box; only the log could be written."""
    from datetime import UTC, datetime

    from contexts.demo import fakes

    events = asyncio_run(fakes.publish("perception", tmp_path / "scratch", datetime.now(UTC).replace(microsecond=0)))
    body = [e.model_dump(mode="json", exclude={"event_id"}) for e in events if e.type == "perception.observation.recorded"]
    with client(tmp_path / "box", token="s3cret") as c:
        auth = {"Authorization": "Bearer s3cret"}
        assert c.post("/api/events", json=body).status_code == 401
        seen = []
        c.app.state.atlas.bus.subscribe("perception.zone", seen.append)
        r = c.post("/api/events", json=body, headers=auth)
        assert r.status_code == 200 and r.json()["published"] == len(body)
        assert seen  # the box's own zone engine made zone events from the pushed observations
        for bad in ({"type": "experience.approval.decided", "action_id": "x", "decision": "approve"},
                    {"type": "agent.alert.raised", "alert_id": "a", "rule_id": "r", "message": "m"},
                    {"type": "perception.zone.event"}):
            assert c.post("/api/events", json=bad, headers=auth).status_code == 403
        assert c.post("/api/events", json={"type": "perception.observation"}, headers=auth).status_code == 422


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)


def test_a_place_pack_cannot_name_a_file_outside_data_and_bad_bodies_are_422(tmp_path):
    with client(tmp_path) as c:
        escape = "name: ../../evil\nkind: campus\nplaces: {a: {name: A}}\nedges: []\n"
        assert c.post("/api/places", content=escape, headers={"content-type": "application/yaml"}).status_code == 422
        assert not (tmp_path.parent.parent / "evil.yaml").exists() and not list(tmp_path.glob("**/evil.yaml"))
        assert c.post("/api/places", content=b"\xff\xfe\x00", headers={"content-type": "application/yaml"}).status_code == 422
        assert c.post("/api/places", content="!!python/object/apply:os.system [echo]", headers={"content-type": "application/yaml"}).status_code == 422
        assert c.post("/api/places", content="[" * 5000, headers={"content-type": "application/yaml"}).status_code == 422
        assert c.get("/api/play/route?item=can&x=nan").status_code == 422
        assert c.get("/api/play/route?item=can&x=2").status_code == 422


def test_preflight_names_what_would_break_the_demo_and_reset_clears_the_stage(tmp_path):
    """D9: the T-15 checklist as checks the box runs on itself."""
    from datetime import UTC, datetime

    from contracts.perception import ZoneEvent

    with client(tmp_path) as c:
        body = c.get("/api/preflight").json()
        checks = {x["name"]: x for x in body["checks"]}
        assert body["ok"] is False  # a laptop with no model, ntfy or cameras is not demo-ready
        assert not checks["phone link"]["ok"] and "LAN" in checks["phone link"]["fix"]
        assert not checks["on-device model"]["ok"] and checks["package alert on"]["ok"]
        atlas = c.app.state.atlas
        door = dict(camera_id="cam-door", track_id=7, label="package", zone_id="door", kind="dwell", dwell_s=9)
        c.portal.call(atlas.bus.publish, ZoneEvent(ts=datetime.now(UTC), **door))
        c.post("/api/teach", json={"task": "t", "instruction": "i"})  # a leftover approval card
        checks = {x["name"]: x for x in c.get("/api/preflight").json()["checks"]}
        assert not checks["package alert ready"]["ok"] and not checks["no stale approval cards"]["ok"]
        reset = c.post("/api/preflight/reset").json()
        assert reset["cooldowns_cleared"] == 1 and reset["cards_denied"] >= 1  # the teach card, plus the Day loop's morning plan card when the day has started
        checks = {x["name"]: x for x in c.get("/api/preflight").json()["checks"]}
        assert checks["package alert ready"]["ok"] and checks["no stale approval cards"]["ok"]


def test_answers_stream_word_by_word_over_the_websocket_before_the_final_answer(tmp_path):
    """A11: the phone shows the answer as it is written; the final event and the log are unchanged."""
    with client(tmp_path) as c, c.websocket_connect("/ws") as ws:
        r = c.post("/api/ask", json={"text": "Where did I leave my keys?"}).json()
        msgs = []
        while not msgs or msgs[-1]["type"] != "agent.answer.generated":
            msgs.append(ws.receive_json())
        deltas = [m for m in msgs if m["type"] == "agent.answer.delta"]
        assert len(deltas) > 3 and "".join(d["text"] for d in deltas) == r["answer"]
        assert msgs.index(deltas[-1]) < len(msgs) - 1  # every piece arrives before the final answer
    log = "".join(p.read_text() for p in (tmp_path / "events").glob("*.jsonl"))
    assert "agent.answer.delta" not in log
