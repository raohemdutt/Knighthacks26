"""Face share's one switch, through the real app: off by default, on with one env line, independent of the others.
In-memory Circle store only (NEON_DATABASE_URL is taken out of the environment)."""

import json
import time

import pytest
import yaml
from fastapi.testclient import TestClient

from contexts.circle.application.protocol import InMemoryCircleStore
from contexts.gateway.app import create_app
from contexts.gateway.bootstrap import Settings, build, circle_from_env

BASE = {"member_id": None, "circle_id": None, "share_presence": False, "share_summaries": False}
OFF = "face sharing between friends' boxes is switched off on this box"


@pytest.fixture
def env(monkeypatch):
    for name in ("NEON_DATABASE_URL", "ATLAS_CIRCLE_MEMBER", "ATLAS_CIRCLE_ID", "ATLAS_CIRCLE_SHARE", "ATLAS_CIRCLE_FACE_SHARE"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def settings(tmp_path) -> Settings:
    return Settings(data_dir=tmp_path, connectors=False, ntfy_url=None, llm_url=None)


def test_the_committed_config_has_it_off():
    circle = yaml.safe_load(open("config/agent.yaml"))["circle"]
    assert circle["face_share"] is False and circle["face_share_days"] == 30


def test_the_env_switch_is_its_own_and_touches_nothing_else():
    assert "face_share" not in circle_from_env(BASE, {"ATLAS_CIRCLE_SHARE": "1"})  # the other switch does not turn it on
    out = circle_from_env(BASE, {"ATLAS_CIRCLE_FACE_SHARE": "1"})
    assert out == {**BASE, "face_share": True}  # and it does not opt in to presence or summaries
    assert "face_share" not in circle_from_env(BASE, {"ATLAS_CIRCLE_FACE_SHARE": "0"})
    assert "face_share" not in BASE


def test_no_circle_no_face_share_and_the_endpoints_say_so(tmp_path, env):
    with TestClient(create_app(settings(tmp_path))) as c:
        r = c.get("/api/circle/face-share")
        assert r.status_code == 409 and r.json()["detail"] == "this box has not joined a Circle"
        assert c.post("/api/circle/face-share/dom/share").status_code == 409
    assert not (tmp_path / "vault" / "circle").exists()


def test_in_a_circle_it_is_still_off_until_its_own_switch(tmp_path, env):
    env.setenv("ATLAS_CIRCLE_MEMBER", "hemdutt")
    env.setenv("ATLAS_CIRCLE_ID", "kh-test")
    env.setenv("ATLAS_CIRCLE_SHARE", "1")  # presence and summaries on: face share stays off
    atlas = build(settings(tmp_path))
    assert atlas.face_share.enabled is False and atlas.face_share.poll() == []
    assert isinstance(atlas.circle.store, InMemoryCircleStore) and atlas.face_share.store.rows == []  # no key published
    assert not (tmp_path / "vault" / "circle").exists()
    with TestClient(create_app(settings(tmp_path))) as c:
        for r in (c.get("/api/circle/face-share"), c.post("/api/circle/face-share/dom/share"),
                  c.post("/api/circle/face-share/dom/revoke"), c.post("/api/circle/face-share/dom/forget")):
            assert r.status_code == 409 and r.json()["detail"].startswith(OFF)
        assert c.get("/api/circle/members").json()["configured"] is True  # the rest of the Circle is as it was
        assert c.get("/identity/settings").json()["identity_enabled"] is False


def test_switched_on_it_answers_publishes_a_key_and_plans_still_get_their_cards(tmp_path, env):
    env.setenv("ATLAS_CIRCLE_MEMBER", "hemdutt")
    env.setenv("ATLAS_CIRCLE_ID", "kh-test")
    env.setenv("ATLAS_CIRCLE_FACE_SHARE", "1")
    atlas = build(settings(tmp_path))
    assert atlas.face_share.enabled and atlas.face_share.days == 30
    assert atlas.face_share.poll() == [] and [r.kind for r in atlas.face_share.store.rows] == ["box_key"]
    assert atlas.identity.settings()["identity_enabled"] is False  # independent of the recognition switch
    with TestClient(create_app(settings(tmp_path))) as c:
        body = c.get("/api/circle/face-share").json()
        assert body["enabled"] and body["member_id"] == "hemdutt" and body["recognition_enabled"] is False
        assert [f["member_id"] for f in body["friends"]] == ["dom", "xander", "ilan"] and not body["me"]["can_share"]
        assert all(f["outgoing"]["state"] == f["incoming"]["state"] == "none" for f in body["friends"])
        r = c.post("/api/circle/face-share/dom/share")
        assert r.status_code == 409 and "has not published a key yet" in r.json()["detail"]
        # the executor it wraps still runs a plan's card
        plan = {"title": "Study session", "start": "2026-10-11T14:00:00-04:00", "end": "2026-10-11T16:00:00-04:00"}
        action_id = c.post("/api/circle/proposals", json=plan).json()["action_id"]
        assert c.post(f"/api/approvals/{action_id}", json={"decision": "approve"}).status_code == 200
        done = None
        for _ in range(100):  # the executor runs in the background
            logged = [json.loads(line) for p in (tmp_path / "events").glob("*.jsonl") for line in p.read_text().splitlines()]
            done = next((e for e in logged if e["type"] == "workspace.action.completed" and e["action_id"] == action_id), None)
            if done:
                break
            time.sleep(0.02)
        assert done and (done["status"], done["result"]) == ("done", "proposed to the circle: Study session")
