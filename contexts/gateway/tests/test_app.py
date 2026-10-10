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
        types = [ws.receive_json()["type"] for _ in range(2)]
        assert types == ["experience.question.asked", "agent.answer.generated"]
    assert len(list((tmp_path / "events").glob("*.jsonl"))) == 1


def test_teach_then_deny_through_the_api(tmp_path):
    with client(tmp_path) as c:
        r = c.post("/api/teach", json={"task": "desk-objects", "instruction": "List objects on the desk."})
        action_id = r.json()["action_id"]
        assert [a["id"] for a in c.get("/api/approvals").json()] == [action_id]
        assert c.post(f"/api/approvals/{action_id}", json={"decision": "deny"}).status_code == 200
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
