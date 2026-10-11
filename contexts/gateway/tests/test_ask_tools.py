"""Ask Atlas's tools through the real gateway (bootstrap + app): ask, see the approval card, approve it, and the
event lands on the box's calendar. The laptop's FakeRuntime picks no tool, so the box's rules choose (ATLAS_ASK_TOOLS=1
turns the router on for the fake; on the Jetson it is on by default)."""

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from contexts.gateway.app import create_app
from contexts.gateway.bootstrap import Settings


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ATLAS_ASK_TOOLS", "1")
    for k in ("AUP_HUB_URL_C", "AUP_HUB_USER_C", "AUP_HUB_TOKEN_C", "ATLAS_NETWORK_BRAIN_URL"):
        monkeypatch.delenv(k, raising=False)
    settings = Settings(data_dir=tmp_path, token=None, connectors=False, ntfy_url=None, llm_url=None)
    with TestClient(create_app(settings)) as c:
        yield c


def test_add_a_meeting_proposes_it_and_approving_lands_it_on_the_calendar(client):
    body = client.post("/api/ask", json={"text": "Add a meeting with Sam at 3 pm tomorrow called demo prep"}).json()
    assert body["tool"] == "calendar.add" and body["answered_by"] == "the box's rules"
    assert "Approve it on your phone" in body["answer"]
    assert body["tool_result"]["event"]["title"] == "demo prep with Sam"
    action_id = body["action_id"]
    assert [a["id"] for a in client.get("/api/approvals").json()] == [action_id]
    assert client.post(f"/api/approvals/{action_id}", json={"decision": "approve"}).json()["status"] == "approved"
    deadline = time.time() + 5
    while time.time() < deadline:
        later = client.post("/api/ask", json={"text": "What's on my calendar tomorrow?"}).json()
        if "demo prep" in later["answer"].lower():
            break
        time.sleep(0.1)
    assert later["tool"] == "calendar.list" and "demo prep" in later["answer"].lower()
    assert any(e["title"] == "demo prep with Sam" for e in later["tool_result"]["events"])


def test_the_other_demo_questions_reach_a_tool_and_say_who_answered(client):
    expect = {
        "Set a reminder to call mom at 6": "reminders.add",
        "Turn off the camera": "camera.pause",
        "Show me the room map": "room.map",
        "What is the box's model right now?": "box.model",
        "How much energy did my last answer cost?": "box.energy_last_answer",
    }
    for question, tool in expect.items():
        body = client.post("/api/ask", json={"text": question}).json()
        assert body["tool"] == tool, question
        assert body["answered_by"] and body["answer"]
    model = client.post("/api/ask", json={"text": "What is the box's model right now?"}).json()
    assert "fake-0" in model["answer"]


def test_a_question_for_no_tool_is_answered_as_before(client):
    body = client.post("/api/ask", json={"text": "Where did I leave my keys?"}).json()
    assert body["tool"] is None and body["answered_by"] == "the box"
    assert body["citations"][0]["frame_ref"] == "frames:cam-desk:k1941"  # the pre-fetch, as before
