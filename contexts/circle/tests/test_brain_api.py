"""The group brain's phone endpoints: the real router over the real GroupBrain and TeacherLink, the fake teacher behind
the fake hub proxy (lab/teacher_fake.py), and the box's log in a temporary folder."""

import pytest
from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

from contexts.circle.brain_api import group_brain_router
from contexts.circle.tests.brain_helpers import FACTS, PLAN, QUESTION, USER_TEXT, World

AUTH = {"Authorization": "Bearer secret"}


def client(brain):
    def guard(authorization: str | None = Header(default=None)) -> None:
        if authorization != AUTH["Authorization"]:
            raise HTTPException(401, "missing or wrong token")

    app = FastAPI()
    app.include_router(group_brain_router(brain, guard=guard))
    return TestClient(app)


def test_without_a_circle_the_status_says_so_and_the_writes_are_not_found():
    api = client(None)

    assert api.get("/api/circle/brain").json() == {
        "configured": False, "circle_id": None, "member_id": None, "opted_in": False, "opt_in_ref": None,
        "changed_at": None, "node": None, "host": None, "badge": "Asked the group brain"}
    assert api.post("/api/circle/brain/consent", json={"on": True}, headers=AUTH).status_code == 404
    assert api.post("/api/circle/brain/ask", json={"question": QUESTION}, headers=AUTH).status_code == 404


def test_the_switch_needs_the_token_and_records_the_opt_in_in_the_log(tmp_path):
    w = World(tmp_path)
    api = client(w.brain)

    assert api.post("/api/circle/brain/consent", json={"on": True}).status_code == 401
    assert w.logged() == []

    switched = api.post("/api/circle/brain/consent", json={"on": True}, headers=AUTH)

    [consent] = w.logged()
    assert switched.status_code == 200
    assert (consent.type, consent.on, consent.source, consent.member_id) == (
        "circle.brain.consent.changed", True, "phone", "ilan")
    assert switched.json() == api.get("/api/circle/brain").json() == {
        "configured": True, "circle_id": "team", "member_id": "ilan", "opted_in": True,
        "opt_in_ref": consent.consent_id, "changed_at": consent.ts.isoformat(), "node": "pod-a",
        "host": "teacher.test", "badge": "Asked the group brain"}

    off = api.post("/api/circle/brain/consent", json={"on": False}, headers=AUTH).json()
    assert (off["opted_in"], off["opt_in_ref"]) == (False, None)


@pytest.mark.parametrize("on", ["yes", 1, None])
def test_the_switch_takes_only_true_or_false(tmp_path, on):
    w = World(tmp_path)

    reply = client(w.brain).post("/api/circle/brain/consent", json={"on": on}, headers=AUTH)

    assert reply.status_code == 422 and w.logged() == []


def test_a_question_before_the_opt_in_is_forbidden_and_nothing_leaves(tmp_path):
    w = World(tmp_path)

    reply = client(w.brain).post("/api/circle/brain/ask", json={"question": QUESTION, "facts": FACTS}, headers=AUTH)

    assert reply.status_code == 403 and reply.json()["detail"]["reason"] == "not_opted_in"
    assert w.fake.requests == [] and w.logged() == []


def test_an_opted_in_question_comes_back_with_the_badge_and_its_log_entries(tmp_path):
    w = World(tmp_path)
    api = client(w.brain)
    api.post("/api/circle/brain/consent", json={"on": True}, headers=AUTH)

    assert api.post("/api/circle/brain/ask", json={"question": QUESTION, "facts": FACTS}).status_code == 401
    reply = api.post("/api/circle/brain/ask", json={"question": QUESTION, "facts": FACTS, "purpose": "where_to_go"},
                     headers=AUTH)

    [consent, asked, answered] = w.logged()
    assert reply.status_code == 200
    assert reply.json() == {"answer": PLAN, "badge": "Asked the group brain", "ask_id": asked.ask_id,
                            "opt_in_ref": consent.consent_id, "model": "group-brain",
                            "latency_ms": answered.latency_ms}
    assert (asked.purpose, asked.messages[1].content) == ("where_to_go", USER_TEXT)
    assert len(w.chats()) == 1


@pytest.mark.parametrize("body, reason", [
    ({"question": QUESTION, "facts": [{"frame_ref": "frames:cam-desk:1728627120000"}]}, None),
    ({"question": QUESTION, "facts": [42]}, None),
    ({"question": 7, "facts": []}, None),
    ({"question": QUESTION, "facts": ["see data:image/jpeg;base64,/9j/4AAQSkZJRg=="]}, "not_text"),
    ({"question": "  ", "facts": FACTS}, "empty"),
    ({"question": QUESTION, "facts": FACTS, "purpose": "Where To Go?"}, "purpose"),
])
def test_only_plain_text_gets_in_and_nothing_leaves_otherwise(tmp_path, body, reason):
    w = World(tmp_path)
    api = client(w.brain)
    api.post("/api/circle/brain/consent", json={"on": True}, headers=AUTH)

    reply = api.post("/api/circle/brain/ask", json=body, headers=AUTH)

    assert reply.status_code == 422
    if reason is not None:
        assert reply.json()["detail"]["reason"] == reason
    assert w.fake.requests == []
    assert [e.type for e in w.logged()] == ["circle.brain.consent.changed"]


def test_a_teacher_that_does_not_answer_is_503_with_its_kind_so_the_phone_can_fall_back(tmp_path):
    w = World(tmp_path)
    api = client(w.brain)
    api.post("/api/circle/brain/consent", json={"on": True}, headers=AUTH)
    api.post("/api/circle/brain/ask", json={"question": QUESTION, "facts": FACTS}, headers=AUTH)
    w.fake.stop_server()

    reply = api.post("/api/circle/brain/ask", json={"question": "And lunch?", "facts": []}, headers=AUTH)

    asked, answered = w.logged()[-2:]
    assert reply.status_code == 503
    assert reply.json()["detail"]["kind"] == "pod_stopped" and reply.json()["detail"]["ask_id"] == asked.ask_id
    assert (answered.ok, answered.error) == (False, "pod_stopped")
