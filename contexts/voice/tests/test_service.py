"""The voice service's HTTP face, and the gateway's /api/voice proxy in front of it (here wired straight together)."""

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from contexts.voice.proxy import OFF, voice_check, voice_router
from contexts.voice.service import create_app
from contexts.voice.session import VoiceSession
from contexts.voice.tests.fakes import FakeMic, FakeSpeaker, FakeStt, FakeTts, make_config


def make_session() -> VoiceSession:
    mic = FakeMic()
    return VoiceSession(make_config(), mic, FakeSpeaker(mic), FakeStt(), FakeTts(), lambda text: ("On the desk.", 1))


def test_status_talk_and_off_over_http():
    s = make_session()
    c = TestClient(create_app(s))
    st = c.get("/status").json()
    assert st["state"] == "idle" and st["mic"]["ok"] and st["speaker"]["ok"] and st["running"]
    assert c.post("/talk").json()["state"] == "listening"
    s.wait()
    st = c.get("/status").json()
    assert st["state"] == "idle" and st["heard"] and st["answer"] == "On the desk."
    assert c.post("/enabled", json={"enabled": False}).json()["state"] == "off"
    assert c.post("/stop").status_code == 200


def gateway_in_front(session: VoiceSession, token: str | None = None) -> TestClient:
    from fastapi import Depends, Header, HTTPException

    def guard(authorization: str | None = Header(default=None)):
        if token and authorization != f"Bearer {token}":
            raise HTTPException(401, "missing or wrong token")

    app = FastAPI()
    app.include_router(voice_router(guard=guard, transport=httpx.ASGITransport(app=create_app(session))), prefix="/api")
    return TestClient(app)


def test_the_gateway_proxy_reaches_the_service_and_guards_the_buttons():
    s = make_session()
    c = gateway_in_front(s, token="secret")
    assert c.get("/api/voice/status").json()["state"] == "idle"  # reads are open
    assert c.post("/api/voice/talk").status_code == 401
    assert c.post("/api/voice/talk", headers={"Authorization": "Bearer secret"}).json()["state"] == "listening"
    s.wait()
    assert s.heard


def test_without_the_service_the_screen_sees_off_not_an_error():
    app = FastAPI()
    dead = httpx.MockTransport(lambda request: (_ for _ in ()).throw(httpx.ConnectError("refused")))
    app.include_router(voice_router(transport=dead), prefix="/api")
    c = TestClient(app)
    assert c.get("/api/voice/status").json() == OFF
    assert c.post("/api/voice/talk").status_code == 503


def test_preflight_line_passes_when_voice_is_not_running_and_fails_on_a_dead_speaker():
    def status_client(body):
        return httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body)))

    def refused(request):
        raise httpx.ConnectError("refused")

    assert voice_check(httpx.Client(transport=httpx.MockTransport(refused)))["ok"]  # optional: nothing to check
    ready = {"state": "idle", **{k: {"ok": True, "detail": "x"} for k in ("mic", "speaker", "speech_to_text", "text_to_speech")}}
    assert voice_check(status_client(ready))["ok"]
    dead = {**ready, "speaker": {"ok": False, "detail": "Bluetooth speaker AA:BB is not connected"}}
    out = voice_check(status_client(dead))
    assert not out["ok"] and "AA:BB" in out["detail"] and out["fix"]
    assert not voice_check(status_client({**ready, "state": "off"}))["ok"]
