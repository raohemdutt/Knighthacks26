"""The voice service's HTTP face, for the gateway and for curl. It listens on localhost only; the gateway proxies it
at /api/voice/* (with the token check on writes), which is how the operator screen's button reaches it.

    GET  /status     state (off | idle | listening | thinking | speaking), what was heard and answered, device health
    POST /talk       push to talk: start listening; again = that was the whole question; while speaking = stop speaking
    POST /stop       stop everything and drop what was heard
    POST /enabled    {enabled: bool} turn the voice off or back on
"""

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

from contexts.voice.audio import Microphone, Speaker
from contexts.voice.config import VoiceConfig, gateway_token
from contexts.voice.engines import PiperTts, WhisperCli
from contexts.voice.session import Ask, VoiceSession

ASK_TIMEOUT_S = 90.0


class EnabledBody(BaseModel):
    enabled: bool


def gateway_ask(cfg: VoiceConfig) -> Ask:
    """The same door the phone's Ask screen uses, so the question lands in the event log as experience.question.asked
    and the answer comes back as agent.answer.generated."""
    def ask(text: str) -> tuple[str, int]:
        headers = {"Authorization": f"Bearer {gateway_token()}"} if gateway_token() else {}
        r = httpx.post(cfg.ask_url.rstrip("/") + "/api/ask", json={"text": text}, headers=headers, timeout=ASK_TIMEOUT_S)
        r.raise_for_status()
        body = r.json()
        return body["answer"], len(body.get("citations", []))
    return ask


def build_session(cfg: VoiceConfig) -> VoiceSession:
    return VoiceSession(cfg, Microphone(cfg.mic_device, cfg.sample_rate), Speaker(cfg.speaker_mac, cfg.speaker_sink),
                        WhisperCli(cfg), PiperTts(cfg), gateway_ask(cfg))


def create_app(session: VoiceSession) -> FastAPI:
    app = FastAPI(title="Atlas voice")

    @app.get("/status")
    def status():
        return session.status()

    @app.post("/talk")
    def talk():
        return session.talk()

    @app.post("/stop")
    def stop():
        return session.stop()

    @app.post("/enabled")
    def enabled(body: EnabledBody):
        return session.set_enabled(body.enabled)

    return app
