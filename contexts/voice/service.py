"""The voice service's HTTP face, for the gateway and for curl. It listens on localhost only; the gateway proxies it
at /api/voice/* (with the token check on writes), which is how the operator screen's button reaches it.

    GET  /status     state (off | idle | listening | thinking | speaking), what was heard and answered, device health
    POST /talk       push to talk: start listening; again = that was the whole question; while speaking = stop speaking
    POST /stop       stop everything and drop what was heard
    POST /enabled    {enabled: bool} turn the voice off or back on
    POST /listen     like /talk, but only listens: what was heard is left in `heard` and nothing is asked or spoken
    POST /say        {text} speak this on the box's speaker (the Ask screen's answers); only while idle
"""

import json
from collections.abc import Callable

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

from contexts.voice.audio import Microphone, Speaker
from contexts.voice.config import VoiceConfig, gateway_token
from contexts.voice.engines import PiperTts, WhisperCli
from contexts.voice.session import Ask, AskStream, VoiceSession

ASK_TIMEOUT_S = 90.0


class EnabledBody(BaseModel):
    enabled: bool


class SayBody(BaseModel):
    text: str


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


def gateway_ask_stream(cfg: VoiceConfig) -> AskStream:
    """/api/ask/stream: the same question and answer, but each piece of the answer is handed over as it is written, so the
    first sentence can be spoken while the rest is still coming."""
    def ask_stream(text: str, on_piece: Callable[[str], None]) -> tuple[str, int]:
        headers = {"Authorization": f"Bearer {gateway_token()}"} if gateway_token() else {}
        with httpx.stream("POST", cfg.ask_url.rstrip("/") + "/api/ask/stream", json={"text": text}, headers=headers,
                          timeout=ASK_TIMEOUT_S) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                msg = json.loads(line)
                if "delta" in msg:
                    on_piece(msg["delta"])
                elif "final" in msg:
                    return msg["final"]["answer"], len(msg["final"].get("citations", []))
                elif "error" in msg:
                    raise RuntimeError(msg["error"])
        raise RuntimeError("the answer ended early")
    return ask_stream


def build_session(cfg: VoiceConfig) -> VoiceSession:
    return VoiceSession(cfg, Microphone(cfg.mic_device, cfg.sample_rate), Speaker(cfg.speaker_mac, cfg.speaker_sink),
                        WhisperCli(cfg), PiperTts(cfg), gateway_ask(cfg), ask_stream=gateway_ask_stream(cfg))


def create_app(session: VoiceSession) -> FastAPI:
    app = FastAPI(title="Atlas voice")

    @app.get("/status")
    def status():
        return session.status()

    @app.post("/talk")
    def talk():
        return session.talk()

    @app.post("/listen")
    def listen():
        return session.talk(ask=False)

    @app.post("/stop")
    def stop():
        return session.stop()

    @app.post("/enabled")
    def enabled(body: EnabledBody):
        return session.set_enabled(body.enabled)

    @app.post("/say")
    def say(body: SayBody):
        return session.say(body.text)

    return app
