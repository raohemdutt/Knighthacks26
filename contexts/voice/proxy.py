"""The gateway's door to the voice service (/api/voice/*), so the operator screen reaches it through the one port the
phone and laptop already use. Reads are open like the other GETs; the buttons need the token when one is set."""

import os
from collections.abc import Callable

import httpx
from fastapi import APIRouter, Depends, HTTPException

OFF = {"state": "off", "running": False, "problem": "The voice service is not running on this box.",
       "heard": "", "answer": "", "seconds_in_state": 0}


def voice_url() -> str:
    return os.environ.get("ATLAS_VOICE_URL", "http://127.0.0.1:8091").rstrip("/")


def voice_router(guard: Callable | None = None, transport: httpx.AsyncBaseTransport | None = None) -> APIRouter:
    router = APIRouter(prefix="/voice")
    deps = [Depends(guard)] if guard else []

    async def call(method: str, path: str, **kwargs) -> dict:
        async with httpx.AsyncClient(timeout=5, transport=transport) as client:
            r = await client.request(method, voice_url() + path, **kwargs)
        r.raise_for_status()
        return r.json()

    @router.get("/status")
    async def status():
        try:
            return await call("GET", "/status")
        except httpx.HTTPError:
            return OFF

    @router.post("/talk", dependencies=deps)
    async def talk():
        return await _write("/talk")

    @router.post("/listen", dependencies=deps)
    async def listen():
        return await _write("/listen")

    @router.post("/stop", dependencies=deps)
    async def stop():
        return await _write("/stop")

    @router.post("/enabled", dependencies=deps)
    async def enabled(body: dict):
        return await _write("/enabled", json={"enabled": bool(body.get("enabled"))})

    @router.post("/say", dependencies=deps)
    async def say(body: dict):
        return await _write("/say", json={"text": str(body.get("text", ""))})

    async def _write(path: str, **kwargs) -> dict:
        try:
            return await call("POST", path, **kwargs)
        except httpx.HTTPError:
            raise HTTPException(503, OFF["problem"]) from None

    return router


def voice_check(client: httpx.Client | None = None) -> dict:
    """One line for the demo pre-flight (D9). Voice is optional, so a service that is not running passes; one that is
    running must have a microphone, a speaker and its speech files. Everything it uses is on the box, so offline is fine."""
    name = "voice on the box"
    try:
        st = (client or httpx).get(voice_url() + "/status", timeout=1.5).json()
    except (httpx.HTTPError, ValueError):
        return {"name": name, "ok": True, "detail": "not running (optional)", "fix": ""}
    problems = [f"{label}: {st[key]['detail']}" for key, label in (("mic", "microphone"), ("speaker", "speaker"),
                ("speech_to_text", "speech-to-text"), ("text_to_speech", "text-to-speech")) if not st.get(key, {}).get("ok", True)]
    if st.get("state") == "off":
        problems.append("switched off")
    ok = not problems
    return {"name": name, "ok": ok, "detail": "ready, all local" if ok else "; ".join(problems),
            "fix": "" if ok else "pair the speaker (jetson/README.md, Voice), plug the webcam in, or switch the voice on from the operator screen"}
