"""The shared brain's watts: the sampler relayed with the token kept on the box, and the last answer's joules."""

import asyncio
import json
from datetime import UTC, datetime

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from contexts.circle.brain_power import BrainPower, brain_power_router, build_shared_brain
from contracts.circle import BrainAnswered, BrainAsked, BrainText

T0 = 1_791_700_000.0  # pod clock, epoch seconds
OFFSET = 60.0  # the pod's clock runs a minute ahead of the box's


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).isoformat().replace("+00:00", "Z")


class FakeSampler:
    """lab/pod_power.py's two routes: 10 W idle, 100 W from T0+20 to T0+30, one sample a second."""

    def __init__(self, now: list[float]) -> None:
        self.now = now
        self.seen_tokens: list[str] = []

    def watts(self, t: float) -> float:
        return 100.0 if T0 + 20 <= t < T0 + 30 else 10.0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.seen_tokens.append(request.headers.get("Authorization", ""))
        pod_now = self.now[0] + OFFSET
        last = int(pod_now - T0)
        stamp = {"pod_received": iso(pod_now), "pod_sent": iso(pod_now)}
        if request.url.path.endswith("/status"):
            return httpx.Response(200, json={"ok": True, "last_seq": last, **stamp})
        after = int(request.url.params.get("after", 0))
        samples = [{"seq": q, "event": {"ts": iso(T0 + q), "socket_w": self.watts(T0 + q)}}
                   for q in range(max(after + 1, 0), last + 1)]
        return httpx.Response(200, json={"ok": True, "last_seq": last, "samples": samples, **stamp})


def asked(ask_id: str, t: float) -> BrainAsked:
    return BrainAsked(ask_id=ask_id, circle_id="team", member_id="ilan", opt_in_ref="c1", purpose="plan", node="pod-a",
                      host="sjc.aupcloud.io", model="group-brain", max_tokens=256,
                      messages=[BrainText(role="user", content="where?")], chars=6,
                      ts=datetime.fromtimestamp(t, UTC))


def answered(ask_id: str, t: float) -> BrainAnswered:
    return BrainAnswered(ask_id=ask_id, circle_id="team", member_id="ilan", ok=True, answer="the library",
                         latency_ms=10_000, ts=datetime.fromtimestamp(t, UTC))


def make(now: list[float]):
    sampler = FakeSampler(now)
    power = BrainPower("https://hub.example/user/u/proxy/8790", "secret-token", host="hub.example",
                       transport=httpx.MockTransport(sampler), clock=lambda: now[0])
    app = FastAPI()
    app.include_router(brain_power_router(power))
    return power, sampler, TestClient(app)


def test_relays_the_last_two_minutes_and_keeps_the_token_server_side():
    now = [T0 + 200 - OFFSET]
    power, sampler, client = make(now)
    body = client.get("/api/circle/brain/power").json()
    assert body["configured"] is True and body["error"] is None
    assert len(body["samples"]) == 120 and body["samples"][-1]["seq"] == 200
    assert abs(body["offset_s"] - OFFSET) < 0.01
    assert all(t == "token secret-token" for t in sampler.seen_tokens)
    assert "secret-token" not in json.dumps(body)
    now[0] += 3
    nxt = client.get("/api/circle/brain/power", params={"after": 200}).json()
    assert [s["seq"] for s in nxt["samples"]] == [201, 202, 203]


def test_the_last_answer_costs_its_window_on_the_pods_clock():
    now = [T0 + 40 - OFFSET]
    power, _, client = make(now)
    power.on_event(asked("a1", T0 + 20 - OFFSET))  # box clock: the pod's minus a minute
    power.on_event(answered("a1", T0 + 30 - OFFSET))
    last = client.get("/api/circle/brain/power").json()["last_answer"]
    assert last["seconds"] == 10.0
    assert abs(last["joules"] - 1000.0) < 50  # 10 s at 100 W, the edges interpolated
    assert abs(last["above_idle_j"] - (last["joules"] - 102.7)) < 0.2
    assert last["low_confidence"] is False or last["samples"] < 10


def test_no_account_answers_unconfigured_in_every_state():
    app = FastAPI()
    app.include_router(brain_power_router(None, "no brain power reading: missing keys"))
    body = TestClient(app).get("/api/circle/brain/power").json()
    assert body["configured"] is False and body["samples"] == [] and "missing keys" in body["error"]


def test_without_a_circle_nothing_is_built():
    assert build_shared_brain({}, bus=None) is None  # type: ignore[arg-type]
    asyncio.run(asyncio.sleep(0))
