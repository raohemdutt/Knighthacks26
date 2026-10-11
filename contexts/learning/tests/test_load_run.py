"""The "16 people asking" button's routes (contexts/learning/load_run.py), with a fake pair server (an httpx mock
transport that streams chat completions) and fake meters: nothing here reaches a pod."""

from __future__ import annotations

import asyncio
import json
import time

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from contexts.circle.brain_pair import BrainPair, Pod
from contexts.learning import load_run as lr
from contexts.learning.lab_deploy import DeployRunner, lab_deploy_router

ENERGY = {"joules": 3000.0, "watts_avg": 100.0, "seconds": 30.0,
          "pods": {"C": {"joules": 1600.0, "watts_avg": 53.3, "idle_w": 10.27, "samples": 30},
                   "D": {"joules": 1400.0, "watts_avg": 46.7, "idle_w": 10.27, "samples": 30}}}
TOKENS_PER_ANSWER = 5


class FakePairServer:
    """Streams a short answer for each chat completion after a small delay; records how many were in flight at once."""

    def __init__(self, *, status: int = 200, delay: float = 0.05) -> None:
        self.status = status
        self.delay = delay
        self.in_flight = 0
        self.max_in_flight = 0
        self.served = 0
        self.bodies: list[dict] = []
        self.paths: set[str] = set()

    async def handler(self, request: httpx.Request) -> httpx.Response:
        self.paths.add(request.url.path)
        body = json.loads(request.content)
        self.bodies.append(body)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.in_flight -= 1
        if self.status != 200:
            return httpx.Response(self.status, text="no")
        self.served += 1
        lines = [f"data: {json.dumps({'choices': [{'delta': {'content': f'w{i} '}}]})}" for i in range(TOKENS_PER_ANSWER)]
        lines.append(f"data: {json.dumps({'choices': [], 'usage': {'completion_tokens': TOKENS_PER_ANSWER}})}")
        lines.append("data: [DONE]")
        return httpx.Response(200, text="\n\n".join(lines) + "\n\n", headers={"content-type": "text/event-stream"})


def fast() -> float:
    return time.monotonic() * 20  # 10 asked seconds pass in half a real second


def build(server: FakePairServer, *, energy=None, available=lambda: None, configured=True):
    windows: list[tuple[float, float]] = []

    def make_pair() -> BrainPair:
        head = Pod("pod C", "head", "pod-a", "https://hub.test/user/c/proxy" if configured else None, "tok-c")
        return BrainPair(head, Pod("pod D", "half", "pod-b"), transport=httpx.MockTransport(server.handler),
                         reason=None if configured else "missing keys: AUP_HUB_TOKEN_C")

    async def measure(pair, start: float, end: float):
        windows.append((start, end))
        if isinstance(energy, Exception):
            raise energy
        return ENERGY if energy is None else energy

    runner = lr.LoadRunner(build_pair=make_pair, energy=measure, available=available, monotonic=fast)
    app = FastAPI()
    app.include_router(lab_deploy_router(DeployRunner("models", "state"), load=runner))
    return TestClient(app), runner, windows


def test_sixteen_streams_run_at_once_and_the_counts_add_up():
    server = FakePairServer()
    c, runner, windows = build(server)
    r = c.post("/api/learning/lab/load", json={"streams": 16, "seconds": 10})
    assert r.status_code == 202 and r.json()["state"] == "running" and r.json()["streams"] == 16
    runner.wait(20)
    v = c.get("/api/learning/lab/load").json()
    assert v["state"] == "done" and v["errors"] == 0 and v["message"] is None
    assert server.max_in_flight == 16  # really concurrent, and never more than asked
    assert v["requests"] == server.served >= 16
    assert v["tokens"] == server.served * TOKENS_PER_ANSWER
    wall = (v["ended_ms"] - v["started_ms"]) / 1000
    assert abs(v["tok_s_total"] - v["tokens"] / wall) < 0.01 * v["tok_s_total"]  # the view rounds to ms
    assert abs(v["tok_s_per_stream"] - v["tok_s_total"] / 16) < 0.02
    assert v["ttft_s"] is not None and v["ttft_s"] > 0
    assert v["energy"] == ENERGY and len(windows) == 1
    assert v["j_per_token"] == round(3000.0 / v["tokens"], 3)
    # only the pair's head, port 8000, plain questions, no tools, thinking off, 96 tokens
    assert server.paths == {"/user/c/proxy/8000/v1/chat/completions"}
    body = server.bodies[0]
    assert body["model"] == "teacher" and body["max_tokens"] == 96 and "tools" not in body
    assert body["chat_template_kwargs"] == {"enable_thinking": False} and body["stream"] is True
    assert {b["messages"][0]["content"] for b in server.bodies} <= set(lr.QUESTIONS)


def test_a_second_press_while_running_is_409():
    server = FakePairServer(delay=0.2)
    c, runner, _ = build(server)
    assert c.post("/api/learning/lab/load", json={"streams": 4, "seconds": 10}).status_code == 202
    again = c.post("/api/learning/lab/load", json={"streams": 4, "seconds": 10})
    assert again.status_code == 409 and "already running" in again.json()["detail"]
    runner.wait(20)


def test_503_with_the_reason_when_the_pair_client_is_unavailable():
    server = FakePairServer()
    c, _, _ = build(server, available=lambda: "missing keys: AUP_HUB_TOKEN_C")
    r = c.post("/api/learning/lab/load", json={})
    assert r.status_code == 503 and "AUP_HUB_TOKEN_C" in r.json()["detail"]
    v = c.get("/api/learning/lab/load").json()
    assert v["available"] is False and v["state"] == "idle" and server.served == 0


def test_the_default_check_reads_the_pairs_keys():
    assert lr.pair_available(lambda: BrainPair(Pod("pod C", "head", "pod-a"), Pod("pod D", "half", "pod-b"),
                                               reason="missing keys: AUP_HUB_USER_C")) == "missing keys: AUP_HUB_USER_C"


def test_bad_bodies_are_400():
    c, _, _ = build(FakePairServer())
    for body in ({"streams": 0}, {"streams": 33}, {"streams": "16"}, {"streams": True}, {"seconds": 5},
                 {"seconds": 500}, {"seconds": "30"}):
        assert c.post("/api/learning/lab/load", json=body).status_code == 400, body


def test_a_server_that_refuses_every_request_fails_honestly():
    server = FakePairServer(status=502, delay=0.01)
    c, runner, _ = build(server)
    assert c.post("/api/learning/lab/load", json={"streams": 2, "seconds": 10}).status_code == 202
    runner.wait(20)
    v = c.get("/api/learning/lab/load").json()
    assert v["state"] == "failed" and v["requests"] == 0 and v["errors"] > 0
    assert "HTTP 502" in v["message"] and v["tok_s_total"] is None and v["j_per_token"] is None


def test_meters_that_do_not_cover_the_window_give_null_energy():
    c, runner, _ = build(FakePairServer(), energy=RuntimeError("sampler down"))
    c.post("/api/learning/lab/load", json={"streams": 2, "seconds": 10})
    runner.wait(20)
    v = c.get("/api/learning/lab/load").json()
    assert v["state"] == "done" and v["energy"] is None and v["j_per_token"] is None and v["tokens"] > 0


def test_the_ramp_adds_streams_without_stopping_any_and_reports_each_stage():
    server = FakePairServer(delay=0.02)
    c, runner, windows = build(server)
    r = c.post("/api/learning/lab/load", json={"ramp": [1, 4, 16, 32], "seconds_per_stage": 10})
    assert r.status_code == 202 and r.json()["ramp"] == [1, 4, 16, 32] and r.json()["stage"] == 1
    runner.wait(30)
    v = c.get("/api/learning/lab/load").json()
    assert v["state"] == "done" and v["streams"] == 32 and v["seconds"] == 40 and v["stage"] is None
    assert server.max_in_flight == 32
    assert [s["streams"] for s in v["stages"]] == [1, 4, 16, 32]
    assert sum(s["requests"] for s in v["stages"]) == v["requests"] == server.served
    assert sum(s["tokens"] for s in v["stages"]) == v["tokens"]
    assert all(s["requests"] > 0 and s["watts_avg"] == 100.0 for s in v["stages"])
    rates = [s["tok_s_total"] for s in v["stages"]]
    assert rates == sorted(rates)  # the staircase rises
    # the stages tile the run with no gap: each starts where the one before ended
    assert all(a["ended_ms"] == b["started_ms"] for a, b in zip(v["stages"], v["stages"][1:], strict=False))
    assert len(windows) == 5  # the whole run, then each stage


def test_bad_ramps_are_400():
    c, _, _ = build(FakePairServer())
    for body in ({"ramp": []}, {"ramp": [4, 1]}, {"ramp": [1, 33]}, {"ramp": [1, 4], "seconds_per_stage": 5},
                 {"ramp": "1,4"}):
        assert c.post("/api/learning/lab/load", json=body).status_code == 400, body
