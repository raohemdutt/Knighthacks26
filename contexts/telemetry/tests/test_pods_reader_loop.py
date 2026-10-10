"""The pod reader loop: what the cursor means, and what one slow pod costs the other.

The hub proxy is the boundary (an httpx MockTransport answering as lab/pod_power.py's endpoint does, from the
recorded samples in fixtures/telemetry/pod_power). The reader and the runner are the real ones.
"""

import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from contexts.telemetry.application.sampling import SamplerRunner
from contexts.telemetry.infrastructure.pods import PodEndpoint, PodReader

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "telemetry" / "pod_power"
TOKEN = "hub-token-for-tests"
NOW = datetime(2026, 10, 10, 18, 2, tzinfo=UTC)


def recorded(name: str) -> list[dict]:
    return [json.loads(line) for line in (FIXTURES / name).read_text(encoding="utf-8").splitlines()]


class FakePod:
    """GET /samples as lab/pod_power.py answers it, over a list of (seq, event) it holds."""

    def __init__(self, node: str, events: list[dict], started: str = "2026-10-10T18:00:00.000Z") -> None:
        self.node = node
        self.held = list(enumerate(events, 1))
        self.started = started
        self.requests: list[httpx.Request] = []

    def answer(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        after, limit = int(request.url.params["after"]), int(request.url.params["limit"])
        page = [(seq, e) for seq, e in self.held if seq > after][:limit]
        return httpx.Response(200, json={
            "ok": True, "node": self.node, "sampler_started": self.started,
            "first_seq": self.held[0][0] if self.held else None, "last_seq": self.held[-1][0] if self.held else 0,
            "samples": [{"seq": seq, "event": e} for seq, e in page]})


def test_a_publish_failure_mid_batch_keeps_the_rest_of_the_batch():
    events = recorded("samples_gpu-a.jsonl")[:10]
    pod = FakePod("pod-a", events)
    reader = PodReader([PodEndpoint("pod-a", "https://hub.test/user/acct/proxy/8790", TOKEN)],
                       transport=httpx.MockTransport(pod.answer), clock=lambda: NOW,
                       sleep=lambda _: asyncio.sleep(0), rounds=1)
    published: list = []
    calls = 0

    async def publish(event) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("the event log could not be written")
        published.append(event)

    async def no_sleep(_seconds: float) -> None:
        await asyncio.sleep(0)

    runner = SamplerRunner(reader, publish, nodes=["pod-a"], sleep=no_sleep)

    async def run() -> None:
        await asyncio.wait_for(runner.run(), timeout=10)

    asyncio.run(run())

    # Every sample the pod held reaches the log once, in order, the one whose write failed on the retry.
    assert [e.event_id for e in published] == [e["event_id"] for e in events]
    # The cursor sits past what was published, not past what was merely fetched.
    assert reader.status()["pod-a"]["after"] == 10


def test_a_stalled_pod_does_not_delay_the_other_pods_samples():
    a_events: list[dict] = []
    b_events = [{**e, "node": "pod-b"} for e in recorded("samples_gpu-b_stale.jsonl")[:3]]
    a, b = FakePod("pod-a", a_events), FakePod("pod-b", b_events)

    async def route(request: httpx.Request) -> httpx.Response:
        if "/acct-a/" in request.url.path:
            await asyncio.sleep(0.6)  # the hub proxy holds this request, it does not refuse it
            return a.answer(request)
        return b.answer(request)

    reader = PodReader([PodEndpoint("pod-a", "https://hub.test/user/acct-a/proxy/8790", TOKEN),
                        PodEndpoint("pod-b", "https://hub.test/user/acct-b/proxy/8790", TOKEN)],
                       transport=httpx.MockTransport(route), clock=lambda: NOW,
                       sleep=lambda _: asyncio.sleep(0), timeout_s=5, rounds=1)

    async def first_b_event_after() -> tuple[float, list[str]]:
        started = time.monotonic()
        waited = None
        nodes = []
        async for event in reader.samples():
            if waited is None:
                waited = time.monotonic() - started
            nodes.append(event.node)
        assert waited is not None, "the reader yielded no event at all"
        return waited, nodes

    waited, nodes = asyncio.run(asyncio.wait_for(first_b_event_after(), timeout=10))

    assert nodes == ["pod-b"] * 3
    assert waited < 0.3, f"pod-b's first sample came {waited:.2f} s after the read began, held up by pod-a's stall"


def test_a_pod_whose_samples_name_another_node_is_logged_with_its_node_the_count_and_the_reason(caplog):
    events = recorded("samples_gpu-a.jsonl")[:3]
    wrong = [{**e, "node": "pod-b"} for e in events[:2]] + [events[2]]
    pod = FakePod("pod-a", wrong)
    reader = PodReader([PodEndpoint("pod-a", "https://hub.test/user/acct/proxy/8790", TOKEN)],
                       transport=httpx.MockTransport(pod.answer), clock=lambda: NOW,
                       sleep=lambda _: asyncio.sleep(0), rounds=1)

    async def run():
        return [e async for e in reader.samples()]

    with caplog.at_level("WARNING", logger="atlas.telemetry.pods"):
        got = asyncio.run(run())

    assert [e.event_id for e in got] == [events[2]["event_id"]]
    assert reader.status()["pod-a"]["rejected"] == 2
    warnings = [r for r in caplog.records if r.name == "atlas.telemetry.pods" and r.levelname == "WARNING"]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]
    message = warnings[0].getMessage()
    assert "pod-a" in message and "2" in message and "pod-b" in message, message
    assert TOKEN not in caplog.text


def test_a_sample_that_is_not_a_contract_event_is_logged_with_the_reason(caplog):
    events = recorded("samples_gpu-a.jsonl")[:2]
    broken = [{**events[0], "socket_w": "lots"}, events[1]]
    pod = FakePod("pod-a", broken)
    reader = PodReader([PodEndpoint("pod-a", "https://hub.test/user/acct/proxy/8790", TOKEN)],
                       transport=httpx.MockTransport(pod.answer), clock=lambda: NOW,
                       sleep=lambda _: asyncio.sleep(0), rounds=1)

    async def run():
        return [e async for e in reader.samples()]

    with caplog.at_level("WARNING", logger="atlas.telemetry.pods"):
        got = asyncio.run(run())

    assert [e.event_id for e in got] == [events[1]["event_id"]]
    warnings = [r.getMessage() for r in caplog.records if r.name == "atlas.telemetry.pods" and r.levelname == "WARNING"]
    assert len(warnings) == 1 and "pod-a" in warnings[0] and "1" in warnings[0] and "socket_w" in warnings[0], warnings
