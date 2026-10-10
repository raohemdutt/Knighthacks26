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
    pod = FakePod("gpu-a", events)
    reader = PodReader([PodEndpoint("gpu-a", "https://hub.test/user/acct/proxy/8790", TOKEN)],
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

    runner = SamplerRunner(reader, publish, nodes=["gpu-a"], sleep=no_sleep)

    async def run() -> None:
        await asyncio.wait_for(runner.run(), timeout=10)

    asyncio.run(run())

    # Every sample the pod held reaches the log once, in order, the one whose write failed on the retry.
    assert [e.event_id for e in published] == [e["event_id"] for e in events]
    # The cursor sits past what was published, not past what was merely fetched.
    assert reader.status()["gpu-a"]["after"] == 10


def test_a_stalled_pod_does_not_delay_the_other_pods_samples():
    a_events: list[dict] = []
    b_events = [{**e, "node": "gpu-b"} for e in recorded("samples_gpu-b_stale.jsonl")[:3]]
    a, b = FakePod("gpu-a", a_events), FakePod("gpu-b", b_events)

    async def route(request: httpx.Request) -> httpx.Response:
        if "/acct-a/" in request.url.path:
            await asyncio.sleep(0.6)  # the hub proxy holds this request, it does not refuse it
            return a.answer(request)
        return b.answer(request)

    reader = PodReader([PodEndpoint("gpu-a", "https://hub.test/user/acct-a/proxy/8790", TOKEN),
                        PodEndpoint("gpu-b", "https://hub.test/user/acct-b/proxy/8790", TOKEN)],
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

    assert nodes == ["gpu-b"] * 3
    assert waited < 0.3, f"gpu-b's first sample came {waited:.2f} s after the read began, held up by gpu-a's stall"
