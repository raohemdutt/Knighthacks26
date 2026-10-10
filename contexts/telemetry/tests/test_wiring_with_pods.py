"""The pods source next to main's tegrastats source: one build_telemetry, one `env` keyword, one status for the reader.

The hub proxy is the boundary: an httpx MockTransport answers as lab/pod_power.py's endpoint does, from the recorded
samples in fixtures/telemetry/pod_power. Everything else (build_telemetry, SamplerRunner, PodReader, the bus) is real.
"""

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from bus import EventBus
from contexts.telemetry.application.projection import TelemetryProjection
from contexts.telemetry.application.sampling import SamplerRunner
from contexts.telemetry.application.service import TelemetryService
from contexts.telemetry.infrastructure.pods import PodReader, pod_endpoints
from contexts.telemetry.wiring import build_telemetry

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "telemetry" / "pod_power"
ENV_A = {"AUP_HUB_USER_A": "acct", "AUP_HUB_TOKEN_A": "t"}


def recorded_samples(count: int) -> list[dict]:
    path = FIXTURES / "samples_gpu-a.jsonl"
    assert path.is_file(), "expected recorded fixture fixtures/telemetry/pod_power/samples_gpu-a.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()][:count]


def fake_pod(events: list[dict]):
    """GET /samples as lab/pod_power.py answers it, over the events it holds (seq counting from 1)."""
    held = list(enumerate(events, 1))

    def answer(request: httpx.Request) -> httpx.Response:
        after, limit = int(request.url.params["after"]), int(request.url.params["limit"])
        page = [(seq, e) for seq, e in held if seq > after][:limit]
        return httpx.Response(200, json={
            "ok": True, "node": "gpu-a", "sampler_started": "2026-10-10T18:00:00.000Z",
            "first_seq": held[0][0], "last_seq": held[-1][0],
            "samples": [{"seq": seq, "event": e} for seq, e in page]})

    return answer


def test_the_wiring_knows_the_fake_tegrastats_and_pods_sources_and_takes_main_env_keyword():
    with pytest.raises(ValueError, match="unknown telemetry source 'nope'; known sources: fake, tegrastats, pods$"):
        build_telemetry("nope", EventBus(persist=False), on_jetson=lambda: False)

    service = build_telemetry("pods", EventBus(persist=False), env=ENV_A, on_jetson=lambda: False)

    assert service.source == "pods"
    assert isinstance(service.runner.sampler, PodReader)


def test_the_status_carries_the_pod_readers_counters_once_under_source():
    events = recorded_samples(5)
    assert len(events) == 5, "expected five recorded samples in fixtures/telemetry/pod_power/samples_gpu-a.jsonl"

    # Built from the real pieces build_telemetry("pods") joins, not through its env keyword, so that this test
    # fails only for what the status shows and not for the wiring the other test in this file covers.
    async def run():
        bus = EventBus(persist=False)
        endpoints = pod_endpoints(ENV_A)
        reader = PodReader(endpoints, transport=httpx.MockTransport(fake_pod(events)), rounds=1)
        runner = SamplerRunner(reader, bus.publish, nodes=[pod.node for pod in endpoints])
        service = TelemetryService(TelemetryProjection(), runner, source="pods")
        service.subscribe(bus)
        await service.run()
        return service

    service = asyncio.run(run())
    status = service.status()

    assert service.runner.sampler.status()["gpu-a"]["fetched"] == 5
    assert status["sampler"]["source"] == service.runner.sampler.status()
    assert "own" not in status["sampler"]
