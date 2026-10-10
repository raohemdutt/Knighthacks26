"""The sampler runner: publishes what a sampler yields, fails closed on what it should not, survives a crash."""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from bus import EventBus
from contexts.telemetry.application.sampling import SamplerRunner
from contracts.agent import AlertRaised
from contracts.telemetry import GpuPower, JetsonPower

T0 = datetime(2026, 10, 10, 14, 0, tzinfo=UTC)


def jetson(s: int) -> JetsonPower:
    return JetsonPower(ts=T0 + timedelta(seconds=s), node="jetson", source="tegrastats", period_ms=1000,
                       rails={"VDD_IN": 6400}, nvpmodel="MAXN_SUPER")


def pod(s: int, node: str = "gpu-a") -> GpuPower:
    return GpuPower(ts=T0 + timedelta(seconds=s), node=node, source="amd-smi", period_ms=1000, socket_w=39.0,
                    gfx_mhz=500, mem_mhz=96, throttled=False)


class Scripted:
    """A sampler that yields its script; an Exception in the script is raised at that point, once."""

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.starts = 0

    async def samples(self):
        self.starts += 1
        while self.script:
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise item
            yield item


class Endless:
    async def samples(self):
        s = 0
        while True:
            yield jetson(s)
            s += 1
            await asyncio.sleep(0)


def logged(tmp_path) -> list:
    return list(EventBus(tmp_path).read_log())


def test_every_sample_goes_onto_the_bus_and_into_the_log(tmp_path):
    events = [jetson(0), jetson(1), jetson(2)]
    seen = []

    async def run():
        bus = EventBus(tmp_path)
        bus.subscribe("telemetry.", seen.append)
        runner = SamplerRunner(Scripted(events), bus.publish, nodes=["jetson"])
        await runner.run()
        return runner

    runner = asyncio.run(run())
    assert seen == events and logged(tmp_path) == events
    assert runner.status() == {"nodes": ["jetson"], "published": 3, "dropped": 0, "errors": 0, "last_error": None}


def test_events_for_another_node_or_another_context_are_dropped_not_published(tmp_path):
    alert = AlertRaised(alert_id="a1", rule_id="r", message="not telemetry")
    script = [jetson(0), pod(0, "gpu-a"), alert, jetson(1)]

    async def run():
        runner = SamplerRunner(Scripted(script), EventBus(tmp_path).publish, nodes=["jetson"])
        await runner.run()
        return runner

    runner = asyncio.run(run())
    assert logged(tmp_path) == [e for e in script if isinstance(e, JetsonPower)]
    assert (runner.published, runner.dropped) == (2, 2)


def test_a_sampler_that_raises_is_started_again_after_the_retry_delay(tmp_path):
    sampler = Scripted([pod(0), RuntimeError("amd-smi exited 1"), pod(1), pod(2)])
    waits = []

    async def fake_sleep(s: float) -> None:
        waits.append(s)

    async def run():
        runner = SamplerRunner(sampler, EventBus(tmp_path).publish, nodes=["gpu-a"], retry_s=7.5, sleep=fake_sleep)
        await runner.run()
        return runner

    runner = asyncio.run(run())
    assert [e.ts for e in logged(tmp_path)] == [T0, T0 + timedelta(seconds=1), T0 + timedelta(seconds=2)]
    assert (waits, sampler.starts) == ([7.5], 2)
    assert runner.status() == {"nodes": ["gpu-a"], "published": 3, "dropped": 0, "errors": 1,
                               "last_error": "RuntimeError: amd-smi exited 1"}


def test_cancelling_the_task_stops_a_sampler_that_never_ends(tmp_path):
    async def run():
        bus = EventBus(tmp_path)
        runner = SamplerRunner(Endless(), bus.publish, nodes=["jetson"])
        task = asyncio.create_task(runner.run())
        while runner.published < 5:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return runner

    runner = asyncio.run(run())
    assert runner.errors == 0 and len(logged(tmp_path)) == runner.published >= 5


def test_a_runner_cannot_be_started_for_a_node_atlas_does_not_know():
    with pytest.raises(ValueError, match="unknown node 'gpu-c'"):
        SamplerRunner(Scripted([]), EventBus(persist=False).publish, nodes=["jetson", "gpu-c"])


class Holding:
    """A sampler that holds something until it is closed (as the tegrastats sampler holds a child process)."""

    def __init__(self) -> None:
        self.closed = False

    async def samples(self):
        try:
            s = 0
            while True:
                yield jetson(s)
                s += 1
        finally:
            self.closed = True


def test_cancelling_the_runner_closes_the_sampler_before_the_task_ends(tmp_path):
    sampler = Holding()
    gate = asyncio.Event()

    async def publish(event) -> None:
        await gate.wait()  # the runner is cancelled while it waits here, the sampler paused at its yield

    async def run():
        runner = SamplerRunner(sampler, publish, nodes=["jetson"])
        task = asyncio.create_task(runner.run())
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return sampler.closed

    assert asyncio.run(run()) is True
