"""The fake sampler publishes the recorded minute on the bus; the projection is then rebuilt from the log alone."""

import asyncio
from datetime import UTC, datetime, timedelta

from bus import EventBus
from contexts.telemetry.application.projection import TelemetryProjection
from contexts.telemetry.application.sampling import SamplerRunner
from contexts.telemetry.domain.nodes import KNOWN_NODES
from contexts.telemetry.fakes import FakeSampler
from contexts.telemetry.infrastructure import recorded

START = datetime(2026, 10, 11, 9, 30, tzinfo=UTC)


def at(s: float) -> datetime:
    return START + timedelta(seconds=s)


async def publish_the_minute(log_dir) -> TelemetryProjection:
    bus = EventBus(log_dir)
    live = TelemetryProjection()
    live.subscribe(bus)
    sampler = FakeSampler(recorded.load(), speed=0, clock=lambda: START)
    await SamplerRunner(sampler, bus.publish, nodes=KNOWN_NODES).run()
    return live


async def rebuild(log_dir) -> tuple[TelemetryProjection, int]:
    bus = EventBus(log_dir)  # a restart: a new bus and an empty projection over the same log
    rebuilt = TelemetryProjection()
    rebuilt.subscribe(bus)
    return rebuilt, await bus.replay_into()


def test_a_projection_rebuilt_from_the_log_matches_the_one_that_watched_it_live(tmp_path):
    live = asyncio.run(publish_the_minute(tmp_path))
    rebuilt, replayed = asyncio.run(rebuild(tmp_path))

    assert replayed == 206 and rebuilt.snapshot() == live.snapshot()
    snap = rebuilt.snapshot()
    assert snap["counts"] == {"telemetry.gpu.power": 120, "telemetry.jetson.power": 60, "telemetry.request": 2,
                              "telemetry.train.step": 24}
    jetson, gpu_a, gpu_b = (snap["nodes"][n]["power"]["fake"] for n in ("jetson", "gpu-a", "gpu-b"))
    assert (jetson["samples"], jetson["latest_w"], jetson["latest_ts"], jetson["measured_period_ms"]) == (
        60, 6.4, at(59), 1000.0)
    assert (gpu_a["samples"], gpu_a["missing"], gpu_a["latest_w"], gpu_a["boundary"]) == (
        60, 1, 39.0, "graphics package")
    assert (gpu_b["samples"], gpu_b["latest_w"], gpu_b["latest_ts"]) == (60, 245.0, at(59.5))
    assert [(r["request_id"], r["duration_s"], r["prefill_s"], r["tokens_out"]) for node in ("jetson", "gpu-a")
            for r in snap["requests"][node]] == [("fake-req-student-1", 4.5, 0.62, 61),
                                                  ("fake-req-teacher-1", 5.3, 1.09, 28)]
    run = snap["train_runs"]["fake-run-0001"]
    assert (run["steps"], run["step"], run["last_ts"], run["node"]) == (24, 124, at(58.5), "gpu-b")


def test_the_teacher_window_can_be_read_back_from_the_rebuilt_projection(tmp_path):
    asyncio.run(publish_the_minute(tmp_path))
    rebuilt, _ = asyncio.run(rebuild(tmp_path))
    [window] = rebuilt.requests("gpu-a")
    inside = rebuilt.readings("gpu-a", window.started, window.ended)
    assert [(r.ts, r.watts) for r in inside] == [(at(30.2), 300.0), (at(31.2), 212.0), (at(32.2), 201.0),
                                                 (at(33.2), 199.0), (at(34.2), 203.0), (at(35.2), 196.0)]


def test_replaying_the_same_log_again_changes_nothing(tmp_path):
    asyncio.run(publish_the_minute(tmp_path))

    async def twice():
        bus = EventBus(tmp_path)
        projection = TelemetryProjection()
        projection.subscribe(bus)
        await bus.replay_into()
        once = projection.snapshot()
        await bus.replay_into()  # the bus has seen every event_id: nothing is folded twice
        return once, projection.snapshot()

    once, again = asyncio.run(twice())
    assert again == once
