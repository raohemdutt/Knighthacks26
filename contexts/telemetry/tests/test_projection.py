"""The projection folds `telemetry.*` events per node and source, delivered through the real bus."""

import asyncio
from datetime import UTC, datetime, timedelta

from bus import EventBus
from contexts.telemetry.application.projection import TelemetryProjection
from contexts.telemetry.domain.power import PowerReading
from contracts.events import Event
from contracts.telemetry import GpuPower, JetsonPower, RequestWindow, TrainStep

T0 = datetime(2026, 10, 10, 14, 0, tzinfo=UTC)


def at(s: float) -> datetime:
    return T0 + timedelta(seconds=s)


def gpu(s: float, w: float | None, node: str = "gpu-a", throttled: bool = False) -> GpuPower:
    return GpuPower(ts=at(s), node=node, source="amd-smi", period_ms=1000, socket_w=w, gfx_mhz=2350, mem_mhz=1258,
                    throttled=throttled, temps_c={"edge": 40.0 + s})


def jetson(s: float, vdd_in_mw: float) -> JetsonPower:
    return JetsonPower(ts=at(s), node="jetson", source="tegrastats", period_ms=1000, nvpmodel="MAXN_SUPER",
                       rails={"VDD_IN": vdd_in_mw, "VDD_CPU_GPU_CV": 1600, "VDD_SOC": 1900}, temps_c={"gpu": 48.0})


def fold(events: list[Event], projection: TelemetryProjection | None = None) -> TelemetryProjection:
    projection = projection or TelemetryProjection()

    async def run():
        bus = EventBus(persist=False)
        projection.subscribe(bus)
        await bus.publish_all(events)

    asyncio.run(run())
    return projection


def test_power_samples_fold_into_latest_watts_counts_and_the_measured_period_per_node_and_source():
    p = fold([jetson(0, 6400), jetson(1, 6500), jetson(2, 12800), jetson(5, 6400),
              gpu(0.2, 39.0), gpu(1.2, 300.0, throttled=True), gpu(2.2, None)])
    snap = p.snapshot()
    assert snap["nodes"]["jetson"]["power"] == {"tegrastats": {
        "boundary": "module", "samples": 4, "missing": 0, "latest_w": 6.4, "latest_ts": at(5), "first_ts": at(0),
        "period_ms": 1000.0, "measured_period_ms": 5000 / 3, "max_gap_ms": 3000.0, "out_of_order": 0}}
    assert snap["nodes"]["gpu-a"]["power"] == {"amd-smi": {
        "boundary": "graphics package", "samples": 3, "missing": 1, "latest_w": None, "latest_ts": at(2.2),
        "first_ts": at(0.2), "period_ms": 1000.0, "measured_period_ms": 1000.0, "max_gap_ms": 1000.0,
        "out_of_order": 0}}
    assert (snap["nodes"]["jetson"]["nvpmodel"], snap["nodes"]["jetson"]["temps_c"]) == ("MAXN_SUPER", {"gpu": 48.0})
    assert (snap["nodes"]["gpu-a"]["throttled"], snap["nodes"]["gpu-a"]["temps_c"]) == (False, {"edge": 42.2})
    assert snap["counts"] == {"telemetry.gpu.power": 3, "telemetry.jetson.power": 4}


def test_a_late_sample_is_kept_for_its_window_but_does_not_become_the_latest():
    p = fold([gpu(0, 40.0), gpu(2, 200.0), gpu(1, 300.0)])
    power = p.snapshot()["nodes"]["gpu-a"]["power"]["amd-smi"]
    assert (power["latest_w"], power["latest_ts"], power["out_of_order"], power["samples"]) == (200.0, at(2), 1, 3)
    assert [r.watts for r in p.readings("gpu-a")] == [40.0, 300.0, 200.0]  # by time, not by arrival


def test_the_same_event_delivered_twice_is_folded_once():
    sample = gpu(0, 40.0)
    p = fold([sample, sample, sample.model_copy()])
    assert p.snapshot()["nodes"]["gpu-a"]["power"]["amd-smi"]["samples"] == 1
    assert p.counts == {"telemetry.gpu.power": 1}


def test_a_source_the_projection_does_not_know_yet_is_counted_for_its_node_and_breaks_nothing():
    new_source = Event(type="telemetry.fan.speed", ts=at(0), node="gpu-a", rpm=1200)  # not registered anywhere
    p = fold([new_source, gpu(1, 40.0)])
    node = p.snapshot()["nodes"]["gpu-a"]
    assert node["events"] == {"telemetry.fan.speed": 1, "telemetry.gpu.power": 1}
    assert node["last_ts"] == at(1) and node["power"]["amd-smi"]["latest_w"] == 40.0


def test_readings_and_requests_answer_for_a_window():
    student = RequestWindow(ts=at(14.5), node="jetson", request_id="s1", started=at(10), first_token=at(10.5),
                            ended=at(14.5), tokens_in=412, tokens_out=61, target="student", model="atlas-base")
    later = RequestWindow(ts=at(40), node="jetson", request_id="s2", started=at(38), ended=at(40), tokens_in=20,
                          tokens_out=5, target="student")
    p = fold([jetson(s, 6400 + 100 * s) for s in range(0, 20)] + [student, later])
    assert p.readings("jetson", at(10), at(12)) == [
        PowerReading(node="jetson", ts=at(s), watts=(6400 + 100 * s) / 1000, source="tegrastats", boundary="module")
        for s in (10, 11, 12)]
    assert p.readings("jetson", at(10), at(12), source="sysfs") == []
    assert p.requests("jetson", at(14), at(15)) == [student]  # overlaps the window's end
    assert p.requests("jetson", at(15), at(37)) == []
    assert p.snapshot()["requests"]["jetson"][0] == {
        "request_id": "s1", "target": "student", "started": at(10), "ended": at(14.5), "duration_s": 4.5,
        "prefill_s": 0.5, "tokens_in": 412, "tokens_out": 61, "model": "atlas-base", "engine": None,
        "since_load": None, "clock": None}


def test_training_steps_keep_the_newest_step_per_run():
    steps = [TrainStep(ts=at(2.5 * k), node="gpu-b", run_id="run-1", step=100 + k, loss=1.2 - 0.1 * k,
                       step_ms=2500.0, socket_w=245.0) for k in (0, 2, 1)]  # one arrives late
    run = fold(steps).snapshot()["train_runs"]["run-1"]
    assert run == {"node": "gpu-b", "steps": 3, "first_ts": at(0), "last_ts": at(5), "step": 102,
                   "loss": 1.2 - 0.2, "step_ms": 2500.0, "socket_w": 245.0}


def test_memory_is_bounded_to_the_newest_readings():
    p = fold([gpu(s, float(s)) for s in range(10)], TelemetryProjection(keep=3))
    assert [r.watts for r in p.readings("gpu-a")] == [7.0, 8.0, 9.0]
    assert p.snapshot()["nodes"]["gpu-a"]["power"]["amd-smi"]["samples"] == 10  # the counts still cover them all
