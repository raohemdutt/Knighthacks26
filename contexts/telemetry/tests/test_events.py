"""The four per-source telemetry events: what they share, and how they survive the event log."""

import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from bus import EventBus
from contracts import REGISTRY
from contracts.telemetry import GpuPower, JetsonPower, RequestWindow, TrainStep

T0 = datetime(2026, 10, 10, 14, 0, tzinfo=UTC)
SOURCES = (JetsonPower, GpuPower, RequestWindow, TrainStep)


def one_of_each() -> list:
    return [
        JetsonPower(ts=T0, node="jetson", source="tegrastats", period_ms=100, nvpmodel="MAXN_SUPER",
                    rails={"VDD_IN": 12800, "VDD_CPU_GPU_CV": 6900, "VDD_SOC": 2600, "VDD_SOMETHING_NEW": 12},
                    temps_c={"gpu": 52.0}, oc_events={"oc1": 0, "oc2": 1}, sensor_avg_samples=4, sensor_update_ms=8.4),
        GpuPower(ts=T0, node="pod-a", source="amdsmi", period_ms=50, socket_w=201.0, gfx_mhz=2350, mem_mhz=1258,
                 throttled=True, power_field="average_socket_power", temps_c={"edge": 55.0, "hotspot": 68.0},
                 energy_counter_j=123456.5, energy_counter_ns=987654321, received_ts=T0 + timedelta(milliseconds=180)),
        RequestWindow(ts=T0 + timedelta(seconds=5), node="pod-a", request_id="r1", started=T0,
                      first_token=T0 + timedelta(seconds=1), ended=T0 + timedelta(seconds=5), tokens_in=1180,
                      tokens_out=28, target="teacher", model="teacher", engine="vllm", since_load=3, clock="jetson"),
        TrainStep(ts=T0, node="pod-b", run_id="run-0003", step=120, loss=1.02, step_ms=2480.0, socket_w=245.0,
                  energy_j=607.6, tokens=4096, received_ts=T0 + timedelta(milliseconds=210)),
    ]


def test_the_four_sources_are_registered_under_their_own_types():
    assert {cls.TYPE for cls in SOURCES} == {
        "telemetry.jetson.power", "telemetry.gpu.power", "telemetry.request", "telemetry.train.step"}
    assert all(REGISTRY[cls.TYPE] is cls for cls in SOURCES)


def test_the_four_sources_share_only_node_and_ts_besides_the_envelope():
    shared = set.intersection(*(set(cls.model_fields) for cls in SOURCES))
    assert shared == {"event_id", "type", "version", "ts", "node"}


def test_each_event_comes_back_from_the_log_as_its_own_type_with_every_field(tmp_path):
    events = one_of_each()

    async def run():
        await EventBus(tmp_path).publish_all(events)

    asyncio.run(run())
    back = list(EventBus(tmp_path).read_log())
    assert back == events
    assert [type(e) for e in back] == [JetsonPower, GpuPower, RequestWindow, TrainStep]


def test_a_reading_the_source_could_not_take_is_null_in_the_log_and_replays_as_none(tmp_path):
    na = GpuPower(ts=T0, node="pod-a", source="amd-smi", period_ms=1000, socket_w=None, gfx_mhz=None, mem_mhz=None,
                  throttled=None)
    nan_loss = TrainStep(ts=T0, node="pod-b", run_id="r", step=1, loss=float("nan"), step_ms=2500.0, socket_w=None)

    async def run():
        await EventBus(tmp_path).publish_all([na, nan_loss])

    asyncio.run(run())
    lines = [json.loads(line) for path in tmp_path.glob("*.jsonl") for line in path.read_text().splitlines()]
    assert [(line["socket_w"], line.get("loss", "absent")) for line in lines] == [(None, "absent"), (None, None)]
    back = list(EventBus(tmp_path).read_log())
    assert back[0] == na and back[1].loss is None and back[1].socket_w is None


def test_a_request_window_cannot_close_before_it_opens_or_put_its_first_token_outside():
    kw = dict(ts=T0, node="jetson", request_id="r", tokens_in=1, tokens_out=1, target="student")
    with pytest.raises(ValidationError, match="ended is before started"):
        RequestWindow(started=T0, ended=T0 - timedelta(milliseconds=1), **kw)
    with pytest.raises(ValidationError, match="first_token is outside"):
        RequestWindow(started=T0, ended=T0 + timedelta(seconds=2), first_token=T0 + timedelta(seconds=3), **kw)
    instant = RequestWindow(started=T0, ended=T0, first_token=T0, **kw)
    assert instant.ended - instant.started == timedelta(0)


def test_every_event_needs_a_node_and_takes_no_fields_its_source_does_not_define():
    with pytest.raises(ValidationError):
        JetsonPower(ts=T0, node="", source="tegrastats", period_ms=1000, rails={}, nvpmodel=None)
    with pytest.raises(ValidationError):
        GpuPower(ts=T0, source="amd-smi", period_ms=1000, socket_w=40.0, gfx_mhz=None, mem_mhz=None, throttled=None)
    with pytest.raises(ValidationError):  # a new metric is a new event type, not a field squeezed into another
        GpuPower(ts=T0, node="pod-a", source="amd-smi", period_ms=1000, socket_w=40.0, gfx_mhz=None, mem_mhz=None,
                 throttled=None, fan_rpm=1200)
