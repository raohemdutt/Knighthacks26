"""Each source's sample in watts, converted in one place, and the node names a sampler may use."""

from datetime import UTC, datetime, timedelta

import pytest

from contexts.telemetry.domain.nodes import KNOWN_NODES, check_node
from contexts.telemetry.domain.power import PowerReading, reading_of
from contracts.telemetry import GpuPower, JetsonPower, RequestWindow, TrainStep

T0 = datetime(2026, 10, 10, 14, 0, tzinfo=UTC)


def jetson(rails: dict) -> JetsonPower:
    return JetsonPower(ts=T0, node="jetson", source="tegrastats", period_ms=1000, rails=rails, nvpmodel="MAXN_SUPER")


def gpu(socket_w: float | None) -> GpuPower:
    return GpuPower(ts=T0, node="gpu-a", source="amd-smi", period_ms=1000, socket_w=socket_w, gfx_mhz=500,
                    mem_mhz=96, throttled=False)


def test_a_jetson_sample_is_module_input_power_in_watts():
    reading = reading_of(jetson({"VDD_IN": 12800, "VDD_CPU_GPU_CV": 6900, "VDD_SOC": 2600}))
    assert reading == PowerReading(node="jetson", ts=T0, watts=12.8, source="tegrastats", boundary="module")


def test_a_jetson_sample_without_the_module_rail_has_no_watts_rather_than_a_partial_sum():
    assert reading_of(jetson({"VDD_CPU_GPU_CV": 6900, "VDD_SOC": 2600})).watts is None


def test_a_pod_sample_is_graphics_package_power_and_a_missing_reading_stays_missing():
    assert reading_of(gpu(201.0)) == PowerReading(node="gpu-a", ts=T0, watts=201.0, source="amd-smi",
                                                  boundary="graphics package")
    assert reading_of(gpu(None)).watts is None


def test_requests_and_training_steps_are_not_power_samples():
    window = RequestWindow(ts=T0, node="jetson", request_id="r", started=T0 - timedelta(seconds=2), ended=T0,
                           tokens_in=10, tokens_out=5, target="student")
    step = TrainStep(ts=T0, node="gpu-b", run_id="r", step=1, loss=1.0, step_ms=2500, socket_w=245.0)
    assert reading_of(window) is None and reading_of(step) is None


def test_only_the_three_known_nodes_pass_the_check():
    assert [check_node(n) for n in KNOWN_NODES] == ["jetson", "gpu-a", "gpu-b"]
    with pytest.raises(ValueError, match="unknown node 'gpu_a'"):
        check_node("gpu_a")
