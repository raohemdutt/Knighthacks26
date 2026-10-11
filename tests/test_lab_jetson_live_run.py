"""The Jetson power sampler's live run (fixtures/telemetry/jetson_power_live) and the metrics rows recorded from it.

The rows in lab/refine/metrics.jsonl state the mean of the VDD_IN rail while the box sat idle, its mean inside our
caption requests and its highest one-second reading under load. These tests recompute each of them from the raw
one-second samples committed beside them, with the telemetry context's own event model and energy integral, so a row
that drifts from its samples, or samples that stop being a valid sampler event stream, fail here.
"""

import asyncio
import json
import statistics
from datetime import datetime
from pathlib import Path

import pytest

from bus import EventBus
from contexts.telemetry.application.joules import Calibration, JoulesPerAnswerProjection
from contexts.telemetry.domain.energy import integrate
from contracts.telemetry import JetsonPower, RequestWindow
from lab.machine import metrics

RUN = Path(__file__).resolve().parents[1] / "fixtures" / "telemetry" / "jetson_power_live"
NODE = "jetson"
RAILS = {"VDD_IN", "VDD_CPU_GPU_CV", "VDD_SOC"}


def epoch(text: str) -> float:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def watts(sample: JetsonPower) -> float:
    return sample.rails["VDD_IN"] / 1000


def row(metric: str) -> dict:
    matches = metrics.history(metric, node=NODE, conditions={"period_ms": 1000, "rail": "VDD_IN"})
    assert len(matches) == 1, f"{metric}: {len(matches)} rows from the one-second live run"
    return matches[0]


@pytest.fixture(scope="module")
def samples() -> list[JetsonPower]:
    lines = (RUN / "maxn_super_1hz_samples.jsonl").read_text(encoding="utf-8").splitlines()
    return [JetsonPower.model_validate_json(line) for line in lines]


@pytest.fixture(scope="module")
def requests() -> list[dict]:
    lines = (RUN / "caption_requests.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


@pytest.fixture(scope="module")
def answered(requests) -> list[dict]:
    return [r for r in requests if "error" not in r]


def test_the_samples_are_one_second_sampler_events_read_in_the_mode_the_box_was_in(samples):
    assert len(samples) == 600
    assert {s.nvpmodel for s in samples} == {"MAXN_SUPER"}
    assert {s.period_ms for s in samples} == {1000.0}
    assert {s.sensor_avg_samples for s in samples} == {512}
    assert all(set(s.rails) == RAILS for s in samples)
    times = [s.ts.timestamp() for s in samples]
    assert all(b > a for a, b in zip(times, times[1:]))


def test_the_longest_gap_between_samples_is_the_one_the_rows_state(samples):
    times = [s.ts.timestamp() for s in samples]
    longest_ms = round(max(b - a for a, b in zip(times, times[1:])) * 1000, 1)
    for metric in ("jetson.idle_watts", "student.watts_avg", "student.watts_peak"):
        assert row(metric)["conditions"]["max_gap_ms"] == longest_ms, metric


def test_idle_row_is_the_mean_of_the_samples_before_our_first_request(samples, answered):
    first_start = min(epoch(r["start"]) for r in answered)
    idle = [s for s in samples if s.ts.timestamp() < first_start]
    stored = row("jetson.idle_watts")
    assert stored["unit"] == "W"
    assert stored["value"] == round(statistics.fmean(watts(s) for s in idle), 3)
    assert stored["conditions"]["samples"] == len(idle)
    assert stored["conditions"]["watts"]["max"] == round(max(watts(s) for s in idle), 3)
    assert stored["conditions"]["watts"]["min"] == round(min(watts(s) for s in idle), 3)
    trace = [(s.ts.timestamp(), watts(s)) for s in samples]
    assert stored["conditions"]["energy_j"] == round(integrate(trace, idle[0].ts.timestamp(), idle[-1].ts.timestamp()), 3)


def test_loaded_row_is_the_mean_of_the_samples_inside_our_request_windows(samples, answered):
    windows = [(epoch(r["start"]), epoch(r["end"])) for r in answered]
    inside = [s for s in samples if any(a <= s.ts.timestamp() <= b for a, b in windows)]
    stored = row("student.watts_avg")
    assert stored["unit"] == "W"
    assert stored["value"] == round(statistics.fmean(watts(s) for s in inside), 3)
    assert stored["conditions"]["samples"] == len(inside)
    assert stored["conditions"]["requests"] == len(answered)


def test_peak_row_is_the_highest_sample_of_the_load_stretch(samples, answered):
    start, end = min(epoch(r["start"]) for r in answered), max(epoch(r["end"]) for r in answered)
    load = [s for s in samples if start <= s.ts.timestamp() <= end]
    stored = row("student.watts_peak")
    assert stored["unit"] == "W"
    assert stored["value"] == round(max(watts(s) for s in load), 3)
    assert stored["conditions"]["samples"] == len(load)


def test_a_loaded_box_draws_more_than_an_idle_one(samples, answered):
    assert row("student.watts_avg")["value"] > row("jetson.idle_watts")["value"]
    assert row("student.watts_peak")["value"] >= row("student.watts_avg")["value"]


def test_each_row_states_how_far_the_overcurrent_counters_rose_over_its_own_window(samples, answered):
    first_start = min(epoch(r["start"]) for r in answered)
    last_end = max(epoch(r["end"]) for r in answered)
    windows = {
        "jetson.idle_watts": [s for s in samples if s.ts.timestamp() < first_start],
        "student.watts_avg": [s for s in samples
                              if any(epoch(r["start"]) <= s.ts.timestamp() <= epoch(r["end"]) for r in answered)],
        "student.watts_peak": [s for s in samples if first_start <= s.ts.timestamp() <= last_end],
    }
    for metric, window in windows.items():
        rise = {name: window[-1].oc_events[name] - window[0].oc_events[name] for name in sorted(window[0].oc_events)}
        assert row(metric)["conditions"]["oc_events_rise"] == rise, metric


def test_the_live_samples_run_through_the_joules_projection_one_value_per_caption(samples, answered):
    windows = [RequestWindow(node="jetson", ts=datetime.fromisoformat(r["end"].replace("Z", "+00:00")),
                             request_id=f"live-{i:02d}", started=datetime.fromisoformat(r["start"].replace("Z", "+00:00")),
                             ended=datetime.fromisoformat(r["end"].replace("Z", "+00:00")), tokens_in=r["prompt_tokens"],
                             tokens_out=r["completion_tokens"], target="student", kind="caption",
                             model="Qwen3.5-4B-Q4_K_M", engine="llama.cpp")
               for i, r in enumerate(answered)]
    projection = JoulesPerAnswerProjection(Calibration(settle_s=0.0))  # the run ends 38 s after its last request

    async def replay() -> list:
        bus = EventBus(persist=False)
        projection.subscribe(bus)
        await bus.publish_all(sorted([*samples, *windows], key=lambda e: e.ts))
        return projection.ready()

    values = sorted(asyncio.run(replay()), key=lambda v: v.started)
    assert projection.waiting == {}
    assert [v.request_id for v in values] == [w.request_id for w in windows]
    trace = [(s.ts.timestamp(), watts(s)) for s in samples]
    for value, request in zip(values, answered):  # the requests run one after another, so nothing is shared
        assert value.node == "jetson"
        assert value.power_field == "VDD_IN"
        assert value.conditions.power_modes == ["MAXN_SUPER"]
        assert value.joules == pytest.approx(integrate(trace, epoch(request["start"]), epoch(request["end"])), rel=1e-8)
        assert "error not measured on this node" in value.reasons


def test_every_live_row_is_low_confidence_and_says_why():
    for metric in ("jetson.idle_watts", "student.watts_avg", "student.watts_peak"):
        stored = row(metric)
        assert stored["low_confidence"] is True
        assert stored["conditions"]["error"].startswith("unmeasured")
        assert stored["conditions"]["low_confidence_reason"].strip() != ""
        assert stored["conditions"]["power_mode"] == {
            "nvpmodel": "MAXN_SUPER", "nvpmodel_id": 2, "read_with": "nvpmodel -q", "changed": False}
