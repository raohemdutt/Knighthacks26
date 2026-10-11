"""Joules per answer replayed from recorded windows (fixtures/telemetry/joules) through the real bus."""

import asyncio
import importlib.util
import itertools
from datetime import timedelta
from pathlib import Path

import pytest

from bus import EventBus
from contexts.telemetry.application.joules import Calibration, JoulesPerAnswerProjection
from contexts.telemetry.domain.energy import integrate
from contexts.telemetry.infrastructure import recorded
from contracts.events import Event
from contracts.telemetry import ClockOffset, GpuPower, JoulesPerAnswer, RequestWindow

# Times are seconds since 1970 in the projection and seconds since the first window here: the two frames round
# differently in the last bits (about 1e-7 s on a 1.8e9 s number), so sums are compared to a part in 1e8.
TOL = 1e-8
FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "telemetry" / "joules"


def load(name: str) -> list[Event]:
    return recorded.load(FIXTURES / f"{name}.jsonl")


def replay(events: list[Event], calibration: Calibration | None = None) -> tuple[JoulesPerAnswerProjection, list]:
    joules = JoulesPerAnswerProjection(calibration)

    async def run():
        bus = EventBus(persist=False)
        joules.subscribe(bus)
        await bus.publish_all(events)
        values = joules.ready()
        await bus.publish_all(values)  # as the service does: the values go into the log and come back folded
        return values

    return joules, asyncio.run(run())


def by_id(values: list[JoulesPerAnswer]) -> dict[str, JoulesPerAnswer]:
    return {v.request_id: v for v in values}


def test_the_committed_fixtures_are_what_the_generator_writes():
    spec = importlib.util.spec_from_file_location("make_windows", FIXTURES / "make_windows.py")
    maker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(maker)
    import json

    for name, make in (("overlap_pod_a", maker.overlap_pod_a), ("conditions", maker.conditions)):
        lines = (FIXTURES / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
        assert [json.loads(line) for line in lines] == make(), "rerun fixtures/telemetry/joules/make_windows.py"


def pod_a_trace_on_jetson_clock(events: list[Event]) -> list[tuple[float, float]]:
    offset = next(e for e in events if isinstance(e, ClockOffset)).offset_s
    epoch = next(e for e in events if isinstance(e, RequestWindow)).started
    return [((e.ts - epoch).total_seconds() - offset, e.socket_w) for e in events if isinstance(e, GpuPower)]


def test_replay_the_answers_joules_never_sum_past_the_nodes_energy_over_any_interval():
    events = load("overlap_pod_a")
    _, values = replay(events)
    windows = [e for e in events if isinstance(e, RequestWindow)]
    assert sorted(v.request_id for v in values) == sorted(w.request_id for w in windows)
    epoch = windows[0].started
    trace = pod_a_trace_on_jetson_clock(events)
    span = {v.request_id: ((v.started - epoch).total_seconds(), (v.ended - epoch).total_seconds()) for v in values}
    edges = sorted({x for s in span.values() for x in s})
    for lo, hi in itertools.combinations(edges, 2):
        answers = sum(v.joules for v in values if lo <= span[v.request_id][0] and span[v.request_id][1] <= hi)
        assert answers <= integrate(trace, lo, hi) * (1 + TOL), (lo, hi)
    busy = sum(integrate(trace, a, b) for a, b in zip(edges, edges[1:])
               if any(s <= a and e >= b for s, e in span.values()))
    assert sum(v.joules for v in values) == pytest.approx(busy)
    for v in values:  # no answer takes more than the node used over its own window
        assert 0 <= v.joules <= v.node_energy_j + 1e-9


def test_overlapping_answers_store_the_token_proportional_method_and_how_many_they_shared_with():
    values = by_id(replay(load("overlap_pod_a"))[1])
    assert {v.attribution for v in values.values()} == {"token_proportional"}
    assert (values["a5"].overlapping, values["a2"].overlapping) == (0, 3)
    assert values["a6"].joules == 0.0  # no output tokens, entirely inside requests that had some
    assert values["a5"].joules == pytest.approx(values["a5"].node_energy_j)  # alone: its whole window


def test_a_lone_answer_is_gross_its_window_integral_with_the_idle_reading_stored_beside_it():
    events = load("overlap_pod_a")
    a5 = by_id(replay(events)[1])["a5"]
    epoch = next(e for e in events if isinstance(e, RequestWindow)).started
    expected = integrate(pod_a_trace_on_jetson_clock(events), (a5.started - epoch).total_seconds(),
                         (a5.ended - epoch).total_seconds())
    assert a5.joules == pytest.approx(expected)  # nothing subtracted
    assert a5.basis == "gross"
    assert a5.idle_w == 27.0
    assert a5.idle_ts < a5.started and a5.idle_ts.microsecond == 609500  # the pod's 8.6 s sample, on the Jetson clock


def test_pod_samples_are_moved_onto_the_jetson_clock_by_the_measured_offset_stored_with_the_value():
    events = load("overlap_pod_a")
    offset = next(e for e in events if isinstance(e, ClockOffset))
    a1 = by_id(replay(events)[1])["a1"]
    assert (a1.clock, a1.clock_offset_s, a1.clock_error_s) == ("jetson", offset.offset_s, offset.error_s)
    # without the offset the window would land 65 s before the pod's busy stretch and read idle watts
    assert a1.node_energy_j > 100 * (a1.ended - a1.started).total_seconds()


def test_a_window_on_another_clock_waits_for_a_measured_offset():
    events = [e for e in load("overlap_pod_a") if not isinstance(e, ClockOffset)]
    joules, values = replay(events)
    assert values == []
    assert joules.waiting == {"no clock offset for the window's clock yet": 6}


def test_a_window_waits_until_the_samples_run_past_its_end_and_the_settle_time():
    events = load("overlap_pod_a")
    a5 = next(e for e in events if isinstance(e, RequestWindow) and e.request_id == "a5")
    offset = next(e for e in events if isinstance(e, ClockOffset)).offset_s
    cut = a5.ended + timedelta(seconds=offset + 100)  # pod clock: 100 s past a5's end, short of the 150 s settle
    joules, values = replay([e for e in events if not isinstance(e, GpuPower) or e.ts <= cut])
    assert "a5" not in by_id(values)
    assert joules.waiting["samples have not run past the window and its settle time"] >= 1
    assert "a5" in by_id(replay(events)[1])


def test_every_value_names_its_sampler_period_samples_error_boundary_and_accelerator():
    values = by_id(replay(load("conditions"))[1])
    pod, jetson = values["pod-throttled"], values["jetson-clean"]
    assert (pod.power_field, pod.sampler, pod.period_ms, pod.samples, pod.error) == (
        "socket_w", "lab/pod_power.py 1", 1000.0, 14, None)
    assert pod.boundary.startswith("APU socket (CPU cores and graphics together")
    assert pod.accelerator == "Radeon 8060S (gfx1151, Strix Halo)"
    assert (jetson.power_field, jetson.sampler, jetson.boundary.split(" (")[0]) == ("VDD_IN", "tegrastats", "module input")
    assert pod.clock_offset_s is None  # the window was stamped on the pod's own clock: no offset needed


def test_one_window_of_each_kind_sets_low_confidence_for_its_reason():
    values = by_id(replay(load("conditions"))[1])
    unmeasured = "error not measured on this node"
    assert values["jetson-clean"].reasons == [unmeasured]
    assert values["jetson-short"].reasons == ["fewer than 10 samples in the window", unmeasured]
    assert values["jetson-modes"].reasons == [unmeasured, "two power modes in the window"]
    assert values["jetson-modes"].conditions.power_modes == ["MAXN_SUPER", "15W"]
    assert values["jetson-oc"].reasons == [unmeasured, "overcurrent event"]
    assert values["jetson-oc"].conditions.overcurrent is True
    assert values["pod-throttled"].reasons == [unmeasured, "throttled"]
    assert values["pod-train"].reasons == [unmeasured, "overlapped a training step"]
    assert all(v.low_confidence for v in values.values())
    assert values["jetson-clean"].conditions.temps_c == {"cpu": 48.0, "gpu": 47.5}  # the sample at the window start
    assert values["jetson-clean"].conditions.overcurrent is False


def test_a_measured_error_lifts_the_flag_only_where_nothing_else_sets_it():
    values = by_id(replay(load("conditions"), Calibration(error={"jetson": 0.05, "pod-b": 0.05}))[1])
    assert values["jetson-clean"].low_confidence is False and values["jetson-clean"].error == 0.05
    assert values["jetson-oc"].low_confidence is True


def test_a_pod_window_shorter_than_the_sensors_measured_averaging_lag_is_low_confidence():
    values = by_id(replay(load("conditions"), Calibration(averaging_lag_s={"pod-b": 20.0}))[1])
    assert "window shorter than the sensor's averaging lag" in values["pod-throttled"].reasons
    assert "window shorter than the sensor's averaging lag" not in values["jetson-clean"].reasons


def test_values_already_in_the_log_are_not_computed_again_and_a_late_overlap_takes_only_what_is_left():
    events = load("overlap_pod_a")
    joules, values = replay(events)
    assert joules.ready() == []
    late = RequestWindow(node="pod-a", request_id="late", started=values[0].started + timedelta(seconds=1),
                         ended=values[-1].ended + timedelta(seconds=2), tokens_in=1, tokens_out=500, target="teacher",
                         clock="jetson", ts=values[-1].ended)
    joules.on_event(late)
    (value,) = joules.ready()
    assert "overlaps an answer reported before this request was known" in value.reasons
    total = sum(v.joules for v in values) + value.joules
    events_on_jetson = pod_a_trace_on_jetson_clock(events)
    epoch = next(e for e in events if isinstance(e, RequestWindow)).started
    first = min((v.started - epoch).total_seconds() for v in values)
    last = (late.ended - epoch).total_seconds()
    assert total <= integrate(events_on_jetson, first, last) * (1 + TOL)


def test_the_pods_old_names_are_folded_under_pod_a_and_pod_b():
    events = [e.model_copy(update={"node": "gpu-a"}) if getattr(e, "node", None) == "pod-a" else e
              for e in load("overlap_pod_a")]
    values = replay(events)[1]
    assert {v.node for v in values} == {"pod-a"}


def test_the_stats_page_gets_watts_now_and_joules_per_answer_and_per_day_per_node():
    joules, values = replay(load("conditions"))
    stats = joules.stats()["nodes"]
    assert set(stats) == {"jetson", "pod-b"}
    jetson = stats["jetson"]
    assert jetson["watts_now"] == 6.4 and jetson["power_field"] == "VDD_IN"
    assert jetson["answers"] == 4
    assert jetson["joules_per_answer_mean"] == pytest.approx(
        sum(v.joules for v in values if v.node == "jetson") / 4)
    (day,) = jetson["per_day"].values()
    assert day["answers"] == 4 and day["answer_j"] <= day["node_j"]
    assert jetson["low_confidence"] is True
