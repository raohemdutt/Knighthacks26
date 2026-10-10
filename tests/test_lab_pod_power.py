"""lab/pod_power.py: the pod power sampler, tested offline on tool output (here the R9700 fixtures, the other chip).

The tools (amd-smi, rocm-smi, rocminfo) are the boundary: tests hand the real parsers and sampler the text those
tools print (fixtures/telemetry/pod_power, laid out by the CLIs' own printer code at therock-7.13) through the
`run` and `which` parameters. The HTTP endpoint and the Jetson-side reader are run for real on localhost.
"""

import asyncio
import importlib.util
import json
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from bus import EventBus
from contexts.telemetry.application.projection import TelemetryProjection
from contexts.telemetry.application.sampling import SamplerRunner
from contexts.telemetry.infrastructure.pods import PodEndpoint, PodReader
from contracts import parse_event
from contracts.telemetry import GpuPower
from lab import pod_power
from lab.pod_power import PodSampler, Reading

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "fixtures" / "telemetry" / "pod_power"
IDLE = (FIXTURES / "amdsmi_metric_idle.txt").read_text(encoding="utf-8")
LOAD = (FIXTURES / "amdsmi_metric_load.txt").read_text(encoding="utf-8")
ROCM = (FIXTURES / "rocmsmi_idle.txt").read_text(encoding="utf-8")
T0 = datetime(2026, 10, 10, 18, 0, tzinfo=UTC).timestamp()
AMDSMI = SimpleNamespace(source="amd-smi")


def lines(name: str) -> list[dict]:
    return [json.loads(line) for line in (FIXTURES / name).read_text(encoding="utf-8").splitlines()]


def sampler(**kw) -> PodSampler:
    kw.setdefault("clock", lambda: T0)
    return PodSampler(kw.pop("reader", AMDSMI), kw.pop("node", "gpu-a"), **kw)


def busy(watts: float, energy: float | None, activity: float = 99.0, gfx: float = 2350.0) -> Reading:
    return Reading(socket_w=watts, gfx_mhz=gfx, mem_mhz=1258.0, temps_c={"edge": 60.0}, gfx_activity_pct=activity,
                   energy_j=energy, power_field="socket_power")


# --- parsing the tools' output -------------------------------------------------------------------------------------

def test_amd_smi_metric_text_at_idle_parses_into_power_clocks_temperatures_activity_and_the_energy_counter():
    assert pod_power.parse_amdsmi(IDLE) == {0: Reading(
        socket_w=40.0, gfx_mhz=500.0, mem_mhz=96.0, temps_c={"edge": 34.0, "hotspot": 37.0, "mem": 44.0},
        throttled=None, gfx_activity_pct=0.0, energy_j=1843212.402, power_field="socket_power")}


def test_amd_smi_metric_text_under_load_parses_the_top_clocks_and_the_busy_engine():
    assert pod_power.parse_amdsmi(LOAD) == {0: Reading(
        socket_w=201.0, gfx_mhz=2350.0, mem_mhz=1258.0, temps_c={"edge": 63.0, "hotspot": 77.0, "mem": 72.0},
        throttled=None, gfx_activity_pct=99.0, energy_j=1851934.775, power_field="socket_power")}


def test_amd_smi_throttle_status_and_n_a_fields_become_a_bool_and_none_never_a_number():
    text = (IDLE.replace("THROTTLE_STATUS: N/A", "THROTTLE_STATUS: THROTTLED")
            .replace("SOCKET_POWER: 40 W", "SOCKET_POWER: N/A")
            .replace("TOTAL_ENERGY_CONSUMPTION: 1843212.402 J", "TOTAL_ENERGY_CONSUMPTION: N/A"))
    reading = pod_power.parse_amdsmi(text)[0]
    assert (reading.throttled, reading.socket_w, reading.energy_j) == (True, None, None)
    assert pod_power.parse_amdsmi(IDLE.replace("THROTTLE_STATUS: N/A", "THROTTLE_STATUS: UNTHROTTLED"))[0].throttled is False


def test_amd_smi_output_with_a_warning_first_two_gpus_and_an_ascii_degree_sign_keeps_each_gpu_apart():
    second = LOAD.replace("GPU: 0", "GPU: 1")
    text = ("WARNING: User is missing the following required groups: render, video\n"
            + IDLE.replace("\N{DEGREE SIGN}C", "C") + "\n" + second)
    parsed = pod_power.parse_amdsmi(text)
    assert sorted(parsed) == [0, 1]
    assert (parsed[0].socket_w, parsed[0].temps_c, parsed[1].socket_w) == (
        40.0, {"edge": 34.0, "hotspot": 37.0, "mem": 44.0}, 201.0)


def test_amd_smi_without_energy_or_usage_sections_reads_what_is_there():
    power_only = "GPU: 0\n    POWER:\n        SOCKET_POWER: 39 W\n        THROTTLE_STATUS: N/A\n"
    assert pod_power.parse_amdsmi(power_only) == {0: Reading(socket_w=39.0, power_field="socket_power")}
    assert pod_power.parse_amdsmi(IDLE.replace("    ENERGY:\n        TOTAL_ENERGY_CONSUMPTION: 1843212.402 J\n",
                                               "    ENERGY: N/A\n"))[0].energy_j is None


def test_rocm_smi_text_parses_the_average_package_power_and_maps_its_sensor_names_to_amd_smi_ones():
    assert pod_power.parse_rocmsmi(ROCM) == {0: Reading(
        socket_w=40.0, gfx_mhz=500.0, mem_mhz=96.0, temps_c={"edge": 34.0, "hotspot": 37.0, "mem": 44.0},
        throttled=None, gfx_activity_pct=0.0, energy_j=None, power_field="average_graphics_package_power")}


def test_rocm_smi_current_socket_power_names_its_own_field():
    text = "GPU[0]\t\t: Current Socket Graphics Package Power (W): 212.5\n"
    assert pod_power.parse_rocmsmi(text) == {0: Reading(socket_w=212.5,
                                                         power_field="current_socket_graphics_package_power")}


def test_numbers_come_from_the_leading_value_and_sentinels_and_text_are_none():
    assert [pod_power.number(v) for v in ("40 W", "35 \N{DEGREE SIGN}C", "2350", "-1.5", 7)] == [40.0, 35.0, 2350.0, -1.5, 7.0]
    assert [pod_power.number(v) for v in ("N/A", "", None, "[0 %, N/A]", True, 65535, "4294967295 W",
                                          float("nan"))] == [None] * 8


# --- choosing the tool -----------------------------------------------------------------------------------------------

class Tools:
    """The boundary: what each command line prints, or the error it raises, recorded in the order asked."""

    def __init__(self, answers: dict[str, str | Exception], installed=("amd-smi", "rocm-smi")) -> None:
        self.answers, self.installed, self.asked = answers, installed, []

    def run(self, argv):
        command = " ".join([Path(argv[0]).name, *argv[1:]])
        self.asked.append(command)
        answer = self.answers.get(command, pod_power.ReadError(f"{argv[0]} exited 2: unrecognized arguments"))
        if isinstance(answer, Exception):
            raise answer
        return answer

    def which(self, name):
        return f"/opt/rocm/bin/{name}" if name in self.installed else None


def test_the_full_amd_smi_read_is_chosen_when_it_reads_power():
    tools = Tools({"amd-smi metric --power --clock --temperature --usage --energy": IDLE})
    reader, attempts = pod_power.choose_reader(run=tools.run, which=tools.which)
    assert (reader.source, reader.options) == ("amd-smi", pod_power.AMDSMI_OPTIONS[0])
    assert attempts == [{"tool": "amd-smi", "options": list(pod_power.AMDSMI_OPTIONS[0]), "gpu": 0, "socket_w": 40.0}]


def test_a_cli_without_energy_falls_back_to_the_read_without_it_and_then_to_rocm_smi():
    no_energy = IDLE.split("    ENERGY:")[0]
    tools = Tools({"amd-smi metric --power --clock --temperature --usage": no_energy})
    reader, attempts = pod_power.choose_reader(run=tools.run, which=tools.which)
    assert reader.options == pod_power.AMDSMI_OPTIONS[1]
    assert [a.get("error", "ok")[:24] for a in attempts] == ["/opt/rocm/bin/amd-smi ex", "ok"]

    tools = Tools({"rocm-smi --showpower --showtemp --showclocks --showuse": ROCM})
    reader, attempts = pod_power.choose_reader(run=tools.run, which=tools.which)
    assert (reader.source, reader.options) == ("rocm-smi", pod_power.ROCMSMI_OPTIONS[0])
    assert tools.asked == ["amd-smi metric --power --clock --temperature --usage --energy",
                           "amd-smi metric --power --clock --temperature --usage",
                           "amd-smi metric --power --clock --temperature", "amd-smi metric --power",
                           "rocm-smi --showpower --showtemp --showclocks --showuse"]


def test_amd_smi_that_reads_no_power_is_passed_over_and_no_tool_at_all_chooses_nothing():
    na = IDLE.replace("SOCKET_POWER: 40 W", "SOCKET_POWER: N/A")
    tools = Tools({"amd-smi metric --power --clock --temperature --usage --energy": na,
                   "rocm-smi --showpower --showtemp --showclocks --showuse": ROCM})
    reader, attempts = pod_power.choose_reader(run=tools.run, which=tools.which)
    assert reader.source == "rocm-smi"
    assert attempts[0]["socket_w"] is None

    tools = Tools({}, installed=())
    reader, attempts = pod_power.choose_reader(run=tools.run, which=tools.which)
    assert reader is None
    assert [a["tool"] for a in attempts] == ["amd-smi", "rocm-smi"]
    assert all("not installed" in a["error"] for a in attempts)


def test_a_reader_asked_for_a_gpu_the_tool_did_not_print_raises():
    reader = pod_power.CommandReader("amd-smi", "amd-smi", ("metric", "--power"), pod_power.parse_amdsmi, gpu=1,
                                     run=lambda argv: IDLE)
    with pytest.raises(pod_power.ReadError, match="printed no GPU 1"):
        reader.read()


def test_run_command_returns_stdout_and_turns_a_failing_or_hanging_program_into_a_read_error():
    assert pod_power.run_command([sys.executable, "-c", "print('GPU: 0')"]).strip() == "GPU: 0"
    with pytest.raises(pod_power.ReadError, match="exited 3: no such option"):
        pod_power.run_command([sys.executable, "-c", "import sys; sys.stderr.write('no such option'); sys.exit(3)"])
    with pytest.raises(pod_power.ReadError, match="no answer within 0.5 s"):
        pod_power.run_command([sys.executable, "-c", "import time; time.sleep(5)"], timeout_s=0.5)


# --- the sampler -----------------------------------------------------------------------------------------------------

def test_every_sample_is_a_contract_valid_gpu_power_event_naming_its_tool_period_field_and_temperatures():
    s = sampler(period_ms=1000, new_id=lambda: "e1")
    event = s.record(pod_power.parse_amdsmi(IDLE)[0], T0 + 0.25)
    parsed = parse_event(event)
    assert isinstance(parsed, GpuPower)
    assert parsed == GpuPower(
        event_id="e1", ts=datetime(2026, 10, 10, 18, 0, 0, 250000, tzinfo=UTC), node="gpu-a", source="amd-smi",
        period_ms=1000.0, socket_w=40.0, gfx_mhz=500.0, mem_mhz=96.0, throttled=None, power_field="socket_power",
        temps_c={"edge": 34.0, "hotspot": 37.0, "mem": 44.0}, energy_counter_j=1843212.402, energy_counter_ns=None,
        received_ts=None, sampler="lab/pod_power.py 1", sampler_started=datetime(2026, 10, 10, 18, tzinfo=UTC),
        polls=1, updates=1, gfx_activity_pct=0.0, under_load=False, stale=False, counter_ok=None, energy_check=None)


def test_polls_and_distinct_updates_are_counted_apart_and_a_failed_read_is_a_null_sample():
    s = sampler()
    idle = pod_power.parse_amdsmi(IDLE)[0]
    events = [s.record(idle, T0), s.record(idle, T0 + 1), s.record(Reading(), T0 + 2, error="ReadError: boom"),
              s.record(idle, T0 + 3), s.record(pod_power.parse_amdsmi(LOAD)[0], T0 + 4)]
    assert [(e["polls"], e["updates"]) for e in events] == [(1, 1), (2, 1), (3, 1), (4, 1), (5, 2)]
    failed = events[2]
    assert (failed["socket_w"], failed["temps_c"], failed["under_load"], failed["stale"]) == (None, {}, None, None)
    assert (s.errors, s.last_error) == (1, "ReadError: boom")


def test_a_poll_whose_read_raises_still_yields_a_sample_stamped_at_the_middle_of_the_read():
    class Broken:
        source = "amd-smi"

        def read(self):
            raise pod_power.ReadError("amd-smi gave no answer within 10 s")

    times = iter([T0, T0 + 10.0, T0 + 10.5])
    s = PodSampler(Broken(), "gpu-a", clock=lambda: next(times))
    event = s.poll()
    assert (event["ts"], event["socket_w"], event["source"]) == ("2026-10-10T18:00:10.250Z", None, "amd-smi")
    assert s.status()["last_error"] == "ReadError: amd-smi gave no answer within 10 s"


def test_readings_frozen_under_load_turn_stale_after_the_window_and_the_run_says_it_saw_it():
    s = sampler(stale_after_s=10.0)
    frozen = busy(245.0, 500.0)
    events = [s.record(frozen, T0 + i) for i in range(13)]
    assert [e["stale"] for e in events] == [False] * 10 + [True] * 3
    assert s.status()["stale_seen"] is True

    moving = sampler(stale_after_s=10.0)
    events = [moving.record(busy(200.0 + i % 2, 500.0 + i), T0 + i) for i in range(13)]
    assert {e["stale"] for e in events} == {False}
    assert moving.status()["stale_seen"] is False


def test_unchanged_readings_at_idle_are_not_stale_and_load_falls_back_to_power_when_activity_is_unread():
    s = sampler(stale_after_s=2.0)
    idle = pod_power.parse_amdsmi(IDLE)[0]
    assert {s.record(idle, T0 + i)["stale"] for i in range(5)} == {False}
    no_activity = sampler(load_watts=100.0)
    assert no_activity.record(Reading(socket_w=150.0), T0)["under_load"] is True
    assert no_activity.record(Reading(socket_w=40.0), T0 + 1)["under_load"] is False


def test_the_start_up_check_trusts_a_counter_that_advances_with_integrated_power():
    s = sampler(check_window_s=10.0)
    events = [s.record(busy(40.0, 1000.0 + 41.0 * i, activity=0.0, gfx=500.0), T0 + i) for i in range(11)]
    assert [e["counter_ok"] for e in events] == [None] * 10 + [True]
    assert events[10]["energy_check"] == {"phase": "start-up", "window_s": 10.0, "samples": 11, "advanced": True,
                                          "counter_delta_j": 410.0, "integrated_j": 400.0, "ratio": 1.025,
                                          "tolerance": 0.1, "agrees": True}
    assert all(e["energy_check"] is None for e in events[:10])


def test_a_counter_off_by_more_than_ten_percent_a_stuck_one_and_an_unreadable_one_are_never_trusted():
    off = sampler()
    last = [off.record(busy(40.0, 1000.0 + 45.0 * i, activity=0.0), T0 + i) for i in range(11)][-1]
    assert (last["energy_check"]["ratio"], last["energy_check"]["agrees"], last["counter_ok"]) == (1.125, False, False)

    stuck = sampler()
    last = [stuck.record(busy(40.0, 1000.0, activity=0.0), T0 + i) for i in range(11)][-1]
    assert (last["energy_check"]["advanced"], last["energy_check"]["agrees"], last["counter_ok"]) == (False, False, False)

    unreadable = sampler()
    last = [unreadable.record(busy(40.0, None, activity=0.0), T0 + i) for i in range(11)][-1]
    check = last["energy_check"]
    assert (check["advanced"], check["counter_delta_j"], check["integrated_j"], check["agrees"]) == (None, None, 400.0, False)


def test_a_counter_that_goes_backwards_inside_the_window_did_not_advance():
    window = [(T0, 40.0, 100.0), (T0 + 1, 40.0, 150.0), (T0 + 2, 40.0, 120.0), (T0 + 3, 40.0, 220.0)]
    check = pod_power.energy_check(window, "start-up")
    assert (check["advanced"], check["counter_delta_j"], check["agrees"]) == (False, 120.0, False)


def test_the_load_check_waits_for_settled_load_runs_once_and_keeps_the_counter_only_if_both_agree():
    s = sampler(check_window_s=10.0, check_settle_s=5.0)
    t = T0
    energy = 1000.0
    for _ in range(11):  # start-up at idle, agreeing
        s.record(busy(40.0, energy, activity=0.0), t)
        t, energy = t + 1, energy + 40.0
    assert s.counter_ok is True
    for _ in range(3):  # a short burst, then idle: the run resets before it settles
        s.record(busy(200.0, energy), t)
        t, energy = t + 1, energy + 200.0
    s.record(busy(40.0, energy, activity=0.0), t)
    t, energy = t + 1, energy + 40.0
    closing = []
    for i in range(30):  # sustained load; the counter now reads 30 percent high
        event = s.record(busy(200.0, energy), t)
        if event["energy_check"]:
            closing.append((i, event["energy_check"]["phase"], event["energy_check"]["agrees"], event["counter_ok"]))
        t, energy = t + 1, energy + 260.0
    assert closing == [(15, "load", False, False)]  # settle 5 s, then a 10 s window
    assert [c["phase"] for c in s.status()["energy_checks"]] == ["start-up", "load"]


def test_samples_after_pages_through_the_buffer_and_reports_what_it_has_dropped():
    s = sampler(keep=5)
    for i in range(8):
        s.record(busy(200.0, float(i)), T0 + i)
    page = s.samples_after(0, limit=2)
    assert (page["first_seq"], page["last_seq"], [x["seq"] for x in page["samples"]]) == (4, 8, [4, 5])
    assert [x["seq"] for x in s.samples_after(5, limit=10)["samples"]] == [6, 7, 8]
    assert s.samples_after(8)["samples"] == []


def test_the_sampler_refuses_an_unknown_node_and_a_period_of_zero():
    with pytest.raises(ValueError, match="unknown node 'pod-a'"):
        sampler(node="pod-a")
    with pytest.raises(ValueError, match="period_ms must be more than 0"):
        sampler(period_ms=0)


def test_run_loop_skips_the_slots_a_slow_read_overran_instead_of_bursting():
    now = [0.0]
    stop = threading.Event()
    waits = []

    class Slow:
        source = "amd-smi"

        def read(self):
            now[0] += 2.5 if len(waits) == 1 else 0.1  # the second read takes two and a half periods
            return Reading(socket_w=40.0)

    s = PodSampler(Slow(), "gpu-a", period_ms=1000, clock=lambda: now[0])
    original_wait = stop.wait

    def wait(seconds):
        waits.append(round(seconds, 3))
        now[0] += max(seconds, 0.0)
        return original_wait(0)

    stop.wait = wait
    pod_power.run_loop(s, stop, monotonic=lambda: now[0], max_polls=4)
    assert waits == [0.9, 0.5, 0.9]  # after the slow read the next poll lands on the next whole slot
    assert s.polls == 4


def test_samples_go_to_one_file_per_utc_day(tmp_path):
    sink = pod_power.DailyJsonl(tmp_path / "power")
    s = sampler(sink=sink)
    s.record(busy(40.0, 1.0), datetime(2026, 10, 10, 23, 59, 59, tzinfo=UTC).timestamp())
    s.record(busy(41.0, 2.0), datetime(2026, 10, 11, 0, 0, 1, tzinfo=UTC).timestamp())
    files = sorted(p.name for p in (tmp_path / "power").iterdir())
    assert files == ["samples-2026-10-10.jsonl", "samples-2026-10-11.jsonl"]
    assert [json.loads(line)["socket_w"] for f in files for line in (tmp_path / "power" / f).read_text().splitlines()] == [40.0, 41.0]


# --- the window summary ----------------------------------------------------------------------------------------------

def test_the_summary_of_the_recorded_pod_a_run_shows_the_restart_the_gap_the_failed_read_and_the_energy_source():
    summary = pod_power.summarize(lines("samples_gpu-a.jsonl"))
    assert summary == {
        "node": "gpu-a", "start": "2026-10-10T18:00:00.200Z", "end": "2026-10-10T18:01:04.640Z", "duration_s": 64.44,
        "samples": 60, "readings": 59, "missing": 1, "distinct_updates": 58, "mean_w": 104.22, "min_w": 39.0,
        "max_w": 203.0, "sampler": "lab/pod_power.py 1", "source": "amd-smi", "power_field": "socket_power",
        "period_ms": 1000.0, "measured_period_ms": 1092.203, "max_gap_ms": 6400.0, "restarts": 1,
        "max_receive_gap_ms": None, "max_receive_delay_s": None,
        "start_temps_c": {"edge": 34.0, "hotspot": 37.0, "mem": 44.0},
        "end_temps_c": {"edge": 46.0, "hotspot": 49.0, "mem": 56.0}, "under_load_samples": 24, "stale_samples": 0,
        "counter_ok": True, "energy_j": 6525.605, "energy_source": "avg_power_integral", "error": "unknown"}


def test_a_window_where_every_sample_trusts_the_counter_takes_energy_from_the_counter():
    events = lines("samples_gpu-a.jsonl")
    start = datetime(2026, 10, 10, 18, 0, 11, tzinfo=UTC)
    end = datetime(2026, 10, 10, 18, 0, 30, tzinfo=UTC)
    summary = pod_power.summarize(events, start=start, end=end)
    inside = [e for e in events if start <= pod_power.parse_ts(e["ts"]) <= end]
    assert summary["energy_source"] == "energy_counter"
    assert summary["energy_j"] == round(inside[-1]["energy_counter_j"] - inside[0]["energy_counter_j"], 3)
    assert (summary["samples"], summary["restarts"], summary["max_gap_ms"]) == (19, 0, 1020.0)


def test_a_window_that_starts_before_the_counter_was_checked_integrates_power_instead():
    events = lines("samples_gpu-a.jsonl")[:15]  # counter_ok is null until the start-up check closes on the 11th
    assert [e["counter_ok"] for e in events] == [None] * 10 + [True] * 5
    summary = pod_power.summarize(events)
    assert (summary["energy_source"], summary["energy_j"]) == ("avg_power_integral", 557.52)


def test_the_stale_pod_b_recording_counts_its_few_updates_and_stale_samples():
    summary = pod_power.summarize(lines("samples_gpu-b_stale.jsonl"))
    assert (summary["samples"], summary["distinct_updates"], summary["stale_samples"], summary["counter_ok"],
            summary["energy_source"]) == (25, 4, 12, False, "avg_power_integral")


def test_the_summary_refuses_an_empty_window_and_samples_from_two_pods_mixed():
    with pytest.raises(ValueError, match="no telemetry.gpu.power samples for gpu-b in the window"):
        pod_power.summarize(lines("samples_gpu-a.jsonl"), node="gpu-b")
    with pytest.raises(ValueError, match="samples from 2 nodes"):
        pod_power.summarize(lines("samples_gpu-a.jsonl") + lines("samples_gpu-b_stale.jsonl"))


# --- the recorded fixtures -------------------------------------------------------------------------------------------

def test_the_recorded_samples_are_exactly_what_the_sampler_writes_for_their_scenario():
    spec = importlib.util.spec_from_file_location("make_samples", FIXTURES / "make_samples.py")
    maker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(maker)
    assert maker.pod_a() == lines("samples_gpu-a.jsonl"), "rerun fixtures/telemetry/pod_power/make_samples.py"
    assert maker.pod_b() == lines("samples_gpu-b_stale.jsonl"), "rerun fixtures/telemetry/pod_power/make_samples.py"


def test_every_recorded_sample_is_a_registered_gpu_power_event_and_the_checks_are_where_the_scenario_puts_them():
    events = [parse_event(e) for e in lines("samples_gpu-a.jsonl")]
    assert {type(e) for e in events} == {GpuPower}
    checks = [(e.ts.isoformat(), e.energy_check.phase, e.energy_check.agrees) for e in events if e.energy_check]
    assert checks == [("2026-10-10T18:00:10.260000+00:00", "start-up", True),
                      ("2026-10-10T18:00:33.230000+00:00", "load", True),
                      ("2026-10-10T18:01:03.690000+00:00", "start-up", True)]


# --- the node gate and the command line ------------------------------------------------------------------------------

def test_the_gate_lets_only_a_known_node_on_a_strix_halo_sample():
    strix_halo = lambda: "  Name:                    gfx1151\n"  # noqa: E731
    assert pod_power.gate("gpu-a", strix_halo) is None
    assert pod_power.gate("pod-a", strix_halo) == "--node must be one of gpu-a, gpu-b, got 'pod-a'"
    assert pod_power.gate("gpu-b", lambda: "  Name:   gfx1201\n") == ("rocminfo names no gfx1151: this is not a Strix "
                                                                      "Halo pod")

    def missing():
        raise pod_power.ReadError("rocminfo is not installed")

    assert pod_power.gate("gpu-b", missing) == "cannot tell what GPU this is: rocminfo failed (rocminfo is not installed)"


def test_serve_refuses_to_start_off_a_strix_halo_and_says_why(capsys):
    args = pod_power.build_parser().parse_args(["serve", "--node", "gpu-a"])
    assert pod_power.serve(args, rocminfo=lambda: "gfx90a") == 2
    assert json.loads(capsys.readouterr().out) == {"ok": False, "outcome": "refused",
                                                   "error": "rocminfo names no gfx1151: this is not a Strix Halo pod"}


def test_check_reads_once_with_every_option_set_and_saves_each_raw_output_for_new_fixtures(tmp_path):
    tools = Tools({"amd-smi metric --power --clock --temperature --usage --energy": IDLE,
                   "amd-smi metric --power": "GPU: 0\n    POWER:\n        SOCKET_POWER: 40 W\n",
                   "rocm-smi --showpower --showtemp --showclocks --showuse": ROCM,
                   "rocm-smi --showpower": "GPU[0]\t\t: Average Graphics Package Power (W): 40.0\n"})
    result = pod_power.check(save=tmp_path / "raw", run=tools.run, which=tools.which)
    assert result["ok"] is True
    assert result["chosen"] == {"tool": "amd-smi", "options": list(pod_power.AMDSMI_OPTIONS[0])}
    assert len(result["attempts"]) == 6
    assert sorted(result["saved"]) == sorted(p.name for p in (tmp_path / "raw").iterdir())
    saved = (tmp_path / "raw" / "amd_smi_metric_power_clock_temperature_usage_energy.txt").read_text(encoding="utf-8")
    assert saved == IDLE


# --- the endpoint, read by the Jetson's reader -----------------------------------------------------------------------

def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as reply:
        return json.loads(reply.read().decode("utf-8"))


@pytest.fixture
def pod_server():
    """A real sampler endpoint on a free localhost port, holding the recorded pod A samples."""
    s = sampler(new_id=iter(e["event_id"] for e in lines("samples_gpu-a.jsonl")).__next__)
    server = pod_power.ThreadingHTTPServer(("127.0.0.1", 0), pod_power.make_handler(s, clock=lambda: T0 + 99))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield s, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def feed(s: PodSampler, events: list[dict]) -> None:
    """Load recorded samples into a sampler's buffer through its own record path."""
    for e in events:
        reading = Reading(socket_w=e["socket_w"], gfx_mhz=e["gfx_mhz"], mem_mhz=e["mem_mhz"], temps_c=e["temps_c"],
                          throttled=e["throttled"], gfx_activity_pct=e["gfx_activity_pct"],
                          energy_j=e["energy_counter_j"], power_field=e["power_field"])
        s.record(reading, pod_power.parse_ts(e["ts"]).timestamp())


def test_the_endpoint_serves_samples_after_a_seq_with_the_pods_clock_and_its_status(pod_server):
    s, base = pod_server
    feed(s, lines("samples_gpu-a.jsonl")[:5])
    page = get_json(f"{base}/samples?after=2&limit=2")
    assert [item["seq"] for item in page["samples"]] == [3, 4]
    assert (page["first_seq"], page["last_seq"], page["node"]) == (1, 5, "gpu-a")
    assert (page["pod_received"], page["pod_sent"]) == ("2026-10-10T18:01:39.000Z", "2026-10-10T18:01:39.000Z")
    assert parse_event(page["samples"][0]["event"]).socket_w == 39.0
    status = get_json(f"{base}/status")
    assert (status["polls"], status["updates"], status["period_ms"], status["counter_ok"]) == (5, 5, 1000.0, None)


def test_the_endpoint_answers_bad_numbers_with_400_and_other_paths_with_404(pod_server):
    _, base = pod_server
    for path, code in (("/samples?after=x", 400), ("/nope", 404)):
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(f"{base}{path}", timeout=5)
        assert caught.value.code == code


def test_the_jetson_reader_fetches_the_pods_samples_over_http_into_the_log_and_the_projection(pod_server, tmp_path):
    s, base = pod_server
    recorded = lines("samples_gpu-a.jsonl")[:40]
    feed(s, recorded)
    received = datetime(2026, 10, 10, 18, 1, 40, tzinfo=UTC)

    async def run():
        bus = EventBus(tmp_path)
        projection = TelemetryProjection()
        projection.subscribe(bus)
        reader = PodReader([PodEndpoint("gpu-a", base, "not-a-real-token")], limit=16, rounds=3,
                           clock=lambda: received, sleep=lambda _: asyncio.sleep(0))
        runner = SamplerRunner(reader, bus.publish, nodes=["gpu-a"])
        await runner.run()
        return projection, runner, reader

    projection, runner, reader = asyncio.run(run())
    logged = [json.loads(line) for path in sorted(tmp_path.glob("*.jsonl")) for line in path.read_text().splitlines()]
    assert [e["event_id"] for e in logged] == [e["event_id"] for e in recorded]
    assert {e["received_ts"] for e in logged} == {"2026-10-10T18:01:40Z"}
    assert [(e["socket_w"], e["polls"], e["updates"]) for e in logged] == [
        (e["socket_w"], e["polls"], e["updates"]) for e in recorded]
    assert runner.status()["published"] == 40
    assert reader.status()["gpu-a"]["after"] == 40
    snap = projection.snapshot()["nodes"]["gpu-a"]
    assert snap["power"]["amd-smi"]["samples"] == 40
    assert (snap["sampler"]["counter_ok"], [c["phase"] for c in snap["sampler"]["energy_checks"]]) == (
        True, ["start-up", "load"])


def test_the_launcher_script_has_unix_line_ends_and_parses_as_bash():
    script = (REPO / "lab" / "pod_power.sh").read_text(encoding="utf-8")
    assert chr(13) not in script  # a carriage return breaks every line of a bash script
    git_bash = str(Path("C:/Program Files/Git/bin/bash.exe"))
    bash = next((b for b in ("bash", git_bash) if _runs(b)), None)
    assert bash is not None, "bash is needed to syntax-check lab/pod_power.sh (Git Bash on Windows)"
    done = subprocess.run([bash, "-n"], input=script.encode("utf-8"), capture_output=True)  # bytes: no CRLF on Windows
    assert (done.returncode, done.stderr) == (0, b"")


def _runs(program: str) -> bool:
    try:
        return subprocess.run([program, "-c", "true"], capture_output=True).returncode == 0
    except OSError:
        return False
