"""lab/sampler_overhead.py, T9: the sampler's own cost on pod B, measured with and without it, offline.

The experiment is a script of steps behind injectable clients (the slot queue, the pod's exec, the bench harness, the
model server), so every test here runs it against fakes on a fake clock: nothing launches, sleeps, or touches Notion
or a hub. Three groups:
- the arithmetic (schedule, CPU from /proc snapshots, spread, verdict) on constructed numbers;
- the experiment's order of steps, its failure paths (the slot is released, the sampler and the server are left as
  found) and the slot wait;
- the real pieces under fakes: the slot rules over lab/machine/slots_fake.py, the bench harness over
  lab/teacher_fake.py, the CPU script against a fake /proc, and the verb in the registry."""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from lab import __main__ as lab_cli
from lab import sampler_overhead as so
from lab.machine import ledger, slots, slots_fake
from lab.machine.slots_notion import NotionClient

REPO = Path(__file__).resolve().parents[1]
UTC = dt.timezone.utc
T0 = dt.datetime(2026, 10, 10, 23, 30, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LAB_ENV_FILES", "")
    for name in ("NOTION_TOKEN", "LAB_QUEUE_DATA_SOURCE", "LAB_NOTION_URL", "LAB_ACTOR", "LAB_TIER", "LAB_TASK"):
        monkeypatch.delenv(name, raising=False)


# --- fakes -----------------------------------------------------------------------------------------------------

class Clock:
    """A clock the experiment's sleeps move; time() is seconds, now() an aware datetime."""

    def __init__(self) -> None:
        self.t = 0.0

    def time(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds

    def now(self) -> dt.datetime:
        return T0 + dt.timedelta(seconds=self.t)


class FakeSlots:
    """The four slot verbs the experiment uses, with a scripted start and a log of what was asked."""

    def __init__(self, events: list, *, starts=("started",), extend_ok=True) -> None:
        self.events = events
        self.starts = list(starts)
        self.extend_ok = extend_ok
        self.released: list[dict] = []

    def request(self, *, task, what, minutes):
        self.events.append(("slot", "request", task, minutes))
        return {"ok": True, "outcome": "queued", "row": "r" * 32}

    def start(self, row):
        outcome = self.starts.pop(0) if len(self.starts) > 1 else self.starts[0]
        self.events.append(("slot", "start", outcome))
        return {"ok": outcome == "started", "outcome": outcome, "row": row}

    def extend(self, row, note):
        self.events.append(("slot", "extend"))
        return {"ok": self.extend_ok, "outcome": "extended" if self.extend_ok else "not_running", "row": row}

    def done(self, row, result, cancel=False):
        self.events.append(("slot", "cancel" if cancel else "done"))
        self.released.append({"row": row, "result": result, "cancel": cancel})
        return {"ok": True, "outcome": "cancelled" if cancel else "done", "row": row}


class FakePod:
    """The pod as the controller's run-script shows it: the sampler script, the teacher script and the CPU script."""

    def __init__(self, events: list, *, sampler_running=False, server_up=False, busy_off=40.0, busy_on=40.5,
                 sampler_cpu=0.8, bench=None, wait_for_bench=False) -> None:
        self.events = events
        self.sampler = sampler_running
        self.server = server_up
        self.busy = {False: busy_off, True: busy_on}
        self.sampler_cpu = sampler_cpu
        self.bench = bench
        self.wait_for_bench = wait_for_bench
        self.fail_serve = False
        self.cpu_calls = 0
        self.calls: list[tuple[str, tuple]] = []

    @staticmethod
    def result(code: int, text: str = "") -> dict:
        return {"ok": code == 0, "outcome": "ok" if code == 0 else "fail", "exit_code": code, "output_tail": text}

    def run_script(self, script, *, args=(), files=(), timeout_s=900.0):
        name = Path(script).name
        self.calls.append((name, tuple(args)))
        if name == "pod_power.sh":
            step = args[0]
            self.events.append(("sampler", step))
            if step == "start":
                self.sampler = True
                return self.result(0, '{"started": true}')
            if step == "stop":
                self.sampler = False
                return self.result(0, "stopped")
            return self.result(0 if self.sampler else 1, "status" if self.sampler else "not running")
        if name == "pod_a_teacher.sh":
            step = args[0]
            self.events.append(("server", step))
            if step == "serve":
                if self.fail_serve:
                    return self.result(1, "the server process ended before it answered")
                self.server = True
            elif step == "stop":
                self.server = False
            return self.result(0, step)
        if name == so.CPU_SCRIPT_NAME:
            self.events.append(("cpu", self.sampler))
            if self.bench is not None:
                self.bench.gate.set()
                if self.wait_for_bench:
                    self.cpu_calls += 1
                    deadline = time.time() + 10
                    while self.bench.done < self.cpu_calls:
                        assert time.time() < deadline, "the fake bench never finished"
                        time.sleep(0.005)
            return self.result(0, "noise\n" + snapshot_line(self.busy[self.sampler], sampler_cpu_pct=self.sampler_cpu
                                                           if self.sampler else None) + "\n")
        raise AssertionError(f"unexpected script {name}")


def snapshot_line(busy_pct: float, *, sampler_cpu_pct: float | None = None, window_s: float = 10.0, ncpu: int = 8,
                  tck: int = 100) -> str:
    """The line the CPU script prints, worked back from the percentages a test wants."""
    total = int(ncpu * tck * window_s)
    busy = round(total * busy_pct / 100.0)
    row = {"schema": so.CPU_SCHEMA, "t0": 1000.0, "t1": 1000.0 + window_s, "tck": tck, "ncpu": ncpu,
           "stat0": [10000, 0, 5000, 80000, 100, 0, 0, 0],
           "stat1": [10000 + busy, 0, 5000, 80000 + total - busy, 100, 0, 0, 0],
           "pid": None, "ticks0": None, "ticks1": None}
    if sampler_cpu_pct is not None:
        row.update(pid=4242, ticks0=500, ticks1=500 + round(tck * window_s * sampler_cpu_pct / 100.0))
    return "SAMPLER_OVERHEAD_CPU " + json.dumps(row)


class FakeBench:
    """The bench harness, one repeat per call: a scripted rate per arm, optionally held until the CPU read has begun."""

    def __init__(self, events: list, rates: dict, *, clock: Clock | None = None, repeat_s=40.0, hold_s=0.0,
                 fail_on: int | None = None) -> None:
        self.events = events
        self.rates = {arm: list(v) for arm, v in rates.items()}
        self.clock = clock
        self.repeat_s = repeat_s
        self.hold_s = hold_s
        self.fail_on = fail_on
        self.calls = 0
        self.done = 0
        self.gate = threading.Event()

    def __call__(self, label):
        self.calls += 1
        self.events.append(("bench", label))
        if self.fail_on == self.calls:
            self.done += 1
            raise RuntimeError("the proxy answered 502")
        arm = label.split("-")[0]
        rate = self.rates[arm].pop(0)
        self.gate.wait(2)
        self.gate.clear()
        time.sleep(self.hold_s)
        if self.clock is not None:
            self.clock.sleep(self.repeat_s)
        self.done += 1
        return {"run_id": f"bench-{self.calls}", "outcome": "ok", "decode_tok_s": rate, "decode_tok_s_total": rate * 3.5,
                "ttft_s": 1.0 + rate / 100.0, "tokens_generated": 384, "window_s": 30.0, "in_flight_mean": 3.4,
                "requests": [{}, {}, {}, {}], "reason": None}


class FakeEndpoint:
    def __init__(self, pod: FakePod) -> None:
        self.pod = pod

    def models(self):
        if not self.pod.server:
            raise ConnectionError("the proxy answered 500")
        return [{"id": "teacher"}]


RATES = {"warmup": [9.0], "off": [10.0, 10.2, 9.8], "on": [9.9, 10.1, 9.7]}


def build(tmp_path, *, rates=None, slot_starts=("started",), sampler_running=False, server_up=False, bench_kwargs=None,
          pod_kwargs=None, settings=None, extend_ok=True):
    events: list = []
    clock = Clock()
    fake_slots = FakeSlots(events, starts=slot_starts, extend_ok=extend_ok)
    bench = FakeBench(events, rates or RATES, clock=clock, **(bench_kwargs or {}))
    pod = FakePod(events, sampler_running=sampler_running, server_up=server_up, bench=bench, **(pod_kwargs or {}))
    experiment = so.Experiment(settings or so.Settings(), slots=fake_slots, pod=pod, bench=bench,
                               endpoint=FakeEndpoint(pod), out_dir=tmp_path / "out", clock=clock.time,
                               sleep=clock.sleep)
    return experiment, events, fake_slots, pod, bench, clock


# --- the arithmetic --------------------------------------------------------------------------------------------

def test_the_schedule_interleaves_off_and_on_three_times_each():
    assert so.schedule(3) == ["off", "on", "off", "on", "off", "on"]
    assert so.schedule(1) == ["off", "on"]
    with pytest.raises(ValueError):
        so.schedule(0)


def test_cpu_busy_and_the_samplers_own_share_come_from_two_proc_snapshots():
    line = snapshot_line(40.0, sampler_cpu_pct=1.5, window_s=10.0, ncpu=8)
    reading = so.parse_cpu(line)
    assert reading["busy_pct"] == pytest.approx(40.0, abs=0.01)
    assert reading["cores_busy"] == pytest.approx(3.2, abs=0.01)
    assert reading["sampler_cpu_pct"] == pytest.approx(1.5, abs=0.01)     # percent of one core
    assert reading["window_s"] == pytest.approx(10.0) and reading["ncpu"] == 8


def test_a_reading_without_the_sampler_has_no_share_and_a_broken_one_is_refused():
    assert so.parse_cpu(snapshot_line(10.0))["sampler_cpu_pct"] is None
    with pytest.raises(so.OverheadError):
        so.parse_cpu("no marker here")
    with pytest.raises(so.OverheadError):
        so.parse_cpu("SAMPLER_OVERHEAD_CPU {not json")
    zero = json.loads(snapshot_line(10.0).split(" ", 1)[1])
    zero["stat1"] = zero["stat0"]
    with pytest.raises(so.OverheadError, match="did not move"):
        so.parse_cpu("SAMPLER_OVERHEAD_CPU " + json.dumps(zero))


def rows_for(off, on, *, cpu_off=40.0, cpu_on=40.5):
    rows = []
    for arm, values, cpu in (("off", off, cpu_off), ("on", on, cpu_on)):
        for k, v in enumerate(values, start=1):
            rows.append({"schema": so.ROW_SCHEMA, "arm": arm, "repeat": k, "outcome": "ok", "decode_tok_s": v,
                         "decode_tok_s_total": v * 3.5, "ttft_s": 1.0, "cpu_busy_pct": cpu,
                         "sampler_cpu_pct": 0.8 if arm == "on" else None})
    return rows


def test_a_difference_inside_the_spread_is_no_interference_and_confirms_the_bound():
    result = so.analyze(rows_for([10.0, 10.2, 9.8], [9.9, 10.1, 9.7]))
    tok = result["metrics"]["decode_tok_s"]
    assert tok["off"]["mean"] == pytest.approx(10.0) and tok["off"]["sd"] == pytest.approx(0.2)
    assert tok["on"]["mean"] == pytest.approx(9.9)
    assert tok["diff"] == pytest.approx(-0.1) and tok["pct"] == pytest.approx(-1.0)
    assert tok["spread"] == pytest.approx(0.2) and tok["within_spread"] is True
    assert tok["verdict"] == "within_spread"
    assert result["verdict"] == "no_interference"
    assert result["bound"]["holds"] is True and result["bound"]["action"] == "confirmed"
    assert result["interference_pct"] == pytest.approx(-1.0)
    assert result["n"] == {"off": 3, "on": 3} and result["low_confidence"] is True


def test_a_slowdown_beyond_the_spread_is_interference_and_corrects_the_bound():
    result = so.analyze(rows_for([10.0, 10.2, 9.8], [8.0, 8.1, 7.9]))
    tok = result["metrics"]["decode_tok_s"]
    assert tok["within_spread"] is False and tok["verdict"] == "worse_outside_spread"
    assert result["verdict"] == "interference"
    assert result["bound"]["holds"] is False and result["bound"]["action"] == "corrected"
    assert result["interference_pct"] == pytest.approx(-20.0)
    assert tok["welch"]["label"] == "a_ahead"       # off is ahead of on


def test_a_speedup_beyond_the_spread_is_not_interference_but_is_flagged_as_drift():
    result = so.analyze(rows_for([10.0, 10.2, 9.8], [11.0, 11.2, 10.8]))
    assert result["metrics"]["decode_tok_s"]["verdict"] == "better_outside_spread"
    assert result["verdict"] == "no_interference" and "drift" in result["notes"][0]


def test_failed_or_missing_repeats_make_the_verdict_inconclusive_not_a_guess():
    rows = rows_for([10.0, 10.2, 9.8], [9.9, 10.1, 9.7])
    rows[4]["outcome"] = "fail"
    rows[4]["decode_tok_s"] = None
    result = so.analyze(rows)
    assert result["n"] == {"off": 3, "on": 2}
    result = so.analyze(rows_for([10.0], [9.9]))
    assert result["verdict"] == "inconclusive" and result["bound"]["holds"] is None
    assert result["bound"]["action"] == "undecided"
    assert so.analyze([])["verdict"] == "inconclusive"


def test_the_warmup_row_never_enters_a_number():
    rows = rows_for([10.0, 10.2, 9.8], [9.9, 10.1, 9.7]) + [
        {"schema": so.ROW_SCHEMA, "arm": "warmup", "repeat": 0, "outcome": "ok", "decode_tok_s": 1.0,
         "ttft_s": 9.0, "cpu_busy_pct": 99.0}]
    assert so.analyze(rows)["metrics"]["decode_tok_s"]["off"]["n"] == 3


def test_cpu_is_reported_per_arm_and_the_samplers_own_share_from_the_on_arm():
    result = so.analyze(rows_for([10.0, 10.2, 9.8], [9.9, 10.1, 9.7], cpu_off=40.0, cpu_on=41.0))
    cpu = result["metrics"]["cpu_busy_pct"]
    assert cpu["off"]["mean"] == 40.0 and cpu["on"]["mean"] == 41.0 and cpu["diff"] == pytest.approx(1.0)
    assert result["sampler_cpu_pct"]["mean"] == pytest.approx(0.8)


# --- the experiment ---------------------------------------------------------------------------------------------

def test_a_full_run_takes_a_slot_serves_measures_interleaved_and_releases(tmp_path):
    experiment, events, fake_slots, pod, bench, _ = build(tmp_path)
    result = experiment.run()
    assert result["ok"] is True and result["outcome"] == "ok", result
    kinds = [e for e in events if e[0] in ("slot", "server", "sampler", "bench")]
    assert kinds[:3] == [("slot", "request", "T9", 15), ("slot", "start", "started"), ("sampler", "status")]
    assert ("server", "serve") in kinds and kinds.index(("server", "serve")) < kinds.index(("bench", "warmup"))
    assert [e[1] for e in kinds if e[0] == "bench"] == ["warmup", "off-1", "on-1", "off-2", "on-2", "off-3", "on-3"]
    # the sampler is off for each off repeat and on for each on repeat
    cpu_states = [e[1] for e in events if e[0] == "cpu"]
    assert cpu_states == [False, False, True, False, True, False, True]
    assert kinds[-2:] == [("server", "stop"), ("slot", "done")]
    assert fake_slots.released[0]["cancel"] is False and "decode" in fake_slots.released[0]["result"]
    assert pod.sampler is False and pod.server is False        # left as found


def test_the_analysis_and_the_rows_are_written_and_the_ledger_has_the_number(tmp_path):
    experiment, *_ = build(tmp_path)
    result = experiment.run()
    rows = [json.loads(line) for line in (tmp_path / "out" / "rows.jsonl").read_text().splitlines()]
    assert [(r["arm"], r["repeat"]) for r in rows] == [("warmup", 0), ("off", 1), ("on", 1), ("off", 2), ("on", 2),
                                                       ("off", 3), ("on", 3)]
    first = rows[1]
    assert first["schema"] == so.ROW_SCHEMA and first["decode_tok_s"] == 10.0 and first["cpu_busy_pct"] == 40.0
    assert first["task"] == "T9" and first["node"] == "pod-b" and first["sampler"] == "off"
    assert rows[2]["sampler_cpu_pct"] == pytest.approx(0.8)
    analysis = json.loads((tmp_path / "out" / "analysis.json").read_text())
    assert analysis["verdict"] == result["analysis"]["verdict"] == "no_interference"
    summary = [r for r in ledger.read() if r["verb"] == "sampler-overhead.summary"]
    assert len(summary) == 1 and summary[0]["node"] == "pod-b" and summary[0]["low_confidence"] is True
    assert summary[0]["metrics"]["telemetry.sampler_interference_pct"] == pytest.approx(-1.0)
    assert summary[0]["units"]["telemetry.sampler_interference_pct"] == "%"
    repeats = [r for r in ledger.read() if r["verb"] == "sampler-overhead.repeat"]
    assert len(repeats) == 7 and {r["conditions"]["sampler"] for r in repeats} == {"off", "on", "warmup-off"}


def test_a_server_that_is_already_up_is_used_and_left_running(tmp_path):
    experiment, events, _, pod, *_ = build(tmp_path, server_up=True)
    experiment.run()
    assert ("server", "serve") not in events and ("server", "stop") not in events
    assert pod.server is True


def test_a_sampler_that_was_running_is_stopped_for_the_off_runs_and_restarted_at_the_end(tmp_path):
    experiment, events, _, pod, *_ = build(tmp_path, sampler_running=True)
    experiment.run()
    steps = [e[1] for e in events if e[0] == "sampler"]
    assert steps[0] == "status" and steps[1] == "stop"
    assert steps[-1] == "start" and pod.sampler is True
    assert [e[1] for e in events if e[0] == "cpu"][:2] == [False, False]


def test_the_sampler_is_started_with_the_pods_node_name_and_period(tmp_path):
    experiment, _, _, pod, *_ = build(tmp_path, settings=so.Settings(period_ms=500))
    experiment.run()
    assert ("pod_power.sh", ("start", "gpu-b", "500")) in pod.calls


def test_a_failure_in_the_middle_still_stops_the_sampler_and_the_server_and_releases_the_slot(tmp_path):
    experiment, events, fake_slots, pod, *_ = build(tmp_path, bench_kwargs={"fail_on": 4})
    result = experiment.run()
    assert result["ok"] is False and result["outcome"] == "fail" and "502" in result["error"]
    assert pod.sampler is False and pod.server is False
    assert fake_slots.released and fake_slots.released[0]["cancel"] is False
    assert "failed" in fake_slots.released[0]["result"] or "502" in fake_slots.released[0]["result"]
    rows = [json.loads(line) for line in (tmp_path / "out" / "rows.jsonl").read_text().splitlines()]
    assert [r["arm"] for r in rows if r["outcome"] == "ok"] == ["warmup", "off", "on"]
    assert rows[-1]["outcome"] == "fail" and "502" in rows[-1]["reason"]


def test_a_server_that_will_not_start_fails_before_any_measuring_and_releases_the_slot(tmp_path):
    experiment, events, fake_slots, pod, bench, _ = build(tmp_path)
    pod.fail_serve = True
    result = experiment.run()
    assert result["outcome"] == "fail" and "server" in result["error"]
    assert bench.calls == 0 and fake_slots.released


def test_the_cpu_read_that_lands_while_the_bench_still_decodes_is_marked_overlapped(tmp_path):
    experiment, *_ = build(tmp_path, bench_kwargs={"hold_s": 0.3})
    experiment.run()
    rows = [json.loads(line) for line in (tmp_path / "out" / "rows.jsonl").read_text().splitlines()]
    assert all(r["cpu_overlap"] is True for r in rows)


def test_a_cpu_read_that_comes_back_after_the_bench_ended_is_marked_and_left_out_of_the_cpu_numbers(tmp_path):
    experiment, *_ = build(tmp_path, pod_kwargs={"wait_for_bench": True})
    result = experiment.run()
    rows = [json.loads(line) for line in (tmp_path / "out" / "rows.jsonl").read_text().splitlines()]
    assert all(r["cpu_overlap"] is False for r in rows)
    assert result["analysis"]["metrics"]["cpu_busy_pct"]["off"]["n"] == 0
    assert result["analysis"]["verdict"] == "no_interference"      # tok/s still decides; CPU is reported apart


# --- the slot ----------------------------------------------------------------------------------------------------

def test_the_slot_is_polled_every_two_minutes_until_it_starts(tmp_path):
    experiment, events, fake_slots, _, _, clock = build(tmp_path, slot_starts=("not_next", "busy", "not_next", "started"))
    experiment.run()
    starts = [e for e in events if e[:2] == ("slot", "start")]
    assert len(starts) == 4
    assert clock.t >= 3 * 120


def test_a_slot_that_never_starts_is_cancelled_and_nothing_runs_on_the_pod(tmp_path):
    experiment, events, fake_slots, pod, bench, _ = build(
        tmp_path, slot_starts=("not_next",), settings=so.Settings(max_wait_s=600, poll_s=120))
    result = experiment.run()
    assert result["ok"] is False and result["outcome"] == "no_slot"
    assert result["harness_ready"] is True and "never started" in result["error"]
    assert fake_slots.released == [{"row": "r" * 32, "result": None, "cancel": True}]
    assert pod.calls == [] and bench.calls == 0
    assert len([e for e in events if e[:2] == ("slot", "start")]) == 6      # 0, 120, ... 600 seconds


def test_a_refused_request_stops_at_once(tmp_path):
    experiment, events, fake_slots, pod, *_ = build(tmp_path)
    fake_slots.request = lambda **kw: {"ok": False, "outcome": "placement_contradiction", "reason": "other card"}
    result = experiment.run()
    assert result["outcome"] == "no_slot" and "placement_contradiction" in result["error"]
    assert pod.calls == [] and fake_slots.released == []


def test_a_start_that_fails_for_a_reason_waiting_will_not_fix_stops_and_cancels(tmp_path):
    experiment, events, fake_slots, pod, *_ = build(tmp_path, slot_starts=("row_not_found",))
    result = experiment.run()
    assert result["outcome"] == "no_slot" and "row_not_found" in result["error"]
    assert pod.calls == []


def test_a_running_slot_can_be_used_by_row_id_and_is_still_released(tmp_path):
    experiment, events, fake_slots, *_ = build(tmp_path, settings=so.Settings(row="a" * 32))
    experiment.run()
    assert not [e for e in events if e[:2] == ("slot", "request")]
    assert fake_slots.released[0]["row"] == "a" * 32


def test_the_slot_is_extended_once_when_the_run_will_not_fit_and_the_run_finishes(tmp_path):
    experiment, events, fake_slots, pod, bench, clock = build(
        tmp_path, bench_kwargs={"repeat_s": 200.0}, settings=so.Settings(slot_minutes=15))
    result = experiment.run()
    assert result["ok"] is True
    assert len([e for e in events if e == ("slot", "extend")]) == 1
    assert [e[1] for e in events if e[0] == "bench"].count("on-3") == 1


def test_when_the_extension_is_refused_the_run_stops_with_what_it_has_and_still_releases(tmp_path):
    experiment, events, fake_slots, pod, bench, clock = build(
        tmp_path, bench_kwargs={"repeat_s": 200.0}, extend_ok=False)
    result = experiment.run()
    assert result["outcome"] == "partial" and result["ok"] is False
    assert fake_slots.released and pod.sampler is False
    assert result["analysis"]["verdict"] == "inconclusive"


# --- the real pieces under fakes ---------------------------------------------------------------------------------

class World:
    def __init__(self) -> None:
        self.now = T0
        self.sleeps: list[float] = []
        self.fake = slots_fake.FakeNotion(clock=lambda: self.now)
        self.client = NotionClient(slots_fake.TOKEN, transport=self.fake.transport, sleep=self.sleeps.append)

    def queue_slots(self) -> so.QueueSlots:
        return so.QueueSlots(client=self.client, now=lambda: self.now)


def test_queue_slots_wait_behind_a_running_row_then_take_the_card_when_it_is_released():
    world = World()
    other = world.fake.add_row("R33: chip sheet", task_id="R33", status="Running", started=world.now, requested=world.now)
    gate = world.queue_slots()
    asked = gate.request(task="T9", what="sampler overhead benchmark, six repeats", minutes=15)
    assert asked["ok"] and asked["outcome"] == "queued"
    refused = gate.start(asked["row"])
    assert refused["ok"] is False and refused["outcome"] == "busy"
    world.fake.requests.clear()
    # R33 ends: its row is Done, then ours starts
    world.client.update_page(other, slots.rest_properties({"Status": "Done", "Ended": world.now}))
    started = gate.start(asked["row"])
    assert started["ok"] is True and started["outcome"] == "started"
    done = gate.done(asked["row"], "10.00 tok/s off, 9.90 on")
    assert done["ok"] is True and done["outcome"] == "done"
    assert world.fake.values(asked["row"])["Status"] == "Done"
    assert [r["verb"] for r in ledger.read()] == ["slot request", "slot start", "slot start", "slot done"]


def test_queue_slots_extend_and_cancel_go_through_the_same_rules():
    world = World()
    gate = world.queue_slots()
    asked = gate.request(task="T9", what="sampler overhead benchmark", minutes=15)
    cancelled = gate.done(asked["row"], None, cancel=True)
    assert cancelled["outcome"] == "cancelled"
    again = gate.request(task="T9", what="sampler overhead benchmark again", minutes=15)
    gate.start(again["row"])
    extended = gate.extend(again["row"], "three of six repeats measured")
    assert extended["ok"] and extended["outcome"] == "extended"


def test_the_attach_bench_runs_the_real_harness_over_a_fake_hub_proxy(tmp_path):
    from lab.proxy import PodLink
    from lab.teacher_fake import FakeTeacher, text_policy

    fake = FakeTeacher(text_policy("word " * 400), user="acct-d", token="hubtoken-" + "k" * 32)
    link = PodLink.for_account(fake.account("D"), transport=fake.transport, sleep=lambda s: None)
    bench = so.AttachBench(lambda: __import__("lab.bench", fromlist=["ProxyEndpoint"]).ProxyEndpoint(link, 8000),
                           out_dir=tmp_path / "bench", batch=4, max_tokens=16, hub="https://sjc.test", account="acct-d",
                           node="pod-b")
    row = bench("off-1")
    assert row["outcome"] == "ok" and row["decode_tok_s_total"] is not None and row["ttft_s"] is not None
    assert row["batch_size"] == 4 and len(row["requests"]) == 4          # the whole fixed prompt set, once
    assert row["warmup"]["requests"] == 0
    body = fake.requests[-1].body
    assert body["max_tokens"] == 16 and body["temperature"] == 0 and body["ignore_eos"] is True
    link.close()


def test_the_cpu_script_runs_against_a_fake_proc_and_prints_the_marked_line(tmp_path):
    proc = tmp_path / "proc"
    (proc / "123").mkdir(parents=True)
    (proc / "stat").write_text("cpu  100 0 50 800 10 0 0 0 0 0\ncpu0 100 0 50 800 10 0 0 0 0 0\n")
    (proc / "123" / "stat").write_text("123 (python3) S 1 1 1 0 -1 4194560 100 0 0 0 7 3 0 0 20 0 1 0 55 0 0\n")
    pidfile = tmp_path / "sampler.pid"
    pidfile.write_text("123\n")
    out = subprocess.run([sys.executable, "-c", so.CPU_PY, "0.2", str(pidfile)], capture_output=True, text=True,
                         env={"PROC_ROOT": str(proc), "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""), "PATH": ""}, timeout=30)
    assert out.returncode == 0, out.stderr
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("SAMPLER_OVERHEAD_CPU ")][0]
    data = json.loads(line.split(" ", 1)[1])
    assert data["schema"] == so.CPU_SCHEMA and data["pid"] == 123 and data["ticks0"] == 10 and data["ticks1"] == 10
    assert len(data["stat0"]) == 8 and data["t1"] > data["t0"]


def test_the_cpu_script_text_carries_the_python_and_runs_under_bash_syntax_check(tmp_path):
    text = so.cpu_script_text()
    assert text.startswith("#!/bin/bash") and so.CPU_PY.strip().splitlines()[0] in text
    checked = subprocess.run(["bash", "-n"], input=text.encode(), capture_output=True, timeout=30)
    assert checked.returncode == 0, checked.stderr


def test_the_verb_is_in_the_registry_in_alphabetical_order_and_has_a_help_line():
    names = list(lab_cli.VERBS)
    assert names == sorted(names) and lab_cli.VERBS["sampler-overhead"] == "lab.sampler_overhead:main"
    assert "pod sampler" in lab_cli.summary(lab_cli.VERBS["sampler-overhead"]).lower()


def test_plan_prints_the_schedule_and_the_steps_and_touches_nothing(tmp_path, capsys):
    answer = so.main(["run", "--account", "D", "--dry-run"])
    assert answer["ok"] is True and answer["outcome"] == "dry_run"
    assert answer["schedule"] == ["off", "on", "off", "on", "off", "on"]
    assert answer["slot"]["minutes"] == 15 and answer["slot"]["poll_s"] == 120 and answer["slot"]["max_wait_s"] == 5400
    assert "pod_a_teacher.sh serve" in " ".join(answer["steps"])


def test_analyze_reads_a_rows_file_and_prints_the_numbers(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows_for([10.0, 10.2, 9.8], [9.9, 10.1, 9.7])))
    answer = so.main(["analyze", "--rows", str(path)])
    assert answer["ok"] is True and answer["verdict"] == "no_interference"
    assert answer["metrics"]["decode_tok_s"]["diff"] == pytest.approx(-0.1)


def test_analyze_refuses_a_missing_file(tmp_path):
    answer = so.main(["analyze", "--rows", str(tmp_path / "nope.jsonl")])
    assert answer["ok"] is False and answer["outcome"] == "error"


def test_the_module_never_prints_or_stores_a_hub_value():
    text = (REPO / "lab" / "sampler_overhead.py").read_text(encoding="utf-8")
    assert "AUP_HUB_TOKEN" not in text and "NOTION_TOKEN" not in text and "AUP_HUB_PASS" not in text
