"""lab/attribution_check.py (T8): joules per answer summed over a run, the gaps between answers, and the whole run.

Everything runs offline on one fake timeline: a fake clock, a fake teacher that takes a fixed time per answer, and
the real pod sampler (lab/pod_power.py PodSampler) recording a synthetic power trace on a pod clock that runs ahead
of the client's, read through bench's proxy reader over an httpx mock transport. The trace sits on a whole-second
grid, so every energy below is worked out by hand from trapezoids.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import httpx
import pytest

from lab import __main__ as lab_cli
from lab import attribution_check as ac
from lab import pod_power
from lab.hub import Account
from lab.machine import ledger
from lab.machine.verdict import verdict
from lab.proxy import PodLink

REPO = Path(__file__).resolve().parents[1]
T0 = 1_760_000_000.0
IDLE_W, BUSY_W = 10.0, 110.0


class World:
    """One fake timeline: the client clock (now, sleep), the pod's clock `offset` seconds ahead, a sampler on the
    pod recording the power of whatever the fake teacher is doing, once a second."""

    def __init__(self, durations, *, offset=46.25, tail=(), inside_only=False, others=0, running=0, waiting=0):
        self.t = T0
        self.offset = offset
        self.durations = list(durations)
        self.tail = list(tail)
        self.inside_only = inside_only      # busy only strictly inside a window: the edges read idle
        self.windows: list[tuple[float, float]] = []
        self.asked: list[int] = []
        self.others = others                # requests other clients finish while the run goes on
        self.running, self.waiting = running, waiting
        self.finished = 1000
        reader = type("Reader", (), {"source": "amd-smi"})()
        self.sampler = pod_power.PodSampler(reader, "gpu-a", period_ms=1000.0, clock=lambda: self.t + offset)
        self.next_sample = T0
        self.link = PodLink.for_account(Account("C", "acct-c", "t" * 40, "https://hub.test"),
                                        transport=httpx.MockTransport(self._handle), sleep=lambda s: None)

    # the clock
    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self._advance(seconds)

    def _advance(self, seconds: float) -> None:
        end = self.t + seconds
        while self.next_sample < end:
            self.sampler.record(pod_power.Reading(socket_w=self.watts(self.next_sample), power_field="socket_power"),
                                self.next_sample + self.offset)
            self.next_sample += 1.0
        self.t = end

    def watts(self, t: float) -> float:
        for s, e in self.windows:
            if (s < t < e) if self.inside_only else (s <= t <= e):
                return BUSY_W
        for _, e in self.windows:
            k = round(t - e)
            if 1 <= k <= len(self.tail) and abs(t - e - k) < 1e-9:
                return self.tail[k - 1]
        return IDLE_W

    # the teacher
    def ask(self, index: int) -> ac.Answer:
        self.asked.append(index)
        start, duration = self.t, self.durations[index % len(self.durations)]
        self.windows.append((start, start + duration))
        self._advance(duration)
        return ac.Answer(index=index, request_id=f"r{index:03d}", prompt_id="text-kitchen", started=start,
                         ended=self.t, tokens_in=40, tokens_out=128, first_token=start + 0.5, finish_reason="length")

    def load(self) -> dict:
        done = self.finished + len(self.asked) + (self.others if self.asked else 0)
        return {"running": self.running, "waiting": self.waiting, "finished": done}

    def ready(self) -> bool:
        return True

    # the pod's sampler endpoint behind the hub proxy
    def _handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if "/proxy/8790/status" in path:
            body = self.sampler.status()
        elif "/proxy/8790/samples" in path:
            body = self.sampler.samples_after(int(request.url.params.get("after", "0")),
                                              int(request.url.params.get("limit", "1000")))
        else:
            return httpx.Response(404, text="no")
        stamp = pod_power.iso(self.t + self.offset)     # the reply leaves the pod at once: round trip 0
        body["pod_received"] = body["pod_sent"] = stamp
        return httpx.Response(200, json=body)

    def power(self) -> ac.SamplerReader:
        return ac.SamplerReader(self.link, 8790, now=self.now)


def run(world: World, n: int, **kwargs) -> dict:
    return ac.run_experiment(world, world.power(), requests=n, idle_s=60.0, gap_s=10.0, settle_s=5.0,
                             now=world.now, sleep=world.sleep, **kwargs)


# --- the computation, worked out by hand -----------------------------------------------------------------------

def test_a_sequential_run_conserves_energy_and_the_gaps_show_the_edge_ramps():
    durations = [40, 41, 42, 40, 41, 42]
    world = World(durations)
    out = run(world, 6)
    r = out["result"]
    assert out["ok"] and r["answers"] == 6 and r["failed"] == 0
    # each window holds busy samples from its start to its end, both included: 110 W times its length
    assert r["answers_energy_j"] == pytest.approx(BUSY_W * sum(durations))
    # an inner gap of 10 s: a ramp down (110 to 10) over its first second, 8 s at 10 W, a ramp up over its last
    # second: 60 + 80 + 60 = 200 J; the last gap has no answer after it: 60 + 90 = 150 J
    assert r["gap_energy_j"] == pytest.approx(5 * 200 + 150)
    assert r["gap_seconds"] == pytest.approx(60.0)
    assert r["run"]["energy_j"] == pytest.approx(BUSY_W * sum(durations) + 1150)
    assert r["conservation"]["ok"] and abs(r["conservation"]["residual_j"]) < 1e-6
    assert r["idle"]["watts"] == pytest.approx(IDLE_W) and r["idle"]["under_load_samples"] == 0
    assert r["idle"]["seconds"] == pytest.approx(60.0) and r["idle"]["samples"] == 61
    # above idle, the gaps hold 100 J per inner gap and 50 J after the last answer
    assert r["missed_j"] == pytest.approx(550.0)
    share = 550.0 / (BUSY_W * sum(durations) + 550.0)
    assert r["missed_share"] == pytest.approx(share)
    assert r["attribution_error_pct"] == pytest.approx(-100.0 * share)
    assert r["samples_per_window"] == {"median": 42.0, "min": 41, "max": 43, "under_min": 0, "min_required": 10}
    a = [BUSY_W * d for d in durations]
    b = [x + (100.0 if i < 5 else 50.0) for i, x in enumerate(a)]
    expected = verdict(a, b)
    assert r["verdict"]["label"] == expected.label and r["verdict"]["p"] == pytest.approx(expected.p)
    assert r["joules_per_answer"]["mean"] == pytest.approx(sum(a) / 6)
    assert r["joules_per_token_mean"] == pytest.approx(sum(a) / 6 / 128)
    assert r["period_ms"] == 1000.0 and r["sampler"] == "lab/pod_power.py 1"


def test_windows_that_hold_every_busy_sample_miss_nothing_and_are_trustworthy():
    durations = [40, 42, 44, 40, 42, 44]
    world = World(durations, inside_only=True)
    r = run(world, 6)["result"]
    # the edges read idle, so each window is a ramp up, D - 2 s at 110 W and a ramp down: 110 D - 100 J
    assert r["answers_energy_j"] == pytest.approx(sum(BUSY_W * d - 100 for d in durations))
    assert r["missed_j"] == pytest.approx(0.0, abs=1e-6)
    assert r["verdict"]["label"] == "no_difference"
    assert r["trustworthy"] is True and r["reasons"] == []


def test_a_tail_after_each_answer_is_what_the_windows_miss():
    world = World([40] * 6, tail=(60.0, 30.0))
    r = run(world, 6)["result"]
    # the tail turns the first 3 s of each gap from 60 + 10 + 10 into 85 + 45 + 20: 70 J more after every answer
    assert r["missed_j"] == pytest.approx(550.0 + 6 * 70.0)
    assert r["verdict"]["label"] == "b_ahead"
    assert r["trustworthy"] is False
    assert any("miss" in reason for reason in r["reasons"])


def test_the_pod_clock_offset_is_measured_and_changes_no_number():
    ahead = run(World([40, 41, 42, 40, 41, 42], offset=46.25), 6)
    same = run(World([40, 41, 42, 40, 41, 42], offset=0.0), 6)
    assert ahead["result"]["clock_offset"]["offset_s"] == pytest.approx(46.25)
    assert ahead["result"]["clock_offset"]["error_s"] == pytest.approx(0.0)
    for key in ("answers_energy_j", "gap_energy_j", "missed_j", "missed_share"):
        assert ahead["result"][key] == pytest.approx(same["result"][key])
    assert ahead["result"]["run"]["energy_j"] == pytest.approx(same["result"]["run"]["energy_j"])


def test_short_answers_have_too_few_samples_per_window():
    r = run(World([3] * 6, inside_only=True), 6)["result"]
    assert r["samples_per_window"]["max"] == 4 and r["samples_per_window"]["under_min"] == 6
    assert r["trustworthy"] is False and any("fewer than 10 samples" in x for x in r["reasons"])


def test_too_few_answers_is_too_early():
    r = run(World([40]), 3)["result"]
    assert r["verdict"]["label"] == "too_early" and r["trustworthy"] is False


# --- the run: preconditions and the other clients ---------------------------------------------------------------

def test_a_teacher_serving_other_requests_is_never_measured():
    world = World([40], running=3, waiting=2)
    out = run(world, 6)
    assert out["ok"] is False and out["outcome"] == "busy" and world.asked == []


def test_requests_from_other_clients_during_the_run_make_it_untrustworthy():
    out = run(World([40, 41, 42, 40, 41, 42], inside_only=True, others=2), 6)
    assert out["result"]["other_requests"] == 2
    assert out["result"]["trustworthy"] is False
    assert any("other clients" in x for x in out["result"]["reasons"])


def test_the_run_writes_its_raw_rows_and_a_replay_gives_the_same_numbers(tmp_path):
    out = run(World([40, 41, 42, 40, 41, 42]), 6, out_dir=tmp_path, tag="test")
    files = {Path(p).name for p in out["files"]}
    assert files == {"attribution_check_test_samples.jsonl", "attribution_check_test_requests.jsonl",
                     "attribution_check_test_run.jsonl"}
    samples = (tmp_path / "attribution_check_test_samples.jsonl").read_text().splitlines()
    assert all("t_local" not in json.loads(line) for line in samples)
    again = ac.replay(tmp_path, "test")
    for key in ("answers_energy_j", "gap_energy_j", "missed_j", "missed_share", "trustworthy"):
        assert again[key] == out["result"][key]
    assert again["verdict"] == out["result"]["verdict"]


# --- the ledger row and the command line -------------------------------------------------------------------------

def test_the_ledger_row_carries_the_numbers_and_stays_low_confidence(tmp_path):
    out = run(World([40, 41, 42, 40, 41, 42]), 6)
    seq = ac.record_result(out["result"], source="test", state_dir=tmp_path)
    row = ledger.read(state_dir=tmp_path)[-1]
    assert row["seq"] == seq and row["verb"] == "attribution-check" and row["task"] == "T8"
    assert row["node"] == "pod-a" and row["low_confidence"] is True
    m = row["metrics"]
    assert m["pod.idle_watts"] == pytest.approx(IDLE_W)
    assert m["telemetry.attribution_error_pct"] == pytest.approx(out["result"]["attribution_error_pct"], abs=1e-3)
    assert m["telemetry.samples_per_window"] == 42.0
    for name in ("telemetry.answers_energy_j", "telemetry.gap_energy_j", "telemetry.run_energy_j",
                 "telemetry.missed_energy_j", "telemetry.joules_per_answer"):
        assert name in m and row["units"][name] == "J"
    assert row["conditions"]["verdict"]["label"] == out["result"]["verdict"]["label"]
    assert row["conditions"]["basis"] == "gross" and row["conditions"]["sampler"] == "lab/pod_power.py 1"


def test_the_verb_is_registered_in_order():
    names = list(lab_cli.VERBS)
    assert lab_cli.VERBS["attribution-check"] == "lab.attribution_check:main"
    assert names == sorted(names)


def test_main_runs_with_injected_clients_and_records_the_row(tmp_path):
    world = World([40, 41, 42, 40, 41, 42])
    out = ac.main(["run", "--requests", "6", "--idle-s", "60", "--gap-s", "10", "--out", str(tmp_path),
                   "--tag", "cli", "--state-dir", str(tmp_path / "state")],
                  teacher=world, power=world.power(), now=world.now, sleep=world.sleep)
    assert out["ok"] is True and out["ledger_seq"] == 1
    assert ledger.read(state_dir=tmp_path / "state")[-1]["verb"] == "attribution-check"


def test_main_waits_for_a_busy_teacher_then_gives_up(tmp_path):
    world = World([40], running=1)
    out = ac.main(["run", "--requests", "6", "--wait-min", "6", "--out", str(tmp_path), "--tag", "busy",
                   "--state-dir", str(tmp_path / "state")],
                  teacher=world, power=world.power(), now=world.now, sleep=world.sleep)
    assert out["ok"] is False and out["outcome"] == "busy" and world.asked == []
    assert world.t - T0 == pytest.approx(360.0)        # three waits of two minutes


# --- the client that talks to the real teacher, against the fake teacher -----------------------------------------

def test_the_proxy_teacher_sends_the_bench_text_prompts_greedy_without_thinking():
    from lab.teacher_fake import FakeTeacher, text_policy

    fake = FakeTeacher(text_policy("word " * 400), user="acct-c", token="hubtoken-" + "k" * 32)
    link = PodLink.for_account(fake.account("C"), transport=fake.transport, sleep=lambda s: None)
    teacher = ac.ProxyTeacher(link, max_tokens=128)
    assert teacher.ready() is True
    first, second = teacher.ask(0), teacher.ask(1)
    assert first.error is None and first.tokens_out == 128 and first.ended >= first.started
    assert first.prompt_id != second.prompt_id
    bodies = [s.body for s in fake.requests if s.method == "POST"]
    for body in bodies:
        assert body["model"] == "teacher" and body["temperature"] == 0 and body["max_tokens"] == 128
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert all(isinstance(m["content"], str) for m in body["messages"])    # text only, no image parts
    assert teacher.load() is None                    # the fake serves no /metrics: unknown, not zero
    link.close()


def test_vllm_metrics_text_gives_running_waiting_and_finished():
    text = "\n".join([
        "# HELP vllm:num_requests_running Number of requests in model execution batches.",
        'vllm:num_requests_running{engine="0",model_name="teacher"} 0.0',
        'vllm:num_requests_waiting{engine="0",model_name="teacher"} 2.0',
        'vllm:request_success_total{engine="0",finished_reason="stop",model_name="teacher"} 44.0',
        'vllm:request_success_total{engine="0",finished_reason="length",model_name="teacher"} 1223.0',
    ])
    assert ac.parse_load(text) == {"running": 0, "waiting": 2, "finished": 1267}
    assert ac.parse_load("nothing here") is None
