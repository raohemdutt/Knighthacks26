"""lab/bench.py, the benchmark harness, one group of tests per clause of R11's Done when.

Two kinds of test, so none depends on how fast the test machine is:
- the arithmetic (decode rate from chunk times, watts over a window, J/token, statistics, bands, the summary) runs on
  recorded or constructed inputs with fixed times;
- whole sessions spawn lab/bench_fake.py as a real server process (a cold spawn, a real token stream, a real stop) and
  read a fake Strix Halo pod, and assert only structure and values the fake decides: counts, roles, conditions,
  cached prompt tokens, mem_busy_percent, and watts bounded by the fake's own power model (lab.bench_fake.FakePower,
  samples on a fixed grid, never a sampling thread).
Every server binds port 0 and names its port in a file, every wait in the harness is bounded, and a watchdog turns a
test that runs past WATCHDOG_S into a crash with every thread's traceback instead of a hang."""
from __future__ import annotations

import base64
import dataclasses
import faulthandler
import json
import os
import socket
import subprocess
import sys
import time
import types
from pathlib import Path

import httpx
import pytest

from lab import bench, bench_fake, pod_power
from lab.machine import ledger, metrics

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "fixtures" / "lab" / "bench"
STRIX = REPO / "fixtures" / "telemetry" / "pod_power"
WATCHDOG_S = 120.0
MAX_TOKENS = 8

# The kernels rubric test 1 (K1 b) conditions, the host facts and the per-run facts every row must carry.
K1_CONDITIONS = ("engine", "engine_commit", "patch_hash", "checkpoint", "quantization", "max_model_len", "batch_size",
                 "node", "rocm", "torch", "triton", "env", "linear_kernels", "attention_backends", "kv_cache_tokens")
HOST_FACTS = ("hub", "account", "hostname", "kernel_release", "amdgpu_driver", "firmware", "asic_serial", "chip")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("LAB_TASK", raising=False)
    faulthandler.dump_traceback_later(WATCHDOG_S, exit=True, file=sys.__stderr__)
    yield
    faulthandler.cancel_dump_traceback_later()


def fake_config(tmp_path, *, engine="vllm", batches=(1, 4), repeats=1, add_64=False, set_env=("GPU_MAX_HW_QUEUES=4",),
                **fake_options) -> tuple[bench.Config, bench_fake.FakeHost, bench_fake.FakePower]:
    root = tmp_path / "pod"
    root.mkdir(parents=True, exist_ok=True)
    host = bench_fake.FakeHost(root)
    power = bench_fake.FakePower(root)
    options = {"engine": engine, "root": str(root), "set_env": list(set_env), **fake_options}
    config = bench.Config(
        engine=engine, command=bench_fake.serve_command(**options), checkpoint="/home/jovyan/models/Qwen3.8-27B-INT4",
        model="teacher", max_len=16384, node="pod-b", port=0, batches=list(batches), add_64=add_64,
        repeats=repeats, max_tokens=MAX_TOKENS, requests_per_stream=2, warmup_rounds=1, idle_s=0.05,
        power_period_s=power.period_s, ready_timeout_s=30.0, request_timeout_s=20.0, out_dir=tmp_path / "out",
        engine_dir="/home/jovyan/vllm", patches=[str(REPO / "lab" / "vllm-gfx1201.diff")],
        launch_env={"GPU_MAX_HW_QUEUES": "2"}, python="python3", hub="https://sjc.aupcloud.io")
    return config, host, power


def run(tmp_path, **kwargs):
    config, host, power = fake_config(tmp_path, **kwargs)
    return config, bench.run_session(config, host=host, power=power)


def synthetic_row(run_id, *, batch=1, decode, outcome="ok", joules=2.0):
    """A row as run_session writes it, with only the fields the summary reads."""
    return {"run_id": run_id, "engine": "vllm", "batch_size": batch, "outcome": outcome, "decode_tok_s": decode,
            "decode_tok_s_total": decode * batch, "ttft_s": 0.5, "joules_per_token": joules,
            "watts": {"avg_w": 100.0, "peak_w": 120.0}, "time_since_spawn_s": 30.0}


# --- the prompt set and the decode settings ---------------------------------------------------------------------

def test_prompt_set_is_text_and_fixture_images_only_and_its_hash_is_stable():
    prompts = bench.PromptSet.load()
    assert prompts.id == "bench-v1"
    assert {p["kind"] for p in prompts.prompts} == {"text", "image"}
    assert len(prompts.sha256) == 64
    assert bench.PromptSet.load().sha256 == prompts.sha256


def test_prompt_set_refuses_a_camera_frame_and_an_image_outside_the_fixtures(tmp_path):
    for bad in ({"id": "x", "kind": "frame", "user": "what is this", "image": "images/bars_1280x720.png"},
                {"id": "x", "kind": "image", "user": "what is this", "image": "../../../data/frames/cam0.jpg"}):
        folder = tmp_path / bad["kind"]
        (folder / "images").mkdir(parents=True)
        (folder / "prompts.json").write_text(json.dumps({"id": "bad", "prompts": [bad]}))
        with pytest.raises(bench.BenchError):
            bench.PromptSet.load(folder / "prompts.json")


def test_request_is_greedy_fixed_length_ignores_eos_and_carries_a_nonce_when_prefix_caching_is_on():
    prompts = bench.PromptSet.load()
    settings = bench.Settings(model="teacher", max_tokens=96, prefix_caching=True, thinking=False)
    first = prompts.request_body(1, settings, nonce="n-1")
    again = prompts.request_body(1, settings, nonce="n-2")
    assert first["temperature"] == 0 and first["max_tokens"] == 96 and first["ignore_eos"] is True
    assert first["stream"] is True and first["stream_options"] == {"include_usage": True}
    assert first["chat_template_kwargs"] == {"enable_thinking": False}
    assert "n-1" in json.dumps(first["messages"]) and "n-2" in json.dumps(again["messages"])
    image = [part for part in first["messages"][-1]["content"] if part["type"] == "image_url"][0]
    data = image["image_url"]["url"]
    assert data.startswith("data:image/png;base64,")
    assert base64.b64decode(data.split(",", 1)[1]) == (FIX / "images" / "bars_1280x720.png").read_bytes()


def test_no_nonce_when_prefix_caching_is_off_and_one_by_default():
    assert bench.Settings(model="m").nonce_on is True                       # vLLM V1 and llama-server cache prefixes
    assert bench.Settings(model="m", prefix_caching=False).nonce_on is False
    prompts = bench.PromptSet.load()
    off = bench.Settings(model="m", prefix_caching=False)
    assert prompts.request_body(0, off, nonce=None) == prompts.request_body(0, off, nonce=None)


# --- batch sizes ------------------------------------------------------------------------------------------------

def test_default_batches_are_1_4_8_16_plus_64_when_it_fits():
    assert bench.DEFAULT_BATCHES == (1, 4, 8, 16)
    plan = bench.plan_batches(None, add_64=True)
    assert [b.size for b in plan] == [1, 4, 8, 16, 64]
    assert [b.when_fits for b in plan] == [False, False, False, False, True]
    assert [b.size for b in bench.plan_batches(None, add_64=False)] == [1, 4, 8, 16]


def test_other_batch_sizes_on_request():
    assert [b.size for b in bench.plan_batches([2, 32], add_64=False)] == [2, 32]
    with pytest.raises(bench.BenchError):
        bench.plan_batches([0], add_64=False)


def test_64_fits_only_when_the_kv_cache_holds_64_requests_of_the_workload():
    assert bench.fits(64, kv_cache_tokens=163_840, tokens_per_request=2_000) is True
    assert bench.fits(64, kv_cache_tokens=100_000, tokens_per_request=2_000) is False
    assert bench.fits(64, kv_cache_tokens=None, tokens_per_request=2_000) is None


# --- client-stream timing (recorded chunk times, no clock) ----------------------------------------------------

def test_decode_rate_is_timed_from_first_to_last_token_of_the_client_stream():
    times = [10.0 + 0.1 * i for i in range(11)]           # 11 tokens, 1.0 s from first to last
    timing = bench.stream_timing(sent=9.5, chunk_times=times, tokens=11)
    assert timing.ttft_s == pytest.approx(0.5)
    assert timing.decode_tok_s == pytest.approx(10.0)
    assert timing.buffered is False


def test_a_stream_the_proxy_held_back_gives_no_decode_rate():
    timing = bench.stream_timing(sent=0.0, chunk_times=[5.0] * 20, tokens=20)
    assert timing.buffered is True and timing.decode_tok_s is None
    assert timing.ttft_s == pytest.approx(5.0)


def test_a_request_is_timed_from_a_recorded_stream_and_reports_cached_prompt_tokens():
    class Recorded:
        def events(self, body):
            yield 2.0, json.dumps({"choices": [{"delta": {"content": "a"}, "finish_reason": None}]})
            yield 2.5, json.dumps({"choices": [{"delta": {"content": "b"}, "finish_reason": None}]})
            yield 3.0, json.dumps({"choices": [{"delta": {"content": "c"}, "finish_reason": "length"}]})
            yield 3.0, json.dumps({"choices": [], "usage": {"prompt_tokens": 40, "completion_tokens": 3,
                                                            "prompt_tokens_details": {"cached_tokens": 32}}})
            yield 3.0, "[DONE]"

    clock = bench.WallClock()
    seen = []
    result = bench.run_request(Recorded(), {}, role="measure", stream=0, index=0, prompt_id="p", clock=clock,
                               on_token=lambda: seen.append(1))
    assert result.tokens == 3 and result.chunks == 3 and len(seen) == 3
    assert result.decode_tok_s == pytest.approx(2.0)        # 2 tokens after the first, over 1.0 s
    assert result.finish_reason == "length" and result.prompt_tokens == 40 and result.cached_tokens == 32
    assert result.error is None


def test_a_stuck_stream_fails_within_its_timeout_instead_of_hanging():
    class Stuck:
        def events(self, body):
            time.sleep(5.0)
            yield 0.0, "[DONE]"

    started = time.monotonic()
    with pytest.raises(bench.BenchError, match="had not finished"):
        bench.run_batch(Stuck(), bench.PromptSet.load(), bench.Settings(model="m"), batch=2, per_stream=1,
                        role="measure", clock=bench.WallClock(), tag="t", timeout_s=0.5, grace_s=0.0)
    assert time.monotonic() - started < 4.0


# --- watts and joules (constructed samples) ---------------------------------------------------------------------

def test_watts_are_integrated_over_the_decode_window_with_its_edges_interpolated():
    points = [(float(t), 100.0) for t in range(0, 21)]
    w = bench.power_window(points, 5.5, 15.5, period_s=1.0)
    assert w["available"] is True
    assert w["energy_j"] == pytest.approx(1000.0) and w["avg_w"] == pytest.approx(100.0)
    ramp = [(float(t), 10.0 * t) for t in range(0, 11)]
    w = bench.power_window(ramp, 2.5, 7.5, period_s=1.0)
    assert w["energy_j"] == pytest.approx(250.0)            # mean 50 W over 5 s
    assert w["peak_w"] == pytest.approx(75.0)


def test_watts_are_unavailable_when_the_samples_do_not_cover_the_window():
    w = bench.power_window([(0.0, 50.0), (1.0, 50.0)], 5.0, 9.0, period_s=1.0)
    assert w["available"] is False and "cover" in w["reason"]


def test_local_power_reads_amd_smi_through_the_pod_power_parsers(tmp_path):
    host = bench_fake.FakeHost(tmp_path)
    power = bench.LocalPower.for_host(host, node="pod-b", period_s=0.01)
    assert power.available, power.reason
    for _ in range(5):
        power.sampler.poll()                                 # read by hand: no thread, no timing
    events = power.events()
    assert len(events) == 5
    assert {e["source"] for e in events} == {"amd-smi"} and {e["power_field"] for e in events} == {"socket_power"}
    assert {e["socket_w"] for e in events} == {bench_fake.IDLE_NO_SERVER_W}
    assert ["/opt/rocm/bin/amd-smi", "metric", "--power", "--clock", "--temperature", "--usage", "--energy"] \
        in host.calls
    # the parser is pod_power's: the recorded strixhalo text reads the same through it
    recorded = pod_power.parse_amdsmi((STRIX / "strixhalo_amdsmi_metric_power.txt").read_text(encoding="utf-8"))
    assert recorded[0].socket_w == 110.0


def test_throttling_is_what_amd_smi_reports(tmp_path):
    host = bench_fake.FakeHost(tmp_path, throttle="THROTTLED")
    power = bench.LocalPower.for_host(host, node="pod-b", period_s=0.01)
    power.sampler.poll()
    assert bench.throttled(power.events()) is True
    assert bench.throttled([{"throttled": None}]) is None      # the Strix Halo reads THROTTLE_STATUS: N/A


def test_fake_power_puts_its_samples_on_a_fixed_grid_from_the_decodes(tmp_path):
    (tmp_path / "decodes.jsonl").write_text(json.dumps({"start": 0.0, "end": 1e12}) + "\n")
    power = bench_fake.FakePower(tmp_path, period_s=0.5)
    power.start()
    events = power.events()
    times = [e["t_local"] for e in events]
    assert all(b - a == pytest.approx(0.5) for a, b in zip(times, times[1:]))
    assert {e["socket_w"] for e in events} == {bench_fake.IDLE_NO_SERVER_W + bench_fake.WATTS_PER_STREAM}


def test_fake_power_draws_the_loaded_idle_while_a_server_is_up_and_until_a_stopped_one_last_decoded(tmp_path):
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()                                              # an exited, reaped process: its pid is not alive
    now = time.time()
    (tmp_path / "servers.jsonl").write_text(
        json.dumps({"pid": gone.pid, "up": now - 9.0}) + "\n" + json.dumps({"pid": os.getpid(), "up": now - 3.0}) + "\n")
    (tmp_path / "decodes.jsonl").write_text(json.dumps({"pid": gone.pid, "start": now - 8.0, "end": now - 6.0}) + "\n")
    power = bench_fake.FakePower(tmp_path, period_s=0.5)
    power.start()
    power.t0 = now - 10.0
    watts = {round(e["t_local"] - now, 1): e["socket_w"] for e in power.events()}
    assert watts[-9.5] == bench_fake.IDLE_NO_SERVER_W                                     # nothing up yet
    assert watts[-8.5] == bench_fake.IDLE_LOADED_W                                        # up, not decoding
    assert watts[-7.0] == bench_fake.IDLE_LOADED_W + bench_fake.WATTS_PER_STREAM          # decoding
    assert watts[-5.5] == watts[-3.5] == bench_fake.IDLE_NO_SERVER_W                      # stopped after its decode
    assert watts[-2.5] == watts[0.0] == bench_fake.IDLE_LOADED_W                          # the live one


def test_the_idle_reading_opens_a_sample_period_after_it_is_asked_for(monkeypatch):
    clock = types.SimpleNamespace(now=100.0)
    monkeypatch.setattr(bench, "time", types.SimpleNamespace(
        time=lambda: clock.now, sleep=lambda s: setattr(clock, "now", clock.now + s), monotonic=lambda: clock.now))

    class JustBusy:                                          # a decode ended at the very moment the reading is asked
        available, reason, period_s = True, None, 0.01

        def events(self):
            grid = [99.0 + k * 0.01 for k in range(int((clock.now + 0.02 - 99.0) / 0.01) + 1)]
            return [{"t_local": t, "socket_w": 23.0 if t <= 100.0 else 8.0} for t in grid]

    assert bench.idle_watts(JustBusy(), 0.05) == 8.0


# --- conditions: serve log, environment, host ---------------------------------------------------------------------

def test_vllm_serve_log_names_the_kernel_classes_and_the_kv_cache_capacity():
    found = bench.parse_serve_log((FIX / "vllm_serve_synthetic.log").read_text(), "vllm")
    assert found["linear_kernels"] == {"CompressedTensorsWNA16": "ConchLinearKernel"}
    assert found["attention_backends"] == ["Triton", "TRITON_ATTN (MMEncoderAttention)"]
    assert found["kv_cache_tokens"] == 163_840
    assert found["max_concurrency"] == 10.0 and found["concurrency_tokens"] == 16_384
    assert found["max_model_len"] == 16_384 and found["quantization"] == "compressed-tensors"
    assert found["prefix_caching"] is True


def test_llamacpp_serve_log_names_attention_and_kv_cache_capacity():
    found = bench.parse_serve_log((FIX / "llamacpp_serve_synthetic.log").read_text(), "llamacpp")
    assert found["kv_cache_tokens"] == 16_384
    assert found["attention_backends"] == ["flash_attn enabled"]
    assert found["linear_kernels"] == {}
    assert found["engine_build"] == "6500 (1a2b3c4d)"
    assert found["quantization"] == "Q4_K - Medium" and found["max_model_len"] == 16_384


def test_host_facts_come_from_amd_smi_static_and_firmware(tmp_path):
    host = bench_fake.FakeHost(tmp_path)
    facts = bench.host_facts(host, hub="https://sjc.aupcloud.io")
    assert facts["hub"] == "https://sjc.aupcloud.io" and facts["account"] == "roscon02"
    assert facts["hostname"] == "jupyter-roscon02" and facts["kernel_release"] == "6.8.0-fake-generic"
    assert facts["amdgpu_driver"] == "6.12.12-fake"
    assert facts["asic_serial"] == "0x00000000FAKE0001"
    assert facts["chip"] == "gfx1151" and facts["market_name"] == "AMD Radeon 8060S"
    assert facts["firmware"]["SMU"] == "100.112.0" and len(facts["firmware"]) == 4
    assert ["/opt/rocm/bin/amd-smi", "static", "--driver"] in host.calls


def test_host_facts_say_why_when_amd_smi_cannot_be_read(tmp_path):
    def refuse(argv):
        raise bench.HostError("no amd-smi")

    host = bench.Host(root=tmp_path, run=refuse, which=lambda name: None, environ={})
    facts = bench.host_facts(host)
    assert facts["amdgpu_driver"] is None and "amdgpu_driver" in facts["unavailable"]


def test_environment_is_read_back_from_the_process_with_gpu_max_hw_queues_and_no_secrets(tmp_path):
    proc = tmp_path / "proc" / "4242"
    proc.mkdir(parents=True)
    proc.joinpath("environ").write_bytes(b"GPU_MAX_HW_QUEUES=4\0HIP_VISIBLE_DEVICES=0\0HF_TOKEN=hf_secret\0"
                                         b"HOME=/home/jovyan\0VLLM_USE_TRITON_FLASH_ATTN=1\0")
    host = bench.Host(root=tmp_path, run=lambda argv: "", which=lambda name: None, environ={})
    env, why = bench.read_environ(host, 4242)
    assert why is None
    assert env == {"GPU_MAX_HW_QUEUES": "4", "HIP_VISIBLE_DEVICES": "0", "VLLM_USE_TRITON_FLASH_ATTN": "1"}
    env, why = bench.read_environ(host, 999)
    assert env is None and "999" in why
    proc.joinpath("environ").write_bytes(b"HOME=/x\0")
    env, _ = bench.read_environ(host, 4242)
    assert env == {"GPU_MAX_HW_QUEUES": None}                # unset, said so


def test_kfd_queue_count_where_the_pod_exposes_it(tmp_path):
    host = bench.Host(root=tmp_path, run=lambda argv: "", which=lambda name: None, environ={})
    assert bench.kfd_queues(host, 7) == (None, "/sys/class/kfd/kfd/proc/7/queues is not exposed")
    for q in range(5):
        (tmp_path / "sys/class/kfd/kfd/proc/7/queues" / str(q)).mkdir(parents=True)
    assert bench.kfd_queues(host, 7) == (5, None)


# --- statistics and bands (constructed values) ------------------------------------------------------------------

def test_stats_report_median_min_max_mean_and_sd():
    s = bench.stats([3.0, 1.0, 2.0, 4.0])
    assert s == {"n": 4, "median": 2.5, "min": 1.0, "max": 4.0, "mean": 2.5, "sd": pytest.approx(1.290994, abs=1e-5)}
    assert bench.stats([5.0])["sd"] is None
    assert bench.stats([])["n"] == 0


def test_bands_come_from_the_spawns_own_decode_rate_clusters():
    rates = {"s1": 2.90, "s2": 2.88, "s3": 1.80, "s4": 2.86, "s5": 1.75}
    bands = bench.classify_bands(rates)
    assert bands == [["s1", "s2", "s4"], ["s3", "s5"]]
    assert bench.classify_bands({"s1": 2.9, "s2": 2.85, "s3": 2.8, "s4": 2.88}) == [["s1", "s4", "s2", "s3"]]
    assert bench.classify_bands({"s1": 2.9, "s2": 1.0}) == [["s1", "s2"]]   # two spawns: too few to band


def test_mem_busy_threshold_is_set_from_the_clusters_never_in_advance():
    rates = {"s1": 2.9, "s2": 2.88, "s3": 1.8, "s4": 2.86, "s5": 1.75}
    busy = {"s1": 55.0, "s2": 56.0, "s3": 85.0, "s4": 54.0, "s5": 86.0}
    rule = bench.band_rule(bench.classify_bands(rates), rates, busy)
    assert [c["band"] for c in rule["clusters"]] == ["b1", "b2"]
    boundary = rule["boundaries"][0]
    assert boundary["rate_threshold_tok_s"] == pytest.approx((2.86 + 1.8) / 2)
    assert boundary["mem_busy_threshold_pct"] == pytest.approx((55.0 + 85.5) / 2)
    assert boundary["mem_busy_separates"] is True
    busy["s2"] = 90.0
    rule = bench.band_rule(bench.classify_bands(rates), rates, busy)
    assert rule["boundaries"][0]["mem_busy_separates"] is False


def test_repeats_are_spawns_summarised_per_configuration_and_per_band():
    probes = {"0": (2.90, 55.0), "1": (2.88, 56.0), "2": (1.80, 85.0), "3": (2.86, 54.0), "4": (1.75, 86.0)}
    rows = [synthetic_row(f"run-{k}", decode=rate, joules=10.0 / rate) for k, (rate, _) in probes.items()]
    rows.append({**synthetic_row("run-skip", batch=64, decode=0.0), "outcome": "skipped"})
    spawns = {k: {"probe": {"decode_tok_s": rate, "mem_busy_pct": {"median": busy}}, "runs": [f"run-{k}"]}
              for k, (rate, busy) in probes.items()}
    summary = bench.summarize_runs(rows, spawns)
    assert set(summary["configurations"]) == {"vllm b1"}               # the skipped 64 arm is not a configuration
    entry = summary["configurations"]["vllm b1"]
    assert [s["run_id"] for s in entry["spawns"]] == ["run-0", "run-1", "run-2", "run-3", "run-4"]
    assert entry["stats"]["decode_tok_s"] == bench.stats([2.90, 2.88, 1.80, 2.86, 1.75])
    assert {s["run_id"]: s["band"] for s in entry["spawns"]} == {
        "run-0": "b1", "run-1": "b1", "run-3": "b1", "run-2": "b2", "run-4": "b2"}
    assert entry["bands"]["b1"]["share"] == pytest.approx(0.6) and entry["bands"]["b2"]["share"] == pytest.approx(0.4)
    assert entry["bands"]["b1"]["stats"]["decode_tok_s"] == bench.stats([2.90, 2.88, 2.86])
    assert entry["bands"]["b2"]["stats"]["joules_per_token"] == bench.stats([10.0 / 1.80, 10.0 / 1.75])
    boundary = summary["band_rule"]["boundaries"][0]
    assert boundary["mem_busy_threshold_pct"] == pytest.approx((55.0 + 85.5) / 2)
    assert boundary["mem_busy_separates"] is True


def test_spawns_without_a_probe_are_banded_by_their_measured_rate():
    rows = [synthetic_row(f"run-{k}", decode=rate) for k, rate in enumerate([3.0, 3.0, 3.0, 1.0])]
    spawns = {str(k): {"probe": None, "runs": [f"run-{k}"]} for k in range(4)}
    entry = bench.summarize_runs(rows, spawns)["configurations"]["vllm b1"]
    assert entry["bands"]["b1"]["spawns"] == 3 and entry["bands"]["b2"]["spawns"] == 1


# --- whole sessions against the spawned fake (structure and the fake's own values) --------------------------------

def test_each_run_is_a_cold_spawn_whose_warmup_is_discarded_and_marked(tmp_path):
    config, result = run(tmp_path, batches=(1, 4), repeats=2)
    rows = result["rows"]
    assert [(r["batch_size"], r["repeat"]) for r in rows] == [(1, 0), (4, 0), (1, 1), (4, 1)]
    assert len({r["spawn"]["pid"] for r in rows}) == 4 and all(r["spawn"]["cold"] for r in rows)
    lines = (config.out_dir / result["session"] / "rows.jsonl").read_text().splitlines()
    assert [json.loads(line)["run_id"] for line in lines] == [r["run_id"] for r in rows]   # one JSON row per run
    n_prompts = len(bench.PromptSet.load().prompts)
    for row in rows:
        assert row["outcome"] == "ok", row.get("reason")
        warm = row["warmup"]
        assert warm["discarded"] is True and warm["requests"] == n_prompts + row["batch_size"]
        assert all(item["role"] == "warmup" for item in warm["items"])
        assert len(row["requests"]) == row["batch_size"] * 2 and all(r["role"] == "measure" for r in row["requests"])
        assert row["tokens_generated"] == row["batch_size"] * 2 * MAX_TOKENS
        assert all(r["finish_reason"] == "length" and r["tokens"] == MAX_TOKENS for r in row["requests"])
        assert row["time_since_spawn_s"] > 0 and row["spawn"]["ready_s"] > 0
        assert row["spawn"]["stopped"] is True
        probe = row["probe"]
        assert probe["in_flight"] == 1 and probe["request"]["role"] == "probe"
        assert probe["mem_busy_pct"]["median"] == bench_fake.MEM_BUSY_ONE           # read while it decoded alone
        assert probe["mem_busy_pct"]["samples"] == MAX_TOKENS
    assert [s["run_id"] for s in result["summary"]["configurations"]["vllm b4"]["spawns"]] ==         [rows[1]["run_id"], rows[3]["run_id"]]


def test_rows_carry_decode_ttft_watts_joules_and_the_idle_reading_beside_it(tmp_path):
    _, result = run(tmp_path, batches=(1, 4))
    for row in result["rows"]:
        for key in ("decode_tok_s", "decode_tok_s_total", "ttft_s"):
            assert row[key] is None or row[key] > 0, key
        w = row["watts"]
        assert w["available"] is True and w["source"] == "amd-smi" and w["power_field"] == "socket_power"
        ceiling = bench_fake.IDLE_LOADED_W + bench_fake.WATTS_PER_STREAM * row["batch_size"]
        assert bench_fake.IDLE_NO_SERVER_W <= w["avg_w"] <= w["peak_w"] <= ceiling
        assert w["energy_j"] == pytest.approx(w["avg_w"] * row["window_s"], rel=1e-3)
        assert row["joules_per_token"] == pytest.approx(w["energy_j"] / row["tokens_generated"], rel=1e-3)
        # exact: each reading opens one sample period after what ran before it ended, so no edge blends with a
        # previous spawn's decode, and the fake reports a live server's idle draw apart from no server at all
        assert row["idle_w"] == {"no_server": bench_fake.IDLE_NO_SERVER_W, "loaded": bench_fake.IDLE_LOADED_W}
        assert row["throttled"] is None


def test_every_row_carries_the_kernels_rubric_test_1_conditions_and_the_host(tmp_path):
    _, result = run(tmp_path, batches=(1,))
    c = result["rows"][0]["conditions"]
    for key in K1_CONDITIONS:
        assert key in c and c[key] is not None, key
    for key in HOST_FACTS:
        assert c["host"][key] is not None, key
    assert c["engine_commit"] == bench_fake.FAKE_COMMIT
    assert c["patch_hash"] == {"vllm-gfx1201.diff": bench.sha256_file(REPO / "lab" / "vllm-gfx1201.diff")}
    assert c["rocm"] == "7.13.0-fake" and c["torch"] == "2.9.1+rocm7.13.0" and c["triton"] == "3.5.1"
    assert c["linear_kernels"] == {"CompressedTensorsWNA16": "ConchLinearKernel"}
    assert c["kv_cache_tokens"] == 163_840 and c["batch_size"] == 1 and c["node"] == "pod-b"
    # read back from the running process: the launch meant 2, the process has 4
    assert c["env"]["GPU_MAX_HW_QUEUES"] == "4" and c["env_meant"]["GPU_MAX_HW_QUEUES"] == "2"
    assert c["kfd_queues"] == 3
    assert c["prompt_set"]["id"] == "bench-v1" and c["max_tokens"] == MAX_TOKENS and c["ignore_eos"] is True
    assert c["temperature"] == 0 and c["nonce"] is True
    assert c["host"]["chip"] == "gfx1151"


def test_rows_reach_the_ledger_and_the_metrics_file_through_the_recorded_hook(tmp_path):
    config, host, power = fake_config(tmp_path, batches=(1,))
    metrics_file = tmp_path / "metrics.jsonl"
    argv = ["run", "--engine", "vllm", "--serve-cmd", json.dumps(config.command), "--checkpoint", config.checkpoint,
            "--node", "pod-b", "--port", "0", "--batches", "1", "--no-64", "--repeats", "1",
            "--max-tokens", str(MAX_TOKENS), "--requests-per-stream", "2", "--idle-s", "0.05",
            "--power-period-s", str(power.period_s), "--request-timeout-s", "20",
            "--out", str(config.out_dir), "--metrics-file", str(metrics_file), "--engine-dir", "/home/jovyan/vllm",
            "--patch", str(REPO / "lab" / "vllm-gfx1201.diff"), "--hub", "https://sjc.aupcloud.io"]
    answer = bench.main(argv, host=host, power=power)
    assert answer["ok"] is True, answer
    rows = ledger.read(state_dir=tmp_path / "state")
    runs = [r for r in rows if r["verb"] == "bench.run"]
    assert len(runs) == 1 and runs[0]["node"] == "pod-b" and runs[0]["refs"]["run_id"] == answer["rows"][0]
    assert runs[0]["metrics"]["teacher.joules_per_token"] > 0 and "teacher.watts_avg" in runs[0]["metrics"]
    assert runs[0]["conditions"]["batch_size"] == 1 and runs[0]["low_confidence"] is True
    whole = [r for r in rows if r["verb"] == "bench"]
    assert len(whole) == 1 and whole[0]["outcome"] == "ok" and "teacher.joules_per_token.b1" in whole[0]["metrics"]
    written = metrics.load(metrics_file)
    names = {r["metric"] for r in written}
    assert {"teacher.joules_per_token", "teacher.watts_avg", "teacher.watts_peak"} <= names
    for r in written:
        assert r["node"] == "pod-b" and r["conditions"]["batch_size"] == 1
        assert r["low_confidence"] is True and "spawn" in r["conditions"]["low_confidence_reason"]


def test_a_told_condition_reaches_every_row_the_ledger_and_the_metrics_file(tmp_path):
    # Two llama.cpp builds of one commit (Vulkan, ROCm) differ only in what the harness is told: --condition.
    config, host, power = fake_config(tmp_path, engine="llamacpp", batches=(1,))
    metrics_file = tmp_path / "metrics.jsonl"
    argv = ["run", "--engine", "llamacpp", "--serve-cmd", json.dumps(config.command), "--checkpoint", "q4.gguf",
            "--node", "pod-b", "--port", "0", "--batches", "1", "--no-64", "--repeats", "1",
            "--max-tokens", str(MAX_TOKENS), "--requests-per-stream", "2", "--idle-s", "0.05",
            "--power-period-s", str(power.period_s), "--request-timeout-s", "20", "--out", str(config.out_dir),
            "--metrics-file", str(metrics_file), "--condition", "backend=vulkan", "--condition", "driver=radv"]
    answer = bench.main(argv, host=host, power=power)
    assert answer["ok"] is True, answer
    row = json.loads(Path(answer["rows_path"]).read_text().splitlines()[0])
    assert row["conditions"]["backend"] == "vulkan" and row["conditions"]["driver"] == "radv"
    runs = [r for r in ledger.read(state_dir=tmp_path / "state") if r["verb"] == "bench.run"]
    assert runs[0]["conditions"]["backend"] == "vulkan"
    written = metrics.load(metrics_file)
    assert written and all(r["conditions"]["backend"] == "vulkan" and r["conditions"]["driver"] == "radv"
                           and r["conditions"]["engine"] == "llamacpp" for r in written)


def test_a_condition_the_harness_records_itself_cannot_be_told(tmp_path):
    answer = bench.main(["run", "--engine", "llamacpp", "--checkpoint", "q4.gguf", "--out", str(tmp_path / "out"),
                         "--condition", "engine=vllm"])
    assert answer["ok"] is False and "engine" in answer["error"]
    assert not (tmp_path / "out").exists()


def test_the_64_arm_is_skipped_with_its_reason_when_it_does_not_fit(tmp_path):
    _, result = run(tmp_path, batches=(1,), add_64=True, kv_tokens=4000)
    rows = result["rows"]
    assert [r["batch_size"] for r in rows] == [1, 64]
    assert rows[1]["outcome"] == "skipped" and "does not fit" in rows[1]["reason"]
    assert rows[1]["fits"]["fits"] is False and rows[1]["fits"]["kv_cache_tokens"] == 4000
    assert rows[1]["spawn"]["stopped"] is True


def test_the_same_harness_runs_unchanged_against_llamacpp(tmp_path):
    _, v = run(tmp_path / "v", batches=(1,))
    _, lc = run(tmp_path / "l", engine="llamacpp", batches=(1,))
    v, lc = v["rows"][0], lc["rows"][0]
    assert set(v) == set(lc) and set(v["conditions"]) == set(lc["conditions"])
    assert lc["outcome"] == "ok" and lc["conditions"]["engine"] == "llamacpp"
    assert lc["conditions"]["attention_backends"] == ["flash_attn enabled"]
    assert lc["conditions"]["kv_cache_tokens"] == 163_840
    assert lc["conditions"]["engine_commit"] == bench_fake.FAKE_COMMIT          # --engine-dir wins over the build line


def test_spawning_refuses_a_port_something_already_listens_on(tmp_path):
    config, host, power = fake_config(tmp_path, batches=(1,))
    with socket.socket() as squatter:
        squatter.bind(("127.0.0.1", 0))
        squatter.listen()
        config.port = squatter.getsockname()[1]
        with pytest.raises(bench.BenchError, match="already"):
            bench.run_session(config, host=host, power=power)


def test_port_0_needs_a_command_that_names_its_port(tmp_path):
    config, host, power = fake_config(tmp_path, batches=(1,))
    config.command = [a for a in config.command if a not in ("--port-file", "{port_file}")]
    with pytest.raises(bench.BenchError, match="port_file"):
        bench.run_session(config, host=host, power=power)


def test_the_nonce_keeps_measured_requests_from_hitting_the_prefix_cache(tmp_path):
    _, with_nonce = run(tmp_path / "a", batches=(1,))
    config, host, power = fake_config(tmp_path / "b", batches=(1,))
    config.prefix_caching = False                        # the harness believes caching is off: no nonce
    without = bench.run_session(config, host=host, power=power)
    assert all(r["cached_prompt_tokens"] == 0 for r in with_nonce["rows"][0]["requests"])
    assert all(r["cached_prompt_tokens"] > 0 for r in without["rows"][0]["requests"])   # the warm-up filled it


def stalling(config, *, timeout_s=1.0):
    """A run whose measured streams the fake slows past the harness's bound (grace 0, timeout_s per request)."""
    return dataclasses.replace(config, request_timeout_s=timeout_s, stream_grace_s=0.0)


def assert_stuck_row_keeps_what_ran(row, n_prompts):
    assert row["outcome"] == "fail" and "had not finished" in row["reason"]
    assert row["warmup"]["requests"] == n_prompts + row["batch_size"]          # the warm-ups that ran are listed
    assert all(item["role"] == "warmup" for item in row["warmup"]["items"])
    assert row["probe"]["request"]["role"] == "probe" and row["probe"]["decode_tok_s"] is not None
    assert row["probe"]["mem_busy_pct"]["median"] == bench_fake.MEM_BUSY_ONE


def test_a_stuck_run_on_a_shared_spawn_ends_the_spawn_and_keeps_its_warmup_and_probe(tmp_path):
    config, host, power = fake_config(tmp_path, batches=(1, 4), stall_match="b1-measure-", stall_tok_s=4.0)
    config = stalling(dataclasses.replace(config, share_spawn=True))
    result = bench.run_session(config, host=host, power=power)
    first, second = result["rows"]
    assert_stuck_row_keeps_what_ran(first, len(bench.PromptSet.load().prompts))
    assert second["batch_size"] == 4 and second["outcome"] == "fail" and second["requests"] == []
    assert "not measured" in second["reason"] and first["run_id"] in second["reason"]
    assert first["spawn"]["stopped"] is True and first["spawn"]["pid"] == second["spawn"]["pid"]


def test_a_stuck_run_while_attached_still_writes_every_row_and_the_summary(tmp_path):
    root = tmp_path / "pod"
    port_file = tmp_path / "serve.port"
    argv = bench.fill_command(bench_fake.serve_command(root=str(root), stall_match="b1-measure-", stall_tok_s=4.0),
                              {"port": 0, "port_file": port_file, "log": tmp_path / "serve.log"})
    server = subprocess.Popen(argv)
    try:
        deadline = time.monotonic() + 30.0
        while not port_file.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert port_file.exists(), "the fake never named its port"
        config = stalling(bench.Config(engine="vllm", model="teacher", node="pod-c", batches=[1, 4], add_64=False,
                                       repeats=1, max_tokens=MAX_TOKENS, requests_per_stream=2, warmup_rounds=1,
                                       idle_s=0.0, out_dir=tmp_path / "out"))
        endpoint = bench.DirectEndpoint(f"http://127.0.0.1:{port_file.read_text()}", timeout_s=config.request_timeout_s)
        result = bench.attach_session(config, endpoint=endpoint, host=bench_fake.FakeHost(root))
        endpoint.close()
    finally:
        server.terminate()
        server.wait(10)
    first, second = result["rows"]
    assert_stuck_row_keeps_what_ran(first, len(bench.PromptSet.load().prompts))
    assert second["outcome"] == "fail" and "not measured" in second["reason"]
    lines = (config.out_dir / result["session"] / "rows.jsonl").read_text().splitlines()
    assert [json.loads(line)["run_id"] for line in lines] == [first["run_id"], second["run_id"]]
    assert (config.out_dir / result["session"] / "summary.json").exists()
    assert result["summary"]["rows"] == 2


# --- attach: a server the harness did not spawn (the live smoke) ---------------------------------------------------

def test_attach_through_the_hub_proxy_marks_watts_unavailable_when_the_sampler_is_down(tmp_path):
    from lab.proxy import PodLink
    from lab.teacher_fake import FakeTeacher, text_policy

    fake = FakeTeacher(text_policy("word " * 400), user="acct-c", token="hubtoken-" + "k" * 32)
    link = PodLink.for_account(fake.account("C"), transport=fake.transport, sleep=lambda s: None)
    config = bench.Config(engine="vllm", model="teacher", node="pod-c", batches=[1], add_64=False, max_tokens=16,
                          requests_per_stream=4, warmup_rounds=0, probe=False, out_dir=tmp_path / "out",
                          hub="https://teacher.test", account="acct-c", request_timeout_s=20.0)
    result = bench.attach_session(config, endpoint=bench.ProxyEndpoint(link, 8000),
                                  power=bench.RemotePower(link, 8790))
    row = result["rows"][0]
    assert row["outcome"] == "ok" and len(row["requests"]) == 4 and row["warmup"]["requests"] == 0
    assert row["spawn"]["attached"] is True and row["spawn"]["cold"] is False and row["time_since_spawn_s"] is None
    assert row["watts"]["available"] is False and "8790" in row["watts"]["reason"]
    assert row["joules_per_token"] is None
    assert row["conditions"]["host"]["hub"] == "https://teacher.test"
    assert row["conditions"]["checkpoint"] == "RedHatAI/Qwen3.8-27B-INT4"      # what GET /v1/models names
    assert row["conditions"]["env"] is None and "attached" in row["conditions"]["unavailable"]["env"]
    body = fake.requests[-1].body
    assert body["ignore_eos"] is True and body["temperature"] == 0
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["model"] == "teacher"
    link.close()


def test_remote_power_reads_the_t6_sampler_through_the_proxy(tmp_path):
    from lab.hub import Account
    from lab.proxy import PodLink

    host = bench_fake.FakeHost(tmp_path)
    reader, _ = pod_power.choose_reader(run=host.run, which=host.which)
    sampler = pod_power.PodSampler(reader, "gpu-a", period_ms=10)
    for _ in range(3):
        sampler.poll()

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "token " + "t" * 40
        path = request.url.path
        if "/proxy/8790/status" in path:
            body = sampler.status()
        elif "/proxy/8790/samples" in path:
            body = sampler.samples_after(int(request.url.params.get("after", "0")))
        else:
            return httpx.Response(404, text="no")
        body["pod_sent"] = pod_power.iso(time.time())
        return httpx.Response(200, json=body)

    link = PodLink.for_account(Account("C", "acct-c", "t" * 40, "https://hub.test"),
                               transport=httpx.MockTransport(handle), sleep=lambda s: None)
    power = bench.RemotePower(link, 8790)
    power.start()
    assert power.available, power.reason
    for _ in range(5):
        sampler.poll()
    power.stop()
    events = power.events()
    assert len(events) == 5                                  # only the samples after start
    points = bench.watt_points(events)
    w = bench.power_window(points, points[0][0], points[-1][0], period_s=0.01)   # the samples' own span
    assert w["available"] is True and w["avg_w"] == pytest.approx(bench_fake.IDLE_NO_SERVER_W)
    link.close()
