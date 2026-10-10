"""lab/bench.py, the benchmark harness, against lab/bench_fake.py: a fake model server spawned as a real process and a
fake Strix Halo pod (recorded amd-smi output, a fake /proc and /sys). One group of tests per clause of R11's Done when:
the prompt set and decode settings, the batch sizes, the client-stream timing, the watts and joules, cold spawns and
warm-up, repeats and bands, the conditions every row carries, the ledger and metrics rows, and both engines."""
from __future__ import annotations

import base64
import json
import socket
import subprocess
import time
from pathlib import Path

import httpx
import pytest

from lab import bench, bench_fake, pod_power
from lab.machine import ledger, metrics

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "fixtures" / "lab" / "bench"
STRIX = REPO / "fixtures" / "telemetry" / "pod_power"

# The kernels rubric test 1 (K1 b) conditions, the host facts and the per-run facts every row must carry.
K1_CONDITIONS = ("engine", "engine_commit", "patch_hash", "checkpoint", "quantization", "max_model_len", "batch_size",
                 "node", "rocm", "torch", "triton", "env", "linear_kernels", "attention_backends", "kv_cache_tokens")
HOST_FACTS = ("hub", "account", "hostname", "kernel_release", "amdgpu_driver", "firmware", "asic_serial", "chip")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("LAB_TASK", raising=False)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def fake_config(tmp_path, *, engine="vllm", batches=(1, 4), repeats=1, add_64=False, set_env=("GPU_MAX_HW_QUEUES=4",),
                **fake_options) -> tuple[bench.Config, bench_fake.FakeHost]:
    root = tmp_path / "pod"
    root.mkdir(parents=True, exist_ok=True)
    host = bench_fake.FakeHost(root)
    options = {"engine": engine, "root": str(root), "set_env": list(set_env), **fake_options}
    config = bench.Config(
        engine=engine, command=bench_fake.serve_command(**options), checkpoint="/home/jovyan/models/Qwen3.8-27B-INT4",
        model="teacher", max_len=16384, node="pod-b", port=free_port(), batches=list(batches), add_64=add_64,
        repeats=repeats, max_tokens=12, requests_per_stream=2, warmup_rounds=1, idle_s=0.15, power_period_s=0.02,
        mem_busy_period_s=0.005, ready_timeout_s=30.0, out_dir=tmp_path / "out", engine_dir="/home/jovyan/vllm",
        patches=[str(REPO / "lab" / "vllm-gfx1201.diff")], launch_env={"GPU_MAX_HW_QUEUES": "2"},
        python="python3", hub="https://sjc.aupcloud.io")
    return config, host


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


# --- client-stream timing ---------------------------------------------------------------------------------------

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


# --- watts and joules -------------------------------------------------------------------------------------------

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
    power.start()
    time.sleep(0.1)
    power.stop()
    events = power.events()
    assert len(events) >= 5
    assert {e["source"] for e in events} == {"amd-smi"} and {e["power_field"] for e in events} == {"socket_power"}
    assert {e["socket_w"] for e in events} == {bench_fake.IDLE_NO_SERVER_W}
    # the parser is pod_power's: the recorded strixhalo text reads the same through it
    recorded = pod_power.parse_amdsmi((STRIX / "strixhalo_amdsmi_metric_power.txt").read_text(encoding="utf-8"))
    assert recorded[0].socket_w == 110.0


def test_throttling_is_what_amd_smi_reports(tmp_path):
    host = bench_fake.FakeHost(tmp_path, throttle="THROTTLED")
    power = bench.LocalPower.for_host(host, node="pod-b", period_s=0.01)
    power.start()
    time.sleep(0.05)
    power.stop()
    assert bench.throttled(power.events()) is True
    assert bench.throttled([{"throttled": None}]) is None      # the Strix Halo reads THROTTLE_STATUS: N/A


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
    host = bench.Host(root=tmp_path, run=lambda argv: (_ for _ in ()).throw(bench.HostError("no amd-smi")),
                      which=lambda name: None, environ={})
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


# --- statistics and bands ---------------------------------------------------------------------------------------

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


# --- whole runs: cold spawns, warm-up, rows -----------------------------------------------------------------------

def test_each_run_is_a_cold_spawn_whose_warmup_is_discarded_and_marked(tmp_path):
    config, host = fake_config(tmp_path, batches=(1, 4), repeats=2)
    result = bench.run_session(config, host=host)
    rows = result["rows"]
    assert [(r["batch_size"], r["repeat"]) for r in rows] == [(1, 0), (4, 0), (1, 1), (4, 1)]
    assert len({r["spawn"]["pid"] for r in rows}) == 4 and all(r["spawn"]["cold"] for r in rows)
    lines = (config.out_dir / result["session"] / "rows.jsonl").read_text().splitlines()
    assert [json.loads(line)["run_id"] for line in lines] == [r["run_id"] for r in rows]   # one JSON row per run
    for row in rows:
        assert row["outcome"] == "ok", row.get("reason")
        warm = row["warmup"]
        assert warm["discarded"] is True and warm["requests"] == len(bench.PromptSet.load().prompts) + row["batch_size"]
        assert len(row["requests"]) == row["batch_size"] * 2 and all(r["role"] == "measure" for r in row["requests"])
        assert row["tokens_generated"] == row["batch_size"] * 2 * 12
        assert row["time_since_spawn_s"] > 0 and row["spawn"]["ready_s"] > 0
        assert row["probe"]["in_flight"] == 1 and row["probe"]["mem_busy_pct"]["median"] == 55.0
    # nothing is left running
    for row in rows:
        assert row["spawn"]["stopped"] is True


def test_rows_carry_decode_ttft_watts_joules_and_the_idle_reading_beside_it(tmp_path):
    config, host = fake_config(tmp_path, batches=(1, 4))
    rows = bench.run_session(config, host=host)["rows"]
    one, four = rows
    for row in rows:
        assert row["decode_tok_s"] > 0 and row["decode_tok_s_total"] > 0 and row["ttft_s"] > 0
        w = row["watts"]
        assert w["available"] is True and w["source"] == "amd-smi" and w["power_field"] == "socket_power"
        assert w["peak_w"] >= w["avg_w"] > bench_fake.IDLE_LOADED_W
        assert row["joules_per_token"] == pytest.approx(w["energy_j"] / row["tokens_generated"], rel=1e-3)
        # the windows are 0.15 s long here, so an edge can interpolate towards the reading next to it
        assert row["idle_w"]["no_server"] == pytest.approx(bench_fake.IDLE_NO_SERVER_W, abs=0.5)
        assert bench_fake.IDLE_NO_SERVER_W <= row["idle_w"]["loaded"] <= bench_fake.IDLE_LOADED_W + 0.01
        assert row["throttled"] is None
    assert four["decode_tok_s_total"] > one["decode_tok_s_total"]   # batching: more in total
    assert four["watts"]["avg_w"] > one["watts"]["avg_w"]


def test_every_row_carries_the_kernels_rubric_test_1_conditions_and_the_host(tmp_path):
    config, host = fake_config(tmp_path, batches=(1,))
    row = bench.run_session(config, host=host)["rows"][0]
    c = row["conditions"]
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
    assert c["prompt_set"]["id"] == "bench-v1" and c["max_tokens"] == 12 and c["ignore_eos"] is True
    assert c["temperature"] == 0 and c["nonce"] is True
    assert c["host"]["chip"] == "gfx1151"


def test_rows_reach_the_ledger_and_the_metrics_file_through_the_recorded_hook(tmp_path):
    config, host = fake_config(tmp_path, batches=(1,))
    metrics_file = tmp_path / "metrics.jsonl"
    argv = ["run", "--engine", "vllm", "--serve-cmd", json.dumps(config.command), "--checkpoint", config.checkpoint,
            "--node", "pod-b", "--port", str(config.port), "--batches", "1", "--no-64", "--repeats", "1",
            "--max-tokens", "12", "--requests-per-stream", "2", "--idle-s", "0.15", "--power-period-s", "0.02",
            "--out", str(config.out_dir), "--metrics-file", str(metrics_file), "--engine-dir", "/home/jovyan/vllm",
            "--patch", str(REPO / "lab" / "vllm-gfx1201.diff"), "--hub", "https://sjc.aupcloud.io"]
    answer = bench.main(argv, host=host)
    assert answer["ok"] is True, answer
    rows = ledger.read(state_dir=tmp_path / "state")
    runs = [r for r in rows if r["verb"] == "bench.run"]
    assert len(runs) == 1 and runs[0]["node"] == "pod-b" and runs[0]["refs"]["run_id"] == answer["rows"][0]
    assert runs[0]["metrics"]["teacher.decode_tok_s"] > 0 and runs[0]["metrics"]["teacher.joules_per_token"] > 0
    assert runs[0]["conditions"]["batch_size"] == 1 and runs[0]["low_confidence"] is True
    whole = [r for r in rows if r["verb"] == "bench"]
    assert len(whole) == 1 and whole[0]["outcome"] == "ok" and "teacher.decode_tok_s.b1" in whole[0]["metrics"]
    written = metrics.load(metrics_file)
    names = {r["metric"] for r in written}
    assert {"teacher.decode_tok_s", "teacher.decode_tok_s_total", "teacher.ttft_s", "teacher.joules_per_token",
            "teacher.watts_avg", "teacher.watts_peak"} <= names
    for r in written:
        assert r["node"] == "pod-b" and r["conditions"]["batch_size"] == 1
        assert r["low_confidence"] is True and "spawn" in r["conditions"]["low_confidence_reason"]


def test_repeats_are_spawns_summarised_per_configuration_and_per_band(tmp_path):
    # slower tokens and a wider band gap than the defaults keep a busy test machine's jitter from splitting a band
    config, host = fake_config(tmp_path, batches=(1,), repeats=5, stream_tok_s=100, slow_factor=0.4)
    config.band_gap = 0.3
    (host.root / "band_plan.txt").write_text("fast fast slow fast slow")
    result = bench.run_session(config, host=host)
    summary = result["summary"]
    config_summary = summary["configurations"]["vllm b1"]
    assert len(config_summary["spawns"]) == 5
    s = config_summary["stats"]["decode_tok_s"]
    assert s["n"] == 5 and s["min"] <= s["median"] <= s["max"] and s["sd"] is not None
    bands = config_summary["bands"]
    assert set(bands) == {"b1", "b2"}
    assert bands["b1"]["share"] == pytest.approx(0.6) and bands["b2"]["share"] == pytest.approx(0.4)
    assert bands["b2"]["stats"]["decode_tok_s"]["max"] < bands["b1"]["stats"]["decode_tok_s"]["min"]
    rule = summary["band_rule"]
    assert 55.0 < rule["boundaries"][0]["mem_busy_threshold_pct"] < 85.0
    assert rule["boundaries"][0]["mem_busy_separates"] is True
    slow_runs = {sp["run_id"] for sp in config_summary["spawns"] if sp["band"] == "b2"}
    assert slow_runs == {result["rows"][2]["run_id"], result["rows"][4]["run_id"]}


def test_the_64_arm_is_skipped_with_its_reason_when_it_does_not_fit(tmp_path):
    config, host = fake_config(tmp_path, batches=(1,), add_64=True, kv_tokens=4000)
    rows = bench.run_session(config, host=host)["rows"]
    assert [r["batch_size"] for r in rows] == [1, 64]
    assert rows[1]["outcome"] == "skipped" and "does not fit" in rows[1]["reason"]
    assert rows[1]["spawn"]["stopped"] is True


def test_the_same_harness_runs_unchanged_against_llamacpp(tmp_path):
    vllm_config, vllm_host = fake_config(tmp_path / "v", batches=(1,))
    llama_config, llama_host = fake_config(tmp_path / "l", engine="llamacpp", batches=(1,))
    v = bench.run_session(vllm_config, host=vllm_host)["rows"][0]
    lc = bench.run_session(llama_config, host=llama_host)["rows"][0]
    assert set(v) == set(lc) and set(v["conditions"]) == set(lc["conditions"])
    assert lc["outcome"] == "ok" and lc["conditions"]["engine"] == "llamacpp"
    assert lc["conditions"]["attention_backends"] == ["flash_attn enabled"]
    assert lc["conditions"]["kv_cache_tokens"] == 163_840


def test_spawning_refuses_a_port_something_already_serves(tmp_path):
    config, host = fake_config(tmp_path, batches=(1,))
    log = tmp_path / "squatter.log"
    argv = [a.replace("{port}", str(config.port)).replace("{log}", str(log)) for a in config.command]
    squatter = subprocess.Popen(argv)
    try:
        deadline = time.time() + 20
        while time.time() < deadline and not log.exists():
            time.sleep(0.05)
        time.sleep(0.2)
        with pytest.raises(bench.BenchError, match="already"):
            bench.run_session(config, host=host)
    finally:
        squatter.terminate()
        squatter.wait(10)


def test_the_nonce_keeps_ttft_from_measuring_a_prefix_cache_hit(tmp_path):
    config, host = fake_config(tmp_path, batches=(1,), ttft_s=0.2, image_ttft_s=0.0)
    config.warmup_rounds = 1
    with_nonce = bench.run_session(config, host=host)["rows"][0]
    config.prefix_caching = False                        # the harness believes caching is off: no nonce
    config.out_dir = tmp_path / "out2"
    without = bench.run_session(config, host=host)["rows"][0]
    assert with_nonce["ttft_s"] >= 0.18                  # every measured request missed the cache
    assert without["ttft_s"] < 0.1                       # the warm-up had already filled it


# --- attach: a server the harness did not spawn (the live smoke) ---------------------------------------------------

def test_attach_through_the_hub_proxy_marks_watts_unavailable_when_the_sampler_is_down(tmp_path):
    from lab.proxy import PodLink
    from lab.teacher_fake import FakeTeacher, text_policy

    fake = FakeTeacher(text_policy("word " * 400), user="acct-c", token="hubtoken-" + "k" * 32)
    link = PodLink.for_account(fake.account("C"), transport=fake.transport, sleep=lambda s: None)
    config = bench.Config(engine="vllm", model="teacher", node="pod-c", batches=[1], add_64=False, max_tokens=16,
                          requests_per_stream=4, warmup_rounds=0, probe=False, out_dir=tmp_path / "out",
                          hub="https://teacher.test", account="acct-c")
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
    start = time.time()
    for _ in range(5):
        sampler.poll()
        time.sleep(0.01)
    end = time.time()
    power.stop()
    events = power.events()
    assert len(events) >= 5                                  # only the samples after start
    w = bench.power_window(bench.watt_points(events), start, end, period_s=0.01)
    assert w["available"] is True and w["avg_w"] == pytest.approx(bench_fake.IDLE_NO_SERVER_W)
    link.close()
