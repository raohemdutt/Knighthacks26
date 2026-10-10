"""lab/profile_decode.py and lab/profile_decode_pod.py: the decode profile's analysis, its file format, the bundle
that crosses the pod's terminal, the script the pod runs, the node gate, and a whole run against lab/bench_fake.py's
server (its torch profiler routes and its rocprofv3 stand-in). The synthetic run in
fixtures/lab/profile_decode/synthetic is made by fixtures/lab/profile_decode/make_fixture.py; its docstring gives the
numbers expected here. No GPU and no network beyond 127.0.0.1."""
from __future__ import annotations

import base64
import gzip
import io
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from lab import bench, bench_fake, pod_power
from lab import profile_decode as pd
from lab import profile_decode_pod as pod
from lab import trace_profile as tp
from lab.machine import ledger

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "fixtures" / "lab" / "profile_decode" / "synthetic"
GIB = 2**30
ARGMAX = "void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<float> > >"
HEAD = "Cijk_Alik_Bljk_BBS_BH_MT64x64x64_SN"
B1_SHARES = {"w4a16_matmul": 0.6, "attention": 0.1, "sampling": round(4 / 300, 6), "host_side": round(55 / 300, 6),
             "copies": round(1 / 300, 6), "other_gpu": 0.1, "gpu_busy": round(245 / 300, 6)}
B4_SHARES = {"w4a16_matmul": 0.59375, "attention": 0.1125, "sampling": 0.015625, "host_side": 0.175,
             "copies": 0.003125, "other_gpu": 0.1, "gpu_busy": 0.825}
ROCMINFO_GFX1151 = "Agent 2\n  Name:                    gfx1151\n  Marketing Name:          AMD Radeon 8060S\n"


def raw() -> dict:
    return json.loads((FIX / "raw.json").read_text(encoding="utf-8"))


def synthetic_result(reference: bool = True) -> dict:
    result = pd.build_result(raw(), FIX)
    if reference:
        pd.apply_reference(result, pd.chip_reference(FIX / "chipsheet.json"))
    return result


# --- the torch trace ------------------------------------------------------------------------------------------------

def test_decode_steps_come_from_vllm_step_ranges():
    trace = tp.TorchTrace.load(FIX / "torch" / "b1.pt.trace.json.gz")
    steps = pd.torch_decode_steps(trace, 1)
    assert [(s.start_us, s.end_us, s.forward_end_us) for s in steps] == [
        (1_100_000 + k * 300_000, 1_400_000 + k * 300_000, 1_140_000 + k * 300_000) for k in range(5)]
    assert pd.torch_decode_steps(trace, 4) == []
    assert pd.step_kind("execute_context_0(0)_generation_4(4)") == (0, 4)
    assert pd.step_kind("execute_context_1(31)_generation_0(0)") == (1, 0)
    assert pd.step_kind("aten::mm") is None


def test_torch_batch_1_step_time_shares_and_kernels():
    a = pd.analyze_torch_decode(FIX / "torch" / "b1.pt.trace.json.gz", 1, pd.categorizer_for(raw()))
    assert a["available"] is True and a["steps"] == 5
    assert (a["step_ms"]["n"], a["step_ms"]["mean"], a["step_ms"]["sd"]) == (5, 300.0, 0.0)
    assert a["shares"] == B1_SHARES
    assert a["ms_per_step"] == {"w4a16_matmul": 180.0, "attention": 30.0, "sampling": 4.0, "host_side": 55.0,
                                "copies": 1.0, "other_gpu": 30.0, "gpu_busy": 245.0, "step": 300.0}
    assert a["attention_split_ms"] == {"full": 20.0, "linear": 10.0}
    assert a["sampling_launched_after_forward_ms"] == 4.0
    assert a["attribution"] == {"kernels": {"by_launch": 85, "by_gpu_time": 0, "outside": 27},
                                "copies": {"by_launch": 5, "by_gpu_time": 0, "outside": 1}}
    assert a["kernel_coverage"] == 0.8167
    assert (a["host"]["forward_cpu_ms"]["mean"], a["host"]["after_forward_cpu_ms"]["mean"]) == (40.0, 260.0)
    assert (a["host"]["runtime_calls_per_step"], a["host"]["graph_launches_per_step"],
            a["host"]["kernel_launch_calls_per_step"]) == (3.0, 1.0, 1.0)
    assert [r["name"] for r in a["kernels_top"]] == ["_gemm_kernel", HEAD, "kernel_unified_attention_2d",
                                                     "fused_recurrent_gated_delta_rule_fwd_kernel",
                                                     "triton_poi_fused_add_rms_norm_0", ARGMAX]
    top = a["kernels_top"][0]
    assert (top["category"], top["launches"], top["launches_per_step"], top["ms_per_step"], top["share_of_step"],
            top["grid"], top["block"]) == ("w4a16_matmul", 50, 10.0, 180.0, 0.6, [80, 1, 1], [256, 1, 1])
    assert (a["copies"]["count"], a["copies"]["ms_per_step"], a["copies"]["bytes_per_step"]) == (5, 1.0, 4.0)
    assert a["memory"]["decode_steps"]["peak_allocated_bytes"] == 20 * GIB + 3_000_000
    assert a["uncategorized_top"] == []


def test_torch_batch_4_shares():
    a = pd.analyze_torch_decode(FIX / "torch" / "b4.pt.trace.json.gz", 4, pd.categorizer_for(raw()))
    assert (a["steps"], a["step_ms"]["mean"]) == (5, 320.0)
    assert a["shares"] == B4_SHARES


def test_a_trace_without_decode_ranges_is_reported_not_guessed(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"traceEvents": [{"ph": "X", "cat": "kernel", "name": "k", "ts": 1, "dur": 2}]}),
                    encoding="utf-8")
    a = pd.analyze_torch_decode(path, 1, tp.Categorizer())
    assert a["available"] is False and a["steps"] == 0
    assert "execute_context_0(0)_generation_1(1)" in a["reason"]


# --- the rocprof trace ----------------------------------------------------------------------------------------------

def rocprof_inputs():
    r = raw()
    kernels = tp.read_rocprof_kernels(FIX / "rocprof" / "4243" / "run_kernel_trace.csv")
    copies = tp.read_rocprof_copies(FIX / "rocprof" / "4243" / "run_memory_copy_trace.csv")
    return r, kernels, copies


def test_the_clock_is_the_one_that_puts_kernels_in_the_decode_windows():
    r, kernels, _ = rocprof_inputs()
    windows = [w for run in r["phases"]["rocprof"]["runs"].values() for w in pd.run_windows(run)]
    clock, counts = pd.choose_clock(kernels, r["clocks_start"], windows)
    assert clock == "boottime"
    assert counts == {"boottime": 170, "monotonic": 0, "realtime": 0}
    moved = {"realtime": r["clocks_start"]["realtime"], "boottime": r["clocks_start"]["boottime"] + 10**13,
             "monotonic": r["clocks_start"]["boottime"]}           # now the kernels sit on the monotonic reading
    assert pd.choose_clock(kernels, moved, windows) == ("monotonic", {"boottime": 0, "monotonic": 170, "realtime": 0})
    assert pd.choose_clock(kernels, {"realtime": 0}, windows) == (None, {"realtime": 0})


def test_rocprof_batch_1_steps_marker_and_shares():
    r, kernels, copies = rocprof_inputs()
    a = pd.analyze_rocprof_decode(kernels, copies, r["phases"]["rocprof"]["runs"]["b1_rocprof"],
                                  offsets=r["clocks_start"], clock="boottime", categorize=pd.categorizer_for(r))
    assert a["available"] is True and (a["steps"], a["windows"]) == (5, 1)
    assert a["step_markers"] == [{"name": HEAD, "launches": 5, "cv": 0.0}]
    assert (a["step_ms"]["n"], a["step_ms"]["mean"]) == (4, 300.0)
    assert a["shares"] == pytest.approx(B1_SHARES, abs=1e-5)
    top = a["kernels_top"][0]
    assert (top["name"], top["launches"], top["grid"], top["block"], top["lds_bytes"], top["vgprs"]) == (
        "_gemm_kernel", 50, [80, 1, 1], [256, 1, 1], 49152, 128)


def test_rocprof_batch_4_shares():
    r, kernels, copies = rocprof_inputs()
    a = pd.analyze_rocprof_decode(kernels, copies, r["phases"]["rocprof"]["runs"]["b4_rocprof"],
                                  offsets=r["clocks_start"], clock="boottime", categorize=pd.categorizer_for(r))
    assert (a["steps"], a["step_ms"]["mean"]) == (5, 320.0)
    assert a["shares"] == pytest.approx(B4_SHARES, abs=1e-5)


def test_rocprof_on_the_wrong_clock_finds_nothing_and_says_so():
    r, kernels, copies = rocprof_inputs()
    a = pd.analyze_rocprof_decode(kernels, copies, r["phases"]["rocprof"]["runs"]["b1_rocprof"],
                                  offsets=r["clocks_start"], clock="monotonic", categorize=tp.Categorizer())
    assert a["available"] is False and "monotonic" in a["reason"]


# --- the client's token stream --------------------------------------------------------------------------------------

def test_client_steps_from_the_gaps_between_tokens():
    runs = raw()["phases"]["torch"]["runs"]
    b1 = pd.client_steps(runs["b1_baseline"])
    assert b1["step_ms"] == pytest.approx({"n": 10, "mean": 300.0, "sd": 5.354, "min": 290.0, "p10": 294.5,
                                           "p50": 300.0, "p90": 305.5, "p99": 309.55, "max": 310.0}, abs=2e-3)
    assert (b1["steps"], b1["step_ms_from_windows"], b1["tok_s"]) == (10, 300.0, 3.333)
    b4 = pd.client_steps(runs["b4_baseline"])
    assert (b4["step_ms"]["n"], b4["steps"], b4["tok_s"]) == (20, 5, 12.5)
    assert b4["step_ms"]["mean"] == pytest.approx(320.0, abs=1e-3)


def test_run_windows_per_request_at_batch_1_and_shared_at_batch_4():
    run = {"batch": 4, "requests": [
        {"first_t": 10.0, "last_t": 12.0, "tokens": 3, "token_times": [10.0, 11.0, 12.0]},
        {"first_t": 10.5, "last_t": 11.5, "tokens": 3, "token_times": [10.5, 11.0, 11.5]},
        {"first_t": 10.0, "last_t": 12.0, "tokens": 3, "token_times": [10.0, 11.0, 12.0]},
        {"first_t": 10.0, "last_t": 12.0, "tokens": 3, "token_times": [10.0, 11.0, 12.0]}]}
    assert pd.run_windows(run) == [(10.5, 11.5, 1)]
    run["batch"] = 1
    assert pd.run_windows(run) == [(10.0, 12.0, 2), (10.5, 11.5, 2), (10.0, 12.0, 2), (10.0, 12.0, 2)]
    run["requests"][0]["error"] = "boom"
    assert len(pd.run_windows(run)) == 3


def test_decode_windows_of_the_pod_record():
    requests = [{"first_t": 1.0, "last_t": 2.0, "error": None}, {"first_t": 1.2, "last_t": 1.8, "error": None}]
    assert pod.decode_windows(requests, 1) == [[1.0, 2.0], [1.2, 1.8]]
    assert pod.decode_windows(requests, 2) == [[1.2, 1.8]]
    assert pod.decode_windows(requests, 4) == []


# --- the result -----------------------------------------------------------------------------------------------------

def test_the_result_of_the_synthetic_run():
    result = synthetic_result()
    assert pd.check_result(result) == []
    b1, b4 = result["batches"]["1"], result["batches"]["4"]
    assert (b1["breakdown_source"], b1["kernels_top_source"]) == ("torch", "rocprof")
    assert b1["shares"] == B1_SHARES and b4["shares"] == B4_SHARES
    assert b1["bandwidth"] == {
        "bytes_per_step": 15_756_000_000, "quantized_bytes_per_step": 13_000_000_000, "achieved_gb_s_step": 52.52,
        "achieved_gb_s_gpu_busy": 64.31, "achieved_gb_s_w4a16": 72.222, "counts": b1["bandwidth"]["counts"],
        "reference_gb_s": 212.5, "fraction_of_reference_step": 0.2472, "fraction_of_reference_gpu_busy": 0.3026,
        "fraction_of_reference_w4a16": 0.3399}
    quant = result["quant_kernel"]
    assert (quant["vllm_class"], quant["vllm_class_source"], quant["conch_version"]) == (
        "ConchLinearKernel", "vllm/model_executor/kernels/linear/mixed_precision/conch.py", "1.2.1")
    assert [(k["name"], k["source"], sorted(k["seen_in"])) for k in quant["gpu_kernels"]] == [
        ("_gemm_kernel", "conch/kernels/quantization/gemm.py", ["rocprof_b1", "rocprof_b4", "torch_b1", "torch_b4"])]
    assert quant["evidence"][0] == "serve log: Using ConchLinearKernel for CompressedTensorsWNA16"
    assert result["memory"]["torch_profiler"] == {"peak_allocated_gib": 20.005, "peak_reserved_gib": 22.5,
                                                  "samples": 6}
    assert result["memory"]["serve_log"] == {"weights_gib": 17.98, "kv_cache_gib": 31.25, "graph_pool_gib": 0.62,
                                             "kv_cache_tokens": 163840}
    assert (result["tools"]["rocprof"]["clock"], result["tools"]["rocprof"]["kernels_read"]) == ("boottime", 173)
    assert result["open"] == []
    assert (result["date"], result["node"], result["chip"]) == ("2026-10-10", "pod-b", "gfx1151")


def test_without_a_chip_sheet_the_comparison_stays_open():
    result = synthetic_result(reference=False)
    pd.apply_reference(result, pd.chip_reference(None))
    assert result["batches"]["1"]["bandwidth"]["fraction_of_reference_step"] is None
    assert result["open"] == ["chip sheet: no chip sheet (chipsheet-gfx1151-*.json) on main when the result was "
                              "collected (R33)"]
    pd.apply_reference(result, pd.chip_reference(FIX / "chipsheet.json"))
    assert result["open"] == []


def test_a_failed_tool_and_a_trace_that_misses_graph_kernels_are_open_items():
    r = raw()
    r["phases"]["rocprof"] = {"kind": "rocprof", "ok": False, "error": "rocprofv3 is not on PATH", "runs": {}}
    result = pd.build_result(r, FIX)
    assert "rocprof: rocprofv3 is not on PATH" in result["open"]
    assert result["batches"]["1"]["kernels_top_source"] == "torch"
    low = {"available": True, "kernel_coverage": 0.1}
    assert pd._primary(low, None) == "torch"
    assert pd._primary(low, {"available": True}) == "rocprof"
    assert pd._primary({"available": True, "kernel_coverage": 0.8}, {"available": True}) == "torch"


def test_chip_sheet_reference_by_name_or_by_key():
    found = pd.chip_reference(FIX / "chipsheet.json")
    assert (found["key"], found["gb_s"]) == ("bandwidth.d2d_copy.max_gb_s", 212.5)
    assert pd.chip_reference(FIX / "chipsheet.json", "peak_theoretical_gb_s")["gb_s"] == 256.0
    bad = pd.chip_reference(FIX / "chipsheet.json", "bandwidth.nope")
    assert bad["gb_s"] is None and "not a number" in bad["note"]


def test_the_checks_catch_a_broken_result():
    good = synthetic_result()
    assert pd.check_result(good) == []
    broken = json.loads(json.dumps(good))
    del broken["batches"]["4"]
    broken["batches"]["1"]["shares"]["w4a16_matmul"] = 2.0
    rows = broken["batches"]["1"]["kernels_top"]
    rows[0], rows[1] = rows[1], rows[0]
    rows[2]["grid"] = [1, 2]
    broken["schema"] = "x"
    problems = pd.check_result(broken)
    assert "schema must be 'lab.profile_decode/1', not 'x'" in problems
    assert "batches.4 is missing" in problems
    assert "batches.1.shares.w4a16_matmul must be a number from 0 to 1.5, not 2.0" in problems
    assert "batches.1.kernels_top[1].rank must be 1" in problems
    assert "batches.1.kernels_top must be sorted by total_ms, biggest first" in problems
    assert "batches.1.kernels_top[3] grid and block must be [x, y, z] or null" in problems
    too_many = json.loads(json.dumps(good))
    too_many["batches"]["1"]["kernels_top"] = [dict(rows[0], rank=i) for i in range(1, 27)]
    assert "batches.1.kernels_top has 26 rows; at most 25" in pd.check_result(too_many)
    assert pd.check_result([]) == ["the result is not a JSON object"]


def test_the_files_are_named_by_date_hold_the_result_and_are_not_overwritten(tmp_path):
    result = synthetic_result()
    json_path, md_path = pd.write_outputs(result, tmp_path)
    assert (json_path.name, md_path.name) == ("decode-profile-gfx1151-2026-10-10.json",
                                              "decode-profile-gfx1151-2026-10-10.md")
    assert json.loads(json_path.read_text(encoding="utf-8")) == json.loads(json.dumps(result))
    text = md_path.read_text(encoding="utf-8")
    assert all(section in text for section in pd.MD_SECTIONS)
    assert "| 1 | torch (used) | 300.0 | 60.0% | 10.0% | 1.3% | 18.3% | 0.3% | 10.0% |" in text
    assert "| 1 | `_gemm_kernel` | w4a16_matmul | 50 | 10.0 | 180.000 | 60.0% | 80x1x1 | 256x1x1 |" in text
    assert "vLLM chose `ConchLinearKernel` (vllm/model_executor/kernels/linear/mixed_precision/conch.py)" in text
    with pytest.raises(pd.ProfileError, match="already exists"):
        pd.write_outputs(result, tmp_path)
    pd.write_outputs(result, tmp_path, force=True)
    broken = dict(result, schema="x")
    with pytest.raises(pd.ProfileError, match="fails its checks"):
        pd.write_outputs(broken, tmp_path / "other")
    assert not (tmp_path / "other").exists()


# --- the bundle -----------------------------------------------------------------------------------------------------

def test_the_bundle_survives_a_terminal_and_a_windows_shell():
    files = {"result.json": b'{"a": 1}', "logs/serve-torch.log": b"line\n" * 50}
    payload = pd.make_bundle(files)
    assert payload == pd.make_bundle(dict(reversed(list(files.items()))))      # same bytes, whatever the order
    text = "echo of the typed command\r\n" + pd.bundle_text(payload).replace("\n", "\r\n") + "### END rc=0\r\n"
    assert pd.read_bundle(text.encode("utf-8")) == files
    assert pd.read_bundle(b"\xef\xbb\xbf" + text.encode("utf-8")) == files
    assert pd.read_bundle(text.encode("utf-16")) == files
    assert pd.read_bundle(payload) == files
    older = pd.bundle_text(pd.make_bundle({"result.json": b"old"}))
    assert pd.read_bundle((older + text).encode()) == files                      # the last block wins


def test_a_damaged_or_missing_bundle_is_refused():
    text = pd.bundle_text(pd.make_bundle({"result.json": b"{}"}))
    lines = text.splitlines()
    lines[1] = ("B" if lines[1][0] != "B" else "C") + lines[1][1:]
    with pytest.raises(pd.ProfileError, match="damaged in transit"):
        pd.read_bundle("\n".join(lines).encode())
    with pytest.raises(pd.ProfileError, match="no '=== R47-RESULT BEGIN' line"):
        pd.read_bundle(b"the pod run was cut short")
    with pytest.raises(pd.ProfileError, match="no end line"):
        pd.read_bundle(text.split(pd.MARK_END)[0].encode())


def test_the_pod_bundle_carries_the_result_raw_logs_and_trace_excerpts():
    result = synthetic_result()
    files = pd.pod_bundle(raw(), result, FIX)
    assert sorted(files) == ["excerpts/rocprof_b1_kernel_trace.csv", "excerpts/rocprof_window.json",
                             "excerpts/torch_b1.pt.trace.json.gz", "raw.json", "result.json", "summary_pod.md"]
    excerpt = tp.TorchTrace(json.loads(gzip.decompress(files["excerpts/torch_b1.pt.trace.json.gz"])))
    assert len(pd.torch_decode_steps(excerpt, 1)) == 2
    window = json.loads(files["excerpts/rocprof_window.json"])
    assert (window["clock"], window["steps"]) == ("boottime", 2)
    rows = list(io.StringIO(files["excerpts/rocprof_b1_kernel_trace.csv"].decode()))
    assert len(rows) == 1 + 2 * 17


# --- the script the pod runs ----------------------------------------------------------------------------------------

def unpack_script(script: str) -> dict[str, bytes]:
    lines = script.splitlines()
    start = lines.index("base64 -d > \"$B/bundle.tar.gz\" <<'R47BUNDLE'") + 1
    body = "".join(lines[start:lines.index("R47BUNDLE")])
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(base64.b64decode(body)))) as tar:
        return {m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}


def test_the_pod_script_can_be_typed_into_a_shell_and_carries_every_file_it_imports():
    script, info = pd.pod_script(expect_user_sha256="ab" * 32, args=["--phases", "torch,rocprof"])
    assert "!" not in script and "\t" not in script
    assert max(len(line) for line in script.splitlines()) < 300
    lines = script.splitlines()
    assert lines[-1] == 'echo "### END rc=$rc"' or lines[-2] == 'echo "### END rc=$rc"'
    assert f"python3 -m lab.profile_decode pod-run --expect-user-sha256 {'ab' * 32} --phases torch,rocprof" in script
    files = unpack_script(script)
    expected = {str(p.relative_to(REPO)).replace("\\", "/") for p in (*pd.module_closure(), *pd.data_files())}
    assert set(files) == expected == set(info["files"])
    for name in ("lab/profile_decode.py", "lab/profile_decode_pod.py", "lab/trace_profile.py", "lab/bench.py",
                 "lab/pod_power.py", "lab/machine/record.py", "fixtures/lab/bench/prompts.json",
                 "fixtures/lab/bench/images/bars_1280x720.png"):
        assert files[name] == (REPO / name).read_bytes()


def test_the_unpacked_bundle_imports_on_its_own(tmp_path):
    script, _ = pd.pod_script(expect_user_sha256="ab" * 32, args=[])
    for name, data in unpack_script(script).items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    done = subprocess.run([sys.executable, "-c", "import lab.profile_decode as m, lab.profile_decode_pod, lab.bench; "
                                                 "print(m.__file__); print(m.data_files()[0])"],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    module, prompts = done.stdout.strip().splitlines()
    assert Path(module).resolve().is_relative_to(tmp_path.resolve())
    assert Path(prompts).resolve().is_relative_to(tmp_path.resolve())


def test_pod_script_main_writes_the_script_with_the_account_hash(tmp_path):
    out = tmp_path / "pod_run.sh"
    args = pd.build_parser().parse_args(["pod-script", "--out", str(out), "--expect-user-sha256", "cd" * 32,
                                         "--max-tokens", "32", "--phases", "torch"])
    answer = pd.pod_script_main(args)
    assert answer["ok"] is True and answer["script"] == str(out)
    text = out.read_text(encoding="ascii")
    assert f"--expect-user-sha256 {'cd' * 32} --phases torch --max-tokens 32 --port 8100" in text
    assert "--deadline-min 30 --hub sjc --account D" in text
    with pytest.raises(SystemExit):
        pd.build_parser().parse_args(["pod-script", "--phases", "torch,nvprof"])


# --- the pod side ---------------------------------------------------------------------------------------------------

def test_the_gate_admits_only_pod_d_on_gfx1151_with_the_port_free():
    want = pod.user_sha256("account-d")
    ok = dict(expect_user_sha256=want, rocminfo=lambda: ROCMINFO_GFX1151, port=8100, port_busy=lambda p: False)
    assert pod.gate(environ={"JUPYTERHUB_USER": "account-d"}, **ok) == []
    assert pod.gate(environ={"JUPYTERHUB_SERVICE_PREFIX": "/user/account-d/"}, **ok) == []
    assert pod.gate(environ={"JUPYTERHUB_USER": "account-c"}, **ok) == [
        "this pod's account is not the one the script was made for (pod D); pod C serves the live teacher and is "
        "never profiled"]
    assert pod.gate(environ={}, **ok) == ["JUPYTERHUB_USER is not set: cannot tell which pod this is"]
    r9700 = dict(ok, rocminfo=lambda: "Name: gfx1201\n")
    assert pod.gate(environ={"JUPYTERHUB_USER": "account-d"}, **r9700) == [
        "rocminfo names no gfx1151: this is not a Strix Halo pod"]
    busy = dict(ok, port_busy=lambda p: True)
    assert pod.gate(environ={"JUPYTERHUB_USER": "account-d"}, **busy) == [
        "something already listens on 127.0.0.1:8100"]
    assert pod.gate(environ={"JUPYTERHUB_USER": "account-d"}, **dict(ok, expect_user_sha256=None))[0].startswith(
        "no --expect-user-sha256")


def write_safetensors(path: Path, tensors: dict[str, int]) -> None:
    header, offset = {}, 0
    for name, size in tensors.items():
        header[name] = {"dtype": "U8", "shape": [size], "data_offsets": [offset, offset + size]}
        offset += size
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + bytes(offset))


def test_weight_bytes_count_what_a_decode_step_reads(tmp_path):
    write_safetensors(tmp_path / "model-1.safetensors", {
        "model.layers.0.mlp.down_proj.weight_packed": 1000, "model.layers.0.mlp.down_proj.weight_scale": 100,
        "model.layers.0.input_layernorm.weight": 10, "model.embed_tokens.weight": 500})
    write_safetensors(tmp_path / "model-2.safetensors", {
        "lm_head.weight": 500, "model.visual.blocks.0.attn.qkv.weight": 300, "mtp.layers.0.weight": 70})
    counted = pod.weight_bytes(tmp_path)
    assert counted == {"files": 2, "tensors": 7, "groups": {"dense": 10, "embedding": 500, "lm_head": 500, "mtp": 70,
                                                            "quantized": 1100, "vision": 300},
                       "total": 2480, "decode_read": 1610, "quantized_read": 1100, "tied_embedding": False}
    (tmp_path / "model-2.safetensors").unlink()
    write_safetensors(tmp_path / "model-2.safetensors", {"model.visual.patch_embed.weight": 300})
    tied = pod.weight_bytes(tmp_path)
    assert (tied["decode_read"], tied["tied_embedding"]) == (1610, True)
    with pytest.raises(pod.PodRunError, match="no .safetensors"):
        pod.weight_bytes(tmp_path / "missing")


def test_memory_lines_of_the_serve_log():
    text = ("(EngineCore_DP0 pid=7) INFO [gpu_model_runner.py:4821] Model loading took 17.98 GiB memory and 4.1 s\n"
            "(EngineCore_DP0 pid=7) INFO [gpu_worker.py:437] Available KV cache memory: 31.25 GiB\n"
            "(EngineCore_DP0 pid=7) INFO [gpu_model_runner.py:5900] Graph capturing finished in 30 secs, "
            "took 0.62 GiB\n")
    assert pod.memory_from_log(text) == {"weights_gib": 17.98, "kv_cache_gib": 31.25, "graph_pool_gib": 0.62}
    assert pod.memory_from_log("") == {"weights_gib": None, "kv_cache_gib": None, "graph_pool_gib": None}


def fake_proc(root: Path, pid: int, ppid: int, state: str, cmdline: str, rss_kib: int = 0) -> None:
    folder = root / "proc" / str(pid)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "stat").write_text(f"{pid} (python3 x) {state} {ppid} 1 1 0", encoding="utf-8")
    (folder / "cmdline").write_bytes(cmdline.replace(" ", "\0").encode() + b"\0")
    (folder / "status").write_text(f"Name: python3\nVmRSS:\t{rss_kib} kB\n", encoding="utf-8")


def test_the_engine_core_and_the_api_server_are_found_under_rocprofv3(tmp_path):
    fake_proc(tmp_path, 100, 1, "S", "python3 /opt/rocm/bin/rocprofv3 --kernel-trace -- python3 -m "
                                     "vllm.entrypoints.openai.api_server --model m")
    fake_proc(tmp_path, 101, 100, "S", "python3 -m vllm.entrypoints.openai.api_server --model m")
    fake_proc(tmp_path, 102, 101, "S", "VLLM::EngineCore_DP0")
    fake_proc(tmp_path, 103, 101, "S", "python3 -c from multiprocessing.resource_tracker import main;main(5)")
    fake_proc(tmp_path, 900, 1, "S", "bash")
    host = bench.Host(root=tmp_path)
    table = pod.proc_table(host)
    assert pod.descendants(table, 100) == [100, 101, 102, 103]
    assert pod.find_engine_core(table, 100) == (102, 101)


def test_without_a_title_the_engine_core_is_the_biggest_spawned_child(tmp_path):
    fake_proc(tmp_path, 200, 1, "S", "python3 -m vllm.entrypoints.openai.api_server")
    fake_proc(tmp_path, 201, 200, "S", "python3 -c from multiprocessing.spawn import spawn_main; spawn_main(7)", 900)
    fake_proc(tmp_path, 202, 200, "S", "python3 -c from multiprocessing.spawn import spawn_main; spawn_main(8)", 90000)
    fake_proc(tmp_path, 203, 200, "S", "python3 -c from multiprocessing.resource_tracker import main;main(5) "
                                       "spawn_main", 999999)
    host = bench.Host(root=tmp_path)
    assert pod.find_engine_core(pod.proc_table(host), 200, rss=lambda p: pod.rss_kib(host, p)) == (202, 200)


class FakeProc:
    def __init__(self) -> None:
        self.code = None

    def poll(self):
        return self.code


class FakeSpawned:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.proc = FakeProc()
        self.stopped_with = None

    def stop(self, timeout_s: float = 30.0) -> bool:
        self.stopped_with = timeout_s
        return True


def test_a_traced_server_is_stopped_engine_core_first_with_the_api_server_paused(tmp_path):
    fake_proc(tmp_path, 101, 1, "S", "python3 -m vllm.entrypoints.openai.api_server")
    fake_proc(tmp_path, 102, 101, "S", "VLLM::EngineCore_DP0")
    host = bench.Host(root=tmp_path)
    spawned = FakeSpawned(101)
    sent = []

    def kill(pid: int, sig: int) -> None:
        sent.append((pid, sig))
        if (pid, sig) == (102, signal.SIGTERM):
            fake_proc(tmp_path, 102, 101, "Z", "VLLM::EngineCore_DP0")    # it wrote its trace and exited
        if (pid, sig) == (101, signal.SIGTERM):
            spawned.proc.code = 0

    out = pod.stop_traced(spawned, host, kill=kill, sleep=lambda s: None, core_timeout_s=5, server_timeout_s=5)
    assert sent == [(101, pod.SIGSTOP), (102, signal.SIGTERM), (101, pod.SIGCONT), (101, signal.SIGTERM)]
    assert (out["core"], out["api"], out["forced"]) == (102, 101, False)
    assert spawned.stopped_with == 30.0


def test_a_traced_server_without_a_visible_engine_core_is_stopped_at_its_root(tmp_path):
    spawned = FakeSpawned(555)
    sent = []

    def kill(pid: int, sig: int) -> None:
        sent.append((pid, sig))
        spawned.proc.code = 0

    out = pod.stop_traced(spawned, bench.Host(root=tmp_path), kill=kill, sleep=lambda s: None,
                          core_timeout_s=1, server_timeout_s=1)
    assert sent == [(555, signal.SIGTERM)]
    assert (out["core"], out["api"]) == (None, 555)


def test_the_jupyter_user_comes_from_the_prefix_when_the_variable_is_missing():
    assert pod.jupyter_user({"JUPYTERHUB_USER": " u1 "}) == "u1"
    assert pod.jupyter_user({"JUPYTERHUB_SERVICE_PREFIX": "/user/u2/"}) == "u2"
    assert pod.jupyter_user({}) is None


# --- a whole run against the fake server, then collect ----------------------------------------------------------------

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_a_whole_run_on_the_fake_pod_then_collect_writes_the_files_and_the_ledger(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pod, "TRACE_POLL_S", 0.05)
    root, ckpt = tmp_path / "root", tmp_path / "ckpt"
    root.mkdir()
    ckpt.mkdir()
    write_safetensors(ckpt / "model.safetensors", {"model.layers.0.mlp.weight_packed": 64, "lm_head.weight": 32})
    args = pd.build_parser().parse_args([
        "pod-run", "--expect-user-sha256", "unused-here", "--port", str(free_port()), "--checkpoint", str(ckpt),
        "--engine-dir", str(tmp_path), "--out", str(tmp_path / "out"), "--serve-dir", str(tmp_path / "serve"),
        "--max-tokens", "6", "--idle-s", "0.02", "--prompts", str(bench.DEFAULT_PROMPTS),
        "--serve-cmd", json.dumps(bench_fake.serve_command(root=str(root), stream_tok_s=200.0)),
        "--rocprof-cmd", json.dumps(bench_fake.rocprof_command())])
    run = pod.PodRun(args, host=bench_fake.FakeHost(root), power=bench_fake.FakePower(root))
    answer = pd.run_and_report(run)
    assert answer["ok"] is True and answer["analysis_error"] is None
    assert answer["phases"] == {"torch": {"ok": True, "error": None}, "rocprof": {"ok": True, "error": None}}
    out = capsys.readouterr().out
    saved = tmp_path / "pod_output.txt"
    saved.write_text(out, encoding="utf-8")
    result = json.loads((run.dir / "result.json").read_text(encoding="utf-8"))
    assert pd.check_result(result) == []
    assert len(result["tools"]["torch_profiler"]["traces"]) == 2
    assert result["tools"]["rocprof"]["clock"] == "realtime" and result["tools"]["rocprof"]["kernels_read"] > 0
    for batch in ("1", "4"):
        entry = result["batches"][batch]
        assert entry["available"] is True
        assert entry["sources"]["torch"]["available"] is True and entry["sources"]["rocprof"]["available"] is True
        assert entry["kernels_top"][0]["name"] == "_gemm_kernel"
        assert entry["step_ms"]["client_unprofiled"]["n"] > 0
    assert result["checkpoint"]["bytes"]["decode_read"] == 96
    raw_record = json.loads((run.dir / "raw.json").read_text(encoding="utf-8"))
    assert raw_record["phases"]["torch"]["server"]["profiler_config"]["profiler"] == "torch"
    assert raw_record["phases"]["rocprof"]["server"]["env"] == {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"}
    assert not bench.port_open(args.port)                              # both servers are gone

    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LAB_ACTOR", "test")
    collected = pd.main(["collect", str(saved), "--out", str(tmp_path / "data"),
                         "--chip-sheet", str(FIX / "chipsheet.json")])
    assert collected["ok"] is True and len(collected["ledger_seqs"]) == 5
    assert Path(collected["json"]).name == f"decode-profile-gfx1151-{result['date']}.json"
    rows = ledger.read(state_dir=tmp_path / "state")
    assert [r["verb"] for r in rows] == ["profile_decode.batch", "profile_decode.batch",
                                         "profile_decode.torch_profiler", "profile_decode.rocprof", "profile_decode"]
    assert {r["node"] for r in rows} == {"pod-b"} and {r["task"] for r in rows} == {"R47"}
    assert {r["refs"]["run_id"] for r in rows} == {run.run_id}
    assert rows[0]["conditions"]["batch"] == 1 and rows[1]["conditions"]["batch"] == 4
    assert rows[0]["units"]["teacher.decode_step_ms"] == "ms"
    assert rows[0]["units"]["teacher.decode_share_w4a16_matmul"] == "fraction"
    again = pd.main(["collect", str(saved), "--out", str(tmp_path / "data"), "--no-ledger"])
    assert again["ok"] is False and "already exists" in again["error"]


def test_the_quick_self_check_passes_on_the_fixture(tmp_path):
    answer = pd.main(["--quick", "--out", str(tmp_path)])
    assert answer["ok"] is True, answer
    assert all(answer["checks"].values())
    assert sorted(p.name for p in tmp_path.iterdir()) == ["decode-profile-gfx1151-2026-10-10.json",
                                                          "decode-profile-gfx1151-2026-10-10.md"]


def test_ledger_rows_of_the_synthetic_result():
    rows = pd.ledger_rows(synthetic_result(), "lab/refine/research/kernels/data/x.json")
    assert [(r["verb"], r["outcome"]) for r in rows] == [
        ("profile_decode.batch", "ok"), ("profile_decode.batch", "ok"), ("profile_decode.torch_profiler", "ok"),
        ("profile_decode.rocprof", "ok"), ("profile_decode", "ok")]
    first = rows[0]
    assert first["metrics"]["teacher.decode_step_ms"] == pytest.approx(300.0, abs=1e-3)
    assert first["metrics"]["teacher.decode_share_w4a16_matmul"] == 0.6
    assert first["metrics"]["teacher.decode_achieved_gb_s"] == 52.52
    assert first["metrics"]["teacher.decode_watts_avg"] == 118.2
    assert first["units"]["teacher.decode_tok_s"] == "tok/s" and first["units"]["teacher.decode_watts_avg"] == "W"
    assert (first["node"], first["low_confidence"], first["refs"]) == ("pod-b", True, {"run_id": "r47-synthetic"})
    assert rows[-1]["metrics"] == {"teacher.decode_peak_allocated_gib": 20.005,
                                   "teacher.decode_peak_reserved_gib": 22.5}
    for row in rows:
        assert ledger.validate({"ts": "2026-10-10T23:00:00Z", "actor": "t", "tier": None, "task": "R47", **row})


def test_the_pod_power_gate_is_the_one_the_run_uses():
    assert pod.SAMPLER_NODE in pod_power.NODES and pod.NODE == pod_power.LEDGER_NODE[pod.SAMPLER_NODE]


def test_recorded_trace_excerpts_from_the_pod_parse_like_the_synthetic_ones():
    """collect --save-fixtures keeps a two-step excerpt of each real trace under recorded-<date>/; every one that is
    committed must parse into decode steps and kernels with their dims (none exist before the pod run)."""
    for folder in sorted((REPO / "fixtures" / "lab" / "profile_decode").glob("recorded-*")):
        torch_excerpt = folder / "torch_b1.pt.trace.json.gz"
        if torch_excerpt.exists():
            trace = tp.TorchTrace.load(torch_excerpt)
            steps = pd.torch_decode_steps(trace, 1)
            assert steps, f"{torch_excerpt}: no decode step ranges"
            assert trace.kernels or trace.graph_launches(), f"{torch_excerpt}: neither kernels nor graph launches"
        rocprof_excerpt = folder / "rocprof_b1_kernel_trace.csv"
        if rocprof_excerpt.exists():
            kernels = tp.read_rocprof_kernels(rocprof_excerpt)
            assert kernels, f"{rocprof_excerpt}: no kernel rows"
            assert all(k.grid and k.block for k in kernels), f"{rocprof_excerpt}: a row without grid or block"
