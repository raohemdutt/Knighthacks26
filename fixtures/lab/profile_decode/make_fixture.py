"""Writes fixtures/lab/profile_decode/synthetic/: a decode-profile run folder made by hand, in the formats the pod
writes, for `python -m lab.profile_decode --quick` and tests/test_lab_profile_decode.py.

    python fixtures/lab/profile_decode/make_fixture.py

Synthetic, not recorded: no GPU took part. The files follow the shapes the tools document, so the parser can be
tested before the pod run: the torch profiler's Chrome trace (kineto: complete events with ts and dur in
microseconds, cat kernel / gpu_memcpy / cuda_runtime / user_annotation, args.correlation linking a runtime call to
its kernels, args.grid and args.block, "[memory]" allocator samples) with vLLM's step ranges named
execute_context_<n>(<tokens>)_generation_<n>(<tokens>) (vllm/v1/worker/gpu_worker.py at b1388b1, annotate_profile),
and rocprofv3's kernel and memory copy trace CSVs (the columns of the rocprofiler-sdk 7.13 documentation, timestamps
in nanoseconds on the boottime clock here). The pod run's own trace excerpts (collect --save-fixtures) are the
recorded fixtures once it has run.

Every number is chosen so the expected results are simple:

    batch 1 decode step: 300 ms. GPU from +5 ms: 10 x _gemm_kernel 18 ms (W4A16, 180 ms), 2 x
    kernel_unified_attention_2d 10 ms, 1 x fused_recurrent_gated_delta_rule_fwd_kernel 10 ms, 2 x
    triton_poi_fused_add_rms_norm_0 5 ms, 1 x Cijk_..._SN 20 ms (lm_head), all from one hipGraphLaunch inside the
    step range (CPU 0 to 40 ms); after the range an argmax reduce_kernel 4 ms at +251 ms and a 4-byte DtoH copy
    1 ms at +256 ms. Busy 245 ms, so host-side 55 ms. Shares: W4A16 0.6, attention 0.1, sampling 4/300, copies
    1/300, other GPU 0.1, host-side 55/300.
    batch 4 decode step: 320 ms; 10 x 19 ms W4A16, 2 x 12 ms + 12 ms attention, 2 x 5 ms norm, 22 ms lm_head,
    argmax 5 ms at +270 ms, a 16-byte copy at +276 ms. Busy 264 ms; shares 190/320, 36/320, 5/320, 1/320, 32/320,
    56/320.
"""
from __future__ import annotations

import csv
import datetime as dt
import gzip
import io
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "synthetic"
T0 = dt.datetime(2026, 10, 10, 23, 0, 0, tzinfo=dt.timezone.utc).timestamp()   # wall clock of the run's start
REALTIME0 = int(T0) * 10**9
MONOTONIC0 = 400_000 * 10**9
BOOTTIME0 = MONOTONIC0 + 1_000 * 10**9          # a thousand seconds apart, so only one clock fits the windows
TRACE_BASE_US = 1_000_000.0
MS = 1000.0                                     # microseconds in a millisecond

STEP = {
    1: {"step_ms": 300, "gemm_ms": 18, "attn_ms": 10, "linear_ms": 10, "norm_ms": 5, "head_ms": 20,
        "argmax_at": 251, "argmax_ms": 4, "copy_at": 256, "copy_bytes": 4},
    4: {"step_ms": 320, "gemm_ms": 19, "attn_ms": 12, "linear_ms": 12, "norm_ms": 5, "head_ms": 22,
        "argmax_at": 270, "argmax_ms": 5, "copy_at": 276, "copy_bytes": 16},
}
GEMM = "_gemm_kernel"
ATTN = "kernel_unified_attention_2d"
LINEAR = "fused_recurrent_gated_delta_rule_fwd_kernel"
NORM = "triton_poi_fused_add_rms_norm_0"
HEAD = "Cijk_Alik_Bljk_BBS_BH_MT64x64x64_SN"
ARGMAX = "void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<float> > >"
DIMS = {GEMM: ([80, 1, 1], [256, 1, 1]), ATTN: ([8, 4, 1], [128, 1, 1]), LINEAR: ([48, 1, 1], [64, 1, 1]),
        NORM: ([20, 1, 1], [512, 1, 1]), HEAD: ([1186, 1, 1], [256, 1, 1]), ARGMAX: ([1, 1, 1], [512, 1, 1])}
LDS = {GEMM: 49152, ATTN: 32768, LINEAR: 16384, NORM: 0, HEAD: 16384, ARGMAX: 4096}


def forward_kernels(batch: int) -> list[tuple[str, float]]:
    """(name, ms) of one step's forward, in launch order."""
    p = STEP[batch]
    return ([(GEMM, p["gemm_ms"])] * 10 + [(ATTN, p["attn_ms"])] * 2 + [(LINEAR, p["linear_ms"])]
            + [(NORM, p["norm_ms"])] * 2 + [(HEAD, p["head_ms"])])


def step_kernels(batch: int, start_ms: float) -> tuple[list[tuple[str, float, float]], tuple[float, float, int]]:
    """Kernels (name, start ms, ms) of one decode step starting at start_ms, and its copy (start, ms, bytes)."""
    p = STEP[batch]
    out, t = [], start_ms + 5
    for name, ms in forward_kernels(batch):
        out.append((name, t, ms))
        t += ms
    out.append((ARGMAX, start_ms + p["argmax_at"], p["argmax_ms"]))
    return out, (start_ms + p["copy_at"], 1.0, p["copy_bytes"])


# --- the torch profiler traces --------------------------------------------------------------------------------------

def torch_trace(batch: int) -> dict:
    p = STEP[batch]
    events: list[dict] = [
        {"name": "process_name", "ph": "M", "pid": 4242, "tid": 0, "args": {"name": "python3"}},
        {"name": "process_name", "ph": "M", "pid": 0, "tid": 0, "args": {"name": "GPU 0"}},
    ]
    corr = 1000

    def kernel(name: str, ts_us: float, dur_us: float, correlation: int) -> dict:
        grid, block = DIMS[name]
        return {"ph": "X", "cat": "kernel", "name": name, "pid": 0, "tid": 7, "ts": ts_us, "dur": dur_us,
                "args": {"External id": correlation, "device": 0, "stream": 7, "correlation": correlation,
                         "registers per thread": 0, "shared memory": LDS[name], "grid": grid, "block": block}}

    def runtime(name: str, ts_us: float, dur_us: float, correlation: int) -> dict:
        return {"ph": "X", "cat": "cuda_runtime", "name": name, "pid": 4242, "tid": 4242, "ts": ts_us,
                "dur": dur_us, "args": {"External id": correlation, "cbid": 0, "correlation": correlation}}

    # a prefill step first: its kernels lie outside every decode step
    pre_name = f"execute_context_{batch}({31 * batch})_generation_0(0)"
    events.append({"ph": "X", "cat": "user_annotation", "name": pre_name, "pid": 4242, "tid": 4242,
                   "ts": TRACE_BASE_US, "dur": 60 * MS, "args": {"External id": 1}})
    events.append(runtime("hipGraphLaunch", TRACE_BASE_US + 1 * MS, 0.3 * MS, corr))
    for i in range(10):                                 # done before the first decode step starts at +100 ms
        events.append(kernel(GEMM, TRACE_BASE_US + 5 * MS + i * 8 * MS, 8 * MS, corr))
    events.append({"ph": "i", "cat": "cpu_instant_event", "s": "t", "name": "[memory]", "pid": 4242, "tid": 4242,
                   "ts": TRACE_BASE_US + 2 * MS, "args": {"Total Reserved": int(22.5 * 2**30),
                                                          "Total Allocated": 20 * 2**30 + 5_000_000,
                                                          "Bytes": 1024, "Device Type": 1, "Device Id": 0}})
    events.append({"ph": "i", "cat": "cpu_instant_event", "s": "t", "name": "[memory]", "pid": 4242, "tid": 4242,
                   "ts": TRACE_BASE_US + 3 * MS, "args": {"Total Reserved": 90 * 2**30, "Total Allocated": 80 * 2**30,
                                                          "Bytes": 64, "Device Type": 0, "Device Id": -1}})
    name = f"execute_context_0(0)_generation_{batch}({batch})"
    for k in range(6):                                  # six decode ranges: five steps have a next one
        start_ms = 100 + k * p["step_ms"]
        s_us = TRACE_BASE_US + start_ms * MS
        corr += 10
        events.append({"ph": "X", "cat": "user_annotation", "name": name, "pid": 4242, "tid": 4242, "ts": s_us,
                       "dur": 40 * MS, "args": {"External id": corr}})
        events.append({"ph": "X", "cat": "cpu_op", "name": "aten::empty", "pid": 4242, "tid": 4242,
                       "ts": s_us + 0.5 * MS, "dur": 0.05 * MS, "args": {"External id": corr}})
        events.append(runtime("hipGraphLaunch", s_us + 1 * MS, 0.3 * MS, corr))
        kernels, (copy_ms, copy_dur, copy_bytes) = step_kernels(batch, start_ms)
        for kname, k_ms, dur in kernels:
            c = corr if kname != ARGMAX else corr + 1
            events.append(kernel(kname, TRACE_BASE_US + k_ms * MS, dur * MS, c))
            if kname == GEMM and k_ms == start_ms + 5:
                events.append({"ph": "s", "id": c, "pid": 4242, "tid": 4242, "ts": s_us + 1 * MS, "cat": "ac2g",
                               "name": "ac2g"})
                events.append({"ph": "f", "id": c, "pid": 0, "tid": 7, "ts": TRACE_BASE_US + k_ms * MS,
                               "cat": "ac2g", "name": "ac2g", "bp": "e"})
        events.append(runtime("hipLaunchKernel", s_us + (p["argmax_at"] - 1) * MS, 0.02 * MS, corr + 1))
        events.append(runtime("hipMemcpyWithStream", s_us + (p["copy_at"] - 0.5) * MS, 1.6 * MS, corr + 2))
        events.append({"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy DtoH (Device -> Pinned)", "pid": 0, "tid": 7,
                       "ts": TRACE_BASE_US + copy_ms * MS, "dur": copy_dur * MS,
                       "args": {"device": 0, "stream": 7, "correlation": corr + 2, "bytes": copy_bytes}})
        events.append({"ph": "X", "cat": "gpu_user_annotation", "name": name, "pid": 0, "tid": 7,
                       "ts": s_us + 5 * MS, "dur": 240 * MS, "args": {}})
        if k in (0, 2):
            extra = 1_000_000 if k == 0 else 3_000_000
            events.append({"ph": "i", "cat": "cpu_instant_event", "s": "t", "name": "[memory]", "pid": 4242,
                           "tid": 4242, "ts": s_us + 0.1 * MS,
                           "args": {"Total Reserved": 22 * 2**30 + (0 if k == 0 else 2 * 2**20),
                                    "Total Allocated": 20 * 2**30 + extra, "Bytes": 512, "Device Type": 1,
                                    "Device Id": 0}})
    events.sort(key=lambda e: e.get("ts", 0.0))
    return {"schemaVersion": 1, "deviceProperties": [{"id": 0, "name": "AMD Radeon 8060S",
                                                       "totalGlobalMem": 68719476736, "warpSize": 32, "numSms": 20}],
            "record_shapes": 0, "with_stack": 0, "profile_memory": 1, "displayTimeUnit": "ms",
            "baseTimeNanoseconds": REALTIME0, "traceEvents": events}


# --- rocprofv3's CSVs -----------------------------------------------------------------------------------------------

ROCPROF_KERNEL_COLUMNS = ("Kind", "Agent_Id", "Queue_Id", "Stream_Id", "Thread_Id", "Dispatch_Id", "Kernel_Id",
                          "Kernel_Name", "Correlation_Id", "Start_Timestamp", "End_Timestamp", "LDS_Block_Size",
                          "Scratch_Size", "VGPR_Count", "Accum_VGPR_Count", "SGPR_Count", "Workgroup_Size_X",
                          "Workgroup_Size_Y", "Workgroup_Size_Z", "Grid_Size_X", "Grid_Size_Y", "Grid_Size_Z")
COPY_COLUMNS = ("Kind", "Direction", "Stream_Id", "Source_Agent_Id", "Destination_Agent_Id", "Correlation_Id",
                "Start_Timestamp", "End_Timestamp")


def boot_ns(wall_s: float) -> int:
    return BOOTTIME0 + round((wall_s - T0) * 1e9)


def rocprof_rows(windows: list[tuple[int, float]]) -> tuple[list[dict], list[dict]]:
    """Kernel and copy rows: three warm-up kernels long before, then five decode steps from each window start."""
    kernels, copies = [], []
    dispatch = 0

    def add(name: str, start_ns: int, dur_ns: int) -> None:
        nonlocal dispatch
        dispatch += 1
        grid, block = DIMS.get(name, ([1, 1, 1], [64, 1, 1]))
        symbol = name + ".kd" if name == GEMM else name     # rocprof shows code-object kernels with .kd
        kernels.append({"Kind": "KERNEL_DISPATCH", "Agent_Id": 2, "Queue_Id": 1, "Stream_Id": 0, "Thread_Id": 4243,
                        "Dispatch_Id": dispatch, "Kernel_Id": 17, "Kernel_Name": symbol, "Correlation_Id": dispatch,
                        "Start_Timestamp": start_ns, "End_Timestamp": start_ns + dur_ns,
                        "LDS_Block_Size": LDS.get(name, 0), "Scratch_Size": 0, "VGPR_Count": 128,
                        "Accum_VGPR_Count": 0, "SGPR_Count": 64, "Workgroup_Size_X": block[0],
                        "Workgroup_Size_Y": block[1], "Workgroup_Size_Z": block[2],
                        "Grid_Size_X": grid[0] * block[0], "Grid_Size_Y": grid[1] * block[1],
                        "Grid_Size_Z": grid[2] * block[2]})

    for i in range(3):
        add("warmup_fill_kernel", boot_ns(T0 + 40.0) + i * 1_000_000, 500_000)
    for batch, start_wall in windows:
        for k in range(5):
            start_ms = k * STEP[batch]["step_ms"]
            ks, (copy_ms, copy_dur, _) = step_kernels(batch, start_ms)
            base = boot_ns(start_wall)
            for name, k_ms, dur in ks:
                add(name, base + round(k_ms * 1e6), round(dur * 1e6))
            copies.append({"Kind": "MEMORY_COPY", "Direction": "MEMORY_COPY_DEVICE_TO_HOST", "Stream_Id": 0,
                           "Source_Agent_Id": 2, "Destination_Agent_Id": 1, "Correlation_Id": dispatch,
                           "Start_Timestamp": base + round(copy_ms * 1e6),
                           "End_Timestamp": base + round((copy_ms + copy_dur) * 1e6)})
    return kernels, copies


def csv_text(columns: tuple[str, ...], rows: list[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, quoting=csv.QUOTE_NONNUMERIC, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


# --- the run record -------------------------------------------------------------------------------------------------

WATTS = {"available": True, "reason": None, "avg_w": 118.2, "peak_w": 126.0, "energy_j": 177.3, "samples": 3,
         "source": "amd-smi", "power_field": "SOCKET_POWER", "sampler": "lab/pod_power.py 1", "period_s": 0.5,
         "throttled": None}


def request(stream: int, index: int, prompt: str, first: float, gaps_ms: list[float], role: str) -> dict:
    times = [first]
    for g in gaps_ms:
        times.append(times[-1] + g / 1000.0)
    tokens = len(times)
    return {"role": role, "stream": stream, "index": index, "prompt": prompt, "sent": None, "first_token": None,
            "last_token": None, "chunks": tokens, "tokens": tokens, "prompt_tokens": 31, "finish_reason": "length",
            "ttft_s": 0.4, "decode_tok_s": round((tokens - 1) / (times[-1] - times[0]), 4), "buffered": False,
            "cached_prompt_tokens": 0, "error": None, "sent_t": first - 0.4, "first_t": first, "last_t": times[-1],
            "token_times": times}


def run(name: str, kind: str, batch: int, requests: list[dict], **extra) -> dict:
    if batch == 1:
        windows = [[r["first_t"], r["last_t"]] for r in requests]
    else:
        windows = [[max(r["first_t"] for r in requests), min(r["last_t"] for r in requests)]]
    return {"name": name, "kind": kind, "batch": batch, "requests": requests, "decode_windows": windows,
            "watts": dict(WATTS), "errors": [], **extra}


PROMPTS = ["text-kitchen", "image-bars", "text-schedule", "image-shapes"]


def raw_record() -> dict:
    b1_base = [request(0, 0, PROMPTS[0], T0 + 10.5, [300, 310, 290, 305, 295], "baseline"),
               request(0, 1, PROMPTS[1], T0 + 13.0, [302, 298, 300, 300, 300], "baseline")]
    b4_base = [request(s, s, PROMPTS[s], T0 + 20.0, [320] * 5, "baseline") for s in range(4)]
    b1_torch = [request(0, 0, PROMPTS[0], T0 + 30.4, [330] * 6, "torch")]
    b4_torch = [request(s, s, PROMPTS[s], T0 + 40.0, [350] * 5, "torch") for s in range(4)]
    b1_roc = [request(0, 0, PROMPTS[0], T0 + 60.0, [300] * 5, "rocprof")]
    b4_roc = [request(s, s, PROMPTS[s], T0 + 70.0, [320] * 5, "rocprof") for s in range(4)]
    log_facts = {"linear_kernels": {"CompressedTensorsWNA16": "ConchLinearKernel"},
                 "attention_backends": ["Triton", "TRITON_ATTN (MMEncoderAttention)"], "kv_cache_tokens": 163840,
                 "max_concurrency": 10.0, "concurrency_tokens": 16384, "max_model_len": 16384,
                 "quantization": "compressed-tensors", "checkpoint": "/home/jovyan/models/Qwen3.8-27B-INT4",
                 "engine_build": None, "prefix_caching": True}
    conch = "/home/jovyan/.local/lib/python3.12/site-packages/conch/kernels/quantization/gemm.py"
    return {
        "schema": "lab.profile_decode.raw/1", "run_id": "r47-synthetic", "task": "R47", "node": "pod-b",
        "hub": "sjc", "account": "D", "started": "2026-10-10T23:00:00.000Z", "ended": "2026-10-10T23:27:00.000Z",
        "args": {"phases": ["torch", "rocprof"], "max_tokens": 6, "port": 8100}, "unavailable": {},
        "facts": {"hub": "sjc", "account": "D", "hostname": "synthetic-pod-d",
                  "kernel_release": "6.18.6-061806-generic",
                  "amdgpu_driver": "6.18.6", "firmware": None, "asic_serial": None, "chip": "gfx1151",
                  "market_name": "AMD Radeon 8060S", "unavailable": {}},
        "versions": {"torch": "2.9.1+rocm7.13.0", "triton": "3.5.1", "hip": "7.13.99004", "python": "3.12.3"},
        "engine": {"vllm_commit": "b1388b1fb", "vllm_version": "0.19.2+gfx1151", "unavailable": {},
                   "introspection": {
                       "classes": {"ConchLinearKernel": "/home/jovyan/vllm/vllm/model_executor/kernels/linear/"
                                                        "mixed_precision/conch.py",
                                   "ExllamaLinearKernel": "/home/jovyan/vllm/vllm/model_executor/kernels/linear/"
                                                          "mixed_precision/exllama.py"},
                       "triton_kernels": {"_gemm_kernel": conch, "_dequantize": conch},
                       "quant_triton_kernels": ["_dequantize", "_gemm_kernel"], "errors": [],
                       "conch_version": "1.2.1"}},
        "checkpoint": {"path": "/home/jovyan/models/Qwen3.8-27B-INT4", "name": "Qwen3.8-27B-INT4",
                       "bytes": {"files": 12, "tensors": 2000,
                                 "groups": {"dense": 1_200_000_000, "embedding": 1_556_000_000,
                                            "lm_head": 1_556_000_000, "mtp": 400_000_000,
                                            "quantized": 13_000_000_000, "vision": 800_000_000},
                                 "total": 18_512_000_000, "decode_read": 15_756_000_000,
                                 "quantized_read": 13_000_000_000, "tied_embedding": False}},
        "tools": {"rocprofv3": {"path": "/opt/rocm/bin/rocprofv3", "version": "rocprofv3 1.0.0 (synthetic)"},
                  "amd_smi": "/opt/rocm/bin/amd-smi"},
        "workload": {"prompt_set": {"id": "bench-v1", "sha256": "0" * 64, "prompts": 4, "images": 2},
                     "prompt_ids": PROMPTS, "max_tokens": 6, "warmup_tokens": 16, "temperature": 0.0,
                     "ignore_eos": True, "thinking": False, "batches": [1, 4],
                     "batch_layout": "batch 1: one request per prompt, one at a time; batch 4: four streams at once"},
        "clocks_start": {"realtime": REALTIME0, "monotonic": MONOTONIC0, "boottime": BOOTTIME0},
        "clocks_end": {"realtime": REALTIME0 + 1620 * 10**9, "monotonic": MONOTONIC0 + 1620 * 10**9,
                       "boottime": BOOTTIME0 + 1620 * 10**9},
        "power": {"idle_no_server_w": 9.0, "period_s": 0.5, "reader": None},
        "power_samples": "power.json",
        "phases": {
            "torch": {"kind": "torch", "ok": True, "error": None,
                      "server": {"argv": ["python3", "-m", "vllm.entrypoints.openai.api_server", "--model",
                                          "/home/jovyan/models/Qwen3.8-27B-INT4"],
                                 "profiler_config": {"profiler": "torch", "torch_profiler_dir": "/home/jovyan/x/torch"},
                                 "ready_s": 297.0, "loaded_idle_w": 9.4, "log": "serve-torch.log",
                                 "log_facts": log_facts,
                                 "memory_log": {"weights_gib": 17.98, "kv_cache_gib": 31.25, "graph_pool_gib": 0.62},
                                 "stopped": True},
                      "amdsmi_memory": {"available": True, "values": {"TOTAL_VRAM": "512 MB", "USED_VRAM": "301 MB",
                                                                      "TOTAL_GTT": "62000 MB",
                                                                      "USED_GTT": "53500 MB"}},
                      "runs": {"b1_baseline": run("b1_baseline", "baseline", 1, b1_base),
                               "b4_baseline": run("b4_baseline", "baseline", 4, b4_base),
                               "b1_torch": run("b1_torch", "torch", 1, b1_torch, trace="torch/b1.pt.trace.json.gz",
                                               trace_bytes=0, stop_profile_s=12.0),
                               "b4_torch": run("b4_torch", "torch", 4, b4_torch, trace="torch/b4.pt.trace.json.gz",
                                               trace_bytes=0, stop_profile_s=9.0)}},
            "rocprof": {"kind": "rocprof", "ok": True, "error": None,
                        "server": {"argv": ["/opt/rocm/bin/rocprofv3", "--kernel-trace"], "output_dir": "rocprof",
                                   "env": {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"}, "ready_s": 330.0,
                                   "log": "serve-rocprof.log", "log_facts": log_facts,
                                   "memory_log": {"weights_gib": 17.98, "kv_cache_gib": 31.2, "graph_pool_gib": 0.62},
                                   "stop": {"core": 4243, "api": 4242, "core_exit_s": 41.0, "server_exit_s": 6.0,
                                            "forced": False}},
                        "files": {"kernel": ["rocprof/4243/run_kernel_trace.csv"],
                                  "copy": ["rocprof/4243/run_memory_copy_trace.csv"]},
                        "runs": {"b1_rocprof": run("b1_rocprof", "rocprof", 1, b1_roc),
                                 "b4_rocprof": run("b4_rocprof", "rocprof", 4, b4_roc)}}},
    }


def main() -> None:
    (OUT / "torch").mkdir(parents=True, exist_ok=True)
    (OUT / "rocprof" / "4243").mkdir(parents=True, exist_ok=True)
    for batch in (1, 4):
        data = json.dumps(torch_trace(batch)).encode()
        (OUT / "torch" / f"b{batch}.pt.trace.json.gz").write_bytes(gzip.compress(data, mtime=0))
    kernels, copies = rocprof_rows([(1, T0 + 60.0), (4, T0 + 70.0)])
    (OUT / "rocprof" / "4243" / "run_kernel_trace.csv").write_text(csv_text(ROCPROF_KERNEL_COLUMNS, kernels),
                                                                   encoding="utf-8", newline="")
    (OUT / "rocprof" / "4243" / "run_memory_copy_trace.csv").write_text(csv_text(COPY_COLUMNS, copies),
                                                                        encoding="utf-8", newline="")
    (OUT / "raw.json").write_text(json.dumps(raw_record(), indent=1) + "\n", encoding="utf-8", newline="\n")
    def copy(label: str, value: float | None, status: str = "ok") -> dict:
        return {"id": f"copy.d2d.size_{label}", "section": "copy", "what": f"device-to-device copy of {label}",
                "unit": "GB/s", "value": value, "status": status, "error": None if status == "ok" else "synthetic"}

    # R33's sheet format (lab/kernels/chipsheet/chipsheet_format.py), the copy entries only, made-up numbers; not
    # in the data folder, so no plan ever reads it
    sheet = {"format": "atlas.chipsheet/1", "gfx": "gfx1151", "node": "pod-b", "mode": "full", "date": "2026-10-10",
             "measurements": [copy("1mib", 40.0), copy("64mib", 180.0), copy("4gib", 212.5),
                              copy("8gib", None, "failed")],
             "notes": ["synthetic, for the R47 self-check and its tests only"]}
    (OUT / "chipsheet.json").write_text(json.dumps(sheet, indent=1) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
