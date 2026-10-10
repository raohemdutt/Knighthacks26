"""The pod side of the decode profile (lab/profile_decode.py): run the stock teacher, profile its decode, keep the raw.

What one run does, on pod D (account D, node pod-b) inside one GPU Queue slot:

1. The node gate, fail closed: rocminfo must name gfx1151 (lab/pod_power.py's gate), the Jupyter user must hash to
   the one the script was made for (pod D's account; the hash, never the name, travels in the script), and the
   server port must be free. Pod C, which serves the live teacher, never passes.
2. Facts: host, ROCm, torch and Triton versions, the vLLM build and commit in ~/vllm, the conch package, rocprofv3
   and amd-smi, the checkpoint's weight bytes from its safetensors headers, and which quantized-matmul kernel class
   and which Triton kernels the build ships (asked of the pod's own Python).
3. amd-smi power sampling for the whole run (lab/bench.py's LocalPower over lab/pod_power.py's reader), and the
   idle watts with no server.
4. Phase "torch": the server is started from lab/bench.py's vLLM launch line (the one lab/pod_a_teacher.sh serves
   with) plus `--profiler-config` (vLLM 0.19's switch for its torch profiler; the VLLM_TORCH_PROFILER_DIR
   variable of older releases is gone at b1388b1), warmed up, then the fixed prompt set runs at batch 1 (one
   request per prompt, one at a time) and at batch 4 (four streams, stream s on prompt s) unprofiled, then the
   same two again between POST /start_profile and POST /stop_profile. Each stop writes one trace file.
5. Phase "rocprof": the same server under `rocprofv3 --kernel-trace --memory-copy-trace --output-format csv`, the
   engine-core child started with spawn (VLLM_WORKER_MULTIPROC_METHOD=spawn) so it leaves through the normal
   interpreter exit that flushes rocprofv3's buffers; the same two batches run once. The stop is ordered so the
   flush can finish: the API server is paused (SIGSTOP) so its five-second kill of the engine core cannot fire,
   the engine core gets SIGTERM and is waited for, then the API server is resumed and stopped.
6. Every server is stopped on the way out, whatever happened, so the pod is left as found.

Each request is timed from its own token stream (lab/bench.py's run_request), with the arrival time of every
content chunk kept, so the client-side time per decode step and its spread come from the unprofiled runs. The
clocks (realtime, monotonic, boottime) are read together at the start and the end, so a rocprof timestamp can be
put on the wall clock whichever clock rocprofiler-sdk used.

Everything lands in ~/r47_decode_profile/<run id>/ on the pod: raw.json (the run as recorded), the torch traces,
the rocprof CSVs, the serve logs and the power samples. lab/profile_decode.py reads raw.json and the traces.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import signal
import struct
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from lab import bench, pod_power
from lab.trace_profile import find_rocprof_csvs

RAW_SCHEMA = "lab.profile_decode.raw/1"
NODE = "pod-b"                      # pod D fills pod B's role (LAB_POD_ROLES D:pod-b)
SAMPLER_NODE = "gpu-b"              # lab/pod_power.py's name for the same pod
DEFAULT_CHECKPOINT = "~/models/Qwen3.8-27B-INT4"
DEFAULT_ENGINE_DIR = "~/vllm"
DEFAULT_OUT = "~/r47_decode_profile"
DEFAULT_PORT = 8100                 # not the teacher's 8000
DEFAULT_MAX_TOKENS = 64
WARMUP_TOKENS = 16
BATCHES = (1, 4)
DEFAULT_PHASES = ("torch", "rocprof")
MAX_LEN = 16384
MODEL_NAME = "teacher"
POWER_PERIOD_S = 0.5
IDLE_S = 10.0
LOADED_IDLE_S = 5.0
READY_TIMEOUT_S = 1500.0
REQUEST_TIMEOUT_S = 600.0
PROFILE_CALL_TIMEOUT_S = 1800.0     # /stop_profile returns once the worker has written its trace
CORE_EXIT_TIMEOUT_S = 900.0         # rocprofv3 writes its CSVs as the engine core exits
SERVER_EXIT_TIMEOUT_S = 900.0
HEARTBEAT_S = 30.0                  # a line at least this often, so the terminal and the proxy stay awake
TRACE_POLL_S = 2.0                  # how often a new trace file's size is read until it stops growing
DEFAULT_DEADLINE_MIN = 30.0         # the GPU Queue slot; a phase starts only with PHASE_MIN_LEFT_S still to go
PHASE_MIN_LEFT_S = 600.0
PROFILER_CONFIG = {"profiler": "torch", "torch_profiler_with_stack": False, "torch_profiler_with_memory": True,
                   "torch_profiler_record_shapes": False, "torch_profiler_dump_cuda_time_total": False,
                   "torch_profiler_use_gzip": True, "ignore_frontend": True}
ROCPROF_ARGS = ("--kernel-trace", "--memory-copy-trace", "--output-format", "csv")
ROCPROF_ENV = {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"}
TRACE_GLOB = "*.pt.trace.json*"
SIGSTOP = getattr(signal, "SIGSTOP", 19)   # Linux numbers where the platform has no such signal (tests on Windows)
SIGCONT = getattr(signal, "SIGCONT", 18)
QUANT_SUFFIXES = ("weight_packed", "weight_scale", "weight_zero_point", "weight_g_idx", "weight_shape")
VISION_MARKERS = ("visual.", "vision_tower", "vision_model", "vision_encoder", "mm_projector", "multi_modal_projector")
MEMORY_LOG = (("weights_gib", re.compile(r"Model loading took ([\d.]+) ?GiB")),
              ("kv_cache_gib", re.compile(r"Available KV cache memory: ([\d.]+) ?GiB")),
              ("graph_pool_gib", re.compile(r"Graph capturing finished in [\d.]+ secs?, took ([\d.]+) ?GiB")))
INTROSPECT = r'''
import importlib, inspect, json, pkgutil
out = {"classes": {}, "triton_kernels": {}, "quant_triton_kernels": [], "errors": [], "conch_version": None}
try:
    from vllm.model_executor.kernels import linear as L
    for name in ("ConchLinearKernel", "ExllamaLinearKernel", "MarlinLinearKernel", "MacheteLinearKernel"):
        cls = getattr(L, name, None)
        if cls is not None:
            out["classes"][name] = inspect.getsourcefile(cls)
except Exception as exc:
    out["errors"].append("vllm linear kernels: %s: %s" % (type(exc).__name__, exc))
try:
    from importlib import metadata
    out["conch_version"] = metadata.version("conch-triton-kernels")
except Exception as exc:
    out["errors"].append("conch version: %s" % type(exc).__name__)
try:
    import conch
    for info in pkgutil.walk_packages(conch.__path__, "conch."):
        if not info.name.startswith("conch.kernels"):
            continue
        try:
            mod = importlib.import_module(info.name)
        except Exception as exc:
            out["errors"].append("%s: %s" % (info.name, type(exc).__name__))
            continue
        for value in list(vars(mod).values()):
            fn = value
            depth = 0
            while hasattr(fn, "fn") and depth < 4:
                fn = fn.fn
                depth += 1
            if depth == 0 or not callable(fn) or getattr(fn, "__module__", None) != info.name:
                continue
            out["triton_kernels"][fn.__name__] = inspect.getsourcefile(fn)
            if info.name.startswith("conch.kernels.quantization"):
                out["quant_triton_kernels"].append(fn.__name__)
except Exception as exc:
    out["errors"].append("conch kernels: %s: %s" % (type(exc).__name__, exc))
out["quant_triton_kernels"] = sorted(set(out["quant_triton_kernels"]))
print(json.dumps(out))
'''


class PodRunError(Exception):
    """A run that cannot go on: the gate refused, a tool is missing, a server never came up."""


def now_iso() -> str:
    return pod_power.iso(time.time())


def say(message: str) -> None:
    print(f"[r47 {time.strftime('%H:%M:%S')}] {message}", flush=True)


class Heartbeat:
    """A line every HEARTBEAT_S seconds while a long call blocks, so a silent wait never looks like a hang."""

    def __init__(self, label: str, every_s: float = HEARTBEAT_S, out: Callable[[str], None] = say) -> None:
        self.label, self.every_s, self.out = label, every_s, out
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.started = 0.0

    def _run(self) -> None:
        while not self._stop.wait(self.every_s):
            self.out(f"still {self.label} after {time.time() - self.started:.0f} s")

    def __enter__(self) -> "Heartbeat":
        self.started = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2.0)


# --- the gate ---------------------------------------------------------------------------------------------------

def jupyter_user(environ: Mapping[str, str]) -> str | None:
    """The hub account this pod belongs to: JUPYTERHUB_USER, else the name in JUPYTERHUB_SERVICE_PREFIX."""
    user = (environ.get("JUPYTERHUB_USER") or "").strip()
    if user:
        return user
    m = re.match(r"^/user/([^/]+)/", environ.get("JUPYTERHUB_SERVICE_PREFIX") or "")
    return m.group(1) if m else None


def user_sha256(user: str) -> str:
    return hashlib.sha256(user.encode("utf-8")).hexdigest()


def gate(*, expect_user_sha256: str | None, environ: Mapping[str, str], rocminfo: Callable[[], str],
         port: int | None, port_busy: Callable[[int], bool] = bench.port_open) -> list[str]:
    """Why this process must not run here (empty when it may): not the Strix Halo, not the account the script was
    made for, or the server port already taken (port None: attaching, the port is meant to be taken)."""
    problems = []
    why = pod_power.gate(SAMPLER_NODE, rocminfo)
    if why:
        problems.append(why)
    user = jupyter_user(environ)
    if not expect_user_sha256:
        problems.append("no --expect-user-sha256: the script was not made for a pod (python -m lab.profile_decode "
                        "pod-script writes it)")
    elif user is None:
        problems.append("JUPYTERHUB_USER is not set: cannot tell which pod this is")
    elif user_sha256(user) != expect_user_sha256.strip().lower():
        problems.append("this pod's account is not the one the script was made for (pod D); pod C serves the "
                        "live teacher and is never profiled")
    if port is not None and port_busy(port):
        problems.append(f"something already listens on 127.0.0.1:{port}")
    return problems


def read_rocminfo() -> str:
    program = pod_power.find_tool("rocminfo")
    if program is None:
        raise pod_power.ReadError("rocminfo is not on PATH or under /opt/rocm/bin")
    return pod_power.run_command([program], timeout_s=30.0)


# --- facts ------------------------------------------------------------------------------------------------------

def clock_offsets() -> dict[str, int]:
    """The three clocks read back to back (nanoseconds): a timestamp on any of them can be put on the wall clock."""
    out = {"realtime": time.time_ns()}
    for name, clock in (("monotonic", getattr(time, "CLOCK_MONOTONIC", None)),
                        ("boottime", getattr(time, "CLOCK_BOOTTIME", None))):
        if clock is not None:
            out[name] = time.clock_gettime_ns(clock)
    return out


def safetensors_header(path: Path) -> dict[str, Any]:
    """The JSON header of a .safetensors file: an 8-byte little-endian length, then that many bytes of JSON."""
    with open(path, "rb") as handle:
        (size,) = struct.unpack("<Q", handle.read(8))
        if size <= 0 or size > 100 * 2**20:
            raise ValueError(f"{path.name}: implausible header length {size}")
        return json.loads(handle.read(size).decode("utf-8"))


def tensor_group(name: str) -> str:
    """Which part of the checkpoint a tensor is, for the bytes a decode step reads."""
    if any(marker in name for marker in VISION_MARKERS):
        return "vision"
    if name.startswith("mtp.") or ".mtp." in name:
        return "mtp"
    if "embed_tokens" in name:
        return "embedding"
    if "lm_head" in name:
        return "lm_head"
    if name.endswith(QUANT_SUFFIXES):
        return "quantized"
    return "dense"


def weight_bytes(folder: str | Path) -> dict[str, Any]:
    """Bytes per tensor group of a safetensors checkpoint, and the bytes one decode step reads: every weight except
    the vision tower and the MTP head, which a text decode step never runs, and the embedding table, of which a step
    gathers only its own rows (unless no lm_head is stored: then the tied table is the output matmul's weight)."""
    folder = Path(folder).expanduser()
    files = sorted(folder.glob("*.safetensors"))
    if not files:
        raise PodRunError(f"no .safetensors files in {folder}")
    groups: dict[str, int] = {}
    tensors = 0
    for path in files:
        for name, meta in safetensors_header(path).items():
            if name == "__metadata__" or not isinstance(meta, Mapping):
                continue
            start, end = meta.get("data_offsets", (0, 0))
            groups[tensor_group(name)] = groups.get(tensor_group(name), 0) + int(end) - int(start)
            tensors += 1
    total = sum(groups.values())
    read = total - groups.get("vision", 0) - groups.get("mtp", 0) - groups.get("embedding", 0)
    tied = groups.get("lm_head", 0) == 0 and groups.get("embedding", 0) > 0
    if tied:
        read += groups["embedding"]
    return {"files": len(files), "tensors": tensors, "groups": dict(sorted(groups.items())), "total": total,
            "decode_read": read, "quantized_read": groups.get("quantized", 0), "tied_embedding": tied}


def run_json(host: bench.Host, argv: Sequence[str]) -> tuple[dict[str, Any] | None, str | None]:
    try:
        text = host.command(list(argv))
        return json.loads(text.strip().splitlines()[-1]), None
    except (bench.HostError, ValueError, IndexError) as exc:
        return None, f"{' '.join(argv[:2])}: {' '.join(str(exc).split())[:300]}"


def engine_facts(host: bench.Host, engine_dir: Path, python: str) -> dict[str, Any]:
    """vLLM's version and commit, and which quantized-matmul classes and conch Triton kernels the build ships."""
    out: dict[str, Any] = {"unavailable": {}}
    try:
        out["vllm_commit"] = host.command(["git", "-C", str(engine_dir), "rev-parse", "--short=9", "HEAD"]).strip()
    except bench.HostError as exc:
        out["vllm_commit"] = None
        out["unavailable"]["vllm_commit"] = str(exc)[:200]
    try:
        # vLLM logs to stdout as it imports: the version is the last line
        lines = host.command([python, "-c", "import vllm; print(vllm.__version__)"]).strip().splitlines()
        out["vllm_version"] = lines[-1].strip() if lines else None
    except bench.HostError as exc:
        out["vllm_version"] = None
        out["unavailable"]["vllm_version"] = str(exc)[:200]
    introspection, error = run_json(host, [python, "-c", INTROSPECT])
    out["introspection"] = introspection
    if error:
        out["unavailable"]["introspection"] = error
    return out


def tool_version(host: bench.Host, program: str) -> str | None:
    try:
        return (host.command([program, "--version"]).strip().splitlines() or [None])[0]
    except bench.HostError:
        return None


def amdsmi_memory(host: bench.Host) -> dict[str, Any]:
    """amd-smi's memory use of the GPU (VRAM and GTT on this APU), as its own key: value pairs."""
    try:
        device = pod_power.amdsmi_tree(host.amdsmi("metric", "--mem-usage")).get(0, {})
    except bench.HostError as exc:
        return {"available": False, "reason": str(exc)[:200]}
    section = device.get("MEM_USAGE") if isinstance(device.get("MEM_USAGE"), Mapping) else device
    return {"available": bool(section), "values": {k: v for k, v in section.items() if isinstance(v, str)}}


def memory_from_log(text: str) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key, pattern in MEMORY_LOG:
        found = pattern.findall(text)
        out[key] = float(found[-1]) if found else None
    return out


# --- requests ---------------------------------------------------------------------------------------------------

def run_requests(endpoint: Any, prompts: bench.PromptSet, settings: bench.Settings, *, batch: int,
                 clock: bench.WallClock, tag: str, role: str,
                 timeout_s: float = REQUEST_TIMEOUT_S) -> list[dict[str, Any]]:
    """Batch 1: one request per prompt, one at a time. Batch B: B streams at once, stream s sending prompt s.
    Each request comes back as lab/bench.py's row plus the wall time of every content chunk."""

    def one(index: int, stream: int) -> dict[str, Any]:
        times: list[float] = []
        nonce = f"{tag}-{role}-{stream}-{index}-{uuid.uuid4().hex[:8]}" if settings.nonce_on else None
        result = bench.run_request(endpoint, prompts.request_body(index, settings, nonce), role=role, stream=stream,
                                   index=index, prompt_id=prompts.prompt_id(index), clock=clock,
                                   on_token=lambda: times.append(clock.wall(time.perf_counter())))
        row = result.as_dict()
        row.update({"sent_t": result.sent, "first_t": result.first, "last_t": result.last, "token_times": times})
        return row

    if batch == 1:
        return [one(i, 0) for i in range(len(prompts.prompts))]
    out: list[dict[str, Any] | None] = [None] * batch
    barrier = threading.Barrier(batch)

    def stream(s: int) -> None:
        try:
            barrier.wait(timeout=60.0)
        except threading.BrokenBarrierError:
            return
        out[s] = one(s % len(prompts.prompts), s)

    threads = [threading.Thread(target=stream, args=(s,), daemon=True) for s in range(batch)]
    for t in threads:
        t.start()
    deadline = time.monotonic() + 60.0 + timeout_s
    for t in threads:
        t.join(max(0.0, deadline - time.monotonic()))
    if any(t.is_alive() for t in threads) or any(r is None for r in out):
        raise PodRunError(f"batch {batch}: a stream had not finished after {timeout_s:g} s")
    return [r for r in out if r is not None]


def decode_windows(requests: Sequence[Mapping[str, Any]], batch: int) -> list[list[float]]:
    """The wall-clock windows in which only decode steps of this batch ran: each request's first to last token at
    batch 1; at batch B the stretch in which all B streams were decoding (latest first token to earliest last)."""
    done = [r for r in requests if r.get("first_t") is not None and r.get("last_t") is not None and not r.get("error")]
    if batch == 1:
        return [[r["first_t"], r["last_t"]] for r in done if r["last_t"] > r["first_t"]]
    if len(done) < batch:
        return []
    start, end = max(r["first_t"] for r in done), min(r["last_t"] for r in done)
    return [[start, end]] if end > start else []


def run_record(name: str, kind: str, batch: int, requests: list[dict[str, Any]], power: Any) -> dict[str, Any]:
    windows = decode_windows(requests, batch)
    start = min((w[0] for w in windows), default=None)
    end = max((w[1] for w in windows), default=None)
    watts = bench.window_watts(power, start, end) if start is not None else {"available": False,
                                                                              "reason": "no decode window"}
    errors = [r["error"] for r in requests if r.get("error")]
    return {"name": name, "kind": kind, "batch": batch, "requests": requests, "decode_windows": windows,
            "watts": watts, "errors": errors}


# --- the server -------------------------------------------------------------------------------------------------

def post(url: str, timeout_s: float) -> tuple[int, str]:
    request = urllib.request.Request(url, data=b"", method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as reply:
            return reply.status, reply.read(2000).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(2000).decode("utf-8", "replace")


def wait_ready(spawned: bench.Spawned, endpoint: Any, timeout_s: float) -> float:
    """bench.wait_ready in slices, with a line now and then; the seconds it took."""
    started = time.time()
    while True:
        remaining = timeout_s - (time.time() - started)
        try:
            bench.wait_ready(spawned, endpoint, max(1.0, min(HEARTBEAT_S, remaining)))
            return time.time() - started
        except bench.BenchError:
            if spawned.proc.poll() is not None or remaining <= HEARTBEAT_S:
                raise
            tail = ""
            try:
                lines = spawned.log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
                tail = lines[-1][-160:] if lines else ""
            except OSError:
                pass
            say(f"server still starting after {time.time() - started:.0f} s; log: {tail}")


def new_files(folder: Path, before: set[Path], *, timeout_s: float = 120.0) -> list[Path]:
    """Trace files that appeared in folder since `before`, once their sizes stop changing."""
    deadline = time.time() + timeout_s
    last: dict[Path, int] = {}
    while time.time() < deadline:
        found = sorted(set(folder.glob(TRACE_GLOB)) - before)
        sizes = {p: p.stat().st_size for p in found if p.exists()}
        if found and sizes == last and all(v > 0 for v in sizes.values()):
            return found
        last = sizes
        time.sleep(TRACE_POLL_S)
    return sorted(set(folder.glob(TRACE_GLOB)) - before)


def proc_table(host: bench.Host) -> dict[int, dict[str, Any]]:
    """pid -> {ppid, state, cmdline} for every process /proc shows."""
    out: dict[int, dict[str, Any]] = {}
    for name in host.listdir("/proc") or []:
        if not name.isdigit():
            continue
        stat = host.read_text(f"/proc/{name}/stat")
        if not stat or ")" not in stat:
            continue
        fields = stat[stat.rfind(")") + 2:].split()
        if len(fields) < 2 or not fields[1].lstrip("-").isdigit():
            continue
        raw = host.read_bytes(f"/proc/{name}/cmdline") or b""
        out[int(name)] = {"ppid": int(fields[1]), "state": fields[0],
                          "cmdline": raw.replace(b"\0", b" ").decode("utf-8", "replace").strip()}
    return out


def descendants(table: Mapping[int, Mapping[str, Any]], root: int) -> list[int]:
    """root and every process below it, parents before children."""
    children: dict[int, list[int]] = {}
    for pid, info in table.items():
        children.setdefault(int(info["ppid"]), []).append(pid)
    out, todo = [], [root]
    while todo:
        pid = todo.pop(0)
        if pid in table and pid not in out:
            out.append(pid)
            todo.extend(sorted(children.get(pid, [])))
    return out


def find_engine_core(table: Mapping[int, Mapping[str, Any]], root: int,
                     rss: Callable[[int], int] | None = None) -> tuple[int | None, int | None]:
    """(engine core pid, API server pid) under root. The engine core is the process whose title names EngineCore
    (vLLM sets it), else the biggest multiprocessing child that is not a tracker; the API server is the topmost
    process running vllm.entrypoints (or titled APIServer), else root."""
    tree = descendants(table, root)
    core = next((p for p in tree if "EngineCore" in table[p]["cmdline"]), None)
    if core is None:
        spawned = [p for p in tree if p != root and "spawn_main" in table[p]["cmdline"]
                   and "tracker" not in table[p]["cmdline"]]
        if spawned:
            core = max(spawned, key=lambda p: (rss(p) if rss else 0, -p))
    api = next((p for p in tree if p != core and "rocprof" not in table[p]["cmdline"]
                and ("vllm.entrypoints" in table[p]["cmdline"] or "APIServer" in table[p]["cmdline"])), root)
    return core, api


def rss_kib(host: bench.Host, pid: int) -> int:
    text = host.read_text(f"/proc/{pid}/status") or ""
    m = re.search(r"^VmRSS:\s+(\d+)", text, re.MULTILINE)
    return int(m.group(1)) if m else 0


def gone(table: Mapping[int, Mapping[str, Any]], pid: int) -> bool:
    info = table.get(pid)
    return info is None or info["state"] in ("Z", "X")


def stop_traced(spawned: bench.Spawned, host: bench.Host, *, kill: Callable[[int, int], None] = os.kill,
                core_timeout_s: float = CORE_EXIT_TIMEOUT_S, server_timeout_s: float = SERVER_EXIT_TIMEOUT_S,
                sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """Stop a server running under rocprofv3 so the engine core can write its trace: pause the API server, end the
    engine core and wait for it, then resume and end the API server, wait for the launched process (rocprofv3 may
    finish its files after the app), and only then clear the process group."""
    out: dict[str, Any] = {"core": None, "api": None, "core_exit_s": None, "server_exit_s": None, "forced": False}
    table = proc_table(host)
    core, api = find_engine_core(table, spawned.pid, rss=lambda p: rss_kib(host, p))
    out["core"], out["api"] = core, api
    t0 = time.time()
    try:
        if core is not None and api is not None and core != api:
            kill(api, SIGSTOP)
            try:
                kill(core, signal.SIGTERM)
                with Heartbeat("waiting for the engine core to write the rocprof trace"):
                    while time.time() - t0 < core_timeout_s and not gone(proc_table(host), core):
                        sleep(1.0)
                out["core_exit_s"] = round(time.time() - t0, 1)
            finally:
                kill(api, SIGCONT)
        target = api if api is not None else spawned.pid
        kill(target, signal.SIGTERM)
    except ProcessLookupError:
        pass
    t1 = time.time()
    with Heartbeat("waiting for the profiled server to exit"):
        while time.time() - t1 < server_timeout_s and spawned.proc.poll() is None:
            sleep(1.0)
    out["server_exit_s"] = round(time.time() - t1, 1)
    out["forced"] = spawned.proc.poll() is None
    spawned.stop(timeout_s=30.0)
    return out


# --- the run ----------------------------------------------------------------------------------------------------

class PodRun:
    """One profiling run on the pod; see the module docstring for the order of things."""

    def __init__(self, args: Any, *, host: bench.Host | None = None, power: Any = None) -> None:
        self.args = args
        self.host = host or bench.Host()
        self.started = time.time()
        self.deadline = self.started + float(args.deadline_min) * 60.0
        self.run_id = "r47-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:4]
        self.dir = Path(args.out).expanduser() / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.prompts = bench.PromptSet.load(args.prompts)
        self.settings = bench.Settings(model=MODEL_NAME, max_tokens=int(args.max_tokens))
        self.warm = bench.Settings(model=MODEL_NAME, max_tokens=WARMUP_TOKENS)
        self.clock = bench.WallClock()
        self.power = power if power is not None else bench.LocalPower.for_host(self.host, node=NODE,
                                                                                period_s=POWER_PERIOD_S)
        self.checkpoint = str(Path(args.checkpoint).expanduser())
        self.raw: dict[str, Any] = {"schema": RAW_SCHEMA, "run_id": self.run_id, "task": "R47", "node": NODE,
                                    "hub": args.hub, "account": args.account, "started": now_iso(), "ended": None,
                                    "args": {k: v for k, v in vars(args).items()
                                             if k not in ("func", "expect_user_sha256")},
                                    "phases": {}, "unavailable": {}}

    def left_s(self) -> float:
        return self.deadline - time.time()

    def save(self) -> Path:
        path = self.dir / "raw.json"
        path.write_text(json.dumps(self.raw, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
        return path

    def serve_argv(self, chip: str | None, log_path: Path) -> list[str]:
        """lab/bench.py's vLLM launch line (lab/pod_a_teacher.sh's serve flags), or args.serve_cmd when a test
        hands in another server (it may use {log} and {port_file} too, and then writes its own log)."""
        command = getattr(self.args, "serve_cmd", None) or bench.default_command("vllm", chip)
        return bench.fill_command(command, {"checkpoint": self.checkpoint, "model": MODEL_NAME,
                                            "port": self.args.port, "max_len": MAX_LEN, "log": log_path,
                                            "port_file": log_path.with_suffix(".port")})

    @property
    def own_log(self) -> bool:
        return bool(getattr(self.args, "serve_cmd", None))

    def facts(self) -> str | None:
        """Everything read before the GPU is touched; returns the chip."""
        args, host = self.args, self.host
        self.raw["facts"] = bench.host_facts(host, hub=args.hub, account=args.account)
        versions, error = bench.software_versions(host, args.python)
        self.raw["versions"] = versions
        if error:
            self.raw["unavailable"]["versions"] = error
        self.raw["engine"] = engine_facts(host, Path(args.engine_dir).expanduser(), args.python)
        try:
            self.raw["checkpoint"] = {"path": self.checkpoint, "name": Path(self.checkpoint).name,
                                      "bytes": weight_bytes(self.checkpoint)}
        except (PodRunError, OSError, ValueError) as exc:
            self.raw["checkpoint"] = {"path": self.checkpoint, "name": Path(self.checkpoint).name, "bytes": None}
            self.raw["unavailable"]["checkpoint_bytes"] = str(exc)[:300]
        rocprof = pod_power.find_tool("rocprofv3")
        if getattr(args, "rocprof_cmd", None):           # tests: a stand-in for rocprofv3
            rocprof = " ".join(args.rocprof_cmd)
        self.raw["tools"] = {"rocprofv3": {"path": rocprof, "version": tool_version(host, rocprof)
                                           if rocprof and not getattr(args, "rocprof_cmd", None) else None},
                             "amd_smi": pod_power.find_tool(pod_power.AMDSMI)}
        self.raw["workload"] = {"prompt_set": self.prompts.describe(), "prompt_ids": [self.prompts.prompt_id(i) for i in
                                                                                     range(len(self.prompts.prompts))],
                                "max_tokens": self.settings.max_tokens, "warmup_tokens": WARMUP_TOKENS,
                                "temperature": self.settings.temperature, "ignore_eos": self.settings.ignore_eos,
                                "thinking": self.settings.thinking, "batches": list(BATCHES),
                                "batch_layout": "batch 1: one request per prompt, one at a time; batch 4: four "
                                                "streams at once, stream s on prompt s"}
        self.raw["clocks_start"] = clock_offsets()
        # the gate has already seen rocminfo name gfx1151; amd-smi may still have printed N/A for the chip
        return self.raw["facts"].get("chip") or pod_power.POD_GFX

    # one server

    def start(self, argv: list[str], env: Mapping[str, str], log_name: str) -> tuple[bench.Spawned, Any, float]:
        log_path = self.dir / log_name
        # never $HOME: the ~/vllm checkout would shadow the installed package as a namespace package
        cwd = Path(getattr(self.args, "serve_dir", None) or "~/serve").expanduser()
        cwd.mkdir(parents=True, exist_ok=True)
        spawned = bench.spawn_server(argv, port=self.args.port, log_path=log_path, env=dict(env), cwd=cwd,
                                     own_log=self.own_log)
        say(f"server pid {spawned.pid} starting: {' '.join(argv[:3])} ... (log {log_path.name})")
        try:
            endpoint = bench.DirectEndpoint(f"http://127.0.0.1:{self.args.port}", timeout_s=REQUEST_TIMEOUT_S)
            ready_s = wait_ready(spawned, endpoint, min(READY_TIMEOUT_S, max(60.0, self.left_s())))
        except BaseException:
            spawned.stop(timeout_s=60.0)    # a server that never came up is not left behind
            raise
        say(f"server ready after {ready_s:.0f} s")
        return spawned, endpoint, ready_s

    def idle(self, seconds: float) -> float | None:
        """Average watts over the next seconds with nothing sent (args.idle_s caps it, for tests)."""
        cap = getattr(self.args, "idle_s", None)
        return bench.idle_watts(self.power, seconds if cap is None else min(seconds, float(cap)))

    def warmup(self, endpoint: Any, tag: str) -> list[dict[str, Any]]:
        out = run_requests(endpoint, self.prompts, self.warm, batch=1, clock=self.clock, tag=tag, role="warmup")
        out += run_requests(endpoint, self.prompts, self.warm, batch=4, clock=self.clock, tag=tag, role="warmup")
        return [{k: v for k, v in r.items() if k != "token_times"} for r in out]

    def measure(self, endpoint: Any, name: str, kind: str, batch: int) -> dict[str, Any]:
        if self.left_s() < 60.0:
            raise PodRunError(f"the run's deadline is near; {name} not started")
        say(f"{name}: batch {batch}, {len(self.prompts.prompts) if batch == 1 else batch} requests of "
            f"{self.settings.max_tokens} tokens")
        requests = run_requests(endpoint, self.prompts, self.settings, batch=batch, clock=self.clock,
                                tag=self.run_id, role=kind)
        record = run_record(name, kind, batch, requests, self.power)
        rates = [r.get("decode_tok_s") for r in requests if r.get("decode_tok_s")]
        say(f"{name}: {len(requests)} done, {len(record['errors'])} errors, decode tok/s per request "
            f"{', '.join(f'{x:.2f}' for x in rates)}")
        return record

    def server_facts(self, phase: dict[str, Any], spawned: bench.Spawned | None, log_name: str) -> None:
        try:
            text = (self.dir / log_name).read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        phase["server"]["log"] = log_name
        phase["server"]["log_facts"] = bench.parse_serve_log(text, "vllm")
        phase["server"]["memory_log"] = memory_from_log(text)
        if spawned is not None:
            env, error = bench.read_environ(self.host, spawned.pid)
            if env is not None:
                phase["server"]["environ"] = env

    def torch_phase(self, chip: str | None) -> dict[str, Any]:
        phase: dict[str, Any] = {"kind": "torch", "ok": False, "error": None, "server": {}, "runs": {}}
        trace_dir = self.dir / "torch"
        trace_dir.mkdir(exist_ok=True)
        config = {**PROFILER_CONFIG, "torch_profiler_dir": str(trace_dir)}
        argv = self.serve_argv(chip, self.dir / "serve-torch.log") + ["--profiler-config",
                                                                      json.dumps(config, separators=(",", ":"))]
        phase["server"].update({"argv": argv, "profiler_config": config})
        spawned = None
        base = f"http://127.0.0.1:{self.args.port}"
        try:
            spawned, endpoint, ready_s = self.start(argv, {}, "serve-torch.log")
            phase["server"]["ready_s"] = round(ready_s, 1)
            phase["server"]["loaded_idle_w"] = self.idle(LOADED_IDLE_S)
            phase["warmup"] = self.warmup(endpoint, self.run_id + "-t")
            for batch in BATCHES:
                phase["runs"][f"b{batch}_baseline"] = self.measure(endpoint, f"b{batch}_baseline", "baseline", batch)
            phase["amdsmi_memory"] = amdsmi_memory(self.host)
            for batch in BATCHES:
                name = f"b{batch}_torch"
                before = set(trace_dir.glob(TRACE_GLOB))
                status, body = post(base + "/start_profile", 120.0)
                if status != 200:
                    raise PodRunError(f"POST /start_profile answered {status}: {body[:200]}")
                record = self.measure(endpoint, name, "torch", batch)
                t0 = time.time()
                with Heartbeat("writing the torch trace (POST /stop_profile)"):
                    status, body = post(base + "/stop_profile", PROFILE_CALL_TIMEOUT_S)
                record["stop_profile_s"] = round(time.time() - t0, 1)
                if status != 200:
                    raise PodRunError(f"POST /stop_profile answered {status}: {body[:200]}")
                files = new_files(trace_dir, before)
                if not files:
                    raise PodRunError(f"{name}: /stop_profile answered 200 but no trace file appeared")
                record["trace"] = str(files[-1].relative_to(self.dir)).replace("\\", "/")
                record["trace_bytes"] = files[-1].stat().st_size
                say(f"{name}: trace {files[-1].name}, {record['trace_bytes'] / 2**20:.1f} MiB, "
                    f"written in {record['stop_profile_s']:.0f} s")
                phase["runs"][name] = record
            phase["ok"] = True
        except Exception as exc:  # one failed phase is recorded; the other phase still runs
            phase["error"] = f"{type(exc).__name__}: {' '.join(str(exc).split())[:400]}"
            say(f"torch phase failed: {phase['error']}")
        finally:
            self.server_facts(phase, spawned, "serve-torch.log")
            if spawned is not None:
                with Heartbeat("stopping the server"):
                    phase["server"]["stopped"] = spawned.stop(timeout_s=120.0)
        return phase

    def rocprof_phase(self, chip: str | None) -> dict[str, Any]:
        phase: dict[str, Any] = {"kind": "rocprof", "ok": False, "error": None, "server": {}, "runs": {}}
        tool = self.raw.get("tools", {}).get("rocprofv3", {}).get("path")
        if not tool:
            phase["error"] = "rocprofv3 is not on PATH or under /opt/rocm/bin"
            return phase
        out_dir = self.dir / "rocprof"
        out_dir.mkdir(exist_ok=True)
        argv = [*(getattr(self.args, "rocprof_cmd", None) or [tool]), *ROCPROF_ARGS, "-d", str(out_dir / "%pid%"),
                "-o", "run", "--",
                *self.serve_argv(chip, self.dir / "serve-rocprof.log")]
        phase["server"].update({"argv": argv, "env": dict(ROCPROF_ENV), "output_dir": "rocprof"})
        spawned = None
        try:
            spawned, endpoint, ready_s = self.start(argv, ROCPROF_ENV, "serve-rocprof.log")
            phase["server"]["ready_s"] = round(ready_s, 1)
            phase["warmup"] = self.warmup(endpoint, self.run_id + "-r")
            for batch in BATCHES:
                phase["runs"][f"b{batch}_rocprof"] = self.measure(endpoint, f"b{batch}_rocprof", "rocprof", batch)
            phase["ok"] = True
        except Exception as exc:
            phase["error"] = f"{type(exc).__name__}: {' '.join(str(exc).split())[:400]}"
            say(f"rocprof phase failed: {phase['error']}")
        finally:
            self.server_facts(phase, spawned, "serve-rocprof.log")
            if spawned is not None:
                phase["server"]["stop"] = stop_traced(spawned, self.host)
        files = {kind: [str(p.relative_to(self.dir)).replace("\\", "/") for p in paths]
                 for kind, paths in find_rocprof_csvs(out_dir).items()}
        phase["files"] = files
        if phase["ok"] and not files["kernel"]:
            phase["ok"] = False
            phase["error"] = ("rocprofv3 wrote no kernel trace CSV under rocprof/ (the server ran; see "
                              "serve-rocprof.log for what rocprofv3 printed)")
        return phase

    def run(self) -> dict[str, Any]:
        chip = self.facts()
        self.save()
        self.power.start()
        try:
            self.raw["power"] = {"idle_no_server_w": self.idle(IDLE_S), "period_s": POWER_PERIOD_S,
                                 "reader": getattr(self.power, "reason", None)}
            for name in self.args.phases:
                if self.left_s() < PHASE_MIN_LEFT_S:
                    self.raw["phases"][name] = {"kind": name, "ok": False, "runs": {},
                                                "error": f"skipped: under {PHASE_MIN_LEFT_S / 60:.0f} minutes of the "
                                                         f"{self.args.deadline_min:g}-minute deadline were left"}
                    continue
                say(f"phase {name}")
                self.raw["phases"][name] = {"torch": self.torch_phase, "rocprof": self.rocprof_phase}[name](chip)
                self.save()
        finally:
            self.power.stop()
            samples = [{"t": t, "w": w} for t, w in bench.watt_points(self.power.events())]
            (self.dir / "power.json").write_text(json.dumps(samples) + "\n", encoding="utf-8", newline="\n")
            self.raw["power_samples"] = "power.json"
            self.raw["clocks_end"] = clock_offsets()
            self.raw["ended"] = now_iso()
            self.save()
        return self.raw

