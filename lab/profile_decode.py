"""Decode profile of the stock teacher on gfx1151: where each decode step's time goes, at batch 1 and batch 4.

    python -m lab.profile_decode --quick [--out DIR]              # this machine: parser and writer on the fixture
    python -m lab.profile_decode pod-script [--out FILE]          # this machine: the one script the pod runs
    python tooling/dispatcher/pod_exec.py D term <script> 3000 > <output file>                 # the pod run
    python -m lab.profile_decode collect <output file> [--chip-sheet FILE] [--save-fixtures]
    python -m lab.profile_decode pod-run --expect-user-sha256 HEX [...]   # on the pod, what the script runs
    python -m lab.profile_decode analyze RUN_DIR [--out DIR]              # re-read a run folder (pod or copy)

`python -m lab profile-decode ...` is the same. Every call answers with one JSON line and exits 0 when it worked.
pod_exec.py is the workspace's own pod tool (not in this repo). Its term mode, the dispatcher's, types the script
into a pod terminal inside a quoted heredoc, runs it and streams its output back until its own end marker; the
script has no tab and no exclamation mark, and its base64 lines are 76 characters, so an interactive shell reads
it as typed. The last argument is the wait in seconds: 3000 covers the 30-minute deadline plus the one 15-minute
extension. Should a 200 KB paste stall the terminal, run mode (`D run <script> r47_pod_run.sh 3000`) uploads the
script through the Jupyter contents API instead and types only its `bash` line, then reads the .out file back
until the script's last line, `### END rc=<code>`.

The run (lab/profile_decode_pod.py): the stock vLLM 0.19.2 teacher build R30 left on pod D (account D, node pod-b)
is started, warmed up, and the fixed prompt set of lab/bench.py (fixtures/lab/bench/prompts.json) runs at batch 1
(one request per prompt, one at a time) and batch 4 (four streams at once): unprofiled, then under the torch
profiler through vLLM's own /start_profile and /stop_profile, then on a second server started under rocprofv3.
Each tool that fails is recorded with its error and the run goes on with the other. With --attach-port the run
profiles a server already running there instead and starts or stops nothing (lab/profile_decode_pod.py says how).

What the result holds (lab/refine/research/kernels/data/decode-profile-gfx1151-<date>.json and its .md):

- time per decode step and its spread: from the client's token stream of the unprofiled runs (every gap between two
  content chunks), from the torch trace (vLLM marks each engine step with a record_function range named
  execute_context_<n>(<tokens>)_generation_<n>(<tokens>); a decode step is one with no context request, and its
  time is the gap to the next decode step's start) and from the rocprof trace (the client's decode windows put on
  rocprof's clock, cut into steps at a kernel that runs exactly once per step);
- the top 25 kernels by self time over the decode steps with launch counts, grid and block (rocprof's dispatch
  record when that run worked, else the torch profiler's launch arguments);
- the share of a step in the W4A16 dequantize-matmul kernel, attention (full and linear), sampling (in the torch
  trace: every kernel launched after the step's forward range, which is vLLM's sample_tokens call), host-side
  scheduling (the part of the step in which the GPU ran nothing), host-device copies, and other GPU work;
- achieved memory bandwidth: the bytes a decode step reads (the checkpoint's safetensors headers: every weight but
  the vision tower, the MTP head and the gathered embedding rows; KV-cache reads are not counted) over the step
  time, and over the W4A16 kernel's own time for the quantized weights, against the chip sheet's measured
  device-to-device copy over its largest buffer (R33's lab/refine/research/kernels/data/chipsheet-gfx1151-<date>.json;
  smaller buffers fit the 32 MiB last-level cache) when one is on main at collect time, never a sample or
  self-check sheet's numbers;
- peak and reserved memory (the torch profiler's allocator samples, the serve log's weights, KV-cache and graph pool
  lines, and amd-smi's memory use), watts during every run from amd-smi, and the exact kernel vLLM selected for the
  quantized matmul (the serve log's "Using <Kernel> for <Scheme>" line, the class's source file, and the Triton
  kernel the class launches with its source file, both asked of the pod's own Python).

The pod sends its result back through the terminal as one base64 block between two marker lines (gzip'd tar of
result.json, raw.json, the pod's summary, the serve log heads and tails and a few-step excerpt of each trace);
`collect` reads it from the saved terminal output, checks its sha256, adds the chip-sheet reference, writes the
JSON and Markdown and the ledger rows (node pod-b), and with --save-fixtures keeps the trace excerpts under
fixtures/lab/profile_decode/recorded/.
"""
from __future__ import annotations

import argparse
import ast
import base64
import bisect
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import math
import os
import re
import shlex
import signal
import statistics
import sys
import tarfile
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if not __package__:  # run by path: the repo root, one folder up, holds lab/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab import trace_profile as tp  # noqa: E402

CLI_VERB = "profile-decode"   # registers python -m lab profile-decode
SCHEMA = "lab.profile_decode/1"
TASK = "R47"
NODE = "pod-b"
CHIP = "gfx1151"
REPO = Path(__file__).resolve().parents[1]
RESULT_PREFIX = f"decode-profile-{CHIP}"
DATA_DIR = REPO / "lab" / "refine" / "research" / "kernels" / "data"
FIXTURES = REPO / "fixtures" / "lab" / "profile_decode"
QUICK_FIXTURE = FIXTURES / "synthetic"
STATE_DIR = REPO / "lab" / "state" / "profile_decode"
TOP_KERNELS = 25
BATCHES = (1, 4)
STEP_RE = re.compile(r"^execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)$")
SHARE_KEYS = ("w4a16_matmul", "attention", "sampling", "host_side", "copies", "other_gpu")
OTHER_GPU = ("dense_matmul", "norm_activation", "elementwise", "other")
MIN_KERNEL_COVERAGE = 0.5        # below this share of a step in kernels, the trace misses kernels (graph replays)
MAX_SHARE = 1.5                  # kernel and copy time may overlap a little; anything beyond this is a bug
MARK_BEGIN = "=== R47-RESULT BEGIN"
MARK_END = "=== R47-RESULT END ==="
B64_LINE = 76
METRIC_PREFIX = "teacher"
MD_SECTIONS = ("## Tools", "## Time per decode step", "## Where a decode step goes", "## Top kernels, batch 1",
               "## Top kernels, batch 4", "## The quantized matmul kernel", "## Memory bandwidth", "## Memory",
               "## Watts", "## Open items")
POD_SCRIPT = STATE_DIR / "pod_run.sh"
DEFAULT_ACCOUNT = "D"
SCRIPT_ROOT = "~/r47_decode_profile"


class ProfileError(Exception):
    """A result or an input that cannot be used: a broken bundle, a result that fails its checks."""


def _r(value: float | None, digits: int = 4) -> float | None:
    return None if value is None or not math.isfinite(float(value)) else round(float(value), digits)


# --- the torch trace ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Step:
    start_us: float        # the step's record_function range starts (CPU)
    end_us: float          # the next decode step starts
    forward_end_us: float  # the range ends: execute_model is done; sampling comes after


def step_kind(name: str) -> tuple[int, int] | None:
    """(context requests, generation requests) of a vLLM step range name; None for any other name."""
    m = STEP_RE.match(name)
    return (int(m.group(1)), int(m.group(3))) if m else None


def torch_decode_steps(trace: tp.TorchTrace, batch: int) -> list[Step]:
    """Decode-only steps with `batch` generation requests whose next step is one too, in order."""
    ranges = sorted((a for a in trace.annotations if step_kind(a.name) is not None), key=lambda a: a.start_us)
    out = []
    for a, b in zip(ranges, ranges[1:]):
        if step_kind(a.name) == (0, batch) and step_kind(b.name) == (0, batch) and b.start_us > a.start_us:
            out.append(Step(a.start_us, b.start_us, a.end_us))
    return out


def attribute(items: Sequence[Any], steps: Sequence[Step],
              launch_of: Callable[[int | None], float | None]) -> tuple[list[list[tuple[Any, bool]]], dict[str, int]]:
    """Each kernel or copy to the step whose CPU interval holds its launch (the runtime call with its correlation
    id), with whether it was launched after the step's forward range; by its own GPU start when the trace has no
    launch for it. Items outside every step are left out."""
    starts = [s.start_us for s in steps]
    per_step: list[list[tuple[Any, bool]]] = [[] for _ in steps]
    how = {"by_launch": 0, "by_gpu_time": 0, "outside": 0}
    for item in items:
        launched = launch_of(item.correlation)
        t = launched if launched is not None else item.start_us
        i = bisect.bisect_right(starts, t) - 1
        if i < 0 or t >= steps[i].end_us:
            how["outside"] += 1
            continue
        how["by_launch" if launched is not None else "by_gpu_time"] += 1
        per_step[i].append((item, launched is not None and t >= steps[i].forward_end_us))
    return per_step, how


def bucket_of(category: str) -> str:
    if category in tp.ATTENTION_GROUP:
        return "attention"
    if category in ("w4a16_matmul", "sampling"):
        return category
    return "other_gpu"


def _breakdown(bucket_us: Mapping[str, float], copies_us: float, busy_us: float, total_us: float,
               steps: int) -> tuple[dict[str, float | None], dict[str, float | None]]:
    """ms per step and share of the step for the SHARE_KEYS buckets, plus gpu_busy and the step itself."""
    values = {key: bucket_us.get(key, 0.0) for key in ("w4a16_matmul", "attention", "sampling", "other_gpu")}
    values["copies"] = copies_us
    values["host_side"] = max(0.0, total_us - busy_us)
    ms = {key: _r(values[key] / steps / 1000.0, 4) for key in SHARE_KEYS}
    ms["gpu_busy"] = _r(busy_us / steps / 1000.0, 4)
    ms["step"] = _r(total_us / steps / 1000.0, 4)
    shares = {key: _r(values[key] / total_us, 6) if total_us else None for key in SHARE_KEYS}
    shares["gpu_busy"] = _r(busy_us / total_us, 6) if total_us else None
    return ms, shares


def analyze_torch_decode(path: str | Path, batch: int, categorize: tp.Categorizer, *,
                         top: int = TOP_KERNELS) -> dict[str, Any]:
    """One torch profiler trace of the decode at `batch`: steps, their spread, shares, top kernels, copies,
    host-side times, allocator peaks."""
    trace = tp.TorchTrace.load(path)
    steps = torch_decode_steps(trace, batch)
    out: dict[str, Any] = {"source": "torch", "batch": batch, "trace": Path(path).name, "events": trace.event_count,
                           "kernels_in_trace": len(trace.kernels), "graph_launches": trace.graph_launches(),
                           "device": trace.device.get("name")}
    if not steps:
        out.update({"available": False, "reason": f"no two consecutive decode-only vLLM step ranges "
                    f"(execute_context_0(0)_generation_{batch}({batch})) in the trace", "steps": 0})
        return out
    n = len(steps)
    kernel_steps, kernel_how = attribute(trace.kernels, steps, trace.launch_us)
    copy_steps, copy_how = attribute(trace.copies, steps, trace.launch_us)
    durations = [s.end_us - s.start_us for s in steps]
    total_us = sum(durations)
    busy = [tp.union_us([(k.start_us, k.end_us) for k, _ in ks] + [(c.start_us, c.end_us) for c, _ in cs])
            for ks, cs in zip(kernel_steps, copy_steps)]
    buckets: dict[str, float] = defaultdict(float)
    categories: dict[str, float] = defaultdict(float)
    post_us = 0.0
    kernels = []
    for ks in kernel_steps:
        for k, post in ks:
            kernels.append(k)
            category = categorize(k.name)
            categories[category] += k.dur_us
            if post:
                post_us += k.dur_us
                buckets["sampling"] += k.dur_us
            else:
                buckets[bucket_of(category)] += k.dur_us
    copies = [c for cs in copy_steps for c, _ in cs]
    copies_us = sum(c.dur_us for c in copies)
    ms, shares = _breakdown(buckets, copies_us, sum(busy), total_us, n)
    runtime = [r for r in trace.runtime if steps[0].start_us <= r.start_us < steps[-1].end_us]
    graph_calls = sum(1 for r in runtime if "graphlaunch" in r.name.lower())
    launch_calls = sum(1 for r in runtime if "launchkernel" in r.name.lower())
    first, last = steps[0].start_us, steps[-1].end_us
    out.update({
        "available": True, "steps": n, "step_ms": tp.spread([d / 1000.0 for d in durations]),
        "ms_per_step": ms, "shares": shares,
        "attention_split_ms": {"full": _r(categories.get("attention", 0.0) / n / 1000.0),
                               "linear": _r(categories.get("attention_linear", 0.0) / n / 1000.0)},
        "categories_ms": {c: _r(v / n / 1000.0) for c, v in sorted(categories.items())},
        "sampling_method": "kernels launched after the step's forward range (vLLM's sample_tokens), by their "
                           "runtime call's correlation id",
        "sampling_launched_after_forward_ms": _r(post_us / n / 1000.0),
        "kernel_coverage": _r(sum(busy) / total_us, 4) if total_us else None,
        "attribution": {"kernels": kernel_how, "copies": copy_how},
        "copies": tp.copy_summary(copies, steps=n),
        "host": {"forward_cpu_ms": tp.spread([(s.forward_end_us - s.start_us) / 1000.0 for s in steps]),
                 "after_forward_cpu_ms": tp.spread([(s.end_us - s.forward_end_us) / 1000.0 for s in steps]),
                 "runtime_calls_per_step": _r(len(runtime) / n, 3),
                 "runtime_cpu_ms_per_step": _r(sum(r.dur_us for r in runtime) / n / 1000.0),
                 "graph_launches_per_step": _r(graph_calls / n, 3),
                 "kernel_launch_calls_per_step": _r(launch_calls / n, 3)},
        "memory": {"whole_trace": tp.memory_peaks(trace.memory),
                   "decode_steps": tp.memory_peaks(trace.memory, start_us=first, end_us=last)},
        "kernels_top": tp.kernel_table(kernels, steps=n, step_us=total_us / n, categorize=categorize, top=top),
        "kernels_distinct": len({k.name for k in kernels}),
        "uncategorized_top": [row for row in tp.kernel_table(kernels, steps=n, step_us=total_us / n,
                                                             categorize=categorize, top=None)
                              if row["category"] == "other"][:5],
    })
    return out


# --- the rocprof trace ----------------------------------------------------------------------------------------------

def to_clock_us(t_wall: float, offsets: Mapping[str, int], clock: str) -> float:
    """A wall-clock time (Unix seconds) on another clock read at the same moment as `offsets`, in microseconds."""
    return (offsets[clock] + (t_wall * 1e9 - offsets["realtime"])) / 1000.0


def run_windows(run: Mapping[str, Any]) -> list[tuple[float, float, int]]:
    """(start, end, decode steps) on the wall clock: at batch 1 each request from its first to its last token
    (tokens - 1 steps), at batch B the stretch in which all B streams decoded (the median count of tokens that
    arrived inside it)."""
    batch = int(run["batch"])
    done = [r for r in run.get("requests", []) if r.get("first_t") is not None and r.get("last_t") is not None
            and not r.get("error")]
    if batch == 1:
        return [(float(r["first_t"]), float(r["last_t"]), int(r["tokens"]) - 1) for r in done
                if int(r.get("tokens") or 0) > 1 and r["last_t"] > r["first_t"]]
    if len(done) < batch:
        return []
    start, end = max(float(r["first_t"]) for r in done), min(float(r["last_t"]) for r in done)
    if end <= start:
        return []
    counts = [sum(1 for t in r.get("token_times", []) if start < t <= end) for r in done]
    steps = int(statistics.median(counts)) if counts else 0
    return [(start, end, steps)] if steps > 0 else []


def choose_clock(kernels: Sequence[tp.Kernel], offsets: Mapping[str, int],
                 windows: Sequence[tuple[float, float, int]]) -> tuple[str | None, dict[str, int]]:
    """Which clock rocprof's timestamps are on: the one that puts the most kernels inside the decode windows."""
    starts = sorted(k.start_us for k in kernels)
    counts: dict[str, int] = {}
    for clock in ("boottime", "monotonic", "realtime"):
        if clock not in offsets:
            continue
        count = 0
        for s, e, _ in windows:
            a, b = to_clock_us(s, offsets, clock), to_clock_us(e, offsets, clock)
            count += bisect.bisect_left(starts, b) - bisect.bisect_left(starts, a)
        counts[clock] = count
    best = max(counts, key=lambda c: (counts[c], c == "boottime")) if counts else None
    return (best if best is not None and counts[best] > 0 else None), counts


def analyze_rocprof_decode(kernels: Sequence[tp.Kernel], copies: Sequence[tp.Copy], run: Mapping[str, Any], *,
                           offsets: Mapping[str, int], clock: str, categorize: tp.Categorizer,
                           top: int = TOP_KERNELS) -> dict[str, Any]:
    """The rocprof kernels and copies inside one run's decode windows: steps, spread, shares, top kernels."""
    batch = int(run["batch"])
    out: dict[str, Any] = {"source": "rocprof", "batch": batch, "clock": clock}
    windows = [(to_clock_us(s, offsets, clock), to_clock_us(e, offsets, clock), n) for s, e, n in run_windows(run)]
    steps = sum(n for _, _, n in windows)
    if not windows or steps < 1:
        out.update({"available": False, "reason": "no decode window in the run's requests", "steps": 0})
        return out
    starts = [k.start_us for k in kernels]
    inside: list[tp.Kernel] = []
    inside_copies: list[tp.Copy] = []
    busy_us = 0.0
    gaps: list[float] = []
    markers = []
    copy_starts = [c.start_us for c in copies]
    for a, b, n in windows:
        ks = kernels[bisect.bisect_left(starts, a):bisect.bisect_left(starts, b)]
        cs = copies[bisect.bisect_left(copy_starts, a):bisect.bisect_left(copy_starts, b)]
        inside.extend(ks)
        inside_copies.extend(cs)
        busy_us += tp.union_us([(max(a, x.start_us), min(b, x.end_us)) for x in (*ks, *cs)])
        marker = tp.pick_marker(ks, n)
        if marker is not None:
            gaps.extend(marker["gaps_us"])
            markers.append({"name": marker["name"], "launches": marker["launches"], "cv": marker["cv"]})
    total_us = sum(b - a for a, b, _ in windows)
    if not inside:
        out.update({"available": False, "reason": f"no kernel of the rocprof trace starts inside the decode "
                    f"windows on the {clock} clock", "steps": steps})
        return out
    buckets: dict[str, float] = defaultdict(float)
    categories: dict[str, float] = defaultdict(float)
    for k in inside:
        category = categorize(k.name)
        categories[category] += k.dur_us
        buckets[bucket_of(category)] += k.dur_us
    ms, shares = _breakdown(buckets, sum(c.dur_us for c in inside_copies), busy_us, total_us, steps)
    step_values = [g / 1000.0 for g in gaps] if gaps else [(b - a) / n / 1000.0 for a, b, n in windows]
    out.update({
        "available": True, "steps": steps, "windows": len(windows),
        "step_ms": tp.spread(step_values),
        "step_ms_method": ("gaps between launches of a kernel that runs once per step" if gaps
                           else "window length over its steps (no once-per-step kernel found)"),
        "step_markers": markers, "ms_per_step": ms, "shares": shares,
        "attention_split_ms": {"full": _r(categories.get("attention", 0.0) / steps / 1000.0),
                               "linear": _r(categories.get("attention_linear", 0.0) / steps / 1000.0)},
        "categories_ms": {c: _r(v / steps / 1000.0) for c, v in sorted(categories.items())},
        "sampling_method": "kernel names (tp.DEFAULT_RULES 'sampling')",
        "kernel_coverage": _r(busy_us / total_us, 4) if total_us else None,
        "copies": tp.copy_summary(inside_copies, steps=steps),
        "kernels_top": tp.kernel_table(inside, steps=steps, step_us=total_us / steps, categorize=categorize, top=top),
        "kernels_distinct": len({k.name for k in inside}),
        "uncategorized_top": [row for row in tp.kernel_table(inside, steps=steps, step_us=total_us / steps,
                                                             categorize=categorize, top=None)
                              if row["category"] == "other"][:5],
    })
    return out


# --- the client's token stream ----------------------------------------------------------------------------------------

def client_steps(run: Mapping[str, Any]) -> dict[str, Any]:
    """Time per decode step as the client saw it: the gaps between content chunks inside the decode windows (every
    request at batch 1; every stream inside the stretch where all decoded at batch B), and the decode rate."""
    batch = int(run["batch"])
    windows = run_windows(run)
    gaps: list[float] = []
    for r in run.get("requests", []):
        times = [float(t) for t in r.get("token_times", [])]
        if batch > 1 and windows:
            start, end, _ = windows[0]
            times = [t for t in times if start <= t <= end]
        gaps.extend((b - a) * 1000.0 for a, b in zip(times, times[1:]))
    steps = sum(n for _, _, n in windows)
    span = sum(e - s for s, e, _ in windows)
    rates = [r.get("decode_tok_s") for r in run.get("requests", []) if r.get("decode_tok_s")]
    return {"source": "client", "batch": batch, "available": bool(gaps), "steps": steps,
            "step_ms": tp.spread(gaps), "step_ms_from_windows": _r(span / steps * 1000.0, 3) if steps else None,
            "tok_s": _r(batch * steps / span, 3) if span else None,
            "tok_s_per_request": [_r(x, 3) for x in rates], "requests": len(run.get("requests", [])),
            "errors": list(run.get("errors", []))}


# --- the result -----------------------------------------------------------------------------------------------------

def quant_names(raw: Mapping[str, Any]) -> list[str]:
    intro = (raw.get("engine") or {}).get("introspection") or {}
    return list(intro.get("quant_triton_kernels") or [])


def categorizer_for(raw: Mapping[str, Any]) -> tp.Categorizer:
    """The default rules, with the Triton kernels of the pod's quantized-matmul package matched by exact name."""
    return tp.Categorizer(exact={name: "w4a16_matmul" for name in quant_names(raw)})


def _short_path(path: str | None) -> str | None:
    """A source path from the package root on (vllm/..., conch/...): the pod's home folders say nothing."""
    if not path:
        return None
    text = path.replace("\\", "/")
    for root in ("/site-packages/", "/dist-packages/", "/vllm/vllm/"):
        if root in text:
            tail = text.split(root, 1)[1]
            return ("vllm/" + tail) if root == "/vllm/vllm/" else tail
    return text


def quant_kernel(raw: Mapping[str, Any], analyses: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The kernel vLLM chose for the W4A16 layers: the serve log's line, the class and its file, and the Triton
    kernels that class's package launches, with where each was seen in the traces and its share of a step."""
    chosen: dict[str, str] = {}
    for phase in (raw.get("phases") or {}).values():
        linear = ((phase.get("server") or {}).get("log_facts") or {}).get("linear_kernels") or {}
        chosen.update(linear)
    intro = (raw.get("engine") or {}).get("introspection") or {}
    classes = intro.get("classes") or {}
    triton = intro.get("triton_kernels") or {}
    names = sorted(set(chosen.values()))
    seen: dict[str, dict[str, Any]] = {}
    for a in analyses:
        if not a.get("available"):
            continue
        for row in a.get("kernels_top", []):
            if row["category"] == "w4a16_matmul":
                entry = seen.setdefault(row["name"], {"name": row["name"], "seen_in": {}})
                entry["seen_in"][f"{a['source']}_b{a['batch']}"] = {"share_of_step": row["share_of_step"],
                                                                   "ms_per_step": row["ms_per_step"],
                                                                   "launches_per_step": row["launches_per_step"],
                                                                   "grid": row["grid"], "block": row["block"]}
    gpu_kernels = []
    for name in sorted(seen) or quant_names(raw):     # the package's kernels only when no trace showed one
        entry = dict(seen.get(name) or {"name": name, "seen_in": {}})
        entry["source"] = _short_path(triton.get(name))
        entry["source_abs"] = triton.get(name)
        gpu_kernels.append(entry)
    return {"scheme_to_kernel": chosen, "vllm_class": names[0] if len(names) == 1 else (names or None),
            "vllm_class_source": _short_path(classes.get(names[0])) if len(names) == 1 else None,
            "vllm_classes_shipped": {k: _short_path(v) for k, v in classes.items()},
            "conch_version": intro.get("conch_version"), "gpu_kernels": gpu_kernels,
            "package_kernels": {k: _short_path(v) for k, v in sorted(triton.items()) if k in quant_names(raw)},
            "evidence": ([f"serve log: Using {k} for {s}" for s, k in sorted(chosen.items())]
                         + [f"pod python: {k} in {_short_path(v)}" for k, v in sorted(classes.items()) if k in names])}


def _load_rocprof(base: Path, files: Mapping[str, Sequence[str]]) -> tuple[list[tp.Kernel], list[tp.Copy]]:
    kernels: list[tp.Kernel] = []
    copies: list[tp.Copy] = []
    for rel in files.get("kernel", []):
        kernels.extend(tp.read_rocprof_kernels(base / rel))
    for rel in files.get("copy", []):
        copies.extend(tp.read_rocprof_copies(base / rel))
    kernels.sort(key=lambda k: k.start_us)
    copies.sort(key=lambda c: c.start_us)
    return kernels, copies


def _primary(torch_a: Mapping[str, Any] | None, rocprof_a: Mapping[str, Any] | None) -> str | None:
    """Which trace the step breakdown comes from: the torch trace when its kernels cover most of a step (its step
    ranges place sampling and host time exactly), else rocprof's, else whatever is there."""
    torch_ok = bool(torch_a and torch_a.get("available"))
    rocprof_ok = bool(rocprof_a and rocprof_a.get("available"))
    if torch_ok and (torch_a.get("kernel_coverage") or 0.0) >= MIN_KERNEL_COVERAGE:
        return "torch"
    if rocprof_ok:
        return "rocprof"
    return "torch" if torch_ok else None


def _bandwidth(weights: Mapping[str, Any] | None, step_ms: float | None, primary: Mapping[str, Any] | None) -> dict:
    out: dict[str, Any] = {"bytes_per_step": None, "quantized_bytes_per_step": None, "achieved_gb_s_step": None,
                           "achieved_gb_s_gpu_busy": None, "achieved_gb_s_w4a16": None,
                           "counts": "weights only: every tensor but the vision tower, the MTP head and the gathered "
                                     "embedding rows (KV-cache reads not counted)"}
    if not weights:
        out["reason"] = "the checkpoint's weight bytes were not read"
        return out
    read, quantized = weights.get("decode_read"), weights.get("quantized_read")
    out["bytes_per_step"], out["quantized_bytes_per_step"] = read, quantized
    ms = (primary or {}).get("ms_per_step") or {}
    if read and step_ms:
        out["achieved_gb_s_step"] = _r(read / (step_ms / 1000.0) / 1e9, 3)
    if read and ms.get("gpu_busy"):
        out["achieved_gb_s_gpu_busy"] = _r(read / (ms["gpu_busy"] / 1000.0) / 1e9, 3)
    if quantized and ms.get("w4a16_matmul"):
        out["achieved_gb_s_w4a16"] = _r(quantized / (ms["w4a16_matmul"] / 1000.0) / 1e9, 3)
    return out


def build_result(raw: Mapping[str, Any], base: str | Path) -> dict[str, Any]:
    """The decode profile from a run folder: raw.json as the pod wrote it and the traces beside it."""
    base = Path(base)
    categorize = categorizer_for(raw)
    phases = raw.get("phases") or {}
    torch_phase, rocprof_phase = phases.get("torch") or {}, phases.get("rocprof") or {}
    open_items: list[str] = []
    batches: dict[str, Any] = {}
    analyses: list[dict[str, Any]] = []
    offsets = raw.get("clocks_start") or {}
    rocprof_kernels: list[tp.Kernel] = []
    rocprof_copies: list[tp.Copy] = []
    rocprof_clock, rocprof_counts = None, {}
    if rocprof_phase.get("runs"):
        try:
            rocprof_kernels, rocprof_copies = _load_rocprof(base, rocprof_phase.get("files") or {})
        except (OSError, ValueError) as exc:
            open_items.append(f"rocprof: its CSV could not be read ({type(exc).__name__}: {exc})")
        windows = [w for run in rocprof_phase["runs"].values() for w in run_windows(run)]
        if rocprof_kernels:
            rocprof_clock, rocprof_counts = choose_clock(rocprof_kernels, offsets, windows)
    weights = ((raw.get("checkpoint") or {}).get("bytes")) or None
    for batch in BATCHES:
        entry: dict[str, Any] = {"batch": batch}
        baseline = (torch_phase.get("runs") or {}).get(f"b{batch}_baseline")
        client = client_steps(baseline) if baseline else None
        torch_run = (torch_phase.get("runs") or {}).get(f"b{batch}_torch")
        rocprof_run = (rocprof_phase.get("runs") or {}).get(f"b{batch}_rocprof")
        torch_a = rocprof_a = None
        if torch_run and torch_run.get("trace"):
            try:
                torch_a = analyze_torch_decode(base / torch_run["trace"], batch, categorize)
            except (OSError, ValueError) as exc:
                torch_a = {"source": "torch", "batch": batch, "available": False,
                           "reason": f"the trace could not be read: {type(exc).__name__}: {exc}"}
            torch_a["client"] = client_steps(torch_run)
        if rocprof_run:
            if rocprof_clock is None:
                rocprof_a = {"source": "rocprof", "batch": batch, "available": False,
                             "reason": "no rocprof kernel falls in the decode windows on any clock "
                                       f"(kernels read: {len(rocprof_kernels)}, counts: {rocprof_counts})"}
            else:
                rocprof_a = analyze_rocprof_decode(rocprof_kernels, rocprof_copies, rocprof_run, offsets=offsets,
                                                   clock=rocprof_clock, categorize=categorize)
            rocprof_a["client"] = client_steps(rocprof_run)
        for a in (torch_a, rocprof_a):
            if a is not None:
                analyses.append(a)
                if not a.get("available"):
                    open_items.append(f"{a['source']} batch {batch}: {a.get('reason')}")
        primary_name = _primary(torch_a, rocprof_a)
        primary = {"torch": torch_a, "rocprof": rocprof_a}.get(primary_name) if primary_name else None
        tables = rocprof_a if rocprof_a and rocprof_a.get("available") else torch_a
        if torch_a and torch_a.get("available") and (torch_a.get("kernel_coverage") or 0) < MIN_KERNEL_COVERAGE:
            open_items.append(f"torch batch {batch}: kernels cover {torch_a['kernel_coverage']:.0%} of a step and the "
                              f"trace holds {torch_a['host']['graph_launches_per_step']} graph launches a step: the "
                              "torch profiler does not show the kernels inside HIP graph replays here")
        step_mean = (client or {}).get("step_ms", {}).get("mean") if client else None
        entry.update({
            "available": bool(primary),
            "step_ms": {"client_unprofiled": (client or {}).get("step_ms"),
                        "client_torch_run": ((torch_a or {}).get("client") or {}).get("step_ms"),
                        "torch": (torch_a or {}).get("step_ms"), "rocprof": (rocprof_a or {}).get("step_ms")},
            "tok_s": {"client_unprofiled": (client or {}).get("tok_s"),
                      "per_request": (client or {}).get("tok_s_per_request")},
            "breakdown_source": primary_name,
            "ms_per_step": (primary or {}).get("ms_per_step"), "shares": (primary or {}).get("shares"),
            "attention_split_ms": (primary or {}).get("attention_split_ms"),
            "kernels_top_source": (tables or {}).get("source") if tables and tables.get("available") else None,
            "kernels_top": (tables or {}).get("kernels_top", []) if tables and tables.get("available") else [],
            "bandwidth": _bandwidth(weights, step_mean, primary),
            "watts": {"unprofiled": (baseline or {}).get("watts"), "torch": (torch_run or {}).get("watts"),
                      "rocprof": (rocprof_run or {}).get("watts")},
            "client_unprofiled": client,
            "sources": {"torch": torch_a, "rocprof": rocprof_a},
        })
        if not primary:
            open_items.append(f"batch {batch}: no trace gave a step breakdown")
        batches[str(batch)] = entry
    torch_traces = [r.get("trace") for r in (torch_phase.get("runs") or {}).values() if r.get("trace")]
    tools = {"torch_profiler": {"ok": bool(torch_phase.get("ok")), "error": torch_phase.get("error"),
                                "traces": torch_traces,
                                "profiler_config": (torch_phase.get("server") or {}).get("profiler_config")},
             "rocprof": {"ok": bool(rocprof_phase.get("ok")), "error": rocprof_phase.get("error"),
                         "tool": ((raw.get("tools") or {}).get("rocprofv3") or {}).get("path"),
                         "version": ((raw.get("tools") or {}).get("rocprofv3") or {}).get("version"),
                         "files": rocprof_phase.get("files"), "clock": rocprof_clock,
                         "kernels_read": len(rocprof_kernels), "stop": (rocprof_phase.get("server") or {}).get("stop")}}
    for name, phase in (("torch profiler", torch_phase), ("rocprof", rocprof_phase)):
        if phase and not phase.get("ok"):
            open_items.append(f"{name}: {phase.get('error') or 'did not run'}")
    torch_mem = [a.get("memory") or {} for a in analyses if a.get("source") == "torch" and a.get("available")]
    peaks = [m.get("whole_trace") or {} for m in torch_mem]
    memory = {
        "torch_profiler": {"peak_allocated_gib": max((p.get("peak_allocated_gib") or 0 for p in peaks), default=None)
                           or None,
                           "peak_reserved_gib": max((p.get("peak_reserved_gib") or 0 for p in peaks), default=None)
                           or None, "samples": sum(p.get("samples") or 0 for p in peaks)},
        "serve_log": {**((torch_phase.get("server") or {}).get("memory_log") or {}),
                      "kv_cache_tokens": ((torch_phase.get("server") or {}).get("log_facts") or {}).get(
                          "kv_cache_tokens")},
        "amd_smi": torch_phase.get("amdsmi_memory")}
    facts = raw.get("facts") or {}
    started = str(raw.get("started") or "")
    engine = raw.get("engine") or {}
    return {
        "schema": SCHEMA, "task": TASK, "run_id": raw.get("run_id"), "date": started[:10] or None,
        "started": raw.get("started"), "ended": raw.get("ended"), "node": raw.get("node") or NODE,
        "hub": raw.get("hub"), "account": raw.get("account"), "chip": facts.get("chip") or CHIP,
        "gpu": facts.get("market_name"), "host": {k: facts.get(k) for k in ("hostname", "kernel_release",
                                                                            "amdgpu_driver")},
        "versions": raw.get("versions"),
        "engine": {"vllm_version": engine.get("vllm_version"), "vllm_commit": engine.get("vllm_commit"),
                   "argv": (torch_phase.get("server") or {}).get("argv"),
                   "attention_backends": ((torch_phase.get("server") or {}).get("log_facts") or {}).get(
                       "attention_backends")},
        "checkpoint": raw.get("checkpoint"), "workload": raw.get("workload"),
        "quant_kernel": quant_kernel(raw, analyses),
        "tools": tools, "batches": batches, "memory": memory,
        "power": {"idle_no_server_w": (raw.get("power") or {}).get("idle_no_server_w"),
                  "loaded_idle_w": (torch_phase.get("server") or {}).get("loaded_idle_w"),
                  "period_s": (raw.get("power") or {}).get("period_s")},
        "categories": categorize.describe(),
        "reference": None, "open": open_items,
    }


# --- the chip sheet -------------------------------------------------------------------------------------------------

def chip_reference(path: str | Path | None = None) -> dict[str, Any]:
    """The copy bandwidth to compare against. Without a path: the newest chipsheet-gfx1151-<date>.json in the data
    folder, else the sample sheet (lab/placement.py's default_chipsheet). R33's measured sheet (atlas.chipsheet/1)
    gives its device-to-device copy over the largest buffer (chipsheet_format.copy_largest: smaller buffers stay in
    the 32 MiB last-level cache and overstate memory several times; a self-check sheet holds invented timings and is
    never used); any other sheet is read the way the placement plan reads it (bandwidth.copy_d2d_peak_gb_s, else
    the best bandwidth.copy_d2d row), and the sample sheet's numbers are never taken as a measurement."""
    from lab import placement   # the box only: collect and --quick run this, the pod never does
    from lab.kernels.chipsheet import chipsheet_format as sheet_format

    target = Path(path) if path is not None else placement.default_chipsheet(CHIP)
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = None
    if isinstance(data, Mapping) and data.get("format") == sheet_format.FORMAT:
        name = placement.relative(target)
        out = {"chip_sheet": name, "sample": False, "source": None, "kind": "measured", "gb_s": None, "note": None,
               "mode": data.get("mode")}
        best = sheet_format.copy_largest(data)
        if data.get("mode") == "self-check":
            out.update({"kind": "self-check", "note": f"{name} is a self-check sheet: its timings are invented"})
        elif best is None:
            out["note"] = f"{name} carries no measured device-to-device copy"
        else:
            out.update({"gb_s": float(best["value"]), "source": f"{name}: {best['id']} ({best.get('what')}, the "
                                                                 "largest buffer measured; bytes read and written "
                                                                 "over the time)"})
        return out
    sheet = placement.read_chipsheet(target)
    found = sheet.get("bandwidth_gb_s")
    out = {"chip_sheet": sheet.path, "sample": sheet.sample, "source": found.source, "kind": found.kind,
           "gb_s": None, "note": None}
    if sheet.problem:
        out["note"] = sheet.problem
    elif sheet.sample:
        out["note"] = (f"only the sample chip sheet ({Path(sheet.path).name}) was on main when the result was "
                       f"collected: R33 has not committed a measured chipsheet-{CHIP}-<date>.json")
    elif not found.known:
        out["note"] = f"{sheet.path} carries no copy bandwidth"
    else:
        out["gb_s"] = found.value
    return out


def apply_reference(result: dict[str, Any], reference: Mapping[str, Any]) -> None:
    """Put the chip-sheet bandwidth beside each batch's achieved bandwidth, as a fraction of it."""
    result["reference"] = dict(reference)
    ref = reference.get("gb_s")
    for entry in (result.get("batches") or {}).values():
        bw = entry.get("bandwidth") or {}
        bw["reference_gb_s"] = ref
        for key in ("achieved_gb_s_step", "achieved_gb_s_gpu_busy", "achieved_gb_s_w4a16"):
            bw[key.replace("achieved_gb_s", "fraction_of_reference")] = (
                _r(bw[key] / ref, 4) if ref and bw.get(key) else None)
    result["open"] = [o for o in result.get("open", []) if not o.startswith("chip sheet:")]
    if not ref:
        result["open"].append(f"chip sheet: {reference.get('note')}")


# --- checks and the files -------------------------------------------------------------------------------------------

def _is_dim3(value: Any) -> bool:
    return value is None or (isinstance(value, list) and len(value) == 3 and all(
        isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in value))


def check_result(result: Any) -> list[str]:
    """Every way a result breaks its format; empty when it is sound."""
    problems: list[str] = []
    if not isinstance(result, Mapping):
        return ["the result is not a JSON object"]
    if result.get("schema") != SCHEMA:
        problems.append(f"schema must be {SCHEMA!r}, not {result.get('schema')!r}")
    for key in ("run_id", "node", "chip"):
        if not isinstance(result.get(key), str) or not result.get(key):
            problems.append(f"{key} must be a non-empty string")
    if not isinstance(result.get("date"), str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", result.get("date") or ""):
        problems.append("date must be YYYY-MM-DD")
    for key in ("quant_kernel", "tools", "memory", "power", "batches"):
        if not isinstance(result.get(key), Mapping):
            problems.append(f"{key} must be an object")
    if not isinstance(result.get("open"), list):
        problems.append("open must be a list")
    batches = result.get("batches") if isinstance(result.get("batches"), Mapping) else {}
    for batch in BATCHES:
        entry = batches.get(str(batch))
        if not isinstance(entry, Mapping):
            problems.append(f"batches.{batch} is missing")
            continue
        where = f"batches.{batch}"
        if not isinstance(entry.get("available"), bool):
            problems.append(f"{where}.available must be true or false")
        if not isinstance(entry.get("step_ms"), Mapping):
            problems.append(f"{where}.step_ms must be an object")
        if entry.get("available"):
            shares = entry.get("shares")
            if not isinstance(shares, Mapping):
                problems.append(f"{where}.shares must be an object")
            else:
                for key in SHARE_KEYS:
                    value = shares.get(key)
                    if value is not None and (not isinstance(value, (int, float)) or not 0 <= value <= MAX_SHARE):
                        problems.append(f"{where}.shares.{key} must be a number from 0 to {MAX_SHARE}, not {value!r}")
                    if key not in shares:
                        problems.append(f"{where}.shares.{key} is missing")
        rows = entry.get("kernels_top")
        if not isinstance(rows, list):
            problems.append(f"{where}.kernels_top must be a list")
            continue
        if len(rows) > TOP_KERNELS:
            problems.append(f"{where}.kernels_top has {len(rows)} rows; at most {TOP_KERNELS}")
        totals = []
        for i, row in enumerate(rows, 1):
            if not isinstance(row, Mapping):
                problems.append(f"{where}.kernels_top[{i}] is not an object")
                continue
            if row.get("rank") != i:
                problems.append(f"{where}.kernels_top[{i}].rank must be {i}")
            if not isinstance(row.get("name"), str) or not row.get("name"):
                problems.append(f"{where}.kernels_top[{i}].name must be a non-empty string")
            if not isinstance(row.get("launches"), int) or row.get("launches", 0) < 1:
                problems.append(f"{where}.kernels_top[{i}].launches must be a whole number of at least 1")
            if not _is_dim3(row.get("grid")) or not _is_dim3(row.get("block")):
                problems.append(f"{where}.kernels_top[{i}] grid and block must be [x, y, z] or null")
            if not isinstance(row.get("category"), str):
                problems.append(f"{where}.kernels_top[{i}].category must be a string")
            totals.append(row.get("total_ms") if isinstance(row.get("total_ms"), (int, float)) else 0.0)
        if any(b > a for a, b in zip(totals, totals[1:])):
            problems.append(f"{where}.kernels_top must be sorted by total_ms, biggest first")
    return problems


def _fmt(value: Any, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    return f"{value}{suffix}"


def _pct(value: Any) -> str:
    return "-" if value is None else f"{100.0 * value:.1f}%"


def _dims(value: Any) -> str:
    return "-" if not value else "x".join(str(v) for v in value)


def _cell(text: Any, limit: int = 70) -> str:
    out = str(text).replace("|", "/").replace("\n", " ")
    return out if len(out) <= limit else out[:limit - 3] + "..."


def render_markdown(result: Mapping[str, Any]) -> str:
    """The Markdown summary beside the JSON: the numbers a reader looks for, each with where it came from."""
    lines: list[str] = []
    add = lines.append
    work = result.get("workload") or {}
    prompt_set = work.get("prompt_set") or {}
    engine = result.get("engine") or {}
    ckpt = result.get("checkpoint") or {}
    add(f"# Decode profile of the stock teacher on {result.get('chip')} ({result.get('date')})")
    add("")
    add(f"Run `{result.get('run_id')}` on node {result.get('node')} (hub {result.get('hub')}, account "
        f"{result.get('account')}), {result.get('gpu') or 'the Strix Halo GPU'}: vLLM "
        f"{engine.get('vllm_version') or '-'} at {engine.get('vllm_commit') or '-'}, checkpoint "
        f"{ckpt.get('name') or '-'}, prompt set {prompt_set.get('id') or '-'} "
        f"(sha256 {str(prompt_set.get('sha256') or '-')[:12]}), {work.get('max_tokens') or '-'} tokens per request, "
        f"greedy, thinking off. Batch 1 is one request per prompt, one at a time; batch 4 is four streams at once.")
    add("")
    add("## Tools")
    add("")
    add("| Tool | Result |")
    add("|---|---|")
    tools = result.get("tools") or {}
    torch_t, roc_t = tools.get("torch_profiler") or {}, tools.get("rocprof") or {}
    torch_said = (f"ok, {len(torch_t.get('traces') or [])} traces" if torch_t.get("ok")
                  else "failed: " + _cell(torch_t.get("error"), 160))
    roc_said = (f"ok, {roc_t.get('kernels_read')} kernel records, clock {roc_t.get('clock')}" if roc_t.get("ok")
                else "failed: " + _cell(roc_t.get("error"), 160))
    add(f"| torch profiler (vLLM --profiler-config, /start_profile and /stop_profile) | {torch_said} |")
    add(f"| rocprofv3 ({_cell(roc_t.get('version') or roc_t.get('tool') or 'not found', 60)}) | {roc_said} |")
    add("")
    add("## Time per decode step")
    add("")
    add("Milliseconds. client: the gaps between streamed tokens of the unprofiled runs; torch: vLLM's step ranges in "
        "the profiled run (profiler overhead included); rocprof: decode windows cut at a once-per-step kernel.")
    add("")
    add("| Batch | Source | n | mean | p50 | p90 | p99 | sd | min | max |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    batches = result.get("batches") or {}
    for batch in BATCHES:
        entry = batches.get(str(batch)) or {}
        for source, values in (entry.get("step_ms") or {}).items():
            if not values:
                continue
            add(f"| {batch} | {source} | {values.get('n')} | {_fmt(values.get('mean'), 1)} | "
                f"{_fmt(values.get('p50'), 1)} | {_fmt(values.get('p90'), 1)} | {_fmt(values.get('p99'), 1)} | "
                f"{_fmt(values.get('sd'), 1)} | {_fmt(values.get('min'), 1)} | {_fmt(values.get('max'), 1)} |")
    rates = [f"batch {b}: {_fmt((batches.get(str(b)) or {}).get('tok_s', {}).get('client_unprofiled'), 2)} tok/s"
             for b in BATCHES]
    add("")
    add(f"Decode rate, unprofiled: {'; '.join(rates)}.")
    add("")
    add("## Where a decode step goes")
    add("")
    add("Share of the step time, per batch, from the trace named in Source. Host-side is the part of the step in "
        "which the GPU ran nothing (scheduling, launches, the wait for the client).")
    add("")
    add("| Batch | Source | step ms | W4A16 matmul | Attention | Sampling | Host-side | Copies | Other GPU |")
    add("|---|---|---|---|---|---|---|---|---|")
    for batch in BATCHES:
        entry = batches.get(str(batch)) or {}
        for name, a in (entry.get("sources") or {}).items():
            if not a or not a.get("available"):
                continue
            s, ms = a.get("shares") or {}, a.get("ms_per_step") or {}
            mark = " (used)" if name == entry.get("breakdown_source") else ""
            add(f"| {batch} | {name}{mark} | {_fmt(ms.get('step'), 1)} | {_pct(s.get('w4a16_matmul'))} | "
                f"{_pct(s.get('attention'))} | {_pct(s.get('sampling'))} | {_pct(s.get('host_side'))} | "
                f"{_pct(s.get('copies'))} | {_pct(s.get('other_gpu'))} |")
    for batch in BATCHES:
        entry = batches.get(str(batch)) or {}
        rows = entry.get("kernels_top") or []
        add("")
        add(f"## Top kernels, batch {batch}")
        add("")
        if not rows:
            add("No kernel table: no trace of this batch was readable.")
            continue
        add(f"From the {entry.get('kernels_top_source')} trace, by total time over the decode steps. Grid in "
            "workgroups, block in work-items.")
        add("")
        add("| # | Kernel | Category | Launches | per step | ms/step | Share of step | Grid | Block |")
        add("|---|---|---|---|---|---|---|---|---|")
        for row in rows:
            add(f"| {row.get('rank')} | `{_cell(row.get('name'))}` | {row.get('category')} | {row.get('launches')} | "
                f"{_fmt(row.get('launches_per_step'), 1)} | {_fmt(row.get('ms_per_step'), 3)} | "
                f"{_pct(row.get('share_of_step'))} | {_dims(row.get('grid'))} | {_dims(row.get('block'))} |")
    quant = result.get("quant_kernel") or {}
    add("")
    add("## The quantized matmul kernel")
    add("")
    add(f"vLLM chose `{quant.get('vllm_class') or '-'}` ({quant.get('vllm_class_source') or 'source not read'}) "
        f"for {', '.join(sorted((quant.get('scheme_to_kernel') or {}).keys())) or 'the W4A16 layers'}; conch "
        f"{quant.get('conch_version') or '-'}.")
    add("")
    for k in quant.get("gpu_kernels") or []:
        seen = "; ".join(f"{where}: {_pct(v.get('share_of_step'))} of a step, {_fmt(v.get('launches_per_step'), 1)} "
                         f"launches, grid {_dims(v.get('grid'))}, block {_dims(v.get('block'))}"
                         for where, v in sorted((k.get("seen_in") or {}).items()))
        add(f"- `{k.get('name')}` in {k.get('source') or 'an unknown file'}: {seen or 'not seen in the traces'}")
    for line in quant.get("evidence") or []:
        add(f"- evidence: {line}")
    ref = result.get("reference") or {}
    add("")
    add("## Memory bandwidth")
    add("")
    said = (f"{_fmt(ref.get('gb_s'), 1, ' GB/s')}, the chip sheet's copy bandwidth ({ref.get('source')})"
            if ref.get("gb_s") else f"none ({ref.get('note') or 'no chip sheet read'})")
    add("Bytes a decode step reads: weights only (every tensor but the vision tower, the MTP head and the gathered "
        f"embedding rows; KV-cache reads not counted). Reference: {said}.")
    add("")
    add("| Batch | GB per step | GB/s over the step | over GPU busy | W4A16 kernel GB/s | fraction of reference |")
    add("|---|---|---|---|---|---|")
    for batch in BATCHES:
        bw = (batches.get(str(batch)) or {}).get("bandwidth") or {}
        gb = bw.get("bytes_per_step") / 1e9 if bw.get("bytes_per_step") else None
        add(f"| {batch} | {_fmt(gb, 2)} | {_fmt(bw.get('achieved_gb_s_step'), 1)} | "
            f"{_fmt(bw.get('achieved_gb_s_gpu_busy'), 1)} | {_fmt(bw.get('achieved_gb_s_w4a16'), 1)} | "
            f"{_pct(bw.get('fraction_of_reference_step'))} |")
    mem = result.get("memory") or {}
    tm, log = mem.get("torch_profiler") or {}, mem.get("serve_log") or {}
    add("")
    add("## Memory")
    add("")
    gib = lambda value: _fmt(value, 2, " GiB")  # noqa: E731
    add(f"- torch profiler, GPU allocator over the profiled runs: peak allocated {gib(tm.get('peak_allocated_gib'))}, "
        f"peak reserved {gib(tm.get('peak_reserved_gib'))} ({tm.get('samples')} samples)")
    add(f"- serve log: weights {gib(log.get('weights_gib'))}, KV cache {gib(log.get('kv_cache_gib'))} "
        f"({log.get('kv_cache_tokens') or '-'} tokens), graph pool {gib(log.get('graph_pool_gib'))}")
    smi = mem.get("amd_smi") or {}
    if smi.get("values"):
        add("- amd-smi metric --mem-usage while loaded: " + ", ".join(f"{k} {v}" for k, v in smi["values"].items()))
    power = result.get("power") or {}
    add("")
    add("## Watts")
    add("")
    add(f"amd-smi SOCKET_POWER every {_fmt(power.get('period_s'), 1)} s. Idle with no server "
        f"{_fmt(power.get('idle_no_server_w'), 1, ' W')}, loaded idle {_fmt(power.get('loaded_idle_w'), 1, ' W')}.")
    add("")
    add("| Batch | Run | average W | peak W | joules | samples |")
    add("|---|---|---|---|---|---|")
    for batch in BATCHES:
        for run, w in ((batches.get(str(batch)) or {}).get("watts") or {}).items():
            if w:
                add(f"| {batch} | {run} | {_fmt(w.get('avg_w'), 1)} | {_fmt(w.get('peak_w'), 1)} | "
                    f"{_fmt(w.get('energy_j'), 1)} | {w.get('samples', '-')} |")
    add("")
    add("## Open items")
    add("")
    for item in result.get("open") or ["none"]:
        add(f"- {item}")
    add("")
    return "\n".join(lines)


def result_paths(result: Mapping[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    stem = f"{RESULT_PREFIX}-{result['date']}"
    out = Path(out_dir)
    return out / f"{stem}.json", out / f"{stem}.md"


def write_outputs(result: Mapping[str, Any], out_dir: str | Path, *, force: bool = False) -> tuple[Path, Path]:
    """decode-profile-gfx1151-<date>.json and .md under out_dir; an existing pair is kept unless force."""
    problems = check_result(result)
    if problems:
        raise ProfileError("the result fails its checks: " + "; ".join(problems[:5]))
    json_path, md_path = result_paths(result, out_dir)
    if not force and (json_path.exists() or md_path.exists()):
        raise ProfileError(f"{json_path.name} already exists; pass --force to replace it")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    md_path.write_text(render_markdown(result), encoding="utf-8", newline="\n")
    return json_path, md_path


# --- the bundle that crosses the terminal ---------------------------------------------------------------------------

def make_bundle(files: Mapping[str, bytes]) -> bytes:
    """A gzip'd tar of name -> bytes, the same bytes for the same files (sorted names, no times, no owners)."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name in sorted(files):
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(files[name]), 0, 0o644
            tar.addfile(info, io.BytesIO(files[name]))
    return gzip.compress(raw.getvalue(), compresslevel=9, mtime=0)


def bundle_text(payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    body = base64.b64encode(payload).decode("ascii")
    lines = [body[i:i + B64_LINE] for i in range(0, len(body), B64_LINE)]
    return f"{MARK_BEGIN} sha256={digest} bytes={len(payload)} ===\n" + "\n".join(lines) + f"\n{MARK_END}\n"


def decode_text(data: bytes) -> str:
    """Saved terminal output as text, whatever the shell wrote: UTF-8 (with or without a BOM) or UTF-16."""
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16")
    if data[:3] == b"\xef\xbb\xbf":
        return data[3:].decode("utf-8", "replace")
    if len(data) > 1 and data[1:2] == b"\x00":
        return data.decode("utf-16-le", "replace")
    return data.decode("utf-8", "replace")


def read_bundle(data: bytes) -> dict[str, bytes]:
    """name -> bytes from a bundle: the saved terminal output (the last marked block), the bare base64 or the
    .tar.gz itself. The sha256 and length in the marker line must match."""
    if data[:2] == b"\x1f\x8b":
        payload = data
    else:
        text = decode_text(data).replace("\r", "")
        begin = text.rfind(MARK_BEGIN)
        if begin < 0:
            raise ProfileError(f"no '{MARK_BEGIN}' line in the input: the pod run did not finish, or the output was "
                               "not saved; the bundle is also on the pod as <run dir>/result_bundle.b64")
        header_end = text.find("\n", begin)
        header = text[begin:header_end]
        end = text.find(MARK_END, header_end)
        if end < 0:
            raise ProfileError("the result block has no end line: the output was cut short")
        body = re.sub(r"[^A-Za-z0-9+/=]", "", text[header_end:end])
        payload = base64.b64decode(body)
        m = re.search(r"sha256=([0-9a-f]{64}) bytes=(\d+)", header)
        if not m:
            raise ProfileError("the result block's first line carries no sha256 and length")
        if len(payload) != int(m.group(2)) or hashlib.sha256(payload).hexdigest() != m.group(1):
            raise ProfileError("the result block does not match its sha256 and length: it was damaged in transit")
    out: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(payload)), mode="r") as tar:
        for member in tar.getmembers():
            if not member.isfile() or member.name.startswith(("/", "..")) or "/../" in member.name:
                continue
            handle = tar.extractfile(member)
            if handle is not None:
                out[member.name] = handle.read()
    return out


def _head_tail(text: str, head: int = 300, tail: int = 150) -> str:
    lines = text.splitlines()
    if len(lines) <= head + tail:
        return text
    return "\n".join(lines[:head] + [f"... {len(lines) - head - tail} lines left out ..."] + lines[-tail:]) + "\n"


def excerpt_files(raw: Mapping[str, Any], base: Path, *, steps: int = 2) -> dict[str, bytes]:
    """A few decode steps of each trace, to keep as recorded fixtures: the batch-1 torch trace cut to `steps`
    steps, and the rocprof kernel rows of the first batch-1 decode window cut to `steps` steps."""
    out: dict[str, bytes] = {}
    phases = raw.get("phases") or {}
    run = ((phases.get("torch") or {}).get("runs") or {}).get("b1_torch") or {}
    if run.get("trace"):
        try:
            data = tp.load_json(base / run["trace"])
            trace = tp.TorchTrace(data)
            found = torch_decode_steps(trace, 1)[:steps]
            if found:
                cut = tp.trim_torch_trace(data, found[0].start_us, found[-1].end_us)
                out["excerpts/torch_b1.pt.trace.json.gz"] = gzip.compress(json.dumps(cut).encode(), mtime=0)
        except (OSError, ValueError):
            pass
    rp = phases.get("rocprof") or {}
    run = (rp.get("runs") or {}).get("b1_rocprof") or {}
    windows = run_windows(run) if run else []
    offsets = raw.get("clocks_start") or {}
    for rel in (rp.get("files") or {}).get("kernel", [])[:1]:
        try:
            kernels = tp.read_rocprof_kernels(base / rel)
        except (OSError, ValueError):
            continue
        clock, _ = choose_clock(kernels, offsets, windows) if windows else (None, {})
        if clock is None:
            continue
        s, e, n = windows[0]
        a = to_clock_us(s, offsets, clock)
        b = a + (to_clock_us(e, offsets, clock) - a) * min(1.0, steps / max(n, 1))
        with open(base / rel, newline="", encoding="utf-8", errors="replace") as handle:
            rows = [row for row in csv.DictReader(handle)
                    if a * 1000.0 <= float(row.get("Start_Timestamp") or 0) < b * 1000.0]
        out["excerpts/rocprof_b1_kernel_trace.csv"] = tp.write_rocprof_kernels(rows).encode()
        out["excerpts/rocprof_window.json"] = json.dumps({"clock": clock, "start_us": a, "end_us": b,
                                                          "steps": min(steps, n)}).encode()
    return out


def pod_bundle(raw: Mapping[str, Any], result: Mapping[str, Any] | None, base: Path,
               error: str | None = None) -> dict[str, bytes]:
    files: dict[str, bytes] = {"raw.json": json.dumps(raw, default=str).encode()}
    if result is not None:
        files["result.json"] = json.dumps(result, indent=1).encode()
        files["summary_pod.md"] = render_markdown(result).encode()
    if error:
        files["analysis_error.txt"] = error.encode()
    for name in ("serve-torch.log", "serve-rocprof.log"):
        path = base / name
        if path.exists():
            files[f"logs/{name}"] = _head_tail(path.read_text(encoding="utf-8", errors="replace")).encode()
    try:
        files.update(excerpt_files(raw, base))
    except Exception as exc:  # an excerpt is a convenience; the result must still come back
        files["excerpts/error.txt"] = f"{type(exc).__name__}: {exc}".encode()
    return files


# --- the script the pod runs ----------------------------------------------------------------------------------------

def _module_file(name: str, root: Path) -> Path | None:
    base = root.joinpath(*name.split("."))
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    return None


# imported only by collect and --quick on the box (the chip sheet readers), never on the pod; a prefix covers the
# package and everything under it
BOX_ONLY = ("lab.placement", "lab.kernels")


def module_closure(start: str = "lab.profile_decode", root: Path = REPO) -> list[Path]:
    """Every lab module the start module imports, directly or not (function-level imports too), with the package
    __init__ files: what must be on the pod for `python3 -m lab.profile_decode` to run there. BOX_ONLY modules and
    what only they import stay behind."""
    seen: dict[str, Path] = {}
    todo = [start, "lab.profile_decode_pod"]
    while todo:
        name = todo.pop()
        if name in seen or not (name == "lab" or name.startswith("lab.")) or any(
                name == box or name.startswith(box + ".") for box in BOX_ONLY):
            continue
        path = _module_file(name, root)
        if path is None:
            continue
        seen[name] = path
        parts = name.split(".")
        todo.extend(".".join(parts[:i]) for i in range(1, len(parts)))
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                todo.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                todo.append(node.module)
                todo.extend(f"{node.module}.{a.name}" for a in node.names)
    return sorted(set(seen.values()))


def data_files(root: Path = REPO) -> list[Path]:
    """The prompt set lab/bench.py loads and the images it names."""
    from lab import bench

    prompts = Path(bench.DEFAULT_PROMPTS)
    data = json.loads(prompts.read_text(encoding="utf-8"))
    images = [prompts.parent / p["image"] for p in data.get("prompts", []) if p.get("kind") == "image"]
    return [prompts, *images]


def pod_script(*, expect_user_sha256: str, args: Sequence[str], root: Path = REPO) -> tuple[str, dict[str, Any]]:
    """The bash script the pod runs (pod_exec.py's run mode uploads it, or term mode types it): the lab files it
    needs as one base64 block (only letters, digits, + / =, lines of 76, so no shell or terminal reads anything
    into it), checked against its sha256, unpacked under ~/r47_decode_profile/bundle-<sha>, then pod-run there. Its
    last line is `### END rc=<code>`, the line run mode waits for."""
    files = {str(p.relative_to(root)).replace("\\", "/"): p.read_bytes() for p in (*module_closure(root=root),
                                                                                    *data_files(root))}
    payload = make_bundle(files)
    digest = hashlib.sha256(payload).hexdigest()
    body = base64.b64encode(payload).decode("ascii")
    lines = [body[i:i + B64_LINE] for i in range(0, len(body), B64_LINE)]
    run_args = " ".join(shlex.quote(a) for a in ["--expect-user-sha256", expect_user_sha256, *args])
    folder = f'"$HOME/r47_decode_profile/bundle-{digest[:12]}"'
    script = "\n".join([
        "# Decode profile of the stock teacher (R47), made by python -m lab.profile_decode pod-script; run with bash.",
        "# It unpacks the lab files below, then runs the profile; the result comes back as one marked base64 block.",
        "# No exclamation mark and no tab anywhere: the script may be typed into an interactive shell.",
        "set -u",
        f"B={folder}",
        'mkdir -p "$B"',
        "base64 -d > \"$B/bundle.tar.gz\" <<'R47BUNDLE'",
        *lines,
        "R47BUNDLE",
        f'echo "{digest}  $B/bundle.tar.gz" | sha256sum -c --quiet - '
        '|| { echo "R47 bundle damaged in transit"; exit 3; }',
        'tar -xzf "$B/bundle.tar.gz" -C "$B" || exit 3',
        'python3 -c "import httpx" 2>/dev/null || pip install --user --quiet httpx',
        'cd "$B" || exit 3',
        f"python3 -m lab.profile_decode pod-run {run_args}",
        "rc=$?",
        'echo "### R47 pod-run rc=$rc"',
        'echo "### END rc=$rc"',
        "exit $rc",
        ""])
    info = {"files": sorted(files), "bundle_bytes": len(payload), "bundle_sha256": digest,
            "script_bytes": len(script.encode()), "lines": script.count("\n")}
    return script, info


# --- verbs ----------------------------------------------------------------------------------------------------------

def _say(message: str) -> None:
    print(f"[r47 {dt.datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def analyze_dir(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any] | None, str | None]:
    raw = json.loads((run_dir / "raw.json").read_text(encoding="utf-8"))
    try:
        result = build_result(raw, run_dir)
    except Exception as exc:  # the raw data must still come back for a second look
        return raw, None, f"{type(exc).__name__}: {exc}"
    return raw, result, None


def pod_run_main(args: argparse.Namespace) -> dict[str, Any]:
    from lab import profile_decode_pod as pod

    from lab import bench

    problems = pod.gate(expect_user_sha256=args.expect_user_sha256, environ=os.environ, rocminfo=pod.read_rocminfo,
                        port=args.port, attach_port=args.attach_port,
                        servers=lambda: pod.vllm_pids(pod.proc_table(bench.Host())))
    if problems:
        return {"ok": False, "outcome": "refused", "problems": problems}

    def stop_here(signum: int, frame: Any) -> None:
        raise SystemExit(f"signal {signum}")

    for name in ("SIGHUP", "SIGTERM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), stop_here)
    return run_and_report(pod.PodRun(args))


def run_and_report(run: Any) -> dict[str, Any]:
    """Run the profile, build the result, keep it in the run folder and print it as the marked block."""
    _say(f"run {run.run_id} in {run.dir}")
    run.run()
    raw, result, error = analyze_dir(run.dir)
    if result is not None:
        (run.dir / "result.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
        (run.dir / "summary_pod.md").write_text(render_markdown(result), encoding="utf-8", newline="\n")
    text = bundle_text(make_bundle(pod_bundle(raw, result, run.dir, error)))
    (run.dir / "result_bundle.b64").write_text(text, encoding="utf-8", newline="\n")
    print(text, flush=True)
    phases = raw.get("phases") or {}
    return {"ok": result is not None and any(p.get("ok") for p in phases.values()), "outcome": "profiled",
            "run_dir": str(run.dir), "analysis_error": error,
            "phases": {k: {"ok": v.get("ok"), "error": v.get("error")} for k, v in phases.items()}}


def analyze_main(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir).expanduser()
    raw, result, error = analyze_dir(run_dir)
    if result is None:
        return {"ok": False, "outcome": "error", "error": error}
    out = Path(args.out) if args.out else run_dir
    json_path, md_path = write_outputs(result, out, force=True)
    return {"ok": True, "outcome": "analyzed", "json": str(json_path), "markdown": str(md_path),
            "open": result["open"]}


def pod_script_main(args: argparse.Namespace) -> dict[str, Any]:
    from lab import hub
    from lab import profile_decode_pod as pod

    if args.expect_user_sha256:
        expect = args.expect_user_sha256
    else:
        try:
            account = hub.load_accounts([args.account])[args.account]
        except hub.MissingCredentials as exc:
            raise ProfileError(f"{exc}: pod-script reads account {args.account}'s user name from the workspace .env "
                               "(or pass --expect-user-sha256)") from exc
        expect = pod.user_sha256(account.user)
    run_args = ["--phases", ",".join(args.phases), "--max-tokens", str(args.max_tokens), "--port", str(args.port),
                "--checkpoint", args.checkpoint, "--deadline-min", f"{args.deadline_min:g}", "--hub", args.hub,
                "--account", args.account]
    if args.attach_port is not None:
        run_args += ["--attach-port", str(args.attach_port), "--serve-log", args.serve_log]
    script, info = pod_script(expect_user_sha256=expect, args=run_args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(script.encode("ascii"))
    return {"ok": True, "outcome": "written", "script": str(out), **info,
            "next": f"python tooling/dispatcher/pod_exec.py {args.account} term <this script> 3000 > <output file>",
            "or": f"python tooling/dispatcher/pod_exec.py {args.account} run <this script> r47_pod_run.sh 3000 "
                  "> <output file>"}


def ledger_rows(result: Mapping[str, Any], source: str) -> list[dict[str, Any]]:
    """The rows `collect` writes: one per batch, one per tool, one for the run; node pod-b, low confidence (one
    spawn per tool)."""
    rows: list[dict[str, Any]] = []
    base_conditions = {"chip": result.get("chip"), "hub": result.get("hub"), "account": result.get("account"),
                       "vllm_commit": (result.get("engine") or {}).get("vllm_commit"),
                       "checkpoint": (result.get("checkpoint") or {}).get("name"),
                       "prompt_set": ((result.get("workload") or {}).get("prompt_set") or {}).get("id"),
                       "max_tokens": (result.get("workload") or {}).get("max_tokens"), "run_id": result.get("run_id")}
    p = METRIC_PREFIX
    for batch in BATCHES:
        entry = (result.get("batches") or {}).get(str(batch)) or {}
        client = (entry.get("step_ms") or {}).get("client_unprofiled") or {}
        shares = entry.get("shares") or {}
        bw = entry.get("bandwidth") or {}
        watts = ((entry.get("watts") or {}).get("unprofiled")) or {}
        metrics = {f"{p}.decode_step_ms": client.get("mean"), f"{p}.decode_step_ms_p50": client.get("p50"),
                   f"{p}.decode_step_ms_p90": client.get("p90"), f"{p}.decode_step_ms_sd": client.get("sd"),
                   f"{p}.decode_tok_s": (entry.get("tok_s") or {}).get("client_unprofiled"),
                   **{f"{p}.decode_share_{k}": shares.get(k) for k in SHARE_KEYS},
                   f"{p}.decode_achieved_gb_s": bw.get("achieved_gb_s_step"),
                   f"{p}.decode_w4a16_gb_s": bw.get("achieved_gb_s_w4a16"),
                   f"{p}.decode_watts_avg": watts.get("avg_w"), f"{p}.decode_watts_peak": watts.get("peak_w")}
        metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        units = {k: ("ms" if "_ms" in k else "tok/s" if k.endswith("tok_s") else "GB/s" if k.endswith("gb_s")
                     else "W" if "watts" in k else "fraction") for k in metrics}
        rows.append({"verb": "profile_decode.batch", "outcome": "ok" if entry.get("available") else "fail",
                     "metrics": metrics, "units": units,
                     "conditions": {**base_conditions, "batch": batch,
                                    "breakdown_source": entry.get("breakdown_source"),
                                    "kernels_source": entry.get("kernels_top_source")},
                     "note": None if entry.get("available") else "no trace gave a step breakdown"})
    for name, tool in (result.get("tools") or {}).items():
        rows.append({"verb": f"profile_decode.{name}", "outcome": "ok" if tool.get("ok") else "fail", "metrics": {},
                     "units": None, "conditions": {**base_conditions, "tool": name},
                     "note": None if tool.get("ok") else str(tool.get("error"))[:400]})
    mem = (result.get("memory") or {}).get("torch_profiler") or {}
    metrics = {f"{p}.decode_peak_allocated_gib": mem.get("peak_allocated_gib"),
               f"{p}.decode_peak_reserved_gib": mem.get("peak_reserved_gib")}
    metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    rows.append({"verb": "profile_decode", "outcome": "ok", "metrics": metrics,
                 "units": {k: "GiB" for k in metrics} or None, "conditions": base_conditions,
                 "note": "; ".join(result.get("open") or [])[:480] or None})
    for row in rows:
        row.update({"node": result.get("node") or NODE, "source": source, "low_confidence": True,
                    "refs": {"run_id": result.get("run_id")}})
    return rows


def collect_main(args: argparse.Namespace) -> dict[str, Any]:
    from lab.machine.record import record

    files = read_bundle(Path(args.input).read_bytes())
    if "result.json" not in files:
        error = files.get("analysis_error.txt", b"no result.json in the bundle").decode("utf-8", "replace")
        return {"ok": False, "outcome": "no_result", "error": error, "files": sorted(files)}
    result = json.loads(files["result.json"].decode("utf-8"))
    apply_reference(result, chip_reference(args.chip_sheet))
    problems = check_result(result)
    if problems:
        return {"ok": False, "outcome": "invalid", "problems": problems}
    out_dir = Path(args.out) if args.out else DATA_DIR
    json_path, md_path = write_outputs(result, out_dir, force=args.force)
    try:
        source = str(json_path.resolve().relative_to(REPO)).replace("\\", "/")
    except ValueError:
        source = str(json_path)
    seqs = []
    if not args.no_ledger:
        for row in ledger_rows(result, source):
            seqs.append(record(row["verb"], row["outcome"], row["metrics"], row["refs"], task=TASK, node=row["node"],
                               units=row["units"] or None, conditions=row["conditions"], source=row["source"],
                               low_confidence=row["low_confidence"], note=row["note"]))
    saved = []
    if args.save_fixtures:
        folder = FIXTURES / f"recorded-{result['date']}"
        for name, data in files.items():
            if name.startswith("excerpts/") and not name.endswith("error.txt"):
                path = folder / name.split("/", 1)[1]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                saved.append(str(path))
    return {"ok": True, "outcome": "collected", "json": str(json_path), "markdown": str(md_path),
            "ledger_seqs": seqs, "fixtures": saved, "reference": result.get("reference"), "open": result["open"]}


def quick_main(args: argparse.Namespace) -> dict[str, Any]:
    """The self-check on this machine: the parsers, the result, its checks and both files on the synthetic run
    in fixtures/lab/profile_decode/synthetic, and a bundle round trip; no GPU, no network."""
    fixture = Path(args.fixture) if args.fixture else QUICK_FIXTURE
    raw = json.loads((fixture / "raw.json").read_text(encoding="utf-8"))
    result = build_result(raw, fixture)
    apply_reference(result, chip_reference(fixture / "chipsheet.json"))
    problems = check_result(result)
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="r47-quick-"))
    json_path, md_path = write_outputs(result, out, force=True) if not problems else (None, None)
    checks: dict[str, bool] = {"result_checks": not problems}
    if json_path is not None:
        checks["json_round_trip"] = json.loads(json_path.read_text(encoding="utf-8")) == json.loads(json.dumps(result))
        text = md_path.read_text(encoding="utf-8")
        checks["markdown_sections"] = all(section in text for section in MD_SECTIONS)
    files = pod_bundle(raw, result, fixture)
    checks["bundle_round_trip"] = read_bundle(("noise\r\n" + bundle_text(make_bundle(files))).encode()) == files
    for batch in BATCHES:
        checks[f"batch_{batch}_breakdown"] = bool((result["batches"][str(batch)] or {}).get("available"))
    return {"ok": all(checks.values()), "outcome": "quick", "checks": checks, "problems": problems,
            "json": str(json_path) if json_path else None, "markdown": str(md_path) if md_path else None,
            "fixture": str(fixture)}


def build_parser() -> argparse.ArgumentParser:
    from lab import profile_decode_pod as pod

    p = argparse.ArgumentParser(prog="python -m lab.profile_decode", description=__doc__.splitlines()[0])
    p.add_argument("--quick", action="store_true", help="self-check on this machine against the recorded fixture")
    p.add_argument("--out", default=None, help="--quick: where the JSON and Markdown go (default: a temp folder)")
    p.add_argument("--fixture", default=None, help=argparse.SUPPRESS)
    sub = p.add_subparsers(dest="cmd")

    def phases(text: str) -> list[str]:
        names = [x for x in text.split(",") if x]
        if not names or any(x not in pod.DEFAULT_PHASES for x in names):
            raise argparse.ArgumentTypeError(f"phases are a comma list of {', '.join(pod.DEFAULT_PHASES)}")
        return names

    script = sub.add_parser("pod-script", help="write the script pod_exec.py types into the pod's terminal")
    script.add_argument("--out", default=str(POD_SCRIPT))
    script.add_argument("--account", default=DEFAULT_ACCOUNT)
    script.add_argument("--expect-user-sha256", default=None, help="the pod account's hash, instead of the .env's")
    script.add_argument("--hub", default="sjc")
    script.add_argument("--phases", type=phases, default=list(pod.DEFAULT_PHASES))
    script.add_argument("--max-tokens", type=int, default=pod.DEFAULT_MAX_TOKENS)
    script.add_argument("--port", type=int, default=pod.DEFAULT_PORT)
    script.add_argument("--checkpoint", default=pod.DEFAULT_CHECKPOINT)
    script.add_argument("--deadline-min", type=float, default=pod.DEFAULT_DEADLINE_MIN)
    script.add_argument("--attach-port", type=int, default=None,
                        help="profile the server already running on this port instead of starting one")
    script.add_argument("--serve-log", default=pod.ATTACH_LOG, help="attached: that server's log")

    run = sub.add_parser("pod-run", help="on the pod: run the profile (what the script runs)")
    run.add_argument("--expect-user-sha256", required=True)
    run.add_argument("--phases", type=phases, default=list(pod.DEFAULT_PHASES))
    run.add_argument("--max-tokens", type=int, default=pod.DEFAULT_MAX_TOKENS)
    run.add_argument("--port", type=int, default=pod.DEFAULT_PORT)
    run.add_argument("--checkpoint", default=pod.DEFAULT_CHECKPOINT)
    run.add_argument("--engine-dir", default=pod.DEFAULT_ENGINE_DIR)
    run.add_argument("--prompts", default=None)
    run.add_argument("--out", default=pod.DEFAULT_OUT)
    run.add_argument("--python", default="python3")
    run.add_argument("--deadline-min", type=float, default=pod.DEFAULT_DEADLINE_MIN)
    run.add_argument("--hub", default="sjc")
    run.add_argument("--account", default=DEFAULT_ACCOUNT)
    run.add_argument("--serve-dir", default="~/serve", help="the server's working folder (never the home itself)")
    run.add_argument("--serve-cmd", type=json.loads, default=None, help=argparse.SUPPRESS)  # tests: another server
    run.add_argument("--idle-s", type=float, default=None, help=argparse.SUPPRESS)        # tests: shorter idle reads
    run.add_argument("--rocprof-cmd", type=json.loads, default=None, help=argparse.SUPPRESS)  # tests: a stand-in
    run.add_argument("--attach-port", type=int, default=None,
                     help="profile the server already running on this port: nothing is started or stopped")
    run.add_argument("--serve-log", default=pod.ATTACH_LOG, help="attached: that server's log, for the log facts")
    run.add_argument("--attach-duration-s", type=float, default=pod.ATTACH_DURATION_S,
                     help="attached: how long rocprofv3 --attach traces")

    analyze = sub.add_parser("analyze", help="build the result from a run folder (raw.json and the traces)")
    analyze.add_argument("run_dir")
    analyze.add_argument("--out", default=None)

    collect = sub.add_parser("collect", help="read the pod's result block, write the JSON, Markdown and ledger rows")
    collect.add_argument("input", help="the saved output of the pod run, its result_bundle.b64, or the .tar.gz")
    collect.add_argument("--chip-sheet", default=None,
                         help=f"default: the newest chipsheet-{CHIP}-<date>.json on main, else the sample")
    collect.add_argument("--out", default=None, help="default: lab/refine/research/kernels/data")
    collect.add_argument("--no-ledger", action="store_true")
    collect.add_argument("--save-fixtures", action="store_true", help="keep the trace excerpts as recorded fixtures")
    collect.add_argument("--force", action="store_true", help="replace a result of the same date")
    return p


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Decode profile of the stock teacher on gfx1151: --quick here, pod-script, pod-run on the pod, collect."""
    parser = build_parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.quick:
        return quick_main(args)
    if args.cmd == "pod-run":
        if args.prompts is None:
            from lab import bench

            args.prompts = str(bench.DEFAULT_PROMPTS)
        return pod_run_main(args)
    handlers = {"pod-script": pod_script_main, "analyze": analyze_main, "collect": collect_main}
    if args.cmd not in handlers:
        parser.print_usage(sys.stderr)
        raise SystemExit(2)
    try:
        return handlers[args.cmd](args)
    except (ProfileError, OSError, ValueError) as exc:
        return {"ok": False, "outcome": "error", "error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    answer = main()
    print(json.dumps(answer))
    sys.exit(0 if answer.get("ok") else 1)
