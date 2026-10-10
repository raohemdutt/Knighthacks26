"""Summaries of GPU profiler traces: the torch profiler's Chrome trace and rocprofv3's CSV kernel trace.

Stdlib only and no GPU: everything here reads files a profiler already wrote, so it runs on a pod right after a run
and on this machine against recorded fixtures. The decode profile (lab/profile_decode.py) and any later step profile
(a training step) share it; what a "step" is stays with the caller, which hands in time windows.

The torch profiler (kineto) writes one JSON object whose `traceEvents` hold complete events (`ph` "X", `ts` and `dur`
in microseconds). This module reads:

    cat "kernel"                         a GPU kernel: args.grid and args.block (HIP gridDim and blockDim as the
                                         profiler reported them), args.correlation (the launching runtime call)
    cat "gpu_memcpy" / "gpu_memset"      a copy or memset on the GPU timeline: args.bytes when present
    cat "cuda_runtime" / "cuda_driver"   a runtime call on the CPU (hipLaunchKernel, hipGraphLaunch, hipMemcpy...):
                                         args.correlation ties it to the kernels and copies it started
    cat "user_annotation"                a record_function range on the CPU (vLLM names its steps this way)
    name "[memory]" (ph "i")             an allocator sample: args "Total Allocated", "Total Reserved", "Device Type"

rocprofv3 writes `<name>_kernel_trace.csv` (Kernel_Name, Start_Timestamp and End_Timestamp in nanoseconds,
Workgroup_Size_X/Y/Z, Grid_Size_X/Y/Z in work-items, LDS_Block_Size, VGPR_Count, SGPR_Count, Scratch_Size; older
releases call the LDS and scratch columns Group_Segment_Size and Private_Segment_Size) and
`<name>_memory_copy_trace.csv` (Direction, Start_Timestamp, End_Timestamp). Every time here is a float in
microseconds; rocprof's nanoseconds are divided by 1000 and keep their own clock, so a caller compares them only
with windows moved onto that clock. A grid is always reported in workgroups (HIP's gridDim): rocprof's work-item
grid is divided by the workgroup size, rounding up.

Kernel names are put in categories by an ordered list of (category, regular expression) rules searched without
regard to case; the first that matches wins, and a caller can put exact names (the quantized matmul kernel the
engine chose) in front. Nothing is dropped: a name no rule matches is "other", and the tables list the biggest of
those so a rule can be added where one is missing.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["Kernel", "Copy", "Span", "MemorySample", "TorchTrace", "Categorizer", "DEFAULT_RULES", "load_json",
           "read_rocprof_kernels", "read_rocprof_copies", "find_rocprof_csvs", "union_us", "percentile", "spread",
           "kernel_table", "category_totals", "copy_summary", "memory_peaks", "pick_marker", "trim_torch_trace",
           "write_rocprof_kernels"]

KERNEL_CATS = frozenset({"kernel"})
COPY_CATS = frozenset({"gpu_memcpy", "gpu_memset"})
RUNTIME_CATS = frozenset({"cuda_runtime", "cuda_driver", "hip_runtime", "runtime"})
ANNOTATION_CATS = frozenset({"user_annotation"})
MEMORY_EVENT = "[memory]"
CPU_DEVICE_TYPE = 0          # torch's DeviceType.CPU in "[memory]" samples; anything else is the GPU's allocator

# Ordered: the first rule that matches a kernel name wins.
DEFAULT_RULES: tuple[tuple[str, str], ...] = (
    ("w4a16_matmul", r"^_gemm_kernel$|gptq|exllama|marlin|machete|conch|w4a16|wna16|dequant"),
    ("attention_linear", r"gated_delta|delta_rule|fused_recurrent|recurrent_gated|chunk_\w*(fwd|bwd)|causal_conv1d|"
                         r"short_conv|fla_|gdn|mamba|selective_scan|ssm_"),
    ("attention", r"attention|attn|flash|fmha|paged|reshape_and_cache|concat_and_cache|cache_kernel|"
                  r"reduce_segments|rotary|rope"),
    ("sampling", r"argmax|arg_max|topk|top_k|top_p|sampl|softmax|multinomial|exponential_|gumbel|penalt|logit"),
    ("dense_matmul", r"cijk_|gemm|gemv|matmul|hipblaslt|rocblas|tensile|wvsplitk|llgemm|skinny|bmm"),
    ("norm_activation", r"rms|norm|silu|gelu|swiglu|act_and_mul|activation|sigmoid"),
    ("elementwise", r"elementwise|vectorized|unrolled|copy|fill|cat_|index|gather|scatter|embedding|triton_poi|"
                    r"triton_red|triton_per|reduce|where"),
)
ATTENTION_GROUP = ("attention", "attention_linear")   # what "attention" means in a step's shares


# --- records ------------------------------------------------------------------------------------------------------

Dim3 = tuple[int, int, int]


@dataclass(frozen=True)
class Kernel:
    """One kernel execution on the GPU. grid in workgroups, block in work-items per workgroup."""

    name: str
    start_us: float
    dur_us: float
    grid: Dim3 | None = None
    block: Dim3 | None = None
    correlation: int | None = None
    stream: int | None = None
    lds_bytes: int | None = None
    scratch_bytes: int | None = None
    vgprs: int | None = None
    sgprs: int | None = None

    @property
    def end_us(self) -> float:
        return self.start_us + self.dur_us


@dataclass(frozen=True)
class Copy:
    """One copy or memset on the GPU timeline. kind is HtoD, DtoH, DtoD, HtoH, memset or other."""

    kind: str
    start_us: float
    dur_us: float
    bytes: int | None = None
    correlation: int | None = None

    @property
    def end_us(self) -> float:
        return self.start_us + self.dur_us


@dataclass(frozen=True)
class Span:
    """A CPU-side interval: a record_function annotation or a runtime call."""

    name: str
    start_us: float
    dur_us: float
    cat: str
    correlation: int | None = None
    tid: Any = None

    @property
    def end_us(self) -> float:
        return self.start_us + self.dur_us


@dataclass(frozen=True)
class MemorySample:
    """One allocator sample of the torch profiler (bytes)."""

    ts_us: float
    allocated: int | None
    reserved: int | None
    device_type: int | None
    device_id: int | None


# --- small readers ------------------------------------------------------------------------------------------------

def _int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"[-+]?\d+", text):
            return int(text)
        if re.fullmatch(r"[-+]?\d+\.\d*", text):
            return int(float(text))
    return None


def _float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def dim3(value: Any) -> Dim3 | None:
    """[x, y, z] (or fewer, padded with 1) from a list, a tuple or text such as "[80, 1, 1]"; None otherwise."""
    if isinstance(value, str):
        value = re.findall(r"-?\d+", value)
    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= 3:
        return None
    parts = [_int(v) for v in value]
    if any(p is None or p < 0 for p in parts):
        return None
    parts += [1] * (3 - len(parts))
    return (parts[0], parts[1], parts[2])


def copy_kind(text: str) -> str:
    """HtoD, DtoH, DtoD, HtoH or memset from a torch copy name or a rocprof Direction; other when neither says."""
    t = text.upper().replace(" ", "_")
    if "MEMSET" in t:
        return "memset"
    for short, words in (("HtoD", ("HTOD", "HOST_TO_DEVICE")), ("DtoH", ("DTOH", "DEVICE_TO_HOST")),
                         ("DtoD", ("DTOD", "DEVICE_TO_DEVICE")), ("HtoH", ("HTOH", "HOST_TO_HOST"))):
        if any(w in t for w in words):
            return short
    return "other"


def load_json(path: str | Path) -> Any:
    """A JSON file, gzip-compressed or not (the torch profiler writes .pt.trace.json.gz)."""
    raw = Path(path).read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


# --- the torch profiler's trace -----------------------------------------------------------------------------------

class TorchTrace:
    """The events of one torch profiler trace this module uses, sorted by start time."""

    def __init__(self, data: Mapping[str, Any]) -> None:
        events = data.get("traceEvents") if isinstance(data, Mapping) else None
        if not isinstance(events, list):
            raise ValueError("not a torch profiler trace: no traceEvents list")
        props = data.get("deviceProperties")
        self.device: dict[str, Any] = dict(props[0]) if isinstance(props, list) and props and isinstance(
            props[0], Mapping) else {}
        self.kernels: list[Kernel] = []
        self.copies: list[Copy] = []
        self.runtime: list[Span] = []
        self.annotations: list[Span] = []
        self.memory: list[MemorySample] = []
        self.event_count = len(events)
        for e in events:
            if not isinstance(e, Mapping):
                continue
            name = str(e.get("name", ""))
            args = e.get("args") if isinstance(e.get("args"), Mapping) else {}
            ph = e.get("ph")
            if ph in ("i", "I") and name == MEMORY_EVENT:
                ts = _float(e.get("ts"))
                if ts is not None:
                    self.memory.append(MemorySample(ts, _int(args.get("Total Allocated")),
                                                    _int(args.get("Total Reserved")), _int(args.get("Device Type")),
                                                    _int(args.get("Device Id"))))
                continue
            if ph != "X":
                continue
            ts, dur = _float(e.get("ts")), _float(e.get("dur"))
            if ts is None or dur is None or dur < 0:
                continue
            cat = str(e.get("cat", "")).lower()
            correlation = _int(args.get("correlation"))
            if cat in KERNEL_CATS:
                self.kernels.append(Kernel(name, ts, dur, dim3(args.get("grid")), dim3(args.get("block")),
                                           correlation, _int(args.get("stream", e.get("tid"))),
                                           _int(args.get("shared memory")), None,
                                           _int(args.get("registers per thread")), None))
            elif cat in COPY_CATS:
                kind = "memset" if cat == "gpu_memset" else copy_kind(name)
                self.copies.append(Copy(kind, ts, dur, _int(args.get("bytes")), correlation))
            elif cat in RUNTIME_CATS:
                self.runtime.append(Span(name, ts, dur, cat, correlation, e.get("tid")))
            elif cat in ANNOTATION_CATS:
                self.annotations.append(Span(name, ts, dur, cat, None, e.get("tid")))
        for items in (self.kernels, self.copies, self.runtime, self.annotations):
            items.sort(key=lambda x: x.start_us)
        self.memory.sort(key=lambda m: m.ts_us)
        self._launch: dict[int, float] = {}
        for call in self.runtime:
            if call.correlation is not None and call.correlation not in self._launch:
                self._launch[call.correlation] = call.start_us

    @classmethod
    def load(cls, path: str | Path) -> "TorchTrace":
        return cls(load_json(path))

    def launch_us(self, correlation: int | None) -> float | None:
        """When the runtime call that started a kernel or copy began on the CPU; None when the trace lacks it."""
        return None if correlation is None else self._launch.get(correlation)

    def graph_launches(self) -> int:
        """How many hipGraphLaunch / cudaGraphLaunch calls the trace holds (HIP graphs replayed)."""
        return sum(1 for r in self.runtime if "graphlaunch" in r.name.lower())


def trim_torch_trace(data: Mapping[str, Any], start_us: float, end_us: float, *, max_events: int = 20000) -> dict:
    """A small copy of a trace that keeps what lies in [start_us, end_us] (kernels and copies by their launch when
    it falls inside, so a step keeps its GPU work), the metadata events and the top-level fields: a recorded
    fixture of a few steps. At most max_events events, earliest first."""
    events = [e for e in data.get("traceEvents", []) if isinstance(e, Mapping)]
    keep_corr: set[int] = set()
    for e in events:
        cat = str(e.get("cat", "")).lower()
        ts = _float(e.get("ts"))
        if cat in RUNTIME_CATS and ts is not None and start_us <= ts <= end_us:
            corr = _int((e.get("args") or {}).get("correlation"))
            if corr is not None:
                keep_corr.add(corr)
    kept = []
    for e in events:
        if e.get("ph") == "M":
            kept.append(e)
            continue
        ts = _float(e.get("ts"))
        cat = str(e.get("cat", "")).lower()
        corr = _int((e.get("args") or {}).get("correlation")) if isinstance(e.get("args"), Mapping) else None
        if (ts is not None and start_us <= ts <= end_us) or (cat in KERNEL_CATS | COPY_CATS and corr in keep_corr):
            kept.append(e)
    kept.sort(key=lambda e: (_float(e.get("ts")) or 0.0))
    out = {k: v for k, v in data.items() if k != "traceEvents"}
    out["traceEvents"] = kept[:max_events]
    return out


# --- rocprofv3's CSV ------------------------------------------------------------------------------------------------

def _row_int(row: Mapping[str, str], *names: str) -> int | None:
    for name in names:
        if name in row and row[name] not in (None, ""):
            value = _int(row[name])
            if value is not None:
                return value
    return None


def _kernel_name(text: str) -> str:
    """rocprof names a code-object kernel by its symbol, sometimes with the descriptor suffix .kd: drop it."""
    name = text.strip()
    return name[:-3] if name.endswith(".kd") else name


def read_rocprof_kernels(path: str | Path) -> list[Kernel]:
    """The rows of a rocprofv3 kernel trace CSV as Kernels (microseconds on rocprof's own clock)."""
    out: list[Kernel] = []
    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            start, end = _row_int(row, "Start_Timestamp"), _row_int(row, "End_Timestamp")
            if start is None or end is None or end < start:
                continue
            block = (_row_int(row, "Workgroup_Size_X", "Workgroup_Size") or 1, _row_int(row, "Workgroup_Size_Y") or 1,
                     _row_int(row, "Workgroup_Size_Z") or 1)
            items = (_row_int(row, "Grid_Size_X", "Grid_Size") or 0, _row_int(row, "Grid_Size_Y") or 1,
                     _row_int(row, "Grid_Size_Z") or 1)
            grid = tuple(-(-g // b) if b else g for g, b in zip(items, block))
            out.append(Kernel(_kernel_name(row.get("Kernel_Name", "")), start / 1000.0, (end - start) / 1000.0,
                              grid, block, _row_int(row, "Correlation_Id"),
                              _row_int(row, "Stream_Id", "Queue_Id"),
                              _row_int(row, "LDS_Block_Size", "Group_Segment_Size", "LDS_Size"),
                              _row_int(row, "Scratch_Size", "Private_Segment_Size"),
                              _row_int(row, "VGPR_Count", "Arch_VGPR_Count", "arch_vgpr_count"),
                              _row_int(row, "SGPR_Count", "sgpr_count")))
    out.sort(key=lambda k: k.start_us)
    return out


def read_rocprof_copies(path: str | Path) -> list[Copy]:
    """The rows of a rocprofv3 memory copy trace CSV as Copies (microseconds on rocprof's own clock)."""
    out: list[Copy] = []
    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            start, end = _row_int(row, "Start_Timestamp"), _row_int(row, "End_Timestamp")
            if start is None or end is None or end < start:
                continue
            out.append(Copy(copy_kind(row.get("Direction", "") or row.get("Kind", "")), start / 1000.0,
                            (end - start) / 1000.0, _row_int(row, "Size", "Bytes", "Copy_Bytes"),
                            _row_int(row, "Correlation_Id")))
    out.sort(key=lambda c: c.start_us)
    return out


ROCPROF_KERNEL_COLUMNS = ("Kind", "Agent_Id", "Queue_Id", "Stream_Id", "Thread_Id", "Dispatch_Id", "Kernel_Id",
                          "Kernel_Name", "Correlation_Id", "Start_Timestamp", "End_Timestamp", "LDS_Block_Size",
                          "Scratch_Size", "VGPR_Count", "Accum_VGPR_Count", "SGPR_Count", "Workgroup_Size_X",
                          "Workgroup_Size_Y", "Workgroup_Size_Z", "Grid_Size_X", "Grid_Size_Y", "Grid_Size_Z")


def write_rocprof_kernels(rows: Iterable[Mapping[str, Any]]) -> str:
    """Kernel trace rows (dicts keyed by rocprofv3's column names) as CSV text: used to keep an excerpt."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=ROCPROF_KERNEL_COLUMNS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def find_rocprof_csvs(root: str | Path) -> dict[str, list[Path]]:
    """Every kernel trace and memory copy trace CSV under root (rocprofv3 writes one set per process)."""
    base = Path(root)
    return {"kernel": sorted(base.rglob("*kernel_trace.csv")),
            "copy": sorted(base.rglob("*memory_copy_trace.csv"))}


# --- categories -----------------------------------------------------------------------------------------------------

class Categorizer:
    """Kernel name -> category by ordered rules; `exact` maps whole names first (case-sensitive)."""

    def __init__(self, rules: Sequence[tuple[str, str]] = DEFAULT_RULES,
                 exact: Mapping[str, str] | None = None) -> None:
        self.rules = [(category, re.compile(pattern, re.IGNORECASE)) for category, pattern in rules]
        self.exact = dict(exact or {})
        self._seen: dict[str, str] = {}

    def __call__(self, name: str) -> str:
        found = self._seen.get(name)
        if found is None:
            found = self.exact.get(name)
            if found is None:
                found = next((c for c, rx in self.rules if rx.search(name)), "other")
            self._seen[name] = found
        return found

    def describe(self) -> dict[str, Any]:
        return {"exact": dict(self.exact), "rules": [[c, rx.pattern] for c, rx in self.rules]}


# --- arithmetic -------------------------------------------------------------------------------------------------------

def union_us(intervals: Iterable[tuple[float, float]]) -> float:
    """Total length of the union of [start, end) intervals."""
    total = 0.0
    cur_s: float | None = None
    cur_e = 0.0
    for s, e in sorted(intervals):
        if e <= s:
            continue
        if cur_s is None or s > cur_e:
            if cur_s is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_s is not None:
        total += cur_e - cur_s
    return total


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """The q-th percentile (0..100) of already sorted values, linear between the two nearest ranks."""
    if not sorted_values:
        raise ValueError("no values")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo))


def _r(value: float | None, digits: int = 3) -> float | None:
    return None if value is None or not math.isfinite(value) else round(float(value), digits)


def spread(values: Iterable[float | None], digits: int = 3) -> dict[str, Any]:
    """n, mean, sd, min, p10, p50, p90, p99 and max of the values that are not None."""
    xs = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not xs:
        return {"n": 0, "mean": None, "sd": None, "min": None, "p10": None, "p50": None, "p90": None, "p99": None,
                "max": None}
    return {"n": len(xs), "mean": _r(statistics.fmean(xs), digits),
            "sd": _r(statistics.stdev(xs), digits) if len(xs) > 1 else None, "min": _r(xs[0], digits),
            "p10": _r(percentile(xs, 10), digits), "p50": _r(percentile(xs, 50), digits),
            "p90": _r(percentile(xs, 90), digits), "p99": _r(percentile(xs, 99), digits), "max": _r(xs[-1], digits)}


# --- tables ---------------------------------------------------------------------------------------------------------

def _top_config(values: Iterable[Any]) -> tuple[Any, int]:
    counts = Counter(v for v in values if v is not None)
    if not counts:
        return None, 0
    value, _ = max(counts.items(), key=lambda kv: (kv[1], str(kv[0])))
    return value, len(counts)


def kernel_table(kernels: Sequence[Kernel], *, steps: int | None, step_us: float | None,
                 categorize: Callable[[str], str], top: int | None = 25) -> list[dict[str, Any]]:
    """Kernels grouped by name, biggest total time first: launches, time, share of all kernel time and of a step,
    the most common grid and block (and how many distinct ones there were), LDS and registers when known."""
    groups: dict[str, list[Kernel]] = defaultdict(list)
    for k in kernels:
        groups[k.name].append(k)
    total = sum(k.dur_us for k in kernels) or 0.0
    rows = []
    for name, items in groups.items():
        time_us = sum(k.dur_us for k in items)
        grid, grids = _top_config(k.grid for k in items)
        block, blocks = _top_config(k.block for k in items)
        per_step = time_us / steps if steps else None
        rows.append({
            "name": name, "category": categorize(name), "launches": len(items),
            "launches_per_step": _r(len(items) / steps, 3) if steps else None,
            "total_ms": _r(time_us / 1000.0, 4),
            "ms_per_step": _r(per_step / 1000.0, 4) if per_step is not None else None,
            "mean_us": _r(time_us / len(items), 2), "max_us": _r(max(k.dur_us for k in items), 2),
            "share_of_kernel_time": _r(time_us / total, 4) if total else None,
            "share_of_step": _r(per_step / step_us, 4) if per_step is not None and step_us else None,
            "grid": list(grid) if grid else None, "distinct_grids": grids,
            "block": list(block) if block else None, "distinct_blocks": blocks,
            "lds_bytes": _top_config(k.lds_bytes for k in items)[0],
            "vgprs": _top_config(k.vgprs for k in items)[0], "sgprs": _top_config(k.sgprs for k in items)[0],
            "scratch_bytes": _top_config(k.scratch_bytes for k in items)[0]})
    rows.sort(key=lambda r: (-r["total_ms"], r["name"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows if top is None else rows[:top]


def category_totals(kernels: Iterable[Kernel], categorize: Callable[[str], str]) -> dict[str, float]:
    """Kernel time in microseconds per category."""
    out: dict[str, float] = defaultdict(float)
    for k in kernels:
        out[categorize(k.name)] += k.dur_us
    return dict(out)


def copy_summary(copies: Sequence[Copy], *, steps: int | None) -> dict[str, Any]:
    """Copies and memsets: count, time and bytes in all and per kind, and per step when steps is given."""
    by_kind: dict[str, dict[str, Any]] = {}
    for c in copies:
        entry = by_kind.setdefault(c.kind, {"count": 0, "time_us": 0.0, "bytes": 0, "bytes_known": 0})
        entry["count"] += 1
        entry["time_us"] += c.dur_us
        if c.bytes is not None:
            entry["bytes"] += c.bytes
            entry["bytes_known"] += 1
    time_us = sum(c.dur_us for c in copies)
    known = [c.bytes for c in copies if c.bytes is not None]
    return {
        "count": len(copies), "time_ms": _r(time_us / 1000.0, 4),
        "bytes": sum(known) if known else None, "bytes_known_for": len(known),
        "count_per_step": _r(len(copies) / steps, 3) if steps else None,
        "ms_per_step": _r(time_us / 1000.0 / steps, 4) if steps else None,
        "bytes_per_step": _r(sum(known) / steps, 1) if steps and known else None,
        "by_kind": {kind: {"count": v["count"], "time_ms": _r(v["time_us"] / 1000.0, 4),
                           "bytes": v["bytes"] if v["bytes_known"] else None}
                    for kind, v in sorted(by_kind.items())}}


def memory_peaks(samples: Iterable[MemorySample], *, start_us: float | None = None,
                 end_us: float | None = None) -> dict[str, Any]:
    """The highest allocated and reserved bytes of the GPU allocator samples, in a window when one is given."""
    gpu = [s for s in samples if s.device_type is not None and s.device_type != CPU_DEVICE_TYPE
           and (start_us is None or s.ts_us >= start_us) and (end_us is None or s.ts_us <= end_us)]
    allocated = [s.allocated for s in gpu if s.allocated is not None]
    reserved = [s.reserved for s in gpu if s.reserved is not None]
    return {"samples": len(gpu), "peak_allocated_bytes": max(allocated) if allocated else None,
            "peak_reserved_bytes": max(reserved) if reserved else None,
            "peak_allocated_gib": _r(max(allocated) / 2**30, 3) if allocated else None,
            "peak_reserved_gib": _r(max(reserved) / 2**30, 3) if reserved else None}


def pick_marker(kernels: Sequence[Kernel], steps: int) -> dict[str, Any] | None:
    """A kernel that runs exactly once per step, to cut a kernel-only trace into steps: among the names launched
    `steps` times (else steps + 1, else steps - 1), the one whose launches are most evenly spaced. Its start times
    are the step boundaries; None when no name fits."""
    if steps < 2:
        return None
    starts: dict[str, list[float]] = defaultdict(list)
    for k in kernels:
        starts[k.name].append(k.start_us)
    for count in (steps, steps + 1, steps - 1):
        candidates = []
        for name, times in starts.items():
            if len(times) != count or count < 3:
                continue
            times = sorted(times)
            gaps = [b - a for a, b in zip(times, times[1:])]
            mean = statistics.fmean(gaps)
            if mean <= 0:
                continue
            cv = statistics.pstdev(gaps) / mean
            candidates.append((cv, name, times, gaps))
        if candidates:
            cv, name, times, gaps = min(candidates, key=lambda c: (c[0], c[1]))
            return {"name": name, "launches": len(times), "cv": round(cv, 4), "starts_us": times, "gaps_us": gaps}
    return None
