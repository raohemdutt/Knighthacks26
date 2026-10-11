"""Training step profile of the student LoRA run: where each optimizer step's time goes.

    python -m lab.profile_train self-check --quick [--out DIR]     # this machine: parser and writer on the fixtures
    python -m lab.profile_train run --dataset D --rows R --model M [--cpu] [--steps 20] [--out DIR] [-- TRAIN ARGS]
    python -m lab.profile_train ingest RUN_DIR [--chip-sheet FILE] [--out DIR] [--no-ledger] [--force]
    python -m lab.profile_train record RESULT_JSON [--out DIR]     # a result made on the pod: file it, ledger row

`python -m lab profile-train ...` is the same. Every call answers with one JSON line and exits 0 when it worked.

`run` trains with lab/train.py for --steps optimizer steps (20 by default; --active N profiles only the last N) in a
process of its own under the torch profiler (record_shapes and profile_memory on, with_stack off; one ProfilerStep
per optimizer step), with forward hooks that open a record_function range around every layer of a kind this profile
reports, named `atlas::<kind>::<module>`:

    gdn          a Gated DeltaNet layer (`*.linear_attn`)
    attention    a full attention layer (`*.self_attn`)
    mlp          a layer's MLP (`*.mlp`)
    lora         a LoRA matmul (`*.lora_A.<adapter>` and `*.lora_B.<adapter>`; it sits inside its layer, and the
                 innermost range wins, so LoRA time is never counted in its layer too)
    embed_head   the token embedding and the LM head

When rocprofv3 is on PATH (and not --cpu), a second process trains the same steps under `rocprofv3 --kernel-trace
--memory-copy-trace` without the torch profiler, for kernel times free of the profiler's own overhead. --cpu is the
rehearsal on a machine without a GPU: the file name says cpu instead of the chip.

`ingest` reads a run folder (run.json, the torch trace, the rocprof CSVs when present), writes
train-profile-<chip|cpu>-<date>.json and .md (default lab/refine/research/kernels/data) and the ledger rows (node
pod-b for a pod run). `run` ingests its own folder at the end, so on the pod one command does both.

How the trace is read. Forward ops are attributed by the innermost `atlas::` range around them on their thread.
Backward ops run on the autograd thread outside every range: each is attributed to the layer of the forward op it
differentiates, through the profiler's `fwdbwd` flow events (a flow starts at the forward op and ends at its
backward function, by sequence number); the ops a backward function calls inherit it. Ops inside the optimizer's
`Optimizer.step#...` and `Optimizer.zero_grad#...` ranges are the optimizer's, and inside `enumerate(DataLoader)#...`
data loading's. A GPU kernel or copy is attributed through its correlation id to the runtime call that launched it,
and so to that call's op. A step is a `ProfilerStep#<n>` range; its work is what was launched inside it.

What a step's share means. On a GPU trace a bucket's share is its kernel time over the step's wall time; copies are
the GPU copies' time; data loading is the wall time of the DataLoader's `__next__` (the GPU waits through it);
gpu_idle is the part of the step in which no kernel or copy ran. On a CPU trace a bucket's share is its ops' self
time (an op's time minus that of the ops it called) over the step time. Shares need not add up to one: kernels can
overlap and host work overlaps the GPU's.

Bandwidth and matmul rate are lower bounds read from the recorded shapes: bytes are every leaf op's input tensors
(outputs are not counted), FLOPs are 2MNK of every aten::mm, addmm, bmm and baddbmm, over the step time and over
the kernel time of those ops; they are set beside the chip sheet's device-to-device copy over its largest buffer
(R33's measured sheet, about 210 GB/s on gfx1151) and its best bf16 matmul.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if not __package__:  # run by path: the repo root, one folder up, holds lab/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab import trace_profile as tp  # noqa: E402

SCHEMA = "lab.profile_train/1"
TASK = "R34"
NODE = "pod-b"
CPU_NODE = "cpu"
CHIP = "gfx1151"
REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "lab" / "refine" / "research" / "kernels" / "data"
FIXTURES = REPO / "fixtures" / "lab" / "train_profile"
DEFAULT_STEPS = 20
TOP = 25
TAG = "atlas::"
LAYER_KINDS = ("gdn", "attention", "mlp", "lora", "embed_head")
BUCKETS = (*LAYER_KINDS, "optimizer", "data_loading", "other")
SHARE_KEYS = ("gdn", "attention", "lora", "mlp", "embed_head", "optimizer", "data_loading", "copies", "other",
              "gpu_idle")
OPTIMIZER_PREFIX = "Optimizer."
DATA_PREFIX = "enumerate(DataLoader)"
STEP_RE = re.compile(r"^ProfilerStep#(\d+)$")
MATMUL_OPS = frozenset({"aten::mm", "aten::addmm", "aten::bmm", "aten::baddbmm"})
CPU_OP_CATS = frozenset({"cpu_op", "operator"})
# flash-linear-attention's kernels and the CUDA causal conv: present only when the fast GDN path ran.
FLA_RE = re.compile(r"chunk_gated_delta|gated_delta_rule|fused_recurrent|chunk_scaled_dot_kkt|"
                    r"solve_tril|wy_fast|chunk_fwd_kernel|chunk_bwd_kernel|fla_|causal_conv1d_(fwd|bwd)",
                    re.IGNORECASE)
DTYPE_BYTES = {"float": 4, "double": 8, "c10::bfloat16": 2, "c10::half": 2, "bfloat16": 2, "half": 2,
               "float16": 2, "long int": 8, "long": 8, "int": 4, "short int": 2, "short": 2, "bool": 1,
               "unsigned char": 1, "signed char": 1, "char": 1, "c10::float8_e4m3fnuz": 1, "c10::float8_e5m2": 1}
MD_SECTIONS = ("## Run", "## Time per step", "## Where a step goes", "## Top kernels", "## Bandwidth and matmul rate",
               "## Memory", "## Fast linear-attention kernels", "## Open items")
L16_CITE = ("L16, the first real fine-tune (pod D, 2026-10-10): 18 steps in 785 s, peak 11.1 GiB, no power "
            "sampler; this profile explains where that run's steps went")


class ProfileError(Exception):
    """A trace or a result that cannot be used."""


def _r(value: float | None, digits: int = 4) -> float | None:
    return None if value is None or not math.isfinite(value) else round(float(value), digits)


# --- the trace: ops, ranges, links -------------------------------------------------------------------------------

@dataclass
class Node:
    """One complete event on a CPU thread: an op, a record_function range or a runtime call."""

    name: str
    cat: str
    start_us: float
    dur_us: float
    tid: Any
    args: Mapping[str, Any]
    parent: "Node | None" = None
    children: list["Node"] = field(default_factory=list)
    linked: "Node | None" = None          # a backward function: the forward op it differentiates
    bucket: str | None = None

    @property
    def end_us(self) -> float:
        return self.start_us + self.dur_us

    @property
    def is_op(self) -> bool:
        return self.cat in CPU_OP_CATS

    def self_us(self) -> float:
        return max(0.0, self.dur_us - sum(c.dur_us for c in self.children))


def tag_kind(name: str) -> str | None:
    """The kind of an `atlas::<kind>::<module>` range, None for any other name."""
    if not name.startswith(TAG):
        return None
    kind = name[len(TAG):].split("::", 1)[0]
    return kind if kind in LAYER_KINDS else None


def _key(tid: Any) -> str:
    return str(tid)


class TrainTrace:
    """A torch profiler trace of training steps, with every CPU event put in its thread's call tree."""

    def __init__(self, data: Mapping[str, Any]) -> None:
        self.base = tp.TorchTrace(data)          # kernels, copies, memory samples, device properties
        self.nodes: list[Node] = []
        self.by_corr: dict[int, Node] = {}
        flows: dict[Any, dict[str, tuple[float, Any]]] = defaultdict(dict)
        for e in data.get("traceEvents", []):
            if not isinstance(e, Mapping):
                continue
            ph, cat = e.get("ph"), str(e.get("cat", "")).lower()
            if ph in ("s", "f") and cat == "fwdbwd":
                ts = tp._float(e.get("ts"))
                if ts is not None and e.get("id") is not None:
                    flows[e.get("id")][ph] = (ts, _key(e.get("tid")))
                continue
            if ph != "X" or not (cat in CPU_OP_CATS or cat in tp.ANNOTATION_CATS or cat in tp.RUNTIME_CATS):
                continue
            ts, dur = tp._float(e.get("ts")), tp._float(e.get("dur"))
            if ts is None or dur is None or dur < 0:
                continue
            args = e.get("args") if isinstance(e.get("args"), Mapping) else {}
            node = Node(str(e.get("name", "")), cat, ts, dur, _key(e.get("tid")), args)
            self.nodes.append(node)
            corr = tp._int(args.get("correlation"))
            if cat in tp.RUNTIME_CATS and corr is not None:
                self.by_corr.setdefault(corr, node)
        self._nest()
        for ends in flows.values():
            if "s" in ends and "f" in ends:
                fwd = self._op_at(*ends["s"])
                bwd = self._op_at(*ends["f"])
                if fwd is not None and bwd is not None and bwd is not fwd:
                    bwd.linked = fwd
        for node in self.nodes:
            self._bucket(node)
        self.steps = sorted((n for n in self.nodes if STEP_RE.match(n.name)), key=lambda n: n.start_us)
        self.link_count = sum(1 for n in self.nodes if n.linked is not None)

    @classmethod
    def load(cls, path: str | Path) -> "TrainTrace":
        return cls(tp.load_json(path))

    def _nest(self) -> None:
        threads: dict[str, list[Node]] = defaultdict(list)
        for node in self.nodes:
            threads[node.tid].append(node)
        self._starts: dict[str, list[float]] = {}
        self._ordered: dict[str, list[Node]] = {}
        for tid, items in threads.items():
            items.sort(key=lambda n: (n.start_us, -n.dur_us))
            stack: list[Node] = []
            for node in items:
                while stack and node.start_us >= stack[-1].end_us:
                    stack.pop()
                if stack and node.end_us <= stack[-1].end_us + 1e-6:
                    node.parent = stack[-1]
                    stack[-1].children.append(node)
                stack.append(node)
            self._ordered[tid] = items
            self._starts[tid] = [n.start_us for n in items]

    def _op_at(self, ts: float, tid: str) -> Node | None:
        """The op a flow end is bound to: the op starting exactly there, else the innermost op containing it."""
        items = self._ordered.get(tid)
        if not items:
            return None
        i = bisect_right(self._starts[tid], ts + 1e-6) - 1
        best = None
        while i >= 0:
            node = items[i]
            if node.is_op and node.start_us <= ts + 1e-6 and ts <= node.end_us + 1e-6:
                if abs(node.start_us - ts) <= 1e-6:
                    return node
                if best is None:
                    best = node
            if node.end_us < ts and node.parent is None:
                break
            i -= 1
        return best

    def _bucket(self, node: Node, depth: int = 0) -> str:
        if node.bucket is not None:
            return node.bucket
        kind = tag_kind(node.name)
        if kind is not None:
            found = kind
        elif node.name.startswith(OPTIMIZER_PREFIX):
            found = "optimizer"
        elif node.name.startswith(DATA_PREFIX):
            found = "data_loading"
        elif node.linked is not None and depth < 64:
            found = self._bucket(node.linked, depth + 1)
        elif node.parent is not None and depth < 512:
            found = self._bucket(node.parent, depth + 1)
        else:
            found = "other"
        node.bucket = found
        return found

    def launcher(self, correlation: int | None) -> Node | None:
        return None if correlation is None else self.by_corr.get(correlation)

    def op_of(self, node: Node | None) -> Node | None:
        """The innermost op a runtime call sits in."""
        while node is not None and not node.is_op:
            node = node.parent
        return node


# --- shapes: bytes and FLOPs --------------------------------------------------------------------------------------

def _dims(value: Any) -> list[list[int]]:
    if not isinstance(value, list):
        return []
    return [[int(d) for d in dims] if isinstance(dims, list) and all(isinstance(d, int) for d in dims) else []
            for dims in value]


def op_bytes(node: Node) -> int:
    """Bytes of the op's input tensors from the recorded shapes and dtypes (outputs are not recorded)."""
    dims, types = _dims(node.args.get("Input Dims")), node.args.get("Input type") or []
    total = 0
    for i, shape in enumerate(dims):
        if not shape:
            continue
        size = DTYPE_BYTES.get(str(types[i]).lower() if i < len(types) else "", 0)
        total += math.prod(shape) * size
    return total


def op_flops(node: Node) -> int:
    """2MNK of a matmul op from its recorded input shapes; 0 for anything else or a shape that does not fit."""
    if node.name not in MATMUL_OPS:
        return 0
    dims = [d for d in _dims(node.args.get("Input Dims"))]
    if node.name in ("aten::addmm", "aten::baddbmm"):
        dims = dims[1:3]
    else:
        dims = dims[:2]
    if len(dims) != 2:
        return 0
    a, b = dims
    if node.name in ("aten::mm", "aten::addmm") and len(a) == 2 and len(b) == 2 and a[1] == b[0]:
        return 2 * a[0] * a[1] * b[1]
    if node.name in ("aten::bmm", "aten::baddbmm") and len(a) == 3 and len(b) == 3 and a[2] == b[1]:
        return 2 * a[0] * a[1] * a[2] * b[2]
    return 0


def _leaf_ops(nodes: Iterable[Node]) -> list[Node]:
    return [n for n in nodes if n.is_op and not any(c.is_op for c in n.children)]


# --- one trace -> the numbers -------------------------------------------------------------------------------------

def _in(ts: float, window: Node) -> bool:
    return window.start_us <= ts < window.end_us


def _step_of(ts: float, steps: Sequence[Node], starts: Sequence[float]) -> int | None:
    i = bisect_right(starts, ts) - 1
    return i if i >= 0 and _in(ts, steps[i]) else None


def _union_in(spans: Iterable[tuple[float, float]], window: Node) -> float:
    return tp.union_us((max(s, window.start_us), min(e, window.end_us)) for s, e in spans
                       if e > window.start_us and s < window.end_us)


def analyze_trace(trace: TrainTrace, *, warmup: int = 0) -> dict[str, Any]:
    """Step times, the share of a step in each bucket, the top kernels (or ops), bytes and FLOPs, memory peaks and
    fast-kernel evidence of one torch profiler trace. The first `warmup` steps are reported apart and left out of
    the means and shares."""
    steps_all = trace.steps
    if not steps_all:
        raise ProfileError("the trace has no ProfilerStep#<n> range: was profiler.step() called each step?")
    measured = steps_all[warmup:] if len(steps_all) > warmup else steps_all
    starts = [s.start_us for s in measured]
    n = len(measured)
    step_total_us = sum(s.dur_us for s in measured)
    gpu = bool(trace.base.kernels)

    bucket_us: dict[str, float] = defaultdict(float)
    kernels_in: list[tp.Kernel] = []
    kernel_bucket: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    matmul_kernel_us = 0.0
    unattributed = 0
    if gpu:
        for k in trace.base.kernels:
            call = trace.launcher(k.correlation)
            when = call.start_us if call is not None else k.start_us
            if _step_of(when, measured, starts) is None:
                continue
            kernels_in.append(k)
            if call is None:
                unattributed += 1
                b = "other"
            else:
                b = call.bucket or "other"
                op = trace.op_of(call)
                if op is not None and op.name in MATMUL_OPS:
                    matmul_kernel_us += k.dur_us
            bucket_us[b] += k.dur_us
            kernel_bucket[k.name][b] += k.dur_us
    else:
        for node in trace.nodes:
            if not node.is_op or node.bucket == "data_loading":
                continue
            if _step_of(node.start_us, measured, starts) is None:
                continue
            bucket_us[node.bucket or "other"] += node.self_us()
            if node.name in MATMUL_OPS:
                matmul_kernel_us += node.self_us()

    copies_in = []
    for c in trace.base.copies:
        call = trace.launcher(c.correlation)
        when = call.start_us if call is not None else c.start_us
        if _step_of(when, measured, starts) is not None:
            copies_in.append(c)
    copies_us = sum(c.dur_us for c in copies_in)

    data_spans = [(n_.start_us, n_.end_us) for n_ in trace.nodes if n_.name.startswith(DATA_PREFIX)]
    data_us = sum(_union_in(data_spans, s) for s in measured)
    busy = [(k.start_us, k.end_us) for k in trace.base.kernels] + [(c.start_us, c.end_us) for c in trace.base.copies]
    busy_us = sum(_union_in(busy, s) for s in measured) if gpu else None

    shares: dict[str, float | None] = {}
    for key in SHARE_KEYS:
        if key == "copies":
            value = copies_us if gpu else None
        elif key == "data_loading":
            value = data_us
        elif key == "gpu_idle":
            value = (step_total_us - busy_us) if busy_us is not None else None
        else:
            value = bucket_us.get(key, 0.0)
        shares[key] = _r(value / step_total_us, 4) if value is not None and step_total_us else None
    ms_per_step = {key: _r((shares[key] or 0.0) * step_total_us / n / 1000.0, 4) if shares[key] is not None
                   else None for key in SHARE_KEYS}

    if gpu:
        table = tp.kernel_table(kernels_in, steps=n, step_us=step_total_us / n, categorize=tp.Categorizer(), top=TOP)
        for row in table:
            per = kernel_bucket.get(row["name"], {})
            row["bucket"] = max(per.items(), key=lambda kv: kv[1])[0] if per else None
        names = {k.name for k in kernels_in}
    else:
        pseudo = [tp.Kernel(node.name, node.start_us, node.self_us()) for node in trace.nodes
                  if node.is_op and _step_of(node.start_us, measured, starts) is not None]
        table = tp.kernel_table(pseudo, steps=n, step_us=step_total_us / n, categorize=tp.Categorizer(), top=TOP)
        per_name: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for node in trace.nodes:
            if node.is_op and _step_of(node.start_us, measured, starts) is not None:
                per_name[node.name][node.bucket or "other"] += node.self_us()
        for row in table:
            per = per_name.get(row["name"], {})
            row["bucket"] = max(per.items(), key=lambda kv: kv[1])[0] if per else None
        names = {node.name for node in trace.nodes if node.is_op}

    leaves = [node for node in _leaf_ops(trace.nodes) if _step_of(node.start_us, measured, starts) is not None]
    step_bytes = sum(op_bytes(node) for node in leaves)
    flops = sum(op_flops(node) for node in trace.nodes
                if node.is_op and _step_of(node.start_us, measured, starts) is not None)
    kernel_us = sum(k.dur_us for k in kernels_in) if gpu else None
    step_s = step_total_us / 1e6
    bandwidth = {
        "bytes_per_step": _r(step_bytes / n, 1), "flops_per_step": _r(flops / n, 1),
        "achieved_gb_s_step": _r(step_bytes / step_s / 1e9, 4) if step_s else None,
        "achieved_gb_s_kernels": _r(step_bytes / (kernel_us / 1e6) / 1e9, 4) if kernel_us else None,
        "matmul_tflops_step": _r(flops / step_s / 1e12, 6) if step_s else None,
        "matmul_tflops_matmul_time": _r(flops / (matmul_kernel_us / 1e6) / 1e12, 6) if matmul_kernel_us else None,
        "matmul_ms_per_step": _r(matmul_kernel_us / n / 1000.0, 4),
        "how": "bytes: every leaf op's input tensors from the recorded shapes (outputs not counted, a lower bound); "
               "FLOPs: 2MNK of aten::mm, addmm, bmm and baddbmm"}

    gpu_mem = tp.memory_peaks(trace.base.memory, start_us=steps_all[0].start_us, end_us=steps_all[-1].end_us)
    cpu_samples = [s for s in trace.base.memory if s.device_type == tp.CPU_DEVICE_TYPE]
    cpu_alloc = [s.allocated for s in cpu_samples if s.allocated is not None]
    memory = {"device": gpu_mem, "host_peak_allocated_gib": _r(max(cpu_alloc) / 2**30, 3) if cpu_alloc else None}

    fla = sorted({name for name in names if FLA_RE.search(name)})
    return {
        "kind": "gpu" if gpu else "cpu",
        "steps_in_trace": len(steps_all), "warmup_steps": len(steps_all) - n, "steps_measured": n,
        "step_ms": tp.spread([s.dur_us / 1000.0 for s in measured]),
        "step_ms_each": [_r(s.dur_us / 1000.0, 3) for s in steps_all],
        "shares": shares, "ms_per_step": ms_per_step,
        "share_basis": ("kernel time over step wall time (data loading: DataLoader wall time; gpu_idle: no kernel "
                        "or copy running)") if gpu else "op self time over step wall time (data loading: wall time)",
        "top": table, "top_basis": "GPU kernels by self time" if gpu else "CPU ops by self time",
        "copies": tp.copy_summary(copies_in, steps=n) if gpu else None,
        "bandwidth": bandwidth, "memory": memory,
        "fast_linear_attention": {"present": bool(fla), "evidence": fla[:10]},
        "attribution": {"backward_links": trace.link_count, "unattributed_kernels": unattributed,
                        "kernels": len(kernels_in), "ops": sum(1 for n_ in trace.nodes if n_.is_op)},
        "device": {k: trace.base.device.get(k) for k in ("name", "computeMajor", "computeMinor", "totalGlobalMem",
                                                          "numSms") if k in trace.base.device},
    }


# --- rocprofv3: kernel times without the torch profiler -------------------------------------------------------------

TRAIN_RULES: tuple[tuple[str, str], ...] = (
    ("optimizer", r"adam|multi_tensor|foreach|fused_adamw"),
    *tp.DEFAULT_RULES[1:],
)


def analyze_rocprof(folder: str | Path, steps: int) -> dict[str, Any] | None:
    """The kernel table and category totals of rocprofv3's CSVs under folder, cut into steps at a kernel that
    runs once per step; None when the folder holds no kernel trace."""
    found = tp.find_rocprof_csvs(folder)
    if not found["kernel"]:
        return None
    kernels = sorted((k for path in found["kernel"] for k in tp.read_rocprof_kernels(path)), key=lambda k: k.start_us)
    copies = sorted((c for path in found["copy"] for c in tp.read_rocprof_copies(path)), key=lambda c: c.start_us)
    categorize = tp.Categorizer(TRAIN_RULES)
    marker = tp.pick_marker(kernels, steps)
    window = None
    step_us = None
    if marker is not None and len(marker["starts_us"]) >= 2:
        window = (marker["starts_us"][0], marker["starts_us"][-1])
        gaps = marker["gaps_us"]
        step_us = sum(gaps) / len(gaps)
        kernels = [k for k in kernels if window[0] <= k.start_us < window[1]]
        copies = [c for c in copies if window[0] <= c.start_us < window[1]]
    n = len(marker["gaps_us"]) if marker else None
    totals = tp.category_totals(kernels, categorize)
    return {
        "marker": {k: marker[k] for k in ("name", "launches", "cv")} if marker else None,
        "steps": n, "step_ms": tp.spread([g / 1000.0 for g in marker["gaps_us"]]) if marker else None,
        "top": tp.kernel_table(kernels, steps=n, step_us=step_us, categorize=categorize, top=TOP),
        "category_ms_per_step": {c: _r(us / n / 1000.0, 4) for c, us in sorted(totals.items())} if n else None,
        "copies": tp.copy_summary(copies, steps=n),
        "fast_linear_attention": sorted({k.name for k in kernels if FLA_RE.search(k.name)})[:10]}


# --- the result -----------------------------------------------------------------------------------------------------

def chip_reference(path: str | Path | None = None) -> dict[str, Any]:
    """The chip sheet's device-to-device copy over its largest buffer and its best bf16 matmul. Without a path: the
    newest chipsheet-gfx1151-<date>.json in the data folder. A self-check sheet's invented numbers are never used."""
    if path is None:
        sheets = sorted(DATA_DIR.glob(f"chipsheet-{CHIP}-*.json"))
        if not sheets:
            return {"chip_sheet": None, "gb_s": None, "matmul_tflops": None,
                    "note": f"no chipsheet-{CHIP}-<date>.json on main"}
        path = sheets[-1]
    target = Path(path)
    try:
        sheet = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"chip_sheet": str(target), "gb_s": None, "matmul_tflops": None, "note": f"unreadable: {exc}"}
    try:
        name = str(target.resolve().relative_to(REPO)).replace("\\", "/")
    except ValueError:
        name = target.name
    out: dict[str, Any] = {"chip_sheet": name, "gb_s": None, "copy_id": None, "matmul_tflops": None,
                           "matmul_id": None, "note": None, "mode": sheet.get("mode")}
    if sheet.get("mode") == "self-check":
        out["note"] = f"{name} is a self-check sheet: its numbers are invented"
        return out
    items = [m for m in sheet.get("measurements") or [] if isinstance(m, Mapping) and m.get("status", "ok") == "ok"
             and isinstance(m.get("value"), (int, float))]
    copies = [m for m in items if str(m.get("id", "")).startswith("copy.d2d.")]
    if copies:
        largest = max(copies, key=lambda m: (m.get("params") or {}).get("bytes") or 0)
        out.update(gb_s=float(largest["value"]), copy_id=largest["id"])
    mms = [m for m in items if str(m.get("id", "")).startswith("matmul.bf16.") and m.get("unit") == "TFLOP/s"]
    if mms:
        best = max(mms, key=lambda m: m["value"])
        out.update(matmul_tflops=float(best["value"]), matmul_id=best["id"])
    if out["gb_s"] is None:
        out["note"] = f"{name} carries no device-to-device copy"
    return out


def apply_reference(result: dict[str, Any], reference: Mapping[str, Any]) -> None:
    result["reference"] = dict(reference)
    bw = result["bandwidth"]
    ref, mm = reference.get("gb_s"), reference.get("matmul_tflops")
    bw["reference_gb_s"], bw["reference_matmul_tflops"] = ref, mm
    for key in ("achieved_gb_s_step", "achieved_gb_s_kernels"):
        bw[key.replace("achieved_gb_s", "fraction_of_copy")] = _r(bw[key] / ref, 4) if ref and bw.get(key) else None
    for key in ("matmul_tflops_step", "matmul_tflops_matmul_time"):
        bw[key.replace("matmul_tflops", "fraction_of_matmul")] = _r(bw[key] / mm, 4) if mm and bw.get(key) else None
    result["open"] = [o for o in result.get("open", []) if not o.startswith("chip sheet:")]
    if not ref:
        result["open"].append(f"chip sheet: {reference.get('note')}")


def build_result(analysis: Mapping[str, Any], run: Mapping[str, Any] | None, rocprof: Mapping[str, Any] | None,
                 *, date: str, cpu: bool) -> dict[str, Any]:
    """The profile's result: the trace analysis, the run's own answer (samples per second, peak memory, the GDN
    kernel lab/train.py saw installed) and rocprof's view when it ran."""
    run = run or {}
    metrics = run.get("metrics") or {}
    refs = run.get("refs") or {}
    reported = str(refs.get("deltanet_kernel") or "") or None
    fla = dict(analysis["fast_linear_attention"])
    fla["run_reported"] = reported
    fla["present"] = bool(fla["present"] or (rocprof or {}).get("fast_linear_attention")
                          or (reported or "").startswith("fla"))
    fla["fallback_share"] = analysis["shares"].get("gdn") if not fla["present"] else 0.0
    fla["note"] = ("flash-linear-attention ran: the GDN share is its kernels" if fla["present"] else
                   "flash-linear-attention absent: the GDN share is the torch fallback (stock image)")
    open_items: list[str] = []
    if analysis["kind"] == "cpu":
        open_items.append("CPU rehearsal: the pod run on gfx1151 is still to do")
    if analysis["attribution"]["backward_links"] == 0:
        open_items.append("no fwdbwd flow in the trace: backward time is attributed to other")
    if not analysis["kind"] == "cpu" and rocprof is None:
        open_items.append("rocprofv3 did not run: kernel times are the torch profiler's only")
    return {
        "schema": SCHEMA, "task": TASK, "date": date, "chip": "cpu" if cpu else CHIP,
        "node": CPU_NODE if cpu else NODE, "run_id": refs.get("run_id"), "outcome": run.get("outcome"),
        "cites": L16_CITE,
        "run": {"samples_per_s": metrics.get("trainer.samples_per_s"),
                "samples_per_s_note": "measured under the torch profiler, as lab/refine/metrics.jsonl:13 measures "
                                      "it; the profiler's overhead makes it a lower bound",
                "steps": metrics.get("trainer.steps"), "train_s": metrics.get("trainer.train_s"),
                "peak_mem_gb": metrics.get("trainer.peak_mem_gb"), "batch": metrics.get("trainer.batch"),
                "grad_accum": metrics.get("trainer.grad_accum"), "lora_rank": metrics.get("trainer.lora_rank"),
                "device": refs.get("device"), "torch": refs.get("torch"), "rocm": refs.get("rocm"),
                "transformers": refs.get("transformers"), "peft": refs.get("peft"), "trl": refs.get("trl")},
        **{k: analysis[k] for k in ("kind", "steps_in_trace", "warmup_steps", "steps_measured", "step_ms",
                                    "step_ms_each", "shares", "ms_per_step", "share_basis", "top", "top_basis",
                                    "copies", "bandwidth", "memory", "attribution", "device")},
        "fast_linear_attention": fla, "rocprof": rocprof, "open": open_items}


def check_result(result: Any) -> list[str]:
    """What is wrong with a result; empty when it can be written."""
    problems: list[str] = []
    if not isinstance(result, Mapping) or result.get("schema") != SCHEMA:
        return ["not a lab.profile_train/1 result"]
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(result.get("date"))):
        problems.append("date is not YYYY-MM-DD")
    step = result.get("step_ms") or {}
    if not step.get("n") or not step.get("mean") or step["mean"] <= 0:
        problems.append("no step time")
    shares = result.get("shares") or {}
    for key in SHARE_KEYS:
        value = shares.get(key, "missing")
        if value == "missing":
            problems.append(f"share {key} missing")
        elif value is not None and not (0.0 <= value <= 1.5):
            problems.append(f"share {key} is {value}, outside 0..1.5")
    top = result.get("top") or []
    if not top or len(top) > TOP:
        problems.append(f"top table has {len(top)} rows")
    if any("launches" not in row for row in top):
        problems.append("a top row has no launch count")
    if "present" not in (result.get("fast_linear_attention") or {}):
        problems.append("fast_linear_attention.present missing")
    return problems


# --- Markdown --------------------------------------------------------------------------------------------------------

def _fmt(value: Any, digits: int = 3, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:,.{digits}f}{suffix}"
    return f"{value}{suffix}"


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _cell(text: Any, limit: int = 80) -> str:
    s = str(text).replace("|", "/").replace("\n", " ")
    return s if len(s) <= limit else s[: limit - 3] + "..."


def render_markdown(result: Mapping[str, Any]) -> str:
    run = result.get("run") or {}
    step = result.get("step_ms") or {}
    lines = [f"# Training step profile: {result['chip']} on {result['node']}, {result['date']}", "",
             f"Task {result['task']}. {result['cites']}.", "", "## Run", "",
             "| What | Value |", "|---|---|"]
    for label, value in (("run id", result.get("run_id")), ("outcome", result.get("outcome")),
                         ("device", run.get("device")), ("torch / ROCm", f"{run.get('torch')} / {run.get('rocm')}"),
                         ("transformers / PEFT / TRL", f"{run.get('transformers')} / {run.get('peft')} / "
                                                       f"{run.get('trl')}"),
                         ("steps in trace (warmup left out)", f"{result['steps_in_trace']} "
                                                             f"({result['warmup_steps']})"),
                         ("batch x grad accum, LoRA rank", f"{run.get('batch')} x {run.get('grad_accum')}, "
                                                           f"r {run.get('lora_rank')}"),
                         ("samples per second (under the profiler)", _fmt(run.get("samples_per_s"), 4)),
                         ("trace", result["kind"])):
        lines.append(f"| {label} | {_cell(value)} |")
    lines += ["", "## Time per step", "",
              f"Mean {_fmt(step.get('mean'))} ms, sd {_fmt(step.get('sd'))}, p10 {_fmt(step.get('p10'))}, "
              f"p50 {_fmt(step.get('p50'))}, p90 {_fmt(step.get('p90'))}, min {_fmt(step.get('min'))}, "
              f"max {_fmt(step.get('max'))} (n {step.get('n')}).", "",
              "Each step (ms): " + ", ".join(_fmt(v, 2) for v in result.get("step_ms_each") or []), "",
              "## Where a step goes", "", f"Basis: {result['share_basis']}.", "",
              "| Bucket | Share of a step | ms per step |", "|---|---|---|"]
    for key in SHARE_KEYS:
        lines.append(f"| {key} | {_pct(result['shares'].get(key))} | {_fmt(result['ms_per_step'].get(key), 3)} |")
    lines += ["", "## Top kernels", "", f"{result['top_basis']}, top {TOP}.", "",
              "| # | Name | Bucket | Launches | per step | ms per step | share of step |",
              "|---|---|---|---|---|---|---|"]
    for row in result.get("top") or []:
        lines.append(f"| {row['rank']} | `{_cell(row['name'], 70)}` | {row.get('bucket')} | {row['launches']} | "
                     f"{_fmt(row.get('launches_per_step'), 2)} | {_fmt(row.get('ms_per_step'), 4)} | "
                     f"{_pct(row.get('share_of_step'))} |")
    rp = result.get("rocprof")
    if rp:
        lines += ["", f"rocprofv3 (no torch profiler): step marker `{_cell((rp.get('marker') or {}).get('name'))}`, "
                      f"step mean {_fmt((rp.get('step_ms') or {}).get('mean'))} ms.", "",
                  "| # | Name | Category | Launches | ms per step |", "|---|---|---|---|---|"]
        for row in rp.get("top") or []:
            lines.append(f"| {row['rank']} | `{_cell(row['name'], 70)}` | {row['category']} | {row['launches']} | "
                         f"{_fmt(row.get('ms_per_step'), 4)} |")
    bw = result["bandwidth"]
    ref = result.get("reference") or {}
    lines += ["", "## Bandwidth and matmul rate", "", f"How: {bw['how']}.", "",
              "| What | Value | Chip sheet | Fraction |", "|---|---|---|---|",
              f"| bytes per step | {_fmt(bw.get('bytes_per_step'), 0)} | | |",
              f"| GB/s over the step | {_fmt(bw.get('achieved_gb_s_step'))} | {_fmt(bw.get('reference_gb_s'))} | "
              f"{_pct(bw.get('fraction_of_copy_step'))} |",
              f"| GB/s over kernel time | {_fmt(bw.get('achieved_gb_s_kernels'))} | {_fmt(bw.get('reference_gb_s'))} "
              f"| {_pct(bw.get('fraction_of_copy_kernels'))} |",
              f"| matmul TFLOP/s over the step | {_fmt(bw.get('matmul_tflops_step'), 4)} | "
              f"{_fmt(bw.get('reference_matmul_tflops'))} | {_pct(bw.get('fraction_of_matmul_step'))} |",
              f"| matmul TFLOP/s over matmul time | {_fmt(bw.get('matmul_tflops_matmul_time'), 4)} | "
              f"{_fmt(bw.get('reference_matmul_tflops'))} | {_pct(bw.get('fraction_of_matmul_matmul_time'))} |", "",
              f"Chip sheet: {ref.get('chip_sheet')} ({ref.get('copy_id')}, {ref.get('matmul_id')})."]
    mem = result["memory"]
    dev = mem.get("device") or {}
    lines += ["", "## Memory", "", f"Peak allocated {_fmt(dev.get('peak_allocated_gib'))} GiB, peak reserved "
              f"{_fmt(dev.get('peak_reserved_gib'))} GiB on the device ({dev.get('samples')} allocator samples); "
              f"host peak allocated {_fmt(mem.get('host_peak_allocated_gib'))} GiB; the trainer's own "
              f"max_memory_allocated {_fmt(run.get('peak_mem_gb'))} GiB."]
    fla = result["fast_linear_attention"]
    lines += ["", "## Fast linear-attention kernels", "",
              f"Present: {'yes' if fla['present'] else 'no'}. {fla['note']}. lab/train.py saw: "
              f"{fla.get('run_reported')}. Fallback share of a step: {_pct(fla.get('fallback_share'))}."
              + (f" Evidence: {', '.join('`' + e + '`' for e in fla['evidence'])}." if fla.get("evidence") else "")]
    att = result["attribution"]
    lines += ["", "## Open items", "",
              f"Attribution: {att['backward_links']} backward links, {att['unattributed_kernels']} of "
              f"{att['kernels']} kernels without a launching call.", ""]
    lines += [f"- {o}" for o in result.get("open") or []] or ["- none"]
    return "\n".join(lines) + "\n"


def result_paths(result: Mapping[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    stem = f"train-profile-{result['chip']}-{result['date']}"
    return Path(out_dir) / f"{stem}.json", Path(out_dir) / f"{stem}.md"


def write_outputs(result: Mapping[str, Any], out_dir: str | Path, *, force: bool = False) -> tuple[Path, Path]:
    json_path, md_path = result_paths(result, out_dir)
    if not force and (json_path.exists() or md_path.exists()):
        raise ProfileError(f"{json_path.name} exists: pass --force to replace it")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(result), encoding="utf-8")
    return json_path, md_path


def ledger_rows(result: Mapping[str, Any], source: str) -> list[dict[str, Any]]:
    """One row for the profile (node pod-b for a pod run, cpu for a rehearsal), low confidence: one run."""
    step = result.get("step_ms") or {}
    bw = result.get("bandwidth") or {}
    dev = (result.get("memory") or {}).get("device") or {}
    metrics = {"train_profile.step_ms": step.get("mean"), "train_profile.step_ms_p50": step.get("p50"),
               "train_profile.step_ms_sd": step.get("sd"),
               **{f"train_profile.share_{k}": (result.get("shares") or {}).get(k) for k in SHARE_KEYS},
               "train_profile.achieved_gb_s": bw.get("achieved_gb_s_step"),
               "train_profile.matmul_tflops": bw.get("matmul_tflops_step"),
               "train_profile.peak_allocated_gib": dev.get("peak_allocated_gib"),
               "train_profile.peak_reserved_gib": dev.get("peak_reserved_gib"),
               "trainer.samples_per_s": (result.get("run") or {}).get("samples_per_s")}
    metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    units = {k: ("ms" if "_ms" in k else "GB/s" if k.endswith("gb_s") else "TFLOP/s" if k.endswith("tflops")
                 else "GiB" if k.endswith("gib") else "samples/s" if k.endswith("samples_per_s") else "fraction")
             for k in metrics}
    return [{"verb": "profile_train", "outcome": "ok", "metrics": metrics, "units": units,
             "conditions": {"chip": result.get("chip"), "trace": result.get("kind"),
                            "steps": result.get("steps_measured"),
                            "fla": (result.get("fast_linear_attention") or {}).get("present")},
             "node": result.get("node"), "source": source, "low_confidence": True,
             "refs": {"run_id": result.get("run_id")},
             "note": "; ".join(result.get("open") or [])[:480] or None}]


# --- a run folder -> the files ---------------------------------------------------------------------------------------

def find_trace(run_dir: Path) -> Path | None:
    found = sorted([*run_dir.rglob("*.pt.trace.json"), *run_dir.rglob("*.pt.trace.json.gz"),
                    *run_dir.rglob("trace.json"), *run_dir.rglob("trace.json.gz")])
    return found[0] if found else None


def ingest(run_dir: str | Path, *, out_dir: str | Path | None = None, chip_sheet: str | Path | None = None,
           ledger: bool = True, force: bool = False, warmup: int | None = None) -> dict[str, Any]:
    base = Path(run_dir)
    trace_path = find_trace(base)
    if trace_path is None:
        return {"ok": False, "outcome": "no_trace", "error": f"no torch trace under {base}"}
    run = None
    for name in ("run.json", "train/run.json"):
        if (base / name).is_file():
            run = json.loads((base / name).read_text(encoding="utf-8"))
            break
    meta = json.loads((base / "profile.json").read_text(encoding="utf-8")) if (base / "profile.json").is_file() else {}
    trace = TrainTrace.load(trace_path)
    if warmup is None:
        warmup = 2 if len(trace.steps) >= 6 else 0
    analysis = analyze_trace(trace, warmup=warmup)
    rocprof = analyze_rocprof(base / "rocprof", len(trace.steps)) if (base / "rocprof").is_dir() else None
    cpu = bool(meta.get("cpu")) if "cpu" in meta else analysis["kind"] == "cpu"
    date = str(meta.get("date") or dt.datetime.now(dt.timezone.utc).date().isoformat())
    result = build_result(analysis, run, rocprof, date=date, cpu=cpu)
    apply_reference(result, chip_reference(chip_sheet))
    problems = check_result(result)
    if problems:
        return {"ok": False, "outcome": "invalid", "problems": problems}
    json_path, md_path = write_outputs(result, out_dir or DATA_DIR, force=force)
    seqs = []
    if ledger:
        from lab.machine.record import record

        try:
            source = str(json_path.resolve().relative_to(REPO)).replace("\\", "/")
        except ValueError:
            source = str(json_path)
        for row in ledger_rows(result, source):
            seqs.append(record(row["verb"], row["outcome"], row["metrics"], row["refs"], task=TASK, node=row["node"],
                               units=row["units"], conditions=row["conditions"], source=row["source"],
                               low_confidence=row["low_confidence"], note=row["note"]))
    return {"ok": True, "outcome": "ingested", "json": str(json_path), "markdown": str(md_path),
            "ledger_seqs": seqs, "step_ms": result["step_ms"].get("mean"), "shares": result["shares"],
            "open": result["open"]}


# --- self-check -------------------------------------------------------------------------------------------------------

def quick_main(args: argparse.Namespace) -> dict[str, Any]:
    """The parser, the result, its checks and both files on the two recorded fixtures (the synthetic ROCm trace and
    the recorded CPU trace); no GPU, no network, no ledger row."""
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="r34-quick-"))
    checks: dict[str, bool] = {}
    answers = {}
    for name in ("rocm", "cpu"):
        folder = FIXTURES / name
        answer = ingest(folder, out_dir=out / name, chip_sheet=folder / "chipsheet.json"
                        if (folder / "chipsheet.json").is_file() else None, ledger=False, force=True)
        answers[name] = answer
        checks[f"{name}_ingested"] = bool(answer.get("ok"))
        if answer.get("ok"):
            text = Path(answer["markdown"]).read_text(encoding="utf-8")
            checks[f"{name}_markdown_sections"] = all(s in text for s in MD_SECTIONS)
    expected = json.loads((FIXTURES / "rocm" / "expected.json").read_text(encoding="utf-8"))
    got = (answers.get("rocm") or {}).get("json")
    if got:
        result = json.loads(Path(got).read_text(encoding="utf-8"))
        checks["rocm_ms_per_step_match"] = all(
            abs((result["ms_per_step"].get(k) or 0.0) - v) <= 1e-3 for k, v in expected["ms_per_step"].items())
    return {"ok": all(checks.values()), "outcome": "quick", "checks": checks, "out": str(out),
            "answers": {k: {kk: v.get(kk) for kk in ("ok", "outcome", "json", "problems", "error")}
                        for k, v in answers.items()}}


# --- the run (lab/train.py under the profilers) ----------------------------------------------------------------------

def layer_kind(module_name: str) -> str | None:
    """The kind of a module this profile wraps in a range, from its qualified name; None for any other module."""
    if re.search(r"\.lora_[AB]\.[^.]+$", module_name):
        return "lora"
    last = module_name.rsplit(".", 1)[-1]
    if last == "linear_attn":
        return "gdn"
    if last == "self_attn":
        return "attention"
    if last == "mlp":
        return "mlp"
    if last in ("embed_tokens", "lm_head"):
        return "embed_head"
    return None


class LayerRanges:
    """Forward pre and post hooks that open and close `atlas::<kind>::<module>` ranges, per thread (gradient
    checkpointing runs a layer's forward again on the autograd thread)."""

    def __init__(self) -> None:
        self.local = threading.local()
        self.handles: list[Any] = []
        self.wrapped: dict[str, int] = defaultdict(int)

    def attach(self, model: Any) -> None:
        from torch.autograd.profiler import record_function

        for name, module in model.named_modules():
            kind = layer_kind(name)
            if kind is None:
                continue
            label = f"{TAG}{kind}::{name}"

            def pre(_module: Any, _inputs: Any, label: str = label) -> None:
                stack = getattr(self.local, "stack", None)
                if stack is None:
                    stack = self.local.stack = []
                ctx = record_function(label)
                ctx.__enter__()
                stack.append(ctx)

            def post(_module: Any, _inputs: Any, _output: Any) -> None:
                stack = getattr(self.local, "stack", None)
                if stack:
                    stack.pop().__exit__(None, None, None)

            self.handles.append(module.register_forward_pre_hook(pre))
            self.handles.append(module.register_forward_hook(post))
            self.wrapped[kind] += 1

    def detach(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def profiler_callback(make_profiler: Any, ranges: LayerRanges) -> Any:
    """A TrainerCallback that, when training begins, puts the layer ranges on the model and starts the profiler
    (so model loading and data preparation stay out of the first step), steps it at the end of every optimizer step
    and stops it when training ends."""
    from transformers import TrainerCallback

    class StepProfiler(TrainerCallback):
        profiler: Any = None

        def on_train_begin(self, args: Any, state: Any, control: Any, model: Any = None, **kwargs: Any) -> None:
            if model is not None and not ranges.handles:
                ranges.attach(model)
            self.profiler = make_profiler()
            self.profiler.start()

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            if self.profiler is not None:
                self.profiler.step()

        def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            if self.profiler is not None:
                self.profiler.stop()
                self.profiler = None
            ranges.detach()

    return StepProfiler()


def child_main(args: argparse.Namespace) -> dict[str, Any]:
    """In the run's own process: lab/train.py with the profiler callback (or plain, under rocprofv3). The profiler
    records every optimizer step (a schedule with one active window of --steps steps, which is what makes it mark
    each step with a ProfilerStep#<n> range) and writes trace.json when the window closes or training ends."""
    from lab import train

    argv = list(args.train_args)
    if args.mode == "rocprof":
        return train.main(argv)
    import torch

    activities = [torch.profiler.ProfilerActivity.CPU]
    if not args.cpu and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    ranges = LayerRanges()

    def make_profiler() -> Any:
        return torch.profiler.profile(
            activities=activities, record_shapes=True, profile_memory=True, with_stack=False,
            schedule=torch.profiler.schedule(wait=max(0, args.steps - args.active), warmup=0, active=args.active,
                                             repeat=1),
            on_trace_ready=lambda prof: prof.export_chrome_trace(str(trace_dir / "trace.json")))

    answer = train.main(argv, callbacks=[profiler_callback(make_profiler, ranges)])
    answer.setdefault("refs", {})["layer_ranges"] = json.dumps(dict(ranges.wrapped), sort_keys=True)
    out = (answer.get("refs") or {}).get("out")
    if out and Path(out).is_dir():
        (Path(out) / "run.json").write_text(json.dumps(answer, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return answer


def train_argv(args: argparse.Namespace, out: Path, dataset: str, rows: str) -> list[str]:
    argv = ["--dataset", dataset, "--rows", rows, "--model", args.model, "--max-steps", str(args.steps),
            "--out", str(out)]
    if args.cpu:
        argv.append("--cpu")
    return argv + [a for a in args.train_args if a != "--"]


def pass_data(args: argparse.Namespace, out: Path, name: str) -> tuple[str, str]:
    """The dataset and rows a pass trains on. On the pod lab/train.py deletes its data however a run ends, so each
    pass trains on a copy of its own (deleted by that pass) and the originals are deleted once every pass is done;
    a --cpu rehearsal keeps its data and trains on the originals."""
    if args.cpu:
        return args.dataset, args.rows
    source = Path(args.dataset).expanduser()
    folder = out / f"data-{name}" / source.name
    if folder.exists():
        shutil.rmtree(folder)
    shutil.copytree(source, folder)
    rows = out / f"data-{name}" / "rows.jsonl"
    shutil.copyfile(Path(args.rows).expanduser(), rows)
    return str(folder), str(rows)


def _read_meta(out: Path) -> dict[str, Any]:
    try:
        return json.loads((out / "profile.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def run_main(args: argparse.Namespace) -> dict[str, Any]:
    """lab/train.py for --steps steps under the torch profiler in a process of its own, then under rocprofv3 when
    it is on PATH (not with --cpu), then the ingest of the folder. --passes picks the passes, so each fits a slot;
    a second call with the same --out adds its pass to the folder."""
    stamp = dt.datetime.now(dt.timezone.utc)
    out = Path(args.out or f"train-profile-{stamp:%Y%m%dT%H%M%SZ}").expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    meta = _read_meta(out)
    meta.update({"cpu": bool(args.cpu), "steps": args.steps})
    meta.setdefault("date", stamp.date().isoformat())
    meta.setdefault("started", stamp.isoformat(timespec="seconds"))
    tools = meta.setdefault("tools", {})
    if not args.cpu and not args.no_install:   # as lab/train.py's own entry point does, around the pod's torch
        from lab import train

        installed = train.install()
        meta["install"] = installed.get("outcome")
        if not installed.get("ok"):
            return {"ok": False, "outcome": "install_failed", "error": installed.get("error"), "dir": str(out)}
    (out / "profile.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    me = [sys.executable, "-m", "lab.profile_train", "_child"]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (str(REPO), os.environ.get("PYTHONPATH")) if p)}
    if "torch" in args.passes:
        dataset, rows = pass_data(args, out, "torch")
        cmd = [*me, "--mode", "torch", "--trace-dir", str(out / "torch"), "--steps", str(args.steps),
               "--active", str(min(args.active or args.steps, args.steps)),
               *(["--cpu"] if args.cpu else []), "--",
               *train_argv(args, out / "train", dataset, rows)]
        done = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, check=False)
        (out / "torch.log").write_text(done.stderr[-200000:], encoding="utf-8")
        tools["torch"] = {"ok": done.returncode == 0, "rc": done.returncode}
        if done.returncode != 0:
            tools["torch"]["error"] = (done.stdout.strip().splitlines() or [done.stderr[-400:]])[-1][:400]
        if (out / "train" / "run.json").is_file():
            shutil.copy(out / "train" / "run.json", out / "run.json")
    if "rocprof" in args.passes:
        rocprof = shutil.which("rocprofv3")
        if rocprof and not args.cpu:
            dataset, rows = pass_data(args, out, "rocprof")
            cmd = [rocprof, "--kernel-trace", "--memory-copy-trace", "--output-format", "csv", "-d",
                   str(out / "rocprof"), "--", *me, "--mode", "rocprof", "--",
                   *train_argv(args, out / "train-rocprof", dataset, rows)]
            done = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, check=False)
            (out / "rocprof.log").write_text(done.stderr[-200000:], encoding="utf-8")
            tools["rocprofv3"] = {"ok": done.returncode == 0, "rc": done.returncode}
        else:
            tools["rocprofv3"] = {"ok": False, "skipped": "--cpu" if args.cpu else "not on PATH"}
    if not args.cpu and not args.keep_data:
        from lab import train

        deleted, how = train.delete_data(args.dataset, args.rows)
        meta["data_deleted"] = "yes" if deleted else f"no: {how}"
    (out / "profile.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    if not (tools.get("torch") or {}).get("ok"):
        return {"ok": False, "outcome": "train_failed", "dir": str(out), "tools": tools,
                "data_deleted": meta.get("data_deleted")}
    answer = ingest(out, out_dir=args.results, chip_sheet=args.chip_sheet, ledger=not args.no_ledger,
                    force=True)
    answer.update(dir=str(out), tools=tools, data_deleted=meta.get("data_deleted"))
    return answer


def record_main(args: argparse.Namespace) -> dict[str, Any]:
    """A result written on the pod (with --no-ledger) brought back: copy it and its Markdown into the data folder
    and write its ledger row here."""
    from lab.machine.record import record

    source = Path(args.result)
    result = json.loads(source.read_text(encoding="utf-8"))
    problems = check_result(result)
    if problems:
        return {"ok": False, "outcome": "invalid", "problems": problems}
    json_path, md_path = write_outputs(result, args.out or DATA_DIR, force=args.force)
    try:
        where = str(json_path.resolve().relative_to(REPO)).replace("\\", "/")
    except ValueError:
        where = str(json_path)
    seqs = [record(row["verb"], row["outcome"], row["metrics"], row["refs"], task=TASK, node=row["node"],
                   units=row["units"], conditions=row["conditions"], source=row["source"],
                   low_confidence=row["low_confidence"], note=row["note"]) for row in ledger_rows(result, where)]
    return {"ok": True, "outcome": "recorded", "json": str(json_path), "markdown": str(md_path),
            "ledger_seqs": seqs}


# --- the entry point ------------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m lab.profile_train", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd")
    check = sub.add_parser("self-check", help="the parser and writer on the recorded fixtures")
    check.add_argument("--quick", action="store_true")
    check.add_argument("--out", default=None)

    run = sub.add_parser("run", help="lab/train.py under the profilers, then ingest")
    run.add_argument("--dataset", required=True)
    run.add_argument("--rows", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    run.add_argument("--cpu", action="store_true", help="the rehearsal: lab/train.py --cpu, a tiny model")
    run.add_argument("--active", type=int, default=None,
                     help="profile only the last N of the steps (default all): a smaller trace when a slot is short")
    run.add_argument("--no-install", action="store_true", help="do not install lab/requirements-train.txt's pins")
    run.add_argument("--out", default=None, help="the run folder (default ./train-profile-<UTC stamp>)")
    run.add_argument("--results", default=None, help="where the JSON and Markdown go (default the data folder)")
    run.add_argument("--chip-sheet", default=None)
    run.add_argument("--passes", type=lambda t: [x for x in t.split(",") if x], default=["torch", "rocprof"],
                     help="comma list of torch and rocprof (default both)")
    run.add_argument("--keep-data", action="store_true", help=argparse.SUPPRESS)   # tests: keep the originals
    run.add_argument("--no-ledger", action="store_true")
    run.add_argument("--force", action="store_true")
    run.add_argument("train_args", nargs=argparse.REMAINDER, help="after --: more lab/train.py arguments")

    ing = sub.add_parser("ingest", help="a run folder -> the JSON, Markdown and ledger rows")
    ing.add_argument("run_dir")
    ing.add_argument("--out", default=None)
    ing.add_argument("--chip-sheet", default=None)
    ing.add_argument("--warmup", type=int, default=None, help="steps left out of the means (default 2 of 6 or more)")
    ing.add_argument("--no-ledger", action="store_true")
    ing.add_argument("--force", action="store_true")

    rec = sub.add_parser("record", help="a result JSON from the pod -> the data folder and its ledger row")
    rec.add_argument("result")
    rec.add_argument("--out", default=None)
    rec.add_argument("--force", action="store_true")

    child = sub.add_parser("_child", help=argparse.SUPPRESS)
    child.add_argument("--mode", choices=("torch", "rocprof"), default="torch")
    child.add_argument("--trace-dir", default="torch")
    child.add_argument("--cpu", action="store_true")
    child.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    child.add_argument("--active", type=int, default=DEFAULT_STEPS)
    child.add_argument("train_args", nargs=argparse.REMAINDER)
    return p


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Training step profile: self-check --quick here, run on the pod (or --cpu here), ingest a run folder."""
    parser = build_parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.cmd == "_child":
        args.train_args = [a for a in args.train_args if a != "--"]
        return child_main(args)
    handlers = {"self-check": quick_main, "run": run_main, "record": record_main}
    try:
        if args.cmd == "ingest":
            return ingest(args.run_dir, out_dir=args.out, chip_sheet=args.chip_sheet, ledger=not args.no_ledger,
                          force=args.force, warmup=args.warmup)
        if args.cmd not in handlers:
            parser.print_usage(sys.stderr)
            raise SystemExit(2)
        return handlers[args.cmd](args)
    except (ProfileError, OSError, ValueError) as exc:
        return {"ok": False, "outcome": "error", "error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    answer = main()
    print(json.dumps(answer))
    sys.exit(0 if answer.get("ok", True) else 1)

CLI_VERB = "profile-train"
