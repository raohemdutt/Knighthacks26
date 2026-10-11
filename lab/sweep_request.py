"""A request-side settings sweep of the teacher on one server spawn: decode tokens per second, time to first token,
watts and joules per token over a fixed grid of the settings a request carries, each cell against a baseline cell.

    python -m lab.sweep_request --dry-run [--quick]        # this machine: the grid, its run order and slot minutes
    python -m lab.sweep_request pod-script [--quick]       # this machine: writes lab/state/sweep_request/pod_run.sh
    python tooling/dispatcher/pod_exec.py D term <script> <timeout_s> > <output>   # the pod run (workspace tool)
    python -m lab.sweep_request collect <output> [--metrics-file]                  # this machine: files and rows
    python -m lab.sweep_request pod-run --expect-user-sha256 HEX [...]             # on the pod: what the script runs

`python -m lab sweep-request ...` is the same. Every call prints one JSON line and exits 0 when it worked.

The grid (lab/refine/research/kernels/data/sweep-request-grid.json, committed, so a run can be repeated) holds only
settings a request carries: the batch size (B concurrent streams, each sending its requests one after another), max
tokens, thinking off or on, the prompt length class (a prompt-set file) and the sampling profile. Each of its slices
changes some of its defaults and runs at every batch size; a cell is one slice at one batch size, `<slice>-b<B>`.
--quick keeps the slices the grid lists under `quick`. The grid names the baseline cell, and the best cell is the one
with the best median of the grid's `best_by` metric. Whatever needs a server restart stays at the pinned baseline of
the inference settings sheet: the server is launched from its search space's baseline mapping and fixed launch
(searchspace-inference-gfx1151.json), with lab/bench.py's stock vLLM line as the fallback when that launch dies
before it is ready; the result says which one served.

One run (pod-run, or run_sweep from Python): idle watts with no server, one spawn, the wait for /v1/models, idle
watts with the server loaded, one discarded warm-up round per batch size and prompt class (the grid's warm-up
tokens), then every cell `repeats` times with the repeats interleaved over the grid (serpentine: forward, backward,
forward, so slow drift lands on every cell alike), then the stop. A cell is measured by lab/bench.py's own
Session.measure on the shared server, so its numbers mean what the benchmark harness's mean: TTFT and decode rate from
the client's token stream, amd-smi SOCKET_POWER integrated over the decode window for watts and joules per token with
the idle readings beside them, and the harness's full conditions on every row (engine commit, patch, checkpoint,
kernels, KV cache, the environment read back from the server process, the host). A run never starts a cell that its
deadline (--deadline-min) cannot hold: that cell is a skipped row with the reason. Stuck streams end the measuring
as they end a shared spawn in the harness: the rest are rows marked not measured. The server is stopped however
the run ends.

The pod sends its result back through the terminal: a few headline lines, then one base64 block between two marker
lines (a gzip'd tar of raw.json and the serve logs' heads and tails, with its sha256); it also keeps the text in
~/sweep_request/last_result.b64. collect reads the block from the saved output and writes
lab/refine/research/kernels/data/sweep-request-gfx1151-<date>.json and .md (per cell, the median of its repeats with
min and max and the change against the baseline cell; the best cell and its change over the baseline with the range
the two cells' spreads allow), one `sweep_request.cell` ledger row per cell on the run's node (pod-b), and with
--metrics-file the baseline and best cells' headline medians and the change as metrics-file rows. Every row is
low_confidence: three repeats on one spawn are fewer than the five verdict() needs to call a difference.

The pod gate (pod-run, fail closed, before anything starts): the node is pod-b, the machine is x86_64, rocminfo names
gfx1151, the Jupyter user hashes to the account the script was made for (the script carries the sha256 of the user
name, never the name), the server port is free and no vLLM server runs there already. Pod C, which serves the live
teacher, never passes.
"""
from __future__ import annotations

import argparse
import ast
import base64
import dataclasses
import datetime as dt
import gzip
import hashlib
import io
import json
import math
import os
import platform
import re
import shlex
import signal
import statistics
import subprocess
import sys
import tarfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if not __package__:  # run by path: the repo root, one folder up, holds lab/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab import bench  # noqa: E402
from lab.machine import metrics as metrics_file  # noqa: E402
from lab.machine.record import record, recorded  # noqa: E402
from lab.machine.verdict import MIN_RUNS_PER_ARM  # noqa: E402

SWEEP = "lab/sweep_request.py 1"
GRID_SCHEMA = "atlas.sweep_request.grid/1"
RAW_SCHEMA = "atlas.sweep_request.raw/1"
RESULT_SCHEMA = "atlas.sweep_request/1"
REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "lab" / "refine" / "research" / "kernels" / "data"
DEFAULT_GRID = DATA_DIR / "sweep-request-grid.json"
STATE_DIR = REPO / "lab" / "state" / "sweep_request"
POD_SCRIPT = STATE_DIR / "pod_run.sh"
POD_DIR = "sweep_request"        # under the pod's home: the bundles, the runs and the last result
CHIP = "gfx1151"
THINKING = {"off": False, "on": True, "unset": None}
CELL_KEYS = ("thinking", "max_tokens", "prompt", "sampling", "requests_per_stream")
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p")
ORDERS = ("serpentine", "forward")
LAUNCH_NAMES = ("pinned", "stock")
# every per-cell number: (unit, whether higher is better)
METRICS: dict[str, tuple[str, bool]] = {
    "decode_tok_s": ("tok/s", True), "decode_tok_s_total": ("tok/s", True), "ttft_s": ("s", False),
    "watts_avg": ("W", False), "watts_peak": ("W", False), "joules_per_token": ("J/token", False),
    "joules_per_token_above_idle": ("J/token", False)}
HEADLINE = ("decode_tok_s", "decode_tok_s_total", "ttft_s", "joules_per_token", "watts_avg")
METRIC_PREFIX = "teacher"
SERVER_KEYS: dict[str, type] = {
    "account": str, "node": str, "launches": list, "search_space": str, "checkpoint": str, "engine_dir": str,
    "patch": str, "model": str, "max_len": int, "port": int, "run_dir": str, "ready_timeout_s": float,
    "request_timeout_s": float, "stream_grace_s": float, "idle_s": float, "power_period_s": float}
TIMING_KEYS = ("ready_s", "stop_s", "decode_tok_s_b1", "per_stream_slowdown", "ttft_s", "prefill_tok_s",
               "per_cell_overhead_s", "margin", "round_to_min")
SERVER_CONDITIONS = ("engine", "engine_commit", "engine_build", "patch_hash", "checkpoint", "served_model",
                     "quantization", "max_model_len", "rocm", "torch", "triton", "hip", "python", "linear_kernels",
                     "attention_backends", "kv_cache_tokens", "max_concurrency", "prefix_caching_log", "env",
                     "env_meant", "kfd_queues", "host", "unavailable")
VLLM_ENTRY = ("python3", "-m", "vllm.entrypoints.openai.api_server")
B64_LINE = 76
MARK_BEGIN = "=== sweep_request result bundle"
MARK_END = "=== end of sweep_request result bundle ==="
LOG_HEAD, LOG_TAIL = 200, 120


class SweepError(Exception):
    """A grid, a run or a result that cannot be used; the message says why."""


def log(message: str) -> None:
    print(f"[sweep {time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _rel(path: str | Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return str(path)


def _r(value: Any, digits: int = 4) -> float | None:
    return None if value is None or not math.isfinite(float(value)) else round(float(value), digits)


# --- the grid ---------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PromptClass:
    name: str
    file: str            # repo-relative, as the grid names it
    tokens_est: int

    @property
    def path(self) -> Path:
        return REPO / self.file


@dataclass(frozen=True)
class Cell:
    id: str
    slice: str
    batch: int
    thinking: str
    max_tokens: int
    prompt: str
    sampling: str
    requests_per_stream: int


@dataclass(frozen=True)
class Timing:
    ready_s: float
    stop_s: float
    decode_tok_s_b1: float
    per_stream_slowdown: float
    ttft_s: float
    prefill_tok_s: float
    per_cell_overhead_s: float
    margin: float
    round_to_min: int


@dataclass(frozen=True)
class Grid:
    data: dict[str, Any]
    sha256: str
    file: str | None
    batches: tuple[int, ...]
    repeats: int
    order: str
    defaults: dict[str, Any]
    prompts: dict[str, PromptClass]
    sampling: dict[str, dict[str, Any]]
    slices: tuple[dict[str, Any], ...]
    quick: tuple[str, ...]
    baseline_cell: str
    best_by: str
    warmup_tokens: int
    server: dict[str, Any]
    timing: Timing
    task: str | None

    def cells(self, quick: bool = False) -> list[Cell]:
        """Every slice (the quick ones only, with quick) at every batch size, in grid order."""
        out = []
        for one in self.slices:
            if quick and one["id"] not in self.quick:
                continue
            s = {**self.defaults, **one.get("set", {})}
            for batch in self.batches:
                out.append(Cell(f"{one['id']}-b{batch}", one["id"], batch, s["thinking"], s["max_tokens"], s["prompt"],
                                s["sampling"], s["requests_per_stream"]))
        return out


def _whole(value: Any, minimum: int = 1) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check_settings(where: str, settings: Any, prompts: Mapping[str, Any], sampling: Mapping[str, Any],
                    problems: list[str]) -> None:
    if not isinstance(settings, Mapping):
        problems.append(f"{where} is not an object")
        return
    for key, value in settings.items():
        if key not in CELL_KEYS:
            problems.append(f"{where} sets {key!r}, which is not a request setting ({', '.join(CELL_KEYS)})")
        elif key == "thinking" and value not in THINKING:
            problems.append(f"{where}: thinking is one of {', '.join(THINKING)}, not {value!r}")
        elif key in ("max_tokens", "requests_per_stream") and not _whole(value):
            problems.append(f"{where}: {key} is a whole number of at least 1, not {value!r}")
        elif key == "prompt" and value not in prompts:
            problems.append(f"{where}: prompt class {value!r} is not under prompts")
        elif key == "sampling" and value not in sampling:
            problems.append(f"{where}: sampling profile {value!r} is not under sampling")


def _check_sampling(name: str, profile: Any, problems: list[str]) -> None:
    if not isinstance(profile, Mapping):
        problems.append(f"sampling {name!r} is not an object")
        return
    for key in profile:
        if key not in (*SAMPLING_KEYS, "why"):
            problems.append(f"sampling {name!r} sets {key!r}; a profile sets {', '.join(SAMPLING_KEYS)}")
    t, p, k, m = (profile.get(key) for key in SAMPLING_KEYS)
    if not _number(t) or t < 0:
        problems.append(f"sampling {name!r} needs a temperature of 0 or more")
    if p is not None and (not _number(p) or not 0 < p <= 1):
        problems.append(f"sampling {name!r}: top_p is above 0 and at most 1")
    if k is not None and (not _whole(k, -1) or k == 0):
        problems.append(f"sampling {name!r}: top_k is -1 (off) or a whole number of at least 1")
    if m is not None and (not _number(m) or not 0 <= m < 1):
        problems.append(f"sampling {name!r}: min_p is at least 0 and below 1")


def _server_value_ok(value: Any, kind: type) -> bool:
    if kind is float:
        return _number(value) and value >= 0
    if kind is int:
        return _whole(value, 0)
    return isinstance(value, kind) and bool(value)


def parse_grid(data: Any, *, file: str | None = None, sha256: str | None = None,
               check_files: bool = True) -> Grid:
    """A grid from its JSON object, every problem named at once (SweepError). check_files also asks that the prompt
    sets and the search space it names exist in this checkout."""
    if not isinstance(data, Mapping):
        raise SweepError("the grid is not a JSON object")
    problems: list[str] = []
    if data.get("schema") != GRID_SCHEMA:
        problems.append(f"schema is {data.get('schema')!r}, not {GRID_SCHEMA}")
    batches = data.get("batches")
    if (not isinstance(batches, list) or not batches or not all(_whole(b) for b in batches)
            or len(set(batches)) != len(batches)):
        problems.append(f"batches is a list of distinct whole numbers of at least 1, not {batches!r}")
        batches = []
    repeats = data.get("repeats")
    if not _whole(repeats):
        problems.append(f"repeats is a whole number of at least 1, not {repeats!r}")
    order = data.get("order", "serpentine")
    if order not in ORDERS:
        problems.append(f"order is one of {', '.join(ORDERS)}, not {order!r}")
    prompts: dict[str, PromptClass] = {}
    entries = data.get("prompts")
    for name, entry in (entries.items() if isinstance(entries, Mapping) else ()):
        file_name = entry.get("file") if isinstance(entry, Mapping) else None
        tokens = entry.get("tokens_est") if isinstance(entry, Mapping) else None
        if (not isinstance(file_name, str) or not file_name or ".." in Path(file_name).parts
                or Path(file_name).is_absolute()):
            problems.append(f"prompt class {name!r} names no repo-relative file")
            continue
        if not _whole(tokens):
            problems.append(f"prompt class {name!r}: tokens_est is a whole number of at least 1")
            continue
        if check_files and not (REPO / file_name).is_file():
            problems.append(f"prompt class {name!r}: {file_name} is not in this checkout")
        prompts[name] = PromptClass(name, file_name, tokens)
    if not prompts:
        problems.append("prompts names no usable prompt class")
    sampling = data.get("sampling")
    if not isinstance(sampling, Mapping) or not sampling:
        problems.append("sampling names no profile")
        sampling = {}
    for name, profile in sampling.items():
        _check_sampling(name, profile, problems)
    defaults = data.get("defaults")
    if not isinstance(defaults, Mapping) or set(defaults) != set(CELL_KEYS):
        problems.append(f"defaults sets exactly {', '.join(CELL_KEYS)}")
        defaults = {}
    _check_settings("defaults", defaults, prompts, sampling, problems)
    slices = data.get("slices")
    if not isinstance(slices, list) or not slices:
        problems.append("slices is a non-empty list")
        slices = []
    seen_ids: list[str] = []
    seen_settings: dict[str, str] = {}
    for k, one in enumerate(slices):
        sid = one.get("id") if isinstance(one, Mapping) else None
        if not isinstance(sid, str) or not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", sid):
            problems.append(f"slice {k} has no id of lowercase letters, digits and hyphens")
            continue
        if sid in seen_ids:
            problems.append(f"slice id {sid!r} is used twice")
        seen_ids.append(sid)
        settings = one.get("set", {})
        _check_settings(f"slice {sid!r}", settings, prompts, sampling, problems)
        if isinstance(settings, Mapping):
            merged = json.dumps({**defaults, **settings}, sort_keys=True)
            if merged in seen_settings:
                problems.append(f"slices {seen_settings[merged]!r} and {sid!r} give the same request settings")
            seen_settings.setdefault(merged, sid)
    quick = data.get("quick")
    if not isinstance(quick, list) or not quick or any(q not in seen_ids for q in quick):
        problems.append(f"quick lists slice ids of this grid, not {quick!r}")
        quick = []
    ids_full = {f"{sid}-b{b}" for sid in seen_ids for b in batches}
    ids_quick = {f"{sid}-b{b}" for sid in quick for b in batches}
    baseline = data.get("baseline_cell")
    if baseline not in ids_full or baseline not in ids_quick:
        problems.append(f"baseline_cell {baseline!r} is not a cell of the grid and of its quick grid")
    best_by = data.get("best_by")
    if best_by not in METRICS:
        problems.append(f"best_by is one of {', '.join(METRICS)}, not {best_by!r}")
    warmup = data.get("warmup")
    warmup_tokens = warmup.get("max_tokens") if isinstance(warmup, Mapping) else None
    if not _whole(warmup_tokens):
        problems.append("warmup.max_tokens is a whole number of at least 1")
    server = data.get("server")
    if not isinstance(server, Mapping):
        problems.append("server is an object")
        server = {}
    for key, kind in SERVER_KEYS.items():
        if not _server_value_ok(server.get(key), kind):
            problems.append(f"server.{key} is missing or not a {kind.__name__}")
    if isinstance(server.get("launches"), list) and any(n not in LAUNCH_NAMES for n in server["launches"]):
        problems.append(f"server.launches holds names from {', '.join(LAUNCH_NAMES)}")
    if check_files and isinstance(server.get("search_space"), str) and not (REPO / server["search_space"]).is_file():
        problems.append(f"server.search_space {server['search_space']} is not in this checkout")
    timing = data.get("timing")
    if not isinstance(timing, Mapping) or any(not _number(timing.get(key)) or timing.get(key) < 0
                                               for key in TIMING_KEYS):
        problems.append(f"timing sets {', '.join(TIMING_KEYS)} as numbers of 0 or more")
        timing = {}
    elif timing["decode_tok_s_b1"] <= 0 or timing["prefill_tok_s"] <= 0 or not _whole(timing["round_to_min"]):
        problems.append("timing: decode_tok_s_b1 and prefill_tok_s are above 0, round_to_min a whole number of minutes")
    if problems:
        raise SweepError("the grid is not usable: " + "; ".join(problems))
    if sha256 is None:
        sha256 = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    return Grid(data=dict(data), sha256=sha256, file=file, batches=tuple(batches), repeats=repeats, order=order,
                defaults=dict(defaults), prompts=prompts, sampling={k: dict(v) for k, v in sampling.items()},
                slices=tuple(dict(s) for s in slices), quick=tuple(quick), baseline_cell=baseline, best_by=best_by,
                warmup_tokens=warmup_tokens, server=dict(server),
                timing=Timing(**{k: (int(timing[k]) if k == "round_to_min" else float(timing[k])) for k in TIMING_KEYS}),
                task=data.get("task") if isinstance(data.get("task"), str) else None)


def load_grid(path: str | Path = DEFAULT_GRID) -> Grid:
    path = Path(path)
    try:
        blob = path.read_bytes()
        data = json.loads(blob)
    except (OSError, ValueError) as exc:
        raise SweepError(f"cannot read the grid {path}: {exc}") from exc
    return parse_grid(data, file=_rel(path), sha256=hashlib.sha256(blob).hexdigest())


# --- the plan: run order and slot minutes -------------------------------------------------------------------------

def stream_rate(timing: Timing, batch: int) -> float:
    """The decode rate one stream is expected to see at this batch size (the timing model)."""
    return timing.decode_tok_s_b1 / (1.0 + timing.per_stream_slowdown * (batch - 1))


def cell_seconds(grid: Grid, cell: Cell, *, tokens: int | None = None) -> float:
    """Expected wall seconds of one run of a cell: each stream's requests one after another, each waiting for the
    prefill of every stream's prompt and then decoding its tokens."""
    t = grid.timing
    n = cell.max_tokens if tokens is None else tokens
    prompt = grid.prompts[cell.prompt].tokens_est
    one = t.ttft_s + cell.batch * prompt / t.prefill_tok_s + n / stream_rate(t, cell.batch)
    return t.per_cell_overhead_s + cell.requests_per_stream * one


def warmup_shapes(cells: Sequence[Cell]) -> list[Cell]:
    """The first cell of each batch size and prompt class: the shapes the warm-up runs once each."""
    out, seen = [], set()
    for cell in cells:
        if (cell.batch, cell.prompt) not in seen:
            seen.add((cell.batch, cell.prompt))
            out.append(cell)
    return out


def run_order(grid: Grid, cells: Sequence[Cell]) -> list[tuple[int, Cell]]:
    """(repeat, cell) in the order they run: every cell once per repeat; serpentine runs odd repeats backward."""
    out: list[tuple[int, Cell]] = []
    for repeat in range(grid.repeats):
        seq = list(cells) if grid.order == "forward" or repeat % 2 == 0 else list(reversed(cells))
        out += [(repeat, cell) for cell in seq]
    return out


def estimate(grid: Grid, cells: Sequence[Cell]) -> dict[str, Any]:
    """The seconds a run needs by the grid's timing model, and the slot minutes with the margin, rounded up."""
    t = grid.timing
    idle = 2.0 * float(grid.server.get("idle_s", 0.0))
    warm = sum(cell_seconds(grid, c, tokens=min(grid.warmup_tokens, c.max_tokens)) for c in warmup_shapes(cells))
    per_repeat = sum(cell_seconds(grid, c) for c in cells)
    total = idle + t.ready_s + warm + grid.repeats * per_repeat + t.stop_s
    minutes = math.ceil(total * (1.0 + t.margin) / 60.0 / t.round_to_min) * t.round_to_min
    return {"idle_s": round(idle, 1), "ready_s": t.ready_s, "warmup_s": round(warm, 1),
            "per_repeat_s": round(per_repeat, 1), "measure_s": round(grid.repeats * per_repeat, 1),
            "stop_s": t.stop_s, "total_s": round(total, 1), "margin": t.margin, "slot_minutes": int(minutes)}


# --- the server launch ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Launch:
    name: str
    argv: tuple[str, ...]           # a template: {port} {checkpoint} {model} {max_len} {log} {port_file}
    env: dict[str, str]
    unset: tuple[str, ...] = ()

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "argv": list(self.argv), "env": dict(self.env), "unset": list(self.unset)}


def pinned_launch(space: Mapping[str, Any], *, model: str, run_dir: str) -> Launch:
    """The vLLM launch of a search space's pinned baseline: its fixed launch arguments and environment, the flags and
    variables its mapping gives each searched setting's baseline value, and its unset list removed from the
    environment (through env -u, which execs the server in the same process). {run_dir} in a value is run_dir."""
    problems = []
    fixed = space.get("fixed_launch") or {}
    args = [str(a) for a in fixed.get("args") or []]
    env = {str(k): str(v) for k, v in (fixed.get("env") or {}).items()}
    for param in space.get("parameters") or []:
        if param.get("type") != "choice":
            continue
        name = param.get("name")
        value = (space.get("baseline") or {}).get(name)
        key = json.dumps(value) if isinstance(value, bool) else str(value)
        entry = ((space.get("mapping") or {}).get(name) or {}).get(key)
        if entry is None:
            problems.append(f"the search space maps no launch for {name} = {key}")
            continue
        args += [str(a) for a in entry.get("args") or []]
        env.update({str(k): str(v).replace("{run_dir}", run_dir) for k, v in (entry.get("env") or {}).items()})
    if "--served-model-name" in args:
        at = args.index("--served-model-name")
        if args[at + 1:at + 2] != [model]:
            problems.append(f"the search space serves {args[at + 1:at + 2]}, the grid asks for {model!r}")
    else:
        args += ["--served-model-name", "{model}"]
    if problems:
        raise SweepError("; ".join(problems))
    unset = tuple(name for name in fixed.get("unset_env") or [] if name not in env)
    argv = [*VLLM_ENTRY, "--model", "{checkpoint}", "--host", "127.0.0.1", "--port", "{port}", *args]
    if unset:
        argv = ["env", *[part for name in unset for part in ("-u", name)], *argv]
    return Launch("pinned", tuple(argv), env, unset)


def stock_launch() -> Launch:
    """lab/bench.py's vLLM line: the flags lab/pod_a_teacher.sh serves with, TRITON_ATTN for the encoder on gfx115x."""
    return Launch("stock", tuple(bench.default_command("vllm", CHIP)), {}, ())


def grid_launches(grid: Grid, *, run_dir: str) -> list[Launch]:
    try:
        space = json.loads((REPO / grid.server["search_space"]).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SweepError(f"cannot read the search space {grid.server['search_space']}: {exc}") from exc
    return [pinned_launch(space, model=grid.server["model"], run_dir=run_dir) if name == "pinned"
            else stock_launch() for name in grid.server["launches"]]


# --- cells on the harness ------------------------------------------------------------------------------------------

@dataclass
class SampledPromptSet(bench.PromptSet):
    """A bench prompt set whose request bodies carry a sampling profile's temperature, top_p, top_k and min_p in place
    of the harness's greedy values."""

    sampling: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def wrap(cls, prompts: bench.PromptSet, profile: Mapping[str, Any]) -> "SampledPromptSet":
        values = {f.name: getattr(prompts, f.name) for f in dataclasses.fields(bench.PromptSet)}
        return cls(**values, sampling={k: profile[k] for k in SAMPLING_KEYS if profile.get(k) is not None})

    def request_body(self, index: int, settings: bench.Settings, nonce: str | None) -> dict[str, Any]:
        body = super().request_body(index, settings, nonce)
        body.update(self.sampling)
        return body


class ReadOnce:
    """A host's command runner that runs each read of a fixed fact once (amd-smi static and firmware, git, the
    interpreter's versions) and every other command, such as the amd-smi metric read the watts come from, each time."""

    def __init__(self, run: Callable[[Sequence[str]], str]) -> None:
        self.run = run
        self.seen: dict[tuple[str, ...], tuple[bool, Any]] = {}

    @staticmethod
    def fixed(argv: Sequence[str]) -> bool:
        tool = Path(argv[0]).name if argv else ""
        return ((tool == "amd-smi" and list(argv[1:2]) in (["static"], ["firmware"])) or tool == "git"
                or tool.startswith("python"))

    def __call__(self, argv: Sequence[str]) -> str:
        argv = [str(a) for a in argv]
        if not self.fixed(argv):
            return self.run(argv)
        key = tuple(argv)
        if key not in self.seen:
            try:
                self.seen[key] = (True, self.run(argv))
            except Exception as exc:  # a fact that cannot be read stays unreadable for the run
                self.seen[key] = (False, exc)
        ok, value = self.seen[key]
        if ok:
            return value
        raise value


def read_once(host: bench.Host) -> bench.Host:
    if not isinstance(host.run, ReadOnce):
        host.run = ReadOnce(host.run)
    return host


@dataclass
class CellRun:
    cell: Cell
    session: bench.Session
    conditions: dict[str, Any]
    seconds: list[float] = field(default_factory=list)


def cell_config(grid: Grid, cell: Cell, base: bench.Config) -> bench.Config:
    """The harness configuration of one cell: the shared server's, with the cell's request settings."""
    return dataclasses.replace(base, batches=[cell.batch], max_tokens=cell.max_tokens, thinking=THINKING[cell.thinking],
                               prompts=grid.prompts[cell.prompt].path, requests_per_stream=cell.requests_per_stream)


def prepare_cell(grid: Grid, cell: Cell, base: bench.Config, *, host: bench.Host, power: Any, endpoint: Any,
                 pid: int | None, log_text: str | None, sweep: Mapping[str, Any]) -> CellRun:
    """A harness session for the cell on the running server, with its conditions read once: the harness's own, the
    sampling profile, and the sweep's (run, cell, slice, launch, grid)."""
    session = bench.Session(cell_config(grid, cell, base), host=host, power=power)
    first_dir = session.dir
    session.id = f"{sweep['run_id']}-{cell.id}"
    session.dir = Path(base.out_dir) / session.id
    session.dir.mkdir(parents=True, exist_ok=True)
    session.rows_path = session.dir / "rows.jsonl"
    if first_dir != session.dir:
        try:
            first_dir.rmdir()
        except OSError:
            pass
    profile = grid.sampling[cell.sampling]
    session.prompts = SampledPromptSet.wrap(session.prompts, profile)
    session.settings.temperature = float(profile.get("temperature", 0.0))
    conditions = session.server_conditions(session.static_conditions(), pid=pid, log_text=log_text, attached=False)
    session.served_model(endpoint, conditions)
    conditions.update({
        "temperature": session.settings.temperature, "top_p": profile.get("top_p", 1.0),
        "sampling": {"profile": cell.sampling, **session.prompts.sampling},
        "sweep": {**sweep, "cell": cell.id, "slice": cell.slice, "prompt_class": cell.prompt,
                  "thinking": cell.thinking, "max_tokens": cell.max_tokens,
                  "requests_per_stream": cell.requests_per_stream}})
    return CellRun(cell, session, conditions)


def _tail(path: Path, lines: int = 20) -> str | None:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return None


def start_server(launches: Sequence[Launch], *, values: Mapping[str, Any], run_dir: Path, port: int,
                 server: Mapping[str, Any], deadline: float | None, tried: list[dict[str, Any]],
                 say: Callable[[str], None]) -> tuple[Any, Any, Launch, list[str], Path]:
    """Spawn the launches in order until one answers /v1/models; each that dies or never answers is stopped and
    listed in tried with its error and the tail of its log."""
    for k, launch in enumerate(launches):
        timeout = float(server["ready_timeout_s"])
        if deadline is not None:
            timeout = min(timeout, deadline - time.monotonic())
            if timeout <= 0:
                tried.append({"launch": launch.name, "argv": None, "error": "no time left before the deadline",
                              "log_tail": None})
                break
        log_path = run_dir / f"serve-{k}-{launch.name}.log"
        port_file = run_dir / f"serve-{k}.port"
        argv = bench.fill_command(launch.argv, {**values, "port_file": port_file, "log": log_path})
        say(f"starting the {launch.name} launch: {shlex.join(argv)[:400]}")
        spawned = endpoint = None
        try:
            spawned = bench.spawn_server(argv, port=port, log_path=log_path, env=launch.env, cwd=run_dir,
                                         own_log=False)
            got = port or bench.wait_port(spawned, port_file, timeout)
            endpoint = bench.DirectEndpoint(f"http://127.0.0.1:{got}", timeout_s=float(server["request_timeout_s"]))
            bench.wait_ready(spawned, endpoint, timeout)
        except bench.BenchError as exc:
            if endpoint is not None:
                endpoint.close()
            if spawned is not None:
                spawned.stop()
            tried.append({"launch": launch.name, "argv": argv, "error": str(exc), "log_tail": _tail(log_path)})
            say(f"the {launch.name} launch did not come up: {exc}")
            continue
        tried.append({"launch": launch.name, "argv": argv, "error": None, "log_tail": None,
                      "ready_s": _r(spawned.ready - spawned.spawned, 3)})
        say(f"the {launch.name} launch is ready after {spawned.ready - spawned.spawned:.0f} s (pid {spawned.pid})")
        return spawned, endpoint, launch, argv, log_path
    raise SweepError("no launch came up: " + "; ".join(f"{t['launch']}: {t['error']}" for t in tried))


def warm_up(endpoint: Any, runs: Sequence[CellRun], *, tokens: int, server: Mapping[str, Any],
            say: Callable[[str], None]) -> list[dict[str, Any]]:
    """One discarded round per batch size and prompt class, at the cell's own settings with at most `tokens` tokens."""
    out = []
    for cell in warmup_shapes([run.cell for run in runs]):
        run = next(r for r in runs if r.cell is cell)
        settings = dataclasses.replace(run.session.settings, max_tokens=min(tokens, cell.max_tokens))
        began = time.monotonic()
        results = bench.run_batch(endpoint, run.session.prompts, settings, batch=cell.batch, per_stream=1,
                                  role="warmup", clock=run.session.clock, tag=f"{run.session.id}-warmup",
                                  timeout_s=float(server["request_timeout_s"]),
                                  grace_s=float(server["stream_grace_s"]))
        prompt_tokens = [r.prompt_tokens for r in results if r.prompt_tokens]
        out.append({"cell": cell.id, "batch": cell.batch, "prompt": cell.prompt, "max_tokens": settings.max_tokens,
                    "requests": len(results), "errors": [r.error for r in results if r.error],
                    "seconds": _r(time.monotonic() - began, 3),
                    "prompt_tokens_median": statistics.median(prompt_tokens) if prompt_tokens else None})
        say(f"warm-up at batch {cell.batch}, {cell.prompt} prompts: {out[-1]['seconds']} s")
    return out


def run_sweep(grid: Grid, cells: Sequence[Cell], launches: Sequence[Launch], *, host: bench.Host, power: Any,
              out_dir: str | Path, checkpoint: str, engine_dir: str | None = None, patches: Sequence[str] = (),
              node: str | None = None, port: int | None = None, hub: str | None = None,
              account: str | None = None, python: str = "python3", deadline_s: float | None = None,
              idle_s: float | None = None, quick: bool = False,
              say: Callable[[str], None] = log) -> dict[str, Any]:
    """Spawn once, measure every cell `repeats` times, stop; the raw result (also written as <run dir>/raw.json).
    Whatever happens is in it, never raised: a launch that never came up, an interrupt, stuck streams. port and
    idle_s replace the grid's server values when given."""
    server = grid.server
    node = node or server["node"]
    port = server["port"] if port is None else port
    idle_s = float(server["idle_s"]) if idle_s is None else idle_s
    deadline = None if deadline_s is None else time.monotonic() + deadline_s
    run_id = "sweep-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + os.urandom(2).hex()
    run_dir = Path(out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    order = run_order(grid, cells)
    raw: dict[str, Any] = {
        "schema": RAW_SCHEMA, "sweep": SWEEP, "harness": bench.HARNESS, "run_id": run_id, "node": node, "hub": hub,
        "account": account, "quick": quick, "started": bench.utc(), "ended": None,
        "grid": {"file": grid.file, "sha256": grid.sha256, "data": grid.data},
        "cells": [dataclasses.asdict(c) for c in cells], "order": [[r, c.id] for r, c in order],
        "deadline_s": deadline_s, "estimate": estimate(grid, cells),
        "launch": {"served": None, "tried": [], "launches": [launch.describe() for launch in launches]},
        "spawn": None, "idle_w": {"no_server": None, "loaded": None},
        "power": {"available": bool(power.available), "reason": power.reason, "period_s": power.period_s},
        "warmup": [], "rows": [], "error": None, "run_dir": str(run_dir)}
    spawned = endpoint = None
    spawn: dict[str, Any] = {}
    power.start()
    try:
        raw["idle_w"]["no_server"] = bench.idle_watts(power, idle_s)
        values = {"port": port, "batch": max(c.batch for c in cells), "max_len": server["max_len"],
                  "model": server["model"], "checkpoint": checkpoint}
        spawned, endpoint, launch, argv, log_path = start_server(
            launches, values=values, run_dir=run_dir, port=port, server=server, deadline=deadline,
            tried=raw["launch"]["tried"], say=say)
        raw["launch"]["served"] = launch.name
        spawn = {"index": 0, "pid": spawned.pid, "cold": True, "attached": False, "spawned": bench.utc(spawned.spawned),
                 "spawned_t": spawned.spawned, "ready_s": _r(spawned.ready - spawned.spawned, 3), "stopped": False,
                 "log": str(log_path), "shared": True, "command": argv, "launch": launch.name}
        raw["spawn"] = {k: v for k, v in spawn.items() if k != "spawned_t"}
        log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else None
        raw["idle_w"]["loaded"] = bench.idle_watts(power, idle_s)
        base = bench.Config(
            engine="vllm", checkpoint=checkpoint, model=server["model"], max_len=server["max_len"],
            engine_dir=engine_dir, patches=list(patches), launch_env=dict(launch.env), node=node, port=port,
            add_64=False, repeats=1, share_spawn=True, warmup_rounds=0, probe=False, idle_s=0.0,
            power_period_s=power.period_s, request_timeout_s=float(server["request_timeout_s"]),
            stream_grace_s=float(server["stream_grace_s"]), ready_timeout_s=float(server["ready_timeout_s"]),
            out_dir=run_dir / "cells", python=python, hub=hub, account=account, record=False)
        sweep = {"run_id": run_id, "launch": launch.name, "grid_sha256": grid.sha256, "shared_spawn": True,
                 "baseline_cell": grid.baseline_cell}
        runs = {c.id: prepare_cell(grid, c, base, host=host, power=power, endpoint=endpoint, pid=spawned.pid,
                                   log_text=log_text, sweep=sweep) for c in cells}
        raw["warmup"] = warm_up(endpoint, [runs[c.id] for c in cells], tokens=grid.warmup_tokens, server=server,
                                say=say)
        idle = dict(raw["idle_w"])
        stuck = None
        for k, (repeat, cell) in enumerate(order):
            run = runs[cell.id]
            if stuck is not None:
                row = run.session.blank_row(repeat=repeat, batch=cell.batch, spawn=spawn, conditions=run.conditions)
                row["outcome"], row["reason"] = "fail", f"not measured: {stuck}"
            else:
                need = run.seconds[-1] if run.seconds else cell_seconds(grid, cell)
                left = None if deadline is None else deadline - grid.timing.stop_s - time.monotonic()
                if left is not None and need > left:
                    row = run.session.blank_row(repeat=repeat, batch=cell.batch, spawn=spawn,
                                                conditions=run.conditions)
                    row["outcome"] = "skipped"
                    row["reason"] = (f"not run: past the deadline (the cell takes about {need:.0f} s and "
                                     f"{max(left, 0.0):.0f} s were left)")
                else:
                    say(f"{k + 1}/{len(order)}: {cell.id}, repeat {repeat}")
                    began = time.monotonic()
                    try:
                        row = run.session.measure(endpoint, bench.PlannedBatch(cell.batch), repeat=repeat, spawn=spawn,
                                                  conditions=run.conditions, idle=idle, probe_now=False)
                    except bench.StuckRun as exc:   # the server may still decode them: nothing more is measured
                        row = exc.row
                        stuck = f"{row['run_id']} ({cell.id}, repeat {repeat}) had stuck streams: {row['reason']}"
                    run.seconds.append(time.monotonic() - began)
                    row["elapsed_s"] = _r(run.seconds[-1], 3)
            row["cell"], row["order"] = cell.id, k
            run.session.keep(row)
            raw["rows"].append(row)
        if stuck is not None:
            raw["error"] = stuck
    except (SweepError, bench.BenchError) as exc:
        raw["error"] = str(exc)
    except (KeyboardInterrupt, SystemExit) as exc:
        raw["error"] = f"interrupted ({type(exc).__name__}: {exc})"
    finally:
        if endpoint is not None:
            endpoint.close()
        if spawned is not None:
            stopped = spawned.stop()
            if raw["spawn"] is not None:
                raw["spawn"]["stopped"] = stopped
        power.stop()
        raw["ended"] = bench.utc()
        for row in raw["rows"]:
            row["spawn"] = dict(raw["spawn"] or {})
        (run_dir / "raw.json").write_text(json.dumps(raw, default=str), encoding="utf-8")
    return raw


# --- the result block that crosses the terminal --------------------------------------------------------------------

def make_bundle(files: Mapping[str, bytes]) -> bytes:
    """A gzip'd tar of name -> bytes, the same bytes for the same files (sorted names, no times, no owners)."""
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name in sorted(files):
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(files[name]), 0, 0o644
            tar.addfile(info, io.BytesIO(files[name]))
    return gzip.compress(out.getvalue(), compresslevel=9, mtime=0)


def b64_lines(payload: bytes) -> list[str]:
    body = base64.b64encode(payload).decode("ascii")
    return [body[i:i + B64_LINE] for i in range(0, len(body), B64_LINE)]


def bundle_text(payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    return (f"{MARK_BEGIN} sha256={digest} bytes={len(payload)} ===\n" + "\n".join(b64_lines(payload))
            + f"\n{MARK_END}\n")


def decode_text(data: bytes) -> str:
    """Saved terminal output as text, whatever wrote it: UTF-8 with or without a BOM, or UTF-16 (PowerShell's >)."""
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16")
    if data[:3] == b"\xef\xbb\xbf":
        return data[3:].decode("utf-8", "replace")
    if len(data) > 1 and data[1:2] == b"\x00":
        return data.decode("utf-16-le", "replace")
    return data.decode("utf-8", "replace")


def read_bundle(data: bytes) -> dict[str, bytes]:
    """name -> bytes from the last result block in saved output (or a bare .tar.gz); its sha256 and length must
    match the marker line."""
    if data[:2] == b"\x1f\x8b":
        payload = data
    else:
        text = decode_text(data).replace("\r", "")
        begin = text.rfind(MARK_BEGIN)
        if begin < 0:
            raise SweepError(f"no '{MARK_BEGIN}' line in the input: the pod run did not finish or its output was not "
                             f"saved; the pod keeps the block in ~/{POD_DIR}/last_result.b64")
        header_end = text.find("\n", begin)
        end = text.find(MARK_END, header_end)
        if header_end < 0 or end < 0:
            raise SweepError("the result block has no end line: the output was cut short")
        m = re.search(r"sha256=([0-9a-f]{64}) bytes=(\d+)", text[begin:header_end])
        if not m:
            raise SweepError("the result block's first line carries no sha256 and length")
        payload = base64.b64decode(re.sub(r"[^A-Za-z0-9+/=]", "", text[header_end:end]))
        if len(payload) != int(m.group(2)) or hashlib.sha256(payload).hexdigest() != m.group(1):
            raise SweepError("the result block does not match its sha256 and length: it was damaged in transit")
    out: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(payload)), mode="r") as tar:
        for member in tar.getmembers():
            name = member.name
            if not member.isfile() or "/" in name or "\\" in name or name.startswith("."):
                continue
            handle = tar.extractfile(member)
            if handle is not None:
                out[name] = handle.read()
    return out


def _head_tail(text: str) -> str:
    lines = text.splitlines()
    if len(lines) <= LOG_HEAD + LOG_TAIL:
        return text
    return "\n".join([*lines[:LOG_HEAD], f"... {len(lines) - LOG_HEAD - LOG_TAIL} lines left out ...",
                      *lines[-LOG_TAIL:]]) + "\n"


def result_files(raw: Mapping[str, Any]) -> dict[str, bytes]:
    """raw.json and the head and tail of each serve log in the run folder."""
    files = {"raw.json": json.dumps(raw, separators=(",", ":"), default=str).encode("utf-8")}
    for path in sorted(Path(raw.get("run_dir") or ".").glob("serve-*.log")):
        files[path.name] = _head_tail(path.read_text(encoding="utf-8", errors="replace")).encode("utf-8")
    return files


def report(raw: Mapping[str, Any]) -> str:
    """What the pod prints: headline lines from the summary, then the result block."""
    try:
        lines = headline_lines(summarize(raw))
    except Exception as exc:  # the block must come back even when the summary cannot be made on the pod
        lines = [f"no summary on the pod: {type(exc).__name__}: {exc}"]
    return "\n".join(lines) + "\n" + bundle_text(make_bundle(result_files(raw)))


# --- the summary ---------------------------------------------------------------------------------------------------

def row_value(row: Mapping[str, Any], metric: str) -> float | None:
    if metric in ("watts_avg", "watts_peak"):
        return (row.get("watts") or {}).get("avg_w" if metric == "watts_avg" else "peak_w")
    return row.get(metric)


def spread(values: Iterable[float | None]) -> dict[str, Any]:
    """bench.stats plus spread_pct, (max - min) / median in percent."""
    s = bench.stats(values)
    s["spread_pct"] = (s["max"] - s["min"]) / s["median"] * 100.0 if s["n"] and s["median"] else None
    return {k: (v if k == "n" else _r(v, 6)) for k, v in s.items()}


def change_pct(value: float | None, base: float | None) -> float | None:
    if value is None or base is None or base == 0:
        return None
    return _r((value / base - 1.0) * 100.0, 3)


def pick(cells: Sequence[Mapping[str, Any]], metric: str) -> Mapping[str, Any] | None:
    """The cell with the best median of metric (higher or lower, by METRICS), the earlier cell on a tie."""
    found = [c for c in cells if c["stats"][metric]["median"] is not None]
    if not found:
        return None
    sign = 1.0 if METRICS[metric][1] else -1.0
    return max(found, key=lambda c: sign * c["stats"][metric]["median"])


def summarize(raw: Mapping[str, Any]) -> dict[str, Any]:
    """The result of a run, pure: per cell its repeats, the median with min and max of every metric and the change
    against the baseline cell; the best cell by the grid's best_by and its change over the baseline with the range
    the two cells' min and max allow; the lowest joules per token; the server's conditions; what stays open."""
    if raw.get("schema") != RAW_SCHEMA:
        raise SweepError(f"not a sweep result: schema {raw.get('schema')!r}")
    grid = parse_grid(raw["grid"]["data"], file=raw["grid"].get("file"), sha256=raw["grid"].get("sha256"),
                      check_files=False)
    cells = [Cell(**c) for c in raw["cells"]]
    by_cell: dict[str, list[Mapping[str, Any]]] = {}
    for row in raw.get("rows") or []:
        by_cell.setdefault(row.get("cell"), []).append(row)
    out: list[dict[str, Any]] = []
    for cell in cells:
        rows = sorted(by_cell.get(cell.id, []), key=lambda r: (r.get("repeat") or 0, r.get("order") or 0))
        ok = [r for r in rows if r.get("outcome") == "ok"]
        tokens_in = [q["prompt_tokens"] for r in ok for q in r.get("requests") or [] if q.get("prompt_tokens")]
        reasons = sorted({str(r.get("reason")) for r in rows if r.get("outcome") != "ok" and r.get("reason")})
        if not rows:
            reasons = [f"not measured: {raw.get('error') or 'no row for this cell'}"]
        out.append({
            **dataclasses.asdict(cell), "outcome": "ok" if len(ok) >= grid.repeats else "partial" if ok else "fail",
            "n": len(ok), "reasons": reasons,
            "prompt_tokens": statistics.median(tokens_in) if tokens_in else None,
            "repeats": [{"repeat": r.get("repeat"), "run_id": r.get("run_id"), "outcome": r.get("outcome"),
                         "reason": r.get("reason"), **{m: row_value(r, m) for m in METRICS},
                         "tokens_generated": r.get("tokens_generated"), "window_s": r.get("window_s"),
                         "in_flight_mean": r.get("in_flight_mean"), "elapsed_s": r.get("elapsed_s")} for r in rows],
            "stats": {m: spread(row_value(r, m) for r in ok) for m in METRICS}})
    base = next((c for c in out if c["id"] == grid.baseline_cell), None)
    for c in out:
        c["vs_baseline_pct"] = {m: change_pct(c["stats"][m]["median"], base["stats"][m]["median"]) if base else None
                                for m in METRICS}
    best = pick(out, grid.best_by)
    best_entry = None
    if best is not None:
        b = base["stats"][grid.best_by] if base else None
        x = best["stats"][grid.best_by]
        best_entry = {"cell": best["id"], "metric": grid.best_by, "higher_is_better": METRICS[grid.best_by][1],
                      "value": x, "baseline": b, "change_pct": None, "range_pct": None}
        if b and b["median"]:
            best_entry["change_pct"] = change_pct(x["median"], b["median"])
            best_entry["range_pct"] = [change_pct(x["min"], b["max"]), change_pct(x["max"], b["min"])]
    lowest = pick(out, "joules_per_token")
    rows_all = raw.get("rows") or []
    sample = next((r for r in rows_all if r.get("outcome") == "ok"), None) or next(iter(rows_all), None)
    conditions = (sample or {}).get("conditions") or {}
    tried = (raw.get("launch") or {}).get("tried") or []
    served = (raw.get("launch") or {}).get("served")
    open_items = []
    if raw.get("error"):
        open_items.append(f"the run ended early: {raw['error']}")
    if served and tried and tried[0].get("error") and tried[0].get("launch") != served:
        open_items.append(f"the {tried[0]['launch']} launch did not come up ({tried[0]['error']}); the {served} "
                          "launch served, so the server is not at the sheet's pinned baseline launch")
    for c in out:
        if c["outcome"] != "ok":
            open_items.append(f"{c['id']}: {c['n']} of {grid.repeats} repeats measured"
                              + (f" ({'; '.join(c['reasons'])})" if c["reasons"] else ""))
    if not (raw.get("power") or {}).get("available"):
        open_items.append(f"no watts: {(raw.get('power') or {}).get('reason')}")
    started = str(raw.get("started") or "")
    return {
        "schema": RESULT_SCHEMA, "sweep": SWEEP, "harness": raw.get("harness"), "run_id": raw.get("run_id"),
        "node": raw.get("node"), "hub": raw.get("hub"), "account": raw.get("account"),
        "chip": (conditions.get("host") or {}).get("chip") or CHIP, "date": started[:10],
        "started": raw.get("started"), "ended": raw.get("ended"), "quick": bool(raw.get("quick")),
        "grid": {"file": grid.file, "sha256": grid.sha256, "task": grid.task, "repeats": grid.repeats,
                 "order": grid.order, "batches": list(grid.batches), "best_by": grid.best_by,
                 "baseline_cell": grid.baseline_cell, "baseline_source": grid.data.get("baseline_source"),
                 "held": grid.data.get("held"),
                 "prompts": {k: {"file": v.file, "tokens_est": v.tokens_est} for k, v in grid.prompts.items()},
                 "sampling": grid.sampling},
        "launch": {"served": served,
                   "tried": [{k: t.get(k) for k in ("launch", "error", "ready_s", "argv")} for t in tried]},
        "spawn": raw.get("spawn"), "idle_w": raw.get("idle_w"), "power": raw.get("power"),
        "warmup": raw.get("warmup"), "estimate": raw.get("estimate"), "deadline_s": raw.get("deadline_s"),
        "error": raw.get("error"), "server": {k: conditions.get(k) for k in SERVER_CONDITIONS if k in conditions},
        "cells": out, "baseline": grid.baseline_cell if base is not None else None, "best": best_entry,
        "lowest_joules_per_token": None if lowest is None else {
            "cell": lowest["id"], "value": lowest["stats"]["joules_per_token"],
            "change_pct": lowest["vs_baseline_pct"]["joules_per_token"]},
        "open": open_items}


def _fmt(value: float | None, digits: int) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _cell_value(stats: Mapping[str, Any], digits: int) -> str:
    if stats["median"] is None:
        return "-"
    if stats["n"] == 1:
        return _fmt(stats["median"], digits)
    return f"{_fmt(stats['median'], digits)} ({_fmt(stats['min'], digits)}-{_fmt(stats['max'], digits)})"


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:+.1f}%"


def headline_lines(summary: Mapping[str, Any]) -> list[str]:
    cells = summary["cells"]
    done = sum(1 for c in cells if c["outcome"] == "ok")
    lines = [f"sweep {summary['run_id']}: {done} of {len(cells)} cells measured in full, launch "
             f"{summary['launch']['served']}" + (f", error: {summary['error']}" if summary.get("error") else "")]
    base = next((c for c in cells if c["id"] == summary.get("baseline")), None)
    if base is not None:
        s = base["stats"]
        lines.append(f"baseline {base['id']}: decode {_fmt(s['decode_tok_s']['median'], 2)} tok/s per stream, "
                     f"{_fmt(s['decode_tok_s_total']['median'], 2)} total, TTFT {_fmt(s['ttft_s']['median'], 2)} s, "
                     f"{_fmt(s['joules_per_token']['median'], 2)} J/token")
    best = summary.get("best")
    if best:
        rng = best.get("range_pct") or [None, None]
        lines.append(f"best {best['cell']} by {best['metric']}: {_fmt(best['value']['median'], 2)} "
                     f"({_pct(best['change_pct'])} over the baseline, {_pct(rng[0])} to {_pct(rng[1])})")
    return lines


def render_markdown(summary: Mapping[str, Any]) -> str:
    """The table of every cell against the baseline cell, the best cell and what stays open."""
    g = summary["grid"]
    server = summary.get("server") or {}
    spawn = summary.get("spawn") or {}
    idle = summary.get("idle_w") or {}
    best_by = g["best_by"]
    head = [f"# Request-side settings sweep on {summary['chip']} ({summary['node']}), {summary['date']}", "",
            f"Run `{summary['run_id']}` from {summary['started']} to {summary['ended']}"
            f"{' (the quick grid)' if summary['quick'] else ''}: one server spawn, the "
            f"{summary['launch']['served'] or 'no'} launch, ready after {spawn.get('ready_s')} s; "
            f"{len(summary['cells'])} cells, {g['repeats']} repeats each in {g['order']} order, one run of a cell "
            "being B concurrent requests of the cell's settings. Every value is the median of the repeats, with the "
            "lowest and highest in brackets. Grid `" + str(g["file"]) + f"` (sha256 {str(g['sha256'])[:12]}).", "",
            f"Server: engine commit {server.get('engine_commit')}, checkpoint {server.get('checkpoint')}, "
            f"quantization {server.get('quantization')}, max model length {server.get('max_model_len')}, linear "
            f"kernels {server.get('linear_kernels')}, attention {server.get('attention_backends')}, KV cache "
            f"{server.get('kv_cache_tokens')} tokens; ROCm {server.get('rocm')}, torch {server.get('torch')}, "
            f"Triton {server.get('triton')}. Idle {idle.get('no_server')} W with no server and {idle.get('loaded')} W "
            "with it loaded; watts are amd-smi SOCKET_POWER (the whole APU package) over each decode window.", "",
            f"Baseline cell: `{summary.get('baseline')}`, {g.get('baseline_source')}.", ""]
    table = ["| Cell | Batch | Thinking | Max tokens | Prompt (tokens in) | Decode tok/s per stream | "
             f"Decode tok/s total | TTFT s | Watts avg | J/token | {best_by} vs baseline | Repeats |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    best = summary.get("best")
    for c in summary["cells"]:
        s = c["stats"]
        name = (f"`{c['id']}`" + (" (baseline)" if c["id"] == summary.get("baseline") else "")
                + (" (best)" if best and c["id"] == best["cell"] else ""))
        tokens_in = "-" if c["prompt_tokens"] is None else f"{c['prompt_tokens']:.0f}"
        table.append(f"| {name} | {c['batch']} | {c['thinking']} | {c['max_tokens']} | {c['prompt']} ({tokens_in}) | "
                     f"{_cell_value(s['decode_tok_s'], 2)} | {_cell_value(s['decode_tok_s_total'], 2)} | "
                     f"{_cell_value(s['ttft_s'], 2)} | {_cell_value(s['watts_avg'], 1)} | "
                     f"{_cell_value(s['joules_per_token'], 2)} | {_pct(c['vs_baseline_pct'][best_by])} | "
                     f"{c['n']} of {g['repeats']} |")
    tail = [""]
    if best and best.get("baseline") and best["baseline"]["median"] is not None:
        rng = best.get("range_pct") or [None, None]
        tail.append(f"Best cell by {best_by}: `{best['cell']}` at {_fmt(best['value']['median'], 2)} "
                    f"({_fmt(best['value']['min'], 2)}-{_fmt(best['value']['max'], 2)}) against the baseline's "
                    f"{_fmt(best['baseline']['median'], 2)} ({_fmt(best['baseline']['min'], 2)}-"
                    f"{_fmt(best['baseline']['max'], 2)}): {_pct(best['change_pct'])} over the baseline, "
                    f"{_pct(rng[0])} to {_pct(rng[1])} from the two cells' lowest and highest repeats.")
    elif best:
        tail.append(f"Best cell by {best_by}: `{best['cell']}`; the baseline cell has no value to compare it with.")
    else:
        tail.append(f"No cell has a value of {best_by}.")
    low = summary.get("lowest_joules_per_token")
    if low:
        tail.append(f"Lowest joules per token: `{low['cell']}` at {_fmt(low['value']['median'], 2)} J/token "
                    f"({_pct(low['change_pct'])} against the baseline).")
    for item in g.get("held") or []:
        tail.append(f"Held: {item.get('setting')} = {item.get('value')}: {item.get('why')}.")
    if summary.get("open"):
        tail += ["", "Open:"] + [f"- {item}" for item in summary["open"]]
    return "\n".join(head + table + tail) + "\n"


# --- rows for the ledger and the metrics file ----------------------------------------------------------------------

def _row_conditions(summary: Mapping[str, Any], cell: Mapping[str, Any]) -> dict[str, Any]:
    g = summary["grid"]
    return {"run_id": summary["run_id"], "cell": cell["id"], "slice": cell["slice"], "batch_size": cell["batch"],
            "thinking": cell["thinking"], "max_tokens": cell["max_tokens"], "prompt_class": cell["prompt"],
            "prompt_file": (g["prompts"].get(cell["prompt"]) or {}).get("file"), "prompt_tokens": cell["prompt_tokens"],
            "sampling": {"profile": cell["sampling"], **(g["sampling"].get(cell["sampling"]) or {})},
            "requests_per_stream": cell["requests_per_stream"], "shared_spawn": True,
            "launch": summary["launch"]["served"], "grid": g["file"], "grid_sha256": g["sha256"],
            "baseline_cell": summary.get("baseline"), "statistic": f"median of {cell['n']} repeats on one shared spawn",
            "stats": cell["stats"], "vs_baseline_pct": cell["vs_baseline_pct"],
            "repeats": [{"run_id": r["run_id"], "outcome": r["outcome"]} for r in cell["repeats"]],
            "hub": summary.get("hub"), "account": summary.get("account"), "chip": summary.get("chip"),
            **{k: v for k, v in (summary.get("server") or {}).items() if k != "unavailable"}}


def ledger_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One `sweep_request.cell` row per cell on the run's node, its medians as metrics, low_confidence."""
    rows = []
    for cell in summary["cells"]:
        metrics = {f"{METRIC_PREFIX}.{m}": _r(cell["stats"][m]["median"], 6) for m in METRICS
                   if cell["stats"][m]["median"] is not None}
        rows.append({"verb": "sweep_request.cell", "outcome": cell["outcome"], "metrics": metrics,
                     "refs": {"run_id": f"{summary['run_id']}/{cell['id']}"}, "node": summary["node"],
                     "units": {k: METRICS[k.split(".", 1)[1]][0] for k in metrics} or None,
                     "conditions": _row_conditions(summary, cell), "low_confidence": True,
                     "source": f"{SWEEP} {summary['run_id']}",
                     "note": "; ".join(cell["reasons"])[:480] if cell["outcome"] != "ok" else None})
    return rows


def _metrics_ts(text: str | None) -> str:
    try:
        moment = dt.datetime.fromisoformat(str(text).replace("Z", "+00:00")).astimezone(dt.timezone.utc)
    except ValueError:
        moment = dt.datetime.now(dt.timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def metric_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The headline rows: the baseline and best cells' medians (each cell once), and the best cell's change over
    the baseline as a derived row."""
    cells = {c["id"]: c for c in summary["cells"]}
    best = summary.get("best") or {}
    rows: list[dict[str, Any]] = []
    done: set[str] = set()
    ts = _metrics_ts(summary.get("ended"))
    source = f"{SWEEP} {summary['run_id']}"
    for role, cell_id in (("baseline", summary.get("baseline")), ("best", best.get("cell"))):
        if cell_id is None or cell_id in done:
            continue
        done.add(cell_id)
        cell = cells[cell_id]
        roles = [r for r, c in (("baseline", summary.get("baseline")), ("best", best.get("cell"))) if c == cell_id]
        for m in HEADLINE:
            s = cell["stats"][m]
            if s["median"] is None:
                continue
            conditions = dict(_row_conditions(summary, cell), role=" and ".join(roles))
            conditions["low_confidence_reason"] = (f"{cell['n']} repeats on one shared spawn, under the "
                                                   f"{MIN_RUNS_PER_ARM} that verdict() needs to call a difference")
            rows.append({"metric": f"{METRIC_PREFIX}.{m}", "value": round(s["median"], 4), "unit": METRICS[m][0],
                         "node": summary["node"], "conditions": conditions, "source": source, "ts": ts,
                         "low_confidence": True})
    if best.get("change_pct") is not None and best.get("cell") != summary.get("baseline"):
        rows.append({"metric": f"{METRIC_PREFIX}.{best['metric']}_change_pct", "value": best["change_pct"],
                     "unit": "%", "node": summary["node"], "source": source, "ts": ts, "low_confidence": True,
                     "conditions": {"derived": True, "run_id": summary["run_id"], "from": best["cell"],
                                    "against": summary.get("baseline"), "metric": best["metric"],
                                    "range_pct": best["range_pct"], "grid_sha256": summary["grid"]["sha256"],
                                    "low_confidence_reason": "derived from two cells of a few repeats each"}})
    return rows


# --- the pod gate --------------------------------------------------------------------------------------------------

def user_sha256(user: str) -> str:
    return hashlib.sha256(user.strip().encode("utf-8")).hexdigest()


def first_gfx(text: str | None) -> str | None:
    m = re.search(r"\bgfx[0-9a-f]+\b", text or "")
    return m.group(0) if m else None


def read_rocminfo() -> str | None:
    for program in ("rocminfo", "/opt/rocm/bin/rocminfo"):
        try:
            done = subprocess.run([program], capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError):
            continue
        if done.stdout:
            return done.stdout
    return None


def running_servers(proc: Path = Path("/proc")) -> list[int]:
    """The pids whose command line runs a vLLM server (none where there is no /proc)."""
    try:
        entries = list(proc.iterdir())
    except OSError:
        return []
    out = []
    for entry in entries:
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            if b"vllm.entrypoints" in (entry / "cmdline").read_bytes():
                out.append(int(entry.name))
        except OSError:
            continue
    return sorted(out)


def gate(*, node: str, expect_user_sha256: str, environ: Mapping[str, str], rocminfo: str | None, machine: str,
         port: int, port_free: bool, servers: Sequence[int], want_node: str = "pod-b") -> list[str]:
    """Every reason the pod run must not start here; [] when it may."""
    problems = []
    if node != want_node:
        problems.append(f"the sweep runs on {want_node}, not {node!r}")
    if machine.lower() in ("aarch64", "arm64"):
        problems.append("this is an aarch64 machine (the Jetson), not a Strix Halo pod")
    elif machine.lower() not in ("x86_64", "amd64"):
        problems.append(f"unexpected machine {machine!r}")
    gfx = first_gfx(rocminfo)
    if gfx is None:
        problems.append("cannot tell what GPU this is: rocminfo is not installed or printed no gfx target")
    elif gfx != CHIP:
        problems.append(f"rocminfo names {gfx}, not {CHIP}")
    user = environ.get("JUPYTERHUB_USER", "")
    if not user.strip():
        problems.append("JUPYTERHUB_USER is not set: cannot tell which hub account this pod belongs to")
    elif user_sha256(user) != expect_user_sha256:
        problems.append("this pod belongs to another hub account than the one the script was made for")
    if not port_free:
        problems.append(f"something already listens on port {port}")
    if servers:
        problems.append(f"a vLLM server already runs here (pid {', '.join(str(p) for p in servers)})")
    return problems


# --- the pod script ------------------------------------------------------------------------------------------------

def module_closure(start: str = "lab.sweep_request", root: Path = REPO) -> list[Path]:
    """The lab modules start imports, at any depth and inside functions too, with their packages' __init__ files."""
    seen: dict[str, Path | None] = {}
    todo = [start]
    while todo:
        name = todo.pop()
        if name in seen or not (name == "lab" or name.startswith("lab.")):
            continue
        path = root / (name.replace(".", "/") + ".py")
        if not path.is_file():
            path = root / name.replace(".", "/") / "__init__.py"
        if not path.is_file():
            seen[name] = None
            continue
        seen[name] = path
        parts = name.split(".")
        todo += [".".join(parts[:i]) for i in range(1, len(parts))]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                todo += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                todo += [node.module, *(f"{node.module}.{a.name}" for a in node.names)]
    return sorted({p for p in seen.values() if p is not None})


def bundle_files(grid: Grid, root: Path = REPO) -> dict[str, bytes]:
    """What the pod needs: the modules, the grid, its prompt sets, the search space and the engine patch."""
    paths = list(module_closure(root=root))
    named = [grid.file or _rel(DEFAULT_GRID), grid.server["search_space"], grid.server["patch"],
             *(p.file for p in grid.prompts.values())]
    for rel in named:
        path = root / rel
        if not path.is_file():
            raise SweepError(f"{rel} is not in this checkout")
        paths.append(path)
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(set(paths))}


def pod_script(grid: Grid, *, quick: bool, expect_user_sha256: str, deadline_min: float, hub: str | None,
               account: str, node: str, root: Path = REPO) -> tuple[str, dict[str, Any]]:
    """The bash script the pod runs (pod_exec.py's term mode types it, run mode uploads it): the lab files as one
    base64 block (letters, digits, + / =, lines of 76) checked against its sha256 and unpacked under
    ~/sweep_request/bundle-<sha>, then pod-run there. No tab and no exclamation mark anywhere, so an interactive shell
    reads it as written. Its last line is `### END rc=<code>`."""
    files = bundle_files(grid, root)
    payload = make_bundle(files)
    digest = hashlib.sha256(payload).hexdigest()
    args = ["pod-run", "--expect-user-sha256", expect_user_sha256, "--deadline-min", f"{deadline_min:g}",
            "--node", node, "--account", account, *(["--hub", hub] if hub else []), *(["--quick"] if quick else [])]
    script = "\n".join([
        "# Request-side settings sweep of the teacher, written by python -m lab.sweep_request pod-script; run with bash.",
        "# It unpacks the lab files below, runs the sweep, and prints the result as one marked base64 block.",
        "set -u",
        f'B="$HOME/{POD_DIR}/bundle-{digest[:12]}"',
        'mkdir -p "$B"',
        "base64 -d > \"$B/bundle.tar.gz\" <<'SWEEPBUNDLE'",
        *b64_lines(payload),
        "SWEEPBUNDLE",
        f'echo "{digest}  $B/bundle.tar.gz" | sha256sum -c --quiet - '
        '|| { echo "the bundle was damaged in transit"; echo "### END rc=3"; exit 3; }',
        'tar -xzf "$B/bundle.tar.gz" -C "$B" || { echo "### END rc=3"; exit 3; }',
        'python3 -c "import httpx" 2>/dev/null || python3 -m pip install --user --quiet httpx',
        'cd "$B" || { echo "### END rc=3"; exit 3; }',
        "python3 -m lab.sweep_request " + " ".join(shlex.quote(a) for a in args),
        "rc=$?",
        'echo "### END rc=$rc"',
        "exit $rc",
        ""])
    if "\t" in script or "!" in script:
        raise SweepError("the pod script would carry a tab or an exclamation mark")
    return script, {"files": sorted(files), "bundle_bytes": len(payload), "bundle_sha256": digest,
                    "script_bytes": len(script.encode("ascii")), "pod_args": args}


# --- verbs ---------------------------------------------------------------------------------------------------------

def plan(grid: Grid, *, quick: bool, deadline_min: float | None = None) -> dict[str, Any]:
    """The cells, their expected seconds, the launches and the slot minutes; nothing runs."""
    cells = grid.cells(quick)
    est = estimate(grid, cells)
    launches = grid_launches(grid, run_dir=grid.server["run_dir"])
    return {"grid": grid.file, "grid_sha256": grid.sha256, "quick": quick, "repeats": grid.repeats,
            "order": grid.order, "baseline_cell": grid.baseline_cell, "best_by": grid.best_by,
            "n_cells": len(cells), "runs": len(cells) * grid.repeats,
            "cells": [{**dataclasses.asdict(c), "est_s": round(cell_seconds(grid, c), 1)} for c in cells],
            "estimate": est, "slot_minutes": est["slot_minutes"],
            "deadline_min": deadline_min if deadline_min is not None else est["slot_minutes"],
            "launches": [launch.describe() for launch in launches]}


def plan_text(p: Mapping[str, Any]) -> str:
    lines = [f"grid {p['grid']} ({'quick, ' if p['quick'] else ''}{p['n_cells']} cells x {p['repeats']} repeats, "
             f"{p['order']} order; baseline {p['baseline_cell']}, best by {p['best_by']})",
             f"{'cell':<20} {'batch':>5} {'thinking':>8} {'max tokens':>10} {'prompt':>7} {'sampling':>9} {'est s':>7}"]
    for c in p["cells"]:
        lines.append(f"{c['id']:<20} {c['batch']:>5} {c['thinking']:>8} {c['max_tokens']:>10} {c['prompt']:>7} "
                     f"{c['sampling']:>9} {c['est_s']:>7.1f}")
    e = p["estimate"]
    lines.append(f"slot minutes: {p['slot_minutes']} (idle {e['idle_s']} s, ready {e['ready_s']:g} s, warm-up "
                 f"{e['warmup_s']} s, {p['repeats']} x {e['per_repeat_s']} s measuring, stop {e['stop_s']:g} s: "
                 f"{e['total_s']} s, plus {e['margin']:.0%}); deadline {p['deadline_min']:g} min")
    for launch in p["launches"]:
        lines.append(f"launch {launch['name']}: {shlex.join(launch['argv'])}"
                     + (f"  env {json.dumps(launch['env'], sort_keys=True)}" if launch["env"] else ""))
    return "\n".join(lines)


def dry_run_main(args: argparse.Namespace) -> dict[str, Any]:
    p = plan(load_grid(args.grid), quick=args.quick, deadline_min=args.deadline_min)
    print(plan_text(p), file=sys.stderr, flush=True)
    return {"ok": True, "outcome": "dry_run", **p}


def pod_script_main(args: argparse.Namespace) -> dict[str, Any]:
    grid = load_grid(args.grid)
    p = plan(grid, quick=args.quick, deadline_min=args.deadline_min)
    letter = str(args.account or grid.server["account"]).upper()
    if args.expect_user_sha256:
        expect, hub_url = args.expect_user_sha256, args.hub
    else:
        from lab import hub as hub_module
        try:
            account = hub_module.load_accounts([letter])[letter]
        except hub_module.MissingCredentials as exc:
            raise SweepError(f"account {letter} has no user name in the environment or the .env files: {exc}") from exc
        expect, hub_url = user_sha256(account.user), args.hub or account.url
    script, info = pod_script(grid, quick=args.quick, expect_user_sha256=expect, deadline_min=p["deadline_min"],
                              hub=hub_url, account=letter, node=grid.server["node"])
    out = Path(args.out)
    timeout = int(p["deadline_min"] * 60 + 300)
    answer = {**info, "slot_minutes": p["slot_minutes"], "deadline_min": p["deadline_min"], "n_cells": p["n_cells"],
              "runs": p["runs"], "quick": args.quick,
              "pod_command": f"python tooling/dispatcher/pod_exec.py {letter} term {out} {timeout} > "
                             f"{out.with_suffix('.out')}",
              "collect": f"python -m lab sweep-request collect {out.with_suffix('.out')} --metrics-file"}
    if args.dry_run:
        print(plan_text(p), file=sys.stderr, flush=True)
        return {"ok": True, "outcome": "dry_run", "script": None, **answer}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(script.encode("ascii"))
    return {"ok": True, "outcome": "written", "script": str(out), **answer}


def pod_run_main(args: argparse.Namespace, *, host: bench.Host | None = None, power: Any = None) -> dict[str, Any]:
    grid = load_grid(args.grid)
    server = grid.server
    node = args.node or server["node"]
    port = server["port"] if args.port is None else args.port
    problems = gate(node=node, expect_user_sha256=args.expect_user_sha256, environ=os.environ,
                    rocminfo=read_rocminfo(), machine=platform.machine(), port=port,
                    port_free=not bench.port_open(port), servers=running_servers(), want_node=server["node"])
    if problems:
        return {"ok": False, "outcome": "refused", "problems": problems, "error": "; ".join(problems)}
    p = plan(grid, quick=args.quick, deadline_min=args.deadline_min)
    if args.dry_run:
        print(plan_text(p), file=sys.stderr, flush=True)
        return {"ok": True, "outcome": "dry_run", "gate": "passed", **p}

    def stop_here(signum: int, frame: Any) -> None:
        raise SystemExit(f"signal {signum}")

    for name in ("SIGHUP", "SIGTERM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), stop_here)
    home = Path.home() / POD_DIR
    host = read_once(host or bench.Host())
    power = power or bench.LocalPower.for_host(host, node=node, period_s=float(server["power_period_s"]))
    engine_dir = os.path.expanduser(args.engine_dir or server["engine_dir"])
    patch = REPO / server["patch"]
    launches = grid_launches(grid, run_dir=os.path.expanduser(server["run_dir"]))
    raw = run_sweep(grid, grid.cells(args.quick), launches, host=host, power=power,
                    out_dir=Path(args.out) if args.out else home / "runs",
                    checkpoint=os.path.expanduser(args.checkpoint or server["checkpoint"]),
                    engine_dir=engine_dir if Path(engine_dir).is_dir() else None,
                    patches=[str(patch)] if patch.is_file() else [], node=node, port=port, hub=args.hub,
                    account=args.account or server["account"], deadline_s=p["deadline_min"] * 60.0, quick=args.quick)
    text = report(raw)
    run_dir = Path(raw["run_dir"])
    (run_dir / "result_bundle.b64").write_text(text, encoding="utf-8", newline="\n")
    home.mkdir(parents=True, exist_ok=True)
    (home / "last_result.b64").write_text(text, encoding="utf-8", newline="\n")
    print(text, flush=True)
    rows = raw["rows"]
    measured = sum(1 for r in rows if r["outcome"] == "ok")
    ok = raw["error"] is None and bool(rows) and measured == len(rows)
    return {"ok": ok, "outcome": "measured" if ok else "incomplete", "run_id": raw["run_id"], "run_dir": str(run_dir),
            "launch": raw["launch"]["served"], "rows": len(rows), "rows_ok": measured,
            "refs": {"run_id": raw["run_id"]},
            **({} if ok else {"error": raw["error"] or f"{len(rows) - measured} of {len(rows)} runs not measured"})}


def result_paths(summary: Mapping[str, Any], out_dir: Path) -> tuple[Path, Path, bool]:
    """Where a run's JSON and Markdown go, and whether that JSON already holds this same run."""
    stem = f"sweep-request-{CHIP}-{summary['date']}" + ("-quick" if summary["quick"] else "")
    k = 1
    while True:
        name = stem if k == 1 else f"{stem}-{k}"
        json_path, md_path = out_dir / f"{name}.json", out_dir / f"{name}.md"
        if not json_path.exists():
            return json_path, md_path, False
        try:
            if json.loads(json_path.read_text(encoding="utf-8")).get("run_id") == summary["run_id"]:
                return json_path, md_path, True
        except (OSError, ValueError):
            pass
        k += 1


def collect_main(args: argparse.Namespace) -> dict[str, Any]:
    try:
        files = read_bundle(Path(args.input).read_bytes())
    except OSError as exc:
        raise SweepError(f"cannot read {args.input}: {exc}") from exc
    if "raw.json" not in files:
        raise SweepError("the result block holds no raw.json")
    summary = summarize(json.loads(files["raw.json"]))
    out_dir = Path(args.out_dir)
    json_path, md_path, again = result_paths(summary, out_dir)
    if again and not args.force:
        return {"ok": False, "outcome": "already_collected", "run_id": summary["run_id"],
                "error": f"{_rel(json_path)} already holds run {summary['run_id']}; --force writes the files and the "
                         "rows again"}
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n")
    md_path.write_text(render_markdown(summary), encoding="utf-8", newline="\n")
    keep = Path(args.state_dir) / summary["run_id"]
    keep.mkdir(parents=True, exist_ok=True)
    for name, blob in files.items():
        (keep / name).write_bytes(blob)
    ledger_n = 0
    if not args.no_ledger:
        for row in ledger_rows(summary):
            record(row.pop("verb"), row.pop("outcome"), row.pop("metrics"), row.pop("refs"),
                   task=summary["grid"]["task"], **row)
            ledger_n += 1
    metrics_n = 0
    if args.metrics_file:
        for row in metric_rows(summary):
            metrics_file.add(row, path=args.metrics_file)
            metrics_n += 1
    best = summary.get("best") or {}
    by = summary["grid"]["best_by"]
    base = next((c for c in summary["cells"] if c["id"] == summary.get("baseline")), None)
    found: dict[str, float] = {}
    if base is not None and base["stats"][by]["median"] is not None:
        found[f"{METRIC_PREFIX}.{by}.baseline"] = round(base["stats"][by]["median"], 4)
    if best.get("value") and best["value"]["median"] is not None:
        found[f"{METRIC_PREFIX}.{by}.best"] = round(best["value"]["median"], 4)
    if best.get("change_pct") is not None:
        found[f"{METRIC_PREFIX}.{by}_change_pct"] = best["change_pct"]
    return {"ok": True, "outcome": "collected", "run_id": summary["run_id"], "json": _rel(json_path),
            "markdown": _rel(md_path), "cells": len(summary["cells"]),
            "cells_ok": sum(1 for c in summary["cells"] if c["outcome"] == "ok"), "baseline": summary.get("baseline"),
            "best": best.get("cell"), "change_pct": best.get("change_pct"), "range_pct": best.get("range_pct"),
            "ledger_rows": ledger_n, "metrics_rows": metrics_n, "open": summary["open"], "metrics": found,
            "refs": {"run_id": summary["run_id"]}}


def build_parser() -> argparse.ArgumentParser:
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--grid", default=argparse.SUPPRESS)
    shared.add_argument("--quick", action="store_true", default=argparse.SUPPRESS)
    shared.add_argument("--dry-run", action="store_true", default=argparse.SUPPRESS)
    shared.add_argument("--deadline-min", type=float, default=argparse.SUPPRESS)
    p = argparse.ArgumentParser(prog="lab/sweep_request.py", description=__doc__.splitlines()[0])
    p.add_argument("--grid", default=str(DEFAULT_GRID), help="the grid file (default: the committed one)")
    p.add_argument("--quick", action="store_true", help="only the slices the grid lists under quick (four cells)")
    p.add_argument("--dry-run", action="store_true", help="print the grid and the slot minutes it needs; run nothing")
    p.add_argument("--deadline-min", type=float, default=None,
                   help="the run starts no cell it cannot finish in this many minutes (default: the slot minutes)")
    sub = p.add_subparsers(dest="cmd")
    script = sub.add_parser("pod-script", parents=[shared], help="write the script the pod runs")
    script.add_argument("--out", default=str(POD_SCRIPT))
    script.add_argument("--account", default=None, help="the hub account letter (default: the grid's)")
    script.add_argument("--expect-user-sha256", default=None,
                        help="the sha256 of the pod's Jupyter user name (default: from the account's .env keys)")
    script.add_argument("--hub", default=None, help="the hub address the rows name (default: the account's)")
    run = sub.add_parser("pod-run", parents=[shared], help="on the pod: gate, spawn, sweep, stop, print the result")
    run.add_argument("--expect-user-sha256", required=True)
    run.add_argument("--node", default=None)
    run.add_argument("--account", default=None)
    run.add_argument("--hub", default=None)
    run.add_argument("--port", type=int, default=None)
    run.add_argument("--checkpoint", default=None)
    run.add_argument("--engine-dir", default=None)
    run.add_argument("--out", default=None, help="where the run folders go (default: ~/sweep_request/runs)")
    collect = sub.add_parser("collect", help="read the pod's output and write the result files and rows")
    collect.add_argument("input", help="the saved output of the pod run (or ~/sweep_request/last_result.b64)")
    collect.add_argument("--out-dir", default=str(DATA_DIR))
    collect.add_argument("--state-dir", default=str(STATE_DIR), help="where the raw result is kept")
    collect.add_argument("--metrics-file", nargs="?", const=str(metrics_file.DEFAULT_PATH), default=None)
    collect.add_argument("--no-ledger", action="store_true")
    collect.add_argument("--force", action="store_true")
    return p


@recorded("sweep-request")
def main(argv: Sequence[str] | None = None, *, host: bench.Host | None = None, power: Any = None) -> dict[str, Any]:
    """Request-side settings sweep of the teacher on one spawn: the dry run, the pod script, the pod run, collect."""
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        if args.cmd is None:
            if not args.dry_run:
                return {"ok": False, "outcome": "usage",
                        "error": "say --dry-run, or one of pod-script, pod-run and collect"}
            return dry_run_main(args)
        if args.cmd == "pod-script":
            return pod_script_main(args)
        if args.cmd == "pod-run":
            return pod_run_main(args, host=host, power=power)
        return collect_main(args)
    except SweepError as exc:
        return {"ok": False, "outcome": "fail", "error": str(exc)}


if __name__ == "__main__":
    answer = main()
    print(json.dumps(answer, default=str))
    sys.exit(0 if answer.get("ok") else 1)
