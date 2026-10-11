"""Settings optimization with Ax and BoTorch: a search-space file, a pluggable trial runner, one JSON file per trial.

    python -m lab optimize install                       pip install -r lab/requirements-optimize.txt (pinned Ax, BoTorch)
    python -m lab optimize --dry-run                     the plan for the sample search space; runs and writes nothing
    python -m lab optimize run --space S.json --fake --budget 6        offline: the fake server or the fake trainer
    python -m lab optimize run --space S.json --node pod-d --minutes 50 --run-dir DIR \\
        --bench "--engine vllm --checkpoint ~/models/Qwen3.8-27B-INT4 --engine-dir ~/vllm"     inference, on a pod
    python -m lab optimize run --space S.json --node pod-d --minutes 50 --run-dir DIR \\
        --train-cmd "python3 lab/smoke_lora.py --steps {steps} --out {out}"                    training, on a pod
    python -m lab optimize summary --run-dir DIR         best configuration, posterior summary, Markdown table

`python lab/optimize_settings.py ...` is the same, by path on a pod. Every call answers with one JSON line on stdout
(exit 0 when ok); a run's progress and the dry run's plan go to stderr.

The search space (SPACE_SCHEMA). A JSON file that the settings sheets produce (R35 for inference, R36 for training);
lab/refine/research/kernels/data/searchspace-sample.json documents the format with illustrative values:

    {"schema": "lab.optimize.searchspace/1", "name": "...", "kind": "inference" | "training",
     "objective": {"metric": "joules_per_token.b1", "goal": "minimize" | "maximize"},
     "tracking": ["decode_tok_s.b1", ...],            reported beside the objective, modelled but not optimized
     "fixed": {"batches": [1]} | {"steps": 20},        what every trial holds fixed (below)
     "parameters": [
        {"name": "max_num_seqs", "type": "range", "bounds": [4, 64], "value_type": "int", "baseline": 16,
         "flag": "--max-num-seqs"},
        {"name": "GPU_MAX_HW_QUEUES", "type": "choice", "values": [1, 2, 4, 8], "baseline": 4, "env": "GPU_MAX_HW_QUEUES"},
        ...]}

A parameter is a range (bounds [low, high], value_type int or float, optional "log_scale": true) or a choice (two or
more values of one type: int, float, str or bool; optional "ordered"). Every parameter carries a pinned "baseline"
inside its range, and exactly one target: "env" (an environment variable of the launched process; true and false are
written 1 and 0) or "flag" (appended to the launch command as `--flag value`; true writes the bare flag, false writes
nothing or the parameter's "false_flag"). Metric names are dotted lowercase words. Inference metrics are lab/bench.py's
per-batch medians, named <metric>.b<batch> (decode_tok_s, decode_tok_s_total, ttft_s, joules_per_token, watts_avg,
watts_peak), and fixed.batches (required) lists the batch sizes each trial runs, every one the metrics name included;
fixed.max_tokens and fixed.share_spawn are optional. Training metrics are the numeric fields of the trainer's
summary.json (samples_per_s, peak_mem_gb, last_loss, ...); fixed.steps is the trainer's {steps}.

A trial. Trial 0 is the pinned baseline; every later trial is Ax's suggestion: Sobol points first (the initialization
budget, --init, else Ax's own choice; the baseline counts toward it), then BoTorch's Gaussian-process model with
expected improvement. A trial runs its settings --repeats times (default 3); each repeat is one call of the runner,
and for inference one call is one lab/bench.py session, so every repeat restarts the server with the trial's settings.
The trial reports the median of its repeats as a noisy observation: the standard error Ax receives is
MEDIAN_SEM_FACTOR x sd / sqrt(n) of the repeats (none with a single repeat, when Ax infers the noise itself). A repeat
that fails ends its trial as failed; Ax is told the arm is abandoned, so it is never suggested again.

The run directory holds run.json (the search space as parsed, its hash, the runner, repeats, seed, init and node:
what a resume must match) and trials/trial-NNNN.json, one file per trial (TRIAL_SCHEMA): its parameters, the env and
flags they became, the conditions (driver, node, runner, versions, git), every repeat (its metrics, the runner's
conditions, start, end, duration, outcome), the median and spread of every metric, the observation Ax got, the
timestamp, the git sha and the ledger seq. A trial file is rewritten after every repeat, so a stop loses at most the
repeat in flight. Each finished trial is an `optimize.trial` ledger row (record(): metrics optimize.<kind>.<metric>,
node, units, conditions, low_confidence while a trial has fewer repeats than verdict()'s 5, refs run_id and slot), and
each session ends with an `optimize.run` row. With --commit each finished trial file (and run.json, and the summary
at the end) is committed to the git repository the run directory is in, so every number is a committed file.

Resuming. Run the same command again: the trial files are the state. Finished trials are replayed into Ax (never
re-run), a trial that was in flight finishes its missing repeats, and the run goes on until it holds --budget trials
(the baseline and failed trials count). --minutes is this call's wall clock: no repeat starts that the median repeat
so far would push past it. lab/state/STOP, three failed trials in a row (--max-failures), Ctrl-C and Ax having nothing
left to suggest also stop a run cleanly; the in-flight trial keeps its finished repeats.

The runners. inference: lab/bench.py run_session with batches from the space, one cold spawn per batch size per
repeat, the trial's env merged into the launch environment and its flags appended to the launch command (--bench
takes lab/bench.py run's own arguments for the rest). training: the --train-cmd template ({steps} {out} {trial}
{repeat} {python}) run with the trial's env and flags, then its <out>/summary.json read (lab/smoke_lora.py writes
one). --fake swaps in lab/optimize_fake.py: its settings-aware server measured by lab/bench.py through
lab/bench_fake.py's fake pod, or its fake trainer, both spawned as real processes. A fake run records node "local"
and defaults to a folder under lab/state/optimize (gitignored), so fake numbers never sit beside real ones; a real
run defaults to lab/refine/research/kernels/data/ax-<kind>-<date>/, must name its --node, must have a --minutes wall
clock, and passes the node gate first: the node is a pod the ledger knows and rocminfo names gfx1151.

Ax is imported only when a suggestion or a posterior is needed, so the dry run, the baseline trial and a summary
without the model work where Ax is not installed; `install` installs the pinned versions with the pod's torch held
where it is (a constraint on the installed torch, so the ROCm build is never replaced).
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import math
import os
import platform
import re
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if not __package__:  # run by path on a pod: the repo root, one folder up, holds lab/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab import bench, pod_power  # noqa: E402
from lab.machine import ledger, stop  # noqa: E402
from lab.machine.record import record  # noqa: E402
from lab.machine.verdict import MIN_RUNS_PER_ARM  # noqa: E402

DRIVER = "lab/optimize_settings.py 1"
SPACE_SCHEMA = "lab.optimize.searchspace/1"
TRIAL_SCHEMA = "lab.optimize.trial/1"
RUN_SCHEMA = "lab.optimize.run/1"
SUMMARY_SCHEMA = "lab.optimize.summary/1"
REPO = Path(__file__).resolve().parents[1]
REQUIREMENTS = REPO / "lab" / "requirements-optimize.txt"
DATA_DIR = REPO / "lab" / "refine" / "research" / "kernels" / "data"
SAMPLE_SPACE = DATA_DIR / "searchspace-sample.json"
STATE_DIR = REPO / "lab" / "state" / "optimize"        # gitignored: fake runs and every run's scratch work
KINDS = ("inference", "training")
GOALS = ("minimize", "maximize")
DEFAULT_REPEATS = 3
DEFAULT_BUDGET = 12
DEFAULT_SEED = 0
DEFAULT_MAX_FAILURES = 3
DEFAULT_TRAIN_TIMEOUT_S = 3600.0
MEDIAN_SEM_FACTOR = math.sqrt(math.pi / 2)   # about 1.2533: a median's standard error over a mean's, normal draws
FAKE_NODE = "local"
TRAINER_CONDITIONS = ("steps", "examples_seen")    # numeric fields of a trainer summary that are not metrics
TRAIN_UNITS = {"samples_per_s": "samples/s", "peak_mem_gb": "GB", "elapsed_s": "s"}
STOP_OK = ("budget", "minutes", "exhausted")       # a session that ends for one of these ended as planned

_METRIC = re.compile(r"[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*")
_INFERENCE_METRIC = re.compile(r"(" + "|".join(bench.METRIC_UNITS) + r")\.b([1-9][0-9]*)")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_FLAG = re.compile(r"--[A-Za-z0-9][A-Za-z0-9_.-]*")
_TRAIN_PLACEHOLDER = re.compile(r"\{(steps|out|trial|repeat|python)\}")
_TRIAL_PLACEHOLDER = re.compile(r"\{(trial|repeat)\}")
_TRIAL_FILE = re.compile(r"trial-(\d{4,})\.json")


class SpaceError(ValueError):
    """A search space that cannot be used; .problems lists every reason."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


class RunError(Exception):
    """A run directory that cannot be resumed as asked, or a run that cannot start."""


class AxUnavailable(RunError):
    """Ax or BoTorch is not importable here."""


def log(message: str) -> None:
    print(f"[optimize] {message}", file=sys.stderr, flush=True)


# --- the search space -------------------------------------------------------------------------------------------

def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _same(a: Any, b: Any) -> bool:
    """Equal and of the same kind, so True is not 1 and 1 is not "1"."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if _is_number(a) and _is_number(b):
        return a == b
    return type(a) is type(b) and a == b


def _value_kind(values: Sequence[Any]) -> str | None:
    if all(isinstance(v, bool) for v in values):
        return "bool"
    if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        return "int"
    if all(_is_number(v) for v in values):
        return "float"
    if all(isinstance(v, str) for v in values):
        return "str"
    return None


@dataclass(frozen=True)
class Parameter:
    name: str
    type: str                       # range | choice
    value_type: str                 # int | float | str | bool
    baseline: Any
    bounds: tuple[float, float] | None = None
    values: tuple[Any, ...] | None = None
    log_scale: bool = False
    ordered: bool | None = None
    env: str | None = None
    flag: str | None = None
    false_flag: str | None = None
    note: str | None = None

    def contains(self, value: Any) -> bool:
        if self.type == "choice":
            return any(_same(value, v) for v in self.values or ())
        if not _is_number(value):
            return False
        if self.value_type == "int" and not float(value).is_integer():
            return False
        low, high = self.bounds or (0, 0)
        return low <= value <= high

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "type": self.type, "value_type": self.value_type,
                               "baseline": self.baseline}
        if self.type == "range":
            out.update(bounds=list(self.bounds or ()), log_scale=self.log_scale)
        else:
            out.update(values=list(self.values or ()), ordered=self.ordered)
        out.update({"env": self.env} if self.env else {"flag": self.flag})
        if self.false_flag:
            out["false_flag"] = self.false_flag
        return out


def _parse_parameter(p: Any, where: str) -> tuple[Parameter | None, list[str]]:
    problems: list[str] = []
    if not isinstance(p, Mapping):
        return None, [f"{where} must be an object"]
    name = p.get("name")
    if not isinstance(name, str) or not name.strip() or name != name.strip() or " " in name:
        problems.append(f"{where}: name must be a non-empty word, not {name!r}")
        name = str(name)
    where = f"{where} ({name})"
    kind = p.get("type")
    env, flag, false_flag = p.get("env"), p.get("flag"), p.get("false_flag")
    if (env is None) == (flag is None):
        problems.append(f"{where}: give exactly one target, env or flag")
    if env is not None and not (isinstance(env, str) and _ENV_NAME.fullmatch(env)):
        problems.append(f"{where}: env must be an environment variable name, not {env!r}")
    if flag is not None and not (isinstance(flag, str) and _FLAG.fullmatch(flag)):
        problems.append(f"{where}: flag must look like --name, not {flag!r}")
    if false_flag is not None and (flag is None or not (isinstance(false_flag, str) and _FLAG.fullmatch(false_flag))):
        problems.append(f"{where}: false_flag goes with a flag target and must look like --name, not {false_flag!r}")
    note = p.get("note")
    if note is not None and not isinstance(note, str):
        problems.append(f"{where}: note must be text")
    baseline = p.get("baseline")
    if "baseline" not in p:
        problems.append(f"{where}: every parameter pins a baseline")
    given_type = p.get("value_type")
    bounds: tuple[float, float] | None = None
    values: tuple[Any, ...] | None = None
    log_scale = p.get("log_scale", False)
    ordered = p.get("ordered")
    value_type = given_type
    if kind == "range":
        raw = p.get("bounds")
        if not (isinstance(raw, list) and len(raw) == 2 and all(_is_number(v) for v in raw) and raw[0] < raw[1]):
            problems.append(f"{where}: bounds must be [low, high] numbers with low < high, not {raw!r}")
        else:
            inferred = "int" if all(isinstance(v, int) for v in raw) else "float"
            value_type = given_type or inferred
            if value_type not in ("int", "float"):
                problems.append(f"{where}: a range's value_type is int or float, not {value_type!r}")
            elif value_type == "int" and inferred != "int":
                problems.append(f"{where}: an int range has whole-number bounds, not {raw!r}")
            bounds = (int(raw[0]), int(raw[1])) if value_type == "int" else (float(raw[0]), float(raw[1]))
            if not isinstance(log_scale, bool):
                problems.append(f"{where}: log_scale is true or false")
            elif log_scale and bounds[0] <= 0:
                problems.append(f"{where}: a log-scale range needs a positive low bound")
        if "values" in p:
            problems.append(f"{where}: a range has bounds, not values")
    elif kind == "choice":
        raw = p.get("values")
        if not (isinstance(raw, list) and len(raw) >= 2):
            problems.append(f"{where}: a choice has a list of two or more values, not {raw!r}")
        else:
            inferred = _value_kind(raw)
            value_type = given_type or inferred
            if inferred is None or value_type not in ("int", "float", "str", "bool"):
                problems.append(f"{where}: a choice's values are all int, float, str or bool, not {raw!r}")
            elif given_type and given_type != inferred and not (given_type == "float" and inferred == "int"):
                problems.append(f"{where}: value_type {given_type!r} does not match the values {raw!r}")
            elif any(_same(a, b) for i, a in enumerate(raw) for b in raw[i + 1:]):
                problems.append(f"{where}: the values repeat: {raw!r}")
            values = tuple(float(v) for v in raw) if value_type == "float" else tuple(raw)
        if ordered is not None and not isinstance(ordered, bool):
            problems.append(f"{where}: ordered is true or false")
        if "bounds" in p:
            problems.append(f"{where}: a choice has values, not bounds")
    else:
        problems.append(f"{where}: type is range or choice, not {kind!r}")
    if problems:
        return None, problems
    param = Parameter(name=name, type=kind, value_type=str(value_type), baseline=baseline, bounds=bounds,
                      values=values, log_scale=bool(log_scale), ordered=ordered, env=env, flag=flag,
                      false_flag=false_flag, note=note)
    if not param.contains(baseline):
        return None, [f"{where}: the baseline {baseline!r} is not in the parameter's "
                      f"{'bounds' if kind == 'range' else 'values'}"]
    if value_type == "float" and _is_number(baseline):
        param = dataclasses.replace(param, baseline=float(baseline))
    if value_type == "int" and _is_number(baseline):
        param = dataclasses.replace(param, baseline=int(baseline))
    return param, []


def ax_metric(name: str) -> str:
    """A metric's name as Ax sees it: Ax's objective strings are expressions, so the dots go."""
    return name.replace(".", "__")


@dataclass(frozen=True)
class SearchSpace:
    name: str
    kind: str
    objective: str
    goal: str
    tracking: tuple[str, ...]
    parameters: tuple[Parameter, ...]
    fixed: Mapping[str, Any]
    raw: Mapping[str, Any]
    sha256: str
    path: str | None = None
    note: str | None = None

    @property
    def minimize(self) -> bool:
        return self.goal == "minimize"

    @property
    def metrics(self) -> list[str]:
        return [self.objective, *self.tracking]

    @property
    def baseline(self) -> dict[str, Any]:
        return {p.name: p.baseline for p in self.parameters}

    def better(self, a: float, b: float) -> bool:
        """Is a better than b for the objective?"""
        return a < b if self.minimize else a > b

    def problems_with(self, params: Mapping[str, Any]) -> list[str]:
        known = {p.name for p in self.parameters}
        out = [f"{name} is not a parameter of {self.name}" for name in params if name not in known]
        for p in self.parameters:
            if p.name not in params:
                out.append(f"{p.name} has no value")
            elif not p.contains(params[p.name]):
                out.append(f"{p.name}={params[p.name]!r} is outside the space")
        return out

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "path": self.path, "sha256": self.sha256,
                "objective": {"metric": self.objective, "goal": self.goal}, "tracking": list(self.tracking),
                "fixed": dict(self.fixed), "parameters": [p.describe() for p in self.parameters],
                "baseline": self.baseline}


def space_hash(data: Any) -> str:
    """The hash of a search space's content, blind to whitespace and key order (so line endings never matter)."""
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def parse_space(data: Any, *, path: str | None = None) -> SearchSpace:
    """A search space from its JSON content; SpaceError lists every problem at once."""
    if not isinstance(data, Mapping):
        raise SpaceError(["a search space is a JSON object"])
    problems: list[str] = []
    if data.get("schema") != SPACE_SCHEMA:
        problems.append(f"schema must be {SPACE_SCHEMA!r}, not {data.get('schema')!r}")
    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        problems.append("name must be non-empty text")
    kind = data.get("kind")
    if kind not in KINDS:
        problems.append(f"kind is {' or '.join(KINDS)}, not {kind!r}")
    objective = data.get("objective")
    metric, goal = None, None
    if not isinstance(objective, Mapping):
        problems.append("objective must be {\"metric\": ..., \"goal\": \"minimize\" | \"maximize\"}")
    else:
        metric, goal = objective.get("metric"), objective.get("goal")
        if not (isinstance(metric, str) and _METRIC.fullmatch(metric)):
            problems.append(f"objective.metric must be a dotted lowercase metric name, not {metric!r}")
        if goal not in GOALS:
            problems.append(f"objective.goal is minimize or maximize, not {goal!r}")
    tracking = data.get("tracking", [])
    if not isinstance(tracking, list) or not all(isinstance(t, str) and _METRIC.fullmatch(t) for t in tracking):
        problems.append(f"tracking must be a list of dotted lowercase metric names, not {tracking!r}")
        tracking = []
    elif len(set(tracking)) != len(tracking) or metric in tracking:
        problems.append("tracking repeats a metric, or repeats the objective")
    fixed = data.get("fixed", {})
    if not isinstance(fixed, Mapping):
        problems.append("fixed must be an object")
        fixed = {}
    raw_params = data.get("parameters")
    params: list[Parameter] = []
    if not isinstance(raw_params, list) or not raw_params:
        problems.append("parameters must be a non-empty list")
    else:
        for i, p in enumerate(raw_params):
            param, found = _parse_parameter(p, f"parameters[{i}]")
            problems.extend(found)
            if param is not None:
                params.append(param)
        for attr, what in (("name", "name"), ("env", "env target"), ("flag", "flag target")):
            seen = [getattr(p, attr) for p in params if getattr(p, attr)]
            for dup in sorted({x for x in seen if seen.count(x) > 1}):
                problems.append(f"two parameters share the {what} {dup!r}")
    metrics = [m for m in [metric, *tracking] if isinstance(m, str)]
    if len({ax_metric(m) for m in metrics}) != len(metrics):
        problems.append("two metric names differ only by their dots")
    if kind == "inference":
        batches = fixed.get("batches")
        if not (isinstance(batches, list) and batches and all(isinstance(b, int) and not isinstance(b, bool)
                                                              and b > 0 for b in batches)):
            problems.append(f"an inference space names fixed.batches, a list of batch sizes, not {batches!r}")
            batches = []
        for m in metrics:
            match = _INFERENCE_METRIC.fullmatch(m)
            if not match:
                problems.append(f"{m!r} is not a lab/bench.py metric: <{'|'.join(bench.METRIC_UNITS)}>.b<batch>")
            elif batches and int(match.group(2)) not in batches:
                problems.append(f"{m!r} names batch {match.group(2)}, which fixed.batches does not run")
        if "max_tokens" in fixed and not (isinstance(fixed["max_tokens"], int) and fixed["max_tokens"] > 0):
            problems.append("fixed.max_tokens is a positive whole number")
        if "share_spawn" in fixed and not isinstance(fixed["share_spawn"], bool):
            problems.append("fixed.share_spawn is true or false")
    elif kind == "training" and "steps" in fixed:
        if not (isinstance(fixed["steps"], int) and not isinstance(fixed["steps"], bool) and fixed["steps"] > 0):
            problems.append("fixed.steps is a positive whole number")
    note = data.get("note")
    if problems:
        raise SpaceError(problems)
    return SearchSpace(name=str(name), kind=str(kind), objective=str(metric), goal=str(goal),
                       tracking=tuple(tracking), parameters=tuple(params), fixed=dict(fixed), raw=data,
                       sha256=space_hash(data), path=path, note=note if isinstance(note, str) else None)


def load_space(path: str | os.PathLike[str]) -> SearchSpace:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SpaceError([f"cannot read {path}: {exc}"]) from exc
    except ValueError as exc:
        raise SpaceError([f"{path.name} is not JSON: {exc}"]) from exc
    return parse_space(data, path=_shown(path))


def _shown(path: Path) -> str:
    """A path as the files record it: relative to the repo when it is inside it, with forward slashes."""
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return path.resolve().as_posix()


# --- settings into a process ------------------------------------------------------------------------------------

def _text(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        return repr(value)
    return str(value)


@dataclass(frozen=True)
class Applied:
    """A trial's parameters as a process receives them."""

    env: dict[str, str]
    flags: list[str]

    def describe(self) -> dict[str, Any]:
        return {"env": dict(self.env), "flags": list(self.flags)}


def apply(space: SearchSpace, params: Mapping[str, Any]) -> Applied:
    env: dict[str, str] = {}
    flags: list[str] = []
    for p in space.parameters:
        value = params[p.name]
        if p.env:
            env[p.env] = _text(value)
        elif value is True:
            flags.append(str(p.flag))
        elif value is False:
            if p.false_flag:
                flags.append(p.false_flag)
        else:
            flags += [str(p.flag), _text(value)]
    return Applied(env, flags)


# --- small facts --------------------------------------------------------------------------------------------------

def now() -> str:
    return ledger.now_text()


def git_facts(where: Path = REPO) -> dict[str, Any]:
    """HEAD and whether tracked files differ from it (untracked files, such as new trial files, do not count)."""
    try:
        sha = subprocess.run(["git", "-C", str(where), "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=20, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(where), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=20, check=True).stdout.strip() != ""
        return {"sha": sha or None, "dirty": dirty}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"sha": None, "dirty": None, "unavailable": f"git: {type(exc).__name__}: {exc}"[:300]}


def _dist_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def versions() -> dict[str, Any]:
    return {"python": platform.python_version(), "ax-platform": _dist_version("ax-platform"),
            "botorch": _dist_version("botorch"), "gpytorch": _dist_version("gpytorch"), "torch": _dist_version("torch")}


def pins(path: Path = REQUIREMENTS) -> dict[str, str]:
    """{distribution: version} of the name==version lines of lab/requirements-optimize.txt."""
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            name, sep, version = line.partition("==")
            if sep:
                out[name.strip().lower()] = version.strip()
    return out


def ax_installed() -> bool:
    return importlib.util.find_spec("ax") is not None and importlib.util.find_spec("botorch") is not None


def host_brief() -> dict[str, Any]:
    return {"hostname": socket.gethostname(), "platform": platform.platform(), "machine": platform.machine(),
            "jupyterhub_user": os.environ.get("JUPYTERHUB_USER")}


def write_json(path: Path, data: Any) -> None:
    """Write the file whole or not at all: a temporary file beside it, then a replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    for _ in range(200):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:   # Windows: a reader has it open for a moment
            time.sleep(0.01)
    raise OSError(f"could not replace {path}")


# --- the node gate ------------------------------------------------------------------------------------------------

def _rocminfo() -> str:
    program = pod_power.find_tool("rocminfo")
    if program is None:
        raise pod_power.ReadError("rocminfo is not installed")
    return pod_power.run_command([program], timeout_s=30.0)


def node_gate(node: str | None, *, fake: bool, rocminfo: Callable[[], str] = _rocminfo) -> str | None:
    """Why this run may not start here, or None. A fake run is node local and nothing else; a real run names a pod
    the ledger knows (lab/bench.py's SAMPLER_NODE) and that pod's rocminfo names gfx1151 (lab/pod_power.py gate)."""
    if fake:
        if node not in (None, FAKE_NODE):
            return f"a fake run records node {FAKE_NODE!r}, not {node!r}: fake numbers never carry a pod's name"
        return None
    if not node:
        return f"a real run names the pod it runs on: --node {' | '.join(bench.SAMPLER_NODE)}"
    if node not in bench.SAMPLER_NODE:
        return f"--node is one of {', '.join(bench.SAMPLER_NODE)} for a real run, not {node!r}"
    return pod_power.gate(bench.SAMPLER_NODE[node], rocminfo)


# --- the runners ----------------------------------------------------------------------------------------------------

@dataclass
class RepeatResult:
    """One repeat of one trial, as a runner reports it."""

    outcome: str                      # ok | fail
    metrics: dict[str, float]
    conditions: dict[str, Any]
    reason: str | None = None


def trainer_metrics(summary: Mapping[str, Any]) -> dict[str, float]:
    """The numeric top-level fields of a trainer's summary.json, except the ones that are conditions."""
    return {k: float(v) for k, v in summary.items()
            if _is_number(v) and k not in TRAINER_CONDITIONS and _METRIC.fullmatch(k)}


def _fill_trial(argv: Sequence[str], trial: int, repeat: int) -> list[str]:
    values = {"trial": trial, "repeat": repeat}
    return [_TRIAL_PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), str(a)) for a in argv]


class InferenceRunner:
    """Each repeat is one lab/bench.py session: one cold spawn per batch size with the trial's env and flags."""

    kind = "inference"

    def __init__(self, config: bench.Config, *, batches: Sequence[int], fake: bool, host: bench.Host | None,
                 power: Callable[[bench.Host | None, bench.Config], Any], share_spawn: bool = False,
                 max_tokens: int | None = None) -> None:
        self.config = config
        self.batches = list(batches)
        self.fake = fake
        self.host = host
        self.power = power
        self.share_spawn = share_spawn
        self.max_tokens = max_tokens
        self._template: list[str] | None = list(config.command) if config.command else None

    def describe(self) -> dict[str, Any]:
        c = self.config
        return {"kind": self.kind, "fake": self.fake, "harness": bench.HARNESS, "engine": c.engine,
                "checkpoint": c.checkpoint or None, "model": c.model, "max_len": c.max_len, "batches": self.batches,
                "share_spawn": self.share_spawn, "max_tokens": self.max_tokens or c.max_tokens,
                "requests_per_stream": c.requests_per_stream, "warmup_rounds": c.warmup_rounds, "probe": c.probe,
                "command": self._template or "lab/bench.py default_command for the engine and the chip",
                "launch_env": dict(c.launch_env), "engine_dir": c.engine_dir, "patches": list(c.patches),
                "prompts": _shown(Path(c.prompts)), "port": c.port}

    def template(self) -> list[str]:
        if self._template is None:
            chip = bench.host_facts(self.host).get("chip") if self.host is not None else None
            self._template = bench.default_command(self.config.engine, chip)
        return self._template

    def run_once(self, applied: Applied, *, trial: int, repeat: int, workdir: Path) -> RepeatResult:
        config = dataclasses.replace(
            self.config, command=_fill_trial([*self.template(), *applied.flags], trial, repeat),
            launch_env={**self.config.launch_env, **applied.env}, batches=list(self.batches), add_64=False,
            repeats=1, share_spawn=self.share_spawn, max_tokens=self.max_tokens or self.config.max_tokens,
            out_dir=workdir, record=True)
        try:
            result = bench.run_session(config, host=self.host, power=self.power(self.host, config))
        except bench.BenchError as exc:
            return RepeatResult("fail", {}, {"command": config.command}, f"lab/bench.py: {exc}")
        rows = result["rows"]
        metrics: dict[str, float] = {}
        for entry in result["summary"]["configurations"].values():
            for name in bench.METRIC_UNITS:
                value = entry["stats"][name]["median"]
                if value is not None:
                    metrics[f"{name}.b{entry['batch_size']}"] = float(value)
        failed = [r for r in rows if r["outcome"] not in ("ok", "skipped")]
        conditions = {
            "bench": {"session": result["session"], "rows_path": result["rows_path"],
                      "runs": [{"run_id": r["run_id"], "batch_size": r["batch_size"], "outcome": r["outcome"],
                                "pid": (r.get("spawn") or {}).get("pid"),
                                "ready_s": (r.get("spawn") or {}).get("ready_s"),
                                "stopped": (r.get("spawn") or {}).get("stopped")} for r in rows]},
            "command": (rows[0].get("spawn") or {}).get("command") if rows else config.command,
            "conditions": rows[0]["conditions"] if rows else None}
        if not rows:
            return RepeatResult("fail", metrics, conditions, "lab/bench.py wrote no rows")
        if failed:
            return RepeatResult("fail", metrics, conditions, f"batch {failed[0]['batch_size']}: {failed[0]['reason']}")
        return RepeatResult("ok", metrics, conditions)


class TrainingRunner:
    """Each repeat runs the trainer command once with the trial's env and flags, then reads its summary.json."""

    kind = "training"

    def __init__(self, command: Sequence[str], *, steps: int | None, fake: bool, env: Mapping[str, str] | None = None,
                 timeout_s: float = DEFAULT_TRAIN_TIMEOUT_S) -> None:
        if not command:
            raise RunError("the training runner needs a command (--train-cmd)")
        self.command = list(command)
        self.steps = steps
        self.fake = fake
        self.env = dict(env or {})
        self.timeout_s = timeout_s

    def describe(self) -> dict[str, Any]:
        return {"kind": self.kind, "fake": self.fake, "command": self.command, "steps": self.steps,
                "env": dict(self.env), "timeout_s": self.timeout_s}

    def run_once(self, applied: Applied, *, trial: int, repeat: int, workdir: Path) -> RepeatResult:
        out = workdir / "out"
        out.mkdir(parents=True, exist_ok=True)
        values = {"steps": self.steps if self.steps is not None else "", "out": out, "trial": trial,
                  "repeat": repeat, "python": sys.executable}
        argv = [_TRAIN_PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), a) for a in self.command] + applied.flags
        log_path = workdir / "train.log"
        started = time.monotonic()
        code: int | None = None
        with open(log_path, "wb") as fh:
            try:
                proc = subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        env={**os.environ, **self.env, **applied.env}, cwd=str(workdir),
                                        start_new_session=os.name == "posix")
            except OSError as exc:
                return RepeatResult("fail", {}, {"argv": argv}, f"could not start {argv[0]}: {exc}")
            try:
                code = proc.wait(self.timeout_s)
            except subprocess.TimeoutExpired:
                _kill(proc)
        conditions: dict[str, Any] = {"argv": argv, "env": {**self.env, **applied.env}, "exit_code": code,
                                      "duration_s": round(time.monotonic() - started, 3), "log": str(log_path),
                                      "host": host_brief()}
        if code is None:
            return RepeatResult("fail", {}, conditions, f"the trainer ran past {self.timeout_s:g} s and was stopped")
        if code != 0:
            return RepeatResult("fail", {}, conditions, f"the trainer exited with code {code}: {_tail(log_path)}")
        try:
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return RepeatResult("fail", {}, conditions, f"no readable {out / 'summary.json'}: {exc}")
        conditions["summary"] = summary
        return RepeatResult("ok", trainer_metrics(summary), conditions)


def _kill(proc: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
        proc.wait(30)
    except (OSError, subprocess.SubprocessError):
        pass


def _tail(path: Path, lines: int = 3) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    except OSError:
        return "(no log)"
    return " | ".join(text[-lines:])[:500] or "(empty log)"


def fake_inference_runner(space: SearchSpace, work_dir: Path, *, noise_seed: int = 0) -> InferenceRunner:
    """lab/optimize_fake.py's settings-aware server, measured by lab/bench.py on lab/bench_fake.py's fake pod."""
    from lab import bench_fake, optimize_fake

    root = work_dir / "fake-pod"
    root.mkdir(parents=True, exist_ok=True)
    config = bench.Config(engine="vllm", command=optimize_fake.serve_command(root, noise_seed=noise_seed),
                          checkpoint="/home/jovyan/models/fake-checkpoint", model="teacher", node=FAKE_NODE, port=0,
                          max_tokens=8, requests_per_stream=2, warmup_rounds=1, idle_s=0.05, power_period_s=0.01,
                          ready_timeout_s=60.0, request_timeout_s=30.0, python="python3")
    host = bench_fake.FakeHost(root)
    return InferenceRunner(config, batches=space.fixed["batches"], fake=True, host=host,
                           power=lambda h, c: bench_fake.FakePower(root, period_s=c.power_period_s),
                           share_spawn=bool(space.fixed.get("share_spawn", False)),
                           max_tokens=space.fixed.get("max_tokens"))


def real_inference_runner(space: SearchSpace, bench_args: str, *, node: str) -> InferenceRunner:
    """lab/bench.py on this pod, configured by its own `run` arguments (--bench)."""
    try:
        args = bench.build_parser().parse_args(["run", *shlex.split(bench_args or "")])
    except SystemExit as exc:
        raise RunError(f"the --bench arguments do not parse as lab/bench.py run's (exit {exc.code})") from None
    config = bench._config(args)   # lab/bench.py's own mapping of its arguments, so the two never drift
    config.node = node
    host = bench.Host()
    return InferenceRunner(config, batches=space.fixed["batches"], fake=False, host=host,
                           power=lambda h, c: bench.LocalPower.for_host(h, node=c.node, period_s=c.power_period_s),
                           share_spawn=bool(space.fixed.get("share_spawn", False)),
                           max_tokens=space.fixed.get("max_tokens"))


def fake_training_runner(space: SearchSpace, *, noise_seed: int = 0, timeout_s: float = 120.0) -> TrainingRunner:
    from lab import optimize_fake

    return TrainingRunner(optimize_fake.train_command(noise_seed=noise_seed), steps=space.fixed.get("steps", 20),
                          fake=True, timeout_s=timeout_s)


# --- the run directory ----------------------------------------------------------------------------------------------

def trial_path(run_dir: Path, n: int) -> Path:
    return run_dir / "trials" / f"trial-{n:04d}.json"


@dataclass
class RunState:
    """What a run directory holds: run.json and the trial files, checked to be whole."""

    run_dir: Path
    manifest: dict[str, Any] | None
    trials: list[dict[str, Any]]

    @classmethod
    def load(cls, run_dir: Path) -> "RunState":
        manifest_path = run_dir / "run.json"
        manifest = None
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except ValueError as exc:
                raise RunError(f"{manifest_path} is not JSON: {exc}") from exc
            if manifest.get("schema") != RUN_SCHEMA:
                raise RunError(f"{manifest_path} is not a {RUN_SCHEMA} file")
        trials: list[dict[str, Any]] = []
        folder = run_dir / "trials"
        names = sorted(p.name for p in folder.iterdir() if _TRIAL_FILE.fullmatch(p.name)) if folder.is_dir() else []
        for i, name in enumerate(names):
            number = int(_TRIAL_FILE.fullmatch(name).group(1))
            if number != i:
                raise RunError(f"{folder / f'trial-{i:04d}.json'} is missing: trial files run 0, 1, 2, ... "
                               "with no gap, so this directory is not whole")
            try:
                trial = json.loads((folder / name).read_text(encoding="utf-8"))
            except ValueError as exc:
                raise RunError(f"{folder / name} is not JSON: {exc}") from exc
            if trial.get("schema") != TRIAL_SCHEMA or trial.get("trial") != i:
                raise RunError(f"{folder / name} is not trial {i} in the {TRIAL_SCHEMA} format")
            if trial.get("status") not in ("running", "completed", "failed"):
                raise RunError(f"{folder / name} has status {trial.get('status')!r}")
            trials.append(trial)
        if trials and manifest is None:
            raise RunError(f"{run_dir} has trial files but no run.json")
        running = [t["trial"] for t in trials if t["status"] == "running"]
        if running and running != [len(trials) - 1]:
            raise RunError(f"trials {running} are in flight; only the last trial can be")
        return cls(run_dir, manifest, trials)

    @property
    def running(self) -> dict[str, Any] | None:
        return self.trials[-1] if self.trials and self.trials[-1]["status"] == "running" else None

    def counts(self) -> dict[str, int]:
        out = {"total": len(self.trials), "completed": 0, "failed": 0, "running": 0}
        for t in self.trials:
            out[t["status"]] += 1
        return out

    def repeat_seconds(self) -> float | None:
        """The median duration of the finished repeats so far: what the next repeat is expected to take."""
        times = [r["duration_s"] for t in self.trials for r in t["repeats"] if r.get("duration_s") is not None]
        return statistics.median(times) if times else None


# --- Ax, imported only when it is needed ----------------------------------------------------------------------------

def _import_ax() -> Any:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import ax
            from ax.utils.common.logger import set_stderr_log_level
    except ImportError as exc:
        raise AxUnavailable(f"Ax is not installed here ({exc}); run: python -m lab optimize install") from exc
    set_stderr_log_level(logging.WARNING)
    return ax


class Optimizer:
    """Ax's Client over a search space, with every trial of the run directory replayed into it in order."""

    def __init__(self, space: SearchSpace, *, name: str, seed: int, init: int | None) -> None:
        ax = _import_ax()
        self.space = space
        self.client = ax.Client(random_seed=seed)
        configs = []
        for p in space.parameters:
            if p.type == "range":
                configs.append(ax.RangeParameterConfig(name=p.name, bounds=tuple(p.bounds or ()),
                                                       parameter_type=p.value_type,
                                                       scaling="log" if p.log_scale else None))
            else:
                configs.append(ax.ChoiceParameterConfig(name=p.name, values=list(p.values or ()),
                                                        parameter_type=p.value_type, is_ordered=p.ordered))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.client.configure_experiment(parameters=configs, name=name)
            self.client.configure_optimization(objective=("-" if space.minimize else "") + ax_metric(space.objective))
            if space.tracking:
                self.client.configure_tracking_metrics([ax_metric(m) for m in space.tracking])
            self.client.configure_generation_strategy(method="fast", initialization_budget=init,
                                                      initialization_random_seed=seed)

    def observation(self, trial: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for metric in self.space.metrics:
            spread = (trial.get("spread") or {}).get(metric)
            if spread and spread.get("median") is not None:
                sem = spread.get("sem")
                out[ax_metric(metric)] = (spread["median"], sem) if sem is not None else spread["median"]
        return out

    def attach(self, trial: Mapping[str, Any]) -> None:
        """Replay one trial file: the baseline as Ax's status quo, then completed, abandoned or still running."""
        params = dict(trial["parameters"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if trial["generator"] == "baseline":
                index = self.client.attach_baseline(parameters=params)
            else:
                index = self.client.attach_trial(parameters=params)
        if index != trial["trial"]:
            raise RunError(f"Ax numbered trial {trial['trial']} as {index}; the run directory and Ax disagree")
        self.finish(trial)

    def finish(self, trial: Mapping[str, Any]) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if trial["status"] == "completed":
                self.client.complete_trial(trial["trial"], raw_data=self.observation(trial))
            elif trial["status"] == "failed":
                self.client.mark_trial_abandoned(trial["trial"])   # abandoned arms are never suggested again

    def suggest(self) -> tuple[int, dict[str, Any], str, str] | None:
        """(trial index, parameters, arm name, generation node) of Ax's next trial, or None when it has none."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                trials = self.client.get_next_trials(max_trials=1)
            except Exception as exc:
                if type(exc).__name__ == "GenerationStrategyCompleted":
                    return None
                raise
            if not trials:
                return None
            index, params = next(iter(trials.items()))
            frame = self.client.summarize(trial_indices=[index])
        arm = str(frame["arm_name"].iloc[0]) if "arm_name" in frame else f"{index}_0"
        node = str(frame["generation_node"].iloc[0]) if "generation_node" in frame else "unknown"
        return index, {k: _plain(v) for k, v in params.items()}, arm, node

    def posterior(self, trials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """The fitted model's mean and standard error at every completed trial, and its best arm."""
        done = [t for t in trials if t["status"] == "completed"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            strategy = self.client._generation_strategy_or_choose()   # Ax 1.3.1: the Client keeps it private
            strategy.fit(experiment=self.client._experiment)
            node = strategy.current_node_name
            try:
                predicted = self.client.predict([dict(t["parameters"]) for t in done])
            except Exception as exc:
                return {"available": False, "model": node,
                        "reason": f"the current generation node ({node}) does not predict yet: {exc}"[:400]}
            best_params, best_pred, best_index, best_arm = self.client.get_best_parameterization(
                use_model_predictions=True)
        names = {ax_metric(m): m for m in self.space.metrics}
        arms = []
        for t, pred in zip(done, predicted):
            arms.append({"trial": t["trial"], "parameters": t["parameters"],
                         "predicted": {names.get(k, k): {"mean": float(v[0]), "sem": float(v[1])}
                                       for k, v in pred.items() if k in names}})
        return {"available": True, "model": node, "arms": arms,
                "best": {"trial": int(best_index), "arm": str(best_arm),
                         "parameters": {k: _plain(v) for k, v in best_params.items()},
                         "predicted": {names.get(k, k): ({"mean": float(v[0]), "sem": float(v[1])}
                                                         if isinstance(v, tuple) else {"mean": float(v), "sem": None})
                                       for k, v in best_pred.items() if k in names}}}


def _plain(value: Any) -> Any:
    """numpy scalars from Ax as plain JSON values."""
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        return value.item()
    return value


def replayed(space: SearchSpace, manifest: Mapping[str, Any], trials: Sequence[Mapping[str, Any]]) -> Optimizer:
    opt = Optimizer(space, name=manifest["name"], seed=manifest["seed"], init=manifest["init"])
    for trial in trials:
        opt.attach(trial)
    return opt


# --- a trial ---------------------------------------------------------------------------------------------------------

def spread_of(values: Sequence[float]) -> dict[str, Any]:
    """bench.stats of the repeats, the values themselves, and the median's standard error (None for one value)."""
    s = bench.stats(values)
    sem = MEDIAN_SEM_FACTOR * s["sd"] / math.sqrt(s["n"]) if s["n"] >= 2 and s["sd"] is not None else None
    return {**s, "sem": sem, "values": [float(v) for v in values]}


def finish_trial(trial: dict[str, Any], space: SearchSpace) -> None:
    """Set a trial's status, medians and spread from its repeats."""
    repeats = trial["repeats"]
    failed = [r for r in repeats if r["outcome"] != "ok"]
    trial["ended"] = trial["timestamp"] = now()
    if failed or len(repeats) < trial["repeats_planned"]:
        trial["status"] = "failed"
        trial["reason"] = failed[0]["reason"] if failed else "not every repeat ran"
        return
    names = sorted({m for r in repeats for m in r["metrics"]})
    trial["spread"] = {m: spread_of([r["metrics"][m] for r in repeats if m in r["metrics"]]) for m in names}
    trial["median"] = {m: s["median"] for m, s in trial["spread"].items()}
    objective = trial["spread"][space.objective]
    trial["observation"] = {"metric": space.objective, "goal": space.goal, "mean": objective["median"],
                            "sem": objective["sem"], "statistic": f"median of {objective['n']} repeats"}
    trial["status"] = "completed"
    trial["reason"] = None


def _units(kind: str, metric: str) -> str | None:
    if kind == "inference":
        return bench.METRIC_UNITS.get(metric.split(".", 1)[0])
    return TRAIN_UNITS.get(metric)


# --- the driver ------------------------------------------------------------------------------------------------------

@dataclass
class Options:
    repeats: int = DEFAULT_REPEATS
    budget: int = DEFAULT_BUDGET
    minutes: float | None = None
    seed: int = DEFAULT_SEED
    init: int | None = None
    node: str = FAKE_NODE
    max_failures: int = DEFAULT_MAX_FAILURES
    commit: bool = False
    slot: str | None = None
    summary: bool = True


class Driver:
    """One session of a run: resume what the run directory holds, then trials until a stop (module docstring)."""

    def __init__(self, space: SearchSpace, runner: Any, run_dir: Path, work_dir: Path, options: Options, *,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.space = space
        self.runner = runner
        self.run_dir = Path(run_dir)
        self.work_dir = Path(work_dir)
        self.o = options
        self.clock = clock
        self.name = self.run_dir.name
        self.git = git_facts()
        self.versions = versions()
        self.optimizer: Optimizer | None = None
        self.commit_errors: list[str] = []

    # the manifest

    def manifest(self) -> dict[str, Any]:
        return {"schema": RUN_SCHEMA, "name": self.name, "created": now(), "driver": DRIVER, "kind": self.space.kind,
                "space": {"path": self.space.path, "sha256": self.space.sha256, "content": self.space.raw},
                "objective": {"metric": self.space.objective, "goal": self.space.goal},
                "runner": self.runner.describe(), "fake": bool(self.runner.fake), "node": self.o.node,
                "repeats": self.o.repeats, "seed": self.o.seed, "init": self.o.init, "git": self.git,
                "versions": self.versions, "pins": pins()}

    def check_manifest(self, manifest: Mapping[str, Any]) -> None:
        wanted = {"space sha256": (manifest["space"]["sha256"], self.space.sha256),
                  "kind": (manifest["kind"], self.space.kind), "repeats": (manifest["repeats"], self.o.repeats),
                  "seed": (manifest["seed"], self.o.seed), "init": (manifest["init"], self.o.init),
                  "node": (manifest["node"], self.o.node), "fake": (manifest["fake"], bool(self.runner.fake))}
        differ = [f"{k}: the run has {a!r}, this call {b!r}" for k, (a, b) in wanted.items() if a != b]
        if differ:
            raise RunError(f"{self.run_dir} was made with other settings, so it cannot be resumed this way: "
                           + "; ".join(differ))

    # one trial

    def new_trial(self, n: int, params: dict[str, Any], *, arm: str, generator: str) -> dict[str, Any]:
        problems = self.space.problems_with(params)
        if problems:
            raise RunError(f"trial {n} is outside the search space: {'; '.join(problems)}")
        return {"schema": TRIAL_SCHEMA, "run": self.name, "trial": n, "arm": arm, "generator": generator,
                "status": "running", "parameters": params, "applied": apply(self.space, params).describe(),
                "objective": {"metric": self.space.objective, "goal": self.space.goal},
                "repeats_planned": self.o.repeats,
                "conditions": {"driver": DRIVER, "node": self.o.node, "fake": bool(self.runner.fake),
                               "space": {"name": self.space.name, "path": self.space.path,
                                         "sha256": self.space.sha256},
                               "runner": self.runner.describe(), "versions": self.versions, "host": host_brief(),
                               "slot": self.o.slot},
                "repeats": [], "median": None, "spread": None, "observation": None, "reason": None,
                "started": now(), "ended": None, "timestamp": None, "git_sha": self.git.get("sha"),
                "git": self.git, "ledger_seq": None}

    def run_repeat(self, trial: dict[str, Any], r: int) -> dict[str, Any]:
        n = trial["trial"]
        workdir = self.work_dir / f"trial-{n:04d}" / f"repeat-{r}"
        workdir.mkdir(parents=True, exist_ok=True)
        applied = Applied(dict(trial["applied"]["env"]), list(trial["applied"]["flags"]))
        log(f"trial {n} repeat {r + 1}/{trial['repeats_planned']}: {trial['parameters']}")
        started, t0 = now(), self.clock()
        try:
            result = self.runner.run_once(applied, trial=n, repeat=r, workdir=workdir)
        except (OSError, bench.BenchError, RunError) as exc:
            result = RepeatResult("fail", {}, {}, f"{type(exc).__name__}: {exc}")
        seconds = self.clock() - t0
        outcome, reason = result.outcome, result.reason
        if outcome == "ok" and self.space.objective not in result.metrics:
            outcome, reason = "fail", f"the runner reported no {self.space.objective}"
        entry = {"repeat": r, "outcome": outcome, "reason": reason, "metrics": result.metrics,
                 "started": started, "ended": now(), "duration_s": round(seconds, 3), "git_sha": self.git.get("sha"),
                 "workdir": _shown(workdir), "conditions": result.conditions}
        value = result.metrics.get(self.space.objective)
        log(f"trial {n} repeat {r + 1}: {outcome} {self.space.objective}={value}" + (f" ({reason})" if reason else ""))
        return entry

    def ledger_row(self, trial: dict[str, Any]) -> None:
        kind = self.space.kind
        metrics = {f"optimize.{kind}.{m}": v for m, v in (trial["median"] or {}).items()
                   if ledger.finite_number(v) is not None and _METRIC.fullmatch(m)}
        units = {k: u for k, u in ((k, _units(kind, k.split(".", 2)[2])) for k in metrics) if u}
        objective = (trial["spread"] or {}).get(self.space.objective)
        conditions = {"run": self.name, "trial": trial["trial"], "arm": trial["arm"], "generator": trial["generator"],
                      "parameters": trial["parameters"], "applied": trial["applied"], "fake": bool(self.runner.fake),
                      "repeats": len(trial["repeats"]), "objective": trial["objective"], "spread": objective,
                      "space_sha256": self.space.sha256, "git_sha": trial["git_sha"],
                      "trial_file": _shown(trial_path(self.run_dir, trial["trial"]))}
        try:
            trial["ledger_seq"] = record(
                "optimize.trial", "ok" if trial["status"] == "completed" else "fail", metrics,
                {"run_id": f"{self.name}/trial-{trial['trial']:04d}", "slot": self.o.slot}, node=self.o.node,
                units=units, conditions=conditions, low_confidence=len(trial["repeats"]) < MIN_RUNS_PER_ARM,
                source=DRIVER, note=trial["reason"])
        except Exception as exc:   # the trial file is the record; a ledger that refuses the row is said, not fatal
            trial["ledger_error"] = f"{type(exc).__name__}: {exc}"[:300]
            log(f"no ledger row for trial {trial['trial']}: {trial['ledger_error']}")

    def commit(self, paths: Sequence[Path], message: str) -> None:
        if not self.o.commit:
            return
        error = git_commit(paths, message)
        if error:
            self.commit_errors.append(error)
            log(f"commit failed: {error}")

    def halt(self, state: RunState, started: float) -> str | None:
        """Why the next repeat must not start, or None."""
        if stop.stopped():
            return "stopped"
        if self.o.minutes is not None:
            elapsed = self.clock() - started
            if elapsed + (state.repeat_seconds() or 0.0) > self.o.minutes * 60.0:
                return "minutes"
        return None

    def optimizer_for(self, state: RunState) -> Optimizer:
        if self.optimizer is None:
            self.optimizer = replayed(self.space, state.manifest or {}, state.trials)
        return self.optimizer

    # the session

    def run(self) -> dict[str, Any]:
        started = self.clock()
        state = RunState.load(self.run_dir)
        if state.manifest is None:
            state.manifest = self.manifest()
            write_json(self.run_dir / "run.json", state.manifest)
        else:
            self.check_manifest(state.manifest)
        resumed = {**state.counts()}
        touched: list[int] = []
        failures = 0
        reason: str | None = None
        error: str | None = None
        try:
            while reason is None:
                trial = state.running
                if trial is None:
                    if len(state.trials) >= self.o.budget:
                        reason = "budget"
                        break
                    reason = self.halt(state, started)
                    if reason:
                        break
                    n = len(state.trials)
                    if n == 0:
                        trial = self.new_trial(0, self.space.baseline, arm="baseline", generator="baseline")
                    else:
                        suggestion = self.optimizer_for(state).suggest()
                        if suggestion is None:
                            reason = "exhausted"
                            break
                        index, params, arm, generator = suggestion
                        if index != n:
                            raise RunError(f"Ax suggested trial {index} where the run directory expects {n}")
                        trial = self.new_trial(n, params, arm=arm, generator=generator)
                    state.trials.append(trial)
                    write_json(trial_path(self.run_dir, n), trial)
                    log(f"trial {n} ({trial['generator']}): {trial['parameters']}")
                touched.append(trial["trial"])
                for r in range(len(trial["repeats"]), trial["repeats_planned"]):
                    reason = self.halt(state, started)
                    if reason:
                        break
                    entry = self.run_repeat(trial, r)
                    trial["repeats"].append(entry)
                    write_json(trial_path(self.run_dir, trial["trial"]), trial)
                    if entry["outcome"] != "ok":
                        break
                if reason:
                    break
                finish_trial(trial, self.space)
                self.ledger_row(trial)
                write_json(trial_path(self.run_dir, trial["trial"]), trial)
                if self.optimizer is not None:
                    self.optimizer.finish(trial)
                self.commit([self.run_dir / "run.json", trial_path(self.run_dir, trial["trial"])],
                            f"Record {self.name} trial {trial['trial']}")
                value = (trial["median"] or {}).get(self.space.objective)
                log(f"trial {trial['trial']} {trial['status']}: {self.space.objective} median {value}"
                    + (f" ({trial['reason']})" if trial["reason"] else ""))
                failures = failures + 1 if trial["status"] == "failed" else 0
                if failures >= self.o.max_failures:
                    reason = "failures"
        except KeyboardInterrupt:
            reason = "interrupted"
            self.finish_session(state, reason, started, touched, resumed, None)
            raise
        except (RunError, SpaceError) as exc:
            reason, error = "error", str(exc)
        except Exception as exc:   # an Ax or runner failure: the files already on disk stay whole
            reason, error = "error", f"{type(exc).__name__}: {exc}"
        return self.finish_session(state, reason or "budget", started, touched, resumed, error)

    def finish_session(self, state: RunState, reason: str, started: float, touched: list[int],
                       resumed: Mapping[str, int], error: str | None) -> dict[str, Any]:
        summary: dict[str, Any] | None = None
        if self.o.summary and any(t["status"] == "completed" for t in state.trials):
            try:
                summary = summarize(self.run_dir, optimizer=self.optimizer)
                self.commit([self.run_dir / "summary.json", self.run_dir / "summary.md"],
                            f"Summarize {self.name} after {len(state.trials)} trials")
            except Exception as exc:
                log(f"no summary: {type(exc).__name__}: {exc}")
        counts = state.counts()
        minutes = (self.clock() - started) / 60.0
        ok = reason in STOP_OK
        best = (summary or {}).get("best")
        metrics = {"optimize.trials_run": len(set(touched)), "optimize.trials_total": counts["total"],
                   "optimize.minutes_used": round(minutes, 3)}
        try:
            record("optimize.run", "ok" if ok else reason, metrics, {"run_id": self.name, "slot": self.o.slot},
                   node=self.o.node, conditions={"stop_reason": reason, "space": self.space.name,
                                                 "space_sha256": self.space.sha256, "budget": self.o.budget,
                                                 "minutes": self.o.minutes, "repeats": self.o.repeats,
                                                 "fake": bool(self.runner.fake), "counts": counts,
                                                 "resumed_with": dict(resumed)},
                   source=DRIVER, note=error)
        except Exception as exc:
            log(f"no ledger row for the session: {type(exc).__name__}: {exc}")
        answer = {"ok": ok, "outcome": "ok" if ok else reason, "stop_reason": reason, "run": self.name,
                  "run_dir": str(self.run_dir), "trials": counts, "resumed_with": dict(resumed),
                  "trials_touched": sorted(set(touched)), "minutes_used": round(minutes, 3), "best": best,
                  "summary_path": str(self.run_dir / "summary.json") if summary else None,
                  "metrics": metrics, "refs": {"run_id": self.name}}
        if error:
            answer["error"] = error
        if self.commit_errors:
            answer["commit_errors"] = self.commit_errors
        return answer


def git_commit(paths: Sequence[Path], message: str) -> str | None:
    """Commit these files, and only these, to the repository they are in; the error text, or None."""
    existing = [Path(p).resolve() for p in paths if Path(p).exists()]
    if not existing:
        return None
    where = existing[0].parent
    try:
        top = Path(subprocess.run(["git", "-C", str(where), "rev-parse", "--show-toplevel"], capture_output=True,
                                  text=True, timeout=30, check=True).stdout.strip()).resolve()
        rel = [p.relative_to(top).as_posix() for p in existing]
        subprocess.run(["git", "-C", str(top), "add", "--", *rel], capture_output=True, text=True, timeout=60,
                       check=True)
        staged = subprocess.run(["git", "-C", str(top), "diff", "--cached", "--quiet", "--", *rel], timeout=60)
        if staged.returncode == 0:
            return None   # nothing changed since the last commit
        subprocess.run(["git", "-C", str(top), "commit", "-q", "-m", message, "--", *rel], capture_output=True,
                       text=True, timeout=60, check=True)
    except subprocess.CalledProcessError as exc:
        return f"git {exc.cmd[3] if len(exc.cmd) > 3 else ''}: {(exc.stderr or exc.stdout or '').strip()[:300]}"
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return f"{type(exc).__name__}: {exc}"[:300]
    return None


# --- the summary -----------------------------------------------------------------------------------------------------

def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def summarize(run_dir: Path, *, model: bool = True, optimizer: Optimizer | None = None) -> dict[str, Any]:
    """Write summary.json and summary.md for a run directory: the best configuration (the model's, or the best
    observed median when there is no model), the posterior at every completed trial, and one table row per trial."""
    run_dir = Path(run_dir)
    state = RunState.load(run_dir)
    if state.manifest is None:
        raise RunError(f"{run_dir} holds no run.json")
    m = state.manifest
    space = parse_space(m["space"]["content"], path=m["space"]["path"])
    done = [t for t in state.trials if t["status"] == "completed"]
    observed_best = None
    for t in done:
        if observed_best is None or space.better(t["median"][space.objective], observed_best["median"][space.objective]):
            observed_best = t
    baseline = next((t for t in state.trials if t["generator"] == "baseline"), None)
    posterior: dict[str, Any] = {"available": False, "reason": "no model was asked for"}
    if model:
        if not ax_installed():
            posterior = {"available": False, "reason": "Ax is not installed here (python -m lab optimize install)"}
        elif len(done) < 2:
            posterior = {"available": False, "reason": f"{len(done)} completed trial(s); a model needs two or more"}
        else:
            try:
                opt = optimizer or replayed(space, m, state.trials)
                posterior = opt.posterior(state.trials)
            except Exception as exc:
                posterior = {"available": False, "reason": f"{type(exc).__name__}: {exc}"[:400]}
    by_trial = {a["trial"]: a["predicted"] for a in posterior.get("arms", [])}
    if posterior.get("available"):
        chosen = next(t for t in state.trials if t["trial"] == posterior["best"]["trial"])
        best = {"source": "model", "trial": chosen["trial"], "arm": chosen["arm"], "parameters": chosen["parameters"],
                "observed": (chosen.get("spread") or {}).get(space.objective),
                "predicted": posterior["best"]["predicted"].get(space.objective)}
    elif observed_best is not None:
        best = {"source": "observed", "trial": observed_best["trial"], "arm": observed_best["arm"],
                "parameters": observed_best["parameters"], "observed": observed_best["spread"][space.objective],
                "predicted": None}
    else:
        best = None
    if best is not None:
        best["applied"] = apply(space, best["parameters"]).describe()
    base_value = (baseline or {}).get("median", None) or {}
    change = None
    if best is not None and best["observed"] and base_value.get(space.objective):
        b0 = base_value[space.objective]
        change = {"baseline": b0, "best": best["observed"]["median"],
                  "relative": (best["observed"]["median"] - b0) / b0}
    summary = {"schema": SUMMARY_SCHEMA, "run": m["name"], "written": now(), "driver": DRIVER, "kind": space.kind,
               "space": {"name": space.name, "sha256": space.sha256, "path": space.path},
               "objective": {"metric": space.objective, "goal": space.goal}, "tracking": list(space.tracking),
               "node": m["node"], "fake": m["fake"], "repeats": m["repeats"], "trials": state.counts(),
               "best": best, "best_observed_trial": observed_best["trial"] if observed_best else None,
               "baseline": {"trial": baseline["trial"], "parameters": baseline["parameters"],
                            "status": baseline["status"], "median": baseline.get("median")} if baseline else None,
               "change_vs_baseline": change, "posterior": posterior, "git": git_facts(), "versions": versions()}
    write_json(run_dir / "summary.json", summary)
    (run_dir / "summary.md").write_text(markdown(summary, state.trials, space, by_trial), encoding="utf-8")
    return summary


def markdown(summary: Mapping[str, Any], trials: Sequence[Mapping[str, Any]], space: SearchSpace,
             predicted: Mapping[int, Mapping[str, Any]]) -> str:
    obj = space.objective
    counts = summary["trials"]
    lines = [f"# Ax settings run {summary['run']}", "",
             f"Objective: {space.goal} {obj}, the median of {summary['repeats']} repeats per trial. "
             f"Runner: {space.kind}{' (fake)' if summary['fake'] else ''}. Node: {summary['node']}. "
             f"Trials: {counts['completed']} completed, {counts['failed']} failed, {counts['running']} in flight. "
             f"Search space: {space.name} ({summary['space']['sha256'][:12]}).", ""]
    best = summary["best"]
    if best:
        settings = ", ".join(f"{k}={_fmt(v)}" for k, v in best["parameters"].items())
        observed = best["observed"] or {}
        text = (f"Best configuration ({'model' if best['source'] == 'model' else 'best observed median'}): "
                f"trial {best['trial']} ({settings}). Observed {_fmt(observed.get('median'))} "
                f"(min {_fmt(observed.get('min'))}, max {_fmt(observed.get('max'))}, n={observed.get('n')})")
        if best.get("predicted"):
            text += f"; posterior {_fmt(best['predicted']['mean'])} +- {_fmt(best['predicted']['sem'])}"
        lines += [text + ".", ""]
    change = summary["change_vs_baseline"]
    if change:
        lines += [f"Baseline (trial 0) median {_fmt(change['baseline'])}; best {_fmt(change['best'])} "
                  f"({change['relative'] * 100:+.1f} %).", ""]
    post = summary["posterior"]
    lines += [f"Posterior: {'model ' + str(post.get('model')) if post.get('available') else post.get('reason')}.", ""]
    names = [p.name for p in space.parameters]
    head = ["Trial", "Status", "Generator", *names, f"{obj} median", "min", "max", "sem", "posterior mean",
            "posterior sem", *[f"{t} median" for t in space.tracking]]
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for t in trials:
        s = (t.get("spread") or {}).get(obj) or {}
        p = (predicted.get(t["trial"]) or {}).get(obj) or {}
        row = [str(t["trial"]), t["status"], t["generator"], *[_fmt(t["parameters"].get(n)) for n in names],
               _fmt(s.get("median")), _fmt(s.get("min")), _fmt(s.get("max")), _fmt(s.get("sem")),
               _fmt(p.get("mean")), _fmt(p.get("sem")),
               *[_fmt((t.get("median") or {}).get(m)) for m in space.tracking]]
        lines.append("| " + " | ".join(row) + " |")
    failed = [t for t in trials if t["status"] == "failed"]
    if failed:
        lines += ["", "Failed trials:", ""] + [f"- trial {t['trial']}: {t['reason']}" for t in failed]
    return "\n".join(lines) + "\n"


# --- install ---------------------------------------------------------------------------------------------------------

def install_command(python: str = sys.executable) -> tuple[list[str], str | None]:
    """The pip command that installs the pinned requirements, and the constraint text that holds torch in place."""
    argv = [python, "-m", "pip", "install", "-r", str(REQUIREMENTS)]
    if sys.prefix == sys.base_prefix:      # not a venv: a pod's interpreter, no root, so the user site
        argv.append("--user")
    torch = _dist_version("torch")
    return argv, (f"torch=={torch}\n" if torch else None)


def install(*, dry_run: bool = False, run: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    argv, constraint = install_command()
    folder = None
    if constraint:
        folder = Path(tempfile.mkdtemp(prefix="optimize-install-"))
        (folder / "torch-constraint.txt").write_text(constraint, encoding="utf-8")
        argv += ["-c", str(folder / "torch-constraint.txt")]
    if dry_run:
        return {"ok": True, "outcome": "dry_run", "argv": argv, "torch_held_at": (constraint or "").strip() or None,
                "pins": pins()}
    log("installing: " + " ".join(argv))
    code = run(argv).returncode
    found = {name: _dist_version(name) for name in pins()}
    mismatched = {k: {"pinned": v, "installed": found.get(k)} for k, v in pins().items() if found.get(k) != v}
    ok = code == 0 and not mismatched
    answer = {"ok": ok, "outcome": "ok" if ok else "fail", "argv": argv, "exit_code": code, "installed": found,
              "torch_held_at": (constraint or "").strip() or None}
    if not ok:
        answer["error"] = (f"pip exited with code {code}" if code else
                           f"installed versions differ from the pins: {json.dumps(mismatched)}")
    return answer


# --- the command line --------------------------------------------------------------------------------------------------

def default_run_dir(kind: str, *, fake: bool) -> Path:
    date = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    return STATE_DIR / f"ax-{kind}-fake-{date}" if fake else DATA_DIR / f"ax-{kind}-{date}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m lab optimize", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run or resume an optimization (the default sub-command)")
    r.add_argument("--space", default=str(SAMPLE_SPACE), help="the search-space JSON file")
    r.add_argument("--runner", choices=KINDS, default=None, help="default: the space's kind")
    r.add_argument("--fake", action="store_true", help="the offline fakes: node local, lab/state/optimize")
    r.add_argument("--run-dir", default=None, help="default: lab/refine/research/kernels/data/ax-<kind>-<date>")
    r.add_argument("--work-dir", default=None, help="scratch for the runners; default lab/state/optimize/<run>/work")
    r.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    r.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help="trials the run holds when it stops")
    r.add_argument("--minutes", type=float, default=None, help="this call's wall clock")
    r.add_argument("--node", default=None, help="the pod (pod-c, pod-d, ...); a fake run is node local")
    r.add_argument("--seed", type=int, default=DEFAULT_SEED)
    r.add_argument("--init", type=int, default=None, help="initialization trials, the baseline included")
    r.add_argument("--max-failures", type=int, default=DEFAULT_MAX_FAILURES)
    r.add_argument("--slot", default=None, help="the GPU Queue row this runs under, for the ledger refs")
    r.add_argument("--commit", action="store_true", help="commit each finished trial file to git")
    r.add_argument("--no-summary", action="store_true")
    r.add_argument("--dry-run", action="store_true", help="print the plan; run and write nothing")
    r.add_argument("--bench", default="", help="lab/bench.py run arguments for the real inference runner")
    r.add_argument("--train-cmd", default=None, help="the trainer template: {steps} {out} {trial} {repeat} {python}")
    r.add_argument("--train-env", action="append", default=[], help="NAME=VALUE for every training run")
    r.add_argument("--train-timeout-s", type=float, default=DEFAULT_TRAIN_TIMEOUT_S)
    r.add_argument("--noise-seed", type=int, default=0, help="the fakes' noise seed")
    s = sub.add_parser("summary", help="write summary.json and summary.md for a run directory")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--no-model", action="store_true", help="best observed median only; no Ax")
    i = sub.add_parser("install", help="pip install -r lab/requirements-optimize.txt, torch held where it is")
    i.add_argument("--dry-run", action="store_true")
    return p


def _env_pairs(items: Sequence[str]) -> dict[str, str]:
    out = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not _ENV_NAME.fullmatch(key):
            raise RunError(f"--train-env takes NAME=VALUE, not {item!r}")
        out[key] = value
    return out


def make_runner(space: SearchSpace, args: argparse.Namespace, work_dir: Path, node: str) -> Any:
    if space.kind == "inference":
        if args.fake:
            return fake_inference_runner(space, work_dir, noise_seed=args.noise_seed)
        return real_inference_runner(space, args.bench, node=node)
    if args.fake:
        return fake_training_runner(space, noise_seed=args.noise_seed)
    if not args.train_cmd:
        raise RunError("a real training run needs --train-cmd, the trainer's command with {steps} and {out}")
    text = args.train_cmd.strip()
    command = json.loads(text) if text.startswith("[") else shlex.split(text)
    return TrainingRunner(command, steps=space.fixed.get("steps"), fake=False, env=_env_pairs(args.train_env),
                          timeout_s=args.train_timeout_s)


def plan(space: SearchSpace, args: argparse.Namespace, run_dir: Path, work_dir: Path, node: str,
         gate: str | None) -> dict[str, Any]:
    """What a run would do, read from the search space and the run directory; nothing is run or written."""
    state = RunState.load(run_dir) if run_dir.exists() else RunState(run_dir, None, [])
    counts = state.counts()
    steps: list[str] = []
    if state.running:
        t = state.running
        steps.append(f"finish trial {t['trial']}: {t['repeats_planned'] - len(t['repeats'])} of "
                     f"{t['repeats_planned']} repeats left")
    total = len(state.trials)
    if total == 0 and args.budget > 0:
        steps.append(f"trial 0: the pinned baseline {space.baseline}")
        total = 1
    if total < args.budget:
        init = f"{args.init} initialization trials" if args.init else "Ax's own initialization budget"
        steps.append(f"trials {total} to {args.budget - 1}: Ax suggestions (Sobol for {init}, the baseline "
                     "included, then BoTorch)")
    estimate = state.repeat_seconds()
    runner_text = {"kind": space.kind, "fake": bool(args.fake)}
    try:
        runner_text = {**runner_text, **(make_runner(space, args, work_dir, node).describe() if args.fake
                                         or space.kind == "training" else {"bench": args.bench or "(defaults)"})}
    except (RunError, OSError) as exc:
        runner_text["problem"] = str(exc)
    return {"run": run_dir.name, "run_dir": str(run_dir), "work_dir": str(work_dir), "exists": run_dir.exists(),
            "resumes": state.manifest is not None, "trials_on_disk": counts, "search_space": space.describe(),
            "runner": runner_text, "repeats": args.repeats, "budget": args.budget, "minutes": args.minutes,
            "seed": args.seed, "init": args.init, "node": node, "gate": gate or "ok", "max_failures": args.max_failures,
            "commit": args.commit, "slot": args.slot, "next": steps,
            "repeat_seconds_so_far": estimate,
            "ax": {"installed": ax_installed(), "pins": pins(), "versions": versions()}}


def _show_plan(p: Mapping[str, Any]) -> None:
    s = p["search_space"]
    log(f"plan for {p['run']} ({'resume' if p['resumes'] else 'new run'}) in {p['run_dir']}")
    log(f"  {s['kind']} space {s['name']}: {s['objective']['goal']} {s['objective']['metric']}, "
        f"tracking {', '.join(s['tracking']) or 'nothing'}, fixed {json.dumps(s['fixed'])}")
    for param in s["parameters"]:
        where = f"env {param['env']}" if param.get("env") else f"flag {param['flag']}"
        span = param.get("bounds") if param["type"] == "range" else param.get("values")
        log(f"  {param['name']}: {param['type']} {span}, baseline {param['baseline']}, {where}")
    log(f"  repeats {p['repeats']}, budget {p['budget']} trials, minutes {p['minutes']}, node {p['node']}, "
        f"gate {p['gate']}, Ax installed {p['ax']['installed']}")
    for line in p["next"]:
        log(f"  next: {line}")


def run_command(args: argparse.Namespace) -> dict[str, Any]:
    space = load_space(args.space)
    if args.runner and args.runner != space.kind:
        raise RunError(f"--runner {args.runner} does not match the space's kind {space.kind}")
    if args.repeats < 1 or args.budget < 1:
        raise RunError("--repeats and --budget are at least 1")
    node = FAKE_NODE if args.fake and args.node is None else args.node
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir(space.kind, fake=args.fake)
    work_dir = Path(args.work_dir) if args.work_dir else STATE_DIR / run_dir.name / "work"
    gate = node_gate(node, fake=args.fake) if not args.dry_run or args.fake or node else node_gate(
        node, fake=args.fake, rocminfo=lambda: "")
    if args.dry_run:
        p = plan(space, args, run_dir, work_dir, node or "(none given)", gate)
        _show_plan(p)
        return {"ok": True, "outcome": "dry_run", "plan": p}
    if gate:
        return {"ok": False, "outcome": "refused", "error": f"node gate: {gate}"}
    if not args.fake and args.minutes is None:
        return {"ok": False, "outcome": "refused",
                "error": "a real run needs --minutes, the wall clock of the GPU slot it runs in"}
    if space.kind == "inference" or args.budget > 1:
        if not ax_installed() and args.budget > 1:
            return {"ok": False, "outcome": "missing_dependency",
                    "error": "Ax and BoTorch are not installed here: python -m lab optimize install"}
    runner = make_runner(space, args, work_dir, str(node))
    options = Options(repeats=args.repeats, budget=args.budget, minutes=args.minutes, seed=args.seed, init=args.init,
                      node=str(node), max_failures=args.max_failures, commit=args.commit, slot=args.slot,
                      summary=not args.no_summary)
    return Driver(space, runner, run_dir, work_dir, options).run()


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Optimize engine or trainer settings with Ax and BoTorch: one committed JSON file per trial (R45)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or (argv[0] not in ("run", "summary", "install") and argv[0] not in ("-h", "--help")):
        argv = ["run", *argv]
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "install":
            return install(dry_run=args.dry_run)
        if args.cmd == "summary":
            summary = summarize(Path(args.run_dir), model=not args.no_model)
            return {"ok": True, "outcome": "ok", "run": summary["run"], "best": summary["best"],
                    "posterior": {k: v for k, v in summary["posterior"].items() if k != "arms"},
                    "summary_path": str(Path(args.run_dir) / "summary.json"),
                    "markdown_path": str(Path(args.run_dir) / "summary.md")}
        return run_command(args)
    except SpaceError as exc:
        return {"ok": False, "outcome": "bad_space", "error": str(exc), "problems": exc.problems}
    except RunError as exc:
        return {"ok": False, "outcome": "refused", "error": str(exc)}


if __name__ == "__main__":
    answer = main()
    print(json.dumps(answer, default=str))
    sys.exit(0 if answer.get("ok") else 1)
