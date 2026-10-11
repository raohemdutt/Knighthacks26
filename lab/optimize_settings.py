"""The settings optimizer: R49's night configuration driven with Ax over R35's and R36's search spaces.

    python -m lab optimize --dry-run                                 the inference lane's plan; runs and writes nothing
    python -m lab optimize --dry-run --lane training --minutes 120
    python -m lab optimize run --lane inference --fake --workers 2 --budget 8         offline, the fakes
    python -m lab optimize run --lane inference --node pod-d --minutes 60 --run-dir DIR \\
        --bench "--engine vllm --checkpoint ~/models/Qwen3.8-27B-INT4 --engine-dir ~/vllm --serve-cmd ..."
    python -m lab optimize run --lane training --node pod-d --minutes 60 --run-dir DIR \\
        --train-cmd "python3 lab/train.py ... --optimizer-settings {settings} --optimizer-out {out} --max-steps {steps}"
    python -m lab optimize confirm --run-dir DIR ...                 the confirmation spawns only
    python -m lab optimize summary --run-dir DIR                     best arm, posterior, Markdown table, verdict
    python -m lab optimize install [--dry-run]                       pip install -r lab/requirements-optimize.txt

`python lab/optimize_settings.py ...` is the same by path on a pod. Every call prints one JSON line on stdout (exit 0
when ok); progress and the dry run's plan go to stderr.

What it reads. The night configuration (--config, default lab/refine/research/kernels/data/ax-config-night.json,
schema lab.optimize.night/1) and the lane's search space (--space, default the lane's search_space; both formats,
lab/optimize_space.py). Where the configuration and R45's Done when differ, the configuration wins: one spawn per
configuration with its requests as the repeats (not N separate repeats), warm start from the baseline and the prior
arms (no Sobol points), asynchronous trials across workers.

The search phase. The lane's initial arms go first: the baseline (attach_baseline) and the prior arms
(attach_trial), as trials 0, 1, 2, ... on the first free workers. Then each free worker asks the strategy for one
trial (get_next_trials with max_trials=1) while the other workers' trials are still running; every second ask of a
worker fixes that worker's last compile class (grouping), so each class is started cold once and reused warm once.
Each trial is one server start (inference: lab/bench.py at the screen fidelity, or finalist when the model ranks the
arm in its top 3) or one trainer process (training), with the restart class's ready timeout; a hopeless trial stops
early (lab/optimize_plan.py). Ax receives each observation as (median, SEM) for the objective, the constraints and
the tracking metrics. A suggestion that hits an invalid combination of the space is written as an invalid trial and
marked failed without running. The objective is the lane's (inference: joules_per_token.b1, minimized) with the
lane's outcome_constraints (inference: decode_tok_s.b1 >= 2.88 and ttft_s.b1 <= 2.0, from R56 until the configuration
on main carries them).

Stops. --budget trials in the run (initial arms and invalid suggestions included), the run's minute budget (the
lane's budget.minutes over all sessions, recorded in run.json) and this call's --minutes (a GPU slot): no trial starts
whose estimated seconds would pass either. max_consecutive_failures failed trials in a row stop a worker; the STOP
file, Ctrl-C and a strategy with nothing left stop the run. The trial in flight at a stop keeps its file as running
and is re-run first by the next session; a finished trial is never re-run.

The confirmation phase (after the search ends on its trial budget or the strategy is done, or with `confirm`).
Spawns of the best feasible arm and the baseline in the order best, baseline, ... on the first worker at the confirm
fidelity, then lab/machine/verdict.py's Welch test (lab/optimize_plan.py confirm_verdict). Each spawn is a file
confirm/confirm-NN.json, so it resumes too. The R21 correctness gate on the best arm (and on any arm with speculative
tokens) is not run here; the summary says it is still owed.

Files. <run-dir>/run.json (lane, space and configuration hashes, strategy, seed, workers, sessions),
trials/trial-NNNN.json per trial (schema lab.optimize.trial/2: parameters, launch args and env, worker, restart
class, compile class, fidelity, the runner's measurement with every request or step, the observation Ax got,
conditions with versions, host and git sha, timestamps, ledger seq), confirm/confirm-NN.json, summary.json and
summary.md. Each finished trial is an optimize.trial ledger row and each session an optimize.run row. --commit
commits each finished file to the git repository the run directory is in. A fake run records node local and
defaults to lab/state/optimize (gitignored); a real run names its --node, needs --minutes and passes the node gate
(the pod's rocminfo names gfx1151). Workers whose runner is "hub" (pods E and F) need lab.controller run-script from
the driver host, which is not built: a real run uses the local worker it runs on.

Ax is imported lazily (lab/optimize_ax.py); --strategy stub runs everything with the stub strategy and no Ax.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import importlib.metadata
import json
import os
import platform
import re
import shlex
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if not __package__:  # run by path on a pod: the repo root, one folder up, holds lab/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab import optimize_plan as rules  # noqa: E402
from lab import optimize_runners as runners  # noqa: E402
from lab.machine import ledger, stop  # noqa: E402
from lab.machine.record import record  # noqa: E402
from lab.optimize_ax import STRATEGIES, ax_installed, make_strategy  # noqa: E402
from lab.optimize_space import (NIGHT_CONFIG, REPO, Lane, Night, SearchSpace, SpaceError, check_lane,  # noqa: E402
                                initial_arms, lane_of, load_night, load_space, same, shown)

DRIVER = "lab/optimize_settings.py 2"
RUN_SCHEMA = "lab.optimize.run/2"
TRIAL_SCHEMA = "lab.optimize.trial/2"
CONFIRM_SCHEMA = "lab.optimize.confirm/1"
SUMMARY_SCHEMA = "lab.optimize.summary/2"
REQUIREMENTS = REPO / "lab" / "requirements-optimize.txt"
STATE_DIR = REPO / "lab" / "state" / "optimize"
DATA_DIR = REPO / "lab" / "refine" / "research" / "kernels" / "data"
FAKE_NODE = runners.FAKE_NODE
FINISHED = ("completed", "failed", "early_stopped", "invalid")
STOP_OK = ("budget", "minutes", "run_minutes", "exhausted", "confirmed")
MAX_INVALID_IN_A_ROW = 20
_TRIAL_FILE = re.compile(r"trial-(\d{4,})\.json")
_CONFIRM_FILE = re.compile(r"confirm-(\d{2,})\.json")


class RunError(Exception):
    """A run that cannot start or resume as asked."""


def log(message: str) -> None:
    print(f"[optimize] {message}", file=sys.stderr, flush=True)


def now() -> str:
    return ledger.now_text()


def git_facts(where: Path = REPO) -> dict[str, Any]:
    try:
        sha = subprocess.run(["git", "-C", str(where), "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=20, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(where), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=20, check=True).stdout.strip() != ""
        return {"sha": sha or None, "dirty": dirty}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"sha": None, "dirty": None, "unavailable": f"git: {exc}"[:300]}


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def versions() -> dict[str, Any]:
    return {"python": platform.python_version(), **{n: _version(n) for n in ("ax-platform", "botorch", "torch")}}


def pins(path: Path = REQUIREMENTS) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        name, sep, version = line.split("#", 1)[0].strip().partition("==")
        if sep:
            out[name.strip().lower()] = version.strip()
    return out


def host_brief() -> dict[str, Any]:
    return {"hostname": socket.gethostname(), "machine": platform.machine(),
            "jupyterhub_user": os.environ.get("JUPYTERHUB_USER")}


def write_json(path: Path, data: Any) -> None:
    """Whole or not at all: a temporary file beside it, then a replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def node_gate(node: str | None, *, fake: bool, rocminfo: Callable[[], str] | None = None) -> str | None:
    """Why this run may not start here, or None: a fake run is node local; a real run names a pod lab/bench.py knows
    and that pod's rocminfo names gfx1151 (lab/pod_power.py gate)."""
    from lab import bench, pod_power

    if fake:
        return None if node in (None, FAKE_NODE) else f"a fake run records node {FAKE_NODE!r}, not {node!r}"
    if not node or node not in bench.SAMPLER_NODE:
        return f"a real run names its pod: --node {' | '.join(bench.SAMPLER_NODE)}"

    def read() -> str:
        program = pod_power.find_tool("rocminfo")
        if program is None:
            raise pod_power.ReadError("rocminfo is not installed")
        return pod_power.run_command([program], timeout_s=30.0)
    return pod_power.gate(bench.SAMPLER_NODE[node], rocminfo or read)


# --- the run directory ------------------------------------------------------------------------------------------------

def trial_path(run_dir: Path, n: int) -> Path:
    return run_dir / "trials" / f"trial-{n:04d}.json"


def confirm_path(run_dir: Path, n: int) -> Path:
    return run_dir / "confirm" / f"confirm-{n:02d}.json"


def _numbered(folder: Path, pattern: re.Pattern, schema: str, key: str) -> list[dict[str, Any]]:
    names = sorted(p.name for p in folder.iterdir() if pattern.fullmatch(p.name)) if folder.is_dir() else []
    out = []
    for i, name in enumerate(names):
        if int(pattern.fullmatch(name).group(1)) != i:
            raise RunError(f"{folder} misses file number {i}: the files run 0, 1, 2, ... with no gap")
        try:
            item = json.loads((folder / name).read_text(encoding="utf-8"))
        except ValueError as exc:
            raise RunError(f"{folder / name} is not JSON: {exc}") from exc
        if item.get("schema") != schema or item.get(key) != i:
            raise RunError(f"{folder / name} is not number {i} in the {schema} format")
        out.append(item)
    return out


@dataclass
class RunState:
    run_dir: Path
    manifest: dict[str, Any] | None
    trials: list[dict[str, Any]]
    confirms: list[dict[str, Any]]

    @classmethod
    def load(cls, run_dir: Path) -> RunState:
        manifest = None
        if (run_dir / "run.json").exists():
            manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            if manifest.get("schema") != RUN_SCHEMA:
                raise RunError(f"{run_dir / 'run.json'} is not a {RUN_SCHEMA} file")
        trials = _numbered(run_dir / "trials", _TRIAL_FILE, TRIAL_SCHEMA, "trial")
        confirms = _numbered(run_dir / "confirm", _CONFIRM_FILE, CONFIRM_SCHEMA, "spawn")
        if (trials or confirms) and manifest is None:
            raise RunError(f"{run_dir} has trial files but no run.json")
        return cls(run_dir, manifest, trials, confirms)

    def counts(self) -> dict[str, int]:
        out = {"total": len(self.trials), "running": 0, **{s: 0 for s in FINISHED}}
        for t in self.trials:
            out[t["status"]] += 1
        return out

    def minutes_before(self) -> float:
        return sum(s.get("minutes") or 0.0 for s in (self.manifest or {}).get("sessions", []))


# --- the driver ---------------------------------------------------------------------------------------------------------

@dataclass
class Options:
    budget: int | None = None
    minutes: float | None = None              # this call's wall clock
    run_minutes: float | None = None          # the run's, over every session (default the lane's budget.minutes)
    strategy: str = "ax"
    seed: int | None = None
    node: str = FAKE_NODE
    fake: bool = True
    commit: bool = False
    slot: str | None = None
    confirm: bool = True
    confirm_minutes: float | None = None


@dataclass
class WorkerSlot:
    name: str
    runner: Any
    stagger_s: float = 0.0
    last_class: dict[str, Any] | None = None
    asks: int = 0
    failures: int = 0
    done: str | None = None
    started: bool = False


class Driver:
    """One session of a run: replay what the run directory holds, then trials until a stop (module docstring)."""

    def __init__(self, night: Night, lane: Lane, space: SearchSpace, workers: Sequence[WorkerSlot], run_dir: Path,
                 work_dir: Path, options: Options) -> None:
        self.night, self.lane, self.space = night, lane, space
        self.workers = list(workers)
        self.run_dir, self.work_dir, self.o = Path(run_dir), Path(work_dir), options
        self.name = self.run_dir.name
        self.git = git_facts()
        self.versions = versions()
        self.lock = threading.RLock()
        self.strategy: Any = None
        self.commit_errors: list[str] = []
        self.arms = initial_arms(lane, space)
        # the lane's own search space must match it exactly; another space (--space, a tiny one for tests or a
        # rehearsal) runs with the difference recorded in run.json
        self.mismatch = check_lane(lane, space)
        if self.mismatch and space.path == lane.search_space:
            raise RunError("; ".join(self.mismatch))

    # manifest and replay

    def manifest(self) -> dict[str, Any]:
        return {"schema": RUN_SCHEMA, "name": self.name, "created": now(), "driver": DRIVER, "lane": self.lane.name,
                "task": self.lane.task, "space": {"path": self.space.path, "sha256": self.space.sha256,
                                                  "format": self.space.format},
                "night": {"path": self.night.path, "sha256": self.night.sha256},
                "objective": {"metric": self.lane.metric, "goal": self.lane.goal},
                "constraints": [c.text() for c in self.lane.constraints], "strategy": self.o.strategy,
                "seed": self.o.seed, "node": self.o.node, "fake": self.o.fake,
                "workers": [{"name": w.name, "runner": w.runner.describe()} for w in self.workers],
                "initial_arms": [a["name"] for a in self.arms], "lane_space_differences": self.mismatch,
                "git": self.git, "versions": self.versions,
                "sessions": []}

    def check_manifest(self, m: Mapping[str, Any]) -> None:
        wanted = {"lane": (m["lane"], self.lane.name), "space sha256": (m["space"]["sha256"], self.space.sha256),
                  "strategy": (m["strategy"], self.o.strategy), "seed": (m["seed"], self.o.seed),
                  "node": (m["node"], self.o.node), "fake": (m["fake"], self.o.fake),
                  "constraints": (m["constraints"], [c.text() for c in self.lane.constraints])}
        differ = [f"{k}: the run has {a!r}, this call {b!r}" for k, (a, b) in wanted.items() if a != b]
        if differ:
            raise RunError(f"{self.run_dir} was made with other settings and cannot resume this way: "
                           + "; ".join(differ))

    def replay(self, state: RunState) -> None:
        """A fresh strategy with every finished trial attached in order; a running trial is re-attached to re-run."""
        self.strategy = make_strategy(self.o.strategy, self.space, self.lane,
                                      generation_strategy=self.night.generation_strategy, seed=self.o.seed)
        for t in state.trials:
            index = self.strategy.attach(t["parameters"], baseline=t["source"] == "baseline")
            if index != t["trial"]:
                raise RunError(f"the strategy numbered trial {t['trial']} as {index}")
            self.tell(t)

    def tell(self, t: Mapping[str, Any]) -> None:
        if t["status"] == "completed":
            self.strategy.complete(t["trial"], {k: tuple(v) for k, v in t["observation"].items()})
        elif t["status"] == "early_stopped":
            self.strategy.early_stopped(t["trial"], {k: tuple(v) for k, v in (t["observation"] or {}).items()})
        elif t["status"] in ("failed", "invalid"):
            self.strategy.fail(t["trial"])

    # one trial

    def baseline_trial(self, state: RunState) -> Mapping[str, Any] | None:
        return next((t for t in state.trials if t["source"] == "baseline" and t["status"] == "completed"), None)

    def new_trial(self, n: int, params: Mapping[str, Any], *, arm: str, source: str, priors: Sequence[str] = (),
                  worker: WorkerSlot | None, fidelity: str = "screen") -> dict[str, Any]:
        workdir = self.work_dir / f"trial-{n:04d}"
        restart = rules.restart_class(self.lane, worker.last_class if worker else None, params)
        return {"schema": TRIAL_SCHEMA, "run": self.name, "lane": self.lane.name, "trial": n, "arm": arm,
                "source": source, "priors": list(priors), "phase": "search", "status": "running",
                "parameters": dict(params), "launch": self.space.launch(params, run_dir=workdir).describe(),
                "worker": worker.name if worker else None, "restart": restart,
                "compile_class": rules.compile_class(self.lane, params), "fidelity": fidelity,
                "objective": {"metric": self.lane.metric, "goal": self.lane.goal},
                "conditions": {"driver": DRIVER, "node": self.o.node, "fake": self.o.fake,
                               "space": {"path": self.space.path, "sha256": self.space.sha256},
                               "night_sha256": self.night.sha256, "strategy": self.o.strategy,
                               "runner": worker.runner.describe() if worker else None, "versions": self.versions,
                               "host": host_brief(), "slot": self.o.slot},
                "measurement": None, "observation": None, "feasible": None, "early_stop": None, "reason": None,
                "started": now(), "ended": None, "timestamp": None, "git_sha": self.git.get("sha"),
                "ledger_seq": None}

    def run_trial(self, trial: dict[str, Any], worker: WorkerSlot, threshold: float | None) -> runners.Measurement:
        """In a worker thread: the runner's one spawn or process for this trial."""
        workdir = self.work_dir / f"trial-{trial['trial']:04d}"
        launch = self.space.launch(trial["parameters"], run_dir=workdir)
        kwargs: dict[str, Any] = {}
        after = rules.percentile_after(self.lane)
        if self.lane.name == "training" and after is not None:
            def progress(steps: list[dict[str, Any]]) -> bool:
                with self.lock:
                    done = sum(1 for t in self.state.trials if t["status"] == "completed")
                    if done < after:
                        return False
                    value = statistics.median(float(s["samples_per_s"]) for s in steps)
                    self.strategy.progress(trial["trial"], len(steps), {self.lane.metric: (value, None)})
                    return self.strategy.should_stop(trial["trial"])
            kwargs["progress"] = progress
        if not worker.started and worker.stagger_s:
            time.sleep(worker.stagger_s)
        worker.started = True
        log(f"trial {trial['trial']} on {worker.name} ({trial['restart']}, {trial['fidelity']}): "
            f"{trial['parameters']}")
        try:
            return worker.runner.run(trial=trial["trial"], launch=launch, fidelity=trial["fidelity"],
                                     workdir=workdir, restart=trial["restart"], stop_below=threshold, **kwargs)
        except Exception as exc:   # a runner bug or a host error ends this trial, not the run
            return runners.Measurement("fail", reason=f"{type(exc).__name__}: {exc}"[:500])

    def finish_trial(self, trial: dict[str, Any], m: runners.Measurement, worker: WorkerSlot) -> None:
        trial["measurement"] = m.as_dict()
        trial["ended"] = trial["timestamp"] = now()
        base = self.baseline_trial(self.state)
        base_samples = ((base or {}).get("measurement") or {}).get("samples", {}).get(rules.sample_key(self.lane))
        base_sd = statistics.stdev(base_samples) if base_samples and len(base_samples) > 1 else None
        early_rule = None
        if m.outcome == "early_stopped" and "strategy" not in (m.reason or ""):
            early_rule = "after_first_request" if self.lane.name == "inference" else "after_steps"
        if m.outcome == "ok" or early_rule:
            obs = rules.observation(self.lane, m.metrics, m.samples, early=bool(early_rule), baseline_sd=base_sd)
            if self.lane.metric not in obs:
                trial["status"], trial["reason"] = "failed", f"the runner reported no {self.lane.metric}"
            else:
                trial["status"], trial["observation"] = "completed", {k: list(v) for k, v in obs.items()}
                trial["feasible"] = rules.feasible(self.lane, {k: v[0] for k, v in obs.items()})
                trial["early_stop"], trial["reason"] = early_rule, m.reason
        elif m.outcome == "early_stopped":
            obs = rules.observation(self.lane, m.metrics, m.samples)
            trial["status"], trial["observation"] = "early_stopped", {k: list(v) for k, v in obs.items()}
            trial["early_stop"], trial["reason"] = "ax_percentile", m.reason
        else:
            trial["status"], trial["reason"] = "failed", m.reason
        worker.last_class = trial["compile_class"] if trial["status"] != "failed" or m.seconds else worker.last_class
        worker.failures = worker.failures + 1 if trial["status"] == "failed" else 0
        if worker.failures >= rules.max_failures(self.lane):
            worker.done = "failures"
        self.ledger_row(trial)
        write_json(trial_path(self.run_dir, trial["trial"]), trial)
        with self.lock:
            self.tell(trial)
        self.commit([self.run_dir / "run.json", trial_path(self.run_dir, trial["trial"])],
                    f"Record {self.name} trial {trial['trial']}")
        value = (trial["observation"] or {}).get(self.lane.metric, [None])[0]
        log(f"trial {trial['trial']} {trial['status']}: {self.lane.metric} {value}"
            + (f" ({trial['reason']})" if trial["reason"] else ""))

    def ledger_row(self, t: dict[str, Any]) -> None:
        lane = self.lane.name
        metrics = {f"optimize.{lane}.{k.replace('.', '_')}": v[0] for k, v in (t["observation"] or {}).items()
                   if ledger.finite_number(v[0]) is not None}
        try:
            t["ledger_seq"] = record(
                "optimize.trial", "ok" if t["status"] == "completed" else t["status"], metrics,
                {"run_id": f"{self.name}/trial-{t['trial']:04d}", "slot": self.o.slot}, node=self.o.node,
                conditions={"run": self.name, "trial": t["trial"], "arm": t["arm"], "source": t["source"],
                            "parameters": t["parameters"], "worker": t["worker"], "restart": t["restart"],
                            "fidelity": t["fidelity"], "fake": self.o.fake, "early_stop": t["early_stop"],
                            "feasible": t["feasible"], "space_sha256": self.space.sha256, "git_sha": t["git_sha"],
                            "trial_file": shown(trial_path(self.run_dir, t["trial"]))},
                low_confidence=True, source=DRIVER, note=t["reason"])
        except Exception as exc:   # the trial file is the record; a refused ledger row is said, not fatal
            t["ledger_error"] = f"{type(exc).__name__}: {exc}"[:300]

    def commit(self, paths: Sequence[Path], message: str) -> None:
        if self.o.commit:
            error = git_commit(paths, message)
            if error:
                self.commit_errors.append(error)
                log(f"commit failed: {error}")

    # budgets

    def est_seconds(self, restart: str, fidelity: str) -> float:
        own = [t["measurement"]["seconds"] for t in self.state.trials
               if t["status"] in FINISHED and t.get("measurement") and t["restart"] == restart]
        if own:
            return statistics.median(own)
        if self.o.fake:
            return 0.0
        return rules.est_seconds(self.lane, fidelity, restart) or 0.0

    def halt(self, started: float, seconds: float) -> str | None:
        if stop.stopped():
            return "stopped"
        elapsed = time.monotonic() - started
        if self.o.minutes is not None and elapsed + seconds > self.o.minutes * 60:
            return "minutes"
        run_minutes = self.o.run_minutes
        if run_minutes is not None and self.state.minutes_before() * 60 + elapsed + seconds > run_minutes * 60:
            return "run_minutes"
        return None

    # the next trial for a free worker

    def next_trial(self, worker: WorkerSlot, started: float) -> dict[str, Any] | str:
        """A new trial file for this worker, or the reason it gets none."""
        n = len(self.state.trials)
        if self.o.budget is not None and n >= self.o.budget:
            return "budget"
        attached = {t["arm"] for t in self.state.trials}
        arm = next((a for a in self.arms if a["name"] not in attached), None)
        restart = rules.restart_class(self.lane, worker.last_class, arm["parameters"]) if arm else "cold"
        why = self.halt(started, self.est_seconds(restart, "screen"))
        if why:
            return why
        with self.lock:
            if arm is not None:
                index = self.strategy.attach(arm["parameters"], baseline=arm["attach_as"] == "baseline")
                return self.new_trial(index, arm["parameters"], arm=arm["name"],
                                      source="baseline" if arm["attach_as"] == "baseline" else "prior",
                                      priors=arm["priors"], worker=worker)
            invalid = 0
            while True:
                fixed = rules.fixed_for_ask(self.lane, worker.asks, worker.last_class)
                worker.asks += 1
                got = self.strategy.next(fixed)
                if got is None and fixed:
                    got = self.strategy.next(None)
                if got is None:
                    return "exhausted"
                index, params = got
                problems = self.space.problems_with(params)
                if not problems:
                    break
                bad = self.new_trial(index, params, arm=f"{index}_0", source="model", worker=None)
                bad.update(status="invalid", reason="; ".join(problems), ended=now(), timestamp=now())
                self.state.trials.append(bad)
                write_json(trial_path(self.run_dir, index), bad)
                self.strategy.fail(index)
                invalid += 1
                if invalid >= MAX_INVALID_IN_A_ROW or (self.o.budget is not None
                                                      and len(self.state.trials) >= self.o.budget):
                    return "budget" if invalid < MAX_INVALID_IN_A_ROW else "exhausted"
            fidelity = "screen"
            if self.lane.name == "inference" and "finalist" in self.lane.fidelity:
                done = [t for t in self.state.trials if t["status"] == "completed"]
                predicted = self.strategy.predict([*(t["parameters"] for t in done), params]) if done else None
                if predicted:
                    values = [p.get(self.lane.metric, {}).get("mean") for p in predicted]
                    if None not in values and rules.is_finalist(self.lane, values[-1], values[:-1]):
                        fidelity = "finalist"
            return self.new_trial(index, params, arm=f"{index}_0", source="model", worker=worker, fidelity=fidelity)

    # the search

    def search(self, state: RunState) -> str:
        self.state = state
        self.search_error: str | None = None
        started = time.monotonic()
        rerun = [t for t in state.trials if t["status"] == "running"]
        pending: dict[cf.Future, tuple[dict[str, Any], WorkerSlot]] = {}
        reason: str | None = None
        with cf.ThreadPoolExecutor(max_workers=len(self.workers)) as pool:
            def submit(trial: dict[str, Any], worker: WorkerSlot) -> None:
                trial["worker"] = worker.name
                write_json(trial_path(self.run_dir, trial["trial"]), trial)
                base = self.baseline_trial(state)
                samples = ((base or {}).get("measurement") or {}).get("samples", {}).get(rules.sample_key(self.lane))
                threshold = rules.early_threshold(self.lane, samples) if trial["source"] != "baseline" else None
                pending[pool.submit(self.run_trial, trial, worker, threshold)] = (trial, worker)

            try:
                while True:
                    busy = {w.name for _, w in pending.values()}
                    for w in self.workers:
                        if w.name in busy or w.done or reason:
                            continue
                        if rerun:
                            submit(rerun.pop(0), w)
                            continue
                        try:
                            got = self.next_trial(w, started)
                        except Exception as exc:   # a strategy failure: the trials in flight still finish
                            self.search_error = f"{type(exc).__name__}: {exc}"[:500]
                            got = "error"
                        if isinstance(got, str):
                            if got in ("stopped", "minutes", "run_minutes", "exhausted", "budget", "error"):
                                reason = reason or got
                            w.done = got
                            continue
                        state.trials.append(got)
                        submit(got, w)
                    if not pending:
                        break
                    finished, _ = cf.wait(list(pending), return_when=cf.FIRST_COMPLETED)
                    for f in finished:
                        trial, worker = pending.pop(f)
                        self.finish_trial(trial, f.result(), worker)
                    if all(w.done == "failures" for w in self.workers):
                        reason = "failures"
            except KeyboardInterrupt:
                reason = "interrupted"
                raise
        return reason or "budget"

    # the confirmation

    def best(self) -> Mapping[str, Any] | None:
        return rules.best_observed(self.lane, self.state.trials)

    def confirm(self, state: RunState, started: float) -> dict[str, Any]:
        self.state = state
        best, base = self.best(), self.baseline_trial(state)
        order = rules.confirmation_order(self.lane)
        if best is None or base is None or not order:
            return {"ran": False, "reason": "no feasible best arm, no baseline, or no confirmation in the lane"}
        if best["trial"] == base["trial"]:
            return {"ran": False, "reason": "the baseline is the best arm; nothing to confirm"}
        worker = self.workers[0]
        minutes = self.o.confirm_minutes or self.lane.confirmation.get("minutes")
        arms = {"best": best, "baseline": base}
        for n in range(len(state.confirms), len(order)):
            elapsed = time.monotonic() - started
            est = self.est_seconds("warm", "confirm")
            if stop.stopped() or (minutes is not None and elapsed + est > float(minutes) * 60):
                return {"ran": True, "complete": False, "reason": "stopped or out of confirmation minutes"}
            source = arms[order[n]]
            item = {"schema": CONFIRM_SCHEMA, "run": self.name, "spawn": n, "arm": order[n],
                    "trial": source["trial"], "parameters": source["parameters"], "worker": worker.name,
                    "fidelity": "confirm" if "confirm" in self.lane.fidelity else "screen", "started": now()}
            workdir = self.work_dir / f"confirm-{n:02d}"
            m = worker.runner.run(trial=10000 + n, launch=self.space.launch(source["parameters"], run_dir=workdir),
                                  fidelity=item["fidelity"], workdir=workdir, restart="warm")
            item.update(outcome=m.outcome, reason=m.reason, measurement=m.as_dict(), ended=now(),
                        value=m.metrics.get(self.lane.metric) if m.outcome == "ok" else None,
                        git_sha=self.git.get("sha"))
            state.confirms.append(item)
            write_json(confirm_path(self.run_dir, n), item)
            self.commit([confirm_path(self.run_dir, n)], f"Record {self.name} confirmation spawn {n}")
            log(f"confirmation {n} ({order[n]}): {m.outcome} {self.lane.metric} {item['value']}")
        return {"ran": True, "complete": True}

    # the session

    def run(self, *, search: bool = True) -> dict[str, Any]:
        started = time.monotonic()
        state = RunState.load(self.run_dir)
        if state.manifest is None:
            state.manifest = self.manifest()
            write_json(self.run_dir / "run.json", state.manifest)
        else:
            self.check_manifest(state.manifest)
        self.state = state
        resumed = state.counts()
        error = None
        confirmation: dict[str, Any] = {"ran": False, "reason": "not asked"}
        try:
            self.replay(state)
            reason = self.search(state) if search else "confirm_only"
            error = getattr(self, "search_error", None)
            if self.o.confirm and reason in ("budget", "exhausted", "confirm_only"):
                confirmation = self.confirm(state, started)
                if confirmation.get("complete"):
                    reason = "confirmed" if search else reason
        except (RunError, SpaceError) as exc:
            reason, error = "error", str(exc)
        except KeyboardInterrupt:
            reason = "interrupted"
        except Exception as exc:   # an Ax or runner failure: the files on disk stay whole
            reason, error = "error", f"{type(exc).__name__}: {exc}"
        minutes = (time.monotonic() - started) / 60
        state.manifest["sessions"].append({"started": now(), "minutes": round(minutes, 3), "stop": reason,
                                           "git_sha": self.git.get("sha")})
        write_json(self.run_dir / "run.json", state.manifest)
        summary = None
        if any(t["status"] == "completed" for t in state.trials):
            try:
                summary = summarize(self.run_dir, strategy=self.strategy if self.o.strategy == "ax" else None)
                self.commit([self.run_dir / "run.json", self.run_dir / "summary.json", self.run_dir / "summary.md"],
                            f"Summarize {self.name} after {len(state.trials)} trials")
            except Exception as exc:
                log(f"no summary: {type(exc).__name__}: {exc}")
        ok = reason in STOP_OK or reason == "confirm_only"
        counts = state.counts()
        try:
            record("optimize.run", "ok" if ok else reason,
                   {"optimize.trials_total": counts["total"], "optimize.minutes_used": round(minutes, 3)},
                   {"run_id": self.name, "slot": self.o.slot}, node=self.o.node,
                   conditions={"stop_reason": reason, "lane": self.lane.name, "counts": counts,
                               "resumed_with": resumed, "fake": self.o.fake}, source=DRIVER, note=error)
        except Exception as exc:
            log(f"no ledger row for the session: {exc}")
        answer = {"ok": ok, "outcome": "ok" if ok else reason, "stop_reason": reason, "run": self.name,
                  "run_dir": str(self.run_dir), "trials": counts, "resumed_with": resumed,
                  "minutes_used": round(minutes, 3), "confirmation": confirmation,
                  "best": (summary or {}).get("best"), "verdict": (summary or {}).get("confirmation")}
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
    try:
        top = Path(subprocess.run(["git", "-C", str(existing[0].parent), "rev-parse", "--show-toplevel"],
                                  capture_output=True, text=True, timeout=30, check=True).stdout.strip()).resolve()
        rel = [p.relative_to(top).as_posix() for p in existing]
        subprocess.run(["git", "-C", str(top), "add", "--", *rel], capture_output=True, timeout=60, check=True)
        if subprocess.run(["git", "-C", str(top), "diff", "--cached", "--quiet", "--", *rel], timeout=60).returncode == 0:
            return None
        subprocess.run(["git", "-C", str(top), "commit", "-q", "-m", message, "--", *rel], capture_output=True,
                       text=True, timeout=60, check=True)
    except subprocess.CalledProcessError as exc:
        return f"git: {(exc.stderr or exc.stdout or b'')!s}"[:300]
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return f"{type(exc).__name__}: {exc}"[:300]
    return None


# --- the summary --------------------------------------------------------------------------------------------------------

def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def summarize(run_dir: Path, *, strategy: Any = None, config: str | Path | None = None) -> dict[str, Any]:
    """summary.json and summary.md: the best feasible arm, the posterior at every completed trial (when Ax is the
    strategy and its model predicts), one table row per trial, and the confirmation verdict."""
    run_dir = Path(run_dir)
    state = RunState.load(run_dir)
    if state.manifest is None:
        raise RunError(f"{run_dir} holds no run.json")
    m = state.manifest
    night = load_night(config or REPO / m["night"]["path"])
    lane = lane_of(night, m["lane"])
    space = load_space(REPO / m["space"]["path"]) if not Path(m["space"]["path"]).is_absolute() \
        else load_space(m["space"]["path"])
    best = rules.best_observed(lane, state.trials)
    base = next((t for t in state.trials if t["source"] == "baseline" and t["status"] == "completed"), None)
    done = [t for t in state.trials if t["status"] == "completed"]
    posterior: dict[str, Any] = {"available": False, "reason": "the strategy is not Ax, or Ax is not installed"}
    if m["strategy"] == "ax" and ax_installed() and done:
        try:
            if strategy is None:
                strategy = make_strategy("ax", space, lane, generation_strategy=night.generation_strategy,
                                         seed=m["seed"])
                for t in state.trials:
                    strategy.attach(t["parameters"], baseline=t["source"] == "baseline")
                    Driver.tell(type("T", (), {"strategy": strategy})(), t)
            predicted = strategy.predict([t["parameters"] for t in done])
            posterior = ({"available": True, "arms": [{"trial": t["trial"], "predicted": p}
                                                      for t, p in zip(done, predicted)]}
                         if predicted else {"available": False, "reason": "the model does not predict yet"})
        except Exception as exc:
            posterior = {"available": False, "reason": f"{type(exc).__name__}: {exc}"[:400]}
    by_trial = {a["trial"]: a["predicted"] for a in posterior.get("arms", [])}
    change = None
    if best and base:
        b0, b1 = base["observation"][lane.metric][0], best["observation"][lane.metric][0]
        change = {"baseline": b0, "best": b1, "relative": (b1 - b0) / b0 if b0 else None}
    runs = {arm: [c["value"] for c in state.confirms if c["arm"] == arm and c.get("value") is not None]
            for arm in ("best", "baseline")}
    confirmation = rules.confirm_verdict(lane, runs["baseline"], runs["best"]) if state.confirms else None
    owed = []
    if lane.raw.get("correctness_gate"):
        owed.append(f"R21 correctness gate: {lane.raw['correctness_gate']}")
    summary = {"schema": SUMMARY_SCHEMA, "run": m["name"], "written": now(), "driver": DRIVER, "lane": lane.name,
               "objective": {"metric": lane.metric, "goal": lane.goal},
               "constraints": [c.text() for c in lane.constraints], "node": m["node"], "fake": m["fake"],
               "strategy": m["strategy"], "trials": state.counts(),
               "best": {"trial": best["trial"], "arm": best["arm"], "parameters": best["parameters"],
                        "observation": best["observation"], "predicted": by_trial.get(best["trial"]),
                        "launch": best["launch"]} if best else None,
               "baseline": {"trial": base["trial"], "observation": base["observation"]} if base else None,
               "change_vs_baseline": change, "posterior": posterior,
               "restarts": {k: sum(1 for t in state.trials if t["restart"] == k and t["status"] != "invalid")
                            for k in ("cold", "warm", "process")},
               "confirmation": confirmation, "still_owed": owed, "git": git_facts(), "versions": versions()}
    write_json(run_dir / "summary.json", summary)
    (run_dir / "summary.md").write_text(markdown(summary, state.trials, lane, space, by_trial), encoding="utf-8")
    return summary


def markdown(summary: Mapping[str, Any], trials: Sequence[Mapping[str, Any]], lane: Lane, space: SearchSpace,
             predicted: Mapping[int, Mapping[str, Any]]) -> str:
    obj = lane.metric
    c = summary["trials"]
    lines = [f"# Settings run {summary['run']} ({lane.name}{', fake' if summary['fake'] else ''})", "",
             f"Objective: {lane.goal} {obj}; constraints: {', '.join(summary['constraints']) or 'none'}. "
             f"Strategy {summary['strategy']}, node {summary['node']}. Trials: {c['completed']} completed, "
             f"{c['failed']} failed, {c['early_stopped']} early-stopped, {c['invalid']} invalid, "
             f"{c['running']} in flight. Restarts: {summary['restarts']}.", ""]
    best = summary["best"]
    if best:
        settings = ", ".join(f"{k}={_fmt(v)}" for k, v in best["parameters"].items()
                             if not same(v, space.baseline.get(k))) or "the baseline"
        lines += [f"Best feasible arm: trial {best['trial']} ({best['arm']}): {settings}; "
                  f"{obj} {_fmt(best['observation'][obj][0])}.", ""]
    if summary["change_vs_baseline"] and summary["change_vs_baseline"]["relative"] is not None:
        ch = summary["change_vs_baseline"]
        lines += [f"Baseline {_fmt(ch['baseline'])}; best {_fmt(ch['best'])} ({ch['relative'] * 100:+.1f} %).", ""]
    post = summary["posterior"]
    lines += [f"Posterior: {'available' if post.get('available') else post.get('reason')}.", ""]
    v = summary["confirmation"]
    if v:
        lines += [f"Confirmation: {v['label']} (runs {v['runs_per_arm']}, one-sided p {_fmt(v['p_one_sided'])}"
                  f"{', low confidence' if v['low_confidence'] else ''}).", ""]
    for item in summary["still_owed"]:
        lines += [f"Still owed: {item}", ""]
    metrics = lane.metrics
    head = ["Trial", "Arm", "Status", "Worker", "Restart", "Fidelity", "Changed settings",
            *metrics, f"{obj} sem", "posterior mean", "posterior sem"]
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for t in trials:
        o = t.get("observation") or {}
        p = (predicted.get(t["trial"]) or {}).get(obj) or {}
        changed = ", ".join(f"{k}={_fmt(v)}" for k, v in t["parameters"].items()
                            if not same(v, space.baseline.get(k))) or "baseline"
        row = [str(t["trial"]), t["arm"], t["status"], _fmt(t["worker"]), _fmt(t["restart"]), t["fidelity"], changed,
               *[_fmt((o.get(mm) or [None])[0]) for mm in metrics], _fmt((o.get(obj) or [None, None])[1]),
               _fmt(p.get("mean")), _fmt(p.get("sem"))]
        lines.append("| " + " | ".join(row) + " |")
    bad = [t for t in trials if t["status"] in ("failed", "invalid")]
    if bad:
        lines += ["", "Failed or invalid trials:", ""] + [f"- trial {t['trial']}: {t['reason']}" for t in bad]
    return "\n".join(lines) + "\n"


# --- install ------------------------------------------------------------------------------------------------------------

def install(*, dry_run: bool = False) -> dict[str, Any]:
    """pip install -r lab/requirements-optimize.txt with the installed torch held where it is (a constraint)."""
    argv = [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)]
    if sys.prefix == sys.base_prefix:
        argv.append("--user")
    torch = _version("torch")
    if torch:
        folder = Path(tempfile.mkdtemp(prefix="optimize-install-"))
        (folder / "torch-constraint.txt").write_text(f"torch=={torch}\n", encoding="utf-8")
        argv += ["-c", str(folder / "torch-constraint.txt")]
    if dry_run:
        return {"ok": True, "outcome": "dry_run", "argv": argv, "torch_held_at": torch, "pins": pins()}
    code = subprocess.run(argv).returncode
    found = {name: _version(name) for name in pins()}
    differ = {k: {"pinned": v, "installed": found.get(k)} for k, v in pins().items() if found.get(k) != v}
    ok = code == 0 and not differ
    return {"ok": ok, "outcome": "ok" if ok else "fail", "argv": argv, "exit_code": code, "installed": found,
            **({} if ok else {"error": f"pip exit {code}; differ from the pins: {json.dumps(differ)}"})}


# --- the command line ---------------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m lab optimize", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "confirm"):
        r = sub.add_parser(name)
        r.add_argument("--lane", choices=("inference", "training"), default="inference")
        r.add_argument("--config", default=str(NIGHT_CONFIG), help="the night configuration")
        r.add_argument("--space", default=None, help="default: the lane's search_space")
        r.add_argument("--run-dir", default=None)
        r.add_argument("--work-dir", default=None)
        r.add_argument("--fake", action="store_true", help="the offline fakes; node local, lab/state/optimize")
        r.add_argument("--workers", type=int, default=None, help="fake workers (default: the lane's enabled ones)")
        r.add_argument("--budget", type=int, default=None, help="trials the run holds when it stops")
        r.add_argument("--minutes", type=float, default=None, help="this call's wall clock (the GPU slot)")
        r.add_argument("--run-minutes", type=float, default=None, help="default: the lane's budget.minutes")
        r.add_argument("--confirm-minutes", type=float, default=None)
        r.add_argument("--no-confirm", action="store_true")
        r.add_argument("--strategy", choices=STRATEGIES, default="ax")
        r.add_argument("--seed", type=int, default=None, help="default: initialization_random_seed")
        r.add_argument("--node", default=None)
        r.add_argument("--slot", default=None, help="the GPU Queue row, for the ledger refs")
        r.add_argument("--commit", action="store_true")
        r.add_argument("--dry-run", action="store_true")
        r.add_argument("--bench", default="", help="lab/bench.py run arguments for the real inference runner")
        r.add_argument("--train-cmd", default=None, help="trainer template: {settings} {out} {steps} {trial}")
        r.add_argument("--noise-seed", type=int, default=0)
    s = sub.add_parser("summary")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--config", default=None)
    i = sub.add_parser("install")
    i.add_argument("--dry-run", action="store_true")
    return p


def lane_workers(night: Night, lane: Lane) -> list[str]:
    return [w for w in lane.worker_names if night.worker(w).available]


def make_workers(night: Night, lane: Lane, args: argparse.Namespace, work_dir: Path, node: str) -> list[WorkerSlot]:
    fidelity = rules.bench_fidelities(lane)
    timeouts = rules.ready_timeouts(lane)
    screen = lane.fidelity.get("screen") or {}
    if args.fake:
        count = args.workers or len(lane_workers(night, lane)) or 1
        out = []
        for i in range(count):
            name = f"fake-{i}"
            if lane.name == "inference":
                runner = runners.fake_inference_runner(work_dir / name, fidelity, noise_seed=args.noise_seed)
            else:
                runner = runners.fake_training_runner(steps=int(screen.get("steps") or 20),
                                                      exclude_first_steps=int(screen.get("exclude_first_steps") or 0),
                                                      noise_seed=args.noise_seed,
                                                      early_after=rules.early_after_steps(lane))
            out.append(WorkerSlot(name, runner))
        return out
    local = [w for w in lane_workers(night, lane) if night.worker(w).runner == "local"]
    if node not in local and not (local and node in ("pod-b", "pod-d") and "pod-d" in local):
        raise RunError(f"{node} is not a local worker of the {lane.name} lane ({', '.join(local) or 'none'})")
    if lane.name == "inference":
        runner = runners.real_inference_runner(shlex.split(args.bench or ""), fidelity, node=node,
                                               ready_timeout_s=timeouts)
    else:
        if not args.train_cmd:
            raise RunError("a real training run needs --train-cmd with {settings} {out} {steps}")
        runner = runners.TrainingRunner(shlex.split(args.train_cmd), steps=int(screen.get("steps") or 20),
                                        exclude_first_steps=int(screen.get("exclude_first_steps") or 0), fake=False,
                                        timeout_s=timeouts.get("process", 3600.0) + 3600.0,
                                        early_after=rules.early_after_steps(lane))
    return [WorkerSlot(node, runner, stagger_s=night.worker(node).stagger_s if node in local else 0.0)]


def dry_plan(night: Night, lane: Lane, space: SearchSpace, args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    state = RunState.load(run_dir) if run_dir.exists() else RunState(run_dir, None, [], [])
    arms = initial_arms(lane, space)
    workers = args.workers or len(lane_workers(night, lane)) or 1
    minutes = args.minutes or args.run_minutes or float(lane.budget.get("minutes") or 60)
    budget = args.budget
    layout = rules.plan_minutes(lane, workers=workers, minutes=minutes, budget=budget, initial=arms,
                                done=len(state.trials))
    return {"lane": lane.name, "task": lane.task, "config": night.path, "space": space.describe() | {"raw": None},
            "run_dir": str(run_dir), "resumes": state.manifest is not None, "trials_on_disk": state.counts(),
            "objective": {"metric": lane.metric, "goal": lane.goal},
            "constraints": [c.text() for c in lane.constraints], "constraint_notes": list(lane.constraint_notes),
            "workers": {"count": workers, "lane": list(lane.worker_names), "available": lane_workers(night, lane)},
            "initial_arms": [{"name": a["name"], "attach_as": a["attach_as"], "restart": a["restart"],
                              "priors": a["priors"]} for a in arms],
            "grouping": dict(lane.grouping), "early_stop": dict(lane.early_stop),
            "minutes": minutes, "budget": budget, "fit": layout["fit"], "restarts": layout["restarts"],
            "trials": layout["trials"], "confirmation": {"order": rules.confirmation_order(lane),
                                                         "minutes": lane.confirmation.get("minutes")},
            "strategy": args.strategy, "ax_installed": ax_installed(), "pins": pins()}


def show_plan(p: Mapping[str, Any]) -> None:
    log(f"plan: lane {p['lane']} ({p['task']}), {p['space']['name']} ({len(p['space']['parameters'])} parameters), "
        f"{p['objective']['goal']} {p['objective']['metric']} subject to {', '.join(p['constraints']) or 'nothing'}")
    log(f"  workers: {p['workers']['count']} (available: {', '.join(p['workers']['available']) or 'none'}); "
        f"run dir {p['run_dir']} ({'resume' if p['resumes'] else 'new'})")
    for a in p["initial_arms"]:
        log(f"  initial arm {a['name']}: {a['attach_as']}, {a['restart']} start, priors {', '.join(a['priors']) or '-'}")
    cap = "" if p["budget"] is None else f" and {p['budget']} trials"
    log(f"  within {p['minutes']:g} minutes{cap}: {p['fit']} trials fit, restarts {p['restarts']}")
    for t in p["trials"]:
        log(f"    trial {t['trial']} {t['arm']} on worker {t['worker']}: {t['restart']} start at minute "
            f"{t['start_min']}, about {t['est_s']:g} s")
    log(f"  then confirmation: {', '.join(p['confirmation']['order'])} ({p['confirmation']['minutes']} minutes)")
    log(f"  strategy {p['strategy']}; Ax installed here: {p['ax_installed']}; pins {p['pins']}")


def run_command(args: argparse.Namespace) -> dict[str, Any]:
    night = load_night(args.config)
    lane = lane_of(night, args.lane)
    space = load_space(args.space or REPO / lane.search_space, kind=lane.name)
    node = FAKE_NODE if args.fake and args.node is None else args.node
    date = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    run_dir = Path(args.run_dir) if args.run_dir else (
        STATE_DIR / f"ax-{lane.name}-fake-{date}" if args.fake else DATA_DIR / f"ax-{lane.name}-{date}")
    work_dir = Path(args.work_dir) if args.work_dir else STATE_DIR / run_dir.name / "work"
    if args.dry_run:
        p = dry_plan(night, lane, space, args, run_dir)
        show_plan(p)
        return {"ok": True, "outcome": "dry_run", "plan": p}
    gate = node_gate(node, fake=args.fake)
    if gate:
        return {"ok": False, "outcome": "refused", "error": f"node gate: {gate}"}
    if not args.fake and args.minutes is None:
        return {"ok": False, "outcome": "refused", "error": "a real run needs --minutes, its GPU slot's wall clock"}
    if args.strategy == "ax" and not ax_installed():
        return {"ok": False, "outcome": "missing_dependency",
                "error": "Ax is not installed here: python -m lab optimize install (or --strategy stub)"}
    seed = args.seed if args.seed is not None else night.generation_strategy.get("initialization_random_seed")
    options = Options(budget=args.budget, minutes=args.minutes,
                      run_minutes=args.run_minutes if args.run_minutes is not None else lane.budget.get("minutes"),
                      strategy=args.strategy, seed=seed, node=str(node), fake=args.fake, commit=args.commit,
                      slot=args.slot, confirm=not args.no_confirm, confirm_minutes=args.confirm_minutes)
    workers = make_workers(night, lane, args, work_dir, str(node))
    return Driver(night, lane, space, workers, run_dir, work_dir, options).run(search=args.cmd == "run")


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Optimize engine or trainer settings to the night configuration (R45)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("run", "confirm", "summary", "install", "-h", "--help"):
        argv = ["run", *argv]
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "install":
            return install(dry_run=args.dry_run)
        if args.cmd == "summary":
            s = summarize(Path(args.run_dir), config=args.config)
            return {"ok": True, "outcome": "ok", "run": s["run"], "best": s["best"], "confirmation": s["confirmation"],
                    "summary_path": str(Path(args.run_dir) / "summary.json")}
        return run_command(args)
    except SpaceError as exc:
        return {"ok": False, "outcome": "bad_space", "error": str(exc), "problems": exc.problems}
    except RunError as exc:
        return {"ok": False, "outcome": "refused", "error": str(exc)}


if __name__ == "__main__":
    answer = main()
    print(json.dumps(answer, default=str))
    sys.exit(0 if answer.get("ok") else 1)
