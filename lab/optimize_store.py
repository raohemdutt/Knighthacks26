"""The settings optimizer's run directory, summary and install (lab/optimize_settings.py is the driver).

    from lab.optimize_store import RunState, summarize, install
    state = RunState.load(run_dir)            # run.json, trials/trial-NNNN.json, confirm/confirm-NN.json, checked whole
    summarize(run_dir)                        # summary.json and summary.md
    install(dry_run=True)                     # the pip command for lab/requirements-optimize.txt, torch held in place

The file formats are described in lab/optimize_settings.py's docstring. summarize() writes the best feasible arm
(lab/optimize_plan.py best_observed), the posterior at every completed trial when the run's strategy is Ax and its
model predicts (the trials are replayed into a fresh Ax client when no live one is passed), the change against the
baseline, the restarts by class, the confirmation verdict and what is still owed (the R21 correctness gate), and a
Markdown table with one row per trial.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import socket
import subprocess
import sys
import tempfile
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lab import optimize_plan as rules
from lab.machine import ledger
from lab.optimize_ax import ax_installed, make_strategy
from lab.optimize_space import REPO, Lane, SearchSpace, lane_of, load_night, load_space, same

DRIVER = "lab/optimize_settings.py 2"
RUN_SCHEMA = "lab.optimize.run/2"
TRIAL_SCHEMA = "lab.optimize.trial/2"
CONFIRM_SCHEMA = "lab.optimize.confirm/1"
SUMMARY_SCHEMA = "lab.optimize.summary/2"
REQUIREMENTS = REPO / "lab" / "requirements-optimize.txt"
FINISHED = ("completed", "failed", "early_stopped", "invalid")
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
                    tell(strategy, t)
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


def tell(strategy: Any, t: Mapping[str, Any]) -> None:
    """Hand one finished trial file to a strategy: completed, early-stopped, or failed (invalid counts as failed)."""
    if t["status"] == "completed":
        strategy.complete(t["trial"], {k: tuple(v) for k, v in t["observation"].items()})
    elif t["status"] == "early_stopped":
        strategy.early_stopped(t["trial"], {k: tuple(v) for k, v in (t["observation"] or {}).items()})
    elif t["status"] in ("failed", "invalid"):
        strategy.fail(t["trial"])
