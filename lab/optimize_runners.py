"""The settings optimizer's trial runners: one server spawn per inference trial, one trainer process per training trial.

    from lab.optimize_runners import InferenceRunner, TrainingRunner, fake_inference_runner, fake_training_runner
    m = runner.run(trial=3, launch=space.launch(params, run_dir=w), fidelity="screen", workdir=w,
                   restart="cold", stop_below=1.4)

Both runners take a trial's Launch (lab/optimize_space.py: args, env, unset_env, settings) and answer a Measurement:
outcome ok, fail or early_stopped, the trial's metrics, the per-request or per-step samples behind them, the
conditions (command, environment, session or log paths) and the seconds it took.

Inference (InferenceRunner). One lab/bench.py session per trial, so one server spawn per configuration: the launch
template (the fake server's, lab/bench.py's default for the engine, or the --serve-cmd given) followed by the trial's
mapped engine args, the trial's env over the runner's launch env, the trial's unset_env removed from the inherited
environment. The fidelity's bench arguments (the night configuration's lanes.inference.fidelity.<name>.bench_args:
batch sizes, --share-spawn, requests per stream, max tokens, warm-up, probe) set what that one spawn measures, so
every restart-free measurement of a configuration happens in its one spawn. The ready timeout is the configuration's
restart class's (cold or warm). stop_below, when given, stops the spawn after the first measured request whose decode
rate is under it (lab/bench.py's stop_measuring hook). Metrics are lab/bench.py's per-batch medians named
<metric>.b<batch>; samples["decode_tok_s.b<batch>"] are the per-request decode rates.

Training (TrainingRunner). One trainer process per trial, started from a command template ({settings} {out}
{steps} {trial} {python}) with the trial's settings written to <workdir>/settings.json and its env and unset_env
applied. The trainer contract (lab/optimize_fake.py's trainer keeps it; lab/train.py takes the same two files once
R39 wires it): <out>/steps.jsonl, one {"step", "samples_per_s", "loss", "mem_gb"} line per optimizer step flushed as
it happens, and <out>/summary.json at the end. The runner watches steps.jsonl: after early_after steps it compares
the mean samples/s over steps 2 to early_after with stop_below and stops a hopeless run. A run whose log names an
out-of-memory error is a failure with reason "out of memory". Metrics: samples_per_s (median over the steps after
exclude_first_steps), peak_mem_gb, first_step_s, last_loss; samples["samples_per_s"] are those per-step rates.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab import bench  # noqa: E402
from lab.optimize_space import Launch, shown  # noqa: E402

FAKE_NODE = "local"
OUTCOMES = ("ok", "fail", "early_stopped")
DEFAULT_TRAIN_TIMEOUT_S = 3600.0
_PLACEHOLDER = re.compile(r"\{(settings|out|steps|trial|python)\}")
_OOM = re.compile(r"out of memory|OutOfMemoryError", re.IGNORECASE)


@dataclass
class Measurement:
    outcome: str
    metrics: dict[str, float] = field(default_factory=dict)
    samples: dict[str, list[float]] = field(default_factory=dict)
    conditions: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None
    seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


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


# --- inference ------------------------------------------------------------------------------------------------------

def fidelity_overrides(bench_args: Sequence[str]) -> dict[str, Any]:
    """The lab/bench.py Config fields a fidelity's bench arguments set (only the ones it names)."""
    ns = bench.build_parser().parse_args(["run", *bench_args])
    given = set(bench_args)
    out: dict[str, Any] = {}
    if "--batches" in given:
        out["batches"] = list(ns.batches)
    if "--no-64" in given:
        out["add_64"] = False
    if "--repeats" in given:
        out["repeats"] = ns.repeats
    if "--max-tokens" in given:
        out["max_tokens"] = ns.max_tokens
    if "--requests-per-stream" in given:
        out["requests_per_stream"] = ns.requests_per_stream
    if "--warmup" in given:
        out["warmup_rounds"] = ns.warmup
    if "--no-probe" in given:
        out["probe"] = False
    if "--share-spawn" in given:
        out["share_spawn"] = True
    return out


class InferenceRunner:
    kind = "inference"

    def __init__(self, config: bench.Config, *, fidelity: Mapping[str, Sequence[str]], fake: bool,
                 host: bench.Host | None, power: Callable[[bench.Host | None, bench.Config], Any],
                 ready_timeout_s: Mapping[str, float] | None = None,
                 template: Callable[[int], list[str]] | None = None) -> None:
        self.config = config
        self.fidelity = {k: list(v) for k, v in fidelity.items()}
        for name, args in self.fidelity.items():
            fidelity_overrides(args)    # a fidelity lab/bench.py cannot parse fails here, before any trial
        self.fake = fake
        self.host = host
        self.power = power
        self.ready_timeout_s = dict(ready_timeout_s or {})
        self._template = template

    def describe(self) -> dict[str, Any]:
        c = self.config
        return {"kind": self.kind, "fake": self.fake, "harness": bench.HARNESS, "engine": c.engine,
                "checkpoint": c.checkpoint or None, "model": c.model, "node": c.node, "fidelity": self.fidelity,
                "ready_timeout_s": self.ready_timeout_s, "launch_env": dict(c.launch_env),
                "command": c.command or ("lab/optimize_fake.py serve" if self.fake else "lab/bench.py default"),
                "prompts": shown(c.prompts)}

    def template(self, trial: int) -> list[str]:
        if self._template is not None:
            return self._template(trial)
        if self.config.command:
            return list(self.config.command)
        chip = bench.host_facts(self.host).get("chip") if self.host is not None else None
        return bench.default_command(self.config.engine, chip)

    def run(self, *, trial: int, launch: Launch, fidelity: str, workdir: Path, restart: str | None = None,
            stop_below: float | None = None) -> Measurement:
        if fidelity not in self.fidelity:
            return Measurement("fail", reason=f"no {fidelity!r} fidelity (have {', '.join(self.fidelity)})")
        workdir.mkdir(parents=True, exist_ok=True)
        early: dict[str, Any] = {"stopped": False}

        def stop(r: bench.RequestResult) -> bool:
            if stop_below is None or r.error is not None or r.decode_tok_s is None:
                return False
            if r.decode_tok_s < stop_below:
                early["stopped"] = True
                return True
            return False

        overrides = fidelity_overrides(self.fidelity[fidelity])
        config = dataclasses.replace(
            self.config, command=[*self.template(trial), *launch.args],
            launch_env={**self.config.launch_env, **launch.env},
            unset_env=[n for n in launch.unset_env if n not in self.config.launch_env],
            out_dir=workdir, record=True, stop_measuring=stop,
            ready_timeout_s=self.ready_timeout_s.get(restart or "", self.config.ready_timeout_s), **overrides)
        started = time.monotonic()
        try:
            result = bench.run_session(config, host=self.host, power=self.power(self.host, config))
        except bench.BenchError as exc:
            return Measurement("fail", conditions={"command": config.command}, reason=f"lab/bench.py: {exc}",
                               seconds=round(time.monotonic() - started, 3))
        seconds = round(time.monotonic() - started, 3)
        rows = result["rows"]
        metrics: dict[str, float] = {}
        samples: dict[str, list[float]] = {}
        for entry in result["summary"]["configurations"].values():
            b = entry["batch_size"]
            for name in bench.METRIC_UNITS:
                value = entry["stats"][name]["median"]
                if value is not None:
                    metrics[f"{name}.b{b}"] = float(value)
        for row in rows:
            rates = [r["decode_tok_s"] for r in row.get("requests") or []
                     if r.get("role") == "measure" and r.get("decode_tok_s") is not None and not r.get("error")]
            samples.setdefault(f"decode_tok_s.b{row['batch_size']}", []).extend(float(x) for x in rates)
        conditions = {
            "bench": {"session": result["session"], "rows_path": result["rows_path"],
                      "runs": [{"run_id": r["run_id"], "batch_size": r["batch_size"], "outcome": r["outcome"],
                                "pid": (r.get("spawn") or {}).get("pid"),
                                "ready_s": (r.get("spawn") or {}).get("ready_s")} for r in rows]},
            "command": (rows[0].get("spawn") or {}).get("command") if rows else config.command,
            "env": dict(config.launch_env), "unset_env": list(config.unset_env), "fidelity": fidelity,
            "restart": restart, "ready_timeout_s": config.ready_timeout_s,
            "conditions": rows[0]["conditions"] if rows else None}
        if not rows:
            return Measurement("fail", metrics, samples, conditions, "lab/bench.py wrote no rows", seconds)
        failed = [r for r in rows if r["outcome"] not in ("ok", "skipped")]
        if early["stopped"]:
            return Measurement("early_stopped", metrics, samples, conditions,
                               f"first measured request under {stop_below:g} tok/s", seconds)
        if failed:
            return Measurement("fail", metrics, samples, conditions,
                               f"batch {failed[0]['batch_size']}: {failed[0]['reason']}", seconds)
        return Measurement("ok", metrics, samples, conditions, None, seconds)


def fake_inference_runner(work_dir: Path, fidelity: Mapping[str, Sequence[str]], *, noise_seed: int = 0,
                          ready_timeout_s: Mapping[str, float] | None = None) -> InferenceRunner:
    """lab/optimize_fake.py's settings-aware server, measured by lab/bench.py on lab/bench_fake.py's fake pod."""
    from lab import bench_fake, optimize_fake

    root = work_dir / "fake-pod"
    root.mkdir(parents=True, exist_ok=True)
    config = bench.Config(engine="vllm", checkpoint="/home/jovyan/models/fake-checkpoint", model="teacher",
                          node=FAKE_NODE, port=0, max_tokens=8, requests_per_stream=2, warmup_rounds=1, idle_s=0.05,
                          power_period_s=0.01, ready_timeout_s=60.0, request_timeout_s=30.0, python="python3")
    return InferenceRunner(config, fidelity=fidelity, fake=True, host=bench_fake.FakeHost(root),
                           power=lambda h, c: bench_fake.FakePower(root, period_s=c.power_period_s),
                           ready_timeout_s=ready_timeout_s,
                           template=lambda trial: optimize_fake.serve_command(root, noise_seed=noise_seed,
                                                                              trial=trial))


def real_inference_runner(bench_args: Sequence[str], fidelity: Mapping[str, Sequence[str]], *, node: str,
                          ready_timeout_s: Mapping[str, float] | None = None) -> InferenceRunner:
    """lab/bench.py on this pod, configured by its own `run` arguments."""
    args = bench.build_parser().parse_args(["run", *bench_args])
    config = bench._config(args)   # lab/bench.py's own mapping of its arguments, so the two never drift
    config.node = node
    host = bench.Host()
    return InferenceRunner(config, fidelity=fidelity, fake=False, host=host,
                           power=lambda h, c: bench.LocalPower.for_host(h, node=c.node, period_s=c.power_period_s),
                           ready_timeout_s=ready_timeout_s)


# --- training -------------------------------------------------------------------------------------------------------

def read_steps(path: Path) -> list[dict[str, Any]]:
    """The whole lines of a trainer's steps.jsonl so far (a line being written is left for the next read)."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    out = []
    for line in text.split("\n")[:-1]:
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict) and isinstance(item.get("step"), int):
            out.append(item)
    return out


class TrainingRunner:
    kind = "training"

    def __init__(self, command: Sequence[str], *, steps: int, exclude_first_steps: int = 0, fake: bool,
                 env: Mapping[str, str] | None = None, timeout_s: float = DEFAULT_TRAIN_TIMEOUT_S,
                 early_after: int = 5, poll_s: float = 0.02) -> None:
        if not command:
            raise ValueError("the training runner needs a command template")
        self.command = list(command)
        self.steps = steps
        self.exclude = exclude_first_steps
        self.fake = fake
        self.env = dict(env or {})
        self.timeout_s = timeout_s
        self.early_after = early_after
        self.poll_s = poll_s

    def describe(self) -> dict[str, Any]:
        return {"kind": self.kind, "fake": self.fake, "command": self.command, "steps": self.steps,
                "exclude_first_steps": self.exclude, "env": dict(self.env), "timeout_s": self.timeout_s,
                "early_after": self.early_after}

    def run(self, *, trial: int, launch: Launch, fidelity: str, workdir: Path, restart: str | None = None,
            stop_below: float | None = None) -> Measurement:
        workdir.mkdir(parents=True, exist_ok=True)
        out = workdir / "out"
        out.mkdir(parents=True, exist_ok=True)
        steps_path = out / "steps.jsonl"
        if steps_path.exists():
            steps_path.unlink()        # a re-run of the same trial starts its own steps
        settings_path = workdir / "settings.json"
        settings_path.write_text(json.dumps(launch.settings, indent=2), encoding="utf-8")
        values = {"settings": settings_path, "out": out, "steps": self.steps, "trial": trial, "python": sys.executable}
        argv = [_PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), a) for a in self.command] + list(launch.args)
        environ = launch.environ({**os.environ, **self.env})
        log_path = workdir / "train.log"
        started = time.monotonic()
        conditions: dict[str, Any] = {"argv": argv, "env": {**self.env, **launch.env},
                                      "unset_env": list(launch.unset_env), "settings_file": shown(settings_path),
                                      "log": shown(log_path), "fidelity": fidelity, "restart": restart}
        stopped_early = False
        code: int | None = None
        with open(log_path, "wb") as fh:
            try:
                proc = subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        env=environ, cwd=str(workdir), start_new_session=os.name == "posix")
            except OSError as exc:
                return Measurement("fail", conditions=conditions, reason=f"could not start {argv[0]}: {exc}")
            while True:
                code = proc.poll()
                if code is not None:
                    break
                if time.monotonic() - started > self.timeout_s:
                    _kill(proc)
                    code = None
                    break
                if stop_below is not None:
                    seen = read_steps(steps_path)
                    if len(seen) >= self.early_after:
                        window = [s["samples_per_s"] for s in seen[1:self.early_after]]
                        if window and sum(window) / len(window) < stop_below:
                            _kill(proc)
                            stopped_early = True
                            break
                time.sleep(self.poll_s)
        seconds = round(time.monotonic() - started, 3)
        conditions["exit_code"] = code
        steps = read_steps(steps_path)
        if stopped_early:
            window = [float(s["samples_per_s"]) for s in steps[1:self.early_after]]
            metrics = {"samples_per_s": sorted(window)[len(window) // 2]}
            return Measurement("early_stopped", metrics, {"samples_per_s": window}, conditions,
                               f"steps 2 to {self.early_after} under {stop_below:g} samples/s", seconds)
        if code is None:
            return Measurement("fail", {}, {}, conditions, f"the trainer ran past {self.timeout_s:g} s", seconds)
        if code != 0:
            tail = _tail(log_path)
            reason = "out of memory" if _OOM.search(tail) else f"the trainer exited with code {code}: {tail}"
            return Measurement("fail", {}, {}, conditions, reason, seconds)
        try:
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return Measurement("fail", {}, {}, conditions, f"no readable summary.json: {exc}", seconds)
        window = [float(s["samples_per_s"]) for s in steps[self.exclude:]] or [float(summary["samples_per_s"])]
        ordered = sorted(window)
        mid = len(ordered) // 2
        median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
        metrics = {"samples_per_s": median}
        for key in ("peak_mem_gb", "first_step_s", "last_loss"):
            if isinstance(summary.get(key), (int, float)):
                metrics[key] = float(summary[key])
        conditions["summary"] = {k: v for k, v in summary.items() if k != "settings"}
        return Measurement("ok", metrics, {"samples_per_s": window}, conditions, None, seconds)


def fake_training_runner(*, steps: int = 20, exclude_first_steps: int = 5, noise_seed: int = 0,
                         early_after: int = 5, time_scale: float = 0.001) -> TrainingRunner:
    from lab import optimize_fake

    return TrainingRunner(optimize_fake.train_command(noise_seed=noise_seed, time_scale=time_scale), steps=steps,
                          exclude_first_steps=exclude_first_steps, fake=True, timeout_s=120.0,
                          early_after=early_after)
