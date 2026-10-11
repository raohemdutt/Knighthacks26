"""The L16 retrain's Bayesian search, driven from the workstation: the Optuna study is held here and Colab runs trials.

    <a Python with optuna> lab/l16v2/bo_local.py --sessions l16v2-a100,l16v2-a100-b --branch ctx/learning/tool-calling-v2 \\
        --began-utc 2026-10-11T08:13:50Z [--budget-s 2400] [--seed-run r01-lr0.0001-rank16-drop0.05]

Run from the repo root (Git Bash) with a Python that has optuna (the lab's own venv does not; any venv with
`pip install optuna` will do). The study is a SQLite file under <workspace>/work/L16v2-out/bo/ and never leaves this
machine. Each Colab session named in --sessions is one worker: when it is free the driver asks the study for the next
point, starts that one trial on the session (lab/l16v2/colab_trial.sh, detached), polls it, reports its step-12
validation loss to the pruner (and interrupts the trial if it is pruned), pulls and verifies the trial's files when
it ends, and tells the study the result. So every point is proposed with every finished result in the model, the
polling is each session's keepalive, and no result lives only on Colab. A session that is not set up yet (no model,
no clone, no dataset) is set up first and joins when its marker appears.

The search design is copied from Ilan's earlier Optuna search (MCA_full_pipeline, PIPELINE/training/
hp_search_modernbert_rocm.py), not its values: TPESampler(multivariate=True, group=True, seed=SEED + trials already in
the study), here with constant_liar=True and RANDOM_STARTUP random trials in total after the seeded ones;
MedianPruner(n_startup_trials=2, n_warmup_steps=1) on the mid-trial validation loss; build_storage(), his
_build_concurrent_storage, with load_if_exists=True, so a stopped driver resumes; the warm start (a finished
observation added, known configurations enqueued with skip_if_exists); the duplicate-parameters guard; only real
trials counted; the boundary check after 5 finished trials; and early stopping with patience 2, the minimum
improvement made relative (3 percent) because this loss is near 0.01, not near 0.7.

The rule (BO_RULE, in lab/L16V2_RULES.md amendment 2 before any trial had a result): trials of 24 optimizer steps with
validation at steps 12 and 24; after the budget the three best configurations train at full length at the same time,
one per session; the frozen choice rule picks among the full-length runs (those and the grid's run 1) by validation
loss and then validation tool-call agreement, scored on the sessions that hold the adapters.

What it writes: one line per poll and per event in <workspace>/work/L16v2-out/progress.log in the fixed form
(stage=bo trial=k/N kind=seed|init|model seen=.. worker=<session> lr=.. rank=.. alpha=.. drop=.. warm=.. step=..
val=.. best=..), a RESULT line per finished trial, `RESULT final adapter ...` for every full-length adapter that is
here and verified (chosen=yes on the chosen one), the bo_table lines (trials in order of proposal, the best-so-far
curve, model-chosen against random startup); bo/bo.json and bo/chosen.json; every product under colab/ with its
sha256 checked (colab/downloads.txt).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

SEED = 42
STUDY_NAME = "l16v2-lora"
FIDELITY_STEPS = 24
EVAL_AT = 12
RANDOM_STARTUP = 3
PRUNER_STARTUP = 2
BOUNDARY_AFTER = 5
EDGE = 0.05
FULL_STEPS = 62
FULL_EVAL = 8
FULL_PATIENCE = 2
FULL_MIN_REL = 0.03
FINALISTS = 3
TOP = 3
EFFECTIVE_BATCH = 16
RANKS = (4, 8, 16, 32)
ALPHA_MULTS = (1, 2)
RANGES = {"lr": [2e-5, 5e-4], "dropout": [0.0, 0.15], "warmup": [0.0, 0.15]}
WIDENED = {"lr": (1e-5, 1e-3), "dropout": (0.0, 0.25), "warmup": (0.0, 0.25)}
WARM_START_CONFIGS = [
    {"lr": 2e-4, "rank": 16, "alpha_mult": 2, "dropout": 0.05, "warmup": 0.10},
    {"lr": 5e-5, "rank": 16, "alpha_mult": 2, "dropout": 0.05, "warmup": 0.10},
]
ROWS_SHA256 = "64e3d4222e9e007e588016a43b8914e1a3f4c9e72b1152727824d8934c135b9d"    # rows-train.jsonl of the frozen dataset
SPLIT_SHA256 = "4776164b103cd9af8ffb877243cb3694cc7c75b22c7e81f78e5ff64351924377"
BO_RULE = "lab/L16V2_RULES.md, amendment 2 (the search, the finalists and the choice), as committed before any trial had a result"

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
W = Path(os.environ.get("L16V2_WORKSPACE", "C:/Users/hackathon/Code/Knighthacks26"))
OUT = W / "work" / "L16v2-out"
COLAB = OUT / "colab"
BO = OUT / "bo"
LAUNCH = COLAB / "launch"
WRAPPER = os.environ.get("L16V2_COLAB", str(W / "tooling" / "dispatcher" / "colab.sh"))
BASH = next((p for p in ("C:/Program Files/Git/usr/bin/bash.exe", "C:/Program Files/Git/bin/bash.exe") if Path(p).exists()),
            shutil.which("bash") or "bash")
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
log_lock = threading.Lock()


def plog(text: str) -> None:
    """One line in progress.log, starting with the Eastern time (UTC minus 4 hours, as the other lines are stamped)."""
    now = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=4)
    stamp = now.strftime("%I:%M %p ET").lstrip("0")
    with log_lock, (OUT / "progress.log").open("a", encoding="ascii", errors="replace", newline="\n") as out:
        out.write(f"{stamp} {text}\n")


def clean(raw: bytes) -> str:
    """The CLI's answer as plain ASCII: bytes in, ANSI codes out, nothing a log holds can break a read."""
    text = ANSI.sub("", raw.decode("utf-8", "replace"))
    return "".join(ch for ch in text if ch in "\t\n\r" or 32 <= ord(ch) < 127)


def colab(*args: str, timeout: int = 200) -> str:
    """One call of the Colab CLI. Its output goes to a file, not a pipe, and on a timeout the whole process tree is
    killed: a pipe held open by a grandchild (bash, wsl, the CLI) once left a worker's thread waiting for ever."""
    env = {**os.environ, "MSYS_NO_PATHCONV": "1"}
    with tempfile.TemporaryFile() as sink:
        process = subprocess.Popen([BASH, WRAPPER, *args], stdout=sink, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   env=env, creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], capture_output=True, check=False)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass
            return "TIMEOUT"
        sink.seek(0)
        return clean(sink.read())


def wsl(path: Path) -> str:
    text = str(path).replace("\\", "/")
    return "/mnt/" + text[0].lower() + text[2:]


def remote(session: str, code: str, word: str, label: str, tries: int = 5, timeout: int = 150) -> str | None:
    """Run Python on the session's VM and return its ASCII output; None after `tries` failed or unreadable answers
    (a failed read is a retry, never a dead poll). `word` must be in a good answer."""
    LAUNCH.mkdir(parents=True, exist_ok=True)
    file = LAUNCH / f"{session}-{label}-{threading.get_ident()}.py"     # one file per calling thread
    file.write_text(code, encoding="ascii", errors="replace", newline="\n")
    for attempt in range(1, tries + 1):
        answer = colab("exec", "-s", session, "--timeout", str(max(30, timeout - 60)), "-f", str(file).replace("\\", "/"),
                       timeout=timeout)
        if word in answer and "Connection was lost" not in answer and "Traceback" not in answer:
            return answer
        last = [line for line in answer.splitlines() if line.strip()][-1:] or ["no answer"]
        plog(f"PROBLEM worker={session} exec {label} try {attempt} failed, retrying: {last[0][:140]}")
        time.sleep(6)
    return None


def shell(session: str, command: str, label: str, timeout: int = 150) -> str | None:
    """A shell command on the VM, its output read as bytes and printed as ASCII, ended with the word END."""
    code = ("import subprocess\n"
            f"raw = subprocess.run({command!r}, shell=True, capture_output=True, executable='/bin/bash')\n"
            "print((raw.stdout + raw.stderr).decode('utf-8', 'replace').encode('ascii', 'replace').decode('ascii'))\n"
            "print('END')\n")
    return remote(session, code, "END", label, timeout=timeout)


def upload(session: str, local: Path, target: str) -> bool:
    for attempt in range(1, 5):
        answer = colab("upload", "-s", session, str(local).replace("\\", "/"), target, timeout=900)
        if "Uploaded" in answer:
            plog(f"upload done worker={session} {target} bytes={local.stat().st_size}")
            return True
        plog(f"PROBLEM worker={session} upload try {attempt} failed for {target}: {answer.strip()[-140:]}")
        time.sleep(8)
    return False


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def get(session: str, name: str) -> Path | None:
    """Pull /content/l16v2/products/<name> with its .sha256 and check size and sha256; up to 4 tries."""
    COLAB.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 5):
        part, check = COLAB / (name + ".part"), COLAB / (name + ".sha256.part")
        part.unlink(missing_ok=True)
        check.unlink(missing_ok=True)
        colab("download", "-s", session, f"/content/l16v2/products/{name}.sha256", wsl(check), timeout=300)
        colab("download", "-s", session, f"/content/l16v2/products/{name}", wsl(part), timeout=1500)
        try:
            want, size = check.read_text(encoding="ascii").split()[:2]
            have = sha256_of(part)
            if want == have and int(size) == part.stat().st_size:
                os.replace(part, COLAB / name)
                os.replace(check, COLAB / (name + ".sha256"))
                with (COLAB / "downloads.txt").open("a", encoding="ascii") as out:
                    out.write(f"{name} {size} {have} verified\n")
                plog(f"download {name} worker={session} bytes={size} sha256={have} verified=yes")
                return COLAB / name
        except (OSError, ValueError):
            pass
        plog(f"PROBLEM worker={session} download try {attempt} of {name} did not verify; retrying")
        time.sleep(8)
    plog(f"PROBLEM worker={session} download of {name} failed 4 times")
    return None


def extract(tarball: Path, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball) as tar:
        tar.extractall(folder, filter="data")


# --- the study --------------------------------------------------------------------------------------------------

def build_storage(path: Path) -> Any:
    """An Optuna RDBStorage on the SQLite file: copied from _build_concurrent_storage of hp_search_modernbert_rocm.py.
    The journal mode (L16V2_SQLITE_JOURNAL, default DELETE as his is since his WAL incident) and a long busy
    timeout are set on the file first, best effort; the storage's own 100 s wait on a write lock is the safety net."""
    import optuna

    path.parent.mkdir(parents=True, exist_ok=True)
    journal = os.environ.get("L16V2_SQLITE_JOURNAL", "DELETE").upper()
    try:
        connection = sqlite3.connect(str(path), timeout=100.0)
        try:
            connection.execute(f"PRAGMA journal_mode={journal};")
            connection.execute("PRAGMA busy_timeout=100000;")
            connection.commit()
        finally:
            connection.close()
    except Exception:   # the pragma is best effort; the storage's timeout still guards
        pass
    return optuna.storages.RDBStorage(url="sqlite:///" + str(path).replace("\\", "/"),
                                      engine_kwargs={"connect_args": {"timeout": 100}})


def trial_name(number: int, params: dict[str, Any]) -> str:
    return (f"t{number:02d}-lr{params['lr']:.3g}-rank{params['rank']}-a{params['rank'] * params['alpha_mult']}"
            f"-drop{params['dropout']:.3f}-warm{params['warmup']:.3f}")


def suggest(trial: Any) -> dict[str, Any]:
    return {"lr": trial.suggest_float("lr", RANGES["lr"][0], RANGES["lr"][1], log=True),
            "rank": trial.suggest_categorical("rank", list(RANKS)),
            "alpha_mult": trial.suggest_categorical("alpha_mult", list(ALPHA_MULTS)),
            "dropout": trial.suggest_float("dropout", RANGES["dropout"][0], RANGES["dropout"][1]),
            "warmup": trial.suggest_float("warmup", RANGES["warmup"][0], RANGES["warmup"][1])}


def on_edge(name: str, value: float) -> str | None:
    low, high = RANGES[name]
    if name == "lr":
        position = (math.log(value) - math.log(low)) / (math.log(high) - math.log(low))
    else:
        position = (value - low) / (high - low)
    return "low" if position <= EDGE else "high" if position >= 1 - EDGE else None


def boundary_check(best: dict[str, Any]) -> None:
    """His boundary check, once: widen the range whose edge the best trial sits on, and say so."""
    for name, value in (("lr", best["lr"]), ("dropout", best["dropout"]), ("warmup", best["warmup_ratio"])):
        edge = on_edge(name, value)
        if edge is None:
            continue
        before = tuple(RANGES[name])
        RANGES[name][0 if edge == "low" else 1] = WIDENED[name][0 if edge == "low" else 1]
        after = tuple(RANGES[name])
        plog(f"decision boundary_check {name}={value:.4g} of trial {best['number']} is on the {edge} edge of "
             f"{before[0]:g}..{before[1]:g}: " + (f"range widened once to {after[0]:g}..{after[1]:g}"
                                                 if after != before else "that edge cannot be widened"))
    plog(f"boundary_check done after {BOUNDARY_AFTER} finished trials (best is trial {best['number']}): ranges now "
         + " ".join(f"{name}={low:g}..{high:g}" for name, (low, high) in RANGES.items()))


def fmt(value: Any, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "na"


# --- one run on one worker ----------------------------------------------------------------------------------------

STATUS_CODE = '''import json, os, subprocess
run = "/content/l16v2/runs/NAME"
def rows(name):
    try:
        return [json.loads(line) for line in open(run + "/" + name) if line.strip()]
    except Exception:
        return []
steps = rows("telemetry.jsonl")
if steps:
    print("T", steps[-1]["step"], steps[-1]["loss"], round(sum(s["step_ms"] for s in steps[1:]) / max(1, len(steps) - 1) / 1000.0, 2))
for check in rows("validation.jsonl"):
    print("V", check["step"], check["eval_loss"])
if os.path.exists(run + "/DONE"):
    print("DONE", open(run + "/DONE").read().strip())
elif os.path.exists(run + "/STARTED"):
    print("RUNNING", subprocess.run("pgrep -f '[r]un-id NAME ' | wc -l", shell=True, capture_output=True).stdout.decode().strip())
else:
    print("ABSENT")
print("GPU", subprocess.run("nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader", shell=True, capture_output=True).stdout.decode("ascii", "replace").strip())
print("UPTIME", open("/proc/uptime").read().split()[0])
'''


def launch(session: str, name: str, adapter: bool, flags: list[str]) -> bool:
    command = (f"setsid nohup bash /content/colab_trial.sh {name} {'yes' if adapter else 'no'} {' '.join(flags)} "
               f"> /content/l16v2/trial-{name}.log 2>&1 &")
    code = f"import subprocess\nsubprocess.Popen({command!r}, shell=True)\nprint('LAUNCHED')\n"
    return remote(session, code, "LAUNCHED", "launch") is not None


def read_status(session: str, name: str) -> dict[str, Any] | None:
    answer = remote(session, STATUS_CODE.replace("NAME", name), "UPTIME", "status", tries=4, timeout=120)
    if answer is None:
        return None
    status: dict[str, Any] = {"step": 0, "loss": None, "step_s": None, "val": {}, "done": None, "state": "absent"}
    for line in answer.splitlines():
        part = line.split()
        try:
            if part[:1] == ["T"]:
                status.update(step=int(part[1]), loss=None if part[2] == "None" else float(part[2]), step_s=float(part[3]))
            elif part[:1] == ["V"]:
                status["val"][int(part[1])] = float(part[2])
            elif part[:1] == ["DONE"]:
                status.update(done=int(part[1]), state="done")
            elif part[:1] == ["RUNNING"]:
                status["state"] = "running"
        except (IndexError, ValueError):
            continue
    return status


def run_on(session: str, name: str, flags: list[str], adapter: bool, line: Any, at_mid: Any = None,
           poll_s: float = 30.0) -> dict[str, Any]:
    """Start (or adopt) one run on the session, follow it to its end, pull and read its files. `line(status)` gives
    the progress line of a poll; `at_mid(loss)` is asked once with the step-12 validation loss and stops the run
    when it says so. Returns the run's facts; "lost" when the session stopped answering."""
    if not launch(session, name, adapter, flags):
        return {"name": name, "outcome": "lost", "ok": False}
    began = time.time()
    asked = pruned = False
    failures = 0
    status: dict[str, Any] = {}
    while True:
        time.sleep(poll_s)
        read = read_status(session, name)
        if read is None:
            failures += 1
            plog(f"PROBLEM worker={session} no status for {name} (round {failures})")
            if failures >= 6:
                return {"name": name, "outcome": "lost", "ok": False}
            continue
        failures, status = 0, read
        plog(line(status))
        if at_mid is not None and not asked and EVAL_AT in status["val"]:
            asked = True
            if at_mid(status["val"][EVAL_AT]):
                pruned = True
                # SIGTERM: the trainer was started detached in the background, where SIGINT is ignored (the first
                # pruned trials ran on to step 24 for that reason); colab_trial.sh then packs what there is.
                shell(session, f"pkill -TERM -f '[r]un-id {name} '; sleep 2; pkill -KILL -f '[r]un-id {name} '; echo stopped", "prune")
                plog(f"decision trial {name} pruned by the median rule at step {EVAL_AT} (val {status['val'][EVAL_AT]:.4f}): "
                     f"its trainer on {session} is stopped now")
        if status["state"] == "done":
            break
        if status["state"] == "absent" and time.time() - began > 120:
            return {"name": name, "outcome": "lost", "ok": False}
    product = get(session, f"run-{name}.tar.gz")
    fact: dict[str, Any] = {"name": name, "worker": session, "wall_s": round(time.time() - began, 1),
                            "step_s": status.get("step_s"), "evaluations": [{"step": s, "eval_loss": v} for s, v in sorted(status["val"].items())],
                            "product": product.name if product else None,
                            "product_sha256": sha256_of(product) if product else None}
    answer: dict[str, Any] = {}
    if product is not None:
        folder = COLAB / "runs"
        extract(product, folder)
        fact["local"] = str(folder / name).replace("\\", "/")
        try:
            answer = json.loads((folder / name / "stdout.json").read_text(encoding="utf-8").strip().splitlines()[-1])
        except (OSError, ValueError, IndexError):
            answer = {}
    metrics, refs = answer.get("metrics", {}), answer.get("refs", {})
    by_step = status["val"]
    if metrics.get("trainer.train_s") and fact["wall_s"] < metrics["trainer.train_s"]:
        fact["wall_s"] = round(metrics["trainer.train_s"] + 40.0, 1)    # a run adopted after it had started: its own time
    fact.update(outcome="pruned" if pruned else answer.get("outcome", f"exit {status.get('done')}"),
                ok=bool(answer.get("ok")) and not pruned, error=answer.get("error"),
                steps=metrics.get("trainer.steps", status.get("step")),
                passes=metrics.get("trainer.examples_seen", (status.get("step") or 0) * EFFECTIVE_BATCH),
                val_at_12=by_step.get(EVAL_AT), val_at_24=by_step.get(FIDELITY_STEPS),
                best_step=metrics.get("trainer.best_step"), best_passes=metrics.get("trainer.best_examples_seen"),
                best_val=metrics.get("trainer.best_eval_loss", min(by_step.values()) if by_step else None),
                final_val=metrics.get("trainer.final_eval_loss"), loss_last=metrics.get("trainer.loss_last"),
                train_s=metrics.get("trainer.train_s"), peak_mem_gb=metrics.get("trainer.peak_mem_gb"),
                device=refs.get("device"), settings=refs.get("settings"), loss_head=refs.get("loss_head"),
                torch=refs.get("torch"), gradient_checkpointing=True, batch=4, grad_accum=4)
    return fact


# --- the search -----------------------------------------------------------------------------------------------------

class Search:
    def __init__(self, args: argparse.Namespace) -> None:
        import optuna
        from optuna.trial import TrialState

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.optuna, self.State, self.args = optuna, TrialState, args
        self.lock = threading.RLock()
        self.began = dt.datetime.fromisoformat(args.began_utc.replace("Z", "+00:00")).timestamp() if args.began_utc else None
        storage = build_storage(BO / "study.db")
        try:
            existing = len(optuna.load_study(study_name=STUDY_NAME, storage=storage).trials)
        except KeyError:
            existing = 0
        self.trials: list[dict[str, Any]] = []
        if existing and (BO / "bo.json").exists():
            self.trials = json.loads((BO / "bo.json").read_text(encoding="utf-8")).get("trials", [])
        self.seeded = 1 + len(WARM_START_CONFIGS)
        self.startup = RANDOM_STARTUP + self.seeded
        sampler = optuna.samplers.TPESampler(multivariate=True, group=True, constant_liar=True, seed=SEED + existing,
                                             n_startup_trials=self.startup)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=PRUNER_STARTUP, n_warmup_steps=1)
        self.study = optuna.create_study(direction="minimize", study_name=STUDY_NAME, sampler=sampler, pruner=pruner,
                                         storage=storage, load_if_exists=True)
        self.engine = (f"optuna {optuna.__version__} TPESampler(multivariate, group, constant_liar, seed {SEED}+{existing}, "
                       f"startup {self.startup} = {RANDOM_STARTUP} random after {self.seeded} seeded) "
                       f"MedianPruner(startup {PRUNER_STARTUP}, warmup 1), sqlite on the workstation, asynchronous workers")
        self.numbers = max((t["number"] for t in self.trials), default=0)
        # Trials an earlier driver left running: each is followed to its end by the worker that holds it.
        self.orphans: dict[str, Any] = {}
        for frozen in self.study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
            holder, name = frozen.user_attrs.get("worker"), frozen.user_attrs.get("name")
            if holder and name and holder not in self.orphans:
                self.orphans[holder] = frozen
                self.numbers = max(self.numbers, int(name[1:3]))
            else:
                self.study.tell(frozen.number, state=TrialState.FAIL)
        self.longest = max((t.get("wall_s") or 0 for t in self.trials if t.get("max_steps") == FIDELITY_STEPS), default=0.0)
        self.stopped = False
        self.checked_boundary = False
        self.duplicates = 0
        self.best = "none"
        self.workers: list[str] = []
        self.slots: dict[str, int] = {}
        plog(f"decision bo engine: {self.engine}; study file {str(BO / 'study.db')}; trials already in it: {existing}")
        if not existing:
            self.seed_from_grid()
            for config in WARM_START_CONFIGS:
                self.study.enqueue_trial(dict(config), skip_if_exists=True)
        self.note_best()
        self.save()

    def seed_from_grid(self) -> None:
        """The grid's run 1 as a finished observation: its own validation loss at step 24 (a 62-step schedule)."""
        from optuna.distributions import CategoricalDistribution, FloatDistribution

        folder = COLAB / "x-run-01"
        fact = None
        for tarball in sorted(COLAB.glob("run-01-*.tar.gz")):
            extract(tarball, folder)
            for result in folder.glob("*/result.json"):
                fact = json.loads(result.read_text(encoding="utf-8"))
        if fact is None:
            plog("PROBLEM bo: the grid's run 1 is not on this machine; the search starts without that observation")
            self.seeded -= 1
            return
        by_step = {int(c["step"]): c["eval_loss"] for c in fact["evaluations"]}
        value = by_step[FIDELITY_STEPS]
        params = {"lr": float(fact["lr"]), "rank": int(fact["rank"]), "alpha_mult": int(fact["alpha"] // fact["rank"]),
                  "dropout": float(fact["dropout"]), "warmup": float(fact["warmup_ratio"])}
        self.study.add_trial(self.optuna.trial.create_trial(
            params=params,
            distributions={"lr": FloatDistribution(RANGES["lr"][0], RANGES["lr"][1], log=True),
                           "rank": CategoricalDistribution(list(RANKS)),
                           "alpha_mult": CategoricalDistribution(list(ALPHA_MULTS)),
                           "dropout": FloatDistribution(*RANGES["dropout"]), "warmup": FloatDistribution(*RANGES["warmup"])},
            value=float(value), intermediate_values={s: float(v) for s, v in by_step.items() if s <= FIDELITY_STEPS},
            user_attrs={"kind": "seed", "name": fact["name"], "note": "grid run 1 at its own step 24 of a 62-step schedule"}))
        seed = {**fact, "number": 1, "kind": "seed", "seen": 0, "worker": "l16v2-a100", "alpha_mult": params["alpha_mult"],
                "value": value, "val_at_12": None, "val_at_24": value, "full_length": True,
                "note": "the grid's run 1: its own step-24 validation loss, from a 62-step schedule with validation every 8 steps"}
        self.trials.append(seed)
        self.numbers = 1
        plog(f"RESULT trial=1 kind=seed seen=0 worker=l16v2-a100 name={fact['name']} lr={fact['lr']:.3g} rank={fact['rank']} "
             f"alpha={fact['alpha']} drop={fact['dropout']:.3f} warm={fact['warmup_ratio']:.3f} outcome=observation "
             f"val24={value:.4f} value={value:.4f} note=grid_run_1_at_its_own_step_24_of_a_62_step_schedule")

    def note_best(self) -> None:
        valued = [t for t in self.trials if t.get("value") is not None and t.get("outcome") != "pruned"]
        if valued:
            top = min(valued, key=lambda t: t["value"])
            self.best = f"trial{top['number']}:{top['value']:.4f}"

    def save(self) -> None:
        BO.mkdir(parents=True, exist_ok=True)
        (BO / "bo.json").write_text(json.dumps({
            "rule": BO_RULE, "engine": self.engine, "study": STUDY_NAME, "fidelity_steps": FIDELITY_STEPS,
            "eval_at": EVAL_AT, "workers": self.workers, "began_utc": self.began,
            "space": {"lr": RANGES["lr"], "rank": RANKS, "alpha_mult": ALPHA_MULTS, "dropout": RANGES["dropout"],
                      "warmup": RANGES["warmup"]},
            "warm_start_configs": WARM_START_CONFIGS, "trials": sorted(self.trials, key=lambda t: t["number"])},
            indent=2) + "\n", encoding="utf-8")

    def real_finished(self) -> int:   # his count: completed, or pruned as a real dud; never a duplicate skip
        return sum(t.state == self.State.COMPLETE
                   or (t.state == self.State.PRUNED and not t.user_attrs.get("duplicate", False))
                   for t in self.study.get_trials(deepcopy=False))

    def next_trial(self, session: str) -> tuple[Any, dict[str, Any], dict[str, Any]] | None:
        """Ask the study for this worker's next point, now, with everything finished so far in the model."""
        with self.lock:
            if self.stopped:
                return None
            if self.began is None:
                self.began = time.time()
            elapsed = time.time() - self.began
            estimate = self.longest or 240.0
            ran = [t for t in self.trials if t.get("max_steps") == FIDELITY_STEPS]
            if ran and elapsed + estimate > self.args.budget_s:
                self.stopped = True
                plog(f"decision bo stops proposing: budget ({int(elapsed)} s of {int(self.args.budget_s)} s used, the next "
                     f"trial needs about {int(estimate)} s); real trials {self.real_finished()}, duplicates skipped {self.duplicates}")
                return None
            seen = self.real_finished()
            if not self.checked_boundary and seen >= BOUNDARY_AFTER:
                self.checked_boundary = True
                valued = [t for t in self.trials if t.get("value") is not None and t.get("outcome") != "pruned"]
                if valued:
                    boundary_check(min(valued, key=lambda t: t["value"]))
            while True:
                every = self.study.get_trials(deepcopy=False)
                counted = sum(t.state in (self.State.COMPLETE, self.State.PRUNED, self.State.RUNNING) for t in every)
                waiting = [t.system_attrs.get("fixed_params") for t in every if t.state == self.State.WAITING]
                trial = self.study.ask()
                params = suggest(trial)
                enqueued = any(fixed and all(params.get(k) == v for k, v in fixed.items()) for fixed in waiting)
                kind = "seed" if enqueued else "init" if counted < self.startup else "model"
                twin = next((t for t in every if t.number != trial.number
                             and t.state in (self.State.COMPLETE, self.State.PRUNED) and t.params == trial.params), None)
                if twin is None:
                    break
                trial.set_user_attr("duplicate", True)    # his duplicate guard: skipped without training, never counted
                self.study.tell(trial, state=self.State.PRUNED)
                self.duplicates += 1
                plog(f"bo duplicate of study trial {twin.number} skipped without training")
                if self.duplicates > 20:
                    self.stopped = True
                    return None
            self.numbers += 1
            number = self.numbers
            total = number - 1 + max(1, int((self.args.budget_s - elapsed) // estimate) * max(1, len(self.workers)))
            fact = {"number": number, "total": total, "kind": kind, "seen": seen, "worker": session,
                    "name": trial_name(number, params), "lr": params["lr"], "rank": params["rank"],
                    "alpha": params["rank"] * params["alpha_mult"], "alpha_mult": params["alpha_mult"],
                    "dropout": params["dropout"], "warmup_ratio": params["warmup"], "max_steps": FIDELITY_STEPS,
                    "study_trial": trial.number, "proposed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            for key in ("kind", "seen", "worker", "name"):
                trial.set_user_attr(key, fact[key])
            return trial, params, fact

    def finish(self, trial: Any, fact: dict[str, Any]) -> None:
        with self.lock:
            if fact["outcome"] == "pruned" and fact.get("val_at_12") is not None:
                fact["value"] = fact["val_at_12"]
                self.study.tell(trial, state=self.State.PRUNED)
            elif fact.get("ok") and fact.get("best_val") is not None:
                fact["value"] = fact["best_val"]
                if fact.get("val_at_24") is not None:
                    trial.report(float(fact["val_at_24"]), FIDELITY_STEPS)
                self.study.tell(trial, float(fact["value"]))
            else:
                fact["value"] = None
                self.study.tell(trial, state=self.State.FAIL)
            if fact.get("wall_s") and fact["outcome"] != "lost":
                self.longest = max(self.longest, fact["wall_s"])
            self.trials.append(fact)
            self.note_best()
            self.save()
            plog(f"RESULT trial={fact['number']}/{fact['total']} kind={fact['kind']} seen={fact['seen']} worker={fact['worker']} "
                 f"name={fact['name']} lr={fact['lr']:.3g} rank={fact['rank']} alpha={fact['alpha']} drop={fact['dropout']:.3f} "
                 f"warm={fact['warmup_ratio']:.3f} outcome={fact['outcome']} steps={fact.get('steps')} passes={fact.get('passes')} "
                 f"val12={fmt(fact.get('val_at_12'))} val24={fmt(fact.get('val_at_24'))} value={fmt(fact.get('value'))} "
                 f"best={self.best} wall_s={fact.get('wall_s')} step_s={fmt(fact.get('step_s'), 2)}"
                 + (" pruned_by_median_at_step_12" if fact["outcome"] == "pruned" else "")
                 + (f" error={str(fact.get('error'))[:100]}" if fact.get("value") is None else ""))

    def adopt(self, worker: str) -> tuple[Any, dict[str, Any], dict[str, Any]] | None:
        """The trial an earlier driver started on this worker and did not see end, as if it had just been asked."""
        with self.lock:
            frozen = self.orphans.pop(worker, None)
            if frozen is None:
                return None
            trial = self.optuna.trial.Trial(self.study, frozen._trial_id)
            params = dict(frozen.params)
            name = frozen.user_attrs["name"]
            fact = {"number": int(name[1:3]), "total": int(name[1:3]), "kind": frozen.user_attrs.get("kind"),
                    "seen": frozen.user_attrs.get("seen"), "worker": worker, "name": name, "lr": params["lr"],
                    "rank": params["rank"], "alpha": params["rank"] * params["alpha_mult"],
                    "alpha_mult": params["alpha_mult"], "dropout": params["dropout"], "warmup_ratio": params["warmup"],
                    "max_steps": FIDELITY_STEPS, "study_trial": frozen.number, "adopted": True}
            plog(f"decision worker={worker} adopts trial {fact['number']} ({name}), left running by the driver before this one")
            return trial, params, fact

    def worker(self, worker: str) -> None:
        """One worker's loop: ask only when free, run the trial, tell. Ends when the study stops proposing."""
        session = worker.split("#")[0]
        while True:
            asked = self.adopt(worker) or self.next_trial(worker)
            if asked is None:
                return
            trial, params, fact = asked
            fact["concurrent_on_card"] = self.slots.get(session, 1)
            flags = ["--max-steps", str(FIDELITY_STEPS), "--eval-steps", str(EVAL_AT), "--lr", f"{params['lr']:.6g}",
                     "--rank", str(params["rank"]), "--alpha", str(fact["alpha"]), "--dropout", f"{params['dropout']:.4f}",
                     "--warmup-ratio", f"{params['warmup']:.4f}"]

            def line(status: dict[str, Any], fact: dict[str, Any] = fact) -> str:
                latest = status["val"][max(status["val"])] if status["val"] else None
                return (f"stage=bo trial={fact['number']}/{fact['total']} kind={fact['kind']} seen={fact['seen']} "
                        f"worker={fact['worker']} lr={fact['lr']:.3g} rank={fact['rank']} alpha={fact['alpha']} "
                        f"drop={fact['dropout']:.3f} warm={fact['warmup_ratio']:.3f} step={status['step']}/{FIDELITY_STEPS} "
                        f"loss={fmt(status['loss'])} val={fmt(latest)} best={self.best} step_s={fmt(status['step_s'], 2)}")

            def at_mid(loss: float, trial: Any = trial) -> bool:
                with self.lock:
                    trial.report(float(loss), EVAL_AT)
                    return bool(trial.should_prune())

            done = run_on(session, fact["name"], flags, False, line, at_mid)
            fact.update(done)
            if done["outcome"] == "lost":    # the session stopped answering: the point goes back to the queue
                with self.lock:
                    self.study.tell(trial, state=self.State.FAIL)
                    self.study.enqueue_trial({k: params[k] for k in ("lr", "rank", "alpha_mult", "dropout", "warmup")})
                    self.numbers -= 0
                    plog(f"PROBLEM worker={worker} lost during trial {fact['number']}; its point is enqueued again and "
                         "this worker leaves the search")
                    if worker in self.workers:
                        self.workers.remove(worker)
                return
            self.finish(trial, fact)

    def table(self) -> None:
        """The trials in order of proposal, the best-so-far curve, and model-chosen against random startup trials."""
        best = None
        ordered = sorted(self.trials, key=lambda t: t["number"])
        for t in ordered:
            if t.get("value") is not None and t.get("outcome") != "pruned":
                best = t["value"] if best is None else min(best, t["value"])
            plog(f"RESULT bo_table n={t['number']} seen={t.get('seen')} kind={t['kind']} worker={t.get('worker')} "
                 f"lr={t['lr']:.3g} rank={t['rank']} alpha={t['alpha']} drop={t['dropout']:.3f} warm={t['warmup_ratio']:.3f} "
                 f"outcome={t.get('outcome', 'observation')} val12={fmt(t.get('val_at_12'))} val24={fmt(t.get('val_at_24'))} "
                 f"value={fmt(t.get('value'))} best_so_far={fmt(best)}")
        parts = []
        for kind in ("seed", "init", "model"):
            ran = [t for t in ordered if t["kind"] == kind and t.get("max_steps") == FIDELITY_STEPS]
            at_12 = [t["val_at_12"] for t in ran if t.get("val_at_12") is not None]
            done = [t["value"] for t in ran if t.get("value") is not None and t.get("outcome") != "pruned"]
            parts.append(f"{kind}: n={len(ran)} mean_val12={fmt(sum(at_12) / len(at_12) if at_12 else None)} "
                         f"completed={len(done)} mean_value={fmt(sum(done) / len(done) if done else None)} "
                         f"best_value={fmt(min(done) if done else None)} pruned={sum(t.get('outcome') == 'pruned' for t in ran)}")
        plog("RESULT bo_summary " + " | ".join(parts))


# --- sessions -------------------------------------------------------------------------------------------------------

def session_ready(session: str, commit: str) -> str:
    """ready, stale (set up, but its clone is another commit) or new, from the session's own marker."""
    answer = shell(session, "cat /content/l16v2/WORKER_READY 2>/dev/null; echo; ls /content/l16v2/data/tool-calling-v2/rows-train.jsonl "
                            "/content/models/Qwen3.5-4B/config.json /content/Knighthacks26/lab/train.py 2>/dev/null | wc -l", "ready")
    if answer is None:
        return "unreachable"
    if ROWS_SHA256 in answer and SPLIT_SHA256 in answer:
        return "ready" if commit in answer else "stale"
    lines = [line.strip() for line in answer.splitlines() if line.strip().isdigit()]
    return "stale" if lines and lines[-1] == "3" else "new"


def prepare(session: str, branch: str, commit: str, bundle: Path, full: bool) -> bool:
    """Bring a session to the branch's commit with the frozen dataset: upload the bundle and the scripts, run
    lab/l16v2/colab_worker_setup.sh detached, wait for its marker and check the dataset's sha256."""
    files = [(bundle, "/content/l16v2.bundle"), (HERE / "colab_worker_setup.sh", "/content/colab_worker_setup.sh"),
             (HERE / "colab_trial.sh", "/content/colab_trial.sh"), (HERE / "vm_score.py", "/content/vm_score.py")]
    if full:
        files += [(REPO / "lab" / "requirements-train.txt", "/content/requirements-train.txt"),
                  (HERE / "colab_setup.sh", "/content/colab_setup.sh")]
    for local, target in files:
        if not upload(session, local, target):
            return False
    code = ("import subprocess\nsubprocess.run('rm -f /content/l16v2/WORKER_READY', shell=True)\n"
            f"subprocess.Popen('setsid nohup bash /content/colab_worker_setup.sh {branch} > /content/colab_worker_setup.log 2>&1 &', shell=True)\n"
            "print('LAUNCHED')\n")
    if remote(session, code, "LAUNCHED", "setup") is None:
        return False
    began = time.time()
    while time.time() - began < 900:
        time.sleep(20)
        answer = shell(session, "cat /content/l16v2/WORKER_READY 2>/dev/null; echo; tail -n 1 /content/colab_worker_setup.log | cut -c1-120", "setup-wait")
        if answer is None:
            continue
        if ROWS_SHA256 in answer and SPLIT_SHA256 in answer and commit in answer:
            plog(f"worker={session} set up in {int(time.time() - began)} s: clone at {commit[:7]}, dataset rebuilt there with the "
                 f"same sha256 (rows {ROWS_SHA256[:12]}, split {SPLIT_SHA256[:12]})")
            return True
        if "WORKER_FAILED" in answer:
            plog(f"PROBLEM worker={session} setup failed: {answer.strip()[-200:]}")
            return False
        plog(f"status stage=setup worker={session} elapsed_s={int(time.time() - began)} last=\"{answer.strip().splitlines()[-2][-100:] if len(answer.strip().splitlines()) > 1 else ''}\"")
    plog(f"PROBLEM worker={session} setup did not finish in 15 minutes")
    return False


def final_adapter_line(fact: dict[str, Any], chosen: bool = False) -> str:
    return (f"RESULT final adapter name={fact['name']} path={fact.get('adapter_path')} sha256={fact.get('product_sha256')} "
            f"adapter_sha256={fact.get('adapter_sha256')} lr={fact['lr']:.3g} rank={fact['rank']} alpha={fact['alpha']} "
            f"drop={fact['dropout']:.3f} warm={fact['warmup_ratio']:.3f} steps={fact.get('best_step')} "
            f"passes={fact.get('best_passes')} val={fmt(fact.get('best_val'))} worker={fact.get('worker')}"
            + (f" val_agreement={fact['val_agreement']['correct']}/{fact['val_agreement']['n']}" if fact.get("val_agreement") else "")
            + (" chosen=yes" if chosen else ""))


def score_on(session: str, label: str, arms: list[str]) -> dict[str, dict[str, Any]] | None:
    """Validation tool-call agreement of adapters that are on this session (and of the base when asked), there."""
    command = (f"cd /content/Knighthacks26 && setsid nohup python3 /content/vm_score.py validation {label} {' '.join(arms)} "
               f"> /content/l16v2/score-{label}.log 2>&1 &")
    code = f"import subprocess\nsubprocess.Popen({command!r}, shell=True)\nprint('LAUNCHED')\n"
    if remote(session, code, "LAUNCHED", "score") is None:
        return None
    began = time.time()
    while time.time() - began < 2400:
        time.sleep(30)
        answer = shell(session, f"cat /content/l16v2/scores/{label}.status 2>/dev/null; echo; ls /content/l16v2/scores/{label}.DONE "
                                f"/content/l16v2/scores/{label}.FAILED 2>/dev/null; tail -n 2 /content/l16v2/score-{label}.log | cut -c1-160", "score-wait")
        if answer is None:
            continue
        first = next((line for line in answer.splitlines() if line.startswith("stage=")), "stage=val_agreement")
        plog(f"{first} worker={session}")
        if f"{label}.FAILED" in answer:
            plog(f"PROBLEM worker={session} validation scoring {label} failed: {answer.strip()[-200:]}")
            return None
        if f"{label}.DONE" in answer:
            product = get(session, f"score-{label}.tar.gz")
            if product is None:
                return None
            extract(product, COLAB / "scores")
            return json.loads((COLAB / "scores" / label / "scores.json").read_text(encoding="utf-8"))
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sessions", required=True, help="Colab sessions, comma separated; each is one worker")
    parser.add_argument("--branch", required=True)
    parser.add_argument("--budget-s", type=float, default=2400.0)
    parser.add_argument("--began-utc", help="when the first trial started, if it was started before this driver")
    parser.add_argument("--skip-search", action="store_true", help="the search is over: finalists and the choice only")
    parser.add_argument("--skip-held-out", action="store_true")
    args = parser.parse_args()
    slots = {item.split(":")[0]: int(item.split(":")[1]) if ":" in item else 1 for item in args.sessions.split(",") if item}
    sessions = list(slots)
    commit = subprocess.run(["git", "rev-parse", args.branch], capture_output=True, text=True, cwd=REPO).stdout.strip()
    if not commit:
        plog(f"PROBLEM bo driver refuses to start: no commit for branch {args.branch}")
        return 1
    bundle = COLAB / "l16v2.bundle"
    bundle.unlink(missing_ok=True)
    subprocess.run(["git", "bundle", "create", str(bundle).replace("\\", "/"), args.branch], capture_output=True, cwd=REPO)
    if not bundle.exists():
        plog("PROBLEM bo driver: git bundle failed")
        return 1
    for session in sessions:
        (COLAB / f"keepalive.{session}.stop").write_text("driver\n", encoding="ascii")
    (COLAB / "keepalive.stop").write_text("driver\n", encoding="ascii")
    plog(f"bo driver start sessions={','.join(sessions)} branch={args.branch} commit={commit[:7]} budget_s={int(args.budget_s)} "
         f"study={str(BO / 'study.db')}")
    search = Search(args)
    search.slots = slots
    ready: dict[str, bool] = {}

    def worker_ids(session: str) -> list[str]:    # the first worker of a session carries the session's own name
        return [session if slot == 1 else f"{session}#{slot}" for slot in range(1, slots.get(session, 1) + 1)]

    def join(session: str) -> None:
        """Set the session up if it needs it, then work trials until the study stops proposing."""
        state = session_ready(session, commit)
        if state in ("ready", "stale"):     # a stale clone runs trials as it is; it is brought to the commit before a finalist
            ready[session] = True
        elif state == "new":
            plog(f"worker={session} is a fresh VM: setting it up (model, pinned stack, clone, dataset)")
            ready[session] = prepare(session, args.branch, commit, bundle, full=True)
        else:
            ready[session] = False
        if not ready[session]:
            plog(f"PROBLEM worker={session} did not join the search ({state})")
            return
        with search.lock:
            search.workers.extend(worker_ids(session))
        plog(f"decision session={session} joins the search ({state}) with {slots.get(session, 1)} worker(s) on its card; "
             f"workers now: {','.join(search.workers)}")
        if not args.skip_search:
            loops = [threading.Thread(target=search.worker, args=(worker,), daemon=True) for worker in worker_ids(session)]
            for loop in loops:
                loop.start()
                time.sleep(5)
            for loop in loops:
                loop.join()

    joined: dict[str, threading.Thread] = {}

    def start(session: str) -> None:
        joined[session] = threading.Thread(target=join, args=(session,), daemon=True)
        joined[session].start()

    for session in list(sessions):
        start(session)
        time.sleep(3)
    while any(thread.is_alive() for thread in joined.values()):
        time.sleep(10)
        try:    # a session granted later is named in this file and joins as a worker
            extra = (BO / "extra-sessions.txt").read_text(encoding="ascii").split()
        except OSError:
            extra = []
        for item in extra:
            session = item.split(":")[0]
            if session not in joined:
                sessions.append(session)
                slots[session] = int(item.split(":")[1]) if ":" in item else 1
                plog(f"decision session {session} was granted later: it is set up and joins the search")
                start(session)
    search.table()
    search.save()

    # --- the finalists: the three best configurations at full length, at the same time, one per session ----------
    completed = [t for t in search.trials if t.get("ok") and t.get("outcome") != "pruned" and t.get("value") is not None
                 and t.get("max_steps") == FIDELITY_STEPS]
    top = sorted(completed, key=lambda t: (t["value"], t["number"]))[:FINALISTS]
    # one finalist per free trainer place, a place on each session first, then the second places
    workers = [worker for slot in range(max(slots.values())) for session in sessions
               if ready.get(session) and slot < slots.get(session, 1) for worker in [worker_ids(session)[slot]]]
    refreshed: dict[str, bool] = {}
    refresh_lock = threading.Lock()
    if not top or not workers:
        plog("PROBLEM no completed trial or no worker: no finalist can train; the fallback adapter (grid run 1) stands")
    plog("decision finalists: " + " | ".join(
        f"{place}: trial {t['number']} ({t['kind']}, seen {t['seen']}) value={fmt(t['value'])} lr={t['lr']:.3g} rank={t['rank']} "
        f"alpha={t['alpha']} drop={t['dropout']:.3f} warm={t['warmup_ratio']:.3f}" for place, t in enumerate(top, 1))
        + f"; on {len(workers)} session(s)")
    finals: list[dict[str, Any]] = []
    queue = list(enumerate(top, 1))
    final_lock = threading.Lock()

    def finalist_worker(worker: str) -> None:
        session = worker.split("#")[0]
        with refresh_lock:      # once per session: its clone at this commit (the full-length flags need it)
            if session not in refreshed:
                refreshed[session] = session_ready(session, commit) == "ready" or prepare(session, args.branch, commit, bundle, full=False)
        if not refreshed[session]:
            plog(f"PROBLEM worker={worker} could not be brought to commit {commit[:7]} for a finalist")
            return
        while True:
            with final_lock:
                if not queue:
                    return
                place, trial = queue.pop(0)
            name = f"final{place}-" + trial["name"]
            flags = ["--max-steps", str(FULL_STEPS), "--eval-steps", str(FULL_EVAL), "--early-stop", str(FULL_PATIENCE),
                     "--early-stop-min-rel", str(FULL_MIN_REL), "--lr", f"{trial['lr']:.6g}", "--rank", str(trial["rank"]),
                     "--alpha", str(trial["alpha"]), "--dropout", f"{trial['dropout']:.4f}",
                     "--warmup-ratio", f"{trial['warmup_ratio']:.4f}"]

            def line(status: dict[str, Any], place: int = place, trial: dict[str, Any] = trial) -> str:
                latest = status["val"][max(status["val"])] if status["val"] else None
                return (f"stage=finalists final={place}/{len(top)} from_trial={trial['number']} worker={worker} lr={trial['lr']:.3g} "
                        f"rank={trial['rank']} alpha={trial['alpha']} drop={trial['dropout']:.3f} warm={trial['warmup_ratio']:.3f} "
                        f"step={status['step']}/{FULL_STEPS} loss={fmt(status['loss'])} val={fmt(latest)} "
                        f"best_val={fmt(min(status['val'].values()) if status['val'] else None)} step_s={fmt(status['step_s'], 2)}")

            fact = run_on(session, name, flags, True, line)
            fact.update(place=place, from_trial=trial["number"], trial_kind=trial["kind"], trial_value=trial["value"],
                        lr=trial["lr"], rank=trial["rank"], alpha=trial["alpha"], alpha_mult=trial["alpha_mult"],
                        dropout=trial["dropout"], warmup_ratio=trial["warmup_ratio"], max_steps=FULL_STEPS, full_length=True,
                        worker=worker, concurrent_on_card=min(slots.get(session, 1), len(top)))
            adapter = COLAB / "runs" / name / "adapter" / "adapter_model.safetensors"
            if fact.get("ok") and adapter.exists():
                fact["adapter_path"] = str(adapter.parent).replace("\\", "/")
                fact["adapter_sha256"] = sha256_of(adapter)
                plog(final_adapter_line(fact))
            else:
                plog(f"PROBLEM finalist {place} ({name}) on worker={worker} did not give an adapter: outcome={fact.get('outcome')} "
                     f"error={str(fact.get('error'))[:120]}")
            with final_lock:
                finals.append(fact)

    threads = [threading.Thread(target=finalist_worker, args=(worker,), daemon=True) for worker in workers[:max(1, len(top))]]
    for thread in threads:
        thread.start()
        time.sleep(3)
    for thread in threads:
        thread.join()

    # --- the choice, by the frozen rule, among the full-length runs (the finalists and the grid's run 1) -----------
    seed = next((t for t in search.trials if t["kind"] == "seed" and t.get("full_length")), None)
    candidates = [f for f in finals if f.get("ok") and f.get("best_val") is not None and f.get("adapter_path")]
    if seed is not None:
        fallback = COLAB / "x-fallback-r01"
        if (COLAB / "fallback-adapter-r01.tar.gz").exists():
            extract(COLAB / "fallback-adapter-r01.tar.gz", fallback)
            seed = {**seed, "place": 9, "adapter_path": str(fallback / "adapter").replace("\\", "/"),
                    "adapter_sha256": sha256_of(fallback / "adapter" / "adapter_model.safetensors"),
                    "product_sha256": sha256_of(COLAB / "fallback-adapter-r01.tar.gz"), "worker": sessions[0]}
            plog(final_adapter_line(seed))
            candidates.append(seed)
    if not candidates:
        plog("PROBLEM no full-length adapter exists; nothing to choose")
        return 2
    ranked = sorted(candidates, key=lambda f: (f["best_val"], f.get("best_passes") or 0, f.get("place", 9)))
    scored = ranked[:TOP]
    plog("decision choice rule, step 1: the full-length runs by validation loss: " + " | ".join(
        f"{f['name']} val={fmt(f['best_val'])} at step {f.get('best_step')}" for f in ranked) + f"; the {len(scored)} lowest are scored")
    by_session: dict[str, list[str]] = {}
    for f in scored:
        by_session.setdefault(f["worker"].split("#")[0], []).append(f["name"])
    by_session.setdefault(scored[0]["worker"].split("#")[0], []).insert(0, "base")
    results: dict[str, dict[str, Any]] = {}

    def score_worker(session: str, arms: list[str]) -> None:
        answer = score_on(session, "val-" + session, arms)
        if answer:
            with final_lock:
                results.update(answer)

    threads = [threading.Thread(target=score_worker, args=item, daemon=True) for item in by_session.items()]
    for thread in threads:
        thread.start()
        time.sleep(3)
    for thread in threads:
        thread.join()
    for f in scored:
        f["val_agreement"] = results.get(f["name"])
    with_score = [f for f in scored if f.get("val_agreement")]
    if not with_score:
        plog("PROBLEM validation agreement could not be scored for any full-length run; the choice falls back to the lowest validation loss")
        best = ranked[0]
    else:
        best = sorted(with_score, key=lambda f: (-f["val_agreement"]["correct"], f["best_val"], f.get("best_passes") or 0))[0]
    stop_step = best.get("best_step") or best["max_steps"]
    chosen = {
        "rule": BO_RULE, "dataset_id": "tool-calling-v2", "split_sha256": SPLIT_SHA256,
        "validation_examples": 126, "validation_groups": 13, "base_validation_agreement": results.get("base"),
        "decode": "transformers greedy, bf16 weights with the adapter attached, max 512 new tokens, prompts of equal "
                  "token length batched (no padding), on the Colab A100 that trained the adapter",
        "full_length_runs": [{k: f.get(k) for k in ("name", "place", "from_trial", "trial_kind", "trial_value", "lr", "rank", "alpha",
                                                    "dropout", "warmup_ratio", "steps", "passes", "best_step", "best_passes",
                                                    "best_val", "final_val", "wall_s", "worker", "val_agreement", "adapter_path",
                                                    "adapter_sha256", "product_sha256", "device", "settings")} for f in ranked],
        "top": [{"name": f["name"], "best_val": f["best_val"], "val_agreement": f.get("val_agreement")} for f in scored],
        "chosen": {"name": best["name"], "lr": best["lr"], "rank": best["rank"], "alpha": best["alpha"],
                   "dropout": best["dropout"], "warmup_ratio": best["warmup_ratio"], "max_steps": best["max_steps"],
                   "best_step": stop_step, "best_passes": best.get("best_passes") or stop_step * EFFECTIVE_BATCH,
                   "best_val": best.get("best_val"), "final_val": best.get("final_val"), "passes": best.get("passes"),
                   "wall_s": best.get("wall_s"), "val_agreement": best.get("val_agreement"), "worker": best.get("worker"),
                   "from_trial": best.get("from_trial"), "trial_kind": best.get("trial_kind"),
                   "adapter_path": best.get("adapter_path"), "adapter_sha256": best.get("adapter_sha256")},
        "differences_a100_vs_pod": "none in the recipe: batch 4 by accumulation 4, gradient checkpointing and the torch "
                                   "fallback kernels on both; the chip and the torch build differ (CUDA here, ROCm there)",
        "recipe_flags": ["--max-steps", str(best["max_steps"]), "--stop-step", str(stop_step), "--lr", f"{best['lr']:.6g}",
                         "--rank", str(best["rank"]), "--alpha", str(best["alpha"]), "--dropout", f"{best['dropout']:.4f}",
                         "--warmup-ratio", f"{best['warmup_ratio']:.4f}", "--batch", "4", "--grad-accum", "4", "--seed", "42"],
        "made_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (BO / "chosen.json").write_text(json.dumps(chosen, indent=2) + "\n", encoding="utf-8")
    search.save()
    base = results.get("base") or {}
    plog(f"RESULT validation tool-call agreement (126 examples, 13 groups; A100, bf16): base {base.get('correct')}/{base.get('n')}; "
         + "; ".join(f"{f['name']} {f['val_agreement']['correct']}/{f['val_agreement']['n']}" for f in with_score))
    plog(final_adapter_line(best, chosen=True))
    plog(f"RESULT chosen recipe: {' '.join(chosen['recipe_flags'])}; chosen.json at {str(BO / 'chosen.json')}")

    # --- after the choice: base against chosen on the held-out side of both datasets, generated on the A100 --------
    if not args.skip_held_out and best.get("worker", "").split("#")[0] in sessions:
        session = best["worker"].split("#")[0]
        command = (f"cd /content/Knighthacks26 && setsid nohup python3 /content/vm_score.py held_out {best['name']} "
                   f"> /content/l16v2/score-held-out.log 2>&1 &")
        code = f"import subprocess\nsubprocess.Popen({command!r}, shell=True)\nprint('LAUNCHED')\n"
        if remote(session, code, "LAUNCHED", "held-out") is not None:
            began = time.time()
            while time.time() - began < 2400:
                time.sleep(30)
                answer = shell(session, "cat /content/l16v2/status.txt 2>/dev/null; echo; ls /content/l16v2/markers/HELD_OUT_DONE "
                                        "/content/l16v2/scores/held_out.FAILED 2>/dev/null; grep -a held_out_bf16 /content/l16v2/finished.txt 2>/dev/null",
                               "held-out-wait")
                if answer is None:
                    continue
                first = next((line for line in answer.splitlines() if line.startswith("stage=")), "stage=held_out_bf16")
                plog(f"{first} worker={session}")
                if "held_out.FAILED" in answer:
                    plog(f"PROBLEM worker={session} held-out bf16 scoring failed: {answer.strip()[-200:]}")
                    break
                if "HELD_OUT_DONE" in answer:
                    for row in answer.splitlines():
                        if row.startswith("held_out_bf16 "):
                            plog("RESULT " + row + " engine=transformers_bf16_on_the_A100 (the pod D numbers on the packaged GGUF are the reported ones)")
                    get(session, "held-out-bf16.tar.gz")
                    break
    plog("bo driver done: the study, every trial's files, every full-length adapter and the choice are on this machine; "
         "sessions are left up for the caller to stop after a last check")
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        if code != 0:    # hand every session back to a keepalive, so no VM is left with nothing polling it
            plog(f"PROBLEM bo driver exited with code {code}; the sessions are left up and a keepalive takes each over")
            names = [a for a in sys.argv[1:] if a.startswith("l16v2")] or []
            for index, arg in enumerate(sys.argv):
                if arg == "--sessions" and index + 1 < len(sys.argv):
                    names = sys.argv[index + 1].split(",")
            for name in names:
                (COLAB / f"keepalive.{name}.stop").unlink(missing_ok=True)
                subprocess.Popen([BASH, str(HERE / "colab_keepalive.sh").replace("\\", "/"), name],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 creationflags=getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    sys.exit(code)
