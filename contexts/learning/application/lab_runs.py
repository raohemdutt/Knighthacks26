"""The trainer's own files as events the Learning lab's run story already draws: a real fine-tune on the "Training
run" tab as soon as its files reach the box, with no change to the trainer.

    runs = TrainerRuns(data_dir / "lab_runs")     # one folder per run, or the run's files directly in it
    runs.sync(view)                               # folds what is new into a LabView; cheap to call per request

The shape it reads is the one lab/train.py (L3, merged as PR #252) writes into its --out folder:

    telemetry.jsonl   one `telemetry.train.step` event per optimizer step, as contracts/telemetry.py TrainStep
                      dumps it: {event_id, type, version, ts, node ("pod-b"; "cpu" for a rehearsal), run_id, step,
                      loss (null when not finite), step_ms, socket_w (null with no sampler reading), energy_j (null
                      on the Strix Halo), tokens, received_ts}. Appended a line at a time while the run trains.
    run.json          written once, when the run ends: {ok, outcome ("trained", "failed", "refused", ...), error?,
                      refs {run_id, node, task, dataset_id, action_id, base_model, base_revision, ...},
                      metrics {"trainer.steps", "trainer.batch", "trainer.grad_accum", "trainer.lora_rank",
                               "trainer.examples", "trainer.epochs", "trainer.samples_per_s", ...}}

The step lines are folded as they are: they are already the event the story's loss curve, step count and power trace
are drawn from, and each number's source is that event's id. run.json becomes one `learning.training.started` event
(the optional input lab_view.py documents), so the story can say how many steps were planned, how many samples a
step holds (per-device batch x gradient accumulation), the base model and the LoRA rank. Its event id is
`<folder>/run.json`, so a number that came from it points at the file, not at an event nobody sent.

What this does not know, and so does not say: the eval, the gate's decision and the swap. Those arrive as their own
events (learning.run.evaluated, learning.model.deployed) under the same run_id and join the same story. Until then a
run read from files shows as "training", even after run.json says it ended; `summary()` carries the trainer's outcome.

Assumptions, each in one place:
- where the files are: `<data_dir>/lab_runs/<anything>/telemetry.jsonl`, or ATLAS_LAB_RUNS_DIR. The trainer writes
  them on the pod (default ~/atlas-runs/<run_id>); how they reach the box is not decided in the repo.
  TODO(L16): point `default_runs_dir` at wherever the first real run's folder is synced to.
- a line that is not a TrainStep (half-written, or another event type) is skipped and counted, never guessed at.
"""

import json
import os
from pathlib import Path
from typing import Any
from weakref import WeakKeyDictionary

from pydantic import ValidationError

from contexts.learning.application.lab_view import LabView
from contracts import parse_event
from contracts.events import Event
from contracts.telemetry import TrainStep

STEPS_FILE = "telemetry.jsonl"
RUN_FILE = "run.json"
ENV = "ATLAS_LAB_RUNS_DIR"


def default_runs_dir(data_dir: str | Path) -> Path:
    return Path(os.environ[ENV]) if os.environ.get(ENV) else Path(data_dir) / "lab_runs"


def read_steps(path: Path) -> tuple[list[TrainStep], int]:
    """The step events of one telemetry.jsonl, in file order, and how many lines were not one."""
    steps, skipped = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = parse_event(line)
        except (ValidationError, ValueError):
            skipped += 1
            continue
        if isinstance(event, TrainStep):
            steps.append(event)
        else:
            skipped += 1
    return steps, skipped


def started_event(run: dict[str, Any], steps: list[TrainStep], ref: str) -> Event | None:
    """The trainer's run.json as the `learning.training.started` event the lab view reads. The one adapter between the
    trainer's ledger-row shape (refs, metrics) and the view; None when the file names no run."""
    refs = run.get("refs") if isinstance(run.get("refs"), dict) else {}
    metrics = run.get("metrics") if isinstance(run.get("metrics"), dict) else {}
    run_id = refs.get("run_id") or (steps[0].run_id if steps else None)
    if not run_id:
        return None
    whole = lambda key: int(metrics[key]) if isinstance(metrics.get(key), int | float) else None  # noqa: E731
    batch = whole("trainer.batch")
    accum = whole("trainer.grad_accum")
    first = steps[0] if steps else None
    body: dict[str, Any] = {
        "type": "learning.training.started", "event_id": ref, "run_id": run_id, "task": refs.get("task"),
        "dataset_id": refs.get("dataset_id"), "node": refs.get("node") or (first.node if first else None),
        "base": refs.get("base_model"), "steps": whole("trainer.steps"),
        "batch": batch * (accum or 1) if batch else None, "train_examples": whole("trainer.examples"),
        "epochs": metrics.get("trainer.epochs") if isinstance(metrics.get("trainer.epochs"), int | float) else None,
        "lora_rank": whole("trainer.lora_rank"),
    }
    if first is not None:  # the run began one step before its first step ended
        body["ts"] = first.ts.timestamp() - first.step_ms / 1000
    return parse_event(body)


def trainer_events(folder: Path) -> tuple[list[Event], dict[str, Any]]:
    """Everything one run folder gives the lab view, and a line about it for the API."""
    steps_path, run_path = folder / STEPS_FILE, folder / RUN_FILE
    steps, skipped = read_steps(steps_path) if steps_path.is_file() else ([], 0)
    events: list[Event] = list(steps)
    info: dict[str, Any] = {"folder": folder.name, "run_id": steps[0].run_id if steps else None, "steps": len(steps),
                            "skipped_lines": skipped, "outcome": None, "error": None, "ended": False,
                            "source": f"{folder.name}/{STEPS_FILE}"}
    if run_path.is_file():
        try:
            run = json.loads(run_path.read_text(encoding="utf-8"))
        except ValueError:
            run = None
        if isinstance(run, dict):
            started = started_event(run, steps, f"{folder.name}/{RUN_FILE}")
            if started is not None:
                events.insert(0, started)
                info["run_id"] = info["run_id"] or started.model_dump().get("run_id")
            info.update(outcome=run.get("outcome"), error=run.get("error"), ended=True)
    return events, info


class TrainerRuns:
    def __init__(self, root: str | Path | None) -> None:
        self.root = Path(root) if root else None
        self._seen: dict[str, tuple[float, int, float, int]] = {}  # folder -> (mtime, size) of its two files
        self.runs: dict[str, dict[str, Any]] = {}

    def folders(self) -> list[Path]:
        if self.root is None or not self.root.is_dir():
            return []
        found = [self.root] if (self.root / STEPS_FILE).is_file() else []
        return found + sorted(p for p in self.root.iterdir() if p.is_dir() and (p / STEPS_FILE).is_file())

    def sync(self, view: LabView) -> list[dict[str, Any]]:
        """Fold every run folder that changed since the last call into `view`. Folding the same step twice counts it
        once (the view keys steps by number), so re-reading a growing file is safe."""
        for folder in self.folders():
            stamp = tuple(x for name in (STEPS_FILE, RUN_FILE) for x in _stamp(folder / name))
            if self._seen.get(str(folder)) == stamp:
                continue
            try:
                events, info = trainer_events(folder)
            except OSError:
                continue  # the file is being replaced; the next call reads it
            for event in events:
                view.on_event(event)
            self._seen[str(folder)] = stamp
            self.runs[str(folder)] = info
        return self.summary()

    def summary(self) -> list[dict[str, Any]]:
        return sorted(self.runs.values(), key=lambda r: r["folder"])


def _stamp(path: Path) -> tuple[float, int]:
    try:
        stat = path.stat()
    except OSError:
        return (0.0, -1)
    return (stat.st_mtime, stat.st_size)


# --- one TrainerRuns per LabView, so the API can find it without a new argument at every call site ---

_ATTACHED: "WeakKeyDictionary[LabView, TrainerRuns]" = WeakKeyDictionary()


def attach(view: LabView, root: str | Path | None) -> TrainerRuns:
    _ATTACHED[view] = TrainerRuns(root)
    return _ATTACHED[view]


def sync(view: LabView) -> list[dict[str, Any]]:
    """Fold the attached run folders into `view`; [] when none is attached."""
    runs = _ATTACHED.get(view)
    return runs.sync(view) if runs is not None else []
