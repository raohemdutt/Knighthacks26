"""The Learning lab's read model: one "run story" per training run, for the /lab screen.

    view = LabView(exports_dir=data_dir / "exports", models_dir=data_dir / "models")
    view.subscribe(bus)            # replay=True only: the log rebuilds it; no model calls, no network
    view.runs()                    # [{run_id, task, status, started, ended, scores, model_version}], newest first
    view.story("run-0003")         # the whole story, or None

A story says what was taught, what the model trained on, how training went (loss and power on one
timeline), what changed on the held-out questions, why the gate deployed it or did not, and what
runs on the box now. While the run is still going it also says so (L30): `run_status` is running, finished or
failed, `steps_so_far` the steps the log holds, and `stage_strip` the four stages of the live loop (labeling,
training, packaging, offered), each {stage, state (done, now, waiting, failed, not reported), started, ended,
seconds, message}, lit from the run's own progress events, steps and package. A live-loop act (L29,
learning.loop.act) adds `members`, `pods` (the training stage names them too) and `parts`: the steps of each extra
pod's rank (R65's two-pod LoRA), whose own run id is left out of the run list. It never invents a number:

- every number is a small dict `{label, value, unit, source, low_confidence, reason}`. `source` names
  the event (type and id), the file and line, or how it was derived and from which events;
- a number the log does not hold is `value: null` with `reason` saying what is missing;
- pictures are never read: an example row carries its `frame_ref` (the box only stores blurred
  keyframes) and its caption, never an image path or bytes.

What it folds, all from the event log:

    learning.task.created, learning.dataset.exported, learning.run.progress, learning.run.evaluated,
    learning.model.deployed, agent.action.proposed, workspace.action.completed (the export approval),
    telemetry.train.step, telemetry.gpu.power, telemetry.request, telemetry.joules_per_answer

and four event types that are NOT in contracts/ yet. They are optional inputs: the view uses them
when the log has them and says what is missing when it does not. Their names stay outside the
`learning.run`, `learning.model` and `learning.dataset` prefixes on purpose, because the Teach
screen's LearningBoard folds everything under those as the events it already knows.

    learning.training.started  {task, run_id, dataset_id, node, base, steps, batch, train_examples, epochs, lora_rank}
    learning.eval.reported     {task, run_id, n_items, n_groups, correct{current,candidate,teacher}, noise, noise_runs,
                                p_value, alpha, test, decode, split_sha256, low_confidence, items[...]}
    learning.package.built     {task, run_id, model_version, base, gguf, bytes, sha256, quantization}
    learning.swap.completed    {model_version, previous_version, started, swap_s, outcome, reason}

It also reads, when they exist on this machine, the export the box wrote (`<exports_dir>/<dataset_id>/
examples.jsonl`, `split.json`) and an installed package's `manifest.json`. A `LabView(sample=True)`
folds the committed sample run instead of the bus and marks everything it returns `sample: true`.
"""

import json
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from bus import EventBus
from contracts.agent import ActionProposed
from contracts.events import Event
from contracts.learning import DatasetExample, DatasetExported, ModelDeployed, ModelManifest, RunEvaluated, RunProgress, \
    TaskCreated
from contracts.telemetry import GpuPower, JoulesPerAnswer, RequestWindow, TrainStep
from contracts.workspace import ActionCompleted

ROWS = 12  # example rows a story shows
ALPHA = 0.05  # the gate's p threshold when the report does not carry one (lab/refine/learning/OWNER_RULINGS.md, 1)


# --- the optional inputs (see the module docstring); not registered in contracts/ ---

class _Optional(BaseModel):
    model_config = ConfigDict(extra="ignore")


class TrainingStarted(_Optional):
    TYPE: ClassVar[str] = "learning.training.started"
    run_id: str
    task: str | None = None
    dataset_id: str | None = None
    node: str | None = None
    base: str | None = None
    steps: int | None = None  # optimizer steps planned
    batch: int | None = None  # samples per optimizer step
    train_examples: int | None = None
    epochs: float | None = None
    lora_rank: int | None = None


class EvalAnswer(_Optional):
    answer: str
    correct: bool


class EvalItem(_Optional):
    id: str
    group: str | None = None
    question: str
    expected: str | None = None
    before: EvalAnswer
    after: EvalAnswer


class EvalReported(_Optional):
    TYPE: ClassVar[str] = "learning.eval.reported"
    run_id: str
    task: str | None = None
    n_items: int
    n_groups: int | None = None
    correct: dict[str, int] = {}  # {"current": 21, "candidate": 41, "teacher": 47}
    noise: float | None = None  # run-to-run spread of the score, as a fraction of 1
    noise_runs: list[int] = []  # items right in each repeated run of the current model
    p_value: float | None = None
    alpha: float | None = None
    test: str | None = None
    decode: str | None = None
    split_sha256: str | None = None
    low_confidence: bool = False
    items: list[EvalItem] = []


class PackageBuilt(_Optional):
    TYPE: ClassVar[str] = "learning.package.built"
    run_id: str
    model_version: str
    base: str | None = None
    gguf: str | None = None
    bytes: int | None = None
    sha256: str | None = None
    quantization: str | None = None


class SwapCompleted(_Optional):
    TYPE: ClassVar[str] = "learning.swap.completed"
    model_version: str
    previous_version: str | None = None
    started: datetime | None = None
    swap_s: float | None = None
    outcome: Literal["serving", "rolled_back"] = "serving"
    reason: str | None = None


class LoopAct(_Optional):
    """The live loop's act (L29): who pooled, which pods train it, and the runs that are its parts (one per extra pod:
    R65's rank 1 on pod C writes its steps under its own run id). Published by lab_import kind=stage from the act's
    start line; not in contracts/, like the others here."""
    TYPE: ClassVar[str] = "learning.loop.act"
    run_id: str
    act: str | None = None
    members: int | None = None
    pods: list[str] = []
    parts: dict[str, str] = {}   # pod -> the run id its rank's steps are posted under


OPTIONAL = {m.TYPE: m for m in (TrainingStarted, EvalReported, PackageBuilt, SwapCompleted, LoopAct)}

SUGGESTED_EVENTS = [
    {"type": TrainingStarted.TYPE, "gives": "total steps, samples per step and the trainer's node, so samples/s and joules per "
                                            "training sample can be computed",
     "fields": "task, run_id, dataset_id, node, base, steps, batch, train_examples, epochs, lora_rank"},
    {"type": EvalReported.TYPE, "gives": "n as items and groups, run-to-run noise, the p-value and each held-out question's "
                                         "before and after answer",
     "fields": "task, run_id, n_items, n_groups, correct, noise, noise_runs, p_value, alpha, test, decode, split_sha256, "
               "low_confidence, items[{id, group, question, expected, before{answer, correct}, after{answer, correct}}]"},
    {"type": PackageBuilt.TYPE, "gives": "the model file, its size, checksum and quantization",
     "fields": "task, run_id, model_version, base, gguf, bytes, sha256, quantization"},
    {"type": SwapCompleted.TYPE, "gives": "how long the hot-swap took and whether the box went back to the old model",
     "fields": "model_version, previous_version, started, swap_s, outcome, reason"},
]


# --- sources and numbers ---

def by_event(e: Event) -> dict[str, Any]:
    return {"kind": "event", "ref": f"{e.type} {e.event_id}", "ids": [e.event_id]}


def by_file(path: str, line: int | None = None) -> dict[str, Any]:
    return {"kind": "file", "ref": f"{path}:{line}" if line else path, "ids": []}


def derived(how: str, *events: Event) -> dict[str, Any]:
    ids = [e.event_id for e in events]
    return {"kind": "derived", "ref": how, "ids": ids[:3] + (["..."] if len(ids) > 3 else []), "events": len(ids)}


def num(label: str, value: Any, unit: str | None, source: dict | None, *, low_confidence: bool = False,
        reason: str | None = None) -> dict[str, Any]:
    """One number with where it came from; `value` None needs a `reason`."""
    if value is None and not reason:
        raise ValueError(f"{label}: a missing number needs a reason")
    return {"label": label, "value": value, "unit": unit, "source": source if value is not None else None,
            "low_confidence": bool(low_confidence), "reason": reason if value is None else None}


def missing(label: str, unit: str | None, reason: str) -> dict[str, Any]:
    return num(label, None, unit, None, reason=reason)


def collect_numbers(obj: Any) -> list[dict[str, Any]]:
    """Every number in a story, in reading order: what the screen's sources drawer lists."""
    out: list[dict[str, Any]] = []
    if isinstance(obj, dict):
        if {"label", "value", "source", "reason"} <= obj.keys():
            return [obj]
        for v in obj.values():
            out.extend(collect_numbers(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(collect_numbers(v))
    return out


def mark_sample(obj: Any) -> Any:
    """`sample: true` on every dict of a payload that came from the sample run."""
    if isinstance(obj, dict):
        return {**{k: mark_sample(v) for k, v in obj.items()}, "sample": True}
    if isinstance(obj, list):
        return [mark_sample(v) for v in obj]
    return obj


def _secs(a: datetime, b: datetime) -> float:
    return round((b - a).total_seconds(), 3)


class LabView:
    def __init__(self, exports_dir: str | Path | None = None, models_dir: str | Path | None = None, *, sample: bool = False,
                 keep: int = 20_000) -> None:
        self.exports_dir = Path(exports_dir) if exports_dir else None
        self.models_dir = Path(models_dir) if models_dir else None
        self.sample = sample
        self.tasks: dict[str, TaskCreated] = {}
        self.proposals: dict[str, ActionProposed] = {}  # export approvals asked for, by action_id
        self.completions: dict[str, ActionCompleted] = {}
        self.exports: list[DatasetExported] = []
        self.run_ids: list[str] = []  # in the order the log first named them
        self.progress: dict[str, list[RunProgress]] = {}
        self.evaluated: dict[str, RunEvaluated] = {}
        self.steps: dict[str, dict[int, TrainStep]] = {}
        self.started: dict[str, tuple[Event, TrainingStarted]] = {}
        self.reports: dict[str, tuple[Event, EvalReported]] = {}
        self.packages: dict[str, tuple[Event, PackageBuilt]] = {}
        self.swaps: list[tuple[Event, SwapCompleted]] = []
        self.loop_acts: dict[str, tuple[Event, LoopAct]] = {}
        self.deployments: list[ModelDeployed] = []
        self.gpu: dict[str, deque[GpuPower]] = {}
        self.answers: deque[JoulesPerAnswer] = deque(maxlen=keep)
        self.requests: deque[RequestWindow] = deque(maxlen=keep)
        self.keep = keep
        self.rejected: list[dict[str, str]] = []  # optional events that did not have the fields this view needs

    def subscribe(self, bus: EventBus) -> None:
        for prefix in ("learning.", "telemetry.train.step", "telemetry.gpu.power", "telemetry.request",
                       "telemetry.joules_per_answer", "agent.action.proposed", "workspace.action.completed"):
            bus.subscribe(prefix, self.on_event, replay=True)

    # --- the fold ---

    def _run(self, run_id: str) -> None:
        if run_id not in self.run_ids:
            self.run_ids.append(run_id)

    def on_event(self, e: Event) -> None:
        if isinstance(e, TaskCreated):
            self.tasks[e.task] = e
        elif isinstance(e, ActionProposed):
            if e.kind == "export_training_data":
                self.proposals[e.action_id] = e
        elif isinstance(e, ActionCompleted):
            if e.action_id in self.proposals:
                self.completions[e.action_id] = e
        elif isinstance(e, DatasetExported):
            self.exports.append(e)
        elif isinstance(e, RunProgress):
            self._run(e.run_id)
            self.progress.setdefault(e.run_id, []).append(e)
        elif isinstance(e, RunEvaluated):
            self._run(e.run_id)
            self.evaluated[e.run_id] = e
        elif isinstance(e, ModelDeployed):
            self.deployments.append(e)
        elif isinstance(e, TrainStep):
            self._run(e.run_id)
            self.steps.setdefault(e.run_id, {})[e.step] = e  # a step sent twice counts once
        elif isinstance(e, GpuPower):
            self.gpu.setdefault(e.node, deque(maxlen=self.keep)).append(e)
        elif isinstance(e, JoulesPerAnswer):
            self.answers.append(e)
        elif isinstance(e, RequestWindow):
            self.requests.append(e)
        elif e.type in OPTIONAL:
            self._on_optional(e)

    def _on_optional(self, e: Event) -> None:
        try:
            body = OPTIONAL[e.type].model_validate(e.model_dump())
        except ValidationError as err:
            self.rejected.append({"type": e.type, "event_id": e.event_id,
                                  "problem": "; ".join(".".join(map(str, x["loc"])) + ": " + x["msg"] for x in err.errors()[:4])})
            return
        if isinstance(body, SwapCompleted):
            self.swaps.append((e, body))
            return
        self._run(body.run_id)
        {TrainingStarted: self.started, EvalReported: self.reports, PackageBuilt: self.packages,
         LoopAct: self.loop_acts}[type(body)][body.run_id] = (e, body)

    def part_of(self, run_id: str) -> str | None:
        """The act this run is a part of (a rank on another pod), or None."""
        return next((parent for parent, (_, act) in self.loop_acts.items() if run_id in act.parts.values()), None)

    # --- reads ---

    def runs(self) -> list[dict[str, Any]]:
        """One line per run, newest first."""
        out = []
        for run_id in self.run_ids:
            if self.part_of(run_id):   # a rank on another pod shows inside its act's story, not as a run of its own
                continue
            s = self.story(run_id)
            out.append({"run_id": run_id, "task": s["task"]["slug"], "status": s["status"], "started": s["marks"]["run_start"],
                        "ended": s["marks"]["end"], "scores": s["eval"]["scores"], "model_version": s["eval"]["model_version"],
                        "verdict": s["eval"]["verdict"], "run_status": s["run_status"]})
        out.sort(key=lambda r: (r["started"] is not None, r["started"]), reverse=True)
        return mark_sample(out) if self.sample else out

    def story(self, run_id: str) -> dict[str, Any] | None:
        if run_id not in self.run_ids:
            return None
        progress = sorted(self.progress.get(run_id, []), key=lambda p: p.ts)
        steps = [s for _, s in sorted(self.steps.get(run_id, {}).items())]
        evaluated = self.evaluated.get(run_id)
        started = self.started.get(run_id)
        report = self.reports.get(run_id)
        package = self.packages.get(run_id)

        slug = next((x.task for x in (evaluated, *progress, started[1] if started else None, report[1] if report else None)
                     if x is not None and x.task), None)
        first = min([p.ts for p in progress] + [s.ts for s in steps] + ([started[0].ts] if started else [])
                    + ([evaluated.ts] if evaluated else []), default=None)
        task = self._task(slug)
        dataset, export = self._dataset(slug, started[1].dataset_id if started else None, first, started, report)
        training, window = self._training(progress, steps, started)
        power = self._power(steps, started, window, training)
        ev = self._eval(evaluated, report)
        version = evaluated.model_version if evaluated else (package[1].model_version if package else None)
        deploy, deployed, swap = self._deploy(version, evaluated)
        status, phase = self._status(progress, evaluated, deploy)
        ev["verdict"], ev["verdict_detail"] = self._verdict(evaluated, report, ev, status)

        created = self.tasks.get(slug) if slug else None
        ends = [x for x in (progress[-1].ts if progress else None, steps[-1].ts if steps else None,
                            evaluated.ts if evaluated else None, package[0].ts if package else None,
                            deployed.ts if deployed else None, swap[0].ts if swap else None) if x]
        marks = {
            "start": min((x for x in (created.ts if created else None, export.ts if export else None, first) if x), default=None),
            "run_start": first,  # the run's own first event; `start` also covers the teaching and the export before it
            "taught": created.ts if created else None,
            "exported": export.ts if export else None,
            "training_start": window[0] if window else None,
            "training_end": window[1] if window else None,
            "evaluated": evaluated.ts if evaluated else (report[0].ts if report else None),
            "packaged": package[0].ts if package else None,
            "deployed": deployed.ts if deployed else None,
            "end": max(ends) if ends else None,
        }
        wall = (num("Start to finish", _secs(first, marks["end"]), "s",
                    derived("last event of the run minus its first (event time); the export and its approval come before"))
                if first and marks["end"] and status in ("deployed", "not deployed", "rolled back", "failed")
                else missing("Start to finish", "s", "the run has not finished"))
        story = {
            "run_id": run_id, "sample": self.sample, "status": status, "phase": phase,
            "task": task, "marks": marks, "wall_s": wall,
            "stages": [{"stage": p.stage, "progress": p.progress, "ts": p.ts, "message": p.message, "event_id": p.event_id}
                       for p in progress],
            "dataset": dataset, "training": training, "power": power, "eval": ev,
            "package": self._package(package, version),
            "deploy": deploy,
            "box": self._box(deployed, swap),
            "ignored_events": [r for r in self.rejected],
        }
        story["stage_strip"] = self._strip(progress, steps, evaluated, package, deployed)
        story.update(self._loop(run_id, story["stage_strip"]))
        story["run_status"] = _run_state(status, progress, story["stage_strip"])
        story["steps_so_far"] = len(steps)
        story["gaps"] = [{"label": n["label"], "reason": n["reason"]} for n in collect_numbers(story) if n["value"] is None]
        return mark_sample(story) if self.sample else story

    # --- the parts of a story ---

    def _task(self, slug: str | None) -> dict[str, Any]:
        created = self.tasks.get(slug) if slug else None
        return {"slug": slug, "instruction": created.instruction if created else None,
                "instruction_reason": None if created else (
                    "no learning.task.created event names this run's task" if slug else "no learning event names this run's task"),
                "created": created.ts if created else None, "source": by_event(created) if created else None}

    def _dataset(self, slug: str | None, dataset_id: str | None, first: datetime | None,
                 started: tuple[Event, TrainingStarted] | None, report: tuple[Event, EvalReported] | None
                 ) -> tuple[dict[str, Any], DatasetExported | None]:
        """The export this run trained on: the one the trainer names, else the task's newest export before the run."""
        candidates = [x for x in self.exports if (dataset_id and x.dataset_id == dataset_id) or
                      (not dataset_id and slug and x.task == slug and (first is None or x.ts <= first))]
        export = candidates[-1] if candidates else None
        dataset_id = export.dataset_id if export else dataset_id
        no_export = "the log has no learning.dataset.exported event for this run's task"
        out: dict[str, Any] = {
            "dataset_id": dataset_id,
            "examples": num("Examples exported", export.examples, "examples", by_event(export)) if export
            else missing("Examples exported", "examples", no_export),
            "approval": self._approval(export),
        }
        files = self._dataset_files(dataset_id)
        split = files.get("split")
        if split:
            src = by_file(files["split_path"])
            out["train"] = num("Examples trained on", len(split["train"]), "examples", src)
            out["held_out"] = num("Examples held out for the test", len(split["held_out"]), "examples", src)
        else:
            why = files.get("split_reason") or "this box has no split.json for the dataset"
            out["train"] = (num("Examples trained on", started[1].train_examples, "examples", by_event(started[0]))
                            if started and started[1].train_examples is not None else missing("Examples trained on", "examples", why))
            out["held_out"] = (num("Examples held out for the test", report[1].n_items, "examples", by_event(report[0]))
                               if report else missing("Examples held out for the test", "examples", why))
        out["held_out_groups"] = (num("Different cases the held-out questions stand on", report[1].n_groups, "groups", by_event(report[0]))
                                  if report and report[1].n_groups is not None else
                                  missing("Different cases the held-out questions stand on", "groups",
                                          "no learning.eval.reported event says how many groups the held-out set spans"))
        out["rows"] = files.get("rows")
        out["rows_reason"] = files.get("rows_reason")
        out["rows_source"] = by_file(files["rows_path"]) if files.get("rows") else None
        return out, export

    def _approval(self, export: DatasetExported | None) -> dict[str, Any]:
        if export is None:
            return {"action_id": None, "asked": None, "approved": None, "reason": None, "approved_by": None,
                    "approved_by_reason": "no export, so no approval", "source": None}
        asked = self.proposals.get(export.action_id)
        done = self.completions.get(export.action_id)
        ok = done is not None and done.status in ("approved", "done")
        return {
            "action_id": export.action_id,
            "asked": asked.ts if asked else None,
            "reason": asked.reason if asked else None,
            # the export only runs after the card is approved, so its completion is the approval the log holds
            "approved": done.ts if ok else None,
            "approved_reason": None if ok else "the log has no workspace.action.completed event for this export's approval",
            "approved_by": None,
            "approved_by_reason": "approval events do not name a person; cards are approved on the owner's phone",
            "source": by_event(done) if ok else by_event(export),
        }

    def _dataset_files(self, dataset_id: str | None) -> dict[str, Any]:
        """A few rows and the split, read from the export folder when this machine has it. Never an image."""
        if not dataset_id or self.exports_dir is None:
            return {"rows_reason": "this box has no export folder for the dataset"}
        if "/" in dataset_id or "\\" in dataset_id or dataset_id.startswith("."):
            return {"rows_reason": "the dataset id is not a folder name"}
        root = self.exports_dir / dataset_id
        path = root / "examples.jsonl"
        if not path.is_file():
            return {"rows_reason": f"{dataset_id}/examples.jsonl is not on this box (the export may live on another machine)"}
        out: dict[str, Any] = {}
        side: dict[str, str] = {}
        split_path = root / "split.json"
        if split_path.is_file():
            try:
                split = json.loads(split_path.read_text())
                out["split"] = {"train": list(split["train"]), "held_out": list(split["held_out"])}
                out["split_path"] = f"{dataset_id}/split.json"
                side = {**{i: "train" for i in out["split"]["train"]}, **{i: "held_out" for i in out["split"]["held_out"]}}
            except (ValueError, KeyError, TypeError):
                out["split_reason"] = f"{dataset_id}/split.json could not be read"
        rows, total = [], 0
        with path.open(encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                total += 1
                try:
                    ex = DatasetExample.model_validate_json(line)
                except ValidationError:
                    continue
                where = side.get(ex.id)
                if len(rows) < ROWS and where != "held_out":  # show what it learned from; held-out rows are in the eval
                    rows.append({"id": ex.id, "frame_ref": ex.frame_ref, "ts": ex.ts, "context": ex.context, "target": ex.target,
                                 "verdict": ex.verdict, "side": where, "line": n})
        out["rows"], out["rows_path"], out["rows_total"] = rows, f"{dataset_id}/examples.jsonl", total
        return out

    def _training(self, progress: list[RunProgress], steps: list[TrainStep], started: tuple[Event, TrainingStarted] | None
                  ) -> tuple[dict[str, Any], tuple[datetime, datetime] | None]:
        cfg = started[1] if started else None
        stage = [p for p in progress if p.stage == "training"]
        later = [p for p in progress if stage and p.ts > stage[-1].ts and p.stage != "training"]
        window: tuple[datetime, datetime] | None = None
        if steps:
            t0 = started[0].ts if started else (stage[0].ts if stage and stage[0].ts < steps[0].ts else steps[0].ts)
            window = (min(t0, steps[0].ts), steps[-1].ts)
            points = [{"t": _secs(window[0], s.ts), "ts": s.ts, "step": s.step, "loss": s.loss, "event_id": s.event_id} for s in steps]
            points_source = derived("one point per telemetry.train.step", *steps)
        elif stage:
            window = (stage[0].ts, later[0].ts if later else stage[-1].ts)
            points = [{"t": _secs(window[0], p.ts), "ts": p.ts, "step": None, "loss": p.loss, "progress": p.progress,
                       "event_id": p.event_id} for p in stage if p.loss is not None]
            points_source = derived("one point per learning.run.progress in the training stage", *stage) if points else None
        else:
            points, points_source = [], None
        no_points = None if points else ("the log has no telemetry.train.step and no learning.run.progress with a loss for this run: "
                                         "only that it started and ended" if window else "the log has no training events for this run")
        losses = [p for p in points if p["loss"] is not None]
        wall = _secs(*window) if window else None
        n_steps = len(steps) if steps else None
        of = cfg.steps if cfg else None
        samples = n_steps * cfg.batch if n_steps and cfg and cfg.batch else None
        no_batch = ("no event says how many samples a training step holds (learning.training.started.batch)" if steps
                    else "the log has no telemetry.train.step events for this run")
        src_steps = derived("count of telemetry.train.step events", *steps) if steps else None
        out = {
            "node": steps[0].node if steps else (cfg.node if cfg else None),
            "base": cfg.base if cfg else None, "lora_rank": cfg.lora_rank if cfg else None, "epochs": cfg.epochs if cfg else None,
            "batch": cfg.batch if cfg else None, "config_source": by_event(started[0]) if started else None,
            "config_reason": None if started else "no learning.training.started event: the trainer's settings are not in the log",
            "steps": num("Training steps done", n_steps, "steps", src_steps) if steps else missing("Training steps done", "steps", no_batch),
            "of": num("Training steps planned", of, "steps", by_event(started[0])) if of is not None
            else missing("Training steps planned", "steps", "no learning.training.started event says how many steps were planned"),
            "wall_s": num("Training time", wall, "s", derived("last training event minus the first (event time)"))
            if wall is not None else missing("Training time", "s", "the log has no training events for this run"),
            "samples": num("Training samples seen", samples, "samples", derived("steps x batch", started[0]))
            if samples else missing("Training samples seen", "samples", no_batch),
            "samples_per_s": num("Training speed", round(samples / wall, 3), "samples/s", derived("samples / training time", started[0]))
            if samples and wall else missing("Training speed", "samples/s", no_batch),
            "loss_first": num("Loss at the first point", losses[0]["loss"], "loss", {"kind": "event", "ref": f"point {losses[0]['event_id']}",
                                                                                  "ids": [losses[0]["event_id"]]})
            if losses else missing("Loss at the first point", "loss", no_points or "no point carries a loss"),
            "loss_last": num("Loss at the last point", losses[-1]["loss"], "loss", {"kind": "event", "ref": f"point {losses[-1]['event_id']}",
                                                                                 "ids": [losses[-1]["event_id"]]})
            if losses else missing("Loss at the last point", "loss", no_points or "no point carries a loss"),
            "points": points, "points_source": points_source, "points_reason": no_points,
        }
        return out, window

    def _power(self, steps: list[TrainStep], started: tuple[Event, TrainingStarted] | None,
               window: tuple[datetime, datetime] | None, training: dict[str, Any]) -> dict[str, Any]:
        """Watts over the training window and their integral. Pod samples when the log has them across the window, else
        each step's own reading (its energy counter, or its mean watts times its length)."""
        labels = ("Energy used by training", "Average power while training", "Peak power while training", "Energy per training sample")
        if window is None:
            why = "the log has no training events for this run"
            return {"points": [], "points_reason": why, "node": None,
                    **dict(zip(("energy_j", "mean_w", "peak_w", "j_per_sample"),
                               (missing(lab, u, why) for lab, u in zip(labels, ("J", "W", "W", "J/sample")))))}
        node = training["node"]
        t0, t1 = window
        samples = sorted((g for g in self.gpu.get(node or "", ()) if g.socket_w is not None and t0 <= g.ts <= t1), key=lambda g: g.ts)
        low, how, src_events = False, "", []
        points: list[dict[str, Any]] = []
        energy = None
        if len(samples) >= 2 and _secs(samples[0].ts, samples[-1].ts) >= 0.9 * _secs(t0, t1):
            points = [{"t": _secs(t0, g.ts), "w": g.socket_w, "event_id": g.event_id} for g in samples]
            energy = sum((a.socket_w + b.socket_w) / 2 * _secs(a.ts, b.ts) for a, b in zip(samples, samples[1:]))
            how, src_events = f"telemetry.gpu.power on {node}, trapezoid rule over {len(samples)} samples", samples
            low = any(g.stale or g.counter_ok is False for g in samples)
        else:
            read = [s for s in steps if s.socket_w is not None or s.energy_j is not None]
            if read:
                points = [{"t": _secs(t0, s.ts), "w": s.socket_w if s.socket_w is not None else round(s.energy_j / (s.step_ms / 1000), 1),
                           "dt": s.step_ms / 1000, "event_id": s.event_id} for s in read]
                energy = sum(s.energy_j if s.energy_j is not None else s.socket_w * s.step_ms / 1000 for s in read)
                how, src_events = "telemetry.train.step: each step's energy (its counter, else mean watts x step time), summed", read
                low = len(read) < len(steps)  # some steps had no reading: the total is short
        if energy is None:
            why = (f"the log has no telemetry.gpu.power samples on {node} across the training window and no telemetry.train.step "
                   "with a power reading" if node else "the log does not say which node trained this run")
            return {"points": [], "points_reason": why, "node": node,
                    **dict(zip(("energy_j", "mean_w", "peak_w", "j_per_sample"),
                               (missing(lab, u, why) for lab, u in zip(labels, ("J", "W", "W", "J/sample")))))}
        src = derived(how, *src_events)
        seen = training["samples"]["value"]
        span = sum(p.get("dt", 0) for p in points) or _secs(samples[0].ts, samples[-1].ts)
        return {
            "node": node, "points": points, "points_reason": None, "points_source": src,
            "energy_j": num(labels[0], round(energy, 1), "J", src, low_confidence=low),
            "mean_w": num(labels[1], round(energy / span, 1), "W", derived("energy / time covered by the readings"), low_confidence=low),
            "peak_w": num(labels[2], max(p["w"] for p in points), "W", src, low_confidence=low),
            "j_per_sample": num(labels[3], round(energy / seen, 1), "J/sample", derived("training energy / training samples"),
                                low_confidence=low) if seen else missing(labels[3], "J/sample", training["samples"]["reason"]),
        }

    def _eval(self, evaluated: RunEvaluated | None, report: tuple[Event, EvalReported] | None) -> dict[str, Any]:
        names = {"current": "Score before", "candidate": "Score after", "teacher": "Teacher's score"}
        no_eval = "the log has no learning.run.evaluated event for this run"
        scores = {}
        for key, label in names.items():
            if evaluated and key in evaluated.scores:
                scores[key] = num(label, evaluated.scores[key], "fraction right", by_event(evaluated))
            else:
                scores[key] = missing(label, "fraction right", f"the eval event has no {key} score" if evaluated else no_eval)
        no_report = "no learning.eval.reported event for this run: the eval event carries scores and a decision only"
        out: dict[str, Any] = {"scores": scores, "decision": evaluated.decision if evaluated else None,
                               "model_version": evaluated.model_version if evaluated else None,
                               "source": by_event(evaluated) if evaluated else None}
        if report is None:
            for key, label, unit in (("n_items", "Held-out questions scored", "items"), ("n_groups", "Groups they stand on", "groups"),
                                     ("noise", "Run-to-run noise", "fraction"), ("p_value", "p-value", None)):
                out[key] = missing(label, unit, no_report)
            out.update(alpha=None, test=None, decode=None, counts=None, items=None, groups=None, items_reason=no_report, correct={})
            return out
        e, r = report
        src = by_event(e)
        low = r.low_confidence
        out["n_items"] = num("Held-out questions scored", r.n_items, "items", src, low_confidence=low)
        out["n_groups"] = (num("Groups they stand on", r.n_groups, "groups", src, low_confidence=low) if r.n_groups is not None
                           else missing("Groups they stand on", "groups", "the eval report does not say"))
        out["noise"] = (num("Run-to-run noise", r.noise, "fraction", src, low_confidence=low) if r.noise is not None
                        else missing("Run-to-run noise", "fraction", "the eval report carries no noise measurement"))
        out["p_value"] = (num("p-value", r.p_value, None, src, low_confidence=low) if r.p_value is not None
                          else missing("p-value", None, "the eval report carries no p-value"))
        out.update(alpha=r.alpha if r.alpha is not None else ALPHA, test=r.test, decode=r.decode, noise_runs=r.noise_runs,
                   split_sha256=r.split_sha256,
                   correct={k: num(f"Right answers ({k})", v, "items", src, low_confidence=low) for k, v in sorted(r.correct.items())})
        if not r.items:
            out.update(items=None, counts=None, groups=None, items_reason="the eval report carries no per-question answers")
            return out
        items, groups = [], {}
        for it in r.items:
            change = {(False, True): "fixed", (True, False): "broke", (True, True): "still_right", (False, False): "still_wrong"}[
                (it.before.correct, it.after.correct)]
            items.append({"id": it.id, "group": it.group, "question": it.question, "expected": it.expected,
                          "before": it.before.model_dump(), "after": it.after.model_dump(), "change": change})
            g = groups.setdefault(it.group or it.id, {"group": it.group or it.id, "n": 0, "before": 0, "after": 0})
            g["n"] += 1
            g["before"] += it.before.correct
            g["after"] += it.after.correct
        tally = {c: sum(i["change"] == c for i in items) for c in ("fixed", "broke", "still_right", "still_wrong")}
        counted = derived("counted from the eval report's per-question answers", e)
        out["items"], out["items_reason"] = items, None
        out["groups"] = [{**g, "diff": round((g["after"] - g["before"]) / g["n"], 4)} for g in groups.values()]
        out["counts"] = {
            "n": num("Questions with both answers", len(items), "items", counted, low_confidence=low),
            "before_right": num("Right before", tally["still_right"] + tally["broke"], "items", counted, low_confidence=low),
            "after_right": num("Right after", tally["still_right"] + tally["fixed"], "items", counted, low_confidence=low),
            "fixed": num("Wrong before, right after", tally["fixed"], "items", counted, low_confidence=low),
            "broke": num("Right before, wrong after", tally["broke"], "items", counted, low_confidence=low),
            "still_right": num("Right both times", tally["still_right"], "items", counted, low_confidence=low),
            "still_wrong": num("Wrong both times", tally["still_wrong"], "items", counted, low_confidence=low),
        }
        return out

    def _deploy(self, version: str | None, evaluated: RunEvaluated | None
                ) -> tuple[dict[str, Any], ModelDeployed | None, tuple[Event, SwapCompleted] | None]:
        deployed = next((d for d in self.deployments if version and d.model_version == version), None)
        swap = next((s for s in self.swaps if version and s[1].model_version == version), None)
        rollback = next((d for d in self.deployments if deployed and d.ts > deployed.ts and d.previous_version == version), None)
        swapped_back = swap is not None and swap[1].outcome == "rolled_back"
        if deployed is None:
            why = ("the gate did not clear this model, so it was not installed" if evaluated and evaluated.decision == "reject" else
                   "the log has no learning.model.deployed event for this run's model")
            if swapped_back:
                why = "the swap was tried and the box went back to the old model" + (f": {swap[1].reason}" if swap[1].reason else "")
        no_swap = "no learning.swap.completed event: the log does not hold how long the swap took"
        out = {
            "model_version": version,
            "previous_version": deployed.previous_version if deployed else (swap[1].previous_version if swap else None),
            "deployed": deployed.ts if deployed else None,
            "deployed_reason": None if deployed else why,
            "source": by_event(deployed) if deployed else None,
            "swap_s": num("Hot-swap time", swap[1].swap_s, "s", by_event(swap[0])) if swap and swap[1].swap_s is not None
            else missing("Hot-swap time", "s", no_swap),
            "swap_started": swap[1].started if swap else None,
            "rolled_back": {"ts": rollback.ts, "to": rollback.model_version, "source": by_event(rollback)} if rollback else (
                {"ts": swap[0].ts, "to": swap[1].previous_version, "reason": swap[1].reason, "source": by_event(swap[0])}
                if swapped_back else None),
        }
        return out, deployed, swap

    def _status(self, progress: list[RunProgress], evaluated: RunEvaluated | None, deploy: dict[str, Any]) -> tuple[str, str | None]:
        phase = progress[-1].stage if progress else None
        if deploy["rolled_back"]:
            return "rolled back", phase
        if deploy["deployed"]:
            return "deployed", phase
        if evaluated and evaluated.decision == "reject":
            return "not deployed", phase
        if phase == "failed":
            return "failed", phase
        if evaluated:
            return "gated", phase  # cleared by the gate, not on the box yet
        return "training", phase

    def _verdict(self, evaluated: RunEvaluated | None, report: tuple[Event, EvalReported] | None, ev: dict[str, Any],
                 status: str) -> tuple[str | None, list[str]]:
        """The gate's decision in plain words. The decision is the event's; the detail only restates what the log holds."""
        if evaluated is None:
            return None, ["The model has not been tested yet."]
        detail = []
        cur, cand = evaluated.scores.get("current"), evaluated.scores.get("candidate")
        if cur is not None and cand is not None:
            detail.append(f"The score went from {cur:.0%} to {cand:.0%} on questions the model never trained on.")
        if report:
            r = report[1]
            if r.p_value is not None:
                alpha = ev["alpha"]
                detail.append(f"p = {r.p_value:.3g}: a difference this large would show up by chance "
                              f"{_one_in(r.p_value)}; the gate needs p below {alpha:g}.")
            if r.noise is not None and cur is not None and cand is not None:
                side = "larger than" if abs(cand - cur) > r.noise else "inside"
                detail.append(f"Scoring the same model again moves the score by about {r.noise:.1%}; the change is {side} that.")
            if r.n_items < 50:
                detail.append(f"Only {r.n_items} held-out questions: the gate needs at least 50 before it may deploy.")
        else:
            detail.append("This run's events carry no n, noise or p-value, so this screen cannot show why the gate decided as it did.")
        if evaluated.decision == "deploy":
            head = {"deployed": "Better by more than chance: deployed.",
                    "rolled back": "Better by more than chance on the test, but the box went back to the old model."}.get(
                status, "Better by more than chance: cleared to deploy, not on the box yet.")
        else:
            head = "Not proven: not deployed."
        return head, detail

    def _package(self, package: tuple[Event, PackageBuilt] | None, version: str | None) -> dict[str, Any]:
        """From the package event, else from the installed package's manifest on this box."""
        e, p = package if package else (None, None)
        manifest, mpath = None, None
        if p is None and version and self.models_dir and "/" not in version and not version.startswith("."):
            mpath = self.models_dir / version / "manifest.json"
            try:
                manifest = ModelManifest.model_validate_json(mpath.read_text()) if mpath.is_file() else None
            except (ValidationError, OSError):
                manifest = None
        why = "no learning.package.built event and no installed package for this model on this box"
        if p is None and manifest is None:
            return {"model_version": version, "gguf": None, "base": None, "quantization": None, "sha256": None, "source": None,
                    "reason": why, "bytes": missing("Model file size", "bytes", why)}
        if p is not None:
            return {"model_version": p.model_version, "gguf": p.gguf, "base": p.base, "quantization": p.quantization,
                    "sha256": p.sha256, "source": by_event(e), "reason": None,
                    "bytes": num("Model file size", p.bytes, "bytes", by_event(e)) if p.bytes is not None
                    else missing("Model file size", "bytes", "the package event carries no size")}
        src = by_file(f"models/{version}/manifest.json")
        gguf = mpath.parent / manifest.gguf
        return {"model_version": manifest.version, "gguf": manifest.gguf, "base": manifest.base, "quantization": None,
                "sha256": manifest.sha256.get(manifest.gguf), "source": src, "reason": None,
                "bytes": num("Model file size", gguf.stat().st_size, "bytes", by_file(f"models/{version}/{manifest.gguf}"))
                if gguf.is_file() else missing("Model file size", "bytes", "the model file is not on this box")}

    def _strip(self, progress: list[RunProgress], steps: list[TrainStep], evaluated: RunEvaluated | None,
               package: tuple[Event, PackageBuilt] | None, deployed: ModelDeployed | None) -> list[dict[str, Any]]:
        """The live run's stage strip (L30): labeling, training, packaging, offered, each with when it began and ended
        from the run's own events. The live loop's stage lines (lab_import kind=stage) report every edge; a run without
        them still lights training from its steps and packaging from its package event."""
        by = {name: [p for p in progress if p.stage == name] for name in ("labeling", "training", "packaging")}
        offered = [p for p in by["packaging"] if p.progress >= 1 and (p.message or "").startswith("offered")]
        done = lambda ps: next((p.ts for p in ps if p.progress >= 1), None)
        started = {"labeling": by["labeling"][0].ts if by["labeling"] else None,
                   "training": min([p.ts for p in by["training"]] + ([steps[0].ts] if steps else []), default=None),
                   "packaging": by["packaging"][0].ts if by["packaging"] else (package[0].ts if package else None),
                   "offered": offered[0].ts if offered else (deployed.ts if deployed else None)}
        ended = {"labeling": done(by["labeling"]),
                 "training": done(by["training"]) or (evaluated.ts if evaluated else None),
                 "packaging": done(by["packaging"]) or (package[0].ts if package else None),
                 "offered": started["offered"]}
        failed = next((p for p in progress if p.stage == "failed"), None)
        out, blocked = [], False
        for i, name in enumerate(STRIP):
            later = any(started[n] or ended[n] for n in STRIP[i + 1:])
            if ended[name] or (started[name] and later):
                state = "done"
            elif later:
                state = "not reported"   # a later stage began; this one sent nothing (a run from files, say)
            elif failed is not None and not blocked:
                state, blocked = "failed", True
            elif started[name]:
                state = "now"
            else:
                state = "waiting"
            a, b = started[name], ended[name]
            out.append({"stage": name, "state": state, "started": a, "ended": b,
                        "seconds": _secs(a, b) if a and b else None,
                        "message": failed.message if state == "failed" and failed else None})
        return out

    def _loop(self, run_id: str, strip: list[dict[str, Any]]) -> dict[str, Any]:
        """The live loop's act (L29): members, the pods it trains on, and each extra pod's rank as a part of the
        training, with its steps and losses. The strip's training stage names the pods."""
        found = self.loop_acts.get(run_id)
        if found is None:
            return {"members": None, "pods": None, "parts": [], "part_of": self.part_of(run_id)}
        act = found[1]
        parts = []
        for node, part in sorted(act.parts.items()):
            steps = [st for _, st in sorted(self.steps.get(part, {}).items())]
            losses = [st.loss for st in steps if st.loss is not None]
            parts.append({"node": node, "run_id": part, "steps": len(steps),
                          "loss_first": losses[0] if losses else None, "loss_last": losses[-1] if losses else None,
                          "points": [{"step": st.step, "ts": st.ts, "loss": st.loss, "w": st.socket_w} for st in steps],
                          "source": derived("one point per telemetry.train.step of the part's run", *steps) if steps
                          else None})
            if steps:   # the other pod's steps light the training stage too
                training = next(x for x in strip if x["stage"] == "training")
                if training["state"] == "waiting":
                    training["state"], training["started"] = "now", training["started"] or steps[0].ts
        for x in strip:
            if x["stage"] == "training":
                x["pods"] = list(act.pods)
        return {"members": act.members, "pods": list(act.pods), "parts": parts, "part_of": None,
                "loop": {"act": act.act, "members": act.members, "pods": list(act.pods), "source": by_event(found[0])}}

    def _box(self, deployed: ModelDeployed | None, swap: tuple[Event, SwapCompleted] | None) -> dict[str, Any]:
        """Energy and speed per answer on the box, for the model before the swap and the one after it."""
        if deployed is None:
            why = "this run's model was not deployed, so there is no before and after on the box"
            return {"before": None, "after": None, "reason": why}
        cut0 = (swap[1].started if swap and swap[1].started else deployed.ts)
        return {"reason": None,
                "before": self._box_side("before", deployed.previous_version, lambda ts: ts < cut0),
                "after": self._box_side("after", deployed.model_version, lambda ts: ts >= deployed.ts)}

    def _box_side(self, when: str, version: str | None, in_window) -> dict[str, Any]:
        answers = [a for a in self.answers if version and a.model == version and in_window(a.ts)]
        reqs = [r for r in self.requests if version and r.model == version and in_window(r.ts) and r.tokens_out]
        rates = []
        for r in reqs:
            secs = _secs(r.first_token or r.started, r.ended)
            if secs > 0:
                rates.append(r.tokens_out / secs)
        no_j = f"the log has no telemetry.joules_per_answer for {version or 'that model'} {when} the swap"
        no_t = f"the log has no telemetry.request with token counts for {version or 'that model'} {when} the swap"
        return {
            "version": version,
            "j_per_answer": num(f"Energy per answer {when}", round(sum(a.joules for a in answers) / len(answers), 1), "J",
                                derived(f"mean of {len(answers)} telemetry.joules_per_answer (gross)", *answers),
                                low_confidence=any(a.low_confidence for a in answers)) if answers
            else missing(f"Energy per answer {when}", "J", no_j),
            "answers": len(answers),
            "tok_s": num(f"Tokens per second {when}", round(sum(rates) / len(rates), 1), "tok/s",
                         derived(f"mean over {len(rates)} telemetry.request of tokens_out / time after the first token", *reqs))
            if rates else missing(f"Tokens per second {when}", "tok/s", no_t),
        }


STRIP = ("labeling", "training", "packaging", "offered")


def _run_state(status: str, progress: list[RunProgress], strip: list[dict[str, Any]]) -> str:
    """running, finished or failed: whether the run is still moving, for a screen that redraws while it does."""
    if status == "failed" or any(s["state"] == "failed" for s in strip):
        return "failed"
    if status in ("deployed", "not deployed", "rolled back") or strip[-1]["state"] == "done":
        return "finished"
    return "running"


def run_status(view: "LabView", run_id: str) -> str | None:
    """The run's `run_status` as its story says it; None when the view does not know the run."""
    story = view.story(run_id)
    return story["run_status"] if story else None


def _one_in(p: float) -> str:
    if p <= 0:
        return "never"
    if p >= 0.5:
        return "more than half the time"
    return f"about 1 time in {round(1 / p):,}"


def fold(view: LabView, events: list[Event]) -> LabView:
    for e in events:
        view.on_event(e)
    return view
