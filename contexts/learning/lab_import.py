"""Bring a training run's files into the box's event log, so the Learning lab shows the real run. No pod access from
the box: someone on the lab's side hands over the file, or pushes it.

    python -m contexts.learning.lab_import --telemetry runs/run-7/telemetry.jsonl [--run-id run-7] [--data-dir data]
        [--eval runs/run-7/gate/scores.json] [--manifest models/atlas-v3/manifest.json]
    python -m contexts.learning.lab_import --ledger lab/state/ledger.jsonl [--data-dir data]

    curl -X POST --data-binary @telemetry.jsonl -H "Authorization: Bearer $ATLAS_TOKEN" \\
        "http://<box>:8000/api/learning/lab/import?run_id=run-7"                  # the steps, finished or still growing
    curl -X POST --data-binary @scores.json   -H "Authorization: Bearer $ATLAS_TOKEN" \\
        "http://<box>:8000/api/learning/lab/import?kind=eval"
    curl -X POST --data-binary @manifest.json -H "Authorization: Bearer $ATLAS_TOKEN" \\
        "http://<box>:8000/api/learning/lab/import?kind=manifest&run_id=run-7"
    curl -X POST --data-binary @ledger.jsonl  -H "Authorization: Bearer $ATLAS_TOKEN" \\
        "http://<box>:8000/api/learning/lab/import?kind=ledger"                   # the lab's ledger, whole, each time

The command appends to the log on disk (`<data-dir>/events`); a gateway that is already running reads the log only
when it starts, so push to a running box with the POST instead. Both go through the same functions.

What is read, and what it becomes:

--telemetry / kind=telemetry   lab/train.py's `<out>/telemetry.jsonl`: one `telemetry.train.step` event per line,
    exactly as contracts/telemetry.py TrainStep dumps it. Each line is validated against that contract and published
    as it is, event id included. Nothing is added, renamed or filled in. A line that does not parse, is another event
    type, or (with a run id given) belongs to another run refuses the whole file: nothing is published, and the
    answer names each line. Importing again adds nothing: a step is known by (run_id, step), so a file that has grown
    since the last import adds only its new steps.
--eval / kind=eval             lab/eval_gate.py's `<out>/scores.json`. Its `event` is the gate's own
    `learning.run.evaluated` and is published as it is. The file's n_items, n_groups, noise, p_value and split_sha256,
    and when it holds them correct, noise_runs, alpha, test and items (the per-question before and after answers),
    become one `learning.eval.reported` event (an optional input of the lab view), id `import:<run_id>:eval.reported`.
    A key the file does not hold is left out of the event.
--manifest / kind=manifest     a package's `manifest.json` (contracts/learning.py ModelManifest) becomes one
    `learning.package.built` event {run_id, model_version, base, gguf, sha256}, id `import:<run_id>:package.built`.
    The manifest does not name its run, so a run id must be given.
--run / kind=run               lab/train.py's `<out>/run.json` (its one JSON answer: ok, outcome, refs, metrics) becomes
    one `learning.training.started` {task, dataset_id, node, base, steps, batch (samples per optimizer step: the
    per-device batch times gradient accumulation), train_examples, epochs, lora_rank}, id
    `import:<run_id>:training.started`, which gives the lab view its samples per second. A run id given that is not
    the file's refs.run_id refuses the file.

Where the stage goes (L29). The phone's Teach screen (application/board.py) and the operator screen follow
`learning.run.progress`, which no file holds, so each import also says where the run is, once the run's task is known
(from `&task=` / `--task`, else from a training.started or an eval already in the log or in the same import):
telemetry -> stage training, progress = the last step over the planned steps (0 while those are not known), loss =
the last finite loss, id `import:<run_id>:progress.training.<step>`; run -> stage training at 1.0 with the final loss
for a trained run, else stage failed with the trainer's error, id `import:<run_id>:progress.trained` or
`.progress.failed`; eval -> stage eval at 1.0, id `import:<run_id>:progress.eval`; manifest -> stage packaging at
1.0 (when the manifest names its task), id `import:<run_id>:progress.packaging`. A telemetry import whose run has no
known task publishes no progress and its answer says so. The loss curve still comes from the steps alone.

--ledger / kind=ledger         the lab's ledger (lab/machine/ledger.py, `<state dir>/ledger.jsonl`), whole. It is not
    turned into events: the box keeps the file as it is at `<data-dir>/lab_state/ledger.jsonl` (never lab/state in the
    checkout), and the Experiments tab reads it there (application/lab_experiments.py) on its next request, with no
    restart. Every line must be a JSON object with an integer `seq` (what the lab's reader needs to count a line as a
    row) that the lab's own `ledger.validate` accepts; one line that is not refuses the whole body, each named by its
    line number and the field at fault, never by its contents. A ledger row carries free text (note, source, the
    experiment's hypothesis and citation), so each line is also put through the repo's credential scan
    (tests/test_no_secrets.py `scan_text`: hub, Hugging Face, Notion, Tailscale and GitHub tokens, private keys,
    passwords), and a line that trips it refuses the body, named by line and by kind of credential only. The file is
    written to a temporary file beside the copy and renamed over it, so a refused or interrupted push leaves the
    previous copy whole. The same bytes again write nothing and say `changed: false`.

kind=stage (L30)               the live loop's stage lines (tooling/demo/live_loop_pod.sh writes ~/l29/<stamp>/stages.jsonl:
    one {act, stage, ts, run_id?, rc?} per line, on pod D's clock), for one run (`&run_id=`, required) and its task
    (`&task=`, or a line's `task`, or what the log knows). Each line becomes a learning.run.progress the stage strip of
    GET /api/learning/lab/{run_id} is drawn from: start -> labeling 0, labeled -> labeling 1, train -> training 0,
    trained -> training 1 and packaging 0, done (rc 0) -> packaging 1 with the message "offered", done with another rc
    or failed -> failed; rank1_train and rank1_trained (R65's rank 1 on pod C, posted under the part's run id
    <run_id>.pod-c) -> training 0 and 1. A start line with members, pods or parts also becomes one learning.loop.act
    (L29: "2 members", the pods act 2 trains on, and the part run each extra pod's steps are posted under). Ids are `import:<run_id>:stage.<stage>`, so posting the same lines again adds nothing. A line
    of another run refuses the body; a stage name not in that list is skipped and counted. "offered" rides on the
    packaging stage, not on "done": the Teach screen reads "done" as deployed.
kind=compare / kind=power (L31) the relay's comparison document and its power windows, kept as files under
    `<data-dir>/lab_loops` (application/lab_loops.py says the shapes), read by GET /api/learning/lab/compare.

Every answer that names a run also says where it is: `status` ("running", "finished" or "failed", lab_view.run_status)
and `steps_so_far` (the steps the box holds for it), so a relay posting a growing telemetry file reads the box's view
of the run in the same answer. A partial run is imported the same way as a finished one.

Once a run's steps are in the log the lab view serves that run instead of the sample, and shows it as "Training"
(live) until an eval arrives.
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import tempfile
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import ValidationError

from bus import EventBus
from contexts.learning.application import lab_loops
from contexts.learning.application.lab_experiments import imported_state_dir, received_at
from contexts.learning.application.lab_view import LabView, run_status
from contracts import parse_event
from contracts.events import Event
from contracts.learning import ModelManifest, RunEvaluated, RunProgress
from contracts.telemetry import TrainStep

MAX_BYTES = 8 * 1024 * 1024  # a 10,000-step telemetry file is about 3 MB
KINDS = ("telemetry", "eval", "manifest", "run")  # the files that become events; "ledger" is kept as a file instead
STAGE = "stage"  # the live loop's stage lines, events too (L30)
FILE_KINDS = ("compare", "power")  # kept as files under <data-dir>/lab_loops (application/lab_loops.py)
# the live loop's stage names -> (event id suffix, stage, progress, message); "done" is handled on its own (rc)
STAGE_LINES: dict[str, list[tuple[str, str, float, str]]] = {
    "start": [("start", "labeling", 0.0, "labeling at the teacher")],
    "labeled": [("labeled", "labeling", 1.0, "labeled")],
    "train": [("train", "training", 0.0, "training started")],
    "trained": [("trained", "training", 1.0, "trained"), ("trained.packaging", "packaging", 0.0, "packaging")],
    # R65's rank 1 on pod C, posted under its own part run id (<act run id>.pod-c) when an act trains on both pods
    "rank1_train": [("rank1_train", "training", 0.0, "rank 1 on pod C started")],
    "rank1_trained": [("rank1_trained", "training", 1.0, "rank 1 on pod C trained")],
}
LOOP_ACT = "learning.loop.act"   # the act's members, pods and parts (lab_view.LoopAct), from its start line
STARTED = "learning.training.started"
LEDGER = "ledger"
PROBLEMS_SHOWN = 50
_FIELD = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,39}")

log = logging.getLogger("atlas.learning.lab_import")


class ImportRefused(ValueError):
    """The file is not what it should be. `problems` names each reason; nothing was published."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("; ".join(problems[:5]) + (f"; and {len(problems) - 5} more" if len(problems) > 5 else ""))


def parse_steps(text: str, run_id: str | None = None) -> list[TrainStep]:
    """Every line of a telemetry.jsonl as a TrainStep, or ImportRefused naming every line that is not one."""
    steps, problems = [], []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = parse_event(line)
        except (ValidationError, ValueError) as err:
            first = err.errors()[0] if isinstance(err, ValidationError) else None
            why = f"{'.'.join(map(str, first['loc']))}: {first['msg']}" if first and first["type"] != "json_invalid" else "not JSON"
            problems.append(f"line {n}: {why}")
            continue
        if not isinstance(event, TrainStep):
            problems.append(f"line {n}: a {event.type} event, not telemetry.train.step")
        elif run_id is not None and event.run_id != run_id:
            problems.append(f"line {n}: run {event.run_id!r}, not {run_id!r}")
        else:
            steps.append(event)
    if not steps and not problems:
        problems.append("the file has no step lines")
    if problems:
        raise ImportRefused(problems)
    return steps


def eval_events(text: str, run_id: str | None = None) -> list[Event]:
    """The eval gate's scores.json as its own learning.run.evaluated event plus one learning.eval.reported."""
    try:
        scores = json.loads(text)
        evaluated = parse_event(scores["event"])
    except (ValueError, KeyError, TypeError, ValidationError) as err:
        raise ImportRefused([f"not an eval gate scores.json with its event: {type(err).__name__}"]) from None
    if not isinstance(evaluated, RunEvaluated):
        raise ImportRefused([f"its event is {evaluated.type}, not learning.run.evaluated"])
    if run_id is not None and evaluated.run_id != run_id:
        raise ImportRefused([f"the eval is for run {evaluated.run_id!r}, not {run_id!r}"])
    out: list[Event] = [evaluated]
    if isinstance(scores.get("n_items"), int):
        body: dict[str, Any] = {"type": "learning.eval.reported", "event_id": f"import:{evaluated.run_id}:eval.reported",
                                "ts": evaluated.ts, "run_id": evaluated.run_id, "task": evaluated.task, "n_items": scores["n_items"]}
        for key in ("n_groups", "noise", "p_value", "split_sha256", "correct", "noise_runs", "alpha", "test", "items", "gate"):  # only what the file holds
            if scores.get(key) is not None:
                body[key] = scores[key]
        if isinstance(scores.get("decode"), dict):
            body["decode"] = ", ".join(f"{k} {v}" for k, v in scores["decode"].items())
        out.append(parse_event(body))
    return out


def manifest_events(text: str, run_id: str | None) -> list[Event]:
    if not run_id:
        raise ImportRefused(["a manifest does not name its run: give the run id"])
    try:
        manifest = ModelManifest.model_validate_json(text)
    except ValidationError as err:
        raise ImportRefused([f"not a package manifest: {err.errors()[0]['msg']}"]) from None
    return [parse_event({"type": "learning.package.built", "event_id": f"import:{run_id}:package.built", "ts": manifest.created,
                         "run_id": run_id, "task": manifest.task, "model_version": manifest.version, "base": manifest.base,
                         "gguf": manifest.gguf, "sha256": manifest.sha256.get(manifest.gguf)})]


@dataclass(frozen=True)
class RunFacts:
    """What the log already says about a run, for the progress an import adds: its task, its planned steps and the
    time of its first and last step."""

    task: str | None = None
    steps: int | None = None
    first_ts: datetime | None = None
    last_ts: datetime | None = None
    past_training: bool = False   # evaluated, packaged or a later stage: a late step file must not move it back


def _facts_of(view: LabView) -> dict[str, RunFacts]:
    out: dict[str, RunFacts] = {}
    for run in {*view.steps, *view.started, *view.evaluated, *view.progress, *view.packages}:
        started = view.started.get(run, (None, None))[1]
        evaluated = view.evaluated.get(run)
        progress = view.progress.get(run) or []
        task = (started.task if started else None) or (evaluated.task if evaluated else None) or (progress[-1].task if progress else None)
        stamps = [s.ts for s in (view.steps.get(run) or {}).values()]
        past = evaluated is not None or run in view.packages or any(p.stage not in ("labeling", "training") for p in progress)
        out[run] = RunFacts(task, started.steps if started else None, min(stamps) if stamps else None,
                            max(stamps) if stamps else None, past)
    return out


def facts_in(events: Iterable[Event]) -> dict[str, RunFacts]:
    """RunFacts from a list of events (the log, or what one import is about to publish), through the lab view's fold."""
    view = LabView()
    for event in events:
        view.on_event(event)
    return _facts_of(view)


def _merge(older: Mapping[str, RunFacts], newer: Mapping[str, RunFacts]) -> dict[str, RunFacts]:
    out = dict(older)
    for run, facts in newer.items():
        was = out.get(run, RunFacts())
        firsts = [t for t in (was.first_ts, facts.first_ts) if t]
        lasts = [t for t in (was.last_ts, facts.last_ts) if t]
        out[run] = RunFacts(facts.task or was.task, facts.steps or was.steps, min(firsts) if firsts else None,
                            max(lasts) if lasts else None, facts.past_training or was.past_training)
    return out


def _number(metrics: Mapping[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def run_events(text: str, run_id: str | None, facts: Mapping[str, RunFacts] | None = None) -> list[Event]:
    """lab/train.py's run.json as one learning.training.started (see the module docstring)."""
    try:
        answer = json.loads(text)
        refs, metrics = answer["refs"], answer.get("metrics") or {}
        file_run = refs["run_id"]
    except (ValueError, KeyError, TypeError):
        raise ImportRefused(["not a trainer run.json (lab/train.py): it needs refs.run_id"]) from None
    if not isinstance(file_run, str) or not isinstance(metrics, dict) or not isinstance(refs, dict):
        raise ImportRefused(["not a trainer run.json (lab/train.py): refs.run_id must be text and metrics an object"])
    if run_id is not None and file_run != run_id:
        raise ImportRefused([f"the run.json is for run {file_run!r}, not {run_id!r}"])
    known = (facts or {}).get(file_run, RunFacts())
    steps, batch, accum = (_number(metrics, f"trainer.{k}") for k in ("steps", "batch", "grad_accum"))
    body: dict[str, Any] = {"type": STARTED, "event_id": f"import:{file_run}:training.started", "run_id": file_run,
                            "ts": known.first_ts or datetime.now(UTC)}
    fields = {"task": refs.get("task"), "dataset_id": refs.get("dataset_id"), "node": refs.get("node"), "base": refs.get("base_model"),
              "steps": int(steps) if steps is not None else None,
              "batch": int(batch * (accum or 1)) if batch is not None else None,
              "train_examples": int(n) if (n := _number(metrics, "trainer.examples")) is not None else None,
              "epochs": _number(metrics, "trainer.epochs"),
              "lora_rank": int(r) if (r := _number(metrics, "trainer.lora_rank")) is not None else None}
    body.update({key: value for key, value in fields.items() if value is not None})
    return [parse_event(body)]


def stage_events(text: str, run_id: str | None, task: str | None = None,
                 facts: Mapping[str, RunFacts] | None = None) -> list[Event]:
    """The live loop's stage lines for one run as learning.run.progress (see the module docstring, kind=stage)."""
    if not run_id:
        raise ImportRefused(["stage lines do not all name their run: give the run id"])
    rows, problems = [], []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            problems.append(f"line {n}: not JSON")
            continue
        if not isinstance(row, dict) or not isinstance(row.get("stage"), str) or not row.get("ts"):
            problems.append(f"line {n}: a stage line needs stage and ts")
            continue
        try:
            ts = datetime.fromisoformat(str(row["ts"]).replace("Z", "+00:00"))
        except ValueError:
            problems.append(f"line {n}: ts is not an ISO time")
            continue
        if row.get("run_id") not in (None, run_id):
            problems.append(f"line {n}: run {row.get('run_id')!r}, not {run_id!r}")
            continue
        rows.append((row, ts))
    if not rows and not problems:
        problems.append("the body has no stage lines")
    run_task = task or next((r.get("task") for r, _ in rows if isinstance(r.get("task"), str) and r.get("task")), None) \
        or (facts or {}).get(run_id, RunFacts()).task
    if not run_task and not problems:
        problems.append("the run's task is not known yet: give &task= or a task on the lines")
    if problems:
        raise ImportRefused(problems)
    out: list[Event] = []
    for row, ts in rows:
        name = row["stage"]
        if name == "done":
            ok = row.get("rc") in (0, None)
            marks = [("done", "packaging", 1.0, "offered")] if ok else [("done", "failed", 0.0, f"the act ended with rc {row.get('rc')}")]
        elif name == "failed":
            marks = [("failed", "failed", 0.0, str(row.get("error") or row.get("message") or "the act failed")[:200])]
        else:
            marks = STAGE_LINES.get(name, [])
        for suffix, stage, progress, message in marks:
            out.append(RunProgress(event_id=f"import:{run_id}:stage.{suffix}", ts=ts, task=run_task, run_id=run_id,
                                   stage=stage, progress=progress, message=message))
        if name == "start" and any(row.get(k) is not None for k in ("members", "pods", "parts")):
            out.append(loop_act(row, run_id, ts))
    return out


def loop_act(row: Mapping[str, Any], run_id: str, ts: datetime) -> Event:
    """A start line's members, pods and parts as one learning.loop.act (L29); ImportRefused when they are not plain."""
    members, pods, parts = row.get("members"), row.get("pods") or [], row.get("parts") or {}
    if members is not None and (not isinstance(members, int) or isinstance(members, bool) or members < 1):
        raise ImportRefused(["members must be a whole number of at least 1"])
    if not isinstance(pods, list) or not all(isinstance(x, str) and _FIELD.fullmatch(x) for x in pods):
        raise ImportRefused(["pods must be a list of pod names"])
    if not isinstance(parts, dict) or not all(isinstance(k, str) and _FIELD.fullmatch(k) and isinstance(v, str)
                                              and v.startswith(run_id + ".") for k, v in parts.items()):
        raise ImportRefused(["parts must map a pod to a run id that starts with the act's own run id and a dot"])
    return parse_event({"type": LOOP_ACT, "event_id": f"import:{run_id}:loop.act", "ts": ts, "run_id": run_id,
                        "act": row.get("act"), "members": members, "pods": pods, "parts": parts})


def skipped_stages(text: str) -> list[str]:
    """The stage names of a stage body that say nothing to the strip (the pod's "go" and "end", or a new name)."""
    out = []
    for line in text.splitlines():
        try:
            name = json.loads(line).get("stage") if line.strip() else None
        except (ValueError, AttributeError):
            continue
        if isinstance(name, str) and name not in STAGE_LINES and name not in ("done", "failed"):
            out.append(name)
    return out


def where_runs_are(view: LabView, runs: list[str]) -> dict[str, Any]:
    """`status` and `steps_so_far` of the one run an answer names (None when it names none or several), and
    `run_status` for each."""
    each = {r: {"status": run_status(view, r), "steps_so_far": len(view.steps.get(r) or {})} for r in runs}
    one = each[runs[0]] if len(runs) == 1 else {"status": None, "steps_so_far": None}
    return {**one, "run_status": each}


def _run_end(text: str, task: str | None, facts: Mapping[str, RunFacts]) -> list[Event]:
    """The progress a run.json closes the training stage with: trained (1.0, final loss) or failed (the error)."""
    answer = json.loads(text)
    refs, metrics = answer["refs"], answer.get("metrics") or {}
    run = refs["run_id"]
    task = task or refs.get("task") or facts.get(run, RunFacts()).task
    if not task or facts.get(run, RunFacts()).past_training:
        return []
    ts = facts.get(run, RunFacts()).last_ts or datetime.now(UTC)
    if answer.get("ok") and answer.get("outcome") == "trained":
        seconds = _number(metrics, "trainer.train_s")
        steps = _number(metrics, "trainer.steps")
        message = "trained" + (f" {int(steps)} steps" if steps is not None else "") + (f" in {seconds:.0f} s" if seconds is not None else "")
        return [RunProgress(event_id=f"import:{run}:progress.trained", ts=ts, task=task, run_id=run, stage="training",
                            progress=1.0, loss=_number(metrics, "trainer.loss_last"), message=message)]
    why = str(answer.get("error") or answer.get("outcome") or "the trainer did not finish")[:200]
    return [RunProgress(event_id=f"import:{run}:progress.failed", ts=ts, task=task, run_id=run, stage="failed",
                        progress=0.0, message=why)]


def _training_progress(steps: list[TrainStep], task: str | None, facts: Mapping[str, RunFacts]) -> list[Event]:
    """One progress per run in a telemetry file: its last step over the planned steps, when the run's task is known."""
    out: list[Event] = []
    for run in sorted({s.run_id for s in steps}):
        mine = sorted((s for s in steps if s.run_id == run), key=lambda s: s.step)
        known = facts.get(run, RunFacts())
        run_task = task or known.task
        if not run_task or known.past_training:
            continue
        last = mine[-1]
        losses = [s.loss for s in mine if s.loss is not None]
        planned = known.steps
        progress = min(1.0, last.step / planned) if planned else 0.0
        message = f"step {last.step}" + (f" of {planned}" if planned else ", planned steps not known yet")
        out.append(RunProgress(event_id=f"import:{run}:progress.training.{last.step}", ts=last.ts, task=run_task, run_id=run,
                               stage="training", progress=progress, loss=losses[-1] if losses else None, message=message))
    return out


def events_of(kind: str, text: str, run_id: str | None, task: str | None = None,
              facts: Mapping[str, RunFacts] | None = None) -> list[Event]:
    """The file's own events, then the learning.run.progress that says where its run is (module docstring, L29).
    `facts` is what the log already knows about each run; the file's own events are added to it."""
    if kind == "telemetry":
        own: list[Event] = list(parse_steps(text, run_id))
    elif kind == "eval":
        own = eval_events(text, run_id)
    elif kind == "manifest":
        own = manifest_events(text, run_id)
    elif kind == "run":
        own = run_events(text, run_id, facts)
    elif kind == STAGE:
        return stage_events(text, run_id, task, facts)
    else:
        raise ImportRefused([f"kind must be one of {[*KINDS, STAGE, LEDGER, *FILE_KINDS]}"])
    known = _merge(facts or {}, facts_in(own))
    if kind == "telemetry":
        return own + _training_progress([e for e in own if isinstance(e, TrainStep)], task, known)
    if kind == "run":
        return own + _run_end(text, task, known)
    run = next((getattr(e, "run_id", None) for e in own if getattr(e, "run_id", None)), None)
    stage_task = task or (known.get(run, RunFacts()).task if run else None) or next((getattr(e, "task", None) for e in own if getattr(e, "task", None)), None)
    if not run or not stage_task:
        return own
    stage = "eval" if kind == "eval" else "packaging"
    return own + [RunProgress(event_id=f"import:{run}:progress.{stage}", ts=own[0].ts, task=stage_task, run_id=run, stage=stage,
                              progress=1.0)]


def progress_note(kind: str, events: list[Event], facts: Mapping[str, RunFacts] | None = None) -> str | None:
    """Why an import said nothing of where its run is, for the answer."""
    if kind not in ("telemetry", "run") or any(isinstance(e, RunProgress) for e in events):
        return None
    runs = {getattr(e, "run_id", None) for e in events} - {None}
    if runs and all((facts or {}).get(r, RunFacts()).past_training for r in runs):
        return "no learning.run.progress: the run is already past training (evaluated or packaged), so its stage stays"
    return ("no learning.run.progress: the run's task is not known yet; give &task= (or --task), or import its "
            "run.json or eval first")


# --- the lab's ledger: kept as a file, not turned into events -------------------------------------------------

def _row_problems(row: Any, validate: Callable[[Any], Any], refused: type[Exception]) -> list[str]:
    """Why the lab's reader or its own row check would not take this line: field names only, never a value."""
    if not isinstance(row, dict):
        return ["not a JSON object"]
    problems = [] if type(row.get("seq")) is int else ["seq is missing or not an integer"]
    try:
        validate({key: value for key, value in row.items() if key != "seq"})  # seq is the ledger's own; the rest is the row
    except refused as err:
        for problem in getattr(err, "problems", []):
            field = problem.split(" ", 1)[0]  # the lab's messages start with the field and may go on to quote its value
            field = field if _FIELD.fullmatch(field) else "a field name"
            text = f"{field} is missing" if problem.endswith(" is missing") else f"{field} is not what the lab's ledger accepts"
            if text not in problems:
                problems.append(text)
    return problems


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, inner in value.items():
            yield str(key)
            yield from _strings(inner)
    elif isinstance(value, list):
        for inner in value:
            yield from _strings(inner)


def _pytest_stand_in() -> None:
    """Where pytest is not installed (a box runs the app, not the tests), give the scanner's module the two decorators
    it uses at import time, as pass-throughs. Nothing is replaced where pytest exists."""
    import importlib.util
    import sys
    import types

    if "pytest" in sys.modules or importlib.util.find_spec("pytest") is not None:
        return

    def passthrough(*args: Any, **kwargs: Any) -> Any:
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda fn: fn

    stand_in = types.ModuleType("pytest")
    stand_in.fixture = passthrough  # type: ignore[attr-defined]
    stand_in.mark = types.SimpleNamespace(parametrize=passthrough)  # type: ignore[attr-defined]
    stand_in.param = lambda *a, **k: a  # type: ignore[attr-defined]
    stand_in.skip = lambda *a, **k: None  # type: ignore[attr-defined]
    stand_in.raises = None  # type: ignore[attr-defined]
    sys.modules["pytest"] = stand_in


def _secret_scan() -> Callable[[str], list[str]]:
    """The repo's own credential scan as text -> labels. ImportRefused where it cannot be loaded: a ledger carries
    free text, so it is not stored unchecked."""
    try:
        _pytest_stand_in()  # the scanner lives in a test module; a box without pytest still needs the scan
        from tests.test_no_secrets import scan_text  # the one scanner the repo has
    except Exception as err:  # noqa: BLE001 - not run from the repo root
        raise ImportRefused([f"the credential check could not be loaded here ({type(err).__name__}), so no ledger is stored"]) from None
    return lambda text: sorted({hit.label for hit in scan_text("ledger", text)})


def check_ledger(text: str) -> dict[str, Any]:
    """The counts of a ledger the box will keep, or ImportRefused naming every line that is not a row the lab's own
    check accepts, or that holds something shaped like a credential. Reads nothing and writes nothing."""
    try:
        from lab.machine import experiments as lab_experiments
        from lab.machine import ledger
    except ImportError as err:  # the command or the gateway run from outside the repo root
        raise ImportRefused([f"the lab's ledger code could not be imported here ({type(err).__name__})"]) from None
    labels_in = _secret_scan()
    rows, problems = [], []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            problems.append(f"line {n}: not JSON")
            continue
        found = _row_problems(row, ledger.validate, ledger.RowError)
        labels = sorted({label for part in (line, *_strings(row)) for label in labels_in(part)})
        found += [f"holds what looks like a credential ({label}); take it out of the ledger and replace the credential" for label in labels]
        if found:
            problems.extend(f"line {n}: {why}" for why in found)
        else:
            rows.append(row)
    if not rows and not problems:
        problems.append("the file has no ledger rows")
    if problems:
        raise ImportRefused(problems)
    folded, _ = lab_experiments.fold(r for r in rows if r.get("verb") in lab_experiments.VERBS)
    verdicts: dict[str, int] = {}
    for experiment in folded.values():
        label = (experiment.get("result") or {}).get("label")
        if isinstance(label, str):
            verdicts[label] = verdicts.get(label, 0) + 1
    return {"rows": len(rows), "experiments": len(folded), "verdicts": dict(sorted(verdicts.items()))}


def store_ledger(raw: bytes, data_dir: str | Path) -> dict[str, Any]:
    """Check the ledger and keep it at <data_dir>/lab_state/ledger.jsonl, replacing the copy there in one rename.
    ImportRefused (nothing written, the previous copy untouched) when it does not pass check_ledger."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ImportRefused(["the body is not UTF-8 text"]) from None
    counts = check_ledger(text)
    from lab.machine import ledger

    folder = imported_state_dir(data_dir)
    path = folder / ledger.LEDGER_FILE
    try:
        changed = path.read_bytes() != raw
    except OSError:
        changed = True
    if changed:
        folder.mkdir(parents=True, exist_ok=True)
        handle, temp = tempfile.mkstemp(dir=folder, prefix=".ledger-", suffix=".tmp")
        try:
            with os.fdopen(handle, "wb") as out:
                out.write(raw)
                out.flush()
                os.fsync(out.fileno())
            os.chmod(temp, 0o644)  # mkstemp makes it owner-only; the gateway may run as another user than the command
            os.replace(temp, path)
        except BaseException:
            Path(temp).unlink(missing_ok=True)
            raise
    out = {"kind": LEDGER, "changed": changed, **counts, "bytes": len(raw), "received_at": received_at(path),
           "stored": f"{folder.name}/{path.name}"}
    if not changed:
        out["note"] = "the same ledger is already here; nothing was written"
    log.info("lab ledger import: changed=%s rows=%d experiments=%d bytes=%d", changed, counts["rows"], counts["experiments"], len(raw))
    return out


def key_of(event: Event) -> tuple:
    """What makes an imported event the same event: a step by its run and number, anything else by its id."""
    return ("step", event.run_id, event.step) if isinstance(event, TrainStep) else ("id", event.event_id)


async def publish_new(events: list[Event], seen: set[tuple], publish: Callable[[Event], Awaitable[None]]) -> dict[str, Any]:
    """Publish the events `seen` does not hold yet, in order; how many were new and how many were already there."""
    new = 0
    for event in events:
        key = key_of(event)
        if key in seen:
            continue
        seen.add(key)
        await publish(event)
        new += 1
    runs = sorted({e.run_id for e in events if getattr(e, "run_id", None)})
    steps = [e.step for e in events if isinstance(e, TrainStep)]
    return {"published": new, "already_there": len(events) - new, "events": len(events), "runs": runs,
            "last_step": max(steps) if steps else None}


def seen_in(events: Iterable[Event]) -> set[tuple]:
    return {key_of(e) for e in events if isinstance(e, TrainStep) or e.type.startswith("learning.")}


def seen_in_view(view: LabView) -> set[tuple]:
    seen = {("step", run_id, step) for run_id, steps in view.steps.items() for step in steps}
    seen |= {("id", e.event_id) for e in view.evaluated.values()}
    seen |= {("id", e.event_id) for table in (view.reports, view.packages, view.started, view.loop_acts)
             for e, _ in table.values()}
    seen |= {("id", e.event_id) for progress in view.progress.values() for e in progress}
    return seen


def lab_import_router(view: LabView, publish: Callable[[Event], Awaitable[None]], guard: Callable[..., None] | None = None,
                      data_dir: str | Path | None = None) -> APIRouter:
    """`data_dir`: the box's data dir, where kind=ledger keeps the lab's ledger; without it that kind is refused."""
    router = APIRouter()

    @router.post("/api/learning/lab/import", dependencies=[Depends(guard)] if guard else [])
    async def lab_import(request: Request, kind: str = "telemetry", run_id: str | None = None, task: str | None = None,
                         stage: str | None = None) -> dict:
        """Body: the file's text. 400 with every problem when the file is refused; 413 over MAX_BYTES."""
        if int(request.headers.get("content-length") or 0) > MAX_BYTES:
            raise HTTPException(413, f"the file is over {MAX_BYTES} bytes")
        raw = await request.body()
        if len(raw) > MAX_BYTES:
            raise HTTPException(413, f"the file is over {MAX_BYTES} bytes")
        if kind == LEDGER:
            if data_dir is None:
                raise HTTPException(400, {"refused": ["this box has no data dir to keep a ledger in"]})
            try:
                return store_ledger(raw, data_dir)
            except ImportRefused as err:
                raise HTTPException(400, {"refused": err.problems[:PROBLEMS_SHOWN], "problems": len(err.problems)}) from None
        if kind in FILE_KINDS:
            keep_in = data_dir if data_dir is not None else lab_loops.env_data_dir()   # an older app.py passes none
            try:
                answer = (lab_loops.store_compare(raw, keep_in) if kind == "compare"
                          else lab_loops.store_power(raw, keep_in, run_id, stage))
            except lab_loops.Refused as err:
                raise HTTPException(400, {"refused": err.problems[:PROBLEMS_SHOWN], "problems": len(err.problems)}) from None
            log.info("lab %s import: %s", kind, {k: v for k, v in answer.items() if k in ("changed", "stamp", "run_id", "stage", "node")})
            return answer
        facts = _facts_of(view)
        try:
            events = events_of(kind, raw.decode("utf-8"), run_id, task, facts)
        except UnicodeDecodeError:
            raise HTTPException(400, {"refused": ["the body is not UTF-8 text"]}) from None
        except ImportRefused as err:
            raise HTTPException(400, {"refused": err.problems[:PROBLEMS_SHOWN], "problems": len(err.problems)}) from None
        answer = {"kind": kind, **await publish_new(events, seen_in_view(view), publish)}
        if note := progress_note(kind, events, facts):
            answer["note"] = note
        if kind == "stage":
            answer["skipped"] = skipped_stages(raw.decode("utf-8"))
        answer.update(where_runs_are(view, answer["runs"]))
        return answer

    return router


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m contexts.learning.lab_import", description=__doc__.split("\n\n")[0])
    parser.add_argument("--telemetry", type=Path, help="lab/train.py's telemetry.jsonl")
    parser.add_argument("--eval", type=Path, help="lab/eval_gate.py's scores.json")
    parser.add_argument("--manifest", type=Path, help="a model package's manifest.json")
    parser.add_argument("--run", type=Path, help="lab/train.py's run.json (becomes learning.training.started)")
    parser.add_argument("--task", help="the run's task, for the learning.run.progress the Teach screen follows")
    parser.add_argument("--ledger", type=Path, help="the lab's ledger.jsonl, kept whole under <data-dir>/lab_state for the Experiments tab")
    parser.add_argument("--run-id", help="the run the files belong to; a line of another run refuses the file")
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("ATLAS_DATA_DIR") or "data"))
    args = parser.parse_args(argv)
    # run.json first, so the steps after it know the run's task and planned steps; then the stages in their order
    files = [(kind, getattr(args, kind)) for kind in ("run", "telemetry", "eval", "manifest") if getattr(args, kind)]
    if not files and not args.ledger:
        parser.error("give --telemetry, --eval, --manifest, --run or --ledger")
    run_id = args.run_id
    log_events = EventBus(log_dir=args.data_dir / "events").read_log() if files and (args.data_dir / "events").is_dir() else []
    facts = facts_in(log_events)
    try:
        batches = []
        raw_ledger = args.ledger.read_bytes() if args.ledger else None
        if raw_ledger is not None:
            if len(raw_ledger) > MAX_BYTES:
                raise ImportRefused([f"the ledger is over {MAX_BYTES} bytes"])
            try:
                check_ledger(raw_ledger.decode("utf-8"))  # refused before any other file is published
            except UnicodeDecodeError:
                raise ImportRefused(["the ledger is not UTF-8 text"]) from None
        for kind, path in files:  # every file is checked before anything is published
            events = events_of(kind, path.read_text(encoding="utf-8"), run_id, args.task, facts)
            run_id = run_id or next((e.run_id for e in events if getattr(e, "run_id", None)), None)
            facts = _merge(facts, facts_in(events))   # a later file of the same command knows what the earlier ones said
            batches.append((kind, events))
    except (ImportRefused, OSError) as err:
        print(json.dumps({"ok": False, "refused": getattr(err, "problems", [str(err)])}, indent=2), file=sys.stderr)
        return 1
    out: dict[str, Any] = {}
    if batches:  # the log is opened (and its folder made) only when there is something for it
        bus = EventBus(log_dir=args.data_dir / "events")
        seen = seen_in(bus.read_log())
        out = {"log": str(args.data_dir / "events"), **{kind: asyncio.run(publish_new(events, seen, bus.publish)) for kind, events in batches}}
    if raw_ledger is not None:
        try:
            out[LEDGER] = store_ledger(raw_ledger, args.data_dir)
        except (ImportRefused, OSError) as err:
            print(json.dumps({"ok": False, "refused": getattr(err, "problems", [str(err)])}, indent=2), file=sys.stderr)
            return 1
    print(json.dumps({"ok": True, **out}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
