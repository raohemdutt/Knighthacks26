"""Bring a training run's files into the box's event log, so the Learning lab shows the real run. No pod access from
the box: someone on the lab's side hands over the file, or pushes it.

    python -m contexts.learning.lab_import --telemetry runs/run-7/telemetry.jsonl [--run-id run-7] [--data-dir data]
        [--eval runs/run-7/gate/scores.json] [--manifest models/atlas-v3/manifest.json]

    curl -X POST --data-binary @telemetry.jsonl -H "Authorization: Bearer $ATLAS_TOKEN" \\
        "http://<box>:8000/api/learning/lab/import?run_id=run-7"                  # the steps, finished or still growing
    curl -X POST --data-binary @scores.json   -H "Authorization: Bearer $ATLAS_TOKEN" \\
        "http://<box>:8000/api/learning/lab/import?kind=eval"
    curl -X POST --data-binary @manifest.json -H "Authorization: Bearer $ATLAS_TOKEN" \\
        "http://<box>:8000/api/learning/lab/import?kind=manifest&run_id=run-7"

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
    `learning.run.evaluated` and is published as it is. The file's n_items, n_groups, noise, p_value and split_sha256
    become one `learning.eval.reported` event (an optional input of the lab view), id `import:<run_id>:eval.reported`.
    The gate does not write per-question answers or right-answer counts into scores.json, so none are published.
--manifest / kind=manifest     a package's `manifest.json` (contracts/learning.py ModelManifest) becomes one
    `learning.package.built` event {run_id, model_version, base, gguf, sha256}, id `import:<run_id>:package.built`.
    The manifest does not name its run, so a run id must be given.

Once a run's steps are in the log the lab view serves that run instead of the sample, and shows it as "Training"
(live) until an eval arrives.
"""

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import ValidationError

from bus import EventBus
from contexts.learning.application.lab_view import LabView
from contracts import parse_event
from contracts.events import Event
from contracts.learning import ModelManifest, RunEvaluated
from contracts.telemetry import TrainStep

MAX_BYTES = 8 * 1024 * 1024  # a 10,000-step telemetry file is about 3 MB
KINDS = ("telemetry", "eval", "manifest")


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
        for key in ("n_groups", "noise", "p_value", "split_sha256"):  # only what the file holds
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


def events_of(kind: str, text: str, run_id: str | None) -> list[Event]:
    if kind == "telemetry":
        return list(parse_steps(text, run_id))
    if kind == "eval":
        return eval_events(text, run_id)
    if kind == "manifest":
        return manifest_events(text, run_id)
    raise ImportRefused([f"kind must be one of {list(KINDS)}"])


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
    seen |= {("id", e.event_id) for table in (view.reports, view.packages) for e, _ in table.values()}
    return seen


def lab_import_router(view: LabView, publish: Callable[[Event], Awaitable[None]], guard: Callable[..., None] | None = None) -> APIRouter:
    router = APIRouter()

    @router.post("/api/learning/lab/import", dependencies=[Depends(guard)] if guard else [])
    async def lab_import(request: Request, kind: str = "telemetry", run_id: str | None = None) -> dict:
        """Body: the file's text. 400 with every problem when the file is refused; 413 over MAX_BYTES."""
        if int(request.headers.get("content-length") or 0) > MAX_BYTES:
            raise HTTPException(413, f"the file is over {MAX_BYTES} bytes")
        raw = await request.body()
        if len(raw) > MAX_BYTES:
            raise HTTPException(413, f"the file is over {MAX_BYTES} bytes")
        try:
            events = events_of(kind, raw.decode("utf-8"), run_id)
        except UnicodeDecodeError:
            raise HTTPException(400, {"refused": ["the body is not UTF-8 text"]}) from None
        except ImportRefused as err:
            raise HTTPException(400, {"refused": err.problems[:50], "problems": len(err.problems)}) from None
        return {"kind": kind, **await publish_new(events, seen_in_view(view), publish)}

    return router


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m contexts.learning.lab_import", description=__doc__.split("\n\n")[0])
    parser.add_argument("--telemetry", type=Path, help="lab/train.py's telemetry.jsonl")
    parser.add_argument("--eval", type=Path, help="lab/eval_gate.py's scores.json")
    parser.add_argument("--manifest", type=Path, help="a model package's manifest.json")
    parser.add_argument("--run-id", help="the run the files belong to; a line of another run refuses the file")
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("ATLAS_DATA_DIR") or "data"))
    args = parser.parse_args(argv)
    files = [(kind, getattr(args, kind)) for kind in KINDS if getattr(args, kind)]
    if not files:
        parser.error("give --telemetry, --eval or --manifest")
    run_id = args.run_id
    try:
        batches = []
        for kind, path in files:  # every file is checked before anything is published
            events = events_of(kind, path.read_text(encoding="utf-8"), run_id)
            run_id = run_id or next((e.run_id for e in events if getattr(e, "run_id", None)), None)
            batches.append((kind, events))
    except (ImportRefused, OSError) as err:
        print(json.dumps({"ok": False, "refused": getattr(err, "problems", [str(err)])}, indent=2), file=sys.stderr)
        return 1
    bus = EventBus(log_dir=args.data_dir / "events")
    seen = seen_in(bus.read_log())
    out = {kind: asyncio.run(publish_new(events, seen, bus.publish)) for kind, events in batches}
    print(json.dumps({"ok": True, "log": str(args.data_dir / "events"), **out}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
