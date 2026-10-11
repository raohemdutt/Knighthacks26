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
    `learning.run.evaluated` and is published as it is. The file's n_items, n_groups, noise, p_value and split_sha256
    become one `learning.eval.reported` event (an optional input of the lab view), id `import:<run_id>:eval.reported`.
    The gate does not write per-question answers or right-answer counts into scores.json, so none are published.
--manifest / kind=manifest     a package's `manifest.json` (contracts/learning.py ModelManifest) becomes one
    `learning.package.built` event {run_id, model_version, base, gguf, sha256}, id `import:<run_id>:package.built`.
    The manifest does not name its run, so a run id must be given.

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
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import ValidationError

from bus import EventBus
from contexts.learning.application.lab_experiments import imported_state_dir, received_at
from contexts.learning.application.lab_view import LabView
from contracts import parse_event
from contracts.events import Event
from contracts.learning import ModelManifest, RunEvaluated
from contracts.telemetry import TrainStep

MAX_BYTES = 8 * 1024 * 1024  # a 10,000-step telemetry file is about 3 MB
KINDS = ("telemetry", "eval", "manifest")  # the files that become events; "ledger" is kept as a file instead
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
    raise ImportRefused([f"kind must be one of {[*KINDS, LEDGER]}"])


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


def _secret_scan() -> Callable[[str], list[str]]:
    """The repo's own credential scan as text -> labels. ImportRefused where it cannot be loaded: a ledger carries
    free text, so it is not stored unchecked."""
    try:
        from tests.test_no_secrets import scan_text  # the one scanner the repo has; it needs pytest to import
    except Exception as err:  # noqa: BLE001 - no pytest, or not run from the repo root
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
    seen |= {("id", e.event_id) for table in (view.reports, view.packages) for e, _ in table.values()}
    return seen


def lab_import_router(view: LabView, publish: Callable[[Event], Awaitable[None]], guard: Callable[..., None] | None = None,
                      data_dir: str | Path | None = None) -> APIRouter:
    """`data_dir`: the box's data dir, where kind=ledger keeps the lab's ledger; without it that kind is refused."""
    router = APIRouter()

    @router.post("/api/learning/lab/import", dependencies=[Depends(guard)] if guard else [])
    async def lab_import(request: Request, kind: str = "telemetry", run_id: str | None = None) -> dict:
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
        try:
            events = events_of(kind, raw.decode("utf-8"), run_id)
        except UnicodeDecodeError:
            raise HTTPException(400, {"refused": ["the body is not UTF-8 text"]}) from None
        except ImportRefused as err:
            raise HTTPException(400, {"refused": err.problems[:PROBLEMS_SHOWN], "problems": len(err.problems)}) from None
        return {"kind": kind, **await publish_new(events, seen_in_view(view), publish)}

    return router


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m contexts.learning.lab_import", description=__doc__.split("\n\n")[0])
    parser.add_argument("--telemetry", type=Path, help="lab/train.py's telemetry.jsonl")
    parser.add_argument("--eval", type=Path, help="lab/eval_gate.py's scores.json")
    parser.add_argument("--manifest", type=Path, help="a model package's manifest.json")
    parser.add_argument("--ledger", type=Path, help="the lab's ledger.jsonl, kept whole under <data-dir>/lab_state for the Experiments tab")
    parser.add_argument("--run-id", help="the run the files belong to; a line of another run refuses the file")
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("ATLAS_DATA_DIR") or "data"))
    args = parser.parse_args(argv)
    files = [(kind, getattr(args, kind)) for kind in KINDS if getattr(args, kind)]
    if not files and not args.ledger:
        parser.error("give --telemetry, --eval, --manifest or --ledger")
    run_id = args.run_id
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
            events = events_of(kind, path.read_text(encoding="utf-8"), run_id)
            run_id = run_id or next((e.run_id for e in events if getattr(e, "run_id", None)), None)
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
