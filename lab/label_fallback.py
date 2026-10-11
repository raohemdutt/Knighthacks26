"""Make the labeled tool-calling dataset that fixtures/learning/tool-calling-labels/ holds, from either source.

    python lab/label_fallback.py convert fixtures/learning/tool-calling-fixture fixtures/learning/tool-calling-labels
    python lab/label_fallback.py package <labeled copy> fixtures/learning/tool-calling-fixture \
        fixtures/learning/tool-calling-labels --state-dir <ledger folder> --run-id <id> [--teacher-info info.json]

Both write the same folder, which lab/dataset.py loads and lab/format_traces.py reads:

    manifest.json        the fixture's DatasetManifest, unchanged
    examples.jsonl       the fixture's examples, each with its target: the JSON text
                         {"tool_calls": [{"name", "arguments"}], "answer"}, the format lab/label.py stores
    split.json           the fixture's frozen split, unchanged
    traces.jsonl         one row per labeled example: {"id", "run_id", "tool_calls": [{"name", "arguments",
                         "result"}], "answer"}, the calls with what each one returned
    label_manifest.json  where the labels came from, what the input was (sha256), the counts, the per-example
                         means and the sha256 of every file here
    rejects.jsonl        one row per example that has no label, with the reason as a word (never the text)
    label_ledger.jsonl   package only: the lab ledger's label-example rows of the run, one per example asked

convert is the cut-off route of owner ruling 12. When the teacher is not labeling in time, the reference traces
already in the fixture's expected.jsonl (the calls the agent's own pre-fetch makes, their results and a reference
answer) become the labels, and the manifest says "source: reference traces, teacher unavailable". Each row is checked
against the task's output schema exactly as lab/label.py checks a teacher's label, and a row that fails is left
without a target and logged in rejects.jsonl, never repaired. expected.jsonl is only read.

package takes the folder lab/label.py labeled (a copy of the fixture, named for its dataset id), re-checks every
target against the schema, and publishes it under the name it will keep, with the run's ledger rows. The teacher's
own facts (checkpoint, the flags the server was seen to honor) come from --teacher-info, a JSON file, because this
module asks no server anything.

Either way the output folder must not exist yet or must be empty, and the command prints one JSON line and exits 0
when every example has a target.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if not __package__:  # run by path: the repo root, one folder up, is what holds contracts/ and lab/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab import label
from lab.dataset import DatasetError, load, write_targets
from lab.machine import ledger, stop
from lab.machine.record import recorded

__all__ = ["SOURCE_TEACHER", "SOURCE_REFERENCE", "MANIFEST", "REJECTS", "LEDGER_ROWS", "LabelsError", "convert",
           "package", "main"]

SOURCE_TEACHER = "teacher"
SOURCE_REFERENCE = "reference traces, teacher unavailable"
MANIFEST = "label_manifest.json"
REJECTS = "rejects.jsonl"
LEDGER_ROWS = "label_ledger.jsonl"
EXPECTED = "expected.jsonl"
COPIED = ("manifest.json", "examples.jsonl", "split.json")
LOCK = ".examples.lock"
PER_EXAMPLE = ("label.seconds", "label.prompt_tokens", "label.completion_tokens", "label.tokens", "label.requests",
               "label.tool_calls")


class LabelsError(Exception):
    """The folder could not be made; str() says why."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            raise LabelsError(f"{path.name} line {number}: not JSON") from None
        if not isinstance(row, dict):
            raise LabelsError(f"{path.name} line {number}: not a JSON object")
        rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8",
                    newline="\n")


def _today_et(now: dt.datetime) -> str:
    try:
        from zoneinfo import ZoneInfo
        return now.astimezone(ZoneInfo("America/New_York")).date().isoformat()
    except Exception:   # no tz database (a bare Windows install): the offset in force on 2026-10-10
        return now.astimezone(dt.timezone(dt.timedelta(hours=-4))).date().isoformat()


def _prepare(source: Path, out: Path) -> None:
    for name in COPIED:
        if not (source / name).is_file():
            raise LabelsError(f"{source.name}: {name} is missing")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise LabelsError(f"{out.name}: already exists and is not empty; remove it first")
    out.mkdir(parents=True, exist_ok=True)
    for name in COPIED:
        shutil.copyfile(source / name, out / name)
    (out / "images").mkdir(exist_ok=True)
    (out / "images" / ".gitkeep").write_bytes(b"")


def _input_facts(source: Path, count: int) -> dict[str, Any]:
    split = json.loads((source / "split.json").read_text(encoding="utf-8"))
    groups = {row.get("frame_ref") for row in _jsonl(source / "examples.jsonl")}
    return {"dataset": source.name, "examples": count, "groups": len(groups),
            "examples_sha256": _sha256(source / "examples.jsonl"),
            "manifest_sha256": _sha256(source / "manifest.json"),
            "split_sha256": _sha256(source / "split.json"), "train": len(split.get("train", [])),
            "held_out": len(split.get("held_out", [])),
            **({"expected_sha256": _sha256(source / EXPECTED)} if (source / EXPECTED).is_file() else {})}


def _publish(source: Path, out: Path, targets: Mapping[str, str], traces: Mapping[str, Mapping[str, Any]],
             rejects: Sequence[Mapping[str, Any]], manifest: dict[str, Any],
             ledger_rows: Sequence[Mapping[str, Any]] | None) -> dict[str, Any]:
    """Write the folder: the fixture's files, the targets, the traces of the labeled examples (in example order),
    the reject log, the optional ledger rows and the manifest, which holds the sha256 of all of them."""
    data = load(out, require_images=False)
    order = [example.id for example in data.examples]
    if targets:
        write_targets(out, dict(targets))
    _write_jsonl(out / "traces.jsonl", [traces[key] for key in order if key in targets and key in traces])
    _write_jsonl(out / REJECTS, list(rejects))
    if ledger_rows is not None:
        _write_jsonl(out / LEDGER_ROWS, list(ledger_rows))
    (out / LOCK).unlink(missing_ok=True)   # the lock file write_targets leaves behind is not part of the dataset
    reread = load(out, require_images=False)
    labeled = len(reread.labeled())
    manifest["counts"].update({"examples": len(order), "labeled": labeled, "unlabeled": len(order) - labeled})
    names = sorted(path.name for path in out.iterdir() if path.is_file() and path.name != MANIFEST)
    manifest["files"] = {name: _sha256(out / name) for name in names}
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
                                newline="\n")
    return manifest


def convert(source: str | Path, out: str | Path, *, now: dt.datetime | None = None) -> dict[str, Any]:
    """Turn the reference traces in source/expected.jsonl into labels in out (see the module docstring); returns the
    manifest written. LabelsError when the fixture cannot be read or out is not empty."""
    source, out = Path(source), Path(out)
    now = now or dt.datetime.now(dt.timezone.utc)
    if not (source / EXPECTED).is_file():
        raise LabelsError(f"{source.name}: {EXPECTED} is missing; there are no reference traces to convert")
    try:
        data = load(source, require_images=False)
    except DatasetError as err:
        raise LabelsError(f"the fixture does not load: {err}") from None
    _prepare(source, out)
    validator = label._validator(data.manifest.spec.output_schema) if data.manifest.spec.output_schema else None
    reference = {row.get("id"): row for row in _jsonl(source / EXPECTED)}
    run_id = f"reference-{now:%Y%m%dT%H%M%SZ}"
    targets: dict[str, str] = {}
    traces: dict[str, dict[str, Any]] = {}
    rejects: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}

    def reject(example_id: str, reason: str, detail: str | None = None) -> None:
        reasons[reason] = reasons.get(reason, 0) + 1
        rejects.append({"id": example_id, "reason": reason, **({"detail": detail} if detail else {})})

    for example in data.examples:
        row = reference.get(example.id)
        if row is None:
            reject(example.id, "no_reference")
            continue
        made = row.get("label") if isinstance(row.get("label"), dict) else {}
        calls = made.get("tool_calls") if isinstance(made.get("tool_calls"), list) else None
        answer = made.get("answer") if isinstance(made.get("answer"), str) else None
        results = row.get("tool_results") if isinstance(row.get("tool_results"), list) else None
        if calls is None or answer is None or results is None:
            reject(example.id, "malformed")
            continue
        if len(results) != len(calls):
            reject(example.id, "results_mismatch", f"{len(calls)} calls, {len(results)} results")
            continue
        problem = label._text_problem(answer.strip())
        if problem is not None:
            reject(example.id, problem)
            continue
        value = {"tool_calls": [{"name": call.get("name"), "arguments": call.get("arguments")} for call in calls
                                if isinstance(call, dict)], "answer": answer.strip()}
        if validator is not None:
            problem = label._schema_problem(validator, value)
            if problem is not None:
                reject(example.id, "schema", problem)
                continue
        targets[example.id] = json.dumps(value, ensure_ascii=False)
        traces[example.id] = {"id": example.id, "run_id": run_id,
                              "tool_calls": [{**call, "result": result} for call, result in
                                             zip(value["tool_calls"], results, strict=True)],
                              "answer": value["answer"]}
    manifest = {"version": 1, "dataset_id": data.dataset_id, "folder": out.name, "source": SOURCE_REFERENCE,
                "created": now.isoformat(timespec="seconds"), "date": _today_et(now), "teacher": None,
                "reference": {"file": EXPECTED, "sha256": _sha256(source / EXPECTED), "run_id": run_id,
                              "what": "the calls the agent's own pre-fetch makes, their results and a reference "
                                      "answer; expected.jsonl is the gold of the agreement check and was not changed"},
                "input": _input_facts(source, len(data.examples)),
                "counts": {"asked": 0, "accepted": len(targets), "rejected": dict(sorted(reasons.items())),
                           "failed": {}}}
    return _publish(source, out, targets, traces, rejects, manifest, None)


def _last_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """The newest ledger row of each example (the rows are in ledger order)."""
    return {row["refs"]["example_id"]: row for row in rows}


def package(labeled: str | Path, source: str | Path, out: str | Path, *, state_dir: str | Path | None = None,
            run_ids: Sequence[str] = (), teacher_info: Mapping[str, Any] | None = None,
            now: dt.datetime | None = None) -> dict[str, Any]:
    """Publish a folder that lab/label.py labeled as out (see the module docstring); returns the manifest written.
    The ledger rows of run_ids (every run of the dataset in the state dir when none are named) are copied into the
    folder and give the counts and the per-example means. LabelsError when the folder cannot be made."""
    labeled, source, out = Path(labeled), Path(source), Path(out)
    now = now or dt.datetime.now(dt.timezone.utc)
    try:
        data = load(labeled, require_images=False)
        original = load(source, require_images=False)
    except DatasetError as err:
        raise LabelsError(f"a dataset does not load: {err}") from None
    if data.dataset_id != original.dataset_id or [e.id for e in data.examples] != [e.id for e in original.examples]:
        raise LabelsError("the labeled folder and the fixture are not the same dataset")
    _prepare(source, out)
    validator = label._validator(original.manifest.spec.output_schema)
    rows = [row for row in ledger.read(state_dir=state_dir, verb="label-example")
            if row.get("refs", {}).get("dataset_id") == data.dataset_id
            and (not run_ids or row["refs"].get("run_id") in run_ids)]
    traces: dict[str, dict[str, Any]] = {}
    if (labeled / label.TRACES).is_file():
        traces = {row["id"]: row for row in _jsonl(labeled / label.TRACES)}   # an id asked twice: the last row counts
    targets: dict[str, str] = {}
    rejects: list[dict[str, Any]] = []
    for example in data.labeled():
        try:
            value = label._parse_json(example.target or "")
        except ValueError:
            rejects.append({"id": example.id, "reason": "not_json"})
            continue
        problem = label._schema_problem(validator, value)
        if problem is not None:
            rejects.append({"id": example.id, "reason": "schema", "detail": problem})
        elif example.id not in traces:
            rejects.append({"id": example.id, "reason": "no_trace"})
        else:
            targets[example.id] = example.target or ""
    newest = _last_rows(rows)
    for example in data.examples:
        if example.id in targets:
            continue
        row = newest.get(example.id)
        if row is not None and row.get("outcome") != "accepted":
            note = row.get("note") or ""
            reason = note.split(":", 1)[0] or "unknown"
            detail = note.split(":", 1)[1].strip() if ":" in note else None
            rejects.append({"id": example.id, "reason": reason, "outcome": row["outcome"],
                            **({"detail": detail} if detail else {})})
        elif not any(item["id"] == example.id for item in rejects):
            rejects.append({"id": example.id, "reason": "not_asked"})
    by_outcome: dict[str, dict[str, int]] = {"rejected": {}, "failed": {}}
    accepted = 0
    sums = {name: [0.0, 0] for name in PER_EXAMPLE}
    for row in rows:
        outcome = row.get("outcome")
        if outcome == "accepted":
            accepted += 1
        elif outcome in by_outcome:
            reason = (row.get("note") or "unknown").split(":", 1)[0]
            by_outcome[outcome][reason] = by_outcome[outcome].get(reason, 0) + 1
        for name in PER_EXAMPLE:
            value = (row.get("metrics") or {}).get(name)
            if isinstance(value, (int, float)):
                sums[name][0] += value
                sums[name][1] += 1
    asked = len(rows)
    runs = sorted({row["refs"].get("run_id") for row in rows} - {None})
    summaries = [row for row in ledger.read(state_dir=state_dir, verb="label")
                 if row.get("refs", {}).get("run_id") in runs]
    wall = round(sum((row.get("metrics") or {}).get("labels.wall_seconds", 0) for row in summaries), 3)
    per_example = {f"{name}.mean": round(total / count, 3) for name, (total, count) in sums.items() if count}
    per_example.update({f"{name}.total": round(total, 3) for name, (total, count) in sums.items() if count})
    if wall and asked:   # examples run side by side, so the wall clock per example is shorter than label.seconds
        per_example.update({"wall_seconds.total": wall, "wall_seconds.per_example": round(wall / asked, 3)})
    conditions = next((row.get("conditions") for row in reversed(rows) if row.get("conditions")), {})
    manifest = {"version": 1, "dataset_id": data.dataset_id, "folder": out.name, "source": SOURCE_TEACHER,
                "created": now.isoformat(timespec="seconds"), "date": _today_et(now),
                "teacher": {"model": conditions.get("model"), "checkpoint": conditions.get("checkpoint"),
                            **(dict(teacher_info) if teacher_info else {})},
                "decode": {key: conditions.get(key) for key in ("thinking", "temperature", "top_p", "max_tokens",
                                                                "seed", "stream")},
                "runs": runs,
                "input": _input_facts(source, len(original.examples)),
                "counts": {"asked": asked, "accepted": accepted, "rejected": dict(sorted(by_outcome["rejected"].items())),
                           "failed": dict(sorted(by_outcome["failed"].items()))},
                "per_example": per_example}
    return _publish(source, out, targets, traces, rejects, manifest, rows)


@recorded("label-fallback")
@stop.guarded
def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Make the tool-calling labels folder from the reference traces (convert) or from a teacher run (package)."""
    parser = argparse.ArgumentParser(prog="python lab/label_fallback.py", description=main.__doc__,
                                     allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("convert", help="labels from expected.jsonl: the teacher was not available")
    one.add_argument("source", help="the fixture folder (with expected.jsonl)")
    one.add_argument("out", help="the folder to make (fixtures/learning/tool-calling-labels)")
    two = commands.add_parser("package", help="publish a folder lab/label.py labeled")
    two.add_argument("labeled", help="the labeled copy of the fixture")
    two.add_argument("source", help="the fixture folder, untouched")
    two.add_argument("out", help="the folder to make")
    two.add_argument("--state-dir", help="the lab ledger's folder (default: LAB_STATE_DIR or lab/state)")
    two.add_argument("--run-id", action="append", default=[], dest="run_ids",
                     help="a label.py run to count (repeat); default: every run of the dataset in the ledger")
    two.add_argument("--teacher-info", help="a JSON file of facts about the server to put under teacher")
    args = parser.parse_args(argv)
    try:
        if args.command == "convert":
            manifest = convert(args.source, args.out)
        else:
            info = json.loads(Path(args.teacher_info).read_text(encoding="utf-8")) if args.teacher_info else None
            manifest = package(args.labeled, args.source, args.out, state_dir=args.state_dir, run_ids=args.run_ids,
                               teacher_info=info)
    except (LabelsError, label.LabelRefused, OSError, ValueError) as err:
        return {"ok": False, "outcome": "refused", "error": str(err)}
    counts = manifest["counts"]
    complete = counts["unlabeled"] == 0
    return {"ok": complete, "outcome": "ok" if complete else "incomplete", "source": manifest["source"],
            "folder": manifest["folder"], "counts": counts,
            "metrics": {"labels.examples": counts["examples"], "labels.labeled": counts["labeled"],
                        "labels.unlabeled": counts["unlabeled"]},
            "refs": {"dataset_id": manifest["dataset_id"]}}


if __name__ == "__main__":
    answer = main()
    print(json.dumps(answer))
    sys.exit(0 if answer.get("ok", True) else 1)
