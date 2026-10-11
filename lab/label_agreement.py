"""Spot-check the teacher's labels: ask again about 30 labeled examples and see how often the second answer agrees.

    python -m lab label-agreement data/exports/desk-objects-3fa91c2b --account C
    python -m lab label-agreement fixtures/learning/tool-calling-fixture --url http://127.0.0.1:8000 --sample seeded

What it reads. A dataset folder with labels in it, as lab/label.py writes them (examples.jsonl, a label in each
example's target), through lab/dataset.py. The folder must be one lab/label.py may label (an approved export or the
fixture). It is never written: the sampled examples are copied, without their labels, to a temporary folder, labeled
there by lab.label.label_dataset and compared, and the copy is deleted.

What it asks. The sample is the first 30 labeled examples, or with --sample seeded 30 drawn with --sample-seed. The
second pass changes two things against the first: the sampling seed (--seed, default 1; the labels were made with
seed 0, the default of lab/label.py, told here with --first-seed) and the prompt, which gets REWORD_PREFIX in front of
the task's instruction. Everything else, the decode settings, the tools and the checks, is lab/label.py's. A first
and a second pass with the same seed are refused: the check would measure nothing.

How it compares. A label that is a JSON object is compared one top-level field at a time (tool-trace labels have
"tool_calls" and "answer"); any other label, free text or a JSON list, is one field called "label". Values are equal
when they are the same JSON value, text with its white space collapsed. An example the second pass could not label
(rejected or failed) disagrees on every field, with None as its second value, and is listed under "unanswered".

The verdict. The agreement rate is agreeing fields over compared fields, per field and overall. It is "consistent"
when the overall rate reaches the threshold, "inconsistent" when it does not, and "insufficient" when fewer than 30
examples were labeled to draw from. The threshold is the number the learning rubric names for LT2's agreement check
(lab/refine/learning/RUBRIC.md; found by rubric_threshold) and 0.9 when it names none, which the report says. This
module never writes the rubric: the report holds the line the merge pass should add, with its citation.

What is recorded. A report in JSON and a short Markdown table (--out), and one ledger row, verb label-agreement,
outcome the verdict, with label.agreement_rate, label.agreement_sample and label.agreement_disagreements and the
seeds and threshold as conditions. The label.agreement_rate here is the teacher agreeing with itself, not
learning.label_agreement (agreement with gold).

Exit code: 0 for a verdict of consistent or inconsistent, 1 for insufficient, a refusal or a teacher that failed.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if not __package__:  # run by path: the repo root, one folder up, is what holds lab/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from lab import hub, label
from lab.dataset import DatasetError, load
from lab.machine import ledger, stop
from lab.machine.record import record

__all__ = ["DEFAULT_THRESHOLD", "SAMPLE_SIZE", "REWORD_PREFIX", "AgreementRefused", "RewordedTeacher",
           "rubric_threshold", "compare_targets", "check_agreement", "write_report", "main"]

CLI_VERB = "label-agreement"   # registers python -m lab label-agreement
DEFAULT_THRESHOLD = 0.9
SAMPLE_SIZE = 30
FIRST_SEED = 0                       # lab/label.py's default seed: what the labels being checked were made with
SECOND_SEED = 1
REWORD_PREFIX = "Read the instructions below with care and follow them exactly.\n\n"
RUBRIC = Path(__file__).resolve().parent / "refine" / "learning" / "RUBRIC.md"
REPORT_JSON = "label_agreement.json"
REPORT_MD = "label_agreement.md"
DEFAULT_SOURCE = "default: lab/refine/learning/RUBRIC.md LT2 names no number, so 0.9 is used"
LT2_SECTION = re.compile(r"^### LT2\..*?(?=^### |\Z)", re.DOTALL | re.MULTILINE)
LT2_NUMBER = re.compile(r"agreement[^.\n]{0,60}?(?:threshold|at least|of)\s+(\d*\.\d+|\d+\s*%)", re.IGNORECASE)


class AgreementRefused(Exception):
    """The check may not run: a bad seed pair, or a dataset lab/label.py would not label."""


# --- the second pass --------------------------------------------------------------------------

class RewordedTeacher:
    """A label.Teacher whose requests carry REWORD_PREFIX in front of the task's instruction: the system message of a
    tool-trace task, else the first text of the user message. Everything else is the wrapped teacher's."""

    def __init__(self, teacher: label.Teacher, prefix: str = REWORD_PREFIX) -> None:
        self._teacher = teacher
        self.prefix = prefix

    def __getattr__(self, name: str) -> Any:
        return getattr(self._teacher, name)

    def __enter__(self) -> "RewordedTeacher":
        return self

    def __exit__(self, *exc: object) -> None:
        self._teacher.close()

    def chat(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> label.Completion:
        return self._teacher.chat(self._reword(messages), **kwargs)

    def _reword(self, messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        out = [dict(message) for message in messages]
        target = next((m for m in out if m.get("role") == "system"), None) or next(
            (m for m in out if m.get("role") == "user"), None)
        if target is None:
            return out
        content = target.get("content")
        if isinstance(content, str):
            target["content"] = self.prefix + content
        elif isinstance(content, list):
            parts = [dict(part) if isinstance(part, Mapping) else part for part in content]
            for part in parts:
                if isinstance(part, dict) and part.get("type") == "text":
                    part["text"] = self.prefix + part.get("text", "")
                    break
            target["content"] = parts
        return out


# --- comparing --------------------------------------------------------------------------------

def _collapse(value: Any) -> Any:
    return " ".join(value.split()) if isinstance(value, str) else value


def _fields(text: str | None) -> dict[str, Any]:
    """A label as {field: value}: a JSON object's top-level fields, else one field "label"."""
    if text is None:
        return {}
    try:
        parsed = json.loads(text)
    except ValueError:
        return {"label": _collapse(text)}
    if isinstance(parsed, dict):
        return {key: _collapse(value) for key, value in parsed.items()}
    return {"label": _collapse(parsed)}


def compare_targets(first: str, second: str | None) -> dict[str, tuple[bool, Any, Any]]:
    """{field: (agree, first value, second value)} for two labels; a label the second pass lacks (None) disagrees on
    every field of the first, and a field one label lacks has None for its value."""
    one, two = _fields(first), _fields(second)
    names = sorted(set(one) | set(two))
    return {name: (second is not None and name in one and name in two and one[name] == two[name],
                   one.get(name), two.get(name)) for name in names}


def rubric_threshold(path: str | Path | None = None) -> float | None:
    """The agreement threshold LT2 in the rubric names, or None when it names none (or the file is missing)."""
    try:
        text = Path(path or RUBRIC).read_text(encoding="utf-8")
    except OSError:
        return None
    section = LT2_SECTION.search(text)
    found = LT2_NUMBER.search(section.group(0)) if section else None
    if found is None:
        return None
    number = found.group(1).strip()
    return float(number[:-1]) / 100 if number.endswith("%") else float(number)


def _rate(agree: int, total: int) -> float:
    return agree / total if total else 0.0


def _sample(labeled: Sequence[Any], mode: str, seed: int, size: int) -> list[Any]:
    if mode == "first":
        return list(labeled[:size])
    if len(labeled) <= size:
        return list(labeled)
    picked = set(random.Random(seed).sample(range(len(labeled)), size))
    return [example for index, example in enumerate(labeled) if index in picked]


def _blank_copy(root: Path, ids: set[str], dataset_id: str, into: Path) -> Path:
    """A copy of the dataset at <into>/data/exports/<dataset_id>/ with the labels of ids removed."""
    folder = into / "data" / "exports" / dataset_id
    shutil.copytree(root, folder, ignore=shutil.ignore_patterns(label.TRACES, label.TRACES_LOCK, ".examples.lock"))
    path = folder / "examples.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        if row.get("id") in ids:
            row["target"] = None
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return folder


# --- the check --------------------------------------------------------------------------------

def check_agreement(root: str | Path, teacher: label.Teacher, *, sample: str = "first", sample_seed: int = 0,
                    size: int = SAMPLE_SIZE, second_seed: int = SECOND_SEED, first_seed: int = FIRST_SEED,
                    threshold: float | None = None, run_id: str | None = None, workers: int = 1,
                    golden_dir: str | None = None, text_only: bool | None = None,
                    max_tokens: int = label.DEFAULT_MAX_TOKENS, expect_model: str | None = label.EXPECTED_MODEL,
                    environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Ask the teacher again about the sample and compare (see the module docstring); the report as a dict.
    DatasetError when the folder does not load, AgreementRefused when the check may not run."""
    if sample not in ("first", "seeded"):
        raise AgreementRefused(f"sample must be 'first' or 'seeded', not {sample!r}")
    if second_seed == first_seed:
        raise AgreementRefused(f"the second pass needs another seed than the first: both are {first_seed}")
    root = Path(root)
    data = load(root, require_images=False)
    problem = label.source_problem(root, data.manifest, environ=environ)
    if problem is not None:
        raise AgreementRefused(problem)
    run_id = run_id or "agr-" + label._new_run_id()
    chosen = _sample(data.labeled(), sample, sample_seed, size)
    first = {example.id: example.target or "" for example in chosen}
    second: dict[str, str | None] = {example.id: None for example in chosen}
    unanswered: list[dict[str, str]] = []
    if chosen:
        with tempfile.TemporaryDirectory(prefix="label-agreement-") as scratch:
            copy = _blank_copy(root, set(first), data.dataset_id, Path(scratch))
            with RewordedTeacher(teacher) as reworded:
                run = label.label_dataset(
                    copy, reworded, decode=label.Decode(seed=second_seed, max_tokens=max_tokens), golden_dir=golden_dir,
                    text_only=text_only, workers=workers, ids=list(first), expect_model=expect_model,
                    run_id=run_id + "-second", environ=environ)
            if run.outcome != "ok" and run.error:
                raise AgreementRefused(f"the second pass stopped: {run.error}")
            for example in load(copy, require_images=False).examples:
                if example.id in second and example.target and example.target.strip():
                    second[example.id] = example.target
    why = _reasons(run_id + "-second") if chosen else {}
    fields: dict[str, dict[str, Any]] = {}
    disagreements: list[dict[str, Any]] = []
    agreeing_examples = 0
    for example_id, text in first.items():
        every = compare_targets(text, second[example_id])
        if second[example_id] is None:
            unanswered.append({"id": example_id, "reason": why.get(example_id, "unknown")})
        for name, (same, one, two) in every.items():
            stats = fields.setdefault(name, {"agree": 0, "total": 0})
            stats["total"] += 1
            stats["agree"] += int(same)
            if not same:
                disagreements.append({"id": example_id, "field": name, "first": one, "second": two})
        agreeing_examples += int(all(same for same, _, _ in every.values()))
    for stats in fields.values():
        stats["rate"] = _rate(stats["agree"], stats["total"])
    agree = sum(stats["agree"] for stats in fields.values())
    total = sum(stats["total"] for stats in fields.values())
    overall = {"agree": agree, "total": total, "rate": _rate(agree, total)}
    named = rubric_threshold() if threshold is None else None
    if threshold is not None:
        value, source = threshold, "--threshold"
    elif named is not None:
        value, source = named, "lab/refine/learning/RUBRIC.md LT2"
    else:
        value, source = DEFAULT_THRESHOLD, DEFAULT_SOURCE
    if len(chosen) < size:
        verdict = "insufficient"
    else:
        verdict = "consistent" if overall["rate"] >= value else "inconsistent"
    report: dict[str, Any] = {
        "run_id": run_id, "dataset_id": data.dataset_id, "task": data.manifest.spec.task, "verdict": verdict,
        "sample": {"mode": sample, "seed": sample_seed if sample == "seeded" else None, "size": len(chosen),
                   "wanted": size, "ids": [example.id for example in chosen]},
        "second_pass": {"seed": second_seed, "first_seed": first_seed, "rewording": REWORD_PREFIX.strip()},
        "threshold": {"value": value, "source": source},
        "overall": overall, "fields": dict(sorted(fields.items())),
        "examples": {"agree": agreeing_examples, "total": len(chosen)},
        "unanswered": unanswered, "disagreements": disagreements,
    }
    report["proposed_rubric_line"] = _rubric_line(report)
    report["proposed_rubric_note"] = ("Not written to lab/refine/learning/RUBRIC.md: tasks never write rubrics, the "
                                      "merge pass does. Add the line to LT2 with its citation.")
    return report


def _reasons(run_id: str) -> dict[str, str]:
    """Why each example of the second pass was not labeled, from its label-example ledger row (a word)."""
    rows = ledger.read(verb="label-example", refs={"run_id": run_id})
    return {row["refs"]["example_id"]: (row.get("note") or row["outcome"]).split(":")[0]
            for row in rows if row["outcome"] != "accepted" and "example_id" in row.get("refs", {})}


def _rubric_line(report: Mapping[str, Any]) -> str:
    threshold, overall, sample = report["threshold"], report["overall"], report["sample"]
    return (f"LT2 label agreement: a second teacher pass (seed {report['second_pass']['seed']}, reworded prompt) over "
            f"{sample['size']} labeled examples must agree field by field on at least {threshold['value']:.2f} "
            f"({threshold['source']}); measured {overall['rate']:.3f}, verdict {report['verdict']}. Source: "
            f"lab/label_agreement.py run {report['run_id']} (ledger verb label-agreement, refs run_id "
            f"{report['run_id']}), dataset {report['dataset_id']}.")


# --- the report -------------------------------------------------------------------------------

def _cell(value: Any, limit: int = 60) -> str:
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    text = " ".join(text.split()).replace("|", "/")
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _markdown(report: Mapping[str, Any]) -> str:
    threshold, overall = report["threshold"], report["overall"]
    lines = [f"# Label agreement: {report['dataset_id']} ({report['run_id']})", "",
             f"Verdict: **{report['verdict']}** - overall agreement {overall['rate']:.3f} against a threshold of "
             f"{threshold['value']:.2f} ({threshold['source']}).",
             f"Sample: {report['sample']['size']} of {report['sample']['wanted']} wanted ({report['sample']['mode']}); "
             f"second pass seed {report['second_pass']['seed']} (first {report['second_pass']['first_seed']}), "
             "prompt reworded.", "",
             "| field | agree | total | rate |", "| --- | --- | --- | --- |"]
    lines += [f"| {name} | {s['agree']} | {s['total']} | {s['rate']:.3f} |" for name, s in report["fields"].items()]
    lines += [f"| overall | {overall['agree']} | {overall['total']} | {overall['rate']:.3f} |", ""]
    if report["unanswered"]:
        lines += [f"Not labeled by the second pass: {', '.join(u['id'] for u in report['unanswered'])}.", ""]
    if report["disagreements"]:
        lines += ["| example | field | first | second |", "| --- | --- | --- | --- |"]
        lines += [f"| {d['id']} | {d['field']} | {_cell(d['first'])} | {_cell(d['second'])} |"
                  for d in report["disagreements"]]
        lines.append("")
    lines += ["Proposed RUBRIC line (for the merge pass):", "", f"> {report['proposed_rubric_line']}", ""]
    return "\n".join(lines)


def write_report(report: Mapping[str, Any], out: str | Path) -> dict[str, Path]:
    """Write label_agreement.json and label_agreement.md into the folder out; the paths."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"json": out / REPORT_JSON, "markdown": out / REPORT_MD}
    paths["json"].write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    paths["markdown"].write_text(_markdown(report), encoding="utf-8")
    return paths


# --- the entry point --------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None, *, transport: httpx.BaseTransport | None = None) -> dict[str, Any]:
    """Re-label 30 labeled examples with a second teacher pass and report how often the labels agree."""
    stop.guard()
    parser = argparse.ArgumentParser(prog="python -m lab label-agreement", description=main.__doc__,
                                     allow_abbrev=False)
    parser.add_argument("dataset", help="a labeled export (data/exports/<dataset_id>/) or the no-camera fixture")
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--account", type=str.upper, help="the hub account whose pod serves the teacher (C, D, ...)")
    where.add_argument("--url", help="the teacher server's address, without /v1 (a server reached directly)")
    parser.add_argument("--port", type=int, default=label.TEACHER_PORT)
    parser.add_argument("--sample", choices=("first", "seeded"), default="first")
    parser.add_argument("--sample-seed", type=int, default=0, help="the draw of --sample seeded")
    parser.add_argument("--size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=SECOND_SEED, help="the second pass's sampling seed")
    parser.add_argument("--first-seed", type=int, default=FIRST_SEED, help="the seed the labels were made with")
    parser.add_argument("--threshold", type=float, help="the agreement rate that makes labels consistent "
                                                        "(default: the rubric's LT2 number, else 0.9)")
    parser.add_argument("--out", help="folder for the JSON and Markdown report (default: label-agreement/<run id> in "
                                      "the lab state folder)")
    parser.add_argument("--run-id")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=label.DEFAULT_MAX_TOKENS)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--golden")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--expect-model", default=label.EXPECTED_MODEL)
    args = parser.parse_args(argv)
    try:
        if args.url:
            teacher = label.Teacher(args.url, transport=transport, stream=not args.no_stream)
        else:
            teacher = label.Teacher.from_hub(hub.load_accounts([args.account])[args.account], port=args.port,
                                             transport=transport, stream=not args.no_stream)
    except hub.MissingCredentials as err:
        return {"ok": False, "outcome": "refused", "error": f"the account's keys are missing: {err}"}
    with teacher:
        try:
            report = check_agreement(
                args.dataset, teacher, sample=args.sample, sample_seed=args.sample_seed, size=args.size,
                second_seed=args.seed, first_seed=args.first_seed, threshold=args.threshold, run_id=args.run_id,
                workers=args.workers, golden_dir=args.golden, text_only=True if args.text_only else None,
                max_tokens=args.max_tokens, expect_model=args.expect_model or None)
        except DatasetError as err:
            return {"ok": False, "outcome": "invalid", "problems": err.problems, "error": str(err)}
        except (AgreementRefused, label.LabelRefused) as err:
            return {"ok": False, "outcome": "refused", "error": str(err)}
    out = Path(args.out) if args.out else ledger.resolve_state_dir() / "label-agreement" / report["run_id"]
    paths = write_report(report, out)
    overall = report["overall"]
    metrics = {"label.agreement_rate": overall["rate"], "label.agreement_sample": report["sample"]["size"],
               "label.agreement_disagreements": len(report["disagreements"])}
    conditions = {"verdict": report["verdict"], "threshold": report["threshold"]["value"],
                  "second_seed": report["second_pass"]["seed"], "first_seed": report["second_pass"]["first_seed"],
                  "sample": report["sample"]["mode"], "reworded": True}
    refs = {"run_id": report["run_id"]}
    record("label-agreement", report["verdict"], metrics, refs, conditions=conditions,
           units={"label.agreement_rate": "ratio", "label.agreement_sample": "examples",
                  "label.agreement_disagreements": "fields"})
    return {"ok": report["verdict"] != "insufficient", "outcome": report["verdict"], "dataset_id": report["dataset_id"],
            "run_id": report["run_id"], "overall": overall, "threshold": report["threshold"],
            "report": {name: str(path) for name, path in paths.items()}, "metrics": metrics, "refs": refs}


if __name__ == "__main__":
    answer = main()
    print(json.dumps(answer))
    sys.exit(0 if answer.get("ok", True) else 1)
