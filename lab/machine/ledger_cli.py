"""python -m lab ledger <sub-verb>: write to and read the lab ledger from the command line.

    python -m lab ledger add VERB OUTCOME [--metric NAME=NUMBER] [--unit NAME=UNIT] [--ref KEY=VALUE]
                             [--task ID] [--node NODE] [--condition KEY=VALUE] [--source TEXT]
                             [--note TEXT] [--low-confidence]
    python -m lab ledger tail [N]
    python -m lab ledger grep [PATTERN] [--verb V] [--outcome O] [--actor A] [--tier T] [--task ID]
                              [--node N] [--ref KEY=VALUE] [--metric NAME] [--since WHEN]
                              [--before WHEN] [--limit N]

add appends one row (lab.machine.record.record). The actor, tier and task come from $LAB_ACTOR,
$LAB_TIER and $LAB_TASK, not from a flag, so a sitting cannot write as another; --task overrides
$LAB_TASK. --metric, --unit, --ref and --condition repeat. A metric value is a finite number. A ref
value is text, except pr, which is a number when it is digits. A condition value is read as JSON
when it is JSON (4, true) and is text otherwise. The answer is {"ok": true, "outcome": "added",
"seq": N, "ledger": path}. A row the ledger refuses writes nothing and answers outcome "refused"
with its problems, exit code 1; a ledger that cannot be written (a lock held too long, a disk error)
answers outcome "error".

tail prints the last N rows (default 10), oldest first:
{"ok": true, "outcome": "tail", "count": n, "rows": [...]}.

grep prints the rows that pass every filter given, oldest first, cut to the newest --limit of them
(default 50; 0 for all): {"ok": true, "outcome": "found" or "none", "matches": all of them, "count":
those printed, "truncated": whether some were cut, "rows": [...]}. PATTERN is a regular expression
searched, ignoring case, in each row's JSON text. --verb, --outcome, --actor, --tier, --task and
--node match the value exactly, or as a shell pattern (bench.*) when it holds * ? or [. --ref
repeats and every pair must hold. --metric NAME keeps the rows that carry that metric. --since and
--before take an age (90m, 2h, 3d) or an ISO 8601 time, UTC when it carries no offset; --since is
inclusive and --before is not.

A sub-verb is a function that gets the rest of argv and answers the way a verb of lab/__main__.py
does (a dict, an int or None). SUBVERBS names them, and a new one takes one line there.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import importlib
import inspect
import json
import math
import re
from collections.abc import Callable, Sequence
from typing import Any

from lab.machine import ledger, record

__all__ = ["SUBVERBS", "DEFAULT_TAIL", "DEFAULT_LIMIT", "main", "add_main", "tail_main", "grep_main"]

DEFAULT_TAIL = 10      # rows `tail` prints without a number
DEFAULT_LIMIT = 50     # rows `grep` prints without --limit
HELP_WORDS = ("-h", "--help", "help")

# One line per sub-verb, alphabetical: "name": "module:function".
SUBVERBS: dict[str, str] = {
    "add": "lab.machine.ledger_cli:add_main",
    "grep": "lab.machine.ledger_cli:grep_main",
    "tail": "lab.machine.ledger_cli:tail_main",
}

_AGE = re.compile(r"(\d+)([smhd])")
_AGE_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
_TEXT_FILTERS = ("verb", "outcome", "actor", "tier", "task", "node")


# --- reading the command line -------------------------------------------------------------

def _parser(sub: str, summary: str | None) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(prog=f"python -m lab ledger {sub}", description=summary)


def _pair(text: str) -> tuple[str, str]:
    name, sep, value = text.partition("=")
    if not sep or not name.strip():
        raise argparse.ArgumentTypeError(f"{text!r} must look like NAME=VALUE")
    return name.strip(), value


def _metric(text: str) -> tuple[str, int | float]:
    name, value = _pair(text)
    try:
        number: int | float = int(value)
    except ValueError:
        try:
            number = float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"the value of metric {name!r} must be a number, not {value!r}") from None
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError(f"the value of metric {name!r} must be finite, not {value!r}")
    return name, number


def _ref(text: str) -> tuple[str, str | int | None]:
    name, value = _pair(text)
    if value == "":
        return name, None
    return name, int(value) if name == "pr" and value.isdigit() else value


def _condition(text: str) -> tuple[str, Any]:
    name, value = _pair(text)
    try:
        return name, json.loads(value)
    except ValueError:
        return name, value


def _count(text: str) -> int:
    try:
        number = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
    if number < 0:
        raise argparse.ArgumentTypeError(f"{text!r} is below zero")
    return number


def _regex(text: str) -> re.Pattern[str]:
    try:
        return re.compile(text, re.IGNORECASE)
    except re.error as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not a regular expression: {exc}") from None


def _when(text: str) -> dt.datetime:
    """An age such as 90m, 2h or 3d (that long before now) or an ISO 8601 time, UTC without an offset."""
    text = text.strip()
    try:
        age = _AGE.fullmatch(text)
        if age:
            return dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=int(age[1]) * _AGE_SECONDS[age[2]])
        moment = dt.datetime.fromisoformat(text)
    except (ValueError, OverflowError):
        raise argparse.ArgumentTypeError(
            f"{text!r} is neither an age such as 90m, 2h or 3d nor an ISO 8601 time") from None
    return moment if moment.utcoffset() is not None else moment.replace(tzinfo=dt.timezone.utc)


def _unique(parser: argparse.ArgumentParser, flag: str, pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """The pairs of a repeated flag as a dict; a name given twice is a usage error."""
    found: dict[str, Any] = {}
    for name, value in pairs:
        if name in found:
            parser.error(f"{flag} {name} is given twice")
        found[name] = value
    return found


def _text_filter(value: str) -> Any:
    """The read() filter for a text field: the value itself, or a shell-pattern test when it holds * ? or [."""
    if any(mark in value for mark in "*?["):
        return lambda have: isinstance(have, str) and fnmatch.fnmatchcase(have, value)
    return value


# --- the sub-verbs ------------------------------------------------------------------------

def add_main(argv: Sequence[str]) -> dict[str, Any]:
    """Append one row to the ledger."""
    parser = _parser("add", add_main.__doc__)
    parser.add_argument("verb", help="what was done, such as bench.run")
    parser.add_argument("outcome", help="how it ended: ok, fail, error, skipped, ...")
    parser.add_argument("--metric", action="append", default=[], type=_metric, metavar="NAME=NUMBER",
                        help="a measured number, such as teacher.decode_tok_s=7.7 (repeatable)")
    parser.add_argument("--unit", action="append", default=[], type=_pair, metavar="NAME=UNIT",
                        help="the unit of one of the metrics, such as teacher.decode_tok_s=tok/s (repeatable)")
    parser.add_argument("--ref", action="append", default=[], type=_ref, metavar="KEY=VALUE",
                        help="an id the row refers to: pr, commit, slot, run_id (repeatable)")
    parser.add_argument("--task", help="the Task Board ID the row is about (default: $LAB_TASK)")
    parser.add_argument("--node", help="where a measurement was taken: pod-a, pod-b, jetson")
    parser.add_argument("--condition", action="append", default=[], type=_condition, metavar="KEY=VALUE",
                        help="what held while measuring, such as batch=4 (repeatable)")
    parser.add_argument("--source", help="where the numbers came from")
    parser.add_argument("--note", help="free text")
    parser.add_argument("--low-confidence", action="store_true", help="mark the numbers as shaky")
    args = parser.parse_args(argv)
    metrics = _unique(parser, "--metric", args.metric)
    refs = _unique(parser, "--ref", args.ref)
    fields: dict[str, Any] = {}
    for key, value in (("node", args.node), ("source", args.source), ("note", args.note),
                       ("units", _unique(parser, "--unit", args.unit) or None),
                       ("conditions", _unique(parser, "--condition", args.condition) or None),
                       ("low_confidence", True if args.low_confidence else None)):
        if value is not None:
            fields[key] = value
    try:
        seq = record.record(args.verb, args.outcome, metrics, refs, task=args.task, **fields)
    except ledger.RowError as exc:
        return {"ok": False, "outcome": "refused", "error": str(exc), "problems": exc.problems}
    except OSError as exc:
        return {"ok": False, "outcome": "error", "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "outcome": "added", "seq": seq, "ledger": str(ledger.ledger_path())}


def tail_main(argv: Sequence[str]) -> dict[str, Any]:
    """Print the last N rows of the ledger, oldest first."""
    parser = _parser("tail", tail_main.__doc__)
    parser.add_argument("n", nargs="?", type=_count, default=DEFAULT_TAIL, metavar="N",
                        help=f"how many rows (default {DEFAULT_TAIL})")
    args = parser.parse_args(argv)
    rows = ledger.tail(args.n)
    return {"ok": True, "outcome": "tail", "count": len(rows), "rows": rows}


def grep_main(argv: Sequence[str]) -> dict[str, Any]:
    """Print the ledger rows that match every filter given, oldest first."""
    parser = _parser("grep", grep_main.__doc__)
    parser.add_argument("pattern", nargs="?", type=_regex,
                        help="regular expression searched, ignoring case, in each row's JSON text")
    for name in _TEXT_FILTERS:
        parser.add_argument(f"--{name}", help=f"the row's {name}, exactly or as a shell pattern (bench.*)")
    parser.add_argument("--ref", action="append", default=[], type=_ref, metavar="KEY=VALUE",
                        help="the row refers to this id (repeatable; every pair must hold)")
    parser.add_argument("--metric", metavar="NAME", help="the row carries this metric")
    parser.add_argument("--since", type=_when, metavar="WHEN", help="at or after an age (2h) or an ISO 8601 time")
    parser.add_argument("--before", type=_when, metavar="WHEN", help="strictly before an age (2h) or an ISO 8601 time")
    parser.add_argument("--limit", type=_count, default=DEFAULT_LIMIT, metavar="N",
                        help=f"print the newest N matches (default {DEFAULT_LIMIT}; 0 for all)")
    args = parser.parse_args(argv)
    filters: dict[str, Any] = {}
    for name in _TEXT_FILTERS:
        value = getattr(args, name)
        if value is not None:
            filters[name] = _text_filter(value.lower() if name == "tier" else value)
    if args.ref:
        filters["refs"] = _unique(parser, "--ref", args.ref)
    if args.metric is not None:
        filters["has_metric"] = args.metric
    for name in ("since", "before"):
        if getattr(args, name) is not None:
            filters[name] = getattr(args, name)
    rows = ledger.read(**filters)
    if args.pattern is not None:
        rows = [row for row in rows if args.pattern.search(json.dumps(row, ensure_ascii=False))]
    matches = len(rows)
    if args.limit and matches > args.limit:
        rows = rows[-args.limit:]
    return {"ok": True, "outcome": "found" if matches else "none", "matches": matches, "count": len(rows),
            "truncated": len(rows) < matches, "rows": rows}


# --- the verb -----------------------------------------------------------------------------

def _load(target: str) -> Callable[[Sequence[str]], Any]:
    module, _, name = target.partition(":")
    return getattr(importlib.import_module(module), name)


def _summary(target: str) -> str:
    try:
        doc = inspect.getdoc(_load(target)) or ""
    except Exception as exc:
        return f"(cannot load: {type(exc).__name__}: {exc})"
    return doc.splitlines()[0] if doc else ""


def help_text() -> str:
    width = max(len(name) for name in SUBVERBS)
    lines = ["usage: python -m lab ledger <sub-verb> [args]    (python -m lab ledger <sub-verb> -h for its arguments)",
             "",
             "sub-verbs:",
             *(f"  {name:<{width}}  {_summary(target)}" for name, target in SUBVERBS.items())]
    return "\n".join(lines)


def main(argv: Sequence[str]) -> dict[str, Any] | int | None:
    """Add to, tail and search the lab ledger (ledger add, tail, grep)."""
    args = list(argv)
    if not args or args[0] in HELP_WORDS:
        print(help_text())
        return 0
    name, rest = args[0], args[1:]
    target = SUBVERBS.get(name)
    if target is None:
        return {"ok": False, "outcome": "unknown_subverb", "error": f"unknown ledger sub-verb {name!r}",
                "subverbs": sorted(SUBVERBS)}
    try:
        fn = _load(target)
    except Exception as exc:
        return {"ok": False, "outcome": "error", "error": f"cannot load {target}: {type(exc).__name__}: {exc}"}
    return fn(rest)
