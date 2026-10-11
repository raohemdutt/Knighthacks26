"""python -m lab <verb> [args]: the lab machine's command line.

Run it from the repo root. `python -m lab` (or `--help`) lists the verbs.

Every call answers with exactly one JSON object on stdout and an exit code: 0 when it worked,
1 when it did not (a refused call, an unknown verb, a failed check, a crash), 2 for bad
arguments (argparse prints its usage to stderr). Tracebacks and logs go to stderr.

While lab/state/STOP exists (see lab/machine/stop.py) every verb is refused before its code
runs, with {"ok": false, "outcome": "stopped", ...} and exit code 1. Only the verbs that
report, set or clear the kill switch are marked @stop.exempt and run anyway.

Adding a verb takes two steps and touches no other line of this file:

1. In lab/machine/<name>.py write `def main(argv: list[str]) -> dict | int | None`. argv is
   everything after the verb name; parse it with argparse. Return a dict and the dispatcher
   prints it as the one JSON line (its "ok" key, true when absent, picks the exit code), or
   print your own line and return the exit code. The first line of the docstring is the
   verb's line in --help. A verb with sub-verbs (`ledger add`, `rulings apply`) registers
   once, under its first word, and reads the rest of argv itself.
2. Add one line to VERBS below, keeping the names alphabetical:

       "name": "lab.machine.<name>:main",

Verbs are imported only when called, so one verb's missing dependency never breaks another.
"""

from __future__ import annotations

import importlib
import inspect
import json
import sys
import traceback
from collections.abc import Callable
from typing import Any

from lab.machine import stop

# One line per verb, alphabetical: "verb": "module:function".
VERBS: dict[str, str] = {
    "alerts": "lab.machine.alerts:main",
    "attribution-check": "lab.attribution_check:main",
    "bench": "lab.bench:main",
    "capacity": "lab.capacity:main",
    "chipsheet": "lab.kernels.chipsheet.ingest:main",
    "correctness": "lab.correctness:main",
    "credits": "lab.credits:main",
    "desk-score": "lab.desk_score:main",
    "digest": "lab.machine.digest:main",
    "draft": "lab.machine.draft:main",
    "eval-gate": "lab.eval_gate:main",
    "experiment": "lab.machine.experiments:main",
    "gpu-power": "lab.machine.gpu_power:main",
    "handoff": "lab.machine.handoff:main",
    "interference": "lab.interference:main",
    "label-agreement": "lab.label_agreement:main",
    "ledger": "lab.machine.ledger_cli:main",
    "link-speed": "lab.link_speed:main",
    "metrics": "lab.machine.metrics_report:main",
    "mirror": "lab.machine.mirror:main",
    "package": "lab.package:main",
    "placement": "lab.placement:main",
    "pool": "lab.machine.pool:main",
    "profile-decode": "lab.profile_decode:main",
    "publish-adapter": "lab.publish_adapter:main",
    "quark-compare": "lab.quark_compare:main",
    "ready": "lab.machine.ready:main",
    "referee": "lab.machine.referee:main",
    "research-check": "lab.machine.research_check:main",
    "resume": "lab.machine.stop:resume_main",
    "review": "lab.machine.review:main",
    "rulings": "lab.machine.rulings:main",
    "sampler-overhead": "lab.sampler_overhead:main",
    "sit": "lab.machine.sit:main",
    "slot": "lab.machine.slots:main",
    "spot-score": "lab.spot_score:main",
    "stop": "lab.machine.stop:stop_main",
    "sweep-request": "lab.sweep_request:main",
}

HELP_WORDS = ("-h", "--help", "help")


def load(target: str) -> Callable[[list[str]], Any]:
    """The function a registry line points at."""
    module, _, name = target.partition(":")
    return getattr(importlib.import_module(module), name)


def emit(result: dict[str, Any]) -> None:
    print(json.dumps(result))


def fail(outcome: str, error: str, **more: Any) -> int:
    emit({"ok": False, "outcome": outcome, "error": error, **more})
    return 1


def summary(target: str) -> str:
    """A verb's line in --help: the first line of its docstring."""
    try:
        doc = inspect.getdoc(load(target)) or ""
    except Exception as exc:
        return f"(cannot load: {type(exc).__name__}: {exc})"
    return doc.splitlines()[0] if doc else ""


def help_text() -> str:
    width = max(len(verb) for verb in VERBS)
    lines = [
        "usage: python -m lab <verb> [args]    (from the repo root; python -m lab <verb> -h for a verb's arguments)",
        "",
        "verbs:",
        *(f"  {verb:<{width}}  {summary(target)}" for verb, target in VERBS.items()),
        "",
        "One JSON line on stdout per call. Exit 0 when ok, 1 when not, 2 for bad arguments.",
        "While lab/state/STOP exists verbs are refused, except those that report, set or clear it.",
    ]
    return "\n".join(lines)


def finish(result: Any, verb: str) -> int:
    """Turn what a verb returned into an exit code, printing a returned dict as the one JSON line."""
    if result is None:
        return 0
    if isinstance(result, dict):
        emit(result)
        return 0 if result.get("ok", True) else 1
    if isinstance(result, int) and not isinstance(result, bool):
        return result
    return fail("error", f"verb returned {type(result).__name__}; expected a dict, an int or None", verb=verb)


def exit_code(exc: SystemExit) -> int:
    """The exit code a SystemExit asks for; a text message goes to stderr and means 1."""
    if exc.code is None:
        return 0
    if isinstance(exc.code, int):
        return exc.code
    print(exc.code, file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in HELP_WORDS:
        print(help_text())
        return 0
    verb, rest = argv[0], argv[1:]
    target = VERBS.get(verb)
    if target is None:
        return fail("unknown_verb", f"unknown verb {verb!r}", verbs=sorted(VERBS))
    try:
        fn = load(target)
    except Exception as exc:
        return fail("error", f"cannot load {target}: {type(exc).__name__}: {exc}", verb=verb)
    try:
        if not stop.is_exempt(fn):
            stop.guard()
        return finish(fn(rest), verb)
    except SystemExit as exc:
        return exit_code(exc)
    except KeyboardInterrupt:
        return fail("interrupted", "KeyboardInterrupt", verb=verb)
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        return fail("error", f"{type(exc).__name__}: {exc}", verb=verb)


if __name__ == "__main__":
    sys.exit(main())
