"""python -m lab <verb> [args]: the lab machine's command line.

Run it from the repo root. `python -m lab` (or `--help`) lists the verbs.

Every call answers with exactly one JSON object on stdout and an exit code: 0 when it worked,
1 when it did not (a refused call, an unknown verb, a failed check, a crash), 2 for bad
arguments (argparse prints its usage to stderr). Tracebacks and logs go to stderr.

While lab/state/STOP exists (see lab/machine/stop.py) every verb is refused before its code
runs, with {"ok": false, "outcome": "stopped", ...} and exit code 1. Only the verbs that
report, set or clear the kill switch are marked @stop.exempt and run anyway.

Adding a verb touches no line of this file:

1. In lab/machine/<name>.py write `def main(argv: list[str]) -> dict | int | None`. argv is
   everything after the verb name; parse it with argparse. Return a dict and the dispatcher
   prints it as the one JSON line (its "ok" key, true when absent, picks the exit code), or
   print your own line and return the exit code. The first line of the docstring is the
   verb's line in --help. A verb with sub-verbs (`ledger add`, `rulings apply`) registers
   once, under its first word, and reads the rest of argv itself.
2. Declare the verb at the top level of that module, as a literal, next to its constants:

       CLI_VERB = "name"

   A module whose entry point is not main, or that holds several verbs, declares
   `CLI_VERBS = {"name": "function", ...}` instead (lab/machine/stop.py registers stop and
   resume this way).
3. Document it in lab/docs/<name>.md (tests/test_lab_verbs.py checks the file exists) and add
   its line to the verbs list in lab/README.md.

The registry below is built when this module loads, by reading the declarations out of every
.py file under lab/ (not by importing them): the help listing is sorted by verb name, and a name
declared by two modules raises RegistryError instead of letting the later file win. Verbs are
imported only when called, so one verb's missing dependency never breaks another.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import re
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lab.machine import stop

LAB_DIR = Path(__file__).resolve().parent
MARKER = re.compile(r"^(?P<marker>CLI_VERBS?)\s*(?::[^=\n]*)?=\s*", re.MULTILINE)
MAX_LITERAL_LINES = 40


class RegistryError(Exception):
    """A verb declaration the dispatcher refuses: a repeated name, a malformed literal, a missing entry point."""


def _literal(text: str, start: int, where: str) -> Any:
    """The literal that starts at text[start:], which may run over several lines (a dict)."""
    lines = text[start:].splitlines()
    for count in range(1, min(len(lines), MAX_LITERAL_LINES) + 1):
        try:
            return ast.literal_eval("\n".join(lines[:count]))
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
    raise RegistryError(f"{where}: CLI_VERB must be a string literal and CLI_VERBS a dict literal of strings")


def declared(text: str, where: str) -> dict[str, str]:
    """The verbs one module's source declares, "verb": "function", without importing the module.

    CLI_VERB = "name" registers main; CLI_VERBS = {"name": "function", ...} registers several entry
    points. The declaration is a top-level literal and every function it names is a top-level def.
    """
    verbs: dict[str, str] = {}
    for match in MARKER.finditer(text):
        value = _literal(text, match.end(), where)
        if match["marker"] == "CLI_VERB":
            if not isinstance(value, str):
                raise RegistryError(f"{where}: CLI_VERB must be a string, not {type(value).__name__}")
            pairs = {value: "main"}
        else:
            if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
                raise RegistryError(f"{where}: CLI_VERBS must be a dict of verb name to function name")
            pairs = dict(value)
        for verb, function in pairs.items():
            if not verb or verb != verb.strip() or any(ch.isspace() for ch in verb):
                raise RegistryError(f"{where}: {verb!r} is not a verb name (one word, no spaces)")
            if verb in verbs:
                raise RegistryError(f"{where}: declares the verb {verb!r} twice")
            if not re.search(rf"^(?:async\s+)?def\s+{re.escape(function)}\s*\(", text, re.MULTILINE):
                raise RegistryError(f"{where}: declares the verb {verb!r} but defines no top-level {function}()")
            verbs[verb] = function
    return verbs


def discover(root: Path = LAB_DIR, package: str = "lab") -> dict[str, str]:
    """Every verb the modules under root register, sorted by name: "verb": "module:function".

    One scan of the sources (not an import, so a verb's missing dependency still breaks only that verb):
    each .py file under root, at any depth, that declares CLI_VERB or CLI_VERBS. A name declared by two
    modules raises RegistryError, so the later file can never silently win.
    """
    found: dict[str, tuple[str, str]] = {}
    for path in sorted(root.rglob("*.py")):
        if path.name == "__main__.py" or "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if "CLI_VERB" not in text:
            continue
        parts = path.relative_to(root).with_suffix("").parts
        module = ".".join((package, *(parts[:-1] if parts[-1] == "__init__" else parts)))
        for verb, function in declared(text, module).items():
            if verb in found:
                raise RegistryError(f"the verb {verb!r} is declared by both {found[verb][0]} and {module}")
            found[verb] = (module, function)
    return {verb: f"{module}:{function}" for verb, (module, function) in sorted(found.items())}


# "verb": "module:function", alphabetical: read from the modules' CLI_VERB declarations when this module loads.
VERBS: dict[str, str] = discover()

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
