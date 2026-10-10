"""The lab's kill switch: one file halts every verb.

The STOP file is lab/state/STOP (or $LAB_STATE_DIR/STOP). While it exists no verb runs:
the dispatcher (lab/__main__.py) refuses each call before its code starts, and a script
that runs outside the dispatcher calls guard() first (or wears @guarded). The file's text
is the reason, shown to whoever is refused.

    python -m lab stop [reason ...]    create it (a second stop adds its reason)
    python -m lab resume               remove it (deleting the file by hand works too)

Every check fails closed: a STOP file that cannot be checked or read still stops the run.
Only the verbs that report, set or clear the kill switch carry @exempt and run anyway.

    from lab.machine.stop import exempt, guard, guarded, stopped

    stopped()      True while the STOP file exists
    stop_reason()  its text on one line, "" when there is no STOP file
    guard()        first line of a verb: while stopped, prints the refusal and exits
                   "stopped" with code 1
    @guarded       the same, as a decorator
    @exempt        marks a verb the dispatcher lets through while stopped
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

STATE_DIR_ENV = "LAB_STATE_DIR"
DEFAULT_STATE_DIR = Path(__file__).resolve().parents[1] / "state"
STOP_NAME = "STOP"
EXIT_STOPPED = 1
NO_REASON = "no reason given"
MAX_REASON_CHARS = 400
EXEMPT_ATTR = "runs_when_stopped"


def state_dir() -> Path:
    """lab/state, or $LAB_STATE_DIR when set; read on every call so a test or a rehearsal can point it elsewhere."""
    override = os.environ.get(STATE_DIR_ENV)
    return Path(override) if override else DEFAULT_STATE_DIR


def stop_file() -> Path:
    return state_dir() / STOP_NAME


def _check() -> tuple[bool, OSError | None]:
    """(stopped, error). A STOP file that cannot be checked counts as present, with the error that hid it."""
    try:
        os.lstat(stop_file())
    except (FileNotFoundError, NotADirectoryError):
        return False, None
    except OSError as exc:
        return True, exc
    return True, None


def stopped() -> bool:
    """True while the STOP file exists. A file that cannot be checked counts as present: the kill switch fails closed."""
    return _check()[0]


def _one_line(text: str) -> str:
    lines = (" ".join(line.split()) for line in text.splitlines())
    return " | ".join(line for line in lines if line)


def stop_reason() -> str:
    """The STOP file's text on one line (lines joined by " | ", cut at MAX_REASON_CHARS); "" when there is none."""
    present, error = _check()
    if not present:
        return ""
    if error is not None:
        return f"(STOP file could not be checked: {type(error).__name__})"
    try:
        text = stop_file().read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""
    except OSError as exc:
        return f"(STOP file unreadable: {type(exc).__name__})"
    return (_one_line(text) or NO_REASON)[:MAX_REASON_CHARS]


class Stopped(SystemExit):
    """What guard() raises: exit code 1; str() is "stopped" and .reason is the STOP file's text."""

    def __init__(self, reason: str) -> None:
        super().__init__(EXIT_STOPPED)
        self.reason = reason

    def __str__(self) -> str:
        return "stopped"


def guard() -> None:
    """First line of every verb: while the STOP file exists, print the refusal and exit with code 1.

    The refusal is one JSON line on stdout and one human line on stderr.
    """
    if not stopped():
        return
    reason = stop_reason() or NO_REASON
    print(json.dumps({"ok": False, "outcome": "stopped", "reason": reason, "file": str(stop_file())}))
    print(f"stopped: {reason}", file=sys.stderr)
    raise Stopped(reason)


def guarded(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator form of guard(): the function does not run while the STOP file exists."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        guard()
        return fn(*args, **kwargs)

    return wrapper


def exempt(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Mark a verb that must run while the STOP file exists: the ones that report, set or clear it."""
    setattr(fn, EXEMPT_ATTR, True)
    return fn


def is_exempt(fn: Callable[..., Any]) -> bool:
    return bool(getattr(fn, EXEMPT_ATTR, False))


def _parser(verb: str, summary: str) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(prog=f"python -m lab {verb}", description=summary)


@exempt
def stop_main(argv: Sequence[str]) -> dict[str, Any]:
    """Create the STOP file: every verb but ready, resume and stop is refused."""
    parser = _parser("stop", stop_main.__doc__)
    parser.add_argument("reason", nargs="*", help="why; stored in the STOP file and shown to whoever is refused")
    args = parser.parse_args(argv)
    reason = " ".join(" ".join(args.reason).split()) or NO_REASON
    path = stop_file()
    already_stopped = stopped()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(reason + "\n")
    except OSError as exc:
        return {"ok": False, "outcome": "error", "error": f"cannot write {path}: {type(exc).__name__}: {exc}"}
    return {
        "ok": True,
        "outcome": "stop_set",
        "already_stopped": already_stopped,
        "reason": stop_reason(),
        "file": str(path),
    }


@exempt
def resume_main(argv: Sequence[str]) -> dict[str, Any]:
    """Remove the STOP file so verbs run again."""
    _parser("resume", resume_main.__doc__).parse_args(argv)
    path = stop_file()
    if not stopped():
        return {"ok": True, "outcome": "not_stopped", "file": str(path)}
    reason = stop_reason()
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        return {
            "ok": False,
            "outcome": "error",
            "error": f"cannot remove {path}: {type(exc).__name__}: {exc}",
            "reason": reason,
            "file": str(path),
        }
    return {"ok": True, "outcome": "resumed", "reason": reason, "file": str(path)}
