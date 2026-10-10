"""Write a row of the lab ledger in one call: record(), and @recorded around a function or a lab verb.

    from lab.machine.record import record, recorded

    record("bench.run", "ok", metrics={"teacher.decode_tok_s": 7.7}, refs={"run_id": "run-0003"})

    @recorded("smoke-lora")
    def main(argv): ...

record() is lab.machine.ledger.append() with the writer filled in from the environment: the actor
from $LAB_ACTOR (UNKNOWN_ACTOR when it is not set), the tier from $LAB_TIER (lowercased, None when
not set) and the task from $LAB_TASK (None when not set). A launcher sets them once for a sitting,
so no script has to know who runs it; each can be passed instead (actor=, tier=, task=). Any other
keyword is a ledger field (node, units, conditions, low_confidence, source, note, ...) and goes into
the row as given. It returns the row's seq. A row the ledger refuses (RowError), a lock that stays
held (LockTimeout) and a disk that cannot be written all raise: a caller that asked for a row wants
to know it did not land.

@recorded(verb) writes one row when the call ends, however it ends, and never gets in the call's
way: the return value or the exception passes through unchanged, nothing is printed to stdout, and
a row that cannot be written is logged as a warning on stderr and dropped. Besides the writer, the
row holds the verb, the outcome, the metrics and refs (below), started and ended (UTC, to the
millisecond), duration_s (seconds of the call by a monotonic clock, ended = started + duration_s)
and, for a call that failed, a note with the reason on one line. The ledger stamps ts when it
writes the row, so ts is the moment the call was recorded and follows the file order.

How a call ends decides its outcome:

    returns a dict          its "outcome" when that is a non-empty string, else ok or fail by its "ok"
                            key (true when absent), the way the dispatcher reads a verb's answer. Its
                            "metrics" and "refs", when it has them, go into the row, and its "error"
                            text is the note of a call that is not ok. A "metrics" or "refs" the ledger
                            refuses is left out and the note says so: the row still records the call.
    returns an int          0 is ok and any other number is fail, the way the dispatcher reads a verb's
                            exit code; a function that returns some other number should return a dict
                            instead. A bool is not an exit code and counts as any other value
    returns anything else   ok (None, text, a list, ...): the function finished
    raises SystemExit       None or 0 is ok, 2 is usage (argparse's code for bad arguments), any other
                            code is fail; stop.Stopped is stopped, its reason the note
    raises KeyboardInterrupt  interrupted
    raises anything else    error, the exception's type and message the note

The decorator takes a plain function: a generator or coroutine function is refused at decoration,
because its call returns before any of the work is done. The verb name is the only required
argument; task=, state_dir= and timeout= are passed to record().
"""
from __future__ import annotations

import datetime as dt
import functools
import inspect
import logging
import os
import time
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple, TypeVar

from lab.machine import ledger, stop

__all__ = ["ENV_ACTOR", "ENV_TIER", "ENV_TASK", "UNKNOWN_ACTOR", "NOTE_CHARS", "current_actor",
           "current_tier", "current_task", "record", "recorded"]

log = logging.getLogger("lab.record")

ENV_ACTOR = "LAB_ACTOR"
ENV_TIER = "LAB_TIER"
ENV_TASK = "LAB_TASK"
UNKNOWN_ACTOR = "unknown"     # the actor of a row written where $LAB_ACTOR is not set
NOTE_CHARS = 500              # the longest note a recorded call writes
EXIT_USAGE = 2                # the exit code argparse uses for bad arguments

F = TypeVar("F", bound=Callable[..., Any])


# --- who is writing ---------------------------------------------------------------------

def _env(name: str, environ: Mapping[str, str] | None) -> str | None:
    """The variable's value without surrounding space; None when it is unset or blank."""
    value = (os.environ if environ is None else environ).get(name)
    return value.strip() or None if value else None


def current_actor(environ: Mapping[str, str] | None = None) -> str:
    """$LAB_ACTOR, or UNKNOWN_ACTOR when it is unset or blank."""
    return _env(ENV_ACTOR, environ) or UNKNOWN_ACTOR


def current_tier(environ: Mapping[str, str] | None = None) -> str | None:
    """$LAB_TIER lowercased (fable, opus, sonnet); None when it is unset or blank."""
    tier = _env(ENV_TIER, environ)
    return tier.lower() if tier else None


def current_task(environ: Mapping[str, str] | None = None) -> str | None:
    """$LAB_TASK, the Task Board ID the sitting works on; None when it is unset or blank."""
    return _env(ENV_TASK, environ)


# --- the row ----------------------------------------------------------------------------

def record(verb: str, outcome: str, metrics: Mapping[str, Any] | None = None,
           refs: Mapping[str, Any] | None = None, *, task: str | None = None,
           actor: str | None = None, tier: str | None = None,
           state_dir: str | os.PathLike[str] | None = None,
           timeout: float = ledger.LOCK_TIMEOUT, **fields: Any) -> int:
    """Append one row to the ledger and return its seq.

    The actor, tier and task come from the environment unless passed; metrics and refs default to
    empty; every other keyword is a ledger field of the row. ts is stamped by the ledger."""
    row = {"actor": actor if actor is not None else current_actor(),
           "tier": tier if tier is not None else current_tier(),
           "task": task if task is not None else current_task(),
           "verb": verb, "outcome": outcome,
           "metrics": {} if metrics is None else metrics,
           "refs": {} if refs is None else refs,
           **fields}
    return ledger.append(row, state_dir=state_dir, timeout=timeout)


# --- what a finished call tells the ledger ------------------------------------------------

class _Found(NamedTuple):
    outcome: str
    metrics: Any = None
    refs: Any = None
    note: str | None = None


def _clip(text: Any) -> str | None:
    """text on one line, cut at NOTE_CHARS; None when there is nothing to say."""
    flat = " ".join(str(text).split())
    if not flat:
        return None
    return flat if len(flat) <= NOTE_CHARS else flat[:NOTE_CHARS - 3] + "..."


def _stamp(moment: dt.datetime) -> str:
    """An aware time as UTC text to the millisecond, ending in Z."""
    text = moment.astimezone(dt.timezone.utc).isoformat(timespec="milliseconds")
    return text[:-len("+00:00")] + "Z" if text.endswith("+00:00") else text


def _from_result(result: Any) -> _Found:
    if isinstance(result, Mapping):
        ok = bool(result.get("ok", True))
        named = result.get("outcome")
        outcome = named.strip() if isinstance(named, str) and named.strip() else ("ok" if ok else "fail")
        return _Found(outcome, result.get("metrics"), result.get("refs"),
                      None if ok else _clip(result.get("error", "")))
    if isinstance(result, int) and not isinstance(result, bool):
        return _Found("ok") if result == 0 else _Found("fail", note=f"exit code {result}")
    return _Found("ok")


def _from_exception(exc: BaseException) -> _Found:
    if isinstance(exc, stop.Stopped):
        return _Found("stopped", note=_clip(exc.reason))
    if isinstance(exc, KeyboardInterrupt):
        return _Found("interrupted")
    if isinstance(exc, SystemExit):
        code = exc.code
        if code is None or code == 0:
            return _Found("ok")
        if code == EXIT_USAGE:
            return _Found("usage", note="exit code 2")
        return _Found("fail", note=_clip(f"exit code {code}" if isinstance(code, int) else code))
    return _Found("error", note=_clip(f"{type(exc).__name__}: {exc}"))


# --- the decorator ----------------------------------------------------------------------

def _write(verb: str, found: _Found, started: dt.datetime, seconds: float, task: str | None,
           state_dir: str | os.PathLike[str] | None, timeout: float) -> None:
    """Append the row of a finished call. Nothing escapes: a row that cannot be written is logged."""
    ended = started + dt.timedelta(seconds=seconds)
    fields: dict[str, Any] = {"started": _stamp(started), "ended": _stamp(ended),
                              "duration_s": round(seconds, 3)}
    if found.note:
        fields["note"] = found.note
    try:
        try:
            record(verb, found.outcome, found.metrics, found.refs, task=task, state_dir=state_dir,
                   timeout=timeout, **fields)
        except ledger.RowError as err:
            problems = "; ".join(err.problems)
            log.warning("the ledger refused the row of %s (%s); writing it without metrics and refs",
                        verb, problems)
            why = f"ledger refused: {problems}"
            fields["note"] = _clip(f"{found.note}; {why}" if found.note else why)
            record(verb, found.outcome, task=task, state_dir=state_dir, timeout=timeout, **fields)
    except Exception as exc:
        log.warning("no ledger row written for %s: %s: %s", verb, type(exc).__name__, exc)


def recorded(verb: str, *, task: str | None = None, state_dir: str | os.PathLike[str] | None = None,
             timeout: float = ledger.LOCK_TIMEOUT) -> Callable[[F], F]:
    """Decorator: write one ledger row when the function's call ends (see the module docstring)."""
    if callable(verb):
        raise TypeError("write @recorded('verb-name'), with the name of the verb in parentheses")
    if not isinstance(verb, str) or not verb.strip():
        raise ValueError(f"the verb of @recorded must be a non-empty string, not {verb!r}")

    def decorate(fn: F) -> F:
        if inspect.isgeneratorfunction(fn) or inspect.iscoroutinefunction(fn):
            raise TypeError(f"@recorded takes a plain function; {fn.__name__} is a generator or coroutine")

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            started = dt.datetime.now(dt.timezone.utc)
            clock = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:
                _write(verb, _from_exception(exc), started, time.perf_counter() - clock, task, state_dir, timeout)
                raise
            _write(verb, _from_result(result), started, time.perf_counter() - clock, task, state_dir, timeout)
            return result

        return wrapper

    return decorate
