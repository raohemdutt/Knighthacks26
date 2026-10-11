"""handoff: the last call of a sitting. It leaves the owner a note and records that the sitting ended.

    python -m lab handoff --sitting <id> --note "<text>"

What it does, in this order:

  1. Appends the note to <state dir>/notes/<day>.md, one file per UTC day, one entry per call: the
     heading "## HH:MM:SSZ, sitting <id>", a blank line, the note, a blank line. The append is made
     under a lock (notes.lock in the state dir), flushed and fsynced, so sittings that hand off at the
     same moment each leave a whole entry; an entry cut short by a crash is closed before the next one.
  2. Appends one ledger row: verb handoff, outcome handoff, the actor from $LAB_ACTOR (the sitting id
     when it is not set), the tier from $LAB_TIER, the sitting id, the note's length (note_chars) and
     whether it reached the file (note_written). The row never holds the note's text.
  3. Answers with one JSON object: ok, outcome handoff, sitting, day, chars, note_written, ledger_seq
     and session_ends. It carries neither the text nor the file's path; a failure is named by its type
     and the system's message (note_error, ledger_error), never by a path. The exit code is 0 when the
     row was written and 1 when it was not.

Why the note is the owner's alone. It is the sitting's own account of itself, kept for the owner's
post-analysis: a wrong line in it would be inherited by every later sitting, so no sitting ever reads
it. The notes folder is on every tier's deny list (lab/machine/allowed.py), the digest is computed from
the ledger and never from the notes, and the text goes nowhere else: not into the ledger, not into the
answer. Both day and time are UTC, like the ledger's ts. A note that starts with a dash is written
--note=TEXT.

The session ends whatever happens to the note. A notes file that cannot be written (or a notes lock that
stays held for the whole wait) is logged and answered note_written false with note_error, still outcome
handoff, and the ledger row still records the handoff with note_written false. Only a ledger row that
cannot be written fails the call, because the ledger is the machine's one record that the sitting ended.

It runs while the STOP file exists (@stop.exempt): the tier prompts tell a sitting that finds the machine
stopped to leave the work safe and hand off, and it writes nothing but the note, the row and their locks.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from lab.machine import ledger, record, stop
from lab.machine.lock import FileLock, LockTimeout

__all__ = ["VERB", "OUTCOME", "NOTES_DIR", "NOTES_LOCK", "ENV_ACTOR", "ENV_TIER", "LOCK_TIMEOUT",
           "notes_path", "append_note", "handoff", "main"]

log = logging.getLogger("lab.handoff")

VERB = "handoff"
CLI_VERB = "handoff"   # registers python -m lab handoff
OUTCOME = "handoff"
NOTES_DIR = "notes"            # under the state dir; allowed.py keeps every sitting out of it
NOTES_LOCK = "notes.lock"      # in the state dir, beside ledger.lock
ENV_ACTOR = record.ENV_ACTOR   # record.py owns the names of the variables that identify a sitting
ENV_TIER = record.ENV_TIER
LOCK_TIMEOUT = ledger.LOCK_TIMEOUT   # seconds a note waits while another sitting is appending

StateDir = str | os.PathLike[str] | None


def notes_path(day: str, state_dir: StateDir = None) -> Path:
    """<state dir>/notes/<day>.md: the owner's file, on no sitting's read list."""
    return ledger.resolve_state_dir(state_dir) / NOTES_DIR / f"{day}.md"


def _sitting_id(text: object) -> str:
    """The sitting id on one line, runs of whitespace made single spaces; ValueError when nothing is left."""
    one_line = " ".join(str(text).split())
    if not one_line:
        raise ValueError("a sitting id must not be empty")
    return one_line


def _moment(now: dt.datetime | None) -> dt.datetime:
    """The time of the call as an aware UTC time to the second; ValueError for a naive time."""
    if now is None:
        return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    if now.utcoffset() is None:
        raise ValueError(f"now must carry a UTC offset, not the naive {now.isoformat()}")
    return now.astimezone(dt.timezone.utc).replace(microsecond=0)


def _entry(moment: dt.datetime, sitting: str, note: str) -> str:
    """One entry of the day file: heading, blank line, the note without its trailing whitespace, blank line."""
    body = note.rstrip()
    return f"## {moment:%H:%M:%S}Z, sitting {sitting}\n\n" + (f"{body}\n\n" if body else "")


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _why(exc: BaseException) -> str:
    """A failure as the answer names it: its type and the system's message, never a path."""
    if isinstance(exc, LockTimeout):
        return "LockTimeout: another writer held the lock for the whole wait"
    if isinstance(exc, OSError):
        return f"{type(exc).__name__}: {exc.strerror or 'not written'}"
    return f"{type(exc).__name__}: {exc}"


def append_note(day: str, text: str, *, state_dir: StateDir = None, timeout: float = LOCK_TIMEOUT) -> None:
    """Append text to the day's notes file under the notes lock, flushed and fsynced before returning. The
    file is written as UTF-8 bytes (a character UTF-8 cannot hold becomes ?) with no newline translation,
    and starts on a new line when the last entry was cut short. OSError, LockTimeout among it, when the
    file or the lock cannot be had; nothing is created then beyond the state dir."""
    state = ledger.resolve_state_dir(state_dir)
    data = text.encode("utf-8", errors="replace")
    with FileLock(state / NOTES_LOCK, timeout=timeout) as lock:
        lock.write_note(f"pid {os.getpid()} appending a handoff note")
        path = notes_path(day, state)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a+b") as fh:
            if fh.seek(0, os.SEEK_END) > 0:
                fh.seek(-1, os.SEEK_END)
                if fh.read(1) != b"\n":
                    data = b"\n" + data
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())


def handoff(sitting: str, note: str, *, state_dir: StateDir = None, actor: str | None = None,
            tier: str | None = None, now: dt.datetime | None = None, timeout: float = LOCK_TIMEOUT) -> dict[str, Any]:
    """End a sitting: append the note, then the ledger row, and answer what the verb prints. now is an
    aware time (default: the current UTC time); actor and tier default to $LAB_ACTOR (else the sitting
    id) and $LAB_TIER. ValueError, with nothing written, for an empty sitting id or a naive now."""
    sitting = _sitting_id(sitting)
    moment = _moment(now)
    note = str(note)
    day = moment.date().isoformat()
    answer: dict[str, Any] = {"ok": False, "outcome": OUTCOME, "sitting": sitting, "day": day, "chars": len(note)}
    try:
        append_note(day, _entry(moment, sitting, note), state_dir=state_dir, timeout=timeout)
    except Exception as exc:  # the note is the owner's extra: whatever goes wrong, the session still ends
        answer.update(note_written=False, note_error=_why(exc))
        log.error("handoff: the note of sitting %s (%d characters) was not written: %s", sitting, len(note),
                  answer["note_error"], exc_info=not isinstance(exc, OSError))
    else:
        answer["note_written"] = True
    row = {
        "ts": moment.isoformat().replace("+00:00", "Z"),  # the given now, so a replayed handoff lands in its own window
        "actor": _text(actor) or _text(os.environ.get(ENV_ACTOR)) or sitting,
        "tier": (_text(tier) or _text(os.environ.get(ENV_TIER))).lower() or None,
        "task": None, "verb": VERB, "outcome": OUTCOME, "metrics": {}, "refs": {},
        "sitting": sitting, "note_chars": len(note), "note_written": answer["note_written"],
    }
    try:
        answer["ledger_seq"] = ledger.append(row, state_dir=state_dir, timeout=timeout)
    except (OSError, ledger.RowError) as exc:
        answer.update(ledger_seq=None, ledger_error=_why(exc))
        log.error("handoff: the ledger row of sitting %s was not written: %s", sitting, answer["ledger_error"])
    else:
        answer["ok"] = True
    answer["session_ends"] = True
    return answer


def _sitting_argument(text: str) -> str:
    try:
        return _sitting_id(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


@stop.exempt
def main(argv: Sequence[str]) -> dict[str, Any]:
    """End the session: leave the owner a note (no sitting reads it) and record the handoff in the ledger."""
    parser = argparse.ArgumentParser(prog="python -m lab handoff", description=main.__doc__, allow_abbrev=False)
    parser.add_argument("--sitting", required=True, type=_sitting_argument, metavar="ID",
                        help="the id of this sitting (it is in your instruction)")
    parser.add_argument("--note", required=True, metavar="TEXT",
                        help="what you did, what is open, what you could not do; write --note=TEXT when it "
                             "starts with a dash; never read by any sitting")
    args = parser.parse_args(argv)
    return handoff(args.sitting, args.note)
