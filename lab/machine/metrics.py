"""Metric baselines: measured numbers kept as append-only JSON lines, read back as a latest value and a history.

The file is lab/refine/metrics.jsonl (DEFAULT_PATH). One JSON object per line, with exactly these fields, in this
order (FIELDS):

  metric          dotted lowercase name that stays stable across runs, e.g. "teacher.decode_tok_s"
  value           finite number (a bool is refused)
  unit            what the value is counted in, e.g. "tok/s"; never blank
  node            lowercase label of the machine that produced it, e.g. "pod-a", "pod-b", "jetson"
  conditions      object holding what was held fixed when it was measured (model, engine, batch, context length,
                  prompt kind, ...). By convention a derived value says so here (derived: true plus its inputs) and a
                  low-confidence value says why under low_confidence_reason.
  source          where the number comes from; never blank
  ts              UTC time as YYYY-MM-DDTHH:MM:SS[.ffffff]Z: when the value was observed or, for a number taken from a
                  write-up that states no observation time, when that write-up recorded it
  low_confidence  true when the value is derived, or is not stated as such by a primary record (for example the
                  midpoint of a stated range, or a figure no primary record carries)

latest and history order rows by ts; rows with the same ts keep file order, so the later line is the newer one. Several
rows can share a metric and node when they were measured under different conditions: pass conditions to latest or
history to compare like with like.

The file is only ever appended to. add holds an exclusive lock on the file itself while it writes, so concurrent
writers, separate processes included, never interleave or lose a row, and load takes the same lock so it never reads a
half-written line. A corrupt line makes load refuse the whole file instead of dropping the row quietly.

load_numbered reads the file the same way, under the same lock, and returns each row with its line number: lines count
from 1 (FIRST_LINE_NUMBER), a blank line is skipped but still counted, and a CRLF ending is one line. These are the
numbers load names in its error text, and load is built on load_numbered, so a reader that cites a row by its line
takes the number from here instead of counting the file again.
"""
from __future__ import annotations

import datetime as dt
import errno
import json
import logging
import math
import os
import re
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

__all__ = [
    "DEFAULT_PATH", "FIELDS", "LOCK_TIMEOUT", "FileLock", "MetricsFormatError", "MetricsLocked",
    "FIRST_LINE_NUMBER", "add", "check_metric_name", "check_row", "history", "latest", "load", "load_numbered",
]

log = logging.getLogger("lab.metrics")

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "refine" / "metrics.jsonl"
FIELDS = ("metric", "value", "unit", "node", "conditions", "source", "ts", "low_confidence")
LOCK_TIMEOUT = 10.0  # seconds add and load wait for another process to let go of the file
FIRST_LINE_NUMBER = 1  # the number load_numbered gives the first line of the file, and load names in its errors

# The lock byte sits this far into the file on Windows, past any real content, so a held lock never overlaps the data a
# reader wants (Windows refuses reads that overlap a locked range) and the file never grows because of it.
_LOCK_OFFSET = 1 << 40
_LOCK_POLL = 0.02  # seconds between attempts while waiting for the lock
# What a non-blocking attempt raises when another handle holds the lock: EACCES from msvcrt.locking,
# EWOULDBLOCK/EAGAIN from flock. The deadlock codes are not defined on every platform.
_LOCK_HELD_ERRNOS = tuple(sorted({
    getattr(errno, name) for name in ("EACCES", "EAGAIN", "EDEADLK", "EDEADLOCK", "EWOULDBLOCK") if hasattr(errno, name)
}))
_BINARY = getattr(os, "O_BINARY", 0)

_METRIC_RE = re.compile(r"[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+")
_NODE_RE = re.compile(r"[a-z0-9][a-z0-9._-]*")
_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z")


class MetricsFormatError(ValueError):
    """A row, or a line of the file, does not match the schema."""


class MetricsLocked(OSError):
    """Another process held the file lock for longer than the timeout."""


# --- the lock ---------------------------------------------------------------------------------------------------------

def _try_lock(fd: int) -> bool:
    """One non-blocking attempt: True when the lock is ours, False when another handle holds it. Any other failure is
    raised."""
    try:
        if sys.platform == "win32":
            os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in _LOCK_HELD_ERRNOS:
            return False
        raise
    return True


def _unlock(fd: int) -> None:
    if sys.platform == "win32":
        os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)


class FileLock:
    """An operating-system lock held on the metrics file itself through an open handle: msvcrt.locking on Windows,
    fcntl.flock elsewhere. The system drops it when the handle closes or the process ends, so it never goes stale the way
    a pid file does, and no lock file is left beside the data. Two handles on one file exclude each other, inside one
    process as across processes. Acquiring creates the file when it does not exist.

    Used as a context manager it waits at most `timeout` seconds and raises MetricsLocked when the lock stays busy.
    Do not call add or load on the same path while holding it: they take the lock themselves and would wait on it."""

    def __init__(self, path: Path | str, *, timeout: float = LOCK_TIMEOUT) -> None:
        self.path = Path(path)
        self.timeout = timeout
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def acquire(self, timeout: float | None = None) -> bool:
        """True once the lock is ours (holding it already counts); False when another handle still holds it after
        `timeout` seconds (the lock's own timeout when None)."""
        if self._fd is not None:
            return True
        wait = self.timeout if timeout is None else timeout
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | _BINARY, 0o644)
        deadline = time.monotonic() + max(0.0, wait)
        try:
            while True:
                if _try_lock(fd):
                    self._fd = fd
                    return True
                if time.monotonic() >= deadline:
                    break
                time.sleep(_LOCK_POLL)
        except BaseException:
            os.close(fd)
            raise
        os.close(fd)
        return False

    def release(self) -> None:
        """Unlock and close; a no-op when the lock is not held."""
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        try:
            _unlock(fd)
        except OSError as exc:
            log.error("could not unlock %s (closing the handle releases it): %s", self.path, exc)
        finally:
            os.close(fd)

    def __enter__(self) -> FileLock:
        if not self.acquire():
            raise MetricsLocked(f"{self.path} stayed locked by another process for {self.timeout:g}s")
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()


# --- rows -------------------------------------------------------------------------------------------------------------

def _parse_ts(ts: str) -> dt.datetime:
    fmt = "%Y-%m-%dT%H:%M:%S.%fZ" if "." in ts else "%Y-%m-%dT%H:%M:%SZ"
    return dt.datetime.strptime(ts, fmt).replace(tzinfo=dt.timezone.utc)


def check_metric_name(name: Any) -> str:
    """`name` unchanged when it is a metric name this file accepts (dotted lowercase words such as
    'teacher.decode_tok_s'), else MetricsFormatError saying what the rule is. Writes nothing. The one rule add, load and
    check_row apply, exposed so a caller can refuse a name before it writes anything anywhere."""
    if not (isinstance(name, str) and _METRIC_RE.fullmatch(name)):
        raise MetricsFormatError(f"metric must be dotted lowercase words such as 'teacher.decode_tok_s', got {name!r}")
    return name


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _check_row(row: Any, *, where: str, stamp_missing_ts: bool) -> dict:
    """The row normalized to FIELDS order, or MetricsFormatError naming `where` and the first field that is wrong."""
    if not isinstance(row, Mapping):
        raise MetricsFormatError(f"{where}: a row must be a JSON object, not {type(row).__name__}")
    for key in row:
        if key not in FIELDS:
            raise MetricsFormatError(f"{where}: unknown field {key!r}; the fields are {', '.join(FIELDS)}")
    for key in FIELDS:
        if key not in row and not (key == "ts" and stamp_missing_ts):
            raise MetricsFormatError(f"{where}: missing field {key!r}")

    metric, value, unit, node = row["metric"], row["value"], row["unit"], row["node"]
    try:
        check_metric_name(metric)
    except MetricsFormatError as exc:
        raise MetricsFormatError(f"{where}: {exc}") from None
    if not _is_finite_number(value):
        raise MetricsFormatError(f"{where}: value must be a finite number (not a bool), got {value!r}")
    if not (isinstance(unit, str) and unit.strip()):
        raise MetricsFormatError(f"{where}: unit must be a non-blank string, got {unit!r}")
    if not (isinstance(node, str) and _NODE_RE.fullmatch(node)):
        raise MetricsFormatError(f"{where}: node must be a lowercase label such as 'pod-a', got {node!r}")

    conditions = row["conditions"]
    if not isinstance(conditions, Mapping):
        raise MetricsFormatError(f"{where}: conditions must be an object, got {type(conditions).__name__}")
    try:
        conditions = json.loads(json.dumps(conditions, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise MetricsFormatError(f"{where}: conditions must be JSON-serializable ({exc})") from exc

    source = row["source"]
    if not (isinstance(source, str) and source.strip()):
        raise MetricsFormatError(f"{where}: source must be a non-blank string, got {source!r}")

    ts = row.get("ts")
    if ts is None and stamp_missing_ts:
        ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        if not (isinstance(ts, str) and _TS_RE.fullmatch(ts)):
            raise ValueError
        _parse_ts(ts)
    except ValueError:
        raise MetricsFormatError(
            f"{where}: ts must be a UTC time such as '2026-10-10T10:20:00Z', got {ts!r}") from None

    low_confidence = row["low_confidence"]
    if type(low_confidence) is not bool:
        raise MetricsFormatError(f"{where}: low_confidence must be true or false, got {low_confidence!r}")

    return {
        "metric": metric, "value": value, "unit": unit, "node": node, "conditions": conditions,
        "source": source, "ts": ts, "low_confidence": low_confidence,
    }


def check_row(row: Mapping[str, Any]) -> dict:
    """The row as add would write it (normalized to FIELDS order, ts stamped with the current UTC time when missing), or
    MetricsFormatError naming the first field that is wrong. Writes nothing and touches no file, so a caller that keeps
    the same reading in another store can refuse it here before writing either."""
    return _check_row(row, where="row", stamp_missing_ts=True)


# --- reading and writing ----------------------------------------------------------------------------------------------

def _read_bytes(path: Path, timeout: float) -> bytes | None:
    """The whole file under the lock, None when it does not exist. A file that cannot be opened for writing (a read-only
    checkout) is read without the lock: nothing can be appending to it through this module."""
    if not path.exists():
        return None
    lock = FileLock(path, timeout=timeout)
    try:
        acquired = lock.acquire()
    except PermissionError:
        return path.read_bytes()
    if not acquired:
        raise MetricsLocked(f"{path} stayed locked by another process for {timeout:g}s")
    try:
        return path.read_bytes()
    finally:
        lock.release()


def load_numbered(
    path: Path | str = DEFAULT_PATH, *, lock_timeout: float = LOCK_TIMEOUT,
) -> list[tuple[int, dict]]:
    """Every row of the file in file order as (line number, row) pairs, each row validated and normalized to FIELDS
    order. The file is read once, under the same lock add takes. Lines count from FIRST_LINE_NUMBER; a blank line
    (whitespace only) is skipped but still counted, and a CRLF ending is one line, so the number is the one an editor
    shows and the one MetricsFormatError names. A missing file is an empty list and is not created. A line that is not
    valid JSON, or a row that breaks the schema, raises MetricsFormatError naming the line; the file is never read past
    a bad line. Raises MetricsLocked when a writer keeps the lock for longer than `lock_timeout`."""
    path = Path(path)
    data = _read_bytes(path, lock_timeout)
    if data is None:
        return []
    numbered: list[tuple[int, dict]] = []
    for number, raw in enumerate(data.split(b"\n"), start=FIRST_LINE_NUMBER):
        raw = raw.strip()
        if not raw:
            continue
        where = f"{path} line {number}"
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise MetricsFormatError(f"{where} is not valid JSON ({exc}); refusing to read a corrupt file") from exc
        numbered.append((number, _check_row(parsed, where=where, stamp_missing_ts=False)))
    return numbered


def load(path: Path | str = DEFAULT_PATH, *, lock_timeout: float = LOCK_TIMEOUT) -> list[dict]:
    """Every row of the file in file order, each validated and normalized to FIELDS order: load_numbered without the
    line numbers. A missing file is an empty list and is not created. Blank lines and CRLF endings are tolerated. A line
    that is not valid JSON, or a row that breaks the schema, raises MetricsFormatError naming the line; the file is
    never read past a bad line. Raises MetricsLocked when a writer keeps the lock for longer than `lock_timeout`."""
    return [row for _, row in load_numbered(path, lock_timeout=lock_timeout)]


def _require_complete_last_line(path: Path) -> None:
    """Refuse to append after a last line that has no newline: the new row would fuse with it into one corrupt line."""
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        if fh.tell() == 0:
            return
        fh.seek(-1, os.SEEK_END)
        if fh.read(1) != b"\n":
            raise MetricsFormatError(
                f"{path} does not end with a newline: its last line was cut short, so nothing was appended")


def add(row: Mapping[str, Any], *, path: Path | str = DEFAULT_PATH, lock_timeout: float = LOCK_TIMEOUT) -> dict:
    """Validate `row` and append it as one line, returning the row as written. Every field of FIELDS is required except
    ts, which defaults to the current UTC time. Unknown fields are refused. Nothing is written, and no file is created,
    for a row that fails validation (MetricsFormatError). The line is on disk (fsynced) before this returns. Waits at
    most `lock_timeout` seconds for the file lock and then raises MetricsLocked. Creates missing parent directories."""
    path = Path(path)
    checked = check_row(row)
    line = (json.dumps(checked, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    with FileLock(path, timeout=lock_timeout):
        _require_complete_last_line(path)
        with open(path, "ab") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
    return json.loads(line)


def _select(rows: list[dict], metric: str, node: str | None, conditions: Mapping[str, Any] | None) -> list[dict]:
    chosen = []
    for position, row in enumerate(rows):
        if row["metric"] != metric:
            continue
        if node is not None and row["node"] != node:
            continue
        if conditions and any(
            key not in row["conditions"] or row["conditions"][key] != wanted for key, wanted in conditions.items()
        ):
            continue
        chosen.append((_parse_ts(row["ts"]), position, row))
    chosen.sort(key=lambda item: item[:2])
    return [row for _, _, row in chosen]


def history(
    metric: str,
    node: str | None = None,
    *,
    conditions: Mapping[str, Any] | None = None,
    path: Path | str = DEFAULT_PATH,
) -> list[dict]:
    """Every row of `metric`, oldest first by ts (file order breaks ties). `node` keeps one node's rows; `conditions` keeps
    rows whose own conditions contain every key and value given. An unknown metric gives an empty list."""
    return _select(load(path), metric, node, conditions)


def latest(
    metric: str,
    node: str | None = None,
    *,
    conditions: Mapping[str, Any] | None = None,
    path: Path | str = DEFAULT_PATH,
) -> dict | None:
    """The newest row of `metric` (the full row, so its unit, node, conditions, source, ts and low_confidence come with the
    value), or None when no row matches. `node` and `conditions` narrow the rows exactly as they do for history."""
    rows = history(metric, node, conditions=conditions, path=path)
    return rows[-1] if rows else None
