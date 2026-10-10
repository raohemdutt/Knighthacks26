"""The lab ledger: an append-only JSON-lines record of what the lab machine
did and measured. Verbs, sittings, GPU slots and experiments write it; the
digest, the metrics, the experiments and the sittings read it back. Nothing
ever rewrites or deletes a line.

Where it lives: <state dir>/ledger.jsonl, the state dir being LAB_STATE_DIR
when that is set and not empty, else lab/state beside this package. Every
function also takes state_dir= to name it directly. Processes working in
separate clones share one ledger only when they share one state dir.

A row is one JSON object on one line, its fields in this order:
  seq      1, 2, 3, ... in file order; assigned by append, never reused
  ts       ISO 8601 text with a UTC offset; append stamps the current UTC
           time when a row has none and stores an aware datetime as UTC text
  actor    who wrote the row: a sitting, a script or a person
  tier     the model tier of the sitting that wrote it (fable, opus,
           sonnet), or None for a row no sitting wrote
  task     the Task Board ID the row is about, or None
  verb     what was done
  outcome  how it ended: ok, fail, error, skipped, ...
  metrics  {name: finite number}, dotted lowercase names such as
           teacher.decode_tok_s; may be empty
  refs     {pr, commit, slot, run_id}: strings, integers or None; append
           fills in the ones a row leaves out as None and keeps any other
           ref a row brings
actor, verb and outcome are non-empty strings; tier and task are non-empty
strings or None. Optional fields, checked when present (None allowed):
node (where a measurement was taken: pod-a, pod-b, jetson), units
({metric: unit} for metrics of the same row), conditions (a dict of what
held while measuring), low_confidence (true or false), source (where the
numbers came from) and note (free text). Any other field is kept as written
as long as it is JSON and its name is not one of read()'s own filter names.
validate() checks a row without writing it; RowError lists every problem.

Writing: append() validates the row, then, holding the ledger lock
(ledger.lock in the state dir, a lock.FileLock), reads the seq of the last
row on disk, writes the row as one line with the next seq and lets go; it
returns the seq once the line is fsynced. The seq comes from the file
itself, never from a counter kept in a second file, so the two cannot
disagree after a crash. A crash mid-write can leave an unfinished last
line: the next append starts its row on a new line and readers skip the
fragment, while a complete row that lost only its newline still counts, for
readers and for the next seq. When the lock stays held past the timeout,
append raises LockTimeout and writes nothing.

Reading takes no lock: read() returns the rows in ledger order, filtered;
tail(n) the last n; latest_metric() the newest measurement of one metric in
the shape of a metrics-file row. A line that is not a row (not JSON, or no
integer seq) is skipped and logged."""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
import numbers
import os
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import IO, Any

from lab.machine.lock import FileLock, LockTimeout

__all__ = ["ENV_STATE_DIR", "DEFAULT_STATE_DIR", "LEDGER_FILE", "LOCK_FILE", "LOCK_TIMEOUT", "FIELDS",
           "REF_KEYS", "OPTIONAL_FIELDS", "FILTER_NAMES", "LockTimeout", "RowError", "resolve_state_dir",
           "ledger_path", "lock_path", "validate", "append", "read", "tail", "latest_metric"]

log = logging.getLogger("lab.ledger")

ENV_STATE_DIR = "LAB_STATE_DIR"
DEFAULT_STATE_DIR = Path(__file__).resolve().parents[1] / "state"
LEDGER_FILE = "ledger.jsonl"
LOCK_FILE = "ledger.lock"
LOCK_TIMEOUT = 10.0          # seconds an append waits while another writer holds the lock

FIELDS = ("seq", "ts", "actor", "tier", "task", "verb", "outcome", "metrics", "refs")
REF_KEYS = ("pr", "commit", "slot", "run_id")
OPTIONAL_FIELDS = ("node", "units", "conditions", "low_confidence", "source", "note")
# read() keywords that are not row fields, so no row may use them as field names
FILTER_NAMES = ("has_metric", "since", "before", "state_dir")

_TEXT_FIELDS = ("actor", "tier", "task", "verb", "outcome")
_MAY_BE_NONE = frozenset({"tier", "task"})
_BLOCK = 16 * 1024           # bytes read at a time when reading the ledger from its end

Row = dict[str, Any]


class RowError(ValueError):
    """A row the ledger refuses. problems lists every reason in field order,
    each one starting with the name of the field it is about."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        super().__init__("ledger row refused: " + "; ".join(self.problems))


# --- where ----------------------------------------------------------------------------

def resolve_state_dir(state_dir: str | os.PathLike[str] | None = None) -> Path:
    """state_dir when given, else LAB_STATE_DIR when set and not empty, else lab/state."""
    if state_dir is not None:
        return Path(state_dir)
    configured = os.environ.get(ENV_STATE_DIR)
    return Path(configured) if configured else DEFAULT_STATE_DIR


def ledger_path(state_dir: str | os.PathLike[str] | None = None) -> Path:
    """The ledger file of the state dir (resolve_state_dir)."""
    return resolve_state_dir(state_dir) / LEDGER_FILE


def lock_path(state_dir: str | os.PathLike[str] | None = None) -> Path:
    """The lock file that every append to that ledger takes."""
    return resolve_state_dir(state_dir) / LOCK_FILE


# --- times ----------------------------------------------------------------------------

def _utc_text(moment: dt.datetime) -> str:
    """An aware time as the ledger writes it: UTC, ending in Z."""
    text = moment.astimezone(dt.timezone.utc).isoformat()
    return text[:-len("+00:00")] + "Z" if text.endswith("+00:00") else text


def _now_text() -> str:
    return _utc_text(dt.datetime.now(dt.timezone.utc).replace(microsecond=0))


def _parse_ts(value: Any) -> dt.datetime | None:
    """value as an aware time when it is ISO 8601 text with a UTC offset (or
    Z), else None."""
    if not isinstance(value, str):
        return None
    try:
        moment = dt.datetime.fromisoformat(value)
    except ValueError:
        return None
    return moment if moment.utcoffset() is not None else None


def _ts_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value == "")


def _bound(value: Any, name: str) -> dt.datetime:
    """A since or before filter as an aware time; ValueError for anything else."""
    if isinstance(value, dt.datetime):
        if value.utcoffset() is None:
            raise ValueError(f"{name} must carry a UTC offset, not the naive {value.isoformat()}")
        return value
    moment = _parse_ts(value)
    if moment is None:
        raise ValueError(f"{name} must be an aware datetime or ISO 8601 text with a UTC offset, not {value!r}")
    return moment


# --- the row schema -------------------------------------------------------------------

def _is_name(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _number(value: Any) -> int | float | None:
    """value as a finite int or float; None when it is not one. A bool is not
    a number here."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    if isinstance(value, numbers.Integral):
        return int(value)
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_problem(name: str, value: Any) -> str | None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        return f"{name} is not JSON: {exc}"
    return None


def _check_ts(value: Any, out: Row, problems: list[str]) -> None:
    if _ts_missing(value):
        out["ts"] = _now_text()
    elif isinstance(value, dt.datetime):
        if value.utcoffset() is None:
            problems.append(f"ts must carry a UTC offset, not the naive {value.isoformat()}")
        else:
            out["ts"] = _utc_text(value)
    elif _parse_ts(value) is None:
        problems.append("ts must be ISO 8601 text with a UTC offset, such as 2026-10-10T10:20:00Z, "
                        f"or an aware datetime, not {value!r}")
    else:
        out["ts"] = value


def _check_text(row: Mapping[str, Any], key: str, out: Row, problems: list[str]) -> None:
    if key not in row:
        problems.append(f"{key} is missing")
        return
    value = row[key]
    if _is_name(value) or (value is None and key in _MAY_BE_NONE):
        out[key] = value
    else:
        allowed = "a non-empty string or None" if key in _MAY_BE_NONE else "a non-empty string"
        problems.append(f"{key} must be {allowed}, not {value!r}")


def _check_metrics(row: Mapping[str, Any], out: Row, problems: list[str]) -> None:
    if "metrics" not in row:
        problems.append("metrics is missing")
        return
    metrics = row["metrics"]
    if not isinstance(metrics, Mapping):
        problems.append(f"metrics must be a dict of metric name to number, not {type(metrics).__name__}")
        return
    clean: dict[str, int | float] = {}
    for name, value in metrics.items():
        number = _number(value)
        if not _is_name(name):
            problems.append(f"metrics name {name!r} must be a non-empty string")
        elif number is None:
            problems.append(f"metrics value for {name!r} must be a finite number, not {value!r}")
        else:
            clean[name] = number
    out["metrics"] = clean


def _check_refs(row: Mapping[str, Any], out: Row, problems: list[str]) -> None:
    if "refs" not in row:
        problems.append("refs is missing")
        return
    refs = row["refs"]
    if not isinstance(refs, Mapping):
        problems.append(f"refs must be a dict of ids ({', '.join(REF_KEYS)}), not {type(refs).__name__}")
        return
    clean: dict[str, Any] = dict.fromkeys(REF_KEYS)
    for name, value in refs.items():
        if not _is_name(name):
            problems.append(f"refs name {name!r} must be a non-empty string")
        elif value is not None and (isinstance(value, bool) or not isinstance(value, (str, int))):
            problems.append(f"refs value for {name!r} must be a string, an integer or None, not {value!r}")
        else:
            clean[name] = value
    out["refs"] = clean


def _check_optional(key: str, value: Any, row: Mapping[str, Any], out: Row, problems: list[str]) -> None:
    found: list[str] = []
    if value is None:
        pass
    elif key == "node":
        if not _is_name(value):
            found.append(f"node must be a non-empty string or None, not {value!r}")
    elif key == "units":
        if not isinstance(value, Mapping):
            found.append(f"units must be a dict of metric name to unit, not {type(value).__name__}")
        else:
            metrics = row.get("metrics")
            names = set(metrics) if isinstance(metrics, Mapping) else set()
            for name, unit in value.items():
                if name not in names:
                    found.append(f"units names {name!r}, which is not a metric of this row")
                elif not _is_name(unit):
                    found.append(f"units value for {name!r} must be a non-empty string, not {unit!r}")
    elif key == "conditions":
        if not isinstance(value, Mapping):
            found.append(f"conditions must be a dict, not {type(value).__name__}")
        else:
            problem = _json_problem("conditions", value)
            if problem:
                found.append(problem)
    elif key == "low_confidence":
        if not isinstance(value, bool):
            found.append(f"low_confidence must be true or false, not {value!r}")
    elif not isinstance(value, str):
        found.append(f"{key} must be a string or None, not {value!r}")
    if found:
        problems.extend(found)
    else:
        out[key] = dict(value) if isinstance(value, Mapping) else value


def _check_extra(key: Any, value: Any, out: Row, problems: list[str]) -> None:
    if not _is_name(key):
        problems.append(f"{key!r} is not a field name: field names are non-empty strings")
    elif key in FILTER_NAMES:
        problems.append(f"{key} is a read() filter name and cannot be a row field")
    else:
        problem = _json_problem(key, value)
        if problem:
            problems.append(problem)
        else:
            out[key] = value


def validate(row: Mapping[str, Any]) -> Row:
    """The row as append() would write it, less its seq: ts stamped when it
    is missing, an aware datetime ts turned into UTC text, refs completed,
    numbers made plain ints and floats, fields in their order. RowError
    lists every problem; TypeError when row is not a mapping at all. Writes
    nothing and leaves row as it was."""
    if not isinstance(row, Mapping):
        raise TypeError(f"a ledger row is a dict, not {type(row).__name__}")
    problems: list[str] = []
    out: Row = {}
    if "seq" in row:
        problems.append("seq is assigned by append; a row may not bring its own")
    _check_ts(row.get("ts"), out, problems)
    for key in _TEXT_FIELDS:
        _check_text(row, key, out, problems)
    _check_metrics(row, out, problems)
    _check_refs(row, out, problems)
    for key in OPTIONAL_FIELDS:
        if key in row:
            _check_optional(key, row[key], row, out, problems)
    for key, value in row.items():
        if key not in FIELDS and key not in OPTIONAL_FIELDS:
            _check_extra(key, value, out, problems)
    if problems:
        raise RowError(problems)
    return out


# --- the file -------------------------------------------------------------------------

def _parse_row(raw: bytes) -> Row | None:
    """The row one line holds; None when it holds none (not JSON, not an
    object, or no integer seq)."""
    try:
        row = json.loads(raw)
    except ValueError:
        return None
    if isinstance(row, dict) and type(row.get("seq")) is int:
        return row
    return None


def _read_at(fh: IO[bytes], pos: int, size: int) -> bytes:
    fh.seek(pos)
    chunks: list[bytes] = []
    while size > 0:
        chunk = fh.read(size)
        if not chunk:
            break
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def _lines_backward(fh: IO[bytes]) -> Iterator[bytes]:
    """The file's lines from the last to the first, without their newlines,
    read in blocks from the end. The first item is whatever follows the last
    newline: b"" when the file is empty or ends in a newline, else an
    unfinished last line."""
    pos = fh.seek(0, os.SEEK_END)
    carry = b""
    while pos > 0:
        step = min(_BLOCK, pos)
        pos -= step
        parts = (_read_at(fh, pos, step) + carry).split(b"\n")
        carry = parts[0]
        yield from reversed(parts[1:])
    yield carry


def _tail_state(fh: IO[bytes], path: Path) -> tuple[int, bool]:
    """(the seq of the last row on disk, 0 for none; whether the file ends
    without a newline, so the next row must start with one)."""
    lines = _lines_backward(fh)
    fragment = next(lines, b"")
    if fragment:
        row = _parse_row(fragment)
        if row is not None:
            log.warning("%s ends in row %d without its newline; the next row closes it", path, row["seq"])
            return row["seq"], True
        log.error("%s ends in an unfinished line (a write that never completed); the next row starts "
                  "on a new line and readers skip the fragment", path)
    for raw in lines:
        row = _parse_row(raw)
        if row is not None:
            return row["seq"], bool(fragment)
    return 0, bool(fragment)


def _write_all(fh: IO[bytes], data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = fh.write(view)
        if not written:
            raise OSError(f"no bytes could be written to {fh.name}")
        view = view[written:]


def append(row: Mapping[str, Any], *, state_dir: str | os.PathLike[str] | None = None,
           timeout: float = LOCK_TIMEOUT) -> int:
    """Write row as the next line of the ledger and return its seq once the
    line is on disk (flushed and fsynced). The row is validated first
    (RowError, and nothing is written); then the ledger lock is held only
    while the last seq is read and the line is written, so a row's seq is
    exactly one more than the row before it, whichever process wrote that.
    A row without ts is stamped with the current UTC time while the lock is
    held, so stamped rows keep file order. LockTimeout, writing nothing,
    when another writer holds the lock for timeout seconds; its message
    names that writer. row itself is not changed."""
    stored = validate(row)
    stamp_ts = _ts_missing(row.get("ts"))
    path = ledger_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+b", buffering=0) as fh:
        with FileLock(lock_path(state_dir), timeout=timeout) as lock:
            lock.write_note(f"pid {os.getpid()} appending since {_now_text()}")
            last_seq, unfinished = _tail_state(fh, path)
            seq = last_seq + 1
            if stamp_ts:
                stored["ts"] = _now_text()
            line = json.dumps({"seq": seq, **stored}, ensure_ascii=False, allow_nan=False).encode("utf-8")
            _write_all(fh, (b"\n" if unfinished else b"") + line + b"\n")
        os.fsync(fh.fileno())
    return seq


def read(*, state_dir: str | os.PathLike[str] | None = None, **filters: Any) -> list[Row]:
    """The rows in ledger order that pass every filter; [] when there is no
    ledger. A filter named after a field keeps the rows whose value
    passes it: a callable is called with the value, a dict must be contained
    in the value (read(refs={"run_id": "run-3"})), anything else must equal
    it; a field a row lacks reads as None. Three filters are not fields:
    has_metric=<name> keeps the rows whose metrics carry that name,
    since=<t> the rows at or after t and before=<t> the rows strictly
    before t, with t an aware datetime or ISO 8601 text with a UTC offset
    (ValueError otherwise), compared as instants. Takes no lock and creates
    nothing."""
    keep = _matcher(filters)
    path = ledger_path(state_dir)
    try:
        fh = open(path, "rb")
    except FileNotFoundError:
        return []
    rows: list[Row] = []
    with fh:
        for number, raw in enumerate(fh, 1):
            row = _parse_row(raw)
            if row is None:
                if not raw.endswith(b"\n"):
                    log.debug("%s line %d is unfinished (a write in progress or one cut short); skipped",
                              path, number)
                elif raw.strip():
                    log.warning("%s line %d is not a ledger row; skipped", path, number)
                continue
            if keep(row):
                rows.append(row)
    return rows


def _matcher(filters: Mapping[str, Any]) -> Callable[[Row], bool]:
    bounds = {name: _bound(filters[name], name) for name in ("since", "before") if name in filters}

    def keep(row: Row) -> bool:
        for key, want in filters.items():
            if key == "has_metric":
                metrics = row.get("metrics")
                if not (isinstance(metrics, dict) and want in metrics):
                    return False
            elif key in bounds:
                moment = _parse_ts(row.get("ts"))
                if moment is None:
                    return False
                too_early = key == "since" and moment < bounds[key]
                too_late = key == "before" and moment >= bounds[key]
                if too_early or too_late:
                    return False
            elif callable(want):
                if not want(row.get(key)):
                    return False
            elif isinstance(want, Mapping):
                have = row.get(key)
                if not isinstance(have, dict) or any(have.get(k) != v for k, v in want.items()):
                    return False
            elif row.get(key) != want:
                return False
        return True

    return keep


def tail(n: int = 10, *, state_dir: str | os.PathLike[str] | None = None) -> list[Row]:
    """The last n rows in ledger order, fewer when the ledger holds fewer;
    [] when n <= 0 or there is no ledger. Reads the file from its end, so
    the cost does not grow with the ledger. Takes no lock."""
    if n <= 0:
        return []
    try:
        fh = open(ledger_path(state_dir), "rb")
    except FileNotFoundError:
        return []
    found: list[Row] = []
    with fh:
        for raw in _lines_backward(fh):
            row = _parse_row(raw)
            if row is not None:
                found.append(row)
                if len(found) == n:
                    break
    found.reverse()
    return found


def latest_metric(name: str, node: str | None = None, *,
                  state_dir: str | os.PathLike[str] | None = None) -> Row | None:
    """The newest measurement of one metric, None when the ledger has none:
    of the rows whose metrics carry name (and whose node is node, when node
    is given), the one with the latest ts, compared as instants, the later
    seq winning a tie. Newest by ts rather than by position, so an older
    number appended late (a backfill) never passes for the current one.

    The answer has the shape of a metrics-file row plus the ledger seq:
    metric, value, unit (from the row's units, else None), node, conditions
    ({} when the row has none), source (the row's own source when it names
    one, else "ledger seq <n>"), ts, low_confidence and seq."""
    best: Row | None = None
    best_key: tuple[dt.datetime, int] | None = None
    for row in read(state_dir=state_dir, has_metric=name):
        if node is not None and row.get("node") != node:
            continue
        moment = _parse_ts(row.get("ts"))
        if moment is None or _number(row["metrics"][name]) is None:
            continue
        key = (moment, row["seq"])
        if best_key is None or key > best_key:
            best, best_key = row, key
    if best is None:
        return None
    seq = best["seq"]
    units = best.get("units")
    conditions = best.get("conditions")
    return {"metric": name, "value": best["metrics"][name],
            "unit": units.get(name) if isinstance(units, dict) else None,
            "node": best.get("node"),
            "conditions": dict(conditions) if isinstance(conditions, dict) else {},
            "source": best.get("source") or f"ledger seq {seq}",
            "ts": best["ts"], "low_confidence": best.get("low_confidence") is True, "seq": seq}
