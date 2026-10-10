"""digest: what a sitting of the lab machine starts from, computed by code from the ledger,
a board snapshot and the rulings files, never from a note.

    python -m lab digest                      the digest as text, at most MAX_LINES lines
    python -m lab digest --json               the same digest as one JSON object
    python -m lab digest --hours 6            a wider window (default: the 2 hours before now)
    python -m lab digest --since <time>       the window starts at <time> (ISO 8601, UTC offset)
    python -m lab digest --board <file>       read this board snapshot instead of the stored one
    python -m lab digest --save-board <file>  check <file>, store it as the board snapshot, digest
    python -m lab digest --schema             the board snapshot schema, with an example

The launcher puts the text in every sitting's prompt, and a sitting reruns it for a fresh
picture. Ported from the phone project's agent digest (social_outreach/tools/agent/digest.py):
everything is computed by code from the machine's own records, every list has a cap and a
count of what it left out, and a last pass trims the longest list until the answer fits. What
changed: the lab runs many sittings at once, so there is no single "where the last session
stopped"; the answer is about the whole lab over a time window.

Never a note. Nothing under lab/state/notes/ (the handoff notes, written for Ilan) is opened,
and no free text a sitting typed into a ledger row (its note, its source, the text inside its
conditions) is copied: a row reaches the digest only through its structured fields (time,
actor, tier, task, verb, outcome, numbers, refs). A lesson reaches the sittings only when Ilan
turns it into a rule, a ruling or a task.

The answer, in the order a sitting reads it:
- header: now, the window, the ledger's size, and the board snapshot's age; past
  BOARD_STALE_MINUTES it is marked STALE (fetch the board fresh before acting on its parts).
- what ran: the ledger rows inside the window: totals by outcome, the tasks they touched
  (newest activity first), the verbs of the rows without a task, and every failure (an outcome
  that starts with one of FAILURE_WORDS), newest first. Read-only verbs (READ_ONLY_VERBS, this
  one among them) are counted, never itemized.
- metrics moved: per series (metric, node and the row's conditions, so a number is only
  compared with one measured the same way), the newest reading in the window against the
  newest reading before it, or against the first reading in the window when nothing came
  before; then the series first read in the window, and the ones that did not change. Seqs
  are printed as ledger:<seq>, the citation form the refinement passes use.
- GPU queue (from the snapshot): running rows with their elapsed minutes against their cap,
  queued rows shortest first per GPU with how long each has waited, and the complaints on
  open rows. The cap and "complaint due" are the queue verbs' own rules, read from
  lab/machine/slots.py and never restated here: the cap is slots.cap_minutes (Minutes,
  DEFAULT_MINUTES when a row names none, plus EXTENSION_MINUTES once Extended), and a queued
  row without a complaint is marked "complaint due" when slots.complaint_due says it may
  complain (it has waited more than COMPLAIN_AFTER_FACTOR times its Minutes, or a Running
  row on its GPU is past its cap).
- rulings waiting on Ilan: every unanswered ruling in lab/refine/<track>/OWNER_RULINGS.md
  (read with lab/machine/refine_formats.py), with the default the passes apply meanwhile.
- Ilan's lane (from the snapshot): his tasks counted by status and tier.
- next unblocked, per tier: his tasks that are Not started, unlocked, his alone, and whose
  every Blocked by task is Done; P0 first, then the earlier wave, stretch last, then the ID.
  Tasks held back because a blocker is missing from the snapshot, or because a teammate is
  also assigned, are named so the gap stays visible.
- problems: what could not be read (a missing or invalid snapshot, a broken rulings file).

Failure. A snapshot or rulings file that cannot be read leaves its part empty and names the
error, so a sitting always gets a digest; a ledger that cannot be opened fails the verb. The
digest writes nothing, except --save-board, which stores the checked snapshot at
<state dir>/board.json (a sitting may not write under lab/state itself).

Output. This verb answers in text because its reader is a sitting's prompt; --json gives the
one JSON object the dispatcher prints for the other verbs, holding exactly what the text
shows. A failure is one JSON line with ok false and exit code 1, as for every verb. The text
is ASCII. compute() builds the digest, render() turns it into text, text() does both; the
launcher may call text() or run the verb.

The board snapshot schema is SCHEMA below (python -m lab digest --schema prints it).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lab.machine import ledger as ledger_mod
from lab.machine import refine_formats, refine_track, slots
from lab.machine.slots import EXTENSION_MINUTES  # noqa: F401  (pool.py reads it as digest.EXTENSION_MINUTES)

__all__ = ["MAX_LINES", "DEFAULT_WINDOW_HOURS", "BOARD_FILE", "BOARD_STALE_MINUTES", "SCHEMA",
           "SCHEMA_VERSION", "BoardError", "Board", "load_board", "parse_board", "save_board",
           "board_path", "queue_slot", "compute", "fit", "render", "text", "main"]

REPO_ROOT = Path(__file__).resolve().parents[2]
RULINGS_FILE = "OWNER_RULINGS.md"
BOARD_FILE = "board.json"
SCHEMA_VERSION = 1

#: The text is at most this many lines ("under 150").
MAX_LINES = 149
DEFAULT_WINDOW_HOURS = 2.0
#: A snapshot older than this is marked STALE.
BOARD_STALE_MINUTES = 60

#: Ilan's Notion user id (AGENTS.md section 1): his lane is the tasks assigned to him.
ILAN_USER_ID = "fbcc0baf-9942-4661-9414-319688df6910"
ILAN_NAMES = ("ilan", "ilan danial")

#: Verbs that only read; counted in the window, never itemized.
READ_ONLY_VERBS = frozenset({
    "digest", "ledger tail", "ledger grep", "rulings", "rulings list", "slot status",
    "metrics report", "metrics compare", "experiment list", "research-check",
})
#: An outcome is a failure when it starts with one of these.
FAILURE_WORDS = ("fail", "error", "crash", "timeout", "timed_out", "refused", "stopped",
                 "interrupted", "invalid", "abort", "denied", "lock_timeout")

STATUS_ORDER = ("Not started", "In progress", "Blocked", "Done")
TIER_ORDER = ("Fable", "Opus", "Sonnet")
NO_TIER = "none"
OPEN_QUEUE_STATUSES = (slots.RUNNING, slots.QUEUED)

# caps: entries listed before "+N more"
MAX_TASKS_RAN = 12
MAX_VERBS_RAN = 8
MAX_FAILURES = 8
MAX_MOVED = 12
MAX_NEW = 6
MAX_UNCHANGED = 8
MAX_RUNNING = 4
MAX_QUEUED = 8
MAX_COMPLAINTS = 5
MAX_RULINGS = 8
MAX_NEXT_PER_TIER = 5
MAX_HELD = 6
MAX_PROBLEMS = 6

# clip widths (characters)
ID_CHARS = 12
NAME_CHARS = 24
TITLE_CHARS = 72
QUESTION_CHARS = 90
DEFAULT_CHARS = 80
COMPLAINT_CHARS = 100
PROBLEM_CHARS = 150
LINE_CHARS = 200

_HEX32 = re.compile(r"[0-9a-f]{32}")
_TASK_ID = re.compile(r"^(?P<letters>[A-Za-z]+)(?P<number>[0-9]+)(?P<rest>.*)$")
_PRIORITY = re.compile(r"P(?P<rank>[0-9])", re.IGNORECASE)
_ASCII = str.maketrans({
    "—": "-", "–": "-", "−": "-", "‘": "'", "’": "'", "“": '"',
    "”": '"', "…": "...", "↔": "<->", "→": "->", "←": "<-",
    "·": "-", "×": "x", " ": " ", "•": "-",
})

SCHEMA = """\
Board snapshot schema, version 1. One JSON object that a sitting writes with its Notion tools
and stores with: python -m lab digest --save-board <file>

  schema     1
  taken_at   when the Notion reads were made: ISO 8601 with a UTC offset (required)
  taken_by   who wrote it, such as the sitting id (optional)
  tasks      Task Board rows (data source collection://68af27b3-01c7-487d-807e-71af7d3fc536):
             every task assigned to Ilan, any status, plus every task named in their Blocked
             by. A Done task may have moved to the archive and no longer come back from a board
             query: fetch such a blocker by its URL. A query in rows mode returns at most 100
             rows, so query per Status when it says has_more.
  gpu_queue  GPU Queue rows (collection://dc46dcb7-79d4-4727-bc04-40426b5d9114) with Status
             Queued or Running; finished rows may be left in and are ignored.

A row may be pasted exactly as the Notion query tool returns it in rows mode, keyed by the
property names. Each field may also be written under a plain lowercase name:

  tasks      id ("userDefined:ID"), title ("Task"), status ("Status"), tier ("Tier"),
             priority ("Priority"), wave ("Wave"), locked ("Locked"), stretch ("Stretch"),
             assignees ("Assignee"), blocked_by ("Blocked by"), context ("Context"), url
  gpu_queue  request ("Request"), task ("Task ID"), gpu ("GPU"), minutes ("Minutes"),
             status ("Status"), agent ("Agent"), requested ("Requested" or
             "date:Requested:start"), started and ended (the same forms), extended
             ("Extended"), complaint ("Complaint")

Values: a checkbox is true or false, or "__YES__" or "__NO__". Assignee and Blocked by are a
list, or a JSON-array string, of user ids (user://<uuid>) and of page URLs; a Blocked by entry
may also be a task ID such as O15. A time is ISO 8601 with a UTC offset (one without is read
as UTC and reported); a date may also be {"start": <time>}. A task without an ID is skipped
and counted.

Example:
{
  "schema": 1,
  "taken_at": "2026-10-10T15:45:00Z",
  "taken_by": "sonnet-3",
  "tasks": [
    {"userDefined:ID": "O17", "Task": "O17: Digest computed from ledger and board",
     "Status": "In progress", "Tier": "Opus", "Priority": "P0 Demo-critical", "Wave": 1,
     "Locked": "__YES__", "Stretch": "__NO__", "Context": "Orchestration",
     "Assignee": "[\\"user://fbcc0baf-9942-4661-9414-319688df6910\\"]",
     "Blocked by": "[\\"https://app.notion.com/p/3f596ef5fec581499019c49da7a9d322\\"]",
     "url": "https://app.notion.com/p/3f596ef5fec58107939aef4e2bd0faf4"}
  ],
  "gpu_queue": [
    {"Request": "R11: decode sweep batch 1 to 16", "Task ID": "R11", "GPU": "B exclusive",
     "Minutes": 15, "Status": "Running", "Agent": "Ilan",
     "date:Requested:start": "2026-10-10T15:30:00Z",
     "date:Started:start": "2026-10-10T15:40:00Z", "Extended": "__NO__", "Complaint": ""}
  ]
}
"""


class BoardError(ValueError):
    """A board snapshot that cannot be used at all; problems lists why."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        super().__init__("board snapshot refused: " + "; ".join(self.problems))


# --- text helpers ---------------------------------------------------------------------------

def ascii_text(text: str) -> str:
    """text with common typographic characters made ASCII and anything else escaped."""
    return text.translate(_ASCII).encode("ascii", "backslashreplace").decode("ascii")


def clip(value: Any, limit: int) -> str:
    """value on one ASCII line, cut to limit characters with "..." marking a cut."""
    one = ascii_text(" ".join(str(value if value is not None else "").split()))
    if len(one) <= limit:
        return one
    return one[:max(0, limit - 3)].rstrip() + "..."


def number_text(value: float | int) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if math.isfinite(value) and value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.6g}"


def delta_text(value: float) -> str:
    text = f"{value:+.4g}"
    return text.replace("e+0", "e+").replace("e-0", "e-")


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" + ("" if count == 1 else "s")


# --- times -------------------------------------------------------------------------------------

def parse_time(value: Any) -> tuple[dt.datetime | None, bool]:
    """(an aware UTC time, whether it had no UTC offset) from ISO 8601 text, a {"start": ...}
    date, or an aware datetime; (None, False) when there is no readable time. Text without an
    offset, or a bare date, is read as UTC."""
    if isinstance(value, Mapping):
        value = value.get("start")
    if isinstance(value, dt.datetime):
        moment = value
    elif isinstance(value, str) and value.strip():
        try:
            moment = dt.datetime.fromisoformat(value.strip())
        except ValueError:
            return None, False
    else:
        return None, False
    if moment.utcoffset() is None:
        return moment.replace(tzinfo=dt.timezone.utc), True
    return moment.astimezone(dt.timezone.utc), False


def iso(moment: dt.datetime) -> str:
    text = moment.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat()
    return text[:-len("+00:00")] + "Z" if text.endswith("+00:00") else text


def clock(moment_text: str | None, now: dt.datetime) -> str:
    """HH:MMZ for a time on now's UTC day, MM-DD HH:MMZ otherwise; "?" when unreadable."""
    moment, _ = parse_time(moment_text)
    if moment is None:
        return "?"
    if moment.date() == now.astimezone(dt.timezone.utc).date():
        return moment.strftime("%H:%MZ")
    return moment.strftime("%m-%d %H:%MZ")


def minutes_between(start: dt.datetime, end: dt.datetime) -> int:
    return int((end - start).total_seconds() // 60)


def span_text(hours: float) -> str:
    if hours == int(hours):
        return f"{int(hours)}h"
    return f"{round(hours * 60)}min"


# --- the board snapshot -------------------------------------------------------------------------

@dataclass
class Task:
    id: str
    title: str
    status: str
    tier: str
    priority: str
    wave: float | None
    locked: bool
    stretch: bool
    assignees: list[str]
    blocked_by: list[str]
    url: str
    context: str = ""

    @property
    def page(self) -> str | None:
        return page_key(self.url)


@dataclass
class QueueRow:
    request: str
    task: str
    gpu: str
    minutes: float | None
    status: str
    requested: dt.datetime | None
    started: dt.datetime | None
    extended: bool
    complaint: str


@dataclass
class Board:
    path: str
    taken_at: dt.datetime | None
    taken_by: str
    tasks: list[Task] = field(default_factory=list)
    queue: list[QueueRow] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


_TASK_KEYS = {
    "id": ("id", "userDefined:ID", "ID"),
    "title": ("title", "Task"),
    "status": ("status", "Status"),
    "tier": ("tier", "Tier"),
    "priority": ("priority", "Priority"),
    "wave": ("wave", "Wave"),
    "locked": ("locked", "Locked"),
    "stretch": ("stretch", "Stretch"),
    "assignees": ("assignees", "Assignee"),
    "blocked_by": ("blocked_by", "Blocked by"),
    "context": ("context", "Context"),
    "url": ("url", "URL"),
}
_QUEUE_KEYS = {
    "request": ("request", "Request"),
    "task": ("task", "Task ID"),
    "gpu": ("gpu", "GPU"),
    "minutes": ("minutes", "Minutes"),
    "status": ("status", "Status"),
    "requested": ("requested", "Requested", "date:Requested:start"),
    "started": ("started", "Started", "date:Started:start"),
    "extended": ("extended", "Extended"),
    "complaint": ("complaint", "Complaint"),
}


def _pick(row: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def _text(value: Any) -> str:
    return "" if value is None else " ".join(str(value).split())


def _flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in ("__yes__", "true", "yes", "1", "checked")


def _list(value: Any) -> list[str]:
    """A list property: a list, a JSON-array string, or one value."""
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            try:
                value = json.loads(stripped)
            except ValueError:
                return [stripped]
        else:
            return [stripped]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def page_key(text: str | None) -> str | None:
    """The 32-hex Notion page id inside a URL or uuid, None when there is none."""
    compact = str(text or "").lower().replace("-", "")
    found = _HEX32.findall(compact)
    return found[-1] if found else None


def _tier(value: Any) -> str:
    name = _text(value)
    for tier in TIER_ORDER:
        if name.lower() == tier.lower():
            return tier
    return name or NO_TIER


def _status(value: Any) -> str:
    name = _text(value)
    for status in STATUS_ORDER:
        if name.lower() == status.lower():
            return status
    return name or "(no status)"


def _task(row: Mapping[str, Any]) -> Task | None:
    values = {key: _pick(row, names) for key, names in _TASK_KEYS.items()}
    task_id = _text(values["id"])
    if not task_id:
        return None
    title = _text(values["title"])
    prefix = f"{task_id}:"
    if title.startswith(prefix):
        title = title[len(prefix):].strip()
    return Task(id=task_id, title=title, status=_status(values["status"]), tier=_tier(values["tier"]),
                priority=_text(values["priority"]), wave=_number(values["wave"]),
                locked=_flag(values["locked"]), stretch=_flag(values["stretch"]),
                assignees=_list(values["assignees"]), blocked_by=_list(values["blocked_by"]),
                url=_text(values["url"]), context=_text(values["context"]))


def _queue_row(row: Mapping[str, Any], naive: list[str]) -> QueueRow:
    values = {key: _pick(row, names) for key, names in _QUEUE_KEYS.items()}
    times: dict[str, dt.datetime | None] = {}
    for key in ("requested", "started"):
        moment, was_naive = parse_time(values[key])
        times[key] = moment
        if was_naive:
            naive.append(key)
    return QueueRow(request=_text(values["request"]), task=_text(values["task"]), gpu=_text(values["gpu"]) or "?",
                    minutes=_number(values["minutes"]), status=_text(values["status"]),
                    requested=times["requested"], started=times["started"],
                    extended=_flag(values["extended"]), complaint=_text(values["complaint"]))


def parse_board(data: Any, path: str = "") -> Board:
    """A Board from a decoded snapshot. BoardError when it cannot be used at all (not an
    object, no tasks list, no readable taken_at); smaller problems are listed on the Board."""
    if not isinstance(data, Mapping):
        raise BoardError([f"the snapshot must be a JSON object, not {type(data).__name__}"])
    fatal: list[str] = []
    problems: list[str] = []
    schema = data.get("schema")
    if schema is not None and schema != SCHEMA_VERSION:
        problems.append(f"schema is {schema!r}; this digest reads version {SCHEMA_VERSION}")
    taken_at, naive_taken = parse_time(data.get("taken_at"))
    if taken_at is None:
        fatal.append("taken_at is missing or not ISO 8601 (when were the Notion reads made?)")
    elif naive_taken:
        problems.append("taken_at has no UTC offset; read as UTC")
    rows = data.get("tasks")
    if not isinstance(rows, list):
        fatal.append("tasks must be a list of Task Board rows")
        rows = []
    queue_rows = data.get("gpu_queue", [])
    if queue_rows is None:
        queue_rows = []
    if not isinstance(queue_rows, list):
        fatal.append("gpu_queue must be a list of GPU Queue rows")
        queue_rows = []
    if fatal:
        raise BoardError(fatal + problems)
    tasks: list[Task] = []
    seen: dict[str, int] = {}
    unnamed = not_rows = 0
    for row in rows:
        if not isinstance(row, Mapping):
            not_rows += 1
            continue
        task = _task(row)
        if task is None:
            unnamed += 1
            continue
        if task.id in seen:
            tasks[seen[task.id]] = task
            problems.append(f"task {clip(task.id, ID_CHARS)} appears twice; the later row is used")
            continue
        seen[task.id] = len(tasks)
        tasks.append(task)
    if unnamed:
        problems.append(f"{_plural(unnamed, 'task row')} without an ID skipped")
    if not_rows:
        problems.append(f"{_plural(not_rows, 'task entry')} that is not an object skipped")
    queue: list[QueueRow] = []
    naive: list[str] = []
    bad_queue = 0
    for row in queue_rows:
        if not isinstance(row, Mapping):
            bad_queue += 1
            continue
        queue.append(_queue_row(row, naive))
    if bad_queue:
        problems.append(f"{_plural(bad_queue, 'queue entry')} that is not an object skipped")
    if naive:
        problems.append(f"{_plural(len(naive), 'queue time')} without a UTC offset read as UTC")
    return Board(path=path, taken_at=taken_at, taken_by=_text(data.get("taken_by")), tasks=tasks,
                 queue=queue, problems=problems)


def board_path(state_dir: str | os.PathLike[str] | None = None) -> Path:
    """Where the stored snapshot lives: <state dir>/board.json."""
    return ledger_mod.resolve_state_dir(state_dir) / BOARD_FILE


def _read_board_file(path: Path) -> Board:
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        raise BoardError([f"no board snapshot (python -m lab digest --schema says how to write one): "
                          f"{_shown(path)}"]) from None
    except (OSError, UnicodeDecodeError) as exc:
        raise BoardError([f"board snapshot unreadable ({type(exc).__name__}): {_shown(path)}"]) from None
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise BoardError([f"board snapshot not JSON ({exc}): {_shown(path)}"]) from None
    return parse_board(data, _shown(path))


def load_board(path: str | os.PathLike[str] | None = None, *,
               state_dir: str | os.PathLike[str] | None = None) -> tuple[Board | None, list[str]]:
    """(the snapshot at path, else the stored one; its problems). (None, problems) when it is
    missing or cannot be used."""
    target = Path(path) if path is not None else board_path(state_dir)
    try:
        board = _read_board_file(target)
    except BoardError as exc:
        return None, exc.problems
    return board, list(board.problems)


def save_board(source: str | os.PathLike[str], *, state_dir: str | os.PathLike[str] | None = None) -> Path:
    """Check the snapshot at source and store it, unchanged, as <state dir>/board.json in one
    atomic replace. BoardError, storing nothing, when it cannot be used."""
    src = Path(source)
    _read_board_file(src)
    data = src.read_bytes()
    target = board_path(state_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(prefix=".board-", suffix=".json", dir=target.parent)
    try:
        with os.fdopen(handle, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp, target)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise
    return target


def _shown(path: Path) -> str:
    """path relative to the repo when inside it, with forward slashes."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


# --- what ran -------------------------------------------------------------------------------------

def is_failure(outcome: Any) -> bool:
    word = str(outcome or "").strip().lower()
    return any(word.startswith(prefix) for prefix in FAILURE_WORDS)


def _capped(items: list[Any], cap: int) -> dict[str, Any]:
    return {"items": items[:cap], "more": max(0, len(items) - cap)}


def _ts(row: Mapping[str, Any]) -> dt.datetime | None:
    moment, _ = parse_time(row.get("ts"))
    return moment


def _sort_key(row: Mapping[str, Any]) -> tuple[dt.datetime, int]:
    return (_ts(row) or dt.datetime.min.replace(tzinfo=dt.timezone.utc), int(row.get("seq") or 0))


def what_ran(rows: list[dict], now: dt.datetime) -> dict[str, Any]:
    """The rows of the window, summarized; rows must be the window's, oldest first."""
    listed = [r for r in rows if str(r.get("verb") or "") not in READ_ONLY_VERBS]
    reads = Counter(clip(r.get("verb"), NAME_CHARS) for r in rows if str(r.get("verb") or "") in READ_ONLY_VERBS)
    outcomes = Counter(clip(r.get("outcome"), NAME_CHARS) for r in listed)
    actors = {clip(r.get("actor"), NAME_CHARS) for r in listed}
    by_task: dict[str, dict[str, Any]] = {}
    by_verb: dict[str, Counter] = {}
    for row in listed:
        task = row.get("task")
        verb = clip(row.get("verb"), NAME_CHARS)
        if task:
            key = clip(task, ID_CHARS)
            entry = by_task.setdefault(key, {"task": key, "rows": 0, "failures": 0, "verbs": [], "pr": None})
            entry["rows"] += 1
            entry["failures"] += int(is_failure(row.get("outcome")))
            if verb not in entry["verbs"]:
                entry["verbs"].append(verb)
            refs = row.get("refs") if isinstance(row.get("refs"), Mapping) else {}
            if refs.get("pr") is not None:
                entry["pr"] = clip(refs.get("pr"), ID_CHARS)
            entry.update({"last_ts": iso(_ts(row) or now), "last_verb": verb,
                          "last_outcome": clip(row.get("outcome"), NAME_CHARS),
                          "last_actor": clip(row.get("actor"), NAME_CHARS), "last_seq": row.get("seq")})
        else:
            by_verb.setdefault(verb, Counter())[clip(row.get("outcome"), NAME_CHARS)] += 1
    tasks = sorted(by_task.values(), key=lambda e: (e["last_ts"], e["last_seq"] or 0), reverse=True)
    for entry in tasks:
        entry["verbs"].sort()
        del entry["last_seq"]
    verbs = [{"verb": verb, "rows": sum(counts.values()), "outcomes": dict(sorted(counts.items()))}
             for verb, counts in by_verb.items()]
    verbs.sort(key=lambda e: (-e["rows"], e["verb"]))
    failures = []
    for row in reversed(listed):
        if not is_failure(row.get("outcome")):
            continue
        refs = row.get("refs") if isinstance(row.get("refs"), Mapping) else {}
        failures.append({"ts": iso(_ts(row) or now), "seq": row.get("seq"),
                         "task": clip(row.get("task"), ID_CHARS) or None, "verb": clip(row.get("verb"), NAME_CHARS),
                         "outcome": clip(row.get("outcome"), NAME_CHARS), "actor": clip(row.get("actor"), NAME_CHARS),
                         "run_id": clip(refs.get("run_id"), NAME_CHARS) or None})
    return {"rows": len(listed), "actors": len(actors), "tasks": len(by_task),
            "outcomes": dict(sorted(outcomes.items(), key=lambda kv: (-kv[1], kv[0]))),
            "reads": dict(sorted(reads.items())),
            "by_task": _capped(tasks, MAX_TASKS_RAN), "by_verb": _capped(verbs, MAX_VERBS_RAN),
            "failures": _capped(failures, MAX_FAILURES)}


# --- metrics --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Reading:
    when: dt.datetime
    seq: int
    value: float
    unit: str | None
    low_confidence: bool


def _conditions_key(conditions: Any) -> str:
    if not isinstance(conditions, Mapping) or not conditions:
        return "{}"
    return json.dumps(conditions, sort_keys=True, default=str)


def _short_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return clip(json.dumps(value, sort_keys=True), 16)
    if isinstance(value, float):
        return number_text(value)
    return clip(value, 16)


def metrics_moved(rows: list[dict], since: dt.datetime, now: dt.datetime) -> dict[str, Any]:
    """Every series read in the window: how it moved against the newest earlier reading.
    rows are every ledger row stamped at or before now."""
    series: dict[tuple[str, str, str], list[Reading]] = {}
    conditions_of: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        when = _ts(row)
        metrics = row.get("metrics")
        if when is None or not isinstance(metrics, Mapping):
            continue
        units = row.get("units") if isinstance(row.get("units"), Mapping) else {}
        conditions = row.get("conditions") if isinstance(row.get("conditions"), Mapping) else {}
        for name, value in metrics.items():
            number = _number(value)
            if number is None:
                continue
            key = (str(name), str(row.get("node") or ""), _conditions_key(conditions))
            series.setdefault(key, []).append(Reading(when, int(row.get("seq") or 0), number,
                                                      units.get(name), row.get("low_confidence") is True))
            conditions_of[key] = dict(conditions)
    moved: list[dict[str, Any]] = []
    new: list[dict[str, Any]] = []
    unchanged: list[dict[str, Any]] = []
    shown = [key for key, readings in series.items() if any(r.when >= since for r in readings)]
    for key in shown:
        readings = sorted(series[key], key=lambda r: (r.when, r.seq))
        inside = [r for r in readings if r.when >= since]
        before = [r for r in readings if r.when < since]
        after = inside[-1]
        base, within = (before[-1], False) if before else ((inside[0], True) if len(inside) > 1 else (None, False))
        entry: dict[str, Any] = {"metric": key[0], "node": key[1] or None, "label": "", "value": after.value,
                                 "unit": after.unit, "seq": after.seq,
                                 "low_confidence": after.low_confidence or bool(base and base.low_confidence)}
        if base is None:
            new.append(entry)
            continue
        entry.update({"before": base.value, "before_seq": base.seq, "within_window": within})
        if base.value == after.value:
            unchanged.append(entry)
            continue
        change = after.value - base.value
        entry["delta"] = change
        entry["pct"] = (change / abs(base.value) * 100.0) if base.value else None
        moved.append(entry)
    # Name a series by the conditions that set it apart from the other series of the same
    # metric and node, so two lines of one metric are never confused.
    for entries in (moved, new, unchanged):
        for entry in entries:
            entry["label"] = _series_label(entry, series, conditions_of)
    moved.sort(key=lambda e: (-(abs(e["pct"]) if e["pct"] is not None else math.inf), e["metric"], e["node"] or ""))
    new.sort(key=lambda e: (e["metric"], e["node"] or "", e["label"]))
    unchanged.sort(key=lambda e: (e["metric"], e["node"] or "", e["label"]))
    return {"moved": _capped(moved, MAX_MOVED), "new": _capped(new, MAX_NEW),
            "unchanged": _capped(unchanged, MAX_UNCHANGED)}


def _series_label(entry: Mapping[str, Any], series: Mapping[tuple[str, str, str], list[Reading]],
                  conditions_of: Mapping[tuple[str, str, str], dict]) -> str:
    siblings = [key for key in series if key[0] == entry["metric"] and key[1] == (entry["node"] or "")]
    if len(siblings) < 2:
        return ""
    mine = next(key for key in siblings
                if any(r.seq == entry["seq"] for r in series[key]))
    keys = sorted({k for key in siblings for k in conditions_of.get(key, {})})
    differing = [k for k in keys
                 if len({json.dumps(conditions_of.get(key, {}).get(k), sort_keys=True, default=str)
                         for key in siblings}) > 1]
    own = conditions_of.get(mine, {})
    # The conditions this series has come first; one it lacks shows as key=-.
    differing.sort(key=lambda k: (k not in own, k))
    parts = [f"{clip(k, 16)}={_short_value(own[k]) if k in own else '-'}" for k in differing[:3]]
    return " ".join(parts)


# --- the GPU queue -----------------------------------------------------------------------------

def queue_slot(row: QueueRow, index: int = 0) -> slots.Slot:
    """A snapshot row as the queue verbs see a row (slots.Slot), so the digest judges it by their rules
    (slots.cap_minutes, slots.is_overrun, slots.complaint_due). The snapshot holds no page id, Agent,
    Ended, Progress note or Result: the id is the row's place in the snapshot and the rest stay empty,
    and none of those rules reads them."""
    return slots.Slot(id=f"snapshot-{index}", url="", request=row.request, task_id=row.task, gpu=row.gpu,
                      minutes=row.minutes, status=row.status, agent="", requested=row.requested, created=None,
                      started=row.started, ended=None, extended=row.extended, progress_note="",
                      complaint=row.complaint, result="")


def gpu_queue(board: Board, now: dt.datetime) -> dict[str, Any]:
    running: list[dict[str, Any]] = []
    queued: list[dict[str, Any]] = []
    complaints: list[dict[str, Any]] = []
    open_rows = [row for row in board.queue if row.status in OPEN_QUEUE_STATUSES]
    seen = [queue_slot(row, index) for index, row in enumerate(open_rows)]
    for row, slot in zip(open_rows, seen):
        base = {"request": clip(row.request, TITLE_CHARS), "task": clip(row.task, ID_CHARS) or None,
                "gpu": clip(row.gpu, NAME_CHARS), "minutes": row.minutes}
        if row.status == slots.RUNNING:
            cap = slots.cap_minutes(slot)
            elapsed = minutes_between(row.started, now) if row.started else None
            running.append({**base, "started": iso(row.started) if row.started else None, "elapsed_min": elapsed,
                            "cap_min": cap, "extended": row.extended,
                            "over_min": (elapsed - cap) if elapsed is not None and elapsed > cap else 0})
        else:
            waited = minutes_between(row.requested, now) if row.requested else None
            may_complain, _, _ = slots.complaint_due(slot, seen, now)
            queued.append({**base, "requested": iso(row.requested) if row.requested else None,
                           "waited_min": waited, "complaint_due": may_complain and not row.complaint,
                           "has_complaint": bool(row.complaint)})
        if row.complaint:
            since = row.started if row.status == slots.RUNNING else row.requested
            complaints.append({"request": clip(row.request, TITLE_CHARS), "status": row.status,
                               "minutes_since": minutes_between(since, now) if since else None,
                               "text": clip(row.complaint, COMPLAINT_CHARS)})
    far = dt.datetime.max.replace(tzinfo=dt.timezone.utc)
    queued.sort(key=lambda e: (e["gpu"], e["minutes"] if e["minutes"] is not None else math.inf,
                               parse_time(e["requested"])[0] or far))
    position: Counter = Counter()
    for entry in queued:
        position[entry["gpu"]] += 1
        entry["position"] = position[entry["gpu"]]
    running.sort(key=lambda e: (e["gpu"], e["started"] or ""))
    gpus = sorted({e["gpu"] for e in running} | {e["gpu"] for e in queued})
    summary = [{"gpu": gpu, "running": sum(e["gpu"] == gpu for e in running),
                "queued": sum(e["gpu"] == gpu for e in queued)} for gpu in gpus]
    return {"summary": summary, "running": _capped(running, MAX_RUNNING), "queued": _capped(queued, MAX_QUEUED),
            "complaints": _capped(complaints, MAX_COMPLAINTS)}


# --- rulings --------------------------------------------------------------------------------------

def open_rulings(refine_root: Path) -> tuple[dict[str, Any], list[str]]:
    """The unanswered rulings of every track under refine_root, and the files that could not
    be read."""
    found: list[dict[str, Any]] = []
    answered = 0
    problems: list[str] = []
    for path in sorted(refine_root.glob(f"*/{RULINGS_FILE}")):
        track = path.parent.name
        try:
            rulings = refine_formats.parse_rulings(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, refine_formats.FormatError) as exc:
            problems.append(f"{track} rulings unreadable: {exc}")
            continue
        for ruling in rulings:
            if ruling.answered:
                answered += 1
                continue
            found.append({"track": track, "number": ruling.number, "origin": clip(ruling.origin, NAME_CHARS),
                          "question": clip(ruling.question, QUESTION_CHARS), "now": clip(ruling.now, DEFAULT_CHARS)})
    return {**_capped(found, MAX_RULINGS), "answered": answered}, problems


# --- the lane and what is next -----------------------------------------------------------------

def is_ilan(assignee: str) -> bool:
    text = assignee.strip().lower()
    if text.removeprefix("user://") == ILAN_USER_ID:
        return True
    return text in ILAN_NAMES


def _priority_rank(priority: str) -> int:
    found = _PRIORITY.search(priority or "")
    return int(found["rank"]) if found else 9


def _id_key(task_id: str) -> tuple[str, int, str]:
    found = _TASK_ID.match(task_id)
    if not found:
        return (task_id.upper(), -1, "")
    return (found["letters"].upper(), int(found["number"]), found["rest"])


def _next_key(task: Task) -> tuple:
    return (_priority_rank(task.priority), task.wave if task.wave is not None else math.inf, task.stretch,
            _id_key(task.id))


def lane(board: Board) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """(Ilan's tasks by status and tier, the next unblocked per tier, the tasks held back)."""
    mine = [task for task in board.tasks if any(is_ilan(a) for a in task.assignees)]
    statuses = [s for s in STATUS_ORDER] + sorted({t.status for t in mine} - set(STATUS_ORDER))
    tiers = [*TIER_ORDER, NO_TIER] + sorted({t.tier for t in mine} - set(TIER_ORDER) - {NO_TIER})
    counts = {status: {tier: sum(1 for t in mine if t.status == status and t.tier == tier) for tier in tiers}
              for status in statuses}
    table = {"tasks": len(mine), "locked": sum(t.locked for t in mine), "tiers": tiers, "counts": counts}

    by_id = {task.id.lower(): task for task in board.tasks}
    by_page = {task.page: task for task in board.tasks if task.page}
    offered: dict[str, list[Task]] = {tier: [] for tier in tiers}
    shared: list[Task] = []
    missing: list[tuple[Task, list[str]]] = []
    for task in mine:
        if task.status != "Not started" or task.locked:
            continue
        if not all(is_ilan(a) for a in task.assignees):
            shared.append(task)
            continue
        unknown: list[str] = []
        waiting = False
        for ref in task.blocked_by:
            blocker = by_page.get(page_key(ref)) if page_key(ref) else by_id.get(ref.strip().lower())
            if blocker is None:
                unknown.append(ref)
            elif blocker.status != "Done":
                waiting = True
        if waiting:
            continue
        if unknown:
            missing.append((task, unknown))
            continue
        offered.setdefault(task.tier, []).append(task)
    nxt = {}
    for tier, tasks in offered.items():
        tasks.sort(key=_next_key)
        items = [{"id": clip(t.id, ID_CHARS), "priority": _priority_short(t.priority),
                  "wave": t.wave, "stretch": t.stretch, "title": clip(t.title, TITLE_CHARS)} for t in tasks]
        nxt[tier] = _capped(items, MAX_NEXT_PER_TIER)
    held = {
        "shared": _capped([clip(t.id, ID_CHARS) for t in sorted(shared, key=_next_key)], MAX_HELD),
        "missing_blocker": _capped(
            [{"id": clip(t.id, ID_CHARS), "blockers": [clip(_ref_label(r), 40) for r in refs]}
             for t, refs in sorted(missing, key=lambda pair: _next_key(pair[0]))], MAX_HELD),
    }
    return table, nxt, held


def _priority_short(priority: str) -> str:
    found = _PRIORITY.search(priority or "")
    return f"P{found['rank']}" if found else "P?"


def _ref_label(ref: str) -> str:
    key = page_key(ref)
    return f"page {key[-8:]}" if key else ref


def _wave_text(wave: float | None) -> str:
    if wave is None:
        return "w?"
    return f"w{number_text(wave)}"


# --- building the digest --------------------------------------------------------------------------

def compute(*, now: dt.datetime | None = None, hours: float = DEFAULT_WINDOW_HOURS,
            since: dt.datetime | None = None, state_dir: str | os.PathLike[str] | None = None,
            board: str | os.PathLike[str] | None = None, refine_root: str | os.PathLike[str] | None = None,
            max_lines: int = MAX_LINES) -> dict[str, Any]:
    """The digest as a JSON-ready dict, already fitted to max_lines. now defaults to the
    current UTC time; the window is [since, now], since defaulting to hours before now. board
    names a snapshot file (default: the stored one); refine_root holds the track folders
    (default: refine_track.REFINE_DIR, lab/refine). OSError when the ledger cannot be opened."""
    now = (now or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc).replace(microsecond=0)
    if since is None:
        since = now - dt.timedelta(hours=hours)
    since = since.astimezone(dt.timezone.utc)
    if since > now:
        raise ValueError(f"the window starts at {iso(since)}, after now ({iso(now)})")
    state = ledger_mod.resolve_state_dir(state_dir)
    path = ledger_mod.ledger_path(state)
    rows = ledger_mod.read(state_dir=state)
    timed = sorted((r for r in rows if _ts(r) is not None), key=_sort_key)
    upto = [r for r in timed if _ts(r) <= now]
    window = [r for r in upto if _ts(r) >= since]

    problems: list[str] = []
    undated = len(rows) - len(timed)
    if undated:
        problems.append(f"{_plural(undated, 'ledger row')} without a readable ts left out")
    snapshot, board_problems = load_board(board, state_dir=state)
    problems.extend(board_problems)
    rulings, ruling_problems = open_rulings(Path(refine_root) if refine_root is not None else refine_track.REFINE_DIR)
    problems.extend(ruling_problems)

    digest: dict[str, Any] = {
        "ok": True, "outcome": "ok", "now": iso(now),
        "window": {"since": iso(since), "until": iso(now), "hours": round((now - since).total_seconds() / 3600, 2)},
        "ledger": {"path": _shown(path), "rows": len(rows),
                   "first_seq": rows[0]["seq"] if rows else None, "last_seq": rows[-1]["seq"] if rows else None},
        "ran": what_ran(window, now),
        "metrics": metrics_moved(upto, since, now),
        "rulings": rulings,
    }
    if snapshot is None:
        digest["board"] = None
    else:
        age = minutes_between(snapshot.taken_at, now) if snapshot.taken_at else None
        digest["board"] = {"path": snapshot.path, "taken_at": iso(snapshot.taken_at) if snapshot.taken_at else None,
                           "taken_by": clip(snapshot.taken_by, NAME_CHARS) or None, "age_min": age,
                           "stale": age is None or age > BOARD_STALE_MINUTES,
                           "tasks": len(snapshot.tasks), "queue_rows": len(snapshot.queue)}
        digest["gpu"] = gpu_queue(snapshot, now)
        digest["lane"], digest["next"], digest["held"] = lane(snapshot)
    digest["problems"] = _capped([clip(p, PROBLEM_CHARS) for p in problems], MAX_PROBLEMS)
    return fit(digest, max_lines)


# --- the size bound -------------------------------------------------------------------------------

def _lists(digest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every capped list of the digest, as {"items", "more"} holders, in trim order."""
    holders: list[Any] = [
        digest["ran"]["by_verb"], digest["ran"]["by_task"], digest["metrics"]["unchanged"],
        digest["metrics"]["new"], digest["metrics"]["moved"], digest["ran"]["failures"],
    ]
    if digest.get("board") is not None:
        holders += [digest["gpu"]["queued"], digest["gpu"]["complaints"], digest["gpu"]["running"],
                    digest["held"]["shared"], digest["held"]["missing_blocker"], *digest["next"].values()]
    holders += [digest["rulings"], digest["problems"]]
    return [h for h in holders if isinstance(h, dict) and "items" in h]


def fit(digest: dict[str, Any], max_lines: int = MAX_LINES) -> dict[str, Any]:
    """Drop the last entry of the longest list, counting it in that list's "more", until the
    text is at most max_lines lines. Lists are ordered most important first."""
    while len(render(digest).splitlines()) > max_lines:
        holders = [h for h in _lists(digest) if h["items"]]
        if not holders:
            break
        longest = max(holders, key=lambda h: len(h["items"]))
        longest["items"].pop()
        longest["more"] += 1
    return digest


# --- the text ---------------------------------------------------------------------------------------

def _more(holder: Mapping[str, Any], indent: str = "  ") -> list[str]:
    return [f"{indent}(+{holder['more']} more)"] if holder.get("more") else []


def _counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{name} {count}" for name, count in counts.items())


def render(digest: Mapping[str, Any]) -> str:
    """The digest as text: ASCII, one entry per line."""
    now, _ = parse_time(digest["now"])
    window = digest["window"]
    led = digest["ledger"]
    lines = [f"LAB DIGEST {digest['now']}: computed from the ledger, the board snapshot and the rulings files; "
             f"never from notes."]
    seqs = f", seq {led['first_seq']} to {led['last_seq']}" if led["rows"] else ""
    lines.append(f"Window: {window['since']} to {window['until']} ({span_text(window['hours'])}).")
    lines.append(f"Ledger: {_plural(led['rows'], 'row')}{seqs}, {led['path']}")
    board = digest.get("board")
    if board is None:
        lines.append("Board snapshot: none readable (see problems); the GPU, lane and next sections are empty.")
    else:
        by = f" by {board['taken_by']}" if board["taken_by"] else ""
        age = f"{board['age_min']} min old" if board["age_min"] is not None else "age unknown"
        stale = " STALE: fetch the board fresh before acting on it." if board["stale"] else ""
        lines.append(f"Board snapshot: taken {board['taken_at']}{by}, {age}, {_plural(board['tasks'], 'task')}, "
                     f"{_plural(board['queue_rows'], 'queue row')}.{stale}")

    ran = digest["ran"]
    lines += ["", f"## What ran ({clock(window['since'], now)} to {clock(window['until'], now)})"]
    if ran["rows"] == 0:
        lines.append("Nothing in the window.")
    else:
        lines.append(f"{_plural(ran['rows'], 'row')} by {_plural(ran['actors'], 'actor')} on "
                     f"{_plural(ran['tasks'], 'task')}; outcomes: {_counts(ran['outcomes'])}.")
    if ran["reads"]:
        lines.append(f"Read-only calls, not listed: {_counts(ran['reads'])}.")
    if ran["by_task"]["items"]:
        lines.append("By task, newest first:")
        for entry in ran["by_task"]["items"]:
            failed = f" ({entry['failures']} failed)" if entry["failures"] else ""
            pr = f", PR {entry['pr']}" if entry["pr"] else ""
            lines.append(f"  {entry['task']}: {_plural(entry['rows'], 'row')}{failed}, last {clock(entry['last_ts'], now)} "
                         f"{entry['last_verb']} {entry['last_outcome']} by {entry['last_actor']}{pr}; "
                         f"verbs: {', '.join(entry['verbs'])}")
        lines += _more(ran["by_task"])
    if ran["by_verb"]["items"]:
        lines.append("Without a task:")
        for entry in ran["by_verb"]["items"]:
            lines.append(f"  {entry['verb']}: {entry['rows']} ({_counts(entry['outcomes'])})")
        lines += _more(ran["by_verb"])
    if ran["failures"]["items"]:
        lines.append("Failures, newest first:")
        for entry in ran["failures"]["items"]:
            run = f", run {entry['run_id']}" if entry["run_id"] else ""
            lines.append(f"  {clock(entry['ts'], now)} {entry['task'] or '-'} {entry['verb']} {entry['outcome']} "
                         f"by {entry['actor']}{run} (ledger:{entry['seq']})")
        lines += _more(ran["failures"])

    metrics = digest["metrics"]
    lines += ["", "## Metrics that moved (newest in the window vs the newest reading before it)"]
    if not (metrics["moved"]["items"] or metrics["new"]["items"] or metrics["unchanged"]["items"]):
        lines.append("No metric was read in the window.")
    for entry in metrics["moved"]["items"]:
        pct = f", {entry['pct']:+.1f}%" if entry["pct"] is not None else ""
        unit = f" {entry['unit']}" if entry["unit"] else ""
        flags = (", low confidence" if entry["low_confidence"] else "") + \
                (", both readings in the window" if entry["within_window"] else "")
        lines.append(f"  {_series_name(entry)}: {number_text(entry['before'])} -> {number_text(entry['value'])}{unit} "
                     f"({delta_text(entry['delta'])}{pct}{flags}) ledger:{entry['before_seq']} -> ledger:{entry['seq']}")
    lines += _more(metrics["moved"])
    if metrics["new"]["items"]:
        lines.append("First readings of a series:")
        for entry in metrics["new"]["items"]:
            unit = f" {entry['unit']}" if entry["unit"] else ""
            low = ", low confidence" if entry["low_confidence"] else ""
            lines.append(f"  {_series_name(entry)}: {number_text(entry['value'])}{unit}{low} (ledger:{entry['seq']})")
        lines += _more(metrics["new"])
    if metrics["unchanged"]["items"]:
        names = "; ".join(f"{_series_name(e)} {number_text(e['value'])}" for e in metrics["unchanged"]["items"])
        more = f" (+{metrics['unchanged']['more']} more)" if metrics["unchanged"]["more"] else ""
        lines.append(f"Unchanged: {names}{more}")

    if board is not None:
        lines += _gpu_lines(digest["gpu"], now)
    lines += _ruling_lines(digest["rulings"])
    if board is not None:
        lines += _lane_lines(digest["lane"])
        lines += _next_lines(digest["next"], digest["held"])
    if digest["problems"]["items"]:
        lines += ["", "## Problems"]
        lines += [f"  {p}" for p in digest["problems"]["items"]]
        lines += _more(digest["problems"])
    return "\n".join(_cut(line) for line in lines) + "\n"


def _cut(line: str) -> str:
    """One output line: ASCII, trailing space removed, cut at LINE_CHARS with "...". Unlike
    clip() it keeps the line's own spacing (indents, table columns)."""
    one = ascii_text(line.replace("\r", " ").replace("\n", " ")).rstrip()
    return one if len(one) <= LINE_CHARS else one[:LINE_CHARS - 3].rstrip() + "..."


def _series_name(entry: Mapping[str, Any]) -> str:
    node = f" @{entry['node']}" if entry["node"] else ""
    label = f" [{entry['label']}]" if entry["label"] else ""
    return f"{entry['metric']}{node}{label}"


def _gpu_lines(gpu: Mapping[str, Any], now: dt.datetime | None) -> list[str]:
    lines = ["", "## GPU queue (open rows in the snapshot)"]
    if not gpu["summary"]:
        lines.append("No open rows: nothing running or queued.")
        return lines
    lines.append("; ".join(f"{s['gpu']}: {s['running']} running, {s['queued']} queued" for s in gpu["summary"]))
    for entry in gpu["running"]["items"]:
        task = f" ({entry['task']})" if entry["task"] else ""
        started = f"started {clock(entry['started'], now)}" if entry["started"] else "start unknown"
        elapsed = f", {entry['elapsed_min']} min elapsed" if entry["elapsed_min"] is not None else ""
        cap = f" of a {number_text(entry['cap_min'])} min cap" if entry["cap_min"] is not None else ""
        extended = ", extended" if entry["extended"] else ""
        over = f", OVER CAP by {number_text(entry['over_min'])} min" if entry["over_min"] else ""
        lines.append(f"  running on {entry['gpu']}: {entry['request']}{task}, {started}{elapsed}{cap}{extended}{over}")
    lines += _more(gpu["running"])
    for entry in gpu["queued"]["items"]:
        task = f" ({entry['task']})" if entry["task"] else ""
        minutes = f"{number_text(entry['minutes'])} min" if entry["minutes"] is not None else "? min"
        waited = f", waiting {entry['waited_min']} min" if entry["waited_min"] is not None else ", waiting ?"
        due = ", complaint due" if entry["complaint_due"] else ""
        lines.append(f"  queued #{entry['position']} on {entry['gpu']}: {entry['request']}{task}, {minutes}{waited}{due}")
    lines += _more(gpu["queued"])
    if gpu["complaints"]["items"]:
        lines.append("Complaints on open rows:")
        for entry in gpu["complaints"]["items"]:
            since = f", {entry['minutes_since']} min" if entry["minutes_since"] is not None else ""
            lines.append(f"  {entry['request']} ({entry['status']}{since}): {entry['text']}")
        lines += _more(gpu["complaints"])
    return lines


def _ruling_lines(rulings: Mapping[str, Any]) -> list[str]:
    lines = ["", "## Rulings waiting on Ilan (unanswered in lab/refine/<track>/OWNER_RULINGS.md)"]
    if not rulings["items"]:
        lines.append(f"None unanswered ({rulings['answered']} answered).")
        return lines
    for entry in rulings["items"]:
        origin = f" ({entry['origin']})" if entry["origin"] else ""
        lines.append(f"  {entry['track']} {entry['number']}{origin}: {entry['question']} Now: {entry['now']}")
    lines += _more(rulings)
    return lines


def _lane_lines(table: Mapping[str, Any]) -> list[str]:
    lines = ["", f"## Ilan's lane: {_plural(table['tasks'], 'task')}, {table['locked']} locked"]
    tiers = table["tiers"]
    width = max(len(s) for s in table["counts"]) if table["counts"] else 11
    lines.append("  " + " " * width + "".join(f"{tier:>8}" for tier in tiers))
    for status, row in table["counts"].items():
        lines.append(f"  {status:<{width}}" + "".join(f"{row[tier]:>8}" for tier in tiers))
    return lines


def _next_lines(nxt: Mapping[str, Any], held: Mapping[str, Any]) -> list[str]:
    lines = ["", "## Next unblocked (Not started, unlocked, Ilan's alone, every blocker Done; "
             "P0 first, then wave)"]
    for tier, holder in nxt.items():
        if tier == NO_TIER and not holder["items"]:
            continue
        name = "No tier (the referee sets one first)" if tier == NO_TIER else tier
        if not holder["items"]:
            lines.append(f"{name}: none")
            continue
        lines.append(f"{name}:")
        for entry in holder["items"]:
            stretch = " stretch" if entry["stretch"] else ""
            lines.append(f"  {entry['id']} {entry['priority']} {_wave_text(entry['wave'])}{stretch} {entry['title']}")
        lines += _more(holder)
    if held["shared"]["items"]:
        more = f" (+{held['shared']['more']} more)" if held["shared"]["more"] else ""
        lines.append(f"Held back, a teammate is also assigned: {', '.join(held['shared']['items'])}{more}")
    if held["missing_blocker"]["items"]:
        parts = [f"{e['id']} (needs {', '.join(e['blockers'])})" for e in held["missing_blocker"]["items"]]
        more = f" (+{held['missing_blocker']['more']} more)" if held["missing_blocker"]["more"] else ""
        lines.append(f"Held back, a blocker is missing from the snapshot: {'; '.join(parts)}{more}")
    return lines


def text(**kwargs: Any) -> str:
    """compute(**kwargs) as text: what the launcher puts in a sitting's prompt."""
    return render(compute(**kwargs))


# --- the verb -------------------------------------------------------------------------------------

def _time_arg(value: str) -> dt.datetime:
    moment, naive = parse_time(value)
    if moment is None or naive:
        raise argparse.ArgumentTypeError(f"{value!r} is not ISO 8601 with a UTC offset, such as 2026-10-10T14:00:00Z")
    return moment


def _hours_arg(value: str) -> float:
    try:
        hours = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number of hours") from None
    if not math.isfinite(hours) or hours <= 0:
        raise argparse.ArgumentTypeError("--hours must be a positive number")
    return hours


def main(argv: Sequence[str]) -> dict[str, Any] | int:
    """What a sitting starts from: what ran, metrics that moved, GPU queue, rulings, Ilan's lane, next tasks."""
    parser = argparse.ArgumentParser(
        prog="python -m lab digest", description=main.__doc__,
        epilog="Answers in text (at most %d lines); --json gives one JSON object. --schema prints how a sitting "
               "writes the board snapshot." % MAX_LINES)
    span = parser.add_mutually_exclusive_group()
    span.add_argument("--hours", type=_hours_arg, default=DEFAULT_WINDOW_HOURS,
                      help=f"the window: this many hours before now (default {number_text(DEFAULT_WINDOW_HOURS)})")
    span.add_argument("--since", type=_time_arg, help="the window starts here (ISO 8601 with a UTC offset)")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--board", metavar="FILE", help="read this board snapshot instead of <state dir>/board.json")
    source.add_argument("--save-board", metavar="FILE",
                        help="check FILE, store it as <state dir>/board.json, then answer with it")
    parser.add_argument("--json", action="store_true", help="answer with one JSON object instead of text")
    parser.add_argument("--schema", action="store_true", help="print the board snapshot schema and stop")
    parser.add_argument("--now", type=_time_arg, help="compute as of this time (replays and tests)")
    parser.add_argument("--refine-root", metavar="DIR", help="the folder of track folders (default lab/refine)")
    args = parser.parse_args(list(argv))
    if args.schema:
        print(SCHEMA, end="")
        return 0
    board = args.board
    if args.save_board:
        try:
            board = str(save_board(args.save_board))
        except BoardError as exc:
            return {"ok": False, "outcome": "invalid_board", "error": str(exc), "problems": exc.problems}
        except OSError as exc:
            return {"ok": False, "outcome": "error", "error": f"cannot store the board snapshot: {type(exc).__name__}: {exc}"}
    try:
        digest = compute(now=args.now, hours=args.hours, since=args.since, board=board, refine_root=args.refine_root)
    except ValueError as exc:
        return {"ok": False, "outcome": "bad_window", "error": str(exc)}
    except OSError as exc:
        return {"ok": False, "outcome": "error", "error": f"cannot read the ledger: {type(exc).__name__}: {exc}"}
    if args.json:
        return digest
    print(render(digest), end="")
    return 0
