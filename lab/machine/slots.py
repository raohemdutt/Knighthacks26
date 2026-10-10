"""slot: the GPU queue for pod B, the one exclusive card that every agent shares.

Anything that trains, benchmarks, restarts a process or needs the whole GPU asks for it here and waits its turn.

    python -m lab slot request  --task L3 --what "smoke LoRA 200 steps" [--minutes 60 --why "..." --measured "..."]
                                [--kind lora] [--override "why it must run there"]
    python -m lab slot start    --row <id or url> [--reply "answer to the critique"] [--minutes 15]
    python -m lab slot extend   --row <id or url> --note "what the slot has already proven"
    python -m lab slot done     --row <id or url> --result "one line with the numbers" [--cancel]
    python -m lab slot complain --row <id or url> [--text "more detail"]
    python -m lab slot status   [--gpu B] [--row <id or url>] [--task L3]

The queue is the Notion database "GPU Queue (R9700 pods)": one row per request, read and written through
the Notion REST API (lab/machine/slots_notion.py) with the NOTION_TOKEN key. The verbs enforce the queue
rules on the client, because Notion cannot:

  request   adds a row: Request "<task>: <what>", Task ID, GPU "B exclusive" (--gpu A for the shared teacher
            pod, which is only queued to restart or change the server itself), Minutes 15, Status Queued,
            Agent Ilan, Requested now. 60 minutes is for a training run or a declared long job and carries a
            justification (below). Asking again with the same words while the first row is still Queued
            returns that row (outcome already_queued) instead of adding a second. Before it adds the row it
            asks the placement plan (lab/placement.py) whether the kind of job (from --what's words, or
            --kind) belongs on the card asked for, and refuses one that contradicts the plan (outcome
            placement_contradiction) unless --override says why it must run there; the reason is then
            written to the Progress note as an Override line. A plan that cannot be built never blocks
            a request: the answer says so (placement verdict unavailable).
  start     sets Status Running and Started now, in one update, and only when the row's justification is in
            order, no row on the GPU is Running and no other Queued row goes before this one. The order is
            shortest first: fewer Minutes goes first, and equal Minutes go to the earlier Requested, except
            that a starved row goes first (below). A Running row blocks even when it looks stale (the answer
            says when it is past its cap); complain instead. Notion cannot make the check and the update one
            step, so the queue is read again after the update, and a row that then finds any other Running
            row on its GPU puts itself back to Queued and answers lost_race: two agents that start in the
            same moment can never both keep the card, and may both step back and try again. --reply and
            --minutes answer the critique or shorten the row before any of this (below).
  extend    one extension of 15 minutes: Extended is checked and the Progress note says what the slot has
            already proven, after any text the note holds. A Running row only, once, while it is still
            inside its cap.
  done      Status Done, Ended now, Result (one line, with the numbers). It releases the card. --cancel
            withdraws a row that never started (Status Cancelled).
  complain  writes the Complaint on the agent's own Queued row, once it has waited more than twice its
            Minutes (or a Running row on its GPU is past its cap): the wait so far, what it is waiting
            behind, and any --text. A later complaint is added after the earlier ones.
  status    the open rows of the GPU in queue order, with who waits how long, when the Running one ends,
            whether it is overrun, and whether each Queued row may start; a Queued row that cannot start
            for want of a critique or a reply is listed apart, under blocked; --row narrows it to one row,
            --task to one task's rows. It changes nothing.

Two guards sit on the order.

Starvation. A Queued row that has been passed over three times, or has waited more than twice its own
Minutes, goes next whatever its length: starvation() says which, and queue_order() puts such rows before every
other row, the earliest request first, the rest staying shortest first. A row is passed over each time a row
requested after it starts while it waits; the count is read from the Started times of the GPU's Running, Done
and Ended by referee rows, so no row carries a counter and no verb writes another agent's row.

Justification. A request for more than 30 minutes carries a justification: --what says what runs, --why why it
cannot be split into 15-minute slots, --measured what was already measured. request refuses it
(justification_required) while --why or --measured is missing, and otherwise writes the justification as the
first line of the row's Progress note. The referee reads each unanswered justification once and adds one
critique line after it (python -m lab referee --critique). The row cannot start until its agent has shortened
it (start --minutes 15) or replied to the critique (start --reply "..."), which adds a reply line. A row that
cannot start for want of a critique or a reply is left out of the order, so it holds nobody up. A request of
30 minutes or less needs none of this.
Idle turn. A startable Queued row first in line whose agent stops acting would hold the card for good, since only
its own agent may start or cancel it. When the card has stood free with it startable for longer than its own
Minutes while a later row waited (idle_turn), that later row may start ahead of it; the start answer names it under
passed_idle. Its row is not written and it keeps its place, so its agent can still start it while the card is free.

A slot is capped at its Minutes (plus 15 once extended): cap_minutes() and ends_at() give the cap, and
is_overrun() says a Running row has passed it. The verbs do not end an overrun slot; the referee does.
A verb writes only a row of this data source whose Agent is its own (--agent, default Ilan); status
reads any row.

Every call answers with one JSON line (ok, outcome, verb, the row and its summary, and when it is refused a
reason and a next step) and an exit code: 0 when it worked, 1 when the rules or Notion refused it, 2 for bad
arguments and for a missing token. Trouble with Notion itself is outcome error, with its kind (unauthorized,
not_found, rate_limited, transport, ...). Each call that gets as far as a decision also appends one row to the
ledger (verb "slot <name>", the outcome, the task - the call's own, the row's, else $LAB_TASK - refs.slot the
row's id, node pod-b or pod-a, the minutes and waits, and for the guards the justification state, the reply or
shortening made, and which starvation rule put a row first or kept another behind it); a refusal is a row too.
The actor is $LAB_ACTOR and the tier $LAB_TIER, as for every ledger row (lab/machine/record.py), and the actor
signs the lines a call adds to a Progress note. A ledger that cannot be written never undoes what Notion
already holds: the answer then carries ledger_error and the exit code stays what the queue's outcome says.
While the STOP file exists (lab/machine/stop.py) every slot verb is refused before it runs.

A request, a start and an extend that went through also carry the numbers lab/machine/alerts.py judges a slot by,
in the spelling it reads: the metric slot.minutes (the Minutes asked for) on a request and a start, and
slot.extend_minutes on an extend. A refused call, and a call that sent nothing to Notion, carry none.

No token. NOTION_TOKEN is looked up in the process environment, then in the .env files of ready.py. When it
is missing nothing is sent anywhere: the verb answers outcome no_token with the exact Notion tool calls a
sitting makes instead (the row to create or the properties to set, in the form its Notion tools take, with
the checks to make first) and exits 2. The ledger still gets its row. A request that lacks its justification
is refused before the token is looked for.

Settings, for rehearsals: LAB_QUEUE_DATA_SOURCE names another data source than the live queue's, and
LAB_NOTION_URL another address than https://api.notion.com. The token is never printed, logged or written to
the ledger; error messages are scrubbed of it.

For other code: Slot, slot_from_page(), queue_order(), starvation(), justification_state(), cap_minutes(),
ends_at(), is_overrun(), complaint_due(), connect() (the client the token makes) and SlotQueue (the six verbs as
methods) are the same rules the verbs use.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import logging
import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from lab.machine import alerts, record, stop
from lab.machine import slots_notion as notion
from lab.machine.allowed import NOTION_SERVER
from lab.machine.ready import collect_env, default_env_files

__all__ = [
    "COLUMNS", "DATA_SOURCE_ID", "GPUS", "Slot", "SlotQueue", "Starvation", "Thread", "board_filter", "cap_minutes",
    "complaint_due", "connect", "data_source_of", "ends_at", "idle_turn", "is_overrun", "jumped", "justification_state",
    "main", "mcp_properties", "note_entry", "parse_moment", "queue_order", "read_thread", "rest_properties", "run",
    "slot_from_page", "starvation",
]

log = logging.getLogger("lab.slots")

UTC = dt.timezone.utc

ENV_TOKEN = "NOTION_TOKEN"
ENV_DATA_SOURCE = "LAB_QUEUE_DATA_SOURCE"
ENV_URL = "LAB_NOTION_URL"

#: The GPU Queue (R9700 pods) data source.
DATA_SOURCE_ID = "dc46dcb7-79d4-4727-bc04-40426b5d9114"

#: The two cards by the letter a caller types: the value of the GPU column, and where the ledger puts the work.
GPUS = {"A": "A teacher shared service", "B": "B exclusive"}
NODES = {"A": "pod-a", "B": "pod-b"}
DEFAULT_GPU = "B"
DEFAULT_AGENT = "Ilan"

QUEUED, RUNNING, DONE, CANCELLED = "Queued", "Running", "Done", "Cancelled"
ENDED_BY_REFEREE = "Ended by referee"
OPEN_STATUSES = (QUEUED, RUNNING)
#: The rows that tell where a Queued row stands: the open ones, and the ones that started and are over.
BOARD_STATUSES = (QUEUED, RUNNING, DONE, ENDED_BY_REFEREE)
#: The rows that have started, whether they still run or not.
STARTED_STATUSES = (RUNNING, DONE, ENDED_BY_REFEREE)

DEFAULT_MINUTES = 15
LONG_MINUTES = 60
SLOT_MINUTES = (DEFAULT_MINUTES, LONG_MINUTES)
LONG_ABOVE_MINUTES = 30          # a request for more minutes than this carries a justification
EXTENSION_MINUTES = 15
COMPLAIN_AFTER_FACTOR = 2        # a row may complain once it has waited more than this many times its Minutes
STARVED_PASSES = 3               # a Queued row passed over this many times goes next
STARVED_AFTER_FACTOR = 2         # so does one that has waited more than this many times its Minutes

MAX_WHAT_CHARS = 200
MAX_WHY_CHARS = 120
MAX_MEASURED_CHARS = 200
MAX_NOTE_CHARS = 1000            # progress note, result, extra complaint text
MAX_REPLY_CHARS = 300            # the reply to a critique
MAX_CRITIQUE_CHARS = 300         # what the referee's critique says
MAX_OVERRIDE_CHARS = 200         # why a request that contradicts the placement plan must run where it asks
#: The kinds of job the placement plan places (lab/placement.py KINDS; spelled out here so this module loads alone).
PLACEMENT_KINDS = ("teacher", "sampler", "bench", "ax", "lora")
MAX_COMPLAINT_CHARS = 2000       # what one Notion text value holds; older entries are dropped to fit
COMPLAINT_JOIN = " | "           # between the entries of a Complaint or of a Progress note

TASK_ID = re.compile(r"[A-Za-z]{1,3}[0-9]{1,3}")

#: Each column of the GPU Queue and the kind of value it holds.
COLUMNS = {
    "Request": "title", "Task ID": "rich_text", "GPU": "select", "Minutes": "number", "Status": "select",
    "Agent": "rich_text", "Requested": "date", "Started": "date", "Ended": "date", "Extended": "checkbox",
    "Progress note": "rich_text", "Complaint": "rich_text", "Result": "rich_text",
}

_LATEST = dt.datetime.max.replace(tzinfo=UTC)


# --- times ----------------------------------------------------------------------------

def utc_now() -> dt.datetime:
    """The current time, aware, UTC, to the millisecond (what Notion stores)."""
    now = dt.datetime.now(UTC)
    return now.replace(microsecond=now.microsecond // 1000 * 1000)


def iso(moment: dt.datetime) -> str:
    """An aware time as UTC text to the millisecond, ending in Z."""
    return moment.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def minutes_between(later: dt.datetime, earlier: dt.datetime) -> float:
    return (later - earlier).total_seconds() / 60.0


def tidy(value: float) -> int | float:
    """A number of minutes rounded to a tenth, a whole number when it is one."""
    rounded = round(value, 1)
    return int(rounded) if rounded == int(rounded) else rounded


def parse_moment(text: Any, zone: str | None = None) -> dt.datetime | None:
    """A Notion date-time as an aware UTC time. None for anything that is not one: a date without a time of
    day cannot order rows within a day. A time with no UTC offset takes the date's time_zone, else UTC."""
    if not isinstance(text, str) or "T" not in text:
        return None
    try:
        moment = dt.datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        try:
            moment = moment.replace(tzinfo=ZoneInfo(zone) if zone else UTC)
        except (ZoneInfoNotFoundError, ValueError):
            moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


# --- a row ----------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Slot:
    """One row of the GPU Queue. Times are aware UTC; a column nobody filled is "" or None."""

    id: str                      # the page id, 32 lowercase hex digits
    url: str
    request: str
    task_id: str
    gpu: str                     # the GPU column's value, e.g. "B exclusive"
    minutes: float | None
    status: str | None
    agent: str
    requested: dt.datetime | None
    created: dt.datetime | None  # when Notion made the row; orders rows whose Requested has no time of day
    started: dt.datetime | None
    ended: dt.datetime | None
    extended: bool
    progress_note: str
    complaint: str
    result: str
    data_source: str | None = None   # the data source the page hangs from, when the page says

    @property
    def minutes_or_default(self) -> float:
        return float(self.minutes) if self.minutes is not None else float(DEFAULT_MINUTES)

    @property
    def requested_at(self) -> dt.datetime | None:
        return self.requested or self.created


def _prop(properties: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = properties.get(name)
    return value if isinstance(value, Mapping) else {}


def _date_prop(properties: Mapping[str, Any], name: str) -> dt.datetime | None:
    value = _prop(properties, name).get("date")
    if not isinstance(value, Mapping):
        return None
    return parse_moment(value.get("start"), value.get("time_zone"))


def _select_prop(properties: Mapping[str, Any], name: str) -> str | None:
    prop = _prop(properties, name)
    chosen = prop.get("select") or prop.get("status")
    return chosen.get("name") if isinstance(chosen, Mapping) and isinstance(chosen.get("name"), str) else None


def slot_from_page(page: Mapping[str, Any]) -> Slot:
    """The Slot a Notion page object holds. ValueError for something that is not a page."""
    page_id = notion.page_id_of(str(page.get("id", "")))
    if page.get("object") not in (None, "page") or page_id is None:
        raise ValueError("not a Notion page: no page id")
    raw = page.get("properties")
    properties: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}
    number = _prop(properties, "Minutes").get("number")
    parent = page.get("parent")
    source = parent.get("data_source_id") if isinstance(parent, Mapping) else None
    return Slot(
        id=page_id,
        url=str(page.get("url") or f"https://www.notion.so/{page_id}"),
        request=notion.plain_text(_prop(properties, "Request")),
        task_id=notion.plain_text(_prop(properties, "Task ID")).strip(),
        gpu=_select_prop(properties, "GPU") or "",
        minutes=float(number) if isinstance(number, (int, float)) and not isinstance(number, bool) else None,
        status=_select_prop(properties, "Status"),
        agent=notion.plain_text(_prop(properties, "Agent")).strip(),
        requested=_date_prop(properties, "Requested"),
        created=parse_moment(page.get("created_time")),
        started=_date_prop(properties, "Started"),
        ended=_date_prop(properties, "Ended"),
        extended=_prop(properties, "Extended").get("checkbox") is True,
        progress_note=notion.plain_text(_prop(properties, "Progress note")),
        complaint=notion.plain_text(_prop(properties, "Complaint")),
        result=notion.plain_text(_prop(properties, "Result")),
        data_source=notion.page_id_of(source) if isinstance(source, str) else None,
    )


# --- the progress note ----------------------------------------------------------------

JUSTIFICATION, CRITIQUE, REPLY, SHORTENED = "Justification", "Critique", "Reply", "Shortened"
OVERRIDE = "Override"            # a Progress note line: why a request runs against the placement plan

#: Where a row stands on its justification.
NOT_REQUIRED, MISSING, AWAITING_CRITIQUE, AWAITING_REPLY, CLEARED = (
    "not_required", "missing", "awaiting_critique", "awaiting_reply", "cleared")
STARTABLE = (NOT_REQUIRED, CLEARED)

#: What a Queued row waits for, by the state of its justification.
WAITING_FOR = {
    MISSING: "a justification (cancel it and ask again with --why and --measured, or shorten it)",
    AWAITING_CRITIQUE: "the referee's critique of its justification",
    AWAITING_REPLY: "its agent's reply to the referee's critique",
}

#: How a start that the state of the justification refuses is named in the answer.
GATE_OUTCOMES = {MISSING: "justification_required", AWAITING_CRITIQUE: "critique_pending",
                 AWAITING_REPLY: "reply_required"}

NOTE_ENTRY = re.compile(r"(Justification|Critique|Reply|Shortened) \(([^()]*), (\d{4}-\d\d-\d\dT\d\d:\d\dZ)\): (.*)",
                        re.DOTALL)
LONG_ENDING = re.compile(r" \(\d+ min: .*\)\Z")


def signer(actor: str) -> str:
    """An actor's name as the signature of a Progress note line: no brackets, commas or spaces in it."""
    return re.sub(r"[^\w.@:/-]+", "-", actor.strip()).strip("-") or record.UNKNOWN_ACTOR


def note_entry(label: str, actor: str, now: dt.datetime, text: str) -> str:
    """One signed line of a Progress note: "<label> (<actor>, <time>Z): <text>". The text cannot hold the
    separator between lines, so it can never pass for a line of its own."""
    return f"{label} ({signer(actor)}, {iso(now)[:16]}Z): {text.replace(COMPLAINT_JOIN, ' / ')}"


@dataclasses.dataclass(frozen=True)
class Thread:
    """The lines of a Progress note that belong to the justification step; None for one not written yet."""

    justification: str | None = None
    critique: str | None = None
    reply: str | None = None


def read_thread(note: str) -> Thread:
    """The justification, the critique after it and the reply after that, from a Progress note: the first line of
    each kind, in that order. Any other text in the note is ignored."""
    found: dict[str, str] = {}
    for part in note.split(COMPLAINT_JOIN):
        match = NOTE_ENTRY.fullmatch(part.strip())
        if match is None:
            continue
        label, text = match.group(1), match.group(4)
        if label == JUSTIFICATION and "justification" not in found:
            found["justification"] = text
        elif label == CRITIQUE and "justification" in found and "critique" not in found:
            found["critique"] = text
        elif label == REPLY and "critique" in found and "reply" not in found:
            found["reply"] = text
    return Thread(**found)


def justification_text(why: str | None, measured: str | None) -> str:
    """What the justification line of a request says, from --why and --measured."""
    return f"cannot be split because: {why}; already measured: {measured}"


# --- the rules ------------------------------------------------------------------------

def justification_state(slot: Slot) -> str:
    """Where a row stands on its justification: NOT_REQUIRED up to 30 minutes, else MISSING without a justification
    line, AWAITING_CRITIQUE with one, AWAITING_REPLY once the referee has written a critique, CLEARED once its
    agent has replied. A row shortened to 30 minutes or less is NOT_REQUIRED again."""
    if slot.minutes_or_default <= LONG_ABOVE_MINUTES:
        return NOT_REQUIRED
    thread = read_thread(slot.progress_note)
    if thread.justification is None:
        return MISSING
    if thread.critique is None:
        return AWAITING_CRITIQUE
    return CLEARED if thread.reply is not None else AWAITING_REPLY


def justification_view(slot: Slot) -> dict[str, Any] | None:
    """The state of a row's justification and the lines it holds, as an answer shows them; None for a row that
    needs none."""
    state = justification_state(slot)
    if state == NOT_REQUIRED:
        return None
    thread = read_thread(slot.progress_note)
    view: dict[str, Any] = {"state": state}
    for name in ("justification", "critique", "reply"):
        text = getattr(thread, name)
        if text is not None:
            view[name] = text
    return view


@dataclasses.dataclass(frozen=True)
class Starvation:
    """Why a Queued row goes next whatever its length: code is passed_over or waited, why a clause that follows
    the row ("has been passed over 3 times by later requests")."""

    code: str
    why: str
    passed_over: int


def passed_over(slot: Slot, rows: Iterable[Slot]) -> int:
    """How many rows on the GPU of slot were requested after it and started after it was requested: each took
    the card while slot waited. Only rows that have started count (Running, Done, Ended by referee)."""
    since = slot.requested_at
    if since is None:
        return 0
    return sum(1 for row in rows
               if row.id != slot.id and row.gpu == slot.gpu and row.status in STARTED_STATUSES
               and row.started is not None and row.started > since
               and row.requested_at is not None and row.requested_at > since)


def starvation(slot: Slot, rows: Iterable[Slot], now: dt.datetime) -> Starvation | None:
    """The reason a Queued row goes next whatever its length, else None: it has been passed over three times, or
    it has waited more than twice its own Minutes. rows are the rows of its GPU, with the ones that started."""
    since = slot.requested_at
    if slot.status != QUEUED or since is None:
        return None
    passes = passed_over(slot, rows)
    if passes >= STARVED_PASSES:
        return Starvation("passed_over", f"has been passed over {passes} times by later requests", passes)
    waited = minutes_between(now, since)
    if waited > STARVED_AFTER_FACTOR * slot.minutes_or_default:
        return Starvation("waited", f"has waited {tidy(waited)} minutes, more than twice the "
                                    f"{tidy(slot.minutes_or_default)} asked for", passes)
    return None


def queue_key(slot: Slot) -> tuple[float, dt.datetime, dt.datetime, str]:
    """Where a row stands in the queue by length alone, smaller going first: fewer Minutes first (shortest job
    first), equal Minutes by the earlier Requested, then by when Notion made the row, then by id so the order is
    total."""
    return (slot.minutes_or_default, slot.requested_at or _LATEST, slot.created or _LATEST, slot.id)


def queue_order(rows: Iterable[Slot], now: dt.datetime) -> list[Slot]:
    """The Queued rows of rows that may take their turn, in the order they may start: first the starved rows, the
    earliest request first, then the others shortest first (queue_key). A row that cannot start for want of a
    critique or a reply is not in the line, so it holds nobody up. rows are the rows of the GPU with the ones
    that started, which is what starvation() reads."""
    board = list(rows)
    line = [row for row in board if row.status == QUEUED and justification_state(row) in STARTABLE]
    hungry = {row.id for row in line if starvation(row, board, now) is not None}
    first = sorted((row for row in line if row.id in hungry),
                   key=lambda row: (row.requested_at or _LATEST, row.created or _LATEST, row.id))
    rest = sorted((row for row in line if row.id not in hungry), key=queue_key)
    return first + rest


def cap_minutes(slot: Slot) -> float:
    """How long the slot may run: its Minutes, and 15 more once it has been extended."""
    return slot.minutes_or_default + (EXTENSION_MINUTES if slot.extended else 0)


def ends_at(slot: Slot) -> dt.datetime | None:
    """When a started slot reaches its cap; None while it has no Started."""
    return slot.started + dt.timedelta(minutes=cap_minutes(slot)) if slot.started else None


def is_overrun(slot: Slot, now: dt.datetime) -> bool:
    """True for a Running row that has passed its cap."""
    end = ends_at(slot)
    return slot.status == RUNNING and end is not None and now > end


def running_on(rows: Iterable[Slot], gpu: str, *, other_than: str | None = None) -> list[Slot]:
    """The Running rows of a GPU (the column's value), oldest start first."""
    found = [row for row in rows if row.status == RUNNING and row.gpu == gpu and row.id != other_than]
    return sorted(found, key=lambda row: (row.started or dt.datetime.min.replace(tzinfo=UTC), row.id))


def ready_since(slot: Slot) -> dt.datetime | None:
    """When a Queued row could first take its turn: its request, or the later reply or shortening that cleared it."""
    moments = [slot.requested_at]
    for part in slot.progress_note.split(COMPLAINT_JOIN):
        match = NOTE_ENTRY.fullmatch(part.strip())
        if match is not None and match.group(1) in (REPLY, SHORTENED):
            moments.append(parse_moment(match.group(3).replace("Z", ":00Z")))
    found = [moment for moment in moments if moment is not None]
    return max(found) if found else None


def idle_turn(slot: Slot, waiting: Slot, rows: Iterable[Slot], now: dt.datetime) -> bool:
    """True when slot, a startable Queued row before waiting in the line, has let its turn go: no row runs on its
    GPU, and the card has stood free with slot startable for longer than slot's own Minutes while waiting waited
    (counted from the latest of slot's ready_since, waiting's request and the last Ended on the GPU). Its agent
    has stopped acting, so it no longer holds waiting up; it keeps its place and may still start while the card
    is free. Nobody writes its row."""
    board = list(rows)
    if slot.status != QUEUED or justification_state(slot) not in STARTABLE or running_on(board, slot.gpu):
        return False
    anchors = [ready_since(slot), waiting.requested_at,
               *(row.ended for row in board if row.gpu == slot.gpu and row.id != slot.id)]
    since = max((moment for moment in anchors if moment is not None), default=None)
    return since is not None and minutes_between(now, since) > slot.minutes_or_default


def _before(slot: Slot, rows: Sequence[Slot], now: dt.datetime) -> list[Slot]:
    line = queue_order(rows, now)
    places = [row.id for row in line]
    if slot.id not in places:
        return []
    return [row for row in line[: places.index(slot.id)] if row.gpu == slot.gpu]


def ahead_of(slot: Slot, rows: Iterable[Slot], now: dt.datetime) -> list[Slot]:
    """The Queued rows on the GPU of slot that go before it, less the ones that let their turn go (idle_turn). A row
    that is not in the line has none."""
    board = list(rows)
    return [row for row in _before(slot, board, now) if not idle_turn(row, slot, board, now)]


def passed_idle(slot: Slot, rows: Iterable[Slot], now: dt.datetime) -> list[Slot]:
    """The Queued rows before slot in the line that it may go ahead of because they let their turn go (idle_turn)."""
    board = list(rows)
    return [row for row in _before(slot, board, now) if idle_turn(row, slot, board, now)]


def jumped(slot: Slot, rows: Iterable[Slot], now: dt.datetime) -> list[Slot]:
    """The Queued rows that go behind slot only because it is starved: shortest first would put them before it."""
    board = list(rows)
    line = queue_order(board, now)
    places = [row.id for row in line]
    if slot.id not in places or starvation(slot, board, now) is None:
        return []
    mine = queue_key(slot)
    return [row for row in line[places.index(slot.id) + 1:] if row.gpu == slot.gpu and queue_key(row) < mine]


def complaint_due(slot: Slot, rows: Iterable[Slot], now: dt.datetime) -> tuple[bool, str, dt.datetime | None]:
    """May this Queued row write a complaint now? (yes or no, the reason, when it first may). It may once it has
    waited more than twice its Minutes, or while a Running row on its GPU is past its cap."""
    since = slot.requested_at
    threshold = COMPLAIN_AFTER_FACTOR * slot.minutes_or_default
    earliest = since + dt.timedelta(minutes=threshold) if since else None
    if since is not None and now > earliest:
        waited = f"waited {tidy(minutes_between(now, since))} minutes"
        return True, f"{waited}, more than twice the {tidy(slot.minutes_or_default)} asked for", earliest
    overrun = [row for row in running_on(rows, slot.gpu) if is_overrun(row, now)]
    if overrun:
        return True, f"the Running row {overrun[0].request!r} is past its cap", earliest
    return False, "it has not waited twice its minutes yet", earliest


# --- columns on the wire --------------------------------------------------------------

def rest_properties(change: Mapping[str, Any]) -> dict[str, Any]:
    """{column: Python value} as the Notion REST API takes it: text, number, option name, bool, aware time or None."""
    out: dict[str, Any] = {}
    for name, value in change.items():
        kind = COLUMNS[name]
        if kind == "title":
            out[name] = notion.title(value)
        elif kind == "rich_text":
            out[name] = notion.rich_text(value)
        elif kind == "number":
            out[name] = notion.number(value)
        elif kind == "select":
            out[name] = notion.select(value)
        elif kind == "checkbox":
            out[name] = notion.checkbox(value)
        else:
            out[name] = notion.date(None if value is None else iso(value))
    return out


def mcp_properties(change: Mapping[str, Any]) -> dict[str, Any]:
    """The same change in the flat form the Notion MCP tools take: a date is two keys, a checkbox __YES__ or __NO__."""
    out: dict[str, Any] = {}
    for name, value in change.items():
        kind = COLUMNS[name]
        if kind == "date":
            out[f"date:{name}:start"] = None if value is None else iso(value)
            if value is not None:
                out[f"date:{name}:is_datetime"] = 1
        elif kind == "checkbox":
            out[name] = "__YES__" if value else "__NO__"
        else:
            out[name] = value
    return out


def _select_is(name: str, value: str) -> dict[str, Any]:
    return {"property": name, "select": {"equals": value}}


def open_rows_filter(gpu: str) -> dict[str, Any]:
    """The REST filter for the Queued and Running rows of one GPU."""
    return {"and": [_select_is("GPU", gpu), {"or": [_select_is("Status", s) for s in OPEN_STATUSES]}]}


def _mcp_select_is(name: str, value: str) -> dict[str, Any]:
    return {"type": "property", "property": name, "propertyType": "select", "operator": "enum_is",
            "value": {"type": "exact", "value": value}}


def open_rows_mcp_filter(gpu: str) -> dict[str, Any]:
    """The same filter as the rows-mode query of the Notion MCP tool takes it."""
    return {"type": "group", "operator": "and", "filters": [
        _mcp_select_is("GPU", gpu),
        {"type": "group", "operator": "or", "filters": [_mcp_select_is("Status", s) for s in OPEN_STATUSES]},
    ]}


def board_filter(gpu: str) -> dict[str, Any]:
    """The REST filter for the rows of one GPU that tell where a Queued row stands: the Queued and Running rows
    and the Done and Ended by referee rows (BOARD_STATUSES)."""
    return {"and": [_select_is("GPU", gpu), {"or": [_select_is("Status", s) for s in BOARD_STATUSES]}]}


def board_mcp_filter(gpu: str) -> dict[str, Any]:
    """The same filter as the rows-mode query of the Notion MCP tool takes it."""
    return {"type": "group", "operator": "and", "filters": [
        _mcp_select_is("GPU", gpu),
        {"type": "group", "operator": "or", "filters": [_mcp_select_is("Status", s) for s in BOARD_STATUSES]},
    ]}


# --- what a call says -----------------------------------------------------------------

def summary(slot: Slot, now: dt.datetime) -> dict[str, Any]:
    """A row as an answer shows it."""
    out: dict[str, Any] = {
        "row": slot.id, "request": slot.request, "task": slot.task_id, "gpu": slot.gpu,
        "minutes": tidy(slot.minutes_or_default), "status": slot.status, "agent": slot.agent,
    }
    if slot.requested_at:
        out["requested"] = iso(slot.requested_at)
        if slot.status == QUEUED:
            out["waited_min"] = tidy(minutes_between(now, slot.requested_at))
    if slot.started:
        out["started"] = iso(slot.started)
        out["cap_min"] = tidy(cap_minutes(slot))
        if slot.status == RUNNING:
            end = ends_at(slot)
            out["ran_min"] = tidy(minutes_between(now, slot.started))
            out["ends_at"] = iso(end)
            out["remaining_min"] = tidy(minutes_between(end, now))
            out["overrun"] = is_overrun(slot, now)
        elif slot.ended:
            out["ran_min"] = tidy(minutes_between(slot.ended, slot.started))
    if slot.extended:
        out["extended"] = True
    if slot.complaint:
        out["complained"] = True
    return out


def refusal(verb: str, outcome: str, reason: str, **more: Any) -> dict[str, Any]:
    return {"ok": False, "outcome": outcome, "verb": verb, "reason": reason, **more}


def same_agent(row_agent: str, agent: str) -> bool:
    """A row with no Agent belongs to nobody in particular; otherwise the names must match, ignoring case."""
    return not row_agent.strip() or row_agent.strip().lower() == agent.strip().lower()


def request_title(task: str, what: str, minutes: int, why: str | None) -> str:
    """The Request column: "<task>: <what>", and for a long slot "(60 min: <why>)" at the end."""
    text = what[len(task) + 1:].strip() if what.upper().startswith(f"{task}:") else what
    title = f"{task}: {text}"
    if why:
        title += f" ({minutes} min: {why})"
    return title


def add_entry(existing: str, entry: str) -> str:
    """A text column with entry added after what it already holds, oldest entries dropped to fit."""
    parts = [part for part in existing.split(COMPLAINT_JOIN) if part] + [entry]
    while len(parts) > 1 and len(COMPLAINT_JOIN.join(parts)) > MAX_COMPLAINT_CHARS:
        parts.pop(0)
    return COMPLAINT_JOIN.join(parts)[-MAX_COMPLAINT_CHARS:]


def add_complaint(existing: str, entry: str) -> str:
    """The Complaint column with entry added after what it already holds, oldest entries dropped to fit."""
    return add_entry(existing, entry)


def justification_missing(minutes: int, why: str | None, measured: str | None) -> list[str]:
    """The options a request for this many minutes lacks: none up to 30 minutes, else --why and --measured."""
    if minutes <= LONG_ABOVE_MINUTES:
        return []
    return [flag for flag, value in (("--why", why), ("--measured", measured)) if not (value or "").strip()]


def justification_refusal(minutes: int, why: str | None, measured: str | None) -> dict[str, Any] | None:
    """The refusal of a request that lacks its justification, else None."""
    lacking = justification_missing(minutes, why, measured)
    if not lacking:
        return None
    return refusal("slot request", "justification_required",
                   f"a request for more than {LONG_ABOVE_MINUTES} minutes carries a justification: what runs (--what), "
                   f"why it cannot be split into {DEFAULT_MINUTES}-minute slots (--why) and what was already measured "
                   f"(--measured); {' and '.join(lacking)} {'is' if len(lacking) == 1 else 'are'} missing",
                   missing=lacking, minutes=minutes,
                   next=f"ask again with {' and '.join(lacking)}, or ask for {DEFAULT_MINUTES} minutes")


def placement_check(gpu: str, what: str, kind: str | None = None) -> dict[str, Any]:
    """What the placement plan (lab/placement.py) says about a request for that card with those words: verdict
    consistent, contradicts, no_line or unavailable, with the kind, the reason and, for a contradiction, the next
    step. Never raises: a plan that cannot be built answers unavailable."""
    from lab import placement   # imported here: lab/placement.py reads the queue through this module
    try:
        return placement.check_request(gpu=gpu, text=what, kind=kind)
    except Exception as exc:    # the queue keeps working whatever happens to the plan
        return {"contradicts": False, "verdict": "unavailable", "kind": kind,
                "reason": f"no placement plan: {type(exc).__name__}: {exc}"}


def placement_view(verdict: Mapping[str, Any], override: str | None) -> dict[str, Any]:
    """The placement part of an answer: the verdict, the kind and how it was found, the reason, and the override
    when the request went ahead against the plan."""
    out = {key: verdict[key] for key in ("verdict", "kind", "kind_from", "node", "reason", "next") if key in verdict}
    plan = verdict.get("plan")
    if isinstance(plan, Mapping):
        out["plan_pods"] = list(plan.get("pods") or [])
    if verdict.get("contradicts") and override:
        out["verdict"] = "overridden"
        out["override"] = override
    return out


def placement_refusal(verdict: Mapping[str, Any], override: str | None) -> dict[str, Any] | None:
    """The refusal of a request that contradicts the placement plan without an --override, else None."""
    if not verdict.get("contradicts") or override:
        return None
    return refusal("slot request", "placement_contradiction", verdict.get("reason", "it contradicts the placement plan"),
                   placement=placement_view(verdict, None), placement_verdict="contradicts",
                   next=verdict.get("next") or 'ask for the card the plan names, or add --override "why"')


def override_text(verdict: Mapping[str, Any], override: str) -> str:
    """The Override line's text: what the plan says, in short, and why the request runs anyway."""
    plan = verdict.get("plan") if isinstance(verdict.get("plan"), Mapping) else {}
    pods = plan.get("pods") or []
    where = " or ".join(pods) if pods else "no pod now"
    return f"against the placement plan ({verdict.get('kind')} on {verdict.get('node')}; the plan says {where}): {override}"


def next_step(slot: Slot, may_start: bool) -> str:
    """What the agent of a Queued row does next, from where the row stands."""
    row = slot.id
    state = justification_state(slot)
    shorten = f"python -m lab slot start --row {row} --minutes {DEFAULT_MINUTES}"
    if state == AWAITING_CRITIQUE:
        return (f"the referee critiques a request for more than {LONG_ABOVE_MINUTES} minutes before it can start: read "
                f"the critique with python -m lab slot status --row {row}, then answer it with python -m lab slot "
                f"start --row {row} --reply \"...\", or shorten the request with {shorten}")
    if state == AWAITING_REPLY:
        return (f"answer the referee's critique with python -m lab slot start --row {row} --reply \"what you will "
                f"change, or why the critique does not apply\", or shorten the request with {shorten}")
    if state == MISSING:
        return (f"the row holds no justification, so it cannot start: shorten it with {shorten}, or cancel it with "
                f"python -m lab slot done --row {row} --cancel and ask again with --why and --measured")
    if may_start:
        return f"python -m lab slot start --row {row}"
    patience = tidy(COMPLAIN_AFTER_FACTOR * slot.minutes_or_default)
    return (f"wait for your turn (python -m lab slot status --row {row}); after {patience} minutes of waiting, "
            f"python -m lab slot complain --row {row}")


# --- the six verbs --------------------------------------------------------------------

class SlotQueue:
    """The GPU queue rules over a NotionClient. Each verb method answers with the dict the CLI prints and raises
    NotionError when Notion cannot be reached or refuses. now is a function returning the current aware time."""

    def __init__(self, client: notion.NotionClient, *, data_source: str = DATA_SOURCE_ID,
                 now: Callable[[], dt.datetime] = utc_now) -> None:
        self.client = client
        plain = notion.page_id_of(data_source)
        self.data_source = notion.dashed_id(plain) if plain else data_source
        self._now = now

    # reading

    def open_rows(self, gpu: str) -> list[Slot]:
        """The Queued and Running rows of a GPU (the column's value, e.g. "B exclusive")."""
        pages = self.client.query_data_source(self.data_source, filter=open_rows_filter(gpu))
        return [slot for slot in (slot_from_page(page) for page in pages) if slot.status in OPEN_STATUSES]

    def board(self, gpu: str) -> list[Slot]:
        """The rows of a GPU that tell where a Queued row stands: the open ones, and the Done and Ended by referee
        rows, whose starts are what passed a waiting row over."""
        pages = self.client.query_data_source(self.data_source, filter=board_filter(gpu))
        return [slot for slot in (slot_from_page(page) for page in pages) if slot.status in BOARD_STATUSES]

    def load(self, row_id: str, agent: str, verb: str, *,
             own: bool = True) -> tuple[Slot | None, dict[str, Any] | None]:
        """(the row, None), or (None, the refusal to answer with) when it is not a row of this queue or, with own, not
        the agent's: another agent's row is only ever read."""
        try:
            slot = slot_from_page(self.client.retrieve_page(row_id))
        except notion.NotionError as exc:
            if exc.kind == "not_found":
                return None, refusal(verb, "row_not_found", f"no page {row_id} that this Notion token can see",
                                     row=row_id)
            raise
        if slot.data_source is not None and slot.data_source != notion.page_id_of(self.data_source):
            return None, refusal(verb, "not_a_queue_row", "that page is not a row of the GPU Queue", row=slot.id)
        if own and not same_agent(slot.agent, agent):
            return None, refusal(verb, "not_your_row", f"the row belongs to {slot.agent!r}, not {agent!r}; another "
                                 "agent's row is only ever read", row=slot.id, slot=summary(slot, self._now()))
        return slot, None

    def _where(self, slot: Slot, rows: Sequence[Slot], now: dt.datetime) -> dict[str, Any]:
        """Where a Queued row stands: its place, who is ahead, what is Running, whether it may start, and what a row
        outside the line waits for."""
        running = [summary(row, now) for row in running_on(rows, slot.gpu, other_than=slot.id)]
        state = justification_state(slot)
        if state not in STARTABLE:
            return {"position": None, "may_start": False, "ahead": [], "running": running,
                    "waiting_for": WAITING_FOR[state]}
        ahead = ahead_of(slot, rows, now)
        where: dict[str, Any] = {"position": len(ahead) + 1, "may_start": not ahead and not running,
                                 "ahead": [summary(row, now) for row in ahead], "running": running}
        hungry = starvation(slot, rows, now)
        if hungry is not None and jumped(slot, rows, now):
            where["starved"] = hungry.why
        return where

    # request

    def request(self, *, task: str, what: str, minutes: int = DEFAULT_MINUTES, why: str | None = None,
                measured: str | None = None, gpu: str = DEFAULT_GPU, agent: str = DEFAULT_AGENT,
                actor: str = record.UNKNOWN_ACTOR, kind: str | None = None, override: str | None = None,
                placement: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Add a row. placement is the placement plan's verdict on the request when the caller already has it;
        otherwise it is asked here (placement_check), so the plan holds for every caller."""
        verb, gpu_name, now = "slot request", GPUS[gpu], self._now()
        refused = justification_refusal(minutes, why, measured)
        if refused is not None:
            return refused
        verdict = dict(placement) if placement is not None else placement_check(gpu, what, kind)
        refused = placement_refusal(verdict, override)
        if refused is not None:
            return refused
        title = request_title(task, what, minutes, why)
        rows = self.board(gpu_name)
        twin = next((r for r in rows if r.status == QUEUED and r.request == title and same_agent(r.agent, agent)), None)
        outcome = "already_queued"
        if twin is None:
            change = {"Request": title, "Task ID": task, "GPU": gpu_name, "Minutes": minutes, "Status": QUEUED,
                      "Agent": agent, "Requested": now}
            if minutes > LONG_ABOVE_MINUTES:
                change["Progress note"] = note_entry(JUSTIFICATION, actor, now, justification_text(why, measured))
            if verdict.get("contradicts") and override:
                change["Progress note"] = add_entry(change.get("Progress note", ""), note_entry(
                    OVERRIDE, actor, now, override_text(verdict, override)))
            twin = slot_from_page(self.client.create_page(self.data_source, rest_properties(change)))
            rows = [row for row in rows if row.id != twin.id] + [twin]
            outcome = "queued"
        answer: dict[str, Any] = {"ok": True, "outcome": outcome, "verb": verb, "row": twin.id, "url": twin.url,
                                  "slot": summary(twin, now), **self._where(twin, rows, now)}
        view = justification_view(twin)
        if view is not None:
            answer["justification"] = view
            answer["justification_state"] = view["state"]
        answer["next"] = next_step(twin, answer["may_start"])
        if gpu == "A":
            answer["warning"] = ("pod A is the shared teacher service: queue it only to restart or change the "
                                 "server itself")
        answer["placement"] = placement_view(verdict, override)
        answer["placement_verdict"] = answer["placement"]["verdict"]
        if answer["placement_verdict"] == "overridden":
            answer["override"] = override
        return answer

    # start

    def _amend(self, slot: Slot, reply: str | None, minutes: int | None, actor: str,
               now: dt.datetime) -> tuple[Slot, dict[str, Any], dict[str, Any] | None]:
        """Write the reply to the critique and the shortened Minutes of a Queued row in one update: (the row as it
        stands, what the answer says was done, or the refusal). A reply needs a critique that waits for one."""
        verb = "slot start"
        if reply is not None and justification_state(slot) != AWAITING_REPLY:
            return slot, {}, refusal(verb, "nothing_to_reply_to", "there is no critique on this row waiting for a "
                                     "reply: the referee writes it after the justification, and one reply answers it",
                                     row=slot.id, slot=summary(slot, now), justification=justification_view(slot))
        if minutes is not None and minutes >= slot.minutes_or_default:
            return slot, {}, refusal(verb, "not_shorter", f"--minutes only shortens a row: it asks for "
                                     f"{tidy(slot.minutes_or_default)} minutes already", row=slot.id,
                                     slot=summary(slot, now))
        change: dict[str, Any] = {}
        said: dict[str, Any] = {}
        note = slot.progress_note
        if minutes is not None:
            title = LONG_ENDING.sub("", slot.request)
            if title != slot.request:
                change["Request"] = title
            change["Minutes"] = minutes
            note = add_entry(note, note_entry(SHORTENED, actor, now,
                                              f"from {tidy(slot.minutes_or_default)} to {minutes} minutes"))
            said["shortened_to"] = minutes
        if reply is not None:
            note = add_entry(note, note_entry(REPLY, actor, now, reply))
            said["replied"] = True
        change["Progress note"] = note
        return slot_from_page(self.client.update_page(slot.id, rest_properties(change))), said, None

    def _not_next(self, slot: Slot, ahead: Sequence[Slot], rows: Sequence[Slot], now: dt.datetime,
                  said: Mapping[str, Any]) -> dict[str, Any]:
        """The refusal of a start behind the rows that go first. When one of them goes first only because it is
        starved, the answer names that row and the rule that put it there."""
        reason = f"{len(ahead)} Queued row(s) go before this one (shortest first, then earliest request)"
        guard: dict[str, Any] = {}
        for row in ahead:
            found = starvation(row, rows, now)
            if found is not None and queue_key(row) > queue_key(slot):
                reason = (f"{len(ahead)} Queued row(s) go before this one; {row.request!r} {found.why} and goes "
                          "next whatever its length")
                guard = {"guard": "starved_ahead", "starved_row": row.id, "passed_over": found.passed_over}
                break
        return refusal("slot start", "not_next", reason, row=slot.id, ahead=[summary(row, now) for row in ahead],
                       slot=summary(slot, now), **guard, **said)

    @staticmethod
    def _guard_note(slot: Slot, rows: Sequence[Slot], now: dt.datetime) -> dict[str, Any]:
        """What a start says when the starvation guard put the row first: the rule, how often the row was passed
        over and how many rows it went ahead of. Nothing when its length alone made it first."""
        hungry = starvation(slot, rows, now)
        passed = jumped(slot, rows, now)
        if hungry is None or not passed:
            return {}
        return {"guard": hungry.code, "starved": hungry.why, "passed_over": hungry.passed_over,
                "went_ahead_of": len(passed)}

    def start(self, row_id: str, *, agent: str = DEFAULT_AGENT, reply: str | None = None, minutes: int | None = None,
              actor: str = record.UNKNOWN_ACTOR) -> dict[str, Any]:
        verb = "slot start"
        slot, bad = self.load(row_id, agent, verb)
        if bad:
            return bad
        now = self._now()
        if slot.status != QUEUED:
            return refusal(verb, "not_queued", f"the row is {slot.status}, not Queued", row=slot.id,
                           slot=summary(slot, now))
        if not slot.gpu:
            return refusal(verb, "no_gpu", "the row has no GPU set, so it is in no queue", row=slot.id,
                           slot=summary(slot, now))
        said: dict[str, Any] = {}
        if reply is not None or minutes is not None:
            slot, said, bad = self._amend(slot, reply, minutes, actor, now)
            if bad:
                return bad
        state = justification_state(slot)
        if state not in STARTABLE:
            return refusal(verb, GATE_OUTCOMES[state], f"the row cannot start yet: it waits for {WAITING_FOR[state]}",
                           row=slot.id, slot=summary(slot, now), justification=justification_view(slot),
                           justification_state=state, next=next_step(slot, False), **said)
        rows = [row for row in self.board(slot.gpu) if row.id != slot.id] + [slot]
        running = running_on(rows, slot.gpu, other_than=slot.id)
        if running:
            stale = [row for row in running if is_overrun(row, now)]
            return refusal(verb, "busy", "a row is Running on this GPU; never start while one is Running, even if it "
                           "looks stale - complain instead" + (" (it is past its cap)" if stale else ""),
                           row=slot.id, running=[summary(row, now) for row in running], slot=summary(slot, now),
                           **said)
        ahead = ahead_of(slot, rows, now)
        if ahead:
            return self._not_next(slot, ahead, rows, now, said)
        guard = self._guard_note(slot, rows, now)
        idle = passed_idle(slot, rows, now)
        if idle:
            guard["passed_idle"] = [summary(row, now) for row in idle]
        view = justification_view(slot)
        if view is not None:
            said["justification_state"] = view["state"]
        began = self._now()
        mine = slot_from_page(self.client.update_page(slot.id, rest_properties({"Status": RUNNING, "Started": began})))
        release = f"python -m lab slot done --row {mine.id} --result \"<one line with the numbers>\""
        try:
            rivals = running_on(self.open_rows(slot.gpu), slot.gpu, other_than=slot.id)
        except notion.NotionError as exc:
            log.warning("slot start: row %s is Running, but the queue could not be read again: %s", slot.id, exc)
            return {"ok": True, "outcome": "started", "verb": verb, "row": mine.id, "slot": summary(mine, began),
                    "release": release, **guard, **said,
                    "warning": f"the queue could not be read again to confirm that no other row started in the same "
                               f"moment ({exc}); run python -m lab slot status and release the card if one did"}
        if rivals:
            answer = refusal(verb, "lost_race", "another row is Running on this GPU after this row's update (two agents "
                             "started in the same moment); this row is back to Queued - try again in a moment",
                             row=slot.id, running=[summary(row, began) for row in rivals], **said)
            try:
                self.client.update_page(slot.id, rest_properties({"Status": QUEUED, "Started": None}))
            except notion.NotionError as exc:
                answer.update(outcome="error_lost_race", revert_error=str(exc),
                              reason="another row is Running on this GPU and this row could not be put back to Queued: "
                              "set its Status to Queued and clear Started by hand")
            return answer
        return {"ok": True, "outcome": "started", "verb": verb, "row": mine.id, "slot": summary(mine, began),
                "release": release, **guard, **said}

    # extend

    def extend(self, row_id: str, note: str, *, agent: str = DEFAULT_AGENT) -> dict[str, Any]:
        verb = "slot extend"
        slot, bad = self.load(row_id, agent, verb)
        if bad:
            return bad
        now = self._now()
        if slot.status != RUNNING:
            return refusal(verb, "not_running", f"only a Running row can be extended; this one is {slot.status}",
                           row=slot.id, slot=summary(slot, now))
        if slot.extended:
            return refusal(verb, "already_extended", f"one extension of {EXTENSION_MINUTES} minutes is allowed; "
                           "finish with done", row=slot.id, slot=summary(slot, now))
        end = ends_at(slot)
        if end is not None and now > end:
            return refusal(verb, "past_cap", f"the slot reached its cap at {iso(end)}; an extension asked for after "
                           "the cap is not one - finish with done", row=slot.id, slot=summary(slot, now))
        text = add_entry(slot.progress_note, note) if slot.progress_note else note
        updated = slot_from_page(self.client.update_page(
            slot.id, rest_properties({"Extended": True, "Progress note": text})))
        return {"ok": True, "outcome": "extended", "verb": verb, "row": updated.id, "slot": summary(updated, now)}

    # done

    def done(self, row_id: str, result: str | None, *, cancel: bool = False,
             agent: str = DEFAULT_AGENT) -> dict[str, Any]:
        verb = "slot done"
        slot, bad = self.load(row_id, agent, verb)
        if bad:
            return bad
        now = self._now()
        if cancel:
            if slot.status != QUEUED:
                return refusal(verb, "not_queued", f"only a Queued row can be cancelled; this one is {slot.status} "
                               "(a Running row is finished with done and a result)", row=slot.id,
                               slot=summary(slot, now))
            change: dict[str, Any] = {"Status": CANCELLED, "Ended": now}
            if result:
                change["Result"] = result
            updated = slot_from_page(self.client.update_page(slot.id, rest_properties(change)))
            return {"ok": True, "outcome": "cancelled", "verb": verb, "row": updated.id,
                    "slot": summary(updated, now), "card_released": True}
        if not result:
            return refusal(verb, "result_required", "finish with a Result: one line with the numbers the slot proved",
                           row=slot.id, slot=summary(slot, now))
        if slot.status != RUNNING:
            return refusal(verb, "not_running", f"only a Running row is finished with done; this one is {slot.status}"
                           + (" (done --cancel withdraws a Queued row)" if slot.status == QUEUED else ""),
                           row=slot.id, slot=summary(slot, now))
        updated = slot_from_page(self.client.update_page(
            slot.id, rest_properties({"Status": DONE, "Ended": now, "Result": result})))
        ran = minutes_between(now, slot.started) if slot.started else None
        answer = {"ok": True, "outcome": "done", "verb": verb, "row": updated.id, "slot": summary(updated, now),
                  "card_released": True}
        if ran is not None:
            answer["ran_min"] = tidy(ran)
            answer["overrun_min"] = tidy(max(0.0, ran - cap_minutes(slot)))
        return answer

    # complain

    def complain(self, row_id: str, text: str | None = None, *, agent: str = DEFAULT_AGENT) -> dict[str, Any]:
        verb = "slot complain"
        slot, bad = self.load(row_id, agent, verb)
        if bad:
            return bad
        now = self._now()
        if slot.status != QUEUED:
            return refusal(verb, "not_waiting", f"only a Queued row waits; this one is {slot.status}", row=slot.id,
                           slot=summary(slot, now))
        if not slot.gpu:
            return refusal(verb, "no_gpu", "the row has no GPU set, so it is in no queue", row=slot.id,
                           slot=summary(slot, now))
        rows = self.board(slot.gpu)
        due, why, earliest = complaint_due(slot, rows, now)
        if not due:
            return refusal(verb, "too_early", "a complaint comes after waiting more than twice the minutes asked for, "
                           "or when the Running row is past its cap", row=slot.id,
                           complain_after_min=tidy(COMPLAIN_AFTER_FACTOR * slot.minutes_or_default),
                           earliest=iso(earliest) if earliest else None, slot=summary(slot, now))
        where = self._where(slot, rows, now)
        behind = [f"Running {row['request']!r} (ends {row.get('ends_at', 'unknown')})" for row in where["running"]]
        if where["ahead"]:
            behind.append(f"{len(where['ahead'])} Queued row(s) ahead")
        if "waiting_for" in where:
            behind.append(f"it waits for {where['waiting_for']}")
        entry = (f"{iso(now)[:16]}Z {why}; waiting behind: {'; '.join(behind) or 'nothing found'}"
                 + (f". {text}" if text else ""))
        updated = slot_from_page(self.client.update_page(
            slot.id, rest_properties({"Complaint": add_complaint(slot.complaint, entry)})))
        return {"ok": True, "outcome": "complaint_recorded", "verb": verb, "row": updated.id, "reason": why,
                "note": entry[:record.NOTE_CHARS], "slot": summary(updated, now), **where}

    # status

    def status(self, *, gpu: str = DEFAULT_GPU, row_id: str | None = None, task: str | None = None,
               agent: str = DEFAULT_AGENT) -> dict[str, Any]:
        verb, gpu_name, now = "slot status", GPUS[gpu], self._now()
        rows = self.board(gpu_name)
        if row_id is not None:
            slot, bad = self.load(row_id, agent, verb, own=False)
            if bad:
                return bad
            answer: dict[str, Any] = {"ok": True, "outcome": "status", "verb": verb, "row": slot.id,
                                      "slot": summary(slot, now)}
            if slot.status == QUEUED:
                answer.update(self._where(slot, rows, now))
                view = justification_view(slot)
                if view is not None:
                    answer["justification"] = view
            return answer
        running = running_on(rows, gpu_name)
        line = queue_order(rows, now)
        in_line = {row.id for row in line}
        held = sorted((row for row in rows if row.status == QUEUED and row.id not in in_line), key=queue_key)
        free = not running
        listed = []
        for index, row in enumerate(line, start=1):
            if task is None or row.task_id == task:
                entry = {**summary(row, now), "position": index, "may_start": free and index == 1}
                hungry = starvation(row, rows, now)
                if hungry is not None and jumped(row, rows, now):
                    entry["starved"] = hungry.why
                listed.append(entry)
        blocked = [{**summary(row, now), "waiting_for": WAITING_FOR[justification_state(row)],
                    "justification_state": justification_state(row)}
                   for row in held if task is None or row.task_id == task]
        return {"ok": True, "outcome": "status", "verb": verb, "gpu": gpu_name,
                "running": [summary(row, now) for row in running if task is None or row.task_id == task],
                "queued": listed, "blocked": blocked, "card_free": free, "next_row": line[0].id if line else None}


# --- no token: what a sitting writes by hand --------------------------------------------

def _tool(name: str) -> str:
    return f"mcp__{NOTION_SERVER}__notion-{name}"


def _query_step(gpu_name: str, data_source: str, why: str) -> dict[str, Any]:
    return {"tool": _tool("query-data-sources"), "why": why, "arguments": {"data": {
        "mode": "rows", "data_source_url": f"collection://{data_source}",
        "filter": board_mcp_filter(gpu_name), "limit": 100}}}


def _update_step(row: str, change: Mapping[str, Any], why: str) -> dict[str, Any]:
    return {"tool": _tool("update-page"), "why": why, "arguments": {
        "page_id": row, "command": "update_properties", "properties": mcp_properties(change)}}


def _fetch_step(row: str, why: str) -> dict[str, Any]:
    return {"tool": _tool("fetch"), "why": why, "arguments": {"id": row}}


RULE_START = ("Start only when your row's justification is in order, no row on the GPU is Running and your row is "
              "first in line. A row asking for more than 30 minutes is in order when its Progress note holds a "
              "Justification line, a Critique line after it and a Reply line after that, or when its Minutes have "
              "been shortened to 30 or less; a row that is not in order is left out of the line. The line is: first "
              "every Queued row that has been passed over three times (three rows requested after it have started "
              "since it was requested, whether they Run, are Done or were Ended by referee) or has waited more than "
              "twice its Minutes, the earliest request first; then the rest, fewest Minutes first and, for equal "
              "Minutes, the earlier request first. Never start while one is Running, even if it looks stale; "
              "complain instead. After the update query again: if any other row on the GPU is Running, set yours "
              "back to Queued, clear Started and try again in a moment.")
RULE_JUSTIFY = ("A request for more than 30 minutes carries a justification, written as the first line of the "
                "Progress note: 'Justification (<who>, <time>Z): cannot be split because: <why it cannot be split "
                "into 15-minute slots>; already measured: <what was measured>'. Without both parts do not add the "
                "row. The referee then adds one 'Critique' line and you add one 'Reply' line, or you shorten the "
                "row to 15 minutes; until then the row cannot start.")
RULE_EXTEND = ("One extension of 15 minutes, only on your Running row, only before Started plus Minutes has "
               "passed, only when Extended is unchecked, and only with a Progress note saying what the slot has "
               "already proven; keep any text the Progress note already holds and add yours after ' | '.")
RULE_COMPLAIN = ("Only on your own Queued row, and only after it has waited more than twice its Minutes (or the "
                 "Running row is past Started plus its Minutes, plus 15 if Extended is checked). Say what you wait "
                 "for and for how long; keep any text the Complaint already holds and add yours after ' | '.")
RULE_DONE = ("Only on your Running row. Result is one line with the numbers the slot proved. "
             "Then stop using the card at once.")


def amendment(args: argparse.Namespace, now: dt.datetime, actor: str) -> dict[str, Any]:
    """The change a start with --reply or --minutes makes before the start, with <> where the row's own text goes."""
    change: dict[str, Any] = {}
    note = "<the Progress note as it is>"
    if args.minutes is not None:
        change["Minutes"] = args.minutes
        note += COMPLAINT_JOIN + note_entry(SHORTENED, actor, now, f"from <Minutes> to {args.minutes} minutes")
    if args.reply is not None:
        note += COMPLAINT_JOIN + note_entry(REPLY, actor, now, args.reply)
    if change or args.reply is not None:
        change["Progress note"] = note
    return change


def offline_plan(args: argparse.Namespace, now: dt.datetime, data_source: str,
                 actor: str = record.UNKNOWN_ACTOR) -> dict[str, Any]:
    """What a sitting does with its Notion tools when no token lets the verb do it: the exact row or properties
    to write and the checks to make, as an answer with outcome no_token."""
    sub = args.sub
    verb = f"slot {sub}"
    plan: dict[str, Any] = {"ok": False, "outcome": "no_token", "verb": verb,
                            "error": f"{ENV_TOKEN} is not set (process environment or a .env file), so nothing was "
                                     "sent to Notion; make the calls below with your Notion tools instead"}
    gpu_name = GPUS[getattr(args, "gpu", DEFAULT_GPU)]
    if sub == "request":
        title = request_title(args.task, args.what, args.minutes, args.why)
        change = {"Request": title, "Task ID": args.task, "GPU": gpu_name, "Minutes": args.minutes, "Status": QUEUED,
                  "Agent": args.agent, "Requested": now}
        long = args.minutes > LONG_ABOVE_MINUTES
        if long:
            change["Progress note"] = note_entry(JUSTIFICATION, actor, now, justification_text(args.why, args.measured))
        verdict = getattr(args, "placement", None) or {}
        if verdict.get("contradicts") and args.override:
            change["Progress note"] = add_entry(change.get("Progress note", ""), note_entry(
                OVERRIDE, actor, now, override_text(verdict, args.override)))
        if verdict:
            plan["placement"] = placement_view(verdict, args.override)
            plan["placement_verdict"] = plan["placement"]["verdict"]
        plan["write"] = mcp_properties(change)
        plan["steps"] = [
            {"tool": _tool("create-pages"), "why": "add your row to the GPU Queue", "arguments": {
                "parent": {"type": "data_source_id", "data_source_id": data_source},
                "pages": [{"properties": mcp_properties(change)}]}},
            _query_step(gpu_name, data_source, "list the Queued, Running, Done and Ended by referee rows of the GPU, "
                        "then apply the start rule"),
        ]
        plan["rule"] = (RULE_JUSTIFY + " " if long else "") + RULE_START
    elif sub == "start":
        change = {"Status": RUNNING, "Started": now}
        plan["write"] = mcp_properties(change)
        steps = [_fetch_step(args.row, "read your row: Status must be Queued; note its GPU, Minutes and Progress note")]
        amend = amendment(args, now, actor)
        if amend:
            plan["amend"] = mcp_properties(amend)
            steps.append(_update_step(args.row, amend, "write the reply to the critique and the shorter Minutes, "
                                      "with the Progress note as the row holds it (a reply needs a Critique line "
                                      "without a Reply line after it; when you shorten a row, drop the ' (N min: "
                                      "...)' ending of its Request)"))
        steps += [
            _query_step(gpu_name, data_source, "list the Queued, Running, Done and Ended by referee rows of your "
                        f"row's GPU (this filter is {gpu_name!r}; use the GPU your row names if it differs), then "
                        "apply the start rule"),
            _update_step(args.row, change, "mark your row Running, Status and Started in one update"),
            _query_step(gpu_name, data_source, "check that no other row is Running; if one is, undo your update"),
        ]
        plan["steps"] = steps
        plan["rule"] = RULE_START
    elif sub == "extend":
        change = {"Extended": True, "Progress note": args.note}
        plan["write"] = mcp_properties(change)
        plan["steps"] = [
            _fetch_step(args.row, "read your row: it must be Running, Extended unchecked, and inside its cap"),
            _update_step(args.row, change, "check Extended and write the Progress note"),
        ]
        plan["rule"] = RULE_EXTEND
    elif sub == "done":
        change = ({"Status": CANCELLED, "Ended": now, **({"Result": args.result} if args.result else {})}
                  if args.cancel else {"Status": DONE, "Ended": now, "Result": args.result})
        plan["write"] = mcp_properties(change)
        plan["steps"] = [_update_step(args.row, change, "withdraw your Queued row" if args.cancel
                                      else "finish the slot and release the card")]
        plan["rule"] = ("Only on your Queued row; a Running row is finished with a result instead." if args.cancel
                        else RULE_DONE)
    elif sub == "complain":
        change = {"Complaint": f"{iso(now)[:16]}Z {args.text or 'waiting longer than twice the minutes asked for'}"}
        plan["write"] = mcp_properties(change)
        plan["steps"] = [
            _fetch_step(args.row, "read your row: it must be Queued and have waited long enough; note its Complaint"),
            _update_step(args.row, change, "write the Complaint (add it after any text already there)"),
        ]
        plan["rule"] = RULE_COMPLAIN
    else:
        plan["steps"] = [_query_step(gpu_name, data_source, "the Queued, Running, Done and Ended by referee rows of "
                                     "the GPU, in no order; apply the start rule to see the line")]
        plan["rule"] = "Read only: nothing is written."
    return plan


# --- the command line ------------------------------------------------------------------

def _text_type(limit: int, what: str) -> Callable[[str], str]:
    def parse(value: str) -> str:
        flat = " ".join(value.split())
        if not flat:
            raise argparse.ArgumentTypeError(f"{what} must not be empty")
        if len(flat) > limit:
            raise argparse.ArgumentTypeError(f"{what} is {len(flat)} characters; the limit is {limit}")
        return flat
    return parse


def _task_type(value: str) -> str:
    if not TASK_ID.fullmatch(value.strip()):
        raise argparse.ArgumentTypeError(f"{value!r} is not a Task Board ID such as L3 or T2")
    return value.strip().upper()


def _row_type(value: str) -> str:
    found = notion.page_id_of(value)
    if found is None:
        raise argparse.ArgumentTypeError(f"{value!r} holds no Notion page id; give the id or the URL of the row")
    return found


def _gpu_type(value: str) -> str:
    letter = value.strip().upper()[:1]
    if letter not in GPUS:
        raise argparse.ArgumentTypeError(f"{value!r} is not a GPU; use A or B")
    return letter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m lab slot", description=main.__doc__,
                                     allow_abbrev=False)
    subs = parser.add_subparsers(dest="sub", required=True, metavar="{request,start,extend,done,complain,status}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--agent", default=DEFAULT_AGENT, metavar="NAME",
                        help=f"whose agent this is (default {DEFAULT_AGENT}); only that agent's rows are touched")

    def sub(name: str, help_: str) -> argparse.ArgumentParser:
        return subs.add_parser(name, help=help_, description=help_, parents=[common], allow_abbrev=False)

    request = sub("request", "add a row to the queue")
    request.add_argument("--task", required=True, type=_task_type, metavar="ID", help="the Task Board ID, e.g. L3")
    request.add_argument("--what", required=True, type=_text_type(MAX_WHAT_CHARS, "--what"), metavar="TEXT",
                         help="one line of what runs in the slot")
    request.add_argument("--minutes", type=int, choices=SLOT_MINUTES, default=DEFAULT_MINUTES,
                         help=f"{DEFAULT_MINUTES}, or {LONG_MINUTES} for a training run or a declared long job "
                              "(then --why and --measured are required)")
    request.add_argument("--why", type=_text_type(MAX_WHY_CHARS, "--why"), metavar="TEXT",
                         help=f"for a {LONG_MINUTES}-minute slot: why it cannot be split into {DEFAULT_MINUTES}-minute "
                              "slots; written into the Request and the justification")
    request.add_argument("--measured", type=_text_type(MAX_MEASURED_CHARS, "--measured"), metavar="TEXT",
                         help=f"for a {LONG_MINUTES}-minute slot: what was already measured (a run, a rate, a size)")
    request.add_argument("--gpu", type=_gpu_type, default=DEFAULT_GPU, metavar="{A,B}",
                         help="B (default) is the exclusive card; A is the shared teacher service")
    request.add_argument("--kind", choices=PLACEMENT_KINDS,
                         help="the kind of job, for the placement plan's check; default: read from --what's words")
    request.add_argument("--override", type=_text_type(MAX_OVERRIDE_CHARS, "--override"), metavar="TEXT",
                         help="why the request must run on this card although the placement plan puts its kind "
                              "elsewhere (written to the Progress note)")

    for name, help_ in (("start", "start the row's slot when its turn has come"),
                        ("extend", "extend a Running slot by 15 minutes, once, with a progress note"),
                        ("done", "finish a Running slot and release the card, or cancel a Queued row"),
                        ("complain", "write a complaint on a row that has waited too long"),
                        ("status", "show the queue, or one row")):
        parser_ = sub(name, help_)
        parser_.add_argument("--row", required=name != "status", type=_row_type, metavar="ID_OR_URL",
                             help="the row's page id or its Notion URL")
        if name == "start":
            parser_.add_argument("--reply", type=_text_type(MAX_REPLY_CHARS, "--reply"), metavar="TEXT",
                                 help="your answer to the referee's critique: what you will change, or why it does "
                                      "not apply (written to the row before the start is tried)")
            parser_.add_argument("--minutes", type=int, choices=(DEFAULT_MINUTES,), metavar=str(DEFAULT_MINUTES),
                                 help=f"shorten the row to {DEFAULT_MINUTES} minutes instead of answering a critique")
        elif name == "extend":
            parser_.add_argument("--note", required=True, type=_text_type(MAX_NOTE_CHARS, "--note"), metavar="TEXT",
                                 help="what the slot has already proven")
        elif name == "done":
            parser_.add_argument("--result", type=_text_type(MAX_NOTE_CHARS, "--result"), metavar="TEXT",
                                 help="one line with the numbers the slot proved (required unless --cancel)")
            parser_.add_argument("--cancel", action="store_true", help="withdraw a Queued row that never started")
        elif name == "complain":
            parser_.add_argument("--text", type=_text_type(MAX_NOTE_CHARS, "--text"), metavar="TEXT",
                                 help="more detail, added to the wait and the row it waits behind")
        elif name == "status":
            parser_.add_argument("--gpu", type=_gpu_type, default=DEFAULT_GPU, metavar="{A,B}")
            parser_.add_argument("--task", type=_task_type, metavar="ID", help="only this task's rows")
    return parser


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """The arguments of a slot call; SystemExit(2), argparse's usage message on stderr, for bad ones."""
    parser = build_parser()
    args = parser.parse_args(list(argv))
    if args.sub == "request" and args.minutes <= LONG_ABOVE_MINUTES and (args.why or args.measured):
        parser.error(f"--why and --measured are for a {LONG_MINUTES}-minute slot; a {DEFAULT_MINUTES}-minute slot "
                     "needs no justification")
    if args.sub == "done" and not args.cancel and not args.result:
        parser.error("--result is required: one line with the numbers the slot proved (or --cancel a Queued row)")
    return args


def find_token(environ: Mapping[str, str], env_files: Sequence[Any] | None) -> str | None:
    """NOTION_TOKEN from the environment, then the .env files; None when it is not there or blank."""
    files = default_env_files(environ) if env_files is None else list(env_files)
    values, _ = collect_env([ENV_TOKEN], environ, files)
    token = values.get(ENV_TOKEN, "").strip()
    return token or None


def connect(environ: Mapping[str, str] | None = None,
            env_files: Sequence[Any] | None = None) -> notion.NotionClient | None:
    """The Notion client the token makes (aimed at LAB_NOTION_URL when that is set), None when there is no token."""
    environ = os.environ if environ is None else environ
    token = find_token(environ, env_files)
    if token is None:
        return None
    return notion.NotionClient(token, base_url=(environ.get(ENV_URL) or "").strip() or notion.API_URL)


def data_source_of(environ: Mapping[str, str] | None = None) -> str:
    """The data source the verbs work on: LAB_QUEUE_DATA_SOURCE when set, else the live GPU Queue."""
    environ = os.environ if environ is None else environ
    return (environ.get(ENV_DATA_SOURCE) or "").strip() or DATA_SOURCE_ID


def _dispatch(queue: SlotQueue, args: argparse.Namespace, actor: str) -> dict[str, Any]:
    if args.sub == "request":
        return queue.request(task=args.task, what=args.what, minutes=args.minutes, why=args.why,
                             measured=args.measured, gpu=args.gpu, agent=args.agent, actor=actor, kind=args.kind,
                             override=args.override, placement=getattr(args, "placement", None))
    if args.sub == "start":
        return queue.start(args.row, agent=args.agent, reply=args.reply, minutes=args.minutes, actor=actor)
    if args.sub == "extend":
        return queue.extend(args.row, args.note, agent=args.agent)
    if args.sub == "done":
        return queue.done(args.row, args.result, cancel=args.cancel, agent=args.agent)
    if args.sub == "complain":
        return queue.complain(args.row, args.text, agent=args.agent)
    return queue.status(gpu=args.gpu, row_id=args.row, task=args.task, agent=args.agent)


def _failure(args: argparse.Namespace, exc: notion.NotionError) -> dict[str, Any]:
    hints = {"unauthorized": f"Notion rejected {ENV_TOKEN}; it is wrong or revoked",
             "not_found": "Notion cannot find the GPU Queue; share it with the integration the token belongs to",
             "restricted": "the integration lacks a capability (read, insert or update content)"}
    answer: dict[str, Any] = {"ok": False, "outcome": "error", "verb": f"slot {args.sub}", "kind": exc.kind,
                              "error": str(exc)}
    if exc.kind in hints:
        answer["reason"] = hints[exc.kind]
    return answer


#: Fields of an answer worth keeping in the ledger row, beside the standard ones.
_LEDGER_KEYS = ("gpu", "minutes", "waited_min", "ran_min", "overrun_min", "position", "reason", "kind", "guard",
                "placement_verdict", "override",
                "passed_over", "went_ahead_of", "justification_state", "replied", "shortened_to", "note")


#: The outcomes of a verb whose call did what the verb names: only such a call carries a number for the alerts.
_NUMBERED_OUTCOMES = {"request": ("queued", "already_queued"), "start": ("started",), "extend": ("extended",)}


def _ledger_metrics(verb: str, outcome: str, slot: Mapping[str, Any]) -> dict[str, int | float]:
    """The numbers lab/machine/alerts.py judges a slot by, in the spelling it reads: the Minutes the slot asked for
    on the row of a request and of a start, the minutes one extension adds on the row of an extend. The alerts take
    any request, start or extend row whose outcome is not a failure word as one that happened, so a refused call, or
    one that sent nothing to Notion, must not hand them a number."""
    if outcome not in _NUMBERED_OUTCOMES.get(verb, ()):
        return {}
    if verb == "extend":
        return {alerts.SLOT_EXTEND_MINUTES: EXTENSION_MINUTES}
    minutes = slot.get("minutes")
    return {alerts.SLOT_MINUTES: minutes} if isinstance(minutes, (int, float)) and not isinstance(minutes, bool) else {}


def _record(args: argparse.Namespace, answer: Mapping[str, Any], environ: Mapping[str, str],
            state_dir: str | os.PathLike[str] | None) -> int:
    """Append the ledger row of a finished call (lab.machine.record) and return its seq."""
    slot = answer.get("slot") if isinstance(answer.get("slot"), Mapping) else {}
    extras: dict[str, Any] = {}
    for key in _LEDGER_KEYS:
        value = answer.get(key, slot.get(key))
        if isinstance(value, (str, int, float, bool)):
            extras[key] = value
    extras["metrics"] = _ledger_metrics(args.sub, answer["outcome"], slot)
    gpu_letter = getattr(args, "gpu", None) or next(
        (letter for letter, name in GPUS.items() if name == slot.get("gpu")), DEFAULT_GPU)
    task = getattr(args, "task", None) or slot.get("task") or record.current_task(environ)
    return record.record(
        f"slot {args.sub}", answer["outcome"], refs={"slot": answer.get("row") or getattr(args, "row", None)},
        task=task, actor=record.current_actor(environ), tier=record.current_tier(environ), state_dir=state_dir,
        node=NODES[gpu_letter], **extras)


def run(argv: Sequence[str], *, client: notion.NotionClient | None = None, environ: Mapping[str, str] | None = None,
        env_files: Sequence[Any] | None = None, now: Callable[[], dt.datetime] | None = None,
        state_dir: str | os.PathLike[str] | None = None) -> tuple[dict[str, Any], int]:
    """One slot call: (the answer, the exit code). client replaces the Notion client the token would make, and
    now the clock; environ and env_files say where NOTION_TOKEN is looked up. The ledger row is written here."""
    environ = os.environ if environ is None else environ
    clock = now or utc_now
    args = parse_args(argv)
    actor = record.current_actor(environ)
    data_source = data_source_of(environ)
    refused = justification_refusal(args.minutes, args.why, args.measured) if args.sub == "request" else None
    if refused is None and args.sub == "request":
        args.placement = placement_check(args.gpu, args.what, args.kind)
        refused = placement_refusal(args.placement, args.override)
    if refused is not None:
        answer, code = refused, 1
    else:
        client = client or connect(environ, env_files)
        if client is None:
            answer, code = offline_plan(args, clock(), data_source, actor), 2
        else:
            try:
                answer = _dispatch(SlotQueue(client, data_source=data_source, now=clock), args, actor)
            except notion.NotionError as exc:
                answer = _failure(args, exc)
            code = 0 if answer["ok"] else 1
    try:
        answer["ledger_seq"] = _record(args, answer, environ, state_dir)
    except Exception as exc:
        answer["ledger_seq"] = None
        answer["ledger_error"] = f"{type(exc).__name__}: {exc}"
        log.error("slot: no ledger row written for %s (%s): %s", answer["verb"], answer["outcome"], exc)
    return answer, code


@stop.guarded
def main(argv: Sequence[str]) -> int:
    """The GPU queue for pod B: request, start, extend, done, complain, status."""
    answer, code = run(argv)
    print(json.dumps(answer))
    return code
