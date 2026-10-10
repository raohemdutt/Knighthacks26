"""slot: the GPU queue for pod B, the one exclusive card that every agent shares.

Anything that trains, benchmarks, restarts a process or needs the whole GPU asks for it here and waits its turn.

    python -m lab slot request  --task L3 --what "smoke LoRA 200 steps" [--minutes 15|60 --why "training run"]
    python -m lab slot start    --row <id or url>
    python -m lab slot extend   --row <id or url> --note "what the slot has already proven"
    python -m lab slot done     --row <id or url> --result "one line with the numbers" [--cancel]
    python -m lab slot complain --row <id or url> [--text "more detail"]
    python -m lab slot status   [--gpu B] [--row <id or url>] [--task L3]

The queue is the Notion database "GPU Queue (R9700 pods)": one row per request, read and written through
the Notion REST API (lab/machine/slots_notion.py) with the NOTION_TOKEN key. The verbs enforce the queue
rules on the client, because Notion cannot:

  request   adds a row: Request "<task>: <what>", Task ID, GPU "B exclusive" (--gpu A for the shared teacher
            pod, which is only queued to restart or change the server itself), Minutes 15, Status Queued,
            Agent Ilan, Requested now. 60 minutes is for a training run or a declared long job, and --why
            says which; it is written into the Request. Asking again with the same words while the first
            row is still Queued returns that row (outcome already_queued) instead of adding a second.
  start     sets Status Running and Started now, in one update, and only when no row on the GPU is Running
            and no other Queued row goes before this one. The order is shortest first: fewer Minutes goes
            first, and equal Minutes go to the earlier Requested. A Running row blocks even when it looks
            stale (the answer says when it is past its cap); complain instead. Notion cannot make the check
            and the update one step, so the queue is read again after the update, and a row that then finds
            any other Running row on its GPU puts itself back to Queued and answers lost_race: two agents
            that start in the same moment can never both keep the card, and may both step back and try again.
  extend    one extension of 15 minutes: Extended is checked and the Progress note says what the slot has
            already proven. A Running row only, once, while it is still inside its cap.
  done      Status Done, Ended now, Result (one line, with the numbers). It releases the card. --cancel
            withdraws a row that never started (Status Cancelled).
  complain  writes the Complaint on the agent's own Queued row, once it has waited more than twice its
            Minutes (or a Running row on its GPU is past its cap): the wait so far, what it is waiting
            behind, and any --text. A later complaint is added after the earlier ones.
  status    the open rows of the GPU in queue order, with who waits how long, when the Running one ends,
            whether it is overrun, and whether each Queued row may start; --row narrows it to one row,
            --task to one task's rows. It changes nothing.

A slot is capped at its Minutes (plus 15 once extended): cap_minutes() and ends_at() give the cap, and
is_overrun() says a Running row has passed it. The verbs do not end an overrun slot; the referee does.
A verb writes only a row of this data source whose Agent is its own (--agent, default Ilan); status
reads any row.

Every call answers with one JSON line (ok, outcome, verb, the row and its summary, and when it is refused a
reason and a next step) and an exit code: 0 when it worked, 1 when the rules or Notion refused it, 2 for bad
arguments and for a missing token. Trouble with Notion itself is outcome error, with its kind (unauthorized,
not_found, rate_limited, transport, ...). Each call that gets as far as a decision also appends one row to the
ledger (verb "slot <name>", the outcome, the task - the call's own, the row's, else $LAB_TASK - refs.slot the
row's id, node pod-b or pod-a, and the minutes and waits); a refusal is a row too. The actor is $LAB_ACTOR and
the tier $LAB_TIER, as for every ledger row (lab/machine/record.py). A ledger that cannot be written never
undoes what Notion already holds: the answer then carries ledger_error and the exit code stays what the
queue's outcome says. While the STOP file exists (lab/machine/stop.py) every slot verb is refused before it
runs.

A request, a start and an extend that went through also carry the numbers lab/machine/alerts.py judges a slot by,
in the spelling it reads: the metric slot.minutes (the Minutes asked for) on a request and a start, and
slot.extend_minutes on an extend. A refused call, and a call that sent nothing to Notion, carry none.

No token. NOTION_TOKEN is looked up in the process environment, then in the .env files of ready.py. When it
is missing nothing is sent anywhere: the verb answers outcome no_token with the exact Notion tool calls a
sitting makes instead (the row to create or the properties to set, in the form its Notion tools take, with
the checks to make first) and exits 2. The ledger still gets its row.

Settings, for rehearsals: LAB_QUEUE_DATA_SOURCE names another data source than the live queue's, and
LAB_NOTION_URL another address than https://api.notion.com. The token is never printed, logged or written to
the ledger; error messages are scrubbed of it.

For other code: Slot, slot_from_page(), queue_order(), cap_minutes(), ends_at(), is_overrun(),
complaint_due(), connect() (the client the token makes) and SlotQueue (the six verbs as methods) are the same
rules the verbs use.
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
    "COLUMNS", "DATA_SOURCE_ID", "GPUS", "Slot", "SlotQueue", "cap_minutes", "complaint_due", "connect",
    "data_source_of", "ends_at", "is_overrun", "main", "mcp_properties", "parse_moment", "queue_order",
    "rest_properties", "run", "slot_from_page",
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
OPEN_STATUSES = (QUEUED, RUNNING)

DEFAULT_MINUTES = 15
LONG_MINUTES = 60
SLOT_MINUTES = (DEFAULT_MINUTES, LONG_MINUTES)
EXTENSION_MINUTES = 15
COMPLAIN_AFTER_FACTOR = 2        # a row may complain once it has waited more than this many times its Minutes

MAX_WHAT_CHARS = 200
MAX_WHY_CHARS = 120
MAX_NOTE_CHARS = 1000            # progress note, result, extra complaint text
MAX_COMPLAINT_CHARS = 2000       # what one Notion text value holds; older complaints are dropped to fit
COMPLAINT_JOIN = " | "

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


# --- the rules ------------------------------------------------------------------------

def queue_key(slot: Slot) -> tuple[float, dt.datetime, dt.datetime, str]:
    """Where a row stands in the queue, smaller going first: fewer Minutes first (shortest job first), equal
    Minutes by the earlier Requested, then by when Notion made the row, then by id so the order is total."""
    return (slot.minutes_or_default, slot.requested_at or _LATEST, slot.created or _LATEST, slot.id)


def queue_order(rows: Iterable[Slot]) -> list[Slot]:
    """The Queued rows of rows in the order they may start."""
    return sorted((row for row in rows if row.status == QUEUED), key=queue_key)


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


def ahead_of(slot: Slot, rows: Iterable[Slot]) -> list[Slot]:
    """The Queued rows on slot's GPU that go before it."""
    mine = queue_key(slot)
    return [row for row in queue_order(rows) if row.gpu == slot.gpu and row.id != slot.id and queue_key(row) < mine]


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
    if minutes != DEFAULT_MINUTES:
        title += f" ({minutes} min: {why})"
    return title


def add_complaint(existing: str, entry: str) -> str:
    """The Complaint column with entry added after what it already holds, oldest entries dropped to fit."""
    parts = [part for part in existing.split(COMPLAINT_JOIN) if part] + [entry]
    while len(parts) > 1 and len(COMPLAINT_JOIN.join(parts)) > MAX_COMPLAINT_CHARS:
        parts.pop(0)
    return COMPLAINT_JOIN.join(parts)[-MAX_COMPLAINT_CHARS:]


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
        """Where a Queued row stands: its place, who is ahead, what is Running, whether it may start."""
        ahead = ahead_of(slot, rows)
        running = running_on(rows, slot.gpu, other_than=slot.id)
        return {"position": len(ahead) + 1, "may_start": not ahead and not running,
                "ahead": [summary(row, now) for row in ahead], "running": [summary(row, now) for row in running]}

    # request

    def request(self, *, task: str, what: str, minutes: int = DEFAULT_MINUTES, why: str | None = None,
                gpu: str = DEFAULT_GPU, agent: str = DEFAULT_AGENT) -> dict[str, Any]:
        verb, gpu_name, now = "slot request", GPUS[gpu], self._now()
        title = request_title(task, what, minutes, why)
        rows = self.open_rows(gpu_name)
        twin = next((r for r in rows if r.status == QUEUED and r.request == title and same_agent(r.agent, agent)), None)
        outcome = "already_queued"
        if twin is None:
            change = {"Request": title, "Task ID": task, "GPU": gpu_name, "Minutes": minutes, "Status": QUEUED,
                      "Agent": agent, "Requested": now}
            twin = slot_from_page(self.client.create_page(self.data_source, rest_properties(change)))
            rows = [row for row in rows if row.id != twin.id] + [twin]
            outcome = "queued"
        answer: dict[str, Any] = {"ok": True, "outcome": outcome, "verb": verb, "row": twin.id, "url": twin.url,
                                  "slot": summary(twin, now), **self._where(twin, rows, now)}
        patience = tidy(COMPLAIN_AFTER_FACTOR * twin.minutes_or_default)
        answer["next"] = (f"python -m lab slot start --row {twin.id}" if answer["may_start"] else
                          f"wait for your turn (python -m lab slot status --row {twin.id}); after {patience} minutes "
                          f"of waiting, python -m lab slot complain --row {twin.id}")
        if gpu == "A":
            answer["warning"] = ("pod A is the shared teacher service: queue it only to restart or change the "
                                 "server itself")
        return answer

    # start

    def start(self, row_id: str, *, agent: str = DEFAULT_AGENT) -> dict[str, Any]:
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
        rows = self.open_rows(slot.gpu)
        running = running_on(rows, slot.gpu, other_than=slot.id)
        if running:
            stale = [row for row in running if is_overrun(row, now)]
            return refusal(verb, "busy", "a row is Running on this GPU; never start while one is Running, even if it "
                           "looks stale - complain instead" + (" (it is past its cap)" if stale else ""),
                           row=slot.id, running=[summary(row, now) for row in running], slot=summary(slot, now))
        ahead = ahead_of(slot, rows)
        if ahead:
            return refusal(verb, "not_next", f"{len(ahead)} Queued row(s) go before this one (shortest first, then "
                           "earliest request)", row=slot.id, ahead=[summary(row, now) for row in ahead],
                           slot=summary(slot, now))
        began = self._now()
        mine = slot_from_page(self.client.update_page(slot.id, rest_properties({"Status": RUNNING, "Started": began})))
        try:
            rivals = running_on(self.open_rows(slot.gpu), slot.gpu, other_than=slot.id)
        except notion.NotionError as exc:
            log.warning("slot start: row %s is Running, but the queue could not be read again: %s", slot.id, exc)
            return {"ok": True, "outcome": "started", "verb": verb, "row": mine.id, "slot": summary(mine, began),
                    "release": f"python -m lab slot done --row {mine.id} --result \"<one line with the numbers>\"",
                    "warning": f"the queue could not be read again to confirm that no other row started in the same "
                               f"moment ({exc}); run python -m lab slot status and release the card if one did"}
        if rivals:
            answer = refusal(verb, "lost_race", "another row is Running on this GPU after this row's update (two agents "
                             "started in the same moment); this row is back to Queued - try again in a moment",
                             row=slot.id, running=[summary(row, began) for row in rivals])
            try:
                self.client.update_page(slot.id, rest_properties({"Status": QUEUED, "Started": None}))
            except notion.NotionError as exc:
                answer.update(outcome="error_lost_race", revert_error=str(exc),
                              reason="another row is Running on this GPU and this row could not be put back to Queued: "
                              "set its Status to Queued and clear Started by hand")
            return answer
        return {"ok": True, "outcome": "started", "verb": verb, "row": mine.id, "slot": summary(mine, began),
                "release": f"python -m lab slot done --row {mine.id} --result \"<one line with the numbers>\""}

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
        updated = slot_from_page(self.client.update_page(
            slot.id, rest_properties({"Extended": True, "Progress note": note})))
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
        rows = self.open_rows(slot.gpu)
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
        entry = (f"{iso(now)[:16]}Z {why}; waiting behind: {'; '.join(behind) or 'nothing found'}"
                 + (f". {text}" if text else ""))
        updated = slot_from_page(self.client.update_page(
            slot.id, rest_properties({"Complaint": add_complaint(slot.complaint, entry)})))
        return {"ok": True, "outcome": "complaint_recorded", "verb": verb, "row": updated.id, "reason": why,
                "slot": summary(updated, now), **where}

    # status

    def status(self, *, gpu: str = DEFAULT_GPU, row_id: str | None = None, task: str | None = None,
               agent: str = DEFAULT_AGENT) -> dict[str, Any]:
        verb, gpu_name, now = "slot status", GPUS[gpu], self._now()
        rows = self.open_rows(gpu_name)
        if row_id is not None:
            slot, bad = self.load(row_id, agent, verb, own=False)
            if bad:
                return bad
            answer: dict[str, Any] = {"ok": True, "outcome": "status", "verb": verb, "row": slot.id,
                                      "slot": summary(slot, now)}
            if slot.status == QUEUED:
                answer.update(self._where(slot, rows, now))
            return answer
        running = running_on(rows, gpu_name)
        queued = queue_order(rows)
        free = not running
        listed = []
        for index, row in enumerate(queued, start=1):
            if task is None or row.task_id == task:
                listed.append({**summary(row, now), "position": index, "may_start": free and index == 1})
        return {"ok": True, "outcome": "status", "verb": verb, "gpu": gpu_name,
                "running": [summary(row, now) for row in running if task is None or row.task_id == task],
                "queued": listed, "card_free": free, "next_row": queued[0].id if queued else None}


# --- no token: what a sitting writes by hand --------------------------------------------

def _tool(name: str) -> str:
    return f"mcp__{NOTION_SERVER}__notion-{name}"


def _query_step(gpu_name: str, data_source: str, why: str) -> dict[str, Any]:
    return {"tool": _tool("query-data-sources"), "why": why, "arguments": {"data": {
        "mode": "rows", "data_source_url": f"collection://{data_source}",
        "filter": open_rows_mcp_filter(gpu_name), "limit": 100}}}


def _update_step(row: str, change: Mapping[str, Any], why: str) -> dict[str, Any]:
    return {"tool": _tool("update-page"), "why": why, "arguments": {
        "page_id": row, "command": "update_properties", "properties": mcp_properties(change)}}


def _fetch_step(row: str, why: str) -> dict[str, Any]:
    return {"tool": _tool("fetch"), "why": why, "arguments": {"id": row}}


RULE_START = ("Start only when no row on the GPU is Running and every other Queued row on it asks for more minutes "
              "than yours, or the same minutes and was requested after yours. Never start while one is Running, "
              "even if it looks stale; complain instead. After the update query again: if any other row on the GPU "
              "is Running, set yours back to Queued, clear Started and try again in a moment.")
RULE_EXTEND = ("One extension of 15 minutes, only on your Running row, only before Started plus Minutes has "
               "passed, only when Extended is unchecked, and only with a Progress note saying what the slot has "
               "already proven.")
RULE_COMPLAIN = ("Only on your own Queued row, and only after it has waited more than twice its Minutes (or the "
                 "Running row is past Started plus its Minutes, plus 15 if Extended is checked). Say what you wait "
                 "for and for how long; keep any text the Complaint already holds and add yours after ' | '.")
RULE_DONE = ("Only on your Running row. Result is one line with the numbers the slot proved. "
             "Then stop using the card at once.")


def offline_plan(args: argparse.Namespace, now: dt.datetime, data_source: str) -> dict[str, Any]:
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
        plan["write"] = mcp_properties(change)
        plan["steps"] = [
            {"tool": _tool("create-pages"), "why": "add your row to the GPU Queue", "arguments": {
                "parent": {"type": "data_source_id", "data_source_id": data_source},
                "pages": [{"properties": mcp_properties(change)}]}},
            _query_step(gpu_name, data_source, "list the Queued and Running rows of the GPU, then apply the start "
                        "rule"),
        ]
        plan["rule"] = RULE_START
    elif sub == "start":
        change = {"Status": RUNNING, "Started": now}
        plan["write"] = mcp_properties(change)
        plan["steps"] = [
            _fetch_step(args.row, "read your row: Status must be Queued; note its GPU and Minutes"),
            _query_step(gpu_name, data_source, "list the Queued and Running rows of your row's GPU (this filter is "
                        f"{gpu_name!r}; use the GPU your row names if it differs), then apply the start rule"),
            _update_step(args.row, change, "mark your row Running, Status and Started in one update"),
            _query_step(gpu_name, data_source, "check that no other row is Running; if one is, undo your update"),
        ]
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
        plan["steps"] = [_query_step(gpu_name, data_source, "the Queued and Running rows of the GPU, in no order; "
                                     "sort them by Minutes, then Requested")]
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
                         help="15, or 60 for a training run or a declared long job (then --why is required)")
    request.add_argument("--why", type=_text_type(MAX_WHY_CHARS, "--why"), metavar="TEXT",
                         help="why a 60-minute slot; written into the Request")
    request.add_argument("--gpu", type=_gpu_type, default=DEFAULT_GPU, metavar="{A,B}",
                         help="B (default) is the exclusive card; A is the shared teacher service")

    for name, help_ in (("start", "start the row's slot when its turn has come"),
                        ("extend", "extend a Running slot by 15 minutes, once, with a progress note"),
                        ("done", "finish a Running slot and release the card, or cancel a Queued row"),
                        ("complain", "write a complaint on a row that has waited too long"),
                        ("status", "show the queue, or one row")):
        parser_ = sub(name, help_)
        parser_.add_argument("--row", required=name != "status", type=_row_type, metavar="ID_OR_URL",
                             help="the row's page id or its Notion URL")
        if name == "extend":
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
    if args.sub == "request" and args.minutes != DEFAULT_MINUTES and not args.why:
        parser.error(f"--minutes {args.minutes} needs --why: say why this is a training run or a declared long job")
    if args.sub == "request" and args.minutes == DEFAULT_MINUTES and args.why:
        parser.error(f"--why is for a {LONG_MINUTES}-minute slot; a {DEFAULT_MINUTES}-minute slot needs no reason")
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


def _dispatch(queue: SlotQueue, args: argparse.Namespace) -> dict[str, Any]:
    if args.sub == "request":
        return queue.request(task=args.task, what=args.what, minutes=args.minutes, why=args.why, gpu=args.gpu,
                             agent=args.agent)
    if args.sub == "start":
        return queue.start(args.row, agent=args.agent)
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
_LEDGER_KEYS = ("gpu", "minutes", "waited_min", "ran_min", "overrun_min", "position", "reason", "kind")


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
    data_source = data_source_of(environ)
    client = client or connect(environ, env_files)
    if client is None:
        answer, code = offline_plan(args, clock(), data_source), 2
    else:
        try:
            answer = _dispatch(SlotQueue(client, data_source=data_source, now=clock), args)
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
