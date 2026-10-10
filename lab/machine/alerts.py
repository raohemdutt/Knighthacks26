"""alerts: the machine tells its owner what went wrong, in one markdown file per UTC day.

    python -m lab alerts [--hours H] [--now T] [--dry-run]

One call reads the ledger, applies the five rules below to the rows of the last H hours (24 unless told
otherwise), and appends one line to <state dir>/alerts/<day>.md for every condition that has not been announced
before. The day is the UTC day of the call and the line starts with the time of the call; the body says when the
thing itself happened. A line looks like

    - 15:00Z **Sitting s-son-2 ended with error** sonnet, task O33, exit code 1, after 15.2 min; ended 2026-10-10 08:20Z (ledger:22) `sitting_error:22`

The backticked key at the end names the condition. Before it writes, a call looks for the keys already in the
files of the days it covers, so running it every few minutes, or twice in a row, announces each condition once.
A file is only ever appended to, under a lock (alerts.lock in the state dir), and a line is flushed and fsynced
before the call answers. Ported from the phone project's alerts (social_outreach/tools/_core/alerts.py): the file
per day and the announce-once rule are the same; the Windows toast is not, because nobody watches this machine's
screen, and the conditions come from the ledger instead of from the code that met them.

The rules. Each reads rows the lab's own verbs write; the words they look for are the constants below, so the
writers of the slot and pod rows (the queue verbs, the pod controller) have one place to check their spelling.

  sitting_error     a row with verb sit and outcome error (or fail, failed, crashed): the launcher's record that
                    a sitting ended with an error. One alert per row, key sitting_error:<seq>. A sitting that
                    timed out, ran out of budget, was stopped or was interrupted is not an error here; the
                    review counts those.
  slot_over_cap     a GPU slot that ran longer than its cap, or is running past it now. Rows with the verbs
                    slot request, slot start, slot extend and slot done (or a referee row) are matched by
                    refs.slot; a row whose outcome says it failed or was refused is ignored. The cap is metrics
                    slot.minutes on the start (else the request) row, plus 15 minutes once a slot extend row
                    exists (metrics slot.extend_minutes overrides the 15); a slot whose rows carry no
                    slot.minutes cannot be judged and raises nothing. The slot ends at its first slot done row;
                    until then it runs until now. A minute of grace is allowed. One alert per slot, key
                    slot_over_cap:<slot>, whether it was first seen running or already done.
  metric_regressed  a reading of a metric that is worse than the previous reading of the same series by at
                    least REGRESSION_FRACTION. A series is the metrics report's (lab/machine/metrics_report.py):
                    metric, node, unit and setup, the setup being the row's conditions less the keys that
                    describe the number itself. "Worse" follows the metric's direction in that report's table
                    (by name, else by unit); a metric with no better side, such as a credit balance or a VRAM
                    share, is never judged. Rows marked low_confidence take no part, and readings are ordered
                    by their ts, not by their seq. One alert per reading and metric, key
                    metric_regressed:<seq>:<metric>. This is a prompt to look, not a verdict: the metrics
                    report and lab/machine/verdict.py judge a comparison with enough runs.
  pod_stopped       a pod that went down without being stopped by us. Rows with a verb that starts with "pod "
                    and a node (pod-a, pod-b) are replayed in time order. pod launch, and an outcome of running,
                    ready, up or alive, mean up; pod stop means down on purpose. A row whose outcome says
                    stopped, down, not_running or gone while the pod was up, or culled, lost or died unless it
                    was known to be down, is the alert. One alert per observing row, key pod_stopped:<seq>.
  credits_low       a pod account whose credits are projected to run out within CREDIT_HORIZON_HOURS. The
                    balance is metrics pod.balance_credits on a row with a node. From the readings of the last
                    CREDIT_WINDOW_HOURS (cut at the last time the balance went up, a top-up) that are at least
                    CREDIT_MIN_SPAN_MINUTES apart, the burn per minute is the fall over the span; the runway is
                    what is left now at that burn. A pod the rule above knows to be stopped burns nothing and
                    raises none. One alert per account per day, key credits_low:<node>:<day>.

A title and a body hold only what the rows' structured fields hold (ids, outcomes, numbers); a free-text note or
a complaint is never copied into the alerts.

Alerts, like the ledger, are read-only for sittings (allowed.py); the verbs write them. The verb runs while the
STOP file exists (@stop.exempt): it only reads the ledger and writes the owner's file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lab.machine import ledger, stop
from lab.machine.lock import FileLock, LockTimeout
from lab.machine.metrics_report import HIGHER, comparable, direction_of

__all__ = [
    "VERB", "ALERTS_DIR", "ALERTS_LOCK", "DEFAULT_HOURS", "RULES", "SITTING_ERROR", "SLOT_OVER_CAP",
    "METRIC_REGRESSED", "POD_STOPPED", "CREDITS_LOW",
    "REGRESSION_FRACTION", "CREDIT_HORIZON_HOURS", "CREDIT_WINDOW_HOURS", "CREDIT_MIN_SPAN_MINUTES",
    "SLOT_COMPLAIN", "BALANCE_METRIC", "EVENT_METRIC_PREFIXES", "SIT_VERB",
    "Alert", "ParsedLine", "Report", "SlotRun", "alerts_dir", "alerts_path", "parse_ts", "number", "metric_value",
    "ident", "label", "one_line", "cite", "timeline", "series_key", "reading_unit", "setup_of",
    "compact_conditions",
    "is_failure", "slot_runs", "replay_pods", "credit_runways", "scan", "format_line", "parse_line", "day_lines",
    "announced_keys", "raise_alerts", "aware_time", "main",
]

UTC = dt.timezone.utc

VERB = "alerts"
ALERTS_DIR = "alerts"            # under the state dir
ALERTS_LOCK = "alerts.lock"      # in the state dir, beside ledger.lock
LOCK_TIMEOUT = ledger.LOCK_TIMEOUT
DEFAULT_HOURS = 24.0             # how far back a call looks for conditions

# --- the rules' names -------------------------------------------------------------------------

SITTING_ERROR = "sitting_error"
SLOT_OVER_CAP = "slot_over_cap"
METRIC_REGRESSED = "metric_regressed"
POD_STOPPED = "pod_stopped"
CREDITS_LOW = "credits_low"
RULES = (SITTING_ERROR, SLOT_OVER_CAP, METRIC_REGRESSED, POD_STOPPED, CREDITS_LOW)

# --- the words the rules look for in ledger rows ----------------------------------------------

SIT_VERB = "sit"
SITTING_ERROR_OUTCOMES = ("error", "fail", "failed", "crashed")

SLOT_REQUEST = "slot request"
SLOT_START = "slot start"
SLOT_EXTEND = "slot extend"
SLOT_DONE = "slot done"
SLOT_COMPLAIN = "slot complain"
SLOT_END_VERBS = (SLOT_DONE, "referee")
SLOT_VERBS = (SLOT_REQUEST, SLOT_START, SLOT_EXTEND, *SLOT_END_VERBS)
SLOT_MINUTES = "slot.minutes"
SLOT_EXTEND_MINUTES = "slot.extend_minutes"
EXTENSION_MINUTES = 15.0         # the one extension a slot may take (open-ticket skill, section 4b)
OVERRUN_GRACE_MINUTES = 1.0

POD_PREFIX = "pod "
POD_LAUNCH = "pod launch"
POD_STOP = "pod stop"
POD_UP_OUTCOMES = ("running", "ready", "up", "alive")
POD_DOWN_OUTCOMES = ("stopped", "down", "not_running", "gone")
POD_LOST_OUTCOMES = ("culled", "lost", "died")

BALANCE_METRIC = "pod.balance_credits"
CREDIT_HORIZON_HOURS = 24.0      # a runway shorter than this raises credits_low
CREDIT_WINDOW_HOURS = 6.0        # balance readings older than this are not used
CREDIT_MIN_SPAN_MINUTES = 30.0   # readings closer together than this say nothing about the burn

#: An outcome that starts with one of these did not happen (a refused slot start is not a start).
FAILURE_WORDS = ("fail", "error", "crash", "refus", "denied", "skipped", "invalid", "abort", "timeout",
                 "timed_out", "lock_timeout")

# --- metric direction ---------------------------------------------------------------------------

REGRESSION_FRACTION = 0.10       # worse by this share of the previous reading or more

#: Metrics with these prefixes describe one event (a sitting, a slot), not a series to compare.
EVENT_METRIC_PREFIXES = ("sitting.", "slot.", "alerts.", "review.")

Row = dict[str, Any]
StateDir = str | os.PathLike[str] | None


# --- small helpers ----------------------------------------------------------------------------

def parse_ts(value: Any) -> dt.datetime | None:
    """A ledger ts (ISO 8601 text, Z or an offset) as an aware UTC time; None for anything else."""
    if not isinstance(value, str) or not value:
        return None
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        moment = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    return moment.astimezone(UTC) if moment.utcoffset() is not None else None


def number(value: Any) -> float | None:
    """value as a finite float; None when it is not a number (a bool is not one)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def metric_value(row: Mapping[str, Any], name: str) -> float | None:
    """The row's reading of one metric as a float; None when it carries none."""
    metrics = row.get("metrics")
    return number(metrics.get(name)) if isinstance(metrics, Mapping) else None


_UNSAFE = re.compile(r"[^A-Za-z0-9._:/@#+=-]")


def ident(value: Any, limit: int = 60) -> str:
    """An id from a row on one line and safe for a title or a key: anything but letters, digits and a
    few punctuation marks becomes an underscore; cut at limit; "?" when nothing is left."""
    text = _UNSAFE.sub("_", str(value if value is not None else "").strip())
    return text[:limit] or "?"


def label(value: Any, missing: str) -> str:
    """ident(value), or the plain words missing ("no node") when the row names nothing."""
    return ident(value) if isinstance(value, str) and value.strip() else missing


def one_line(text: str) -> str:
    """text without line breaks, backticks or bold marks, which would break the file's line shape."""
    return " ".join(text.replace("`", "'").replace("**", "*").split())


def is_failure(outcome: Any) -> bool:
    """Whether an outcome says the call did not happen or did not finish well (it starts with one of
    FAILURE_WORDS)."""
    return isinstance(outcome, str) and outcome.lower().startswith(FAILURE_WORDS)


def timeline(rows: Iterable[Row], now: dt.datetime) -> list[tuple[dt.datetime, Row]]:
    """The rows that carry a readable ts at or before now, as (time, row) in time order (the seq breaks a
    tie). Ledger order is not always time order: a backfilled row keeps its old ts."""
    found = []
    for row in rows:
        moment = parse_ts(row.get("ts"))
        if moment is not None and moment <= now:
            found.append((moment, row))
    found.sort(key=lambda item: (item[0], item[1].get("seq") if isinstance(item[1].get("seq"), int) else 0))
    return found


def cite(row: Mapping[str, Any]) -> str:
    """A row in the citation form the refinement passes use: ledger:<seq>."""
    return f"ledger:{row.get('seq')}"


def _minutes_text(minutes: float) -> str:
    return f"{minutes:.1f} min"


def stamp(moment: dt.datetime) -> str:
    """A moment as the bodies of alerts write it: 2026-10-10 08:20Z (UTC, to the minute)."""
    return f"{moment.astimezone(UTC):%Y-%m-%d %H:%M}Z"


def reading_unit(row: Mapping[str, Any], name: str) -> str | None:
    """The unit the row gives one of its metrics; None when it gives none."""
    units = row.get("units")
    unit = units.get(name) if isinstance(units, Mapping) else None
    return unit if isinstance(unit, str) else None


def setup_of(row: Mapping[str, Any]) -> dict[str, Any]:
    """The row's conditions without the keys that describe the number rather than how it was measured (the
    metrics report's comparable())."""
    conditions = row.get("conditions")
    return comparable(conditions) if isinstance(conditions, Mapping) else {}


def series_key(name: str, row: Mapping[str, Any]) -> tuple[str, str | None, str | None, str]:
    """What makes two readings comparable, as in the metrics report: the metric, the node, the unit and the
    setup. A reading is only ever set against one measured the same way."""
    setup = setup_of(row)
    canon = json.dumps(setup, sort_keys=True, separators=(",", ":"), default=str) if setup else ""
    return name, row.get("node") if isinstance(row.get("node"), str) else None, reading_unit(row, name), canon


def compact_conditions(row: Mapping[str, Any], limit: int = 70) -> str:
    """The setup of the row as "key=value, ..." for a line, cut at limit; "" when it has none."""
    setup = setup_of(row)
    text = ", ".join(f"{key}={one_line(str(setup[key]))}" for key in sorted(setup, key=str))
    return text if len(text) <= limit else text[: limit - 3] + "..."


# --- the alert ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class Alert:
    """One condition the rules found. key is what makes the same condition announce once."""

    rule: str
    key: str
    title: str
    body: str
    when: dt.datetime        # when the condition arose (the ledger row's time)


def _alert(rule: str, key: str, title: str, body: str, when: dt.datetime) -> Alert:
    return Alert(rule, one_line(key).replace(" ", "_"), one_line(title), one_line(body), when)


# --- rule: a sitting ended with an error ---------------------------------------------------------

def sitting_errors(events: Sequence[tuple[dt.datetime, Row]], since: dt.datetime) -> list[Alert]:
    found = []
    for when, row in events:
        if when < since or row.get("verb") != SIT_VERB or row.get("outcome") not in SITTING_ERROR_OUTCOMES:
            continue
        refs = row.get("refs") if isinstance(row.get("refs"), Mapping) else {}
        sitting = ident(refs.get("run_id") or row.get("actor"))
        parts = [label(row.get("tier"), "no tier"), f"task {ident(row['task'])}" if row.get("task") else "no task"]
        exit_code = row.get("exit_code")
        if isinstance(exit_code, int) and not isinstance(exit_code, bool):
            parts.append(f"exit code {exit_code}")
        seconds = metric_value(row, "sitting.duration_s")
        if seconds is not None:
            parts.append(f"after {_minutes_text(seconds / 60)}")
        found.append(_alert(SITTING_ERROR, f"{SITTING_ERROR}:{row['seq']}",
                            f"Sitting {sitting} ended with {row['outcome']}",
                            f"{', '.join(parts)}; ended {stamp(when)} ({cite(row)})", when))
    return found


# --- rule: a slot past its cap -------------------------------------------------------------------

@dataclass(frozen=True)
class SlotRun:
    """One GPU slot as the ledger tells it."""

    slot: str
    node: str | None
    task: str | None
    start: dt.datetime
    start_seq: int
    minutes: float                   # what was asked for
    extended: bool
    cap: float                       # minutes, plus the extension
    end: dt.datetime | None          # None while it runs
    elapsed: float                   # minutes from the start to the end, or to now

    @property
    def over_cap(self) -> bool:
        return self.elapsed > self.cap + OVERRUN_GRACE_MINUTES


def slot_runs(events: Sequence[tuple[dt.datetime, Row]], now: dt.datetime) -> list[SlotRun]:
    """Every slot with a successful start row and a known number of minutes, in order of start. A slot
    whose rows name no slot.minutes cannot be judged and is left out."""
    by_slot: dict[str, list[tuple[dt.datetime, Row]]] = {}
    for when, row in events:
        refs = row.get("refs")
        slot = refs.get("slot") if isinstance(refs, Mapping) else None
        if (slot is None or isinstance(slot, bool) or row.get("verb") not in SLOT_VERBS
                or is_failure(row.get("outcome"))):
            continue
        by_slot.setdefault(str(slot), []).append((when, row))
    runs = []
    for slot, group in by_slot.items():
        start = next(((when, row) for when, row in group if row["verb"] == SLOT_START), None)
        if start is None:
            continue
        start_when, start_row = start
        minutes = metric_value(start_row, SLOT_MINUTES)
        if minutes is None:
            minutes = next((metric_value(row, SLOT_MINUTES) for _, row in group
                            if row["verb"] == SLOT_REQUEST and metric_value(row, SLOT_MINUTES) is not None), None)
        if minutes is None:
            continue
        later = [(when, row) for when, row in group if (when, row.get("seq")) >= (start_when, start_row.get("seq"))]
        extension = next((row for _, row in later if row["verb"] == SLOT_EXTEND), None)
        extra = 0.0
        if extension is not None:
            extra = metric_value(extension, SLOT_EXTEND_MINUTES) or EXTENSION_MINUTES
        end = next((when for when, row in later if row["verb"] in SLOT_END_VERBS), None)
        elapsed = ((end or now) - start_when).total_seconds() / 60
        node = start_row.get("node") if isinstance(start_row.get("node"), str) else None
        task = start_row.get("task") if isinstance(start_row.get("task"), str) else None
        runs.append(SlotRun(slot, node, task, start_when, start_row["seq"], minutes, extension is not None,
                            minutes + extra, end, elapsed))
    runs.sort(key=lambda run: (run.start, run.start_seq))
    return runs


def slot_overruns(events: Sequence[tuple[dt.datetime, Row]], since: dt.datetime, now: dt.datetime) -> list[Alert]:
    found = []
    for run in slot_runs(events, now):
        if not run.over_cap or (run.end or now) < since:
            continue
        went_over = run.end or run.start + dt.timedelta(minutes=run.cap)
        state = "is past" if run.end is None else "ran past"
        detail = f"{run.elapsed:.0f} of {run.cap:.0f} min" + (" (extended)" if run.extended else "")
        where = f"{label(run.node, 'no node')}, {'task ' + ident(run.task) if run.task else 'no task'}"
        found.append(_alert(SLOT_OVER_CAP, f"{SLOT_OVER_CAP}:{ident(run.slot)}",
                            f"Slot {ident(run.slot)} {state} its cap",
                            f"{where}: {detail}, started {stamp(run.start)}"
                            f"{', still running' if run.end is None else ', ended ' + stamp(run.end)} "
                            f"(ledger:{run.start_seq})", went_over))
    return found


# --- rule: a metric regressed --------------------------------------------------------------------

def metric_regressions(events: Sequence[tuple[dt.datetime, Row]], since: dt.datetime,
                       fraction: float = REGRESSION_FRACTION) -> list[Alert]:
    series: dict[tuple[str, str | None, str | None, str], list[tuple[dt.datetime, Row, float]]] = {}
    for when, row in events:
        metrics = row.get("metrics")
        if row.get("low_confidence") is True or not isinstance(metrics, Mapping):
            continue
        for name in metrics:
            value = number(metrics[name])
            if value is None or not isinstance(name, str) or name.startswith(EVENT_METRIC_PREFIXES):
                continue
            series.setdefault(series_key(name, row), []).append((when, row, value))
    found = []
    for (name, node, _unit, _setup), readings in series.items():
        for (_, before_row, before), (when, row, after) in zip(readings, readings[1:]):
            direction = direction_of(name, reading_unit(row, name))
            if direction is None or when < since or before == 0:
                continue
            worse = (before - after) / abs(before) if direction == HIGHER else (after - before) / abs(before)
            if worse < fraction:
                continue
            unit = reading_unit(row, name)
            conditions = compact_conditions(row)
            body = (f"{label(node, 'no node')}: {before:g} -> {after:g}{' ' + unit if unit else ''} "
                    f"({(after - before) / abs(before) * 100:+.1f}%, {direction} is better)"
                    f"{'; ' + conditions if conditions else ''}; read {stamp(when)} "
                    f"({cite(before_row)} -> {cite(row)})")
            found.append(_alert(METRIC_REGRESSED, f"{METRIC_REGRESSED}:{row['seq']}:{ident(name)}",
                                f"{ident(name)} regressed", body, when))
    return found


# --- rule: a pod stopped without being stopped -----------------------------------------------------

def replay_pods(events: Sequence[tuple[dt.datetime, Row]],
                since: dt.datetime) -> tuple[list[Alert], dict[str, bool]]:
    """Replay the pod rows in time order: the alerts for pods that went down unexpectedly at or after since, and
    the last known state of each node (True up, False down); a node no row has spoken of is not in it."""
    up: dict[str, bool | None] = {}
    found = []
    for when, row in events:
        verb, outcome = row.get("verb"), row.get("outcome")
        if not (isinstance(verb, str) and verb.startswith(POD_PREFIX)) or not isinstance(outcome, str):
            continue
        node = row.get("node") if isinstance(row.get("node"), str) else "unknown"
        state = up.get(node)
        if verb == POD_STOP and not is_failure(outcome):
            up[node] = False
        elif verb == POD_LAUNCH and not is_failure(outcome):
            up[node] = True
        elif outcome in POD_UP_OUTCOMES:
            up[node] = True
        elif outcome in POD_DOWN_OUTCOMES or outcome in POD_LOST_OUTCOMES:
            unexpected = state is True if outcome in POD_DOWN_OUTCOMES else state is not False
            up[node] = False
            if unexpected and when >= since:
                found.append(_alert(POD_STOPPED, f"{POD_STOPPED}:{row['seq']}",
                                    f"Pod {ident(node)} stopped unexpectedly",
                                    f"{verb} reported {outcome} at {stamp(when)} and no pod stop was recorded "
                                    f"since it was up ({cite(row)})", when))
    return found, {node: state for node, state in up.items() if state is not None}


def pod_stops(events: Sequence[tuple[dt.datetime, Row]], since: dt.datetime) -> list[Alert]:
    """The pods that went down without a pod stop (see replay_pods)."""
    return replay_pods(events, since)[0]


# --- rule: credits projected to run out ----------------------------------------------------------

def credit_runways(events: Sequence[tuple[dt.datetime, Row]], now: dt.datetime, *,
                   window_hours: float = CREDIT_WINDOW_HOURS,
                   min_span_minutes: float = CREDIT_MIN_SPAN_MINUTES) -> dict[str, dict[str, Any]]:
    """Per node with a usable projection: the balance now, the burn per minute, the hours left, and the
    first and last readings the projection stands on."""
    cutoff = now - dt.timedelta(hours=window_hours)
    by_node: dict[str, list[tuple[dt.datetime, Row, float]]] = {}
    for when, row in events:
        balance = metric_value(row, BALANCE_METRIC)
        node = row.get("node")
        if balance is not None and isinstance(node, str) and when >= cutoff:
            by_node.setdefault(node, []).append((when, row, balance))
    runways = {}
    for node, readings in by_node.items():
        run = [readings[-1]]
        for reading in reversed(readings[:-1]):
            if reading[2] < run[0][2]:         # the balance rose after this reading: a top-up, stop here
                break
            run.insert(0, reading)
        (first_when, first_row, first), (last_when, last_row, last) = run[0], run[-1]
        span = (last_when - first_when).total_seconds() / 60
        burn = (first - last) / span if span >= min_span_minutes else 0.0
        if burn <= 0:
            continue
        left = max(0.0, last - burn * (now - last_when).total_seconds() / 60)
        runways[node] = {"balance": left, "burn_per_min": burn, "hours_left": left / burn / 60,
                         "first": first_row, "last": last_row, "when": last_when}
    return runways


def credit_alerts(events: Sequence[tuple[dt.datetime, Row]], now: dt.datetime,
                  horizon_hours: float = CREDIT_HORIZON_HOURS, down: Iterable[str] = ()) -> list[Alert]:
    """The accounts whose runway is under horizon_hours. A node in down is a pod known to be stopped, which
    burns nothing, so it raises none."""
    found = []
    stopped = set(down)
    for node, runway in sorted(credit_runways(events, now).items()):
        if runway["hours_left"] >= horizon_hours or node in stopped:
            continue
        found.append(_alert(
            CREDITS_LOW, f"{CREDITS_LOW}:{ident(node)}:{now:%Y-%m-%d}",
            f"Credits on {ident(node)} are projected to run out",
            f"{runway['balance']:,.0f} left at {runway['burn_per_min']:.2f} credits/min, about "
            f"{runway['hours_left']:.1f} h (alert horizon {horizon_hours:g} h), from the readings up to "
            f"{stamp(runway['when'])} ({cite(runway['first'])} to {cite(runway['last'])})", runway["when"]))
    return found


# --- all of the rules ----------------------------------------------------------------------------

def scan(rows: Iterable[Row], *, now: dt.datetime, since: dt.datetime, regression_fraction: float = REGRESSION_FRACTION,
         credit_horizon_hours: float = CREDIT_HORIZON_HOURS) -> list[Alert]:
    """Every condition the rules find in the rows as of now, with events from since on, in the order they arose
    (the rule order breaks a tie). Nothing is read or written besides the rows."""
    events = timeline(rows, now)
    pod_alerts, pod_states = replay_pods(events, since)
    down = [node for node, up in pod_states.items() if not up]
    found = [*sitting_errors(events, since), *slot_overruns(events, since, now),
             *metric_regressions(events, since, regression_fraction), *pod_alerts,
             *credit_alerts(events, now, credit_horizon_hours, down)]
    order = {rule: index for index, rule in enumerate(RULES)}
    found.sort(key=lambda alert: (alert.when, order[alert.rule], alert.key))
    return found


# --- the files ---------------------------------------------------------------------------------

_LINE = re.compile(r"- (?P<time>\d\d:\d\d)Z \*\*(?P<title>.+?)\*\* (?P<body>.*) `(?P<key>[^`\s]+)`")


@dataclass(frozen=True)
class ParsedLine:
    """The parts of an alert line: the time of the call that wrote it, the title, the body and the key."""

    time: str
    title: str
    body: str
    key: str


def alerts_dir(state_dir: StateDir = None) -> Path:
    """<state dir>/alerts."""
    return ledger.resolve_state_dir(state_dir) / ALERTS_DIR


def alerts_path(day: str, state_dir: StateDir = None) -> Path:
    """<state dir>/alerts/<day>.md."""
    return alerts_dir(state_dir) / f"{day}.md"


def format_line(alert: Alert, now: dt.datetime) -> str:
    """The line an alert becomes in the day's file, newline included."""
    return f"- {now.astimezone(UTC):%H:%M}Z **{alert.title}** {alert.body} `{alert.key}`\n"


def parse_line(line: str) -> ParsedLine | None:
    """A line of an alerts file as its parts; None for a heading, a blank line or anything else."""
    found = _LINE.fullmatch(line.rstrip("\r\n"))
    return ParsedLine(found["time"], found["title"], found["body"], found["key"]) if found else None


def day_lines(day: str, state_dir: StateDir = None) -> list[str]:
    """The alert lines of one day's file as written, without their newlines; [] when there is no file (nor a
    folder where it would be: some systems answer a path under a plain file with NotADirectoryError)."""
    try:
        text = alerts_path(day, state_dir).read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, NotADirectoryError):
        return []
    return [line for line in text.splitlines() if parse_line(line) is not None]


def announced_keys(first_day: dt.date, last_day: dt.date, state_dir: StateDir = None) -> set[str]:
    """The keys of every alert in the files of the days first_day to last_day, both included."""
    keys: set[str] = set()
    day = first_day
    while day <= last_day:
        for line in day_lines(day.isoformat(), state_dir):
            keys.add(parse_line(line).key)          # type: ignore[union-attr]
        day += dt.timedelta(days=1)
    return keys


def _append(path: Path, text: str, header: str) -> None:
    """Append text to the file, flushed and fsynced; a new file starts with header, and a last line that lost
    its newline is closed first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+b") as fh:
        size = fh.seek(0, os.SEEK_END)
        data = text.encode("utf-8", errors="replace")
        if size == 0:
            data = header.encode("utf-8") + data
        else:
            fh.seek(-1, os.SEEK_END)
            if fh.read(1) != b"\n":
                data = b"\n" + data
        fh.seek(0, os.SEEK_END)
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())


@dataclass(frozen=True)
class Report:
    """What one call found: every condition, the ones announced before, and the new ones."""

    day: str
    since: dt.datetime
    rows: int
    found: tuple[Alert, ...]
    known: tuple[Alert, ...]
    new: tuple[Alert, ...]
    written: bool


def raise_alerts(*, state_dir: StateDir = None, now: dt.datetime | None = None, hours: float = DEFAULT_HOURS,
                 dry_run: bool = False, timeout: float = LOCK_TIMEOUT,
                 regression_fraction: float = REGRESSION_FRACTION,
                 credit_horizon_hours: float = CREDIT_HORIZON_HOURS) -> Report:
    """Scan the ledger and append the alerts not announced yet to the file of now's UTC day. now is an aware
    time (default: the current time); events from now minus hours on are considered. Nothing is written for a
    dry run. ValueError for a naive now or hours that are not positive; LockTimeout or OSError when the file
    or its lock cannot be had."""
    if now is None:
        now = dt.datetime.now(UTC)
    if now.utcoffset() is None:
        raise ValueError(f"now must carry a UTC offset, not the naive {now.isoformat()}")
    if not hours > 0 or not math.isfinite(hours):
        raise ValueError(f"hours must be a positive number, not {hours!r}")
    now = now.astimezone(UTC)
    since = now - dt.timedelta(hours=hours)
    state = ledger.resolve_state_dir(state_dir)
    rows = ledger.read(state_dir=state)
    found = tuple(scan(rows, now=now, since=since, regression_fraction=regression_fraction,
                       credit_horizon_hours=credit_horizon_hours))
    day = f"{now:%Y-%m-%d}"
    path = alerts_path(day, state)
    header = f"# Alerts, {day} (UTC)\n\n"

    def split() -> tuple[tuple[Alert, ...], tuple[Alert, ...]]:
        keys = announced_keys(since.date(), now.date(), state)
        return (tuple(a for a in found if a.key in keys), tuple(a for a in found if a.key not in keys))

    if dry_run:
        known, new = split()
        return Report(day, since, len(rows), found, known, new, False)
    if not found:
        return Report(day, since, len(rows), found, (), (), False)
    with FileLock(state / ALERTS_LOCK, timeout=timeout) as lock:
        lock.write_note(f"pid {os.getpid()} appending alerts")
        known, new = split()
        if new:
            _append(path, "".join(format_line(alert, now) for alert in new), header)
    return Report(day, since, len(rows), found, known, new, bool(new))


# --- the verb --------------------------------------------------------------------------------------

def aware_time(text: str) -> dt.datetime:
    moment = parse_ts(text)
    if moment is None:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not an ISO 8601 time with a UTC offset, such as 2026-10-10T14:00:00Z")
    return moment


def _positive_hours(text: str) -> float:
    try:
        hours = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not hours > 0 or not math.isfinite(hours):
        raise argparse.ArgumentTypeError("hours must be a positive number")
    return hours


def _why(exc: BaseException) -> str:
    """A failure as the answer names it: its type and the system's message, never a path."""
    if isinstance(exc, LockTimeout):
        return "LockTimeout: another writer held the alerts lock for the whole wait"
    if isinstance(exc, OSError):
        return f"{type(exc).__name__}: {exc.strerror or 'not written'}"
    return f"{type(exc).__name__}: {exc}"


@stop.exempt
def main(argv: Sequence[str]) -> dict[str, Any]:
    """Scan the ledger for trouble and append what is new to today's alerts file."""
    parser = argparse.ArgumentParser(prog="python -m lab alerts", description=main.__doc__, allow_abbrev=False)
    parser.add_argument("--hours", type=_positive_hours, default=DEFAULT_HOURS, metavar="H",
                        help=f"how far back to look for conditions (default {DEFAULT_HOURS:g})")
    parser.add_argument("--now", type=aware_time, default=None, metavar="T",
                        help="act as if it were this time (ISO 8601 with a UTC offset): replay a ledger as of T")
    parser.add_argument("--dry-run", action="store_true", help="say what would be announced and write nothing")
    args = parser.parse_args(argv)
    try:
        report = raise_alerts(now=args.now, hours=args.hours, dry_run=args.dry_run)
    except (OSError, LockTimeout) as exc:
        return {"ok": False, "outcome": "error", "error": _why(exc)}
    return {
        "ok": True,
        "outcome": "dry_run" if args.dry_run else "scanned",
        "day": report.day,
        "file": f"{ALERTS_DIR}/{report.day}.md",
        "since": report.since.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rows": report.rows,
        "found": len(report.found),
        "already_announced": len(report.known),
        "new": len(report.new),
        "written": report.written,
        "alerts": [{"rule": alert.rule, "key": alert.key, "title": alert.title} for alert in report.new],
    }
