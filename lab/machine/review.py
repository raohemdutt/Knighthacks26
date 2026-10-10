"""review: the owner's morning file, <state dir>/review/<day>.md.

    python -m lab review [--day YYYY-MM-DD] [--prs-file F | --no-prs] [--no-alerts] [--now T]

The machine's own account of one UTC day (today unless --day says another), written for Ilan to read first
thing and for nobody else: what happened, which numbers moved, what is waiting on him. Ported from the phone
project's agent review (social_outreach/tools/agent/review.py): the file is computed by code from the
machine's records, rewritten whole on every run through a temporary file and os.replace (a second run on one
day replaces the first), and nothing else is written. The ledger is the record; the notes folder (the
handoff notes, for Ilan alone) is never opened.

What goes in, in this order, all of the day asked for (the UTC day of the rows' ts):
- Status: the counts of everything below, and a line saying so when lab/state/STOP exists.
- Alerts: the lines of <state dir>/alerts/<day>.md. When the day is today, the verb first brings that file up
  to date (alerts.raise_alerts), because nothing else may have run the scan; --no-alerts leaves it as it is.
- What happened: the sittings that ended that day with their outcome, length and cost (a sitting that started
  and never ended is named; a cost the launcher did not report reads "cost unknown" and makes the total a
  floor), the other verbs the ledger saw and how they ended, and the tasks the rows were about with the PRs
  and commits their refs name.
- Numbers moved: per series (metric, node and conditions, so a number is only compared with one measured the
  same way) the newest reading of the day against the newest before it, or against the day's first reading
  when nothing came before; readings first seen that day; how many series did not move. Event numbers
  (sitting.*, slot.*) are not series. Seqs are cited as ledger:<seq>.
- GPU slots and complaints: the slots that ran that day against their caps, and every complaint the queue
  recorded (a slot complain row with the outcome complaint_recorded; one the queue refused is not a complaint).
- Rulings waiting: every ruling without an answer in lab/refine/<track>/OWNER_RULINGS.md, read with
  refine_formats.parse_rulings, with the default the passes apply meanwhile.
- PRs open: the open pull requests of the team repository, oldest first, from `gh pr list` (a failure is
  named in the file, never fatal), or from a JSON file in the same shape (--prs-file), or skipped (--no-prs).
  The ledger cannot know when Ilan merges, so it is not asked.
- Problems: what could not be read (a broken rulings file, a failed alert scan).

It runs while the STOP file exists (@stop.exempt), like the original: it reads records and writes the owner's
file. The answer is one JSON line with the day, the file (relative to the state dir) and the counts; the text
of the file is not in it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import subprocess
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lab.machine import alerts, ledger, metrics_report, rulings, stop

__all__ = ["VERB", "REVIEW_DIR", "REFINE_ROOT", "REPO", "GH_COMMAND", "PrsUnavailable", "Review", "review_path",
           "gh_open_prs", "file_open_prs", "build", "review", "main"]

log = logging.getLogger("lab.review")

UTC = dt.timezone.utc

VERB = "review"
REVIEW_DIR = "review"                                   # under the state dir
REFINE_ROOT = rulings.REFINE_DIR
REPO = "raohemdutt/Knighthacks26"
GH_COMMAND = ("gh", "pr", "list", "--repo", REPO, "--state", "open", "--limit", "100", "--json",
              "number,title,headRefName,url,createdAt,isDraft,author")
PR_TIMEOUT = 30.0               # seconds gh may take

STARTED = "started"             # the outcome of a sitting's start row (lab sit)
NONE_LINE = "- none"
LINE_CHARS = 200                # how much of a free-text field a line shows
TITLE_CHARS = 100

# lines listed before "+N more"
MAX_SITTINGS = 30
MAX_ACTIVITY = 12
MAX_TASKS = 15
MAX_MOVED = 15
MAX_NEW = 10
MAX_SLOTS = 15
MAX_COMPLAINTS = 15
MAX_RULINGS = 15
MAX_PRS = 15

HEAD_ALERTS = "## Alerts"
HEAD_HAPPENED = "## What happened"
HEAD_SITTINGS = "### Sittings"
HEAD_ACTIVITY = "### Other activity"
HEAD_TASKS = "### Tasks touched"
HEAD_MOVED = "## Numbers moved"
HEAD_GPU = "## GPU slots and complaints"
HEAD_SLOTS = "### Slots"
HEAD_COMPLAINTS = "### Complaints"
HEAD_RULINGS = "## Rulings waiting"
HEAD_PRS = "## PRs open"
HEAD_PROBLEMS = "## Problems"

Row = dict[str, Any]
Event = tuple[dt.datetime, Row]
StateDir = str | os.PathLike[str] | None


class PrsUnavailable(RuntimeError):
    """The open pull requests could not be listed; the message says why, without any output of gh."""


# --- small helpers -----------------------------------------------------------------------------

def review_path(day: str, state_dir: StateDir = None) -> Path:
    """<state dir>/review/<day>.md."""
    return ledger.resolve_state_dir(state_dir) / REVIEW_DIR / f"{day}.md"


def _clip(text: Any, limit: int = LINE_CHARS) -> str:
    """text on one line, cut at limit with "..."."""
    one = " ".join(str(text).split())
    return one if len(one) <= limit else one[: limit - 3] + "..."


def _count(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def _more(shown: list[str], total: int) -> list[str]:
    return shown + ([f"- +{total - len(shown)} more"] if total > len(shown) else [])


def _clock(moment: dt.datetime) -> str:
    return f"{moment:%H:%M:%S}Z"


def _refs(row: Mapping[str, Any]) -> Mapping[str, Any]:
    refs = row.get("refs")
    return refs if isinstance(refs, Mapping) else {}


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _day_bounds(day: str) -> tuple[dt.datetime, dt.datetime]:
    start = dt.datetime.combine(dt.date.fromisoformat(day), dt.time(), tzinfo=UTC)
    return start, start + dt.timedelta(days=1)


# --- sittings, activity, tasks ---------------------------------------------------------------------

def _sitting_id(row: Mapping[str, Any]) -> str:
    run = _refs(row).get("run_id")
    return str(run) if run else f"seq-{row.get('seq')}"


def _who(row: Row) -> str:
    """A sitting row's id: the run id the launcher gave it, else the actor that wrote the row."""
    return alerts.ident(_refs(row).get("run_id") or row.get("actor"))


def _item(row: Row) -> str:
    """The tier and the task of a row for a line: "opus, task O20", "sonnet, no task"."""
    task = _text(row.get("task"))
    return f"{alerts.label(row.get('tier'), 'no tier')}, {'task ' + alerts.ident(task) if task else 'no task'}"


def _sitting_line(when: dt.datetime, row: Row) -> str:
    seconds = alerts.metric_value(row, "sitting.duration_s")
    cost = alerts.metric_value(row, "sitting.cost_usd")
    length = "duration unknown" if seconds is None else f"{seconds / 60:.1f} min"
    price = "cost unknown" if cost is None else f"${cost:.2f}"
    return f"- {_clock(when)} {_who(row)} ({_item(row)}): {row.get('outcome')}, {length}, {price} ({alerts.cite(row)})"


def sitting_section(day_events: Sequence[Event], all_events: Sequence[Event]) -> tuple[list[str], dict[str, Any]]:
    """The day's ended sittings, the ones that started that day and never ended, and the counts."""
    ended_ever = {_sitting_id(row) for _, row in all_events
                  if row.get("verb") == alerts.SIT_VERB and row.get("outcome") != STARTED}
    ended = [(when, row) for when, row in day_events
             if row.get("verb") == alerts.SIT_VERB and row.get("outcome") != STARTED]
    open_ones = [(when, row) for when, row in day_events
                 if row.get("verb") == alerts.SIT_VERB and row.get("outcome") == STARTED
                 and _sitting_id(row) not in ended_ever]
    lines = _more([_sitting_line(when, row) for when, row in ended[:MAX_SITTINGS]], len(ended))
    for when, row in open_ones:
        lines.append(f"- {_clock(when)} {_who(row)} ({_item(row)}): started, no end row ({alerts.cite(row)})")
    costs = [alerts.metric_value(row, "sitting.cost_usd") for _, row in ended]
    known = [cost for cost in costs if cost is not None]
    outcomes = Counter(str(row.get("outcome")) for _, row in ended)
    return lines, {
        "sittings": len(ended),
        "outcomes": dict(sorted(outcomes.items(), key=lambda item: (-item[1], item[0]))),
        "never_ended": len(open_ones),
        "cost_usd": round(sum(known), 4),
        "cost_unknown": len(costs) - len(known),
    }


def activity_section(day_events: Sequence[Event]) -> list[str]:
    """The other verbs of the day, most frequent first, each with how its calls ended."""
    by_verb: dict[str, Counter[str]] = {}
    for _, row in day_events:
        if row.get("verb") != alerts.SIT_VERB:
            by_verb.setdefault(str(row.get("verb")), Counter())[str(row.get("outcome"))] += 1
    ordered = sorted(by_verb.items(), key=lambda item: (-sum(item[1].values()), item[0]))
    lines = [f"- {verb}: {sum(done.values())} ("
             + ", ".join(f"{outcome} {n}" for outcome, n in sorted(done.items(), key=lambda item: (-item[1], item[0])))
             + ")" for verb, done in ordered]
    return _more(lines[:MAX_ACTIVITY], len(lines))


def tasks_section(day_events: Sequence[Event]) -> list[str]:
    """The tasks the day's rows were about, the latest activity first, with the PRs and commits named."""
    tasks: dict[str, dict[str, Any]] = {}
    for when, row in day_events:
        task = _text(row.get("task"))
        if not task:
            continue
        entry = tasks.setdefault(task, {"rows": 0, "last": when, "prs": set(), "commits": set()})
        entry["rows"] += 1
        entry["last"] = max(entry["last"], when)
        refs = _refs(row)
        if refs.get("pr") is not None:
            entry["prs"].add(str(refs["pr"]))
        if refs.get("commit") is not None:
            entry["commits"].add(str(refs["commit"]))
    lines = []
    for task, entry in sorted(tasks.items(), key=lambda item: (-item[1]["last"].timestamp(), item[0])):
        parts = [_count(entry["rows"], "row", "rows")]
        parts += [f"PR {pr}" for pr in sorted(entry["prs"])]
        parts += [f"commit {commit}" for commit in sorted(entry["commits"])]
        lines.append(f"- {alerts.ident(task)}: {', '.join(parts)}")
    return _more(lines[:MAX_TASKS], len(lines))


# --- numbers moved -----------------------------------------------------------------------------

@dataclass
class _Moves:
    moved: list[tuple[float, str]] = field(default_factory=list)       # (absolute change in percent, line)
    new: list[str] = field(default_factory=list)
    unchanged: int = 0


def moved_section(all_events: Sequence[Event], start: dt.datetime,
                  end: dt.datetime) -> tuple[list[str], dict[str, int]]:
    """What the day's readings did against the ones before: moved, new and unchanged series."""
    series: dict[tuple[str, str | None, str | None, str], list[tuple[dt.datetime, Row, float]]] = {}
    for when, row in all_events:
        metrics = row.get("metrics")
        if when >= end or not isinstance(metrics, Mapping):
            continue
        for name in metrics:
            value = alerts.number(metrics[name])
            if value is not None and isinstance(name, str) and not name.startswith(alerts.EVENT_METRIC_PREFIXES):
                series.setdefault(alerts.series_key(name, row), []).append((when, row, value))
    result = _Moves()
    for (name, node, _unit, _setup), readings in series.items():
        today = [reading for reading in readings if start <= reading[0] < end]
        if not today:
            continue
        earlier = [reading for reading in readings if reading[0] < start]
        after = today[-1]
        if earlier:
            before = earlier[-1]
        elif len(today) > 1:
            before = today[0]
        else:
            row, value = after[1], after[2]
            unit = alerts.reading_unit(row, name)
            result.new.append(f"- {alerts.ident(name)} ({_where(node, row)}): first reading {value:g}"
                              f"{' ' + unit if unit else ''}{_confidence(row)} ({alerts.cite(row)})")
            continue
        if before[2] == after[2]:
            result.unchanged += 1
            continue
        unit = alerts.reading_unit(after[1], name)
        change = (after[2] - before[2]) / abs(before[2]) * 100 if before[2] != 0 else None
        direction = metrics_report.direction_of(name, unit)
        if direction is None:
            word = "up" if after[2] > before[2] else "down"
        else:
            better = after[2] > before[2] if direction == metrics_report.HIGHER else after[2] < before[2]
            word = "better" if better else "worse"
        low = before[1].get("low_confidence") is True or after[1].get("low_confidence") is True
        result.moved.append((abs(change) if change is not None else -1.0,
                             f"- {alerts.ident(name)} ({_where(node, after[1])}): {before[2]:g} -> {after[2]:g}"
                             f"{' ' + unit if unit else ''} "
                             f"({'from 0' if change is None else f'{change:+.1f}%'}, {word})"
                             f"{' (low confidence)' if low else ''} "
                             f"({alerts.cite(before[1])} -> {alerts.cite(after[1])})"))
    result.moved.sort(key=lambda item: (-item[0], item[1]))
    lines = _more([line for _, line in result.moved[:MAX_MOVED]], len(result.moved))
    result.new.sort()
    if result.new:
        lines += ["", "First readings:"] + _more(result.new[:MAX_NEW], len(result.new))
    if result.unchanged:
        lines += ["", f"- {_count(result.unchanged, 'series', 'series')} did not change"]
    lines = lines[1:] if lines[:1] == [""] else lines
    counts = {"moved": len(result.moved), "new": len(result.new), "unchanged": result.unchanged}
    return lines, counts


def _where(node: str | None, row: Row) -> str:
    conditions = alerts.compact_conditions(row)
    return alerts.label(node, "no node") + (f"; {conditions}" if conditions else "")


def _confidence(row: Row) -> str:
    return " (low confidence)" if row.get("low_confidence") is True else ""


# --- GPU slots and complaints -------------------------------------------------------------------

def slot_lines(all_events: Sequence[Event], now: dt.datetime, start: dt.datetime, end: dt.datetime) -> list[str]:
    """The slots that began, or ended, that day, with their time against their cap."""
    lines = []
    for run in alerts.slot_runs(all_events, min(now, end)):
        if not (start <= run.start < end or (run.end is not None and start <= run.end < end)):
            continue
        state = "still running" if run.end is None else "done"
        extra = " (extended)" if run.extended else ""
        flag = ", past its cap" if run.over_cap else ""
        lines.append(f"- slot {alerts.ident(run.slot)} on {alerts.label(run.node, 'no node')}, "
                     f"{'task ' + alerts.ident(run.task) if run.task else 'no task'}: {run.elapsed:.0f} of "
                     f"{run.cap:.0f} min{extra}, {state}{flag} (ledger:{run.start_seq})")
    return _more(lines[:MAX_SLOTS], len(lines))


def complaint_lines(day_events: Sequence[Event]) -> tuple[list[str], int]:
    """Every complaint the queue recorded that day (alerts.slot_happened: a refused slot complain, such as one
    answered too_early or not_waiting, filed nothing), with the complaint's first line."""
    lines = []
    for when, row in day_events:
        if row.get("verb") != alerts.SLOT_COMPLAIN or not alerts.slot_happened(row):
            continue
        slot = _refs(row).get("slot")
        what = _clip(_text(row.get("note")) or _text(row.get("complaint")) or "(no text)")
        lines.append(f"- {_clock(when)} slot {alerts.ident(slot)} ({alerts.ident(row.get('actor'))}): {what} "
                     f"({alerts.cite(row)})")
    return _more(lines[:MAX_COMPLAINTS], len(lines)), len(lines)


# --- rulings and pull requests --------------------------------------------------------------------

def rulings_section(refine_root: Path) -> tuple[list[str], int, list[str]]:
    """The rulings still waiting on the owner (those the rulings verb lists as open, in its track order), and
    what could not be read: a track whose rulings file breaks the format."""
    standings, errors, _ = rulings.collect(refine_root)
    waiting = [item for item in standings if not item.answered]
    lines = [f"- {item.track} ruling {item.number}: {_clip(item.question)} "
             f"(applied meanwhile: {_clip(item.now, 120)}; recommended: {_clip(item.recommend, 120)})"
             for item in waiting]
    problems = [f"{error['track']}/{rulings.RULINGS_NAME} could not be read: "
                f"{_clip(error['error'].split(': ', 1)[-1])}" for error in errors]
    return _more(lines[:MAX_RULINGS], len(lines)), len(lines), problems


def gh_open_prs(*, run: Callable[..., Any] = subprocess.run, timeout: float = PR_TIMEOUT) -> list[dict[str, Any]]:
    """The open pull requests of the team repository, from gh. PrsUnavailable when gh is missing, fails, times
    out or prints something that is not a JSON list."""
    try:
        done = run(GH_COMMAND, capture_output=True, encoding="utf-8", errors="replace", timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PrsUnavailable(f"gh could not run: {type(exc).__name__}") from None
    if done.returncode != 0:
        raise PrsUnavailable(f"gh pr list exited with code {done.returncode}")
    return _pr_list(done.stdout)


def _pr_list(text: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(text)
    except ValueError:
        raise PrsUnavailable("the pull request list is not JSON") from None
    if not isinstance(data, list):
        raise PrsUnavailable(f"the pull request list is {type(data).__name__}, not a list")
    return [item for item in data if isinstance(item, dict)]


def file_open_prs(path: str | os.PathLike[str]) -> Callable[[], list[dict[str, Any]]]:
    """A source of open pull requests that reads a JSON file in the shape gh pr list prints."""

    def read() -> list[dict[str, Any]]:
        try:
            return _pr_list(Path(path).read_text(encoding="utf-8"))
        except OSError as exc:
            raise PrsUnavailable(f"the pull request file could not be read: {type(exc).__name__}") from None

    return read


def pr_lines(prs: Sequence[Mapping[str, Any]], now: dt.datetime) -> list[str]:
    """Open pull requests, the longest open first."""

    def opened(pr: Mapping[str, Any]) -> dt.datetime | None:
        return alerts.parse_ts(pr.get("createdAt"))

    ordered = sorted(prs, key=lambda pr: (opened(pr) or now, str(pr.get("number"))))
    lines = []
    for pr in ordered[:MAX_PRS]:
        author = pr.get("author")
        login = author.get("login") if isinstance(author, Mapping) else None
        created = opened(pr)
        age = f", open {max(0.0, (now - created).total_seconds() / 3600):.1f} h" if created else ""
        branch = _clip(pr.get("headRefName") or "no branch", 60)
        lines.append(f"- #{pr.get('number')} {_clip(pr.get('title') or '(no title)', TITLE_CHARS)} "
                     f"({branch}{', draft' if pr.get('isDraft') is True else ''}), by {_clip(login or 'unknown', 40)}"
                     f"{age}, {_clip(pr.get('url') or 'no url', 120)}")
    return _more(lines, len(ordered))


# --- the file --------------------------------------------------------------------------------------

@dataclass
class Review:
    """A composed review: the file's text, the counts the verb answers with, and what could not be read."""

    day: str
    text: str
    counts: dict[str, Any]
    problems: list[str]


def _stop_line(state: Path) -> str | None:
    """The line for a stopped machine, from the STOP file of this state dir; None when there is none. A file
    that cannot be read still counts as stopped."""
    path = state / stop.STOP_NAME
    try:
        os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError:
        return "The machine is stopped (the STOP file could not be checked)."
    try:
        reason = _clip(path.read_text(encoding="utf-8", errors="replace"), 300)
    except OSError:
        reason = ""
    return f"The machine is stopped: {reason or 'no reason given'}."


def build(day: str, *, state_dir: StateDir = None, refine_root: Path | None = None,
          prs: Callable[[], list[dict[str, Any]]] | None = None, now: dt.datetime | None = None,
          scan: bool = True) -> Review:
    """Compose the review of one UTC day from the ledger, the alerts file, the rulings files and the pull request
    source (None: skipped), as of now (default: the current time). With scan, and day being now's day, the
    alerts file is first brought up to date. Writes nothing but, through the scan, the alerts file. ValueError
    for a day that is not YYYY-MM-DD or a naive now."""
    now = (now or dt.datetime.now(UTC))
    if now.utcoffset() is None:
        raise ValueError(f"now must carry a UTC offset, not the naive {now.isoformat()}")
    now = now.astimezone(UTC)
    start, end = _day_bounds(day)
    state = ledger.resolve_state_dir(state_dir)
    problems: list[str] = []

    if scan and start <= now < end:
        try:
            alerts.raise_alerts(state_dir=state, now=now)
        except Exception as exc:    # the review is the owner's last resort: whatever fails, the file is written
            log.error("review: the alert scan failed: %s", exc, exc_info=True)
            problems.append(f"the alert scan failed: {type(exc).__name__}: {_clip(exc)}")

    all_events = alerts.timeline(ledger.read(state_dir=state), now)
    day_events = [event for event in all_events if start <= event[0] < end]
    alert_lines = []
    try:
        written = alerts.day_lines(day, state)
    except OSError as exc:
        problems.append(f"the alerts file could not be read: {type(exc).__name__}: {exc.strerror or 'not read'}")
        written = []
    for line in written:
        parsed = alerts.parse_line(line)
        alert_lines.append(f"- {parsed.time}Z **{parsed.title}** {parsed.body}")     # type: ignore[union-attr]

    sittings, sitting_counts = sitting_section(day_events, all_events)
    moved, moved_counts = moved_section(all_events, start, end)
    slots = slot_lines(all_events, now, start, end)
    complaints, complaint_total = complaint_lines(day_events)
    rulings, ruling_total, ruling_problems = rulings_section(refine_root if refine_root is not None else REFINE_ROOT)
    problems += ruling_problems

    pr_total: int | None = None
    if prs is None:
        pr_body = ["- not asked for"]
    else:
        try:
            listed = prs()
        except PrsUnavailable as exc:
            pr_body = [f"- could not be listed: {_clip(exc)}"]
        else:
            pr_total = len(listed)
            pr_body = pr_lines(listed, now) or [NONE_LINE]

    outcomes = ", ".join(f"{n} {outcome}" for outcome, n in sitting_counts["outcomes"].items())
    unknown = sitting_counts["cost_unknown"]
    cost = f"${sitting_counts['cost_usd']:.2f}" if not unknown else f"at least ${sitting_counts['cost_usd']:.2f}"
    status = ", ".join((
        _count(len(day_events), "ledger row", "ledger rows"),
        _count(sitting_counts["sittings"], "sitting ended", "sittings ended") + (f" ({outcomes})" if outcomes else ""),
        f"costing {cost}" + (f" ({unknown} cost unknown)" if unknown else ""),
        _count(len(alert_lines), "alert", "alerts"),
        _count(moved_counts["moved"], "metric moved", "metrics moved"),
        _count(ruling_total, "ruling waiting", "rulings waiting"),
        _count(complaint_total, "complaint", "complaints"),
        "PRs open unknown" if pr_total is None else _count(pr_total, "PR open", "PRs open"),
    ))
    lines = [f"# Lab review, {day} (UTC)", "", f"Status: {status}."]
    stopped = _stop_line(state)
    if stopped:
        lines += ["", stopped]
    if sitting_counts["never_ended"]:
        lines += ["", f"{_count(sitting_counts['never_ended'], 'sitting started', 'sittings started')} that day and "
                      "never ended; the launcher may have died."]
    for heading, body in (
        (HEAD_ALERTS, alert_lines),
        (HEAD_HAPPENED, None),
        (HEAD_SITTINGS, sittings),
        (HEAD_ACTIVITY, activity_section(day_events)),
        (HEAD_TASKS, tasks_section(day_events)),
        (HEAD_MOVED, moved),
        (HEAD_GPU, None),
        (HEAD_SLOTS, slots),
        (HEAD_COMPLAINTS, complaints),
        (HEAD_RULINGS, rulings),
        (HEAD_PRS, pr_body),
        (HEAD_PROBLEMS, [f"- {problem}" for problem in problems]),
    ):
        lines += ["", heading]
        if body is not None:
            lines += [""] + (body or [NONE_LINE])
    counts = {
        "rows": len(day_events),
        "sittings": sitting_counts["sittings"],
        "sittings_never_ended": sitting_counts["never_ended"],
        "alerts": len(alert_lines),
        "metrics_moved": moved_counts["moved"],
        "metrics_new": moved_counts["new"],
        "rulings_waiting": ruling_total,
        "complaints": complaint_total,
        "prs_open": pr_total,
        "cost_usd": sitting_counts["cost_usd"],
    }
    if unknown:
        counts["cost_unknown"] = unknown
    return Review(day, "\n".join(lines) + "\n", counts, problems)


def _write(path: Path, text: str) -> None:
    """Through a fresh temporary file beside it and os.replace; the temporary file is removed when the write
    or the replace fails, and the failure is raised."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError as exc:
            log.error("could not remove the temporary review file %s: %s", tmp, exc)
        raise


def review(day: str | None = None, *, state_dir: StateDir = None, refine_root: Path | None = None,
           prs: Callable[[], list[dict[str, Any]]] | None = None, now: dt.datetime | None = None,
           scan: bool = True) -> dict[str, Any]:
    """Write the day's review file and answer what the verb prints. See build for the arguments."""
    now = (now or dt.datetime.now(UTC))
    if now.utcoffset() is None:
        raise ValueError(f"now must carry a UTC offset, not the naive {now.isoformat()}")
    now = now.astimezone(UTC)
    day = day or f"{now:%Y-%m-%d}"
    built = build(day, state_dir=state_dir, refine_root=refine_root, prs=prs, now=now, scan=scan)
    path = review_path(day, state_dir)
    _write(path, built.text)
    log.info("review: wrote %s", path)
    return {"ok": True, "outcome": "written", "day": day, "file": f"{REVIEW_DIR}/{day}.md", "counts": built.counts,
            "problems": len(built.problems)}


# --- the verb --------------------------------------------------------------------------------------

def _day_argument(text: str) -> str:
    try:
        return dt.date.fromisoformat(text).isoformat()
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a date (YYYY-MM-DD)") from None


@stop.exempt
def main(argv: Sequence[str]) -> dict[str, Any]:
    """Write the owner's morning file: what happened, numbers moved, rulings, complaints and open PRs."""
    parser = argparse.ArgumentParser(prog="python -m lab review", description=main.__doc__, allow_abbrev=False)
    parser.add_argument("--day", type=_day_argument, default=None, metavar="YYYY-MM-DD",
                        help="the UTC day to review (default today)")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--prs-file", default=None, metavar="FILE",
                        help="read the open pull requests from this JSON file (the shape gh pr list prints) "
                             "instead of asking gh")
    source.add_argument("--no-prs", action="store_true", help="leave the pull request list out")
    parser.add_argument("--no-alerts", action="store_true",
                        help="do not bring today's alerts file up to date before reading it")
    parser.add_argument("--now", type=alerts.aware_time, default=None, metavar="T",
                        help="act as if it were this time (ISO 8601 with a UTC offset): review a ledger as of T")
    args = parser.parse_args(argv)
    prs = None if args.no_prs else file_open_prs(args.prs_file) if args.prs_file else gh_open_prs
    try:
        return review(args.day, prs=prs, now=args.now, scan=not args.no_alerts)
    except OSError as exc:
        return {"ok": False, "outcome": "error", "error": f"{type(exc).__name__}: {exc.strerror or 'not written'}"}
