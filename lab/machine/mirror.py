"""mirror: the lab's account of itself on Notion, written by code from the digest and the refinement logs.

    python -m lab mirror                         both jobs, against Notion (needs NOTION_TOKEN)
    python -m lab mirror --only overnight        only the Ilan section of Overnight Status
    python -m lab mirror --only plan-changes     only the Plan changes lines on Ilan's Team Comms page
    python -m lab mirror --print                 touch nothing; return the exact texts and where they go
    python -m lab mirror --hours 12 | --since <time>     the digest window (default: the 12 hours before now)
    python -m lab mirror --board <file> --refine-dir <folder>    other inputs than the stored ones

The team reads Notion and the machine writes the ledger; this verb is the bridge. It does two jobs, and
each is judged on its own.

Overnight Status. The page keeps one section per person; "## Ilan" is rewritten from the digest
(lab/machine/digest.py), and nothing outside that section is ever touched. Its five bullets read:

    Last update    the signed time, the digest window and the board snapshot it used, and the id of the facts
    Working on     Ilan's tasks In progress (board snapshot) and the GPU rows Running
    Done tonight   Ilan's tasks that are Done on the board and had ledger rows in the window
    Blocked on     Ilan's tasks that are Blocked, GPU queue waits, overruns and complaints, failed ledger rows
                   of the window, and tasks held back because a blocker is missing from the snapshot
    Needs a human  this flow's segment: tasks Ilan shares with a teammate, which no sitting takes

The first four bullets belong to this flow and are replaced whole. The Needs a human bullet is shared: it
also carries the "Lab rulings waiting on you" segment that `python -m lab rulings mirror` keeps there, and
whatever else a person or another agent wrote. Only the segment this flow wrote (the signed prefix, "Lab
digest:", and a closing "(digest <8 hex digits>)") is ever replaced, in place; everything else on the line is
kept as it stands, the rulings segment included. The answer says whether that rulings segment is current,
stale or missing, so a stale one can be refreshed with `python -m lab rulings mirror`.

Plan changes. Every board change a refinement pass or the merge pass makes is one line on the "Plan changes"
item under "## FYI" on Ilan's Team Comms page (lab/refine/PASS_RULES.md, "Telling Ilan what changed"). The
passes post their own lines as they work; this verb makes sure each change in lab/refine/<track>/REFINE_LOG.md
and lab/refine/MERGE_LOG.md has its line, once. A line has the shape the rules give,
`<track> pass <N>, <task ID>: <what changed>. Was: "<before>". Why: <reason>. Source: <citation>.`, with the
signed prefix, and goes after the item's last child (the item is created first when it is missing). A change
counts as posted when the item already has a line, from a pass or from an earlier run, that starts with the
same `<label>, <task ID>:` after its signed prefix; each such line covers one change of that label and task,
the earliest first, and a "reverted change" line covers none.

Idempotent. The facts of the section are named by an id (8 hex digits over the four bodies, not the clock or
the ledger's size), shown on the Last update bullet. A page that already shows the current id, with the Needs
bullet in its place, is left alone; a changed digest replaces the earlier text in place. The verb's own ledger
row (verb mirror) and its own failures are left out of the facts, so running it never changes them.

Notion. Page text is read and edited through the page-markdown boundary of lab/machine/rulings.py
(NotionPages, and RestNotionClient over NOTION_TOKEN); no second client is written. An edit is one exact
replacement anchored on the "## Ilan" heading or on the end of the Plan changes item, which Notion refuses
unless the text occurs exactly once, and the page is read again afterwards to see the change. The tests run
the same code against the in-memory pages of lab/machine/pages_fake.py.

No token. NOTION_TOKEN is looked up in the process environment, then in the .env files of ready.py. Without
it nothing is sent anywhere: the answer is outcome no_token with the exact lines to write and the rules to
apply them with, for a sitting to do through its Notion tools, and the exit code is 2. --print returns the same
texts as outcome printed and exits 0.

Answer and exit code. One JSON object: ok, outcome (ok, failed, no_token, printed, error, usage), mode (live or
print), section_id, the window, targets (overnight_status and plan_changes, each with its action: posted,
updated, unchanged or failed), rulings, warnings and digest_problems, and ledger_seq. Exit 0 when it worked, 1
when a target or an input failed, 2 for bad arguments and for a missing token. Each call that gets as far as a
decision appends one row to the ledger (verb mirror). While lab/state/STOP exists the verb is refused before it
runs. The token is never printed, logged or written to the ledger.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import logging
import os
import re
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

from lab.machine import digest as digest_mod
from lab.machine import ledger as ledger_mod
from lab.machine import ready, record, refine_formats, refine_track, rulings, stop
from lab.machine.digest import Board

__all__ = [
    "MirrorError", "PlanChange", "Rendered", "Section", "active_tasks", "build_section", "collect_plan_changes",
    "main", "overnight_edit", "plan_edit", "print_plan", "render", "rewrite_needs", "rulings_segment_state", "run",
]

log = logging.getLogger("lab.mirror")

UTC = dt.timezone.utc
VERB = "mirror"
ENV_TOKEN = rulings.TOKEN_ENV

OVERNIGHT_PAGE = rulings.OVERNIGHT_PAGE
OVERNIGHT_URL = rulings.OVERNIGHT_URL
TEAM_COMMS_PAGE = rulings.TEAM_COMMS_PAGE
TEAM_COMMS_URL = rulings.TEAM_COMMS_URL

LOG_NAME = rulings.LOG_NAME
MERGE_LOG_NAME = "MERGE_LOG.md"

#: The digest window of a mirror run: what "tonight" means unless --hours or --since says otherwise.
WINDOW_HOURS = 12.0

# The five bullets of an Ilan section, by the label inside the bold marks.
LAST, WORKING, DONE, BLOCKED, NEEDS = "Last update", "Working on", "Done tonight", "Blocked on", "Needs a human (Ilan)"
LABELS = (LAST, WORKING, DONE, BLOCKED, NEEDS)

#: What this flow's segment of the Needs bullet starts with, after the signed prefix.
SEGMENT_PHRASE = "Lab digest:"
#: The item that holds the plan-change lines, as lab/refine/PASS_RULES.md and the pass skills write it.
PLAN_ITEM = "- **Plan changes** (one line per board change a refinement pass makes; undo a line by restoring its Was value)"

MAX_LISTED = 5       # task ids named in a list before "and N more"
MAX_FAILURES = 3     # failed rows named before "and N more"
CLIP_WHAT = 160      # characters of a change's heading in its plan-change line
CLIP_WAS = 900       # of its BEFORE text: Ilan undoes a change by pasting this back
CLIP_WHY = 300
CLIP_SOURCE = 300
CLIP_NOTE = 200      # of a free-text field in an error

_TIME = r" [A-Z][a-z]{2} [0-9]{1,2}:[0-9]{2} [AP]M"
_STAMP = re.escape(rulings.SIGNER) + _TIME + " ET:"
_LOOSE_STAMP = re.escape(rulings.SIGNER) + _TIME + "(?: ET)?:"  # a pass may leave out the ET
_DIGEST_MARK = re.compile(r"\(digest (?P<id>[0-9a-f]{8})\)")
_OWN_SEGMENT = re.compile(_STAMP + " " + re.escape(SEGMENT_PHRASE) + r".*?" + _DIGEST_MARK.pattern)
_RULINGS_SEGMENT = re.compile(_STAMP + " " + re.escape(rulings.PHRASE) + r".*?\(list (?P<id>[0-9a-f]{8})\)")
_ILAN_HEADING = re.compile(r"^##[ \t]+Ilan[ \t]*$")
_HEADING = re.compile(r"^#{1,6}[ \t]")
_FYI_HEADING = re.compile(r"^##[ \t]+FYI\b")
_BULLET = re.compile(r"^(?P<indent>[ \t]*)- \*\*(?P<label>[^*\n]+?):\*\*(?P<body>.*)$")
_NEEDS_LINE = re.compile(r"^(?P<head>[ \t]*- \*\*" + re.escape(NEEDS) + r":\*\*)(?P<body>.*)$")
_PLAN_ITEM = re.compile(r"^(?P<indent>[ \t]*)- \*\*Plan changes\*\*")
_PLAN_CHILD = re.compile(
    r"^(?P<indent>[ \t]+)- (?:" + _LOOSE_STAMP + r" )?"
    r"(?P<label>[a-z][a-z0-9-]* pass [0-9]+|merge [0-9]+), (?P<task>[A-Za-z0-9]+): (?P<rest>.*)$"
)
_TASK_ID = re.compile(r"\b[A-Z]{1,3}[0-9]{1,3}[a-z]?\b")
_ID_PARTS = re.compile(r"^([A-Za-z]+)([0-9]+)(.*)$")
_EMPTY_BODY = frozenset({"", "-", "\u2014", "\u2013", "none", "nothing"})


class MirrorError(Exception):
    """A page that does not have the shape this flow edits; the message says what is missing."""


# ---------------------------------------------------------------------------------------------
# Small text helpers


def _et(moment: dt.datetime) -> str:
    """The moment as the day and Eastern time the pages use: Sat 11:15 AM ET."""
    et = rulings.eastern(moment)
    return f"{rulings.DAYS[et.weekday()]} {et.hour % 12 or 12}:{et.minute:02d} {'AM' if et.hour < 12 else 'PM'} ET"


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" + ("" if count == 1 else "s")


def _id_key(task_id: str) -> tuple[str, int, str]:
    found = _ID_PARTS.match(task_id)
    return (found[1].upper(), int(found[2]), found[3]) if found else (task_id.upper(), -1, "")


def _listed(items: Sequence[str], cap: int = MAX_LISTED) -> str:
    """The first cap items, then how many were left out."""
    shown = ", ".join(items[:cap])
    return f"{shown}, and {len(items) - cap} more" if len(items) > cap else shown


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",;:- ") + " ..."


def _bare(text: str) -> str:
    """The text without the full stop or semicolon it ends on, so a sentence can add its own."""
    return " ".join(str(text).split()).rstrip(" .;:")


def _lane(board: Board) -> list[Any]:
    """Ilan's tasks: any task he is assigned to, as the digest counts them."""
    return [task for task in board.tasks if any(digest_mod.is_ilan(who) for who in task.assignees)]


def _sorted_ids(tasks: Sequence[Any]) -> list[str]:
    return [task.id for task in sorted(tasks, key=lambda task: _id_key(task.id))]


# ---------------------------------------------------------------------------------------------
# The Ilan section of Overnight Status, from the digest


@dataclasses.dataclass(frozen=True)
class Section:
    """What this flow says about the lab in the Ilan section, before it is signed and laid out.

    working, done and blocked are the bodies of their bullets and needs is this flow's part of the Needs bullet
    ("" when it has nothing to add), all in Notion's markup. section_id names exactly these four texts."""

    section_id: str
    since: dt.datetime
    until: dt.datetime
    board: str
    working: str
    done: str
    blocked: str
    needs: str


@dataclasses.dataclass(frozen=True)
class Rendered:
    """The section laid out for the page: the four whole bullets, and the segment for the Needs bullet (None
    when this flow has nothing to say there)."""

    last: str
    working: str
    done: str
    blocked: str
    segment: str | None


def active_tasks(since: dt.datetime, until: dt.datetime, *, state_dir: str | os.PathLike[str] | None = None) -> set[str]:
    """The task ids of the ledger rows in [since, until], not counting this verb's own rows."""
    rows = ledger_mod.read(state_dir=state_dir, since=since, before=until + dt.timedelta(seconds=1))
    return {str(row["task"]).strip() for row in rows if row.get("task") and row.get("verb") != VERB}


def _gpu_parts(gpu: Mapping[str, Any] | None) -> tuple[list[str], list[str], int]:
    """(running rows for Working on, waits and overruns for Blocked on, open complaints)."""
    if not gpu:
        return [], [], 0
    running_items = gpu["running"]["items"]
    working = [f"GPU {row['gpu']}: {row['task'] or digest_mod.clip(row['request'], 40)} running" for row in running_items[:3]]
    blocked = []
    for card in gpu["summary"]:
        over = [row["task"] or "a row" for row in running_items if row["gpu"] == card["gpu"] and row["over_min"] > 0]
        pieces = []
        if over:
            pieces.append(f"{_listed(over)} past its cap")
        if card["queued"]:
            pieces.append(f"{card['queued']} queued")
        if pieces:
            blocked.append(f"GPU {card['gpu']}: {', '.join(pieces)}")
    complaints = gpu["complaints"]
    return working, blocked, len(complaints["items"]) + complaints["more"]


def build_section(digest: Mapping[str, Any], board: Board | None, active: Collection[str]) -> Section:
    """The facts of the Ilan section from a digest (digest.compute), the board snapshot it was computed from
    and the tasks that had ledger rows in its window."""
    since = digest_mod.parse_time(digest["window"]["since"])[0]
    until = digest_mod.parse_time(digest["window"]["until"])[0]
    assert since is not None and until is not None, "the digest window must be readable"
    seen = digest.get("board")
    if seen is None or board is None:
        board_text = "missing"
    else:
        taken = digest_mod.parse_time(seen["taken_at"])[0]
        board_text = f"from {_et(taken)}" + (", stale" if seen["stale"] else "") if taken else "of unknown time"
    gpu_working, gpu_blocked, complaints = _gpu_parts(digest.get("gpu"))
    held = digest.get("held") or {}
    failures = [row for row in digest["ran"]["failures"]["items"] if row["verb"] != VERB]
    failed_total = len(failures) + digest["ran"]["failures"]["more"]

    working: list[str]
    done: str
    blocked: list[str] = []
    if board is None:
        hint = "`python -m lab digest --save-board <file>` stores one"
        working = [f"no board snapshot, so the tasks In progress are unknown ({hint})"]
        done = "unknown without a board snapshot"
        if active:
            working.append(f"ledger rows in the window for {_listed(sorted(active, key=_id_key))}")
    else:
        mine = _lane(board)
        in_progress = sorted((t for t in mine if t.status == "In progress"), key=lambda t: _id_key(t.id))
        labels = [t.id if t.tier == digest_mod.NO_TIER else f"{t.id} ({t.tier})" for t in in_progress]
        working = [f"{len(in_progress)} In progress: {_listed(labels)}" if in_progress else "nothing In progress"]
        finished = [t for t in mine if t.status == "Done"]
        tonight = [t for t in finished if t.id in active]
        total = f"{len(finished)} of Ilan's {len(mine)} tasks Done in all"
        done = (f"{_listed(_sorted_ids(tonight))} (Done on the board, with ledger rows in the window); {total}"
                if tonight else f"none yet in the window; {total}")
        stuck = [t for t in mine if t.status == "Blocked"]
        if stuck:
            blocked.append(f"{_listed(_sorted_ids(stuck))} {'is' if len(stuck) == 1 else 'are'} Blocked")
        missing = [row["id"] for row in (held.get("missing_blocker") or {}).get("items", [])]
        if missing:
            blocked.append(f"{_listed(missing)} held back, a blocker is missing from the board snapshot")
    working += gpu_working
    blocked += gpu_blocked
    if complaints:
        blocked.append(f"{_plural(complaints, 'complaint')} open on the GPU queue")
    if failed_total:
        named = [f"{row['task'] or 'no task'} {row['verb']} {row['outcome']}" for row in failures[:MAX_FAILURES]]
        more = failed_total - len(named)
        blocked.append(f"{_plural(failed_total, 'failed ledger row')} in the window ({'; '.join(named)}{f'; and {more} more' if more > 0 else ''})")

    shared = (held.get("shared") or {}).get("items", [])
    needs = (f"{_listed(shared)} {'is' if len(shared) == 1 else 'are'} shared with a teammate, so no sitting takes "
             f"{'it' if len(shared) == 1 else 'them'}") if shared else ""

    bodies = [rulings.notion_text("; ".join(working)), rulings.notion_text(done),
              rulings.notion_text("; ".join(blocked) or "nothing"), rulings.notion_text(needs)]
    ident = hashlib.sha256(json.dumps(bodies, ensure_ascii=True).encode("ascii")).hexdigest()[:8]
    return Section(ident, since, until, board_text, *bodies)


def render(section: Section, moment: dt.datetime) -> Rendered:
    """The section as the page shows it, signed with moment."""
    sign = rulings.stamp(moment)
    mark = f"(digest {section.section_id})"
    last = (f"- **{LAST}:** {sign} lab digest, {_et(section.since)} to {_et(section.until)}; "
            f"board snapshot {section.board} {mark}")
    segment = f"{sign} {SEGMENT_PHRASE} {section.needs} {mark}" if section.needs else None
    return Rendered(last, f"- **{WORKING}:** {section.working}", f"- **{DONE}:** {section.done}",
                    f"- **{BLOCKED}:** {section.blocked}", segment)


def rewrite_needs(line: str, segment: str | None, section_id: str) -> tuple[str, str]:
    """The Needs bullet with this flow's segment in place, and what was done: posted, updated, removed or
    unchanged. Only a segment this flow wrote is replaced or removed; the rest of the line stays as it is."""
    found_line = _NEEDS_LINE.match(line)
    if found_line is None:
        raise MirrorError(f"not a Needs a human bullet: {_clip(line, CLIP_NOTE)}")
    head, body = found_line["head"], found_line["body"]
    found = _OWN_SEGMENT.search(body)
    if segment is None:
        if not found:
            return line, "unchanged"
        before = body[: found.start()].strip().rstrip(";").rstrip()
        after = body[found.end():].strip().lstrip(";").lstrip()
        rest = f"{before}; {after}" if before and after else before or after
        return f"{head} {rest or 'none'}", "removed"
    if found:
        if found["id"] == section_id:
            return line, "unchanged"
        return head + body[: found.start()] + segment + body[found.end():], "updated"
    rest = body.strip()
    if rest.lower() in _EMPTY_BODY:
        return f"{head} {segment}", "posted"
    glue = " " if rest.endswith((";", ".")) else "; "
    return f"{head} {rest}{glue}{segment}", "posted"


@dataclasses.dataclass(frozen=True)
class _Item:
    """A bullet of the Ilan section: where it starts, where its children end, and its own line."""

    start: int
    end: int
    head: str


def _ilan_items(lines: Sequence[str]) -> tuple[int, dict[str, _Item]]:
    """(the index of the "## Ilan" heading, the five bullets by label). MirrorError unless the page has exactly
    one such heading and each of the five bullets exactly once below it."""
    heads = [index for index, line in enumerate(lines) if _ILAN_HEADING.match(line)]
    if len(heads) != 1:
        raise MirrorError(f"expected one '## Ilan' heading on the Overnight Status page, found {len(heads)}")
    start = heads[0]
    stop_at = next((index for index in range(start + 1, len(lines))
                    if _HEADING.match(lines[index]) or lines[index].strip() == "---"), len(lines))
    found: dict[str, list[_Item]] = {label: [] for label in LABELS}
    index = start + 1
    while index < stop_at:
        match = _BULLET.match(lines[index])
        end = index + 1
        while end < stop_at and lines[end][:1] in (" ", "\t"):
            end += 1
        if match and not match["indent"] and match["label"] in found:
            found[match["label"]].append(_Item(index, end, lines[index]))
        index = end
    bad = [f"{label!r} found {len(items)} times" for label, items in found.items() if len(items) != 1]
    if bad:
        raise MirrorError("the Ilan section needs each of its five bullets once: " + "; ".join(bad))
    return start, {label: items[0] for label, items in found.items()}


def overnight_edit(page: str, section: Section, moment: dt.datetime) -> tuple[str, str, str]:
    """(action, old text, new text) for the Overnight Status page: the one exact replacement that brings the Ilan
    section in line with the section, from the "## Ilan" heading to the end of its last bullet. The action is
    posted (the section was not this flow's yet), updated, or unchanged (then both texts are empty). MirrorError
    when the page does not have the shape."""
    lines = page.split("\n")
    start, items = _ilan_items(lines)
    shown = _DIGEST_MARK.search(items[LAST].head)
    rendered = render(section, moment)
    needs_line, needs_action = rewrite_needs(items[NEEDS].head, rendered.segment, section.section_id)
    if shown and shown["id"] == section.section_id and needs_action == "unchanged":
        return "unchanged", "", ""
    needs = items[NEEDS]
    replacement = {
        LAST: [rendered.last], WORKING: [rendered.working], DONE: [rendered.done], BLOCKED: [rendered.blocked],
        NEEDS: [needs_line, *lines[needs.start + 1:needs.end]],
    }
    first = min(item.start for item in items.values())
    last = max(item.end for item in items.values())
    at = {item.start: label for label, item in items.items()}
    new_lines = list(lines[start:first])
    index = first
    while index < last:
        label = at.get(index)
        if label is None:
            new_lines.append(lines[index])
            index += 1
        else:
            new_lines.extend(replacement[label])
            index = items[label].end
    return ("updated" if shown else "posted"), "\n".join(lines[start:last]), "\n".join(new_lines)


def rulings_segment_state(page: str | None, waiting: int, current: str | None) -> dict[str, Any]:
    """Whether the rulings segment on the Needs bullet matches the rulings now waiting. page is the Overnight Status
    text, or None when it was not read; current is the list id of the rulings now waiting (None when their
    files could not be read). state is current, stale, missing, none_waiting, or unknown."""
    on_page = None
    if page is not None:
        for line in page.split("\n"):
            if _NEEDS_LINE.match(line):
                found = _RULINGS_SEGMENT.search(line)
                on_page = found["id"] if found else None
    if page is None or current is None:
        state = "unknown"
    elif not waiting:
        state = "stale" if on_page else "none_waiting"
    elif on_page is None:
        state = "missing"
    else:
        state = "current" if on_page == current else "stale"
    answer: dict[str, Any] = {"waiting": waiting, "segment": state, "on_page": on_page, "current": current}
    if state in ("stale", "missing"):
        answer["next"] = "python -m lab rulings mirror"
    return answer


# ---------------------------------------------------------------------------------------------
# Plan changes, from the refinement logs


@dataclasses.dataclass(frozen=True)
class PlanChange:
    """One change of a pass or merge entry, as its plan-change line (unsigned, in Notion's markup)."""

    label: str
    task: str
    number: int
    order: tuple[Any, ...]
    text: str

    @property
    def covers(self) -> str:
        """What a line on the page starts with, after its signed prefix, to count as this change's."""
        return f"{self.label}, {self.task}:"


def _stop(text: str) -> str:
    """The text closed by a full stop, unless clipping already ended it with an ellipsis."""
    return text if text.endswith("...") else text + "."


def _plan_text(label: str, task: str, change: refine_formats.Change) -> str:
    what = rulings.notion_text(_clip(_bare(change.target), CLIP_WHAT)) or "change"
    was = rulings.notion_text(_clip(change.before, CLIP_WAS)) or "(not recorded)"
    why = rulings.notion_text(_clip(_bare(change.why), CLIP_WHY)) or "(not recorded)"
    source = rulings.notion_text(_clip(_bare(change.source), CLIP_SOURCE)) or "(not recorded)"
    return f'{label}, {task}: {_stop(what)} Was: "{was}". Why: {_stop(why)} Source: {_stop(source)}'


def _log_folders(refine_root: Path) -> list[str]:
    found = [folder.name for folder in refine_root.iterdir() if (folder / LOG_NAME).is_file()] if refine_root.is_dir() else []
    return rulings.track_order(found)


def collect_plan_changes(refine_root: Path) -> tuple[list[PlanChange], list[str]]:
    """Every change of every pass entry (lab/refine/<track>/REFINE_LOG.md) and merge entry (MERGE_LOG.md) as a
    plan-change line, oldest entry first, and the logs that could not be read (their changes are missing)."""
    logs = [(name, refine_root / name / LOG_NAME) for name in _log_folders(refine_root)]
    if (refine_root / MERGE_LOG_NAME).is_file():
        logs.append(("", refine_root / MERGE_LOG_NAME))
    found: list[PlanChange] = []
    problems: list[str] = []
    for track, path in logs:
        try:
            parsed = refine_formats.parse_log(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, refine_formats.FormatError) as exc:
            problems.append(f"{path.parent.name}/{path.name}: {_clip(str(exc), CLIP_NOTE)}")
            continue
        for entry in parsed.entries:
            # a track log holds pass entries and the merge log holds merge entries; anything else is not a plan change
            if not isinstance(entry, refine_formats.PassEntry) or entry.kind != ("pass" if track else "merge"):
                continue
            label = f"{track} pass {entry.number}" if entry.kind == "pass" else f"merge {entry.number}"
            tracks = refine_track.TRACKS
            rank = (entry.date, 0 if entry.kind == "pass" else 1,
                    tracks.index(track) if track in tracks else len(tracks), track, entry.number)
            for change in entry.changes:
                match = _TASK_ID.search(change.target)
                task = match[0] if match else "board"
                found.append(PlanChange(label, task, change.number, (*rank, change.number), _plan_text(label, task, change)))
    return sorted(found, key=lambda item: item.order), problems


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def plan_edit(page: str, changes: Sequence[PlanChange], moment: dt.datetime) -> dict[str, Any]:
    """What to do on Ilan's Team Comms page for these changes: a dict with `old` and `new` (the one exact
    replacement; both None when every change already has its line), `post` (the PlanChange list to be added),
    `already` (how many changes were found on the page) and `item` (existing or created). MirrorError when the
    page has no FYI heading."""
    lines = page.split("\n")
    heading = next((index for index, line in enumerate(lines) if _FYI_HEADING.match(line)), None)
    if heading is None:
        raise MirrorError("Ilan's Team Comms page has no '## FYI' heading")
    section_end = next((index for index in range(heading + 1, len(lines)) if _HEADING.match(lines[index])), len(lines))
    item = next((index for index in range(heading + 1, section_end) if _PLAN_ITEM.match(lines[index])), None)
    covered: Counter[tuple[str, str]] = Counter()
    indent, last = 0, -1
    if item is not None:
        indent, last = _indent(lines[item]), item
        while last + 1 < section_end and _indent(lines[last + 1]) > indent:
            last += 1
            child = _PLAN_CHILD.match(lines[last])
            if child and not child["rest"].startswith("reverted change"):
                covered[(child["label"], child["task"])] += 1
    seen: Counter[tuple[str, str]] = Counter()
    post: list[PlanChange] = []
    for change in changes:
        key = (change.label, change.task)
        seen[key] += 1
        if seen[key] > covered[key]:
            post.append(change)
    result: dict[str, Any] = {"old": None, "new": None, "post": post, "already": len(changes) - len(post),
                              "item": "existing" if item is not None else "created"}
    if not post:
        return result
    sign = rulings.stamp(moment)
    if item is None:
        result["old"] = lines[heading]
        result["new"] = "\n".join([lines[heading], PLAN_ITEM, *[f"\t- {sign} {change.text}" for change in post]])
    else:
        old = "\n".join(lines[max(item, last - 1):last + 1])  # the last two lines, so the anchor is unlikely to repeat
        child_indent = lines[item][:indent] + "\t"
        result["old"] = old
        result["new"] = "\n".join([old, *[f"{child_indent}- {sign} {change.text}" for change in post]])
    return result


# ---------------------------------------------------------------------------------------------
# No client: the texts a sitting writes with its Notion tools


def print_plan(section: Section | None, changes: Sequence[PlanChange] | None, moment: dt.datetime) -> list[dict[str, Any]]:
    """Where each text goes and how a session applies it, for use when no Notion client is available."""
    sign = rulings.stamp(moment)
    earlier = f"an earlier segment of this flow: it starts with '{rulings.SIGNER}', a time and '{SEGMENT_PHRASE}' and ends at '(digest <8 hex digits>)'"
    posts: list[dict[str, Any]] = []
    if section is not None:
        rendered = render(section, moment)
        posts.append({
            "target": "overnight_status",
            "action": "replace_lines",
            "page_url": OVERNIGHT_URL,
            "where": "under '## Ilan': replace the whole bullets '- **Last update:**', '- **Working on:**', '- **Done tonight:**' "
                     "and '- **Blocked on:**' (any lines indented under them go too), and change the bullet '- **Needs a human (Ilan):**' "
                     "only as described in needs_segment",
            "lines": {LAST: rendered.last, WORKING: rendered.working, DONE: rendered.done, BLOCKED: rendered.blocked},
            "needs_segment": rendered.segment or "",
            "needs_how": ("add the segment at the end of the Needs a human bullet after '; ', or replace " + earlier) if rendered.segment
                         else ("remove " + earlier + " from the bullet if it has one, and write 'none' there if nothing else is left"),
            "skip_if_page_shows": f"(digest {section.section_id}) on the Last update bullet, and the segment in place on the Needs a human bullet",
            "keep": "every other piece of the Needs a human bullet, the 'Lab rulings waiting on you' segment included, and the rest of the page, "
                    "above all every other person's section",
        })
    if changes is not None:
        posts.append({
            "target": "plan_changes",
            "action": "append_lines",
            "page_url": TEAM_COMMS_URL,
            "item": PLAN_ITEM,
            "where": "as child bullets of the item above, which is under '## FYI (no reply needed)'; add each line after the item's last child, "
                     "one tab deeper than the item, and create the item first, directly under the heading, when it is missing",
            "lines": [{"covers": change.covers, "text": f"- {sign} {change.text}"} for change in changes],
            "skip_if_page_shows": "a child line that already starts, after its signed prefix, with the same '<label>, <task>:'; each such line "
                                  "covers one change with that label and task, the earliest first, and a 'reverted change' line covers none",
        })
    return posts


# ---------------------------------------------------------------------------------------------
# The verb


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m lab mirror",
                                     description="Mirror the lab digest and the plan changes to Notion.")
    parser.add_argument("--only", choices=("overnight", "plan-changes"), default=None,
                        help="do one job only (default: both)")
    parser.add_argument("--print", dest="print_only", action="store_true",
                        help="touch nothing; return the exact texts and where they go")
    window = parser.add_mutually_exclusive_group()
    window.add_argument("--hours", type=float, default=None, help=f"digest window length (default {WINDOW_HOURS:g})")
    window.add_argument("--since", default=None, help="digest window start, ISO 8601 with a UTC offset")
    parser.add_argument("--board", default=None, help="board snapshot file (default: the stored one)")
    parser.add_argument("--refine-dir", type=Path, default=None, help="folder that holds the track folders (default: lab/refine)")
    return parser


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(list(argv))
    if args.hours is not None and not 0 < args.hours <= 24 * 31:
        parser.error("--hours must be more than 0 and at most 744")
    if args.since is not None and digest_mod.parse_time(args.since)[0] is None:
        parser.error(f"--since {args.since!r} is not an ISO 8601 time")
    return args


def _find_token(environ: Mapping[str, str], env_files: Sequence[Path] | None) -> str | None:
    files = ready.default_env_files(environ) if env_files is None else list(env_files)
    values, _ = ready.collect_env((ENV_TOKEN,), environ, files)
    return values.get(ENV_TOKEN, "").strip() or None


def _rulings_facts(refine_root: Path, moment: dt.datetime) -> tuple[int, str | None, list[str]]:
    """(rulings waiting, the id of their list, problems) from the rulings files; the id is None when a track's
    rulings file cannot be read, because the list would then be incomplete."""
    rows, errors, warnings = rulings.collect(refine_root)
    problems = [f"{item['track']}: {_clip(item['error'], CLIP_NOTE)}" for item in [*errors, *warnings]]
    waiting = [row for row in rows if row.state == rulings.OPEN]
    return len(waiting), (None if errors else rulings.build_posts(waiting, moment).list_id), problems


def _overnight(client: rulings.NotionPages, section: Section, moment: dt.datetime) -> tuple[dict[str, Any], str | None]:
    """(the target's result, the page text as last read)."""
    page: str | None = None
    try:
        page = client.read_page(OVERNIGHT_PAGE)
        action, old, new = overnight_edit(page, section, moment)
        if action == "unchanged":
            return {"action": "unchanged"}, page
        client.replace_text(OVERNIGHT_PAGE, old, new)
        page = client.read_page(OVERNIGHT_PAGE)
        _, items = _ilan_items(page.split("\n"))
        shown = _DIGEST_MARK.search(items[LAST].head)
        if not shown or shown["id"] != section.section_id:
            return {"action": "failed", "error": "the Overnight Status page does not show the change after the write"}, page
        return {"action": action}, page
    except (rulings.NotionError, MirrorError) as exc:
        return {"action": "failed", "error": str(exc)}, page


def _plan_changes(client: rulings.NotionPages, changes: Sequence[PlanChange], moment: dt.datetime) -> dict[str, Any]:
    try:
        edit = plan_edit(client.read_page(TEAM_COMMS_PAGE), changes, moment)
        if edit["old"] is None:
            return {"action": "unchanged", "posted": 0, "already": edit["already"], "changes": len(changes)}
        client.replace_text(TEAM_COMMS_PAGE, edit["old"], edit["new"])
        if plan_edit(client.read_page(TEAM_COMMS_PAGE), changes, moment)["old"] is not None:
            return {"action": "failed", "error": "Ilan's Team Comms page does not show the lines after the write"}
        return {"action": "posted", "posted": len(edit["post"]), "already": edit["already"], "changes": len(changes),
                "item": edit["item"], "lines": [f"{change.covers} (change {change.number})" for change in edit["post"]]}
    except (rulings.NotionError, MirrorError) as exc:
        return {"action": "failed", "error": str(exc)}


@dataclasses.dataclass
class _Facts:
    """What a call works from: the section and the plan changes (None for a job not asked for), the rulings now
    waiting, and what could not be read."""

    section: Section | None = None
    changes: list[PlanChange] | None = None
    waiting: int = 0
    rulings_list: str | None = None
    warnings: list[str] = dataclasses.field(default_factory=list)
    digest_problems: list[str] = dataclasses.field(default_factory=list)


def _gather(args: argparse.Namespace, moment: dt.datetime, refine_root: Path,
            state_dir: str | os.PathLike[str] | None) -> _Facts:
    """The facts for the jobs asked for. OSError when the ledger cannot be opened."""
    facts = _Facts()
    if args.only in (None, "overnight"):
        since = digest_mod.parse_time(args.since)[0] if args.since else moment - dt.timedelta(hours=args.hours or WINDOW_HOURS)
        digest = digest_mod.compute(now=moment, since=since, state_dir=state_dir, board=args.board,
                                    refine_root=refine_root, max_lines=10 ** 9)
        board, _ = digest_mod.load_board(args.board, state_dir=state_dir)
        facts.section = build_section(digest, board, active_tasks(since, moment, state_dir=state_dir))
        facts.digest_problems = list(digest["problems"]["items"])
        facts.waiting, facts.rulings_list, problems = _rulings_facts(refine_root, moment)
        facts.warnings += problems
    if args.only in (None, "plan-changes"):
        facts.changes, problems = collect_plan_changes(refine_root)
        facts.warnings += problems
    return facts


def run(argv: Sequence[str], *, client: rulings.NotionPages | None = None, transport: rulings.Transport | None = None,
        now: dt.datetime | None = None, environ: Mapping[str, str] | None = None,
        env_files: Sequence[Path] | None = None,
        state_dir: str | os.PathLike[str] | None = None) -> tuple[dict[str, Any], int]:
    """One mirror call: (the answer, the exit code). client replaces the Notion client the token would make,
    transport the HTTP under that client, now the clock; environ and env_files say where NOTION_TOKEN is
    looked up. The ledger row is written here."""
    args = parse_args(argv)
    environ = os.environ if environ is None else environ
    moment = (now or dt.datetime.now(UTC)).astimezone(UTC).replace(microsecond=0)
    if args.since and digest_mod.parse_time(args.since)[0] > moment:
        return {"ok": False, "outcome": "usage", "verb": VERB, "error": f"--since {args.since} is after now"}, 2
    try:
        facts = _gather(args, moment, args.refine_dir or refine_track.REFINE_DIR, state_dir)
    except (OSError, ValueError) as exc:
        failure = {"ok": False, "outcome": "error", "verb": VERB, "error": f"cannot compute the digest: {_clip(str(exc), CLIP_NOTE)}"}
        return _finish(failure, environ, state_dir), 1
    answer: dict[str, Any] = {"ok": True, "outcome": "ok", "verb": VERB, "mode": "live"}
    if facts.section is not None:
        answer["section_id"] = facts.section.section_id
        answer["window"] = {"since": digest_mod.iso(facts.section.since), "until": digest_mod.iso(facts.section.until)}
        answer["digest_problems"] = facts.digest_problems
    if facts.warnings:
        answer["warnings"] = facts.warnings

    token = None if client is not None or args.print_only else _find_token(environ, env_files)
    if args.print_only or (client is None and token is None):
        answer.update({"mode": "print", "rulings": rulings_segment_state(None, facts.waiting, facts.rulings_list),
                       "posts": print_plan(facts.section, facts.changes, moment),
                       "how": "Fetch each page first. Where it already shows what a post says (skip_if_page_shows), post nothing there; "
                              "where it shows an earlier post of this flow, replace that post in place. Edit nothing else on either page."})
        if args.print_only:
            answer["outcome"] = "printed"
            return _finish(answer, environ, state_dir), 0
        answer.update({"ok": False, "outcome": "no_token",
                       "error": f"{ENV_TOKEN} is not set (process environment or a .env file), so nothing was sent to Notion; "
                                "write the posts below with your Notion tools instead"})
        return _finish(answer, environ, state_dir), 2

    notion = client if client is not None else rulings.RestNotionClient(token or "", transport=transport)
    targets: dict[str, Any] = {}
    if facts.section is not None:
        targets["overnight_status"], page = _overnight(notion, facts.section, moment)
        answer["rulings"] = rulings_segment_state(page, facts.waiting, facts.rulings_list)
    if facts.changes is not None:
        targets["plan_changes"] = _plan_changes(notion, facts.changes, moment)
    answer["targets"] = targets
    if any(result["action"] == "failed" for result in targets.values()):
        answer.update({"ok": False, "outcome": "failed"})
    return _finish(answer, environ, state_dir), (0 if answer["ok"] else 1)


def _finish(answer: dict[str, Any], environ: Mapping[str, str],
            state_dir: str | os.PathLike[str] | None) -> dict[str, Any]:
    """Append the ledger row of a finished call and put its seq on the answer."""
    extras: dict[str, Any] = {"mode": answer.get("mode", "live")}
    if isinstance(answer.get("section_id"), str):
        extras["section_id"] = answer["section_id"]
    for name, result in (answer.get("targets") or {}).items():
        extras[name] = result["action"]
        if isinstance(result.get("posted"), int):
            extras["plan_lines_posted"] = result["posted"]
    try:
        answer["ledger_seq"] = record.record(VERB, answer["outcome"], task=record.current_task(environ),
                                             actor=record.current_actor(environ), tier=record.current_tier(environ),
                                             state_dir=state_dir, **extras)
    except Exception as exc:
        answer["ledger_seq"] = None
        answer["ledger_error"] = f"{type(exc).__name__}: {_clip(str(exc), CLIP_NOTE)}"
        log.error("mirror: no ledger row written (%s): %s", answer["outcome"], exc)
    return answer


@stop.guarded
def main(argv: Sequence[str]) -> int:
    """Mirror the lab digest to Overnight Status and the refinement plan changes to Ilan's Team Comms page."""
    answer, code = run(argv)
    print(json.dumps(answer))
    return code
