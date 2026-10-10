"""Checks on what refinement passes leave in a track folder, and the dry-run record format.

A pass on one track (the refine-track skill, which lives in the workspace outside this repo) appends one entry
to lab/refine/<track>/REFINE_LOG.md, saves its report as pass_reports/pass_<NN>.md and writes the counter last.
Its rehearsal mode, --dry-run, writes one record under lab/refine/<track>/dry_runs/ and nothing else.

check_track() reads a track folder and returns one sentence per problem: an entry that check_pass_entry()
rejects, pass numbers that do not run from 1 to the counter, a pass without its report, a metrics-file line
cited for a value it does not hold, a dry-run record that does not parse. An empty list means the folder holds
together, which is what a pass checks before it commits and what the suite checks for every track.

A dry-run record is a pass report (TEMPLATES.md section 4) with three differences:

    Dry run of <track> refinement pass <N> (<who>), <YYYY-MM-DD>. <what it did not touch>
    Ended: <ISO 8601 date and time with a UTC offset>
    ...
    ```text dry-run:pass-entry
    ## Pass <N>, <YYYY-MM-DD>
    ...
    ```

Its file is named <YYYY-MM-DD>-pass-<NN>.md after the date and the pass it rehearses. A dry run is not a pass:
its entry never counts toward the counter, and no pass cites a record as a source. parse_dry_run() reads one
and raises FormatError when its structure is wrong.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from lab.machine import refine_formats as rf

__all__ = [
    "METRICS_PATH",
    "REFINE_DIR",
    "TRACKS",
    "DryRun",
    "check_dry_run",
    "check_track",
    "metric_citation_problems",
    "parse_dry_run",
]

TRACKS = rf.TRACKS
REFINE_DIR = Path(__file__).resolve().parents[1] / "refine"
METRICS_PATH = REFINE_DIR / "metrics.jsonl"

_DRY_RUN_HEADER = re.compile(
    r"^Dry run of (?P<track>[a-z]+) refinement pass (?P<number>[0-9]+) \((?P<who>[^()\n]*)\), "
    r"(?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2})\."
)
_DRY_RUN_NAME = re.compile(r"^(?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2})-pass-(?P<number>[0-9]{2,})\.md$")
_DRY_RUN_FENCE = re.compile(r"^```text dry-run:pass-entry[ \t]*\n(?P<body>.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)
_REPORT_FIRST_LINE = re.compile(r"^Verbatim report returned by (?P<track>[a-z]+) refinement pass (?P<number>[0-9]+) \(")
_METRICS_LINE = re.compile(r"(?:^|/)metrics\.jsonl:(?P<line>[0-9]+)$")


@dataclass(frozen=True)
class DryRun:
    """A parsed dry-run record: which pass of which track was rehearsed, by whom, when it ended, and its entry."""

    track: str
    number: int
    who: str
    date: str
    ended: datetime
    entry: rf.PassEntry


def parse_dry_run(text: str) -> DryRun:
    """Read a dry-run record: its header line, its Ended line and the one pass entry in its fenced block."""
    lines = _normalize(text).split("\n")
    header = _DRY_RUN_HEADER.match(lines[0])
    if not header:
        raise rf.FormatError(
            "a dry-run record starts 'Dry run of <track> refinement pass <N> (<who>), <YYYY-MM-DD>.'"
        )
    if len(lines) < 2 or not lines[1].startswith("Ended:"):
        raise rf.FormatError("the second line of a dry-run record states when it ended: 'Ended: <date>T<time><offset>'")
    ended = rf.parse_report_end(lines[1])
    fences = list(_DRY_RUN_FENCE.finditer("\n".join(lines)))
    if len(fences) != 1:
        raise rf.FormatError(f"a dry-run record holds exactly one ```text dry-run:pass-entry block, found {len(fences)}")
    log = rf.parse_log(fences[0]["body"])
    entries = [entry for entry in log.entries if isinstance(entry, rf.PassEntry) and entry.kind == "pass"]
    if len(log.entries) != 1 or len(entries) != 1:
        raise rf.FormatError(f"the dry-run block holds one '## Pass N, YYYY-MM-DD' entry and nothing else, found {len(log.entries)} entries")
    entry = entries[0]
    number = int(header["number"])
    if (entry.number, entry.date) != (number, header["date"]):
        raise rf.FormatError(
            f"the record rehearses pass {number} on {header['date']} but its entry is pass {entry.number} on {entry.date}"
        )
    return DryRun(header["track"], number, header["who"].strip(), header["date"], ended, entry)


def metric_citation_problems(readings: Iterable[rf.Reading], metrics_lines: list[str], where: str) -> list[str]:
    """Check every reading that cites metrics.jsonl:<line> against that line: same metric, node and value.

    Readings whose source is 'none' or a ledger row are skipped; the ledger is not kept in this repo.
    """
    problems: list[str] = []
    for reading in readings:
        found = _METRICS_LINE.search(reading.source)
        if not found:
            continue
        line = int(found["line"])
        text = metrics_lines[line - 1].strip() if 1 <= line <= len(metrics_lines) else ""
        if not text:
            problems.append(f"{where}: {reading.metric} cites metrics.jsonl:{line}, which holds no row")
            continue
        try:
            row = json.loads(text)
        except ValueError:
            problems.append(f"{where}: {reading.metric} cites metrics.jsonl:{line}, which is not a JSON row")
            continue
        value = row.get("value")
        differ = []
        if row.get("metric") != reading.metric:
            differ.append(f"metric {row.get('metric')!r}")
        if row.get("node") != reading.node:
            differ.append(f"node {row.get('node')!r}")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or reading.number != float(value):
            differ.append(f"value {value!r}")
        if differ:
            problems.append(
                f"{where}: {reading.metric} = {reading.value} on {reading.node} cites metrics.jsonl:{line}, "
                f"which holds {', '.join(differ)}"
            )
    return problems


def check_dry_run(path: Path | str, metrics_path: Path | str = METRICS_PATH) -> list[str]:
    """List what is wrong with one dry-run record; an empty list means it is a valid rehearsal of one pass."""
    path = Path(path)
    name = path.name
    try:
        record = parse_dry_run(path.read_text(encoding="utf-8"))
    except (OSError, rf.FormatError) as exc:
        return [f"{name}: {exc}"]
    problems = [f"{name}: {problem}" for problem in rf.check_pass_entry(record.entry)]
    named = _DRY_RUN_NAME.match(name)
    if not named:
        problems.append(f"{name}: a dry-run record is named <YYYY-MM-DD>-pass-<NN>.md")
    elif (named["date"], int(named["number"])) != (record.date, record.number):
        problems.append(
            f"{name}: the name says pass {int(named['number'])} on {named['date']} "
            f"but the record rehearses pass {record.number} on {record.date}"
        )
    if record.track not in TRACKS:
        problems.append(f"{name}: {record.track!r} is not a track; the tracks are {', '.join(TRACKS)}")
    folder = path.parent
    if folder.name == "dry_runs" and record.track != folder.parent.name:
        problems.append(f"{name}: the record rehearses the {record.track} track but sits in the {folder.parent.name} folder")
    entry = record.entry
    problems += metric_citation_problems(entry.metrics_before + entry.metrics_after, _lines(metrics_path), name)
    return problems


def check_track(track_dir: Path | str, metrics_path: Path | str = METRICS_PATH) -> list[str]:
    """List what is wrong with a track folder's log, counter, reports and dry-run records; [] when they hold together."""
    track_dir = Path(track_dir)
    track = track_dir.name
    try:
        log = rf.parse_log(_read(track_dir / "REFINE_LOG.md"))
    except (OSError, rf.FormatError) as exc:
        return [f"REFINE_LOG.md: {exc}"]
    try:
        counter = rf.parse_counter(_read(track_dir / "refine_pass.txt"))
    except (OSError, rf.FormatError) as exc:
        return [f"refine_pass.txt: {exc}"]
    problems: list[str] = []
    numbers = [entry.number for entry in log.passes]
    if numbers != list(range(1, counter + 1)):
        problems.append(
            f"refine_pass.txt reads {counter} but the log's pass entries are numbered {numbers}; "
            "they must run from 1 to the counter, which a pass writes last"
        )
    metrics_lines = _lines(metrics_path)
    for entry in log.passes:
        problems += [f"REFINE_LOG.md: {problem}" for problem in rf.check_pass_entry(entry)]
        problems += metric_citation_problems(
            entry.metrics_before + entry.metrics_after, metrics_lines, f"REFINE_LOG.md pass {entry.number}"
        )
        problems += _report_problems(track_dir, track, entry)
    for record in sorted((track_dir / "dry_runs").glob("*.md")):
        problems += [f"dry_runs/{problem}" for problem in check_dry_run(record, metrics_path)]
    return problems


def _report_problems(track_dir: Path, track: str, entry: rf.PassEntry) -> list[str]:
    name = f"pass_{entry.number:02d}.md"
    path = track_dir / "pass_reports" / name
    if not path.is_file():
        return [f"pass {entry.number} has no report at pass_reports/{name}"]
    lines = _read(path).split("\n")
    problems: list[str] = []
    first = _REPORT_FIRST_LINE.match(lines[0])
    if not first or (first["track"], int(first["number"])) != (track, entry.number):
        problems.append(
            f"pass_reports/{name}: the first line must read "
            f"'Verbatim report returned by {track} refinement pass {entry.number} (...'"
        )
    if len(lines) < 2 or not lines[1].startswith("Ended:"):
        problems.append(f"pass_reports/{name}: the second line must state when the pass ended: 'Ended: <date>T<time><offset>'")
    else:
        try:
            rf.parse_report_end(lines[1])
        except rf.FormatError as exc:
            problems.append(f"pass_reports/{name}: {exc}")
    heading = f"## Pass {entry.number}, {entry.date}"
    if heading not in (line.rstrip() for line in lines):
        problems.append(f"pass_reports/{name}: does not quote its entry; no line reads '{heading}'")
    return problems


def _normalize(text: str) -> str:
    return text.replace("\r\n", "\n")


def _read(path: Path) -> str:
    return _normalize(path.read_text(encoding="utf-8"))


def _lines(path: Path | str) -> list[str]:
    try:
        return _read(Path(path)).split("\n")
    except OSError:
        return []
