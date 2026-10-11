"""Checks for the merge pass: lab/refine/MERGE_LOG.md, the round it reconciles, and the track files it judges.

The merge pass (the refine-merge skill) runs once after every round of track passes. It walks every pass
entry of the round and the board items that cross tracks, then appends one '## Merge N, YYYY-MM-DD' entry
to lab/refine/MERGE_LOG.md in the pass-entry format (lab/refine/TEMPLATES.md, section 1.4). Parsing is
lab/machine/refine_formats.py; this module states what a merge entry keeps beyond that format, and the
invariants of a track's files that the merge pass relies on when it judges a round:

- A merge walks with merge check ids only (CHECKS) and runs every one of them each round.
- A walked pass entry is named '<track> pass <N>', one per line, and is checked against M6. It counts as
  reconciled unless its walk says BROKE M6 (the merge found it invalid and reverted it). The next round is
  every pass entry above the highest reconciled pass of its track, and each pass is reconciled once, in
  order.
- A ruling a merge lists names where it was filed, '<question> -> <track> <n>', and ruling <n> of that
  track's OWNER_RULINGS.md asks the same question. Every ruling a reconciled pass of the round raised is
  listed this way, filed in that pass's own track.
- A change to a RUBRIC.md or to a Priority cites an answered ruling as '<track>/OWNER_RULINGS.md <n>'. A file
  name Notion turned into a link, [OWNER_RULINGS.md](http://OWNER_RULINGS.md), reads as the name.
- A track's pass entries are numbered 1, 2, 3 in file order and each is valid; refine_pass.txt equals the
  last pass number; every pass has its report, whose first line names its track and pass, whose second line
  says when it ended and which quotes its entry's heading; and no report lacks its pass. check_track states
  these once, and refine_track.check_track reads a track folder through it.
- A track's rulings are numbered 1, 2, 3 in file order, and each names its origin.

Every check returns a list of problems, one sentence each; an empty list means everything holds.
check_tree() runs them all on a lab/refine folder, which is what tests/test_lab_refine_merge.py does with
the committed files.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from lab.machine import refine_formats as rf

__all__ = [
    "CHECKS",
    "TRACKS",
    "WalkedPass",
    "check_merge_entry",
    "check_merge_log",
    "check_rulings",
    "check_track",
    "check_tree",
    "next_merge_number",
    "read_reports",
    "reconciled",
    "round_passes",
    "walked_passes",
]

TRACKS = rf.TRACKS

#: The checks every merge runs, by id. A merge entry's walk, findings and changes use these ids as
#: their dimensions, and its walk names every one of them.
CHECKS: dict[str, str] = {
    "M1": "blockers that cross tracks",
    "M2": "shared contract changes, and names that two tracks must spell the same way",
    "M3": "the demo flow: every demo beat Ilan owns still has a path",
    "M4": "duplicate tasks across tracks, and a task that is in no track or in two",
    "M5": "Tier consistency",
    "M6": "every entry of the round is valid",
    "M7": "answered rulings that need the merge pass's powers are carried out",
}

_TRACK = "|".join(TRACKS)
_CHECK_ID = re.compile(r"M[0-9]{1,2}")
_PASS_TARGET = re.compile(r"^(?P<track>[a-z]+) pass (?P<number>[0-9]+)$")
_PASS_MENTION = re.compile(rf"\b(?:{_TRACK}) pass [0-9]")
_BROKE = re.compile(r"^BROKE\s+(?P<ids>[A-Z0-9]+(?:\s*,\s*[A-Z0-9]+)*)")
_DESTINATION = re.compile(r"\s+->\s+(?P<track>[a-z]+) (?P<number>[0-9]+)\s*$")
_RULING_CITATION = re.compile(rf"\b(?P<track>{_TRACK})/OWNER_RULINGS\.md,?\s+(?:ruling\s+)?(?P<number>[0-9]+)\b")
_NEEDS_RULING = re.compile(r"RUBRIC\.md|\bPriority\b")
_ORIGIN = re.compile(r"pass [0-9]+|merge [0-9]+|owner, [0-9]{4}-[0-9]{2}-[0-9]{2}")
_REPORT_NAME = re.compile(r"pass_(?P<number>[0-9]{2,})\.md")
_REPORT_FIRST_LINE = re.compile(r"^Verbatim report returned by (?P<track>[a-z]+) refinement pass (?P<number>[0-9]+) \(")


@dataclass(frozen=True)
class WalkedPass:
    """A pass entry a merge walked. reconciled is False when the walk says BROKE M6: the entry was invalid
    and the merge reverted it, so the pass number is free for the track's next valid pass."""

    track: str
    number: int
    reconciled: bool


def walked_passes(entry: rf.PassEntry) -> tuple[WalkedPass, ...]:
    """The '<track> pass <N>' lines of a merge entry's walk, in the order they were written."""
    found: list[WalkedPass] = []
    for walk in entry.walked:
        target = _PASS_TARGET.match(walk.target)
        if not target:
            continue
        broke = _BROKE.match(walk.result)
        broken = {part.strip() for part in broke["ids"].split(",")} if broke else set()
        found.append(WalkedPass(target["track"], int(target["number"]), "M6" not in broken))
    return tuple(found)


def _merges(merge_log: rf.Log) -> list[rf.PassEntry]:
    return [entry for entry in merge_log.entries if isinstance(entry, rf.PassEntry) and entry.kind == "merge"]


def reconciled(merge_log: rf.Log) -> dict[str, int]:
    """The highest pass number of each track that a merge has reconciled, 0 for a track with none."""
    highest = dict.fromkeys(TRACKS, 0)
    for entry in _merges(merge_log):
        for walked in walked_passes(entry):
            if walked.reconciled and walked.track in highest:
                highest[walked.track] = max(highest[walked.track], walked.number)
    return highest


def round_passes(track_logs: Mapping[str, rf.Log], merge_log: rf.Log) -> dict[str, tuple[rf.PassEntry, ...]]:
    """The pass entries the next merge reconciles: each track's passes above its highest reconciled one."""
    done = reconciled(merge_log)
    return {
        track: tuple(entry for entry in log.passes if entry.number > done.get(track, 0))
        for track, log in track_logs.items()
    }


def next_merge_number(merge_log: rf.Log) -> int:
    """The number of the next merge entry: the merge entries in the file, plus one."""
    return len(_merges(merge_log)) + 1


def _norm(text: str) -> str:
    return " ".join(text.split())


def _filed_as(question: str) -> tuple[str, re.Match[str] | None]:
    """Split a listed ruling's question into the question itself and its ' -> <track> <n>' suffix."""
    destination = _DESTINATION.search(question)
    if not destination:
        return _norm(question), None
    return _norm(question[: destination.start()]), destination


def check_merge_entry(
    entry: rf.PassEntry,
    track_logs: Mapping[str, rf.Log] | None = None,
    rulings: Mapping[str, Sequence[rf.Ruling]] | None = None,
) -> list[str]:
    """List what a merge entry breaks; an empty list means it is valid.

    The entry must pass refine_formats.check_pass_entry and the merge rules in this module's docstring.
    With track_logs, every reconciled pass it walked must exist and have each of its raised rulings listed.
    With rulings (each track's parsed OWNER_RULINGS.md), every listed ruling must be filed where the entry
    says, and every RUBRIC.md or Priority change must cite an answered ruling that exists. A track missing
    from track_logs or rulings is skipped: whoever read the files reports the missing one.
    """
    if entry.kind != "merge":
        return [f"{entry.kind} {entry.number}: MERGE_LOG.md holds merge entries only"]
    who = f"merge {entry.number}"
    problems = list(rf.check_pass_entry(entry))

    def add(message: str) -> None:
        problems.append(f"{who}: {message}")

    ids = ", ".join(CHECKS)
    seen: set[str] = set()
    for walk in entry.walked:
        seen.update(walk.dimensions)
        strange = [d for d in walk.dimensions if not _CHECK_ID.fullmatch(d)]
        if strange:
            add(f"walked {walk.target}: {', '.join(strange)} is not a merge check id ({ids})")
        target = _PASS_TARGET.match(walk.target)
        if target:
            if target["track"] not in TRACKS:
                add(f"walked {walk.target}: {target['track']!r} is not a track ({', '.join(TRACKS)})")
            if "M6" not in walk.dimensions:
                add(f"walked {walk.target}: a pass entry of the round is checked against M6")
        elif _PASS_MENTION.search(walk.target):
            add(f"walked {walk.target}: name one pass entry per line, as '<track> pass <N>'")
    missing = [check for check in CHECKS if check not in seen]
    if missing:
        add(f"the walk never runs {', '.join(missing)}; every merge runs all of {ids}")

    for finding in entry.findings:
        if not _CHECK_ID.fullmatch(finding.dimension):
            add(f"the finding for {finding.dimension} {finding.where}: {finding.dimension} is not a merge check id")

    filed = {track: {r.number: r for r in found} for track, found in (rulings or {}).items()}
    for change in entry.changes:
        strange = [d for d in change.dimensions if not _CHECK_ID.fullmatch(d)]
        if strange:
            add(f"change {change.number}: {', '.join(strange)} is not a merge check id")
        if not _NEEDS_RULING.search(rf.unlink_file_names(change.target)):
            continue
        cited = _RULING_CITATION.search(rf.unlink_file_names(change.source))
        if not cited:
            add(
                f"change {change.number} edits a RUBRIC.md or a Priority, so its SOURCE cites the answered "
                "ruling as '<track>/OWNER_RULINGS.md <n>'"
            )
        elif cited["track"] in filed:
            track, number = cited["track"], int(cited["number"])
            ruling = filed[track].get(number)
            if ruling is None:
                add(f"change {change.number} cites {track} ruling {number}, which does not exist")
            elif not ruling.answered:
                add(f"change {change.number} cites {track} ruling {number}, which has no Answer yet")

    listed: dict[str, set[str]] = {}
    for raised in entry.rulings_raised:
        question, destination = _filed_as(raised.question)
        if destination is None:
            add(f"ruling ({raised.label}) does not say where it was filed; end its question with ' -> <track> <n>'")
            continue
        track, number = destination["track"], int(destination["number"])
        if track not in TRACKS:
            add(f"ruling ({raised.label}) is filed under {track!r}, which is not a track")
            continue
        listed.setdefault(track, set()).add(question)
        if track in filed:
            ruling = filed[track].get(number)
            if ruling is None:
                add(f"ruling ({raised.label}) says it was filed as {track} {number}, but that file has no ruling {number}")
            elif _norm(ruling.question) != question:
                add(f"ruling ({raised.label}) and {track} ruling {number} do not ask the same question")

    if track_logs is not None:
        for walked in walked_passes(entry):
            log = track_logs.get(walked.track)
            if log is None or not walked.reconciled:
                continue
            passes = {p.number: p for p in log.passes}
            entry_of_pass = passes.get(walked.number)
            if entry_of_pass is None:
                add(f"walked {walked.track} pass {walked.number}, which is not in that track's REFINE_LOG.md")
                continue
            for raised in entry_of_pass.rulings_raised:
                if _norm(raised.question) not in listed.get(walked.track, set()):
                    add(
                        f"{walked.track} pass {walked.number} raised ruling ({raised.label}), which this merge "
                        f"does not list as filed in the {walked.track} rulings"
                    )
    return problems


def check_merge_log(
    merge_log: rf.Log,
    track_logs: Mapping[str, rf.Log] | None = None,
    rulings: Mapping[str, Sequence[rf.Ruling]] | None = None,
) -> list[str]:
    """List what a parsed MERGE_LOG.md breaks: its entries (check_merge_entry), their numbering and dates,
    and the rule that each track's passes are reconciled once each, in order, from pass 1."""
    problems: list[str] = []
    for entry in merge_log.entries:
        if isinstance(entry, rf.PassEntry) and entry.kind == "pass":
            problems.append(f"pass {entry.number}: a pass entry belongs in its track's REFINE_LOG.md, not in MERGE_LOG.md")
        elif isinstance(entry, rf.RunState):
            problems.append("a 'State at the end of the run' section belongs in its track's REFINE_LOG.md, not in MERGE_LOG.md")
    merges = _merges(merge_log)
    numbers = [entry.number for entry in merges]
    if numbers != list(range(1, len(numbers) + 1)):
        problems.append(f"merge entries are numbered 1 to {len(numbers)} in file order, found {numbers}")
    for earlier, later in zip(merges, merges[1:]):
        if later.date < earlier.date:
            problems.append(f"merge {later.number} is dated {later.date}, before merge {earlier.number} ({earlier.date})")
    order: dict[str, list[int]] = {}
    for entry in merges:
        for walked in sorted(walked_passes(entry), key=lambda w: w.number):
            if walked.reconciled:
                order.setdefault(walked.track, []).append(walked.number)
    for track, found in order.items():
        expected = list(range(1, len(found) + 1))
        if found != expected:
            problems.append(
                f"{track}: the merges reconcile passes {found}; each pass is reconciled once, in order, from pass 1 "
                f"(expected {expected})"
            )
    for entry in merges:
        problems.extend(check_merge_entry(entry, track_logs, rulings))
    return problems


def read_reports(track_dir: Path) -> dict[int, str]:
    """The pass reports in a track folder, pass_reports/pass_NN.md, by pass number; {} when there is no such folder."""
    folder = Path(track_dir) / "pass_reports"
    if not folder.is_dir():
        return {}
    reports: dict[int, str] = {}
    for path in sorted(folder.iterdir()):
        name = _REPORT_NAME.fullmatch(path.name)
        if name and path.is_file():
            reports[int(name["number"])] = path.read_text(encoding="utf-8")
    return reports


def check_track(track: str, log: rf.Log, counter: int | None, reports: Mapping[int, str]) -> list[str]:
    """List what one track's files break, each problem named by the file it is in. counter is the value of
    refine_pass.txt (None skips that check); reports maps a pass number to the text of pass_reports/pass_NN.md.

    This is the one statement of a track folder's rules. The merge pass reads it through check_tree, and a track
    pass through refine_track.check_track, which adds what only a pass checks (metric citations, dry runs).
    """
    problems: list[str] = []
    numbers = [entry.number for entry in log.passes]
    if numbers != list(range(1, len(numbers) + 1)):
        problems.append(f"REFINE_LOG.md: pass entries must be numbered 1 to {len(numbers)} in file order, found {numbers}")
    last = numbers[-1] if numbers else 0
    if counter is not None and counter != last:
        problems.append(
            f"refine_pass.txt reads {counter} but the log's last pass entry is "
            f"{f'pass {last}' if numbers else 'none'}; a pass writes the counter last, after its entry"
        )
    for entry in log.passes:
        problems.extend(f"REFINE_LOG.md: {problem}" for problem in rf.check_pass_entry(entry))
    for entry in log.passes:
        text = reports.get(entry.number)
        if text is None:
            problems.append(f"pass {entry.number} has no report at pass_reports/pass_{entry.number:02d}.md")
        else:
            problems.extend(_report_problems(track, entry, text))
    for number in sorted(set(reports) - set(numbers)):
        problems.append(f"pass_reports/pass_{number:02d}.md has no pass {number} in REFINE_LOG.md")
    return problems


def _report_problems(track: str, entry: rf.PassEntry, text: str) -> list[str]:
    """A pass report's first line names its track and pass, its second line says when the pass ended, and it
    quotes its entry's heading (lab/refine/TEMPLATES.md, section 4)."""
    name = f"pass_reports/pass_{entry.number:02d}.md"
    lines = text.replace("\r\n", "\n").split("\n")
    problems: list[str] = []
    first = _REPORT_FIRST_LINE.match(lines[0])
    if not first or (first["track"], int(first["number"])) != (track, entry.number):
        problems.append(
            f"{name}: the first line must read 'Verbatim report returned by {track} refinement pass {entry.number} (...'"
        )
    if len(lines) < 2 or not lines[1].startswith("Ended:"):
        problems.append(f"{name}: the second line must state when the pass ended: 'Ended: <date>T<time><offset>'")
    else:
        try:
            rf.parse_report_end(lines[1])
        except rf.FormatError as error:
            problems.append(f"{name}: {error}")
    heading = f"## Pass {entry.number}, {entry.date}"
    if heading not in (line.rstrip() for line in lines):
        problems.append(f"{name}: does not quote its entry; no line reads '{heading}'")
    return problems


def check_rulings(track: str, rulings: Sequence[rf.Ruling]) -> list[str]:
    """List what one track's parsed OWNER_RULINGS.md breaks: numbering in order of arrival, and origins."""
    problems: list[str] = []
    numbers = [ruling.number for ruling in rulings]
    if numbers != list(range(1, len(numbers) + 1)):
        problems.append(f"{track}: rulings are numbered 1 to {len(numbers)} in file order, found {numbers}")
    for ruling in rulings:
        if not _ORIGIN.fullmatch(ruling.origin):
            problems.append(
                f"{track} ruling {ruling.number}: the origin {ruling.origin!r} is not 'pass <N>', 'merge <N>' "
                "or 'owner, <YYYY-MM-DD>'"
            )
    return problems


def check_tree(root: Path) -> list[str]:
    """Run every check on a lab/refine folder: each track's log, counter, reports and rulings, then
    MERGE_LOG.md against them. A file that does not parse is reported, and the checks that need it skip it."""
    problems: list[str] = []
    track_logs: dict[str, rf.Log] = {}
    rulings: dict[str, tuple[rf.Ruling, ...]] = {}

    def read(path: Path) -> str | None:
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            problems.append(f"{path.relative_to(root).as_posix()} is missing")
            return None

    for track in TRACKS:
        folder = root / track
        log: rf.Log | None = None
        counter: int | None = None
        log_text = read(folder / "REFINE_LOG.md")
        if log_text is not None:
            try:
                log = rf.parse_log(log_text)
            except rf.FormatError as error:
                problems.append(f"{track} REFINE_LOG.md: {error}")
        counter_text = read(folder / "refine_pass.txt")
        if counter_text is not None:
            try:
                counter = rf.parse_counter(counter_text)
            except rf.FormatError as error:
                problems.append(f"{track}: {error}")
        rulings_text = read(folder / "OWNER_RULINGS.md")
        if log is not None:
            track_logs[track] = log
            problems.extend(f"{track}: {problem}" for problem in check_track(track, log, counter, read_reports(folder)))
        if rulings_text is not None:
            try:
                rulings[track] = rf.parse_rulings(rulings_text)
            except rf.FormatError as error:
                problems.append(f"{track} OWNER_RULINGS.md: {error}")
            else:
                problems.extend(check_rulings(track, rulings[track]))

    merge_path = root / "MERGE_LOG.md"
    if merge_path.is_file():
        try:
            merge_log = rf.parse_log(merge_path.read_text(encoding="utf-8"))
        except rf.FormatError as error:
            problems.append(f"MERGE_LOG.md: {error}")
        else:
            problems.extend(check_merge_log(merge_log, track_logs, rulings))
    return problems
