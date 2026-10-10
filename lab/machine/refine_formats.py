"""Parsers for the refinement loop files described in lab/refine/TEMPLATES.md.

Every track under lab/refine/<track>/ keeps three machine-read files: REFINE_LOG.md (pass entries,
owner corrections and the closing state of each run), OWNER_RULINGS.md (numbered questions with an
Answer line the owner fills in) and refine_pass.txt (the pass counter). The pass runner, the merge
pass and the rulings flow read them through this module, so they all see the shapes the passes write.

Structure is strict: a heading that is not one of the documented forms, a duplicate number, a
malformed counter or a ruling without its Answer line raises FormatError. Content inside a pass entry
is read leniently, and check_pass_entry() lists what an entry is missing so a pass can be told exactly
what to fix.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date as _date
from datetime import datetime

__all__ = [
    "Change",
    "CodeBlock",
    "Finding",
    "FormatError",
    "Log",
    "METRIC_NAME",
    "OwnerCorrection",
    "PassEntry",
    "RaisedRuling",
    "Reading",
    "Ruling",
    "RunState",
    "Source",
    "TRACKS",
    "Walk",
    "check_pass_entry",
    "code_blocks",
    "parse_counter",
    "parse_log",
    "parse_report_end",
    "parse_rulings",
]

#: The four refinement tracks, in the order lab/refine/PASS_RULES.md lists them ("The four tracks"). Every module of
#: the loop reads this tuple; changing the set takes a ruling and a board task.
TRACKS: tuple[str, ...] = ("telemetry", "learning", "plumbing", "kernels")

#: A metric name: dotted lowercase words, each starting with a letter (teacher.decode_tok_s). It is the rule
#: lab/machine/metrics.py applies to lab/refine/metrics.jsonl, so a name a reading cites or an experiment moves is
#: one that file can hold.
METRIC_NAME = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+")

# A dimension id is the label a track's RUBRIC.md gives one of its tests (D1, T3, ...); U1 and U2 are
# the two universal rules.
_DIM = r"[A-Z]{1,3}[0-9]{1,2}"
_DIM_LIST = rf"{_DIM}(?:\s*,\s*{_DIM})*"
_DATE = r"[0-9]{4}-[0-9]{2}-[0-9]{2}"

_PASS_HEADING = re.compile(rf"^## (?P<kind>Pass|Merge) (?P<number>[0-9]+), (?P<date>{_DATE})\s*$")
_STATE_HEADING = re.compile(
    rf"^## State at the end of the run(?:, passes (?P<first>[0-9]+) to (?P<last>[0-9]+), (?P<date>{_DATE}))?\s*$"
)
_RUN_ENDED = re.compile(r"^Run ended:\s*(?P<how>converged|cap reached|ended by the owner)\b", re.IGNORECASE)
_ENDING = re.compile(
    r"^(?:CONVERGED at pass (?P<converged>[0-9]+)|CAP REACHED at pass (?P<cap>[0-9]+), not converged)\s*$"
)
_REPORT_END = re.compile(r"^Ended:\s*(?P<when>\S+)\s*$", re.MULTILINE)
_RULING_HEADING = re.compile(r"^## (?P<number>[0-9]+)\.\s+(?P<title>.*?)\s*(?:\((?P<origin>[^()]*)\))?\s*$")

_SECTION_NAMES = {
    "Research step": "research",
    "Metrics before": "metrics_before",
    "Walked": "walked",
    "Findings": "findings",
    "Changes": "changes",
    "Rulings raised": "rulings",
    "Metrics after": "metrics_after",
    "Converged": "converged",
    "Left for the next pass": "next",
    "Checks run": "checks",
}
# A section starts at the beginning of a line with its label, an optional free-text parenthetical,
# and a colon. The parenthetical may itself contain colons.
_SECTION_LINE = re.compile(
    r"^(?P<label>" + "|".join(re.escape(label) for label in _SECTION_NAMES) + r")(?:\s*\([^)\n]*\))?:[ \t]*(?P<rest>.*)$"
)

_COLUMNS = ("metric", "value", "unit", "node", "conditions", "source")
_LINE_REF = re.compile(r"[\w./-]*(?:metrics\.jsonl|ledger(?:\.jsonl)?):[0-9]+")

_CHANGE_START = re.compile(r"^(?P<number>[0-9]+)\.\s+(?P<head>.+)$")
_CHANGE_FIELD = re.compile(r"^\s*(?P<label>BEFORE|AFTER|SOURCE|WHY|METRIC):[ \t]*(?P<value>.*)$")
_CHANGE_DIMS = re.compile(rf"\((?P<dims>{_DIM_LIST})\)\s*$")
_FINDING_LEFT = re.compile(rf"^(?P<dim>{_DIM})\s+(?P<where>.+)$")
_FIXED = re.compile(r"\bFIXED \((?P<numbers>[0-9]+(?:\s*,\s*[0-9]+)*)\)")
_RULING_REF = re.compile(r"\bRULING \((?P<labels>[a-z](?:\s*,\s*[a-z])*)\)")
_RAISED_START = re.compile(r"^\((?P<label>[a-z])\)\s+(?P<question>.+)$")
_RULINGS_APPLIED = re.compile(r"^Rulings applied:\s*(?P<numbers>[0-9]+(?:\s*,\s*[0-9]+)*)\s*$")

# A code fence: three or more backticks or tildes, after any indentation; what follows is the info string.
_FENCE = re.compile(r"^\s*(?P<fence>`{3,}|~{3,})(?P<info>.*)$")

_RAISED_FIELDS = {"Now": "now", "Recommend": "recommend", "Source of tension": "tension", "Answer": "answer"}
_RULING_FIELDS = {**_RAISED_FIELDS, "Tension": "tension"}


class FormatError(ValueError):
    """A refinement file breaks the structure documented in lab/refine/TEMPLATES.md."""


@dataclass(frozen=True)
class CodeBlock:
    """A fenced code block of a markdown text, by line index: the opening fence (start), the closing fence (end, or
    the text's last line when the block never closes), the opener's info string and the lines between the fences."""

    start: int
    end: int
    info: str
    body: tuple[str, ...]
    closed: bool

    @property
    def indexes(self) -> range:
        """The indexes of every line the block covers, its fences included."""
        return range(self.start, self.end + 1)


def code_blocks(lines: Sequence[str]) -> list[CodeBlock]:
    """The fenced code blocks among a markdown text's lines, read as CommonMark reads them.

    A block opens at a line of three or more backticks or tildes (any indentation, then an optional info string)
    and closes at the next line holding only a fence of the same character at least as long. A shorter fence, a
    fence of the other character or a fence with text after it is a line of the block, so a block can quote
    another. A block that never closes runs to the end of the text.
    """
    blocks: list[CodeBlock] = []
    opened: tuple[int, str, str] | None = None
    for index, line in enumerate(lines):
        if opened is None:
            found = _FENCE.match(line)
            if found:
                opened = (index, found["fence"], found["info"].strip())
            continue
        start, fence, info = opened
        if re.fullmatch(rf"\s*{re.escape(fence[0])}{{{len(fence)},}}\s*", line):
            blocks.append(CodeBlock(start, index, info, tuple(lines[start + 1:index]), True))
            opened = None
    if opened is not None:
        start, _, info = opened
        blocks.append(CodeBlock(start, len(lines) - 1, info, tuple(lines[start + 1:]), False))
    return blocks


@dataclass(frozen=True)
class Source:
    """One bullet of the research step: a citation and what it says that matters."""

    citation: str
    note: str


@dataclass(frozen=True)
class Reading:
    """One row of a metrics table: a metric, its value and conditions, and where it was read."""

    metric: str
    value: str
    unit: str
    node: str
    conditions: str
    source: str

    @property
    def cited(self) -> bool:
        """True when the source column names a metrics-file or ledger line."""
        return _LINE_REF.fullmatch(self.source) is not None

    @property
    def absent(self) -> bool:
        """True when the row states that no value is stored for this metric yet."""
        return self.value.lower() == "none" and self.source.lower() == "none"

    @property
    def number(self) -> float | None:
        """The value as a number, or None when it is not one."""
        try:
            return float(self.value)
        except ValueError:
            return None


@dataclass(frozen=True)
class Walk:
    """One walked task or experiment, the dimensions it was checked against, and the result."""

    target: str
    dimensions: tuple[str, ...]
    result: str

    @property
    def held(self) -> bool:
        return self.result.startswith("held")


@dataclass(frozen=True)
class Finding:
    """A finding against one dimension. status is fixed, ruling, held or open ('' when none was written)."""

    dimension: str
    where: str
    text: str
    status: str
    fixed: tuple[int, ...] = ()
    rulings: tuple[str, ...] = ()


@dataclass(frozen=True)
class Change:
    """A numbered change with the text before and after, its source, its reason and the metric it moves."""

    number: int
    target: str
    dimensions: tuple[str, ...]
    before: str = ""
    after: str = ""
    source: str = ""
    why: str = ""
    metric: str = ""


@dataclass(frozen=True)
class RaisedRuling:
    """A candidate ruling listed in a pass entry; the merge pass numbers it into OWNER_RULINGS.md.

    answer is None when the candidate has no Answer line; a pass leaves the line blank.
    """

    label: str
    question: str
    now: str = ""
    recommend: str = ""
    tension: str = ""
    answer: str | None = None


@dataclass(frozen=True)
class PassEntry:
    """A '## Pass N' entry of a track log, or a '## Merge N' entry of the merge log (kind 'merge').

    ending is converged or cap when the entry closes with 'CONVERGED at pass N' or
    'CAP REACHED at pass N, not converged' (ending_number is that N), and '' otherwise.
    """

    kind: str
    number: int
    date: str
    summary: str
    sections: frozenset[str] = frozenset()
    sources: tuple[Source, ...] = ()
    metrics_before: tuple[Reading, ...] = ()
    walked: tuple[Walk, ...] = ()
    findings: tuple[Finding, ...] = ()
    changes: tuple[Change, ...] = ()
    rulings_raised: tuple[RaisedRuling, ...] = ()
    metrics_after: tuple[Reading, ...] = ()
    metrics_after_note: str = ""
    converged: bool | None = None
    converged_note: str = ""
    next_pass: str = ""
    checks: str = ""
    ending: str = ""
    ending_number: int | None = None
    unparsed: tuple[str, ...] = ()


@dataclass(frozen=True)
class OwnerCorrection:
    """An owner correction made outside a pass; it never increments the counter."""

    date: str
    title: str
    rulings: tuple[int, ...]
    summary: str
    changes: tuple[Change, ...]


@dataclass(frozen=True)
class RunState:
    """The closing 'State at the end of the run' section.

    first_pass, last_pass and date are None when the heading is the bare title. ended is converged,
    cap or owner according to its 'Run ended:' line, and '' when it has none.
    """

    first_pass: int | None
    last_pass: int | None
    date: str | None
    ended: str
    text: str


@dataclass(frozen=True)
class Log:
    """A parsed log: the header text before the first entry, then the entries in file order."""

    header: str
    entries: tuple[PassEntry | OwnerCorrection | RunState, ...]

    @property
    def passes(self) -> tuple[PassEntry, ...]:
        """The '## Pass N' entries only."""
        return tuple(e for e in self.entries if isinstance(e, PassEntry) and e.kind == "pass")

    @property
    def current_run(self) -> tuple[PassEntry, ...]:
        """The pass entries after the latest run-state section, or all of them when no run has closed.

        The next pass is number len(current_run) + 1 of its run.
        """
        start = 0
        for index, entry in enumerate(self.entries):
            if isinstance(entry, RunState):
                start = index + 1
        return tuple(e for e in self.entries[start:] if isinstance(e, PassEntry) and e.kind == "pass")


@dataclass(frozen=True)
class Ruling:
    """A numbered owner ruling. The Answer line is blank (underscores or nothing) until the owner fills it."""

    number: int
    question: str
    origin: str
    now: str
    recommend: str
    tension: str
    answer: str

    @property
    def answered(self) -> bool:
        return not _blank_answer(self.answer)


def parse_counter(text: str) -> int:
    """Read refine_pass.txt: one non-negative integer, optionally followed by a newline."""
    value = text.strip()
    if not re.fullmatch(r"[0-9]+", value):
        raise FormatError(f"refine_pass.txt must hold one non-negative integer, found {text!r}")
    return int(value)


def parse_report_end(text: str) -> datetime:
    """Read the 'Ended:' line of a pass report: an ISO 8601 date and time with a UTC offset."""
    found = _REPORT_END.search(text)
    if not found:
        raise FormatError("a pass report must state when the pass ended, on a line 'Ended: <date>T<time><offset>'")
    try:
        ended = datetime.fromisoformat(found["when"])
    except ValueError:
        raise FormatError(f"'Ended: {found['when']}' is not an ISO 8601 date and time") from None
    if ended.tzinfo is None:
        raise FormatError(f"'Ended: {found['when']}' has no UTC offset, for example 2026-10-10T06:52:00-04:00")
    return ended


def parse_log(text: str) -> Log:
    """Split a REFINE_LOG.md (or MERGE_LOG.md) into its entries.

    Every '## ' heading must be a pass, merge, owner correction or run-state heading. Headings inside
    fenced code blocks are ignored. A pass or merge number may appear only once.
    """
    header, blocks = _split_h2(text)
    entries: list[PassEntry | OwnerCorrection | RunState] = []
    seen: set[tuple[str, int]] = set()
    for heading, body in blocks:
        entry = _parse_entry(heading, body)
        if isinstance(entry, PassEntry):
            if (entry.kind, entry.number) in seen:
                raise FormatError(f"{entry.kind} {entry.number} appears twice in one log")
            seen.add((entry.kind, entry.number))
        entries.append(entry)
    return Log("\n".join(header).strip(), tuple(entries))


def parse_rulings(text: str) -> tuple[Ruling, ...]:
    """Read an OWNER_RULINGS.md into its numbered rulings.

    A heading that starts with a digit must be a complete ruling: '## <n>. <question> (<origin>)' with
    Now, Recommend, Source of tension and Answer lines. Any other '## ' heading is a free-form note and
    is skipped. Ruling numbers must be unique.
    """
    _, blocks = _split_h2(text)
    rulings: list[Ruling] = []
    seen: set[int] = set()
    for heading, body in blocks:
        if not re.match(r"^## [0-9]", heading):
            continue
        match = _RULING_HEADING.match(heading)
        if not match:
            raise FormatError(f"{heading!r} is not a ruling heading; expected '## <n>. <question> (<origin>)'")
        number = int(match["number"])
        if number in seen:
            raise FormatError(f"ruling {number} appears twice")
        seen.add(number)
        fields = _fields(body, _RULING_FIELDS)
        for key, label in (("now", "Now"), ("recommend", "Recommend"), ("tension", "Source of tension"), ("answer", "Answer")):
            if key not in fields:
                raise FormatError(f"ruling {number} has no '{label}:' line")
        rulings.append(
            Ruling(
                number=number,
                question=match["title"].strip(),
                origin=(match["origin"] or "").strip(),
                now=fields["now"],
                recommend=fields["recommend"],
                tension=fields["tension"],
                answer=fields["answer"],
            )
        )
    return tuple(rulings)


def check_pass_entry(entry: PassEntry) -> list[str]:
    """List what a pass or merge entry is missing; an empty list means the entry is valid.

    A pass entry needs a research step with sources, a metrics-before table whose rows cite a
    metrics-file or ledger line, a walk, dispositioned findings, complete numbered changes that trace
    to findings, rulings raised with a blank Answer line, and a Converged verdict that agrees with its
    closing CONVERGED or CAP REACHED line; a YES verdict comes with no change and no new ruling. A merge entry needs the same except the research step, the
    metrics-before table, the METRIC line and the finding for each change.
    """
    who = f"{entry.kind} {entry.number}"
    full = entry.kind == "pass"
    problems: list[str] = []

    def add(message: str) -> None:
        problems.append(f"{who}: {message}")

    if not entry.summary:
        add("no summary paragraph under the heading")

    if full and not entry.sources:
        add("'Research step' needs at least one source, written '- <citation> :: <what it says>'")
    if any(not source.citation for source in entry.sources):
        add("a research-step bullet has an empty citation")

    if full and not entry.metrics_before:
        add("'Metrics before' needs a table with at least one reading")
    _check_readings(add, "metrics before", entry.metrics_before, allow_absent=True)

    if not entry.walked:
        add("'Walked' needs at least one '- <id> :: <dimension ids> :: held|BROKE ...' line")
    for walk in entry.walked:
        if not walk.dimensions or not all(re.fullmatch(_DIM, d) for d in walk.dimensions):
            add(f"walked {walk.target}: dimensions must be ids like D1, U2; found {list(walk.dimensions)}")
        if not walk.result.startswith(("held", "BROKE")):
            add(f"walked {walk.target}: the result must start with held or BROKE")

    if "findings" not in entry.sections:
        add("no 'Findings' section (write 'Findings (by dimension): none' when there is nothing to report)")
    for finding in entry.findings:
        if not finding.status:
            add(f"the finding for {finding.dimension} {finding.where} has no disposition (FIXED (n), RULING (x), HELD or OPEN)")

    if "changes" not in entry.sections:
        add("no 'Changes' section (write 'Changes (0 of at most N): none' when nothing changed)")
    numbers = [change.number for change in entry.changes]
    if numbers != list(range(1, len(numbers) + 1)):
        add(f"changes must be numbered 1 to {len(numbers)} in order, found {numbers}")
    for change in entry.changes:
        _check_change(add, change, full)
    referenced = {n for finding in entry.findings for n in finding.fixed}
    for n in sorted(referenced - set(numbers)):
        add(f"a finding says FIXED ({n}) but there is no change {n}")
    if full:
        for n in sorted(set(numbers) - referenced):
            add(f"change {n} is not referenced by any finding as FIXED ({n})")

    if "rulings" not in entry.sections:
        add("no 'Rulings raised' section (write 'Rulings raised: none')")
    labels = {ruling.label for ruling in entry.rulings_raised}
    for ruling in entry.rulings_raised:
        fields = (("Now", ruling.now), ("Recommend", ruling.recommend), ("Source of tension", ruling.tension))
        missing = [name for name, value in fields if not value]
        if missing:
            add(f"ruling ({ruling.label}) is missing {', '.join(missing)}")
        if ruling.answer is None:
            add(f"ruling ({ruling.label}) is missing its blank Answer line (write 'Answer: ____')")
        elif not _blank_answer(ruling.answer):
            add(f"ruling ({ruling.label}) has an Answer filled in; only the owner writes an Answer")
    for finding in entry.findings:
        for label in finding.rulings:
            if label not in labels:
                add(f"a finding says RULING ({label}) but no ruling ({label}) is raised")

    if "metrics_after" in entry.sections:
        if not entry.metrics_after and not entry.metrics_after_note.lower().startswith("not run"):
            add("'Metrics after' needs a table or 'not run: <why>'")
        _check_readings(add, "metrics after", entry.metrics_after, allow_absent=False)

    if entry.converged is None and (full or "converged" in entry.sections):
        add("'Converged' must start with YES or NO")
    if entry.converged and entry.changes:
        add(f"Converged is YES but the entry makes {len(entry.changes)} change(s); a converged walk changes nothing")
    if entry.converged and entry.rulings_raised:
        add(
            f"Converged is YES but the entry raises {len(entry.rulings_raised)} ruling(s); "
            "a converged walk raises no new ruling"
        )
    if entry.converged and entry.ending != "converged":
        add(f"Converged is YES, so the entry must end with the line 'CONVERGED at pass {entry.number}'")
    if entry.ending == "converged" and entry.converged is False:
        add("the entry ends with a CONVERGED line but its Converged line says NO")
    if entry.ending == "cap":
        if entry.converged:
            add("the entry ends with a CAP REACHED line but its Converged line says YES")
        if not entry.rulings_raised:
            add("a CAP REACHED entry must raise the ruling that asks whether to extend the run")
    if entry.ending and entry.ending_number != entry.number:
        add(f"the closing line names pass {entry.ending_number} but this is pass {entry.number}")

    for line in entry.unparsed:
        add(f"unparsed line in {line}")
    return problems


def _check_readings(add, name: str, readings: tuple[Reading, ...], allow_absent: bool) -> None:
    for reading in readings:
        if not METRIC_NAME.fullmatch(reading.metric):
            add(f"{name}: {reading.metric!r} is not a dotted lowercase metric name")
        if not reading.value:
            add(f"{name}: {reading.metric} has no value")
        if not (reading.cited or (allow_absent and reading.absent)):
            add(
                f"{name}: {reading.metric} needs a metrics-file or ledger line in the source column "
                f"(metrics.jsonl:<line> or ledger:<seq>), found {reading.source!r}"
            )


def _check_change(add, change: Change, full: bool) -> None:
    n = change.number
    if not change.dimensions:
        add(f"change {n} names no dimension in its heading, e.g. '... (D1, D3)'")
    for field, value in (("BEFORE", change.before), ("AFTER", change.after), ("SOURCE", change.source), ("WHY", change.why)):
        if not value:
            add(f"change {n} has no {field}")
    if full:
        if not change.metric:
            add(f"change {n} has no METRIC")
        elif not (METRIC_NAME.match(change.metric) or change.metric.lower().startswith("none")):
            add(f"change {n}: METRIC must start with a dotted metric name or 'none - <reason>'")


def _split_h2(text: str) -> tuple[list[str], list[tuple[str, list[str]]]]:
    """Return the lines before the first '## ' heading and each heading with its body lines. A '## ' line inside a
    fenced code block (code_blocks) is a line of the body, not a heading."""
    header: list[str] = []
    blocks: list[tuple[str, list[str]]] = []
    lines = text.splitlines()
    fenced = {index for block in code_blocks(lines) for index in block.indexes}
    for index, line in enumerate(lines):
        if index not in fenced and line.startswith("## "):
            blocks.append((line.rstrip(), []))
        elif blocks:
            blocks[-1][1].append(line)
        else:
            header.append(line)
    return header, blocks


def _valid_date(value: str, heading: str) -> str:
    try:
        _date.fromisoformat(value)
    except ValueError:
        raise FormatError(f"{heading!r}: {value!r} is not a calendar date") from None
    return value


def _parse_entry(heading: str, body: list[str]) -> PassEntry | OwnerCorrection | RunState:
    match = _PASS_HEADING.match(heading)
    if match:
        return _parse_pass(match, heading, body)
    if heading.startswith("## Owner correction"):
        return _parse_correction(heading, body)
    match = _STATE_HEADING.match(heading)
    if match:
        first = int(match["first"]) if match["first"] else None
        last = int(match["last"]) if match["last"] else None
        if first is not None and last < first:
            raise FormatError(f"{heading!r}: the run ends before it starts")
        ended = ""
        for line in body:
            how = _RUN_ENDED.match(line)
            if how:
                ended = {"converged": "converged", "cap reached": "cap", "ended by the owner": "owner"}[how["how"].lower()]
                break
        text = "\n".join(line for line in body if line.strip() != "---").strip()
        return RunState(first, last, _valid_date(match["date"], heading) if match["date"] else None, ended, text)
    raise FormatError(
        f"unrecognized log heading {heading!r}; expected '## Pass N, YYYY-MM-DD', '## Merge N, YYYY-MM-DD', "
        "'## Owner correction, YYYY-MM-DD (no increment)' or '## State at the end of the run'"
    )


def _parse_pass(match: re.Match[str], heading: str, body: list[str]) -> PassEntry:
    summary_lines: list[str] = []
    sections: dict[str, list[str]] = {}
    current = summary_lines
    ending, ending_number = "", None
    for line in body:
        section = _SECTION_LINE.match(line)
        closing = _ENDING.match(line)
        if closing:
            if ending:
                raise FormatError(f"{heading!r}: more than one closing CONVERGED or CAP REACHED line")
            ending = "converged" if closing["converged"] else "cap"
            ending_number = int(closing["converged"] or closing["cap"])
        elif section:
            name = _SECTION_NAMES[section["label"]]
            if name in sections:
                raise FormatError(f"{heading!r}: section '{section['label']}' appears twice")
            current = sections[name] = []
            if section["rest"].strip():
                current.append(section["rest"])
        elif line.strip() != "---":
            current.append(line)

    unparsed: list[str] = []

    def items(name: str) -> list[str]:
        found, stray = _bullets(sections.get(name, []))
        unparsed.extend(f"{name}: {line}" for line in stray)
        return found

    sources = tuple(Source(c.strip(), n.strip()) for c, _, n in (item.partition(" :: ") for item in items("research")))

    walked: list[Walk] = []
    for item in items("walked"):
        parts = [part.strip() for part in item.split(" :: ", 2)]
        if len(parts) == 3:
            walked.append(Walk(parts[0], tuple(d.strip() for d in parts[1].split(",") if d.strip()), parts[2]))
        else:
            unparsed.append(f"walked: {item}")

    findings: list[Finding] = []
    for item in items("findings"):
        left, separator, text = item.partition(" :: ")
        parsed = _FINDING_LEFT.match(left.strip())
        if separator and parsed:
            findings.append(_finding(parsed["dim"], parsed["where"].strip(), text.strip()))
        else:
            unparsed.append(f"findings: {item}")

    changes, stray = _parse_changes(sections.get("changes", []))
    unparsed.extend(f"changes: {line}" for line in stray)

    raised, stray = _parse_raised(sections.get("rulings", []))
    unparsed.extend(f"rulings: {line}" for line in stray)

    before, _ = _readings("metrics_before", sections.get("metrics_before", []), unparsed)
    after, after_note = _readings("metrics_after", sections.get("metrics_after", []), unparsed)

    verdict_text = _join(sections.get("converged", []))
    verdict = re.match(r"^(YES|NO)\b[\s:-]*(.*)$", verdict_text)

    return PassEntry(
        kind=match["kind"].lower(),
        number=int(match["number"]),
        date=_valid_date(match["date"], heading),
        summary=_join(summary_lines),
        sections=frozenset(sections),
        sources=sources,
        metrics_before=before,
        walked=tuple(walked),
        findings=tuple(findings),
        changes=changes,
        rulings_raised=raised,
        metrics_after=after,
        metrics_after_note=after_note,
        converged=None if not verdict else verdict[1] == "YES",
        converged_note=verdict[2] if verdict else verdict_text,
        next_pass=_join(sections.get("next", [])),
        checks=_join(sections.get("checks", [])),
        ending=ending,
        ending_number=ending_number,
        unparsed=tuple(unparsed),
    )


def _parse_correction(heading: str, body: list[str]) -> OwnerCorrection:
    rest = heading[len("## Owner correction"):]
    found = re.search(_DATE, rest)
    if not found:
        raise FormatError(f"{heading!r}: an owner correction heading needs a date, e.g. '## Owner correction, 2026-10-10 (no increment)'")
    first_change = next((i for i, line in enumerate(body) if _CHANGE_START.match(line)), len(body))
    rulings: tuple[int, ...] = ()
    summary: list[str] = []
    for line in body[:first_change]:
        applied = _RULINGS_APPLIED.match(line)
        if applied:
            rulings = tuple(int(n) for n in re.split(r"\s*,\s*", applied["numbers"]))
        elif line.strip() != "---":
            summary.append(line)
    changes, _ = _parse_changes([line for line in body[first_change:] if line.strip() != "---"])
    return OwnerCorrection(
        date=_valid_date(found.group(0), heading),
        title=rest[: found.start()].strip(" ,:"),
        rulings=rulings,
        summary=_join(summary),
        changes=changes,
    )


def _finding(dimension: str, where: str, text: str) -> Finding:
    fixed = tuple(int(n) for group in _FIXED.findall(text) for n in re.split(r"\s*,\s*", group))
    rulings = tuple(label for group in _RULING_REF.findall(text) for label in re.split(r"\s*,\s*", group))
    if fixed:
        status = "fixed"
    elif rulings:
        status = "ruling"
    elif re.search(r"\bHELD\b", text):
        status = "held"
    elif re.search(r"\bOPEN\b", text):
        status = "open"
    else:
        status = ""
    return Finding(dimension, where, text, status, fixed, rulings)


def _bullets(lines: list[str]) -> tuple[list[str], list[str]]:
    """Group '- ' bullets with their indented continuation lines; return (bullets, stray lines)."""
    bullets: list[str] = []
    stray: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        if line.startswith("- "):
            bullets.append(line[2:].strip())
        elif bullets and line[:1] in (" ", "\t"):
            bullets[-1] += " " + line.strip()
        elif line.strip().lower().rstrip(".") != "none":
            stray.append(line.strip())
    return bullets, stray


def _cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def _readings(name: str, lines: list[str], unparsed: list[str]) -> tuple[tuple[Reading, ...], str]:
    """Read a pipe table of metric readings; return (readings, any non-table text)."""
    rows = [line.strip() for line in lines if line.strip().startswith("|")]
    note = _join(line for line in lines if line.strip() and not line.strip().startswith("|"))
    readings: list[Reading] = []
    if rows:
        if tuple(cell.lower() for cell in _cells(rows[0])) != _COLUMNS:
            unparsed.append(f"{name}: the table header must be | {' | '.join(_COLUMNS)} |, found {rows[0]}")
        else:
            for row in rows[1:]:
                cells = _cells(row)
                if all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
                    continue
                if len(cells) == len(_COLUMNS):
                    readings.append(Reading(*cells))
                else:
                    unparsed.append(f"{name}: expected {len(_COLUMNS)} cells, found {len(cells)}: {row}")
    return tuple(readings), note


def _parse_changes(lines: list[str]) -> tuple[tuple[Change, ...], list[str]]:
    """Read numbered change blocks; return (changes, lines that belong to no change)."""
    blocks: list[dict[str, str]] = []
    stray: list[str] = []
    field = "head"
    for line in lines:
        if not line.strip():
            continue
        start = _CHANGE_START.match(line)
        labelled = _CHANGE_FIELD.match(line)
        if start:
            blocks.append({"number": start["number"], "head": start["head"].strip()})
            field = "head"
        elif labelled and blocks:
            field = labelled["label"].lower()
            blocks[-1][field] = labelled["value"].strip()
        elif blocks and line[:1] in (" ", "\t"):
            blocks[-1][field] = (blocks[-1].get(field, "") + " " + line.strip()).strip()
        elif line.strip().lower().rstrip(".") != "none":
            stray.append(line.strip())
    changes = []
    for block in blocks:
        head = block["head"]
        dims = _CHANGE_DIMS.search(head)
        changes.append(
            Change(
                number=int(block["number"]),
                target=(head[: dims.start()] if dims else head).strip(),
                dimensions=tuple(d.strip() for d in dims["dims"].split(",")) if dims else (),
                before=_unquote(block.get("before", "")),
                after=_unquote(block.get("after", "")),
                source=block.get("source", ""),
                why=block.get("why", ""),
                metric=block.get("metric", ""),
            )
        )
    return tuple(changes), stray


def _parse_raised(lines: list[str]) -> tuple[tuple[RaisedRuling, ...], list[str]]:
    """Read lettered candidate rulings; return (rulings, lines that belong to none)."""
    blocks: list[tuple[str, str, list[str]]] = []
    stray: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        start = _RAISED_START.match(line)
        if start:
            blocks.append((start["label"], start["question"].strip(), []))
        elif blocks:
            blocks[-1][2].append(line)
        elif line.strip().lower().rstrip(".") != "none":
            stray.append(line.strip())
    raised = []
    for label, question, body in blocks:
        fields = _fields(body, _RAISED_FIELDS)
        raised.append(
            RaisedRuling(
                label, question, fields.get("now", ""), fields.get("recommend", ""), fields.get("tension", ""), fields.get("answer")
            )
        )
    return tuple(raised), stray


def _fields(lines: list[str], labels: dict[str, str]) -> dict[str, str]:
    """Collect 'Label: text' fields; a field runs until the next label. Text before the first label is skipped."""
    pattern = re.compile(r"^\s*(" + "|".join(re.escape(label) for label in labels) + r"):[ \t]*(.*)$")
    collected: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in lines:
        match = pattern.match(line)
        if match:
            current = collected.setdefault(labels[match[1]], [])
            current.append(match[2])
        elif current is not None:
            current.append(line)
    return {key: _join(parts) for key, parts in collected.items()}


def _blank_answer(value: str) -> bool:
    """True for an Answer that is empty or only underscores, the blank the file starts with."""
    return value.strip("_ \t") == ""


def _join(lines) -> str:
    return " ".join(part for part in (line.strip() for line in lines) if part)


def _unquote(value: str) -> str:
    if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    return value
