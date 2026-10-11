"""Check research notes against the format in lab/refine/RESEARCH_FORMAT.md.

A note is a markdown file under lab/refine/research/<track>/. Each finding is one top-level bullet under a topic
heading. It starts with one of four labels (measured, practitioner, anecdote, opinion), carries its citation, and has
an implication and a candidate experiment under it. A track folder also holds a SOURCES.md index of every source cited.

    python -m lab research-check lab/refine/research/telemetry
    python -m lab research-check --ref lab/refine/research/telemetry/sampling.md#idle-baseline

check(path) returns the violations as strings shaped "<file>:<line>: <code>: <message>" (no line for a problem with
the whole file); an empty list means the path conforms. The path is a note, a SOURCES.md, or a folder: a folder is
checked note by note against its SOURCES.md, and a folder with no notes of its own is checked subfolder by subfolder.
check_reference(ref) checks that "path.md#heading" names a real file and heading, which is how a pass cites a finding.

The research-check verb is main(argv). Its one JSON line carries ok, outcome ("clean" or "violations"), count,
violations (the strings above), notes and findings; the exit code is 0 when clean and 1 otherwise.
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lab.machine import refine_formats as rf

__all__ = ["LABELS", "check", "check_reference", "label_of", "main", "slugify"]

CLI_VERB = "research-check"   # registers python -m lab research-check
LABELS = ("measured", "practitioner", "anecdote", "opinion")

# What a track asks of a candidate experiment besides its expected movement. A folder that is not a track gets the default.
TRACK_EXTRAS = {
    "telemetry": ("minimum",),
    "learning": ("minimum",),
    "plumbing": (),
    "kernels": ("minimum", "confidence"),
}
DEFAULT_EXTRAS = ("minimum",)

INDEX_NAME = "SOURCES.md"
NOT_NOTES = frozenset({"sources.md", "readme.md"})
RECORD_PATTERN = "measurements*.md"  # a raw measurement record that notes cite, not a note

# The framing headings RESEARCH_FORMAT.md section 2 names, whose bullets are not findings, each in the singular and the
# plural: Scope, Summary, Overview, Background, Method, Search log, Queries, Sources, References, Open questions, Gaps,
# Limitations, Caveats, Next steps, Takeaways, Inferences, Conclusions, and Candidate experiments (ranked). A heading
# matches when, after numbering and a trailing parenthesis are dropped, it is one of these names, or it starts with one
# of the prefixes: a multi-word framing name followed by what it is about ("Open questions for R33"), the forms of the
# ranked experiment list section 6 asks the kernels note to end with ("candidate experiment", "ranked", "ranking"),
# and "search notes", the search log's own record of refused pages in the committed kernels hardware note. Any other
# heading is a topic: a one-word name followed by more words ("Sources of error", "Sources opened") is a topic, and so
# is a heading the format does not name ("Decisions").
_EXEMPT_WORDS = frozenset(
    {
        "scope", "summary", "overview", "background", "method", "methods", "query", "queries",
        "source", "sources", "reference", "references", "gap", "gaps",
        "limitation", "limitations", "caveat", "caveats", "takeaway", "takeaways",
        "inference", "inferences", "conclusion", "conclusions",
    }
)
_EXEMPT_PREFIXES = ("search log", "open question", "next step", "candidate experiment", "ranked", "ranking", "search notes")

_CODE_SPAN = re.compile(r"`[^`\n]*`")
_HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$")
_ITEM = re.compile(r"^(?P<indent>[ \t]*)(?:[-*+]|\d{1,3}[.)])[ \t]+(?P<text>\S.*)$")
_RULE = re.compile(r"^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$")
_COMMENT = re.compile(r"<!--.*?-->", re.S)

# A bullet that continues the finding above it instead of starting a new one.
_ATTACH = re.compile(
    r"^[*_`\s]*(?:implications?|implies|metric implications?|candidate experiments?|experiments?|expected|minimum"
    r"|confidence|sources?|citations?|cite|references?)\b",
    re.I,
)
_SOURCE_LEAD = re.compile(r"^[*_`\s]*(?:sources?|citations?|cite|references?|refs?)\b", re.I)

# The four ways a label is written at the start of a bullet: [label], (label), **label**, and "label:".
_SQUARE = re.compile(r"^[*_`\s]*\[(?P<inner>[^\]\n]{1,80})\](?!\()")
_PAREN = re.compile(r"^[*_`\s]*\((?P<inner>[^)\n]{1,80})\)")
_EMPHASIZED = re.compile(
    r"^(?:\*{1,3}|_{1,3}|`)\s*(?P<inner>[A-Za-z]+(?:\s*[,;/][^*_`\n]{0,60})?)\s*[:.]?\s*(?:\*{1,3}|_{1,3}|`)"
)
_PLAIN = re.compile(r"^(?P<inner>[A-Za-z]+)\s*(?::|\s[-\u2013\u2014]\s)")
_LATER_LABEL = re.compile(r"\[(?:" + "|".join(LABELS) + r")\b[^\]\n]*\](?!\()", re.I)

# Citation forms: a URL, a DOI, an arXiv id, an issue or PR, a ledger line, or a source file (and function).
_URL = re.compile(r"https?://[\w-]+(?:\.[\w-]+)+(?::\d+)?(?:[/?#][^\s<>\"'`)\]*]*)?", re.I)
_DOI = re.compile(r"(?<![\w/.-])(?:doi:\s*)?(?P<doi>10\.\d{4,9}/[^\s<>\"'`)\]]+)", re.I)
_ARXIV = re.compile(r"\barxiv[:\s]\s*(?P<id>\d{4}\.\d{4,5})(?:v\d+)?\b", re.I)
_TRACKER = re.compile(r"(?<![\w/.-])[\w.-]+/[\w.-]+#\d+\b|\bPR\s*#\d+\b")
_LEDGER = re.compile(r"\bledger(?:\s+(?:seq|line|row|entry))?\s*[:#]?\s*\d+\b", re.I)
_EXT = "(?:" + "|".join(rf.FILE_EXTENSIONS) + ")"
_WHERE = r"(?:::[A-Za-z_][\w.]*(?:\(\))?|:\d+(?:-\d+)?|#L\d+(?:-L?\d+)?)"
_CODE = re.compile(
    r"(?<![\w/.@:-])(?:[\w.-]+/)+[\w.-]+\." + _EXT + r"\b(?:" + _WHERE + r")?"
    r"|(?<![\w/.@:-])[\w.-]+\." + _EXT + r"\b" + _WHERE
)

# A source ID such as [S49] or [S3, S7] points at an entry of SOURCES.md; a relative link such as
# [log](measurements.md#m1) points at a record in this repo.
_SOURCE_ID = re.compile(r"\[([A-Z]\d{1,4}(?:\s*,\s*[A-Z]\d{1,4})*)\]")
_REL_LINK = re.compile(r"\]\((?![A-Za-z][\w+.-]*://|#)[^)\s]+\." + _EXT + r"(?:#[^)\s]*)?\)", re.I)
_INDEX_ID = re.compile(r"^[\s|*_`\[(]*([A-Z])(\d+)\b")

# An experiment line that points at another experiment of the note instead of repeating it.
_XREF = re.compile(
    r"(?i:\bsee\b[^.;]{0,60}\b(?:previous|preceding|above|below|earlier|next|finding|experiment|section|test)\b)"
    r"|(?i:\bcovered by\b|\bspecified in\b|\bsame (?:experiment )?as\b)"
    r"|^(?i:experiments?)\W+(?:\([^)]*\)\W*)?(?:(?i:none)\b|[A-Z]{1,3}-?\d{1,3}\b)"
)

_IMPLICATION = re.compile(r"\bimplic\w*|\bimplies\b|\bimply\w*", re.I)
_EXPERIMENT = re.compile(r"\bexperiments?\b", re.I)
_EXPERIMENT_LINE = re.compile(r"\bexperiments?\b[^:\n]{0,40}:", re.I)  # "Experiment:", "Candidate experiment (EN-6):"
_EXPECTED = re.compile(r"\bexpect\w*|\bmovement\b|\bdelta\b", re.I)
_MINIMUM = re.compile(
    r"\bminimum\b|\bmin\.?\s+(?:n\b|samples?|runs?)|\bsample size\b|\bn\s*(?:=|>=|>|\u2265)\s*\d+|\bat least\s+\d+"
    r"|\b\d+\s*\+?\s*(?:runs?|samples?|trials?|requests?|repeats?|repetitions?|seeds?|prompts?|questions?|launches?|windows?)\b",
    re.I,
)
_CONFIDENCE = re.compile(r"\bconfiden(?:ce|t)\b|\bconf\b", re.I)

# An index entry may record a lead that was seen but never opened; it has no link and is not cited.
_UNOPENED = re.compile(r"\bnot (?:opened|citable)\b", re.I)

_MONTH =r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?"
_DATE = re.compile(
    r"\b\d{4}[-/]\d{2}[-/]\d{2}\b|\b" + _MONTH + r"\s+\d{1,2},?\s+\d{4}\b|\b\d{1,2}\s+" + _MONTH + r"\s+\d{4}\b",
    re.I,
)


@dataclass
class _Item:
    line: int
    indent: int
    text: str
    more: list[str] = field(default_factory=list)

    @property
    def full(self) -> str:
        return " ".join([self.text, *self.more])


@dataclass
class _Section:
    heading: str
    level: int
    line: int
    exempt: bool
    items: list[_Item] = field(default_factory=list)


@dataclass
class _Block:
    """A top-level bullet with everything under it."""

    line: int
    lead: str
    details: list[str] = field(default_factory=list)
    source_lines: list[str] = field(default_factory=list)


@dataclass
class _Report:
    violations: list[str] = field(default_factory=list)
    notes: int = 0
    findings: int = 0

    def add(self, where: str, line: int | None, code: str, message: str) -> None:
        self.violations.append(f"{where}:{line}: {code}: {message}" if line else f"{where}: {code}: {message}")


# ---- reading markdown -------------------------------------------------------------------------------------


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="replace")


def _clean(text: str) -> list[str]:
    """The file's lines with front matter and HTML comments blanked out, so line numbers still match the file, and
    file names Notion turned into links read back as names (refine_formats.unlink_file_names)."""
    lines = re.split(r"\r\n|\r|\n", text)
    if lines and lines[0].strip() == "---":
        for end in range(1, len(lines)):
            if lines[end].strip() in ("---", "..."):
                lines[: end + 1] = [""] * (end + 1)
                break
    joined = rf.unlink_file_names("\n".join(lines))
    return _COMMENT.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), joined).split("\n")


def _visible(lines: list[str]):
    """(line number, line) for every line outside a fenced code block (refine_formats.code_blocks)."""
    expanded = [raw.expandtabs(4) for raw in lines]
    fenced = {index for block in rf.code_blocks(expanded) for index in block.indexes}
    for index, line in enumerate(expanded):
        if index not in fenced:
            yield index + 1, line


def _normalized_heading(text: str) -> str:
    text = re.sub(r"[*_`]", "", text).strip()
    text = re.sub(r"^(?:\d+(?:\.\d+)*[.)]?|[A-Za-z][.)])\s+", "", text)
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text)
    return re.sub(r"\s+", " ", text.rstrip(":").strip().lower())


def _is_exempt(heading: str) -> bool:
    name = _normalized_heading(heading)
    return name in _EXEMPT_WORDS or name.startswith(_EXEMPT_PREFIXES)


def _sections(lines: list[str]) -> list[_Section]:
    """Split a note at its headings and collect the list items under each. Section 0 holds what precedes the first heading."""
    sections = [_Section("", 0, 0, True)]
    stack: list[tuple[int, str]] = []
    item: _Item | None = None
    blank = False
    for number, line in _visible(lines):
        if not line.strip():
            blank = True
            continue
        heading = _HEADING.match(line)
        if heading:
            level, text = len(heading.group(1)), heading.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text))
            exempt = level == 1 or any(_is_exempt(t) for lvl, t in stack if lvl >= 2)
            sections.append(_Section(text, level, number, exempt))
            item = None
        elif (found := _ITEM.match(line)) and not _RULE.match(line):
            item = _Item(number, len(found.group("indent")), found.group("text").strip())
            sections[-1].items.append(item)
        elif item is not None and (not blank or len(line) - len(line.lstrip()) > item.indent):
            item.more.append(line.strip())
        else:
            item = None
        blank = False
    return sections


def _blocks(items: list[_Item], attach: bool) -> list[_Block]:
    """Group a section's items into top-level bullets, each with the deeper items under it."""
    if not items:
        return []
    base = min(i.indent for i in items)
    blocks: list[_Block] = []
    for item in items:
        under = blocks and (item.indent > base or (attach and _ATTACH.match(item.text)))
        if not under:
            blocks.append(_Block(item.line, item.full))
            continue
        blocks[-1].details.append(item.full)
        if _SOURCE_LEAD.match(item.text):
            blocks[-1].source_lines.append(item.full)
    return blocks


# ---- labels -----------------------------------------------------------------------------------------------


def _classify(inner: str, bracketed: bool) -> tuple[str, str | None]:
    parts = [p.strip().lower() for p in re.split(r"[,;/|]", inner)]
    if parts[0] in LABELS:
        return ("multiple", inner.strip()) if any(p in LABELS for p in parts[1:]) else ("label", parts[0])
    if bracketed and re.fullmatch(r"[A-Za-z][A-Za-z '/-]{0,40}", inner.strip()):
        return "unknown", inner.strip()
    return "none", None


def _parse_label(text: str) -> tuple[str, str | None, int]:
    """("label" | "multiple" | "unknown" | "none", the label or the offending text, where the label ends)."""
    for pattern, bracketed in ((_SQUARE, True), (_PAREN, False), (_EMPHASIZED, False), (_PLAIN, False)):
        match = pattern.match(text)
        if match:
            kind, value = _classify(match.group("inner"), bracketed)
            if kind != "none":
                return kind, value, match.end()
    return "none", None, 0


def label_of(text: str) -> str | None:
    """The label a bullet's text starts with, or None when it has none, an unknown one, or several."""
    kind, value, _ = _parse_label(text.lstrip())
    return value if kind == "label" else None


# ---- citations and the keys that match them to the SOURCES index ------------------------------------------


def _urls(text: str) -> list[str]:
    """The URLs in the text, leaving out templates such as https://host/user/<name>/ or .../users/{name}/server."""
    found = []
    for match in _URL.finditer(text):
        url = match.group(0)
        if "{" in url or text[match.end():match.end() + 1] == "<":
            continue
        found.append(url.rstrip(".,;:!?"))
    return found


def _has_citation(text: str) -> bool:
    return bool(
        _urls(text) or _DOI.search(text) or _ARXIV.search(text) or _TRACKER.search(text) or _LEDGER.search(text)
        or _CODE.search(text) or _SOURCE_ID.search(_CODE_SPAN.sub(" ", text)) or _REL_LINK.search(text)
    )


def _source_ids(text: str) -> list[str]:
    """The source IDs cited in the text (S49 for [S49]), leaving out `code spans`."""
    found = _CODE_SPAN.sub(" ", text)
    return [f"{letter}{int(number)}" for m in _SOURCE_ID.finditer(found) for letter, number in re.findall(r"([A-Z])(\d+)", m.group(1))]


def _doi_keys(doi: str) -> set[str]:
    doi = doi.rstrip(".,;:)").lower()
    keys = {"doi:" + doi}
    arxiv = re.fullmatch(r"10\.48550/arxiv\.(\d{4}\.\d{4,5})(?:v\d+)?", doi)
    if arxiv:
        keys.add("arxiv:" + arxiv.group(1))
    return keys


def _url_keys(url: str) -> set[str]:
    path = re.sub(r"^www\.", "", re.sub(r"^https?://", "", url, flags=re.I), flags=re.I)
    path = path.split("#", 1)[0].rstrip("/")
    keys = {path.lower()}
    arxiv = re.match(r"^arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?$", path, re.I)
    if arxiv:
        keys.add("arxiv:" + arxiv.group(1))
    doi = re.search(r"(10\.\d{4,9}/[^\s?#]+)", path)
    if doi:
        keys |= _doi_keys(doi.group(1))
    # Source code is read at a ref, so a file URL also stands for the repo at that tag or commit.
    code = re.match(r"^(github\.com/[^/]+/[^/]+)/(?:blob|tree)/([^/]+)/", path, re.I)
    if code:
        repo, ref = code.group(1).lower(), code.group(2).lower()
        keys.add(f"repo:{repo}@{ref}")
        if re.fullmatch(r"[0-9a-f]{7,40}", ref):
            keys.add(f"repo:{repo}@{ref[:7]}")
    return keys


def _cited_keys(text: str) -> list[tuple[str, set[str]]]:
    """Each URL, DOI and arXiv id in the text with the keys that identify it, for matching against the index.

    Text in `code spans` is an example or a literal, not a source, so it is left out.
    """
    text = _CODE_SPAN.sub(" ", text)
    cited = [(url, _url_keys(url)) for url in _urls(text)]
    cited += [(f"doi:{m.group('doi')}", _doi_keys(m.group("doi"))) for m in _DOI.finditer(text)]
    cited += [(f"arXiv:{m.group('id')}", {"arxiv:" + m.group("id")}) for m in _ARXIV.finditer(text)]
    return cited


# ---- notes ------------------------------------------------------------------------------------------------


def _check_note(path: Path, report: _Report, index_keys: set[str] | None) -> None:
    where = path.as_posix()
    sections = _sections(_clean(_read(path)))
    extras = TRACK_EXTRAS.get(path.absolute().parent.name.lower(), DEFAULT_EXTRAS)
    problems: list[tuple[int, str, str]] = []
    report.notes += 1

    first_seen: dict[str, int] = {}
    for section in sections:
        if section.level == 2:
            slug = slugify(section.heading)
            if slug in first_seen:
                problems.append((
                    section.line, "duplicate-heading",
                    f"'## {section.heading}' repeats line {first_seen[slug]}; a pass cites a finding by its heading, "
                    "so each ## heading must be unique in the note",
                ))
            first_seen.setdefault(slug, section.line)

    count = 0
    for section in sections:
        if section.exempt:
            continue
        for block in _blocks(section.items, attach=True):
            count += 1
            _check_finding(block, extras, index_keys, problems)
    report.findings += count
    if not count:
        problems.append((0, "no-findings", "no findings: put each finding in a bullet under a ## topic heading"))

    for line, code, message in sorted(problems, key=lambda p: p[0]):
        report.add(where, line or None, code, message)


def _check_finding(block: _Block, extras: tuple[str, ...], index_keys: set[str] | None, problems: list) -> None:
    line = block.line

    def problem(code: str, message: str) -> None:
        problems.append((line, code, message))

    names = ", ".join(LABELS)
    kind, value, end = _parse_label(block.lead)
    if kind == "label" and _LATER_LABEL.search(block.lead[end:]):
        kind = "multiple"
    if kind == "none":
        problem("no-label", f"bullet has no label; start it with one of {names}, for example **[measured]**")
    elif kind == "unknown":
        problem("unknown-label", f"'{value}' is not a label; use one of {names}")
    elif kind == "multiple":
        problem("multiple-labels", "a finding takes exactly one label; give each finding its own bullet")

    cited = " ".join([block.lead, *block.source_lines])
    if not _has_citation(cited):
        problem(
            "no-citation",
            "finding has no citation; add a URL, DOI, arXiv id, owner/repo#N, ledger:N, or source file and function",
        )
    elif index_keys is not None:
        for shown, keys in _cited_keys(cited):
            if not keys & index_keys:
                problem("source-not-indexed", f"{shown} is cited but not listed in {INDEX_NAME}")
        for source_id in _source_ids(cited):
            if f"id:{source_id}" not in index_keys:
                problem("source-not-indexed", f"[{source_id}] is cited but {INDEX_NAME} has no entry with that ID")

    whole = " ".join([block.lead, *block.details])
    if not _IMPLICATION.search(whole):
        problem("no-implication", "finding has no implication for our metric set; add an 'Implication:' line under it")
    experiment = _EXPERIMENT_LINE.search(whole) or _EXPERIMENT.search(whole)
    if not experiment:
        problem("no-experiment", "finding has no candidate experiment; add a 'Candidate experiment:' line under it")
        return
    text = whole[experiment.start():]
    if _XREF.search(text):
        return
    if not _EXPECTED.search(text):
        problem("no-expected-movement", "candidate experiment does not state the expected movement")
    if "minimum" in extras and not _MINIMUM.search(text):
        problem("no-minimum-sample", "candidate experiment does not state the minimum sample size or number of runs")
    if "confidence" in extras and not _CONFIDENCE.search(text):
        problem("no-confidence", "candidate experiment does not state a confidence")


# ---- the SOURCES index ------------------------------------------------------------------------------------


def _table_rows(lines: list[str]) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    skipping = False
    for number, line in _visible(lines):
        text = line.strip()
        if not text.startswith("|"):
            skipping = False
            continue
        cells = [cell.strip() for cell in text.strip("|").split("|")]
        if all(re.fullmatch(r":?-+:?", cell) for cell in cells):
            if rows and rows[-1][0] == number - 1:
                header = rows.pop()[1].lower()  # the row above a separator is the header
                skipping = not re.search(r"url|doi|link|opened|date", header)  # e.g. a table that lists the notes
            continue
        if not skipping:
            rows.append((number, " | ".join(cells)))
    return rows


def _index_entries(lines: list[str]) -> list[tuple[int, str]]:
    """One (line, text) per source: each table row, and each top-level bullet with the items under it."""
    entries = list(_table_rows(lines))
    for section in _sections(lines):
        entries += [(b.line, " ".join([b.lead, *b.details])) for b in _blocks(section.items, attach=False)]
    return sorted(entries)


def _index_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    for _, text in _index_entries(_clean(_read(path))):
        for _, found in _cited_keys(text):
            keys |= found
        entry_id = _INDEX_ID.match(text)
        if entry_id:
            keys.add(f"id:{entry_id.group(1)}{int(entry_id.group(2))}")
    return keys


def _check_index(path: Path, report: _Report) -> None:
    where = path.as_posix()
    entries = _index_entries(_clean(_read(path)))
    if not entries:
        report.add(where, None, "sources-empty", "no sources are listed; list every source opened, one per row or bullet")
    for line, text in entries:
        if not _has_citation(text) and not _UNOPENED.search(text):
            report.add(where, line, "source-no-url", "source has no URL or DOI; give the link that was opened")
        if not _DATE.search(text):
            report.add(where, line, "source-no-date", "source has no date opened; add it as YYYY-MM-DD")


# ---- paths ------------------------------------------------------------------------------------------------


def _find_index(folder: Path) -> Path | None:
    return next((p for p in sorted(folder.iterdir()) if p.is_file() and p.name.lower() == INDEX_NAME.lower()), None)


def _notes_in(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".md" and p.name.lower() not in NOT_NOTES
        and not fnmatch.fnmatch(p.name.lower(), RECORD_PATTERN)
    )


def _check_folder(folder: Path, report: _Report, explicit: bool) -> None:
    notes, index = _notes_in(folder), _find_index(folder)
    if notes:
        keys = _index_keys(index) if index else None
        for note in notes:
            _check_note(note, report, keys)
        if index:
            _check_index(index, report)
        else:
            report.add(folder.as_posix(), None, "sources-missing", f"folder has notes but no {INDEX_NAME}")
        return
    subfolders = [d for d in sorted(folder.iterdir()) if d.is_dir() and not d.name.startswith(".")]
    for sub in subfolders:
        _check_folder(sub, report, explicit=False)
    if explicit and not subfolders:
        report.add(folder.as_posix(), None, "no-notes", "no notes found in this folder")


def _check_path(path: Path, report: _Report) -> None:
    where = path.as_posix()
    if not path.exists():
        report.add(where, None, "missing-path", "no such file or folder")
    elif path.is_dir():
        _check_folder(path, report, explicit=True)
    elif path.suffix.lower() != ".md":
        report.add(where, None, "not-markdown", "notes and indexes are markdown (.md) files")
    elif path.name.lower() == INDEX_NAME.lower():
        _check_index(path, report)
    else:
        index = _find_index(path.parent)
        _check_note(path, report, _index_keys(index) if index else None)


def check(path: str | os.PathLike[str]) -> list[str]:
    """Violations in a note, a SOURCES.md, or a folder of notes; an empty list means it conforms."""
    report = _Report()
    _check_path(Path(path), report)
    return report.violations


# ---- citing a finding by file and heading -----------------------------------------------------------------


def slugify(heading: str) -> str:
    """The anchor GitHub gives a heading: lowercase, punctuation dropped, each space a hyphen."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)
    text = re.sub(r"<[^>]+>", "", text).strip().lower()
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def _heading_slugs(path: Path) -> set[str]:
    seen: dict[str, int] = {}
    slugs: set[str] = set()
    for section in _sections(_clean(_read(path)))[1:]:
        base = slugify(section.heading)
        slugs.add(base if base not in seen else f"{base}-{seen[base]}")
        seen[base] = seen.get(base, 0) + 1
    return slugs


def check_reference(ref: str, root: str | os.PathLike[str] | None = None) -> list[str]:
    """Check that "path/to/note.md#heading" names a markdown file under root and a heading in it.

    The heading is its GitHub slug ("sampler-trust") or its text ("Sampler trust"); a repeated heading takes -1, -2, ...
    """
    base = Path(root) if root is not None else Path.cwd()
    file_part, _, heading = ref.partition("#")

    def bad(message: str) -> list[str]:
        return [f"{ref}: bad-reference: {message}"]

    if not file_part.lower().endswith(".md"):
        return bad("the part before # must be a markdown file")
    target = (base / file_part).resolve()
    try:
        target.relative_to(base.resolve())
    except ValueError:
        return bad("the file is outside the repository")
    if not target.is_file():
        return bad(f"no such file: {file_part}")
    if not heading.strip():
        return bad("name the heading after #, as in path/to/note.md#heading-slug")
    if slugify(heading) not in _heading_slugs(target):
        return bad(f"no heading with the slug '{slugify(heading)}' in {file_part}")
    return []


# ---- the research-check verb ------------------------------------------------------------------------------


def _expand(arg: str) -> list[str]:
    """PowerShell passes wildcards through untouched, so expand them here."""
    if any(c in arg for c in "*?[") and not os.path.exists(arg):
        return sorted(glob.glob(arg)) or [arg]
    return [arg]


def main(argv: Sequence[str]) -> dict[str, Any]:
    """Check research notes and SOURCES.md indexes against lab/refine/RESEARCH_FORMAT.md."""
    parser = argparse.ArgumentParser(prog="python -m lab research-check", description=main.__doc__)
    parser.add_argument("paths", nargs="*", help="a note, a SOURCES.md, or a folder of notes; wildcards are expanded")
    parser.add_argument(
        "--ref", action="append", default=[], metavar="FILE#HEADING",
        help="check that a file-and-heading reference resolves (repeatable)",
    )
    parser.add_argument("--root", default=None, help="folder that --ref files are relative to (default: the current folder)")
    args = parser.parse_args(argv)
    if not args.paths and not args.ref:
        parser.error("give at least one path or --ref")

    report = _Report()
    for arg in args.paths:
        for path in _expand(arg):
            _check_path(Path(path), report)
    for ref in args.ref:
        report.violations += check_reference(ref, args.root)
    return {
        "ok": not report.violations,
        "outcome": "violations" if report.violations else "clean",
        "count": len(report.violations),
        "violations": report.violations,
        "notes": report.notes,
        "findings": report.findings,
    }
