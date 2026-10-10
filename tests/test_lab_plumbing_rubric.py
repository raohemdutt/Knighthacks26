"""Shape checks for the plumbing track rubric, lab/refine/plumbing/RUBRIC.md.

Every refinement pass on the plumbing track reads this file to learn what binds it
(the two universal rules), what it checks (the eight plumbing tests), what it
measures (the metric set), and what it may cite (the source list). The rubric is a
document, so there is no function to call; these tests parse it by its headings and
fail when one of the parts the passes rely on goes missing or loses its content.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUBRIC = ROOT / "lab" / "refine" / "plumbing" / "RUBRIC.md"

SECTION_PREFIXES = [
    "The goal",
    "The two universal rules",
    "Primary sources",
    "The rubric: eight plumbing tests",
    "The metric set",
    "The walk list",
    "What a pass may NOT change",
    "The pass procedure",
    "Log entry format",
    "Known open items for pass 1",
    "Sources",
]

TEST_FIELDS = ["Passes when:", "Sources predict:", "Checked by:", "Moves:", "Today:"]

METRIC_NAMES = [
    "pod.credits_per_hour",
    "pod.launch_to_serving_s",
    "link.push_mb_s",
    "link.pull_mb_s",
    "pod.recovery_s",
    "link.failed_call_rate",
]


def _text() -> str:
    assert RUBRIC.is_file(), f"expected the plumbing rubric at {RUBRIC.relative_to(ROOT)}"
    return RUBRIC.read_text(encoding="utf-8")


def _split(text: str, marker: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, body) pairs at lines starting with marker.

    Lines inside fenced code blocks are never treated as headings.
    """
    parts: list[tuple[str, str]] = []
    heading = None
    body: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
        if not in_fence and line.startswith(marker):
            if heading is not None:
                parts.append((heading, "\n".join(body)))
            heading = line[len(marker):].strip()
            body = []
        elif heading is not None:
            body.append(line)
    if heading is not None:
        parts.append((heading, "\n".join(body)))
    return parts


def _section(prefix: str) -> str:
    matches = [body for heading, body in _split(_text(), "## ") if heading.startswith(prefix)]
    assert len(matches) == 1, f"expected exactly one section starting with {prefix!r}, found {len(matches)}"
    return matches[0]


def test_sections_follow_the_refine_goal_shape_in_order():
    headings = [heading for heading, _ in _split(_text(), "## ")]
    found = []
    for prefix in SECTION_PREFIXES:
        hits = [i for i, heading in enumerate(headings) if heading.startswith(prefix)]
        assert len(hits) == 1, f"expected one section starting with {prefix!r}, got headings {headings}"
        found.append(hits[0])
    assert found == sorted(found), f"sections out of order: {headings}"


def test_two_universal_rules_are_old_versus_new_then_source_forced():
    body = _section("The two universal rules")
    rules = re.findall(r"^(\d+)\. (.+)$", body, flags=re.MULTILINE)
    assert [number for number, _ in rules] == ["1", "2"]
    assert rules[0][1].startswith("Old versus new.")
    assert rules[1][1].startswith("Source-forced.")
    assert "lab/refine/research/plumbing/" in body


def test_eight_plumbing_tests_are_numbered_and_fully_specified():
    body = _section("The rubric: eight plumbing tests")
    blocks = _split(body, "### ")
    assert [heading.split(".", 1)[0] for heading, _ in blocks] == [f"T{n}" for n in range(1, 9)]
    for heading, block in blocks:
        for field in TEST_FIELDS:
            assert f"- {field}" in block, f"{heading} has no '{field}' line"
        predicted = block.split("- Sources predict:", 1)[1].split("- Checked by:", 1)[0]
        cites = re.findall(r"https://\S+|[\w./-]+\.(?:py|md|yaml|yml|sh)\b|task page|Design Doc", predicted)
        assert cites, f"{heading} predicts without citing a source"


def test_metric_set_names_every_task_metric_with_unit_and_direction():
    body = _section("The metric set")
    rows = [line for line in body.splitlines() if line.startswith("|")]
    names, directions = [], []
    for row in rows[2:]:
        cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
        assert len(cells) == 6, f"metric row should have 6 cells: {row}"
        name = cells[1].strip("`")
        assert re.fullmatch(r"[a-z]+(\.[a-z0-9_]+)+", name), f"metric name not dotted lowercase: {cells[1]}"
        assert cells[2], f"metric {name} has no unit"
        names.append(name)
        directions.append(cells[5].split()[0])
    assert names == METRIC_NAMES
    assert directions == ["lower", "lower", "higher", "higher", "lower", "lower"]


def test_every_source_is_a_link_and_source_files_are_pinned():
    body = _section("Sources")
    bullets = [line for line in body.splitlines() if line.startswith("- ")]
    assert len(bullets) >= 20, f"expected the full source list, found {len(bullets)} entries"
    for bullet in bullets:
        assert "https://" in bullet, f"source without a link: {bullet}"
    refs = re.findall(r"https://github\.com/[^/\s]+/[^/\s]+/blob/([^/\s]+)/", body)
    assert refs, "expected pinned GitHub source links"
    for ref in refs:
        pinned = re.fullmatch(r"[0-9a-f]{7,40}|v?\d+(\.\d+)+", ref)
        assert pinned, f"GitHub link not pinned to a tag or commit: {ref}"


def test_plain_hyphens_and_no_token_in_context():
    # Notion page ids in the source links are 32 hex characters too, so a token is
    # recognised by its context (an Authorization header or a token= assignment).
    text = _text()
    assert "—" not in text and "–" not in text, "use plain hyphens, not em or en dashes"
    leaked = re.findall(r"(?i)(?:authorization:\s*token\s+|token\s*[=:]\s*[\"']?)[0-9a-f]{32}\b", text)
    assert leaked == [], "the rubric must never carry a hub token"
