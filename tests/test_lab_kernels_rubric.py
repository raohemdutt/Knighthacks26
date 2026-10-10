"""Structure checks for the kernels and speed track rubric.

The rubric is the file every refinement pass on the kernels track is judged
against. These checks keep its required parts in place: the section order of
the refinement-goal shape, the two universal rules and the eight kernel tests,
the metric set, the recorded baseline, and a source list in which every cited
source exists, carries a link, and is actually cited.
"""

from __future__ import annotations

import re
from pathlib import Path

RUBRIC = Path(__file__).resolve().parents[1] / "lab" / "refine" / "kernels" / "RUBRIC.md"

SECTIONS = [
    "The goal",
    "How passes are run",
    "Primary sources",
    "The rubric",
    "The walk list",
    "The metric set",
    "The baseline to beat",
    "What a pass may NOT change",
    "The pass procedure",
    "Known open items for pass 1",
    "Sources",
]

RULE_IDS = ["U1", "U2"] + [f"K{n}" for n in range(1, 9)]

CITATION = re.compile(r"\[S(\d+)\]")
SOURCE_LINE = re.compile(r"^- \[S(\d+)\] (.+)$")
RULE_HEADING = re.compile(r"^### ([UK]\d+)\. (.+)$")


def _text() -> str:
    assert RUBRIC.is_file(), f"expected the kernels rubric at {RUBRIC}"
    return RUBRIC.read_text(encoding="utf-8")


def _sections(text: str) -> list[tuple[str, str]]:
    """Split the file into (level-2 heading, body) pairs, in file order."""
    parts: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        if line.startswith("## "):
            parts.append((line[3:].strip(), []))
        elif parts:
            parts[-1][1].append(line)
    return [(title, "\n".join(body)) for title, body in parts]


def _section(text: str, name: str) -> str:
    matches = [body for title, body in _sections(text) if title.startswith(name)]
    assert len(matches) == 1, f"expected exactly one section starting with {name!r}, found {len(matches)}"
    return matches[0]


def _rules(text: str) -> list[tuple[str, str, str]]:
    """Return (id, title, body) for each level-3 rule heading inside The rubric."""
    rules: list[tuple[str, str, list[str]]] = []
    for line in _section(text, "The rubric").splitlines():
        found = RULE_HEADING.match(line)
        if found:
            rules.append((found.group(1), found.group(2), []))
        elif rules:
            rules[-1][2].append(line)
    return [(rule_id, title, "\n".join(body).strip()) for rule_id, title, body in rules]


def _sources(text: str) -> dict[int, str]:
    entries: dict[int, str] = {}
    for line in _section(text, "Sources").splitlines():
        found = SOURCE_LINE.match(line)
        if found:
            number = int(found.group(1))
            assert number not in entries, f"source S{number} is listed twice"
            entries[number] = found.group(2)
    return entries


def _text_outside_sources(text: str) -> str:
    return "\n".join(body for title, body in _sections(text) if not title.startswith("Sources"))


def test_sections_follow_the_refine_goal_order() -> None:
    titles = [title for title, _ in _sections(_text())]
    expected_in_file = [name for title in titles for name in SECTIONS if title.startswith(name)]
    assert expected_in_file == SECTIONS


def test_rubric_holds_two_universal_rules_and_eight_kernel_tests() -> None:
    rules = _rules(_text())
    assert [rule_id for rule_id, _, _ in rules] == RULE_IDS
    empty = [rule_id for rule_id, _, body in rules if not body]
    assert empty == []


def test_every_rule_cites_a_listed_source() -> None:
    text = _text()
    listed = set(_sources(text))
    uncited = [rule_id for rule_id, _, body in _rules(text) if not CITATION.search(body)]
    assert uncited == []
    dangling = sorted(
        {int(n) for _, _, body in _rules(text) for n in CITATION.findall(body)} - listed
    )
    assert dangling == []


def test_sources_are_numbered_linked_and_all_cited() -> None:
    text = _text()
    sources = _sources(text)
    assert len(sources) >= 20
    assert sorted(sources) == list(range(1, len(sources) + 1))
    unlinked = [n for n, entry in sources.items() if "https://" not in entry]
    assert unlinked == []
    cited = {int(n) for n in CITATION.findall(_text_outside_sources(text))}
    assert sorted(set(sources) - cited) == []
    assert sorted(cited - set(sources)) == []


def test_metric_set_names_every_metric_and_batch_size() -> None:
    text = _text()
    metric_set = _section(text, "The metric set").lower()
    for metric in ("tokens per second", "time to first token", "joules per token", "vram", "output match rate"):
        assert metric in metric_set, f"metric set does not name {metric!r}"
    stock_rule = dict((rule_id, body) for rule_id, _, body in _rules(text))["K1"]
    assert "1, 4, 8, and 16" in stock_rule


def test_baseline_keeps_the_first_recorded_numbers() -> None:
    baseline = _section(_text(), "The baseline to beat")
    rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in baseline.splitlines()
        if line.startswith("|") and not line.startswith("|---")
    ]
    assert rows == [
        ["Prompt", "Time to first token", "Decode"],
        ["Text, thinking off, 76 tokens out", "0.55 s", "7.8 tok/s"],
        ["Text, thinking on, 160 tokens out", "0.28 s", "7.7 tok/s"],
        ["1280x720 JPEG caption, 28 tokens out", "1.09 s", "6.6 tok/s"],
    ]
    for fact in (
        "RedHatAI/Qwen3.8-27B-INT4",
        "b1388b1",
        "ROCm 7.13",
        "300 W peak",
        "about 26 J per generated token",
        "ConchLinearKernel",
    ):
        assert fact in baseline, f"baseline section lost {fact!r}"


def test_lds_rule_names_the_limit_and_the_first_example() -> None:
    lds_rule = dict((rule_id, body) for rule_id, _, body in _rules(_text()))["K5"]
    for fact in ("64 KB", "65,536", "num_stages=1", "lab/vllm-gfx1201.diff"):
        assert fact in lds_rule, f"K5 does not name {fact!r}"


def test_source_forced_rule_names_the_research_folder() -> None:
    source_rule = dict((rule_id, body) for rule_id, _, body in _rules(_text()))["U2"]
    assert "lab/refine/research/kernels/" in source_rule


def test_file_uses_plain_hyphens_only() -> None:
    assert "—" not in _text()
