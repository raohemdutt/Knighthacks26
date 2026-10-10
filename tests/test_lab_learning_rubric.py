"""The learning track's rubric keeps the shape the refinement passes read.

Every refinement pass on the learning track walks the board against
lab/refine/learning/RUBRIC.md. These checks parse that file and fail when a
section, one of the two universal rules, one of the nine tests, a metric, or a
cited repo file goes missing, so a pass is never judged against less than the plan.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUBRIC = ROOT / "lab" / "refine" / "learning" / "RUBRIC.md"

# Top-level sections, in order: the shape every track rubric copies.
SECTION_ORDER = [
    "The goal",
    "How passes are run",
    "Universal rules",
    "Primary sources",
    "The rubric",
    "The metric set",
    "The walk list",
    "What a pass may NOT change",
    "Known open items",
    "Sources",
]

METRICS = {
    "learning.heldout.current",
    "learning.heldout.candidate",
    "learning.heldout.teacher",
    "learning.label_agreement",
    "learning.loop_minutes",
    "trainer.samples_per_s",
    "learning.package_mb",
}

# The starting source list the rubric must carry.
REQUIRED_SOURCES = [
    "https://arxiv.org/abs/2106.09685",  # LoRA
    "https://arxiv.org/abs/2305.14314",  # QLoRA
    "https://huggingface.co/docs/peft/",
    "https://huggingface.co/docs/trl/",
    "https://huggingface.co/Qwen/Qwen3.5-4B",
    "https://huggingface.co/Qwen/Qwen3.8-27B",
    "- `contracts/learning.py`",
    "- `contexts/learning/application/export.py`",
]


def read_rubric() -> str:
    assert RUBRIC.is_file(), f"missing {RUBRIC.relative_to(ROOT).as_posix()}"
    return RUBRIC.read_text(encoding="utf-8")


def sections(text: str) -> dict[str, str]:
    """Body of each top-level (##) section, keyed by its heading, in file order."""
    out: dict[str, str] = {}
    heading = None
    for line in text.splitlines():
        if line.startswith("## "):
            heading = line[3:].strip()
            out[heading] = ""
        elif heading is not None:
            out[heading] += line + "\n"
    return out


def section(text: str, prefix: str) -> str:
    bodies = [body for heading, body in sections(text).items() if heading.startswith(prefix)]
    assert len(bodies) == 1, f"expected one section starting with {prefix!r}, found {len(bodies)}"
    return bodies[0]


def test_sections_follow_the_refine_goal_shape_in_order():
    headings = list(sections(read_rubric()))
    matched = [prefix for heading in headings for prefix in SECTION_ORDER if heading.startswith(prefix)]
    assert matched == SECTION_ORDER


def test_both_universal_rules_are_stated():
    body = section(read_rubric(), "Universal rules")
    rules = re.findall(r"^(\d+)\. \*\*(.+?)\*\*", body, flags=re.M)
    assert rules == [("1", "Old versus new."), ("2", "Source-forced.")]
    assert "`lab/refine/research/learning/`" in body


def test_nine_tests_each_judged_on_the_board_and_on_evidence():
    body = section(read_rubric(), "The rubric")
    blocks = re.split(r"^### ", body, flags=re.M)[1:]
    assert [block.split(".", 1)[0] for block in blocks] == [f"LT{n}" for n in range(1, 10)]
    for block in blocks:
        fields = re.findall(r"^- (Board|Evidence|Moves|Sources):", block, flags=re.M)
        assert fields == ["Board", "Evidence", "Moves", "Sources"], block.splitlines()[0]


def test_metric_set_is_complete_and_every_metric_is_moved_by_a_test():
    text = read_rubric()
    names = set(re.findall(r"^\|[^|\n]+\| `([a-z_.]+)`", section(text, "The metric set"), flags=re.M))
    assert names == METRICS
    moved: set[str] = set()
    for line in re.findall(r"^- Moves:(.*)$", section(text, "The rubric"), flags=re.M):
        moved |= set(re.findall(r"`([a-z_.]+)`", line))
    assert moved == METRICS


def test_sources_are_links_or_files_that_exist_in_the_repo():
    body = section(read_rubric(), "Sources")
    entries = [line for line in body.splitlines() if line.startswith("- ")]
    for entry in entries:
        path = re.match(r"- `([^`]+)`", entry)
        if path:
            assert (ROOT / path.group(1)).exists(), entry
        else:
            assert re.search(r"https://\S+", entry), entry
    missing = [source for source in REQUIRED_SOURCES if source not in body]
    assert missing == []


def test_plain_hyphens_only():
    text = read_rubric()
    assert [ch for ch in ("—", "–") if ch in text] == []
