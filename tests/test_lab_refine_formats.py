"""Tests for lab/machine/refine_formats.py, the parsers for the refinement loop files.

The filled examples in lab/refine/TEMPLATES.md are the sample entries: each one is parsed here with the
real parser, so an example that stops parsing fails the suite. The track files committed under
lab/refine/<track>/ are parsed too, and a hand-written sample covers every field exactly.
"""

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lab.machine import refine_formats as rf

REFINE = Path(__file__).resolve().parents[1] / "lab" / "refine"
TRACKS = ("telemetry", "learning", "plumbing", "kernels")

SAMPLE = """\
## Pass 7, 2026-10-12

Walked T9 and E9 against D1 and D2. One change to a task, one to an experiment. NOT CONVERGED.

Research step (sources opened):
- Vendor guide, section "Power sampling" (https://example.org/guide), opened 2026-10-12 :: states the sampling
  interval and the unit of the power field.
- lab/README.md, section "Teacher (pod A)" :: the measured idle and load power.

Metrics before (read before changing anything):
| metric | value | unit | node | conditions | source |
| --- | --- | --- | --- | --- | --- |
| pod.idle_watts | 39 | W | pod-a | no server loaded | metrics.jsonl:5 |
| pod.sampler_overhead_pct | none | percent | pod-a | never measured | none |

Walked (task or experiment :: dimensions checked :: result):
- T9 :: D1, D2 :: BROKE D1; FIXED (1)
- E9 :: D2, U2 :: BROKE D2; FIXED (2)

Findings (by dimension):
- D1 T9 :: the done-when names no sampling interval. FIXED (1).
- D2 E9 :: the experiment moves two levers at once.
  It must be split. FIXED (2). RULING (a).
- U2 T9, E9 :: both changes below carry a source. HELD.
- D2 E9 :: the repeat count is still open. OPEN.

Changes (2 of at most 5):
1. T9 on the board, rewritten (D1)
   BEFORE: "Sampler publishes power."
   AFTER:  "Sampler publishes power with
      its interval in ms."
   SOURCE: Vendor guide, section "Power sampling" (https://example.org/guide)
   WHY: a reading without an interval cannot be trusted.
   METRIC: pod.sampler_overhead_pct, measured for the first time.
2. E9 on T0, Tier set (D2)
   BEFORE: "Tier: Sonnet"
   AFTER:  "Tier: Opus"
   SOURCE: lab/README.md, section "Teacher (pod A)"
   WHY: the run needs judgment about when to stop.
   METRIC: none - process change, no product metric moves.

Rulings raised (candidates; the merge pass numbers them into OWNER_RULINGS.md):
(a) Should E9 run on pod A or pod B?
    Now: pod B.
    Recommend: pod B, because pod A serves the teacher.
    Source of tension: lab/README.md, section "Teacher (pod A)" against the GPU queue rules.
    Answer: ____

Metrics after (only if the pass ran anything):
| metric | value | unit | node | conditions | source |
| --- | --- | --- | --- | --- | --- |
| pod.sampler_overhead_pct | 0.8 | percent | pod-b | 10 s window, 4 runs | ledger:41 |

Converged: NO - two changes made.

Left for the next pass: re-walk T9 and E9.

---
"""

MERGE_SAMPLE = """\
## Merge 1, 2026-10-12

Checked the four tracks after pass 3 of each. One duplicate task found and merged.

Walked (task or experiment :: dimensions checked :: result):
- R24 :: M1, M5 :: held
- T3 :: M4 :: BROKE M4; FIXED (1)

Findings (by dimension):
- M4 T3 :: duplicates a task on the plumbing track. FIXED (1).

Changes (1 of at most 5):
1. T3 on the board, merged into its twin (M4)
   BEFORE: "T3: Pod GPU power sampler"
   AFTER:  "T3: Pod GPU power sampler (also covers the plumbing sampler)"
   SOURCE: Notion task T3 and the plumbing track log, pass 2
   WHY: two tasks wrote the same sampler.

Rulings raised: none

---
"""


def example(name: str) -> str:
    """Return the body of the ```text example:<name> block in lab/refine/TEMPLATES.md."""
    text = (REFINE / "TEMPLATES.md").read_text(encoding="utf-8")
    found = re.search(rf"^```text example:{re.escape(name)}\n(.*?)^```\s*$", text, flags=re.MULTILINE | re.DOTALL)
    assert found, f"lab/refine/TEMPLATES.md has no ```text example:{name} block"
    return found.group(1)


def only_entry(text: str):
    log = rf.parse_log(text)
    assert len(log.entries) == 1
    return log.entries[0]


def edit(text: str, old: str, new: str) -> str:
    """Replace exactly one occurrence, so a test never passes because its edit matched nothing."""
    assert text.count(old) == 1, f"expected exactly one {old!r} in the text"
    return text.replace(old, new)


def test_sample_pass_entry_parses_header_summary_and_sources():
    entry = only_entry(SAMPLE)

    assert (entry.kind, entry.number, entry.date) == ("pass", 7, "2026-10-12")
    assert entry.summary == "Walked T9 and E9 against D1 and D2. One change to a task, one to an experiment. NOT CONVERGED."
    assert entry.sections == frozenset(
        {"research", "metrics_before", "walked", "findings", "changes", "rulings", "metrics_after", "converged", "next"}
    )
    assert entry.sources == (
        rf.Source(
            'Vendor guide, section "Power sampling" (https://example.org/guide), opened 2026-10-12',
            "states the sampling interval and the unit of the power field.",
        ),
        rf.Source('lab/README.md, section "Teacher (pod A)"', "the measured idle and load power."),
    )
    assert entry.metrics_before == (
        rf.Reading("pod.idle_watts", "39", "W", "pod-a", "no server loaded", "metrics.jsonl:5"),
        rf.Reading("pod.sampler_overhead_pct", "none", "percent", "pod-a", "never measured", "none"),
    )
    assert entry.metrics_after == (
        rf.Reading("pod.sampler_overhead_pct", "0.8", "percent", "pod-b", "10 s window, 4 runs", "ledger:41"),
    )
    assert entry.metrics_after_note == ""
    assert entry.unparsed == ()


def test_sample_pass_entry_parses_walk_findings_and_verdict():
    entry = only_entry(SAMPLE)

    assert entry.walked == (
        rf.Walk("T9", ("D1", "D2"), "BROKE D1; FIXED (1)"),
        rf.Walk("E9", ("D2", "U2"), "BROKE D2; FIXED (2)"),
    )
    assert entry.findings == (
        rf.Finding("D1", "T9", "the done-when names no sampling interval. FIXED (1).", "fixed", (1,), ()),
        rf.Finding(
            "D2", "E9", "the experiment moves two levers at once. It must be split. FIXED (2). RULING (a).", "fixed", (2,), ("a",)
        ),
        rf.Finding("U2", "T9, E9", "both changes below carry a source. HELD.", "held", (), ()),
        rf.Finding("D2", "E9", "the repeat count is still open. OPEN.", "open", (), ()),
    )
    assert entry.rulings_raised == (
        rf.RaisedRuling(
            "a",
            "Should E9 run on pod A or pod B?",
            now="pod B.",
            recommend="pod B, because pod A serves the teacher.",
            tension='lab/README.md, section "Teacher (pod A)" against the GPU queue rules.',
            answer="____",
        ),
    )
    assert entry.converged is False
    assert entry.converged_note == "two changes made."
    assert entry.next_pass == "re-walk T9 and E9."
    assert entry.checks == ""


def test_sample_pass_entry_parses_numbered_changes_with_wrapped_lines():
    first, second = only_entry(SAMPLE).changes

    assert first == rf.Change(
        1,
        "T9 on the board, rewritten",
        ("D1",),
        before="Sampler publishes power.",
        after="Sampler publishes power with its interval in ms.",
        source='Vendor guide, section "Power sampling" (https://example.org/guide)',
        why="a reading without an interval cannot be trusted.",
        metric="pod.sampler_overhead_pct, measured for the first time.",
    )
    assert second == rf.Change(
        2,
        "E9 on T0, Tier set",
        ("D2",),
        before="Tier: Sonnet",
        after="Tier: Opus",
        source='lab/README.md, section "Teacher (pod A)"',
        why="the run needs judgment about when to stop.",
        metric="none - process change, no product metric moves.",
    )


def test_sample_pass_entry_is_valid():
    assert rf.check_pass_entry(only_entry(SAMPLE)) == []


def test_reading_and_walk_properties():
    stored, never = only_entry(SAMPLE).metrics_before
    assert (stored.cited, stored.absent, stored.number) == (True, False, 39.0)
    assert (never.cited, never.absent, never.number) == (False, True, None)
    assert rf.Walk("R24", ("D2",), "held").held is True
    assert rf.Walk("E3", ("D1",), "BROKE D1; FIXED (1)").held is False


def test_template_pass_entry_example_parses_and_is_valid():
    entry = only_entry(example("pass-entry"))

    assert (entry.kind, entry.number, entry.date) == ("pass", 3, "2026-10-10")
    assert rf.check_pass_entry(entry) == []
    assert [r.metric for r in entry.metrics_before] == [
        "teacher.decode_tok_s",
        "teacher.watts_avg",
        "teacher.joules_per_token",
        "teacher.decode_tok_s",
    ]
    assert [r.source for r in entry.metrics_before] == ["metrics.jsonl:3", "metrics.jsonl:7", "metrics.jsonl:12", "none"]
    assert [w.target for w in entry.walked] == ["E2", "E3", "E5", "R24"]
    assert [w.held for w in entry.walked] == [False, False, False, True]
    assert [(f.dimension, f.status) for f in entry.findings] == [
        ("K1", "fixed"),
        ("K3", "fixed"),
        ("K2", "fixed"),
        ("K8", "fixed"),
        ("K6", "fixed"),
        ("K6", "held"),
    ]
    assert [c.number for c in entry.changes] == [1, 2, 3]
    assert [c.dimensions for c in entry.changes] == [("K1", "K3", "K6"), ("K2",), ("K8",)]
    assert [(r.label, r.answer) for r in entry.rulings_raised] == [("a", "____")]
    assert (entry.ending, entry.ending_number) == ("", None)
    assert entry.metrics_after == ()
    assert entry.metrics_after_note.startswith("not run")
    assert entry.converged is False


def test_template_owner_correction_example_parses():
    entry = only_entry(example("owner-correction"))

    assert isinstance(entry, rf.OwnerCorrection)
    assert (entry.date, entry.title, entry.rulings) == ("2026-10-10", "", (4,))
    assert [(c.number, c.dimensions) for c in entry.changes] == [(1, ("K3",))]
    assert entry.changes[0].source == "OWNER_RULINGS.md 4 (owner, 2026-10-10)"
    assert entry.changes[0].before != entry.changes[0].after


def test_template_run_state_example_parses():
    entry = only_entry(example("run-state"))

    assert isinstance(entry, rf.RunState)
    assert (entry.first_pass, entry.last_pass, entry.date, entry.ended) == (1, 5, "2026-10-10", "converged")


def test_template_ruling_example_is_unanswered():
    (ruling,) = rf.parse_rulings(example("ruling"))

    assert (ruling.number, ruling.origin, ruling.answer, ruling.answered) == (4, "pass 3", "____", False)
    assert ruling.question == "Must a kernel win beat the run-to-run spread, or a fixed percentage?"
    assert ruling.now and ruling.recommend and ruling.tension


def test_template_answered_example_marks_the_ruling_answered():
    answer_line = example("ruling-answered").strip()
    text = edit(example("ruling"), "Answer: ____", answer_line)

    (ruling,) = rf.parse_rulings(text)

    assert ruling.answered is True
    assert ruling.answer == answer_line.removeprefix("Answer: ")


def test_template_counter_example_is_a_counter():
    assert rf.parse_counter(example("counter")) == 3


def test_merge_entry_uses_the_pass_format_without_research_or_metrics():
    log = rf.parse_log(MERGE_SAMPLE)
    (entry,) = log.entries

    assert (entry.kind, entry.number, entry.date) == ("merge", 1, "2026-10-12")
    assert entry.converged is None
    assert rf.check_pass_entry(entry) == []
    assert log.passes == ()


def test_the_same_entry_as_a_pass_is_missing_what_a_pass_needs():
    as_pass = edit(MERGE_SAMPLE, "## Merge 1, 2026-10-12", "## Pass 1, 2026-10-12")

    assert rf.check_pass_entry(only_entry(as_pass)) == [
        "pass 1: 'Research step' needs at least one source, written '- <citation> :: <what it says>'",
        "pass 1: 'Metrics before' needs a table with at least one reading",
        "pass 1: change 1 has no METRIC",
        "pass 1: 'Converged' must start with YES or NO",
    ]


@pytest.mark.parametrize("track", TRACKS)
def test_committed_track_log_is_a_header_only_log(track):
    log = rf.parse_log((REFINE / track / "REFINE_LOG.md").read_text(encoding="utf-8"))

    assert log.entries == ()
    assert log.header.startswith(f"# Refinement log for the {track} track")
    assert "lab/refine/TEMPLATES.md" in log.header


@pytest.mark.parametrize("track", TRACKS)
def test_committed_rulings_file_holds_no_rulings_yet(track):
    text = (REFINE / track / "OWNER_RULINGS.md").read_text(encoding="utf-8")

    assert text.startswith(f"# Owner rulings for the {track} track")
    assert rf.parse_rulings(text) == ()


@pytest.mark.parametrize("track", TRACKS)
def test_committed_counter_reads_zero(track):
    assert rf.parse_counter((REFINE / track / "refine_pass.txt").read_text(encoding="utf-8")) == 0


@pytest.mark.parametrize("track", TRACKS)
def test_committed_pass_reports_folder_is_kept_in_git(track):
    assert (REFINE / track / "pass_reports" / ".gitkeep").is_file()


CORRECTION = """
## Owner correction after pass 7 (2): the sweep, 2026-10-13 (no increment)

Rulings applied: 2, 3

The owner changed the sweep.

1. E3 on T0, repeat count (D2)
   BEFORE: "five runs"
   AFTER:  "three runs"
   SOURCE: OWNER_RULINGS.md 2 (owner, 2026-10-13)
   WHY: it fits the slot.

---
"""


def test_log_keeps_entry_order_and_kinds():
    text = (
        "# Refinement log for the kernels track\n\nHeader paragraph.\n\n---\n\n"
        + SAMPLE
        + CORRECTION
        + "\n## Pass 8, 2026-10-14\n\nNothing changed.\n\n---\n\n"
        + "## State at the end of the run, passes 7 to 8, 2026-10-14\n\nRun ended: converged at pass 8.\n"
    )

    log = rf.parse_log(text)

    assert log.header == "# Refinement log for the kernels track\n\nHeader paragraph.\n\n---"
    assert [type(entry).__name__ for entry in log.entries] == ["PassEntry", "OwnerCorrection", "PassEntry", "RunState"]
    assert [entry.number for entry in log.passes] == [7, 8]
    correction = log.entries[1]
    assert (correction.date, correction.title, correction.rulings) == ("2026-10-13", "after pass 7 (2): the sweep", (2, 3))
    assert correction.summary == "The owner changed the sweep."
    assert [(c.number, c.before, c.after) for c in correction.changes] == [(1, "five runs", "three runs")]
    assert log.entries[3].ended == "converged"


def test_unrecognized_log_heading_raises():
    with pytest.raises(rf.FormatError, match="unrecognized log heading '## Notes'"):
        rf.parse_log(SAMPLE + "\n## Notes\n\nA free note.\n")


def test_a_pass_number_may_appear_only_once():
    with pytest.raises(rf.FormatError, match="pass 7 appears twice"):
        rf.parse_log(SAMPLE + "\n" + SAMPLE)


def test_an_impossible_date_raises():
    with pytest.raises(rf.FormatError, match="not a calendar date"):
        rf.parse_log("## Pass 1, 2026-13-45\n\nSummary.\n")


def test_a_heading_inside_a_code_fence_is_not_an_entry():
    fenced = edit(SAMPLE, "Left for the next pass: re-walk T9 and E9.", "Left for the next pass: re-walk T9 and E9.\n\n```\n## Notes\n```")

    entry = only_entry(fenced)

    assert entry.number == 7
    assert entry.next_pass == "re-walk T9 and E9. ``` ## Notes ```"


def test_a_section_label_may_appear_only_once_in_an_entry():
    doubled = edit(SAMPLE, "Converged: NO - two changes made.", "Converged: NO - two changes made.\nConverged: NO - again.")

    with pytest.raises(rf.FormatError, match="section 'Converged' appears twice"):
        rf.parse_log(doubled)


def test_a_run_cannot_end_before_it_starts():
    with pytest.raises(rf.FormatError, match="the run ends before it starts"):
        rf.parse_log("## State at the end of the run, passes 5 to 3, 2026-10-10\n\nConverged: NO - x.\n")


def test_an_owner_correction_needs_a_date():
    with pytest.raises(rf.FormatError, match="needs a date"):
        rf.parse_log("## Owner correction (no increment)\n\nText.\n")


ROW_IDLE = "| pod.idle_watts | 39 | W | pod-a | no server loaded | metrics.jsonl:5 |\n"
ROW_NEVER = "| pod.sampler_overhead_pct | none | percent | pod-a | never measured | none |\n"
ROW_AFTER = "| pod.sampler_overhead_pct | 0.8 | percent | pod-b | 10 s window, 4 runs | ledger:41 |\n"
FINDING_HELD = "- U2 T9, E9 :: both changes below carry a source. HELD."

CHECK_CASES = [
    pytest.param(
        [(ROW_IDLE, ""), (ROW_NEVER, "")],
        ["pass 7: 'Metrics before' needs a table with at least one reading"],
        id="no-metrics-before-rows",
    ),
    pytest.param(
        [("no server loaded | metrics.jsonl:5 |", "no server loaded | PR 37 |")],
        [
            "pass 7: metrics before: pod.idle_watts needs a metrics-file or ledger line in the source column "
            "(metrics.jsonl:<line> or ledger:<seq>), found 'PR 37'"
        ],
        id="reading-without-a-line",
    ),
    pytest.param(
        [("| pod.idle_watts |", "| idle |")],
        ["pass 7: metrics before: 'idle' is not a dotted lowercase metric name"],
        id="metric-name-not-dotted",
    ),
    pytest.param(
        [('   SOURCE: lab/README.md, section "Teacher (pod A)"\n', "")],
        ["pass 7: change 2 has no SOURCE"],
        id="change-without-source",
    ),
    pytest.param(
        [("METRIC: none - process change, no product metric moves.", "METRIC: faster")],
        ["pass 7: change 2: METRIC must start with a dotted metric name or 'none - <reason>'"],
        id="metric-line-not-a-metric",
    ),
    pytest.param(
        [("FIXED (2). RULING (a).", "FIXED (9). RULING (a).")],
        [
            "pass 7: a finding says FIXED (9) but there is no change 9",
            "pass 7: change 2 is not referenced by any finding as FIXED (2)",
        ],
        id="fixed-reference-to-a-missing-change",
    ),
    pytest.param(
        [("2. E9 on T0, Tier set (D2)", "3. E9 on T0, Tier set (D2)")],
        [
            "pass 7: changes must be numbered 1 to 2 in order, found [1, 3]",
            "pass 7: a finding says FIXED (2) but there is no change 2",
            "pass 7: change 3 is not referenced by any finding as FIXED (3)",
        ],
        id="change-numbers-with-a-gap",
    ),
    pytest.param(
        [(FINDING_HELD, "- U2 T9, E9 :: both changes below carry a source.")],
        ["pass 7: the finding for U2 T9, E9 has no disposition (FIXED (n), RULING (x), HELD or OPEN)"],
        id="finding-without-a-disposition",
    ),
    pytest.param(
        [("RULING (a)", "RULING (z)")],
        ["pass 7: a finding says RULING (z) but no ruling (z) is raised"],
        id="ruling-reference-to-nothing",
    ),
    pytest.param(
        [("    Recommend: pod B, because pod A serves the teacher.\n", "")],
        ["pass 7: ruling (a) is missing Recommend"],
        id="raised-ruling-without-recommend",
    ),
    pytest.param(
        [(FINDING_HELD, FINDING_HELD + "\nnot a bullet")],
        ["pass 7: unparsed line in findings: not a bullet"],
        id="stray-line-in-findings",
    ),
    pytest.param(
        [("- T9 :: D1, D2 ::", "- T9 :: d1, D2 ::")],
        ["pass 7: walked T9: dimensions must be ids like D1, U2; found ['d1', 'D2']"],
        id="walk-dimension-not-an-id",
    ),
    pytest.param(
        [("BROKE D1; FIXED (1)", "passed")],
        ["pass 7: walked T9: the result must start with held or BROKE"],
        id="walk-result-not-held-or-broke",
    ),
    pytest.param(
        [(ROW_AFTER, "")],
        ["pass 7: 'Metrics after' needs a table or 'not run: <why>'"],
        id="metrics-after-empty",
    ),
    pytest.param(
        [("Converged: NO - two changes made.\n", "")],
        ["pass 7: 'Converged' must start with YES or NO"],
        id="no-converged-line",
    ),
    pytest.param(
        [("Converged: NO - two changes made.", "Converged: YES - nothing to do.")],
        [
            "pass 7: Converged is YES but the entry makes 2 change(s); a converged walk changes nothing",
            "pass 7: Converged is YES, so the entry must end with the line 'CONVERGED at pass 7'",
        ],
        id="converged-yes-with-changes",
    ),
]


@pytest.mark.parametrize("edits,expected", CHECK_CASES)
def test_check_reports_exactly_what_is_wrong(edits, expected):
    text = SAMPLE
    for old, new in edits:
        text = edit(text, old, new)

    assert rf.check_pass_entry(only_entry(text)) == expected


RULINGS = """\
# Owner rulings for the kernels track

Header paragraph with no heading.

## Notes for the owner

This heading does not start with a digit, so the parser skips it.

## 1. Is one idle-baseline policy enough for every node? (pass 2)

Now: one policy, the node's idle watts measured in the same run.
Recommend: **keep one policy.** Two policies make the numbers
incomparable.
Source of tension: kernel test 4 against the Jetson's idle floor.
Answer: ____

## 2. Does "a win" mean 5 percent (informational; default applied)? (merge 1)

Now: the spread.
Recommend: the spread.
Tension: kernel test 3.
Answer:

## 3. Which pod runs the sweep? (owner, 2026-10-12)

Now: pod B.
Recommend: pod B.
Source of tension: the GPU queue rules.
Answer: Pod B, always.
  Pod A only serves the teacher (owner, 2026-10-12).
"""

RULING_BODY = "\n\nNow: a.\nRecommend: b.\nSource of tension: c.\nAnswer: ____\n"


def test_rulings_parse_headings_fields_and_skip_notes():
    rulings = rf.parse_rulings(RULINGS)

    assert [(r.number, r.question, r.origin) for r in rulings] == [
        (1, "Is one idle-baseline policy enough for every node?", "pass 2"),
        (2, 'Does "a win" mean 5 percent (informational; default applied)?', "merge 1"),
        (3, "Which pod runs the sweep?", "owner, 2026-10-12"),
    ]
    assert rulings[0].now == "one policy, the node's idle watts measured in the same run."
    assert rulings[0].recommend == "**keep one policy.** Two policies make the numbers incomparable."
    assert rulings[0].tension == "kernel test 4 against the Jetson's idle floor."
    assert rulings[0].answer == "____"
    assert rulings[1].tension == "kernel test 3."
    assert rulings[1].answer == ""
    assert rulings[2].answer == "Pod B, always. Pod A only serves the teacher (owner, 2026-10-12)."
    assert [r.answered for r in rulings] == [False, False, True]


@pytest.mark.parametrize(
    "answer_line,answered",
    [
        ("Answer: ____", False),
        ("Answer:", False),
        ("Answer:   ", False),
        ("Answer: ______", False),
        ("Answer: yes", True),
        ("Answer: ____ keep it", True),
        ("Answer: **B (owner, 2026-10-12)**", True),
    ],
)
def test_an_answer_is_blank_only_when_nothing_but_underscores_follow(answer_line, answered):
    text = f"## 1. Question? (pass 1)\n\nNow: a.\nRecommend: b.\nSource of tension: c.\n{answer_line}\n"

    (ruling,) = rf.parse_rulings(text)

    assert ruling.answered is answered


@pytest.mark.parametrize("label", ["Now", "Recommend", "Source of tension", "Answer"])
def test_a_ruling_without_one_of_its_four_lines_raises(label):
    lines = {"Now": "a.", "Recommend": "b.", "Source of tension": "c.", "Answer": "____"}
    body = "".join(f"{key}: {value}\n" for key, value in lines.items() if key != label)

    with pytest.raises(rf.FormatError, match=f"ruling 2 has no '{label}:' line"):
        rf.parse_rulings("## 2. Question? (pass 1)\n\n" + body)


def test_a_ruling_number_may_appear_only_once():
    entry = "## 1. Question? (pass 1)" + RULING_BODY

    with pytest.raises(rf.FormatError, match="ruling 1 appears twice"):
        rf.parse_rulings(entry + "\n" + entry)


def test_a_numbered_heading_that_is_not_a_ruling_raises():
    with pytest.raises(rf.FormatError, match="is not a ruling heading"):
        rf.parse_rulings("## 3 Missing the period" + RULING_BODY)


@pytest.mark.parametrize("text,expected", [("0\n", 0), ("0", 0), ("12\n", 12), (" 7 \n", 7), ("007\n", 7)])
def test_counter_reads_one_integer(text, expected):
    assert rf.parse_counter(text) == expected


@pytest.mark.parametrize("text", ["", "\n", "abc\n", "1.5\n", "-1\n", "1 2\n", "0\n0\n", chr(0x663) + "\n"])
def test_counter_rejects_anything_but_one_non_negative_integer(text):
    with pytest.raises(rf.FormatError, match="refine_pass.txt must hold one non-negative integer"):
        rf.parse_counter(text)


HEADER_ROW = "| metric | value | unit | node | conditions | source |"
BEFORE_LABEL = "Metrics before (read before changing anything):\n"


def test_a_metrics_table_with_the_wrong_header_is_reported_and_not_read():
    text = edit(SAMPLE, BEFORE_LABEL + HEADER_ROW, BEFORE_LABEL + HEADER_ROW.replace("metric", "name"))

    entry = only_entry(text)

    assert entry.metrics_before == ()
    assert rf.check_pass_entry(entry) == [
        "pass 7: 'Metrics before' needs a table with at least one reading",
        "pass 7: unparsed line in metrics_before: the table header must be "
        "| metric | value | unit | node | conditions | source |, found "
        "| name | value | unit | node | conditions | source |",
    ]


def test_a_metrics_row_with_the_wrong_number_of_cells_is_reported_and_skipped():
    text = edit(SAMPLE, ROW_IDLE, "| pod.idle_watts | 39 | W | pod-a | no server loaded |\n")

    entry = only_entry(text)

    assert [reading.metric for reading in entry.metrics_before] == ["pod.sampler_overhead_pct"]
    assert rf.check_pass_entry(entry) == [
        "pass 7: unparsed line in metrics_before: expected 6 cells, found 5: | pod.idle_watts | 39 | W | pod-a | no server loaded |"
    ]


def test_a_merge_entry_that_writes_a_verdict_must_make_it_yes_or_no():
    text = edit(MERGE_SAMPLE, "Rulings raised: none\n", "Rulings raised: none\n\nConverged: maybe\n")

    assert rf.check_pass_entry(only_entry(text)) == ["merge 1: 'Converged' must start with YES or NO"]


def test_crlf_line_endings_parse_like_lf():
    assert rf.parse_log(SAMPLE.replace("\n", "\r\n")) == rf.parse_log(SAMPLE)
    assert rf.parse_rulings(RULINGS.replace("\n", "\r\n")) == rf.parse_rulings(RULINGS)


def test_empty_files_have_no_entries_and_no_rulings():
    assert rf.parse_log("") == rf.Log("", ())
    assert rf.parse_rulings("") == ()


def test_a_run_state_without_a_run_ended_line_reads_empty():
    (state,) = rf.parse_log("## State at the end of the run, passes 1 to 2, 2026-10-10\n\nNo verdict written.\n").entries

    assert (state.first_pass, state.last_pass, state.ended) == (1, 2, "")
    assert state.text == "No verdict written."


def test_template_pass_entry_example_keeps_its_checks_and_next_steps():
    entry = only_entry(example("pass-entry"))

    assert entry.checks.startswith("Re-fetched T0 after changes 1 to 3")
    assert entry.next_pass.endswith("Waits on the owner: ruling (a).")
    assert "checks" in entry.sections and "next" in entry.sections


CONVERGED_PASS = """\
## Pass 5, 2026-10-13

Walked T9 and E9 against D1 and D2 and changed nothing. CONVERGED.

Research step (sources opened):
- lab/README.md, section "Teacher (pod A)" :: still the current baseline.

Metrics before (read before changing anything):
| metric | value | unit | node | conditions | source |
| --- | --- | --- | --- | --- | --- |
| pod.idle_watts | 39 | W | pod-a | no server loaded | metrics.jsonl:5 |

Walked (task or experiment :: dimensions checked :: result):
- T9 :: D1, D2 :: held
- E9 :: D1, D2 :: held

Findings (by dimension):
- D1 T9, E9 :: both name their metric and its unit. HELD.

Changes (0 of at most 5): none

Rulings raised: none

Metrics after (only if the pass ran anything):
not run: nothing changed, so there was nothing to measure.

Converged: YES - a full walk changed nothing.

CONVERGED at pass 5

---
"""

CAP_RULING = """\
(a) Extend the run, open a new one, or leave the track as it is?
    Now: the track stays as it is.
    Recommend: open a new run, because the numbers moved.
    Source of tension: lab/refine/PASS_RULES.md, section "The cap".
    Answer: ____
"""


def cap_pass(rulings: str = CAP_RULING, closing: str = "CAP REACHED at pass 10, not converged") -> str:
    return (
        "## Pass 10, 2026-10-14\n\nTenth pass of the run, with one finding still open.\n\n"
        'Research step (sources opened):\n- lab/README.md, section "Teacher (pod A)" :: the baseline.\n\n'
        "Metrics before (read before changing anything):\n"
        "| metric | value | unit | node | conditions | source |\n| --- | --- | --- | --- | --- | --- |\n"
        "| pod.idle_watts | 39 | W | pod-a | no server loaded | metrics.jsonl:5 |\n\n"
        "Walked (task or experiment :: dimensions checked :: result):\n- T9 :: D1 :: held\n\n"
        "Findings (by dimension):\n- D1 T9 :: the interval is still not in the done-when. OPEN.\n\n"
        "Changes (0 of at most 5): none\n\n"
        "Rulings raised (candidates; the merge pass numbers them into OWNER_RULINGS.md):\n" + rulings + "\n"
        "Converged: NO - the cap is reached.\n\n" + closing + "\n\n---\n"
    )


def test_a_converged_entry_closes_with_its_own_pass_number():
    entry = only_entry(CONVERGED_PASS)

    assert (entry.converged, entry.ending, entry.ending_number) == (True, "converged", 5)
    assert entry.converged_note == "a full walk changed nothing."
    assert entry.changes == ()
    assert entry.metrics_after_note == "not run: nothing changed, so there was nothing to measure."
    assert rf.check_pass_entry(entry) == []


@pytest.mark.parametrize(
    "edits,expected",
    [
        pytest.param(
            [("CONVERGED at pass 5\n\n", "")],
            ["pass 5: Converged is YES, so the entry must end with the line 'CONVERGED at pass 5'"],
            id="yes-without-the-closing-line",
        ),
        pytest.param(
            [("CONVERGED at pass 5", "CONVERGED at pass 6")],
            ["pass 5: the closing line names pass 6 but this is pass 5"],
            id="closing-line-names-another-pass",
        ),
        pytest.param(
            [("Converged: YES - a full walk changed nothing.", "Converged: NO - one thing left.")],
            ["pass 5: the entry ends with a CONVERGED line but its Converged line says NO"],
            id="closing-line-contradicts-the-verdict",
        ),
    ],
)
def test_the_converged_closing_line_must_agree_with_the_entry(edits, expected):
    text = CONVERGED_PASS
    for old, new in edits:
        text = edit(text, old, new)

    assert rf.check_pass_entry(only_entry(text)) == expected


def test_a_cap_entry_closes_with_the_cap_line_and_raises_a_ruling():
    entry = only_entry(cap_pass())

    assert (entry.converged, entry.ending, entry.ending_number) == (False, "cap", 10)
    assert [r.label for r in entry.rulings_raised] == ["a"]
    assert rf.check_pass_entry(entry) == []


def test_a_cap_entry_without_the_extend_the_run_ruling_is_reported():
    entry = only_entry(cap_pass(rulings="none\n"))

    assert rf.check_pass_entry(entry) == ["pass 10: a CAP REACHED entry must raise the ruling that asks whether to extend the run"]


def test_a_cap_line_naming_another_pass_is_reported():
    entry = only_entry(cap_pass(closing="CAP REACHED at pass 9, not converged"))

    assert rf.check_pass_entry(entry) == ["pass 10: the closing line names pass 9 but this is pass 10"]


def test_two_closing_lines_in_one_entry_raise():
    with pytest.raises(rf.FormatError, match="more than one closing CONVERGED or CAP REACHED line"):
        rf.parse_log(cap_pass(closing="CONVERGED at pass 10\nCAP REACHED at pass 10, not converged"))


@pytest.mark.parametrize(
    "answer_line,expected",
    [
        ("", ["pass 7: ruling (a) is missing its blank Answer line (write 'Answer: ____')"]),
        ("    Answer: yes\n", ["pass 7: ruling (a) has an Answer filled in; only the owner writes an Answer"]),
    ],
)
def test_a_raised_ruling_keeps_its_answer_line_blank(answer_line, expected):
    text = edit(SAMPLE, "    Answer: ____\n", answer_line)

    assert rf.check_pass_entry(only_entry(text)) == expected


def stub(number: int) -> str:
    return f"## Pass {number}, 2026-10-1{number}\n\nSummary.\n\n---\n\n"


STATE_BARE = "## State at the end of the run\n\nRun ended: converged at pass 2.\n\n---\n\n"


def test_current_run_is_every_pass_when_no_run_has_closed():
    log = rf.parse_log(stub(1) + stub(2))

    assert [entry.number for entry in log.current_run] == [1, 2]


def test_current_run_starts_after_the_latest_run_state_and_skips_corrections():
    log = rf.parse_log(stub(1) + stub(2) + STATE_BARE + stub(3) + CORRECTION + stub(4))

    assert [entry.number for entry in log.current_run] == [3, 4]
    assert [entry.number for entry in log.passes] == [1, 2, 3, 4]


def test_current_run_is_empty_right_after_a_run_state():
    assert rf.parse_log(stub(1) + stub(2) + STATE_BARE).current_run == ()


@pytest.mark.parametrize(
    "line,expected",
    [
        ("Run ended: converged at pass 5.", "converged"),
        ("Run ended: Converged at pass 5.", "converged"),
        ("Run ended: cap reached at pass 10.", "cap"),
        ("Run ended: ended by the owner after pass 4.", "owner"),
    ],
)
def test_a_bare_run_state_heading_reads_how_the_run_ended(line, expected):
    (state,) = rf.parse_log(f"## State at the end of the run\n\n{line}\n").entries

    assert (state.first_pass, state.last_pass, state.date, state.ended) == (None, None, None, expected)


def test_report_end_reads_the_ended_line_as_a_datetime_with_an_offset():
    ended = rf.parse_report_end(example("pass-report"))

    assert ended == datetime(2026, 10, 10, 7, 12, tzinfo=timezone(timedelta(hours=-4)))
    assert ended.utcoffset() == timedelta(hours=-4)


def test_report_end_accepts_a_z_suffix():
    assert rf.parse_report_end("Ended: 2026-10-10T11:12:00Z\n") == datetime(2026, 10, 10, 11, 12, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "text,message",
    [
        ("No end time here.\n", "must state when the pass ended"),
        ("Ended: 2026-10-10T07:12:00\n", "has no UTC offset"),
        ("Ended: yesterday\n", "is not an ISO 8601 date and time"),
    ],
)
def test_report_end_rejects_a_missing_naive_or_invalid_time(text, message):
    with pytest.raises(rf.FormatError, match=message):
        rf.parse_report_end(text)


def test_template_example_cites_real_rows_of_the_metrics_file():
    rows = (REFINE / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    cited = [reading for reading in only_entry(example("pass-entry")).metrics_before if reading.cited]

    assert len(cited) == 3
    for reading in cited:
        row = json.loads(rows[int(reading.source.rsplit(":", 1)[1]) - 1])
        assert (row["metric"], row["value"]) == (reading.metric, float(reading.value)), reading


def test_dimension_ids_may_be_a_rubrics_own_labels():
    text = edit(SAMPLE, "- T9 :: D1, D2 :: BROKE D1; FIXED (1)", "- T9 :: LT1, LT2 :: BROKE LT1; FIXED (1)")
    text = edit(text, "- D1 T9 :: the done-when", "- LT1 T9 :: the done-when")
    text = edit(text, "1. T9 on the board, rewritten (D1)", "1. T9 on the board, rewritten (LT1)")

    entry = only_entry(text)

    assert entry.walked[0].dimensions == ("LT1", "LT2")
    assert entry.findings[0].dimension == "LT1"
    assert entry.changes[0].dimensions == ("LT1",)
    assert rf.check_pass_entry(entry) == []
