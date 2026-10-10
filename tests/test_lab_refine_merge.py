"""Tests for lab/machine/refine_merge.py, the checks the merge pass's log and the track files must pass.

The committed lab/refine folder is checked as it stands, so a merge entry, a pass entry, a counter, a pass
report or a ruling that breaks the rules fails the suite on the pull request that writes it. A filled
example round (four pass entries, their reports and rulings, and the merge entry that reconciles them,
the same example the refine-merge skill shows) is built in a temporary folder and must pass every check;
edits of it show that each rule reports exactly what is wrong.
"""

import json
from pathlib import Path

import pytest

from lab.machine import refine_formats as rf
from lab.machine import refine_merge as rm
from lab.machine import refine_track as rt

REFINE = Path(__file__).resolve().parents[1] / "lab" / "refine"

TELEMETRY_QUESTION = "Does an answer whose request window spans a sampler restart get a joules-per-answer number?"
KERNELS_QUESTION = "At which batch size does the checkpoint shootout compare tok/s and joules per token?"
LEARNING_QUESTION = "Who sets the Tier of a lab task that is shared with another person?"

MERGE_EXAMPLE = """\
## Merge 1, 2026-10-10

EXAMPLE, not the record of a real merge. The task IDs B1, L6, R11 and T3, their properties and the quoted Done when are real (Notion board, 2026-10-10); the round, the tasks T13 and R29 and the rulings are invented. Reconciled the first round: pass 1 of each of the four tracks, all four valid. Ran the checks M1 to M7 over the four pass entries, every open task of the four tracks and the three demo beats Ilan owns. Three board changes, one proposal for a shared task, two rulings filed from the passes and one raised by the merge.

Walked (pass entry, task, demo beat or file :: checks :: result):
- telemetry pass 1 :: M6 :: held
- learning pass 1 :: M6 :: held
- plumbing pass 1 :: M6 :: held
- kernels pass 1 :: M6 :: held
- T13, R29 :: M1, M4, M5 :: BROKE M4; FIXED (1)
- B1 :: M4, M5 :: BROKE M4, M5; FIXED (2, 3)
- L6 :: M3, M5 :: BROKE M5; OPEN, proposal on Dominick and Team
- R11 :: M1, M2 :: held
- demo beats "It learns", "Where are my keys?" and "Under the hood" :: M3 :: held
- every other open task of the four tracks :: M1, M4, M5 :: held
- commits to contracts/, runtime/, bus/, contexts/gateway/bootstrap.py, pyproject.toml and AGENTS.md since the round started :: M2 :: held
- answered rulings in the four OWNER_RULINGS.md files :: M7 :: held

Findings (by dimension):
- M4 T13, R29 :: telemetry pass 1 created T13 and plumbing pass 1 created R29, and both measure the sampler's own cost on pod B, so one number would take two GPU slots. FIXED (1).
- M4 B1 :: a lab task whose only Assignee is Ilan and that no track claims (it is in neither PASS_RULES.md "The four tracks" nor any RUBRIC.md walk list), so no pass ever walks it. FIXED (2).
- M5 B1 :: the Tier is empty, so the open-ticket skill stops before it starts. FIXED (3).
- M5 L6 :: the Tier is empty on a P0 task, but L6 is shared with Xander, so the merge may not set it; the proposal to set Opus is posted on Dominick and Team. OPEN. RULING (c).
- M1 R11 :: R11 (kernels) samples pod power during its runs, and it is Blocked by T3 (telemetry), the pod power sampler, already. HELD.
- M2 contracts/ and runtime/ :: no commit since the round started touches a shared file, and no task changed this round plans to. HELD.
- M3 "It learns", "Where are my keys?", "Under the hood" :: every task these beats need from Ilan's lane is open with a chain of blockers that ends on the board, and none lost a clause this round. HELD.
- M6 telemetry pass 1, learning pass 1, plumbing pass 1, kernels pass 1 :: each entry passes the format checks, every board change it made reads as its AFTER value, and each change has its Plan changes line. HELD.
- M7 the four OWNER_RULINGS.md files :: no ruling has an Answer yet. HELD.

Changes (3 of at most 10):
1. R29 merged into T13, and R29 moved under Completed tasks (archive) (M4)
   BEFORE: "T13 and R29 both open and Not started; R29 created by plumbing pass 1"
   AFTER:  "T13 open, its Done when also covering pod B; R29 under Completed tasks (archive) with the line 'merged into T13 by merge 1'"
   SOURCE: lab/refine/telemetry/REFINE_LOG.md, "## Pass 1, 2026-10-10", change 2; lab/refine/plumbing/REFINE_LOG.md, "## Pass 1, 2026-10-10", change 4; lab/refine/PASS_RULES.md, "The merge pass" (merging duplicates across tracks)
   WHY: one measurement should take one GPU slot; T13 is the older task, so it stays.
2. B1 added to the kernels track (M4)
   BEFORE: "B1 in no track"
   AFTER:  "B1 in the kernels track"
   SOURCE: Notion task B1, Done when ("used only for kernel development and short benchmarks"); lab/refine/PASS_RULES.md, "The four tracks" (a task belongs to exactly one track, and the merge pass records a move)
   WHY: a task no track claims is never walked, and kernel development is the kernels track's.
3. B1 Tier set (M5)
   BEFORE: "Tier: (empty)"
   AFTER:  "Tier: Opus"
   SOURCE: lab/refine/PASS_RULES.md, "What a pass may change" (Opus for hard builds); Notion task B1, Done when (a local vLLM build that serves a small model on the RX 9070 XT)
   WHY: an empty Tier stops the open-ticket skill, and repairing a ROCm build is a hard build.

Rulings raised (filed into OWNER_RULINGS.md; the arrow names the track and the ruling's number there):
(a) Does an answer whose request window spans a sampler restart get a joules-per-answer number? -> telemetry 1
    Now: no; the answer is counted as unmeasured.
    Recommend: no number, and a count of such answers beside the metric, so a gap never reads as a low energy cost.
    Source of tension: lab/refine/telemetry/RUBRIC.md, TT2 (joules per answer from the request window) against TT1 (source, sampling rate and error on every power number).
    Answer: ____
(b) At which batch size does the checkpoint shootout compare tok/s and joules per token? -> kernels 1
    Now: batch 1, the only batch size with a stored baseline.
    Recommend: batch 1 and the batching sweep's best size, so the shootout answers both the single-user case and the busy one.
    Source of tension: lab/refine/kernels/RUBRIC.md, K1 (stock numbers first, at several batch sizes) against the 15-minute slot of the GPU queue.
    Answer: ____
(c) Who sets the Tier of a lab task that is shared with another person? -> learning 1
    Now: nobody; L6 keeps an empty Tier until Xander answers the proposal on Dominick and Team.
    Recommend: the person who builds the next piece of it, through a proposal the other person answers; until then the open-ticket skill asks the referee, as it does today.
    Source of tension: lab/refine/PASS_RULES.md, "What a pass may never change" (a shared task) against the open-ticket skill, section 1 (an empty Tier stops the agent).
    Answer: ____

Left for the next pass: Track map beyond the rubric walk lists: telemetry T13; kernels B1. Telemetry walks T13 as rewritten, since it now covers pod B; kernels walks B1, new to the track, against K1 to K8; L6 keeps an empty Tier until Xander answers. Waits on Ilan: telemetry 1, kernels 1, learning 1.

Checks run (commands and their actual output):
python -m pytest -q tests/test_lab_refine_merge.py tests/test_lab_refine_formats.py, then its real output pasted here.
Re-fetched T13, R29 and B1 after changes 1 to 3: each reads as its AFTER value.

---
"""

LOG_HEADER = "# Refinement log for the {track} track\n\nHeader paragraph.\n\n---\n\n"
RULINGS_HEADER = "# Owner rulings for the {track} track\n\nHeader paragraph.\n"
MERGE_HEADER = "# Merge log for the four refinement tracks\n\nHeader paragraph.\n\n---\n\n"


def pass_entry(number: int, date: str = "2026-10-10", ruling: str | None = None) -> str:
    """A valid pass entry; with a question, it raises that question as ruling (a)."""
    reference = " RULING (a)." if ruling else ""
    raised = (
        "none"
        if ruling is None
        else f"(a) {ruling}\n"
        "    Now: the default stands.\n"
        "    Recommend: keep the default until Ilan answers.\n"
        '    Source of tension: lab/refine/PASS_RULES.md, "The merge pass" against the track rubric.\n'
        "    Answer: ____"
    )
    return f"""## Pass {number}, {date}

Walked T9 against D1. One change. NOT CONVERGED.

Research step (sources opened):
- lab/README.md, section "Teacher (pod A)" :: the measured idle and load power.

Metrics before (read before changing anything):
| metric | value | unit | node | conditions | source |
| --- | --- | --- | --- | --- | --- |
| pod.idle_watts | 39 | W | pod-a | no server loaded | metrics.jsonl:5 |

Walked (task or experiment :: dimensions checked :: result):
- T9 :: D1 :: BROKE D1; FIXED (1)

Findings (by dimension):
- D1 T9 :: the done-when names no sampling interval. FIXED (1).{reference}

Changes (1 of at most 5):
1. T9 on the board, rewritten (D1)
   BEFORE: "Sampler publishes power."
   AFTER:  "Sampler publishes power with its interval in ms."
   SOURCE: lab/README.md, section "Teacher (pod A)"
   WHY: a reading without an interval cannot be trusted.
   METRIC: none - plan change, nothing measured.

Rulings raised (candidates; the merge pass numbers them into OWNER_RULINGS.md):
{raised}

Metrics after (only if the pass ran anything):
not run: the pass changed the plan only.

Converged: NO - one change made.

Left for the next pass: re-walk T9.

---

"""


def report(track: str, number: int, date: str = "2026-10-10") -> str:
    return (
        f"Verbatim report returned by {track} refinement pass {number} (Opus sitting), {date}. "
        f"The log entry it appended is in lab/refine/{track}/REFINE_LOG.md.\n"
        f"Ended: {date}T11:00:00-04:00\n\n---\n\nThe report.\n\n```text\n## Pass {number}, {date}\n```\n"
    )


def ruling(number: int, question: str, origin: str, answer: str = "____") -> str:
    return (
        f"\n## {number}. {question} ({origin})\n\n"
        "Now: the default stands.\n"
        "Recommend: keep the default until Ilan answers.\n"
        'Source of tension: lab/refine/PASS_RULES.md, "The merge pass" against the track rubric.\n'
        f"Answer: {answer}\n"
    )


EXAMPLE_PASSES = {
    "telemetry": [TELEMETRY_QUESTION],
    "learning": [None],
    "plumbing": [None],
    "kernels": [KERNELS_QUESTION],
}
EXAMPLE_RULINGS = {
    "telemetry": [ruling(1, TELEMETRY_QUESTION, "pass 1")],
    "learning": [ruling(1, LEARNING_QUESTION, "merge 1")],
    "plumbing": [],
    "kernels": [ruling(1, KERNELS_QUESTION, "pass 1")],
}


def make_tree(root: Path, passes=None, rulings=None, merge: str = MERGE_EXAMPLE) -> Path:
    """Write a lab/refine folder: each track's passes (one entry per item, the item being the question that
    pass raises, or None), counter, reports and rulings, and MERGE_LOG.md holding the given entries."""
    passes = EXAMPLE_PASSES if passes is None else passes
    rulings = EXAMPLE_RULINGS if rulings is None else rulings
    for track in rm.TRACKS:
        folder = root / track
        (folder / "pass_reports").mkdir(parents=True)
        questions = passes.get(track, [])
        entries = "".join(pass_entry(n, ruling=q) for n, q in enumerate(questions, start=1))
        (folder / "REFINE_LOG.md").write_text(LOG_HEADER.format(track=track) + entries, encoding="utf-8")
        (folder / "refine_pass.txt").write_text(f"{len(questions)}\n", encoding="utf-8")
        for n in range(1, len(questions) + 1):
            (folder / "pass_reports" / f"pass_{n:02d}.md").write_text(report(track, n), encoding="utf-8")
        (folder / "OWNER_RULINGS.md").write_text(
            RULINGS_HEADER.format(track=track) + "".join(rulings.get(track, [])), encoding="utf-8"
        )
    (root / "MERGE_LOG.md").write_text(MERGE_HEADER + merge, encoding="utf-8")
    return root


def read_tree(root: Path):
    logs = {t: rf.parse_log((root / t / "REFINE_LOG.md").read_text(encoding="utf-8")) for t in rm.TRACKS}
    rulings = {t: rf.parse_rulings((root / t / "OWNER_RULINGS.md").read_text(encoding="utf-8")) for t in rm.TRACKS}
    merge_log = rf.parse_log((root / "MERGE_LOG.md").read_text(encoding="utf-8"))
    return logs, rulings, merge_log


def edit(text: str, old: str, new: str) -> str:
    """Replace exactly one occurrence, so a test never passes because its edit matched nothing."""
    assert text.count(old) == 1, f"expected exactly one {old!r} in the text"
    return text.replace(old, new)


def only_merge(text: str) -> rf.PassEntry:
    (entry,) = rf.parse_log(text).entries
    return entry


# --- the committed files -------------------------------------------------------------------------------


@pytest.fixture(params=["committed", "with-a-merge-entry"])
def merge_log_text(request):
    """The committed MERGE_LOG.md, then its header with a merge entry under it, as the first merge leaves the file."""
    text = (REFINE / "MERGE_LOG.md").read_text(encoding="utf-8")
    if request.param == "with-a-merge-entry":
        return text.split("\n## ", 1)[0].rstrip("\n") + "\n\n" + MERGE_EXAMPLE
    return text


def test_committed_merge_log_is_a_header_that_points_at_the_format_and_the_checks(merge_log_text):
    log = rf.parse_log(merge_log_text)

    assert log.header.startswith("# Merge log for the four refinement tracks")
    for name in ("lab/refine/TEMPLATES.md", "lab/refine/PASS_RULES.md", "lab/machine/refine_merge.py"):
        assert name in log.header


def test_committed_refinement_files_pass_every_merge_check():
    assert rm.check_tree(REFINE) == []


# --- the example round ---------------------------------------------------------------------------------


def test_example_round_passes_every_check(tmp_path):
    assert rm.check_tree(make_tree(tmp_path)) == []


def test_example_merge_entry_reads_as_written():
    entry = only_merge(MERGE_EXAMPLE)

    assert (entry.kind, entry.number, entry.date) == ("merge", 1, "2026-10-10")
    assert rm.walked_passes(entry) == (
        rm.WalkedPass("telemetry", 1, True),
        rm.WalkedPass("learning", 1, True),
        rm.WalkedPass("plumbing", 1, True),
        rm.WalkedPass("kernels", 1, True),
    )
    assert [change.number for change in entry.changes] == [1, 2, 3]
    assert [(r.label, r.question.rsplit(" -> ", 1)[1]) for r in entry.rulings_raised] == [
        ("a", "telemetry 1"),
        ("b", "kernels 1"),
        ("c", "learning 1"),
    ]
    assert entry.converged is None
    assert rm.check_merge_entry(entry) == []


def test_every_check_has_an_id_and_a_name():
    assert list(rm.CHECKS) == ["M1", "M2", "M3", "M4", "M5", "M6", "M7"]
    assert all(name for name in rm.CHECKS.values())


# --- the round: which pass entries the next merge reconciles --------------------------------------------


def walk_lines(*lines: str) -> str:
    return "\n".join(f"- {line}" for line in lines)


def merge_entry(number: int, walked: str, date: str = "2026-10-10") -> str:
    """A minimal valid merge entry with the given walk lines plus one line that runs every check."""
    return (
        f"## Merge {number}, {date}\n\nReconciled a round.\n\n"
        "Walked (pass entry, task, demo beat or file :: checks :: result):\n"
        f"{walked}\n- every open task of the four tracks :: M1, M2, M3, M4, M5, M7 :: held\n\n"
        "Findings (by dimension): none\n\nChanges: none\n\nRulings raised: none\n\n---\n\n"
    )


def test_round_passes_starts_after_the_highest_reconciled_pass_of_each_track():
    logs = {
        "telemetry": rf.parse_log(pass_entry(1) + pass_entry(2) + pass_entry(3)),
        "kernels": rf.parse_log(pass_entry(1)),
    }
    merge_log = rf.parse_log(merge_entry(1, walk_lines("telemetry pass 1 :: M6 :: held", "telemetry pass 2 :: M6 :: held")))

    found = rm.round_passes(logs, merge_log)

    assert {track: [p.number for p in passes] for track, passes in found.items()} == {"telemetry": [3], "kernels": [1]}
    assert rm.reconciled(merge_log) == {"telemetry": 2, "learning": 0, "plumbing": 0, "kernels": 0}


def test_a_pass_the_merge_reverted_is_not_reconciled_and_its_number_comes_back():
    merge_log = rf.parse_log(
        merge_entry(
            1,
            walk_lines(
                "telemetry pass 1 :: M6 :: held",
                "telemetry pass 2 :: M6 :: BROKE M6; FIXED (1)",
                "kernels pass 1 :: M1, M6 :: BROKE M1; FIXED (2)",
            ),
        )
    )

    assert rm.walked_passes(merge_log.entries[0]) == (
        rm.WalkedPass("telemetry", 1, True),
        rm.WalkedPass("telemetry", 2, False),
        rm.WalkedPass("kernels", 1, True),
    )
    assert rm.reconciled(merge_log)["telemetry"] == 1
    assert rm.reconciled(merge_log)["kernels"] == 1


def test_next_merge_number_counts_the_merge_entries():
    assert rm.next_merge_number(rf.parse_log(MERGE_HEADER)) == 1
    two = rf.parse_log(
        MERGE_HEADER
        + merge_entry(1, walk_lines("telemetry pass 1 :: M6 :: held"))
        + merge_entry(2, walk_lines("telemetry pass 2 :: M6 :: held"))
    )
    assert rm.next_merge_number(two) == 3


# --- what a merge entry must keep ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        (
            "- R11 :: M1, M2 :: held",
            "- R11 :: M1, K2 :: held",
            ["merge 1: walked R11: K2 is not a merge check id (M1, M2, M3, M4, M5, M6, M7)"],
        ),
        (
            "- answered rulings in the four OWNER_RULINGS.md files :: M7 :: held\n",
            "",
            ["merge 1: the walk never runs M7; every merge runs all of M1, M2, M3, M4, M5, M6, M7"],
        ),
        (
            "- plumbing pass 1 :: M6 :: held",
            "- plumbing pass 1 :: M1 :: held",
            ["merge 1: walked plumbing pass 1: a pass entry of the round is checked against M6"],
        ),
        (
            "- learning pass 1 :: M6 :: held\n- plumbing pass 1 :: M6 :: held",
            "- learning pass 1, plumbing pass 1 :: M6 :: held",
            ["merge 1: walked learning pass 1, plumbing pass 1: name one pass entry per line, as '<track> pass <N>'"],
        ),
        (
            "- kernels pass 1 :: M6 :: held",
            "- kernel pass 1 :: M6 :: held",
            ["merge 1: walked kernel pass 1: 'kernel' is not a track (telemetry, learning, plumbing, kernels)"],
        ),
        (
            "At which batch size does the checkpoint shootout compare tok/s and joules per token? -> kernels 1",
            "At which batch size does the checkpoint shootout compare tok/s and joules per token?",
            ["merge 1: ruling (b) does not say where it was filed; end its question with ' -> <track> <n>'"],
        ),
        (
            "- M5 B1 :: the Tier is empty",
            "- D5 B1 :: the Tier is empty",
            ["merge 1: the finding for D5 B1: D5 is not a merge check id"],
        ),
        (
            "3. B1 Tier set (M5)",
            "3. B1 Tier set (M5, U1)",
            ["merge 1: change 3: U1 is not a merge check id"],
        ),
    ],
)
def test_a_merge_entry_reports_exactly_what_breaks_the_merge_rules(old, new, expected):
    assert rm.check_merge_entry(only_merge(edit(MERGE_EXAMPLE, old, new))) == expected


def test_a_merge_entry_must_also_keep_the_pass_format():
    text = edit(MERGE_EXAMPLE, "   WHY: one measurement should take one GPU slot; T13 is the older task, so it stays.\n", "")

    assert rm.check_merge_entry(only_merge(text)) == ["merge 1: change 1 has no WHY"]


def test_a_pass_entry_is_not_a_merge_entry():
    entry = rf.parse_log(pass_entry(1)).entries[0]

    assert rm.check_merge_entry(entry) == ["pass 1: MERGE_LOG.md holds merge entries only"]


RUBRIC_CHANGE = """\
4. lab/refine/learning/RUBRIC.md, LT4 rewritten (M7)
   BEFORE: "a margin larger than measured run-to-run noise"
   AFTER:  "a margin larger than measured run-to-run noise, from a paired test on the held-out set"
   SOURCE: {source}
   WHY: Ilan answered the ruling and named this change.

Rulings raised"""

PRIORITY_CHANGE = """\
4. L9 Priority raised to P0 Demo-critical (M3)
   BEFORE: "P1 Strengthens demo"
   AFTER:  "P0 Demo-critical"
   SOURCE: {source}
   WHY: Ilan answered the ruling and named this change.

Rulings raised"""


def with_change(template: str, source: str) -> str:
    """The example merge entry with a fourth change, a rubric edit or a Priority change, citing source."""
    return edit(MERGE_EXAMPLE, "\nRulings raised", "\n" + template.format(source=source))


def answered_learning_rulings():
    return {
        **EXAMPLE_RULINGS,
        "learning": [ruling(1, LEARNING_QUESTION, "merge 1", answer="the builder sets it (owner, 2026-10-11)")],
    }


@pytest.mark.parametrize("template", [RUBRIC_CHANGE, PRIORITY_CHANGE])
def test_a_rubric_or_priority_change_carried_out_on_an_answered_ruling_is_valid(tmp_path, template):
    merge = with_change(template, "lab/refine/learning/OWNER_RULINGS.md 1 (owner, 2026-10-11)")
    root = make_tree(tmp_path, rulings=answered_learning_rulings(), merge=merge)

    assert rm.check_tree(root) == []


@pytest.mark.parametrize("template", [RUBRIC_CHANGE, PRIORITY_CHANGE])
def test_a_rubric_or_priority_change_without_a_ruling_is_reported(template):
    text = with_change(template, "lab/refine/PASS_RULES.md, \"The merge pass\"")

    assert rm.check_merge_entry(only_merge(text)) == [
        "merge 1: change 4 edits a RUBRIC.md or a Priority, so its SOURCE cites the answered ruling as "
        "'<track>/OWNER_RULINGS.md <n>'"
    ]


@pytest.mark.parametrize("template", [RUBRIC_CHANGE, PRIORITY_CHANGE])
def test_a_rubric_or_priority_change_on_an_unanswered_ruling_is_reported(tmp_path, template):
    merge = with_change(template, "lab/refine/learning/OWNER_RULINGS.md 1")
    logs, rulings, merge_log = read_tree(make_tree(tmp_path, merge=merge))

    assert rm.check_merge_entry(merge_log.entries[0], logs, rulings) == [
        "merge 1: change 4 cites learning ruling 1, which has no Answer yet"
    ]


def test_file_names_notion_turned_into_links_still_name_the_rubric_and_cite_the_ruling(tmp_path):
    """A merge entry that quotes a change as Notion shows it carries file names as links,
    [OWNER_RULINGS.md](http://OWNER_RULINGS.md), because .md is also a web domain ending. The rubric target and the
    ruling citation read through them, so a change carried out on an answered ruling is still valid."""
    template = RUBRIC_CHANGE.replace("lab/refine/learning/RUBRIC.md,", "lab/refine/learning/[RUBRIC.md](http://RUBRIC.md),")
    merge = with_change(template, "lab/refine/learning/[OWNER_RULINGS.md](http://OWNER_RULINGS.md) 1 (owner, 2026-10-11)")
    root = make_tree(tmp_path, rulings=answered_learning_rulings(), merge=merge)

    assert rm.check_tree(root) == []


def test_a_change_citing_a_ruling_that_does_not_exist_is_reported(tmp_path):
    merge = with_change(RUBRIC_CHANGE, "lab/refine/learning/OWNER_RULINGS.md 7")
    logs, rulings, merge_log = read_tree(make_tree(tmp_path, merge=merge))

    assert rm.check_merge_entry(merge_log.entries[0], logs, rulings) == [
        "merge 1: change 4 cites learning ruling 7, which does not exist"
    ]


def test_a_listed_ruling_must_exist_where_the_entry_says_it_was_filed(tmp_path):
    rulings = {**EXAMPLE_RULINGS, "kernels": []}
    logs, parsed, merge_log = read_tree(make_tree(tmp_path, rulings=rulings))

    assert rm.check_merge_entry(merge_log.entries[0], logs, parsed) == [
        "merge 1: ruling (b) says it was filed as kernels 1, but that file has no ruling 1"
    ]


def test_a_listed_ruling_must_ask_the_question_the_file_holds(tmp_path):
    rulings = {**EXAMPLE_RULINGS, "telemetry": [ruling(1, "Is idle draw part of an answer's energy?", "pass 1")]}
    logs, parsed, merge_log = read_tree(make_tree(tmp_path, rulings=rulings))

    assert rm.check_merge_entry(merge_log.entries[0], logs, parsed) == [
        "merge 1: ruling (a) and telemetry ruling 1 do not ask the same question"
    ]


def test_every_ruling_a_reconciled_pass_raised_must_be_listed(tmp_path):
    passes = {**EXAMPLE_PASSES, "plumbing": ["Should the controller retry a refused launch?"]}
    logs, rulings, merge_log = read_tree(make_tree(tmp_path, passes=passes))

    assert rm.check_merge_entry(merge_log.entries[0], logs, rulings) == [
        "merge 1: plumbing pass 1 raised ruling (a), which this merge does not list as filed in the plumbing rulings"
    ]


def test_a_reconciled_pass_must_be_in_its_track_log(tmp_path):
    passes = {**EXAMPLE_PASSES, "learning": []}
    logs, rulings, merge_log = read_tree(make_tree(tmp_path, passes=passes))

    assert rm.check_merge_entry(merge_log.entries[0], logs, rulings) == [
        "merge 1: walked learning pass 1, which is not in that track's REFINE_LOG.md"
    ]


def test_a_reverted_pass_may_be_gone_from_its_log_and_its_rulings_are_not_filed(tmp_path):
    merge = edit(MERGE_EXAMPLE, "- learning pass 1 :: M6 :: held", "- learning pass 1 :: M6 :: BROKE M6; FIXED (1)")
    merge = edit(merge, "- M4 T13, R29 ::", "- M6 learning pass 1 :: the entry has no metrics-before block, so it was reverted. FIXED (1).\n- M4 T13, R29 ::")
    passes = {**EXAMPLE_PASSES, "learning": []}
    logs, rulings, merge_log = read_tree(make_tree(tmp_path, passes=passes, merge=merge))

    assert rm.check_merge_entry(merge_log.entries[0], logs, rulings) == []
    assert rm.reconciled(merge_log)["learning"] == 0


# --- what MERGE_LOG.md as a whole must keep -----------------------------------------------------------


def test_merge_numbers_run_from_one_in_file_order():
    text = MERGE_HEADER + edit(MERGE_EXAMPLE, "## Merge 1, 2026-10-10", "## Merge 2, 2026-10-10")

    assert rm.check_merge_log(rf.parse_log(text)) == ["merge entries are numbered 1 to 1 in file order, found [2]"]


def test_a_merge_dated_before_the_one_above_it_is_reported():
    text = (
        MERGE_HEADER
        + merge_entry(1, walk_lines("telemetry pass 1 :: M6 :: held"), date="2026-10-11")
        + merge_entry(2, walk_lines("telemetry pass 2 :: M6 :: held"), date="2026-10-10")
    )

    assert rm.check_merge_log(rf.parse_log(text)) == ["merge 2 is dated 2026-10-10, before merge 1 (2026-10-11)"]


def test_a_pass_reconciled_twice_is_reported():
    text = (
        MERGE_HEADER
        + merge_entry(1, walk_lines("telemetry pass 1 :: M6 :: held"))
        + merge_entry(2, walk_lines("telemetry pass 1 :: M6 :: held"))
    )

    assert rm.check_merge_log(rf.parse_log(text)) == [
        "telemetry: the merges reconcile passes [1, 1]; each pass is reconciled once, in order, from pass 1 "
        "(expected [1, 2])"
    ]


def test_a_pass_no_merge_reconciled_is_reported():
    text = MERGE_HEADER + merge_entry(1, walk_lines("kernels pass 2 :: M6 :: held"))

    assert rm.check_merge_log(rf.parse_log(text)) == [
        "kernels: the merges reconcile passes [2]; each pass is reconciled once, in order, from pass 1 (expected [1])"
    ]


def test_pass_entries_and_run_states_do_not_belong_in_the_merge_log():
    text = MERGE_HEADER + pass_entry(1) + "## State at the end of the run\n\nRun ended: converged at pass 1.\n\n---\n"

    assert rm.check_merge_log(rf.parse_log(text)) == [
        "pass 1: a pass entry belongs in its track's REFINE_LOG.md, not in MERGE_LOG.md",
        "a 'State at the end of the run' section belongs in its track's REFINE_LOG.md, not in MERGE_LOG.md",
    ]


# --- what a track's files must keep -------------------------------------------------------------------


def track_files(tmp_path, entries: str, counter: str, reports: dict[int, str]) -> Path:
    passes = {track: [] for track in rm.TRACKS}
    rulings = {track: [] for track in rm.TRACKS}
    root = make_tree(tmp_path, passes=passes, rulings=rulings, merge="")
    folder = root / "kernels"
    (folder / "REFINE_LOG.md").write_text(LOG_HEADER.format(track="kernels") + entries, encoding="utf-8")
    (folder / "refine_pass.txt").write_text(counter, encoding="utf-8")
    for number, text in reports.items():
        (folder / "pass_reports" / f"pass_{number:02d}.md").write_text(text, encoding="utf-8")
    return root


def test_a_track_with_two_valid_passes_their_counter_and_reports_passes(tmp_path):
    root = track_files(tmp_path, pass_entry(1) + pass_entry(2), "2\n", {1: report("kernels", 1), 2: report("kernels", 2)})

    assert rm.check_tree(root) == []


def test_the_merge_and_the_track_check_report_the_same_track_problems(tmp_path):
    """refine_track.check_track and the merge's check_tree read one track folder by the same rules: here a report
    written for another pass, which only the track check used to see, and a report with no pass in the log, which
    only the merge check used to see."""
    wrong_pass = report("kernels", 1).replace("kernels refinement pass 1 (", "kernels refinement pass 2 (")
    root = track_files(tmp_path, pass_entry(1), "1\n", {1: wrong_pass, 3: report("kernels", 3)})
    metrics = tmp_path / "metrics.jsonl"
    row = {"metric": "pod.idle_watts", "value": 39, "unit": "W", "node": "pod-a"}
    metrics.write_text("{}\n" * 4 + json.dumps(row) + "\n", encoding="utf-8")

    track_side = rt.check_track(root / "kernels", metrics)

    assert track_side == [
        "pass_reports/pass_01.md: the first line must read 'Verbatim report returned by kernels refinement pass 1 (...'",
        "pass_reports/pass_03.md has no pass 3 in REFINE_LOG.md",
    ]
    assert rm.check_tree(root) == [f"kernels: {problem}" for problem in track_side]


@pytest.mark.parametrize(
    ("entries", "counter", "reports", "expected"),
    [
        (
            pass_entry(1) + pass_entry(2),
            "1\n",
            {1: report("kernels", 1), 2: report("kernels", 2)},
            ["kernels: refine_pass.txt reads 1 but the log's last pass entry is pass 2; a pass writes the counter last, after its entry"],
        ),
        (
            pass_entry(1),
            "1\n",
            {},
            ["kernels: pass 1 has no report at pass_reports/pass_01.md"],
        ),
        (
            pass_entry(1),
            "1\n",
            {1: report("kernels", 1), 2: report("kernels", 2)},
            ["kernels: pass_reports/pass_02.md has no pass 2 in REFINE_LOG.md"],
        ),
        (
            pass_entry(1),
            "1\n",
            {1: report("kernels", 1).replace("T11:00:00-04:00", "T11:00:00")},
            [
                "kernels: pass_reports/pass_01.md: 'Ended: 2026-10-10T11:00:00' has no UTC offset, "
                "for example 2026-10-10T06:52:00-04:00"
            ],
        ),
        (
            pass_entry(1),
            "1\n",
            {1: report("kernels", 1).replace("## Pass 1, 2026-10-10", "## Pass 1, 2026-10-11")},
            ["kernels: pass_reports/pass_01.md: does not quote its entry; no line reads '## Pass 1, 2026-10-10'"],
        ),
        (
            pass_entry(2),
            "2\n",
            {2: report("kernels", 2)},
            ["kernels: REFINE_LOG.md: pass entries must be numbered 1 to 1 in file order, found [2]"],
        ),
        (
            edit(pass_entry(1), "| pod.idle_watts | 39 | W | pod-a | no server loaded | metrics.jsonl:5 |\n", ""),
            "1\n",
            {1: report("kernels", 1)},
            ["kernels: REFINE_LOG.md: pass 1: 'Metrics before' needs a table with at least one reading"],
        ),
    ],
)
def test_a_track_reports_exactly_what_its_files_break(tmp_path, entries, counter, reports, expected):
    assert rm.check_tree(track_files(tmp_path, entries, counter, reports)) == expected


@pytest.mark.parametrize(
    ("entries", "expected"),
    [
        (ruling(2, LEARNING_QUESTION, "merge 1"), ["learning: rulings are numbered 1 to 1 in file order, found [2]"]),
        (
            ruling(1, LEARNING_QUESTION, "pass 1") + ruling(2, KERNELS_QUESTION, "the merge"),
            ["learning ruling 2: the origin 'the merge' is not 'pass <N>', 'merge <N>' or 'owner, <YYYY-MM-DD>'"],
        ),
    ],
)
def test_a_rulings_file_reports_its_numbering_and_origins(entries, expected):
    rulings = rf.parse_rulings(RULINGS_HEADER.format(track="learning") + entries)

    assert rm.check_rulings("learning", rulings) == expected


def test_a_rulings_file_with_rulings_in_order_and_known_origins_passes():
    text = (
        RULINGS_HEADER.format(track="learning")
        + ruling(1, LEARNING_QUESTION, "pass 1")
        + ruling(2, KERNELS_QUESTION, "merge 1")
        + ruling(3, TELEMETRY_QUESTION, "owner, 2026-10-11", answer="only the window (owner, 2026-10-11)")
    )

    assert rm.check_rulings("learning", rf.parse_rulings(text)) == []


def test_a_file_that_does_not_parse_or_is_missing_is_reported(tmp_path):
    root = make_tree(tmp_path)
    (root / "plumbing" / "refine_pass.txt").write_text("one\n", encoding="utf-8")
    (root / "telemetry" / "OWNER_RULINGS.md").unlink()

    assert rm.check_tree(root) == [
        "telemetry/OWNER_RULINGS.md is missing",
        "plumbing: refine_pass.txt must hold one non-negative integer, found 'one\\n'",
    ]
