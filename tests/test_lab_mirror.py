"""Tests for the Notion mirror (python -m lab mirror): the digest into the Ilan section of Overnight Status, and the
refinement logs' plan changes into the Plan changes item on Ilan's Team Comms page.

The facts come from recorded files under fixtures/lab/mirror/ (a ledger, a board snapshot, the four tracks' rulings
and logs, a merge log), read by the real digest and the real log parsers. Only Notion is faked, at its boundary: an
in-memory page store that behaves like the page-markdown endpoints (lab/machine/pages_fake.py), used both as the
protocol the flows edit through and as the wire under the real REST client. Every test has its own state dir and sees
no .env file and no NOTION_TOKEN, so none of them can write the real ledger or reach the real pages.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from lab import __main__ as lab_cli
from lab.machine import ledger, mirror, pages_fake, record, rulings
from lab.machine.pages_fake import FakePages

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "fixtures" / "lab" / "mirror"
LEDGER = FIXTURES / "ledger.jsonl"
BOARD = FIXTURES / "board.json"
REFINE = FIXTURES / "refine"

UTC = dt.timezone.utc
NOW = dt.datetime(2026, 10, 10, 16, 0, tzinfo=UTC)  # Saturday 12:00 PM Eastern
LATER = NOW + dt.timedelta(minutes=5)
HEART = "\U0001F916"
DOT = "\u00b7"
SIGN = f"{HEART} Ilan's agent {DOT} Sat 12:00 PM ET:"
TOKEN = pages_fake.TOKEN
OVERNIGHT = rulings.OVERNIGHT_PAGE
TEAM = rulings.TEAM_COMMS_PAGE

NEEDS_OLD = "- **Needs a human (Ilan):** Q2 and Q3 on Ilan \u2194 Dominick (dataset and package formats)"
OVERNIGHT_TEXT = "\n".join(
    [
        '<callout icon="\U0001F319" color="purple_bg">',
        "\tEach person's coding agent rewrites **its own section** every cycle (about every 30 minutes).",
        "</callout>",
        "## How agents update this page",
        "- Replace only your own section's lines. Never edit another person's section.",
        "- **Needs a human:** one line per item, linking the question on your person's Team Comms page.",
        "---",
        "## Dominick",
        f"- **Last update:** {HEART} Dominick's agent {DOT} Sat 6:10 AM ET (earlier timestamps in my section were off by hours)",
        "- **Working on:** waiting for Xander's answer (Q5 on his page)",
        "- **Done tonight:** PRs #11\u2013#32 of mine: e2e test, Circle presence + leaderboards",
        "- **Blocked on:** `main` has 4 failing perception tests since Hemdutt's #17",
        "- **Needs a human (Dominick):** (1) who owns Sort It S1/S2 (Dominick \u2194 Team Q1)",
        "## Hemdutt",
        "- **Last update:** \u2014",
        "- **Working on:** \u2014",
        "- **Done tonight:** \u2014",
        "- **Blocked on:** \u2014",
        "- **Needs a human (Hemdutt):** \u2014",
        "## Ilan",
        f"- **Last update:** {HEART} Ilan's agent {DOT} Sat 6:50 AM ET",
        "- **Working on:** Ilan's lab machine, the Orchestration tasks O1 to O41",
        "- **Done tonight:** R2 merged (`bb351e6`, PR #37)",
        "- **Blocked on:** nothing on the lab side",
        NEEDS_OLD,
        "## Xander",
        f"- **Last update:** {HEART} Xander's agent {DOT} Sat 11:03 AM ET (from `date` on the Jetson)",
        "- **Working on:** nothing new is unblocked; the gateway holds steady at \\~630 MB",
        "- **Done tonight:** R8 `atlas-llm` (#36, cache fix #70)",
        "- **Blocked on:** a USB webcam on the Jetson",
        "- **Needs a human (Xander):** (1) a USB webcam for the Jetson",
    ]
)
FYI = "## FYI (no reply needed)"
TEAM_TEXT = "\n".join(
    [
        '<callout icon="\U0001F9EA" color="orange_bg">',
        "\tIlan owns: GPU lab setup (R2), teacher labeling (L2).",
        "</callout>",
        "## Open questions",
        "- [x] **Q1. Can the R9700 servers join Tailscale?**",
        "\t- Answer: No.",
        "- [ ] **Q2. Dataset format for `label.py`.**",
        "\t- Answer:",
        "## Answered",
        FYI,
        f"- {HEART} Ilan's agent {DOT} Sat 6:40 AM: **Ilan's lane split into finer tasks on the board** so several agents can build in parallel.",
        "- **Start here with your coding agent:** have it read `AGENTS.md` in the repo.",
        "\t- A child of that item.",
        "## Questions for Dominick",
        "-",
    ]
)
PLAN_ITEM = "- **Plan changes** (one line per board change a refinement pass makes; undo a line by restoring its Was value)"

# What the recorded ledger, board snapshot and rulings say at NOW (12 hours of window), laid out as the page shows it.
WORKING = "- **Working on:** 5 In progress: O14 (Sonnet), O17 (Opus), O20 (Opus), O24 (Sonnet), S4 (Opus); GPU B exclusive: R11 running"
DONE = "- **Done tonight:** O15 (Done on the board, with ledger rows in the window); 5 of Ilan's 22 tasks Done in all"
BLOCKED = (
    "- **Blocked on:** L11 is Blocked; O41 held back, a blocker is missing from the board snapshot; "
    "GPU A teacher shared service: 1 queued; GPU B exclusive: R11 past its cap, 2 queued; "
    "1 complaint open on the GPU queue; 2 failed ledger rows in the window (L13 label error; L13 label fail)"
)
LAST = (
    f"- **Last update:** {SIGN} lab digest, Sat 12:00 AM ET to Sat 12:00 PM ET; "
    "board snapshot from Sat 11:45 AM ET (digest {id})"
)
SEGMENT = f"{SIGN} Lab digest: R5 is shared with a teammate, so no sitting takes it (digest {{id}})"

E3_LINE = (
    'kernels pass 3, E3: E3 on T0, rewritten. Was: "E3 batching sweep on vLLM (batch 1, 4, 16, 64: J/token curve)". '
    "Why: the sweep skipped batch 8, which the stock baseline requires, named no repeat count and reported J/token "
    "without tok/s, so no win could be told from noise and speed was not paired with energy; five runs is a starting "
    "count, to be raised if the spread is wide. Source: Notion task O4, \"Kernel tests\" 1, 3 and 6 "
    "(https://app.notion.com/p/3f596ef5fec5816aae04d8fb488302c4); `lab/refine/PASS_RULES.md`, section "
    '"The four tracks" (a queued experiment line names its track).'
)
MERGE_LINE = (
    'merge 1, T3: T3 on the board, Blocked by gains L4. Was: "Blocked by: T1". Why: the sampler cannot tag a run '
    "before the gate that names it exists. Source: Notion task T3, \"Done when\" (the sampler reads the eval gate's "
    "run id); `lab/refine/learning/RUBRIC.md`, LT4."
)
# The nine changes of the recorded logs, in the order they are posted: (label, task).
ALL_CHANGES = [
    ("telemetry pass 1", "T4"), ("telemetry pass 1", "T3"), ("telemetry pass 1", "T2"), ("telemetry pass 1", "T0"),
    ("telemetry pass 1", "T1"), ("kernels pass 3", "E3"), ("kernels pass 3", "E5"), ("kernels pass 3", "T0"),
    ("merge 1", "T3"),
]


@pytest.fixture(autouse=True)
def state(tmp_path, monkeypatch):
    """A private state dir holding the recorded ledger, no .env files, no Notion token, no sitting identity."""
    folder = tmp_path / "state"
    folder.mkdir()
    shutil.copyfile(LEDGER, folder / "ledger.jsonl")
    monkeypatch.setenv("LAB_STATE_DIR", str(folder))
    monkeypatch.setenv("LAB_ENV_FILES", "")
    for name in ("NOTION_TOKEN", "LAB_ACTOR", "LAB_TIER", "LAB_TASK"):
        monkeypatch.delenv(name, raising=False)
    return folder


def pages(overnight: str = OVERNIGHT_TEXT, team: str = TEAM_TEXT) -> FakePages:
    return FakePages({OVERNIGHT: overnight, TEAM: team})


def go(*argv: str, client=None, now: dt.datetime = NOW, board=BOARD, refine=REFINE, **kwargs):
    """One mirror call on the recorded inputs: (the answer, the exit code)."""
    kwargs.setdefault("environ", {})
    kwargs.setdefault("env_files", [])
    return mirror.run([*argv, "--board", str(board), "--refine-dir", str(refine)], client=client, now=now, **kwargs)


def lines(text: str) -> list[str]:
    return text.split("\n")


def ilan_section(text: str) -> list[str]:
    """The lines under '## Ilan', up to the next heading."""
    rows = lines(text)
    start = rows.index("## Ilan")
    end = next((i for i in range(start + 1, len(rows)) if rows[i].startswith("## ")), len(rows))
    return rows[start + 1:end]


def outside_ilan(text: str) -> list[str]:
    rows = lines(text)
    start = rows.index("## Ilan")
    end = next((i for i in range(start + 1, len(rows)) if rows[i].startswith("## ")), len(rows))
    return rows[:start + 1] + rows[end:]


def marker(line: str) -> str:
    found = re.search(r"\(digest ([0-9a-f]{8})\)", line)
    assert found, f"no digest marker in {line!r}"
    return found[1]


def needs_line(text: str) -> str:
    return next(row for row in lines(text) if row.startswith("- **Needs a human (Ilan):**"))


def plan_children(text: str) -> list[str]:
    """The child lines of the Plan changes item."""
    rows = lines(text)
    start = rows.index(PLAN_ITEM)
    end = start + 1
    while end < len(rows) and rows[end].startswith("\t"):
        end += 1
    return rows[start + 1:end]


def outside_plan_item(text: str) -> list[str]:
    """The page without the Plan changes item and its children."""
    rows = lines(text)
    start = rows.index(PLAN_ITEM)
    end = start + 1
    while end < len(rows) and rows[end].startswith("\t"):
        end += 1
    return rows[:start] + rows[end:]


def labels_of(children: list[str]) -> list[str]:
    """Each child line's '<label>, <task>:' after its signed prefix."""
    found = [re.match(r"\t- \S+ Ilan's agent \S+ \w{3} \d+:\d{2} [AP]M(?: ET)?: ([a-z0-9 -]+, [A-Za-z0-9]+):", row) for row in children]
    assert all(found), children
    return [match[1] for match in found]


def signed(text: str, sign: str = SIGN) -> str:
    return f"\t- {sign} {text}"


def open_rulings():
    rows, errors, warnings = rulings.collect(REFINE)
    assert (errors, warnings) == ([], [])
    return [row for row in rows if row.state == rulings.OPEN]


def ledger_rows(verb: str = "mirror"):
    return ledger.read(verb=verb)


def add_row(**fields) -> int:
    """Append a row to the ledger as a sitting's verb would, at a fixed time so no test depends on the clock."""
    row = {"ts": "2026-10-10T15:50:00Z", "actor": "sonnet-9", "tier": "sonnet", "task": None, "verb": "label",
           "outcome": "ok", "metrics": {}, "refs": {}, **fields}
    return ledger.append(row)


def quiet_board(folder: Path) -> Path:
    """A board snapshot with no tasks and no GPU rows: nothing for the section to say."""
    path = folder / "quiet-board.json"
    path.write_text(json.dumps({"schema": 1, "taken_at": "2026-10-10T15:55:00Z", "tasks": [], "gpu_queue": []}), encoding="utf-8")
    return path


# ------------------------------------------------------------------ the section, from the recorded inputs


def test_the_section_is_what_the_recorded_ledger_board_and_rulings_say():
    answer, code = go("--print", "--only", "overnight")

    assert (code, answer["ok"], answer["outcome"], answer["mode"]) == (0, True, "printed", "print")
    post = answer["posts"][0]
    ident = answer["section_id"]
    assert re.fullmatch(r"[0-9a-f]{8}", ident)
    assert post["lines"] == {
        "Last update": LAST.format(id=ident),
        "Working on": WORKING,
        "Done tonight": DONE,
        "Blocked on": BLOCKED,
    }
    assert post["needs_segment"] == SEGMENT.format(id=ident)
    assert answer["window"] == {"since": "2026-10-10T04:00:00Z", "until": "2026-10-10T16:00:00Z"}
    assert answer["digest_problems"] == ["1 task row without an ID skipped"]


def test_the_id_follows_the_facts_and_not_the_clock_or_the_verbs_own_rows():
    first, _ = go("--print", "--only", "overnight")
    again, _ = go("--print", "--only", "overnight", now=LATER)  # by now the first call has left a mirror row in the ledger

    assert [row["outcome"] for row in ledger_rows()] == ["failed", "ok", "printed", "printed"]
    assert again["section_id"] == first["section_id"]

    add_row(task="O99", outcome="error")
    changed, _ = go("--print", "--only", "overnight", now=LATER)
    assert changed["section_id"] != first["section_id"]
    assert "3 failed ledger rows in the window (O99 label error; L13 label error; L13 label fail)" in changed["posts"][0]["lines"]["Blocked on"]


def test_the_verbs_own_rows_never_count_as_failures_or_as_work_done_on_a_task():
    answer, _ = go("--print", "--only", "overnight")

    lines_ = answer["posts"][0]["lines"]
    assert "mirror" not in lines_["Blocked on"]  # the recorded failed mirror row
    assert "O28" not in lines_["Done tonight"]  # Done on the board, but its only row in the window is a mirror row


def test_a_task_done_on_the_board_is_done_tonight_only_when_the_ledger_shows_work_in_the_window():
    wide, _ = go("--print", "--only", "overnight", "--hours", "24")
    narrow, _ = go("--print", "--only", "overnight", "--hours", "1")

    assert wide["posts"][0]["lines"]["Done tonight"].startswith("- **Done tonight:** O15 (Done on the board")
    assert narrow["posts"][0]["lines"]["Done tonight"] == "- **Done tonight:** none yet in the window; 5 of Ilan's 22 tasks Done in all"


def test_since_names_the_start_of_the_window_and_excludes_hours():
    answer, code = go("--print", "--only", "overnight", "--since", "2026-10-10T14:35:00Z")

    assert code == 0
    assert answer["window"]["since"] == "2026-10-10T14:35:00Z"
    assert "2 failed ledger rows" not in answer["posts"][0]["lines"]["Blocked on"]
    assert "1 failed ledger row in the window (L13 label error)" in answer["posts"][0]["lines"]["Blocked on"]
    with pytest.raises(SystemExit) as refused:
        go("--print", "--hours", "2", "--since", "2026-10-10T14:35:00Z")
    assert refused.value.code == 2


@pytest.mark.parametrize("argv", [["--hours", "0"], ["--hours", "-1"], ["--hours", "100000"], ["--since", "yesterday"], ["--only", "both"]])
def test_bad_window_arguments_exit_2_and_write_nothing(argv):
    before = ledger.read()

    with pytest.raises(SystemExit) as refused:
        go("--print", *argv)

    assert refused.value.code == 2
    assert ledger.read() == before


def test_a_window_that_starts_after_now_is_a_usage_error_and_writes_nothing():
    before = ledger.read()

    answer, code = go("--print", "--since", "2026-10-10T17:00:00Z")

    assert (code, answer["ok"], answer["outcome"]) == (2, False, "usage")
    assert ledger.read() == before


def test_without_a_board_snapshot_the_section_says_so_and_guesses_nothing(tmp_path):
    answer, code = go("--print", "--only", "overnight", board=tmp_path / "no-such-board.json")

    lines_ = answer["posts"][0]["lines"]
    assert code == 0
    assert lines_["Working on"] == (
        "- **Working on:** no board snapshot, so the tasks In progress are unknown (`python -m lab digest --save-board <file>` stores one); "
        "ledger rows in the window for L3, L13, O15, O17, O19, and 2 more"
    )
    assert lines_["Done tonight"] == "- **Done tonight:** unknown without a board snapshot"
    assert lines_["Blocked on"] == "- **Blocked on:** 2 failed ledger rows in the window (L13 label error; L13 label fail)"
    assert lines_["Last update"].endswith(f"board snapshot missing (digest {answer['section_id']})")
    assert answer["posts"][0]["needs_segment"] == ""
    assert any("no board snapshot" in problem for problem in answer["digest_problems"])


def test_a_board_snapshot_older_than_an_hour_is_marked_stale():
    fresh, _ = go("--print", "--only", "overnight", now=NOW + dt.timedelta(minutes=30))
    stale, _ = go("--print", "--only", "overnight", now=NOW + dt.timedelta(minutes=80))

    assert "board snapshot from Sat 11:45 AM ET (digest" in fresh["posts"][0]["lines"]["Last update"]
    assert "board snapshot from Sat 11:45 AM ET, stale (digest" in stale["posts"][0]["lines"]["Last update"]


def test_the_stored_board_snapshot_is_used_when_none_is_named(state):
    shutil.copyfile(BOARD, state / "board.json")

    answer, _ = mirror.run(["--print", "--only", "overnight", "--refine-dir", str(REFINE)], now=NOW, environ={}, env_files=[])

    assert answer["posts"][0]["lines"]["Working on"] == WORKING


def test_a_quiet_lab_has_nothing_blocked_and_nothing_for_a_human(tmp_path):
    board = tmp_path / "board.json"
    board.write_text(json.dumps({
        "schema": 1, "taken_at": "2026-10-10T15:55:00Z",
        "tasks": [{"userDefined:ID": "O1", "Task": "O1: one", "Status": "Not started", "Tier": "Opus", "Locked": "__NO__",
                   "Assignee": ["user://fbcc0baf-9942-4661-9414-319688df6910"], "Blocked by": []}],
        "gpu_queue": [],
    }), encoding="utf-8")
    quiet = tmp_path / "quiet-state"
    quiet.mkdir()

    answer, code = go("--print", "--only", "overnight", board=board, state_dir=quiet, refine=tmp_path)

    lines_ = answer["posts"][0]["lines"]
    assert code == 0
    assert lines_["Working on"] == "- **Working on:** nothing In progress"
    assert lines_["Done tonight"] == "- **Done tonight:** none yet in the window; 0 of Ilan's 1 tasks Done in all"
    assert lines_["Blocked on"] == "- **Blocked on:** nothing"
    assert answer["posts"][0]["needs_segment"] == ""


def test_text_from_a_ledger_row_is_escaped_for_notion():
    add_row(task="O99", verb="run*it", outcome="fail_[x]")

    answer, _ = go("--print", "--only", "overnight")

    assert "O99 run\\*it `fail_[x]`" in answer["posts"][0]["lines"]["Blocked on"]


# ------------------------------------------------------------------ the Ilan section on the page


def test_the_first_run_rewrites_the_ilan_section_and_nothing_else_on_the_page():
    notion = pages()

    answer, code = go("--only", "overnight", client=notion)

    assert (code, answer["ok"], answer["outcome"]) == (0, True, "ok")
    assert answer["targets"] == {"overnight_status": {"action": "posted"}}
    after = notion.text(OVERNIGHT)
    assert outside_ilan(after) == outside_ilan(OVERNIGHT_TEXT)
    section = ilan_section(after)
    ident = marker(section[0])
    assert section == [
        LAST.format(id=ident), WORKING, DONE, BLOCKED, NEEDS_OLD + "; " + SEGMENT.format(id=ident),
    ]
    assert len(notion.edits) == 1 and notion.edits[0][1].startswith("## Ilan\n")


def test_an_ilan_section_whose_lines_repeat_another_persons_still_edits_only_the_ilan_section():
    same = OVERNIGHT_TEXT.replace(
        "\n".join(ilan_section(OVERNIGHT_TEXT)[:4]),
        "- **Last update:** \u2014\n- **Working on:** \u2014\n- **Done tonight:** \u2014\n- **Blocked on:** \u2014",
    )
    assert same.count("- **Working on:** \u2014") == 2
    notion = pages(overnight=same)

    answer, _ = go("--only", "overnight", client=notion)

    assert answer["targets"]["overnight_status"] == {"action": "posted"}
    assert outside_ilan(notion.text(OVERNIGHT)) == outside_ilan(same)
    assert ilan_section(notion.text(OVERNIGHT))[1] == WORKING


def test_a_second_run_changes_nothing_even_at_another_time_and_after_the_first_left_a_ledger_row():
    notion = pages()
    go("--only", "overnight", client=notion)
    page, edits = notion.text(OVERNIGHT), list(notion.edits)

    answer, code = go("--only", "overnight", client=notion, now=LATER)

    assert (code, answer["targets"]) == (0, {"overnight_status": {"action": "unchanged"}})
    assert notion.text(OVERNIGHT) == page
    assert notion.edits == edits


def test_a_board_snapshot_that_goes_stale_rewrites_the_last_update_once():
    notion = pages()
    since = "2026-10-10T04:00:00Z"                       # one window start for every call
    # the snapshot was taken at 15:45 and an hour is the limit; the last ledger row is at 16:30, so from 16:31 on
    # the facts stay the same and only the snapshot's age moves
    go("--only", "overnight", "--since", since, client=notion, now=NOW + dt.timedelta(minutes=31))
    first = ilan_section(notion.text(OVERNIGHT))
    fresh = first[0]

    stale_at = NOW + dt.timedelta(minutes=50)
    answer, code = go("--only", "overnight", "--since", since, client=notion, now=stale_at)
    later, _ = go("--only", "overnight", "--since", since, client=notion, now=stale_at + dt.timedelta(minutes=5))

    assert (code, answer["targets"]) == (0, {"overnight_status": {"action": "updated"}})
    assert "board snapshot from Sat 11:45 AM ET (digest" in fresh
    section = ilan_section(notion.text(OVERNIGHT))
    stale = section[0]
    assert stale.startswith(f"- **Last update:** {HEART} Ilan's agent {DOT} Sat 12:50 PM ET: lab digest,")
    assert "board snapshot from Sat 11:45 AM ET, stale (digest" in stale
    assert marker(stale) == answer["section_id"] != marker(fresh)
    assert section[1:4] == first[1:4]                    # Working on, Done tonight and Blocked on: the same facts
    assert later["targets"] == {"overnight_status": {"action": "unchanged"}}


def test_a_segment_someone_removed_comes_back_even_though_the_facts_did_not_change():
    notion = pages()
    go("--only", "overnight", client=notion)
    ident = marker(ilan_section(notion.text(OVERNIGHT))[0])
    notion.replace_text(OVERNIGHT, "; " + SEGMENT.format(id=ident), "")
    assert needs_line(notion.text(OVERNIGHT)) == NEEDS_OLD

    answer, _ = go("--only", "overnight", client=notion, now=LATER)

    assert answer["targets"]["overnight_status"] == {"action": "updated"}
    assert needs_line(notion.text(OVERNIGHT)).endswith(SEGMENT.format(id=ident).replace(SIGN, SIGN.replace("12:00", "12:05")))


def test_each_job_reads_only_its_own_page_and_the_plan_changes_need_no_digest(tmp_path):
    overnight_only, plans_only = pages(), pages()
    broken = tmp_path / "broken-state"
    (broken / "ledger.jsonl").mkdir(parents=True)

    go("--only", "overnight", client=overnight_only)
    answer, code = go("--only", "plan-changes", client=plans_only, state_dir=broken)

    assert {page for page in overnight_only.reads} == {OVERNIGHT}
    assert {page for page in plans_only.reads} == {TEAM}
    assert code == 0 and answer["targets"]["plan_changes"]["action"] == "posted"  # no digest, so the broken ledger is no obstacle
    assert "section_id" not in answer and "window" not in answer


def test_more_failed_rows_than_are_named_are_counted_and_the_rest_left_out():
    for number in range(8):
        add_row(task=f"P{number + 1}", outcome="error", ts=f"2026-10-10T15:4{number}:00Z")

    answer, _ = go("--print", "--only", "overnight")

    assert ("- **Blocked on:** L11 is Blocked; O41 held back, a blocker is missing from the board snapshot; "
            "GPU A teacher shared service: 1 queued; GPU B exclusive: R11 past its cap, 2 queued; 1 complaint open on the GPU queue; "
            "10 failed ledger rows in the window (P8 label error; P7 label error; P6 label error; and 7 more)") == answer["posts"][0]["lines"]["Blocked on"]


def test_more_tasks_in_progress_than_are_named_are_counted_and_the_rest_left_out(tmp_path):
    board = tmp_path / "board.json"
    ilan = ["user://fbcc0baf-9942-4661-9414-319688df6910"]
    board.write_text(json.dumps({
        "schema": 1, "taken_at": "2026-10-10T15:55:00Z", "gpu_queue": [],
        "tasks": [{"userDefined:ID": f"O{n}", "Task": f"O{n}: t", "Status": "In progress", "Tier": None, "Locked": "__YES__",
                   "Assignee": ilan, "Blocked by": []} for n in (30, 4, 12, 101, 7, 9, 21)],
    }), encoding="utf-8")

    answer, _ = go("--print", "--only", "overnight", board=board)

    assert answer["posts"][0]["lines"]["Working on"] == "- **Working on:** 7 In progress: O4, O7, O9, O12, O21, and 2 more"


def test_a_changed_digest_replaces_the_earlier_text_in_place_instead_of_adding_to_it():
    notion = pages()
    go("--only", "overnight", client=notion)
    first = ilan_section(notion.text(OVERNIGHT))
    add_row(task="O99", outcome="error")

    answer, _ = go("--only", "overnight", client=notion, now=LATER)

    assert answer["targets"]["overnight_status"] == {"action": "updated"}
    section = ilan_section(notion.text(OVERNIGHT))
    ident = marker(section[0])
    assert ident != marker(first[0])
    later_sign = SIGN.replace("12:00", "12:05")
    assert section == [
        LAST.format(id=ident).replace("Sat 12:00 AM ET to Sat 12:00 PM ET", "Sat 12:05 AM ET to Sat 12:05 PM ET").replace(SIGN, later_sign),
        WORKING,
        DONE,
        BLOCKED.replace("2 failed ledger rows in the window (L13 label error; L13 label fail)",
                        "3 failed ledger rows in the window (O99 label error; L13 label error; L13 label fail)"),
        NEEDS_OLD + "; " + SEGMENT.format(id=ident).replace(SIGN, later_sign),
    ]
    assert needs_line(notion.text(OVERNIGHT)).count("Lab digest:") == 1
    assert outside_ilan(notion.text(OVERNIGHT)) == outside_ilan(OVERNIGHT_TEXT)


def test_the_rulings_segment_and_the_people_text_on_the_needs_line_are_kept_whatever_their_order():
    notion = pages()
    rulings.mirror_rulings(open_rulings(), notion, NOW)
    with_rulings = needs_line(notion.text(OVERNIGHT))
    segment = rulings.build_posts(open_rulings(), NOW).segment
    assert with_rulings == NEEDS_OLD + "; " + segment

    go("--only", "overnight", client=notion)

    line = needs_line(notion.text(OVERNIGHT))
    ident = marker(ilan_section(notion.text(OVERNIGHT))[0])
    assert line == NEEDS_OLD + "; " + segment + "; " + SEGMENT.format(id=ident)


def test_both_flows_leave_each_others_segments_alone_in_either_order():
    notion = pages()
    go("--only", "overnight", client=notion)
    mine_first = needs_line(notion.text(OVERNIGHT))
    posts = rulings.build_posts(open_rulings(), NOW)

    posted = rulings.mirror_rulings(open_rulings(), notion, NOW)
    both = needs_line(notion.text(OVERNIGHT))
    again = rulings.mirror_rulings(open_rulings(), notion, LATER)
    mine_again, _ = go("--only", "overnight", client=notion, now=LATER)

    assert posted["targets"]["overnight_status"] == {"action": "posted"}
    assert both == mine_first + "; " + posts.segment
    assert again["targets"]["overnight_status"] == {"action": "unchanged"}
    assert mine_again["targets"]["overnight_status"] == {"action": "unchanged"}
    assert needs_line(notion.text(OVERNIGHT)) == both
    assert mine_again["rulings"] == {"waiting": 2, "segment": "current", "on_page": posts.list_id, "current": posts.list_id}


def test_when_every_ruling_is_answered_the_rulings_flow_removes_its_segment_and_ours_stays_whole():
    notion = pages()
    rulings.mirror_rulings(open_rulings(), notion, NOW)
    go("--only", "overnight", client=notion)
    ident = marker(ilan_section(notion.text(OVERNIGHT))[0])

    rulings.mirror_rulings([], notion, LATER)

    line = needs_line(notion.text(OVERNIGHT))
    assert "Lab rulings waiting on you" not in line
    assert line.startswith(NEEDS_OLD)
    assert line.endswith(SEGMENT.format(id=ident))
    assert line.count("Lab digest:") == 1


def test_the_answer_says_when_the_rulings_segment_is_missing_or_stale():
    notion = pages()
    missing, _ = go("--only", "overnight", client=notion)
    posts = rulings.build_posts(open_rulings(), NOW)
    rulings.mirror_rulings(open_rulings(), notion, NOW)
    current, _ = go("--only", "overnight", client=notion, now=LATER)
    notion.replace_text(OVERNIGHT, f"(list {posts.list_id})", "(list 00000000)")
    stale, _ = go("--only", "overnight", client=notion, now=LATER)

    assert missing["rulings"] == {"waiting": 2, "segment": "missing", "on_page": None, "current": posts.list_id,
                                  "next": "python -m lab rulings mirror"}
    assert current["rulings"]["segment"] == "current"
    assert stale["rulings"] == {"waiting": 2, "segment": "stale", "on_page": "00000000", "current": posts.list_id,
                                "next": "python -m lab rulings mirror"}
    assert "00000000" in needs_line(notion.text(OVERNIGHT))  # reported, not repaired: that segment is the rulings flow's


def test_a_rulings_segment_left_on_the_page_when_nothing_is_waiting_is_reported_stale(tmp_path):
    notion = pages()
    rulings.mirror_rulings(open_rulings(), notion, NOW)
    empty = tmp_path / "refine"
    shutil.copytree(REFINE, empty)
    for track in ("telemetry", "kernels"):
        text = (empty / track / "OWNER_RULINGS.md").read_text(encoding="utf-8")
        (empty / track / "OWNER_RULINGS.md").write_text(text.replace("Answer: ____", "Answer: yes (owner, 2026-10-10)"), encoding="utf-8")

    answer, _ = go("--only", "overnight", client=notion, refine=empty)

    assert answer["rulings"]["waiting"] == 0 and answer["rulings"]["segment"] == "stale"


def test_a_rulings_file_that_cannot_be_read_is_a_warning_and_the_segment_state_is_unknown(tmp_path):
    refine = tmp_path / "refine"
    shutil.copytree(REFINE, refine)
    (refine / "telemetry" / "OWNER_RULINGS.md").write_text("# Owner rulings\n\n## 1. A question without its fields (pass 1)\n", encoding="utf-8")
    notion = pages()

    answer, code = go("--only", "overnight", client=notion, refine=refine)

    assert (code, answer["ok"]) == (0, True)
    assert len(answer["warnings"]) == 1 and answer["warnings"][0].startswith("telemetry: ")
    assert any(problem.startswith("telemetry rulings unreadable") for problem in answer["digest_problems"])
    assert (answer["rulings"]["segment"], answer["rulings"]["current"]) == ("unknown", None)
    assert answer["targets"]["overnight_status"]["action"] == "posted"


def test_our_segment_goes_and_the_line_keeps_the_rest_when_there_is_nothing_left_to_say():
    notion = pages()
    go("--only", "overnight", client=notion)

    answer, _ = go("--only", "overnight", client=notion, board=quiet_board(Path(os.environ["LAB_STATE_DIR"])), now=LATER)

    assert answer["targets"]["overnight_status"] == {"action": "updated"}
    assert needs_line(notion.text(OVERNIGHT)) == NEEDS_OLD


def test_a_needs_line_that_only_had_our_segment_reads_none_when_it_goes():
    only_ours = OVERNIGHT_TEXT.replace(NEEDS_OLD, "- **Needs a human (Ilan):** \u2014")
    notion = pages(overnight=only_ours)
    go("--only", "overnight", client=notion)
    assert "Lab digest:" in needs_line(notion.text(OVERNIGHT))

    go("--only", "overnight", client=notion, board=quiet_board(Path(os.environ["LAB_STATE_DIR"])), now=LATER)

    assert needs_line(notion.text(OVERNIGHT)) == "- **Needs a human (Ilan):** none"


@pytest.mark.parametrize("body,glue", [("see Q2", "; "), ("see Q2.", " "), ("see Q2;", " ")])
def test_our_segment_goes_on_the_end_of_whatever_the_line_already_says(body, glue):
    line = f"- **Needs a human (Ilan):** {body}"
    new, action = mirror.rewrite_needs(line, "SEGMENT (digest 12345678)", "12345678")

    assert (new, action) == (f"{line}{glue}SEGMENT (digest 12345678)", "posted")


@pytest.mark.parametrize("empty", ["", "-", "\u2014", "none", "nothing", "None"])
def test_a_placeholder_needs_line_is_replaced_by_our_segment(empty):
    line = f"- **Needs a human (Ilan):** {empty}".rstrip()

    new, action = mirror.rewrite_needs(line, "SEGMENT (digest 12345678)", "12345678")

    assert (new, action) == ("- **Needs a human (Ilan):** SEGMENT (digest 12345678)", "posted")


def test_children_under_the_needs_bullet_stay_and_children_under_the_replaced_bullets_go():
    page = OVERNIGHT_TEXT.replace(
        "- **Working on:** Ilan's lab machine, the Orchestration tasks O1 to O41",
        "- **Working on:** Ilan's lab machine, the Orchestration tasks O1 to O41\n\t- an old detail\n\t\t- and a deeper one",
    ).replace(NEEDS_OLD, NEEDS_OLD + "\n\t- a detail of the question")
    notion = pages(overnight=page)

    go("--only", "overnight", client=notion)

    after = notion.text(OVERNIGHT)
    assert "an old detail" not in after and "and a deeper one" not in after
    section = ilan_section(after)
    assert section[-1] == "\t- a detail of the question"
    assert section[-2].startswith(NEEDS_OLD + "; ")


def test_a_page_with_other_text_inside_the_section_keeps_it_in_place():
    page = OVERNIGHT_TEXT.replace("- **Blocked on:** nothing on the lab side", "- **Blocked on:** nothing on the lab side\n- **Notes:** a note from a person")
    notion = pages(overnight=page)

    go("--only", "overnight", client=notion)

    section = ilan_section(notion.text(OVERNIGHT))
    assert section[4] == "- **Notes:** a note from a person"
    assert section[5].startswith(NEEDS_OLD)


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda text: text.replace("## Ilan\n", ""), "expected one '## Ilan' heading on the Overnight Status page, found 0"),
        (lambda text: text + "\n## Ilan\n- **Last update:** x", "expected one '## Ilan' heading on the Overnight Status page, found 2"),
        (lambda text: text.replace("- **Done tonight:** R2 merged (`bb351e6`, PR #37)\n", ""), "'Done tonight' found 0 times"),
        (lambda text: text.replace("- **Blocked on:** nothing on the lab side", "- **Blocked on:** a\n- **Blocked on:** b"), "'Blocked on' found 2 times"),
    ],
)
def test_a_page_without_the_expected_section_fails_that_target_and_edits_nothing(mutate, message):
    notion = pages(overnight=mutate(OVERNIGHT_TEXT))

    answer, code = go("--only", "overnight", client=notion)

    assert (code, answer["ok"], answer["outcome"]) == (1, False, "failed")
    assert message in answer["targets"]["overnight_status"]["error"]
    assert notion.edits == []


def test_a_write_that_does_not_show_up_on_the_page_is_reported_as_failed():
    notion = pages()
    notion.replace_text = lambda page_id, old, new: None  # Notion accepts the write and shows nothing

    answer, code = go("--only", "overnight", client=notion)

    assert code == 1
    assert answer["targets"]["overnight_status"] == {
        "action": "failed", "error": "the Overnight Status page does not show the change after the write"}


def test_plan_changes_that_do_not_show_up_on_the_page_are_reported_as_failed():
    notion = pages()
    notion.replace_text = lambda page_id, old, new: None  # Notion accepts the write and shows nothing

    answer, code = go("--only", "plan-changes", client=notion)

    assert code == 1
    assert answer["targets"]["plan_changes"] == {
        "action": "failed", "error": "Ilan's Team Comms page does not show the lines after the write"}


def test_a_notion_error_on_one_page_fails_that_target_only():
    notion = pages()
    notion.fail_next(1, status=500)

    answer, code = go(client=notion)

    assert code == 1 and answer["outcome"] == "failed"
    assert answer["targets"]["overnight_status"]["action"] == "failed"
    assert "500" in answer["targets"]["overnight_status"]["error"]
    assert answer["targets"]["plan_changes"]["action"] == "posted"
    assert len(plan_children(notion.text(TEAM))) == 9


def test_a_page_changed_by_someone_else_between_the_read_and_the_write_fails_and_the_next_run_recovers():
    notion = pages()

    def change_the_section_first(fake, kind, page_id):
        if kind == "replace" and page_id == OVERNIGHT:
            fake.on_call(lambda *_: None)
            fake.add_page(OVERNIGHT, fake.text(OVERNIGHT).replace("nothing on the lab side", "something else"))

    notion.on_call(change_the_section_first)
    first, code = go("--only", "overnight", client=notion)
    second, code2 = go("--only", "overnight", client=notion, now=LATER)

    assert code == 1 and first["targets"]["overnight_status"]["action"] == "failed"
    assert code2 == 0 and second["targets"]["overnight_status"]["action"] == "posted"
    assert ilan_section(notion.text(OVERNIGHT))[1] == WORKING


# ------------------------------------------------------------------ the plan changes on Ilan's Team Comms page


def test_with_no_plan_changes_item_the_first_run_creates_it_under_the_fyi_heading_with_every_change_signed():
    notion = pages()

    answer, code = go("--only", "plan-changes", client=notion)

    assert code == 0
    result = answer["targets"]["plan_changes"]
    assert (result["action"], result["posted"], result["already"], result["changes"], result["item"]) == ("posted", 9, 0, 9, "created")
    rows = lines(notion.text(TEAM))
    at = rows.index(FYI)
    assert rows[:at + 1] == lines(TEAM_TEXT)[:at + 1]
    assert rows[at + 1] == PLAN_ITEM
    children = rows[at + 2:at + 11]
    assert all(row.startswith(f"\t- {SIGN} ") for row in children)
    assert labels_of(children) == [f"{a}, {b}" for a, b in ALL_CHANGES]
    assert signed(E3_LINE) == children[5]
    assert signed(MERGE_LINE) == children[8]
    assert rows[at + 11:] == lines(TEAM_TEXT)[at + 1:]  # the other FYI items and the sections below are untouched
    assert len(notion.edits) == 1


def test_a_second_run_posts_each_change_once_and_touches_nothing():
    notion = pages()
    go("--only", "plan-changes", client=notion)
    page, edits = notion.text(TEAM), list(notion.edits)

    answer, code = go("--only", "plan-changes", client=notion, now=LATER)

    assert code == 0
    assert answer["targets"]["plan_changes"] == {"action": "unchanged", "posted": 0, "already": 9, "changes": 9}
    assert notion.text(TEAM) == page and notion.edits == edits


def test_new_lines_go_after_the_items_last_child_one_tab_deeper_and_the_old_lines_stay():
    existing = [signed("telemetry pass 1, T4: T4 on the board, Done when rewritten. Was: x.", f"{HEART} Ilan's agent {DOT} Sat 9:02 AM ET:"),
                signed("telemetry pass 1, T3: T3 on the board, Done when rewritten. Was: y.", f"{HEART} Ilan's agent {DOT} Sat 9:03 AM ET:")]
    team = TEAM_TEXT.replace(FYI, FYI + "\n" + PLAN_ITEM + "\n" + "\n".join(existing))
    notion = pages(team=team)

    answer, _ = go("--only", "plan-changes", client=notion)

    result = answer["targets"]["plan_changes"]
    assert (result["action"], result["posted"], result["already"], result["item"]) == ("posted", 7, 2, "existing")
    children = plan_children(notion.text(TEAM))
    assert children[:2] == existing
    assert labels_of(children[2:]) == [f"{a}, {b}" for a, b in ALL_CHANGES[2:]]
    assert all(row.startswith(f"\t- {SIGN} ") for row in children[2:])
    assert outside_plan_item(notion.text(TEAM)) == outside_plan_item(team)
    assert len(notion.edits) == 1 and notion.edits[0][1] == "\n".join(existing)  # anchored on the last two lines of the item


def test_an_item_without_children_gets_its_first_line_straight_after_it():
    team = TEAM_TEXT.replace(FYI, FYI + "\n" + PLAN_ITEM)
    notion = pages(team=team)

    answer, _ = go("--only", "plan-changes", client=notion)

    assert answer["targets"]["plan_changes"]["item"] == "existing"
    assert labels_of(plan_children(notion.text(TEAM))) == [f"{a}, {b}" for a, b in ALL_CHANGES]
    assert notion.edits[0][1] == PLAN_ITEM


def test_lines_a_pass_posted_in_its_own_words_cover_the_changes_with_the_same_label_and_task():
    own = [
        signed("telemetry pass 1, T4: Done when rewritten to define joules per answer. Was: \"old\". Why: because. Source: here."),
        signed("telemetry pass 1, T0: E6 queued. Was: no telemetry line. Why: no sensor lag measured. Source: there."),
        signed("kernels pass 3, E3: sweep rewritten. Was: \"old\".", f"{HEART} Ilan's agent {DOT} Sat 8:00 AM:"),
    ]
    team = TEAM_TEXT.replace(FYI, FYI + "\n" + PLAN_ITEM + "\n" + "\n".join(own))
    notion = pages(team=team)

    answer, _ = go("--only", "plan-changes", client=notion)

    result = answer["targets"]["plan_changes"]
    assert (result["posted"], result["already"]) == (6, 3)
    assert [line.split(" (change")[0] for line in result["lines"]] == [
        "telemetry pass 1, T3:", "telemetry pass 1, T2:", "telemetry pass 1, T1:", "kernels pass 3, E5:",
        "kernels pass 3, T0:", "merge 1, T3:"]
    assert plan_children(notion.text(TEAM))[:3] == own


def test_a_pass_that_posted_one_of_two_changes_to_a_task_leaves_only_the_later_change_to_post(tmp_path):
    refine = tmp_path / "refine"
    shutil.copytree(REFINE, refine)
    log = refine / "kernels" / "REFINE_LOG.md"
    log.write_text(log.read_text(encoding="utf-8").replace("2. E5 on T0, split into E5a and E5b (K2)", "2. E3 on T0, repeat count raised (K3)"),
                   encoding="utf-8")
    team = TEAM_TEXT.replace(FYI, FYI + "\n" + PLAN_ITEM + "\n" + signed("kernels pass 3, E3: first change. Was: x."))
    notion = pages(team=team)

    answer, _ = go("--only", "plan-changes", client=notion, refine=refine)

    posted = answer["targets"]["plan_changes"]["lines"]
    assert "kernels pass 3, E3: (change 2)" in posted
    assert "kernels pass 3, E3: (change 1)" not in posted
    assert sorted(labels_of(plan_children(notion.text(TEAM)))).count("kernels pass 3, E3") == 2


def test_a_reverted_change_line_does_not_stand_for_the_change():
    revert = signed("kernels pass 3, E3: reverted change 1. Was: \"new\". Why: the pass was invalid.")
    team = TEAM_TEXT.replace(FYI, FYI + "\n" + PLAN_ITEM + "\n" + revert)
    notion = pages(team=team)

    answer, _ = go("--only", "plan-changes", client=notion)

    assert answer["targets"]["plan_changes"]["posted"] == 9
    assert plan_children(notion.text(TEAM))[0] == revert


def test_a_change_recorded_later_is_added_at_the_end_signed_with_its_own_time_and_the_earlier_lines_keep_theirs(tmp_path):
    notion = pages()
    go("--only", "plan-changes", client=notion)
    first_nine = plan_children(notion.text(TEAM))
    refine = tmp_path / "refine"
    shutil.copytree(REFINE, refine)
    merge_log = refine / "MERGE_LOG.md"
    merge_log.write_text(
        merge_log.read_text(encoding="utf-8")
        + "\n## Merge 2, 2026-10-11\n\nA later round.\n\nChanges (1 of at most 10):\n"
        '1. L4 on the board, Tier set (M5)\n   BEFORE: "Tier: none"\n   AFTER:  "Tier: Opus"\n'
        "   SOURCE: Notion task L4\n   WHY: a gate is a hard build.\n\n---\n",
        encoding="utf-8",
    )

    answer, _ = go("--only", "plan-changes", client=notion, now=NOW + dt.timedelta(hours=1), refine=refine)

    children = plan_children(notion.text(TEAM))
    assert answer["targets"]["plan_changes"]["posted"] == 1
    assert children[:9] == first_nine
    assert children[9] == (f"\t- {HEART} Ilan's agent {DOT} Sat 1:00 PM ET: merge 2, L4: L4 on the board, Tier set. "
                           'Was: "Tier: none". Why: a gate is a hard build. Source: Notion task L4.')
    assert len(children) == 10


def test_markup_in_a_log_is_escaped_file_names_go_in_code_and_long_text_is_clipped(tmp_path):
    refine = tmp_path / "refine"
    refine.mkdir()
    (refine / "plumbing").mkdir()
    long_before = " ".join(f"word{i}" for i in range(400))
    (refine / "plumbing" / "REFINE_LOG.md").write_text(
        "# Refinement log for the plumbing track\n\n---\n\n## Pass 2, 2026-10-10\n\nA pass.\n\nChanges (2 of at most 5):\n"
        "1. R9 on the board, *Done when* [rewritten] (D1)\n"
        f"   BEFORE: \"{long_before}\"\n"
        "   AFTER:  \"new\"\n"
        "   SOURCE: lab/pod_a_teacher.sh, <section 3> | table\n"
        "   WHY: `a` span and a_b name.\n"
        "2. R10 on the board, renamed (D2)\n"
        "   BEFORE: \"keep `a` and lab/old_name.py, 5 * 3 < 7\"\n"
        "   AFTER:  \"new\"\n"
        "   SOURCE: s\n"
        "   WHY: w\n\n---\n", encoding="utf-8")

    changes, problems = mirror.collect_plan_changes(refine)

    assert problems == []
    assert 'Was: "keep `a` and `lab/old_name.py`, 5 \\* 3 \\< 7".' in changes[1].text
    text = changes[0].text
    assert text.startswith("plumbing pass 2, R9: R9 on the board, \\*Done when\\* \\[rewritten\\]. Was: \"word0 word1")
    was = re.search(r'Was: "(.*?)"\. Why:', text)[1]
    assert was.endswith(" ...") and mirror.CLIP_WAS - 12 <= len(was) <= mirror.CLIP_WAS + 4  # cut at a word, with the cut marked
    assert "Why: `a` span and `a_b` name." in text
    assert "Source: `lab/pod_a_teacher.sh`, \\<section 3\\> \\| table." in text
    assert len(text) < 1700


def test_the_task_of_a_change_is_the_first_task_id_in_its_heading_and_board_when_it_names_none(tmp_path):
    refine = tmp_path / "refine"
    (refine / "learning").mkdir(parents=True)
    (refine / "learning" / "REFINE_LOG.md").write_text(
        "# Refinement log for the learning track\n\n---\n\n## Pass 1, 2026-10-10\n\nA pass.\n\nChanges (3 of at most 5):\n"
        "1. Experiment queue on T0, E7 queued (LT1)\n   BEFORE: \"a\"\n   AFTER:  \"b\"\n   SOURCE: s\n   WHY: w\n"
        "2. The rubric wording tightened (LT2)\n   BEFORE: \"a\"\n   AFTER:  \"b\"\n   SOURCE: s\n   WHY: w\n"
        "3. L3 split into L3a and L3b (LT3)\n   BEFORE: \"a\"\n   AFTER:  \"b\"\n   SOURCE: s\n   WHY: w\n\n---\n", encoding="utf-8")

    changes, problems = mirror.collect_plan_changes(refine)

    assert problems == []
    assert [(change.label, change.task, change.number) for change in changes] == [
        ("learning pass 1", "T0", 1), ("learning pass 1", "board", 2), ("learning pass 1", "L3", 3)]
    assert changes[1].text == 'learning pass 1, board: The rubric wording tightened. Was: "a". Why: w. Source: s.'


def test_a_log_that_cannot_be_read_is_a_warning_and_the_other_logs_are_still_posted(tmp_path):
    refine = tmp_path / "refine"
    shutil.copytree(REFINE, refine)
    (refine / "telemetry" / "REFINE_LOG.md").write_text("# Log\n\n## Whatever\n", encoding="utf-8")
    notion = pages()

    answer, code = go("--only", "plan-changes", client=notion, refine=refine)

    assert code == 0 and answer["ok"] is True
    assert answer["targets"]["plan_changes"]["posted"] == 4
    assert len(answer["warnings"]) == 1 and answer["warnings"][0].startswith("telemetry/REFINE_LOG.md:")


def test_with_no_changes_anywhere_nothing_is_created(tmp_path):
    empty = tmp_path / "refine"
    (empty / "telemetry").mkdir(parents=True)
    (empty / "telemetry" / "REFINE_LOG.md").write_text("# Refinement log for the telemetry track\n\n---\n", encoding="utf-8")
    notion = pages()

    answer, code = go("--only", "plan-changes", client=notion, refine=empty)

    assert code == 0
    assert answer["targets"]["plan_changes"] == {"action": "unchanged", "posted": 0, "already": 0, "changes": 0}
    assert notion.text(TEAM) == TEAM_TEXT and notion.edits == []


def test_a_team_comms_page_without_an_fyi_heading_fails_that_target_and_the_other_is_still_done():
    notion = pages(team=TEAM_TEXT.replace(FYI, "## Notes"))

    answer, code = go(client=notion)

    assert code == 1
    assert answer["targets"]["plan_changes"] == {"action": "failed", "error": "Ilan's Team Comms page has no '## FYI' heading"}
    assert answer["targets"]["overnight_status"]["action"] == "posted"


def test_the_plan_changes_item_and_the_rulings_post_share_the_fyi_section_without_touching_each_other():
    notion = pages()
    go("--only", "plan-changes", client=notion)
    rulings.mirror_rulings(open_rulings(), notion, NOW)
    page = notion.text(TEAM)
    block = rulings.build_posts(open_rulings(), NOW).block

    again = rulings.mirror_rulings(open_rulings(), notion, LATER)
    answer, _ = go("--only", "plan-changes", client=notion, now=LATER)

    assert block in page and PLAN_ITEM in page
    assert again["targets"]["team_comms_fyi"] == {"action": "unchanged"}
    assert answer["targets"]["plan_changes"]["action"] == "unchanged"
    assert notion.text(TEAM) == page
    assert len(plan_children(page)) == 9


# ------------------------------------------------------------------ print mode and the missing token


def forbidden_transport(*args):
    raise AssertionError("the network must not be touched")


def test_print_mode_touches_no_client_and_returns_the_texts_and_where_they_go():
    answer, code = go("--print", environ={"NOTION_TOKEN": TOKEN}, transport=forbidden_transport)

    assert (code, answer["ok"], answer["outcome"], answer["mode"]) == (0, True, "printed", "print")
    overnight, plans = answer["posts"]
    ident = answer["section_id"]
    assert (overnight["target"], overnight["action"], overnight["page_url"]) == ("overnight_status", "replace_lines", mirror.OVERNIGHT_URL)
    assert overnight["lines"]["Working on"] == WORKING
    assert overnight["needs_segment"] == SEGMENT.format(id=ident)
    assert overnight["skip_if_page_shows"].startswith(f"(digest {ident})")
    assert "Lab rulings waiting on you" in overnight["keep"]
    assert (plans["target"], plans["action"], plans["page_url"], plans["item"]) == ("plan_changes", "append_lines", mirror.TEAM_COMMS_URL, PLAN_ITEM)
    assert [item["covers"] for item in plans["lines"]] == [f"{a}, {b}:" for a, b in ALL_CHANGES]
    assert plans["lines"][5]["text"] == f"- {SIGN} {E3_LINE}"
    assert answer["rulings"]["segment"] == "unknown" and answer["rulings"]["waiting"] == 2


def test_print_mode_reads_and_writes_nothing_even_when_a_client_is_at_hand():
    notion = pages()

    answer, code = go("--print", client=notion)

    assert (code, answer["outcome"], answer["mode"]) == (0, "printed", "print")
    assert notion.reads == [] and notion.edits == [] and notion.requests == []
    assert notion.text(OVERNIGHT) == OVERNIGHT_TEXT and notion.text(TEAM) == TEAM_TEXT


def test_the_token_is_not_in_the_printed_texts_even_when_the_environment_has_one():
    answer, _ = go("--print", environ={"NOTION_TOKEN": TOKEN}, transport=forbidden_transport)

    assert TOKEN not in json.dumps(answer)


def test_the_texts_a_session_would_write_are_the_ones_the_live_run_posts():
    printed, _ = go("--print")
    notion = pages()
    go(client=notion)

    section = ilan_section(notion.text(OVERNIGHT))
    overnight = printed["posts"][0]
    assert section[:4] == list(overnight["lines"].values())
    assert section[4].endswith("; " + overnight["needs_segment"])
    live_lines = [row.removeprefix("\t") for row in plan_children(notion.text(TEAM))]
    assert live_lines == [item["text"] for item in printed["posts"][1]["lines"]]


def test_print_mode_for_one_job_returns_that_job_only():
    only_plans, _ = go("--print", "--only", "plan-changes")
    only_overnight, _ = go("--print", "--only", "overnight")

    assert [post["target"] for post in only_plans["posts"]] == ["plan_changes"]
    assert [post["target"] for post in only_overnight["posts"]] == ["overnight_status"]
    assert "section_id" not in only_plans and "window" not in only_plans


def test_with_nothing_to_say_the_print_plan_tells_the_session_to_remove_an_earlier_segment():
    answer, _ = go("--print", "--only", "overnight", board=quiet_board(Path(os.environ["LAB_STATE_DIR"])))

    post = answer["posts"][0]
    assert post["needs_segment"] == ""
    assert post["needs_how"].startswith("remove an earlier segment of this flow") and "write 'none' there" in post["needs_how"]


def test_without_a_token_nothing_is_sent_and_the_texts_come_back_with_exit_code_2():
    answer, code = go(transport=forbidden_transport)

    assert (code, answer["ok"], answer["outcome"], answer["mode"]) == (2, False, "no_token", "print")
    assert "NOTION_TOKEN is not set" in answer["error"]
    printed, _ = go("--print")
    assert answer["posts"] == printed["posts"]
    assert ledger_rows()[-1]["outcome"] == "printed"
    assert [row["outcome"] for row in ledger_rows()[-2:]] == ["no_token", "printed"]


def test_a_blank_token_counts_as_missing(tmp_path):
    env = tmp_path / ".env"
    env.write_text("NOTION_TOKEN=\n", encoding="utf-8")

    answer, code = go(env_files=[env], transport=forbidden_transport)

    assert (code, answer["outcome"]) == (2, "no_token")


def test_the_token_is_read_from_the_environment_then_from_env_files(tmp_path):
    env = tmp_path / ".env"
    env.write_text(f"OTHER=1\nNOTION_TOKEN={TOKEN}\n", encoding="utf-8")
    from_file = pages()
    from_environment = pages()

    first, code_a = go("--only", "overnight", env_files=[env], transport=from_file.transport)
    second, code_b = go("--only", "overnight", environ={"NOTION_TOKEN": TOKEN}, env_files=[], transport=from_environment.transport)

    assert (code_a, code_b) == (0, 0)
    assert first["targets"] == second["targets"] == {"overnight_status": {"action": "posted"}}
    assert ilan_section(from_file.text(OVERNIGHT))[1] == WORKING


def test_the_token_never_reaches_the_answer_the_ledger_or_an_error_message(state):
    def leaky(method, url, headers, body, timeout):
        raise OSError(f"connection to {url} failed with the header {headers['Authorization']} and the token {TOKEN}")

    answer, code = go(environ={"NOTION_TOKEN": TOKEN}, transport=leaky)

    assert code == 1
    assert answer["targets"]["overnight_status"]["action"] == "failed"
    assert answer["targets"]["plan_changes"]["action"] == "failed"
    everything = json.dumps(answer) + (state / "ledger.jsonl").read_text(encoding="utf-8")
    assert TOKEN not in everything
    assert "***" in answer["targets"]["overnight_status"]["error"]


# ------------------------------------------------------------------ the real REST client over the fake wire


def test_the_rest_client_speaks_the_page_markdown_api_and_the_mirror_works_through_it():
    notion = pages()

    answer, code = go(environ={"NOTION_TOKEN": TOKEN}, transport=notion.transport)

    assert (code, answer["outcome"]) == (0, "ok")
    methods = [(method, path.split("/")[3]) for method, path, _ in notion.requests]
    assert methods == [
        ("GET", OVERNIGHT), ("PATCH", OVERNIGHT), ("GET", OVERNIGHT),
        ("GET", TEAM), ("PATCH", TEAM), ("GET", TEAM),
    ]
    patch = notion.requests[1][2]
    assert patch["type"] == "update_content"
    (update,) = patch["update_content"]["content_updates"]
    assert update["old_str"].startswith("## Ilan\n- **Last update:** ") and update["new_str"].startswith("## Ilan\n- **Last update:** ")
    assert ilan_section(notion.text(OVERNIGHT))[1] == WORKING
    assert len(plan_children(notion.text(TEAM))) == 9


def test_a_second_run_through_the_rest_client_sends_no_patch():
    notion = pages()
    go(environ={"NOTION_TOKEN": TOKEN}, transport=notion.transport)
    sent = len(notion.requests)

    answer, _ = go(environ={"NOTION_TOKEN": TOKEN}, transport=notion.transport, now=LATER)

    assert [method for method, _, _ in notion.requests[sent:]] == ["GET", "GET"]
    assert answer["targets"]["overnight_status"]["action"] == "unchanged"
    assert answer["targets"]["plan_changes"]["action"] == "unchanged"


def test_a_page_the_api_returns_in_part_is_refused_before_anything_is_written():
    notion = pages()
    notion.set_truncated(OVERNIGHT)
    notion.set_truncated(TEAM)

    answer, code = go(environ={"NOTION_TOKEN": TOKEN}, transport=notion.transport)

    assert code == 1
    for target in ("overnight_status", "plan_changes"):
        assert answer["targets"][target]["action"] == "failed"
        assert "only part of the page" in answer["targets"][target]["error"]
    assert [method for method, _, _ in notion.requests] == ["GET", "GET"]


def test_a_wrong_token_is_reported_as_the_notion_error_it_is():
    notion = pages()

    answer, code = go(environ={"NOTION_TOKEN": "not-the-token-0123456789"}, transport=notion.transport)

    assert code == 1
    assert "Notion answered 401" in answer["targets"]["overnight_status"]["error"]
    assert "not-the-token" not in json.dumps(answer)


# ------------------------------------------------------------------ the ledger row and the verb through the dispatcher


def test_every_call_that_gets_as_far_as_a_decision_leaves_one_ledger_row(state):
    notion = pages()
    before = len(ledger_rows())

    go(client=notion)
    go("--print")
    go(transport=forbidden_transport)
    go(client=notion, now=LATER)

    rows = ledger_rows()[before:]
    assert [(row["outcome"], row.get("mode")) for row in rows] == [("ok", "live"), ("printed", "print"), ("no_token", "print"), ("ok", "live")]
    first = rows[0]
    assert (first["verb"], first["overnight_status"], first["plan_changes"], first["plan_lines_posted"]) == ("mirror", "posted", "posted", 9)
    assert re.fullmatch(r"[0-9a-f]{8}", first["section_id"])
    assert (rows[3]["overnight_status"], rows[3]["plan_changes"]) == ("unchanged", "unchanged")
    assert (first["actor"], first["tier"], first["task"]) == ("unknown", None, None)


def test_the_row_carries_the_sittings_identity(monkeypatch):
    monkeypatch.setenv("LAB_ACTOR", "fable-1")
    monkeypatch.setenv("LAB_TIER", "Fable")
    monkeypatch.setenv("LAB_TASK", "O41")

    answer, _ = mirror.run(["--print", "--board", str(BOARD), "--refine-dir", str(REFINE)], now=NOW, environ=dict(os.environ), env_files=[])

    row = ledger_rows()[-1]
    assert (row["actor"], row["tier"], row["task"], row["seq"]) == ("fable-1", "fable", "O41", answer["ledger_seq"])


def test_a_ledger_that_cannot_be_written_never_undoes_what_notion_holds(monkeypatch):
    def refused(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(record, "record", refused)
    notion = pages()

    answer, code = go(client=notion)

    assert code == 0 and answer["ok"] is True
    assert answer["ledger_seq"] is None and answer["ledger_error"] == "OSError: disk full"
    assert ilan_section(notion.text(OVERNIGHT))[1] == WORKING


def test_a_ledger_that_cannot_be_opened_stops_the_digest_with_an_error_answer(tmp_path):
    broken = tmp_path / "broken-state"
    (broken / "ledger.jsonl").mkdir(parents=True)  # a folder where the ledger file should be

    answer, code = go("--print", "--only", "overnight", state_dir=broken)

    assert (code, answer["ok"], answer["outcome"]) == (1, False, "error")
    assert answer["error"].startswith("cannot compute the digest:")


def run_verb(capsys, *argv: str):
    code = lab_cli.main(["mirror", *argv])
    out, err = capsys.readouterr()
    return code, out, err


def test_the_verb_is_registered_alphabetically_and_listed_in_help(capsys):
    names = list(lab_cli.VERBS)
    assert lab_cli.VERBS["mirror"] == "lab.machine.mirror:main"
    assert names.index("metrics") < names.index("mirror") < names.index("ready")  # the registry stays alphabetical

    assert lab_cli.main(["--help"]) == 0
    listed = [row for row in capsys.readouterr().out.splitlines() if row.strip().startswith("mirror ")]
    assert len(listed) == 1
    assert re.fullmatch(r"\s+mirror\s+Mirror the lab digest to Overnight Status and the refinement plan changes to Ilan's Team Comms page\.", listed[0])


def test_the_dispatcher_prints_one_json_line_and_exits_0_in_print_mode(capsys):
    code, out, err = run_verb(capsys, "--print", "--board", str(BOARD), "--refine-dir", str(REFINE))

    assert code == 0 and err == ""
    (line,) = out.splitlines()
    assert json.loads(line)["outcome"] == "printed"
    assert out.isascii()


def test_the_dispatcher_exits_2_with_the_texts_when_there_is_no_token(capsys):
    code, out, _ = run_verb(capsys, "--board", str(BOARD), "--refine-dir", str(REFINE))

    answer = json.loads(out)
    assert code == 2 and answer["outcome"] == "no_token" and len(answer["posts"]) == 2


def test_the_dispatcher_exits_2_for_bad_arguments(capsys):
    code, out, err = run_verb(capsys, "--only", "nowhere")

    assert code == 2 and out == "" and "usage" in err.lower()


def test_while_the_stop_file_exists_the_verb_is_refused_before_it_runs(state, capsys):
    (state / "STOP").write_text("hold everything\n", encoding="utf-8")
    before = ledger.read()

    code, out, _ = run_verb(capsys, "--print", "--board", str(BOARD), "--refine-dir", str(REFINE))

    refused = json.loads(out)
    assert code == 1 and refused["outcome"] == "stopped" and refused["reason"] == "hold everything"
    assert ledger.read() == before


def test_called_without_the_dispatcher_the_verb_still_honors_the_stop_file(state, capsys):
    (state / "STOP").write_text("hold everything\n", encoding="utf-8")

    with pytest.raises(SystemExit) as refused:
        mirror.main(["--print"])

    assert refused.value.code == 1
    assert json.loads(capsys.readouterr().out)["outcome"] == "stopped"


def test_python_dash_m_lab_mirror_runs_from_the_repo_root(state):
    env = {**os.environ, "LAB_STATE_DIR": str(state), "LAB_ENV_FILES": ""}
    env.pop("NOTION_TOKEN", None)

    done = subprocess.run(
        [sys.executable, "-m", "lab", "mirror", "--print", "--board", str(BOARD), "--refine-dir", str(REFINE)],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=120, check=False,
    )

    assert done.returncode == 0, done.stderr
    answer = json.loads(done.stdout)
    assert answer["outcome"] == "printed" and answer["posts"][0]["lines"]["Working on"] == WORKING


# ------------------------------------------------------------------ the in-memory pages behave like the API


def test_the_fake_pages_refuse_a_replacement_unless_the_text_occurs_exactly_once():
    notion = FakePages({OVERNIGHT: "a\nb\na"})

    with pytest.raises(rulings.NotionError, match="matched nothing"):
        notion.replace_text(OVERNIGHT, "zzz", "y")
    with pytest.raises(rulings.NotionError, match="matched 2 places"):
        notion.replace_text(OVERNIGHT, "a", "y")
    notion.replace_text(OVERNIGHT, "b", "y")

    assert notion.text(OVERNIGHT) == "a\ny\na" and notion.edits == [(OVERNIGHT, "b", "y")]


def test_the_fake_wire_checks_the_token_the_version_and_the_page():
    notion = FakePages({OVERNIGHT: "text"})
    url = f"https://api.notion.com/v1/pages/{OVERNIGHT}/markdown"
    good = {"Authorization": f"Bearer {TOKEN}", "Notion-Version": pages_fake.API_VERSION}

    assert notion.transport("GET", url, {**good, "Authorization": "Bearer nope"}, None, 5)[0] == 401
    assert notion.transport("GET", url, {**good, "Notion-Version": "2025-09-03"}, None, 5)[0] == 400
    assert notion.transport("GET", url.replace(OVERNIGHT, TEAM), good, None, 5)[0] == 404
    status, body = notion.transport("GET", url, good, None, 5)
    assert (status, body["markdown"], body["truncated"], body["unknown_block_ids"]) == (200, "text", False, [])
    patch = json.dumps({"type": "update_content", "update_content": {"content_updates": [{"old_str": "text", "new_str": "more"}]}}).encode()
    assert notion.transport("PATCH", url, good, patch, 5)[0] == 200 and notion.text(OVERNIGHT) == "more"
    nothing = json.dumps({"type": "update_content", "update_content": {"content_updates": [{"old_str": "gone", "new_str": "x"}]}}).encode()
    assert notion.transport("PATCH", url, good, nothing, 5)[0] == 400
    assert all(token not in json.dumps(request) for request in notion.requests for token in (TOKEN,))
