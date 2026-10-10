"""Tests for the handoff verb (python -m lab handoff): the owner's note and the ledger row.

Every test gets its own state dir through LAB_STATE_DIR, so none of them writes the real lab/state.
"""

from __future__ import annotations

import datetime as dt
import json
import multiprocessing
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from lab import __main__ as lab_cli
from lab.machine import allowed, handoff, ledger, stop
from lab.machine.lock import FileLock

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS = REPO_ROOT / "lab" / "machine" / "prompts"
UTC = dt.timezone.utc
NOW = dt.datetime(2026, 10, 10, 14, 2, 3, tzinfo=UTC)
TIERS = ("fable", "opus", "sonnet")
SPAWN = multiprocessing.get_context("spawn")
JOIN_TIMEOUT = 300.0     # seconds; a spawned child still running after this has hung
START_TIMEOUT = 60.0     # seconds a spawned child waits for its partners to start
SITTINGS = 5
HANDOFFS_PER_SITTING = 30


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """A private state dir for every test, and no actor or tier from the environment."""
    state = tmp_path / "state"
    monkeypatch.setenv("LAB_STATE_DIR", str(state))
    monkeypatch.delenv("LAB_ACTOR", raising=False)
    monkeypatch.delenv("LAB_TIER", raising=False)
    return state


def notes_file(state: Path, day: str = "2026-10-10") -> bytes:
    return (state / "notes" / f"{day}.md").read_bytes()


def note_files(state: Path) -> list[str]:
    return sorted(path.name for path in (state / "notes").iterdir())


def aware(ts: str) -> dt.datetime:
    """A ledger ts (ISO 8601 text ending in Z) as an aware time."""
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))


def run_cli(argv, capsys):
    code = lab_cli.main(argv)
    out, err = capsys.readouterr()
    return code, out, err


def one_json_line(text: str) -> dict:
    lines = text.splitlines()
    assert len(lines) == 1, f"expected exactly one stdout line, got {lines!r}"
    return json.loads(lines[0])


# --- the note file ----------------------------------------------------------------------

def test_the_note_is_appended_under_a_heading_with_the_time_and_the_sitting_id(isolated_state):
    handoff.handoff("s-12", "Opened PR 61.\nLeft O19 open.", now=NOW)
    assert note_files(isolated_state) == ["2026-10-10.md"]
    assert notes_file(isolated_state) == b"## 14:02:03Z, sitting s-12\n\nOpened PR 61.\nLeft O19 open.\n\n"


def test_a_second_note_the_same_day_is_appended_after_the_first(isolated_state):
    handoff.handoff("s-12", "first note", now=NOW)
    handoff.handoff("s-13", "second note", now=NOW.replace(hour=15, minute=30, second=0))
    assert note_files(isolated_state) == ["2026-10-10.md"]
    assert notes_file(isolated_state) == (
        b"## 14:02:03Z, sitting s-12\n\nfirst note\n\n"
        b"## 15:30:00Z, sitting s-13\n\nsecond note\n\n"
    )


def test_each_utc_day_has_its_own_file(isolated_state):
    handoff.handoff("s-1", "late", now=dt.datetime(2026, 10, 10, 23, 59, 59, tzinfo=UTC))
    handoff.handoff("s-2", "early", now=dt.datetime(2026, 10, 11, 0, 0, 1, tzinfo=UTC))
    assert note_files(isolated_state) == ["2026-10-10.md", "2026-10-11.md"]
    assert notes_file(isolated_state, "2026-10-10") == b"## 23:59:59Z, sitting s-1\n\nlate\n\n"
    assert notes_file(isolated_state, "2026-10-11") == b"## 00:00:01Z, sitting s-2\n\nearly\n\n"


def test_a_time_with_another_utc_offset_is_filed_by_its_utc_day_and_headed_in_utc(isolated_state):
    eastern = dt.timezone(dt.timedelta(hours=-4))
    result = handoff.handoff("s-1", "evening", now=dt.datetime(2026, 10, 10, 22, 30, 0, tzinfo=eastern))
    assert result["day"] == "2026-10-11"
    assert notes_file(isolated_state, "2026-10-11") == b"## 02:30:00Z, sitting s-1\n\nevening\n\n"


def test_without_a_time_the_note_is_stamped_with_the_current_utc_time(isolated_state):
    before = dt.datetime.now(UTC).replace(microsecond=0)
    result = handoff.handoff("s-1", "now")
    after = dt.datetime.now(UTC)
    text = notes_file(isolated_state, result["day"]).decode("utf-8")
    found = re.fullmatch(r"## (\d\d:\d\d:\d\d)Z, sitting s-1\n\nnow\n\n", text)
    assert found, text
    stamped = dt.datetime.fromisoformat(f"{result['day']}T{found.group(1)}+00:00")
    assert before <= stamped <= after


def test_only_trailing_whitespace_is_trimmed_from_the_note(isolated_state):
    handoff.handoff("s-1", "  indented\n\n\tsecond line  \n\n\n", now=NOW)
    assert notes_file(isolated_state) == b"## 14:02:03Z, sitting s-1\n\n  indented\n\n\tsecond line\n\n"


def test_an_empty_note_still_leaves_its_heading_and_counts_zero_characters(isolated_state):
    result = handoff.handoff("s-1", "", now=NOW)
    assert result["chars"] == 0
    assert notes_file(isolated_state) == b"## 14:02:03Z, sitting s-1\n\n"
    assert ledger.read()[0]["note_chars"] == 0


def test_the_note_is_utf8_and_gets_no_carriage_returns_from_the_platform(isolated_state):
    handoff.handoff("s-1", "caf\u00e9 \u2713\nnext line", now=NOW)
    raw = notes_file(isolated_state)
    assert raw == "## 14:02:03Z, sitting s-1\n\ncaf\u00e9 \u2713\nnext line\n\n".encode("utf-8")
    assert b"\r" not in raw


def test_a_character_utf8_cannot_hold_is_replaced_instead_of_losing_the_note(isolated_state):
    result = handoff.handoff("s-1", "before \ud800 after", now=NOW)
    assert result["note_written"] is True
    assert notes_file(isolated_state).decode("utf-8") == "## 14:02:03Z, sitting s-1\n\nbefore ? after\n\n"


def test_an_unfinished_last_entry_is_closed_before_the_next_one_begins(isolated_state):
    (isolated_state / "notes").mkdir(parents=True)
    (isolated_state / "notes" / "2026-10-10.md").write_bytes(b"## 13:00:00Z, sitting old\n\nhalf a no")
    handoff.handoff("s-1", "whole note", now=NOW)
    assert notes_file(isolated_state) == (
        b"## 13:00:00Z, sitting old\n\nhalf a no\n"
        b"## 14:02:03Z, sitting s-1\n\nwhole note\n\n"
    )


def test_the_sitting_id_stays_on_one_line_of_the_heading(isolated_state):
    result = handoff.handoff("  s-1\nsitting   evil  ", "x", now=NOW)
    assert result["sitting"] == "s-1 sitting evil"
    assert notes_file(isolated_state) == b"## 14:02:03Z, sitting s-1 sitting evil\n\nx\n\n"


@pytest.mark.parametrize("sitting", ["", "   ", "\n\t"])
def test_a_sitting_id_with_nothing_in_it_is_refused_before_anything_is_written(isolated_state, sitting):
    with pytest.raises(ValueError, match="sitting"):
        handoff.handoff(sitting, "note", now=NOW)
    assert not isolated_state.exists()


def test_a_time_without_a_utc_offset_is_refused_before_anything_is_written(isolated_state):
    with pytest.raises(ValueError, match="UTC offset"):
        handoff.handoff("s-1", "note", now=dt.datetime(2026, 10, 10, 14, 2, 3))
    assert not isolated_state.exists()


def test_notes_path_names_the_days_file_in_the_notes_folder_of_the_state_dir(tmp_path):
    assert handoff.notes_path("2026-10-10", state_dir=tmp_path) == tmp_path / "notes" / "2026-10-10.md"
    assert handoff.notes_path("2026-10-10") == Path(os.environ["LAB_STATE_DIR"]) / "notes" / "2026-10-10.md"


def test_a_handoff_leaves_only_the_notes_the_ledger_and_their_locks_in_the_state_dir(isolated_state):
    handoff.handoff("s-1", "x", now=NOW)
    assert sorted(path.name for path in isolated_state.iterdir()) == [
        "ledger.jsonl", "ledger.lock", "notes", "notes.lock"]
    assert note_files(isolated_state) == ["2026-10-10.md"]


# --- the ledger row ---------------------------------------------------------------------

def test_one_ledger_row_says_handoff_and_holds_no_note_text(isolated_state, monkeypatch):
    monkeypatch.setenv("LAB_ACTOR", "opus-7")
    monkeypatch.setenv("LAB_TIER", "Opus")
    note = "KEEPOUT-4711 the PR is open; O19 still holds the lock"
    before = dt.datetime.now(UTC).replace(microsecond=0)
    handoff.handoff("s-12", note, now=NOW)
    after = dt.datetime.now(UTC)
    rows = ledger.read()
    assert len(rows) == 1
    assert before <= aware(rows[0].pop("ts")) <= after
    assert rows[0] == {
        "seq": 1, "actor": "opus-7", "tier": "opus", "task": None, "verb": "handoff", "outcome": "handoff",
        "metrics": {}, "refs": {"pr": None, "commit": None, "slot": None, "run_id": None},
        "sitting": "s-12", "note_chars": len(note), "note_written": True,
    }
    assert "KEEPOUT-4711" not in (isolated_state / "ledger.jsonl").read_text(encoding="utf-8")


def test_the_actor_is_the_sitting_and_the_tier_empty_when_the_environment_names_neither(isolated_state, monkeypatch):
    handoff.handoff("s-12", "x", now=NOW)
    monkeypatch.setenv("LAB_ACTOR", "  ")
    monkeypatch.setenv("LAB_TIER", "")
    handoff.handoff("s-13", "y", now=NOW)
    assert [(row["actor"], row["tier"]) for row in ledger.read()] == [("s-12", None), ("s-13", None)]


def test_an_actor_and_a_tier_passed_in_win_over_the_environment(isolated_state, monkeypatch):
    monkeypatch.setenv("LAB_ACTOR", "from-env")
    monkeypatch.setenv("LAB_TIER", "sonnet")
    handoff.handoff("s-1", "x", actor="from-caller", tier="fable", now=NOW)
    assert [(row["actor"], row["tier"]) for row in ledger.read()] == [("from-caller", "fable")]


def test_the_row_follows_the_rows_already_in_the_ledger(isolated_state):
    for verb in ("ready", "slot.start"):
        ledger.append({"actor": "a", "tier": None, "task": None, "verb": verb, "outcome": "ok",
                       "metrics": {}, "refs": {}})
    result = handoff.handoff("s-1", "x", now=NOW)
    assert result["ledger_seq"] == 3
    assert [(row["seq"], row["verb"]) for row in ledger.read()] == [(1, "ready"), (2, "slot.start"), (3, "handoff")]


def test_the_answer_has_the_day_and_the_length_but_neither_the_text_nor_a_path(isolated_state):
    note = "KEEPOUT-4711 an account of the sitting"
    result = handoff.handoff("s-12", note, now=NOW)
    assert result == {
        "ok": True, "outcome": "handoff", "sitting": "s-12", "day": "2026-10-10", "chars": len(note),
        "note_written": True, "ledger_seq": 1, "session_ends": True,
    }
    text = json.dumps(result)
    assert "KEEPOUT-4711" not in text and "notes" not in text and str(isolated_state) not in text


# --- through the dispatcher -------------------------------------------------------------

def test_the_verb_is_registered_and_listed_in_the_help(capsys):
    assert lab_cli.VERBS["handoff"] == "lab.machine.handoff:main"
    code, out, _ = run_cli(["--help"], capsys)
    assert code == 0
    assert re.search(r"^  handoff +End the session: leave the owner a note", out, re.MULTILINE), out


def test_handoff_answers_one_json_line_and_exit_code_0(isolated_state, capsys, monkeypatch):
    monkeypatch.setenv("LAB_ACTOR", "sonnet-3")
    monkeypatch.setenv("LAB_TIER", "sonnet")
    code, out, err = run_cli(["handoff", "--sitting", "s-9", "--note", "did the task"], capsys)
    assert code == 0
    answer = one_json_line(out)
    assert answer == {
        "ok": True, "outcome": "handoff", "sitting": "s-9", "day": answer["day"], "chars": len("did the task"),
        "note_written": True, "ledger_seq": 1, "session_ends": True,
    }
    assert err == ""
    assert note_files(isolated_state) == [f"{answer['day']}.md"]
    text = notes_file(isolated_state, answer["day"]).decode("utf-8")
    assert re.fullmatch(r"## \d\d:\d\d:\d\dZ, sitting s-9\n\ndid the task\n\n", text), text
    row = ledger.read()[0]
    assert (row["actor"], row["tier"], row["verb"], row["outcome"], row["sitting"]) == (
        "sonnet-3", "sonnet", "handoff", "handoff", "s-9")


def test_python_dash_m_lab_handoff_works_as_a_real_process(isolated_state):
    note = "KEEPOUT-4711 opened 'the' PR, then left; nothing else | open: none"
    done = subprocess.run(
        [sys.executable, "-m", "lab", "handoff", "--sitting", "s-9", "--note", note],
        cwd=REPO_ROOT, env=dict(os.environ), capture_output=True, text=True, timeout=120,
    )
    assert done.returncode == 0, done.stderr
    answer = one_json_line(done.stdout)
    assert (answer["ok"], answer["outcome"], answer["note_written"], answer["ledger_seq"]) == (True, "handoff", True, 1)
    assert done.stderr == ""
    assert notes_file(isolated_state, answer["day"]).decode("utf-8").endswith(f"sitting s-9\n\n{note}\n\n")
    assert "KEEPOUT-4711" not in (isolated_state / "ledger.jsonl").read_text(encoding="utf-8")


@pytest.mark.parametrize("tier", TIERS)
def test_the_command_a_tier_prompt_gives_a_sitting_runs_as_written(isolated_state, capsys, tier):
    prompt = (PROMPTS / f"{tier}.md").read_text(encoding="utf-8")
    shown = re.search(r'python -m lab handoff (--sitting <[^>]+> --note "<[^>]+>")', prompt)
    assert shown, f"{tier}.md no longer shows the handoff command"
    filled = shown.group(1).replace("<id>", "s-9").replace("<text>", "what I did and what is open")
    code, out, _ = run_cli(["handoff", *shlex.split(filled)], capsys)
    assert code == 0, out
    answer = one_json_line(out)
    assert (answer["ok"], answer["sitting"], answer["chars"]) == (True, "s-9", len("what I did and what is open"))


@pytest.mark.parametrize("argv", [
    ["--note", "no sitting"],
    ["--sitting", "s-1"],
    ["--sitting", "", "--note", "empty id"],
    ["--sitting", "  ", "--note", "blank id"],
    ["--sitting", "s-1", "--note"],
    ["--sitting", "s-1", "--note", "x", "--extra"],
])
def test_bad_arguments_exit_2_and_write_nothing(isolated_state, capsys, argv):
    code, out, err = run_cli(["handoff", *argv], capsys)
    assert code == 2
    assert out == ""
    assert "usage: python -m lab handoff" in err
    assert not isolated_state.exists()


def test_a_note_that_starts_with_dashes_is_taken_after_an_equals_sign(isolated_state, capsys):
    code, out, _ = run_cli(["handoff", "--sitting", "s-1", "--note=--force was refused"], capsys)
    assert code == 0, out
    day = one_json_line(out)["day"]
    assert notes_file(isolated_state, day).decode("utf-8").endswith("sitting s-1\n\n--force was refused\n\n")


def test_handoff_help_is_answered_without_writing_anything(isolated_state, capsys):
    code, out, _ = run_cli(["handoff", "-h"], capsys)
    assert code == 0
    assert "usage: python -m lab handoff" in out and "--sitting" in out and "--note" in out
    assert not isolated_state.exists()


def test_a_sitting_can_hand_off_while_the_stop_file_exists(isolated_state, capsys):
    isolated_state.mkdir(parents=True)
    (isolated_state / "STOP").write_text("rehearsal of a halt\n", encoding="utf-8")
    assert stop.stopped() is True
    code, out, _ = run_cli(["handoff", "--sitting", "s-9", "--note", "stopped midway"], capsys)
    assert code == 0, out
    answer = one_json_line(out)
    assert (answer["ok"], answer["note_written"], answer["ledger_seq"]) == (True, True, 1)
    assert notes_file(isolated_state, answer["day"]).decode("utf-8").endswith("sitting s-9\n\nstopped midway\n\n")
    assert [row["verb"] for row in ledger.read()] == ["handoff"]


# --- when a write fails -----------------------------------------------------------------

def test_a_note_that_cannot_be_written_still_ends_the_session_and_the_row_says_so(isolated_state, capsys):
    isolated_state.mkdir(parents=True)
    (isolated_state / "notes").write_text("a file where the folder should be", encoding="utf-8")
    code, out, err = run_cli(["handoff", "--sitting", "s-9", "--note", "KEEPOUT-4711 lost"], capsys)
    assert code == 0
    answer = one_json_line(out)
    assert (answer["ok"], answer["outcome"], answer["session_ends"], answer["note_written"]) == (
        True, "handoff", True, False)
    assert answer["note_error"].startswith("FileExistsError: ")
    assert answer["ledger_seq"] == 1
    for shown in (out, err):
        assert "KEEPOUT-4711" not in shown and str(isolated_state) not in shown
    row = ledger.read()[0]
    assert (row["outcome"], row["sitting"], row["note_written"], row["note_chars"]) == (
        "handoff", "s-9", False, len("KEEPOUT-4711 lost"))
    assert "KEEPOUT-4711" not in (isolated_state / "ledger.jsonl").read_text(encoding="utf-8")


def test_a_notes_lock_held_by_another_writer_is_waited_for_only_so_long(isolated_state):
    with FileLock(isolated_state / "notes.lock") as lock:
        lock.write_note("pid 4242 (a stuck writer)")
        result = handoff.handoff("s-1", "KEEPOUT-4711 blocked", now=NOW, timeout=0.2)
    text = json.dumps(result)
    assert result["ok"] is True and result["note_written"] is False and result["ledger_seq"] == 1
    assert result["note_error"].startswith("LockTimeout: ")
    assert "KEEPOUT-4711" not in text and str(isolated_state) not in text
    assert not (isolated_state / "notes").exists()
    handoff.handoff("s-2", "after the wait", now=NOW)
    assert notes_file(isolated_state) == b"## 14:02:03Z, sitting s-2\n\nafter the wait\n\n"
    assert [row["note_written"] for row in ledger.read()] == [False, True]


def test_a_ledger_that_cannot_be_written_fails_the_call_but_the_note_is_kept(isolated_state, capsys):
    (isolated_state / "ledger.jsonl").mkdir(parents=True)
    code, out, err = run_cli(["handoff", "--sitting", "s-9", "--note", "KEEPOUT-4711 kept"], capsys)
    assert code == 1
    answer = one_json_line(out)
    assert (answer["ok"], answer["outcome"], answer["session_ends"]) == (False, "handoff", True)
    assert (answer["note_written"], answer["ledger_seq"]) == (True, None)
    assert re.match(r"(PermissionError|IsADirectoryError): ", answer["ledger_error"]), answer
    for shown in (out, err):
        assert "KEEPOUT-4711" not in shown and str(isolated_state) not in shown
    assert notes_file(isolated_state, answer["day"]).decode("utf-8").endswith("sitting s-9\n\nKEEPOUT-4711 kept\n\n")


# --- many sittings at once --------------------------------------------------------------

def _long_note(sitting: str) -> str:
    """Forty dense lines, none blank, so a torn or interleaved entry cannot parse back."""
    return "\n".join(f"{sitting} line {line} " + "x" * 90 for line in range(40))


def _hand_off_repeatedly(state: str, number: int, start, out: str) -> None:
    """Spawned child: wait for its partners, then hand off HANDOFFS_PER_SITTING times as fast as it can."""
    start.wait(START_TIMEOUT)
    answers = [handoff.handoff(f"s-{number}-{i}", _long_note(f"s-{number}-{i}"), state_dir=state)
               for i in range(HANDOFFS_PER_SITTING)]
    Path(out).write_text(json.dumps(answers), encoding="utf-8")


def test_sittings_handing_off_at_the_same_moment_each_leave_a_whole_note_and_a_row(tmp_path, isolated_state):
    start = SPAWN.Barrier(SITTINGS)
    children = [SPAWN.Process(target=_hand_off_repeatedly,
                              args=(str(isolated_state), n, start, str(tmp_path / f"{n}.json")), daemon=True)
                for n in range(SITTINGS)]
    for child in children:
        child.start()
    for child in children:
        child.join(JOIN_TIMEOUT)
    assert [child.exitcode for child in children] == [0] * SITTINGS
    expected = {f"s-{n}-{i}": _long_note(f"s-{n}-{i}") for n in range(SITTINGS) for i in range(HANDOFFS_PER_SITTING)}
    assert len(note_files(isolated_state)) in (1, 2), "the run crossed more than one UTC midnight"
    parts: list[str] = []
    for name in note_files(isolated_state):
        text = (isolated_state / "notes" / name).read_bytes().decode("utf-8")
        assert text.endswith("\n\n")
        parts += text.split("\n\n")[:-1]
    assert len(parts) == 2 * len(expected), "headings and bodies do not pair up"
    found: dict[str, str] = {}
    for heading, body in zip(parts[0::2], parts[1::2]):
        match = re.fullmatch(r"## \d\d:\d\d:\d\dZ, sitting (s-\d+-\d+)", heading)
        assert match, f"torn heading {heading!r}"
        assert match.group(1) not in found, f"{match.group(1)} appears twice"
        found[match.group(1)] = body
    assert found == expected
    rows = ledger.read()
    assert [row["seq"] for row in rows] == list(range(1, len(expected) + 1))
    assert sorted(row["sitting"] for row in rows) == sorted(expected)
    assert all(row["note_written"] is True for row in rows)
    for n in range(SITTINGS):
        answers = json.loads((tmp_path / f"{n}.json").read_text(encoding="utf-8"))
        assert [a["ledger_seq"] for a in answers] == [r["seq"] for r in rows if r["sitting"].startswith(f"s-{n}-")]


# --- the notes folder and what a sitting may do with it ---------------------------------

@pytest.mark.parametrize("tier", TIERS)
def test_a_sitting_cannot_read_or_change_the_folder_the_notes_are_written_to(tier):
    day = "2026-10-10"
    paths = [
        handoff.notes_path(day, state_dir=Path(allowed.STATE_DIR)).as_posix(),
        (Path("work") / "T1" / allowed.STATE_DIR / handoff.NOTES_DIR / f"{day}.md").as_posix(),
        "C:\\Users\\someone\\work\\T1\\" + allowed.STATE_DIR.replace("/", "\\") + f"\\{handoff.NOTES_DIR}\\{day}.md",
    ]
    for tool in ("Read", "Grep", "Glob", "Edit", "Write"):
        assert [p for p in paths if allowed.allows(tier, tool, p)] == [], (tier, tool)
    assert [p for p in paths if not allowed.denies("Read", p)] == []


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("python", allowed.PYTHONS)
def test_a_sitting_of_every_tier_may_call_handoff_even_when_its_note_names_the_notes_folder(tier, python):
    plain = f'{python} -m lab handoff --sitting s-12 --note "Opened the PR and left the lock on O19"'
    naming = f'{python} -m lab handoff --sitting s-12 --note "I did not read lab/state/notes"'
    assert allowed.allows(tier, "Bash", plain)
    assert allowed.allows(tier, "Bash", naming)


def test_notes_in_the_default_state_dir_are_ignored_by_git():
    ignored = subprocess.run(["git", "check-ignore", "-q", "lab/state/notes/2026-10-10.md"], cwd=REPO_ROOT)
    assert ignored.returncode == 0, "git must ignore lab/state/notes/<day>.md (1: it does not; 128: not a checkout)"
