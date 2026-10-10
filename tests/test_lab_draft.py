"""Tests for the draft verb (python -m lab draft): a sitting asks for a verb it lacks.

Every test gets its own state dir through LAB_STATE_DIR, so none of them writes the real lab/state. The stub
a draft leaves is loaded and called for real, with goals built to break out of a string, because the stub is
the one file here that carries a sitting's own words as code.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import multiprocessing
import os
import re
import shlex
import shutil
from pathlib import Path

import pytest

from lab import __main__ as lab_cli
from lab.machine import allowed, draft, ledger, record
from lab.machine.lock import FileLock

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS = REPO_ROOT / "lab" / "machine" / "prompts"
UTC = dt.timezone.utc
NOW = dt.datetime(2026, 10, 10, 14, 2, 3, tzinfo=UTC)
TIERS = ("fable", "opus", "sonnet")
SPAWN = multiprocessing.get_context("spawn")
JOIN_TIMEOUT = 300.0     # seconds; a spawned child still running after this has hung
START_TIMEOUT = 60.0     # seconds a spawned child waits for its partners to start
CALLERS = 4


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """A private state dir for every test, and no actor or tier from the environment."""
    state = tmp_path / "state"
    monkeypatch.setenv("LAB_STATE_DIR", str(state))
    monkeypatch.delenv("LAB_ACTOR", raising=False)
    monkeypatch.delenv("LAB_TIER", raising=False)
    return state


def run_cli(argv, capsys):
    code = lab_cli.main(argv)
    out, err = capsys.readouterr()
    return code, out, err


def one_json_line(text: str) -> dict:
    lines = text.splitlines()
    assert len(lines) == 1, f"expected exactly one stdout line, got {lines!r}"
    return json.loads(lines[0])


def draft_folders(state: Path) -> list[str]:
    """Every entry of the drafts dir, hidden ones included, so a leftover staging folder shows."""
    root = state / "drafts"
    return sorted(path.name for path in root.iterdir()) if root.exists() else []


def request_of(state: Path, name: str) -> dict:
    return json.loads((state / "drafts" / name / "request.json").read_text(encoding="utf-8"))


def load_stub(state: Path, name: str):
    """Import the stub module of a draft from its file, the way the owner would after moving it."""
    module_name = name.replace("-", "_")
    path = state / "drafts" / name / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"draft_stub_{module_name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- what a draft leaves behind -----------------------------------------------------------

def test_a_draft_leaves_a_request_and_a_stub_module_named_after_the_verb(isolated_state):
    result = draft.draft("gpu-peek", "Show the GPU B queue in one line.", sitting="sit-1", actor="sit-1",
                         tier="Sonnet", now=NOW)
    assert result == {
        "ok": True, "outcome": "drafted", "draft": "gpu-peek", "module": "gpu_peek",
        "files": ["gpu_peek.py", "request.json"], "sitting": "sit-1", "ledger_seq": 1,
        "next": "carry on without this verb; the draft waits for the owner and nothing in it runs",
    }
    assert draft_folders(isolated_state) == ["gpu-peek"]
    made = sorted(p.name for p in (isolated_state / "drafts" / "gpu-peek").iterdir())
    assert made == ["gpu_peek.py", "request.json"]
    assert request_of(isolated_state, "gpu-peek") == {
        "name": "gpu-peek", "module": "gpu_peek", "goal": "Show the GPU B queue in one line.",
        "sitting": "sit-1", "tier": "sonnet", "actor": "sit-1", "requested": "2026-10-10T14:02:03Z",
    }


def test_the_stub_is_a_verb_the_dispatcher_accepts_that_answers_not_built(isolated_state, capsys):
    draft.draft("gpu-peek", "Show the GPU B queue.", now=NOW)
    stub = load_stub(isolated_state, "gpu-peek")
    assert stub.NAME == "gpu-peek"
    assert stub.GOAL == "Show the GPU B queue."
    answer = stub.main([])
    assert answer == {"ok": False, "outcome": "not_built", "verb": "gpu-peek",
                      "error": "this verb is only a draft; see GOAL"}
    assert lab_cli.finish(answer, "gpu-peek") == 1
    assert one_json_line(capsys.readouterr().out) == answer


def test_the_stub_names_the_registry_line_that_adopting_it_needs(isolated_state):
    draft.draft("slot-bulk", "Request several slots at once.", now=NOW)
    source = (isolated_state / "drafts" / "slot-bulk" / "slot_bulk.py").read_text(encoding="utf-8")
    assert '    "slot-bulk": "lab.machine.slot_bulk:main",' in source
    assert "tests/test_lab_slot_bulk.py" in source
    target = re.search(r'"slot-bulk": "([^"]+)"', source).group(1)
    assert re.fullmatch(r"lab(\.[a-z_][a-z0-9_]*)+:[a-z_][a-z0-9_]*", target)


HOSTILE_GOALS = {
    "triple double quotes": 'close it """ and carry on',
    "triple single quotes": "close it " + "'" * 3 + " and carry on",
    "backslashes": "C:\\temp\\new\\table and a trailing backslash \\",
    "a newline then code": "first line\nimport os\nos.environ['O32_PWNED'] = '1'\nlast line",
    "a quote then code": "'; __import__('os').environ.update(O32_PWNED='1'); '",
    "format fields and percent codes": "{name} {{goal}} {0} %s %(x)s",
    "comment markers": "# not a comment\n# neither",
    "non-ascii text": "caf\u00e9 \u2713 \U0001f600",
    "control characters": "bell \x07 nul \x00 escape \x1b",
    "unicode line separators": "a\u2028b\u2029c\x85d",
}


@pytest.mark.parametrize("label", sorted(HOSTILE_GOALS))
def test_a_goal_stays_a_string_in_the_stub_and_the_request_whatever_it_contains(isolated_state, monkeypatch, label):
    goal = HOSTILE_GOALS[label]
    monkeypatch.delenv("O32_PWNED", raising=False)
    result = draft.draft("hostile-goal", goal, now=NOW)
    assert result["ok"] is True, result
    stub = load_stub(isolated_state, "hostile-goal")
    assert stub.GOAL == goal.strip()
    assert request_of(isolated_state, "hostile-goal")["goal"] == goal.strip()
    assert "O32_PWNED" not in os.environ, "the goal ran as code"
    assert stub.main([])["outcome"] == "not_built"


def test_a_goal_with_a_character_utf8_cannot_hold_is_kept_with_a_question_mark(isolated_state):
    result = draft.draft("odd-goal", "before \ud800 after", now=NOW)
    assert result["ok"] is True, result
    assert load_stub(isolated_state, "odd-goal").GOAL == "before ? after"
    assert request_of(isolated_state, "odd-goal")["goal"] == "before ? after"


def test_the_goal_is_stripped_and_counted_without_its_padding(isolated_state):
    draft.draft("padded-goal", "  needs a verb \n\n", now=NOW)
    assert request_of(isolated_state, "padded-goal")["goal"] == "needs a verb"
    assert ledger.read()[0]["goal_chars"] == len("needs a verb")


def test_a_goal_of_exactly_the_limit_is_accepted_and_one_more_character_is_not(isolated_state):
    assert draft.draft("at-limit", "x" * draft.MAX_GOAL_CHARS, now=NOW)["ok"] is True
    refused = draft.draft("over-limit", "x" * (draft.MAX_GOAL_CHARS + 1), now=NOW)
    assert (refused["ok"], refused["outcome"]) == (False, "usage")
    assert refused["error"].startswith(f"--goal is {draft.MAX_GOAL_CHARS + 1} characters; the limit is 1000")
    assert draft_folders(isolated_state) == ["at-limit"]


def test_a_blank_goal_is_refused(isolated_state):
    for goal in ("", "   ", "\n\t "):
        refused = draft.draft("some-verb", goal, now=NOW)
        assert refused == {"ok": False, "outcome": "usage", "error": "--goal is blank"}
    assert draft_folders(isolated_state) == []
    assert ledger.read() == []


def test_a_naive_time_is_a_programming_error_and_writes_nothing(isolated_state):
    with pytest.raises(ValueError, match="UTC offset"):
        draft.draft("some-verb", "a goal", now=dt.datetime(2026, 10, 10, 14, 2, 3))
    assert draft_folders(isolated_state) == []


def test_without_a_time_the_request_is_stamped_with_the_current_utc_time(isolated_state):
    before = dt.datetime.now(UTC).replace(microsecond=0)
    draft.draft("stamped", "a goal")
    after = dt.datetime.now(UTC)
    stamped = dt.datetime.fromisoformat(request_of(isolated_state, "stamped")["requested"].replace("Z", "+00:00"))
    assert before <= stamped <= after


# --- the name -------------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "", "   ", "Upper", "snake_case", "two words", "-lead", "trail-", "dou--ble", "9lives", "a/b", "../escape",
    "a\\b", "dot.ted", "x" * (draft.MAX_NAME_CHARS + 1), "class", "import", "nul", "con", "aux", "prn", "com1", "lpt9",
    "rejected", "caf\u00e9",
])
def test_a_name_that_cannot_be_a_verb_or_a_folder_is_refused_and_nothing_is_written(isolated_state, name):
    refused = draft.draft(name, "a goal", now=NOW)
    assert (refused["ok"], refused["outcome"]) == (False, "usage"), refused
    assert refused["error"].startswith("--name ")
    assert draft_folders(isolated_state) == []
    assert ledger.read() == []


@pytest.mark.parametrize("name", ["a", "x1", "slot-bulk", "research-check2", "a-b-c-d", "z" * draft.MAX_NAME_CHARS])
def test_a_name_that_could_be_registered_is_accepted(isolated_state, name):
    result = draft.draft(name, "a goal", now=NOW)
    assert result["ok"] is True, result
    assert draft_folders(isolated_state) == [name]


def test_a_name_longer_than_the_limit_says_how_long_it_is(isolated_state):
    refused = draft.draft("x" * 41, "a goal", now=NOW)
    assert refused["error"] == "--name is 41 characters; the limit is 40"


@pytest.mark.parametrize("name", ["handoff", "draft", "stop", "metrics"])
def test_a_registered_verb_cannot_be_drafted_and_the_refusal_points_at_it(isolated_state, name):
    assert name in lab_cli.VERBS
    refused = draft.draft(name, "a goal", now=NOW)
    assert refused == {"ok": False, "outcome": "exists",
                       "error": f"the verb {name} exists: run python -m lab {name} -h and use it"}
    assert draft_folders(isolated_state) == []


@pytest.mark.parametrize("name, shown", [
    ("allowed", "lab/machine/allowed.py"), ("lock", "lab/machine/lock.py"), ("verdict", "lab/machine/verdict.py"),
    ("prompts", "lab/machine/prompts/"),
])
def test_a_name_that_would_collide_with_a_module_of_the_lab_is_refused(isolated_state, name, shown):
    refused = draft.draft(name, "a goal", now=NOW)
    assert refused == {"ok": False, "outcome": "exists",
                       "error": f"{shown} exists, so a verb named {name} would collide with it; choose another name"}
    assert draft_folders(isolated_state) == []


def test_a_verb_registered_in_the_dispatcher_is_found_even_when_no_module_of_that_name_exists(isolated_state,
                                                                                              monkeypatch):
    monkeypatch.setitem(lab_cli.VERBS, "ghost-verb", "lab.machine.ghost_verb:main")
    refused = draft.draft("ghost-verb", "a goal", now=NOW)
    assert (refused["ok"], refused["outcome"]) == (False, "exists")


def test_a_name_that_already_has_a_draft_is_taken_for_every_sitting(isolated_state):
    assert draft.draft("slot-bulk", "first goal", sitting="s-1", tier="sonnet", now=NOW)["ok"] is True
    refused = draft.draft("slot-bulk", "second goal", sitting="s-2", tier="opus", now=NOW)
    assert (refused["ok"], refused["outcome"]) == (False, "name_taken")
    assert request_of(isolated_state, "slot-bulk")["goal"] == "first goal"
    assert [row["draft"] for row in ledger.read()] == ["slot-bulk"]


# --- one draft per sitting -------------------------------------------------------------------

def test_a_second_draft_from_the_same_sitting_is_refused_and_writes_nothing(isolated_state):
    first = draft.draft("slot-bulk", "first goal", sitting="s-1", tier="sonnet", now=NOW)
    second = draft.draft("gpu-peek", "second goal", sitting="s-1", tier="sonnet", now=NOW)
    assert first["ok"] is True
    assert second == {"ok": False, "outcome": "already_drafted", "draft": "slot-bulk",
                      "error": "this sitting already drafted slot-bulk: one draft per sitting"}
    assert draft_folders(isolated_state) == ["slot-bulk"]
    assert [row["draft"] for row in ledger.read()] == ["slot-bulk"]


def test_each_sitting_gets_its_own_draft(isolated_state):
    for number in range(3):
        result = draft.draft(f"verb-{number}", "a goal", sitting=f"s-{number}", tier="opus", now=NOW)
        assert result["ok"] is True, result
    assert draft_folders(isolated_state) == ["verb-0", "verb-1", "verb-2"]
    assert [row["sitting"] for row in ledger.read()] == ["s-0", "s-1", "s-2"]


def test_the_owner_outside_a_sitting_has_no_limit(isolated_state):
    for name in ("first-verb", "second-verb", "third-verb"):
        result = draft.draft(name, "a goal", now=NOW)
        assert (result["ok"], result["sitting"]) == (True, None)
    rows = ledger.read()
    assert [(row["actor"], row["tier"], row["sitting"]) for row in rows] == [(record.UNKNOWN_ACTOR, None, None)] * 3


def test_a_refused_call_does_not_spend_the_sitting_draft(isolated_state):
    draft.draft("taken-name", "someone else", sitting="s-other", now=NOW)
    for name, outcome in (("Bad_Name", "usage"), ("handoff", "exists"), ("taken-name", "name_taken")):
        assert draft.draft(name, "a goal", sitting="s-1", now=NOW)["outcome"] == outcome
    ok = draft.draft("good-name", "a goal", sitting="s-1", now=NOW)
    assert (ok["ok"], ok["draft"]) == (True, "good-name")
    assert [row["draft"] for row in ledger.read()] == ["taken-name", "good-name"]


def test_a_draft_the_owner_rejected_still_spends_the_sitting_and_keeps_its_name(isolated_state):
    draft.draft("alpha", "a goal", sitting="s-1", now=NOW)
    rejected = isolated_state / "drafts" / "rejected"
    rejected.mkdir()
    shutil.move(str(isolated_state / "drafts" / "alpha"), str(rejected / "alpha"))
    again = draft.draft("beta", "a goal", sitting="s-1", now=NOW)
    assert (again["outcome"], again["draft"]) == ("already_drafted", "alpha")
    retry = draft.draft("alpha", "a goal", sitting="s-2", now=NOW)
    assert retry["outcome"] == "name_taken"
    assert draft_folders(isolated_state) == ["rejected"]


def test_a_draft_the_owner_deleted_still_spends_the_sitting_through_the_ledger(isolated_state):
    draft.draft("alpha", "a goal", sitting="s-1", now=NOW)
    shutil.rmtree(isolated_state / "drafts" / "alpha")
    again = draft.draft("beta", "a goal", sitting="s-1", now=NOW)
    assert (again["outcome"], again["draft"]) == ("already_drafted", "alpha")
    assert draft_folders(isolated_state) == []


def test_requests_lists_every_draft_with_the_rejected_ones_marked(isolated_state):
    draft.draft("alpha", "first", sitting="s-1", tier="sonnet", now=NOW)
    draft.draft("beta", "second", sitting="s-2", tier="opus", now=NOW)
    rejected = isolated_state / "drafts" / "rejected"
    rejected.mkdir()
    shutil.move(str(isolated_state / "drafts" / "beta"), str(rejected / "beta"))
    listed = draft.requests()
    assert [(r["name"], r["sitting"], r["rejected"]) for r in listed] == [("alpha", "s-1", False), ("beta", "s-2", True)]
    assert listed[0]["goal"] == "first"


def test_requests_skips_a_hidden_staging_folder_and_a_folder_without_a_readable_request(isolated_state):
    draft.draft("alpha", "first", now=NOW)
    drafts = isolated_state / "drafts"
    (drafts / ".gamma.123.tmp").mkdir()
    (drafts / ".gamma.123.tmp" / "request.json").write_text(json.dumps({"name": "gamma"}), encoding="utf-8")
    (drafts / "broken").mkdir()
    (drafts / "broken" / "request.json").write_text("{not json", encoding="utf-8")
    (drafts / "list-not-object").mkdir()
    (drafts / "list-not-object" / "request.json").write_text("[1, 2]", encoding="utf-8")
    assert [r["name"] for r in draft.requests()] == ["alpha"]


def test_requests_reads_the_name_from_the_folder_when_the_request_has_none(isolated_state):
    folder = isolated_state / "drafts" / "nameless"
    folder.mkdir(parents=True)
    (folder / "request.json").write_text(json.dumps({"goal": "g", "sitting": "s-1"}), encoding="utf-8")
    assert draft.requests() == [{"goal": "g", "sitting": "s-1", "name": "nameless", "rejected": False}]
    assert draft.draft("other", "g", sitting="s-1", now=NOW)["outcome"] == "already_drafted"


# --- who is asking ---------------------------------------------------------------------------

def test_a_sitting_is_the_actor_when_the_tier_is_also_set(monkeypatch):
    monkeypatch.setenv("LAB_ACTOR", " sit-1 ")
    monkeypatch.setenv("LAB_TIER", " Opus ")
    assert draft.identity() == ("sit-1", "opus", "sit-1")


@pytest.mark.parametrize("actor, tier, expected", [
    ("ilan", None, ("ilan", None, None)),
    (None, "sonnet", (None, "sonnet", None)),
    (None, None, (None, None, None)),
    ("  ", "opus", (None, "opus", None)),
    ("sit-1", "  ", ("sit-1", None, None)),
])
def test_anyone_without_both_variables_is_not_a_sitting(monkeypatch, actor, tier, expected):
    for key, value in (("LAB_ACTOR", actor), ("LAB_TIER", tier)):
        if value is not None:
            monkeypatch.setenv(key, value)
    assert draft.identity() == expected


def test_identity_reads_the_mapping_it_is_given():
    assert draft.identity({"LAB_ACTOR": "sit-7", "LAB_TIER": "fable"}) == ("sit-7", "fable", "sit-7")


# --- the ledger row --------------------------------------------------------------------------

def test_an_actor_and_tier_not_given_come_from_the_environment_like_every_other_verb(isolated_state, monkeypatch):
    monkeypatch.setenv("LAB_ACTOR", "ilan")
    monkeypatch.setenv("LAB_TIER", "Fable")
    draft.draft("queue-list", "a goal", now=NOW)
    request = request_of(isolated_state, "queue-list")
    assert (request["actor"], request["tier"], request["sitting"]) == ("ilan", "fable", None)
    row = ledger.read()[0]
    assert (row["actor"], row["tier"], row["sitting"]) == ("ilan", "fable", None)


def test_the_ledger_row_names_the_draft_and_the_goal_length_but_never_the_goal(isolated_state):
    goal = "KEEPOUT-4711 needs a verb that lists the queue"
    result = draft.draft("queue-list", goal, sitting="sit-1", actor="sit-1", tier="Opus", now=NOW)
    rows = ledger.read()
    assert len(rows) == 1
    row = rows[0]
    assert (row["seq"], row["actor"], row["tier"], row["task"], row["verb"], row["outcome"]) == (
        1, "sit-1", "opus", None, "draft", "drafted")
    assert (row["metrics"], row["sitting"], row["draft"], row["goal_chars"]) == ({}, "sit-1", "queue-list", len(goal))
    assert result["ledger_seq"] == 1
    assert "KEEPOUT-4711" not in (isolated_state / "ledger.jsonl").read_text(encoding="utf-8")
    assert "KEEPOUT-4711" in (isolated_state / "drafts" / "queue-list" / "request.json").read_text(encoding="utf-8")


def test_the_row_carries_the_task_board_id_the_sitting_works_on(isolated_state, monkeypatch):
    monkeypatch.setenv("LAB_TASK", "O99")
    draft.draft("queue-list", "a goal", sitting="sit-1", tier="sonnet", now=NOW)
    assert ledger.read()[0]["task"] == "O99"


def test_the_answer_has_neither_the_goal_nor_a_path(isolated_state, capsys):
    code, out, err = run_cli(["draft", "--name", "queue-list", "--goal", "KEEPOUT-4711 lists the queue"], capsys)
    assert code == 0
    answer = one_json_line(out)
    assert answer["ok"] is True and answer["draft"] == "queue-list"
    for shown in (out, err):
        assert "KEEPOUT-4711" not in shown and str(isolated_state) not in shown


# --- failures leave nothing half-made ----------------------------------------------------------

def test_a_ledger_row_that_cannot_be_written_withdraws_the_draft_and_does_not_spend_the_sitting(isolated_state,
                                                                                                  capsys):
    (isolated_state / "ledger.jsonl").mkdir(parents=True)
    code, out, err = run_cli(["draft", "--name", "alpha", "--goal", "KEEPOUT-4711 lost"], capsys)
    assert code == 1
    answer = one_json_line(out)
    assert (answer["ok"], answer["outcome"]) == (False, "error")
    assert answer["error"].startswith("the ledger row was not written, so the draft was withdrawn: ")
    for shown in (out, err):
        assert "KEEPOUT-4711" not in shown and str(isolated_state) not in shown
    assert draft_folders(isolated_state) == []
    shutil.rmtree(isolated_state / "ledger.jsonl")
    retry = draft.draft("alpha", "a goal", sitting="s-1", now=NOW)
    assert (retry["ok"], retry["ledger_seq"]) == (True, 1)
    assert draft_folders(isolated_state) == ["alpha"]


def test_a_drafts_dir_that_cannot_be_made_is_an_error_answer_and_writes_no_row(isolated_state):
    isolated_state.mkdir(parents=True)
    (isolated_state / "drafts").write_text("in the way", encoding="utf-8")
    result = draft.draft("alpha", "KEEPOUT-4711 a goal", sitting="s-1", now=NOW)
    assert (result["ok"], result["outcome"]) == (False, "error")
    assert result["error"].startswith("the draft was not written: ")
    assert "KEEPOUT-4711" not in json.dumps(result) and str(isolated_state) not in json.dumps(result)
    assert ledger.read() == []


def test_a_claim_lock_held_by_another_writer_is_waited_for_only_so_long(isolated_state):
    with FileLock(isolated_state / "drafts.lock") as lock:
        lock.write_note("pid 4242 (a stuck writer)")
        busy = draft.draft("alpha", "a goal", sitting="s-1", now=NOW, timeout=0.2)
    assert busy == {"ok": False, "outcome": "busy", "error": "another draft is being written; call again in a moment"}
    assert draft_folders(isolated_state) == [] and ledger.read() == []
    assert draft.draft("alpha", "a goal", sitting="s-1", now=NOW)["ok"] is True


# --- many sittings at once ---------------------------------------------------------------------

def _draft_in_a_child(state: str, sitting: str, name: str, start, out: str) -> None:
    """Spawned child: wait for its partners, then ask for one draft as fast as it can."""
    start.wait(START_TIMEOUT)
    answer = draft.draft(name, f"goal of {name}", sitting=sitting, tier="sonnet", state_dir=state)
    Path(out).write_text(json.dumps(answer), encoding="utf-8")


def _drafts_at_the_same_moment(tmp_path: Path, state: Path, callers: list[tuple[str, str]]) -> list[dict]:
    start = SPAWN.Barrier(len(callers))
    outs = [tmp_path / f"answer-{n}.json" for n in range(len(callers))]
    children = [SPAWN.Process(target=_draft_in_a_child, args=(str(state), sitting, name, start, str(out)), daemon=True)
                for (sitting, name), out in zip(callers, outs)]
    for child in children:
        child.start()
    for child in children:
        child.join(JOIN_TIMEOUT)
    assert [child.exitcode for child in children] == [0] * len(callers)
    return [json.loads(out.read_text(encoding="utf-8")) for out in outs]


def test_one_sitting_calling_at_the_same_moment_still_gets_exactly_one_draft(tmp_path, isolated_state):
    answers = _drafts_at_the_same_moment(tmp_path, isolated_state,
                                         [("sit-shared", f"verb-{n}") for n in range(CALLERS)])
    assert sorted(a["outcome"] for a in answers) == ["already_drafted"] * (CALLERS - 1) + ["drafted"]
    winner = next(a["draft"] for a in answers if a["outcome"] == "drafted")
    assert all(a["draft"] == winner for a in answers if a["outcome"] == "already_drafted")
    assert draft_folders(isolated_state) == [winner]
    assert [(row["draft"], row["sitting"]) for row in ledger.read()] == [(winner, "sit-shared")]


def test_sittings_calling_at_the_same_moment_each_get_a_draft_and_a_row(tmp_path, isolated_state):
    answers = _drafts_at_the_same_moment(tmp_path, isolated_state,
                                         [(f"sit-{n}", f"verb-{n}") for n in range(CALLERS)])
    assert [a["outcome"] for a in answers] == ["drafted"] * CALLERS
    assert draft_folders(isolated_state) == [f"verb-{n}" for n in range(CALLERS)]
    assert sorted(row["seq"] for row in ledger.read()) == list(range(1, CALLERS + 1))
    assert sorted(a["ledger_seq"] for a in answers) == list(range(1, CALLERS + 1))


def test_sittings_asking_for_the_same_name_at_the_same_moment_make_one_draft(tmp_path, isolated_state):
    answers = _drafts_at_the_same_moment(tmp_path, isolated_state,
                                         [(f"sit-{n}", "same-verb") for n in range(CALLERS)])
    assert sorted(a["outcome"] for a in answers) == ["drafted"] + ["name_taken"] * (CALLERS - 1)
    assert draft_folders(isolated_state) == ["same-verb"]
    assert len(ledger.read()) == 1


# --- through the dispatcher --------------------------------------------------------------------

def test_the_verb_is_registered_and_listed_in_the_help(capsys):
    assert lab_cli.VERBS["draft"] == "lab.machine.draft:main"
    code, out, _ = run_cli(["--help"], capsys)
    assert code == 0
    assert re.search(r"^  draft +Ask for a verb the lab lacks", out, re.MULTILINE), out


def test_draft_answers_one_json_line_and_exit_code_0(isolated_state, capsys, monkeypatch):
    monkeypatch.setenv("LAB_ACTOR", "sit-20261010-1402-sonnet-ab12")
    monkeypatch.setenv("LAB_TIER", "sonnet")
    code, out, err = run_cli(["draft", "--name", "queue-list", "--goal", "List the queue in one line."], capsys)
    assert code == 0 and err == ""
    answer = one_json_line(out)
    assert (answer["ok"], answer["outcome"], answer["draft"], answer["sitting"]) == (
        True, "drafted", "queue-list", "sit-20261010-1402-sonnet-ab12")
    request = request_of(isolated_state, "queue-list")
    assert (request["sitting"], request["tier"], request["actor"]) == (
        "sit-20261010-1402-sonnet-ab12", "sonnet", "sit-20261010-1402-sonnet-ab12")


def test_a_sitting_that_drafts_twice_through_the_dispatcher_is_refused_the_second_time(isolated_state, capsys,
                                                                                       monkeypatch):
    monkeypatch.setenv("LAB_ACTOR", "sit-1")
    monkeypatch.setenv("LAB_TIER", "opus")
    first = run_cli(["draft", "--name", "alpha", "--goal", "one"], capsys)
    second = run_cli(["draft", "--name", "beta", "--goal", "two"], capsys)
    assert first[0] == 0 and second[0] == 1
    answer = one_json_line(second[1])
    assert (answer["outcome"], answer["draft"]) == ("already_drafted", "alpha")
    monkeypatch.setenv("LAB_ACTOR", "sit-2")
    third = run_cli(["draft", "--name", "beta", "--goal", "two"], capsys)
    assert third[0] == 0
    assert draft_folders(isolated_state) == ["alpha", "beta"]


def test_a_sitting_cannot_dodge_the_limit_by_naming_itself_differently(isolated_state, capsys, monkeypatch):
    monkeypatch.setenv("LAB_ACTOR", "sit-1")
    monkeypatch.setenv("LAB_TIER", "opus")
    assert run_cli(["draft", "--name", "alpha", "--goal", "one"], capsys)[0] == 0
    code, out, err = run_cli(["draft", "--name", "beta", "--goal", "two", "--sitting", "sit-2"], capsys)
    assert code == 2
    assert "unrecognized arguments: --sitting" in err
    assert draft_folders(isolated_state) == ["alpha"]


def test_an_actor_without_a_tier_is_not_a_sitting_and_has_no_limit(isolated_state, capsys, monkeypatch):
    monkeypatch.setenv("LAB_ACTOR", "ilan")
    for name in ("alpha", "beta"):
        code, out, _ = run_cli(["draft", "--name", name, "--goal", "a goal"], capsys)
        assert code == 0
        assert one_json_line(out)["sitting"] is None
    assert [row["actor"] for row in ledger.read()] == ["ilan", "ilan"]


def test_a_goal_that_starts_with_a_dash_is_written_with_an_equals_sign(isolated_state, capsys):
    code, out, _ = run_cli(["draft", "--name", "dashed", "--goal=--dry-run flag for the metrics verb"], capsys)
    assert code == 0, out
    assert request_of(isolated_state, "dashed")["goal"] == "--dry-run flag for the metrics verb"


@pytest.mark.parametrize("argv", [["draft"], ["draft", "--name", "alpha"], ["draft", "--goal", "a goal"]])
def test_name_and_goal_are_both_required(isolated_state, capsys, argv):
    code, out, err = run_cli(argv, capsys)
    assert code == 2 and out == ""
    assert "the following arguments are required" in err
    assert draft_folders(isolated_state) == []


def test_draft_help_describes_both_arguments(capsys):
    code, out, _ = run_cli(["draft", "-h"], capsys)
    assert code == 0
    assert "usage: python -m lab draft" in out and "--name VERB" in out and "--goal TEXT" in out


def test_the_stop_file_refuses_a_draft_before_anything_is_written(isolated_state, capsys):
    isolated_state.mkdir(parents=True)
    (isolated_state / "STOP").write_text("maintenance\n", encoding="utf-8")
    code, out, _ = run_cli(["draft", "--name", "alpha", "--goal", "a goal"], capsys)
    assert code == 1
    answer = one_json_line(out)
    assert (answer["ok"], answer["outcome"], answer["reason"]) == (False, "stopped", "maintenance")
    assert draft_folders(isolated_state) == [] and ledger.read() == []


# --- the tier prompts and the allow-lists -------------------------------------------------------

def _prompt_command(tier: str) -> str:
    text = (PROMPTS / f"{tier}.md").read_text(encoding="utf-8")
    found = re.search(r'python -m lab draft --name <verb> --goal "<text>"', text)
    assert found, f"{tier}.md does not show the draft command"
    return found.group(0).replace("<verb>", "prompt-check").replace("<text>", "a goal from the prompt")


@pytest.mark.parametrize("tier", TIERS)
def test_the_tier_prompt_tells_a_sitting_not_to_script_around_a_missing_verb(tier):
    text = (PROMPTS / f"{tier}.md").read_text(encoding="utf-8")
    assert "Never script around a missing verb: a draft is only a request for Ilan." in text.splitlines()


@pytest.mark.parametrize("tier", TIERS)
def test_the_command_a_tier_prompt_shows_runs_through_the_dispatcher(isolated_state, capsys, tier):
    argv = shlex.split(_prompt_command(tier))[3:]
    assert argv[:1] == ["draft"]
    code, out, _ = run_cli(argv, capsys)
    assert code == 0, out
    assert one_json_line(out)["draft"] == "prompt-check"
    assert request_of(isolated_state, "prompt-check")["goal"] == "a goal from the prompt"


@pytest.mark.parametrize("tier", TIERS)
def test_a_sitting_of_every_tier_may_run_the_command_its_prompt_shows(tier):
    assert allowed.allows(tier, "Bash", _prompt_command(tier))
