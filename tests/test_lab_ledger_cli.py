"""python -m lab ledger add / tail / grep: what each sub-verb writes and prints, the usage errors, the STOP
refusal, the way a later sub-verb is added, and one run through a real process. Every test works in a
temporary state dir; nothing here touches lab/state."""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from lab import __main__ as lab_cli
from lab.machine import ledger, ledger_cli

REPO_ROOT = Path(__file__).resolve().parents[1]
NO_REFS = {"pr": None, "commit": None, "slot": None, "run_id": None}


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """A private state dir and no writer identity in the environment."""
    state = tmp_path / "state"
    monkeypatch.setenv("LAB_STATE_DIR", str(state))
    for name in ("LAB_ACTOR", "LAB_TIER", "LAB_TASK"):
        monkeypatch.delenv(name, raising=False)
    return state


def run(capsys, *argv):
    """(exit code, the one JSON object on stdout, stderr) of python -m lab ledger <argv>."""
    code = lab_cli.main(["ledger", *argv])
    out, err = capsys.readouterr()
    lines = out.splitlines()
    assert len(lines) == 1, f"expected exactly one stdout line, got {lines!r}"
    return code, json.loads(lines[0]), err


def seed(verb="bench.run", outcome="ok", ts=None, **more):
    row = {"actor": "opus-1", "tier": "opus", "task": "O16", "verb": verb, "outcome": outcome,
           "metrics": {}, "refs": {}, **more}
    if ts is not None:
        row["ts"] = ts
    return ledger.append(row)


def seed_standard():
    """Four rows of different writers, verbs, outcomes, nodes, metrics and refs: seqs 1 to 4."""
    seed("bench.run", "ok", ts="2026-10-10T10:00:00Z", node="pod-a", metrics={"teacher.decode_tok_s": 7.7},
         refs={"pr": 37, "commit": "bb351e6"}, note="Warm cache")
    seed("bench.sweep", "fail", ts="2026-10-10T11:00:00Z", actor="sonnet-2", tier="sonnet", task="O17",
         node="pod-b", refs={"pr": 38})
    seed("smoke-lora", "ok", ts="2026-10-10T12:00:00Z", metrics={"train.samples_per_s": 3.5})
    seed("ready", "ok", ts="2026-10-10T13:00:00Z", actor="fable-1", tier="fable", task=None)


def seqs(answer):
    return [row["seq"] for row in answer["rows"]]


def without(row, *names):
    return {key: value for key, value in row.items() if key not in names}


# --- ledger add ---------------------------------------------------------------------------

def test_add_writes_one_row_and_answers_its_seq(isolated, capsys):
    code, answer, err = run(capsys, "add", "bench.run", "ok")
    assert code == 0 and err == ""
    assert answer == {"ok": True, "outcome": "added", "seq": 1, "ledger": str(isolated / "ledger.jsonl")}
    assert without(ledger.tail(1)[0], "ts") == {
        "seq": 1, "actor": "unknown", "tier": None, "task": None, "verb": "bench.run", "outcome": "ok",
        "metrics": {}, "refs": NO_REFS}


def test_add_takes_the_writer_from_the_environment_and_every_option_into_the_row(monkeypatch, capsys):
    monkeypatch.setenv("LAB_ACTOR", "opus-3")
    monkeypatch.setenv("LAB_TIER", "Opus")
    monkeypatch.setenv("LAB_TASK", "O16")
    code, answer, _ = run(
        capsys, "add", "bench.run", "ok",
        "--metric", "teacher.decode_tok_s=7.7", "--metric", "teacher.watts_avg=202",
        "--unit", "teacher.decode_tok_s=tok/s",
        "--ref", "pr=37", "--ref", "commit=1234567", "--ref", "slot=",
        "--node", "pod-a", "--condition", "batch=4", "--condition", "model=Qwen3.8-27B",
        "--condition", "thinking=false", "--source", "run 3", "--note", "warm cache", "--low-confidence")
    assert (code, answer["seq"]) == (0, 1)
    assert without(ledger.tail(1)[0], "ts") == {
        "seq": 1, "actor": "opus-3", "tier": "opus", "task": "O16", "verb": "bench.run", "outcome": "ok",
        "metrics": {"teacher.decode_tok_s": 7.7, "teacher.watts_avg": 202},
        "refs": {"pr": 37, "commit": "1234567", "slot": None, "run_id": None},
        "node": "pod-a", "units": {"teacher.decode_tok_s": "tok/s"},
        "conditions": {"batch": 4, "model": "Qwen3.8-27B", "thinking": False},
        "low_confidence": True, "source": "run 3", "note": "warm cache"}


def test_add_task_flag_beats_the_environment(monkeypatch, capsys):
    monkeypatch.setenv("LAB_TASK", "O1")
    run(capsys, "add", "a.b", "ok", "--task", "O99")
    assert ledger.tail(1)[0]["task"] == "O99"


def test_add_has_no_flag_to_write_as_someone_else(capsys):
    with pytest.raises(SystemExit) as exited:
        ledger_cli.add_main(["a.b", "ok", "--actor", "somebody-else"])
    assert exited.value.code == 2
    assert "unrecognized arguments: --actor somebody-else" in capsys.readouterr().err
    assert ledger.read() == []


@pytest.mark.parametrize("flag, message", [
    (["--metric", "teacher.decode_tok_s=fast"], "must be a number"),
    (["--metric", "teacher.decode_tok_s=nan"], "must be finite"),
    (["--metric", "teacher.decode_tok_s"], "must look like NAME=VALUE"),
    (["--metric", "a.b=1", "--metric", "a.b=2"], "--metric a.b is given twice"),
    (["--ref", "pr=1", "--ref", "pr=2"], "--ref pr is given twice"),
])
def test_add_refuses_bad_arguments_with_exit_code_2_and_writes_nothing(flag, message, capsys):
    code = lab_cli.main(["ledger", "add", "a.b", "ok", *flag])
    assert code == 2
    assert message in capsys.readouterr().err
    assert ledger.read() == []


def test_add_refuses_a_metric_int_beyond_float_range_with_exit_code_2_and_writes_nothing(capsys):
    huge = "1" + "0" * 399
    code = lab_cli.main(["ledger", "add", "bench.run", "ok", "--metric", f"teacher.decode_tok_s={huge}"])
    captured = capsys.readouterr()
    assert code == 2, (code, captured.out, captured.err)
    assert "must be finite" in captured.err
    assert ledger.read() == []


def test_add_passes_the_ledgers_refusal_on_and_writes_nothing(capsys):
    code, answer, _ = run(capsys, "add", "bench.run", "ok", "--unit", "teacher.decode_tok_s=tok/s")
    assert code == 1
    assert answer == {"ok": False, "outcome": "refused",
                      "error": "ledger row refused: units names 'teacher.decode_tok_s', which is not a metric of this row",
                      "problems": ["units names 'teacher.decode_tok_s', which is not a metric of this row"]}
    assert ledger.read() == []


def test_add_refuses_a_blank_verb(capsys):
    code, answer, _ = run(capsys, "add", " ", "ok")
    assert (code, answer["outcome"], answer["problems"]) == (1, "refused", ["verb must be a non-empty string, not ' '"])


def test_add_answers_error_when_the_ledger_cannot_be_written(tmp_path, monkeypatch, capsys):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where the state dir should be", encoding="utf-8")
    monkeypatch.setenv("LAB_STATE_DIR", str(blocker / "state"))
    code, answer, _ = run(capsys, "add", "bench.run", "ok")
    assert (code, answer["ok"], answer["outcome"]) == (1, False, "error")
    assert answer["error"].startswith(("FileExistsError", "NotADirectoryError"))


# --- ledger tail --------------------------------------------------------------------------

def test_tail_of_an_empty_ledger_is_empty(capsys):
    code, answer, _ = run(capsys, "tail")
    assert (code, answer) == (0, {"ok": True, "outcome": "tail", "count": 0, "rows": []})


def test_tail_prints_the_last_ten_rows_oldest_first_by_default(capsys):
    for number in range(12):
        seed(f"step.{number}")
    code, answer, _ = run(capsys, "tail")
    assert (code, answer["outcome"], answer["count"]) == (0, "tail", 10)
    assert seqs(answer) == list(range(3, 13))
    assert [row["verb"] for row in answer["rows"]][:2] == ["step.2", "step.3"]


def test_tail_takes_a_number(capsys):
    seed_standard()
    assert seqs(run(capsys, "tail", "2")[1]) == [3, 4]
    assert run(capsys, "tail", "0")[1]["rows"] == []
    assert seqs(run(capsys, "tail", "99")[1]) == [1, 2, 3, 4]


@pytest.mark.parametrize("number", ["-1", "many", "1.5"])
def test_tail_refuses_a_number_that_is_not_a_count(number, capsys):
    assert lab_cli.main(["ledger", "tail", number]) == 2
    assert "python -m lab ledger tail: error" in capsys.readouterr().err


# --- ledger grep --------------------------------------------------------------------------

def test_grep_without_filters_prints_every_row_oldest_first(capsys):
    seed_standard()
    code, answer, _ = run(capsys, "grep")
    assert (code, answer["outcome"], answer["matches"], answer["count"], answer["truncated"]) == (0, "found", 4, 4, False)
    assert seqs(answer) == [1, 2, 3, 4]


def test_grep_on_a_ledger_that_does_not_exist_finds_none(capsys):
    code, answer, _ = run(capsys, "grep", "--verb", "bench.run")
    assert (code, answer) == (0, {"ok": True, "outcome": "none", "matches": 0, "count": 0, "truncated": False, "rows": []})


@pytest.mark.parametrize("flags, expected", [
    (["--verb", "bench.run"], [1]),
    (["--verb", "bench.*"], [1, 2]),
    (["--verb", "smoke-?ora"], [3]),
    (["--verb", "no.such.verb"], []),
    (["--outcome", "fail"], [2]),
    (["--outcome", "ok"], [1, 3, 4]),
    (["--actor", "opus-1"], [1, 3]),
    (["--actor", "*-2"], [2]),
    (["--tier", "sonnet"], [2]),
    (["--tier", "OPUS"], [1, 3]),
    (["--task", "O17"], [2]),
    (["--task", "O16"], [1, 3]),
    (["--node", "pod-a"], [1]),
    (["--node", "pod-?"], [1, 2]),
    (["--ref", "pr=37"], [1]),
    (["--ref", "pr=37", "--ref", "commit=bb351e6"], [1]),
    (["--ref", "pr=37", "--ref", "commit=other"], []),
    (["--ref", "pr=38"], [2]),
    (["--metric", "teacher.decode_tok_s"], [1]),
    (["--metric", "no.such.metric"], []),
    (["--verb", "bench.*", "--outcome", "ok"], [1]),
    (["--tier", "opus", "--metric", "train.samples_per_s"], [3]),
])
def test_grep_filters_by_field_and_combines_them_with_and(flags, expected, capsys):
    seed_standard()
    code, answer, _ = run(capsys, "grep", *flags)
    assert code == 0
    assert seqs(answer) == expected
    assert answer["outcome"] == ("found" if expected else "none")


@pytest.mark.parametrize("flags, expected", [
    (["--since", "2026-10-10T11:00:00Z"], [2, 3, 4]),
    (["--since", "2026-10-10T07:00:00-04:00"], [2, 3, 4]),
    (["--since", "2026-10-10T11:00:00"], [2, 3, 4]),
    (["--before", "2026-10-10T11:00:00Z"], [1]),
    (["--since", "2026-10-10T10:30:00Z", "--before", "2026-10-10T13:00:00Z"], [2, 3]),
    (["--since", "2026-10-10"], [1, 2, 3, 4]),
    (["--before", "2026-10-10"], []),
])
def test_grep_since_is_inclusive_and_before_is_not(flags, expected, capsys):
    seed_standard()
    assert seqs(run(capsys, "grep", *flags)[1]) == expected


def test_grep_reads_an_age_as_that_long_before_now(capsys):
    now = dt.datetime.now(dt.timezone.utc)

    def stamp(ago):
        return (now - ago).strftime("%Y-%m-%dT%H:%M:%SZ")

    seed("old", ts=stamp(dt.timedelta(hours=3)))
    seed("recent", ts=stamp(dt.timedelta(minutes=30)))
    seed("fresh", ts=stamp(dt.timedelta(minutes=2)))
    assert [row["verb"] for row in run(capsys, "grep", "--since", "1h")[1]["rows"]] == ["recent", "fresh"]
    assert [row["verb"] for row in run(capsys, "grep", "--before", "10m")[1]["rows"]] == ["old", "recent"]
    assert [row["verb"] for row in run(capsys, "grep", "--since", "2h", "--before", "10m")[1]["rows"]] == ["recent"]
    assert [row["verb"] for row in run(capsys, "grep", "--since", "90s")[1]["rows"]] == []


def test_grep_pattern_is_a_case_insensitive_regular_expression_over_the_row_text(capsys):
    seed_standard()
    assert seqs(run(capsys, "grep", "warm cache")[1]) == [1]
    assert seqs(run(capsys, "grep", "pod-[ab]")[1]) == [1, 2]
    assert seqs(run(capsys, "grep", '"outcome": "fail"')[1]) == [2]
    assert seqs(run(capsys, "grep", "pod-[ab]", "--outcome", "fail")[1]) == [2]
    assert run(capsys, "grep", "nothing like this")[1]["outcome"] == "none"


def test_grep_refuses_a_pattern_that_is_not_a_regular_expression(capsys):
    assert lab_cli.main(["ledger", "grep", "("]) == 2
    assert "is not a regular expression" in capsys.readouterr().err


@pytest.mark.parametrize("flags", [["--since", "yesterday"], ["--before", "5x"], ["--limit", "-3"], ["--limit", "all"]])
def test_grep_refuses_a_time_or_limit_it_cannot_read(flags, capsys):
    assert lab_cli.main(["ledger", "grep", *flags]) == 2
    assert "python -m lab ledger grep: error" in capsys.readouterr().err


def test_grep_keeps_the_newest_rows_when_it_cuts_to_the_limit(capsys):
    for number in range(5):
        seed(f"step.{number}")
    code, answer, _ = run(capsys, "grep", "--limit", "2")
    assert (code, answer["matches"], answer["count"], answer["truncated"], seqs(answer)) == (0, 5, 2, True, [4, 5])
    assert seqs(run(capsys, "grep", "--limit", "0")[1]) == [1, 2, 3, 4, 5]
    assert run(capsys, "grep", "--limit", "5")[1]["truncated"] is False


def test_grep_prints_at_most_fifty_rows_unless_told_otherwise(capsys):
    for number in range(60):
        seed("step.run")
    answer = run(capsys, "grep")[1]
    assert (answer["matches"], answer["count"], answer["truncated"]) == (60, 50, True)
    assert seqs(answer) == list(range(11, 61))


# --- the verb: help, unknown sub-verbs, STOP, new sub-verbs ----------------------------------

@pytest.mark.parametrize("words", [[], ["-h"], ["--help"], ["help"]])
def test_ledger_without_a_sub_verb_or_with_help_lists_the_sub_verbs(words, capsys):
    assert lab_cli.main(["ledger", *words]) == 0
    out, err = capsys.readouterr()
    assert err == ""
    assert out.splitlines()[0].startswith("usage: python -m lab ledger <sub-verb> [args]")
    assert [line.split()[0] for line in out.splitlines()[3:]] == ["add", "backfill", "grep", "tail"]
    assert "Append one row to the ledger." in out


def test_an_unknown_sub_verb_answers_with_the_ones_that_exist(capsys):
    code, answer, _ = run(capsys, "rewrite")
    assert code == 1
    assert answer == {"ok": False, "outcome": "unknown_subverb", "error": "unknown ledger sub-verb 'rewrite'",
                      "subverbs": ["add", "backfill", "grep", "tail"]}


def test_each_sub_verb_has_its_own_help(capsys):
    for name in ("add", "grep", "tail"):
        with pytest.raises(SystemExit) as exited:
            ledger_cli.main([name, "-h"])
        assert exited.value.code == 0
        assert f"usage: python -m lab ledger {name}" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [["add", "bench.run", "ok"], ["tail"], ["grep"]])
def test_the_stop_file_refuses_every_ledger_sub_verb_before_it_runs(argv, isolated, capsys):
    isolated.mkdir(parents=True)
    (isolated / "STOP").write_text("maintenance\n", encoding="utf-8")
    code = lab_cli.main(["ledger", *argv])
    out, _ = capsys.readouterr()
    assert code == 1
    assert json.loads(out)["outcome"] == "stopped"
    assert ledger.read() == []


def probe_sub(argv):
    """A stand-in for a sub-verb that a later task registers."""
    return {"ok": True, "outcome": "probed", "argv": list(argv)}


def test_a_sub_verb_added_to_the_table_gets_the_rest_of_argv(monkeypatch, capsys):
    monkeypatch.setitem(ledger_cli.SUBVERBS, "probe", f"{__name__}:probe_sub")
    code, answer, _ = run(capsys, "probe", "--flag", "value")
    assert (code, answer) == (0, {"ok": True, "outcome": "probed", "argv": ["--flag", "value"]})
    listing = ledger_cli.help_text().splitlines()
    assert [line.split()[0] for line in listing[3:]] == ["add", "backfill", "grep", "tail", "probe"]
    assert listing[7].endswith("A stand-in for a sub-verb that a later task registers.")


def test_a_sub_verb_that_cannot_be_loaded_answers_error_without_breaking_the_others(monkeypatch, capsys):
    monkeypatch.setitem(ledger_cli.SUBVERBS, "broken", "lab.machine.no_such_module:main")
    code, answer, _ = run(capsys, "broken")
    assert (code, answer["outcome"]) == (1, "error")
    assert answer["error"].startswith("cannot load lab.machine.no_such_module:main: ModuleNotFoundError")
    assert run(capsys, "tail")[0] == 0
    assert "(cannot load: ModuleNotFoundError" in ledger_cli.help_text()


# --- the registry line ----------------------------------------------------------------------

def test_the_verb_registry_names_ledger_once():
    """The registry is read from the modules' CLI_VERB declarations (lab/__main__.py, O70): ledger_cli
    declares the verb, the scan finds it once, and a second module declaring it would raise."""
    assert lab_cli.declared((REPO_ROOT / "lab" / "machine" / "ledger_cli.py").read_text(encoding="utf-8"),
                            "lab.machine.ledger_cli") == {"ledger": "main"}
    assert lab_cli.VERBS["ledger"] == "lab.machine.ledger_cli:main"
    assert list(lab_cli.VERBS) == sorted(lab_cli.VERBS)


def test_the_dispatcher_help_lists_the_ledger_verb():
    line = next(line for line in lab_cli.help_text().splitlines() if line.strip().startswith("ledger"))
    assert line.split(None, 1)[1] == "Add to, tail and search the lab ledger (ledger add, tail, grep)."


# --- through a real process -----------------------------------------------------------------

def test_the_ledger_verb_works_from_the_command_line(isolated):
    env = dict(os.environ, LAB_STATE_DIR=str(isolated), LAB_ACTOR="sonnet-9", LAB_TIER="sonnet")

    def call(*argv):
        done = subprocess.run([sys.executable, "-m", "lab", "ledger", *argv], cwd=REPO_ROOT, env=env,
                              capture_output=True, text=True, timeout=60)
        return done.returncode, done.stdout, done.stderr

    code, out, err = call("add", "bench.run", "ok", "--metric", "teacher.decode_tok_s=7.7", "--ref", "pr=37")
    assert (code, err) == (0, "")
    assert json.loads(out)["seq"] == 1
    code, out, err = call("grep", "--actor", "sonnet-9", "--ref", "pr=37")
    assert (code, err) == (0, "")
    answer = json.loads(out)
    assert [(row["seq"], row["verb"], row["metrics"]) for row in answer["rows"]] == [
        (1, "bench.run", {"teacher.decode_tok_s": 7.7})]
    code, out, err = call("tail", "x")
    assert code == 2
    assert "argument N: 'x' is not a whole number" in err


# --- the ledger verb leans on the dispatcher's own load, summary and help words ---------------

def test_ledger_cli_dispatches_through_the_dispatchers_load(monkeypatch, capsys):
    seen = []

    def stub_load(target):
        seen.append(target)
        return lambda argv: {"ok": True, "outcome": "stubbed"}

    monkeypatch.setattr(lab_cli, "load", stub_load)
    assert ledger_cli.main(["tail"]) == {"ok": True, "outcome": "stubbed"}
    assert seen == ["lab.machine.ledger_cli:tail_main"]


def test_ledger_cli_help_summaries_come_from_the_dispatchers_summary(monkeypatch):
    monkeypatch.setattr(lab_cli, "summary", lambda target: "STUB")
    listing = ledger_cli.help_text().splitlines()[3:]
    assert [line.split()[0] for line in listing] == list(ledger_cli.SUBVERBS)
    assert all("STUB" in line for line in listing)


def test_ledger_cli_has_no_private_copies_of_the_dispatcher_helpers(capsys):
    assert not hasattr(ledger_cli, "_load")
    assert not hasattr(ledger_cli, "_summary")
    assert not hasattr(ledger_cli, "HELP_WORDS")
    for word in ("-h", "help"):
        assert ledger_cli.main([word]) == 0
        out, err = capsys.readouterr()
        assert err == ""
        assert out.splitlines()[0].startswith("usage: python -m lab ledger <sub-verb> [args]")
