"""record() and @recorded: the row they write, who it is written as, how the end of a call becomes its
outcome, and that a ledger which cannot be written never changes what the call does. Every test works in a
temporary state dir; nothing here touches lab/state."""
from __future__ import annotations

import datetime as dt
import logging
import re
import time

import pytest

from lab import __main__ as lab_cli
from lab.machine import ledger, stop
from lab.machine.lock import FileLock
from lab.machine.record import (NOTE_CHARS, UNKNOWN_ACTOR, current_actor, current_task, current_tier, record,
                                recorded)

NO_REFS = {"pr": None, "commit": None, "slot": None, "run_id": None}
STAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """A private state dir and no writer identity in the environment."""
    state = tmp_path / "state"
    monkeypatch.setenv("LAB_STATE_DIR", str(state))
    for name in ("LAB_ACTOR", "LAB_TIER", "LAB_TASK"):
        monkeypatch.delenv(name, raising=False)
    return state


def only_row():
    rows = ledger.read()
    assert len(rows) == 1, rows
    return rows[0]


def without(row, *names):
    return {key: value for key, value in row.items() if key not in names}


def when(text):
    return dt.datetime.fromisoformat(text)


# --- record() -----------------------------------------------------------------------------

def test_record_takes_the_writer_from_the_environment(monkeypatch):
    monkeypatch.setenv("LAB_ACTOR", "opus-3")
    monkeypatch.setenv("LAB_TIER", " Opus ")
    monkeypatch.setenv("LAB_TASK", "O16")
    seq = record("bench.run", "ok", metrics={"teacher.decode_tok_s": 7.7}, refs={"run_id": "run-0003"})
    assert seq == 1
    assert without(only_row(), "ts") == {
        "seq": 1, "actor": "opus-3", "tier": "opus", "task": "O16", "verb": "bench.run", "outcome": "ok",
        "metrics": {"teacher.decode_tok_s": 7.7}, "refs": {**NO_REFS, "run_id": "run-0003"}}


def test_record_without_an_environment_writes_an_unknown_actor_and_no_tier_or_task():
    record("bench.run", "skipped")
    assert without(only_row(), "ts") == {
        "seq": 1, "actor": UNKNOWN_ACTOR, "tier": None, "task": None, "verb": "bench.run",
        "outcome": "skipped", "metrics": {}, "refs": NO_REFS}


def test_blank_environment_values_count_as_unset(monkeypatch):
    monkeypatch.setenv("LAB_ACTOR", "   ")
    monkeypatch.setenv("LAB_TIER", "")
    monkeypatch.setenv("LAB_TASK", "\t")
    assert (current_actor(), current_tier(), current_task()) == (UNKNOWN_ACTOR, None, None)


def test_the_current_writer_can_be_read_from_a_given_mapping():
    environ = {"LAB_ACTOR": "sonnet-1", "LAB_TIER": "SONNET", "LAB_TASK": "O17"}
    assert (current_actor(environ), current_tier(environ), current_task(environ)) == ("sonnet-1", "sonnet", "O17")


def test_explicit_arguments_beat_the_environment(monkeypatch):
    monkeypatch.setenv("LAB_ACTOR", "from-env")
    monkeypatch.setenv("LAB_TIER", "fable")
    monkeypatch.setenv("LAB_TASK", "O1")
    record("x.y", "ok", actor="ilan", tier="opus", task="O2")
    row = only_row()
    assert (row["actor"], row["tier"], row["task"]) == ("ilan", "opus", "O2")


def test_record_passes_other_keywords_on_as_ledger_fields():
    record("bench.run", "ok", metrics={"teacher.decode_tok_s": 7.7}, node="pod-a",
           units={"teacher.decode_tok_s": "tok/s"}, conditions={"batch": 4}, low_confidence=True,
           source="run 3", note="warm cache")
    row = only_row()
    assert (row["node"], row["units"], row["conditions"], row["low_confidence"], row["source"], row["note"]) == (
        "pod-a", {"teacher.decode_tok_s": "tok/s"}, {"batch": 4}, True, "run 3", "warm cache")


def test_record_returns_one_more_seq_each_time():
    assert [record("a.b", "ok"), record("a.b", "ok"), record("a.c", "fail")] == [1, 2, 3]


def test_record_refuses_a_row_the_ledger_refuses_and_writes_nothing():
    with pytest.raises(ledger.RowError) as refused:
        record("bench.run", "ok", metrics={"teacher.decode_tok_s": float("nan")})
    assert refused.value.problems == ["metrics value for 'teacher.decode_tok_s' must be a finite number, not nan"]
    assert ledger.read() == []


def test_record_writes_to_the_state_dir_it_is_given(tmp_path):
    other = tmp_path / "elsewhere"
    record("bench.run", "ok", state_dir=other)
    assert [row["verb"] for row in ledger.read(state_dir=other)] == ["bench.run"]
    assert ledger.read() == []


# --- @recorded: the row -------------------------------------------------------------------

def test_a_recorded_call_returns_its_value_and_writes_one_timed_row(monkeypatch):
    monkeypatch.setenv("LAB_ACTOR", "sonnet-2")
    monkeypatch.setenv("LAB_TIER", "sonnet")

    @recorded("demo.sleep")
    def work(x):
        time.sleep(0.05)
        return f"doubled {x * 2}"

    slack = dt.timedelta(milliseconds=1)
    before = dt.datetime.now(dt.timezone.utc)
    assert work(21) == "doubled 42"
    after = dt.datetime.now(dt.timezone.utc)
    row = only_row()
    assert STAMP.fullmatch(row["started"]) and STAMP.fullmatch(row["ended"])
    started, ended = when(row["started"]), when(row["ended"])
    assert before - slack <= started <= ended <= after
    assert 0.04 <= row["duration_s"] <= (after - before).total_seconds() + 0.001
    assert abs((ended - started).total_seconds() - row["duration_s"]) <= 0.002
    assert (row["actor"], row["tier"], row["verb"], row["outcome"], row["metrics"]) == (
        "sonnet-2", "sonnet", "demo.sleep", "ok", {})
    assert "note" not in row


@pytest.mark.parametrize("result, outcome, note", [
    ({"ok": True}, "ok", None),
    ({}, "ok", None),
    ({"count": 3}, "ok", None),
    ({"ok": False}, "fail", None),
    ({"ok": True, "outcome": "clean"}, "clean", None),
    ({"ok": False, "outcome": "violations", "error": "3 violations"}, "violations", "3 violations"),
    ({"ok": False, "outcome": "  ", "error": "no hub"}, "fail", "no hub"),
    ({"ok": True, "outcome": 5}, "ok", None),
    ({"ok": True, "outcome": "done", "error": "left out when ok"}, "done", None),
])
def test_a_dict_answer_gives_the_outcome_the_dispatcher_would_report(result, outcome, note):
    @recorded("demo.verb")
    def verb(argv):
        return result

    assert verb([]) is result
    row = only_row()
    assert row["outcome"] == outcome
    assert row.get("note") == note


@pytest.mark.parametrize("result, outcome, note", [
    (0, "ok", None),
    (3, "fail", "exit code 3"),
    (-1, "fail", "exit code -1"),
    (None, "ok", None),
    (True, "ok", None),
    (False, "ok", None),
    ("text", "ok", None),
    ([], "ok", None),
])
def test_an_exit_code_or_any_other_return_value_gives_ok_or_fail(result, outcome, note):
    @recorded("demo.verb")
    def verb(argv):
        return result

    assert verb([]) is result
    row = only_row()
    assert (row["outcome"], row.get("note")) == (outcome, note)


def test_a_dict_answer_carries_its_metrics_and_refs_into_the_row():
    @recorded("bench.run")
    def verb(argv):
        return {"ok": True, "outcome": "ok", "metrics": {"teacher.decode_tok_s": 7.7, "teacher.watts_avg": 202},
                "refs": {"pr": 37, "run_id": "run-1"}}

    verb([])
    row = only_row()
    assert row["metrics"] == {"teacher.decode_tok_s": 7.7, "teacher.watts_avg": 202}
    assert row["refs"] == {**NO_REFS, "pr": 37, "run_id": "run-1"}


def test_metrics_the_ledger_refuses_are_left_out_and_the_row_says_so(caplog):
    answer = {"ok": True, "outcome": "ok", "metrics": {"teacher.decode_tok_s": "fast"}, "refs": {"pr": 37}}

    @recorded("bench.run")
    def verb(argv):
        return answer

    with caplog.at_level(logging.WARNING, logger="lab.record"):
        assert verb([]) is answer
    row = only_row()
    assert (row["outcome"], row["metrics"], row["refs"]) == ("ok", {}, NO_REFS)
    assert row["note"] == "ledger refused: metrics value for 'teacher.decode_tok_s' must be a finite number, not 'fast'"
    assert [record_.levelname for record_ in caplog.records] == ["WARNING"]


def test_a_refused_metric_keeps_the_error_the_call_reported():
    @recorded("bench.run")
    def verb(argv):
        return {"ok": False, "outcome": "error", "error": "disk full", "metrics": {"a.b": None}}

    verb([])
    assert only_row()["note"] == "disk full; ledger refused: metrics value for 'a.b' must be a finite number, not None"


# --- @recorded: how a call ends -----------------------------------------------------------

def test_an_exception_is_recorded_as_an_error_and_reaches_the_caller_unchanged():
    boom = ValueError("boom\nsecond line")

    @recorded("demo.crash")
    def crash():
        raise boom

    with pytest.raises(ValueError) as raised:
        crash()
    assert raised.value is boom
    row = only_row()
    assert (row["outcome"], row["note"]) == ("error", "ValueError: boom second line")


@pytest.mark.parametrize("code, outcome, note", [
    (None, "ok", None),
    (0, "ok", None),
    (1, "fail", "exit code 1"),
    (2, "usage", "exit code 2"),
    (7, "fail", "exit code 7"),
    ("bad thing", "fail", "bad thing"),
])
def test_a_system_exit_is_recorded_by_its_code(code, outcome, note):
    @recorded("demo.exit")
    def leave():
        raise SystemExit(code)

    with pytest.raises(SystemExit) as raised:
        leave()
    assert raised.value.code == code
    row = only_row()
    assert (row["outcome"], row.get("note")) == (outcome, note)


def test_a_keyboard_interrupt_is_recorded_and_passes_through():
    @recorded("demo.wait")
    def wait():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        wait()
    assert only_row()["outcome"] == "interrupted"


def test_a_call_refused_by_the_stop_file_is_recorded_as_stopped(isolated):
    isolated.mkdir(parents=True)
    (isolated / "STOP").write_text("maintenance window\n", encoding="utf-8")

    @recorded("demo.verb")
    def verb(argv):
        stop.guard()
        raise AssertionError("ran while the STOP file existed")

    with pytest.raises(SystemExit):
        verb([])
    row = only_row()
    assert (row["outcome"], row["note"]) == ("stopped", "maintenance window")


def test_a_long_message_is_cut_to_one_line_of_note_chars():
    @recorded("demo.crash")
    def crash():
        raise RuntimeError("x" * 2000 + "\nmore")

    with pytest.raises(RuntimeError):
        crash()
    note = only_row()["note"]
    assert len(note) == NOTE_CHARS
    assert note.startswith("RuntimeError: " + "x" * 50) and note.endswith("...") and "\n" not in note


# --- @recorded: the ledger never gets in the call's way -----------------------------------

def test_a_recorded_call_prints_nothing(capsys):
    @recorded("demo.quiet")
    def quiet():
        return 1

    assert quiet() == 1
    assert capsys.readouterr() == ("", "")


def test_a_ledger_that_cannot_be_written_does_not_change_the_call(tmp_path, monkeypatch, caplog):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where the state dir should be", encoding="utf-8")
    monkeypatch.setenv("LAB_STATE_DIR", str(blocker / "state"))

    @recorded("demo.verb")
    def verb(x):
        return x + 1

    @recorded("demo.crash")
    def crash():
        raise KeyError("k")

    with caplog.at_level(logging.WARNING, logger="lab.record"):
        assert verb(1) == 2
        with pytest.raises(KeyError):
            crash()
    assert [record_.getMessage().split(":")[0] for record_ in caplog.records] == [
        "no ledger row written for demo.verb", "no ledger row written for demo.crash"]


def test_a_lock_held_too_long_drops_the_row_and_not_the_call(caplog):
    @recorded("demo.verb", timeout=0.05)
    def verb():
        return "done"

    with FileLock(ledger.lock_path(), timeout=1):
        with caplog.at_level(logging.WARNING, logger="lab.record"):
            assert verb() == "done"
    assert ledger.read() == []
    assert len(caplog.records) == 1
    assert "no ledger row written for demo.verb: LockTimeout" in caplog.records[0].getMessage()


def test_recorded_needs_the_name_of_the_verb():
    with pytest.raises(TypeError, match="verb-name"):
        recorded(lambda: None)
    with pytest.raises(ValueError):
        recorded("  ")
    with pytest.raises(ValueError):
        recorded(5)


def test_a_generator_or_coroutine_function_is_refused_at_decoration():
    def numbers():
        yield 1

    async def later():
        return 1

    with pytest.raises(TypeError, match="numbers"):
        recorded("demo.gen")(numbers)
    with pytest.raises(TypeError, match="later"):
        recorded("demo.async")(later)


def test_the_wrapper_keeps_the_name_the_help_line_and_the_stop_exemption():
    @recorded("demo.verb")
    @stop.exempt
    def verb_main(argv):
        """Say hello."""

    assert (verb_main.__name__, verb_main.__doc__) == ("verb_main", "Say hello.")
    assert stop.is_exempt(verb_main)


def test_the_task_of_the_decorator_beats_the_environment_and_the_environment_fills_the_gap(monkeypatch):
    monkeypatch.setenv("LAB_TASK", "O16")

    @recorded("demo.own", task="O99")
    def own():
        return None

    @recorded("demo.env")
    def env():
        return None

    own()
    env()
    assert [(row["verb"], row["task"]) for row in ledger.read()] == [("demo.own", "O99"), ("demo.env", "O16")]


def test_the_decorator_can_write_to_its_own_state_dir(tmp_path):
    other = tmp_path / "elsewhere"

    @recorded("demo.verb", state_dir=other)
    def verb():
        return None

    verb()
    assert [row["verb"] for row in ledger.read(state_dir=other)] == ["demo.verb"]
    assert ledger.read() == []


def test_a_verb_that_calls_another_recorded_function_gets_both_rows_inner_first():
    @recorded("inner.step")
    def inner():
        time.sleep(0.02)

    @recorded("outer.verb")
    def outer():
        inner()
        return {"ok": True, "outcome": "ok"}

    outer()
    first, second = ledger.read()
    assert (first["seq"], first["verb"], second["seq"], second["verb"]) == (1, "inner.step", 2, "outer.verb")
    assert when(second["started"]) <= when(first["started"]) <= when(first["ended"]) <= when(second["ended"])


# --- through the dispatcher ---------------------------------------------------------------

@recorded("probe")
def probe_main(argv):
    """A verb that reports a metric."""
    return {"ok": True, "outcome": "probed", "metrics": {"probe.args": len(argv)}}


@recorded("probe-crash")
def probe_crash_main(argv):
    """A verb that crashes."""
    raise RuntimeError("boom")


def test_a_recorded_verb_runs_through_the_dispatcher_and_leaves_its_row(monkeypatch, capsys):
    monkeypatch.setitem(lab_cli.VERBS, "probe", f"{__name__}:probe_main")
    assert lab_cli.main(["probe", "a", "b"]) == 0
    out, err = capsys.readouterr()
    assert out == '{"ok": true, "outcome": "probed", "metrics": {"probe.args": 2}}\n'
    assert err == ""
    row = only_row()
    assert (row["verb"], row["outcome"], row["metrics"]) == ("probe", "probed", {"probe.args": 2})
    assert lab_cli.summary(f"{__name__}:probe_main") == "A verb that reports a metric."


def test_a_crashing_recorded_verb_still_answers_error_and_leaves_an_error_row(monkeypatch, capsys):
    monkeypatch.setitem(lab_cli.VERBS, "probe-crash", f"{__name__}:probe_crash_main")
    assert lab_cli.main(["probe-crash"]) == 1
    out, _ = capsys.readouterr()
    assert out == '{"ok": false, "outcome": "error", "error": "RuntimeError: boom", "verb": "probe-crash"}\n'
    row = only_row()
    assert (row["verb"], row["outcome"], row["note"]) == ("probe-crash", "error", "RuntimeError: boom")
