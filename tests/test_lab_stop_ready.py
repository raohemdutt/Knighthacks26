"""Tests for the lab verb dispatcher (python -m lab), the STOP rail and the ready preflight.

Every test gets its own state dir and sees no real secrets, no .env file and no hub:
credentials are fake and the hub is a stand-in HTTP server on 127.0.0.1.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from lab import __main__ as lab_cli
from lab.machine import stop

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_lab(tmp_path, monkeypatch):
    """A private state dir, no .env files, no hub credentials, and loopback exempt from proxies."""
    state = tmp_path / "state"
    monkeypatch.setenv("LAB_STATE_DIR", str(state))
    monkeypatch.setenv("LAB_ENV_FILES", "")
    for name in list(os.environ):
        if name.startswith("AUP_HUB_") or name == "LAB_HUB_URL":
            monkeypatch.delenv(name)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    return state


def write_stop(state_dir: Path, text: str) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "STOP").write_text(text, encoding="utf-8")


def one_json_line(text: str) -> dict:
    lines = text.splitlines()
    assert len(lines) == 1, f"expected exactly one stdout line, got {lines!r}"
    return json.loads(lines[0])


def run_cli(argv, capsys):
    code = lab_cli.main(argv)
    out, err = capsys.readouterr()
    return code, out, err


def run_module(*args):
    return subprocess.run(
        [sys.executable, "-m", "lab", *args],
        cwd=REPO_ROOT,
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=60,
    )


# --- verbs the dispatcher tests register under throwaway names -------------------------

PROBE_CALLS: list[list[str]] = []


def probe_verb(argv):
    """A stand-in for a verb that a later task registers."""
    PROBE_CALLS.append(list(argv))
    return {"ok": True, "outcome": "probed", "argv": list(argv)}


def verb_returns_code(argv):
    """Returns the exit code it is given."""
    return int(argv[0])


def verb_prints_its_own_line(argv):
    """Prints its own JSON line and returns 0."""
    print(json.dumps({"ok": True, "outcome": "own"}))
    return 0


def verb_returns_nothing(argv):
    """Returns None."""


def verb_returns_a_plain_dict(argv):
    """Returns a dict with no ok key."""
    return {"count": 3}


def verb_fails_with_a_dict(argv):
    """Returns a dict with ok false."""
    return {"ok": False, "outcome": "nope"}


def verb_returns_text(argv):
    """Returns a str."""
    return "oops"


def verb_crashes(argv):
    """Raises."""
    raise RuntimeError("boom")


def verb_exits_with_text(argv):
    """Exits with a message."""
    sys.exit("bad usage")


def verb_is_interrupted(argv):
    """Raises KeyboardInterrupt."""
    raise KeyboardInterrupt


def register(monkeypatch, name, func):
    monkeypatch.setitem(lab_cli.VERBS, name, f"{__name__}:{func.__name__}")


@pytest.fixture
def probe(monkeypatch):
    PROBE_CALLS.clear()
    register(monkeypatch, "probe", probe_verb)
    return PROBE_CALLS


# --- the STOP file ---------------------------------------------------------------------


def test_default_state_dir_is_lab_state_beside_the_package(monkeypatch):
    monkeypatch.delenv("LAB_STATE_DIR")
    assert stop.state_dir() == REPO_ROOT / "lab" / "state"
    assert stop.stop_file() == REPO_ROOT / "lab" / "state" / "STOP"


def test_lab_state_dir_moves_the_stop_file(isolated_lab):
    assert stop.stop_file() == isolated_lab / "STOP"


def test_not_stopped_without_a_stop_file():
    assert stop.stopped() is False
    assert stop.stop_reason() == ""


def test_stopped_while_the_stop_file_exists_and_its_text_is_the_reason(isolated_lab):
    write_stop(isolated_lab, "pod credits low\n")
    assert stop.stopped() is True
    assert stop.stop_reason() == "pod credits low"
    (isolated_lab / "STOP").unlink()
    assert stop.stopped() is False


def test_an_empty_stop_file_still_stops_with_a_default_reason(isolated_lab):
    write_stop(isolated_lab, "")
    assert stop.stopped() is True
    assert stop.stop_reason() == "no reason given"


def test_a_multi_line_reason_is_shown_on_one_line(isolated_lab):
    write_stop(isolated_lab, "  first   reason \n\n second reason\n")
    assert stop.stop_reason() == "first reason | second reason"


def test_a_very_long_reason_is_cut(isolated_lab):
    write_stop(isolated_lab, "x" * 1000)
    assert stop.stop_reason() == "x" * 400


def test_a_stop_file_that_cannot_be_read_still_stops(isolated_lab):
    (isolated_lab / "STOP").mkdir(parents=True)
    assert stop.stopped() is True
    assert stop.stop_reason().startswith("(STOP file unreadable: ")


def test_a_stop_file_that_cannot_be_checked_still_stops(monkeypatch):
    real_lstat = os.lstat

    def deny_the_stop_file(path, *args, **kwargs):
        if Path(path) == stop.stop_file():
            raise PermissionError(13, "Permission denied")
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(os, "lstat", deny_the_stop_file)
    assert stop.stopped() is True
    assert stop.stop_reason() == "(STOP file could not be checked: PermissionError)"


# --- guard() and @guarded --------------------------------------------------------------


def test_guard_does_nothing_while_not_stopped(capsys):
    assert stop.guard() is None
    assert capsys.readouterr() == ("", "")


def test_guard_exits_stopped_with_code_1_and_the_reason(isolated_lab, capsys):
    write_stop(isolated_lab, "pod credits low")
    with pytest.raises(SystemExit) as caught:
        stop.guard()
    assert caught.value.code == 1
    assert str(caught.value) == "stopped"
    out, err = capsys.readouterr()
    assert one_json_line(out) == {
        "ok": False,
        "outcome": "stopped",
        "reason": "pod credits low",
        "file": str(isolated_lab / "STOP"),
    }
    assert err == "stopped: pod credits low\n"


def test_guarded_function_does_not_run_while_stopped(isolated_lab):
    calls = []

    @stop.guarded
    def work(x, *, y=2):
        calls.append((x, y))
        return x + y

    assert work(1, y=5) == 6
    assert calls == [(1, 5)]
    write_stop(isolated_lab, "halt")
    with pytest.raises(SystemExit) as caught:
        work(1)
    assert caught.value.code == 1
    assert calls == [(1, 5)]
    assert work.__name__ == "work"


# --- python -m lab stop / resume -------------------------------------------------------


def test_stop_verb_writes_the_reason_into_the_stop_file(isolated_lab, capsys):
    code, out, _ = run_cli(["stop", "pod", "credits", "low"], capsys)
    assert code == 0
    assert one_json_line(out) == {
        "ok": True,
        "outcome": "stop_set",
        "already_stopped": False,
        "reason": "pod credits low",
        "file": str(isolated_lab / "STOP"),
    }
    assert (isolated_lab / "STOP").read_text(encoding="utf-8") == "pod credits low\n"
    assert stop.stopped() is True


def test_stop_verb_without_a_reason_says_so(isolated_lab, capsys):
    code, out, _ = run_cli(["stop"], capsys)
    assert code == 0
    assert one_json_line(out)["reason"] == "no reason given"
    assert (isolated_lab / "STOP").read_text(encoding="utf-8") == "no reason given\n"


def test_a_second_stop_keeps_the_first_reason_and_adds_its_own(isolated_lab, capsys):
    run_cli(["stop", "first"], capsys)
    code, out, _ = run_cli(["stop", "second"], capsys)
    assert code == 0
    result = one_json_line(out)
    assert result["already_stopped"] is True
    assert result["reason"] == "first | second"
    assert (isolated_lab / "STOP").read_text(encoding="utf-8") == "first\nsecond\n"


def test_stop_verb_reports_a_state_dir_it_cannot_write(tmp_path, monkeypatch, capsys):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where the state dir should be", encoding="utf-8")
    monkeypatch.setenv("LAB_STATE_DIR", str(blocker / "state"))
    code, out, _ = run_cli(["stop", "why"], capsys)
    assert code == 1
    result = one_json_line(out)
    assert result["ok"] is False
    assert result["outcome"] == "error"
    assert result["error"].startswith("cannot write ")


def test_resume_verb_removes_the_stop_file_and_reports_what_it_cleared(isolated_lab, capsys):
    write_stop(isolated_lab, "pod credits low")
    code, out, _ = run_cli(["resume"], capsys)
    assert code == 0
    assert one_json_line(out) == {
        "ok": True,
        "outcome": "resumed",
        "reason": "pod credits low",
        "file": str(isolated_lab / "STOP"),
    }
    assert stop.stopped() is False


def test_resume_without_a_stop_file_is_a_no_op(isolated_lab, capsys):
    code, out, _ = run_cli(["resume"], capsys)
    assert code == 0
    assert one_json_line(out) == {"ok": True, "outcome": "not_stopped", "file": str(isolated_lab / "STOP")}


def test_resume_fails_loudly_when_the_stop_file_cannot_be_removed(isolated_lab, capsys):
    (isolated_lab / "STOP").mkdir(parents=True)
    code, out, _ = run_cli(["resume"], capsys)
    assert code == 1
    result = one_json_line(out)
    assert result["ok"] is False
    assert result["outcome"] == "error"
    assert result["error"].startswith("cannot remove ")
    assert stop.stopped() is True


# --- the dispatcher honors STOP for every verb -----------------------------------------


def test_a_registered_verb_is_refused_while_stopped_and_runs_again_after_resume(isolated_lab, probe, capsys):
    run_cli(["stop", "halt everything"], capsys)

    code, out, err = run_cli(["probe", "x"], capsys)

    assert code == 1
    assert one_json_line(out) == {
        "ok": False,
        "outcome": "stopped",
        "reason": "halt everything",
        "file": str(isolated_lab / "STOP"),
    }
    assert err == "stopped: halt everything\n"
    assert probe == []

    run_cli(["resume"], capsys)
    code, out, _ = run_cli(["probe", "x"], capsys)
    assert code == 0
    assert one_json_line(out) == {"ok": True, "outcome": "probed", "argv": ["x"]}
    assert probe == [["x"]]


def test_stop_and_resume_themselves_run_while_stopped(isolated_lab, capsys):
    write_stop(isolated_lab, "already stopped")
    assert run_cli(["stop", "again"], capsys)[0] == 0
    assert run_cli(["resume"], capsys)[0] == 0
    assert stop.stopped() is False


def test_help_works_while_stopped(isolated_lab, capsys):
    write_stop(isolated_lab, "halt")
    code, out, _ = run_cli([], capsys)
    assert code == 0
    assert out.startswith("usage: python -m lab <verb>")


# --- the dispatcher contract with verbs ------------------------------------------------


def test_a_verb_that_returns_an_int_sets_the_exit_code(monkeypatch, capsys):
    register(monkeypatch, "code", verb_returns_code)
    assert run_cli(["code", "7"], capsys) == (7, "", "")


def test_a_verb_that_prints_its_own_line_is_left_alone(monkeypatch, capsys):
    register(monkeypatch, "own", verb_prints_its_own_line)
    assert run_cli(["own"], capsys) == (0, '{"ok": true, "outcome": "own"}\n', "")


def test_a_verb_that_returns_nothing_succeeds_silently(monkeypatch, capsys):
    register(monkeypatch, "nothing", verb_returns_nothing)
    assert run_cli(["nothing"], capsys) == (0, "", "")


def test_a_dict_without_ok_counts_as_success_and_is_printed(monkeypatch, capsys):
    register(monkeypatch, "plain", verb_returns_a_plain_dict)
    code, out, _ = run_cli(["plain"], capsys)
    assert code == 0
    assert one_json_line(out) == {"count": 3}


def test_a_dict_with_ok_false_exits_1_and_is_printed(monkeypatch, capsys):
    register(monkeypatch, "fails", verb_fails_with_a_dict)
    code, out, _ = run_cli(["fails"], capsys)
    assert code == 1
    assert one_json_line(out) == {"ok": False, "outcome": "nope"}


def test_a_verb_that_returns_something_else_is_reported_as_an_error(monkeypatch, capsys):
    register(monkeypatch, "text", verb_returns_text)
    code, out, _ = run_cli(["text"], capsys)
    assert code == 1
    assert one_json_line(out) == {
        "ok": False,
        "outcome": "error",
        "error": "verb returned str; expected a dict, an int or None",
        "verb": "text",
    }


def test_a_crashing_verb_gives_one_json_error_line_and_the_traceback_on_stderr(monkeypatch, capsys):
    register(monkeypatch, "crash", verb_crashes)
    code, out, err = run_cli(["crash"], capsys)
    assert code == 1
    assert one_json_line(out) == {"ok": False, "outcome": "error", "error": "RuntimeError: boom", "verb": "crash"}
    assert "Traceback (most recent call last)" in err


def test_sys_exit_with_text_goes_to_stderr_and_exits_1(monkeypatch, capsys):
    register(monkeypatch, "exits", verb_exits_with_text)
    assert run_cli(["exits"], capsys) == (1, "", "bad usage\n")


def test_an_interrupted_verb_answers_interrupted(monkeypatch, capsys):
    register(monkeypatch, "interrupted", verb_is_interrupted)
    code, out, _ = run_cli(["interrupted"], capsys)
    assert code == 1
    assert one_json_line(out) == {
        "ok": False,
        "outcome": "interrupted",
        "error": "KeyboardInterrupt",
        "verb": "interrupted",
    }


@pytest.mark.parametrize("target", ["lab.machine.no_such_module:main", "lab.machine.stop:no_such_function"])
def test_a_verb_that_cannot_be_loaded_is_an_error_not_a_crash(monkeypatch, capsys, target):
    monkeypatch.setitem(lab_cli.VERBS, "ghost", target)
    code, out, _ = run_cli(["ghost"], capsys)
    assert code == 1
    result = one_json_line(out)
    assert result["outcome"] == "error"
    assert result["error"].startswith(f"cannot load {target}: ")
    assert result["verb"] == "ghost"


def test_an_unknown_verb_lists_the_verbs(capsys):
    code, out, _ = run_cli(["nope"], capsys)
    assert code == 1
    assert one_json_line(out) == {
        "ok": False,
        "outcome": "unknown_verb",
        "error": "unknown verb 'nope'",
        "verbs": sorted(lab_cli.VERBS),
    }


def test_no_arguments_and_help_list_every_registered_verb(capsys):
    for argv in ([], ["--help"], ["-h"], ["help"]):
        code, out, _ = run_cli(argv, capsys)
        assert code == 0
        assert out.startswith("usage: python -m lab <verb>")
        for verb in lab_cli.VERBS:
            assert re.search(rf"^  {re.escape(verb)} ", out, re.MULTILINE), f"{verb} missing from help for {argv}"


def test_registry_is_alphabetical_with_a_module_and_function_per_verb():
    verbs = list(lab_cli.VERBS)
    assert verbs == sorted(verbs)
    for verb, target in lab_cli.VERBS.items():
        assert re.fullmatch(r"[a-z][a-z0-9-]*", verb), verb
        assert re.fullmatch(r"lab(\.[a-z_][a-z0-9_]*)+:[a-z_][a-z0-9_]*", target), target


@pytest.mark.parametrize("verb", ["resume", "stop"])
def test_stop_verbs_are_registered_and_answer_help(verb, capsys):
    assert verb in lab_cli.VERBS
    code, out, _ = run_cli([verb, "-h"], capsys)
    assert code == 0
    assert f"usage: python -m lab {verb}" in out


# --- the real entry point, in a child process ------------------------------------------


def test_python_dash_m_lab_stop_then_resume(isolated_lab):
    first = run_module("stop", "rehearsal", "halt")
    assert first.returncode == 0
    assert one_json_line(first.stdout)["reason"] == "rehearsal halt"
    assert (isolated_lab / "STOP").read_text(encoding="utf-8") == "rehearsal halt\n"

    second = run_module("resume")
    assert second.returncode == 0
    assert one_json_line(second.stdout)["outcome"] == "resumed"
    assert not (isolated_lab / "STOP").exists()


def test_python_dash_m_lab_with_an_unknown_verb_exits_1():
    result = run_module("nope")
    assert result.returncode == 1
    assert one_json_line(result.stdout)["outcome"] == "unknown_verb"


def test_python_dash_m_lab_with_no_arguments_prints_the_help():
    result = run_module()
    assert result.returncode == 0
    assert result.stdout.startswith("usage: python -m lab <verb>")
