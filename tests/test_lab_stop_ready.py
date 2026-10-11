"""Tests for the lab verb dispatcher (python -m lab), the STOP rail and the ready preflight.

Every test gets its own state dir and sees no real secrets, no .env file and no hub:
credentials are fake and the hub is a stand-in HTTP server on 127.0.0.1.
"""

import http.client
import json
import os
import re
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from lab import __main__ as lab_cli
from lab.machine import ready, stop

REPO_ROOT = Path(__file__).resolve().parents[1]

# Fake credentials. Long, distinctive values let the tests prove nothing ever echoes one.
SECRETS = {
    "AUP_HUB_USER_A": "fake-user-alpha",
    "AUP_HUB_USER_B": "fake-user-bravo",
    "AUP_HUB_TOKEN_A": "fake-token-alpha-0123456789",
    "AUP_HUB_TOKEN_B": "fake-token-bravo-0123456789",
}
REQUIRED = ["AUP_HUB_TOKEN_A", "AUP_HUB_TOKEN_B", "AUP_HUB_USER_A", "AUP_HUB_USER_B"]


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


@pytest.mark.parametrize("verb", ["ready", "resume", "stop"])
def test_stop_and_ready_verbs_are_registered_and_answer_help(verb, capsys):
    assert verb in lab_cli.VERBS
    code, out, _ = run_cli([verb, "-h"], capsys)
    assert code == 0
    assert f"usage: python -m lab {verb}" in out


# --- ready: reading keys ---------------------------------------------------------------


def test_dotenv_files_are_read_the_usual_way(tmp_path):
    path = tmp_path / "x.env"
    path.write_text(
        "﻿# a comment\n"
        "\n"
        "PLAIN=value one\n"
        "export EXPORTED=exported\n"
        'DOUBLE="double quoted # not a comment"\n'
        "SINGLE='single quoted'\n"
        "TRAILING=before # a trailing comment\n"
        "  SPACED  =  spaced out  \n"
        "BLANK=\n"
        "no equals sign here\n"
        "=no key\n"
        "REPEATED=first\n"
        "REPEATED=second\n",
        encoding="utf-8",
    )
    assert ready.read_env_file(path) == {
        "PLAIN": "value one",
        "EXPORTED": "exported",
        "DOUBLE": "double quoted # not a comment",
        "SINGLE": "single quoted",
        "TRAILING": "before",
        "SPACED": "spaced out",
        "BLANK": "",
        "REPEATED": "second",
    }
    assert ready.read_env_file(tmp_path / "missing.env") == {}


def test_the_first_definition_wins_and_a_blank_value_does_not_count(tmp_path):
    first, second = tmp_path / "first.env", tmp_path / "second.env"
    first.write_text("ONE=from-first\nTWO=\n", encoding="utf-8")
    second.write_text("ONE=from-second\nTWO=from-second\nTHREE=from-second\n", encoding="utf-8")

    values, used = ready.collect_env(
        ["ONE", "TWO", "THREE", "FOUR"], {"THREE": "from-environ"}, [first, tmp_path / "missing.env", second]
    )

    assert values == {"ONE": "from-first", "TWO": "from-second", "THREE": "from-environ"}
    assert used == [first, second]


def test_env_files_default_to_the_clone_and_the_two_folders_above_it(monkeypatch):
    monkeypatch.delenv("LAB_ENV_FILES")
    assert ready.default_env_files() == [
        REPO_ROOT / ".env",
        REPO_ROOT.parent / ".env",
        REPO_ROOT.parent.parent / ".env",
    ]


def test_lab_env_files_replaces_the_search(monkeypatch, tmp_path):
    first, second = tmp_path / "a.env", tmp_path / "b.env"
    monkeypatch.setenv("LAB_ENV_FILES", os.pathsep.join([str(first), str(second)]))
    assert ready.default_env_files() == [first, second]
    monkeypatch.setenv("LAB_ENV_FILES", "")
    assert ready.default_env_files() == []


# --- ready: the checks, against a stand-in hub -----------------------------------------


class FakeHubHandler(BaseHTTPRequestHandler):
    """The hub's user endpoint as far as ready can tell: it records every request it gets."""

    def answer(self):
        hub = self.server
        hub.requests.append(
            {
                "method": self.command,
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "user_agent": self.headers.get("User-Agent"),
            }
        )
        if self.command != "GET":
            self.send_json(405, {"message": "this stand-in hub is read-only"})
        elif hub.mode == "redirect":
            self.send_response(302)
            self.send_header("Location", "/somewhere-else")
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif hub.mode == "html":
            self.send_json(200, "<html>sign in</html>", content_type="text/html")
        elif (self.headers.get("User-Agent") or "").startswith("Python-urllib"):
            self.send_json(403, {"message": "blocked by the front end"})
        else:
            prefix = "/hub/api/users/"
            user = self.path[len(prefix):] if self.path.startswith(prefix) else None
            token = hub.tokens.get(user)
            if token is None:
                self.send_json(404, {"message": "no such user"})
            elif self.headers.get("Authorization") != f"token {token}":
                self.send_json(403, {"message": "forbidden"})
            else:
                self.send_json(200, {"name": user, "servers": {}})

    def send_json(self, status, body, content_type="application/json"):
        payload = (body if isinstance(body, str) else json.dumps(body)).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = answer

    def log_message(self, *args):
        pass


@pytest.fixture
def hub():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeHubHandler)
    server.requests = []
    server.mode = "normal"
    server.tokens = {
        SECRETS["AUP_HUB_USER_A"]: SECRETS["AUP_HUB_TOKEN_A"],
        SECRETS["AUP_HUB_USER_B"]: SECRETS["AUP_HUB_TOKEN_B"],
    }
    server.url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def configure(monkeypatch, tmp_path, hub_url, values=None):
    """Put the credentials in a .env file and point ready at the hub."""
    values = SECRETS if values is None else values
    env_file = tmp_path / "secrets.env"
    env_file.write_text("".join(f"{key}={value}\n" for key, value in values.items()), encoding="utf-8")
    monkeypatch.setenv("LAB_ENV_FILES", str(env_file))
    monkeypatch.setenv("LAB_HUB_URL", hub_url)
    return env_file


def core(accounts):
    """Each account's verdict without the fields the live-hub report adds (reachable, running, ready, role, hub, url)."""
    keep = ("ok", "status", "error", "skipped")
    if "ok" in accounts:
        return {k: v for k, v in accounts.items() if k in keep}
    return {letter: {k: v for k, v in probe.items() if k in keep} for letter, probe in accounts.items()}


def assert_no_secret_values(*outputs):
    for text in outputs:
        for name, value in SECRETS.items():
            assert value not in text, f"the value of {name} leaked into: {text[:200]}"


def test_ready_passes_when_every_precondition_holds(isolated_lab, hub, monkeypatch, tmp_path, capsys):
    env_file = configure(monkeypatch, tmp_path, hub.url)

    code, out, err = run_cli(["ready"], capsys)

    assert code == 0
    verdict = one_json_line(out)
    assert verdict.pop("warnings", []) == []
    accounts = verdict["checks"]["hub"]["accounts"]
    assert core(accounts) == {"A": {"ok": True, "status": 200}, "B": {"ok": True, "status": 200}}
    assert accounts["A"] == {"ok": True, "reachable": True, "running": False, "ready": False, "status": 200,
                             "role": "pod-a", "hub": hub.url.split("//")[1], "url": hub.url}
    verdict["checks"]["hub"]["accounts"] = "ACCOUNTS"
    assert verdict == {
        "ok": True,
        "ready": True,
        "outcome": "ready",
        "reasons": [],
        "checks": {
            "stop": {"ok": True, "stopped": False, "reason": "", "file": str(isolated_lab / "STOP")},
            "state_dir": {"ok": True, "path": str(isolated_lab), "writable": True},
            "env": {"ok": True, "present": REQUIRED, "missing": [], "files": [str(env_file)]},
            "hub": {"ok": True, "url": hub.url, "hubs": [hub.url.split("//")[1]], "accounts": "ACCOUNTS"},
        },
    }
    assert [(r["method"], r["path"]) for r in hub.requests] == [
        ("GET", "/hub/api/users/fake-user-alpha"),
        ("GET", "/hub/api/users/fake-user-bravo"),
    ]
    assert [r["authorization"] for r in hub.requests] == [
        f"token {SECRETS['AUP_HUB_TOKEN_A']}",
        f"token {SECRETS['AUP_HUB_TOKEN_B']}",
    ]
    assert list(isolated_lab.iterdir()) == []
    assert_no_secret_values(out, err)


def test_ready_names_missing_keys_without_printing_values(isolated_lab, hub, monkeypatch, tmp_path, capsys):
    only_account_a = {key: value for key, value in SECRETS.items() if key.endswith("_A")}
    configure(monkeypatch, tmp_path, hub.url, only_account_a)

    code, out, err = run_cli(["ready"], capsys)

    assert code == 1
    result = one_json_line(out)
    assert (result["ok"], result["ready"], result["outcome"]) == (False, False, "not_ready")
    assert result["reasons"] == ["env keys missing: AUP_HUB_TOKEN_B, AUP_HUB_USER_B"]
    assert result["checks"]["env"]["present"] == ["AUP_HUB_TOKEN_A", "AUP_HUB_USER_A"]
    assert result["checks"]["env"]["missing"] == ["AUP_HUB_TOKEN_B", "AUP_HUB_USER_B"]
    assert core(result["checks"]["hub"]["accounts"]) == {
        "A": {"ok": True, "status": 200},
        "B": {"ok": False, "skipped": "credentials missing"},
    }
    assert len(hub.requests) == 1
    assert_no_secret_values(out, err)


def test_the_process_environment_beats_the_env_file(isolated_lab, hub, monkeypatch, tmp_path, capsys):
    configure(monkeypatch, tmp_path, hub.url, dict(SECRETS, AUP_HUB_TOKEN_A="wrong-token-in-the-file"))
    monkeypatch.setenv("AUP_HUB_TOKEN_A", SECRETS["AUP_HUB_TOKEN_A"])

    code, out, _ = run_cli(["ready"], capsys)

    assert code == 0
    assert one_json_line(out)["ready"] is True
    assert hub.requests[0]["authorization"] == f"token {SECRETS['AUP_HUB_TOKEN_A']}"


def test_ready_says_not_ready_while_stopped_but_still_runs_every_check(
    isolated_lab, hub, monkeypatch, tmp_path, capsys
):
    configure(monkeypatch, tmp_path, hub.url)
    write_stop(isolated_lab, "pod credits low")

    code, out, err = run_cli(["ready"], capsys)

    assert code == 1
    result = one_json_line(out)
    assert result["reasons"] == ["STOP file present: pod credits low"]
    assert result["checks"]["stop"] == {
        "ok": False,
        "stopped": True,
        "reason": "pod credits low",
        "file": str(isolated_lab / "STOP"),
    }
    assert result["checks"]["hub"]["ok"] is True
    assert len(hub.requests) == 2
    assert err == ""


def test_ready_reports_a_state_dir_it_cannot_write(tmp_path, hub, monkeypatch, capsys):
    configure(monkeypatch, tmp_path, hub.url)
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where the state dir should be", encoding="utf-8")
    state = blocker / "state"
    monkeypatch.setenv("LAB_STATE_DIR", str(state))

    code, out, _ = run_cli(["ready"], capsys)

    assert code == 1
    result = one_json_line(out)
    check = result["checks"]["state_dir"]
    assert (check["ok"], check["writable"], check["path"]) == (False, False, str(state))
    assert isinstance(check["error"], str) and check["error"] != ""
    assert f"state dir not writable: {state}" in result["reasons"]


@pytest.mark.parametrize(
    ("break_hub", "status"),
    [
        (lambda hub: hub.tokens.update({SECRETS["AUP_HUB_USER_B"]: "a-token-the-hub-never-issued"}), 403),
        (lambda hub: hub.tokens.pop(SECRETS["AUP_HUB_USER_B"]), 404),
    ],
    ids=["token the hub refuses", "user the hub does not know"],
)
def test_ready_reports_an_account_the_hub_turns_away(
    break_hub, status, isolated_lab, hub, monkeypatch, tmp_path, capsys
):
    configure(monkeypatch, tmp_path, hub.url)
    break_hub(hub)

    code, out, err = run_cli(["ready"], capsys)

    assert code == 0
    result = one_json_line(out)
    assert core(result["checks"]["hub"]["accounts"]) == {
        "A": {"ok": True, "status": 200},
        "B": {"ok": False, "status": status},
    }
    assert (result["ok"], result["ready"], result["outcome"]) == (True, True, "ready")
    assert result["reasons"] == []
    assert result["warnings"] == [f"hub account B answered {status}"]
    assert_no_secret_values(out, err)


def test_ready_does_not_follow_a_redirect_with_the_token(isolated_lab, hub, monkeypatch, tmp_path, capsys):
    configure(monkeypatch, tmp_path, hub.url)
    hub.mode = "redirect"

    code, out, _ = run_cli(["ready"], capsys)

    assert code == 0
    result = one_json_line(out)
    assert core(result["checks"]["hub"]["accounts"]) == {
        "A": {"ok": False, "status": 302},
        "B": {"ok": False, "status": 302},
    }
    assert (result["ready"], result["reasons"]) == (True, [])
    assert result["warnings"] == ["hub account A answered 302", "hub account B answered 302"]
    assert [r["path"] for r in hub.requests] == [
        "/hub/api/users/fake-user-alpha",
        "/hub/api/users/fake-user-bravo",
    ]


def test_ready_rejects_a_200_that_is_not_a_json_object(isolated_lab, hub, monkeypatch, tmp_path, capsys):
    configure(monkeypatch, tmp_path, hub.url)
    hub.mode = "html"

    code, out, _ = run_cli(["ready"], capsys)

    assert code == 0
    result = one_json_line(out)
    assert core(result["checks"]["hub"]["accounts"]["A"]) == {
        "ok": False,
        "status": 200,
        "error": "response is not a JSON object",
    }
    assert (result["ready"], result["reasons"]) == (True, [])
    assert result["warnings"] == [
        "hub account A answered 200 (response is not a JSON object)",
        "hub account B answered 200 (response is not a JSON object)",
    ]


def test_ready_reports_a_hub_that_cannot_be_reached(isolated_lab, monkeypatch, tmp_path, capsys):
    configure(monkeypatch, tmp_path, f"http://127.0.0.1:{closed_port()}")

    code, out, err = run_cli(["ready"], capsys)

    assert code == 0
    result = one_json_line(out)
    for letter in ("A", "B"):
        probe = result["checks"]["hub"]["accounts"][letter]
        assert probe["ok"] is False
        assert "status" not in probe
        # The one hub client's exception for a connection that never got an answer (lab/hub.py HubUnreachable).
        assert probe["error"].startswith("HubUnreachable: ")
    assert (result["ready"], result["reasons"]) == (True, [])
    assert result["warnings"] == ["hub account A unreachable", "hub account B unreachable"]
    assert_no_secret_values(out, err)


def test_errors_from_the_transport_never_echo_a_credential():
    def leaky(url, headers, timeout):
        raise OSError(f"reset while talking to {SECRETS['AUP_HUB_USER_A']} with {SECRETS['AUP_HUB_TOKEN_A']}")

    result = ready.check(environ=dict(SECRETS), env_files=[], fetch=leaky)

    assert core(result["checks"]["hub"]["accounts"]["A"]) == {"ok": False, "error": "OSError: reset while talking to *** with ***"}
    assert_no_secret_values(json.dumps(result))


# A token sourced from a CRLF .env reaches the check with a CR or LF on its end, or a hand-typed one with spaces
# around it. The text the check then has to clean spells the token bare, or with the line break escaped the way
# http.client's "Invalid header value" does, so hiding only the exact text it was given leaves the token showing.
TOKEN = SECRETS["AUP_HUB_TOKEN_A"]
LINE_BREAKS = {"cr": "\r", "lf": "\n", "crlf": "\r\n"}
PADDED = {**{name: TOKEN + ending for name, ending in LINE_BREAKS.items()}, "spaces": "  " + TOKEN + "  "}


def send_through_http_client(url, headers, timeout):
    """A fetch that puts the headers through the real http.client, which refuses a value ending in a line break."""
    connection = http.client.HTTPConnection("hub.invalid")
    connection.putrequest("GET", "/")
    for name, value in headers.items():
        connection.putheader(name, value)
    return 200, {}


@pytest.mark.parametrize("padded", PADDED.values(), ids=PADDED.keys())
def test_scrub_hides_a_padded_secret_from_a_message_that_spells_it_stripped(padded):
    message = f"the hub refused Authorization: token {TOKEN} for this call"

    assert ready.scrub(message, [padded]) == "the hub refused Authorization: token *** for this call"


@pytest.mark.parametrize("padded", PADDED.values(), ids=PADDED.keys())
def test_scrub_still_hides_a_padded_secret_spelled_exactly_as_given(padded):
    assert ready.scrub(f"echoed {padded}back", [padded]) == "echoed ***back"


@pytest.mark.parametrize("blank", ["", " ", "   ", "\r\n"], ids=["empty", "space", "spaces", "crlf"])
def test_scrub_leaves_the_text_alone_for_a_secret_that_is_blank(blank):
    assert ready.scrub("two words,   a gap\r\n", [blank]) == "two words,   a gap\r\n"


def test_scrub_skips_a_blank_secret_and_still_hides_the_others():
    assert ready.scrub(f"token {TOKEN} ok", ["", TOKEN, "  "]) == "token *** ok"


@pytest.mark.parametrize("ending", LINE_BREAKS.values(), ids=LINE_BREAKS.keys())
def test_a_token_ending_in_a_line_break_is_hidden_from_the_error_http_client_raises(ending):
    environ = dict(SECRETS, AUP_HUB_TOKEN_A=TOKEN + ending)

    result = ready.check(environ=environ, env_files=[], fetch=send_through_http_client)

    escaped = repr(ending)[1:-1]  # the two characters backslash and r, not the CR itself
    accounts = result["checks"]["hub"]["accounts"]
    assert core(accounts["A"]) == {"ok": False, "error": f"ValueError: Invalid header value b'token ***{escaped}'"}
    assert core(accounts["B"]) == {"ok": True, "status": 200}
    assert_no_secret_values(json.dumps(result))


def test_a_token_with_spaces_around_it_is_hidden_from_an_error_that_spells_it_bare():
    def echoes_the_token_bare(url, headers, timeout):
        raise OSError(f"the hub said no to token {TOKEN}")

    result = ready.check(environ=dict(SECRETS, AUP_HUB_TOKEN_A=f"  {TOKEN}  "), env_files=[], fetch=echoes_the_token_bare)

    assert core(result["checks"]["hub"]["accounts"]["A"]) == {"ok": False, "error": "OSError: the hub said no to token ***"}
    assert_no_secret_values(json.dumps(result))


def test_the_hub_address_defaults_to_the_aup_hub_and_follows_lab_hub_url():
    seen = []

    def record(url, headers, timeout):
        seen.append(url)
        return 200, {}

    ready.check(environ=dict(SECRETS), env_files=[], fetch=record)
    assert seen == [
        "https://dub.aupcloud.io/hub/api/users/fake-user-alpha",
        "https://dub.aupcloud.io/hub/api/users/fake-user-bravo",
    ]

    seen.clear()
    ready.check(environ=dict(SECRETS, LAB_HUB_URL="http://hub.invalid/"), env_files=[], fetch=record)
    assert seen == [
        "http://hub.invalid/hub/api/users/fake-user-alpha",
        "http://hub.invalid/hub/api/users/fake-user-bravo",
    ]


def test_a_user_name_is_percent_encoded_into_the_path():
    seen = []

    def record(url, headers, timeout):
        seen.append(url)
        return 200, {}

    ready.check(environ=dict(SECRETS, AUP_HUB_USER_A="a b/c"), env_files=[], fetch=record)

    assert seen[0] == "https://dub.aupcloud.io/hub/api/users/a%20b%2Fc"


# --- the real entry point, in a child process ------------------------------------------


def test_python_dash_m_lab_ready_prints_one_json_line(isolated_lab, hub, monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path, hub.url)

    result = run_module("ready")

    assert result.returncode == 0
    verdict = one_json_line(result.stdout)
    assert (verdict["ready"], verdict["reasons"]) == (True, [])
    assert_no_secret_values(result.stdout, result.stderr)


def test_python_dash_m_lab_ready_exits_0_when_only_the_hub_is_down(isolated_lab, monkeypatch, tmp_path):
    # A hub outage must not stop GPU-free sittings: stop, state dir and env are fine, every probe fails.
    configure(monkeypatch, tmp_path, f"http://127.0.0.1:{closed_port()}")

    result = run_module("ready")

    verdict = one_json_line(result.stdout)
    assert result.returncode == 0, result.stdout
    assert (verdict["ok"], verdict["ready"], verdict["outcome"]) == (True, True, "ready")
    assert verdict["reasons"] == []
    assert verdict["checks"]["hub"]["ok"] is False
    assert len(verdict["warnings"]) == 2
    assert "account A" in verdict["warnings"][0]
    assert "account B" in verdict["warnings"][1]
    assert_no_secret_values(result.stdout, result.stderr)


def test_python_dash_m_lab_ready_exits_1_while_stopped(isolated_lab, hub, monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path, hub.url)
    write_stop(isolated_lab, "halt")

    result = run_module("ready")

    assert result.returncode == 1
    assert one_json_line(result.stdout)["reasons"] == ["STOP file present: halt"]


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


class RecordingHub:
    """Stands in for lab.hub.Hub (the hub boundary) and records how ready used it."""

    constructed: list = []
    gets: list = []

    def __init__(self, account, *, base_url=None, **options):
        self.account = account
        self.base_url = base_url
        type(self).constructed.append((account.user, account.token, base_url))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def close(self):
        return None

    def user_url(self, suffix=""):
        return f"/hub/api/users/{self.account.user}{suffix}"

    def get(self, path, **kwargs):
        from lab import hub as lab_hub

        type(self).gets.append((self.account.user, path))
        return lab_hub.Reply(status=200, body={"name": self.account.user})


def _imports_of(module) -> set:
    import ast

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            names.add(base)
            names.update(f"{base}.{alias.name}" for alias in node.names)
    return names


def test_ready_probes_the_hub_through_the_one_hub_client(monkeypatch):
    """ready carries no second hub transport: it asks lab.hub.Hub, built per account, for user_url()."""
    from lab import hub as lab_hub

    problems = []
    for leftover in ("fetch_json",):
        if hasattr(ready, leftover):
            problems.append(f"ready still defines {leftover}")

    RecordingHub.constructed = []
    RecordingHub.gets = []
    monkeypatch.setattr(lab_hub, "Hub", RecordingHub)
    # Keep an unfixed ready from reaching a network: its own transport is replaced by one that refuses.
    def no_network(url, headers, timeout):
        raise OSError("network is not allowed in a test")

    monkeypatch.setattr(ready, "fetch_json", no_network, raising=False)

    result = ready.check(environ=dict(SECRETS), env_files=[], base_url="http://hub.test")
    hub_check = result["checks"]["hub"]

    constructed = [(user, token, (base or "").rstrip("/")) for user, token, base in RecordingHub.constructed]
    expected = [
        (SECRETS["AUP_HUB_USER_A"], SECRETS["AUP_HUB_TOKEN_A"], "http://hub.test"),
        (SECRETS["AUP_HUB_USER_B"], SECRETS["AUP_HUB_TOKEN_B"], "http://hub.test"),
    ]
    if constructed != expected:
        problems.append(f"Hub constructed as {constructed!r}, expected {expected!r}")
    expected_gets = [(user, f"/hub/api/users/{user}") for user in (SECRETS["AUP_HUB_USER_A"], SECRETS["AUP_HUB_USER_B"])]
    if RecordingHub.gets != expected_gets:
        problems.append(f"Hub.get called as {RecordingHub.gets!r}, expected {expected_gets!r}")
    answers = {letter: probe["ok"] for letter, probe in hub_check["accounts"].items()}
    if hub_check["ok"] is not True or answers != {"A": True, "B": True}:
        problems.append(f"hub check answered {hub_check!r}, expected ok for both accounts")

    assert problems == []


def test_the_ledger_resolves_the_state_dir_through_stop(monkeypatch, tmp_path):
    """One resolver for the state dir: stop owns the variable name and the default, and the ledger follows it on every call."""
    from lab.machine import ledger

    assert ledger.ENV_STATE_DIR == stop.STATE_DIR_ENV
    assert ledger.DEFAULT_STATE_DIR == stop.DEFAULT_STATE_DIR

    monkeypatch.delenv("LAB_STATE_DIR", raising=False)
    moved_default = tmp_path / "moved-default"
    monkeypatch.setattr(stop, "DEFAULT_STATE_DIR", moved_default)
    assert ledger.resolve_state_dir() == moved_default, "the ledger must follow stop's default, not keep a copy"

    moved_env = tmp_path / "moved-env"
    monkeypatch.setenv("LAB_STATE_DIR", str(moved_env))
    assert stop.state_dir() == moved_env
    assert ledger.resolve_state_dir() == moved_env, "the ledger must read LAB_STATE_DIR through stop"


# --- ready: the accounts LAB_POD_ROLES names, each at its own hub ----------------------

SJC_SECRETS = {
    "AUP_HUB_USER_C": "fake-user-charlie",
    "AUP_HUB_TOKEN_C": "fake-token-charlie-0123456789",
    "AUP_HUB_USER_D": "fake-user-delta",
    "AUP_HUB_TOKEN_D": "fake-token-delta-0123456789",
}


def two_hubs():
    """A DUB fake with accounts A and B and an SJC fake with C and D, as one network."""
    from lab import hub_fake

    clock = hub_fake.FakeClock()
    dub, sjc = hub_fake.FakeHub(clock), hub_fake.FakeHub(clock)
    for letter in "AB":
        dub.add_account(letter, user=SECRETS[f"AUP_HUB_USER_{letter}"], token=SECRETS[f"AUP_HUB_TOKEN_{letter}"])
    for letter in "CD":
        sjc.add_account(letter, user=SJC_SECRETS[f"AUP_HUB_USER_{letter}"],
                        token=SJC_SECRETS[f"AUP_HUB_TOKEN_{letter}"], quota_enabled=False)
    return dub, sjc, hub_fake.network({"dub.test": dub, "sjc.test": sjc})


def sjc_environ(**extra):
    return {**SJC_SECRETS, "LAB_POD_ROLES": "C:pod-a,D:pod-b", "AUP_HUB_URL_C": "https://sjc.test",
            "AUP_HUB_URL_D": "https://sjc.test", **extra}


def test_ready_probes_c_and_d_at_their_own_hub_when_lab_pod_roles_names_them(isolated_lab):
    dub, sjc, net = two_hubs()
    sjc.start_server("C")
    sjc.start_server("D", ready=False)

    result = ready.check(environ=sjc_environ(), env_files=[], transport=net)

    assert (result["ready"], result["reasons"], result["warnings"]) == (True, [], [])
    assert result["checks"]["env"]["missing"] == []
    accounts = result["checks"]["hub"]["accounts"]
    assert list(accounts) == ["C", "D"]
    assert {k: accounts["C"][k] for k in ("ok", "reachable", "running", "ready", "status", "role", "hub")} == {
        "ok": True, "reachable": True, "running": True, "ready": True, "status": 200, "role": "pod-a",
        "hub": "sjc.test"}
    assert {k: accounts["D"][k] for k in ("reachable", "running", "ready", "role")} == {
        "reachable": True, "running": True, "ready": False, "role": "pod-b"}
    assert result["checks"]["hub"]["ok"] is True
    assert result["checks"]["hub"]["hubs"] == ["sjc.test"]
    assert dub.calls == []
    assert [c.path for c in sjc.calls] == ["/hub/api/users/fake-user-charlie", "/hub/api/users/fake-user-delta"]
    assert all(c.method == "GET" for c in sjc.calls)


def test_ready_reports_an_account_with_no_pod_running(isolated_lab):
    _, _, net = two_hubs()

    result = ready.check(environ=sjc_environ(), env_files=[], transport=net)

    c = result["checks"]["hub"]["accounts"]["C"]
    assert (c["reachable"], c["running"], c["ready"]) == (True, False, False)
    assert result["ready"] is True


def test_ready_defaults_to_a_and_b_on_the_dub_hub(isolated_lab):
    dub, sjc, net = two_hubs()
    environ = {**SECRETS, **SJC_SECRETS, "AUP_HUB_URL_A": "https://dub.test", "AUP_HUB_URL_B": "https://dub.test"}

    result = ready.check(environ=environ, env_files=[], transport=net)

    accounts = result["checks"]["hub"]["accounts"]
    assert accounts["A"]["reachable"] and accounts["B"]["reachable"]
    assert accounts["A"]["hub"] == "dub.test"
    assert accounts["C"] == {"ok": False, "reachable": False, "running": False, "ready": False,
                             "skipped": "fills no pod role in LAB_POD_ROLES"}
    assert accounts["D"]["skipped"] == "fills no pod role in LAB_POD_ROLES"
    assert result["checks"]["hub"]["ok"] is True      # the accounts without a role do not count
    assert sjc.calls == []
    assert result["ready"] is True


def test_each_account_is_asked_at_its_own_hub_when_they_differ(isolated_lab):
    dub, sjc, net = two_hubs()
    environ = {**SECRETS, **SJC_SECRETS, "LAB_POD_ROLES": "A:pod-a,D:pod-b",
               "AUP_HUB_URL_A": "https://dub.test", "AUP_HUB_URL_D": "https://sjc.test"}

    result = ready.check(environ=environ, env_files=[], transport=net)

    hub_check = result["checks"]["hub"]
    assert hub_check["hubs"] == ["dub.test", "sjc.test"]
    assert hub_check["accounts"]["A"]["hub"] == "dub.test"
    assert hub_check["accounts"]["D"]["hub"] == "sjc.test"
    assert len(dub.calls) == 1 and len(sjc.calls) == 1


def test_an_account_without_keys_is_reported_and_not_probed_and_its_keys_are_named(isolated_lab):
    _, sjc, net = two_hubs()
    environ = sjc_environ()
    del environ["AUP_HUB_TOKEN_D"]

    result = ready.check(environ=environ, env_files=[], transport=net)

    assert result["checks"]["hub"]["accounts"]["D"] == {
        "ok": False, "reachable": False, "running": False, "ready": False, "skipped": "credentials missing",
        "role": "pod-b"}
    assert result["reasons"] == ["env keys missing: AUP_HUB_TOKEN_D"]
    assert result["ready"] is False
    assert len(sjc.calls) == 1


def test_the_hub_being_down_is_still_not_a_hand_off_for_c_and_d(isolated_lab):
    from lab import hub_fake

    nothing = hub_fake.network({})
    result = ready.check(environ=sjc_environ(), env_files=[], transport=nothing)

    assert (result["ok"], result["ready"], result["reasons"]) == (True, True, [])
    assert result["checks"]["hub"]["ok"] is False
    assert result["warnings"] == ["hub account C unreachable", "hub account D unreachable"]
    for letter in "CD":
        probe = result["checks"]["hub"]["accounts"][letter]
        assert (probe["reachable"], probe["running"], probe["ready"]) == (False, False, False)
        assert probe["hub"] == "sjc.test"


def test_lab_pod_roles_in_an_env_file_is_read(isolated_lab, tmp_path):
    _, sjc, net = two_hubs()
    env_file = tmp_path / "roles.env"
    environ = sjc_environ()
    roles = environ.pop("LAB_POD_ROLES")
    env_file.write_text(f"LAB_POD_ROLES={roles}\n", encoding="utf-8")

    result = ready.check(environ=environ, env_files=[env_file], transport=net)

    assert list(result["checks"]["hub"]["accounts"]) == ["C", "D"]
    assert result["ready"] is True


def test_a_lab_pod_roles_that_cannot_be_read_is_a_reason_not_a_crash(isolated_lab):
    result = ready.check(environ=sjc_environ(LAB_POD_ROLES="C:teacher"), env_files=[], transport=None,
                         fetch=lambda url, headers, timeout: (200, {}))

    assert result["ready"] is False
    assert len(result["reasons"]) == 1 and result["reasons"][0].startswith("LAB_POD_ROLES unreadable: ")


def test_the_cli_reports_c_and_d_and_exits_0(isolated_lab, monkeypatch, tmp_path):
    env_file = tmp_path / "sjc.env"
    environ = sjc_environ(AUP_HUB_URL_C=f"http://127.0.0.1:{closed_port()}",
                          AUP_HUB_URL_D=f"http://127.0.0.1:{closed_port()}")
    env_file.write_text("".join(f"{k}={v}\n" for k, v in environ.items()), encoding="utf-8")
    monkeypatch.setenv("LAB_ENV_FILES", str(env_file))

    result = run_module("ready")

    verdict = one_json_line(result.stdout)
    assert result.returncode == 0, result.stdout
    assert list(verdict["checks"]["hub"]["accounts"]) == ["C", "D"]
    assert verdict["checks"]["env"]["missing"] == []
    assert verdict["warnings"] == ["hub account C unreachable", "hub account D unreachable"]
    for value in SJC_SECRETS.values():
        assert value not in result.stdout + result.stderr
