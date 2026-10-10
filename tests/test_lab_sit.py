"""Tests for python -m lab sit, the launcher of one headless sitting.

No test starts a real agent. The CLI is a fake: a small Python script this file writes
into tmp_path, which the launcher runs as a real subprocess and which records the argv,
working directory, environment and stdin it was given. The digest is a stand-in verb
registered in the dispatcher. Every test has its own state dir and workspace, and an
autouse fixture makes any attempt to find the real agent CLI fail the test.
"""

import datetime as dt
import json
import os
import re
import sys
import time
from pathlib import Path

import pytest

from lab import __main__ as lab_cli
from lab.machine import allowed, ledger, sit

REAL_RESOLVE_COMMAND = sit.resolve_command
OPUS_PROMPT = (sit.PROMPTS_DIR / "opus.md").read_text(encoding="utf-8")
DIGEST_TEXT = "ledger: 3 sittings in the last window\nnext unblocked: O22 (opus), O24 (sonnet)"
MODEL_WORDS = "MODEL-WORDS-THAT-NEVER-REACH-THE-LEDGER"
ENV_KEYS = ["LAB_STATE_DIR", "LAB_ACTOR", "LAB_TIER", "LAB_TASK", "PYTHONPATH", "CLAUDE_CODE_DISABLE_AUTO_MEMORY",
            "ENABLE_CLAUDEAI_MCP_SERVERS",
            "CLAUDECODE", "CLAUDE_CODE_SESSION_ID", "KEEP_ME"]

FAKE_CLI = r'''
import json, os, subprocess, sys, time
from pathlib import Path

HEARTBEAT = """
import pathlib, sys, time
path = pathlib.Path(sys.argv[1])
end = time.time() + 20
while time.time() < end:
    with open(path, "ab") as handle:
        handle.write(b".")
    time.sleep(0.05)
"""

spec = json.loads(Path(os.environ["FAKE_CLI_SPEC"]).read_text(encoding="utf-8"))
prompt = sys.stdin.buffer.read().decode("utf-8")
record = {"argv": sys.argv[1:], "cwd": os.getcwd(), "stdin": prompt,
          "env": {key: os.environ.get(key) for key in spec.get("env_keys", [])}}
Path(os.environ["FAKE_CLI_RECORD"]).write_text(json.dumps(record), encoding="utf-8")
heartbeat = spec.get("heartbeat")
if heartbeat:
    subprocess.Popen([sys.executable, "-c", HEARTBEAT, heartbeat],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 10
    while not Path(heartbeat).exists() and time.time() < deadline:
        time.sleep(0.02)
out = sys.stdout.buffer
for line in spec.get("lines", []):
    out.write(line.encode("utf-8") + b"\n")
    out.flush()
if spec.get("stderr"):
    sys.stderr.buffer.write(spec["stderr"].encode("utf-8"))
    sys.stderr.flush()
if spec.get("stop_reason"):
    state = Path(os.environ["LAB_STATE_DIR"])
    state.mkdir(parents=True, exist_ok=True)
    (state / "STOP").write_text(spec["stop_reason"], encoding="utf-8")
time.sleep(spec.get("sleep", 0))
for line in spec.get("after", []):
    out.write(line.encode("utf-8") + b"\n")
    out.flush()
sys.exit(spec.get("exit", 0))
'''


# --- the CLI's events, in the shapes of the Agent SDK's system/init and result messages ------


def init_event(notion: str | None = "connected", model: str = "opus-model-id") -> str:
    servers = [{"name": "plugin:playwright:playwright", "status": "pending", "source": "plugin"}]
    if notion is not None:
        servers.insert(0, {"name": "plugin:design:notion", "status": notion, "source": "plugin"})
    return json.dumps({"type": "system", "subtype": "init", "session_id": "session-1", "model": model,
                       "permissionMode": "dontAsk", "tools": ["Bash", "Read", "mcp__plugin_design_notion__notion-fetch"],
                       "mcp_servers": servers})


def assistant_event(text: str = MODEL_WORDS) -> str:
    return json.dumps({"type": "assistant", "session_id": "session-1",
                       "message": {"role": "assistant", "content": [{"type": "text", "text": text}]}})


def result_event(subtype: str = "success", *, is_error: bool = False, cost: float | None = 0.42, turns: int = 7,
                 denials: int = 1) -> str:
    doc = {"type": "result", "subtype": subtype, "is_error": is_error, "duration_ms": 1234, "num_turns": turns,
           "result": MODEL_WORDS, "session_id": "session-1",
           "permission_denials": [{"tool_name": "Bash", "tool_use_id": f"t{n}", "tool_input": {}}
                                  for n in range(denials)]}
    if cost is not None:
        doc["total_cost_usd"] = cost
    return json.dumps(doc)


# --- fixtures ---------------------------------------------------------------------------------


def refuse_real_cli(command: str = sit.COMMAND) -> list[str]:
    raise AssertionError("a test reached for the real agent CLI")


@pytest.fixture(autouse=True)
def lab(tmp_path, monkeypatch):
    """A private state dir and workspace, no digest verb, fast clocks, and no real CLI."""
    state = tmp_path / "state"
    workspace = tmp_path / "workspace"
    workspace.joinpath(*sit.WORKSPACE_MARKER).parent.mkdir(parents=True)
    workspace.joinpath(*sit.WORKSPACE_MARKER).write_text("# open-ticket\n", encoding="utf-8")
    monkeypatch.setenv("LAB_STATE_DIR", str(state))
    monkeypatch.setenv(sit.WORKSPACE_ENV, str(workspace))
    monkeypatch.delenv(sit.ACTOR_ENV, raising=False)
    # A copy without the digest verb: deleting and restoring the key would move it to the end.
    monkeypatch.setattr(lab_cli, "VERBS", {verb: target for verb, target in lab_cli.VERBS.items()
                                           if verb != sit.DIGEST_VERB})
    monkeypatch.setattr(sit, "resolve_command", refuse_real_cli)
    monkeypatch.setattr(sit, "POLL_SECONDS", 0.05)
    monkeypatch.setattr(sit, "STOP_GRACE_SECONDS", 0.5)
    monkeypatch.setattr(sit, "TERM_WAIT_SECONDS", 1.0)
    monkeypatch.setattr(sit, "KILL_WAIT_SECONDS", 10.0)
    monkeypatch.setattr(sit, "READER_JOIN_SECONDS", 5.0)
    return {"state": state, "workspace": workspace}


def digest_verb(argv):
    """Prints a recorded digest."""
    print(DIGEST_TEXT)
    return 0


def failing_digest_verb(argv):
    """Answers that the digest could not be computed."""
    return {"ok": False, "outcome": "error", "error": "ledger unreadable"}


def long_digest_verb(argv):
    """Prints a digest longer than the launcher keeps."""
    print("x" * 120)
    return 0


@pytest.fixture
def digest(monkeypatch):
    monkeypatch.setitem(lab_cli.VERBS, sit.DIGEST_VERB, f"{__name__}:{digest_verb.__name__}")


class FakeCli:
    """The fake CLI: what it does is written to a spec file; what it saw is read back."""

    def __init__(self, tmp_path: Path, monkeypatch) -> None:
        self.script = tmp_path / "fake_cli.py"
        self.script.write_text(FAKE_CLI, encoding="utf-8")
        self.spec_path = tmp_path / "fake_spec.json"
        self.record_path = tmp_path / "fake_record.json"
        self.heartbeat = tmp_path / "heartbeat.txt"
        monkeypatch.setenv("FAKE_CLI_SPEC", str(self.spec_path))
        monkeypatch.setenv("FAKE_CLI_RECORD", str(self.record_path))
        monkeypatch.setattr(sit, "resolve_command", lambda command=sit.COMMAND: self.program)
        self.does(lines=[init_event(), assistant_event(), result_event()])

    @property
    def program(self) -> list[str]:
        return [sys.executable, str(self.script)]

    def does(self, **spec) -> None:
        spec.setdefault("env_keys", ENV_KEYS)
        self.spec_path.write_text(json.dumps(spec), encoding="utf-8")

    def ran(self) -> bool:
        return self.record_path.exists()

    def record(self) -> dict:
        return json.loads(self.record_path.read_text(encoding="utf-8"))


@pytest.fixture
def cli(tmp_path, monkeypatch):
    return FakeCli(tmp_path, monkeypatch)


def run_cli(argv, capsys):
    code = lab_cli.main(argv)
    out, _ = capsys.readouterr()
    lines = out.splitlines()
    assert len(lines) == 1, f"expected one JSON line on stdout, got {lines!r}"
    return code, json.loads(lines[0])


def rows(state: Path) -> list[dict]:
    return ledger.read(state_dir=state)


def without_ts(row: dict) -> dict:
    return {key: value for key, value in row.items() if key != "ts"}


def expected_argv(tier: str, *extra: str) -> list[str]:
    return ["-p", "--model", tier, "--output-format", "stream-json", "--verbose", "--no-session-persistence",
            "--permission-prompts", "none", *extra, *allowed.cli_flags(tier)]


def split_prompt(prompt: str) -> tuple[str, str, str]:
    """The tier prompt, the digest and the instruction of a prompt the launcher built."""
    head, rest = prompt.split(f"\n\n{sit.DIGEST_HEADING}\n\n")
    digest_part, instruction_part = rest.split(f"\n\n{sit.INSTRUCTION_HEADING}\n\n")
    return head, digest_part, instruction_part


# --- a sitting that runs ---------------------------------------------------------------------


def test_a_task_sitting_runs_the_cli_with_the_tier_model_flags_and_prompt_on_stdin(lab, cli, digest, capsys):
    code, answer = run_cli(["sit", "--tier", "opus", "--task", "o20"], capsys)

    assert code == 0
    assert answer["ok"] is True and answer["outcome"] == "ok"
    seen = cli.record()
    assert seen["argv"] == expected_argv("opus")
    assert Path(seen["cwd"]).resolve() == lab["workspace"].resolve()
    tier_part, digest_part, instruction_part = split_prompt(seen["stdin"])
    assert tier_part == OPUS_PROMPT.strip()
    assert digest_part == DIGEST_TEXT
    sitting = answer["sitting"]
    assert re.fullmatch(r"sit-\d{8}-\d{6}-opus-[0-9a-f]{4}", sitting)
    assert instruction_part.startswith(f"You are sitting {sitting}, at the Opus tier, started ")
    assert f"Your sitting id, for python -m lab handoff --sitting, is {sitting}." in instruction_part
    assert f"You run in the workspace {lab['workspace'].resolve()};" in instruction_part
    assert instruction_part.endswith("\n\nYour item is task O20: run /open-ticket O20.\n")


def test_the_session_environment_names_the_sitting_and_drops_the_parent_session(lab, cli, digest, capsys,
                                                                                 monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "parent-session")
    monkeypatch.setenv("KEEP_ME", "kept")
    monkeypatch.setenv("ENABLE_CLAUDEAI_MCP_SERVERS", "true")
    monkeypatch.setenv("LAB_TASK", "X9")
    monkeypatch.setenv("PYTHONPATH", "elsewhere")

    _, answer = run_cli(["sit", "--tier", "sonnet", "--task", "L3"], capsys)

    assert cli.record()["env"] == {
        "LAB_STATE_DIR": str(lab["state"].resolve()),
        "LAB_ACTOR": answer["sitting"],
        "LAB_TIER": "sonnet",
        "LAB_TASK": "L3",
        "PYTHONPATH": os.pathsep.join([str(sit.REPO_ROOT), "elsewhere"]),
        "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
        "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
        "CLAUDECODE": None,
        "CLAUDE_CODE_SESSION_ID": None,
        "KEEP_ME": "kept",
    }


def test_the_ledger_records_the_start_and_the_end_of_a_sitting(lab, cli, digest, capsys):
    _, answer = run_cli(["sit", "--tier", "opus", "--task", "O20"], capsys)

    start, end = rows(lab["state"])
    sitting = answer["sitting"]
    refs = {"pr": None, "commit": None, "slot": None, "run_id": sitting}
    assert without_ts(start) == {
        "seq": 1, "actor": "lab sit", "tier": "opus", "task": "O20", "verb": "sit", "outcome": "started",
        "metrics": {}, "refs": refs, "item": "task O20", "model": "opus", "rehearse": False, "cap_minutes": 240.0,
    }
    duration = end["metrics"]["sitting.duration_s"]
    assert isinstance(duration, float) and 0 <= duration < 60
    assert without_ts(end) == {
        "seq": 2, "actor": "lab sit", "tier": "opus", "task": "O20", "verb": "sit", "outcome": "ok",
        "metrics": {"sitting.duration_s": duration, "sitting.cost_usd": 0.42, "sitting.turns": 7,
                    "sitting.denials": 1},
        "units": {"sitting.duration_s": "s", "sitting.cost_usd": "USD"},
        "refs": refs, "item": "task O20", "rehearse": False, "exit_code": 0, "killed": None, "stop_seen": False,
        "notion": "connected", "model_used": "opus-model-id",
    }
    assert answer["ledger"] == {"start": 1, "end": 2}
    assert (answer["exit_code"], answer["duration_s"], answer["cost_usd"], answer["turns"], answer["denials"],
            answer["notion"], answer["model_used"]) == (0, duration, 0.42, 7, 1, "connected", "opus-model-id")


def test_the_stream_file_keeps_every_line_and_the_prompt_is_kept_beside_it(lab, cli, digest, capsys):
    lines = [init_event(), assistant_event(), result_event(), '{"type": "system", "subtype": "task_updated"}']
    cli.does(lines=lines)

    _, answer = run_cli(["sit", "--tier", "opus", "--task", "O20"], capsys)

    stream = Path(answer["stream"])
    assert stream == lab["state"].resolve() / "sittings" / f"{answer['sitting']}.jsonl"
    assert stream.read_text(encoding="utf-8").splitlines() == lines
    prompt_copy = stream.with_name(f"{answer['sitting']}.prompt.md")
    assert prompt_copy.read_text(encoding="utf-8") == cli.record()["stdin"]
    assert not stream.with_name(f"{answer['sitting']}.stderr.txt").exists()


def test_the_models_own_words_reach_only_the_stream_file(lab, cli, digest, capsys):
    _, answer = run_cli(["sit", "--tier", "opus", "--task", "O20"], capsys)

    assert MODEL_WORDS not in json.dumps(answer)
    assert MODEL_WORDS not in ledger.ledger_path(lab["state"]).read_text(encoding="utf-8")
    assert MODEL_WORDS in Path(answer["stream"]).read_text(encoding="utf-8")


def test_the_result_line_is_found_among_lines_printed_after_it(lab, cli, digest, capsys):
    cli.does(lines=[init_event(), result_event(cost=1.01), '{"type": "system", "subtype": "task_notification"}',
                    "not json at all"])

    code, answer = run_cli(["sit", "--tier", "opus", "--task", "O20"], capsys)

    assert (code, answer["outcome"], answer["cost_usd"]) == (0, "ok", 1.01)


@pytest.mark.parametrize(("argv", "body"), [
    (["--tier", "sonnet", "--task", "L3"], "Your item is task L3: run /open-ticket L3."),
    (["--tier", "opus", "--pass", "telemetry"],
     "Your item is a refinement pass on the telemetry track: run /refine-track telemetry."),
    (["--tier", "fable", "--pass", "kernels"],
     "Your item is a refinement pass on the kernels track: run /refine-track kernels."),
    (["--tier", "fable", "--merge"], "Your duty is the merge pass: run /refine-merge."),
    (["--tier", "fable", "--referee"],
     "Your duty is the referee: run python -m lab referee first, then follow the Referee duty section."),
])
def test_each_item_becomes_the_last_line_of_the_instruction(lab, cli, digest, capsys, argv, body):
    code, answer = run_cli(["sit", *argv, "--dry-run"], capsys)

    assert code == 0 and answer["outcome"] == "dry_run"
    assert answer["instruction"].endswith(f"\n\n{body}")
    tier = argv[1]
    assert answer["prompt_file"] == str(sit.PROMPTS_DIR / f"{tier}.md")
    assert answer["command"] == [*cli.program, *expected_argv(tier)]


def test_the_budget_goes_to_the_cli_before_the_allow_list(lab, cli, digest, capsys):
    run_cli(["sit", "--tier", "opus", "--task", "O20", "--budget-usd", "3"], capsys)

    assert cli.record()["argv"] == expected_argv("opus", "--max-budget-usd", "3")


# --- how a sitting ends ----------------------------------------------------------------------


def test_a_session_stopped_at_its_budget_ends_budget_with_its_cost(lab, cli, digest, capsys):
    cli.does(lines=[init_event(), result_event("error_max_budget_usd", is_error=True, cost=3.02)], exit=1)

    code, answer = run_cli(["sit", "--tier", "opus", "--task", "O20", "--budget-usd", "3"], capsys)

    assert (code, answer["ok"], answer["outcome"], answer["cost_usd"]) == (1, False, "budget", 3.02)
    assert rows(lab["state"])[-1]["outcome"] == "budget"


def test_an_error_result_is_an_error_without_the_models_words(lab, cli, digest, capsys):
    cli.does(lines=[init_event(), result_event("error_during_execution", is_error=True)], exit=1)

    code, answer = run_cli(["sit", "--tier", "opus", "--task", "O20"], capsys)

    assert (code, answer["outcome"]) == (1, "error")
    assert answer["error"] == "the CLI reported error_during_execution (exit 1)"
    assert rows(lab["state"])[-1]["note"] == "the CLI reported error_during_execution (exit 1)"


def test_a_cli_that_prints_no_result_is_an_error_quoting_what_it_printed(lab, cli, digest, capsys):
    cli.does(lines=["Error: not logged in"], stderr="please log in first", exit=1)

    code, answer = run_cli(["sit", "--tier", "opus", "--task", "O20"], capsys)

    assert (code, answer["outcome"], answer["exit_code"]) == (1, "error", 1)
    assert answer["error"] == "the CLI exited 1 and printed no result: please log in first"
    assert (answer["cost_usd"], answer["turns"], answer["denials"], answer["notion"]) == (None, None, None, None)
    end = rows(lab["state"])[-1]
    assert (end["outcome"], end["note"], end["metrics"].keys()) == ("error", answer["error"], {"sitting.duration_s"})
    stderr_file = Path(answer["stream"]).with_name(f"{answer['sitting']}.stderr.txt")
    assert stderr_file.read_text(encoding="utf-8") == "please log in first"


def test_a_program_that_cannot_start_still_gets_its_end_row(lab, cli, digest, capsys, tmp_path, monkeypatch):
    missing = tmp_path / "no-such-cli.exe"
    monkeypatch.setattr(sit, "resolve_command", lambda command=sit.COMMAND: [str(missing)])

    code, answer = run_cli(["sit", "--tier", "opus", "--task", "O20"], capsys)

    assert (code, answer["outcome"], answer["exit_code"]) == (1, "error", None)
    assert answer["error"].startswith(f"could not run {missing}: ")
    assert [row["outcome"] for row in rows(lab["state"])] == ["started", "error"]


def test_a_session_past_its_cap_is_killed(lab, cli, digest, capsys):
    cli.does(lines=[init_event()], sleep=60)

    code, answer = run_cli(["sit", "--tier", "opus", "--task", "O20", "--minutes", "0.02"], capsys)

    assert (code, answer["outcome"], answer["killed"], answer["cost_usd"]) == (1, "timeout", "timeout", None)
    assert 1.0 <= answer["duration_s"] < 30
    end = rows(lab["state"])[-1]
    assert (end["outcome"], end["killed"], set(end["metrics"])) == ("timeout", "timeout", {"sitting.duration_s"})


def test_stop_during_a_session_gives_it_the_grace_and_then_kills_it(lab, cli, digest, capsys):
    cli.does(lines=[init_event()], stop_reason="owner halt", sleep=60)

    code, answer = run_cli(["sit", "--tier", "opus", "--task", "O20"], capsys)

    assert (code, answer["outcome"], answer["killed"], answer["stop_seen"]) == (1, "stopped", "stopped", True)
    assert 0.5 <= answer["duration_s"] < 30
    assert rows(lab["state"])[-1]["outcome"] == "stopped"


def test_a_killed_session_takes_every_process_it_started_with_it(lab, cli, digest, capsys):
    # The fake writes STOP only once its own child is running, so the kill always finds it.
    cli.does(lines=[init_event()], heartbeat=str(cli.heartbeat), stop_reason="owner halt", sleep=60)

    _, answer = run_cli(["sit", "--tier", "opus", "--task", "O20"], capsys)

    assert answer["killed"] == "stopped"
    size = cli.heartbeat.stat().st_size
    time.sleep(0.5)
    assert cli.heartbeat.stat().st_size == size, "a process the session started is still running"


def test_a_result_printed_before_the_kill_decides_the_outcome(lab, cli, digest, capsys):
    # The CLI can linger after its result line (it waits for background work); the result decides.
    cli.does(lines=[init_event(), result_event(cost=0.3)], stop_reason="owner halt", sleep=60)

    code, answer = run_cli(["sit", "--tier", "opus", "--task", "O20"], capsys)

    assert (code, answer["outcome"], answer["killed"], answer["cost_usd"]) == (0, "ok", "stopped", 0.3)


def test_a_session_that_hands_off_within_the_grace_ends_on_its_own(lab, cli, digest, capsys, monkeypatch):
    monkeypatch.setattr(sit, "STOP_GRACE_SECONDS", 20.0)
    cli.does(lines=[init_event()], stop_reason="owner halt", sleep=1.0, after=[result_event()])

    code, answer = run_cli(["sit", "--tier", "opus", "--task", "O20"], capsys)

    assert (code, answer["outcome"], answer["killed"], answer["stop_seen"]) == (0, "ok", None, True)


@pytest.mark.parametrize(("init", "status"), [
    ({"mcp_servers": [{"name": allowed.NOTION_SERVER, "status": "connected"}]}, "connected"),
    ({"mcp_servers": [{"name": allowed.NOTION_SERVER, "status": "failed"}]}, "failed"),
    ({"mcp_servers": [{"name": "plugin:design:notion", "status": "needs-auth"}]}, "needs-auth"),
    ({"mcp_servers": [{"name": "notion", "status": "connected"}]}, "absent"),
    ({"mcp_servers": [{"name": "claude.ai Notion", "status": "needs-auth", "source": "claudeai"}]}, "absent"),
    ({"mcp_servers": []}, "absent"),
    (None, None),
])
def test_the_notion_server_status_comes_from_the_init_event(init, status):
    assert sit.notion_status(init) == status


# --- refusals --------------------------------------------------------------------------------


@pytest.mark.parametrize("argv", [
    ["--tier", "opus", "--merge"],
    ["--tier", "sonnet", "--referee"],
    ["--tier", "sonnet", "--pass", "telemetry"],
    ["--tier", "opus", "--pass", "gardening"],
    ["--tier", "opus", "--task", "O-20"],
    ["--tier", "opus", "--task", "O20; rm"],
    ["--tier", "haiku", "--task", "O20"],
    ["--tier", "opus"],
    ["--tier", "opus", "--task", "O20", "--merge"],
    ["--task", "O20"],
    ["--tier", "opus", "--task", "O20", "--minutes", "0"],
    ["--tier", "opus", "--task", "O20", "--budget-usd", "nan"],
])
def test_bad_arguments_are_usage_errors_and_start_nothing(lab, cli, digest, capsys, argv):
    code = lab_cli.main(["sit", *argv])

    out, err = capsys.readouterr()
    assert code == 2 and out == ""
    assert "usage: python -m lab sit" in err
    assert not cli.ran() and rows(lab["state"]) == []


@pytest.mark.parametrize(("tier", "item", "minutes", "error"), [
    ("opus", sit.Item("merge"), None, "--merge is a fable duty, not one for the opus tier"),
    ("sonnet", sit.Item("pass", "telemetry"), None, "a refinement pass is for the fable or opus tier, not sonnet"),
    ("opus", sit.Item("task", "o20"), None, "--task takes a Task Board ID such as O20, not 'o20'"),
    ("opus", sit.Item("rehearse", "x"), None, "--rehearse takes no name"),
    ("opus", sit.Item("lunch"), None, "unknown item 'lunch'; expected one of task, pass, merge, referee, rehearse"),
    ("opus", sit.Item("task", "O20"), 0, "minutes must be a number above 0, not 0"),
])
def test_run_refuses_what_the_parser_would(lab, cli, tier, item, minutes, error):
    assert sit.run(tier, item, minutes=minutes) == {"ok": False, "outcome": "usage", "error": error}
    assert not cli.ran()


def test_the_stop_file_refuses_a_sitting_before_anything_runs(lab, cli, digest, capsys):
    lab["state"].mkdir(parents=True)
    (lab["state"] / "STOP").write_text("owner halt", encoding="utf-8")

    code, answer = run_cli(["sit", "--tier", "opus", "--task", "O20"], capsys)
    direct = sit.run("opus", sit.Item("task", "O20"))

    assert (code, answer["outcome"], answer["reason"]) == (1, "stopped", "owner halt")
    assert direct == {"ok": False, "outcome": "stopped", "reason": "owner halt",
                      "file": str(lab["state"] / "STOP")}
    assert not cli.ran()
    assert rows(lab["state"]) == [] and not (lab["state"] / "sittings").exists()


def test_without_a_digest_a_real_sitting_is_refused(lab, cli, capsys):
    code, answer = run_cli(["sit", "--tier", "opus", "--task", "O20"], capsys)

    assert (code, answer["outcome"]) == (1, "no_digest")
    assert answer["error"] == ("a sitting starts from the digest, and there is none: "
                               "python -m lab digest is not registered in lab/__main__.py")
    assert not cli.ran() and rows(lab["state"]) == []


def test_a_digest_that_fails_refuses_the_sitting(lab, cli, capsys, monkeypatch):
    monkeypatch.setitem(lab_cli.VERBS, sit.DIGEST_VERB, f"{__name__}:{failing_digest_verb.__name__}")

    code, answer = run_cli(["sit", "--tier", "opus", "--task", "O20"], capsys)

    assert (code, answer["outcome"]) == (1, "no_digest")
    assert answer["digest"]["error"].startswith('python -m lab digest exited 1: {"ok": false, "outcome": "error"')
    assert not cli.ran()


def test_a_digest_longer_than_the_launcher_keeps_is_cut_with_a_note(monkeypatch):
    monkeypatch.setitem(lab_cli.VERBS, sit.DIGEST_VERB, f"{__name__}:{long_digest_verb.__name__}")
    monkeypatch.setattr(sit, "MAX_DIGEST_CHARS", 50)

    digest = sit.compute_digest()

    assert (digest.ok, digest.truncated) == (True, True)
    assert digest.text == "x" * 50 + "\n[digest cut at 50 of 120 characters]"


def test_no_workspace_no_sitting(lab, cli, digest, capsys, monkeypatch, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv(sit.WORKSPACE_ENV, str(empty))

    code, answer = run_cli(["sit", "--tier", "opus", "--task", "O20"], capsys)

    assert (code, answer["outcome"]) == (1, "no_workspace")
    assert answer["error"] == f"no workspace: {Path(*sit.WORKSPACE_MARKER)} is not in {empty}"
    assert not cli.ran()


def test_no_cli_no_sitting(lab, digest, capsys, monkeypatch):
    monkeypatch.setattr(sit, "resolve_command", REAL_RESOLVE_COMMAND)
    monkeypatch.setattr(sit.shutil, "which", lambda name: None)

    code, answer = run_cli(["sit", "--tier", "opus", "--task", "O20"], capsys)

    assert (code, answer["outcome"]) == (1, "no_cli")
    assert answer["error"] == f"{sit.COMMAND!r} is not on PATH; install the agent CLI and log in"
    assert rows(lab["state"]) == []


def test_a_start_that_cannot_be_recorded_is_no_start(lab, cli, digest, capsys, monkeypatch, tmp_path):
    blocker = tmp_path / "a-file"
    blocker.write_text("not a folder", encoding="utf-8")
    monkeypatch.setenv("LAB_STATE_DIR", str(blocker / "state"))

    code, answer = run_cli(["sit", "--tier", "opus", "--task", "O20"], capsys)

    assert (code, answer["outcome"]) == (1, "ledger_error")
    assert answer["error"].startswith("the start row could not be written, so the sitting did not start: ")
    assert not cli.ran()


def test_the_launcher_never_runs_inside_a_sitting(lab, cli, digest, capsys, monkeypatch):
    monkeypatch.setenv(sit.ACTOR_ENV, "sit-20261010-101500-opus-ab12")

    code, answer = run_cli(["sit", "--tier", "opus", "--task", "O20"], capsys)

    assert (code, answer["outcome"]) == (1, "inside_sitting")
    assert not cli.ran()


# --- rehearsal and dry run -------------------------------------------------------------------


def test_a_rehearsal_runs_a_fake_task_even_without_a_digest(lab, cli, capsys, monkeypatch):
    monkeypatch.setenv("LAB_TASK", "X9")

    code, answer = run_cli(["sit", "--tier", "sonnet", "--rehearse"], capsys)

    assert (code, answer["outcome"], answer["item"], answer["rehearse"], answer["minutes"]) == (
        0, "ok", "rehearse", True, 2.0)
    sitting = answer["sitting"]
    _, digest_part, instruction_part = split_prompt(cli.record()["stdin"])
    assert digest_part == "(no digest: python -m lab digest is not registered in lab/__main__.py)"
    assert instruction_part.endswith("\n\n" + sit.REHEARSAL_TASK.format(minutes=2.0, sitting=sitting, tier="sonnet")
                                     + "\n")
    assert "/open-ticket" not in instruction_part
    assert cli.record()["env"]["LAB_TASK"] is None
    start, end = rows(lab["state"])
    assert (start["task"], start["item"], start["rehearse"], start["cap_minutes"]) == (None, "rehearse", True, 2.0)
    assert (end["outcome"], end["rehearse"]) == ("ok", True)


@pytest.mark.parametrize(("minutes", "cap"), [(None, 2.0), ("30", 2.0), ("0.5", 0.5)])
def test_a_rehearsal_never_runs_past_two_minutes(lab, cli, capsys, minutes, cap):
    argv = ["sit", "--tier", "opus", "--rehearse", "--dry-run"] + (["--minutes", minutes] if minutes else [])

    _, answer = run_cli(argv, capsys)

    assert answer["minutes"] == cap


def test_a_dry_run_answers_what_would_run_and_starts_and_writes_nothing(lab, cli, digest, capsys, monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")

    code, answer = run_cli(["sit", "--tier", "opus", "--task", "O20", "--dry-run"], capsys)

    assert (code, answer["ok"], answer["outcome"]) == (0, True, "dry_run")
    assert answer["command"] == [*cli.program, *expected_argv("opus")]
    assert answer["cwd"] == str(lab["workspace"].resolve())
    assert answer["env"] == {
        "LAB_STATE_DIR": str(lab["state"].resolve()), "LAB_ACTOR": answer["sitting"], "LAB_TIER": "opus",
        "LAB_TASK": "O20",
        "PYTHONPATH": answer["env"]["PYTHONPATH"], "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
        "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
    }
    assert answer["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(sit.REPO_ROOT)
    assert "CLAUDECODE" in answer["env_removed"]
    assert answer["digest"] == {"ok": True, "chars": len(DIGEST_TEXT)}
    assert answer["prompt_chars"] > len(OPUS_PROMPT.strip()) + len(DIGEST_TEXT)
    assert not cli.ran()
    assert not lab["state"].exists()


# --- pieces ----------------------------------------------------------------------------------


def test_sit_is_registered_and_answers_help(capsys):
    assert lab_cli.VERBS["sit"] == "lab.machine.sit:main"
    code = lab_cli.main(["sit", "-h"])
    out, _ = capsys.readouterr()
    assert code == 0
    assert out.startswith("usage: python -m lab sit")


def test_a_sitting_id_names_the_time_and_tier_and_differs_within_one_second():
    now = dt.datetime(2026, 10, 10, 10, 5, 9, tzinfo=dt.timezone(dt.timedelta(hours=-4)))
    ids = {sit.new_sitting_id("fable", now) for _ in range(20)}
    assert all(re.fullmatch(r"sit-20261010-140509-fable-[0-9a-f]{4}", one) for one in ids)
    assert len(ids) > 1


def test_find_workspace_takes_the_nearest_folder_that_holds_the_skill(tmp_path, monkeypatch):
    monkeypatch.delenv(sit.WORKSPACE_ENV, raising=False)
    workspace = tmp_path / "Knighthacks26"
    workspace.joinpath(*sit.WORKSPACE_MARKER).parent.mkdir(parents=True)
    workspace.joinpath(*sit.WORKSPACE_MARKER).write_text("", encoding="utf-8")
    task_clone = workspace / "work" / "O20"
    own_clone = workspace / "atlas"
    task_clone.mkdir(parents=True)
    own_clone.mkdir()
    too_deep = workspace / "a" / "b" / "c" / "d"
    too_deep.mkdir(parents=True)

    assert sit.find_workspace(task_clone) == workspace.resolve()
    assert sit.find_workspace(own_clone) == workspace.resolve()
    assert sit.find_workspace(too_deep) is None


def write_shim(folder: Path, target: str) -> Path:
    shim = folder / f"{sit.COMMAND}.cmd"
    shim.write_text('@ECHO off\r\nGOTO start\r\n:start\r\nSETLOCAL\r\n'
                    f'"%dp0%\\{target}"   %*\r\n', encoding="utf-8")
    return shim


def test_resolve_command_runs_the_exe_an_npm_shim_names(tmp_path, monkeypatch):
    exe = tmp_path / "node_modules" / "@vendor" / "agent-cli" / "bin" / f"{sit.COMMAND}.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    shim = write_shim(tmp_path, f"node_modules\\@vendor\\agent-cli\\bin\\{sit.COMMAND}.exe")
    monkeypatch.setattr(sit.shutil, "which", lambda name: str(shim) if name == sit.COMMAND else None)

    assert REAL_RESOLVE_COMMAND() == [str(exe)]


def test_resolve_command_runs_a_js_target_with_node(tmp_path, monkeypatch):
    script = tmp_path / "node_modules" / "@vendor" / "agent-cli" / "cli.js"
    script.parent.mkdir(parents=True)
    script.write_text("", encoding="utf-8")
    shim = write_shim(tmp_path, r"node_modules\@vendor\agent-cli\cli.js")
    node = str(tmp_path / "node.exe")
    monkeypatch.setattr(sit.shutil, "which", lambda name: {sit.COMMAND: str(shim), "node": node}.get(name))

    assert REAL_RESOLVE_COMMAND() == [node, str(script)]


def test_resolve_command_takes_a_plain_executable_as_it_is(tmp_path, monkeypatch):
    program = str(tmp_path / "agent-cli")
    monkeypatch.setattr(sit.shutil, "which", lambda name: program if name == sit.COMMAND else None)

    assert REAL_RESOLVE_COMMAND() == [program]


def test_the_command_ends_with_the_allow_list_flags_and_never_holds_the_prompt():
    cmd = sit.session_command(["agent-cli"], "fable", 2.5)

    assert cmd[-2:] == ["--permission-mode", "dontAsk"]
    assert cmd[:12] == ["agent-cli", "-p", "--model", "fable", "--output-format", "stream-json", "--verbose",
                        "--no-session-persistence", "--permission-prompts", "none", "--max-budget-usd", "2.5"]
    assert cmd[12:] == allowed.cli_flags("fable")
