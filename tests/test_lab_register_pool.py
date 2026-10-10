"""Tests for lab/scheduler: the Windows scheduled task that keeps python -m lab pool running.

Nothing here registers a task, starts the pool or runs the batch file. Both scripts are read as
text, with their comment lines removed first so that a word in a comment proves nothing, and the
lines that carry the contract are checked against the code they depend on: the pool verb, the
STOP file and the state directory. Two tests use PowerShell: one has its parser read
register_pool.ps1 (parsing executes none of it) and one checks that this parse check reports a
syntax error, so that it cannot pass by reading nothing.

PowerShell is needed for those two: powershell.exe on Windows, pwsh elsewhere (the CI image has
it). A machine without either fails them with an install hint.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from lab import __main__ as lab_cli
from lab.machine import ledger, stop

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = REPO_ROOT / "lab" / "scheduler"
REGISTER = SCHEDULER / "register_pool.ps1"
WRAPPER = SCHEDULER / "run_pool.cmd"
SCRIPTS = [REGISTER, WRAPPER]

# The settings that keep the pool alive, as register_pool.ps1 must hand them to Task Scheduler.
EXPECTED_SETTINGS = {
    "AllowStartIfOnBatteries": "$true",
    "DontStopIfGoingOnBatteries": "$true",
    "MultipleInstances": '"IgnoreNew"',
    "ExecutionTimeLimit": "[TimeSpan]::Zero",
    "RestartCount": "999",
    "RestartInterval": "(New-TimeSpan -Minutes 1)",
}

PARSE_COMMAND = (
    "$tokens = $null; $errors = $null; "
    "[void][System.Management.Automation.Language.Parser]::ParseFile($env:LAB_PARSE_TARGET, [ref]$tokens, [ref]$errors); "
    "foreach ($e in $errors) { '{0}:{1}: {2}' -f $e.Extent.StartLineNumber, $e.Extent.StartColumnNumber, $e.Message }; "
    "if ($null -eq $tokens) { 'tokens=0' } else { 'tokens={0}' -f $tokens.Count }"
)


def powershell_code(path: Path) -> str:
    """A PowerShell script without its full-line comments and blank lines."""
    lines = path.read_text(encoding="ascii").splitlines()
    return "\n".join(line for line in lines if line.strip() and not line.lstrip().startswith("#"))


def batch_code(path: Path) -> str:
    """A batch file without its rem lines and blank lines."""
    lines = path.read_text(encoding="ascii").splitlines()
    return "\n".join(line for line in lines if line.strip() and not re.match(r"\s*rem(\s|$)", line, re.IGNORECASE))


def position(code: str, pattern: str) -> int:
    found = re.search(pattern, code, re.MULTILINE)
    assert found, f"no line matches {pattern!r}"
    return found.start()


def parse_errors(path: Path) -> list[str]:
    """What PowerShell's own parser reports for the file, one 'line:column: message' per error.

    The parser must have read the file: a run that found no tokens is a failure, not a clean parse."""
    exe = shutil.which("powershell") or shutil.which("pwsh")
    if exe is None:
        pytest.fail("PowerShell is not installed: the parse tests need powershell.exe (Windows) or pwsh "
                    "(brew install powershell; the CI image already has it)")
    done = subprocess.run([exe, "-NoProfile", "-NonInteractive", "-Command", PARSE_COMMAND],
                          env={**os.environ, "LAB_PARSE_TARGET": str(path)},
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, f"PowerShell failed: {done.stderr!r}"
    lines = [line for line in done.stdout.splitlines() if line.strip()]
    counted = [line for line in lines if re.fullmatch(r"tokens=\d+", line)]
    assert len(counted) == 1 and counted[0] != "tokens=0", f"the parser did not read {path.name}: {done.stdout!r} {done.stderr!r}"
    return [line for line in lines if line not in counted]


# --- the files themselves ------------------------------------------------------------------------


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda path: path.name)
def test_the_scripts_are_plain_ascii_without_a_byte_order_mark(path):
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), f"{path.name} starts with a byte order mark"
    bad = [(offset, byte) for offset, byte in enumerate(raw) if byte > 127]
    assert not bad, f"{path.name} has non-ASCII bytes (Windows PowerShell 5.1 reads a file without a mark as ANSI): {bad[:5]}"


def test_the_batch_file_keeps_windows_line_endings_in_every_checkout():
    raw = WRAPPER.read_bytes()
    assert raw.count(b"\n") == raw.count(b"\r\n") > 0, "run_pool.cmd has a line ending that is not CRLF"
    attributes = (SCHEDULER / ".gitattributes").read_text(encoding="ascii").splitlines()
    assert "*.cmd text eol=crlf" in attributes


def test_register_pool_parses_without_errors():
    assert parse_errors(REGISTER) == []


def test_the_parse_check_reports_a_syntax_error(tmp_path):
    broken = tmp_path / "broken.ps1"
    broken.write_text("param([switch]$Remove\nif ($Remove) {\n", encoding="ascii")
    errors = parse_errors(broken)
    assert errors, "the parser said nothing about a script with an unclosed parameter list and block"
    assert all(re.match(r"\d+:\d+: ", line) for line in errors)


# --- register_pool.ps1: the task it registers ------------------------------------------------------


def test_register_pool_takes_one_switch_and_it_is_remove():
    code = powershell_code(REGISTER)
    assert re.findall(r"^param\(.*\)[ \t]*$", code, re.MULTILINE) == ["param([switch]$Remove)"]


def test_the_task_starts_at_logon_of_the_current_user_and_not_at_boot():
    code = powershell_code(REGISTER)
    assert "$user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name" in code
    assert re.findall(r"New-ScheduledTaskTrigger\b.*", code) == ["New-ScheduledTaskTrigger -AtLogOn -User $user"]
    assert re.findall(r"New-ScheduledTaskPrincipal\b.*", code) == [
        "New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited"]


def test_the_task_restarts_never_times_out_and_runs_one_copy():
    code = powershell_code(REGISTER)
    block = re.search(r"\$settingsArgs = @\{\n(?P<body>.*?)\n\}", code, re.DOTALL)
    assert block, "no $settingsArgs hashtable"
    found = dict(re.findall(r"^\s*(\w+)\s*=\s*(.+?)\s*$", block["body"], re.MULTILINE))
    assert found == EXPECTED_SETTINGS
    assert "New-ScheduledTaskSettingsSet @settingsArgs" in code


def test_the_task_runs_the_batch_file_headless_from_the_clone_it_sits_in():
    code = powershell_code(REGISTER)
    assert '$wrapper = Join-Path $PSScriptRoot "run_pool.cmd"' in code
    assert WRAPPER.is_file()
    assert '$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path' in code
    assert '$conhost = Join-Path $env:SystemRoot "System32\\conhost.exe"' in code
    assert re.findall(r"New-ScheduledTaskAction\b.*", code) == [
        'New-ScheduledTaskAction -Execute $conhost -Argument "--headless `"$wrapper`"" -WorkingDirectory $repoRoot']
    assert "Test-Path -LiteralPath $wrapper" in code


def test_there_is_one_task_name_and_one_registration():
    code = powershell_code(REGISTER)
    assert re.findall(r'"AtlasLab-Pool"', code) == ['"AtlasLab-Pool"']
    assert len(re.findall(r"^\s*Register-ScheduledTask\b", code, re.MULTILINE)) == 1
    commands = re.findall(r"\b(?:Get|Stop|Unregister|Register)-ScheduledTask\b[^|\n]*", code)
    assert len(commands) == 4
    assert all("-TaskName $taskName -TaskPath $taskPath" in command for command in commands), commands
    assert "-Force" in re.search(r"^\s*Register-ScheduledTask\b.*", code, re.MULTILINE)[0]


def test_remove_stops_a_running_task_then_unregisters_it_before_any_registration_code():
    code = powershell_code(REGISTER)
    order = [position(code, pattern) for pattern in (
        r"^if \(\$Remove\) \{", r"^\s*Stop-ScheduledTask\b", r"^\s*Unregister-ScheduledTask\b",
        r"^\s*\$action = New-ScheduledTaskAction\b", r"^\s*Register-ScheduledTask\b")]
    assert order == sorted(order)
    assert "-Confirm:$false" in re.search(r"^\s*Unregister-ScheduledTask\b.*", code, re.MULTILINE)[0]


def test_registering_never_starts_the_pool_or_any_task():
    code = powershell_code(REGISTER)
    started = re.findall(r"^\s*(?:&\s*)?(?:Start-ScheduledTask|Start-Process|schtasks(?:\.exe)?|Invoke-Expression|iex)\b.*",
                         code, re.MULTILINE | re.IGNORECASE)
    assert started == []


# --- run_pool.cmd: the loop the task runs ---------------------------------------------------------


def wrapper_setting(code: str, name: str) -> str:
    found = re.search(rf'^set "{name}=(.*)"\s*$', code, re.MULTILINE)
    assert found, f"run_pool.cmd does not set {name}"
    return found[1]


def test_the_batch_file_runs_the_registered_pool_verb_with_its_arguments_into_the_log():
    code = batch_code(WRAPPER)
    assert lab_cli.VERBS["pool"] == "lab.machine.pool:main"
    assert '"%PY%" -m lab pool %* >> "%LOG%" 2>&1' in code.splitlines()
    assert 'cd /d "%~dp0..\\.." || exit /b 1' in code.splitlines()


def test_the_batch_file_logs_to_the_ignored_pool_log_in_the_state_dir():
    code = batch_code(WRAPPER)
    state = wrapper_setting(code, "STATE")
    assert Path(*state.split("\\")) == stop.DEFAULT_STATE_DIR.relative_to(REPO_ROOT)
    assert wrapper_setting(code, "LOG") == "%STATE%\\logs\\pool.log"
    assert 'if not exist "%STATE%\\logs" mkdir "%STATE%\\logs"' in code.splitlines()
    ignored = subprocess.run(["git", "check-ignore", "--no-index", "-q", "lab/state/logs/pool.log"], cwd=REPO_ROOT)
    assert ignored.returncode == 0, "lab/state/logs/pool.log is not ignored by git"


def test_the_batch_file_watches_the_stop_file_the_pool_obeys_and_clears_a_state_dir_override():
    code = batch_code(WRAPPER)
    stop_check = f'if exist "%STATE%\\{stop.STOP_NAME}" goto stopped'
    assert code.splitlines().count(stop_check) == 2
    assert f"set {stop.STATE_DIR_ENV}=" in code.splitlines()
    assert stop.STATE_DIR_ENV == ledger.ENV_STATE_DIR


def test_the_batch_file_checks_stop_before_every_start_and_restarts_after_any_exit():
    lines = batch_code(WRAPPER).splitlines()
    loop = lines[lines.index(":again") + 1:lines.index(":stopped")]
    assert loop[0] == f'if exist "%STATE%\\{stop.STOP_NAME}" goto stopped'
    assert loop.index('"%PY%" -m lab pool %* >> "%LOG%" 2>&1') > 0
    assert loop[-2:] == ["call :wait %RESTART_SECONDS%", "goto again"]
    waiting = lines[lines.index(":stopped") + 1:lines.index(":note")]
    assert waiting[-2:] == ["call :wait %POLL_SECONDS%", "goto again"]


def test_the_batch_file_waits_between_starts_so_a_failing_pool_cannot_spin():
    code = batch_code(WRAPPER)
    assert int(wrapper_setting(code, "RESTART_SECONDS")) >= 10
    assert int(wrapper_setting(code, "POLL_SECONDS")) >= 1
    assert "ping -n %PINGS% 127.0.0.1 > nul" in code.splitlines()
