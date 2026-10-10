"""python -m lab sit: start one headless sitting of the lab machine and record it.

A sitting is one agent session, the agent CLI in print mode (-p), that does one item and
ends. The pool starts one per seat and the owner may start one by hand; a sitting never
starts another (allowed.OWNER_VERBS denies this verb to every tier, and the launcher
refuses to run while LAB_ACTOR names a sitting).

    python -m lab sit --tier <fable|opus|sonnet> --task <ID>    a board task: /open-ticket <ID>
    python -m lab sit --tier <opus|fable> --pass <track>        a refinement pass: /refine-track <track>
    python -m lab sit --tier fable --merge                      the merge pass: /refine-merge
    python -m lab sit --tier fable --referee                    the referee duty: python -m lab referee first
    python -m lab sit --tier <tier> --rehearse                  a fake task, capped at REHEARSE_MINUTES
        [--minutes N] [--budget-usd X] [--dry-run]

One call, in order:
1. Refusals before anything runs or is written: the STOP file (stop.py), an item the
   tier cannot do (exit 2: --merge and --referee are Fable duties, a pass is Opus's or
   Fable's), no workspace, no tier prompt, no agent CLI and, for anything but a
   rehearsal, no digest.
2. The prompt: the tier prompt (prompts/<tier>.md), the digest (python -m lab digest,
   run through the dispatcher in this process just before the session starts) and,
   last, the instruction: the sitting id, the cap and the item.
3. A ledger row, verb "sit", outcome "started", refs.run_id the sitting id. A start
   that cannot be recorded is no start.
4. The session, with the prompt on stdin; the launcher waits for it.
5. The end row: outcome ok, error, budget, timeout, stopped or interrupted, with
   exit_code, the metrics sitting.duration_s, sitting.cost_usd (only when the CLI
   reported a cost), sitting.turns and sitting.denials, and the Notion server's status.
The answer is one JSON line with the same facts; ok is true only for the outcome ok.

The command: <cli> -p --model <tier> --output-format stream-json --verbose
--no-session-persistence --permission-prompts none [--max-budget-usd X], then
allowed.cli_flags(tier) last (--tools, --allowedTools, --disallowedTools and
--permission-mode dontAsk, which allowed.py requires to end the vector because the list
flags take every argument up to the next option). The tier name is the CLI's model
alias. The prompt never goes on the command line: it holds newlines, which cmd.exe cuts
an argument at, and runs to tens of thousands of characters. On Windows the CLI on PATH
is an npm .cmd shim; resolve_command runs the .exe it names directly.

Where it runs. The working directory is the workspace, the folder holding
.claude/skills/open-ticket ($LAB_WORKSPACE, else the nearest of the repo root and its
three parents that has it), so the workspace skills, settings, hooks and CLAUDE.md
load. User settings load as well: the Notion MCP server comes from a user plugin and
must connect as allowed.NOTION_SERVER; the end row records its status from the
stream's init event. The session's environment is the launcher's without the
variables that tie a process to the agent session it was started from (SESSION_VARS:
a sitting is a session of its own), plus LAB_STATE_DIR (the launcher's state dir, made
absolute, so every verb a sitting runs from any clone writes the one ledger),
LAB_ACTOR (the sitting id), LAB_TIER and, for a task, LAB_TASK (the three the ledger
helper record.py writes rows with; an inherited LAB_TASK is dropped otherwise),
PYTHONPATH (this repo root first, so python -m lab runs from the workspace),
CLAUDE_CODE_DISABLE_AUTO_MEMORY=1 (a sitting starts
from the digest, never from notes an earlier session left) and
ENABLE_CLAUDEAI_MCP_SERVERS=false (none of the connectors added to the owner's web
account: no tier may call one, and the account's Notion connector points at the same
endpoint as the plugin's Notion server, so at a headless start it can take that
server's place, unauthorised, and leave the sitting with no Notion tools).

While it runs. The CLI's stdout is copied line by line, as it arrives, to
<state dir>/sittings/<sitting>.jsonl; the prompt is kept beside it as
<sitting>.prompt.md, and stderr, when there is any, as <sitting>.stderr.txt. The
launcher looks at the session every POLL_SECONDS. At the cap (--minutes, default
DEFAULT_MINUTES) it kills the session's process tree: outcome timeout. When the STOP
file appears it gives the session STOP_GRACE_SECONDS to hand off on its own (every
verb now answers stopped, and the tier prompt says to hand off then), and kills it
after that: outcome stopped. Ctrl+C kills it too: outcome interrupted. The CLI reports
cost only in its result line, so a session killed before that line has no cost, and
the row says nothing rather than zero. A session that printed its result and was
killed while it lingered (the CLI waits for background work) is judged by its result.
Nothing here keeps the model's own words: the ledger and the answer carry numbers and
codes, and the transcript lives only in the stream file.

--rehearse runs the same command, environment, prompt shape and ledger rows with a fake
task (run python -m lab --help once, answer one line, change nothing) and a cap of
REHEARSE_MINUTES (a smaller --minutes lowers it); without a digest it still runs and
the prompt says why there is none. --dry-run answers the command, the launcher's
environment variables, the working directory, the instruction and the prompt's size;
it starts nothing and writes nothing.

run() is the same call for a caller in this process, such as the pool:
run("opus", Item("task", "O20")) returns the answer as a dict.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import io
import json
import logging
import math
import os
import re
import secrets
import shutil
import signal
import subprocess
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from lab.machine import allowed, ledger, record, refine_merge, stop
from lab.machine.lock import LockTimeout

log = logging.getLogger("lab.sit")

VERB = "sit"
#: The actor of the launcher's own ledger rows; the sitting id is in refs.run_id.
ACTOR = "lab sit"
REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = Path(__file__).with_name("prompts")

#: The refinement tracks a pass may name (lab/refine/<track>/): the one list, refine_merge.TRACKS,
#: so --pass and item_problem accept exactly the tracks the merge pass reconciles.
TRACKS: tuple[str, ...] = refine_merge.TRACKS
#: The tiers whose allow-lists carry the refine-track skill.
PASS_TIERS: tuple[str, ...] = ("fable", "opus")
#: The tier of the merge pass and the referee duty.
DUTY_TIER = "fable"
KINDS: tuple[str, ...] = ("task", "pass", "merge", "referee", "rehearse")
#: A Task Board ID: a context letter or two and a number (O20, L3, R11).
TASK_ID = re.compile(r"[A-Z]{1,4}[0-9]{1,4}")

WORKSPACE_ENV = "LAB_WORKSPACE"
WORKSPACE_MARKER: tuple[str, ...] = (".claude", "skills", "open-ticket", "SKILL.md")
#: The actor, tier and task names a sitting is started under: record.py's, the names every
#: ledger row of the sitting is read back from, so the launcher sets exactly what record() reads.
ACTOR_ENV = record.ENV_ACTOR
TIER_ENV = record.ENV_TIER
TASK_ENV = record.ENV_TASK
PYTHONPATH_ENV = "PYTHONPATH"
#: What the launcher sets in a session's environment beyond the state dir, actor and tier:
#: no auto memory, and no connectors of the owner's web account (see "Where it runs" above).
ISOLATION_ENV: dict[str, str] = {"CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1", "ENABLE_CLAUDEAI_MCP_SERVERS": "false"}
#: Variables that tie a process to the agent session it was started from. A launcher
#: started from inside a session inherits them; a sitting must not.
SESSION_VARS: tuple[str, ...] = (
    "CLAUDECODE",
    "CLAUDE_PID",
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_CHILD_SESSION",
    "CLAUDE_CODE_BRIDGE_SESSION_ID",
    "CLAUDE_CODE_MESSAGING_SOCKET",
    "CLAUDE_CODE_MESSAGING_TOKEN",
    "CLAUDE_CODE_SESSION_ATTENDED",
    "CLAUDE_CODE_SSE_PORT",
)
#: The variables a dry run shows: the ones the launcher decides. The rest of the
#: environment is never printed.
LAUNCHER_ENV_KEYS: tuple[str, ...] = (ledger.ENV_STATE_DIR, ACTOR_ENV, TIER_ENV, TASK_ENV, PYTHONPATH_ENV,
                                      *ISOLATION_ENV)

#: The agent CLI's command name on PATH.
COMMAND = "claude"
OUTPUT_FORMAT = "stream-json"
BATCH_SUFFIXES: tuple[str, ...] = (".cmd", ".bat")
# The target an npm shim forwards to: "%dp0%\node_modules\...\<cli>.exe" (or a .js run by node).
_SHIM_TARGET = re.compile(r'"%dp0%\\([^"\r\n]+?\.(?:exe|js))"', re.IGNORECASE)

SITTING_PREFIX = "sit-"
DIGEST_VERB = "digest"
DIGEST_HEADING = "# The digest (python -m lab digest, computed by code just before this sitting started)"
INSTRUCTION_HEADING = "# Your instruction"

DEFAULT_MINUTES = 240.0        # the cap of a sitting that names none
REHEARSE_MINUTES = 2.0         # the cap of a rehearsal
STOP_GRACE_SECONDS = 300.0     # how long a session may take to hand off after STOP appears
POLL_SECONDS = 1.0             # how often the launcher looks at a running session
TERM_WAIT_SECONDS = 5.0        # POSIX: how long SIGTERM gets before SIGKILL
KILL_WAIT_SECONDS = 30.0       # how long a killed session is waited for
READER_JOIN_SECONDS = 10.0     # how long the end of a finished session's output is waited for
MAX_DIGEST_CHARS = 60_000      # a longer digest is cut, with a note
STDERR_KEEP = 64 * 1024        # bytes of stderr kept (the last ones)
TAIL_CHARS = 300               # characters of output an error message quotes
SITTINGS_DIR = "sittings"      # under the state dir: streams, prompts, stderr

# outcomes of a sitting that ran (and of its end row)
OK = "ok"
ERROR = "error"
BUDGET = "budget"
TIMEOUT = "timeout"
STOPPED = "stopped"
INTERRUPTED = "interrupted"
STARTED = "started"            # the outcome of the start row
# outcomes of a call that started nothing
OUTCOME_USAGE = "usage"
OUTCOME_INSIDE = "inside_sitting"
OUTCOME_NO_WORKSPACE = "no_workspace"
OUTCOME_NO_PROMPT = "no_prompt"
OUTCOME_NO_CLI = "no_cli"
OUTCOME_NO_DIGEST = "no_digest"
OUTCOME_LEDGER = "ledger_error"
OUTCOME_DRY_RUN = "dry_run"

#: The CLI's result subtypes the launcher tells apart.
SUCCESS_SUBTYPE = "success"
BUDGET_SUBTYPE = "error_max_budget_usd"

REHEARSAL_TASK = (
    "This sitting is a rehearsal of the launcher with a fake task: it changes nothing and it is "
    "ended after {minutes:g} minutes. Ignore the Start, item, duty and End sections of the prompt "
    "above, and use no skill and no Notion tool. Do exactly two things: run python -m lab --help "
    "once (it only lists the verbs), then reply with one line, rehearsal ok {sitting} {tier} "
    "<how many verbs it listed>, and end your turn."
)


class CliMissing(RuntimeError):
    """There is no agent CLI to run."""


@dataclass(frozen=True)
class Item:
    """What a sitting is for: kind task (name is the Task Board ID), pass (name is the
    track), merge, referee or rehearse (no name)."""

    kind: str
    name: str | None = None

    @property
    def label(self) -> str:
        """"task O20", "pass telemetry", "merge", "referee" or "rehearse"."""
        return f"{self.kind} {self.name}" if self.name else self.kind

    @property
    def task(self) -> str | None:
        """The Task Board ID a ledger row names, or None."""
        return self.name if self.kind == "task" else None


def item_problem(tier: str, item: Item) -> str | None:
    """Why this tier cannot sit for this item, or None when it can."""
    if tier not in allowed.TIERS:
        return f"unknown tier {tier!r}; expected one of {', '.join(allowed.TIERS)}"
    if item.kind not in KINDS:
        return f"unknown item {item.kind!r}; expected one of {', '.join(KINDS)}"
    if item.kind == "task":
        if not isinstance(item.name, str) or TASK_ID.fullmatch(item.name) is None:
            return f"--task takes a Task Board ID such as O20, not {item.name!r}"
    elif item.kind == "pass":
        if item.name not in TRACKS:
            return f"--pass takes one of {', '.join(TRACKS)}, not {item.name!r}"
        if tier not in PASS_TIERS:
            return f"a refinement pass is for the {' or '.join(PASS_TIERS)} tier, not {tier}"
    elif item.name is not None:
        return f"--{item.kind} takes no name"
    elif item.kind in ("merge", "referee") and tier != DUTY_TIER:
        return f"--{item.kind} is a {DUTY_TIER} duty, not one for the {tier} tier"
    return None


# --- what the session is given ---------------------------------------------------------------


def resolve_command(command: str = COMMAND) -> list[str]:
    """The head of the session's argv: the CLI on PATH or, for an npm .cmd shim, the
    .exe it names (else a .js it names, run by node), so cmd.exe never touches the
    arguments. A shim with nothing runnable behind it is called as it is: the prompt goes
    on stdin, so only cmd.exe's 8191-character limit applies, which the vector keeps under.
    CliMissing when nothing is on PATH."""
    found = shutil.which(command)
    if not found:
        raise CliMissing(f"{command!r} is not on PATH; install the agent CLI and log in")
    shim = Path(found)
    if shim.suffix.lower() not in BATCH_SUFFIXES:
        return [found]
    try:
        text = shim.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        log.error("lab sit: could not read the CLI shim %s: %s", shim, exc)
        text = ""
    targets = [shim.parent.joinpath(*rel.replace("\\", "/").split("/")) for rel in _SHIM_TARGET.findall(text)]
    for target in targets:
        if target.suffix.lower() == ".exe" and target.name.lower().startswith(command) and target.is_file():
            return [str(target)]
    node = shutil.which("node")
    for target in targets:
        if target.suffix.lower() == ".js" and target.is_file() and node:
            return [node, str(target)]
    log.warning("lab sit: %s is a batch shim with nothing runnable found behind it; it is called as it is", found)
    return [found]


def session_command(program: Sequence[str], tier: str, budget_usd: float | None = None) -> list[str]:
    """The session's argv: the program, the print-mode flags, then allowed.cli_flags(tier),
    which must come last. The prompt is not in it: it goes on stdin."""
    cmd = [*program, "-p", "--model", tier, "--output-format", OUTPUT_FORMAT, "--verbose",
           "--no-session-persistence", "--permission-prompts", "none"]
    if budget_usd is not None:
        cmd += ["--max-budget-usd", f"{budget_usd:g}"]
    return cmd + allowed.cli_flags(tier)


def session_env(base: Mapping[str, str], *, state_dir: Path, sitting: str, tier: str, task: str | None = None,
                repo_root: Path = REPO_ROOT) -> dict[str, str]:
    """The session's whole environment: base without SESSION_VARS and without a LAB_TASK
    of its own, then the launcher's variables (LAB_TASK only for a task). PYTHONPATH keeps
    what base had, after this repo root."""
    env = {key: value for key, value in base.items() if key not in SESSION_VARS and key != TASK_ENV}
    env[ledger.ENV_STATE_DIR] = str(state_dir)
    env[ACTOR_ENV] = sitting
    env[TIER_ENV] = tier
    if task:
        env[TASK_ENV] = task
    inherited = base.get(PYTHONPATH_ENV)
    env[PYTHONPATH_ENV] = os.pathsep.join([str(repo_root), inherited]) if inherited else str(repo_root)
    env.update(ISOLATION_ENV)
    return env


def find_workspace(repo_root: Path = REPO_ROOT) -> Path | None:
    """The folder the session runs in: $LAB_WORKSPACE when set, else the nearest of the
    repo root and its three parents that holds .claude/skills/open-ticket/SKILL.md; None
    when the folder found has no such file."""
    configured = os.environ.get(WORKSPACE_ENV)
    candidates = [Path(configured)] if configured else [repo_root, *list(repo_root.parents)[:3]]
    for folder in candidates:
        if folder.joinpath(*WORKSPACE_MARKER).is_file():
            return folder.resolve()
    return None


@dataclass
class Digest:
    """What python -m lab digest answered: its stdout when ok, else why there is none."""

    ok: bool
    text: str = ""
    error: str | None = None
    truncated: bool = False

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {"ok": self.ok, "chars": len(self.text)}
        if self.truncated:
            out["truncated"] = True
        if self.error:
            out["error"] = self.error
        return out


def compute_digest() -> Digest:
    """Run python -m lab digest through the dispatcher in this process; whatever it prints
    is the digest. Not ok when the verb is not registered, exits non-zero or prints nothing."""
    from lab import __main__ as dispatcher  # imported here: the dispatcher loads verbs lazily

    if DIGEST_VERB not in dispatcher.VERBS:
        return Digest(False, error=f"python -m lab {DIGEST_VERB} is not registered in lab/__main__.py")
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            code = dispatcher.main([DIGEST_VERB])
    except Exception as exc:  # the dispatcher answers a verb's own errors; this is a failure around it
        return Digest(False, error=f"python -m lab {DIGEST_VERB} failed: {type(exc).__name__}: {exc}")
    text = buffer.getvalue().strip()
    if code != 0:
        return Digest(False, error=f"python -m lab {DIGEST_VERB} exited {code}: {_tail(text) or 'no output'}")
    if not text:
        return Digest(False, error=f"python -m lab {DIGEST_VERB} printed nothing")
    if len(text) > MAX_DIGEST_CHARS:
        cut = text[:MAX_DIGEST_CHARS] + f"\n[digest cut at {MAX_DIGEST_CHARS} of {len(text)} characters]"
        return Digest(True, cut, truncated=True)
    return Digest(True, text)


def instruction(item: Item, sitting: str, tier: str, workspace: Path, minutes: float,
                started: dt.datetime) -> str:
    """The last part of the prompt: who the sitting is, its cap, and its item."""
    stamp = started.astimezone(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    head = (f"You are sitting {sitting}, at the {tier.capitalize()} tier, started {stamp} by python -m lab sit, "
            f"which ends the sitting after {minutes:g} minutes. Your sitting id, for python -m lab handoff "
            f"--sitting, is {sitting}. You run in the workspace {workspace}; python -m lab works from any folder.")
    if item.kind == "task":
        body = f"Your item is task {item.name}: run /open-ticket {item.name}."
    elif item.kind == "pass":
        body = f"Your item is a refinement pass on the {item.name} track: run /refine-track {item.name}."
    elif item.kind == "merge":
        body = "Your duty is the merge pass: run /refine-merge."
    elif item.kind == "referee":
        body = "Your duty is the referee: run python -m lab referee first, then follow the Referee duty section."
    else:
        body = REHEARSAL_TASK.format(minutes=minutes, sitting=sitting, tier=tier)
    return f"{head}\n\n{body}"


def build_prompt(tier_prompt: str, digest: Digest, instruction_text: str) -> str:
    """The tier prompt, the digest (or why there is none), then the instruction, last."""
    body = digest.text if digest.ok else f"(no digest: {digest.error})"
    return "\n\n".join([tier_prompt.strip(), DIGEST_HEADING, body, INSTRUCTION_HEADING, instruction_text.strip()]) + "\n"


def new_sitting_id(tier: str, now: dt.datetime) -> str:
    """sit-YYYYMMDD-HHMMSS-<tier>-xxxx, in UTC; four hex so two launchers in one second differ."""
    return f"{SITTING_PREFIX}{now.astimezone(dt.timezone.utc):%Y%m%d-%H%M%S}-{tier}-{secrets.token_hex(2)}"


def sitting_paths(state_dir: Path, sitting: str) -> tuple[Path, Path, Path]:
    """The stream, prompt and stderr files of one sitting."""
    folder = state_dir / SITTINGS_DIR
    return folder / f"{sitting}.jsonl", folder / f"{sitting}.prompt.md", folder / f"{sitting}.stderr.txt"


# --- the session -----------------------------------------------------------------------------


@dataclass
class Session:
    """What the launcher saw of one CLI process."""

    returncode: int | None = None
    duration_s: float = 0.0
    killed: str | None = None          # timeout, stopped or interrupted: why the launcher ended it
    stop_seen: bool = False
    error: str | None = None           # the process could not be started
    init: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    other: list[str] = field(default_factory=list)   # the last stdout lines that were no JSON object
    stderr: str = ""
    stream_error: str | None = None


class _StdoutReader(threading.Thread):
    """Reads the session's stdout to its end: each line goes to the stream file as it
    arrives, the first init event and the last result event are kept. A stream file that
    cannot be written costs the record, never the session."""

    def __init__(self, pipe: IO[bytes], path: Path) -> None:
        super().__init__(name="lab-sit-stdout", daemon=True)
        self._pipe = pipe
        self._path = path
        self.init: dict[str, Any] | None = None
        self.result: dict[str, Any] | None = None
        self.other: deque[str] = deque(maxlen=20)
        self.error: str | None = None

    def run(self) -> None:
        sink: IO[bytes] | None = None
        try:
            sink = open(self._path, "ab")
        except OSError as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            log.error("lab sit: the stream file %s cannot be written (%s); the session runs unrecorded",
                      self._path, exc)
        try:
            for raw in iter(self._pipe.readline, b""):
                if sink is not None:
                    try:
                        sink.write(raw if raw.endswith(b"\n") else raw + b"\n")
                        sink.flush()
                    except OSError as exc:
                        self.error = f"{type(exc).__name__}: {exc}"
                        log.error("lab sit: writing the stream file %s failed (%s); the rest is unrecorded",
                                  self._path, exc)
                        with contextlib.suppress(OSError):
                            sink.close()
                        sink = None
                self._take(raw)
        except (OSError, ValueError) as exc:
            log.error("lab sit: reading the session's output stopped: %s", exc)
        finally:
            if sink is not None:
                with contextlib.suppress(OSError):
                    sink.close()

    def _take(self, raw: bytes) -> None:
        text = raw.decode("utf-8", errors="replace").strip()
        if not text:
            return
        try:
            doc = json.loads(text)
        except ValueError:
            doc = None
        if not isinstance(doc, dict):
            self.other.append(text)
        elif doc.get("type") == "result":
            self.result = doc
        elif doc.get("type") == "system" and doc.get("subtype") == "init" and self.init is None:
            self.init = doc


class _StderrReader(threading.Thread):
    """Reads the session's stderr to its end, keeping the last STDERR_KEEP bytes."""

    def __init__(self, pipe: IO[bytes]) -> None:
        super().__init__(name="lab-sit-stderr", daemon=True)
        self._pipe = pipe
        self._kept = bytearray()

    def run(self) -> None:
        read = getattr(self._pipe, "read1", self._pipe.read)
        try:
            for chunk in iter(lambda: read(4096), b""):
                self._kept += chunk
                if len(self._kept) > STDERR_KEEP:
                    del self._kept[:-STDERR_KEEP]
        except (OSError, ValueError) as exc:
            log.error("lab sit: reading the session's stderr stopped: %s", exc)

    def text(self) -> str:
        return self._kept.decode("utf-8", errors="replace")


def _feed(pipe: IO[bytes], data: bytes) -> None:
    """Write the prompt to the session's stdin and close it, in a thread of its own so a
    session that never reads cannot hold the launcher."""
    try:
        pipe.write(data)
        pipe.flush()
    except (OSError, ValueError) as exc:
        log.warning("lab sit: the session closed its stdin before taking the whole prompt: %s", exc)
    finally:
        with contextlib.suppress(OSError, ValueError):
            pipe.close()


def _new_group() -> dict[str, Any]:
    """POSIX: the session leads a process group of its own, so kill_tree reaches its children."""
    return {} if os.name == "nt" else {"start_new_session": True}


def kill_tree(proc: subprocess.Popen) -> None:
    """End the session and every process it started: taskkill /T /F on Windows; on POSIX
    SIGTERM to its process group (the CLI's own clean exit), then SIGKILL."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=KILL_WAIT_SECONDS, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            log.error("lab sit: taskkill of the session (pid %s) failed: %s", proc.pid, exc)
    else:
        with contextlib.suppress(OSError):
            os.killpg(proc.pid, signal.SIGTERM)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=TERM_WAIT_SECONDS)
        with contextlib.suppress(OSError):
            os.killpg(proc.pid, signal.SIGKILL)
    if proc.poll() is None:
        with contextlib.suppress(OSError):
            proc.kill()


def run_session(cmd: Sequence[str], prompt: str, *, cwd: Path, env: Mapping[str, str], stream_path: Path,
                cap_seconds: float) -> Session:
    """Run one session to its end: the prompt on stdin, stdout copied to stream_path as it
    comes, the process tree killed at cap_seconds, or STOP_GRACE_SECONDS after the STOP
    file appeared, or on Ctrl+C."""
    session = Session()
    started = time.monotonic()
    with contextlib.suppress(OSError):  # a folder that cannot be made shows up as the stream's error
        stream_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.Popen(list(cmd), cwd=str(cwd), env=dict(env), stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, **_new_group())
    except OSError as exc:
        session.error = f"could not run {cmd[0]}: {exc}"
        session.duration_s = round(time.monotonic() - started, 1)
        return session
    reader = _StdoutReader(proc.stdout, stream_path)
    errors = _StderrReader(proc.stderr)
    reader.start()
    errors.start()
    threading.Thread(target=_feed, args=(proc.stdin, prompt.encode("utf-8")), name="lab-sit-stdin",
                     daemon=True).start()
    deadline = started + cap_seconds
    stop_deadline: float | None = None
    try:
        while True:
            try:
                proc.wait(timeout=POLL_SECONDS)
                break
            except subprocess.TimeoutExpired:
                pass
            now = time.monotonic()
            if stop.stopped():
                session.stop_seen = True
                if stop_deadline is None:
                    stop_deadline = now + STOP_GRACE_SECONDS
                    log.warning("lab sit: the STOP file appeared; the session has %.0f seconds to hand off",
                                STOP_GRACE_SECONDS)
            else:
                stop_deadline = None
            if now >= deadline:
                session.killed = TIMEOUT
            elif stop_deadline is not None and now >= stop_deadline:
                session.killed = STOPPED
            if session.killed:
                log.warning("lab sit: the session is killed (%s)", session.killed)
                kill_tree(proc)
                break
    except KeyboardInterrupt:
        session.killed = INTERRUPTED
        kill_tree(proc)
    finally:
        if proc.poll() is None:
            kill_tree(proc)
        try:
            proc.wait(timeout=KILL_WAIT_SECONDS)
        except subprocess.TimeoutExpired:
            log.error("lab sit: the session (pid %s) has not exited %.0f seconds after it was killed",
                      proc.pid, KILL_WAIT_SECONDS)
        reader.join(READER_JOIN_SECONDS)
        errors.join(READER_JOIN_SECONDS)
    session.returncode = proc.returncode
    session.duration_s = round(time.monotonic() - started, 1)
    session.init, session.result = reader.init, reader.result
    session.other = list(reader.other)
    session.stderr = errors.text()
    session.stream_error = reader.error
    return session


# --- how it ended ----------------------------------------------------------------------------


def _tail(text: str) -> str:
    one = " ".join((text or "").split())
    return one if len(one) <= TAIL_CHARS else "..." + one[-TAIL_CHARS:]


def _number(value: Any) -> int | float | None:
    """value as a finite int or float; None for anything else, a bool included."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def exit_reason(session: Session) -> tuple[str, str | None]:
    """The outcome of a session that ran, and for an error what was seen. A result line
    decides when there is one, even if the launcher killed the process after it; the
    model's own words (the result's text) are never quoted."""
    if session.error:
        return ERROR, session.error
    doc = session.result
    if doc is not None:
        subtype = str(doc.get("subtype") or "")
        if subtype == BUDGET_SUBTYPE:
            return BUDGET, None
        if subtype == SUCCESS_SUBTYPE and not doc.get("is_error"):
            return OK, None
        return ERROR, f"the CLI reported {subtype or 'an error'} (exit {session.returncode})"
    if session.killed:
        return session.killed, None
    seen = _tail(session.stderr) or _tail(" ".join(session.other)) or "no output"
    return ERROR, f"the CLI exited {session.returncode} and printed no result: {seen}"


def tool_prefix_name(server: str) -> str:
    """A server's name as its tools carry it (mcp__<name>__<tool>): every character other
    than a letter, a digit, _ or - becomes _ (plugin:design:notion is plugin_design_notion)."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", server)


def notion_status(init: Mapping[str, Any] | None) -> str | None:
    """The status the init event gives the MCP server whose tools are named
    mcp__<allowed.NOTION_SERVER>__...; "absent" when the session started without it,
    None when there was no init event."""
    if not init:
        return None
    for server in init.get("mcp_servers") or []:
        name = server.get("name") if isinstance(server, Mapping) else None
        if isinstance(name, str) and tool_prefix_name(name) == allowed.NOTION_SERVER:
            return str(server.get("status") or "unknown")
    return "absent"


def session_facts(session: Session) -> dict[str, Any]:
    """Cost, turns and permission denials from the result line, and the model and the
    Notion server's status from the init event; None for whatever the CLI did not say."""
    doc = session.result or {}
    denials = doc.get("permission_denials")
    model = (session.init or {}).get("model")
    return {
        "cost_usd": _number(doc.get("total_cost_usd")),
        "turns": _number(doc.get("num_turns")),
        "denials": len(denials) if isinstance(denials, list) else None,
        "notion": notion_status(session.init),
        "model_used": model if isinstance(model, str) else None,
    }


def _start_row(sitting: str, tier: str, item: Item, cap: float) -> dict[str, Any]:
    return {"actor": ACTOR, "tier": tier, "task": item.task, "verb": VERB, "outcome": STARTED,
            "metrics": {}, "refs": {"run_id": sitting}, "item": item.label, "model": tier,
            "rehearse": item.kind == "rehearse", "cap_minutes": cap}


def _end_row(sitting: str, tier: str, item: Item, session: Session, reason: str, error: str | None,
             facts: Mapping[str, Any]) -> dict[str, Any]:
    metrics: dict[str, int | float] = {"sitting.duration_s": session.duration_s}
    units = {"sitting.duration_s": "s"}
    if facts["cost_usd"] is not None:
        metrics["sitting.cost_usd"] = facts["cost_usd"]
        units["sitting.cost_usd"] = "USD"
    if facts["turns"] is not None:
        metrics["sitting.turns"] = facts["turns"]
    if facts["denials"] is not None:
        metrics["sitting.denials"] = facts["denials"]
    row: dict[str, Any] = {
        "actor": ACTOR, "tier": tier, "task": item.task, "verb": VERB, "outcome": reason,
        "metrics": metrics, "units": units, "refs": {"run_id": sitting}, "item": item.label,
        "rehearse": item.kind == "rehearse", "exit_code": session.returncode, "killed": session.killed,
        "stop_seen": session.stop_seen, "notion": facts["notion"], "model_used": facts["model_used"],
    }
    if error:
        row["note"] = error
    return row


# --- the verb --------------------------------------------------------------------------------


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        log.error("lab sit: could not read %s: %s", path, exc)
        return ""


def _save(path: Path, text: str) -> None:
    """Keep a file beside the stream; a failure is logged and costs nothing else."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    except OSError as exc:
        log.error("lab sit: could not write %s: %s", path, exc)


def _stopped_answer() -> dict[str, Any]:
    return {"ok": False, "outcome": STOPPED, "reason": stop.stop_reason() or stop.NO_REASON,
            "file": str(stop.stop_file())}


def run(tier: str, item: Item, *, minutes: float | None = None, budget_usd: float | None = None,
        dry_run: bool = False) -> dict[str, Any]:
    """Start one sitting of this tier for this item and wait for it to end; the answer
    as a dict (see the module docstring). minutes is the cap, budget_usd the CLI's
    --max-budget-usd; dry_run answers what would run and starts nothing."""
    tier = str(tier).strip().lower()
    problem = item_problem(tier, item)
    for name, value in (("minutes", minutes), ("budget_usd", budget_usd)):
        if problem is None and value is not None and not (_number(value) is not None and value > 0):
            problem = f"{name} must be a number above 0, not {value!r}"
    if problem:
        return {"ok": False, "outcome": OUTCOME_USAGE, "error": problem}
    inside = os.environ.get(ACTOR_ENV, "")
    if inside.startswith(SITTING_PREFIX):
        return {"ok": False, "outcome": OUTCOME_INSIDE,
                "error": f"python -m lab sit never runs inside a sitting, and this is sitting {inside}"}
    if stop.stopped():
        return _stopped_answer()
    rehearse = item.kind == "rehearse"
    cap = min(minutes or REHEARSE_MINUTES, REHEARSE_MINUTES) if rehearse else (minutes or DEFAULT_MINUTES)
    base: dict[str, Any] = {"tier": tier, "item": item.label, "model": tier, "rehearse": rehearse, "minutes": cap}
    workspace = find_workspace()
    if workspace is None:
        where = os.environ.get(WORKSPACE_ENV) or f"{REPO_ROOT} or the three folders above it"
        return {"ok": False, "outcome": OUTCOME_NO_WORKSPACE, **base,
                "error": f"no workspace: {Path(*WORKSPACE_MARKER)} is not in {where}"}
    prompt_file = PROMPTS_DIR / f"{tier}.md"
    tier_prompt = _read(prompt_file)
    if not tier_prompt.strip():
        return {"ok": False, "outcome": OUTCOME_NO_PROMPT, **base, "error": f"no tier prompt at {prompt_file}"}
    try:
        program = resolve_command()
    except CliMissing as exc:
        return {"ok": False, "outcome": OUTCOME_NO_CLI, **base, "error": str(exc)}
    digest = compute_digest()
    if not digest.ok and not (rehearse or dry_run):
        return {"ok": False, "outcome": OUTCOME_NO_DIGEST, **base, "digest": digest.summary(),
                "error": f"a sitting starts from the digest, and there is none: {digest.error}"}
    state_dir = ledger.resolve_state_dir().resolve()
    started = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    sitting = new_sitting_id(tier, started)
    text = instruction(item, sitting, tier, workspace, cap, started)
    prompt = build_prompt(tier_prompt, digest, text)
    cmd = session_command(program, tier, budget_usd)
    env = session_env(os.environ, state_dir=state_dir, sitting=sitting, tier=tier, task=item.task)
    base = {"sitting": sitting, **base}
    if dry_run:
        return {"ok": True, "outcome": OUTCOME_DRY_RUN, **base, "command": cmd, "cwd": str(workspace),
                "env": {key: env.get(key) for key in LAUNCHER_ENV_KEYS},
                "env_removed": sorted(key for key in SESSION_VARS if key in os.environ),
                "instruction": text, "prompt_file": str(prompt_file), "prompt_chars": len(prompt),
                "digest": digest.summary(), "state_dir": str(state_dir)}
    if stop.stopped():
        return _stopped_answer()
    stream, prompt_copy, stderr_file = sitting_paths(state_dir, sitting)
    try:
        start_seq = ledger.append(_start_row(sitting, tier, item, cap), state_dir=state_dir)
    except (OSError, LockTimeout, ledger.RowError) as exc:
        return {"ok": False, "outcome": OUTCOME_LEDGER, **base,
                "error": f"the start row could not be written, so the sitting did not start: "
                         f"{type(exc).__name__}: {exc}"}
    _save(prompt_copy, prompt)
    log.info("lab sit: sitting %s starts (%s, %s, %d prompt characters, cap %g minutes)",
             sitting, tier, item.label, len(prompt), cap)
    try:
        session = run_session(cmd, prompt, cwd=workspace, env=env, stream_path=stream, cap_seconds=cap * 60)
    except Exception as exc:  # whatever happened, the start row gets its end row
        log.exception("lab sit: sitting %s failed", sitting)
        session = Session(error=f"{type(exc).__name__}: {exc}")
    if session.stderr.strip():
        _save(stderr_file, session.stderr)
    reason, error = exit_reason(session)
    facts = session_facts(session)
    answer: dict[str, Any] = {
        "ok": reason == OK, "outcome": reason, **base, "exit_code": session.returncode,
        "duration_s": session.duration_s, **facts, "killed": session.killed, "stop_seen": session.stop_seen,
        "digest": digest.summary(), "stream": str(stream), "ledger": {"start": start_seq, "end": None},
    }
    if error:
        answer["error"] = error
    if session.stream_error:
        answer["stream_error"] = session.stream_error
    try:
        answer["ledger"]["end"] = ledger.append(_end_row(sitting, tier, item, session, reason, error, facts),
                                                state_dir=state_dir)
    except (OSError, LockTimeout, ledger.RowError) as exc:
        log.error("lab sit: the end row of sitting %s could not be written: %s", sitting, exc)
        answer["ledger_error"] = f"the end row could not be written: {type(exc).__name__}: {exc}"
    return answer


def _task_id(text: str) -> str:
    value = text.strip().upper()
    if TASK_ID.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(f"a Task Board ID such as O20, not {text!r}")
    return value


def _positive(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"a number above 0, not {text!r}") from None
    if not (math.isfinite(value) and value > 0):
        raise argparse.ArgumentTypeError(f"a number above 0, not {text!r}")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lab sit",
        description="Start one headless agent session (the agent CLI in print mode) for one item, wait for it to end, "
                    "and record its start and end in the ledger.")
    parser.add_argument("--tier", required=True, type=str.lower, choices=allowed.TIERS,
                        help="the model tier, which picks the model, the allow-list and the prompt")
    item = parser.add_mutually_exclusive_group(required=True)
    item.add_argument("--task", type=_task_id, metavar="ID", help="a Task Board task, through /open-ticket <ID>")
    item.add_argument("--pass", dest="track", type=str.lower, choices=TRACKS,
                      help="a refinement pass on one track, through /refine-track <track> (opus or fable)")
    item.add_argument("--merge", action="store_true", help="the merge pass, through /refine-merge (fable)")
    item.add_argument("--referee", action="store_true", help="the referee duty (fable)")
    item.add_argument("--rehearse", action="store_true",
                      help=f"a fake task that changes nothing, capped at {REHEARSE_MINUTES:g} minutes")
    parser.add_argument("--minutes", type=_positive,
                        help=f"kill the session after this many minutes (default {DEFAULT_MINUTES:g}; "
                             f"a rehearsal never runs past {REHEARSE_MINUTES:g})")
    parser.add_argument("--budget-usd", type=_positive, help="passed to the CLI as --max-budget-usd")
    parser.add_argument("--dry-run", action="store_true",
                        help="answer the command, environment and prompt size; start and write nothing")
    return parser


def _item(args: argparse.Namespace) -> Item:
    if args.task:
        return Item("task", args.task)
    if args.track:
        return Item("pass", args.track)
    for kind in ("merge", "referee", "rehearse"):
        if getattr(args, kind):
            return Item(kind)
    raise AssertionError("argparse requires one item")


def main(argv: Sequence[str]) -> dict[str, Any]:
    """Start one headless agent sitting for a task, a pass, the merge or the referee, and record it."""
    parser = _parser()
    args = parser.parse_args(list(argv))
    item = _item(args)
    problem = item_problem(args.tier, item)
    if problem:
        parser.error(problem)
    return run(args.tier, item, minutes=args.minutes, budget_usd=args.budget_usd, dry_run=args.dry_run)
