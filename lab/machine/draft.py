"""draft: a sitting asks for a verb it lacks. It leaves the owner a request and a stub module.

    python -m lab draft --name <verb> --goal "<what the verb should do and why>"

A sitting that needs a verb the lab does not have runs this once and carries on without it, instead
of scripting around the gap. Nothing here builds the verb: the draft is a request for the owner, and
no code in it is imported or run by the lab. What the call does, in this order, holding one lock
(drafts.lock in the state dir) from the first check to the ledger row, so two calls never interleave:

  1. Checks the arguments. The name is the verb as it would be registered: lowercase words and
     digits joined by single hyphens, starting with a letter, at most 40 characters, a legal module
     name once the hyphens become underscores, not a Windows device name, not "rejected". It must
     not be a registered verb or a module of lab/machine. The goal is not blank and at most 1000
     characters. A refused call spends nothing: the sitting may call again with a better argument.
  2. Refuses a name that already has a draft (a rejected one included), and refuses a second draft
     from the same sitting. A sitting's draft is spent as soon as it exists: any request.json under
     <state dir>/drafts/ naming the sitting counts, those the owner moved under drafts/rejected/
     too, and so does a ledger row of a drafted verb for it. This is enforced here, never by the
     prompt. Outside a sitting (the owner at a keyboard) there is no per-sitting limit.
  3. Writes <state dir>/drafts/<name>/ with two files. request.json holds the request: name,
     module (the name with its hyphens as underscores), goal, sitting, tier, actor and requested
     (UTC). <module>.py is a stub that follows the dispatcher's verb contract (lab/__main__.py):
     main(argv) answers not_built, and the goal sits in a string constant, escaped, so no goal can
     become code. Both files appear together or not at all: they are built in a hidden folder that
     is renamed into place.
  4. Appends one ledger row with record.record(): verb draft, outcome drafted, the actor, tier and
     task from $LAB_ACTOR, $LAB_TIER and $LAB_TASK (record.py says what an unset one becomes), the
     sitting, the draft's name and the goal's length (goal_chars). The row never holds the goal:
     sittings read the ledger and the digest, and the goal is a sitting's own words. A ledger row
     that cannot be written withdraws the draft and fails the call, so a draft exists exactly when
     the ledger says so.
  5. Answers with one JSON object: ok, outcome drafted, draft, module, files, sitting, ledger_seq
     and next. Neither the goal nor a path is in it; a failure is named by its type and the
     system's message. The exit code is 0 when the draft was written and 1 when it was not.

Who the sitting is. The launcher gives every sitting LAB_ACTOR (its id) and LAB_TIER. The call
belongs to a sitting only when both are set; the id cannot be chosen by an argument, so a sitting
cannot dodge the limit by naming itself differently.

The owner reviews a draft by reading its folder. To adopt one, build the verb as
lab/machine/<module>.py, add the line the stub names to VERBS in lab/__main__.py and write its tests.
To reject one, move its folder under <state dir>/drafts/rejected/: its name stays taken, and the
sitting that asked stays spent.

The dispatcher refuses it while the STOP file exists (it is not @stop.exempt, as handoff is): asking
for a new verb is starting something new.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import keyword
import logging
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lab.machine import ledger, record
from lab.machine.lock import FileLock, LockTimeout

__all__ = ["VERB", "OUTCOME", "DRAFTS_DIR", "REJECTED_DIR", "REQUEST_FILE", "CLAIM_LOCK", "ENV_ACTOR", "ENV_TIER",
           "MAX_NAME_CHARS", "MAX_GOAL_CHARS", "LOCK_TIMEOUT", "identity", "module_name", "name_problem",
           "drafts_path", "requests", "draft", "main"]

log = logging.getLogger("lab.draft")

VERB = "draft"
CLI_VERB = "draft"   # registers python -m lab draft
OUTCOME = "drafted"
DRAFTS_DIR = "drafts"             # under the state dir
REJECTED_DIR = "rejected"         # under the drafts dir: where the owner moves a draft he turns down
REQUEST_FILE = "request.json"
CLAIM_LOCK = "drafts.lock"        # in the state dir, beside ledger.lock
ENV_ACTOR = record.ENV_ACTOR
ENV_TIER = record.ENV_TIER
MAX_NAME_CHARS = 40
MAX_GOAL_CHARS = 1000
LOCK_TIMEOUT = ledger.LOCK_TIMEOUT   # seconds a call waits while another draft is being written
NEXT = "carry on without this verb; the draft waits for the owner and nothing in it runs"

MACHINE_DIR = Path(__file__).resolve().parent
_NAME = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")
_WINDOWS_DEVICE = re.compile(r"(?:con|prn|aux|nul|com[1-9]|lpt[1-9])")

StateDir = str | os.PathLike[str] | None

_STUB = '''\
"""Draft of the lab verb {name}: a stub, not part of the lab.

Written by python -m lab draft; request.json beside it says who asked and when. Nothing in lab/
imports or runs this file. To adopt the draft, build the verb as lab/machine/{module}.py (main
gets everything after the verb name; return a dict and its "ok" key picks the exit code), add this
line to VERBS in lab/__main__.py, keeping the names alphabetical, and test it in
tests/test_lab_{module}.py:

    "{name}": "lab.machine.{module}:main",

To reject the draft, move this folder into ../rejected/.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

NAME = "{name}"
# What the requester wants this verb to do and why, in its own words.
GOAL = {goal}


def main(argv: Sequence[str]) -> dict[str, Any]:
    """{name}: a draft that is not built yet."""
    return {{"ok": False, "outcome": "not_built", "verb": NAME, "error": "this verb is only a draft; see GOAL"}}
'''


# --- who is asking ------------------------------------------------------------------------

def identity(environ: Mapping[str, str] | None = None) -> tuple[str | None, str | None, str | None]:
    """(actor, tier, sitting) of the calling process: $LAB_ACTOR, $LAB_TIER lower-cased, and the sitting
    id, which is the actor when the tier is also set (the launcher sets both for every sitting) and None
    for anyone else. Blank values count as not set."""
    env = os.environ if environ is None else environ
    actor = (env.get(ENV_ACTOR) or "").strip() or None
    tier = (env.get(ENV_TIER) or "").strip().lower() or None
    return actor, tier, (actor if actor and tier else None)


# --- the arguments ------------------------------------------------------------------------

def module_name(name: str) -> str:
    """The module a verb is built in: the name with its hyphens as underscores."""
    return name.replace("-", "_")


def name_problem(name: str) -> str | None:
    """Why name cannot be a draft's name, as a sentence; None when it can. Says nothing about whether
    a verb or a draft of that name already exists."""
    if not name:
        return "--name is blank"
    shown = name[:60] + ("..." if len(name) > 60 else "")
    if not _NAME.fullmatch(name):
        return (f"--name must be the verb as it would be registered: lowercase letters and digits in words "
                f"joined by single hyphens, starting with a letter, such as research-check; got {shown!r}")
    if len(name) > MAX_NAME_CHARS:
        return f"--name is {len(name)} characters; the limit is {MAX_NAME_CHARS}"
    if keyword.iskeyword(module_name(name)):
        return f"--name {name!r} is a Python keyword, so no module can carry it"
    if _WINDOWS_DEVICE.fullmatch(name):
        return f"--name {name!r} is a Windows device name, so no folder can carry it"
    if name == REJECTED_DIR:
        return f"--name {name!r} is the folder of rejected drafts"
    return None


def _goal(text: object) -> str:
    """The goal as written, stripped, with any character UTF-8 cannot hold replaced by ?."""
    return str(text).encode("utf-8", errors="replace").decode("utf-8").strip()


def _goal_problem(goal: str) -> str | None:
    if not goal:
        return "--goal is blank"
    if len(goal) > MAX_GOAL_CHARS:
        return (f"--goal is {len(goal)} characters; the limit is {MAX_GOAL_CHARS}: say what the verb should do "
                f"and why in a sentence or two")
    return None


def _collision(name: str) -> str | None:
    """Why a verb of this name cannot be drafted because one already exists; None when none does."""
    from lab.__main__ import VERBS   # here, not at the top: the dispatcher loads verbs lazily, so no cycle

    if name in VERBS:
        return f"the verb {name} exists: run python -m lab {name} -h and use it"
    module = module_name(name)
    for taken in (MACHINE_DIR / f"{module}.py", MACHINE_DIR / module):
        if taken.exists():
            shown = f"lab/machine/{taken.name}" + ("/" if taken.is_dir() else "")
            return f"{shown} exists, so a verb named {name} would collide with it; choose another name"
    return None


def _moment(now: dt.datetime | None) -> dt.datetime:
    """The time of the call as an aware UTC time to the second; ValueError for a naive time."""
    if now is None:
        return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    if now.utcoffset() is None:
        raise ValueError(f"now must carry a UTC offset, not the naive {now.isoformat()}")
    return now.astimezone(dt.timezone.utc).replace(microsecond=0)


# --- the drafts on disk -------------------------------------------------------------------

def drafts_path(state_dir: StateDir = None) -> Path:
    """<state dir>/drafts: one folder per draft, the rejected ones under rejected/."""
    return ledger.resolve_state_dir(state_dir) / DRAFTS_DIR


def requests(state_dir: StateDir = None) -> list[dict[str, Any]]:
    """Every draft's request, the owner's rejected ones included, in folder order: the stored fields
    plus rejected (true for a draft moved under rejected/). A name missing from request.json is read from
    the folder. A folder without a readable request is skipped and logged; a hidden folder is a draft
    still being written. Takes no lock and creates nothing."""
    root = drafts_path(state_dir)
    found: list[dict[str, Any]] = []
    for rejected, pattern in ((False, f"*/{REQUEST_FILE}"), (True, f"{REJECTED_DIR}/*/{REQUEST_FILE}")):
        for path in sorted(root.glob(pattern)):
            folder = path.parent
            if folder.name.startswith(".") or (not rejected and folder.name == REJECTED_DIR):
                continue
            try:
                request = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                log.warning("draft: %s/%s is unreadable (%s); not counted", folder.name, REQUEST_FILE, exc)
                continue
            if isinstance(request, dict):
                found.append({**request, "name": request.get("name") or folder.name, "rejected": rejected})
    return found


def _prior_draft(sitting: str, state_dir: StateDir) -> str | None:
    """The name of the draft this sitting already made, from the folders first, then the ledger (the owner
    may have deleted a folder; the row stays). None when the sitting has made none."""
    for request in requests(state_dir):
        if request.get("sitting") == sitting:
            return str(request["name"])
    for row in ledger.read(state_dir=state_dir, verb=VERB, outcome=OUTCOME, sitting=sitting):
        return str(row.get("draft") or "an earlier draft")
    return None


def _write(path: Path, text: str) -> None:
    """text as UTF-8 with plain newlines, flushed and fsynced."""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())


def _stage(drafts: Path, name: str, module: str, request: Mapping[str, Any]) -> Path:
    """Build the draft's folder under a hidden name beside its final place and rename it into place, so
    both files appear together. Nothing is left behind when it fails."""
    staging = drafts / f".{name}.{os.getpid()}.tmp"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    final = drafts / name
    try:
        _write(staging / f"{module}.py", _STUB.format(name=name, module=module, goal=ascii(request["goal"])))
        _write(staging / REQUEST_FILE, json.dumps(request, ensure_ascii=False, indent=2) + "\n")
        os.replace(staging, final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return final


def _why(exc: BaseException) -> str:
    """A failure as the answer names it: its type and the system's message, never a path."""
    if isinstance(exc, LockTimeout):
        return "LockTimeout: another writer held the lock for the whole wait"
    if isinstance(exc, OSError):
        return f"{type(exc).__name__}: {exc.strerror or 'not written'}"
    return f"{type(exc).__name__}: {exc}"


def _refused(outcome: str, error: str, **more: Any) -> dict[str, Any]:
    return {"ok": False, "outcome": outcome, "error": error, **more}


# --- the verb -----------------------------------------------------------------------------

def _claim(name: str, module: str, request: dict[str, Any], state_dir: StateDir, timeout: float) -> dict[str, Any]:
    """Everything after the lock is held: the two refusals, the files, the ledger row."""
    drafts = drafts_path(state_dir)
    if (drafts / name).exists() or (drafts / REJECTED_DIR / name).exists():
        return _refused("name_taken", f"a draft named {name} already exists or was rejected; choose another name "
                                      f"or carry on without it")
    sitting = request["sitting"]
    if sitting is not None:
        prior = _prior_draft(sitting, state_dir)
        if prior is not None:
            return _refused("already_drafted", f"this sitting already drafted {prior}: one draft per sitting",
                            draft=prior)
    drafts.mkdir(parents=True, exist_ok=True)
    final = _stage(drafts, name, module, request)
    try:
        seq = record.record(VERB, OUTCOME, actor=request["actor"], tier=request["tier"], state_dir=state_dir,
                            timeout=timeout, sitting=sitting, draft=name, goal_chars=len(request["goal"]))
    except (OSError, ledger.RowError) as exc:
        shutil.rmtree(final, ignore_errors=True)
        log.error("draft: the ledger row of %s was not written; the draft was withdrawn: %s", name, _why(exc))
        return _refused("error", f"the ledger row was not written, so the draft was withdrawn: {_why(exc)}")
    return {"ok": True, "outcome": OUTCOME, "draft": name, "module": module,
            "files": sorted([REQUEST_FILE, f"{module}.py"]), "sitting": sitting, "ledger_seq": seq, "next": NEXT}


def draft(name: str, goal: str, *, sitting: str | None = None, actor: str | None = None, tier: str | None = None,
          state_dir: StateDir = None, now: dt.datetime | None = None, timeout: float = LOCK_TIMEOUT) -> dict[str, Any]:
    """Draft the verb name for the goal and answer what the verb prints. sitting is the id the one-draft
    limit is kept for (None: no limit). actor and tier go into the request and the ledger row; an actor
    not given is the sitting, else $LAB_ACTOR, and a tier not given is $LAB_TIER. now is an aware time
    (default: the current UTC time). A refusal is an answer with ok false and an outcome of usage, exists,
    name_taken, already_drafted, busy or error, and has written nothing. ValueError, with nothing written,
    for a naive now."""
    name = str(name).strip()
    goal = _goal(goal)
    problem = name_problem(name) or _goal_problem(goal)
    if problem:
        return _refused("usage", problem)
    collision = _collision(name)
    if collision:
        return _refused("exists", collision)
    moment = _moment(now)
    module = module_name(name)
    sitting = (sitting or "").strip() or None
    tier = (tier or "").strip().lower() or record.current_tier()
    actor = (actor or "").strip() or sitting or record.current_actor()
    request = {"name": name, "module": module, "goal": goal, "sitting": sitting, "tier": tier, "actor": actor,
               "requested": moment.strftime("%Y-%m-%dT%H:%M:%SZ")}
    try:
        with FileLock(ledger.resolve_state_dir(state_dir) / CLAIM_LOCK, timeout=timeout) as lock:
            lock.write_note(f"pid {os.getpid()} drafting {name}")
            return _claim(name, module, request, state_dir, timeout)
    except LockTimeout:
        return _refused("busy", "another draft is being written; call again in a moment")
    except Exception as exc:
        log.error("draft: %s was not written: %s", name, _why(exc), exc_info=not isinstance(exc, OSError))
        return _refused("error", f"the draft was not written: {_why(exc)}")


def main(argv: Sequence[str]) -> dict[str, Any]:
    """Ask for a verb the lab lacks: leaves the owner a request and a stub (once per sitting; nothing runs)."""
    parser = argparse.ArgumentParser(prog="python -m lab draft", description=main.__doc__, allow_abbrev=False)
    parser.add_argument("--name", required=True, metavar="VERB",
                        help="the verb as it would be registered: lowercase words joined by hyphens, such as "
                             "research-check")
    parser.add_argument("--goal", required=True, metavar="TEXT",
                        help="what the verb should do and why you need it, in a sentence or two (write "
                             "--goal=TEXT when it starts with a dash); the owner reads it, no sitting does")
    args = parser.parse_args(argv)
    actor, tier, sitting = identity()
    return draft(args.name, args.goal, sitting=sitting, actor=actor, tier=tier)
