"""python -m lab pool: keep the lab's sittings running under a hard budget of 16 units.

    python -m lab pool [--units N] [--minutes M] [--budget-usd X] [--board FILE] [--refine-ref REF]
                       [--board-max-age MIN] [--tick S] [--exit-when-idle]
    python -m lab pool --dry-run      the schedule it would start now; starts nothing, writes nothing
    python -m lab pool status         whether a pool runs, the units in use, its seats, what waits

The budget. At most BUDGET_UNITS (16) units run at once: a Sonnet sitting costs 1, an Opus
sitting 2 and a Fable sitting 8 (COST). Every sitting counts, not only the pool's own: one
started by hand (python -m lab sit), or by an earlier pool process that died, holds its units
from its start row in the ledger until its end row, or until its cap plus LIVE_GRACE_MINUTES
has passed, and its item is not started a second time meanwhile. --units lowers the budget, to
leave room for agents the ledger does not see; nothing raises it.

No fixed cycle. Each sitting runs in a seat: a thread that runs python -m lab sit as a child
process and waits for its one-line answer. When a seat's sitting ends, the pool records it and
refills the freed units in the same step. Between ends it looks again every --tick seconds
(TICK_SECONDS), because the board snapshot, the refinement files, STOP and the clock change
without a sitting ending. A child process rather than sit.run() in a thread: sit reads the
digest through contextlib.redirect_stdout, which swaps sys.stdout for the whole process, so two
seats starting in threads at once would capture each other's output.

What it starts. Every candidate is one sit item for one tier:
- a task (any tier): a Task Board row of the board snapshot (python -m lab digest
  --save-board; --board reads another file) that is Ilan's alone, Not started, not Locked, has
  a Tier, and whose every Blocked by task is Done in the snapshot (a blocker missing from the
  snapshot holds it back). A slot verification is such a task at the Sonnet tier: sit has no
  item of its own for one, and the Sonnet prompt runs the GPU queue flow for a task that needs
  a slot. A task whose title names a refinement pass, a track loop or a merge pass is never
  started as a task; the pass and merge items below run it, so one pass never starts twice.
- a due pass (Opus, sit --pass <track>), by lab/refine/PASS_RULES.md. The track's files are
  read at --refine-ref (default origin/main, fetched every FETCH_MINUTES and read with git show,
  so a pass counts once its pull request has merged; "worktree" reads the working tree). A
  track is due on one of: its first pass (counter 0), a run in progress (its latest pass neither
  converged nor reached the cap), or PASS_EVERY_HOURS since its latest pass ended (the Ended
  line of its report). A first pass, and the first merge, start only through their open task on
  the board, never without one. It is not due while capped (cap reached and no answered ruling from that
  pass), while a pass of it runs (a seat, a live sitting in the ledger, or a Locked open pass
  task on the board), while its pass of the current round waits for the merge, while the merge
  runs, waits for its pull request or is due, or while its own track task on the board is
  blocked. Triggers 1 and 2 of PASS_RULES.md (a task reached Done, an unwalked queue
  complaint) need history the snapshot does not keep; the pass skill names the trigger it finds.
- the merge pass (Fable, sit --merge): once a track has a pass the merge log has not
  reconciled, no pass runs, and the open merge task on the board, if there is one, has its
  blockers Done. A Locked open merge task means a merge runs elsewhere: no merge starts and no
  pass either, until it has finished (PASS_RULES.md, "Work happens in rounds").
- the referee duty (Fable, sit --referee): only once python -m lab referee is registered, then
  when a fresh board snapshot newer than the last referee sitting shows GPU queue trouble (a
  running slot past its cap, a complaint, a queued row that waited twice its minutes), at most
  every REFEREE_TROUBLE_MINUTES, or REFEREE_EVERY_MINUTES after the last one ended.
Board-derived items (tasks, passes, the merge) wait while the snapshot is older than
--board-max-age minutes (digest.BOARD_STALE_MINUTES): a stale snapshot hides locks taken since.

The order. P0 before P1 before P2, then the earlier wave (none last), then stretch last, then
the merge, the referee, passes and tasks in that order, then the ID. A duty or pass takes the
priority and wave of its task on the board, else DEFAULT_RANK; a referee called by queue trouble
ranks as P0, wave 0. The pool starts candidates in that order while they fit in the free units and stops
at the first that does not fit, holding the free units for it rather than letting a later item
take them: a Fable sitting therefore starts only when 8 units are free, and smaller work never
starves it. An item that costs more than the budget is never offered.

After a sitting ends. The ledger rows the pool writes are its memory, so a restarted pool knows
what its predecessor did (the "pool end" rows of the last LEDGER_LOOKBACK_HOURS):
- ok: the item is not offered again until what it changes has moved: a task until a snapshot
  taken after the sitting ended, a pass until the track's counter passes the one it started from
  (its pull request merged), the merge until the merge log has a new entry. A pass or merge
  whose pull request has not merged after PROGRESS_WAIT_HOURS is offered again, so a sitting
  that ended without one cannot hold its track for ever.
- failed (error, budget, timeout): offered again at once, even if the board shows the task
  Locked or In progress, since that is most likely the failed sitting's own claim, which the
  open-ticket skill continues. After MAX_RESTARTS restarts in a row fail, the item is benched for
  BENCH_HOURS after its last failure. BREAKER_FAILURES failed sittings in a row, whatever the
  item, pause the pool for PAUSE_MINUTES.
- stopped or interrupted: offered again like a failure, without counting one.
- refused (python -m lab sit started nothing: no CLI, no workspace, no digest, a ledger that
  cannot be written): the pool pauses for PAUSE_MINUTES; the item counts no failure.

STOP. While lab/state/STOP exists the pool starts nothing; the sittings that run hand off and
end on their own (sit gives them a grace and then kills them), and the pool exits once the last
has ended. It does not wait for a resume: python -m lab resume, then start the pool again.
Ctrl+C drains the same way (the sittings in the same console get it too and end); a second
Ctrl+C leaves at once. The run is refused while STOP exists; status and --dry-run are not, as
status is how the owner watches a pool drain (main is marked exempt and guards the run itself).

Ledger rows, actor ACTOR, stamped with the pool's clock, one per decision: pool begin, pool
start (a seat starts: tier, task, item, seat, units_held, source, why, the progress it must
move), pool end (outcome ended; the sitting's own outcome in sitting_outcome, how the pool judged
it in result, refs.run_id the sitting id), pool bench, pool pause, pool hold (the head of the
order waits for units; written when the held item changes), pool stop (STOP seen), pool drain
(Ctrl+C) and pool exit. Nothing a model wrote enters a row; an error text is sit's own message,
cut at ERROR_CHARS.

State. The running pool holds <state dir>/pool.lock for its whole life, so a second pool is
refused and status can tell a live pool from a dead one, and rewrites <state dir>/pool.json
(its seats, units, what is held, benched, paused) after every step. status reads both and the
ledger; --dry-run computes the same view a pool would and prints the order, starting nothing
and writing nothing (it may fetch --refine-ref).
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import logging
import math
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from lab.machine import digest, ledger, refine_merge, sit, stop
from lab.machine import refine_formats as rf
from lab.machine.lock import FileLock, LockTimeout

log = logging.getLogger("lab.pool")

ACTOR = "lab pool"
REPO_ROOT = Path(__file__).resolve().parents[2]
REFINE_DIR = "lab/refine"

#: Units a sitting of each tier holds while it runs.
COST: dict[str, int] = {"sonnet": 1, "opus": 2, "fable": 8}
#: At most this many units run at once.
BUDGET_UNITS = 16
#: The board's Tier select, as sit's tier names.
BOARD_TIERS: dict[str, str] = {"Fable": "fable", "Opus": "opus", "Sonnet": "sonnet"}
PASS_TIER = "opus"
DUTY_TIER = "fable"

TICK_SECONDS = 30.0            # how often the pool looks again while no sitting ends
MAX_RESTARTS = 2               # restarts of an item after failures in a row; one more failure benches it
BENCH_HOURS = 6.0              # how long a benched item waits after its last failure
BREAKER_FAILURES = 5           # failed sittings in a row, any items, that pause the pool
PAUSE_MINUTES = 10.0           # how long a pause lasts
PASS_EVERY_HOURS = 2.0         # PASS_RULES.md trigger 3: two hours without a pass
PROGRESS_WAIT_HOURS = 6.0      # an ok pass or merge whose pull request has not merged is offered again after this
REFEREE_EVERY_MINUTES = 120.0  # the longest the referee duty waits without queue trouble
REFEREE_TROUBLE_MINUTES = 30.0 # the shortest gap between referee sittings called by queue trouble
LIVE_GRACE_MINUTES = 10.0      # past its cap, a sitting with no end row counts as gone after this
LEDGER_LOOKBACK_HOURS = 24.0   # how far back the ledger is read for sittings and the pool's history
LOCK_WAIT_SECONDS = 2.0        # how long a starting pool waits for pool.lock
FETCH_MINUTES = 5.0            # how often --refine-ref is fetched
DEFAULT_REF = "origin/main"
WORKTREE = "worktree"
DEFAULT_RANK = 1               # the priority of a pass or duty with no task on the board: P1
ERROR_CHARS = 200
STATE_FILE = "pool.json"
LOCK_FILE = "pool.lock"

NOT_STARTED = "Not started"
IN_PROGRESS = "In progress"
DONE = "Done"
KIND_ORDER: dict[str, int] = {"merge": 0, "referee": 1, "pass": 2, "task": 3}

# how the pool judges a sitting that ended (History)
RESULT_OK = "ok"
RESULT_FAILED = "failed"
RESULT_NEUTRAL = "neutral"     # stopped or interrupted: no fault of the item
RESULT_REFUSED = "refused"     # python -m lab sit started nothing

# the pool's own ledger verbs
V_BEGIN, V_START, V_END, V_BENCH = "pool begin", "pool start", "pool end", "pool bench"
V_PAUSE, V_HOLD, V_STOP, V_DRAIN, V_EXIT = "pool pause", "pool hold", "pool stop", "pool drain", "pool exit"

_PASS_TITLE = re.compile(
    r"^(?:first refinement pass|refinement pass [0-9]+), (?P<track>telemetry|learning|plumbing|kernels) track\b",
    re.IGNORECASE)
_LOOP_TITLE = re.compile(r"^(?P<track>telemetry|learning|plumbing|kernels) loop, passes\b", re.IGNORECASE)
_MERGE_TITLE = re.compile(r"^(?:first merge pass|merge pass [0-9]+)\b", re.IGNORECASE)
_PRIORITY = re.compile(r"P(?P<rank>[0-9])", re.IGNORECASE)
_ID = re.compile(r"^(?P<letters>[A-Za-z]+)(?P<number>[0-9]+)(?P<rest>.*)$")

Launcher = Callable[..., Mapping[str, Any]]


# --- small helpers -------------------------------------------------------------------------------


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def iso(moment: dt.datetime | None) -> str | None:
    return digest.iso(moment) if moment is not None else None


def parse_iso(value: Any) -> dt.datetime | None:
    moment, _ = digest.parse_time(value)
    return moment


def minutes(delta: dt.timedelta) -> float:
    return round(delta.total_seconds() / 60.0, 1)


def clip(text: Any, limit: int = ERROR_CHARS) -> str:
    return digest.clip(text, limit)


def priority_rank(priority: str) -> int:
    """0 for P0, 1 for P1, 2 for P2; 9 for a priority with no P number."""
    found = _PRIORITY.search(priority or "")
    return int(found["rank"]) if found else 9


def priority_label(rank: int) -> str:
    return f"P{rank}" if rank < 9 else "P?"


def id_key(task_id: str) -> tuple[str, int, str]:
    found = _ID.match(task_id or "")
    if not found:
        return ((task_id or "").upper(), -1, "")
    return found["letters"].upper(), int(found["number"]), found["rest"]


def item_args(item: sit.Item) -> list[str]:
    """The sit arguments that name an item."""
    if item.kind == "task":
        return ["--task", str(item.name)]
    if item.kind == "pass":
        return ["--pass", str(item.name)]
    return [f"--{item.kind}"]


def command_text(tier: str, item: sit.Item) -> str:
    return " ".join(["python -m lab sit --tier", tier, *item_args(item)])


def item_from_label(label: str) -> sit.Item:
    """The inverse of sit.Item.label: "task <ID>" -> Item("task", "<ID>"), "merge" -> Item("merge")."""
    kind, _, name = str(label).strip().partition(" ")
    return sit.Item(kind, name or None)


# --- the board ----------------------------------------------------------------------------------


def pass_track(task: digest.Task) -> str | None:
    """The track a board task is a refinement pass or track loop of, else None."""
    found = _PASS_TITLE.match(task.title) or _LOOP_TITLE.match(task.title)
    return found["track"].lower() if found else None


def is_merge_task(task: digest.Task) -> bool:
    return _MERGE_TITLE.match(task.title) is not None


def ilan_alone(task: digest.Task) -> bool:
    return bool(task.assignees) and all(digest.is_ilan(a) for a in task.assignees)


class BoardIndex:
    """Lookups on one snapshot: tasks by ID and by page, and the state of a task's blockers."""

    def __init__(self, board: digest.Board) -> None:
        self.board = board
        self.by_id = {task.id.lower(): task for task in board.tasks}
        self.by_page = {task.page: task for task in board.tasks if task.page}

    def blocker(self, ref: str) -> digest.Task | None:
        key = digest.page_key(ref)
        return self.by_page.get(key) if key else self.by_id.get(ref.strip().lower())

    def blockers(self, task: digest.Task) -> tuple[list[str], list[str]]:
        """(the blockers that are not Done, by ID; the references missing from the snapshot)."""
        waiting: list[str] = []
        unknown: list[str] = []
        for ref in task.blocked_by:
            found = self.blocker(ref)
            if found is None:
                unknown.append(ref)
            elif found.status != DONE:
                waiting.append(found.id)
        return waiting, unknown


def queue_trouble(board: digest.Board, now: dt.datetime) -> list[str]:
    """What on the GPU queue needs the referee: running slots past their cap, complaints on open
    rows, and queued rows that waited twice their minutes without a complaint."""
    found: list[str] = []
    for row in board.queue:
        if row.status not in digest.OPEN_QUEUE_STATUSES:
            continue
        name = row.task or clip(row.request, 40) or "a row"
        if row.complaint:
            found.append(f"a complaint on {name}")
        if row.status == "Running" and row.started is not None and row.minutes is not None:
            cap = row.minutes + (digest.EXTENSION_MINUTES if row.extended else 0)
            elapsed = digest.minutes_between(row.started, now)
            if elapsed > cap:
                found.append(f"{name} running {elapsed} min against a cap of {cap:g}")
        if (row.status == "Queued" and row.requested is not None and row.minutes is not None
                and not row.complaint and digest.minutes_between(row.requested, now) >= 2 * row.minutes):
            found.append(f"{name} queued {digest.minutes_between(row.requested, now)} min, a complaint is due")
    return found


# --- the refinement files ------------------------------------------------------------------------


class RefineFiles(Protocol):
    """Where the refinement files are read from: refresh() before a look, then read() each file
    by its path under lab/refine (None when it does not exist)."""

    where: str

    def refresh(self) -> list[str]: ...

    def read(self, rel: str) -> str | None: ...


class TreeFiles:
    """The refinement files in a folder: the working tree's lab/refine by default."""

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        self.root = Path(root) if root is not None else REPO_ROOT / REFINE_DIR
        self.where = str(self.root)

    def refresh(self) -> list[str]:
        return []

    def read(self, rel: str) -> str | None:
        try:
            return (self.root / rel).read_text(encoding="utf-8")
        except FileNotFoundError:
            return None


class GitFiles:
    """The refinement files at a git ref (git show <commit>:lab/refine/<file>), the ref's
    remote fetched at most every fetch_minutes. A ref that cannot be resolved reads as no
    files, with the problem named."""

    def __init__(self, ref: str = DEFAULT_REF, *, repo: str | os.PathLike[str] = REPO_ROOT,
                 fetch_minutes: float | None = FETCH_MINUTES) -> None:
        self.ref = ref
        self.repo = Path(repo)
        self.fetch_minutes = fetch_minutes
        self.where = f"{ref} (git)"
        self.commit: str | None = None
        self._fetched: float | None = None
        self._cache: dict[str, str | None] = {}

    def _git(self, *args: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(["git", "-C", str(self.repo), *args], stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120, check=False)

    def refresh(self) -> list[str]:
        problems: list[str] = []
        remote = self.ref.split("/", 1)[0] if "/" in self.ref else None
        clock = time.monotonic()
        if (remote and self.fetch_minutes is not None
                and (self._fetched is None or clock - self._fetched >= self.fetch_minutes * 60)):
            self._fetched = clock
            try:
                done = self._git("fetch", "--quiet", remote)
                if done.returncode != 0:
                    problems.append(f"git fetch {remote} failed: {clip(done.stderr.decode('utf-8', 'replace'))}")
            except (OSError, subprocess.SubprocessError) as exc:
                problems.append(f"git fetch {remote} failed: {type(exc).__name__}: {exc}")
        try:
            found = self._git("rev-parse", "--verify", "--quiet", f"{self.ref}^{{commit}}")
        except (OSError, subprocess.SubprocessError) as exc:
            found = None
            problems.append(f"git cannot resolve {self.ref}: {type(exc).__name__}: {exc}")
        commit = found.stdout.decode("ascii", "replace").strip() if found is not None and found.returncode == 0 else None
        if commit is None and found is not None:
            problems.append(f"git cannot resolve {self.ref}; no pass or merge is offered")
        if commit != self.commit:
            self._cache.clear()
        self.commit = commit
        return problems

    def read(self, rel: str) -> str | None:
        if self.commit is None:
            return None
        if rel not in self._cache:
            try:
                shown = self._git("show", f"{self.commit}:{REFINE_DIR}/{rel}")
            except (OSError, subprocess.SubprocessError):
                return None
            self._cache[rel] = shown.stdout.decode("utf-8", "replace") if shown.returncode == 0 else None
        return self._cache[rel]


def refine_files(ref: str, repo: Path = REPO_ROOT) -> RefineFiles:
    """--refine-ref as a reader: "worktree" for the working tree, else a git ref."""
    if ref == WORKTREE:
        return TreeFiles(repo / REFINE_DIR)
    return GitFiles(ref, repo=repo)


@dataclass
class TrackState:
    """What the pool reads of one track: the counter, whether a run is open or capped, when the
    latest pass ended, and how many of its passes wait for the merge."""

    track: str
    counter: int | None = None
    run_open: bool = False
    capped: bool = False
    ended_at: dt.datetime | None = None
    unreconciled: int = 0
    problem: str | None = None
    refusal: str | None = None


@dataclass
class RefineState:
    where: str
    tracks: dict[str, TrackState]
    merges: int | None
    problems: list[str] = field(default_factory=list)

    def waiting_for_merge(self) -> list[str]:
        """The tracks with a pass the merge log has not reconciled."""
        return sorted(t.track for t in self.tracks.values() if t.unreconciled > 0)


def _cap_pass(track_log: rf.Log) -> int | None:
    """The number of the pass that capped the track's latest run, None when it is not capped."""
    run = track_log.current_run
    if run:
        last = run[-1]
        return (last.ending_number or last.number) if last.ending == "cap" else None
    states = [entry for entry in track_log.entries if isinstance(entry, rf.RunState)]
    if states and states[-1].ended == "cap":
        return states[-1].last_pass if states[-1].last_pass is not None else -1
    return None


def pass_refusal(track: str, log: rf.Log, counter: int) -> str | None:
    """Why a sitting for the track's next pass (counter + 1) must not start, None when it may: the log
    already records that pass as finished, it would skip a number, or the track's files break the
    numbering rules of the combined track check (refine_merge.check_track)."""
    numbers = [entry.number for entry in log.passes]
    following = counter + 1
    if following in numbers:
        return (f"pass {following} is already finished in REFINE_LOG.md, but refine_pass.txt reads {counter}; "
                "the counter is written last, so a pass left it behind")
    expected = (numbers[-1] if numbers else 0) + 1
    if following != expected:
        return (f"pass {following} would skip pass {expected}: refine_pass.txt reads {counter}, "
                f"the last pass in REFINE_LOG.md is {expected - 1}")
    # the numbering rules of the combined track check; its entry-content and report rules are the suite's
    # business, not a reason to hold a pass
    problems = [p for p in refine_merge.check_track(track, log, counter, {})
                if p.startswith((f"{track}: pass entries are numbered", f"{track}: refine_pass.txt reads"))]
    return f"the track's files break the track rules: {problems[0]}" if problems else None


def read_refine(files: RefineFiles) -> RefineState:
    """The refinement loop's state from its files. A track whose log or counter cannot be read
    has a problem and is never offered a pass; an unreadable merge log offers no merge."""
    problems = list(files.refresh())
    tracks: dict[str, TrackState] = {}
    logs: dict[str, rf.Log] = {}
    for track in refine_merge.TRACKS:
        state = TrackState(track)
        tracks[track] = state
        try:
            log_text = files.read(f"{track}/REFINE_LOG.md")
            counter_text = files.read(f"{track}/refine_pass.txt")
            if log_text is None or counter_text is None:
                raise rf.FormatError("REFINE_LOG.md or refine_pass.txt is missing")
            track_log = rf.parse_log(log_text)
            state.counter = rf.parse_counter(counter_text)
            rulings_text = files.read(f"{track}/OWNER_RULINGS.md")
            rulings = rf.parse_rulings(rulings_text) if rulings_text else ()
        except rf.FormatError as exc:
            state.problem = f"{track}: {exc}"
            problems.append(state.problem)
            continue
        logs[track] = track_log
        state.refusal = pass_refusal(track, track_log, state.counter)
        run = track_log.current_run
        state.run_open = bool(run) and not run[-1].ending
        cap = _cap_pass(track_log)
        state.capped = cap is not None and not any(
            ruling.answered and ruling.origin == f"pass {cap}" for ruling in rulings)
        if state.counter:
            report = files.read(f"{track}/pass_reports/pass_{state.counter:02d}.md")
            try:
                if report is None:
                    raise rf.FormatError(f"no report for pass {state.counter}")
                state.ended_at = rf.parse_report_end(report).astimezone(dt.timezone.utc)
            except rf.FormatError as exc:
                problems.append(f"{track}: {exc}; the two-hour trigger is off")
    merges: int | None = None
    try:
        merge_text = files.read("MERGE_LOG.md")
        merge_log = rf.parse_log(merge_text) if merge_text is not None else rf.Log("", ())
    except rf.FormatError as exc:
        problems.append(f"MERGE_LOG.md: {exc}; no merge is offered")
        merge_log = None
    if merge_log is not None:
        merges = refine_merge.next_merge_number(merge_log) - 1
        for track, entries in refine_merge.round_passes(logs, merge_log).items():
            tracks[track].unreconciled = len(entries)
    return RefineState(files.where, tracks, merges, problems)


# --- the ledger: sittings that hold units, and the pool's own history -----------------------------


@dataclass(frozen=True)
class SittingInfo:
    """A sitting the ledger shows as running: a start row without an end row, inside its cap."""

    sitting: str
    tier: str
    item: str
    started: dt.datetime
    units: int

    def summary(self) -> dict[str, Any]:
        return {"sitting": self.sitting, "tier": self.tier, "item": self.item, "units": self.units,
                "started": iso(self.started)}


def live_sittings(rows: Iterable[Mapping[str, Any]], now: dt.datetime) -> list[SittingInfo]:
    """The sittings whose start row has no end row and whose cap, plus LIVE_GRACE_MINUTES, has
    not passed. rows are the ledger's sit rows."""
    starts: dict[str, Mapping[str, Any]] = {}
    ended: set[str] = set()
    for row in rows:
        if row.get("verb") != sit.VERB:
            continue
        run_id = (row.get("refs") or {}).get("run_id")
        if not isinstance(run_id, str):
            continue
        if row.get("outcome") == sit.STARTED:
            starts[run_id] = row
        else:
            ended.add(run_id)
    live: list[SittingInfo] = []
    for run_id, row in starts.items():
        if run_id in ended:
            continue
        started = parse_iso(row.get("ts"))
        tier = str(row.get("tier") or "")
        if started is None or tier not in COST:
            continue
        cap = row.get("cap_minutes")
        cap_minutes = float(cap) if isinstance(cap, (int, float)) and not isinstance(cap, bool) else sit.DEFAULT_MINUTES
        if started + dt.timedelta(minutes=cap_minutes + LIVE_GRACE_MINUTES) <= now:
            continue
        live.append(SittingInfo(run_id, tier, str(row.get("item") or ""), started, COST[tier]))
    live.sort(key=lambda s: (s.started, s.sitting))
    return live


@dataclass
class Seat:
    """One sitting the pool runs: what it is for and since when."""

    seat: str
    tier: str
    item: sit.Item
    started: dt.datetime
    source: str = "task"
    why: str = ""
    progress: dict[str, Any] = field(default_factory=dict)

    @property
    def units(self) -> int:
        return COST[self.tier]

    @property
    def label(self) -> str:
        return self.item.label

    def summary(self, now: dt.datetime | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {"seat": self.seat, "tier": self.tier, "item": self.label, "units": self.units,
                               "source": self.source, "started": iso(self.started), "why": self.why,
                               "command": command_text(self.tier, self.item)}
        if now is not None:
            out["minutes"] = minutes(now - self.started)
        return out

    @classmethod
    def from_summary(cls, data: Mapping[str, Any]) -> Seat | None:
        started = parse_iso(data.get("started"))
        tier = str(data.get("tier") or "")
        if started is None or tier not in COST or not data.get("item"):
            return None
        return cls(str(data.get("seat") or "?"), tier, item_from_label(str(data["item"])), started,
                   str(data.get("source") or ""), str(data.get("why") or ""))


def foreign_sittings(live: Sequence[SittingInfo], seats: Iterable[Seat]) -> list[SittingInfo]:
    """The live sittings that are not the pool's own seats. A seat owns the earliest live
    sitting of its tier and item that started no earlier than a minute before the seat."""
    left = list(live)
    for seat in sorted(seats, key=lambda s: s.started):
        for index, info in enumerate(left):
            if (info.tier == seat.tier and info.item == seat.label
                    and info.started >= seat.started - dt.timedelta(minutes=1)):
                del left[index]
                break
    return left


@dataclass
class Record:
    """What the pool remembers of one item."""

    label: str
    tier: str | None = None
    last_start: dt.datetime | None = None
    last_end: dt.datetime | None = None
    last_result: str | None = None
    ok_progress: dict[str, Any] | None = None
    failures: int = 0
    last_failure: dt.datetime | None = None

    def bench_until(self, now: dt.datetime) -> dt.datetime | None:
        if self.failures <= MAX_RESTARTS or self.last_failure is None:
            return None
        until = self.last_failure + dt.timedelta(hours=BENCH_HOURS)
        return until if until > now else None

    @property
    def restarting(self) -> bool:
        """The last sitting failed or was cut off, so the next one continues its work."""
        return self.last_result in (RESULT_FAILED, RESULT_NEUTRAL)


class History:
    """Every item's record, and the failures in a row across items."""

    def __init__(self) -> None:
        self.items: dict[str, Record] = {}
        self.streak = 0

    def get(self, label: str) -> Record | None:
        return self.items.get(label)

    def _record(self, label: str) -> Record:
        return self.items.setdefault(label, Record(label))

    def started(self, label: str, tier: str, when: dt.datetime) -> None:
        record = self._record(label)
        record.tier = tier
        record.last_start = when

    def ended(self, label: str, tier: str, result: str, when: dt.datetime, progress: Mapping[str, Any],
              started: dt.datetime | None = None) -> Record:
        record = self._record(label)
        record.tier = tier
        if started is not None:
            record.last_start = started
        record.last_end = when
        record.last_result = result
        if result == RESULT_OK:
            record.failures = 0
            record.ok_progress = dict(progress)
            self.streak = 0
        elif result == RESULT_FAILED:
            record.failures += 1
            record.last_failure = when
            self.streak += 1
        return record

    @classmethod
    def from_rows(cls, rows: Iterable[Mapping[str, Any]]) -> History:
        """Replay the pool's end rows, oldest first."""
        history = cls()
        for row in rows:
            if row.get("verb") != V_END:
                continue
            when = parse_iso(row.get("ts"))
            label, tier, result = row.get("item"), row.get("tier"), row.get("result")
            if when is None or not isinstance(label, str) or result not in (
                    RESULT_OK, RESULT_FAILED, RESULT_NEUTRAL, RESULT_REFUSED):
                continue
            progress = row.get("progress") if isinstance(row.get("progress"), Mapping) else {}
            history.ended(label, str(tier or ""), str(result), when, progress, parse_iso(row.get("started_at")))
        return history


def classify(answer: Mapping[str, Any]) -> str:
    """How the pool judges a sit answer: ok, failed, neutral (stopped or interrupted) or refused
    (nothing ran: sit wrote no start row). A refused item itself (usage) counts as failed."""
    outcome = str(answer.get("outcome") or "")
    started = answer.get("ledger")
    ran = isinstance(started, Mapping) and started.get("start") is not None
    if not ran:
        if outcome in (sit.STOPPED, sit.INTERRUPTED):
            return RESULT_NEUTRAL
        return RESULT_FAILED if outcome == sit.OUTCOME_USAGE else RESULT_REFUSED
    if outcome == sit.OK and answer.get("ok") is True:
        return RESULT_OK
    if outcome in (sit.STOPPED, sit.INTERRUPTED):
        return RESULT_NEUTRAL
    return RESULT_FAILED


# --- candidates and the order ------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One item the pool could start now, with what puts it in order."""

    tier: str
    item: sit.Item
    source: str
    rank: int = DEFAULT_RANK
    wave: float | None = None
    stretch: bool = False
    why: str = ""
    progress: Mapping[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.item.label

    @property
    def units(self) -> int:
        return COST[self.tier]

    def key(self) -> tuple:
        return (self.rank, self.wave if self.wave is not None else math.inf, self.stretch,
                KIND_ORDER.get(self.item.kind, 9), id_key(self.item.name or ""))

    def summary(self) -> dict[str, Any]:
        return {"tier": self.tier, "item": self.label, "units": self.units, "source": self.source,
                "priority": priority_label(self.rank), "wave": self.wave, "why": self.why,
                "command": command_text(self.tier, self.item)}


@dataclass(frozen=True)
class Skip:
    """An item the pool does not offer now, and why."""

    item: str
    reason: str
    tier: str | None = None

    def summary(self) -> dict[str, Any]:
        return {"item": self.item, "tier": self.tier, "reason": self.reason}


@dataclass
class Plan:
    """What a fill does: the candidates that start, the one held for units, and the rest."""

    starts: list[Candidate]
    held: Candidate | None
    after: list[Candidate]
    never_fits: list[Candidate]


def plan(candidates: Sequence[Candidate], free: int, budget: int) -> Plan:
    """Walk the candidates in order, starting each that fits in the free units, and stop at the
    first that does not: its units are held for it. One that costs more than the budget is set
    aside."""
    starts: list[Candidate] = []
    never: list[Candidate] = []
    ordered = sorted(candidates, key=Candidate.key)
    for index, candidate in enumerate(ordered):
        if candidate.units > budget:
            never.append(candidate)
            continue
        if candidate.units <= free:
            starts.append(candidate)
            free -= candidate.units
            continue
        rest = [c for c in ordered[index + 1:] if c.units <= budget]
        never += [c for c in ordered[index + 1:] if c.units > budget]
        return Plan(starts, candidate, rest, never)
    return Plan(starts, None, [], never)


def _gate(tasks: Sequence[digest.Task], index: BoardIndex, label: str, skips: list[Skip], tier: str,
          restarting: bool) -> tuple[bool, digest.Task | None]:
    """(may the pass or merge start, the board task it inherits its order from). The open tasks of
    the item: a Locked one means someone runs it (unless the pool's own last sitting of it failed);
    otherwise the first must be Ilan's alone with its blockers Done."""
    open_tasks = sorted((t for t in tasks if t.status != DONE),
                        key=lambda t: (not t.title.lower().startswith("first"), id_key(t.id)))
    locked = [t.id for t in open_tasks if t.locked]
    if locked and not restarting:
        skips.append(Skip(label, f"{', '.join(locked)} is Locked on the board: a session runs it", tier))
        return False, None
    if not open_tasks:
        return True, None
    gate = open_tasks[0]
    if not ilan_alone(gate):
        skips.append(Skip(label, f"{gate.id} is not Ilan's alone", tier))
        return False, None
    waiting, unknown = index.blockers(gate)
    if waiting:
        return False, None  # blocked, like any task whose blockers are not Done
    if unknown:
        skips.append(Skip(label, f"{gate.id} has a blocker missing from the board snapshot", tier))
        return False, None
    return True, gate


def _order_of(task: digest.Task | None) -> tuple[int, float | None]:
    if task is None:
        return DEFAULT_RANK, None
    return priority_rank(task.priority), task.wave


@dataclass
class Settings:
    """What a pool, a dry run and status share."""

    state_dir: Path
    budget: int = BUDGET_UNITS
    board_file: Path | None = None
    board_max_age: float = float(digest.BOARD_STALE_MINUTES)
    refine: RefineFiles | None = None
    verbs: Callable[[], Iterable[str]] | None = None


@dataclass
class View:
    """Everything one decision is made from."""

    now: dt.datetime
    budget: int
    board: dict[str, Any] | None
    refine: RefineState | None
    candidates: list[Candidate]
    skips: list[Skip]
    seats: list[Seat]
    foreign: list[SittingInfo]
    problems: list[str]
    blocked: str | None = None

    @property
    def in_use(self) -> int:
        return sum(seat.units for seat in self.seats) + sum(info.units for info in self.foreign)

    @property
    def free(self) -> int:
        return max(0, self.budget - self.in_use)


def registered_verbs() -> list[str]:
    from lab import __main__ as dispatcher  # imported here: the dispatcher imports verbs lazily
    return list(dispatcher.VERBS)


def build_view(settings: Settings, seats: Sequence[Seat], history: History, now: dt.datetime) -> View:
    """The candidates in order, what is not offered and why, and the units in use."""
    problems: list[str] = []
    skips: list[Skip] = []
    candidates: list[Candidate] = []

    try:
        rows = ledger.read(state_dir=settings.state_dir, verb=sit.VERB,
                           since=now - dt.timedelta(hours=LEDGER_LOOKBACK_HOURS))
        foreign = foreign_sittings(live_sittings(rows, now), seats)
    except OSError as exc:
        blocked = f"the ledger cannot be read ({type(exc).__name__}: {exc}); nothing starts"
        return View(now, settings.budget, None, None, [], skips, list(seats), [], [blocked], blocked)
    # a sitting of the same item runs, whoever started it: a seat, a hand sitting, an earlier pool
    running = {seat.label for seat in seats} | {info.item for info in foreign}

    board, board_problems = digest.load_board(settings.board_file, state_dir=settings.state_dir)
    problems += [f"board: {p}" for p in board_problems]
    board_info: dict[str, Any] | None = None
    fresh = False
    if board is not None and board.taken_at is not None:
        age = minutes(now - board.taken_at)
        fresh = age <= settings.board_max_age
        board_info = {"path": board.path, "taken_at": iso(board.taken_at), "age_min": age, "stale": not fresh,
                      "tasks": len(board.tasks)}
        if not fresh:
            skips.append(Skip("board", f"the snapshot is {age:g} min old, over {settings.board_max_age:g}: no task, "
                                       "pass or merge starts until a fresh one is saved (python -m lab digest "
                                       "--save-board <file>)"))
    elif board is None:
        skips.append(Skip("board", "no board snapshot: no task, pass or merge starts"))

    def admit(candidate: Candidate) -> None:
        if candidate.label in running:
            return
        record = history.get(candidate.label)
        if record is not None:
            until = record.bench_until(now)
            if until is not None:
                skips.append(Skip(candidate.label, f"benched after {record.failures} failed sittings in a row, "
                                                   f"until {iso(until)}", candidate.tier))
                return
        problem = sit.item_problem(candidate.tier, candidate.item)
        if problem:
            skips.append(Skip(candidate.label, f"sit would refuse it: {problem}", candidate.tier))
            return
        candidates.append(candidate)

    refine: RefineState | None = None
    if board is not None and fresh:
        index = BoardIndex(board)
        for task in board.tasks:
            _task_candidate(task, index, board, history, admit, skips)
        if settings.refine is None:
            settings.refine = refine_files(DEFAULT_REF)
        refine = read_refine(settings.refine)
        problems += [f"refine: {p}" for p in refine.problems]
        _refine_candidates(refine, index, running, history, now, admit, skips)

    _referee_candidate(settings, board if fresh else None, running, history, now, admit, skips)
    return View(now, settings.budget, board_info, refine, sorted(candidates, key=Candidate.key), skips, list(seats),
                foreign, problems)


def _task_candidate(task: digest.Task, index: BoardIndex, board: digest.Board, history: History,
                    admit: Callable[[Candidate], None], skips: list[Skip]) -> None:
    if not ilan_alone(task) or pass_track(task) or is_merge_task(task):
        return
    label = sit.Item("task", task.id).label
    record = history.get(label)
    restarting = record is not None and record.restarting
    allowed_status = (NOT_STARTED, IN_PROGRESS) if restarting else (NOT_STARTED,)
    if task.status not in allowed_status or (task.locked and not restarting):
        return
    waiting, unknown = index.blockers(task)
    if waiting:
        return
    tier = BOARD_TIERS.get(task.tier)
    if unknown:
        skips.append(Skip(label, "a blocker is missing from the board snapshot", tier))
        return
    if tier is None:
        skips.append(Skip(label, "no Tier on the board (the referee sets it)", None))
        return
    if (record is not None and record.last_result == RESULT_OK and record.last_end is not None
            and board.taken_at is not None and board.taken_at <= record.last_end):
        skips.append(Skip(label, f"its last sitting ended ok at {iso(record.last_end)}; waiting for a board "
                                 "snapshot taken after that", tier))
        return
    wave = f"wave {digest.number_text(task.wave)}" if task.wave is not None else "no wave"
    again = f", restart after a {record.last_result} sitting" if restarting and record is not None else ""
    why = f"{task.id} {priority_label(priority_rank(task.priority))}, {wave}, {task.status}{again}: {clip(task.title, 60)}"
    admit(Candidate(tier, sit.Item("task", task.id), "task", priority_rank(task.priority), task.wave, task.stretch,
                    why, {"board": iso(board.taken_at)}))


def _waiting_for_progress(record: Record | None, key: str, value: int, now: dt.datetime) -> bool:
    """Whether the last sitting of the item ended ok and what it changes (the counter, the merge
    log) has not moved since it started, for less than PROGRESS_WAIT_HOURS."""
    if record is None or record.last_result != RESULT_OK or not record.ok_progress or record.last_end is None:
        return False
    before = record.ok_progress.get(key)
    if not isinstance(before, int) or value > before:
        return False
    return now - record.last_end < dt.timedelta(hours=PROGRESS_WAIT_HOURS)


def _refine_candidates(refine: RefineState, index: BoardIndex, running: set[str], history: History,
                       now: dt.datetime, admit: Callable[[Candidate], None], skips: list[Skip]) -> None:
    board = index.board
    pass_tasks: dict[str, list[digest.Task]] = {track: [] for track in refine.tracks}
    merge_tasks: list[digest.Task] = []
    for task in board.tasks:
        track = pass_track(task)
        if track in pass_tasks:
            pass_tasks[track].append(task)
        elif is_merge_task(task):
            merge_tasks.append(task)
    passes_running = {label.partition(" ")[2] for label in running if label.startswith("pass ")}
    passes_locked = {track for track, tasks in pass_tasks.items() if any(t.locked and t.status != DONE for t in tasks)}

    # the merge first: while it runs, waits for its pull request or is due, no pass starts
    merge_label = sit.Item("merge").label
    merge_record = history.get(merge_label)
    merge_restart = merge_record is not None and merge_record.restarting
    merge_locked = [t.id for t in merge_tasks if t.locked and t.status != DONE]
    merge_blocks_passes = False
    waiting_tracks = refine.waiting_for_merge()
    if merge_label in running:
        merge_blocks_passes = True
    elif merge_locked and not merge_restart:
        merge_blocks_passes = True
        skips.append(Skip(merge_label, f"{', '.join(merge_locked)} is Locked on the board: a session runs the merge, "
                                       "and no pass starts until it has finished", DUTY_TIER))
    elif refine.merges is None:
        merge_blocks_passes = True
    elif waiting_tracks and _waiting_for_progress(merge_record, "merges", refine.merges, now):
        merge_blocks_passes = True
        skips.append(Skip(merge_label, "its last sitting ended ok; waiting for its pull request to merge "
                                       "(the merge log has no new entry)", DUTY_TIER))
    elif waiting_tracks:
        running_now = sorted(passes_running | passes_locked)
        if running_now:
            skips.append(Skip(merge_label, f"waits for the passes still running: {', '.join(running_now)}", DUTY_TIER))
        else:
            ok, gate = _gate(merge_tasks, index, merge_label, skips, DUTY_TIER, merge_restart)
            if ok and gate is None and refine.merges == 0:
                ok = False
                skips.append(Skip(merge_label, "the first merge runs as its board task, and the snapshot has no open "
                                               "merge task", DUTY_TIER))
            if ok:
                merge_blocks_passes = True
                rank, wave = _order_of(gate)
                passes = ", ".join(f"{t} ({refine.tracks[t].unreconciled})" for t in waiting_tracks)
                why = f"the round's passes wait for the merge: {passes}" + (f"; board task {gate.id}" if gate else "")
                admit(Candidate(DUTY_TIER, sit.Item("merge"), "merge", rank, wave, False, why,
                                {"merges": refine.merges}))

    for track, state in sorted(refine.tracks.items()):
        label = sit.Item("pass", track).label
        if state.problem or state.counter is None:
            continue
        if track in passes_running:
            continue
        if state.refusal:
            skips.append(Skip(label, f"refused: {state.refusal}", PASS_TIER))
            continue
        if state.capped:
            skips.append(Skip(label, "the run reached its cap and the cap ruling has no Answer", PASS_TIER))
            continue
        if merge_blocks_passes:
            continue
        if state.unreconciled:
            continue  # its pass of this round waits for the merge
        record = history.get(label)
        if _waiting_for_progress(record, "counter", state.counter, now):
            skips.append(Skip(label, f"its last sitting ended ok; waiting for its pull request to merge "
                                     f"(the counter is still {state.counter})", PASS_TIER))
            continue
        if state.counter == 0:
            trigger = "the track's first pass"
        elif state.run_open:
            trigger = f"pass {state.counter + 1} continues the open run"
        elif state.ended_at is not None and now - state.ended_at >= dt.timedelta(hours=PASS_EVERY_HOURS):
            trigger = f"{minutes(now - state.ended_at):g} min since pass {state.counter} ended"
        else:
            continue
        restarting = record is not None and record.restarting
        ok, gate = _gate(pass_tasks[track], index, label, skips, PASS_TIER, restarting)
        if not ok:
            continue
        if gate is None and state.counter == 0:
            skips.append(Skip(label, "a track's first pass runs as its board task, and the snapshot has no open "
                                     "first-pass task for it", PASS_TIER))
            continue
        rank, wave = _order_of(gate)
        why = f"{trigger}" + (f"; board task {gate.id}" if gate else "")
        admit(Candidate(PASS_TIER, sit.Item("pass", track), "pass", rank, wave, False, why,
                        {"counter": state.counter}))


def _referee_candidate(settings: Settings, board: digest.Board | None, running: set[str], history: History,
                       now: dt.datetime, admit: Callable[[Candidate], None], skips: list[Skip]) -> None:
    label = sit.Item("referee").label
    try:
        verbs = set((settings.verbs or registered_verbs)())
    except Exception as exc:  # the registry is read for one name; a failure there costs only the referee
        skips.append(Skip(label, f"the verb registry cannot be read: {type(exc).__name__}: {exc}", DUTY_TIER))
        return
    if "referee" not in verbs:
        skips.append(Skip(label, "python -m lab referee is not registered yet", DUTY_TIER))
        return
    if label in running:
        return
    record = history.get(label)
    last = None if record is None else (record.last_end or record.last_start)
    newer = board is not None and board.taken_at is not None and (last is None or board.taken_at > last)
    spaced = last is None or now - last >= dt.timedelta(minutes=REFEREE_TROUBLE_MINUTES)
    trouble = queue_trouble(board, now) if board is not None and newer and spaced else []
    wave: float | None = None
    if trouble:
        more = f"; +{len(trouble) - 3} more" if len(trouble) > 3 else ""
        why, rank, wave = "GPU queue: " + "; ".join(trouble[:3]) + more, 0, 0.0
    elif record is not None and record.restarting:
        why, rank = f"restart after a {record.last_result} sitting", DEFAULT_RANK
    elif last is None:
        why, rank = "no referee sitting in the pool's records", DEFAULT_RANK
    elif now - last >= dt.timedelta(minutes=REFEREE_EVERY_MINUTES):
        why, rank = f"{minutes(now - last):g} min since the last referee sitting", DEFAULT_RANK
    else:
        return
    admit(Candidate(DUTY_TIER, sit.Item("referee"), "referee", rank, wave, False, why, {}))


# --- launching a sitting -------------------------------------------------------------------------


def parse_answer(stdout: bytes | str, returncode: int | None) -> dict[str, Any]:
    """sit's answer: the last line of its stdout that is a JSON object; an error answer when there
    is none."""
    text = stdout.decode("utf-8", "replace") if isinstance(stdout, bytes) else (stdout or "")
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            doc = json.loads(line)
        except ValueError:
            continue
        if isinstance(doc, dict):
            return doc
    return {"ok": False, "outcome": "error",
            "error": f"python -m lab sit exited {returncode} without an answer: {clip(text) or 'no output'}"}


class SitProcess:
    """The default launcher: python -m lab sit for one item, as a child process of the pool,
    from the repo root, with the pool's state dir; it waits for the child and returns its
    answer. head replaces the command before the sit arguments."""

    def __init__(self, state_dir: Path, *, head: Sequence[str] | None = None, cwd: Path = REPO_ROOT) -> None:
        self.state_dir = Path(state_dir).resolve()
        self.head = list(head) if head else [sys.executable, "-m", "lab", "sit"]
        self.cwd = cwd

    def argv(self, tier: str, item: sit.Item, *, minutes: float | None = None,
             budget_usd: float | None = None) -> list[str]:
        args = [*self.head, "--tier", tier, *item_args(item)]
        if minutes is not None:
            args += ["--minutes", f"{minutes:g}"]
        if budget_usd is not None:
            args += ["--budget-usd", f"{budget_usd:g}"]
        return args

    def __call__(self, tier: str, item: sit.Item, *, minutes: float | None = None,
                 budget_usd: float | None = None) -> dict[str, Any]:
        env = dict(os.environ)
        env[ledger.ENV_STATE_DIR] = str(self.state_dir)
        try:
            done = subprocess.run(self.argv(tier, item, minutes=minutes, budget_usd=budget_usd), cwd=str(self.cwd),
                                  env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, check=False)
        except OSError as exc:
            return {"ok": False, "outcome": "error", "error": f"could not run python -m lab sit: {exc}"}
        return parse_answer(done.stdout, done.returncode)


# --- the pool ------------------------------------------------------------------------------------


def state_path(state_dir: Path) -> Path:
    return Path(state_dir) / STATE_FILE


def lock_path(state_dir: Path) -> Path:
    return Path(state_dir) / LOCK_FILE


def write_state(path: Path, data: Mapping[str, Any]) -> None:
    """pool.json through a temporary file and one replace; a failure is logged, never raised."""
    temp: str | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temp = tempfile.mkstemp(prefix=".pool-", suffix=".json", dir=path.parent)
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(data, fh, indent=1)
        os.replace(temp, path)
    except OSError as exc:
        log.error("lab pool: could not write %s: %s", path, exc)
        if temp is not None:
            with contextlib.suppress(OSError):
                Path(temp).unlink()


def read_state(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def pool_running(state_dir: Path) -> tuple[bool, str]:
    """(whether a pool holds pool.lock, the holder's note). A pool always has the file, so a
    missing one answers no pool and nothing is created."""
    if not lock_path(state_dir).exists():
        return False, ""
    probe = FileLock(lock_path(state_dir), timeout=0)
    try:
        if probe.acquire(0):
            probe.release()
            return False, ""
    except OSError as exc:
        return False, f"pool.lock cannot be opened: {exc}"
    return True, probe.read_note()


def restore_history(state_dir: Path, now: dt.datetime) -> History:
    rows = ledger.read(state_dir=state_dir, actor=ACTOR, verb=V_END,
                       since=now - dt.timedelta(hours=LEDGER_LOOKBACK_HOURS))
    return History.from_rows(rows)


class Pool:
    """The pool's state and its steps: begin(), tick() until done, finish(); run() does all."""

    def __init__(self, settings: Settings, *, launcher: Launcher | None = None, minutes: float | None = None,
                 budget_usd: float | None = None, tick_seconds: float = TICK_SECONDS, exit_when_idle: bool = False,
                 clock: Callable[[], dt.datetime] = utc_now) -> None:
        self.settings = settings
        self.state_dir = Path(settings.state_dir)
        self.launcher: Launcher = launcher or SitProcess(self.state_dir)
        self.minutes = minutes
        self.budget_usd = budget_usd
        self.tick_seconds = tick_seconds
        self.exit_when_idle = exit_when_idle
        self.clock = clock
        self.history = History()
        self.seats: dict[str, Seat] = {}
        self.ended: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()
        self.lock = FileLock(lock_path(self.state_dir), timeout=LOCK_WAIT_SECONDS)
        self.began: dt.datetime | None = None
        self.paused_until: dt.datetime | None = None
        self.pause_reason = ""
        self.stopping = False
        self.draining = False
        self.idle = False
        self.held: Candidate | None = None
        self.view: View | None = None
        self.counts = {"started": 0, "ended": 0, "failed": 0, "benched": 0}
        self.error: str | None = None
        self._next_seat = 1

    # rows ------------------------------------------------------------------------------------

    def _row(self, verb: str, outcome: str, *, tier: str | None = None, task: str | None = None,
             metrics: Mapping[str, float] | None = None, refs: Mapping[str, Any] | None = None,
             required: bool = False, **more: Any) -> int | None:
        row = {"ts": self.clock(), "actor": ACTOR, "tier": tier, "task": task, "verb": verb, "outcome": outcome,
               "metrics": dict(metrics or {}), "refs": dict(refs or {}), **more}
        try:
            return ledger.append(row, state_dir=self.state_dir)
        except (OSError, LockTimeout, ledger.RowError) as exc:
            log.error("lab pool: the %s row could not be written: %s", verb, exc)
            if required:
                raise
            return None

    def _units_metrics(self, view: View | None = None) -> dict[str, float]:
        in_use = sum(s.units for s in self.seats.values()) + (sum(f.units for f in view.foreign) if view else 0)
        return {"pool.units_in_use": in_use, "pool.units_free": max(0, self.settings.budget - in_use)}

    # the steps -------------------------------------------------------------------------------

    def begin(self) -> dict[str, Any] | None:
        """Take pool.lock, restore the history, write the begin row; an answer when it refuses."""
        if stop.stopped():
            return {"ok": False, "outcome": sit.STOPPED, "reason": stop.stop_reason() or stop.NO_REASON,
                    "file": str(stop.stop_file())}
        try:
            got = self.lock.acquire()
        except OSError as exc:
            return {"ok": False, "outcome": "error", "error": f"pool.lock cannot be opened: {exc}"}
        if not got:
            return {"ok": False, "outcome": "already_running", "holder": self.lock.read_note(),
                    "error": f"another pool holds {lock_path(self.state_dir)}"}
        now = self.clock()
        self.began = now
        try:
            self.lock.write_note(f"pid {os.getpid()}, pool since {iso(now)}")
            self.history = restore_history(self.state_dir, now)
            self._row(V_BEGIN, "started", metrics={"pool.budget": self.settings.budget}, required=True,
                      pid=os.getpid(), budget=self.settings.budget, minutes=self.minutes, budget_usd=self.budget_usd)
        except (OSError, LockTimeout, ledger.RowError) as exc:
            self.lock.release()
            return {"ok": False, "outcome": sit.OUTCOME_LEDGER,
                    "error": f"the pool could not start its record: {type(exc).__name__}: {exc}"}
        log.info("lab pool: started with a budget of %d units", self.settings.budget)
        self._save(now)
        return None

    def _check_stop(self, now: dt.datetime) -> None:
        if not self.stopping and stop.stopped():
            self.stopping = True
            log.warning("lab pool: STOP is set; nothing new starts, %d seats run to their end", len(self.seats))
            self._row(V_STOP, sit.STOPPED, metrics=self._units_metrics(),
                      reason=stop.stop_reason() or stop.NO_REASON, seats=len(self.seats))

    def _pause(self, now: dt.datetime, reason: str) -> None:
        if self.paused_until is not None and self.paused_until > now:
            log.warning("lab pool: already paused until %s; also: %s", iso(self.paused_until), reason)
            return
        self.paused_until = now + dt.timedelta(minutes=PAUSE_MINUTES)
        self.pause_reason = reason
        log.warning("lab pool: paused until %s: %s", iso(self.paused_until), reason)
        self._row(V_PAUSE, "paused", until=iso(self.paused_until), reason=clip(reason))

    def fill(self, now: dt.datetime) -> Plan | None:
        """Start what fits, in order; hold the free units for the first that does not. pool.json
        is rewritten after it."""
        chosen = self._fill(now)
        self._save(now)
        return chosen

    def _fill(self, now: dt.datetime) -> Plan | None:
        self._check_stop(now)
        if self.stopping or self.draining:
            self.idle = False
            return None
        if self.paused_until is not None and now < self.paused_until:
            self.idle = not self.seats
            return None
        self.paused_until = None
        view = build_view(self.settings, list(self.seats.values()), self.history, now)
        self.view = view
        if view.blocked:
            self._pause(now, view.blocked)
            return None
        chosen = plan(view.candidates, view.free, self.settings.budget)
        for candidate in chosen.starts:
            if stop.stopped():
                self._check_stop(now)
                break
            if not self._start(candidate, view, now):
                break
        if chosen.held is not None and (self.held is None or self.held.label != chosen.held.label):
            self._row(V_HOLD, "held", tier=chosen.held.tier, task=chosen.held.item.task, metrics=self._units_metrics(view),
                      item=chosen.held.label, units_held=chosen.held.units, why=chosen.held.why)
        self.held = chosen.held
        self.idle = not self.seats and not chosen.starts and chosen.held is None
        return chosen

    def _start(self, candidate: Candidate, view: View, now: dt.datetime) -> bool:
        seat = Seat(f"seat-{self._next_seat}", candidate.tier, candidate.item, now, candidate.source, candidate.why,
                    dict(candidate.progress))
        in_use = sum(s.units for s in self.seats.values()) + sum(f.units for f in view.foreign) + seat.units
        try:
            self._row(V_START, "started", tier=seat.tier, task=seat.item.task,
                      metrics={"pool.units_in_use": in_use, "pool.units_free": max(0, self.settings.budget - in_use)},
                      required=True, item=seat.label, seat=seat.seat, units_held=seat.units, source=seat.source,
                      why=seat.why, progress=seat.progress)
        except (OSError, LockTimeout, ledger.RowError) as exc:
            self._pause(now, f"the start row of {seat.label} could not be written, so it did not start: "
                             f"{type(exc).__name__}: {exc}")
            return False
        self._next_seat += 1
        self.seats[seat.seat] = seat
        self.history.started(seat.label, seat.tier, now)
        self.counts["started"] += 1
        log.info("lab pool: %s starts %s %s (%d units, %d of %d in use)", seat.seat, seat.tier, seat.label,
                 seat.units, in_use, self.settings.budget)
        threading.Thread(target=self._sit, args=(seat,), name=f"lab-pool-{seat.seat}", daemon=True).start()
        return True

    def _sit(self, seat: Seat) -> None:
        """A seat's thread: run the sitting and hand its answer to the main loop, whatever
        happens, so the seat always ends and its units come back."""
        answer: dict[str, Any] = {"ok": False, "outcome": "error", "error": "the launcher did not answer"}
        try:
            got = self.launcher(seat.tier, seat.item, minutes=self.minutes, budget_usd=self.budget_usd)
            answer = dict(got) if isinstance(got, Mapping) else {
                "ok": False, "outcome": "error", "error": f"the launcher answered {type(got).__name__}"}
        except BaseException as exc:  # a seat thread reports back whatever ended it
            answer = {"ok": False, "outcome": "error", "error": f"the launcher failed: {type(exc).__name__}: {exc}"}
        finally:
            self.ended.put((seat.seat, answer))

    def handle(self, seat_id: str, answer: Mapping[str, Any], now: dt.datetime) -> None:
        """Record a seat whose sitting ended: its end row, its record, a bench or a pause."""
        seat = self.seats.pop(seat_id, None)
        if seat is None:
            return
        result = classify(answer)
        record = self.history.ended(seat.label, seat.tier, result, now, seat.progress, seat.started)
        self.counts["ended"] += 1
        outcome = str(answer.get("outcome") or "unknown")
        sitting_id = answer.get("sitting") if isinstance(answer.get("sitting"), str) else None
        more: dict[str, Any] = {}
        if answer.get("error"):
            more["error"] = clip(answer["error"])
        self._row(V_END, "ended", tier=seat.tier, task=seat.item.task,
                  metrics={**self._units_metrics(self.view), "pool.seat_s": round((now - seat.started).total_seconds(), 1)},
                  refs={"run_id": sitting_id}, item=seat.label, seat=seat.seat, units_held=seat.units,
                  sitting_outcome=outcome, result=result, failures=record.failures, progress=seat.progress,
                  started_at=iso(seat.started), **more)
        log.info("lab pool: %s ended %s %s: %s (%s)", seat.seat, seat.tier, seat.label, outcome, result)
        if result == RESULT_FAILED:
            self.counts["failed"] += 1
            until = record.bench_until(now)
            if until is not None:
                self.counts["benched"] += 1
                self._row(V_BENCH, "benched", tier=seat.tier, task=seat.item.task, item=seat.label,
                          failures=record.failures, until=iso(until))
                log.warning("lab pool: %s is benched until %s after %d failed sittings in a row", seat.label,
                            iso(until), record.failures)
            if self.history.streak >= BREAKER_FAILURES:
                self.history.streak = 0
                self._pause(now, f"{BREAKER_FAILURES} sittings in a row failed")
        elif result == RESULT_REFUSED:
            self._pause(now, f"python -m lab sit started nothing for {seat.label}: {outcome}"
                             + (f": {answer.get('error')}" if answer.get("error") else ""))

    def tick(self, timeout: float) -> None:
        """One step: fill, wait up to timeout for a seat to end, record every seat that ended, and
        refill at once."""
        self.fill(self.clock())
        try:
            seat_id, answer = self.ended.get(timeout=max(0.0, timeout))
        except queue.Empty:
            return
        self.handle(seat_id, answer, self.clock())
        while True:
            try:
                seat_id, answer = self.ended.get_nowait()
            except queue.Empty:
                break
            self.handle(seat_id, answer, self.clock())
        self.fill(self.clock())

    def done(self) -> str | None:
        """Why the run is over, or None while it goes on."""
        if self.seats:
            return None
        if self.stopping:
            return sit.STOPPED
        if self.draining:
            return sit.INTERRUPTED
        if self.exit_when_idle and self.idle:
            return "idle"
        return None

    def run(self) -> dict[str, Any]:
        refused = self.begin()
        if refused is not None:
            return refused
        outcome = "error"
        try:
            while True:
                try:
                    self.tick(self.tick_seconds)
                except KeyboardInterrupt:
                    if self.draining:
                        outcome = sit.INTERRUPTED
                        break
                    self.draining = True
                    log.warning("lab pool: interrupted; nothing new starts, %d seats run to their end "
                                "(Ctrl+C again to leave now)", len(self.seats))
                    self._row(V_DRAIN, "draining", metrics=self._units_metrics(), seats=len(self.seats))
                    continue
                why = self.done()
                if why is not None:
                    outcome = why
                    break
        except Exception as exc:
            log.exception("lab pool: the pool failed")
            outcome = "error"
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            answer = self.finish(outcome)
        return answer

    def finish(self, outcome: str) -> dict[str, Any]:
        now = self.clock()
        left = [seat.summary(now) for seat in self.seats.values()]
        ran = minutes(now - self.began) if self.began else 0.0
        self._row(V_EXIT, outcome, metrics={"pool.minutes": ran, "pool.started": self.counts["started"]},
                  started=self.counts["started"], ended=self.counts["ended"], seats_left=[s["item"] for s in left])
        self._save(now, running=False)
        self.lock.release()
        answer: dict[str, Any] = {"ok": outcome in (sit.STOPPED, "idle"), "outcome": outcome, "minutes": ran,
                                  **self.counts, "seats_left": left}
        if self.error:
            answer["error"] = self.error
        return answer

    def _save(self, now: dt.datetime, *, running: bool = True) -> None:
        view = self.view
        foreign = [f.summary() for f in view.foreign] if view else []
        own = sum(s.units for s in self.seats.values())
        data = {
            "running": running, "pid": os.getpid(), "since": iso(self.began), "updated": iso(now),
            "budget": self.settings.budget, "units_in_use": own + sum(f["units"] for f in foreign),
            "seats": [seat.summary() for seat in self.seats.values()], "foreign": foreign,
            "held": self.held.summary() if self.held else None,
            "benched": [{"item": r.label, "tier": r.tier, "failures": r.failures, "until": iso(r.bench_until(now))}
                        for r in self.history.items.values() if r.bench_until(now) is not None],
            "paused_until": iso(self.paused_until) if self.paused_until and self.paused_until > now else None,
            "pause_reason": self.pause_reason if self.paused_until and self.paused_until > now else None,
            "stopping": self.stopping, "draining": self.draining,
            "board": view.board if view else None, "counts": dict(self.counts),
        }
        write_state(state_path(self.state_dir), data)


# --- dry run and status --------------------------------------------------------------------------


def running_seats(state_dir: Path) -> list[Seat]:
    """The seats of the pool that runs now, from pool.json; [] when no pool holds pool.lock."""
    running, _ = pool_running(state_dir)
    data = read_state(state_path(state_dir)) if running else None
    seats = [Seat.from_summary(s) for s in (data or {}).get("seats") or [] if isinstance(s, Mapping)]
    return [seat for seat in seats if seat is not None]


def dry_run(settings: Settings, now: dt.datetime | None = None) -> dict[str, Any]:
    """The schedule a pool would start now: the order with each item's decision (start, hold,
    after), what is not offered and why. Starts nothing and writes nothing."""
    now = now or utc_now()
    history = restore_history(settings.state_dir, now)
    seats = running_seats(settings.state_dir)
    view = build_view(settings, seats, history, now)
    chosen = plan(view.candidates, view.free, settings.budget)
    schedule: list[dict[str, Any]] = []
    for decision, entries in (("start", chosen.starts), ("hold", [chosen.held] if chosen.held else []),
                              ("after", chosen.after)):
        schedule += [{"decision": decision, **c.summary()} for c in entries]
    stopped = stop.stopped()
    skips = [s.summary() for s in view.skips] + [
        {"item": c.label, "tier": c.tier, "reason": f"costs {c.units} units, more than the budget of {settings.budget}"}
        for c in chosen.never_fits]
    return {"ok": True, "outcome": sit.OUTCOME_DRY_RUN, "now": iso(now), "stop": stopped,
            "note": "STOP is set: a pool would start nothing" if stopped else None,
            "budget": settings.budget, "units_in_use": view.in_use, "units_free": view.free,
            "units_to_start": sum(c.units for c in chosen.starts),
            "seats": [seat.summary(now) for seat in seats], "foreign": [f.summary() for f in view.foreign],
            "schedule": schedule, "not_offered": skips, "board": view.board,
            "refine": view.refine.where if view.refine else None, "problems": view.problems}


def status(state_dir: Path, now: dt.datetime | None = None) -> dict[str, Any]:
    """Whether a pool runs, its units in use and seats, and the sittings that hold units outside it."""
    now = now or utc_now()
    running, holder = pool_running(state_dir)
    data = read_state(state_path(state_dir)) or {}
    seats = [Seat.from_summary(s) for s in data.get("seats") or [] if isinstance(s, Mapping)] if running else []
    seats = [seat for seat in seats if seat is not None]
    try:
        rows = ledger.read(state_dir=state_dir, verb=sit.VERB, since=now - dt.timedelta(hours=LEDGER_LOOKBACK_HOURS))
        foreign = foreign_sittings(live_sittings(rows, now), seats)
        problem = None
    except OSError as exc:
        foreign, problem = [], f"the ledger cannot be read: {type(exc).__name__}: {exc}"
    budget = int(data.get("budget") or BUDGET_UNITS) if running else BUDGET_UNITS
    in_use = sum(seat.units for seat in seats) + sum(info.units for info in foreign)
    answer: dict[str, Any] = {
        "ok": True, "outcome": "running" if running else "not_running", "running": running,
        "holder": holder or None, "stop": stop.stopped(), "budget": budget, "units_in_use": in_use,
        "units_free": max(0, budget - in_use), "by_tier": {tier: sum(s.units for s in seats if s.tier == tier)
                                                           + sum(f.units for f in foreign if f.tier == tier)
                                                           for tier in COST},
        "seats": [seat.summary(now) for seat in seats], "foreign": [f.summary() for f in foreign],
    }
    if running:
        for key in ("since", "updated", "held", "benched", "paused_until", "pause_reason", "stopping", "draining",
                    "board", "counts"):
            answer[key] = data.get(key)
    elif data:
        answer["last_pool"] = {"since": data.get("since"), "updated": data.get("updated")}
    if problem:
        answer["problem"] = problem
    return answer


# --- the verb ------------------------------------------------------------------------------------


def _units(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"a whole number from 1 to {BUDGET_UNITS}, not {text!r}") from None
    if not 1 <= value <= BUDGET_UNITS:
        raise argparse.ArgumentTypeError(f"a whole number from 1 to {BUDGET_UNITS}, not {text!r}")
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
        prog="python -m lab pool",
        description="Keep headless sittings running under a hard budget of 16 units (Sonnet 1, Opus 2, Fable 8), "
                    "refilling freed units at once in order. python -m lab pool status shows the units in use.")
    parser.add_argument("--units", type=_units, default=BUDGET_UNITS,
                        help=f"the budget to fill, at most {BUDGET_UNITS} (default {BUDGET_UNITS})")
    parser.add_argument("--minutes", type=_positive, help="each sitting's cap, passed to sit (default: sit's own)")
    parser.add_argument("--budget-usd", type=_positive, help="each sitting's --max-budget-usd, passed to sit")
    parser.add_argument("--board", type=Path, metavar="FILE", help="read this board snapshot instead of the stored one")
    parser.add_argument("--refine-ref", default=DEFAULT_REF, metavar="REF",
                        help=f"the git ref the refinement files are read at (default {DEFAULT_REF}); "
                             f"'{WORKTREE}' reads the working tree")
    parser.add_argument("--board-max-age", type=_positive, default=float(digest.BOARD_STALE_MINUTES), metavar="MIN",
                        help="board-derived items wait while the snapshot is older than this (default "
                             f"{digest.BOARD_STALE_MINUTES})")
    parser.add_argument("--tick", type=_positive, default=TICK_SECONDS, metavar="S",
                        help=f"how often to look again while no sitting ends (default {TICK_SECONDS:g})")
    parser.add_argument("--exit-when-idle", action="store_true",
                        help="exit once nothing runs and nothing can start, instead of waiting for work")
    parser.add_argument("--dry-run", action="store_true", help="print the schedule it would start; start nothing")
    return parser


def settings_from(args: argparse.Namespace) -> Settings:
    state_dir = ledger.resolve_state_dir().resolve()
    return Settings(state_dir=state_dir, budget=args.units, board_file=args.board, board_max_age=args.board_max_age,
                    refine=refine_files(args.refine_ref))


def status_main(argv: Sequence[str]) -> dict[str, Any]:
    argparse.ArgumentParser(prog="python -m lab pool status",
                            description="Whether a pool runs, the units in use and its seats.").parse_args(list(argv))
    return status(ledger.resolve_state_dir().resolve())


@stop.exempt
def main(argv: Sequence[str]) -> dict[str, Any] | int:
    """Keep sittings running under the 16-unit budget (Sonnet 1, Opus 2, Fable 8); pool status shows the units."""
    args = list(argv)
    if args and args[0] == "status":
        return status_main(args[1:])
    parsed = _parser().parse_args(args)
    settings = settings_from(parsed)
    if parsed.dry_run:
        return dry_run(settings)
    stop.guard()  # main is exempt for status and the dry run; the pool itself never starts under STOP
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(message)s")
    return Pool(settings, minutes=parsed.minutes, budget_usd=parsed.budget_usd, tick_seconds=parsed.tick,
                exit_when_idle=parsed.exit_when_idle).run()
