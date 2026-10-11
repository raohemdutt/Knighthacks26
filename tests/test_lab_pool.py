"""Tests for python -m lab pool, the pool that keeps sittings running under the 16-unit budget.

No test starts a real sitting. The pool is given a fake launcher: a callable that records which
tier and item each seat asked for and answers the way python -m lab sit does, either at once
(scripted answers) or when the test releases it, so a test can hold seats open and watch the
pool refill. An autouse fixture makes the pool's real launcher, and any reach for the agent CLI,
fail the test. Each test has its own state dir, board snapshot and refinement files, and a fixed
clock it can move.
"""

import datetime as dt
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from lab import __main__ as lab_cli
from lab.machine import digest, ledger, pool, sit, slots, slots_fake, slots_notion, stop

NOW = dt.datetime(2026, 10, 10, 15, 0, tzinfo=dt.timezone.utc)
ILAN = "user://fbcc0baf-9942-4661-9414-319688df6910"
HEMDUTT = "user://3f5d872b-594c-8144-b8ea-0002ce9d65db"
TRACKS = ("telemetry", "learning", "plumbing", "kernels")
REAL_SIT_PROCESS_CALL = pool.SitProcess.__call__


# --- the fake launcher -----------------------------------------------------------------------------


def ran(outcome: str = "ok", sitting: str = "sit-fake") -> dict:
    """An answer of python -m lab sit for a sitting that ran (it wrote a start row)."""
    return {"ok": outcome == "ok", "outcome": outcome, "sitting": sitting, "exit_code": 0 if outcome == "ok" else 1,
            "ledger": {"start": 1, "end": 2}}


def refused(outcome: str = "no_cli") -> dict:
    """An answer of python -m lab sit that started nothing."""
    return {"ok": False, "outcome": outcome, "error": f"{outcome}: refused before anything ran"}


class FakeLauncher:
    """Stands in for python -m lab sit. With script, a label's answers come back at once, in order
    (the last one repeats); without, each call waits until the test calls finish()."""

    def __init__(self, script: dict | None = None, default: dict | None = None) -> None:
        self.script = {label: list(answers) for label, answers in (script or {}).items()}
        self.default = default
        self.calls: list[tuple[str, str, dict]] = []
        self._lock = threading.Lock()
        self._gates: dict[str, list[tuple[threading.Event, list]]] = {}

    def __call__(self, tier, item, *, minutes=None, budget_usd=None):
        with self._lock:
            self.calls.append((tier, item.label, {"minutes": minutes, "budget_usd": budget_usd}))
            answers = self.script.get(item.label)
            if answers:
                return answers.pop(0) if len(answers) > 1 else dict(answers[0])
            if self.default is not None:
                return dict(self.default)
            gate, box = threading.Event(), []
            self._gates.setdefault(item.label, []).append((gate, box))
        gate.wait(30)
        return box[0] if box else ran("interrupted")

    def wait_calls(self, count: int, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if len(self.calls) >= count:
                    return
            time.sleep(0.005)
        raise AssertionError(f"expected {count} launcher calls, saw {len(self.calls)}: {self.labels()}")

    def labels(self) -> list[str]:
        with self._lock:
            return [label for _, label, _ in self.calls]

    def finish(self, label: str, answer: dict | None = None) -> None:
        with self._lock:
            gate, box = self._gates[label].pop(0)
        box.append(answer if answer is not None else ran())
        gate.set()

    def release_all(self) -> None:
        with self._lock:
            pending = [entry for entries in self._gates.values() for entry in entries]
            self._gates.clear()
        for gate, box in pending:
            box.append(ran("interrupted"))
            gate.set()


class Clock:
    def __init__(self, now: dt.datetime = NOW) -> None:
        self.now = now

    def __call__(self) -> dt.datetime:
        return self.now

    def advance(self, **delta) -> None:
        self.now += dt.timedelta(**delta)


# --- the board snapshot and the refinement files ------------------------------------------------


def task(task_id: str, *, tier: str | None = "Opus", status: str = "Not started", priority: str = "P0 Demo-critical",
         wave: float | None = 1, locked: bool = False, blocked_by=(), title: str | None = None, assignees=(ILAN,),
         stretch: bool = False) -> dict:
    return {"userDefined:ID": task_id, "Task": f"{task_id}: {title or 'build the ' + task_id.lower() + ' piece'}",
            "Status": status, "Tier": tier, "Priority": priority, "Wave": wave,
            "Locked": "__YES__" if locked else "__NO__", "Stretch": "__YES__" if stretch else "__NO__",
            "Assignee": json.dumps(list(assignees)), "Blocked by": json.dumps(list(blocked_by)),
            "Context": "Orchestration", "url": f"https://app.notion.com/p/{abs(hash(task_id)) % 10**32:032x}"}


def save_board(state: Path, tasks, *, taken_at: dt.datetime = NOW, queue=()) -> Path:
    path = digest.board_path(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": 1, "taken_at": digest.iso(taken_at), "taken_by": "test",
                                "tasks": list(tasks), "gpu_queue": list(queue)}), encoding="utf-8")
    return path


PASS_ENTRY = "## Pass {n}, 2026-10-10\nWalked the track.\n{ending}\n"
RUN_STATE = "## State at the end of the run, passes 1 to {n}, 2026-10-10\n\nRun ended: {how} at pass {n}.\n"


def write_track(root: Path, track: str, *, counter: int = 1, ending: str = "converged", ended: dt.datetime | None = None,
                closed: bool = True, rulings: str = "") -> None:
    """One track folder: passes 1..counter, the last ending as given (converged, cap, or "" for an
    open run), the closing section when closed, and the last pass's report ended at ended."""
    folder = root / track
    (folder / "pass_reports").mkdir(parents=True, exist_ok=True)
    entries = []
    for n in range(1, counter + 1):
        last = n == counter
        line = ""
        if last and ending == "converged":
            line = f"CONVERGED at pass {n}"
        elif last and ending == "cap":
            line = f"CAP REACHED at pass {n}, not converged"
        entries.append(PASS_ENTRY.format(n=n, ending=line))
    if counter and closed and ending in ("converged", "cap"):
        entries.append(RUN_STATE.format(n=counter, how="converged" if ending == "converged" else "cap reached"))
    (folder / "REFINE_LOG.md").write_text(f"# {track} log\n\n" + "\n".join(entries), encoding="utf-8")
    (folder / "refine_pass.txt").write_text(f"{counter}\n", encoding="utf-8")
    (folder / "OWNER_RULINGS.md").write_text(f"# Owner rulings for the {track} track\n\n{rulings}", encoding="utf-8")
    if counter:
        ended = ended or NOW - dt.timedelta(minutes=10)
        (folder / "pass_reports" / f"pass_{counter:02d}.md").write_text(
            f"Verbatim report returned by {track} refinement pass {counter} (test), 2026-10-10.\n"
            f"Ended: {ended.isoformat()}\n", encoding="utf-8")


def write_merge_log(root: Path, reconciled: dict[str, int]) -> None:
    """MERGE_LOG.md with one merge that walked each track's passes up to the number given."""
    walked = "".join(f"- {track} pass {n} :: M6 :: held\n" for track, upto in reconciled.items()
                     for n in range(1, upto + 1))
    body = "# Merge log\n\n"
    if walked:
        body += f"## Merge 1, 2026-10-10\n\nWalked (pass entry :: checks :: result):\n{walked}"
    (root / "MERGE_LOG.md").write_text(body, encoding="utf-8")


def quiet_refine(root: Path) -> Path:
    """Four tracks that each converged at pass 1 ten minutes ago, all reconciled: no pass or merge is due."""
    for track in TRACKS:
        write_track(root, track)
    write_merge_log(root, {track: 1 for track in TRACKS})
    return root


def rows(state: Path, verb_prefix: str = "pool") -> list[dict]:
    return [row for row in ledger.read(state_dir=state) if str(row.get("verb", "")).startswith(verb_prefix)]


def verbs(state: Path) -> list[str]:
    return [row["verb"] for row in rows(state)]


# --- fixtures --------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def lab(tmp_path, monkeypatch):
    """A private state dir, quiet refinement files, no referee verb, and no real sitting or CLI."""
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setenv("LAB_STATE_DIR", str(state))
    monkeypatch.delenv(sit.ACTOR_ENV, raising=False)

    def no_real_sitting(*args, **kwargs):
        raise AssertionError("a test reached for a real sitting")

    monkeypatch.setattr(pool.SitProcess, "__call__", no_real_sitting)
    monkeypatch.setattr(sit, "resolve_command", no_real_sitting)
    quiet_refine(tmp_path / "refine")
    return state


@pytest.fixture
def refine_root(tmp_path) -> Path:
    return tmp_path / "refine"


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def make_pool(lab, refine_root, clock):
    made: list[tuple[pool.Pool, FakeLauncher]] = []

    def make(launcher: FakeLauncher | None = None, *, budget: int = 16, referee: bool = False,
             exit_when_idle: bool = False, **settings) -> tuple[pool.Pool, FakeLauncher]:
        fake = launcher or FakeLauncher()
        names = ["digest", "sit", "referee"] if referee else ["digest", "sit"]
        conf = pool.Settings(state_dir=lab, budget=budget, refine=pool.TreeFiles(refine_root),
                             verbs=lambda: names, **settings)
        made.append((pool.Pool(conf, launcher=fake, tick_seconds=0.01, exit_when_idle=exit_when_idle,
                               clock=clock), fake))
        return made[-1]

    yield make
    for each, fake in made:
        fake.release_all()
        each.lock.release()


def started(p: pool.Pool) -> list[str]:
    return sorted(seat.label for seat in p.seats.values())


def settle(p: pool.Pool, fake: FakeLauncher, label: str, answer: dict | None = None) -> None:
    """End one held seat and let the pool record it and refill."""
    fake.finish(label, answer)
    p.tick(10)


# --- the order -------------------------------------------------------------------------------------


def candidate(tier: str, name: str, rank: int = 0, wave=1, kind: str = "task", stretch: bool = False):
    item = sit.Item(kind, name if kind in ("task", "pass") else None)
    return pool.Candidate(tier, item, kind, rank, wave, stretch)


def test_plan_starts_in_order_and_holds_the_free_units_for_the_first_that_does_not_fit():
    chosen = pool.plan([candidate("sonnet", "S1", rank=1), candidate("fable", "F1", rank=0, wave=2),
                        candidate("opus", "O1", rank=0, wave=1), candidate("opus", "O2", rank=1)], free=9, budget=16)
    assert [c.label for c in chosen.starts] == ["task O1"]
    assert chosen.held.label == "task F1"
    assert [c.label for c in chosen.after] == ["task O2", "task S1"]
    assert chosen.never_fits == []


def test_a_fable_sitting_starts_only_when_eight_units_are_free_and_nothing_smaller_jumps_it():
    head = candidate("fable", "F1", rank=0)
    assert pool.plan([head, candidate("sonnet", "S1", rank=1)], free=7, budget=16).starts == []
    chosen = pool.plan([head, candidate("sonnet", "S1", rank=1)], free=9, budget=16)
    assert [c.label for c in chosen.starts] == ["task F1", "task S1"]


def test_an_item_that_costs_more_than_the_budget_is_set_aside_and_never_holds_units():
    chosen = pool.plan([candidate("fable", "F1", rank=0), candidate("sonnet", "S1", rank=1)], free=4, budget=4)
    assert [c.label for c in chosen.starts] == ["task S1"]
    assert chosen.held is None
    assert [c.label for c in chosen.never_fits] == ["task F1"]


def test_the_order_is_priority_then_wave_then_stretch_then_duties_before_passes_before_tasks_then_id():
    ordered = [
        candidate("fable", "", rank=0, wave=1, kind="merge"),
        candidate("fable", "", rank=0, wave=1, kind="referee"),
        candidate("opus", "telemetry", rank=0, wave=1, kind="pass"),
        candidate("opus", "O9", rank=0, wave=1),
        candidate("opus", "O10", rank=0, wave=1),
        candidate("opus", "O3", rank=0, wave=1, stretch=True),
        candidate("opus", "O1", rank=0, wave=2),
        candidate("opus", "O2", rank=0, wave=None),
        candidate("sonnet", "S1", rank=1, wave=0),
        candidate("sonnet", "S2", rank=9, wave=0),
    ]
    shuffled = ordered[::-1]
    assert [c.label for c in sorted(shuffled, key=pool.Candidate.key)] == [c.label for c in ordered]


# --- filling the budget ----------------------------------------------------------------------------


def test_the_pool_fills_sixteen_units_and_never_more(make_pool, lab):
    save_board(lab, [task(f"O{n}", wave=n) for n in range(1, 11)])
    p, fake = make_pool()
    assert p.begin() is None
    p.fill(NOW)
    fake.wait_calls(8)
    assert started(p) == sorted(f"task O{n}" for n in range(1, 9))
    assert sum(seat.units for seat in p.seats.values()) == 16
    assert p.held.label == "task O9"
    assert pool.status(lab, NOW)["units_in_use"] == 16
    assert {tier for tier, _, _ in fake.calls} == {"opus"}


def test_each_tier_holds_its_units_one_eight_and_two(make_pool, lab):
    save_board(lab, [task("F1", tier="Fable", wave=1), *[task(f"O{n}", wave=2) for n in range(1, 5)],
                     *[task(f"S{n}", tier="Sonnet", priority="P1 Strengthens demo", wave=1) for n in range(1, 4)]])
    p, fake = make_pool()
    p.begin()
    p.fill(NOW)
    fake.wait_calls(5)
    assert started(p) == ["task F1", "task O1", "task O2", "task O3", "task O4"]
    assert p.held.label == "task S1"
    answer = pool.status(lab, NOW)
    assert (answer["units_in_use"], answer["units_free"]) == (16, 0)
    assert answer["by_tier"] == {"sonnet": 0, "opus": 8, "fable": 8}


def test_when_a_sitting_ends_its_units_are_refilled_at_once_with_the_next_item_in_order(make_pool, lab):
    save_board(lab, [task(f"O{n}", wave=n) for n in range(1, 11)])
    p, fake = make_pool()
    p.begin()
    p.fill(NOW)
    fake.wait_calls(8)
    settle(p, fake, "task O3")
    fake.wait_calls(9)
    assert "task O9" in started(p) and "task O3" not in started(p)
    assert len(p.seats) == 8
    assert fake.labels()[-1] == "task O9"
    end = [row for row in rows(lab) if row["verb"] == pool.V_END]
    assert [(row["item"], row["sitting_outcome"], row["result"]) for row in end] == [("task O3", "ok", "ok")]


def test_tasks_that_are_blocked_locked_shared_untiered_or_not_started_never_start(make_pool, lab):
    save_board(lab, [
        task("O1", status="Done"),
        task("O2", blocked_by=["O1"]),
        task("O3", blocked_by=["O4"]),
        task("O4", status="In progress"),
        task("O5", locked=True),
        task("O6", assignees=(ILAN, HEMDUTT)),
        task("O7", assignees=(HEMDUTT,)),
        task("O8", tier=None),
        task("O9", blocked_by=["O99"]),
        task("O10", status="Blocked"),
    ])
    p, fake = make_pool()
    p.begin()
    p.fill(NOW)
    fake.wait_calls(1)
    assert started(p) == ["task O2"]
    reasons = {skip.item: skip.reason for skip in p.view.skips}
    assert reasons["task O8"] == "no Tier on the board (the referee sets it)"
    assert reasons["task O9"] == "a blocker is missing from the board snapshot"
    assert "task O3" not in reasons and "task O5" not in reasons


def test_a_stale_board_starts_no_task_and_says_why(make_pool, lab):
    save_board(lab, [task("O1")], taken_at=NOW - dt.timedelta(minutes=61))
    p, fake = make_pool()
    p.begin()
    p.fill(NOW)
    assert p.seats == {}
    assert fake.calls == []
    assert [skip.reason for skip in p.view.skips if skip.item == "board"] == [
        "the snapshot is 61 min old, over 60: no task, pass or merge starts until a fresh one is saved "
        "(python -m lab digest --save-board <file>)"]


def test_without_a_board_snapshot_nothing_starts(make_pool):
    p, fake = make_pool()
    p.begin()
    p.fill(NOW)
    assert p.seats == {} and fake.calls == []
    assert any(skip.item == "board" and skip.reason.startswith("no board snapshot") for skip in p.view.skips)


def test_a_lower_budget_is_filled_and_never_exceeded(make_pool, lab):
    save_board(lab, [task(f"S{n}", tier="Sonnet", wave=n) for n in range(1, 8)])
    p, fake = make_pool(budget=5)
    p.begin()
    p.fill(NOW)
    fake.wait_calls(5)
    assert started(p) == [f"task S{n}" for n in range(1, 6)]


# --- sittings outside the pool -------------------------------------------------------------------


def sit_row(state: Path, run_id: str, tier: str, item: str, *, outcome: str = "started", ts: dt.datetime,
            cap: float = 240) -> None:
    ledger.append({"ts": ts, "actor": sit.ACTOR, "tier": tier, "task": None, "verb": sit.VERB, "outcome": outcome,
                   "metrics": {}, "refs": {"run_id": run_id}, "item": item, "cap_minutes": cap}, state_dir=state)


def test_sittings_started_outside_the_pool_hold_their_units_and_their_items_never_start_twice(make_pool, lab):
    sit_row(lab, "sit-hand-1", "opus", "task O1", ts=NOW - dt.timedelta(minutes=30))
    sit_row(lab, "sit-hand-2", "fable", "merge", ts=NOW - dt.timedelta(minutes=300), cap=240)  # past its cap
    sit_row(lab, "sit-hand-3", "sonnet", "task S1", ts=NOW - dt.timedelta(minutes=50))
    sit_row(lab, "sit-hand-3", "sonnet", "task S1", outcome="ok", ts=NOW - dt.timedelta(minutes=5))  # ended
    save_board(lab, [task(f"O{n}", wave=n) for n in range(1, 10)])
    p, fake = make_pool()
    p.begin()
    p.fill(NOW)
    fake.wait_calls(7)
    assert "task O1" not in started(p)
    assert started(p) == sorted(f"task O{n}" for n in range(2, 9))
    answer = pool.status(lab, NOW)
    assert [f["sitting"] for f in answer["foreign"]] == ["sit-hand-1"]
    assert answer["units_in_use"] == 16


def test_status_without_a_pool_counts_the_live_sittings_in_the_ledger(lab):
    sit_row(lab, "sit-hand-1", "fable", "referee", ts=NOW - dt.timedelta(minutes=10))
    sit_row(lab, "sit-hand-2", "sonnet", "task S1", ts=NOW - dt.timedelta(minutes=10))
    answer = pool.status(lab, NOW)
    assert answer["running"] is False and answer["outcome"] == "not_running"
    assert (answer["units_in_use"], answer["units_free"]) == (9, 7)
    assert answer["seats"] == []


def test_a_seat_does_not_count_twice_once_its_sitting_writes_its_start_row(make_pool, lab):
    save_board(lab, [task("O1"), task("O2", wave=2)])
    p, fake = make_pool()
    p.begin()
    p.fill(NOW)
    fake.wait_calls(2)
    sit_row(lab, "sit-own-1", "opus", "task O1", ts=NOW)
    answer = pool.status(lab, NOW)
    assert answer["units_in_use"] == 4
    assert answer["foreign"] == []
    assert [seat["item"] for seat in answer["seats"]] == ["task O1", "task O2"]


# --- STOP --------------------------------------------------------------------------------------------


def test_under_stop_nothing_new_starts_and_the_pool_exits_once_its_seats_have_ended(make_pool, lab):
    save_board(lab, [task(f"O{n}", wave=n) for n in range(1, 11)])
    p, fake = make_pool()
    p.begin()
    p.fill(NOW)
    fake.wait_calls(8)
    stop.stop_main(["owner", "pause"])
    settle(p, fake, "task O1")
    assert len(p.seats) == 7 and len(fake.calls) == 8
    assert p.done() is None
    for label in sorted(seat.label for seat in list(p.seats.values())):
        fake.finish(label)
    while p.seats:
        p.tick(10)
    assert p.done() == "stopped"
    answer = p.finish("stopped")
    assert answer["ok"] is True and answer["outcome"] == "stopped" and answer["started"] == 8
    stop_rows = [row for row in rows(lab) if row["verb"] == pool.V_STOP]
    assert [(row["outcome"], row["reason"], row["seats"]) for row in stop_rows] == [("stopped", "owner pause", 8)]
    assert pool.status(lab, NOW)["running"] is False


def test_the_pool_run_is_refused_under_stop_but_status_and_the_dry_run_answer(capsys, lab, monkeypatch, refine_root):
    monkeypatch.setattr(pool, "refine_files", lambda ref, repo=pool.REPO_ROOT: pool.TreeFiles(refine_root))
    stop.stop_main(["maintenance"])
    assert lab_cli.main(["pool"]) == 1
    refused = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert (refused["ok"], refused["outcome"], refused["reason"]) == (False, "stopped", "maintenance")
    assert lab_cli.main(["pool", "status"]) == 0
    assert json.loads(capsys.readouterr().out)["stop"] is True
    assert lab_cli.main(["pool", "--dry-run"]) == 0
    dry = json.loads(capsys.readouterr().out)
    assert (dry["outcome"], dry["stop"], dry["note"]) == ("dry_run", True, "STOP is set: a pool would start nothing")
    assert rows(lab) == []


# --- the ledger and the state file ---------------------------------------------------------------


def test_every_decision_writes_one_ledger_row_and_the_run_ends_when_idle(make_pool, lab):
    save_board(lab, [task("O1"), task("S1", tier="Sonnet", wave=2)])
    p, fake = make_pool(FakeLauncher(script={"task O1": [ran(sitting="sit-a")], "task S1": [ran("error", "sit-b")]}),
                        budget=2, exit_when_idle=True)
    answer = p.run()
    assert answer["outcome"] == "idle" and answer["ok"] is True
    assert answer["started"] == 4 and answer["failed"] == 3 and answer["benched"] == 1
    assert verbs(lab) == [pool.V_BEGIN, pool.V_START, pool.V_HOLD, pool.V_END, pool.V_START, pool.V_END,
                          pool.V_START, pool.V_END, pool.V_START, pool.V_END, pool.V_BENCH, pool.V_EXIT]
    start = next(row for row in rows(lab) if row["verb"] == pool.V_START)
    assert {k: start[k] for k in ("actor", "tier", "task", "outcome", "item", "seat", "units_held", "source", "progress")} == {
        "actor": "lab pool", "tier": "opus", "task": "O1", "outcome": "started", "item": "task O1", "seat": "seat-1",
        "units_held": 2, "source": "task", "progress": {"board": "2026-10-10T15:00:00Z"}}
    assert start["metrics"] == {"pool.units_in_use": 2, "pool.units_free": 0}
    end = next(row for row in rows(lab) if row["verb"] == pool.V_END and row["item"] == "task O1")
    assert (end["outcome"], end["sitting_outcome"], end["result"], end["refs"]["run_id"]) == ("ended", "ok", "ok", "sit-a")
    exit_row = rows(lab)[-1]
    assert (exit_row["outcome"], exit_row["started"], exit_row["ended"]) == ("idle", 4, 4)
    saved = json.loads(pool.state_path(lab).read_text(encoding="utf-8"))
    assert saved["running"] is False and saved["seats"] == []
    assert [b["item"] for b in saved["benched"]] == ["task S1"]


def test_status_shows_the_running_pool_its_seats_and_what_is_held(make_pool, lab):
    save_board(lab, [task("F1", tier="Fable"), task("F2", tier="Fable", wave=2), task("O1", wave=3)])
    p, fake = make_pool()
    p.begin()
    p.fill(NOW)
    fake.wait_calls(2)
    answer = pool.status(lab, NOW + dt.timedelta(minutes=12))
    assert (answer["ok"], answer["outcome"], answer["running"]) == (True, "running", True)
    assert (answer["units_in_use"], answer["units_free"], answer["budget"]) == (16, 0, 16)
    assert [(s["seat"], s["tier"], s["item"], s["units"], s["minutes"]) for s in answer["seats"]] == [
        ("seat-1", "fable", "task F1", 8, 12.0), ("seat-2", "fable", "task F2", 8, 12.0)]
    assert answer["seats"][0]["command"] == "python -m lab sit --tier fable --task F1"
    assert answer["held"]["item"] == "task O1"
    assert answer["holder"].startswith("pid ")


def test_a_second_pool_is_refused_while_one_runs(make_pool):
    first, _ = make_pool()
    second, _ = make_pool()
    assert first.begin() is None
    refusal = second.begin()
    assert (refusal["ok"], refusal["outcome"]) == (False, "already_running")
    assert refusal["holder"].startswith("pid ")


def test_the_held_item_is_written_once_until_it_changes(make_pool, lab):
    save_board(lab, [task("O1"), task("F1", tier="Fable", wave=2)])
    p, fake = make_pool(budget=9)
    p.begin()
    p.fill(NOW)
    p.fill(NOW)
    fake.wait_calls(1)
    holds = [row for row in rows(lab) if row["verb"] == pool.V_HOLD]
    assert [(row["item"], row["units_held"]) for row in holds] == [("task F1", 8)]


# --- restarts -------------------------------------------------------------------------------------


def test_a_failing_item_is_restarted_twice_then_benched_while_other_work_goes_on(make_pool, lab, clock):
    save_board(lab, [task("O1"), task("O2", wave=2)])
    p, fake = make_pool()
    p.begin()
    p.fill(NOW)
    fake.wait_calls(2)
    for attempt in range(3):
        settle(p, fake, "task O1", ran("error"))
        if attempt < 2:
            fake.wait_calls(3 + attempt)
            assert "task O1" in started(p)
    assert started(p) == ["task O2"]
    assert fake.labels().count("task O1") == 3
    bench = [row for row in rows(lab) if row["verb"] == pool.V_BENCH]
    assert [(row["item"], row["failures"], row["until"]) for row in bench] == [("task O1", 3, "2026-10-10T21:00:00Z")]
    p.fill(NOW)
    assert "task O1" not in started(p)
    assert any(s.item == "task O1" and s.reason.startswith("benched after 3 failed sittings") for s in p.view.skips)
    clock.advance(hours=6, minutes=1)
    save_board(lab, [task("O1"), task("O2", wave=2)], taken_at=clock.now)
    p.fill(clock.now)
    fake.wait_calls(5)
    assert "task O1" in started(p)


def test_a_pool_started_again_remembers_the_failures_its_predecessor_wrote(make_pool, lab):
    save_board(lab, [task("O1")])
    first, fake = make_pool(FakeLauncher(script={"task O1": [ran("timeout"), ran("budget")]}))
    first.begin()
    first.fill(NOW)
    fake.wait_calls(1)
    first.tick(10)
    fake.wait_calls(2)
    first.tick(10)
    first.finish("interrupted")
    second, fake2 = make_pool(FakeLauncher(script={"task O1": [ran("error")]}))
    second.begin()
    assert second.history.get("task O1").failures == 2
    second.fill(NOW)
    fake2.wait_calls(1)
    second.tick(10)
    assert started(second) == []
    assert [row["failures"] for row in rows(lab) if row["verb"] == pool.V_BENCH] == [3]


def test_an_ok_end_resets_the_failures(make_pool, lab):
    save_board(lab, [task("O1")])
    p, fake = make_pool(FakeLauncher(script={"task O1": [ran("error"), ran("error"), ran("ok")]}))
    p.begin()
    for _ in range(3):
        p.fill(NOW)
        p.tick(10)
    record = p.history.get("task O1")
    assert (record.failures, record.last_result) == (0, "ok")
    assert [row["result"] for row in rows(lab) if row["verb"] == pool.V_END] == ["failed", "failed", "ok"]


def test_stopped_and_interrupted_sittings_count_no_failure_and_are_offered_again(make_pool, lab):
    save_board(lab, [task("O1", status="In progress", locked=True)])
    p, fake = make_pool()
    p.begin()
    p.history.ended("task O1", "opus", pool.RESULT_NEUTRAL, NOW - dt.timedelta(minutes=5), {})
    p.fill(NOW)
    fake.wait_calls(1)
    assert started(p) == ["task O1"]
    settle(p, fake, "task O1", ran("interrupted"))
    assert p.history.get("task O1").failures == 0
    assert pool.classify(ran("stopped")) == pool.RESULT_NEUTRAL
    assert pool.classify(refused("stopped")) == pool.RESULT_NEUTRAL


def test_a_sitting_that_started_nothing_pauses_the_pool_without_a_failure(make_pool, lab, clock):
    save_board(lab, [task("O1"), task("O2", wave=2)])
    p, fake = make_pool(FakeLauncher(script={"task O1": [refused("no_cli")]}))
    p.begin()
    p.fill(NOW)
    fake.wait_calls(2)
    p.tick(10)
    pause = [row for row in rows(lab) if row["verb"] == pool.V_PAUSE]
    assert [(row["outcome"], row["until"]) for row in pause] == [("paused", "2026-10-10T15:10:00Z")]
    assert pause[0]["reason"].startswith("python -m lab sit started nothing for task O1: no_cli")
    assert p.history.get("task O1").failures == 0
    p.fill(NOW)
    assert "task O1" not in started(p)
    clock.advance(minutes=11)
    p.fill(clock.now)
    fake.wait_calls(3)
    assert "task O1" in started(p)


def test_a_child_that_died_after_its_sitting_started_counts_as_a_failed_sitting_not_a_refusal(make_pool, lab):
    save_board(lab, [task("S1", tier="Sonnet")])
    p, fake = make_pool()
    p.begin()
    p.fill(NOW)
    fake.wait_calls(1)
    assert started(p) == ["task S1"]
    # The sit child wrote its start row, then died before printing its answer: the launcher
    # answers with the no-answer error and no ledger key.
    sit_row(lab, "sit-crashed-1", "sonnet", "task S1", ts=NOW)
    crashed = {"ok": False, "outcome": "error",
               "error": "python -m lab sit exited 1 without an answer: Traceback (most recent call last)"}
    settle(p, fake, "task S1", crashed)
    record = p.history.get("task S1")
    assert (record.last_result, record.failures) == (pool.RESULT_FAILED, 1)
    assert [row for row in rows(lab) if row["verb"] == pool.V_PAUSE] == []


def test_five_failed_sittings_in_a_row_pause_the_pool(make_pool, lab):
    save_board(lab, [task(f"S{n}", tier="Sonnet", wave=n) for n in range(1, 8)])
    p, fake = make_pool(budget=5)
    p.begin()
    p.fill(NOW)
    fake.wait_calls(5)
    for n in range(1, 6):
        fake.finish(f"task S{n}", ran("error"))
    deadline = time.monotonic() + 10
    while p.ended.qsize() < 5 and time.monotonic() < deadline:
        time.sleep(0.005)
    p.tick(10)
    assert p.seats == {}
    pauses = [row for row in rows(lab) if row["verb"] == pool.V_PAUSE]
    assert [(row["reason"], row["until"]) for row in pauses] == [("5 sittings in a row failed", "2026-10-10T15:10:00Z")]
    assert len(fake.calls) == 5


def test_a_failed_task_is_restarted_even_when_the_board_shows_its_own_claim(make_pool, lab):
    save_board(lab, [task("O1")])
    p, fake = make_pool()
    p.begin()
    p.fill(NOW)
    fake.wait_calls(1)
    save_board(lab, [task("O1", status="In progress", locked=True)], taken_at=NOW + dt.timedelta(seconds=30))
    settle(p, fake, "task O1", ran("timeout"))
    fake.wait_calls(2)
    assert started(p) == ["task O1"]
    assert "restart after a failed sitting" in p.seats["seat-2"].why


def test_a_task_that_ended_ok_waits_for_a_board_snapshot_taken_after_it(make_pool, lab, clock):
    save_board(lab, [task("O1")])
    p, fake = make_pool()
    p.begin()
    p.fill(NOW)
    fake.wait_calls(1)
    clock.advance(minutes=20)
    settle(p, fake, "task O1")
    assert started(p) == []
    assert [s.reason for s in p.view.skips if s.item == "task O1"] == [
        "its last sitting ended ok at 2026-10-10T15:20:00Z; waiting for a board snapshot taken after that"]
    save_board(lab, [task("O1")], taken_at=clock.now + dt.timedelta(minutes=1))
    clock.advance(minutes=2)
    p.fill(clock.now)
    fake.wait_calls(2)
    assert started(p) == ["task O1"]


# --- refinement passes and the merge --------------------------------------------------------------


def pass_task(task_id: str, track: str, *, first: bool = True, **more) -> dict:
    title = f"First refinement pass, {track} track" if first else f"{track.capitalize()} loop, passes 2 onward until converged or cap"
    return task(task_id, title=title, **more)


def merge_task(task_id: str = "O41", **more) -> dict:
    return task(task_id, tier="Fable", title="First merge pass across the four tracks", **more)


def offered(p: pool.Pool) -> list[tuple[str, str]]:
    return [(c.tier, c.label) for c in p.view.candidates]


def test_a_first_pass_is_due_once_its_board_task_is_unblocked_and_never_runs_as_a_task(make_pool, lab, refine_root):
    write_track(refine_root, "telemetry", counter=0)
    write_track(refine_root, "learning", counter=0)
    save_board(lab, [task("O1", status="Done"), task("O2"),
                     pass_task("O37", "telemetry", blocked_by=["O1"], priority="P1 Strengthens demo", wave=3),
                     pass_task("O38", "learning", blocked_by=["O2"])])
    p, fake = make_pool()
    p.begin()
    p.fill(NOW)
    fake.wait_calls(2)
    assert started(p) == ["pass telemetry", "task O2"]
    seat = next(s for s in p.seats.values() if s.item.kind == "pass")
    assert (seat.tier, seat.why, seat.progress) == ("opus", "the track's first pass; board task O37", {"counter": 0})
    assert ("opus", "task O37") not in [(t, label) for t, label, _ in fake.calls]
    start = next(row for row in rows(lab) if row["verb"] == pool.V_START and row["item"] == "pass telemetry")
    assert start["task"] is None and start["units_held"] == 2


def test_an_open_run_continues_and_a_converged_track_waits_two_hours(make_pool, lab, refine_root, clock):
    write_track(refine_root, "telemetry", counter=3, ending="", closed=False)
    write_track(refine_root, "learning", counter=2, ending="converged", ended=NOW - dt.timedelta(minutes=119))
    write_merge_log(refine_root, {"telemetry": 3, "learning": 2, "plumbing": 1, "kernels": 1})
    save_board(lab, [])
    p, _ = make_pool()
    p.begin()
    p.fill(NOW)
    assert [(c.label, c.why) for c in p.view.candidates] == [("pass telemetry", "pass 4 continues the open run")]
    clock.advance(minutes=2)
    save_board(lab, [], taken_at=clock.now)
    p.fill(clock.now)
    assert ("pass learning", "121 min since pass 2 ended") in [(c.label, c.why) for c in p.view.candidates]


def test_a_capped_track_waits_for_the_answer_to_its_cap_ruling(make_pool, lab, refine_root):
    write_track(refine_root, "kernels", counter=10, ending="cap", ended=NOW - dt.timedelta(hours=5))
    write_merge_log(refine_root, {"telemetry": 1, "learning": 1, "plumbing": 1, "kernels": 10})
    save_board(lab, [])
    p, _ = make_pool()
    p.begin()
    p.fill(NOW)
    assert offered(p) == []
    assert [s.reason for s in p.view.skips if s.item == "pass kernels"] == [
        "the run reached its cap and the cap ruling has no Answer"]
    ruling = ("## 1. Extend the kernels run or open a new one? (pass 10)\n\nNow: the track is capped.\n"
              "Recommend: open a new run.\nSource of tension: the cap.\nAnswer: open a new run.\n")
    write_track(refine_root, "kernels", counter=10, ending="cap", ended=NOW - dt.timedelta(hours=5), rulings=ruling)
    p.fill(NOW)
    assert offered(p) == [("opus", "pass kernels")]


def test_a_pass_that_ended_ok_waits_for_its_pull_request_then_the_merge_comes_next(make_pool, lab, refine_root, clock):
    write_track(refine_root, "telemetry", counter=1, ending="", closed=False)
    write_merge_log(refine_root, {"telemetry": 1, "learning": 1, "plumbing": 1, "kernels": 1})
    save_board(lab, [])
    p, fake = make_pool(referee=False)
    p.begin()
    p.fill(NOW)
    fake.wait_calls(1)
    assert started(p) == ["pass telemetry"]
    clock.advance(minutes=30)
    settle(p, fake, "pass telemetry")
    save_board(lab, [], taken_at=clock.now)
    p.fill(clock.now)
    assert started(p) == []
    assert [s.reason for s in p.view.skips if s.item == "pass telemetry"] == [
        "its last sitting ended ok; waiting for its pull request to merge (the counter is still 1)"]
    write_track(refine_root, "telemetry", counter=2, ending="", closed=False)  # the pull request merged
    p.fill(clock.now)
    fake.wait_calls(2)
    assert started(p) == ["merge"]
    seat = p.seats["seat-2"]
    assert (seat.tier, seat.why, seat.progress) == ("fable", "the round's passes wait for the merge: telemetry (1)",
                                                   {"merges": 1})


def test_the_merge_waits_for_the_running_passes_and_a_due_track_may_still_join_the_round(make_pool, lab, refine_root):
    write_track(refine_root, "telemetry", counter=2, ending="", closed=False)   # pass 2 waits for the merge
    write_track(refine_root, "learning", counter=1, ending="", closed=False)    # its run goes on: due
    write_merge_log(refine_root, {"telemetry": 1, "learning": 1, "plumbing": 1, "kernels": 1})
    sit_row(lab, "sit-hand-plumbing", "opus", "pass plumbing", ts=NOW - dt.timedelta(minutes=20))
    save_board(lab, [])
    p, fake = make_pool()
    p.begin()
    p.fill(NOW)
    fake.wait_calls(1)
    assert started(p) == ["pass learning"]
    assert [s.reason for s in p.view.skips if s.item == "merge"] == ["waits for the passes still running: plumbing"]
    p.fill(NOW)
    assert [s.reason for s in p.view.skips if s.item == "merge"] == [
        "waits for the passes still running: learning, plumbing"]
    assert offered(p) == []


def test_while_the_merge_is_due_or_running_no_pass_starts(make_pool, lab, refine_root):
    write_track(refine_root, "telemetry", counter=2, ending="", closed=False)
    write_track(refine_root, "learning", counter=1, ending="", closed=False)
    write_merge_log(refine_root, {"telemetry": 1, "learning": 1, "plumbing": 1, "kernels": 1})
    sit_row(lab, "sit-hand-learning", "opus", "pass learning", ts=NOW - dt.timedelta(hours=1))
    sit_row(lab, "sit-hand-learning", "opus", "pass learning", outcome="ok", ts=NOW - dt.timedelta(minutes=1))
    save_board(lab, [])
    p, fake = make_pool()
    p.begin()
    p.fill(NOW)
    fake.wait_calls(1)
    assert started(p) == ["merge"]
    assert offered(p) == [("fable", "merge")]
    p.fill(NOW)
    assert offered(p) == []


def test_a_locked_merge_task_means_a_merge_runs_elsewhere_and_holds_every_pass(make_pool, lab, refine_root):
    write_track(refine_root, "learning", counter=1, ending="", closed=False)
    save_board(lab, [merge_task(status="In progress", locked=True)])
    p, _ = make_pool()
    p.begin()
    p.fill(NOW)
    assert offered(p) == []
    assert [s.reason for s in p.view.skips if s.item == "merge"] == [
        "O41 is Locked on the board: a session runs the merge, and no pass starts until it has finished"]


def test_a_locked_pass_task_means_that_pass_runs_elsewhere(make_pool, lab, refine_root):
    write_track(refine_root, "telemetry", counter=0)
    save_board(lab, [pass_task("O37", "telemetry", status="In progress", locked=True)])
    p, _ = make_pool()
    p.begin()
    p.fill(NOW)
    assert offered(p) == []
    assert [s.reason for s in p.view.skips if s.item == "pass telemetry"] == [
        "O37 is Locked on the board: a session runs it"]


def test_the_first_merge_waits_for_its_board_task_blockers(make_pool, lab, refine_root):
    write_track(refine_root, "telemetry", counter=1, ending="", closed=False)
    write_merge_log(refine_root, {})
    save_board(lab, [pass_task("O37", "telemetry", status="Done"), pass_task("O38", "learning"),
                     merge_task(blocked_by=["O37", "O38"])])
    write_track(refine_root, "learning", counter=0)
    p, fake = make_pool()
    p.begin()
    p.fill(NOW)
    fake.wait_calls(1)
    assert started(p) == ["pass learning"]
    assert "merge" not in [c.label for c in p.view.candidates]


# --- the referee ---------------------------------------------------------------------------------


def queue_row(task_id: str, status: str, *, minutes: float = 15, started=None, requested=None, complaint: str = "") -> dict:
    return {"Request": f"{task_id}: a short check", "Task ID": task_id, "GPU": "B exclusive", "Minutes": minutes,
            "Status": status, "Agent": "Ilan", "date:Requested:start": digest.iso(requested or NOW - dt.timedelta(minutes=5)),
            "date:Started:start": digest.iso(started) if started else None, "Extended": "__NO__", "Complaint": complaint}


def test_no_referee_sitting_starts_until_the_referee_verb_is_registered(make_pool, lab):
    save_board(lab, [], queue=[queue_row("R11", "Running", started=NOW - dt.timedelta(minutes=40))])
    p, _ = make_pool(referee=False)
    p.begin()
    p.fill(NOW)
    assert offered(p) == []
    assert [s.reason for s in p.view.skips if s.item == "referee"] == ["python -m lab referee is not registered yet"]


def test_queue_trouble_calls_the_referee_first_and_it_then_waits(make_pool, lab, clock):
    save_board(lab, [task("O1")], queue=[queue_row("R11", "Running", started=NOW - dt.timedelta(minutes=40)),
                                         queue_row("R12", "Queued", requested=NOW - dt.timedelta(minutes=31))])
    p, fake = make_pool(referee=True, budget=10)
    p.history.ended("referee", "fable", pool.RESULT_OK, NOW - dt.timedelta(minutes=45), {})
    p.fill(NOW)
    fake.wait_calls(2)
    assert started(p) == ["referee", "task O1"]
    assert p.seats["seat-1"].why == ("GPU queue: R11 running 40 min against a cap of 15; R12 queued 31 min, a complaint "
                                     "is due")
    clock.advance(minutes=5)
    settle(p, fake, "referee")
    assert "referee" not in [c.label for c in p.view.candidates]  # the snapshot is older than its end
    clock.advance(minutes=pool.REFEREE_EVERY_MINUTES)
    save_board(lab, [task("O1")], taken_at=clock.now - dt.timedelta(minutes=50))
    p.fill(clock.now)
    assert [(c.label, c.rank) for c in p.view.candidates if c.label == "referee"] == [("referee", 1)]


def test_queue_trouble_calls_the_referee_at_most_every_half_hour(make_pool, lab):
    save_board(lab, [], queue=[queue_row("R11", "Queued", complaint="waited 40 minutes")])
    p, _ = make_pool(referee=True)
    p.begin()
    p.history.ended("referee", "fable", pool.RESULT_OK, NOW - dt.timedelta(minutes=20), {})
    p.fill(NOW)
    assert offered(p) == []


# --- queue trouble is judged by the queue verbs' rules -------------------------------------------

B = "B exclusive"
A = "A teacher shared service"


class QueueWorld:
    """The GPU queue as the slot verbs see it: a fake Notion under the real client and the real verbs
    (lab/machine/slots.py), and the board snapshot the pool reads, taken from the same rows. The verbs
    answer what a sitting would be told; queue_trouble must agree."""

    def __init__(self) -> None:
        self.fake = slots_fake.FakeNotion(clock=lambda: NOW)
        client = slots_notion.NotionClient(slots_fake.TOKEN, transport=self.fake.transport, sleep=lambda seconds: None)
        self.verbs = slots.SlotQueue(client, now=lambda: NOW)
        self.ids: dict[str, str] = {}

    def row(self, task: str, status: str, *, minutes: float | None = 15, waited: float | None = None,
            ran: float | None = None, extended: bool = False, complaint: str = "", gpu: str = B) -> str:
        """A row in the fake Notion: waited minutes since Requested, ran minutes since Started."""
        requested = NOW - dt.timedelta(minutes=waited if waited is not None else (ran or 0) + 5)
        row_id = self.fake.add_row(f"{task}: a short check", task_id=task, gpu=gpu, minutes=minutes, status=status,
                                   requested=requested, started=None if ran is None else NOW - dt.timedelta(minutes=ran),
                                   extended=extended, complaint=complaint)
        self.ids[task] = row_id
        return row_id

    def board(self, *, unrequested: tuple[str, ...] = ()) -> digest.Board:
        """The snapshot of every row so far, as the board file holds the queue; the tasks named in unrequested
        lose their Requested time, as a row made by hand may."""
        rows = [{**self.fake.values(row_id), **({"Requested": None} if task in unrequested else {})}
                for task, row_id in self.ids.items()]
        return digest.parse_board({"schema": 1, "taken_at": digest.iso(NOW), "taken_by": "test", "tasks": [],
                                   "gpu_queue": rows})

    def would_complain(self, task: str) -> bool:
        """True when the complain verb records a complaint on the row now, False when it answers too_early."""
        outcome = self.verbs.complain(self.ids[task])["outcome"]
        assert outcome in ("complaint_recorded", "too_early")
        return outcome == "complaint_recorded"

    def is_overrun(self, task: str, card: str = "B") -> bool:
        """True when the status verb marks the Running row of that card (A or B) overrun."""
        running = {entry["task"]: entry for entry in self.verbs.status(gpu=card)["running"]}
        return running[task]["overrun"]


def test_a_queued_row_is_due_a_complaint_while_a_running_row_on_its_card_is_past_its_cap():
    world = QueueWorld()
    world.row("R11", "Running", ran=20)
    world.row("L3", "Queued", minutes=60, waited=60)  # 60 minutes of the 60 asked for, not yet twice
    board = world.board()
    assert world.is_overrun("R11") and world.would_complain("L3")
    assert pool.queue_trouble(board, NOW) == ["R11 running 20 min against a cap of 15",
                                              "L3 queued 60 min, a complaint is due"]


def test_an_overrun_on_the_other_card_makes_no_complaint_due():
    world = QueueWorld()
    world.row("T1", "Running", ran=20, gpu=A)
    world.row("L3", "Queued", waited=20)
    board = world.board()
    assert world.is_overrun("T1", "A") and world.would_complain("L3") is False
    assert pool.queue_trouble(board, NOW) == ["T1 running 20 min against a cap of 15"]


@pytest.mark.parametrize("waited", [29, 30, 30.5, 45])
def test_a_waiting_row_is_flagged_exactly_when_the_complain_verb_would_record_a_complaint(waited):
    world = QueueWorld()
    world.row("L3", "Queued", minutes=15, waited=waited)
    board = world.board()
    due = world.would_complain("L3")
    assert pool.queue_trouble(board, NOW) == ([f"L3 queued {waited:g} min, a complaint is due"] if due else [])


@pytest.mark.parametrize("ran", [14, 15, 15.5, 20])
def test_a_running_row_is_flagged_exactly_when_the_status_verb_says_it_is_overrun(ran):
    world = QueueWorld()
    world.row("R11", "Running", minutes=15, ran=ran)
    board = world.board()
    overrun = world.is_overrun("R11")
    assert pool.queue_trouble(board, NOW) == ([f"R11 running {ran:g} min against a cap of 15"] if overrun else [])


def test_a_row_that_names_no_minutes_is_judged_by_the_default_slot():
    world = QueueWorld()
    world.row("R11", "Running", minutes=None, ran=20)
    world.row("L3", "Queued", minutes=None, waited=31)
    world.row("L4", "Queued", minutes=None, waited=29)
    board = world.board()
    assert world.is_overrun("R11") and world.would_complain("L3")
    assert pool.queue_trouble(board, NOW) == ["R11 running 20 min against a cap of 15",
                                              "L3 queued 31 min, a complaint is due",
                                              "L4 queued 29 min, a complaint is due"]


def test_one_extension_of_15_minutes_lifts_the_cap():
    world = QueueWorld()
    world.row("R11", "Running", ran=25, extended=True)
    world.row("R12", "Running", ran=31, extended=True, gpu=A)
    board = world.board()
    assert world.is_overrun("R11") is False
    assert pool.queue_trouble(board, NOW) == ["R12 running 31 min against a cap of 30"]


def test_a_queued_row_with_no_requested_time_is_due_a_complaint_only_while_an_overrun_runs():
    overrun = QueueWorld()
    overrun.row("R11", "Running", ran=20)
    overrun.row("L3", "Queued")
    assert pool.queue_trouble(overrun.board(unrequested=("L3",)), NOW) == [
        "R11 running 20 min against a cap of 15", "L3 queued, a complaint is due"]
    quiet = QueueWorld()
    quiet.row("R11", "Running", ran=10)
    quiet.row("L3", "Queued")
    assert pool.queue_trouble(quiet.board(unrequested=("L3",)), NOW) == []


def test_a_row_that_already_complained_is_listed_by_its_complaint_alone():
    world = QueueWorld()
    world.row("R11", "Running", ran=20)
    world.row("L3", "Queued", waited=60, complaint="waited 60 minutes behind R11")
    assert pool.queue_trouble(world.board(), NOW) == ["R11 running 20 min against a cap of 15", "a complaint on L3"]


def test_rows_that_are_not_open_make_no_trouble():
    world = QueueWorld()
    world.row("R11", "Done", ran=90)
    world.row("R12", "Cancelled", waited=300)
    world.row("R13", "Ended by referee", ran=90, complaint="ended after 90 minutes")
    assert pool.queue_trouble(world.board(), NOW) == []


# --- the dry run --------------------------------------------------------------------------------


def test_the_dry_run_prints_the_schedule_and_starts_and_writes_nothing(lab, refine_root):
    write_track(refine_root, "plumbing", counter=0)
    board = save_board(lab, [task("F1", tier="Fable", wave=1), task("O1", wave=2), task("S1", tier="Sonnet", wave=3),
                             task("O2", wave=4), task("O3", wave=5), task("O4", wave=6), task("S2", tier="Sonnet", wave=9),
                             pass_task("O39", "plumbing", wave=7)])
    settings = pool.Settings(state_dir=lab, refine=pool.TreeFiles(refine_root), verbs=lambda: ["digest", "sit"])
    answer = json.loads(json.dumps(pool.dry_run(settings, NOW)))
    assert (answer["ok"], answer["outcome"], answer["budget"], answer["units_in_use"]) == (True, "dry_run", 16, 0)
    assert [(e["decision"], e["item"], e["units"]) for e in answer["schedule"]] == [
        ("start", "task F1", 8), ("start", "task O1", 2), ("start", "task S1", 1), ("start", "task O2", 2),
        ("start", "task O3", 2), ("hold", "task O4", 2), ("after", "pass plumbing", 2), ("after", "task S2", 1)]
    assert answer["units_to_start"] == 15
    assert answer["schedule"][0]["command"] == "python -m lab sit --tier fable --task F1"
    assert answer["schedule"][6]["command"] == "python -m lab sit --tier opus --pass plumbing"
    assert ledger.read(state_dir=lab) == []
    assert not pool.state_path(lab).exists()
    assert sorted(p.name for p in lab.iterdir()) == [board.name]


# --- the default launcher -----------------------------------------------------------------------

FAKE_SIT = r'''
import json, os, sys
from pathlib import Path
Path(os.environ["FAKE_SIT_RECORD"]).write_text(json.dumps({"argv": sys.argv[1:], "cwd": os.getcwd(),
    "state": os.environ.get("LAB_STATE_DIR")}), encoding="utf-8")
print("a log line that is not the answer")
if os.environ.get("FAKE_SIT_ANSWER"):
    print(os.environ["FAKE_SIT_ANSWER"])
sys.exit(int(os.environ.get("FAKE_SIT_EXIT", "0")))
'''


def test_the_default_launcher_runs_sit_as_a_child_with_the_pool_state_dir_and_reads_its_answer(tmp_path, lab, monkeypatch):
    script = tmp_path / "fake_sit.py"
    script.write_text(FAKE_SIT, encoding="utf-8")
    record = tmp_path / "record.json"
    monkeypatch.setenv("FAKE_SIT_RECORD", str(record))
    monkeypatch.setenv("FAKE_SIT_ANSWER", json.dumps(ran(sitting="sit-child")))
    launcher = pool.SitProcess(lab, head=[sys.executable, str(script)], cwd=tmp_path)
    answer = REAL_SIT_PROCESS_CALL(launcher, "opus", sit.Item("task", "O5"), minutes=30, budget_usd=2.5)
    assert answer == ran(sitting="sit-child")
    seen = json.loads(record.read_text(encoding="utf-8"))
    assert seen["argv"] == ["--tier", "opus", "--task", "O5", "--minutes", "30", "--budget-usd", "2.5"]
    assert Path(seen["state"]) == lab.resolve()
    assert pool.SitProcess(lab).argv("fable", sit.Item("merge")) == [sys.executable, "-m", "lab", "sit", "--tier",
                                                                       "fable", "--merge"]


def test_a_child_that_prints_no_answer_is_read_as_a_refusal(tmp_path, lab, monkeypatch):
    script = tmp_path / "fake_sit.py"
    script.write_text(FAKE_SIT, encoding="utf-8")
    monkeypatch.setenv("FAKE_SIT_RECORD", str(tmp_path / "record.json"))
    monkeypatch.delenv("FAKE_SIT_ANSWER", raising=False)
    monkeypatch.setenv("FAKE_SIT_EXIT", "3")
    answer = REAL_SIT_PROCESS_CALL(pool.SitProcess(lab, head=[sys.executable, str(script)], cwd=tmp_path), "sonnet",
                                   sit.Item("task", "S1"))
    assert answer == {"ok": False, "outcome": "error",
                      "error": "python -m lab sit exited 3 without an answer: a log line that is not the answer"}
    assert pool.classify(answer) == pool.RESULT_REFUSED


# --- reading the refinement files at a git ref ------------------------------------------------------


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_git_files_read_the_refinement_files_at_the_ref_not_the_working_tree(tmp_path):
    repo = tmp_path / "repo"
    counter = repo / "lab" / "refine" / "telemetry" / "refine_pass.txt"
    counter.parent.mkdir(parents=True)
    counter.write_bytes(b"3\n")
    git(repo, "init", "--quiet")
    git(repo, "add", "-A")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false", "commit", "--quiet",
        "-m", "counter")
    counter.write_text("5\n", encoding="utf-8")
    files = pool.GitFiles("HEAD", repo=repo, fetch_minutes=None)
    assert files.refresh() == []
    assert files.read("telemetry/refine_pass.txt") == "3\n"
    assert files.read("telemetry/REFINE_LOG.md") is None
    missing = pool.GitFiles("no-such-ref", repo=repo, fetch_minutes=None)
    assert missing.refresh() == ["git cannot resolve no-such-ref; no pass or merge is offered"]
    state = pool.read_refine(missing)
    assert state.problems[0] == "git cannot resolve no-such-ref; no pass or merge is offered"
    assert all(track.problem for track in state.tracks.values())


def test_the_quiet_refinement_files_offer_nothing_and_the_repo_files_parse(refine_root):
    quiet = pool.read_refine(pool.TreeFiles(refine_root))
    assert quiet.problems == []
    assert [(t.track, t.counter, t.run_open, t.capped, t.unreconciled) for t in quiet.tracks.values()] == [
        (track, 1, False, False, 0) for track in TRACKS]
    assert quiet.merges == 1
    repo = pool.read_refine(pool.TreeFiles())
    assert [t.problem for t in repo.tracks.values()] == [None, None, None, None]
    assert repo.merges is not None


def test_the_pool_verb_is_registered_and_owner_only():
    assert lab_cli.VERBS["pool"] == "lab.machine.pool:main"
    assert stop.is_exempt(pool.main)
    from lab.machine import allowed
    for tier in allowed.TIERS:
        assert not allowed.allows(tier, "Bash", "python -m lab pool")
        assert not allowed.allows(tier, "Bash", "python -m lab pool status")


# --- interrupts, broken launchers, waits that end ----------------------------------------------------


class InterruptOnce(pool.Pool):
    """A pool whose main loop gets one Ctrl+C once a seat runs, the way a console delivers it."""

    raised = False

    def tick(self, timeout):
        if self.seats and not self.raised:
            self.raised = True
            raise KeyboardInterrupt
        return super().tick(timeout)


def test_ctrl_c_drains_the_pool_starting_nothing_new_until_its_seats_end(lab, refine_root, clock):
    save_board(lab, [task("O1"), task("O2", wave=2)])
    fake = FakeLauncher()
    conf = pool.Settings(state_dir=lab, budget=2, refine=pool.TreeFiles(refine_root), verbs=lambda: ["digest", "sit"])
    p = InterruptOnce(conf, launcher=fake, tick_seconds=0.01, clock=clock)
    ender = threading.Thread(target=lambda: (fake.wait_calls(1), time.sleep(0.2), fake.finish("task O1")))
    ender.start()
    try:
        answer = p.run()
    finally:
        ender.join(10)
        fake.release_all()
    assert (answer["ok"], answer["outcome"], answer["started"], answer["seats_left"]) == (False, "interrupted", 1, [])
    assert fake.labels() == ["task O1"]
    drain = [row for row in rows(lab) if row["verb"] == pool.V_DRAIN]
    assert [(row["outcome"], row["seats"]) for row in drain] == [("draining", 1)]
    assert rows(lab)[-1]["verb"] == pool.V_EXIT and rows(lab)[-1]["outcome"] == "interrupted"


def test_a_launcher_that_raises_still_ends_its_seat(make_pool, lab):
    def broken(tier, item, **kwargs):
        raise KeyboardInterrupt

    save_board(lab, [task("O1")])
    p, _ = make_pool(launcher=None)
    p.launcher = broken
    p.begin()
    p.fill(NOW)
    p.tick(10)
    end = [row for row in rows(lab) if row["verb"] == pool.V_END]
    assert [(row["item"], row["sitting_outcome"], row["result"], row["error"]) for row in end] == [
        ("task O1", "error", "refused", "the launcher failed: KeyboardInterrupt:")]
    assert p.seats == {}


def test_a_pass_whose_pull_request_never_merged_is_offered_again_after_the_wait(make_pool, lab, refine_root, clock):
    write_track(refine_root, "telemetry", counter=1, ending="", closed=False)
    save_board(lab, [])
    p, fake = make_pool()
    p.begin()
    p.history.ended("pass telemetry", "opus", pool.RESULT_OK, NOW - dt.timedelta(hours=5, minutes=59), {"counter": 1})
    p.fill(NOW)
    assert offered(p) == []
    clock.advance(minutes=2)
    save_board(lab, [], taken_at=clock.now)
    p.fill(clock.now)
    assert offered(p) == [("opus", "pass telemetry")]


def test_a_pause_is_written_once_while_it_lasts(make_pool, lab):
    save_board(lab, [task(f"S{n}", tier="Sonnet", wave=n) for n in range(1, 4)])
    p, fake = make_pool()
    p.begin()
    p.fill(NOW)
    fake.wait_calls(3)
    for n in range(1, 4):
        fake.finish(f"task S{n}", refused("no_workspace"))
    deadline = time.monotonic() + 10
    while p.ended.qsize() < 3 and time.monotonic() < deadline:
        time.sleep(0.005)
    p.tick(10)
    assert p.seats == {}
    assert [row["until"] for row in rows(lab) if row["verb"] == pool.V_PAUSE] == ["2026-10-10T15:10:00Z"]


def test_a_first_pass_or_the_first_merge_never_starts_without_its_board_task(make_pool, lab, refine_root):
    write_track(refine_root, "kernels", counter=0)
    write_merge_log(refine_root, {})
    save_board(lab, [])
    p, _ = make_pool()
    p.begin()
    p.fill(NOW)
    assert offered(p) == []
    reasons = {s.item: s.reason for s in p.view.skips}
    assert reasons["pass kernels"] == ("a track's first pass runs as its board task, and the snapshot has no open "
                                       "first-pass task for it")
    assert reasons["merge"] == "the first merge runs as its board task, and the snapshot has no open merge task"


def test_the_title_recognisers_are_built_from_the_refine_merge_track_list():
    from lab.machine import refine_merge

    source = Path(pool.__file__).read_text(encoding="utf-8")
    assert "telemetry|learning" not in source, (
        "pool.py types the track alternation into its regexes; build it from refine_merge.TRACKS")
    for name, regex in (("_PASS_TITLE", pool._PASS_TITLE), ("_LOOP_TITLE", pool._LOOP_TITLE)):
        missing = [track for track in refine_merge.TRACKS if track not in regex.pattern]
        assert missing == [], f"{name} does not name every refine_merge.TRACKS entry, missing {missing}"


class FailingFetchFiles:
    """A RefineFiles stand-in for a ref whose git fetch fails: refresh() reports it, nothing can be read."""

    where = "origin/main"

    def __init__(self, problem: str) -> None:
        self.problem = problem

    def refresh(self) -> list[str]:
        return [self.problem]

    def read(self, rel: str) -> str | None:
        return None


def test_a_failed_refine_fetch_is_written_to_pool_json_and_answered_by_status(lab, clock):
    fetch_failure = "git fetch origin failed: fatal: unable to access"
    save_board(lab, [task("O1")])
    conf = pool.Settings(state_dir=lab, budget=16, refine=FailingFetchFiles(fetch_failure),
                         verbs=lambda: ["digest", "sit"])
    fake = FakeLauncher()
    p = pool.Pool(conf, launcher=fake, tick_seconds=0.01, exit_when_idle=False, clock=clock)
    try:
        assert p.begin() is None
        p.fill(NOW)
        fake.wait_calls(1)
        saved = json.loads(pool.state_path(lab).read_text(encoding="utf-8"))
        answer = pool.status(lab, NOW + dt.timedelta(minutes=1))
    finally:
        fake.release_all()
        p.lock.release()
    assert f"refine: {fetch_failure}" in saved.get("problems", [])
    assert (answer["ok"], answer["outcome"], answer["running"]) == (True, "running", True)
    assert f"refine: {fetch_failure}" in (answer.get("problems") or [])


# --- a restarted pool adopts the outcomes its predecessor never recorded ---------------------------


def test_a_restarted_pool_adopts_the_outcome_of_a_sitting_its_predecessor_never_recorded(make_pool, lab):
    started_at = NOW - dt.timedelta(hours=2)
    # an earlier pool: it began, started a seat for task T1, and died before it wrote the pool end row
    ledger.append({"ts": started_at - dt.timedelta(minutes=1), "actor": pool.ACTOR, "tier": None, "task": None,
                   "verb": pool.V_BEGIN, "outcome": "started", "metrics": {}, "refs": {}}, state_dir=lab)
    ledger.append({"ts": started_at, "actor": pool.ACTOR, "tier": "opus", "task": "T1", "verb": pool.V_START,
                   "outcome": "started", "metrics": {}, "refs": {}, "item": "task T1", "seat": "seat-1",
                   "units_held": 2, "source": "board", "why": "test", "progress": {}}, state_dir=lab)
    # its sitting ran on its own and ended in error
    sit_row(lab, "sit-old", "opus", "task T1", ts=started_at + dt.timedelta(seconds=30))
    sit_row(lab, "sit-old", "opus", "task T1", outcome="error", ts=started_at + dt.timedelta(minutes=40))
    assert [row["verb"] for row in rows(lab) if row["verb"] == pool.V_END] == []

    p, _ = make_pool()
    assert p.begin() is None

    record = p.history.get("task T1")
    assert record is not None
    assert (record.last_result, record.failures) == ("failed", 1)
    ends = [row for row in rows(lab) if row["verb"] == pool.V_END]
    assert [(row["item"], row["result"], row["refs"].get("run_id")) for row in ends] == [
        ("task T1", "failed", "sit-old")]


# --- the breaker belongs to the history, not to one caller -----------------------------------------


def test_an_outage_that_trips_the_breaker_every_round_benches_no_item(make_pool, lab, clock):
    """Five failed sittings in a row, whatever the items, are the pool's problem and not the items':
    an outage that keeps tripping the breaker must not bench the items it failed."""
    save_board(lab, [task(f"S{n}", tier="Sonnet", wave=n) for n in range(1, 7)], taken_at=clock.now)
    p, fake = make_pool(FakeLauncher(default=ran("error")), referee=True)
    assert p.begin() is None
    for outage_round in range(3):
        if outage_round:
            clock.advance(minutes=pool.PAUSE_MINUTES + 1)
            save_board(lab, [task(f"S{n}", tier="Sonnet", wave=n) for n in range(1, 7)], taken_at=clock.now)
        deadline = time.monotonic() + 20
        p.tick(1)
        while p.seats and time.monotonic() < deadline:
            p.tick(0.2)
        assert p.seats == {}, f"round {outage_round + 1}: the seats did not all end"
        pauses = [row for row in rows(lab) if row["verb"] == pool.V_PAUSE]
        assert len(pauses) == outage_round + 1, f"round {outage_round + 1}: expected one breaker pause per round"
        assert pauses[-1]["reason"] == f"{pool.BREAKER_FAILURES} sittings in a row failed"
    benched = [(row["item"], row["failures"]) for row in rows(lab) if row["verb"] == pool.V_BENCH]
    assert benched == []
    saved = json.loads(pool.state_path(lab).read_text(encoding="utf-8"))
    assert saved["benched"] == []
    worst = {label: record.failures for label, record in p.history.items.items()}
    assert all(count < pool.MAX_RESTARTS + 1 for count in worst.values()), worst


def test_replaying_the_end_rows_reproduces_the_breaker_the_live_history_applies(lab):
    """Seven failed sittings of seven different items, oldest first: the fifth trips the breaker, so
    the streak starts over and the items that failed before it are cleared. The replay of the pool's
    end rows and a live history fed the same endings must agree on that."""
    labels = [f"task S{n}" for n in range(1, 8)]
    stamps = [NOW + dt.timedelta(minutes=n) for n in range(7)]
    end_rows = [{"verb": pool.V_END, "ts": pool.iso(when), "item": label, "tier": "sonnet",
                 "result": pool.RESULT_FAILED, "progress": {}} for label, when in zip(labels, stamps)]
    replayed = pool.History.from_rows(end_rows)
    live = pool.History()
    for label, when in zip(labels, stamps):
        live.ended(label, "sonnet", pool.RESULT_FAILED, when, {})
    for name, history in (("replayed", replayed), ("live", live)):
        assert history.streak == 2, name
        assert [history.get(label).failures for label in labels[:5]] == [0, 0, 0, 0, 0], name
    assert {label: replayed.get(label).failures for label in labels} == {
        label: live.get(label).failures for label in labels}


# --- a sitting with a long cap holds its units for its whole cap -----------------------------------


def test_a_sitting_with_a_cap_over_a_day_still_holds_its_units_and_its_item_is_not_offered(lab, refine_root):
    long_cap_minutes = 1600  # 26.7 h: still running 25 h after its start row, which a 24 h read of the ledger misses
    sit_row(lab, "sit-long-1", "sonnet", "task S1", ts=NOW - dt.timedelta(hours=25), cap=long_cap_minutes)
    save_board(lab, [task("S1", tier="Sonnet", wave=1)])

    answer = pool.status(lab, NOW)
    assert answer["units_in_use"] == 1
    assert [(f["sitting"], f["tier"], f["item"], f["units"]) for f in answer["foreign"]] == [
        ("sit-long-1", "sonnet", "task S1", 1)]

    settings = pool.Settings(state_dir=lab, refine=pool.TreeFiles(refine_root), verbs=lambda: ["digest", "sit"])
    dry = pool.dry_run(settings, NOW)
    assert dry["units_in_use"] == 1
    assert [f["sitting"] for f in dry["foreign"]] == ["sit-long-1"]
    assert [entry["item"] for entry in dry["schedule"]] == []
