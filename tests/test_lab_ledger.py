"""The lab ledger and its file lock: the row schema, append and read, the
torn-tail rule, latest_metric, the lock's timeout and its exclusion across
handles and processes, and two processes appending at once. Every test
works in a temporary state dir; nothing here touches lab/state."""
from __future__ import annotations

import datetime as dt
import json
import multiprocessing
import os
import subprocess
import time
from pathlib import Path

import pytest

from lab.machine import ledger
from lab.machine.lock import FileLock, LockTimeout

REPO_ROOT = Path(__file__).resolve().parents[1]
SPAWN = multiprocessing.get_context("spawn")
JOIN_TIMEOUT = 300.0     # seconds; a spawned child still running after this has hung
START_TIMEOUT = 60.0     # seconds a spawned child waits for its partner to start
ROWS_PER_CHILD = 1000
UTC = dt.timezone.utc


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """The default state dir of every test is a temporary one, so a call
    that leaves out state_dir still never writes the real lab/state."""
    state = tmp_path / "state"
    monkeypatch.setenv("LAB_STATE_DIR", str(state))
    return state


def make_row(**changes):
    row = {"ts": "2026-10-10T10:20:00Z", "actor": "opus-1", "tier": "opus", "task": "L3",
           "verb": "bench.run", "outcome": "ok",
           "metrics": {"teacher.decode_tok_s": 7.7, "teacher.watts_avg": 202},
           "refs": {"pr": 37, "commit": "bb351e6", "slot": "gpu-b-0001", "run_id": "run-0001"}}
    row.update(changes)
    return row


def seqs(rows):
    return [r["seq"] for r in rows]


def free(path) -> bool:
    """Whether nothing holds the lock at path, leaving it as found."""
    probe = FileLock(path)
    got = probe.acquire(timeout=0)
    probe.release()
    return got


# --- the row schema -------------------------------------------------------------------

def test_append_writes_one_json_line_per_row_and_returns_its_seq(isolated_state):
    first = make_row()
    assert ledger.append(first) == 1
    assert "seq" not in first, "the caller's dict is not changed"
    path = isolated_state / "ledger.jsonl"
    written = path.read_bytes()
    assert written.endswith(b"\n") and written.count(b"\n") == 1
    stored = json.loads(written)
    assert stored == {"seq": 1, **first}
    assert list(stored)[:9] == ["seq", "ts", "actor", "tier", "task", "verb", "outcome", "metrics", "refs"]
    assert ledger.append(make_row(verb="bench.done", metrics={})) == 2
    after = path.read_bytes()
    assert after.startswith(written), "an append never rewrites an earlier line"
    assert seqs(json.loads(line) for line in after.splitlines()) == [1, 2]


def test_append_stamps_ts_in_utc_when_the_row_has_none():
    row = make_row()
    del row["ts"]
    before = dt.datetime.now(UTC).replace(microsecond=0)
    assert ledger.append(row) == 1
    assert ledger.append(make_row(ts=None)) == 2
    after = dt.datetime.now(UTC)
    for stored in ledger.read():
        assert stored["ts"].endswith("Z")
        stamp = dt.datetime.fromisoformat(stored["ts"])
        assert stamp.utcoffset() == dt.timedelta(0)
        assert before <= stamp <= after


def test_an_aware_datetime_ts_is_stored_as_utc_iso_and_a_naive_one_is_refused():
    eastern = dt.timezone(dt.timedelta(hours=-4))
    assert ledger.append(make_row(ts=dt.datetime(2026, 10, 10, 6, 20, 5, tzinfo=eastern))) == 1
    assert ledger.read()[0]["ts"] == "2026-10-10T10:20:05Z"
    with pytest.raises(ledger.RowError) as err:
        ledger.append(make_row(ts=dt.datetime(2026, 10, 10, 6, 20, 5)))
    assert "ts" in str(err.value)
    assert seqs(ledger.read()) == [1]


@pytest.mark.parametrize("key", ["actor", "tier", "task", "verb", "outcome", "metrics", "refs"])
def test_a_row_missing_a_required_key_is_refused_and_nothing_is_written(key, isolated_state):
    row = make_row()
    del row[key]
    with pytest.raises(ledger.RowError) as err:
        ledger.append(row)
    assert isinstance(err.value, ValueError)
    assert err.value.problems == [f"{key} is missing"]
    assert not (isolated_state / "ledger.jsonl").exists()


def test_every_problem_in_a_row_is_reported_together():
    row = make_row(ts="yesterday", outcome="", metrics={"teacher.decode_tok_s": "7.7"})
    del row["actor"]
    with pytest.raises(ledger.RowError) as err:
        ledger.append(row)
    problems = err.value.problems
    assert len(problems) == 4, problems
    assert [p.split()[0] for p in problems] == ["ts", "actor", "outcome", "metrics"], "in field order"
    assert problems[1] == "actor is missing"
    assert "teacher.decode_tok_s" in problems[3]
    assert ledger.read() == []


@pytest.mark.parametrize("ts", ["yesterday", "2026-10-10T10:20:00", "2026-13-40T10:20:00Z",
                                "10/10/2026 10:20", 1728555600, ["2026-10-10T10:20:00Z"]])
def test_ts_must_be_iso_8601_with_a_utc_offset(ts):
    with pytest.raises(ledger.RowError) as err:
        ledger.append(make_row(ts=ts))
    assert [p.split()[0] for p in err.value.problems] == ["ts"]


@pytest.mark.parametrize("ts", ["2026-10-10T10:20:00Z", "2026-10-10T06:20:00-04:00",
                                "2026-10-10T10:20:00.123456+00:00"])
def test_an_iso_ts_with_an_offset_is_kept_as_written(ts):
    assert ledger.append(make_row(ts=ts)) == 1
    assert ledger.read()[0]["ts"] == ts


@pytest.mark.parametrize("metrics", [{"x": "7.7"}, {"x": True}, {"x": None}, {"x": float("nan")},
                                     {"x": float("inf")}, {"": 1.0}, {3: 1.0}, [("x", 1.0)], "x=1"])
def test_metrics_must_map_names_to_finite_numbers(metrics):
    with pytest.raises(ledger.RowError) as err:
        ledger.append(make_row(metrics=metrics))
    assert [p.split()[0] for p in err.value.problems] == ["metrics"]


def test_metrics_take_ints_floats_and_may_be_empty():
    for metrics in ({"x": 7}, {"x": 7.7, "y": -0.5}, {}):
        ledger.append(make_row(metrics=metrics))
    assert [r["metrics"] for r in ledger.read()] == [{"x": 7}, {"x": 7.7, "y": -0.5}, {}]


def test_refs_are_stored_with_all_four_keys():
    ledger.append(make_row(refs={"pr": 37}))
    ledger.append(make_row(refs={}))
    ledger.append(make_row(refs={"run_id": "run-7", "queue_row": "abc"}))
    assert [r["refs"] for r in ledger.read()] == [
        {"pr": 37, "commit": None, "slot": None, "run_id": None},
        {"pr": None, "commit": None, "slot": None, "run_id": None},
        {"pr": None, "commit": None, "slot": None, "run_id": "run-7", "queue_row": "abc"},
    ]


@pytest.mark.parametrize("refs", ["#37", ["pr", 37], {"pr": [37]}, {"pr": True}, {"commit": 1.5}, {"": "x"}])
def test_refs_must_be_a_dict_of_ids(refs):
    with pytest.raises(ledger.RowError) as err:
        ledger.append(make_row(refs=refs))
    assert [p.split()[0] for p in err.value.problems] == ["refs"]


def test_a_row_may_not_bring_its_own_seq():
    with pytest.raises(ledger.RowError) as err:
        ledger.append(make_row(seq=99))
    assert [p.split()[0] for p in err.value.problems] == ["seq"]
    assert ledger.read() == []


@pytest.mark.parametrize("key,value", [("actor", ""), ("actor", None), ("verb", "  "), ("outcome", 3),
                                       ("tier", ""), ("task", 15)])
def test_identity_fields_must_be_non_empty_strings(key, value):
    with pytest.raises(ledger.RowError) as err:
        ledger.append(make_row(**{key: value}))
    assert [p.split()[0] for p in err.value.problems] == [key]


def test_tier_and_task_may_be_none_for_rows_outside_a_sitting_or_a_task():
    assert ledger.append(make_row(tier=None, task=None)) == 1
    stored = ledger.read()[0]
    assert stored["tier"] is None and stored["task"] is None


def test_optional_fields_are_checked_and_kept_with_any_extra_field():
    row = make_row(node="pod-a", units={"teacher.decode_tok_s": "tok/s"}, conditions={"batch": 1, "ctx": 16384},
                   low_confidence=True, source="PR #37", note="first run", gpu="r9700")
    assert ledger.append(row) == 1
    assert ledger.read() == [{"seq": 1, **row}]


@pytest.mark.parametrize("changes", [{"node": ""}, {"node": 3}, {"units": {"other.metric": "W"}},
                                     {"units": {"teacher.decode_tok_s": 1}}, {"units": ["tok/s"]},
                                     {"conditions": ["batch", 1]}, {"low_confidence": "yes"},
                                     {"source": 37}, {"note": ["a"]}])
def test_optional_fields_of_the_wrong_shape_are_refused(changes):
    with pytest.raises(ledger.RowError) as err:
        ledger.append(make_row(**changes))
    assert [p.split()[0] for p in err.value.problems] == list(changes)


def test_extra_fields_must_be_json_with_string_names():
    with pytest.raises(ledger.RowError) as err:
        ledger.append(make_row(when=dt.date(2026, 10, 10)))
    assert [p.split()[0] for p in err.value.problems] == ["when"]
    with pytest.raises(ledger.RowError):
        ledger.append({**make_row(), 3: "x"})
    with pytest.raises(TypeError):
        ledger.append([("actor", "x")])
    assert ledger.read() == []


def test_a_row_may_carry_its_own_metric_field_but_not_a_read_filter_name():
    experiment = make_row(verb="experiment.add", metrics={}, metric="teacher.decode_tok_s", expected_delta=0.2)
    assert ledger.append(experiment) == 1
    assert ledger.append(make_row(metrics={"teacher.decode_tok_s": 8.1})) == 2
    assert seqs(ledger.read(metric="teacher.decode_tok_s")) == [1], "metric is a field like any other"
    assert seqs(ledger.read(has_metric="teacher.decode_tok_s")) == [2]
    for name in ("has_metric", "since", "before", "state_dir"):
        with pytest.raises(ledger.RowError) as err:
            ledger.append(make_row(**{name: "x"}))
        assert [p.split()[0] for p in err.value.problems] == [name]
    assert seqs(ledger.read()) == [1, 2]


# --- reading --------------------------------------------------------------------------

def test_read_on_a_missing_ledger_is_empty_and_creates_nothing(isolated_state):
    assert ledger.read() == []
    assert ledger.tail(5) == []
    assert ledger.latest_metric("teacher.decode_tok_s") is None
    assert not isolated_state.exists()


def test_read_filters_by_value_dict_subset_predicate_metric_and_time():
    rows = [make_row(ts="2026-10-10T10:00:00Z", task="L3", outcome="ok", refs={"run_id": "run-1"}, metrics={"a": 1}),
            make_row(ts="2026-10-10T11:00:00Z", task="L4", outcome="fail", refs={"run_id": "run-2"}, metrics={"b": 2}),
            make_row(ts="2026-10-10T12:00:00Z", task="L3", outcome="ok", refs={"run_id": "run-2"}, metrics={"a": 3})]
    for row in rows:
        ledger.append(row)
    assert seqs(ledger.read()) == [1, 2, 3]
    assert ledger.read()[0] == {"seq": 1, **rows[0], "refs": {"pr": None, "commit": None, "slot": None, "run_id": "run-1"}}
    assert seqs(ledger.read(task="L3")) == [1, 3]
    assert seqs(ledger.read(refs={"run_id": "run-2"})) == [2, 3]
    assert seqs(ledger.read(outcome=lambda o: o != "ok")) == [2]
    assert seqs(ledger.read(has_metric="a")) == [1, 3]
    assert seqs(ledger.read(since="2026-10-10T11:00:00Z")) == [2, 3]
    assert seqs(ledger.read(before="2026-10-10T07:00:00-04:00")) == [1], "before is exclusive, compared as instants"
    assert seqs(ledger.read(task="L3", since=dt.datetime(2026, 10, 10, 11, tzinfo=UTC))) == [3]
    assert seqs(ledger.read(node=None)) == [1, 2, 3], "a missing field reads as None"
    assert ledger.read(task="L9") == []
    with pytest.raises(ValueError):
        ledger.read(since="2026-10-10T11:00:00")


def test_tail_returns_the_last_n_rows_in_ledger_order():
    for i in range(5):
        ledger.append(make_row(metrics={"i": i}))
    assert [r["metrics"]["i"] for r in ledger.tail(2)] == [3, 4]
    assert seqs(ledger.tail(10)) == [1, 2, 3, 4, 5]
    assert ledger.tail(5) == ledger.read()
    assert ledger.tail(0) == []


def test_tail_and_seq_work_across_rows_longer_than_a_read_block():
    long_note = "x" * 150_000
    for i in range(3):
        ledger.append(make_row(note=long_note, metrics={"i": i}))
    assert ledger.append(make_row(metrics={"i": 3})) == 4
    tail = ledger.tail(3)
    assert [r["metrics"]["i"] for r in tail] == [1, 2, 3]
    assert tail[0]["note"] == long_note


def test_latest_metric_is_the_newest_measurement_by_time_then_seq_with_the_metrics_row_shape():
    ledger.append(make_row(ts="2026-10-10T10:00:00Z", node="pod-a", metrics={"m": 1.0}))
    ledger.append(make_row(ts="2026-10-10T12:00:00Z", node="pod-b", metrics={"m": 2.0}))
    ledger.append(make_row(ts="2026-10-10T11:00:00Z", node="pod-a", metrics={"m": 3.0, "n": 4.0},
                           low_confidence=True))
    ledger.append(make_row(ts="2026-10-10T09:00:00Z", node="pod-c", metrics={"m": 9.0},
                           units={"m": "tok/s"}, conditions={"batch": 1}, source="PR #37"))
    assert ledger.latest_metric("m") == {
        "metric": "m", "value": 2.0, "unit": None, "node": "pod-b", "conditions": {},
        "source": "ledger seq 2", "ts": "2026-10-10T12:00:00Z", "low_confidence": False, "seq": 2}
    assert ledger.latest_metric("m", node="pod-a") == {
        "metric": "m", "value": 3.0, "unit": None, "node": "pod-a", "conditions": {},
        "source": "ledger seq 3", "ts": "2026-10-10T11:00:00Z", "low_confidence": True, "seq": 3}
    assert ledger.latest_metric("m", node="pod-c") == {
        "metric": "m", "value": 9.0, "unit": "tok/s", "node": "pod-c", "conditions": {"batch": 1},
        "source": "PR #37", "ts": "2026-10-10T09:00:00Z", "low_confidence": False, "seq": 4}
    assert ledger.latest_metric("m", node="jetson") is None
    assert ledger.latest_metric("absent") is None
    # instants, not strings: 08:30 at -04:00 is 12:30Z, later than 12:00Z
    ledger.append(make_row(ts="2026-10-10T08:30:00-04:00", node="pod-b", metrics={"m": 5.0}))
    assert ledger.latest_metric("m")["value"] == 5.0
    # the same instant twice: the later seq wins
    ledger.append(make_row(ts="2026-10-10T12:30:00Z", node="pod-b", metrics={"m": 6.0}))
    assert (ledger.latest_metric("m")["value"], ledger.latest_metric("m")["seq"]) == (6.0, 6)


def test_latest_metric_narrows_by_conditions_as_the_metrics_file_does():
    ledger.append(make_row(ts="2026-10-10T10:00:00Z", node="pod-a", metrics={"m": 1.0},
                           conditions={"prompt": "text", "batch": 1}))
    ledger.append(make_row(ts="2026-10-10T11:00:00Z", node="pod-a", metrics={"m": 2.0},
                           conditions={"prompt": "image", "batch": 1}))
    ledger.append(make_row(ts="2026-10-10T12:00:00Z", node="pod-a", metrics={"m": 3.0}))
    assert ledger.latest_metric("m")["value"] == 3.0
    assert ledger.latest_metric("m", conditions={"prompt": "text"})["value"] == 1.0
    assert ledger.latest_metric("m", "pod-a", conditions={"batch": 1})["value"] == 2.0
    assert ledger.latest_metric("m", conditions={"batch": None}) is None, "a key the row lacks never matches"
    assert ledger.latest_metric("m", conditions={"prompt": "audio"}) is None


# --- crashes mid-write ----------------------------------------------------------------

def test_a_torn_last_line_is_skipped_and_the_next_append_starts_a_new_line(isolated_state):
    assert ledger.append(make_row()) == 1
    path = isolated_state / "ledger.jsonl"
    fragment = b'{"seq": 2, "ts": "2026-10-10T10:2'
    with open(path, "ab") as fh:
        fh.write(fragment)          # a write cut short by a crash
    assert seqs(ledger.read()) == [1]
    assert seqs(ledger.tail(5)) == [1]
    assert ledger.append(make_row()) == 2
    raw = path.read_bytes().split(b"\n")
    assert len(raw) == 4 and raw[1] == fragment and raw[3] == b""
    assert seqs(ledger.read()) == [1, 2]
    assert seqs(ledger.tail(5)) == [1, 2]


def test_a_complete_row_that_lost_only_its_newline_still_counts(isolated_state):
    assert ledger.append(make_row()) == 1
    path = isolated_state / "ledger.jsonl"
    with open(path, "ab") as fh:
        fh.write(json.dumps({"seq": 2, **make_row()}).encode("utf-8"))   # all but the newline
    assert seqs(ledger.read()) == [1, 2]
    assert ledger.append(make_row()) == 3, "a seq already on disk is never handed out again"
    assert seqs(ledger.read()) == [1, 2, 3]
    assert path.read_bytes().count(b"\n") == 3


# --- where the ledger lives -----------------------------------------------------------

def test_the_ledger_lives_in_lab_state_unless_lab_state_dir_says_otherwise(tmp_path, monkeypatch):
    monkeypatch.delenv("LAB_STATE_DIR", raising=False)
    assert ledger.ledger_path() == REPO_ROOT / "lab" / "state" / "ledger.jsonl"
    monkeypatch.setenv("LAB_STATE_DIR", "")
    assert ledger.ledger_path() == REPO_ROOT / "lab" / "state" / "ledger.jsonl"
    shared = tmp_path / "shared-state"
    monkeypatch.setenv("LAB_STATE_DIR", str(shared))
    assert ledger.ledger_path() == shared / "ledger.jsonl"
    assert ledger.append(make_row()) == 1
    assert (shared / "ledger.jsonl").is_file()
    explicit = tmp_path / "explicit"
    assert ledger.append(make_row(), state_dir=explicit) == 1
    assert seqs(ledger.read(state_dir=explicit)) == [1]
    assert seqs(ledger.read()) == [1]


def test_lab_state_is_ignored_by_git_except_its_gitkeep():
    assert (REPO_ROOT / "lab" / "state" / ".gitkeep").is_file()

    def ignored(rel: str) -> bool:
        res = subprocess.run(["git", "check-ignore", "--no-index", "-q", rel], cwd=REPO_ROOT)
        assert res.returncode in (0, 1), f"git check-ignore failed on {rel}"
        return res.returncode == 0

    assert [ignored(f"lab/state/{name}") for name in ("ledger.jsonl", "ledger.lock", "STOP", "sub/x.json")] == [
        True, True, True, True]
    assert ignored("lab/state/.gitkeep") is False


# --- the lock -------------------------------------------------------------------------

def test_a_second_handle_is_refused_until_the_first_releases(tmp_path):
    path = tmp_path / "x.lock"
    first = FileLock(path)
    assert first.held is False and first.acquire(timeout=0) is True and first.held is True
    assert first.acquire(timeout=0) is True, "holding it already counts"
    second = FileLock(path)
    assert second.acquire(timeout=0) is False and second.held is False
    first.release()
    assert first.held is False
    assert second.acquire(timeout=0) is True
    second.release()
    first.release()     # releasing a lock that is not held does nothing


def test_acquire_waits_out_its_timeout_and_the_context_manager_raises_lock_timeout(tmp_path):
    path = tmp_path / "x.lock"
    with FileLock(path):
        waiter = FileLock(path)
        start = time.monotonic()
        assert waiter.acquire(timeout=0.3) is False
        assert 0.3 <= time.monotonic() - start < 5.0
        start = time.monotonic()
        with pytest.raises(LockTimeout) as err:
            with FileLock(path, timeout=0.3):
                pytest.fail("entered a lock another handle holds")
        assert 0.3 <= time.monotonic() - start < 5.0
        assert isinstance(err.value, TimeoutError) and str(path) in str(err.value)
    assert free(path) is True


def test_the_context_manager_releases_on_exit_and_on_an_exception_and_is_not_reentrant(tmp_path):
    path = tmp_path / "x.lock"
    with FileLock(path) as lock:
        assert lock.held is True and free(path) is False
    assert lock.held is False and free(path) is True
    with pytest.raises(ValueError):
        with FileLock(path):
            raise ValueError("boom")
    assert free(path) is True
    outer = FileLock(path)
    with outer:
        with pytest.raises(RuntimeError):
            with outer:
                pytest.fail("a held FileLock entered twice")
        assert outer.held is True and free(path) is False
    assert free(path) is True


def test_a_refused_caller_reads_the_holders_note(tmp_path):
    path = tmp_path / "x.lock"
    with FileLock(path) as lock:
        lock.write_note("pid 1 appending")
        assert FileLock(path).read_note() == "pid 1 appending"
        with pytest.raises(LockTimeout) as err:
            with FileLock(path, timeout=0.05):
                pytest.fail("entered a lock another handle holds")
        assert "pid 1 appending" in str(err.value)
    assert FileLock(tmp_path / "missing" / "y.lock").read_note() == ""


def _try_lock(path: str, timeout: float) -> None:
    """Spawned child: exit 10 when it got the lock, 11 when it was refused."""
    lock = FileLock(path)
    got = lock.acquire(timeout=timeout)
    lock.release()
    raise SystemExit(10 if got else 11)


def _hold_lock_until_told(path: str, held, done) -> None:
    """Spawned child: take the lock, say so, then exit without releasing it."""
    lock = FileLock(path)
    if not lock.acquire(timeout=30):
        raise SystemExit(3)
    held.set()
    done.wait(JOIN_TIMEOUT)
    os._exit(0)


def spawn(target, *args):
    """A spawned child, daemonic so a hung one cannot keep the test run alive."""
    return SPAWN.Process(target=target, args=args, daemon=True)


def run_child(target, *args) -> int:
    child = spawn(target, *args)
    child.start()
    child.join(JOIN_TIMEOUT)
    assert not child.is_alive(), f"{target.__name__} hung"
    return child.exitcode


def test_the_lock_excludes_another_process_and_is_freed_when_its_holder_dies(tmp_path):
    path = str(tmp_path / "x.lock")
    with FileLock(path):
        assert run_child(_try_lock, path, 0.2) == 11
    assert run_child(_try_lock, path, 0.2) == 10
    held, done = SPAWN.Event(), SPAWN.Event()
    holder = spawn(_hold_lock_until_told, path, held, done)
    holder.start()
    try:
        assert held.wait(JOIN_TIMEOUT), "the child never took the lock"
        assert free(path) is False
    finally:
        done.set()
        holder.join(JOIN_TIMEOUT)
    assert holder.exitcode == 0
    survivor = FileLock(path)
    assert survivor.acquire(timeout=10) is True, "the lock outlived the process that held it"
    survivor.release()


def test_append_gives_up_with_lock_timeout_while_the_ledger_lock_stays_busy():
    assert ledger.append(make_row()) == 1
    with FileLock(ledger.lock_path()) as lock:
        lock.write_note("pid 4242 (a stuck writer)")
        start = time.monotonic()
        with pytest.raises(LockTimeout) as err:
            ledger.append(make_row(), timeout=0.3)
        assert 0.3 <= time.monotonic() - start < 5.0
        assert "pid 4242 (a stuck writer)" in str(err.value)
    assert seqs(ledger.read()) == [1]
    assert ledger.append(make_row()) == 2


# --- two processes at once ------------------------------------------------------------

def _append_rows(state: str, actor: str, start, out: str) -> None:
    """Spawned child: wait for the other child, append ROWS_PER_CHILD rows as
    fast as it can, then write down the seq each append returned."""
    start.wait(START_TIMEOUT)
    returned = [ledger.append({"actor": actor, "tier": "sonnet", "task": "L3", "verb": "ledger.race",
                               "outcome": "ok", "metrics": {"i": i}, "refs": {}}, state_dir=state)
                for i in range(ROWS_PER_CHILD)]
    Path(out).write_text(json.dumps(returned), encoding="utf-8")


def test_two_processes_appending_1000_rows_each_lose_nothing(tmp_path, isolated_state):
    start = SPAWN.Barrier(2)
    children = {name: spawn(_append_rows, str(isolated_state), name, start, str(tmp_path / f"{name}.json"))
                for name in ("a", "b")}
    for child in children.values():
        child.start()
    for child in children.values():
        child.join(JOIN_TIMEOUT)
    assert {name: child.exitcode for name, child in children.items()} == {"a": 0, "b": 0}
    raw = (isolated_state / "ledger.jsonl").read_bytes()
    assert raw.endswith(b"\n") and raw.count(b"\n") == 2 * ROWS_PER_CHILD
    rows = ledger.read(state_dir=isolated_state)
    assert seqs(rows) == list(range(1, 2 * ROWS_PER_CHILD + 1))
    for name in ("a", "b"):
        mine = [r for r in rows if r["actor"] == name]
        assert [r["metrics"]["i"] for r in mine] == list(range(ROWS_PER_CHILD))
        returned = json.loads((tmp_path / f"{name}.json").read_text(encoding="utf-8"))
        assert returned == seqs(mine), "every seq append returned is the seq on disk"
    switches = sum(1 for prev, cur in zip(rows, rows[1:]) if prev["actor"] != cur["actor"])
    assert switches > 1, "the two children never interleaved, so nothing raced"
