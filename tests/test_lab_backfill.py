"""The ledger backfill (python -m lab ledger backfill): the metrics file's rows and the lab setup facts go into the
ledger once, whatever happens to a run. Every test works in a temporary state dir; nothing here touches lab/state."""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from lab import __main__ as lab_cli
from lab.machine import backfill, ledger, metrics
from lab.machine.lock import FileLock, LockTimeout

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED_FILE = REPO_ROOT / "lab" / "refine" / "metrics.jsonl"
SHARED = ("metric", "value", "unit", "node", "conditions", "source", "ts", "low_confidence")


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """The default state dir of every test is a temporary one, so a call that leaves out state_dir still never
    writes the real lab/state."""
    state = tmp_path / "state"
    monkeypatch.setenv("LAB_STATE_DIR", str(state))
    return state


def make_metric(**changes):
    row = {"metric": "teacher.decode_tok_s", "value": 7.7, "unit": "tok/s", "node": "pod-a",
           "conditions": {"model": "unit-model", "batch": 1}, "source": "unit test", "ts": "2026-10-10T10:20:00Z",
           "low_confidence": False}
    row.update(changes)
    return row


def metrics_file(tmp_path, *rows, name="metrics.jsonl"):
    """A metrics file holding rows, written through the real metrics writer."""
    path = tmp_path / name
    path.write_bytes(b"")
    for row in rows:
        metrics.add(row, path=path)
    return path


def backfill_rows(**filters):
    return ledger.read(verb="backfill", **filters)


def run_cli(argv, capsys):
    code = lab_cli.main(argv)
    out, err = capsys.readouterr()
    return code, out, err


def one_json_line(text):
    lines = text.splitlines()
    assert len(lines) == 1, f"expected exactly one stdout line, got {lines!r}"
    return json.loads(lines[0])


def run_module(*args):
    return subprocess.run([sys.executable, "-m", "lab", *args], cwd=REPO_ROOT, env=dict(os.environ),
                          capture_output=True, text=True, timeout=120)


def total_rows(metrics_path=SEED_FILE):
    return len(backfill.candidates(metrics_path))


# --- the metrics file ------------------------------------------------------------------------------------------

def test_every_row_of_the_seed_file_lands_in_the_ledger_in_file_order():
    seed = metrics.load(SEED_FILE)
    assert len(seed) >= 22

    backfill.run()

    rows = backfill_rows(backfill={"kind": "metric"})
    assert len(rows) == len(seed)
    for entry, row in zip(seed, rows):
        assert row["metrics"] == {entry["metric"]: entry["value"]}, entry
        assert row["units"] == {entry["metric"]: entry["unit"]}, entry
        for name in ("node", "conditions", "source", "ts", "low_confidence"):
            assert row[name] == entry[name], (name, entry)
        assert (row["verb"], row["outcome"], row["actor"], row["tier"], row["task"]) == (
            "backfill", "ok", "backfill", None, None)
        assert row["backfill"]["origin"] == "lab/refine/metrics.jsonl"


def test_the_ledger_answers_latest_metric_like_the_metrics_file():
    seed = metrics.load(SEED_FILE)
    backfill.run()

    for entry in seed:
        by_name = ledger.latest_metric(entry["metric"], entry["node"])
        want = metrics.latest(entry["metric"], entry["node"], path=SEED_FILE)
        assert {k: by_name[k] for k in SHARED} == want, (entry["metric"], entry["node"])

        narrowed = ledger.latest_metric(entry["metric"], entry["node"], conditions=entry["conditions"])
        want = metrics.latest(entry["metric"], entry["node"], conditions=entry["conditions"], path=SEED_FILE)
        assert {k: narrowed[k] for k in SHARED} == want, entry


def test_a_metric_row_is_stored_exactly_and_keeps_the_time_it_was_measured(tmp_path):
    conditions = {"derived": True, "inputs": {"teacher.watts_avg": 200, "teacher.decode_tok_s": 7.7},
                  "range_w": [38, 40]}
    source_file = metrics_file(tmp_path, make_metric(
        metric="teacher.joules_per_token", value=25.97, unit="J/token", conditions=conditions,
        source="derived from two rows", ts="2026-10-10T10:22:13.250000Z", low_confidence=True))

    backfill.run(metrics_path=source_file)

    stored = next(row for row in backfill_rows() if row["backfill"]["kind"] == "metric")
    key = stored["backfill"].pop("key")
    assert re.fullmatch(r"metric:[0-9a-f]{16}", key)
    assert stored == {
        "seq": 1, "ts": "2026-10-10T10:22:13.250000Z", "actor": "backfill", "tier": None, "task": None,
        "verb": "backfill", "outcome": "ok", "metrics": {"teacher.joules_per_token": 25.97},
        "refs": {"pr": None, "commit": None, "slot": None, "run_id": None},
        "node": "pod-a", "units": {"teacher.joules_per_token": "J/token"}, "conditions": conditions,
        "low_confidence": True, "source": "derived from two rows",
        "backfill": {"kind": "metric", "origin": source_file.as_posix()},
    }


def test_an_old_backfilled_number_never_passes_for_a_newer_measurement(tmp_path):
    ledger.append({"ts": "2026-10-10T12:00:00Z", "actor": "opus-1", "tier": "opus", "task": "L3", "verb": "bench",
                   "outcome": "ok", "metrics": {"teacher.decode_tok_s": 9.1}, "refs": {}, "node": "pod-a",
                   "units": {"teacher.decode_tok_s": "tok/s"}, "source": "a later bench"})
    source_file = metrics_file(tmp_path, make_metric(value=7.7, ts="2026-10-10T10:20:00Z"))

    backfill.run(metrics_path=source_file)

    latest = ledger.latest_metric("teacher.decode_tok_s", "pod-a")
    assert (latest["value"], latest["source"], latest["seq"]) == (9.1, "a later bench", 1)


# --- the facts -------------------------------------------------------------------------------------------------

def test_every_fact_lands_as_one_row_with_its_own_details():
    backfill.run()

    rows = backfill_rows(backfill={"kind": "fact"})
    assert [row["backfill"]["key"] for row in rows] == ["fact:" + fact.key for fact in backfill.FACTS]
    for fact, row in zip(backfill.FACTS, rows):
        assert row["ts"] == fact.ts, fact.key
        assert row["note"] == fact.note, fact.key
        assert row["source"] == fact.source, fact.key
        assert row["conditions"] == dict(fact.conditions), fact.key
        assert row.get("node") == fact.node, fact.key
        assert row["backfill"]["topic"] == fact.topic, fact.key
        assert row["metrics"] == {}, fact.key
        assert row["task"] == "R2", fact.key
        assert row["refs"] == {"pr": 37, "commit": "bb351e6", "slot": None, "run_id": None}, fact.key
        assert (row["verb"], row["outcome"], row["actor"], row["tier"], row["low_confidence"]) == (
            "backfill", "ok", "backfill", None, False), fact.key


def test_the_facts_cover_build_fixes_checkpoints_and_pod_sessions_of_both_pods():
    backfill.run()

    facts = backfill_rows(backfill={"kind": "fact"})
    topics = {row["backfill"]["topic"] for row in facts}
    assert {"build_fix", "checkpoint", "pod_session"} <= topics
    served = [row for row in facts if row["backfill"]["key"] == "fact:checkpoint.teacher_in_use"]
    assert len(served) == 1
    assert served[0]["node"] == "pod-a"
    assert served[0]["conditions"]["model"] == "RedHatAI/Qwen3.8-27B-INT4"
    assert served[0]["conditions"]["served_model_name"] == "teacher"
    sessions = {row["node"] for row in facts if row["backfill"]["topic"] == "pod_session" and "node" in row}
    assert sessions == {"pod-a", "pod-b"}
    patches = {row["backfill"]["key"] for row in facts if row["backfill"]["topic"] == "build_fix"}
    assert {"fact:build.fix.attention_num_stages", "fact:build.fix.gptq_half_atomicadd",
            "fact:build.fix.torch_pins_removed"} <= patches


def test_fact_keys_are_unique_and_every_fact_is_a_row_the_ledger_accepts():
    assert len({fact.key for fact in backfill.FACTS}) == len(backfill.FACTS)
    found = backfill.fact_candidates()
    assert [c.key for c in found] == ["fact:" + fact.key for fact in backfill.FACTS]
    for candidate in found:
        assert ledger.validate(candidate.row)["backfill"]["key"] == candidate.key


def test_facts_name_pods_by_role_never_by_account_and_never_carry_a_credential():
    text = json.dumps([backfill.fact_row(fact) for fact in backfill.FACTS])
    accounts = re.findall(r"/user/([^/\s\"]+)/", text)
    assert len(accounts) >= 2
    assert set(accounts) == {"<account>"}
    assert not re.search(r"(?i)token\s*[:=]\s*\S", text)
    assert not re.search(r"[A-Za-z0-9_\-]{40,}", text)


# --- once, whatever happens ------------------------------------------------------------------------------------

def test_the_first_run_reports_what_it_wrote():
    total = total_rows()
    metric_count = len(metrics.load(SEED_FILE))

    result = backfill.run()

    assert result == {
        "ok": True, "outcome": "backfilled", "dry_run": False, "ledger": str(ledger.ledger_path()),
        "metrics_file": str(SEED_FILE),
        "rows": {"metrics": metric_count, "facts": len(backfill.FACTS), "total": total},
        "present": 0, "to_add": total, "added": total, "first_seq": 1, "last_seq": total,
    }
    assert [row["seq"] for row in backfill_rows()] == list(range(1, total + 1))


def test_a_second_run_adds_nothing():
    total = total_rows()
    backfill.run()
    before = ledger.ledger_path().read_bytes()

    result = backfill.run()

    assert result["outcome"] == "up_to_date"
    assert (result["ok"], result["present"], result["to_add"], result["added"]) == (True, total, 0, 0)
    assert (result["first_seq"], result["last_seq"]) == (None, None)
    assert ledger.ledger_path().read_bytes() == before


@pytest.mark.parametrize("failure", [OSError("disk full"), LockTimeout("the ledger stayed locked")])
def test_a_run_cut_short_is_finished_by_the_next_one(monkeypatch, failure):
    total = total_rows()
    real_append = ledger.append
    written = []

    def append_then_fail(row, **kwargs):
        if len(written) == 5:
            raise failure
        written.append(row["backfill"]["key"])
        return real_append(row, **kwargs)

    monkeypatch.setattr(backfill.ledger, "append", append_then_fail)
    if isinstance(failure, LockTimeout):
        cut = backfill.run()
        assert (cut["ok"], cut["outcome"], cut["added"], cut["error"]) == (False, "locked", 5, str(failure))
    else:
        with pytest.raises(OSError, match="disk full"):
            backfill.run()
    monkeypatch.setattr(backfill.ledger, "append", real_append)
    assert len(backfill_rows()) == 5

    result = backfill.run()

    assert (result["outcome"], result["present"], result["added"]) == ("backfilled", 5, total - 5)
    keys = [row["backfill"]["key"] for row in backfill_rows()]
    assert len(keys) == total
    assert len(set(keys)) == total


def test_rows_added_to_the_metrics_file_later_are_picked_up(tmp_path):
    source_file = metrics_file(tmp_path, make_metric(), make_metric(metric="teacher.ttft_s", value=0.55, unit="s"))
    first = backfill.run(metrics_path=source_file)
    assert first["rows"]["metrics"] == 2

    metrics.add(make_metric(metric="teacher.watts_avg", value=200, unit="W"), path=source_file)
    second = backfill.run(metrics_path=source_file)

    assert (second["present"], second["added"]) == (first["added"], 1)
    assert second["first_seq"] == second["last_seq"] == first["added"] + 1
    assert ledger.latest_metric("teacher.watts_avg", "pod-a")["value"] == 200


def test_identical_rows_in_the_metrics_file_are_each_imported_once(tmp_path):
    source_file = metrics_file(tmp_path, make_metric(), make_metric())

    first = backfill.run(metrics_path=source_file)
    second = backfill.run(metrics_path=source_file)

    imported = backfill_rows(backfill={"kind": "metric"})
    keys = [row["backfill"]["key"] for row in imported]
    assert len(keys) == 2
    assert keys[1] == keys[0] + "#2"
    assert first["rows"]["metrics"] == 2
    assert second["added"] == 0


def test_reordering_the_metrics_file_imports_nothing_twice(tmp_path):
    one, two, three = (make_metric(value=v, ts=f"2026-10-10T10:2{v}:00Z") for v in (1, 2, 3))
    original = metrics_file(tmp_path, one, two, three, name="original.jsonl")
    shuffled = metrics_file(tmp_path, three, one, two, name="shuffled.jsonl")
    backfill.run(metrics_path=original)

    result = backfill.run(metrics_path=shuffled)

    assert (result["outcome"], result["added"]) == ("up_to_date", 0)
    assert len(backfill_rows(backfill={"kind": "metric"})) == 3


def test_a_row_that_changed_in_the_file_is_a_new_row_and_the_old_one_stays(tmp_path):
    backfill.run(metrics_path=metrics_file(tmp_path, make_metric(value=7.7), name="before.jsonl"))

    result = backfill.run(metrics_path=metrics_file(tmp_path, make_metric(value=7.9), name="after.jsonl"))

    assert result["added"] == 1
    assert [row["metrics"]["teacher.decode_tok_s"] for row in backfill_rows(backfill={"kind": "metric"})] == [7.7, 7.9]


def test_rows_other_writers_made_are_left_alone_and_only_keyed_backfill_rows_count(tmp_path):
    ledger.append({"ts": "2026-10-10T09:00:00Z", "actor": "opus-1", "tier": "opus", "task": "L3", "verb": "bench",
                   "outcome": "ok", "metrics": {}, "refs": {}})
    ledger.append({"ts": "2026-10-10T09:01:00Z", "actor": "someone", "tier": None, "task": None, "verb": "backfill",
                   "outcome": "ok", "metrics": {}, "refs": {}, "note": "hand-written, no backfill key"})
    source_file = metrics_file(tmp_path, make_metric())
    before = ledger.read()

    result = backfill.run(metrics_path=source_file)

    assert ledger.read()[:2] == before
    assert (result["present"], result["added"], result["first_seq"]) == (0, 1 + len(backfill.FACTS), 3)


def test_two_runs_at_once_write_every_row_once():
    total = total_rows()
    command = [sys.executable, "-m", "lab", "ledger", "backfill"]
    runs = [subprocess.Popen(command, cwd=REPO_ROOT, env=dict(os.environ), stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True) for _ in range(2)]
    outputs = [run.communicate(timeout=120) for run in runs]

    assert [run.returncode for run in runs] == [0, 0], outputs
    answers = [one_json_line(out) for out, _ in outputs]
    assert sorted(answer["added"] for answer in answers) == [0, total]
    rows = backfill_rows()
    assert [row["seq"] for row in rows] == list(range(1, total + 1))
    assert len({row["backfill"]["key"] for row in rows}) == total


# --- dry run ---------------------------------------------------------------------------------------------------

def test_a_dry_run_counts_what_a_run_would_add_and_writes_nothing(isolated_state):
    total = total_rows()

    result = backfill.run(dry_run=True)

    assert result["outcome"] == "dry_run"
    assert (result["ok"], result["dry_run"], result["present"], result["to_add"], result["added"]) == (
        True, True, 0, total, 0)
    assert (result["first_seq"], result["last_seq"]) == (None, None)
    assert not isolated_state.exists()


def test_a_dry_run_after_a_full_run_finds_nothing_to_add():
    backfill.run()
    before = ledger.ledger_path().read_bytes()

    result = backfill.run(dry_run=True)

    assert (result["outcome"], result["present"], result["to_add"]) == ("dry_run", total_rows(), 0)
    assert ledger.ledger_path().read_bytes() == before


# --- refusals: nothing is written ------------------------------------------------------------------------------

def test_a_corrupt_metrics_file_refuses_the_whole_run(tmp_path, isolated_state):
    corrupt = tmp_path / "corrupt.jsonl"
    metrics.add(make_metric(), path=corrupt)
    with open(corrupt, "ab") as fh:
        fh.write(b"this is not json\n")

    result = backfill.run(metrics_path=corrupt)

    assert (result["ok"], result["outcome"]) == (False, "refused")
    assert "line 2" in result["error"]
    assert not isolated_state.exists()


def test_a_missing_metrics_file_is_refused_not_treated_as_empty(tmp_path, isolated_state):
    missing = tmp_path / "nope.jsonl"

    result = backfill.run(metrics_path=missing)

    assert (result["ok"], result["outcome"]) == (False, "refused")
    assert result["error"] == f"metrics file {missing} does not exist"
    assert not isolated_state.exists()


def test_a_fact_the_ledger_would_refuse_stops_the_run_before_any_row_is_written(monkeypatch, isolated_state):
    bad = backfill.Fact(key="bad", topic="build", ts="yesterday", note="n", source="s")
    monkeypatch.setattr(backfill, "FACTS", backfill.FACTS + (bad,))

    result = backfill.run()

    assert (result["ok"], result["outcome"]) == (False, "refused")
    assert result["error"].startswith("ledger row refused: ts must be ISO 8601 text")
    assert not isolated_state.exists()


def test_a_fact_key_used_twice_stops_the_run(monkeypatch, isolated_state):
    monkeypatch.setattr(backfill, "FACTS", backfill.FACTS + backfill.FACTS[:1])

    result = backfill.run()

    assert (result["ok"], result["outcome"]) == (False, "refused")
    assert result["error"] == f"fact key {backfill.FACTS[0].key!r} appears twice in FACTS"
    assert not isolated_state.exists()


# --- the lock --------------------------------------------------------------------------------------------------

def test_a_run_gives_up_while_another_backfill_holds_the_lock(isolated_state):
    with FileLock(isolated_state / backfill.LOCK_FILE, timeout=5):
        result = backfill.run(lock_timeout=0.2)

    assert (result["ok"], result["outcome"], result["added"]) == (False, "locked", 0)
    assert "backfill.lock" in result["error"]
    assert not ledger.ledger_path().exists()
    assert backfill.run()["outcome"] == "backfilled"


# --- python -m lab ledger backfill -----------------------------------------------------------------------------

def test_the_verb_answers_one_json_line_and_fills_the_ledger(capsys):
    total = total_rows()

    code, out, err = run_cli(["ledger", "backfill"], capsys)

    assert code == 0
    assert err == ""
    answer = one_json_line(out)
    assert (answer["ok"], answer["outcome"], answer["added"], answer["rows"]["total"]) == (
        True, "backfilled", total, total)
    assert len(backfill_rows()) == total


def test_the_verb_is_safe_to_repeat(capsys):
    run_cli(["ledger", "backfill"], capsys)

    code, out, _ = run_cli(["ledger", "backfill"], capsys)

    assert code == 0
    assert one_json_line(out)["outcome"] == "up_to_date"
    assert one_json_line(out)["added"] == 0


def test_the_verb_takes_dry_run_and_a_metrics_file(tmp_path, capsys, isolated_state):
    source_file = metrics_file(tmp_path, make_metric())

    code, out, _ = run_cli(["ledger", "backfill", "--dry-run", "--metrics", str(source_file)], capsys)
    assert code == 0
    answer = one_json_line(out)
    assert (answer["outcome"], answer["to_add"], answer["rows"]["metrics"]) == ("dry_run", 1 + len(backfill.FACTS), 1)
    assert answer["metrics_file"] == str(source_file)
    assert not isolated_state.exists()

    code, out, _ = run_cli(["ledger", "backfill", "--metrics", str(source_file)], capsys)
    assert code == 0
    assert one_json_line(out)["added"] == 1 + len(backfill.FACTS)


def test_the_verb_exits_1_with_the_reason_when_the_run_is_refused(tmp_path, capsys):
    code, out, _ = run_cli(["ledger", "backfill", "--metrics", str(tmp_path / "nope.jsonl")], capsys)

    assert code == 1
    answer = one_json_line(out)
    assert (answer["ok"], answer["outcome"]) == (False, "refused")


def test_an_unknown_sub_verb_and_a_missing_one_are_bad_arguments(capsys):
    code, out, err = run_cli(["ledger", "nope"], capsys)
    assert (code, out) == (2, "")
    assert "invalid choice: 'nope'" in err

    code, out, err = run_cli(["ledger"], capsys)
    assert (code, out) == (2, "")
    assert "sub_verb" in err


def test_help_shows_the_sub_verb_and_its_arguments(capsys):
    code, out, _ = run_cli(["ledger", "--help"], capsys)
    assert code == 0
    assert "backfill" in out

    code, out, _ = run_cli(["ledger", "backfill", "-h"], capsys)
    assert code == 0
    assert "--dry-run" in out
    assert "--metrics" in out


def test_the_dispatcher_lists_the_ledger_verb_with_its_summary():
    lines = [line.strip() for line in lab_cli.help_text().splitlines()]
    assert any(line.startswith("ledger ") and line.endswith("backfill puts the lab's past into the ledger.")
               for line in lines), lines


def test_the_verb_is_refused_while_the_machine_is_stopped(isolated_state, capsys):
    isolated_state.mkdir(parents=True)
    (isolated_state / "STOP").write_text("pod credits low\n", encoding="utf-8")

    code, out, _ = run_cli(["ledger", "backfill"], capsys)

    assert code == 1
    assert one_json_line(out)["outcome"] == "stopped"
    assert not ledger.ledger_path().exists()


def test_the_verb_works_as_a_real_process():
    total = total_rows()

    first = run_module("ledger", "backfill")
    again = run_module("ledger", "backfill", "--dry-run")

    assert first.returncode == 0, first.stderr
    assert (one_json_line(first.stdout)["outcome"], one_json_line(first.stdout)["added"]) == ("backfilled", total)
    assert again.returncode == 0, again.stderr
    assert (one_json_line(again.stdout)["outcome"], one_json_line(again.stdout)["to_add"]) == ("dry_run", 0)


def test_the_verb_registry_names_every_verb_once():
    tree = ast.parse((REPO_ROOT / "lab" / "__main__.py").read_text(encoding="utf-8"))
    registries = [node.value for node in ast.walk(tree)
                  if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "VERBS"]
    assert len(registries) == 1
    names = [key.value for key in registries[0].keys]
    repeated = sorted({name for name in names if names.count(name) > 1})
    assert repeated == [], f"VERBS repeats {repeated}: a repeated name silently replaces the verb before it"
    assert "ledger" in names
