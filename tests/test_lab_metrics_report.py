"""Tests for python -m lab metrics (lab/machine/metrics_report.py): report and compare over the metrics file and the
ledger, with verdicts from lab/machine/verdict.py.

Every test writes its own metrics file with metrics.add() and its own ledger with ledger.append() under a private
state dir, so nothing reads or writes the real lab/state. The tests that read the committed seed file only read it.
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from lab import __main__ as lab_cli
from lab.machine import ledger, metrics
from lab.machine import metrics_report as mr

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = {"model": "m", "engine": "e", "batch": 1}


@pytest.fixture(autouse=True)
def state_dir(tmp_path, monkeypatch):
    """A private state dir, so the ledger the code finds by default is this test's own."""
    state = tmp_path / "state"
    monkeypatch.setenv("LAB_STATE_DIR", str(state))
    return state


@pytest.fixture
def mfile(tmp_path):
    return tmp_path / "metrics.jsonl"


def ts(hour, minute=0):
    return f"2026-10-10T{hour:02d}:{minute:02d}:00Z"


def file_add(path, value, *, at, metric="teacher.decode_tok_s", unit="tok/s", node="pod-a", conditions=None,
             spread=None, source="unit test", low_confidence=False):
    conditions = dict(BASE if conditions is None else conditions)
    if spread is not None:
        conditions["spread"] = spread
    return metrics.add({"metric": metric, "value": value, "unit": unit, "node": node, "conditions": conditions,
                        "source": source, "ts": at, "low_confidence": low_confidence}, path=path)


def ledger_add(values, *, at, node="pod-a", units="tok/s", conditions=None, source="ledger test",
               low_confidence=None):
    row = {"ts": at, "actor": "test", "tier": None, "task": None, "verb": "measure", "outcome": "ok",
           "metrics": dict(values), "refs": {}, "node": node,
           "conditions": dict(BASE if conditions is None else conditions), "source": source}
    if units is not None:
        row["units"] = units if isinstance(units, dict) else {name: units for name in values}
    if low_confidence is not None:
        row["low_confidence"] = low_confidence
    return ledger.append(row)


def only_row(doc):
    assert doc["ok"] is True, doc
    assert doc["count"] == 1, [row["conditions"] for row in doc["rows"]]
    return doc["rows"][0]


def cli(argv, capsys):
    code = lab_cli.main(argv)
    out, err = capsys.readouterr()
    return code, out, err


def one_json_line(text):
    lines = text.splitlines()
    assert len(lines) == 1, f"expected exactly one stdout line, got {lines!r}"
    return json.loads(lines[0])


# --- the committed seed file --------------------------------------------------------------------------------------------

def test_report_on_the_seed_file_gives_every_row_as_a_latest_reading_at_its_own_line():
    doc = mr.report(source="metrics")
    lines = metrics.DEFAULT_PATH.read_text(encoding="utf-8").splitlines()
    seen = []
    for row in doc["rows"]:
        stored = metrics.latest(row["metric"], row["node"], conditions=row["conditions"])
        assert row["latest"]["value"] == stored["value"]
        assert row["latest"]["ts"] == stored["ts"]
        (ref,) = row["latest"]["refs"]
        name, _, line = ref.partition(":")
        assert name == "metrics.jsonl"
        on_disk = json.loads(lines[int(line) - 1])
        assert (on_disk["metric"], on_disk["node"], on_disk["value"]) == (row["metric"], row["node"],
                                                                         row["latest"]["value"])
        seen.append(ref)
    assert len(seen) == len(set(seen))
    assert doc["source"]["metrics_file"]["readings"] == len(metrics.load())


def test_seed_series_follow_the_rubric_directions():
    rows = {(row["metric"], row["node"]): row for row in mr.report(source="metrics")["rows"]}
    assert rows[("teacher.joules_per_token", "pod-a")]["direction"] == mr.LOWER
    assert rows[("trainer.samples_per_s", "pod-b")]["direction"] == mr.HIGHER
    assert rows[("teacher.vram_pct", "pod-a")]["direction"] is None
    assert rows[("pod.idle_watts", "pod-a")]["direction"] is None
    decode = [row for row in mr.report(source="metrics", metric_names=["teacher.decode_tok_s"])["rows"]
              if row["node"] == "pod-a"]
    assert [row["direction"] for row in decode] == [mr.HIGHER] * 3


# --- report: latest against previous -----------------------------------------------------------------------------------

def test_a_gain_beyond_the_recorded_spread_reads_better_with_the_welch_numbers(mfile):
    file_add(mfile, 7.7, at=ts(10), spread={"runs": 5, "sd": 0.03})
    file_add(mfile, 8.1, at=ts(12), spread={"runs": 5, "sd": 0.04})
    row = only_row(mr.report(metrics_path=mfile))
    assert row["verdict"] == mr.BETTER
    assert row["previous"] == {"value": 7.7, "runs": 5, "sd": 0.03, "readings": 1, "ts": ts(10),
                               "refs": ["metrics.jsonl:1"], "also_in": [], "low_confidence": False}
    assert row["latest"]["refs"] == ["metrics.jsonl:2"]
    assert (row["latest"]["value"], row["latest"]["runs"], row["latest"]["sd"]) == (8.1, 5, 0.04)
    assert row["change"] == 0.4
    assert row["change_pct"] == pytest.approx(0.4 / 7.7 * 100)
    # by hand: se = sqrt(0.03^2/5 + 0.04^2/5) = sqrt(0.0005); t = 0.4 / se; df = 0.0005^2 / (0.00018^2/4 + 0.00032^2/4)
    stats = row["stats"]
    assert stats["label"] == "b_ahead"
    assert stats["statistic"] == pytest.approx(0.4 / math.sqrt(0.0005), rel=1e-9)
    assert stats["df"] == pytest.approx(0.0005 ** 2 / (0.00018 ** 2 / 4 + 0.00032 ** 2 / 4), rel=1e-9)
    assert stats["p"] < 1e-5
    assert (stats["n_a"], stats["n_b"], stats["more_a"], stats["more_b"]) == (5, 5, 0, 0)


@pytest.mark.parametrize("before, after, expected", [(26.0, 21.0, mr.BETTER), (21.0, 26.0, mr.WORSE)])
def test_lower_is_better_for_joules_per_token(mfile, before, after, expected):
    for value, hour in ((before, 10), (after, 12)):
        file_add(mfile, value, at=ts(hour), metric="teacher.joules_per_token", unit="J/token",
                 spread={"runs": 5, "sd": 0.2})
    row = only_row(mr.report(metrics_path=mfile))
    assert row["direction"] == mr.LOWER
    assert row["verdict"] == expected
    assert row["change"] == after - before


@pytest.mark.parametrize("metric, unit, before, after, expected", [
    ("teacher.vram_pct", "%", 84, 90, mr.HIGHER),
    ("pod.idle_watts", "W", 39.0, 30.0, mr.LOWER),
])
def test_a_metric_with_no_better_side_reads_higher_or_lower(mfile, metric, unit, before, after, expected):
    for value, hour in ((before, 10), (after, 12)):
        file_add(mfile, value, at=ts(hour), metric=metric, unit=unit, spread={"runs": 5, "sd": 0.5})
    row = only_row(mr.report(metrics_path=mfile))
    assert row["direction"] is None
    assert row["verdict"] == expected
    assert row["note"].startswith("no side of this metric is better, so this reads higher or lower: ")


def test_overlapping_spreads_read_no_difference_and_size_the_runs_needed(mfile):
    file_add(mfile, 7.7, at=ts(10), spread={"runs": 5, "sd": 0.3})
    file_add(mfile, 7.75, at=ts(12), spread={"runs": 5, "sd": 0.3})
    row = only_row(mr.report(metrics_path=mfile))
    assert row["verdict"] == mr.NO_DIFFERENCE
    stats = row["stats"]
    assert stats["p"] > 0.05
    assert stats["needed_per_arm"] > 5
    assert stats["more_a"] == stats["more_b"] == stats["needed_per_arm"] - 5


def test_single_runs_are_too_early_with_four_more_runs_each_and_still_show_the_change(mfile):
    file_add(mfile, 7.7, at=ts(10))
    file_add(mfile, 7.8, at=ts(12))
    row = only_row(mr.report(metrics_path=mfile))
    assert row["verdict"] == mr.TOO_EARLY
    assert (row["previous"]["runs"], row["latest"]["runs"]) == (1, 1)
    assert row["change"] == 0.1
    assert row["change_pct"] == pytest.approx(0.1 / 7.7 * 100)
    stats = row["stats"]
    assert (stats["n_a"], stats["n_b"], stats["more_a"], stats["more_b"]) == (1, 1, 4, 4)
    assert row["note"] == stats["note"]


def test_one_reading_has_no_previous_and_says_so(mfile):
    file_add(mfile, 7.7, at=ts(10))
    row = only_row(mr.report(metrics_path=mfile))
    assert row["previous"] is None
    assert row["change"] is None and row["change_pct"] is None
    assert row["verdict"] == mr.TOO_EARLY
    assert row["note"] == "one reading only: nothing earlier to compare it with"


def test_latest_is_by_time_not_by_file_order(mfile):
    file_add(mfile, 8.0, at=ts(12))
    file_add(mfile, 7.0, at=ts(9))  # an older number appended late
    row = only_row(mr.report(metrics_path=mfile))
    assert (row["previous"]["value"], row["latest"]["value"]) == (7.0, 8.0)
    assert (row["previous"]["refs"], row["latest"]["refs"]) == (["metrics.jsonl:2"], ["metrics.jsonl:1"])


def test_a_tie_on_time_keeps_file_order(mfile):
    file_add(mfile, 7.0, at=ts(10))
    file_add(mfile, 7.5, at=ts(10))
    row = only_row(mr.report(metrics_path=mfile))
    assert (row["previous"]["value"], row["latest"]["value"]) == (7.0, 7.5)


def test_a_blank_line_still_counts_in_the_line_refs(mfile):
    file_add(mfile, 7.0, at=ts(10))
    with open(mfile, "ab") as fh:
        fh.write(b"\n")
    file_add(mfile, 7.5, at=ts(11))
    row = only_row(mr.report(metrics_path=mfile))
    assert row["latest"]["refs"] == ["metrics.jsonl:3"]


def test_the_previous_reading_marked_low_confidence_flags_the_row(mfile):
    file_add(mfile, 7.7, at=ts(10), low_confidence=True)
    file_add(mfile, 7.8, at=ts(12))
    row = only_row(mr.report(metrics_path=mfile))
    assert row["low_confidence"] is True
    assert row["previous"]["low_confidence"] is True and row["latest"]["low_confidence"] is False
    assert row["note"].endswith("; low confidence: metrics.jsonl:1")


# --- the recorded spread --------------------------------------------------------------------------------------------

def test_a_recorded_mean_is_what_the_verdict_reads_while_the_value_is_what_is_shown(mfile):
    file_add(mfile, 7.0, at=ts(10), spread={"runs": 5, "sd": 0.1, "mean": 7.2, "median": 7.0, "min": 7.0, "max": 7.4})
    file_add(mfile, 7.1, at=ts(12), spread={"runs": 5, "sd": 0.1})
    row = only_row(mr.report(metrics_path=mfile))
    assert row["previous"]["value"] == 7.0
    assert row["change"] == 0.1
    assert row["stats"]["diff"] == pytest.approx(7.1 - 7.2)
    assert row["verdict"] == mr.NO_DIFFERENCE


@pytest.mark.parametrize("spread, problem", [
    ({"runs": 5}, "a spread of 5 runs needs sd"),
    ({"runs": 5, "sd": -0.1}, "a spread of 5 runs needs sd"),
    ({"runs": 0, "sd": 0.1}, "spread runs must be a whole number of at least 1"),
    ({"runs": 2.5, "sd": 0.1}, "spread runs must be a whole number of at least 1"),
    ({"runs": True, "sd": 0.1}, "spread runs must be a whole number of at least 1"),
    ({"runs": 5, "sd": 0.1, "mean": "high"}, "spread mean must be a finite number"),
    ("five runs", "conditions.spread must be an object"),
])
def test_a_spread_that_cannot_be_read_is_one_run_and_a_warning(mfile, spread, problem):
    file_add(mfile, 7.7, at=ts(10), spread=spread)
    doc = mr.report(metrics_path=mfile)
    row = only_row(doc)
    assert (row["latest"]["runs"], row["latest"]["sd"], row["latest"]["value"]) == (1, None, 7.7)
    assert len(doc["warnings"]) == 1
    warning = doc["warnings"][0]
    assert warning.startswith("metrics.jsonl:1 teacher.decode_tok_s: ")
    assert problem in warning
    assert warning.endswith("; read as one run")


def test_a_spread_of_one_run_has_no_sd(mfile):
    file_add(mfile, 7.7, at=ts(10), spread={"runs": 1, "sd": 0.5})
    row = only_row(mr.report(metrics_path=mfile))
    assert (row["latest"]["runs"], row["latest"]["sd"]) == (1, None)


def test_warnings_cover_only_the_readings_asked_about(mfile):
    file_add(mfile, 7.7, at=ts(10), spread={"runs": 5})
    file_add(mfile, 0.3, at=ts(10), metric="teacher.ttft_s", unit="s")
    assert mr.report(metrics_path=mfile, metric_names=["teacher.ttft_s"])["warnings"] == []
    assert len(mr.report(metrics_path=mfile)["warnings"]) == 1


# --- series -----------------------------------------------------------------------------------------------------------

def test_keys_that_describe_the_number_do_not_split_a_series(mfile):
    file_add(mfile, 26.0, at=ts(10), metric="teacher.joules_per_token", unit="J/token",
             conditions={**BASE, "derived": True, "inputs": {"w": 200, "tok_s": 7.7}, "stated_as": "about 26 J",
                         "low_confidence_reason": "one window", "value_is": "rounded", "range_w": [199, 201],
                         "approx": True, "note": "first"})
    file_add(mfile, 25.0, at=ts(12), metric="teacher.joules_per_token", unit="J/token",
             conditions={**BASE, "derived": True, "inputs": {"w": 195, "tok_s": 7.8}}, spread={"runs": 5, "sd": 0.1})
    row = only_row(mr.report(metrics_path=mfile))
    assert row["conditions"] == {**BASE, "derived": True}
    assert (row["previous"]["value"], row["latest"]["value"]) == (26.0, 25.0)


@pytest.mark.parametrize("change", [
    {"conditions": {**BASE, "batch": 8}},
    {"node": "pod-b"},
    {"unit": "tok/min"},
])
def test_a_different_condition_node_or_unit_is_another_series(mfile, change):
    file_add(mfile, 7.7, at=ts(10))
    file_add(mfile, 8.0, at=ts(12), **change)
    doc = mr.report(metrics_path=mfile)
    assert doc["count"] == 2
    assert all(row["previous"] is None for row in doc["rows"])


# --- filters ----------------------------------------------------------------------------------------------------------

def test_report_filters_by_metric_node_and_where_with_json_values(mfile):
    file_add(mfile, 7.7, at=ts(10))
    file_add(mfile, 20.0, at=ts(10), conditions={**BASE, "batch": 8})
    file_add(mfile, 9.0, at=ts(10), conditions={**BASE, "batch": "1"})
    file_add(mfile, 7.9, at=ts(10), node="pod-b")
    file_add(mfile, 0.3, at=ts(10), metric="teacher.ttft_s", unit="s")
    doc = mr.report(metrics_path=mfile, metric_names=["teacher.decode_tok_s"], node="pod-a", where={"batch": 1})
    row = only_row(doc)
    assert row["latest"]["value"] == 7.7
    assert doc["filters"] == {"metric": ["teacher.decode_tok_s"], "node": "pod-a", "where": {"batch": 1}}


def test_report_lists_asked_for_metrics_that_have_no_reading(mfile):
    file_add(mfile, 7.7, at=ts(10))
    doc = mr.report(metrics_path=mfile, metric_names=["teacher.decode_tok_s", "teacher.output_match_rate",
                                                      "teacher.output_match_rate"])
    assert doc["count"] == 1
    assert doc["no_readings"] == ["teacher.output_match_rate"]
    assert "No readings, so no baseline yet: teacher.output_match_rate." in mr.render_md(doc)


# --- the ledger -------------------------------------------------------------------------------------------------------

def test_ledger_readings_carry_their_seq_unit_node_and_flags():
    ledger_add({"teacher.decode_tok_s": 7.6}, at=ts(10), low_confidence=True)
    ledger_add({"teacher.decode_tok_s": 7.9}, at=ts(12), conditions={**BASE, "spread": {"runs": 5, "sd": 0.05}})
    row = only_row(mr.report(source="ledger"))
    assert row["unit"] == "tok/s" and row["node"] == "pod-a"
    assert row["previous"] == {"value": 7.6, "runs": 1, "sd": None, "readings": 1, "ts": ts(10),
                               "refs": ["ledger:1"], "also_in": [], "low_confidence": True}
    assert (row["latest"]["refs"], row["latest"]["runs"], row["latest"]["sd"]) == (["ledger:2"], 5, 0.05)


def test_the_newest_ledger_reading_of_each_series_is_what_latest_metric_answers():
    ledger_add({"teacher.decode_tok_s": 7.6, "teacher.ttft_s": 0.5}, at=ts(10),
               units={"teacher.decode_tok_s": "tok/s", "teacher.ttft_s": "s"})
    ledger_add({"teacher.decode_tok_s": 7.9}, at=ts(12))
    ledger_add({"teacher.decode_tok_s": 7.0}, at=ts(9))  # a backfill: older, appended last
    ledger_add({"teacher.decode_tok_s": 20.0}, at=ts(11), conditions={**BASE, "batch": 8})
    ledger_add({"teacher.decode_tok_s": 6.0}, at=ts(11), node="pod-b")
    doc = mr.report(source="ledger")
    latest = {(row["metric"], row["node"], row["conditions"]["batch"]): row["latest"] for row in doc["rows"]}
    assert {key: arm["value"] for key, arm in latest.items()} == {
        ("teacher.decode_tok_s", "pod-a", 1): 7.9, ("teacher.decode_tok_s", "pod-a", 8): 20.0,
        ("teacher.decode_tok_s", "pod-b", 1): 6.0, ("teacher.ttft_s", "pod-a", 1): 0.5}
    for row in doc["rows"]:
        newest = ledger.latest_metric(row["metric"], row["node"], conditions=row["conditions"])
        assert row["latest"]["value"] == newest["value"]
        assert row["latest"]["ts"] == newest["ts"]
        assert row["latest"]["refs"] == [f"ledger:{newest['seq']}"]
        assert row["unit"] == newest["unit"]
        assert row["latest"]["low_confidence"] == newest["low_confidence"]


def test_a_multi_metric_ledger_row_keys_its_spread_by_metric():
    spread = {"teacher.decode_tok_s": {"runs": 5, "sd": 0.05}, "teacher.ttft_s": {"runs": 4, "sd": 0.01}}
    ledger_add({"teacher.decode_tok_s": 7.9, "teacher.ttft_s": 0.5}, at=ts(10),
               units={"teacher.decode_tok_s": "tok/s", "teacher.ttft_s": "s"}, conditions={**BASE, "spread": spread})
    doc = mr.report(source="ledger")
    arms = {row["metric"]: row["latest"] for row in doc["rows"]}
    assert (arms["teacher.decode_tok_s"]["runs"], arms["teacher.decode_tok_s"]["sd"]) == (5, 0.05)
    assert (arms["teacher.ttft_s"]["runs"], arms["teacher.ttft_s"]["sd"]) == (4, 0.01)
    assert doc["warnings"] == []


def test_a_flat_spread_on_a_multi_metric_ledger_row_is_refused_with_a_warning():
    ledger_add({"teacher.decode_tok_s": 7.9, "teacher.ttft_s": 0.5}, at=ts(10),
               units={"teacher.decode_tok_s": "tok/s", "teacher.ttft_s": "s"},
               conditions={**BASE, "spread": {"runs": 5, "sd": 0.05}})
    doc = mr.report(source="ledger")
    assert [row["latest"]["runs"] for row in doc["rows"]] == [1, 1]
    assert len(doc["warnings"]) == 2
    assert all("not keyed by metric on a row with 2 metrics" in warning for warning in doc["warnings"])


def test_a_ledger_reading_with_no_unit_is_its_own_series_and_a_warning(mfile):
    file_add(mfile, 7.7, at=ts(10))
    ledger_add({"teacher.decode_tok_s": 7.9}, at=ts(12), units=None)
    doc = mr.report(metrics_path=mfile)
    assert doc["count"] == 2
    assert doc["warnings"] == ["teacher.decode_tok_s has no unit at ledger:1, so those readings are a series of "
                               "their own"]


def test_auto_reads_the_file_and_the_ledger_together(mfile):
    file_add(mfile, 7.7, at=ts(10), spread={"runs": 5, "sd": 0.03})
    ledger_add({"teacher.decode_tok_s": 8.1}, at=ts(12), conditions={**BASE, "spread": {"runs": 5, "sd": 0.04}})
    doc = mr.report(metrics_path=mfile)
    row = only_row(doc)
    assert (row["previous"]["refs"], row["latest"]["refs"]) == (["metrics.jsonl:1"], ["ledger:1"])
    assert row["verdict"] == mr.BETTER
    assert doc["source"]["metrics_file"] == {"path": str(mfile), "present": True, "readings": 1}
    assert doc["source"]["ledger"]["present"] is True and doc["source"]["ledger"]["readings"] == 1
    assert (doc["source"]["copies_merged"], doc["source"]["readings"]) == (0, 2)


@pytest.mark.parametrize("copy_at, copy_source", [(ts(10), "somewhere else"), (ts(15), "unit test")])
def test_a_ledger_copy_of_a_file_line_counts_once(mfile, copy_at, copy_source):
    file_add(mfile, 7.7, at=ts(10), conditions={**BASE, "stated_as": "7.7 tok/s"})
    ledger_add({"teacher.decode_tok_s": 7.7}, at=copy_at, source=copy_source)
    doc = mr.report(metrics_path=mfile)
    row = only_row(doc)
    assert row["previous"] is None
    assert row["latest"]["refs"] == ["metrics.jsonl:1"]
    assert row["latest"]["also_in"] == ["ledger:1"]
    assert doc["source"]["copies_merged"] == 1


def test_a_repeat_measurement_with_the_same_value_is_not_a_copy(mfile):
    file_add(mfile, 4, at=ts(10), metric="pod.credits_per_min", unit="credits/min")
    ledger_add({"pod.credits_per_min": 4}, at=ts(15), units="credits/min", source="quota/me reading")
    row = only_row(mr.report(metrics_path=mfile))
    assert (row["previous"]["refs"], row["latest"]["refs"]) == (["metrics.jsonl:1"], ["ledger:1"])
    assert row["change"] == 0


def test_each_file_line_absorbs_one_copy_at_most(mfile):
    file_add(mfile, 7.7, at=ts(10))
    ledger_add({"teacher.decode_tok_s": 7.7}, at=ts(10))
    ledger_add({"teacher.decode_tok_s": 7.7}, at=ts(10))
    doc = mr.report(metrics_path=mfile)
    row = only_row(doc)
    assert doc["source"]["copies_merged"] == 1
    assert (row["previous"]["refs"], row["latest"]["refs"]) == (["metrics.jsonl:1"], ["ledger:2"])


def test_source_metrics_ignores_the_ledger_and_source_ledger_ignores_the_file(mfile):
    file_add(mfile, 7.7, at=ts(10))
    ledger_add({"teacher.decode_tok_s": 8.1}, at=ts(12))
    from_file = only_row(mr.report(metrics_path=mfile, source="metrics"))
    from_ledger = only_row(mr.report(metrics_path=mfile, source="ledger"))
    assert from_file["latest"]["refs"] == ["metrics.jsonl:1"]
    assert from_ledger["latest"]["refs"] == ["ledger:1"]


def test_source_ledger_with_no_ledger_warns_and_reports_nothing(mfile, state_dir):
    file_add(mfile, 7.7, at=ts(10))
    doc = mr.report(metrics_path=mfile, source="ledger")
    assert doc["count"] == 0
    assert doc["warnings"] == [f"no ledger at {state_dir / 'ledger.jsonl'}"]


def test_ledger_rows_without_metrics_are_not_readings(mfile):
    ledger.append({"actor": "test", "tier": None, "task": None, "verb": "ready", "outcome": "ok", "metrics": {},
                   "refs": {}})
    file_add(mfile, 7.7, at=ts(10))
    doc = mr.report(metrics_path=mfile)
    assert doc["source"]["ledger"] == {"path": doc["source"]["ledger"]["path"], "present": True, "readings": 0}
    assert only_row(doc)["latest"]["refs"] == ["metrics.jsonl:1"]


# --- compare ----------------------------------------------------------------------------------------------------------

def test_compare_pools_each_window_and_leaves_out_the_gap(mfile):
    for value, minute in ((7.6, 0), (7.7, 10), (7.8, 20)):
        file_add(mfile, value, at=ts(9, minute))
    file_add(mfile, 9.9, at=ts(11))  # between the two times: in neither arm
    for value, minute in ((8.0, 0), (8.1, 10), (8.2, 20)):
        file_add(mfile, value, at=ts(12, minute))
    doc = mr.compare("teacher.decode_tok_s", before=mr.parse_time(ts(10)), after=mr.parse_time(ts(12)),
                     metrics_path=mfile)
    row = only_row(doc)
    assert doc["window"] == {"before": ts(10), "after": ts(12)}
    assert row["before"]["runs"] == 3 and row["before"]["readings"] == 3
    assert row["before"]["value"] == pytest.approx(7.7)
    assert row["before"]["sd"] == pytest.approx(0.1)  # stdev of 7.6, 7.7, 7.8
    assert row["before"]["refs"] == ["metrics.jsonl:1", "metrics.jsonl:2", "metrics.jsonl:3"]
    assert row["after"]["refs"] == ["metrics.jsonl:5", "metrics.jsonl:6", "metrics.jsonl:7"]
    assert row["after"]["value"] == pytest.approx(8.1)
    assert row["change"] == pytest.approx(0.4)
    assert row["verdict"] == mr.TOO_EARLY
    assert (row["stats"]["more_a"], row["stats"]["more_b"]) == (2, 2)


def test_compare_pools_recorded_spreads_exactly_and_calls_the_gain(mfile):
    file_add(mfile, 7.70, at=ts(8), spread={"runs": 5, "sd": 0.03})
    file_add(mfile, 7.72, at=ts(9), spread={"runs": 5, "sd": 0.03})
    file_add(mfile, 8.10, at=ts(13), spread={"runs": 5, "sd": 0.04})
    row = only_row(mr.compare("teacher.decode_tok_s", before=mr.parse_time(ts(10)), metrics_path=mfile))
    # by hand: n 10, mean 7.71, sum of squares 4(0.03^2) + 4(0.03^2) + 5(0.01^2) + 5(0.01^2) = 0.0082 over 9
    assert row["before"]["runs"] == 10
    assert row["before"]["value"] == pytest.approx(7.71)
    assert row["before"]["sd"] == pytest.approx(math.sqrt(0.0082 / 9))
    assert (row["after"]["runs"], row["after"]["value"], row["after"]["sd"]) == (5, 8.1, 0.04)
    assert row["verdict"] == mr.BETTER
    assert row["stats"]["n_a"] == 10 and row["stats"]["n_b"] == 5


def test_one_time_splits_the_history(mfile):
    file_add(mfile, 7.7, at=ts(9))
    file_add(mfile, 7.9, at=ts(10))  # exactly at the split: after
    only_after = only_row(mr.compare("teacher.decode_tok_s", after=mr.parse_time(ts(10)), metrics_path=mfile))
    only_before = only_row(mr.compare("teacher.decode_tok_s", before=mr.parse_time(ts(10)), metrics_path=mfile))
    for row in (only_after, only_before):
        assert (row["before"]["refs"], row["after"]["refs"]) == (["metrics.jsonl:1"], ["metrics.jsonl:2"])


def test_compare_says_which_arm_is_empty(mfile):
    file_add(mfile, 7.7, at=ts(12))
    file_add(mfile, 0.5, at=ts(9), metric="teacher.ttft_s", unit="s")
    file_add(mfile, 0.3, at=ts(11), metric="pod.start_s", unit="s")
    window = {"before": mr.parse_time(ts(10)), "after": mr.parse_time(ts(12))}
    assert only_row(mr.compare("teacher.decode_tok_s", metrics_path=mfile, **window))["note"] == (
        f"no reading before {ts(10)}")
    assert only_row(mr.compare("teacher.ttft_s", metrics_path=mfile, **window))["note"] == (
        f"no reading at or after {ts(12)}")
    gap = only_row(mr.compare("pod.start_s", metrics_path=mfile, **window))
    assert gap["note"] == f"every reading falls between {ts(10)} and {ts(12)}, so neither arm has one"
    assert gap["before"] is None and gap["after"] is None and gap["verdict"] == mr.TOO_EARLY


def test_compare_gives_one_row_per_series_of_the_metric(mfile):
    file_add(mfile, 7.7, at=ts(9))
    file_add(mfile, 20.0, at=ts(9), conditions={**BASE, "batch": 8})
    file_add(mfile, 0.3, at=ts(9), metric="teacher.ttft_s", unit="s")
    doc = mr.compare("teacher.decode_tok_s", before=mr.parse_time(ts(10)), metrics_path=mfile)
    assert [row["conditions"]["batch"] for row in doc["rows"]] == [1, 8]


def test_compare_of_a_metric_with_no_readings_names_the_close_ones(mfile):
    file_add(mfile, 7.7, at=ts(9))
    doc = mr.compare("teacher.decod_tok_s", before=mr.parse_time(ts(10)), metrics_path=mfile)
    assert doc["ok"] is False and doc["outcome"] == "no_readings"
    assert doc["close"] == ["teacher.decode_tok_s"]
    assert doc["error"] == "no readings of teacher.decod_tok_s in 1 readings"


@pytest.mark.parametrize("kwargs, message", [
    ({}, "compare needs --before, --after or both"),
    ({"before": "2026-10-10T12:00:00Z", "after": "2026-10-10T11:00:00Z"}, "is earlier than --before"),
])
def test_compare_refuses_a_missing_or_backward_window(mfile, kwargs, message):
    times = {key: mr.parse_time(value) for key, value in kwargs.items()}
    with pytest.raises(ValueError, match=message):
        mr.compare("teacher.decode_tok_s", metrics_path=mfile, **times)


def test_compare_refuses_a_naive_time(mfile):
    import datetime as dt
    with pytest.raises(ValueError, match="must carry a UTC offset"):
        mr.compare("teacher.decode_tok_s", before=dt.datetime(2026, 10, 10, 12), metrics_path=mfile)


# --- times ------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("2026-10-10", "2026-10-10T00:00:00+00:00"),
    ("2026-10-10T12:30:00Z", "2026-10-10T12:30:00+00:00"),
    ("2026-10-10T08:30:00-04:00", "2026-10-10T08:30:00-04:00"),
    ("2026-10-10T12:30Z", "2026-10-10T12:30:00+00:00"),
])
def test_parse_time_takes_a_date_or_a_time_with_an_offset(text, expected):
    assert mr.parse_time(text).isoformat() == expected


@pytest.mark.parametrize("text", ["2026-10-10T12:30:00", "yesterday", "2026-13-01", ""])
def test_parse_time_refuses_a_time_without_an_offset_or_not_a_time(text):
    with pytest.raises(Exception, match="is not a time with a UTC offset"):
        mr.parse_time(text)


# --- directions -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("metric, unit, expected", [
    ("teacher.decode_tok_s", None, mr.HIGHER),
    ("teacher.vram_pct", "%", None),
    ("lab.some_new_time_s", "s", mr.LOWER),
    ("lab.some_new_rate", "MB/s", mr.HIGHER),
    ("lab.some_new_cost", "credits/min", mr.LOWER),
    ("lab.some_new_share", "%", None),
    ("lab.some_new_thing", None, None),
])
def test_direction_by_name_then_by_unit(metric, unit, expected):
    assert mr.direction_of(metric, unit) == expected


# --- markdown ---------------------------------------------------------------------------------------------------------

def test_markdown_report_has_one_row_per_series_and_notes_for_two_armed_rows(mfile):
    file_add(mfile, 7.7, at=ts(10), spread={"runs": 5, "sd": 0.03})
    file_add(mfile, 8.1, at=ts(12), spread={"runs": 5, "sd": 0.04})
    file_add(mfile, 0.3, at=ts(10), metric="teacher.ttft_s", unit="s", low_confidence=True)
    text = mr.render_md(mr.report(metrics_path=mfile))
    lines = text.splitlines()
    assert lines[0] == "# Metrics report"
    header = lines.index("| # | metric | node | conditions | previous | latest | change | verdict | read from |")
    assert lines[header + 2] == ("| 1 | teacher.decode_tok_s | pod-a | model=m, engine=e, batch=1 | 7.7 tok/s, 5 runs, "
                                 "sd 0.03 | 8.1 tok/s, 5 runs, sd 0.04 | +0.4 tok/s (+5.2%) | better | "
                                 "metrics.jsonl:1 -> metrics.jsonl:2 |")
    assert lines[header + 3] == ("| 2 | teacher.ttft_s | pod-a | model=m, engine=e, batch=1 | - | 0.3 s, 1 run, low "
                                 "confidence | - | too early | none -> metrics.jsonl:3 |")
    notes = lines.index("Notes (rows with both readings):")
    assert lines[notes + 1].startswith("1. teacher.decode_tok_s on pod-a: B ahead of A: mean 8.1 against mean 7.7 (t = ")
    assert not any(line.startswith("2. ") for line in lines[notes:])


def test_markdown_escapes_pipes_and_spells_out_big_numbers(mfile):
    file_add(mfile, 1234567.0, at=ts(10), metric="pod.balance_credits", unit="credits",
             conditions={"stated_as": "x", "how": "a | b"})
    text = mr.render_md(mr.report(metrics_path=mfile))
    row = next(line for line in text.splitlines() if line.startswith("| 1 |"))
    assert "how=a \\| b" in row
    assert "1234567 credits" in row
    assert row.replace("\\|", "").count("|") == 10


def test_markdown_compare_names_the_window_and_the_arms(mfile):
    file_add(mfile, 7.7, at=ts(9))
    file_add(mfile, 7.9, at=ts(13))
    text = mr.render_md(mr.compare("teacher.decode_tok_s", before=mr.parse_time(ts(10)),
                                   after=mr.parse_time(ts(12)), metrics_path=mfile))
    assert text.startswith("# Metrics compare: teacher.decode_tok_s\n")
    assert f"Before (A) pools the readings before {ts(10)}; after (B) pools the readings at or after {ts(12)}." in text
    assert "| # | metric | node | conditions | before | after | change | verdict | read from |" in text


# --- the verb ---------------------------------------------------------------------------------------------------------

def test_the_verb_answers_one_json_line(mfile, capsys):
    file_add(mfile, 7.7, at=ts(10))
    file_add(mfile, 7.8, at=ts(12))
    code, out, _ = cli(["metrics", "report", "--metrics-file", str(mfile)], capsys)
    assert code == 0
    doc = one_json_line(out)
    assert doc["ok"] is True and doc["outcome"] == "report" and doc["count"] == 1
    assert doc["rows"][0]["latest"]["value"] == 7.8


def test_the_verb_prints_markdown_on_request(mfile, capsys):
    file_add(mfile, 7.7, at=ts(10))
    code, out, _ = cli(["metrics", "report", "--metrics-file", str(mfile), "--format", "md"], capsys)
    assert code == 0
    assert out.startswith("# Metrics report\n")
    assert "| 1 | teacher.decode_tok_s | pod-a |" in out


def test_the_verb_compare_with_where_and_node(mfile, capsys):
    file_add(mfile, 7.7, at=ts(9))
    file_add(mfile, 20.0, at=ts(13), conditions={**BASE, "batch": 8})
    file_add(mfile, 7.9, at=ts(13))
    file_add(mfile, 6.0, at=ts(13), node="pod-b")
    code, out, _ = cli(["metrics", "compare", "teacher.decode_tok_s", "--before", ts(10), "--after", ts(12),
                        "--node", "pod-a", "--where", "batch=1", "--metrics-file", str(mfile)], capsys)
    assert code == 0
    doc = one_json_line(out)
    row = only_row(doc)
    assert (row["before"]["value"], row["after"]["value"]) == (7.7, 7.9)
    assert doc["filters"] == {"metric": ["teacher.decode_tok_s"], "node": "pod-a", "where": {"batch": 1}}


def test_the_verb_compare_of_an_unknown_metric_exits_1(mfile, capsys):
    file_add(mfile, 7.7, at=ts(9))
    code, out, _ = cli(["metrics", "compare", "teacher.decod_tok_s", "--before", ts(10), "--format", "md",
                        "--metrics-file", str(mfile)], capsys)
    assert code == 1
    doc = one_json_line(out)
    assert doc["outcome"] == "no_readings" and doc["close"] == ["teacher.decode_tok_s"]


@pytest.mark.parametrize("argv, message", [
    (["metrics", "compare", "teacher.decode_tok_s"], "give --before, --after or both"),
    (["metrics", "compare", "teacher.decode_tok_s", "--before", ts(12), "--after", ts(11)],
     "--after is earlier than --before"),
    (["metrics", "compare", "teacher.decode_tok_s", "--before", "2026-10-10T12:00:00"],
     "is not a time with a UTC offset"),
    (["metrics", "report", "--where", "batch"], "is not KEY=VALUE"),
    (["metrics", "report", "--source", "both"], "invalid choice"),
    (["metrics"], "required"),
])
def test_bad_arguments_exit_2_with_the_reason_on_stderr(argv, message, capsys):
    code, out, err = cli(argv, capsys)
    assert code == 2
    assert out == ""
    assert message in err


def test_a_corrupt_metrics_file_is_an_error_line_and_exit_1(mfile, capsys):
    file_add(mfile, 7.7, at=ts(10))
    with open(mfile, "ab") as fh:
        fh.write(b"{not json\n")
    code, out, _ = cli(["metrics", "report", "--metrics-file", str(mfile)], capsys)
    assert code == 1
    doc = one_json_line(out)
    assert doc["ok"] is False and doc["outcome"] == "error"
    assert doc["error"].startswith("MetricsFormatError: ") and "line 2 is not valid JSON" in doc["error"]


def test_the_verb_is_refused_while_the_lab_is_stopped(state_dir, mfile, capsys):
    state_dir.mkdir(parents=True)
    (state_dir / "STOP").write_text("maintenance", encoding="utf-8")
    code, out, _ = cli(["metrics", "report", "--metrics-file", str(mfile)], capsys)
    assert code == 1
    assert one_json_line(out)["outcome"] == "stopped"


def test_the_dispatcher_lists_the_verb(capsys):
    code, out, _ = cli(["--help"], capsys)
    assert code == 0
    assert any(line.split() and line.split()[0] == "metrics" and "Report each metric's latest reading" in line
               for line in out.splitlines())


def test_python_dash_m_lab_metrics_runs_from_the_repo_root(tmp_path):
    env = {**os.environ, "LAB_STATE_DIR": str(tmp_path / "state")}
    done = subprocess.run([sys.executable, "-m", "lab", "metrics", "report", "--source", "metrics", "--format", "md"],
                          cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith("# Metrics report\n")
    assert "metrics.jsonl:3" in done.stdout
