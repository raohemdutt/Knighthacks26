"""python -m lab gpu-power record: a row the metrics file refuses must leave the ledger untouched.

The ledger accepts any non-empty metric name while the metrics file requires dotted lowercase words, so a bare
name such as idle_watts is refused by the metrics file. Whichever file refuses, the two stores must not disagree:
nothing may be left in the ledger for a reading the metrics file never got.

LAB_STATE_DIR and the metrics file are isolated per test; the samples are the recorded fixture in
fixtures/telemetry/pod_power.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab import __main__ as lab_cli
from lab.machine import ledger

REPO = Path(__file__).resolve().parents[1]
POD_A = REPO / "fixtures" / "telemetry" / "pod_power" / "samples_gpu-a.jsonl"
IDLE_WINDOW = ["--until", "2026-10-10T18:00:15Z"]  # the first 15 samples of pod A, all at idle


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    state = tmp_path / "state"
    monkeypatch.setenv("LAB_STATE_DIR", str(state))
    monkeypatch.setenv("LAB_ACTOR", "tester")
    monkeypatch.delenv("LAB_TIER", raising=False)
    monkeypatch.delenv("LAB_TASK", raising=False)
    return state


def test_a_metric_name_the_metrics_file_refuses_writes_no_ledger_row_and_no_metrics_row(
        capsys, isolated_state, tmp_path):
    assert POD_A.is_file(), "expected recorded fixture under fixtures/telemetry/pod_power for this test"
    target = tmp_path / "metrics.jsonl"

    try:
        code = lab_cli.main(["gpu-power", "record", "--samples", str(POD_A), "--node", "pod-a",
                             "--metric", "idle_watts", "--metrics-file", str(target), *IDLE_WINDOW])
    except SystemExit as exc:  # an argparse usage error is an acceptable way to refuse the name
        code = exc.code if isinstance(exc.code, int) else 2
    capsys.readouterr()

    assert code != 0, "a metric name the metrics file refuses must not succeed"
    assert ledger.read(state_dir=isolated_state) == [], "the ledger must hold no row for a refused reading"
    assert not target.exists(), "the metrics file must not be created for a refused reading"


# --- record can store a statistic other than the window mean -------------------------------------------------------

POD_B_STALE = REPO / "fixtures" / "telemetry" / "pod_power" / "samples_gpu-b_stale.jsonl"


def test_record_can_store_the_window_peak_as_the_value_of_a_peak_metric(capsys, isolated_state):
    assert POD_B_STALE.is_file(), "expected recorded fixture under fixtures/telemetry/pod_power for this test"

    summary_code = lab_cli.main(["gpu-power", "summary", "--samples", str(POD_B_STALE), "--node", "pod-b"])
    summary_lines = capsys.readouterr().out.strip().splitlines()
    assert summary_code == 0 and len(summary_lines) == 1, summary_lines
    summary = json.loads(summary_lines[0])["summary"]
    assert summary["max_w"] is not None and summary["mean_w"] is not None
    assert summary["max_w"] != summary["mean_w"], "the fixture must tell the peak from the mean"

    try:
        code = lab_cli.main(["gpu-power", "record", "--samples", str(POD_B_STALE), "--node", "pod-b",
                             "--metric", "pod.peak_watts", "--value", "peak"])
    except SystemExit as exc:  # argparse rejecting --value is the failure this test is about
        code = exc.code if isinstance(exc.code, int) else 2
    out_lines = capsys.readouterr().out.strip().splitlines()

    assert code == 0, "record must accept a choice of the statistic to store"
    assert len(out_lines) == 1, out_lines
    assert json.loads(out_lines[0])["outcome"] == "recorded"
    [row] = ledger.read(state_dir=isolated_state)
    assert row["metrics"] == {"pod.peak_watts": summary["max_w"]}
    assert row["conditions"]["value"] == "peak"
