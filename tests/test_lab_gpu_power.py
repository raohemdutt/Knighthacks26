"""python -m lab gpu-power: pod power samples summarized, and a window stored as a row that says how it was measured.

Every test gets its own state dir through LAB_STATE_DIR and its own metrics file, so none writes lab/state or
lab/refine/metrics.jsonl. The samples are the recorded fixtures in fixtures/telemetry/pod_power.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab import __main__ as lab_cli
from lab.machine import ledger, metrics

REPO = Path(__file__).resolve().parents[1]
POD_A = REPO / "fixtures" / "telemetry" / "pod_power" / "samples_gpu-a.jsonl"
POD_B = REPO / "fixtures" / "telemetry" / "pod_power" / "samples_gpu-b_stale.jsonl"
IDLE_WINDOW = ["--until", "2026-10-10T18:00:15Z"]  # the first 15 samples of pod A, all at idle


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    state = tmp_path / "state"
    monkeypatch.setenv("LAB_STATE_DIR", str(state))
    monkeypatch.setenv("LAB_ACTOR", "tester")
    monkeypatch.delenv("LAB_TIER", raising=False)
    monkeypatch.delenv("LAB_TASK", raising=False)
    return state


def run(capsys, *argv: str):
    code = lab_cli.main(["gpu-power", *argv])
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1, out
    return code, json.loads(out[0])


def test_the_verb_is_registered_alphabetically():
    names = list(lab_cli.VERBS)
    assert lab_cli.VERBS["gpu-power"] == "lab.machine.gpu_power:main"
    assert names.index("experiment") < names.index("gpu-power") < names.index("handoff")


def test_summary_reads_the_samples_and_writes_nothing(capsys, isolated_state):
    code, answer = run(capsys, "summary", "--samples", str(POD_B))
    assert code == 0
    summary = answer["summary"]
    assert (summary["node"], summary["samples"], summary["distinct_updates"], summary["stale_samples"],
            summary["mean_w"], summary["error"]) == ("gpu-b", 25, 4, 12, 220.52, "unknown")
    assert not isolated_state.exists()


def test_record_stores_the_idle_window_as_a_low_confidence_pod_a_row_carrying_how_it_was_measured(capsys, isolated_state):
    code, answer = run(capsys, "record", "--samples", str(POD_A), "--node", "pod-a", "--metric", "pod.idle_watts",
                       "--state", "idle", *IDLE_WINDOW)
    assert (code, answer["outcome"]) == (0, "recorded")
    [row] = ledger.read(state_dir=isolated_state)
    assert row["seq"] == answer["seq"] == 1
    assert (row["verb"], row["outcome"], row["actor"], row["node"]) == ("gpu-power record", "ok", "tester", "pod-a")
    assert (row["metrics"], row["units"], row["low_confidence"]) == ({"pod.idle_watts": 39.8}, {"pod.idle_watts": "W"},
                                                                      True)
    assert row["source"] == "lab/pod_power.py 1 samples from samples_gpu-a.jsonl"
    assert row["conditions"] == {
        "node": "gpu-a", "start": "2026-10-10T18:00:00.200Z", "end": "2026-10-10T18:00:14.200Z", "duration_s": 14.0,
        "samples": 15, "readings": 15, "missing": 0, "distinct_updates": 14, "mean_w": 39.8, "min_w": 39.0,
        "max_w": 41.0, "sampler": "lab/pod_power.py 1", "source": "amd-smi", "power_field": "socket_power",
        "period_ms": 1000.0, "measured_period_ms": 1000.0, "max_gap_ms": 1020.0, "restarts": 0,
        "max_receive_gap_ms": None, "max_receive_delay_s": None,
        "start_temps_c": {"edge": 34.0, "hotspot": 37.0, "mem": 44.0},
        "end_temps_c": {"edge": 38.0, "hotspot": 41.0, "mem": 48.0}, "under_load_samples": 0, "stale_samples": 0,
        "counter_ok": True, "energy_j": 557.52, "energy_source": "avg_power_integral", "error": "unknown",
        "state": "idle"}


def test_a_window_spanning_the_restart_shows_it_in_the_rows_measured_period_and_longest_gap(capsys, isolated_state):
    code, _ = run(capsys, "record", "--samples", str(POD_A), "--node", "gpu-a", "--metric", "pod.mixed_watts",
                  "--since", "2026-10-10T18:00:40Z")
    assert code == 0
    [row] = ledger.read(state_dir=isolated_state)
    conditions = row["conditions"]
    assert (row["node"], conditions["restarts"], conditions["max_gap_ms"], conditions["missing"]) == (
        "pod-a", 1, 6400.0, 0)
    assert conditions["measured_period_ms"] > conditions["period_ms"]


def test_metrics_file_gets_the_same_reading_in_its_own_format(capsys, isolated_state, tmp_path):
    target = tmp_path / "metrics.jsonl"
    code, answer = run(capsys, "record", "--samples", str(POD_A), "--node", "pod-a", "--metric", "pod.idle_watts",
                       *IDLE_WINDOW, "--metrics-file", str(target))
    assert code == 0
    [stored] = metrics.load(target)
    assert stored == answer["metrics_row"]
    assert (stored["metric"], stored["value"], stored["unit"], stored["node"], stored["ts"], stored["low_confidence"]) == (
        "pod.idle_watts", 39.8, "W", "pod-a", "2026-10-10T18:00:14.200000Z", True)
    assert stored["conditions"]["error"] == "unknown"
    assert stored["conditions"]["sampler"] == "lab/pod_power.py 1"
    assert metrics.latest("pod.idle_watts", "pod-a", path=target)["value"] == 39.8


def test_dry_run_prints_the_row_and_writes_neither_the_ledger_nor_the_metrics_file(capsys, isolated_state, tmp_path):
    target = tmp_path / "metrics.jsonl"
    code, answer = run(capsys, "record", "--samples", str(POD_A), "--node", "pod-a", "--metric", "pod.idle_watts",
                       *IDLE_WINDOW, "--metrics-file", str(target), "--dry-run")
    assert (code, answer["outcome"]) == (0, "dry_run")
    assert answer["row"]["metrics"] == {"pod.idle_watts": 39.8}
    assert answer["metrics_row"]["node"] == "pod-a"
    assert not isolated_state.exists() and not target.exists()


def test_a_window_without_a_power_reading_or_without_samples_records_nothing(capsys, isolated_state, tmp_path):
    lines = POD_A.read_text(encoding="utf-8").splitlines()
    failed_only = tmp_path / "failed.jsonl"
    failed_only.write_text(next(line for line in lines if '"socket_w":null' in line) + "\n", encoding="utf-8")
    code, answer = run(capsys, "record", "--samples", str(failed_only), "--node", "pod-a", "--metric", "pod.idle_watts")
    assert (code, answer["outcome"]) == (1, "no_readings")

    code, answer = run(capsys, "record", "--samples", str(POD_A), "--node", "pod-b", "--metric", "pod.idle_watts")
    assert (code, answer["outcome"], answer["error"]) == (
        1, "error", "no telemetry.gpu.power samples for gpu-b in the window")
    assert not isolated_state.exists()


def test_a_node_that_is_not_a_pod_is_a_usage_error(capsys):
    assert lab_cli.main(["gpu-power", "record", "--samples", str(POD_A), "--node", "jetson",
                         "--metric", "pod.idle_watts"]) == 2
    assert "'jetson' is not a pod" in capsys.readouterr().err


def test_summary_reads_a_jetson_log_copy_whose_pod_samples_carry_the_pod_names_today(capsys, tmp_path):
    # since telemetry ruling 3 the Jetson's pod reader logs a pod's samples as pod-a and pod-b, while the pod's own
    # files keep the sampler's gpu-a and gpu-b: a Jetson-log copy must still be read under either name
    log = tmp_path / "events.jsonl"
    lines = []
    for line in POD_A.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("node") == "gpu-a":
            event["node"] = "pod-a"
        lines.append(json.dumps(event))
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    want = run(capsys, "summary", "--samples", str(POD_A), "--node", "pod-a", *IDLE_WINDOW)
    for node in ("pod-a", "gpu-a"):
        assert run(capsys, "summary", "--samples", str(log), "--node", node, *IDLE_WINDOW) == want
