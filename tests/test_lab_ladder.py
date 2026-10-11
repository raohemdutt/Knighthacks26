"""Tests for the model ladder (lab/ladder.py): the committed student candidates, the pick for one chip and the
two-chip pick through the placement plan (lab/placement.py).

Capacity records come from the probe on the fake chip client (the R33 sample sheet); nothing reaches a pod, the
hub or a GPU.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from lab import capacity, ladder, placement
from lab.capacity_fake import FakeChipClient

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS = REPO_ROOT / "lab" / "refine" / "metrics.jsonl"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LAB_ENV_FILES", "")
    return tmp_path


def record(node: str = "pod-b", free: float | None = None, tflops: float | None = None) -> dict:
    """A capacity record from the fake (the sample: 58 GiB free of 64), with free memory or bf16 rate replaced."""
    rec = capacity.probe(FakeChipClient(), node=node)
    if free is not None:
        rec["free_gib"] = {"value": free, "unit": "GiB", "source": "test", "kind": "recorded"}
    if tflops is not None:
        rec["bf16_tflops"] = {"value": tflops, "unit": "TFLOPS", "source": "test", "kind": "measured"}
    return rec


# --- the committed ladder ---------------------------------------------------------------------------------------

def test_ladder_starts_at_the_student_and_climbs():
    models = [c.model for c in ladder.LADDER]
    assert models[0] == "Qwen/Qwen3.5-4B"
    assert models == ["Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-27B"]
    sizes = [c.params_b for c in ladder.LADDER]
    assert sizes == sorted(sizes)


def test_every_rung_says_measured_or_predicted_and_names_its_source():
    for c in ladder.LADDER:
        for num in (c.step_mem, c.samples_per_s):
            assert num.kind in ("measured", "predicted")
            assert num.source
        assert c.size_source
    first = ladder.LADDER[0]
    assert first.step_mem.kind == "measured" and first.samples_per_s.kind == "measured"
    assert all(c.step_mem.kind == "predicted" for c in ladder.LADDER[1:])


def test_the_measured_rung_matches_the_metrics_file():
    first = ladder.LADDER[0]
    peak = placement.metric("trainer.peak_mem_gb", where={"model": "Qwen/Qwen3.5-4B"}, path=METRICS)
    rate = placement.metric("trainer.samples_per_s", where={"model": "Qwen/Qwen3.5-4B"}, path=METRICS)
    assert first.step_mem.value == peak.value == 8.7
    assert first.samples_per_s.value == rate.value == 0.51
    assert "metrics.jsonl:14" in first.step_mem.source and "metrics.jsonl:13" in first.samples_per_s.source


def test_predicted_rungs_scale_by_parameter_count():
    nine = ladder.candidate("Qwen/Qwen3.5-9B")
    assert nine.step_mem.value == pytest.approx(8.7 * 9 / 4, abs=1e-3)
    assert nine.samples_per_s.value == pytest.approx(0.51 * 4 / 9, abs=1e-3)


def test_settings_are_the_pinned_baseline_of_the_search_space():
    settings = ladder.settings()
    assert settings["per_device_train_batch_size"] == 4
    assert settings["gradient_accumulation_steps"] == 4
    assert settings["gradient_checkpointing"] is True
    assert settings["max_length"] == 256
    assert settings["lora_r"] == 16


# --- run-time arithmetic ----------------------------------------------------------------------------------------

def test_run_time_is_samples_times_epochs_over_rate():
    assert ladder.run_seconds(1000, 3, 0.51) == pytest.approx(1000 * 3 / 0.51)
    assert ladder.run_seconds(1000, 3, 0.51) == pytest.approx(5882.353, abs=1e-3)
    with pytest.raises(ValueError):
        ladder.run_seconds(1000, 1, 0.0)
    with pytest.raises(ValueError):
        ladder.run_seconds(0, 1, 0.5)


def test_memory_need_is_step_peak_plus_margin_plus_host_reserve():
    rules = placement.Rules()
    four = ladder.candidate("Qwen/Qwen3.5-4B")
    assert ladder.need_gib(four, rules) == pytest.approx(8.7 * 1.5 + 4.0)
    assert ladder.need_gib(four, placement.Rules(margin=0.0, host_reserve_gib=0.0)) == pytest.approx(8.7)


def test_rate_scales_with_the_chips_bf16_rate_only_when_both_are_known():
    four = ladder.candidate("Qwen/Qwen3.5-4B")
    rate, why = ladder.rate_on(four, record(tflops=40.0), reference_tflops=20.0)
    assert rate == pytest.approx(1.02)
    assert "40" in why and "20" in why
    rate, why = ladder.rate_on(four, record(), reference_tflops=20.0)       # the sample: rate unknown
    assert rate == pytest.approx(0.51)
    assert "unknown" in why


# --- the pick for one chip --------------------------------------------------------------------------------------

def test_pick_for_a_64_gb_record():
    answer = ladder.pick([record()], samples=2000, epochs=2)
    assert answer["ok"] is True and answer["outcome"] == "picked"
    assert answer["model"] == "Qwen/Qwen3.5-9B"            # 27B needs about 92 GiB, more than the 58 free
    assert answer["node"] == "pod-b"
    assert answer["run_s"] == pytest.approx(2000 * 2 / (0.51 * 4 / 9), rel=1e-3)
    assert answer["reason"].count(".") >= 1 and "Qwen/Qwen3.5-27B" in answer["reason"]
    rows = {row["model"]: row for row in answer["options"]}
    assert rows["Qwen/Qwen3.5-27B"]["fits"] is False
    assert rows["Qwen/Qwen3.5-9B"]["fits"] is True and rows["Qwen/Qwen3.5-4B"]["fits"] is True


def test_pick_for_a_small_record():
    answer = ladder.pick([record(free=24.0)], samples=500, epochs=1)
    assert answer["model"] == "Qwen/Qwen3.5-4B"
    assert answer["run_s"] == pytest.approx(500 / 0.51)


def test_nothing_fits_a_tiny_record_and_says_why():
    answer = ladder.pick([record(free=10.0)], samples=500, epochs=1)
    assert answer["ok"] is False and answer["outcome"] == "none_fits"
    assert answer["model"] is None
    assert "17.05" in answer["reason"] and "10" in answer["reason"]


def test_a_deadline_drops_the_rungs_that_finish_late():
    four_s = 2000 * 2 / 0.51
    answer = ladder.pick([record()], samples=2000, epochs=2, within_s=four_s + 1)
    assert answer["model"] == "Qwen/Qwen3.5-4B"
    rows = {row["model"]: row for row in answer["options"]}
    assert rows["Qwen/Qwen3.5-9B"]["fits"] is True and rows["Qwen/Qwen3.5-9B"]["in_time"] is False


def test_a_record_with_unknown_free_memory_is_refused():
    rec = record()
    rec["free_gib"] = {"value": None, "unit": "GiB", "source": "x", "kind": "unknown"}
    answer = ladder.pick([rec], samples=10, epochs=1)
    assert answer["ok"] is False and answer["outcome"] == "none_fits"


# --- two chips, through the placement plan ----------------------------------------------------------------------

def test_two_chips_run_two_jobs_at_once_and_pick_the_larger():
    answer = ladder.pick([record("pod-a"), record("pod-b")], samples=2000, epochs=2, teacher_share=0.0)
    assert answer["outcome"] == "picked"
    assert sorted(answer["placement"]["lora"]) == ["pod-a", "pod-b"]
    jobs = {job["model"]: job["node"] for job in answer["jobs"]}
    assert set(jobs) == {"Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-4B"}
    assert set(jobs.values()) == {"pod-a", "pod-b"}
    assert answer["model"] == "Qwen/Qwen3.5-9B"
    assert "at the same time" in answer["reason"]
    assert answer["placement"]["one_lora_per_chip"] is True


def test_the_teacher_holding_a_chip_leaves_one_job():
    answer = ladder.pick([record("pod-a"), record("pod-b")], samples=2000, epochs=2, teacher_on="pod-a")
    assert answer["placement"]["lora"] == ["pod-b"]
    assert [job["node"] for job in answer["jobs"]] == ["pod-b"]
    assert answer["model"] == "Qwen/Qwen3.5-9B"
    assert any("pod-a" in why for why in answer["placement"]["left_out"])


def test_the_larger_model_goes_where_it_finishes_soonest():
    answer = ladder.pick([record("pod-a", tflops=30.0), record("pod-b", tflops=15.0)], samples=2000, epochs=2,
                         teacher_share=0.0, reference_tflops=15.0)
    jobs = {job["model"]: job for job in answer["jobs"]}
    assert jobs["Qwen/Qwen3.5-9B"]["node"] == "pod-a"
    assert jobs["Qwen/Qwen3.5-9B"]["run_s"] == pytest.approx(2000 * 2 / (0.51 * 4 / 9 * 2), rel=1e-3)
    assert answer["model"] == "Qwen/Qwen3.5-9B" and answer["node"] == "pod-a"


def test_ties_in_quality_go_to_the_one_that_finishes_soonest():
    # only the 4B fits either chip; it runs once, on the faster one
    answer = ladder.pick([record("pod-a", free=24.0, tflops=10.0), record("pod-b", free=24.0, tflops=20.0)],
                         samples=100, epochs=1, teacher_share=0.0, reference_tflops=10.0)
    assert answer["model"] == "Qwen/Qwen3.5-4B" and answer["node"] == "pod-b"


def test_more_than_one_chip_must_be_the_placement_plans_pods():
    with pytest.raises(ValueError):
        ladder.pick([record("pod-a"), record("friend-1")], samples=10, epochs=1)
    with pytest.raises(ValueError):
        ladder.pick([record("pod-b"), record("pod-b")], samples=10, epochs=1)


def test_data_parallel_is_never_assumed():
    answer = ladder.pick([record("pod-a"), record("pod-b")], samples=2000, epochs=2, teacher_share=0.0)
    assert answer["data_parallel"]["assumed"] is False
    assert "R44_REAL_RUN.md" in answer["data_parallel"]["see"]


# --- the command line -------------------------------------------------------------------------------------------

def test_command_line_reads_capacity_records(tmp_path, capsys):
    a = capacity.write_record(record("pod-a"), tmp_path / "a.json")
    b = capacity.write_record(record("pod-b"), tmp_path / "b.json")
    code = ladder.main(["--capacity", str(a), "--capacity", str(b), "--samples", "2000", "--epochs", "2",
                        "--teacher-share", "0", "--json"])
    answer = json.loads(capsys.readouterr().out)
    assert code == 0 and answer["model"] == "Qwen/Qwen3.5-9B"
    assert len(answer["jobs"]) == 2
    assert math.isfinite(answer["run_s"])
