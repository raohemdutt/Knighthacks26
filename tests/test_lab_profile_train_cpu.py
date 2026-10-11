"""lab/profile_train.py end to end on the CPU: `run --cpu` trains a tiny model of the student's architecture (one
Gated DeltaNet layer and one attention layer, built as tests/test_lab_train_cpu.py builds it) for three steps under
the torch profiler in a process of its own, and ingests the trace into the result files.

Skipped where the training stack (lab/requirements-train.txt and torch) is not installed, like the CPU trainer test.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("trl")
pytest.importorskip("datasets")
pytest.importorskip("tokenizers")

from lab import profile_train as pt  # noqa: E402
from tests.test_lab_train_cpu import copy_fixture, text_model  # noqa: E402,F401  (text_model is a fixture)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LAB_ACTOR", "opus-test")
    monkeypatch.delenv("LAB_TASK", raising=False)


def test_a_cpu_profile_run_attributes_the_step_to_the_layers(tmp_path, text_model):  # noqa: F811
    folder, rows = copy_fixture(tmp_path)
    answer = pt.main(["run", "--cpu", "--dataset", str(folder), "--rows", str(rows), "--model", str(text_model),
                      "--steps", "3", "--out", str(tmp_path / "run"), "--results", str(tmp_path / "data"),
                      "--no-ledger", "--", "--batch", "2", "--grad-accum", "1", "--workers", "0"])
    assert answer["ok"] is True, answer
    assert answer["tools"]["torch"]["ok"] is True
    assert answer["tools"]["rocprofv3"] == {"ok": False, "skipped": "--cpu"}
    assert Path(answer["json"]).name.startswith("train-profile-cpu-")
    result = json.loads(Path(answer["json"]).read_text(encoding="utf-8"))
    assert result["kind"] == "cpu" and result["node"] == "cpu" and result["steps_in_trace"] == 3
    for bucket in ("gdn", "attention", "mlp", "lora", "optimizer"):
        assert result["shares"][bucket] > 0, bucket
    assert result["attribution"]["backward_links"] > 0
    assert result["fast_linear_attention"]["present"] is False
    assert result["run"]["samples_per_s"] > 0
    assert result["bandwidth"]["flops_per_step"] > 0
    run = json.loads((tmp_path / "run" / "run.json").read_text(encoding="utf-8"))
    wrapped = json.loads(run["refs"]["layer_ranges"])
    assert wrapped["gdn"] == 1 and wrapped["attention"] == 1 and wrapped["mlp"] == 2 and wrapped["lora"] > 0
    assert (tmp_path / "run" / "torch" / "trace.json").is_file()
    assert folder.is_dir()      # a --cpu rehearsal keeps its data
