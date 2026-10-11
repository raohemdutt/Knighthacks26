"""The settings optimizer reads both search-space formats on main as they are, and the night configuration."""
from __future__ import annotations

import json

import pytest

from lab import optimize_space as osp

DATA = osp.DATA_DIR


def test_the_r35_inference_space_reads_as_it_is_and_launches_through_its_mapping(tmp_path):
    space = osp.load_space(DATA / "searchspace-inference-gfx1151.json")
    assert space.format == osp.R35_SCHEMA and space.kind == "inference"
    assert len(space.names) == 10 and space.names[0] == "linear_kernel"
    assert space.fixed["max_model_len"] == 16384
    base = space.launch(space.baseline, run_dir=tmp_path)
    assert base.args[:2] == ["--served-model-name", "teacher"]
    assert "--dtype" in base.args and base.args[base.args.index("--dtype") + 1] == "bfloat16"
    assert base.env["GPU_MAX_HW_QUEUES"] == "4" and base.env["TRITON_CACHE_DIR"] == f"{tmp_path}/triton-cache"
    assert "VLLM_DISABLED_KERNELS" in base.unset_env
    trial = space.launch({**space.baseline, "linear_kernel": "exllama_fp16", "cudagraph_mode": "NONE"})
    assert trial.env["VLLM_DISABLED_KERNELS"] == "ConchLinearKernel"
    assert "VLLM_DISABLED_KERNELS" not in trial.unset_env and "--enforce-eager" in trial.args
    assert trial.environ({"VLLM_DISABLED_KERNELS": "x", "HSA_ENABLE_SDMA": "0", "KEEP": "1"}) == {
        **trial.env, "KEEP": "1"}
    assert space.problems_with({**space.baseline, "gpu_max_hw_queues": 3}) == [
        "gpu_max_hw_queues=3 is outside the space"]


def test_the_r36_training_space_reads_as_it_is_with_derived_values_invalid_combinations_and_env():
    space = osp.load_space(DATA / "searchspace-training-gfx1151.json")
    assert space.format == osp.AX_CONFIG and space.kind == "training"
    assert "gradient_accumulation_steps" not in space.names and len(space.names) == 10
    base = space.launch(space.baseline)
    assert base.settings["gradient_accumulation_steps"] == 4 and base.settings["pinned"]["seed"] == 42
    assert base.env == {"MIOPEN_FIND_MODE": "NORMAL"} and "GPU_MAX_HW_QUEUES" in base.unset_env
    trial = space.launch({**space.baseline, "per_device_train_batch_size": 8, "tunableop": True,
                          "blas_backend": "hipblaslt"})
    assert trial.settings["gradient_accumulation_steps"] == 2
    assert trial.env["PYTORCH_TUNABLEOP_ENABLED"] == "1" and trial.env["PYTORCH_TUNABLEOP_TUNING"] == "0"
    assert trial.env["TORCH_BLAS_PREFER_HIPBLASLT"] == "1"
    bad = space.problems_with({**space.baseline, "packing": True})
    assert any("cu_seqlens" in b for b in bad)


def test_a_space_in_neither_format_or_with_a_bad_baseline_lists_every_problem():
    with pytest.raises(osp.SpaceError, match="neither format"):
        osp.parse_space({"parameters": [{"name": "x"}]})
    data = json.loads((DATA / "searchspace-sample.json").read_text(encoding="utf-8"))
    data["baseline"]["gpu_max_hw_queues"] = 2
    del data["mapping"]["linear_kernel"]["exllama_fp16"]
    with pytest.raises(osp.SpaceError) as caught:
        osp.parse_space(data)
    assert any("gpu_max_hw_queues=2 is outside" in p for p in caught.value.problems)
    assert any("mapping.linear_kernel has no entry for exllama_fp16" in p for p in caught.value.problems)


def test_derived_expressions_are_plain_arithmetic_only():
    assert osp.evaluate("16 / b", {"b": 4}) == 4
    with pytest.raises(osp.SpaceError):
        osp.evaluate("__import__('os')", {})


def test_the_night_configuration_reads_its_lanes_workers_constraints_and_initial_arms():
    night = osp.load_night()
    inference = osp.lane_of(night, "inference")
    assert inference.metric == "joules_per_token.b1" and inference.minimize
    texts = [c.text() for c in inference.constraints]
    assert "decode_tok_s.b1 >= 2.88" in texts and "ttft_s.b1 <= 2" in texts
    assert night.worker("pod-d").available and not night.worker("pod-e").available
    space = osp.load_space(DATA / "searchspace-inference-gfx1151.json")
    assert osp.check_lane(inference, space) == []
    arms = osp.initial_arms(inference, space)
    assert [a["name"] for a in arms] == ["baseline", "hw-queues-1", "exllama", "exllama-mtp1", "mtp1"]
    assert arms[0]["parameters"] == space.baseline
    assert arms[3]["parameters"]["num_speculative_tokens"] == 1 and arms[3]["parameters"]["gpu_max_hw_queues"] == 4
    training = osp.lane_of(night, "training")
    tspace = osp.load_space(DATA / "searchspace-training-gfx1151.json")
    assert osp.check_lane(training, tspace) == []
    assert [a["name"] for a in osp.initial_arms(training, tspace)][0] == "baseline"
    assert training.constraints == () and len(training.constraint_notes) == 2


def test_an_initial_arm_outside_the_space_is_refused():
    night = osp.load_night()
    raw = json.loads(json.dumps(night.raw))
    raw["lanes"]["inference"]["initial_arms"].append({"name": "bad", "parameters": {"gpu_max_hw_queues": 3}})
    space = osp.load_space(DATA / "searchspace-inference-gfx1151.json")
    with pytest.raises(osp.SpaceError, match="initial arm bad"):
        osp.initial_arms(osp.lane_of(osp.parse_night(raw), "inference"), space)


def test_the_committed_sample_reads_in_r35_format():
    space = osp.load_space(osp.SAMPLE_SPACE)
    assert space.kind == "inference" and space.names == ["linear_kernel", "num_speculative_tokens",
                                                          "gpu_max_hw_queues"]
