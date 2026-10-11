"""The optimizer's runners against the fakes: one fake vLLM spawn per inference trial, one fake trainer per
training trial, each reading its settings the way the real process does."""
from __future__ import annotations

import json

from lab import bench, optimize_fake, optimize_runners as runners, optimize_space as osp

SCREEN = ["--batches", "1", "--no-64", "--repeats", "1", "--max-tokens", "8", "--requests-per-stream", "3",
          "--warmup", "1", "--no-probe"]


def test_the_fake_server_reads_the_r35_mapping_back_from_engine_args_and_env():
    space = osp.load_space(osp.DATA_DIR / "searchspace-inference-gfx1151.json")
    params = {**space.baseline, "linear_kernel": "exllama_fp16", "num_speculative_tokens": 2,
              "cudagraph_mode": "NONE", "gpu_max_hw_queues": 1, "enable_prefix_caching": False}
    launch = space.launch(params)
    read = optimize_fake.engine_settings(optimize_fake.parse_args_list(launch.args), launch.environ({}))
    assert read == params
    assert optimize_fake.decode_rate(read) > optimize_fake.decode_rate(space.baseline)


def test_one_inference_trial_is_one_spawn_measured_by_bench_and_a_slow_one_stops_after_one_request(tmp_path):
    space = osp.load_space(osp.SAMPLE_SPACE)
    runner = runners.fake_inference_runner(tmp_path / "work", {"screen": SCREEN})
    m = runner.run(trial=0, launch=space.launch(space.baseline), fidelity="screen", workdir=tmp_path / "t0",
                   restart="cold")
    assert m.outcome == "ok", m.reason
    assert {"decode_tok_s.b1", "joules_per_token.b1", "ttft_s.b1"} <= set(m.metrics)
    assert len(m.samples["decode_tok_s.b1"]) == 3 and len(m.conditions["bench"]["runs"]) == 1
    assert "--dtype" in m.conditions["command"] and "VLLM_DISABLED_KERNELS" in m.conditions["unset_env"]
    stopped = runner.run(trial=1, launch=space.launch(space.baseline), fidelity="screen", workdir=tmp_path / "t1",
                         restart="cold", stop_below=1e9)
    assert stopped.outcome == "early_stopped" and len(stopped.samples["decode_tok_s.b1"]) == 1


def test_the_training_runner_drives_the_fake_trainer_through_a_settings_file(tmp_path):
    space = osp.load_space(osp.DATA_DIR / "searchspace-training-gfx1151.json")
    runner = runners.fake_training_runner(steps=12, exclude_first_steps=3)
    m = runner.run(trial=0, launch=space.launch(space.baseline), fidelity="screen", workdir=tmp_path / "t0")
    assert m.outcome == "ok", m.reason
    assert len(m.samples["samples_per_s"]) == 9 and m.metrics["peak_mem_gb"] > 0
    settings = json.loads((tmp_path / "t0" / "settings.json").read_text(encoding="utf-8"))
    assert settings["gradient_accumulation_steps"] == 4
    faster = runner.run(trial=1, launch=space.launch({**space.baseline, "gdn_kernel": "fla",
                                                       "blas_backend": "hipblaslt"}),
                        fidelity="screen", workdir=tmp_path / "t1")
    assert faster.metrics["samples_per_s"] > m.metrics["samples_per_s"]
    oom = runner.run(trial=2, launch=space.launch({**space.baseline, "per_device_train_batch_size": 16,
                                                    "gradient_checkpointing": False}),
                     fidelity="screen", workdir=tmp_path / "t2")
    assert oom.outcome == "fail" and oom.reason == "out of memory"
    slow = runners.fake_training_runner(steps=200, exclude_first_steps=3, time_scale=0.02)
    early = slow.run(trial=3, launch=space.launch(space.baseline), fidelity="screen", workdir=tmp_path / "t3",
                     stop_below=1e9)
    assert early.outcome == "early_stopped" and len(early.samples["samples_per_s"]) == 4


def test_the_condition_the_harness_writes_for_unset_variables_cannot_be_told(tmp_path):
    """lab/bench.py records a spawn's unset variables as env_unset itself, so --condition refuses that name like
    every other condition the harness fills."""
    answer = bench.main(["run", "--engine", "llamacpp", "--checkpoint", "q4.gguf", "--out", str(tmp_path / "out"),
                         "--condition", "env_unset=HSA_ENABLE_SDMA"])
    assert answer["ok"] is False and "env_unset" in answer["error"]
    assert not (tmp_path / "out").exists()
