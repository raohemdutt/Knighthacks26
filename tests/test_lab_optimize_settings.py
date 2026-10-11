"""The settings optimizer end to end on the fakes: asynchronous trials across workers, the initial arms first,
one JSON file per trial and a ledger row, resume without re-running, confirmation, summary and the dry run."""
from __future__ import annotations

import json

import pytest

from lab import optimize_ax, optimize_settings as opt, optimize_space as osp
from lab.machine import ledger

SCREEN = ["--batches", "1", "--no-64", "--repeats", "1", "--max-tokens", "8", "--requests-per-stream", "3",
          "--warmup", "1", "--no-probe"]


@pytest.fixture
def config(tmp_path, monkeypatch):
    """The night configuration on main, cut down to the three-parameter sample and quick fidelities."""
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    raw = json.loads(osp.NIGHT_CONFIG.read_text(encoding="utf-8"))
    lane = raw["lanes"]["inference"]
    lane["search_space"] = "lab/refine/research/kernels/data/searchspace-sample.json"
    lane["parameters"] = ["linear_kernel", "num_speculative_tokens", "gpu_max_hw_queues"]
    lane["grouping"]["class_keys"] = ["linear_kernel", "num_speculative_tokens"]
    lane["fidelity"] = {"screen": {"bench_args": SCREEN}, "confirm": {"bench_args": SCREEN}}
    lane["confirmation"]["spawns_per_arm"] = 1
    train = raw["lanes"]["training"]
    train["fidelity"]["screen"].update(steps=8, exclude_first_steps=2)
    path = tmp_path / "night.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def run(config, tmp_path, *extra):
    return opt.main(["run", "--fake", "--strategy", "stub", "--config", str(config), "--run-dir",
                     str(tmp_path / "run"), "--work-dir", str(tmp_path / "work"), *extra])


def files(tmp_path):
    return {p.name: p.read_text(encoding="utf-8") for p in sorted((tmp_path / "run" / "trials").glob("trial-*.json"))}


def test_trials_run_asynchronously_from_the_initial_arms_and_resume_without_re_running(config, tmp_path):
    first = run(config, tmp_path, "--workers", "2", "--budget", "6", "--no-confirm")
    assert first["ok"] and first["stop_reason"] == "budget", first
    trials = [json.loads(t) for t in files(tmp_path).values()]
    assert [t["arm"] for t in trials[:5]] == ["baseline", "hw-queues-1", "exllama", "exllama-mtp1", "mtp1"]
    assert trials[0]["source"] == "baseline" and trials[1]["priors"] == ["P3"] and trials[5]["source"] == "model"
    assert {t["worker"] for t in trials} == {"fake-0", "fake-1"}
    assert all(t["status"] == "completed" and t["restart"] in ("cold", "warm") for t in trials)
    t2 = trials[2]
    assert t2["launch"]["env"]["VLLM_DISABLED_KERNELS"] == "ConchLinearKernel"
    assert len(t2["measurement"]["samples"]["decode_tok_s.b1"]) == 3
    assert t2["observation"]["joules_per_token.b1"][1] is not None and t2["git_sha"]
    assert t2["conditions"]["versions"]["python"] and t2["started"] and t2["ended"]
    rows = [r for r in ledger.read() if r["verb"] == "optimize.trial"]
    assert len(rows) == 6 and rows[0]["refs"]["run_id"].endswith("trial-0000")
    assert (tmp_path / "run" / "summary.md").read_text(encoding="utf-8").count("| completed |") == 6

    before = files(tmp_path)
    second = run(config, tmp_path, "--workers", "2", "--budget", "7")
    assert second["ok"] and second["resumed_with"]["completed"] == 6, second
    after = files(tmp_path)
    assert {k: after[k] for k in before} == before and len(after) == 7
    confirms = sorted((tmp_path / "run" / "confirm").glob("confirm-*.json"))
    assert [json.loads(p.read_text())["arm"] for p in confirms] == ["best", "baseline"]
    summary = json.loads((tmp_path / "run" / "summary.json").read_text(encoding="utf-8"))
    assert summary["best"]["parameters"]["linear_kernel"] == "exllama_fp16"
    assert summary["confirmation"]["runs_per_arm"] == [1, 1] and summary["confirmation"]["low_confidence"]
    assert summary["still_owed"] and "R21" in summary["still_owed"][0]

    refused = run(config, tmp_path, "--seed", "5", "--budget", "8")
    assert refused["outcome"] == "refused" and "seed" in refused["error"]


def test_a_trial_in_flight_at_a_stop_is_run_again_and_the_finished_ones_are_not(config, tmp_path):
    assert run(config, tmp_path, "--workers", "1", "--budget", "3", "--no-confirm")["ok"]
    path = tmp_path / "run" / "trials" / "trial-0002.json"
    t = json.loads(path.read_text(encoding="utf-8"))
    t.update(status="running", measurement=None, observation=None, ended=None)
    path.write_text(json.dumps(t), encoding="utf-8")
    before = files(tmp_path)
    again = run(config, tmp_path, "--workers", "1", "--budget", "3", "--no-confirm")
    assert again["ok"] and again["resumed_with"]["running"] == 1
    after = files(tmp_path)
    assert after["trial-0000.json"] == before["trial-0000.json"] and after["trial-0001.json"] == before["trial-0001.json"]
    assert json.loads(after["trial-0002.json"])["status"] == "completed"


def test_the_minute_budget_stops_a_run_before_a_trial_that_would_not_fit(config, tmp_path):
    out = run(config, tmp_path, "--workers", "1", "--minutes", "0", "--no-confirm")
    assert out["ok"] and out["stop_reason"] == "minutes" and out["trials"]["total"] == 0


def test_training_runs_the_trainer_and_an_invalid_suggestion_is_recorded_not_run(config, tmp_path, monkeypatch):
    real = optimize_ax.make_strategy

    class OneBad(optimize_ax.StubStrategy):
        sent = False

        def next(self, fixed=None):
            if not OneBad.sent:
                OneBad.sent = True
                bad = {**self.space.baseline, "packing": True}
                return self.attach(bad), bad
            return super().next(fixed)

    monkeypatch.setattr(opt, "make_strategy", lambda kind, space, lane, **kw: OneBad(space, lane, seed=0)
                        if kind == "stub" else real(kind, space, lane, **kw))
    out = run(config, tmp_path, "--lane", "training", "--workers", "1", "--budget", "7", "--no-confirm")
    assert out["ok"], out
    trials = [json.loads(t) for t in files(tmp_path).values()]
    assert [t["arm"] for t in trials[:5]] == ["baseline", "batch-8", "no-checkpointing", "hipblaslt", "fla"]
    assert trials[5]["status"] == "invalid" and "cu_seqlens" in trials[5]["reason"]
    assert trials[5]["measurement"] is None and trials[6]["status"] in ("completed", "failed")
    base = trials[0]
    assert base["restart"] == "process" and base["launch"]["settings"]["gradient_accumulation_steps"] == 4
    assert len(base["measurement"]["samples"]["samples_per_s"]) == 6
    fla = trials[4]
    assert fla["observation"]["samples_per_s"][0] > base["observation"]["samples_per_s"][0]


def test_the_dry_run_prints_the_plan_restarts_and_the_trials_that_fit(capsys):
    out = opt.main(["--dry-run"])
    assert out["ok"] and out["outcome"] == "dry_run"
    plan = out["plan"]
    assert plan["fit"] == 17 and plan["restarts"] == {"cold": 10, "warm": 7}
    assert [a["name"] for a in plan["initial_arms"]][0] == "baseline"
    assert "17 trials fit" in capsys.readouterr().err
    training = opt.main(["--dry-run", "--lane", "training", "--minutes", "120"])["plan"]
    assert training["fit"] == 10 and training["minutes"] == 120


def test_a_real_run_needs_a_pod_that_passes_the_node_gate(config, tmp_path):
    out = opt.main(["run", "--config", str(config), "--node", "pod-d", "--minutes", "1", "--strategy", "stub",
                    "--run-dir", str(tmp_path / "run")])
    assert out["outcome"] == "refused" and "node gate" in out["error"]
    assert opt.main(["run", "--fake", "--node", "pod-d", "--config", str(config), "--strategy", "stub",
                     "--run-dir", str(tmp_path / "run")])["outcome"] == "refused"


def test_the_requirements_pin_ax_and_botorch_outside_pyproject():
    pins = opt.pins()
    assert set(pins) == {"ax-platform", "botorch"} and pins["ax-platform"] == "1.3.1"
    assert "ax-platform" not in (opt.REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert opt.install(dry_run=True)["argv"][-2:] != ["-r", ""]


@pytest.mark.skipif(not optimize_ax.ax_installed(), reason="Ax is not installed (lab/requirements-optimize.txt)")
def test_ax_takes_the_initial_arms_and_suggests_with_fixed_parameters():
    night = osp.load_night()
    lane = osp.lane_of(night, "inference")
    space = osp.load_space(osp.SAMPLE_SPACE)
    s = optimize_ax.make_strategy("ax", space, lane, generation_strategy=night.generation_strategy, seed=49)
    for arm in osp.initial_arms(lane, space):
        i = s.attach(arm["parameters"], baseline=arm["attach_as"] == "baseline")
        rate = 3.0 * (2 if arm["parameters"]["linear_kernel"] == "exllama_fp16" else 1)
        s.complete(i, {"joules_per_token.b1": (120 / rate, 0.5), "decode_tok_s.b1": (rate, 0.05),
                       "ttft_s.b1": (0.5, None), "watts_avg.b1": (120.0, None)})
    index, params = s.next({"linear_kernel": "exllama_fp16"})
    assert index == 5 and params["linear_kernel"] == "exllama_fp16"
