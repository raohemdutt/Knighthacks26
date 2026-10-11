"""The night configuration's rules (lab/optimize_plan.py): restart classes, grouping, early stop, observations,
the confirmation verdict and the minute budget."""
from __future__ import annotations

import pytest

from lab import optimize_plan as rules, optimize_space as osp


@pytest.fixture(scope="module")
def lanes():
    night = osp.load_night()
    return osp.lane_of(night, "inference"), osp.lane_of(night, "training")


def test_a_server_is_warm_only_when_the_worker_last_ran_the_same_compile_class(lanes):
    inference, training = lanes
    space = osp.load_space(osp.DATA_DIR / "searchspace-inference-gfx1151.json")
    base = dict(space.baseline)
    assert rules.restart_class(inference, None, base) == "cold"
    last = rules.compile_class(inference, base)
    assert rules.restart_class(inference, last, {**base, "gpu_max_hw_queues": 1}) == "warm"
    assert rules.restart_class(inference, last, {**base, "linear_kernel": "exllama_fp16"}) == "cold"
    assert rules.restart_class(training, None, {}) == "process"


def test_every_second_ask_of_a_worker_fixes_its_last_compile_class(lanes):
    inference, training = lanes
    last = {"linear_kernel": "exllama_fp16"}
    assert [rules.fixed_for_ask(inference, n, last) for n in range(4)] == [None, last, None, last]
    assert rules.fixed_for_ask(inference, 1, None) is None
    assert rules.fixed_for_ask(training, 1, last) is None


def test_the_minute_budget_fits_the_configurations_own_estimate(lanes):
    inference, training = lanes
    space = osp.load_space(osp.DATA_DIR / "searchspace-inference-gfx1151.json")
    plan = rules.plan_minutes(inference, workers=1, minutes=180, budget=None,
                              initial=osp.initial_arms(inference, space))
    assert plan["fit"] == inference.budget["trials_per_worker_est"] == 17
    assert plan["restarts"] == {"cold": 10, "warm": 7} and plan["minutes_used"] <= 180
    assert rules.plan_minutes(inference, workers=2, minutes=180, budget=20,
                              initial=osp.initial_arms(inference, space))["fit"] == 20
    tspace = osp.load_space(osp.DATA_DIR / "searchspace-training-gfx1151.json")
    tplan = rules.plan_minutes(training, workers=1, minutes=180, budget=None,
                               initial=osp.initial_arms(training, tspace))
    assert tplan["fit"] == training.budget["trials_per_worker_est"] == 15


def test_early_stop_thresholds_and_the_sem_an_early_stopped_trial_reports(lanes):
    inference, training = lanes
    assert rules.early_threshold(inference, [2.8, 3.0, 2.9]) == pytest.approx(1.45)
    assert rules.early_threshold(training, [1.0, 4.0, 4.0, 6.0, 6.0, 9.0]) == pytest.approx(2.5)
    assert rules.early_threshold(inference, None) is None
    obs = rules.observation(inference, {"decode_tok_s.b1": 1.0, "joules_per_token.b1": 100.0},
                            {"decode_tok_s.b1": [1.0]}, early=True, baseline_sd=0.1)
    assert obs["decode_tok_s.b1"] == (1.0, pytest.approx(0.2))
    assert obs["joules_per_token.b1"] == (100.0, pytest.approx(20.0))


def test_an_observation_is_the_median_with_the_median_sem_and_jpt_takes_the_relative_sem(lanes):
    inference, _ = lanes
    obs = rules.observation(inference, {"decode_tok_s.b1": 3.0, "joules_per_token.b1": 40.0, "ttft_s.b1": 0.5},
                            {"decode_tok_s.b1": [2.9, 3.0, 3.1]})
    sem = rules.MEDIAN_SEM_FACTOR * 0.1 / 3 ** 0.5
    assert obs["decode_tok_s.b1"] == (3.0, pytest.approx(sem))
    assert obs["joules_per_token.b1"][1] == pytest.approx(40.0 * sem / 3.0)
    assert obs["ttft_s.b1"] == (0.5, None)
    assert rules.feasible(inference, {"decode_tok_s.b1": 3.0, "ttft_s.b1": 0.5})
    assert not rules.feasible(inference, {"decode_tok_s.b1": 2.0, "ttft_s.b1": 0.5})


def test_the_confirmation_order_and_verdict(lanes):
    inference, _ = lanes
    assert rules.confirmation_order(inference) == ["best", "baseline"] * 3
    v = rules.confirm_verdict(inference, [41.0, 42.0, 41.5], [30.0, 30.5, 29.8])
    assert v["best_ahead"] and v["low_confidence"] and v["p_one_sided"] < 0.05
    assert rules.confirm_verdict(inference, [41.0], [30.0])["label"] == "too_early"
