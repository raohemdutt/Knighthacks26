"""The night optimization config (lab/refine/research/kernels/data/ax-config-night.json) keeps the format its note documents.

The R45 driver reads this file; these tests hold the keys, the cross-references between its sections and the
trial-budget arithmetic together, so an edit that breaks one of them fails here instead of on a pod at night.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "lab" / "refine" / "research" / "kernels" / "data" / "ax-config-night.json"
NOTE = REPO / "lab" / "refine" / "research" / "kernels" / "fast_optimization.md"

# The keyword arguments of ax.Client.configure_generation_strategy in ax-platform 1.3.1.
AX_GS_ARGS = {
    "method", "initialization_budget", "initialization_random_seed", "initialize_with_center",
    "use_existing_trials_for_initialization", "min_observed_initialization_trials",
    "allow_exceeding_initialization_budget", "torch_device", "simplify_parameter_changes",
}
TOP_LEVEL = {"schema", "decision", "tool", "asynchronous", "window", "generation_strategy", "workers", "driver_host",
             "group_by_state", "lanes", "priors", "expected_ranges", "transfer", "schedule"}
SEARCH_LANES = ("inference", "training")
CLOCK = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


@pytest.fixture(scope="module")
def config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_schema_and_top_level_keys(config: dict) -> None:
    assert config["schema"] == "lab.optimize.night/1"
    assert TOP_LEVEL <= set(config)
    assert config["asynchronous"] is True
    assert config["tool"]["name"] == "ax-platform"


def test_generation_strategy_uses_ax_argument_names(config: dict) -> None:
    gs = config["generation_strategy"]
    ax_keys = {k for k in gs if not k.startswith(("fallback_", "_"))}
    assert ax_keys <= AX_GS_ARGS, ax_keys - AX_GS_ARGS
    assert gs["method"] in {"fast", "quality", "random_search"}
    assert gs["fallback_method"] in {"fast", "quality", "random_search"}


def test_workers_are_well_formed(config: dict) -> None:
    names = [w["name"] for w in config["workers"]]
    assert len(names) == len(set(names))
    lanes = set(config["lanes"]) | {"baseline_attach"}
    for worker in config["workers"]:
        assert worker["enabled"] in (True, "if_precondition", "if_granted")
        assert worker["runner"] in ("local", "hub")
        assert isinstance(worker["exclusive"], bool)
        assert set(worker["lanes"]) <= lanes
        assert worker["enabled"] is True or worker["precondition"]
        for conditional in worker.get("conditional_lanes", []):
            assert conditional["lane"] in lanes and conditional["precondition"]
            assert conditional["plan"] in ("A", "B")
    for lane in config["lanes"].values():
        for key, value in lane.items():
            if key.endswith("worker_names"):
                assert set(value) <= set(names), key


@pytest.mark.parametrize("lane_name", SEARCH_LANES)
def test_restart_classes_partition_the_parameters(config: dict, lane_name: str) -> None:
    lane = config["lanes"][lane_name]
    params = lane["parameters"]
    assert len(params) == len(set(params))
    restart = lane["restart"]
    if lane_name == "inference":
        classes = restart["warm_restart"] + restart["cold_restart"] + restart["cold_restart_unverified"]
    else:
        classes = restart["in_process"] + restart["process_restart"]
    assert sorted(classes) == sorted(params)
    assert set(lane["grouping"]["class_keys"]) <= set(params)


@pytest.mark.parametrize("lane_name", SEARCH_LANES)
def test_initial_arms_start_at_the_baseline_and_cite_known_priors(config: dict, lane_name: str) -> None:
    lane = config["lanes"][lane_name]
    priors = {p["id"]: p for p in config["priors"]}
    arms = lane["initial_arms"]
    assert arms[0]["parameters"] == {} and arms[0]["attach_as"] == "baseline"
    assert all(arm["attach_as"] == "trial" for arm in arms[1:])
    for arm in arms:
        assert set(arm["parameters"]) <= set(lane["parameters"]), arm["name"]
        for prior_id in arm["priors"]:
            assert priors[prior_id]["lane"] == lane_name


def test_priors_are_well_formed(config: dict) -> None:
    ids = [p["id"] for p in config["priors"]]
    assert len(ids) == len(set(ids))
    for prior in config["priors"]:
        assert prior["p"] is None or 0 < prior["p"] < 1
        assert prior["sources"]
        assert set(prior["parameters"]) <= set(config["lanes"][prior["lane"]]["parameters"])


def test_budget_arithmetic_matches_the_trial_estimates(config: dict) -> None:
    inference = config["lanes"]["inference"]
    restart = inference["restart"]
    screen = inference["fidelity"]["screen"]["est_trial_s"]
    overhead = screen["cold"] - restart["cold_s"]
    assert screen["warm"] - restart["warm_s"] == overhead
    assert screen["grouped_mean"] == round((restart["cold_s"] + restart["warm_s"]) / 2 + overhead)
    for lane_name, trial_s in (("inference", screen["grouped_mean"]),
                               ("training", config["lanes"]["training"]["fidelity"]["screen"]["est_trial_s"])):
        budget = config["lanes"][lane_name]["budget"]
        assert budget["slots"] * budget["slot_minutes"] == budget["minutes"]
        assert budget["trials_per_worker_est"] == math.floor(budget["minutes"] * 60 / trial_s)


def test_group_by_state_covers_every_lane_and_card(config: dict) -> None:
    rule = config["group_by_state"]
    assert {"principle", "inference", "training", "card"} <= set(rule)
    names = {w["name"] for w in config["workers"]}
    lanes = set(config["lanes"]) | {"baseline_attach"}
    for worker, order in rule["card"].items():
        if worker == "rule":
            continue
        assert worker in names
        assert set(order) <= lanes
    est = rule["inference"]["restarts_per_night_est"]
    assert est["grouped"]["cold"] + est["grouped"]["warm"] == est["grouped"]["total"] == est["configurations"]


def test_kernel_lane_kill_rule(config: dict) -> None:
    kernel = config["lanes"]["kernel"]
    assert 0 < kernel["kill"]["spearman_min"] < 1
    assert kernel["kill"]["flat_space_gain_min"] > 0
    assert kernel["top_k_measure"] < kernel["initial_random_configs"]


def test_schedule_names_known_workers_and_lanes(config: dict) -> None:
    names = {w["name"] for w in config["workers"]}
    lanes = set(config["lanes"]) | {"baseline_attach"}
    for item in config["schedule"]:
        assert CLOCK.match(item["start"]) and CLOCK.match(item["end"]), item
        assert item["plan"] in ("A", "B", "A and B")
        assert item["worker"] in names
        assert item["lane"] in lanes


def test_note_documents_every_top_level_key(config: dict) -> None:
    approach = NOTE.read_text(encoding="utf-8").split("## Approach", 1)[1].split("\n## ", 1)[0]
    for key in TOP_LEVEL:
        assert f"`{key}`" in approach, key
