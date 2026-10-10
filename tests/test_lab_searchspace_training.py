"""Structure checks for the gfx1151 training search space that Ax reads.

The search space is the contract between the settings sheet and the optimizer run: every parameter
must carry a type and a range or choices in the shape Ax's parameter configs take, the pinned baseline
must be a point inside the space, the shortlist must stay at ten settings or fewer, and every setting
the space names must be one the settings sheet documents.
"""

from __future__ import annotations

import json
from pathlib import Path

KERNELS = Path(__file__).resolve().parents[1] / "lab" / "refine" / "research" / "kernels"
SPACE = KERNELS / "data" / "searchspace-training-gfx1151.json"
NOTE = KERNELS / "settings_training_gfx1151.md"

TYPES = {"float": float, "int": int, "str": str, "bool": bool}


def _space() -> dict:
    return json.loads(SPACE.read_text(encoding="utf-8"))


def _matches(value, parameter_type: str) -> bool:
    expected = TYPES[parameter_type]
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if expected is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, expected)


def test_every_parameter_has_an_ax_type_and_a_range_or_choices():
    for p in _space()["parameters"]:
        assert p["kind"] in {"range", "choice", "derived"}, p["name"]
        if p["kind"] == "range":
            assert p["parameter_type"] in {"float", "int"}, p["name"]
            low, high = p["bounds"]
            assert low < high, p["name"]
        elif p["kind"] == "choice":
            assert p["parameter_type"] in TYPES, p["name"]
            assert len(p["values"]) >= 2, p["name"]
            assert len(set(map(repr, p["values"]))) == len(p["values"]), p["name"]
            assert all(_matches(v, p["parameter_type"]) for v in p["values"]), p["name"]
        else:
            assert p["parameter_type"] in TYPES and p["expression_str"], p["name"]


def test_parameter_names_are_unique():
    names = [p["name"] for p in _space()["parameters"]]
    assert len(names) == len(set(names))


def test_the_baseline_is_a_point_inside_the_space():
    space = _space()
    baseline = space["baseline"]
    by_name = {p["name"]: p for p in space["parameters"]}
    assert set(baseline) == set(by_name)
    for name, value in baseline.items():
        p = by_name[name]
        if p["kind"] == "choice":
            assert value in p["values"], name
        elif p["kind"] == "range":
            assert p["bounds"][0] <= value <= p["bounds"][1], name
    assert baseline["per_device_train_batch_size"] * baseline["gradient_accumulation_steps"] == space[
        "fixed_effective_batch_size"
    ]


def test_every_micro_batch_divides_the_fixed_effective_batch():
    space = _space()
    batch = next(p for p in space["parameters"] if p["name"] == "per_device_train_batch_size")
    assert all(space["fixed_effective_batch_size"] % b == 0 for b in batch["values"])


def test_the_shortlist_holds_at_most_ten_distinct_ranks():
    ranks = [p["shortlist_rank"] for p in _space()["parameters"] if p["shortlist_rank"] is not None]
    assert 0 < len(ranks) <= 10
    assert sorted(ranks) == list(range(1, len(ranks) + 1))


def test_every_parameter_points_at_a_section_and_experiment_of_the_note():
    note = NOTE.read_text(encoding="utf-8")
    for p in _space()["parameters"]:
        assert f"\n## {p['note_section']}. " in note, p["name"]
        assert f"{p['experiment']}," in note or f"{p['experiment']} " in note, p["name"]


def test_invalid_combinations_name_only_known_parameters_and_values():
    space = _space()
    by_name = {p["name"]: p for p in space["parameters"]}
    for rule in space["invalid_combinations"]:
        for name, value in rule["when"].items():
            assert name in by_name, name
            assert value in by_name[name]["values"], name
