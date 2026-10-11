"""lab/l16v2/bo_edge.py and lab/l16v2/bo_ext.py: the edge round and the extended search, with a fake worker.

No Colab session and no model: a run on a worker is a function of its flags, and validation agreement comes from a
list. What is checked is what the rules say (lab/L16V2_RULES.md amendments 5 and 6): the forced corners and the
noise repeats come in the fixed order, a fourth finalist needs a gain larger than the measured noise, and in the
extended search a challenger replaces the choice only on the mean of two seeds. Needs optuna; skipped without it.
"""
from __future__ import annotations

import datetime as dt
import importlib
import json
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("optuna")

HERE = Path(__file__).resolve().parents[1] / "lab" / "l16v2"


def trial(number, kind, lr, rank, alpha, value, at_12, outcome="trained", steps=24):
    return {"number": number, "kind": kind, "seen": number - 1, "worker": "w1", "name": f"t{number:02d}", "lr": lr, "rank": rank,
            "alpha": alpha, "alpha_mult": alpha // rank, "dropout": 0.05, "warmup_ratio": 0.10, "max_steps": steps,
            "outcome": outcome, "ok": outcome == "trained", "value": value, "val_at_12": at_12,
            "val_at_24": None if outcome == "pruned" else value}


def full_run(name, lr, rank, alpha, best_val, agree, place):
    return {"name": name, "place": place, "lr": lr, "rank": rank, "alpha": alpha, "dropout": 0.05, "warmup_ratio": 0.10,
            "steps": 48, "passes": 768, "best_step": 32, "best_passes": 512, "best_val": best_val, "final_val": best_val,
            "wall_s": 500.0, "worker": "w1", "val_agreement": {"correct": agree, "n": 126},
            "adapter_path": f"X:/{name}/adapter", "adapter_sha256": "0" * 64, "product_sha256": "1" * 64}


@pytest.fixture
def rounds(tmp_path, monkeypatch):
    monkeypatch.setenv("L16V2_WORKSPACE", str(tmp_path).replace("\\", "/"))
    monkeypatch.syspath_prepend(str(HERE))
    for name in ("bo_local", "bo_edge", "bo_ext"):
        sys.modules.pop(name, None)
    bl = importlib.import_module("bo_local")
    be = importlib.import_module("bo_edge")
    bx = importlib.import_module("bo_ext")
    (bl.BO).mkdir(parents=True)
    bl.COLAB.mkdir(parents=True)
    seed = {**trial(1, "seed", 1e-4, 16, 32, 0.0133, None, steps=62), "full_length": True, "name": "r01"}
    first = [seed, trial(2, "seed", 2e-4, 16, 32, 0.0159, 0.0237), trial(3, "seed", 5e-5, 16, 32, 0.0293, 0.0293, "pruned"),
             trial(9, "model", 2.48e-4, 16, 32, 0.0150, 0.0227), trial(12, "model", 4.28e-4, 4, 8, 0.0142, 0.0212)]
    (bl.BO / "bo.json").write_text(json.dumps({"trials": first}), encoding="utf-8")
    chosen = {"full_length_runs": [full_run("r01", 1e-4, 16, 32, 0.0081, 110, 9), full_run("final2", 4.28e-4, 4, 8, 0.0098, 113, 2),
                                   full_run("final3", 2.48e-4, 16, 32, 0.0107, 111, 3)],
              "chosen": {"name": "final2", "val_agreement": {"correct": 113, "n": 126}}, "recipe_flags": []}
    (bl.BO / "chosen.json").write_text(json.dumps(chosen), encoding="utf-8")
    lines: list[str] = []
    monkeypatch.setattr(bl, "plog", lines.append)
    clock = {"skew": 0.0}
    real = time.time
    monkeypatch.setattr(time, "time", lambda: real() + clock["skew"])
    return bl, be, bx, lines, clock


def test_the_edge_round_runs_corners_and_noise_in_order_and_names_no_finalist_inside_the_noise(rounds, monkeypatch):
    bl, be, bx, lines, clock = rounds
    launched: list[tuple[str, str | None]] = []

    def run_on(session, name, flags, adapter, line, at_mid=None, poll_s=30.0):
        value = dict(zip(flags[::2], flags[1::2]))
        launched.append((name, value.get("--seed")))
        clock["skew"] += 100.0
        at_12 = 0.0200
        at_24 = {"n": 0.0170}.get(name[0], 0.0125)          # every edge trial gains 0.0017 on the best 24-step value
        pruned = bool(at_mid and at_mid(at_12))
        return {"name": name, "worker": session, "wall_s": 100.0, "outcome": "pruned" if pruned else "trained", "ok": not pruned,
                "val_at_12": at_12, "val_at_24": None if pruned else at_24, "best_val": at_12 if pruned else at_24}

    monkeypatch.setattr(bl, "run_on", run_on)
    monkeypatch.setattr(be, "TRIAL_ALONE_S", 100.0)
    monkeypatch.setattr(sys, "argv", ["bo_edge.py", "--sessions", "w1", "--budget-s", "750", "--no-wait"])
    assert be.main() == 0
    names = [name for name, _ in launched]
    assert [name[:1] for name in names[:7]] == ["e", "e", "n", "n", "e", "e", "e"]
    assert ["rank32" in names[0] and "lr0.0002" in names[0], "rank16" in names[1] and "lr0.0004" in names[1],
            "rank32" in names[4], "rank4" in names[5] and "lr0.0008" in names[5], "rank2" in names[6]] == [True] * 5
    assert launched[2][1] == "43" and launched[3][1] is None            # trial 9 again with another seed; the seed recipe
    edge = json.loads((bl.BO / "bo-edge.json").read_text(encoding="utf-8"))
    assert set(edge["noise"]) == {"trial9", "seed24"} and edge["best_24_step_value_before"] == 0.0142
    result = json.loads((bl.BO / "edge-result.json").read_text(encoding="utf-8"))
    assert result["fourth"] is None and result["floor"] == pytest.approx(0.0037)      # |0.0170 - 0.0133|, the larger
    assert any("no edge-round trial beats" in line and "the choice stands" in line for line in lines)
    assert any("left out, below the new learning-rate floor: trials [3]" in line for line in lines)


def test_the_extended_search_replaces_the_choice_only_on_the_mean_of_two_seeds(rounds, monkeypatch):
    bl, be, bx, lines, clock = rounds
    agreement = iter([105, 116, 108, 112, 113, 110, 109, 108, 107, 106, 105, 104])
    order: list[str] = []

    def run_on(session, name, flags, adapter, line, at_mid=None, poll_s=30.0):
        order.append(name)
        clock["skew"] += 400.0
        return {"name": name, "worker": session, "wall_s": 600.0, "outcome": "trained", "ok": True, "steps": 48, "passes": 768,
                "best_val": 0.0090, "best_step": 32, "best_passes": 512, "final_val": 0.0095}

    def score_on(session, label, arms):
        clock["skew"] += 80.0
        return {arms[0]: {"correct": next(agreement), "n": 126}}

    monkeypatch.setattr(bl, "run_on", run_on)
    monkeypatch.setattr(bl, "score_on", score_on)
    monkeypatch.setattr(bx, "pull_adapter", lambda session, name: {"adapter_path": f"X:/{name}/adapter", "adapter_sha256": "2" * 64})
    monkeypatch.setattr(be, "read_held_out", lambda session, name: pytest.fail("the held-out side is read only by the final line's caller"))
    until = dt.datetime.fromtimestamp(time.time() + 3300, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    monkeypatch.setattr(sys, "argv", ["bo_ext.py", "--sessions", "w1", "--until-utc", until, "--skip-held-out"])
    assert bx.main() == 0
    # the choice's own repeat first (113 and 105: mean 109); r01 (110) and final3 (111) are then above it and repeat next
    assert order[0] == bx.CHOICE_REPEAT and all("repeat" in name for name in order[1:3])
    saved = json.loads((bl.BO / "bo-ext.json").read_text(encoding="utf-8"))
    by_name = {run["name"]: run for run in saved["runs"]}
    assert by_name[order[1]]["repeat_of"] in ("r01", "final3") and by_name[order[1]]["seed"] == 43
    # the first repeat scored 116: with its first run's score its mean beats 109, so that configuration is the choice
    replaced = saved["choice"]["replaced"]
    assert len(replaced) >= 1 and replaced[0]["mean"] == 109.0 and replaced[0]["mean_by"] > 109.0
    final = json.loads((bl.BO / "chosen.json").read_text(encoding="utf-8"))
    assert final["previous_choice"] == "final2" and final["chosen"]["seeds_run"] == 2
    assert final["chosen"]["mean_val_agreement_over_seeds"] == pytest.approx(replaced[-1]["mean_by"])
    assert any("NEW CHOICE" in line for line in lines) and any("chosen=yes" in line for line in lines)
    assert sum("RESULT ext_table noise" in line for line in lines) >= 2
    assert any("stops starting runs" in line for line in lines)
