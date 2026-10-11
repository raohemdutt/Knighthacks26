"""lab/l16v2/bo_local.py: the workstation-held Bayesian search of the L16 retrain, with a fake worker.

No Colab session, no model and no trainer run here: a run on a worker is replaced by a function of its parameters.
What is checked is what makes the search honest: the warm start comes first, each point is asked for only when its
worker is free (`seen` is the finished trials the sampler had), the random startup is three trials in total after
the seeded ones, a pruned trial is recorded as pruned with its step-12 loss, and the budget stops the asking.
Needs optuna (lab/l16v2's own dependency, not in pyproject.toml), so the test is skipped where it is missing.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import tarfile
import time
from pathlib import Path

import pytest

optuna = pytest.importorskip("optuna")

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def bo(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("l16v2_bo_local", REPO_ROOT / "lab" / "l16v2" / "bo_local.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "out"
    (out / "colab").mkdir(parents=True)
    for name, value in (("OUT", out), ("COLAB", out / "colab"), ("BO", out / "bo"), ("LAUNCH", out / "colab" / "launch")):
        monkeypatch.setattr(module, name, value)
    monkeypatch.setattr(module, "RANGES", {key: list(value) for key, value in module.RANGES.items()})
    module.lines = []
    monkeypatch.setattr(module, "plog", module.lines.append)
    # the grid's run 1, as its metrics product came back: the observation the search is seeded with
    run = tmp_path / "r01-lr0.0001-rank16-drop0.05"
    run.mkdir()
    (run / "result.json").write_text(json.dumps({
        "name": run.name, "lr": 1e-4, "rank": 16, "alpha": 32, "dropout": 0.05, "warmup_ratio": 0.1, "max_steps": 62,
        "ok": True, "outcome": "trained", "best_val": 0.0081, "best_step": 62, "best_passes": 992,
        "evaluations": [{"step": 8, "eval_loss": 0.0355}, {"step": 16, "eval_loss": 0.0188},
                        {"step": 24, "eval_loss": 0.0133}, {"step": 62, "eval_loss": 0.0081}]}), encoding="utf-8")
    with tarfile.open(out / "colab" / "run-01-r01-lr0.0001-rank16-drop0.05.tar.gz", "w:gz") as tar:
        tar.add(run, arcname=run.name)
    return module


def arguments(budget_s: float = 1000.0) -> argparse.Namespace:
    return argparse.Namespace(budget_s=budget_s, began_utc=None)


def fake_run(loss_of, clock):
    """A worker whose trials each take 100 s of the search's clock (`clock` holds the search once it exists): the
    losses are a function of the flags; a pruned trial stops at step 12."""
    def run_on(session, name, flags, adapter, line, at_mid=None, poll_s=30.0):
        clock["search"].began -= 100.0
        value = dict(zip(flags[::2], flags[1::2]))
        at_12, at_24 = loss_of(float(value["--lr"]), int(value["--rank"]))
        pruned = bool(at_mid and at_mid(at_12))
        return {"name": name, "worker": session, "wall_s": 100.0, "step_s": 6.5, "outcome": "pruned" if pruned else "trained",
                "ok": not pruned, "steps": 12 if pruned else 24, "passes": 192 if pruned else 384, "val_at_12": at_12,
                "val_at_24": None if pruned else at_24, "best_val": at_12 if pruned else min(at_12, at_24),
                "best_step": 12 if pruned else 24, "error": None}
    return run_on


def test_the_names_the_edges_and_the_ascii_read(bo):
    assert bo.trial_name(2, {"lr": 2e-4, "rank": 16, "alpha_mult": 2, "dropout": 0.05, "warmup": 0.10}) == \
        "t02-lr0.0002-rank16-a32-drop0.050-warm0.100"
    assert (bo.on_edge("lr", 2.1e-5), bo.on_edge("lr", 1e-4), bo.on_edge("lr", 4.9e-4)) == ("low", None, "high")
    assert (bo.on_edge("dropout", 0.0), bo.on_edge("dropout", 0.07), bo.on_edge("warmup", 0.149)) == ("low", None, "high")
    assert bo.clean("\x1b[0;31mUnicodeDecodeError\x1b[0m \u2501\u2501 ok\n".encode("utf-8")) == "UnicodeDecodeError  ok\n"
    assert bo.wsl(Path("C:/Users/x/out")) == "/mnt/c/Users/x/out"


def test_the_boundary_check_widens_a_range_once_and_only_the_edge_it_sits_on(bo):
    bo.boundary_check({"number": 4, "lr": 4.95e-4, "dropout": 0.0, "warmup_ratio": 0.08})
    assert bo.RANGES == {"lr": [2e-5, 1e-3], "dropout": [0.0, 0.15], "warmup": [0.0, 0.15]}
    assert any("lr=0.000495" in line and "widened once to 2e-05..0.001" in line for line in bo.lines)
    assert any("dropout=0" in line and "cannot be widened" in line for line in bo.lines)


def test_the_search_starts_from_the_seeds_asks_only_when_a_worker_is_free_and_stops_on_the_budget(bo, monkeypatch):
    clock: dict = {}
    monkeypatch.setattr(bo, "run_on", fake_run(lambda lr, rank: (0.03 + abs(lr - 1.5e-4) * 40, 0.01 + abs(lr - 1.5e-4) * 40), clock))
    search = clock["search"] = bo.Search(arguments())
    search.workers = ["w1"]
    assert [t["kind"] for t in search.trials] == ["seed"] and search.trials[0]["value"] == 0.0133
    search.began = time.time() - 300          # 700 s of budget left, 100 s a trial: seven more trials
    search.longest = 100.0
    search.worker("w1")
    ran = sorted((t for t in search.trials if t["number"] > 1), key=lambda t: t["number"])
    assert [(t["lr"], t["rank"], t["alpha"], t["kind"]) for t in ran[:2]] == [(2e-4, 16, 32, "seed"), (5e-5, 16, 32, "seed")]
    assert [t["kind"] for t in ran[2:5]] == ["init", "init", "init"]      # three random startup trials in total
    assert all(t["kind"] == "model" for t in ran[5:]) and len(ran) > 5
    assert [t["seen"] for t in ran] == list(range(1, len(ran) + 1))       # every ask had every finished trial
    assert search.stopped and any("stops proposing: budget" in line for line in bo.lines)
    states = [t.state.name for t in search.study.trials]
    assert states.count("RUNNING") == 0 and states.count("WAITING") == 0
    saved = json.loads((bo.BO / "bo.json").read_text(encoding="utf-8"))
    assert [t["number"] for t in saved["trials"]] == list(range(1, len(ran) + 2))
    search.table()
    table = [line for line in bo.lines if "bo_table" in line]
    assert len(table) == len(search.trials) and "kind=seed" in table[0] and "best_so_far=0.0133" in table[0]
    assert any("bo_summary" in line and "model: n=" in line for line in bo.lines)


def test_two_workers_ask_one_at_a_time_and_a_running_trial_counts_toward_the_startup_not_toward_seen(bo, monkeypatch):
    search = bo.Search(arguments())
    search.workers = ["w1", "w2"]
    first = search.next_trial("w1")
    second = search.next_trial("w2")       # asked while the first is still running
    assert (first[2]["seen"], second[2]["seen"]) == (1, 1)
    assert (first[2]["kind"], second[2]["kind"]) == ("seed", "seed")
    third = search.next_trial("w1")        # both warm-start points are out: the next is a random startup draw
    assert (third[2]["kind"], third[2]["seen"]) == ("init", 1)
    done = {"outcome": "trained", "ok": True, "val_at_12": 0.03, "val_at_24": 0.02, "best_val": 0.02, "wall_s": 100.0}
    search.finish(first[0], {**first[2], **done})
    fourth = search.next_trial("w2")
    assert fourth[2]["seen"] == 2 and fourth[2]["number"] == 5


def test_a_trial_the_median_pruner_stops_is_recorded_as_pruned_with_its_step_12_loss(bo, monkeypatch):
    losses = iter([(0.02, 0.01), (0.021, 0.011), (0.022, 0.012), (0.5, 0.4)])
    clock: dict = {}
    monkeypatch.setattr(bo, "run_on", fake_run(lambda lr, rank: next(losses), clock))
    search = clock["search"] = bo.Search(arguments())
    search.workers = ["w1"]
    search.began = time.time() - 550          # room for four trials of 100 s
    search.longest = 100.0
    search.worker("w1")
    last = max(search.trials, key=lambda t: t["number"])
    assert (last["outcome"], last["value"], last["val_at_24"]) == ("pruned", 0.5, None)
    assert search.study.trials[last["study_trial"]].state.name == "PRUNED"
    assert search.best.startswith("trial2:")          # a pruned trial is never the best
    assert any("pruned_by_median_at_step_12" in line for line in bo.lines)
