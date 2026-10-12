"""lab/l16v2/report.py: the extended search's tables from a bo-ext.json, with a run that was lost.

No model and no Colab here: the runs are given as the driver writes them. What is checked is that a lost run shows
as lost with no numbers (not as None), that a configuration run with two seeds gets its mean and difference, and
that the changes of the choice are listed.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_report():
    spec = importlib.util.spec_from_file_location("l16v2_report", REPO_ROOT / "lab" / "l16v2" / "report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(name: str, number: int, kind: str, seed: int, lr: float, rank: int, agree: int | None, val: float | None) -> dict:
    fact = {"name": name, "number": number, "kind": kind, "seed": seed, "lr": lr, "rank": rank, "alpha": 2 * rank,
            "dropout": 0.05, "warmup_ratio": 0.05, "worker": "vm-a"}
    if agree is None:
        return {**fact, "outcome": "lost", "ok": False}
    return {**fact, "ok": True, "outcome": "trained", "steps": 62, "best_step": 62, "best_passes": 992, "best_val": val,
            "agree": agree, "wall_s": 600.0}


def test_a_lost_run_shows_as_lost_and_the_pairs_and_the_changes_of_the_choice_are_listed(tmp_path: Path) -> None:
    runs = [run("first", 0, "earlier", 42, 4e-4, 4, 113, 0.0098), run("first-again", 1, "repeat", 43, 4e-4, 4, 106, 0.0073),
            run("corner", 2, "edge", 42, 2e-4, 32, 115, 0.0087), run("corner-again", 3, "repeat", 43, 2e-4, 32, 117, 0.0068),
            run("gone", 4, "edge", 42, 4e-4, 2, None, None)]
    choice = {"key": "corner", "first": "first", "replaced": [{"key": "first", "mean": 109.5, "by": "corner", "mean_by": 116.0}]}
    (tmp_path / "bo-ext.json").write_text(json.dumps({"runs": runs, "choice": choice}), encoding="utf-8")

    text = load_report().ext_tables(tmp_path)

    lost = next(line for line in text.splitlines() if "| gone |" in line)
    assert "| lost | n/a | n/a | not scored | vm-a | n/a |" in lost and "None" not in lost
    assert "| 113 / 106 | 109.5 | 7 |" in text and "| 115 / 117 | 116.0 | 2 |" in text
    assert "differ by 2 to 7 examples of 126 (mean 4.5)" in text
    assert "Runs that did not finish: gone (lost)." in text
    assert "Changes of the choice, in order: first (mean 109.5/126) replaced by corner (mean 116.0/126)." in text
