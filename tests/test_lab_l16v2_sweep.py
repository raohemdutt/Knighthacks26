"""lab/l16v2/sweep.py: the choice rule and the products of the L16 retrain's sweep, with a fake generator.

No model, no GPU and no trainer run here: the runs are given as facts, and the generator writes the reference call or
a wrong answer by a fixed pattern. What is checked is what the rule says: the three lowest validation losses are
scored, the highest validation agreement wins, the held-out side is read only after the choice is written, and every
product has its size and sha256 beside it before its marker appears.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import tarfile
from pathlib import Path

import pytest

from lab import fixture_dataset_v2 as v2
from lab import format_traces as ft
from lab import tool_call_agreement as tca
from lab.dataset import load_split
from lab.fixture_dataset import load_expected

REPO_ROOT = Path(__file__).resolve().parents[1]
OLD = REPO_ROOT / "fixtures" / "learning" / "tool-calling-fixture"


def load_sweep():
    spec = importlib.util.spec_from_file_location("l16v2_sweep", REPO_ROOT / "lab" / "l16v2" / "sweep.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LAB_ACTOR", "opus-test")
    monkeypatch.setenv("LAB_TIER", "opus")
    monkeypatch.delenv("LAB_TASK", raising=False)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> Path:
    return v2.build(tmp_path_factory.mktemp("v2")).root


def reference_outputs(*folders: Path) -> tuple[dict[str, str], set[str]]:
    """{prompt: the reference call as the template writes it} for every example, and the held-out prompts."""
    gold, held = {}, set()
    for folder in folders:
        split = load_split(folder, require_images=False)
        expected = load_expected(folder)
        held_ids = {example.id for example in split.held_out}
        for example in split.data.examples:
            calls = [{"type": "function", "function": {"name": call.name, "arguments": call.arguments}}
                     for call in expected[example.id].label.tool_calls]
            text, spans = ft.render_with_spans([{"role": "user", "content": "q"},
                                                {"role": "assistant", "content": "", "tool_calls": calls}], ft.TOOLS)
            prompt = tca.prompt_for(v2.system_for(split.data.manifest, example), example.context)
            gold[prompt] = text[spans[0][0]:spans[0][1]]
            if example.id in held_ids:
                held.add(prompt)
    return gold, held


def facts_for(sweep, job) -> list[dict]:
    """Three runs that trained and one that failed; the lowest validation loss is not the best at the calls."""
    facts = []
    for index, ((lr, rank, dropout), best_val, ok) in enumerate(
            zip(sweep.GRID, (0.030, 0.020, 0.040), (True, True, True)), 1):
        name = sweep.run_name(index, lr, rank, dropout)
        (job.runs / name / "adapter").mkdir(parents=True)
        (job.runs / name / "adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
        (job.runs / name / "run.json").write_text("{}", encoding="utf-8")
        facts.append({"name": name, "index": index, "lr": lr, "rank": rank, "alpha": 2 * rank, "dropout": dropout,
                      "max_steps": 62, "warmup_ratio": sweep.WARMUP_RATIO, "ok": ok, "eligible": True,
                      "outcome": "trained", "best_val": best_val, "best_step": 40, "best_passes": 640,
                      "final_val": best_val + 0.01, "passes": 992, "planned_passes": 992, "wall_s": 600.0,
                      "torch_loss_head": False, "batch": 4, "grad_accum": 4, "gradient_checkpointing": True,
                      "gdn_kernel": "torch", "causal_conv1d_kernel": "torch", "loss_head": "trl triton kernel",
                      "device": "fake"})
    facts.append({**facts[0], "name": "r04-failed", "index": 4, "ok": False, "outcome": "failed", "best_val": None})
    (job.out / "sweep.json").write_text(json.dumps({"runs": facts}), encoding="utf-8")
    return facts


def test_the_rule_scores_the_lowest_losses_and_chooses_by_validation_agreement_before_held_out_is_read(
        dataset, tmp_path, monkeypatch):
    sweep = load_sweep()
    gold, held = reference_outputs(dataset, OLD)
    job = sweep.Job(tmp_path / "out", 100.0)
    facts = facts_for(sweep, job)
    wrong_every = {"base": 3, facts[0]["name"]: 50, facts[1]["name"]: 5, facts[2]["name"]: 9}
    seen_held_before_choice: list[str] = []

    class FakeGenerator:
        def __init__(self, model: str) -> None:
            self.arm = "base"

        def attach(self, adapter: Path | None) -> None:
            self.arm = "base" if adapter is None else adapter.parent.name

        def generate(self, prompts, max_new_tokens, tick=None):
            if not (job.out / "chosen.json").exists():
                seen_held_before_choice.extend(prompt for prompt in prompts if prompt in held)
            every = wrong_every[self.arm]
            out = {prompt: ("I do not know.<|im_end|>" if number % every == 0 else gold[prompt])
                   for number, prompt in enumerate(dict.fromkeys(prompts))}
            if tick:
                tick(len(out), len(out))
            return out

    monkeypatch.setattr(sweep, "Generator", FakeGenerator)
    args = argparse.Namespace(dataset=str(dataset), model="none")
    chosen = sweep.choose(job, args, facts)
    # the run with the lowest validation loss (the second) misses more calls than the first, so the first is chosen
    assert [run["name"] for run in chosen["ranking_by_validation_loss"]] == \
        [facts[1]["name"], facts[0]["name"], facts[2]["name"]]
    assert chosen["chosen"]["name"] == facts[0]["name"]
    assert chosen["chosen"]["val_agreement"]["correct"] == max(run["val_agreement"]["correct"] for run in chosen["top"])
    assert chosen["rule"] == sweep.CHOICE_RULE and chosen["scoring_rule"] == tca.RULE
    assert chosen["validation_examples"] == 126 and chosen["base_validation_agreement"]["n"] == 126
    assert "--stop-step" in chosen["recipe_flags"] and chosen["recipe_flags"][chosen["recipe_flags"].index("--batch") + 1] == "4"
    assert seen_held_before_choice == []
    assert (job.markers / "BEST_DONE").read_text(encoding="ascii").strip() == "best-adapter.tar.gz"

    sweep.held_out(job, args, chosen)
    assert seen_held_before_choice == []
    for name in ("best-adapter.tar.gz", "held-out-bf16.tar.gz"):
        digest, size, listed = (job.products / (name + ".sha256")).read_text(encoding="ascii").split()
        payload = (job.products / name).read_bytes()
        assert (digest, int(size), listed) == (hashlib.sha256(payload).hexdigest(), len(payload), name)
    with tarfile.open(job.products / "best-adapter.tar.gz") as tar:
        names = tar.getnames()
    assert {"adapter/adapter_config.json", "chosen.json", "run.json", "sweep.json"} <= set(names)
    summary = json.loads((job.out / "held-out-bf16" / "v2" / "held_out.summary.json").read_text(encoding="utf-8"))
    assert (summary["n_items"], summary["n_groups"], summary["side"]) == (228, 18, "held_out")
    old = json.loads((job.out / "held-out-bf16" / "fixture" / "held_out.summary.json").read_text(encoding="utf-8"))
    assert (old["n_items"], old["n_groups"]) == (58, 4)
    lines = (job.out / "finished.txt").read_text(encoding="ascii").splitlines()
    assert any(line.startswith("chosen name=") for line in lines)
    assert sum(line.startswith("held_out_bf16 ") for line in lines) == 2
    assert (job.out / "status.txt").read_text(encoding="ascii").isascii()


def test_a_sweep_with_no_run_that_trained_chooses_nothing(dataset, tmp_path, monkeypatch):
    sweep = load_sweep()
    job = sweep.Job(tmp_path / "out", 100.0)
    monkeypatch.setattr(sweep, "Generator", lambda model: pytest.fail("no model is loaded when nothing trained"))
    with pytest.raises(RuntimeError, match="no eligible run"):
        sweep.choose(job, argparse.Namespace(dataset=str(dataset), model="none"),
                     [{"name": "r01", "index": 1, "ok": False, "eligible": True, "best_val": None}])
    assert not (job.out / "chosen.json").exists()


def test_the_status_line_is_ascii_key_value_and_every_run_is_within_the_pod_budget():
    sweep = load_sweep()
    assert sweep.MAX_STEPS * sweep.BATCH * sweep.GRAD_ACCUM <= sweep.POD_PASSES
    assert sweep.BATCH * sweep.GRAD_ACCUM == sweep.EFFECTIVE_BATCH
    assert sweep.ascii_only("lr=1e-4 ✓") == "lr=1e-4 ?"
    assert len(sweep.GRID) == len(set(sweep.GRID)) >= 3
