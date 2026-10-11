"""The L16 retrain on the Colab A100: a small LoRA sweep on tool-calling-v2, chosen by validation only.

    python lab/l16v2/sweep.py --dataset /content/l16v2/data/tool-calling-v2 --model /content/models/Qwen3.5-4B \\
        --out /content/l16v2 [--budget-s 2400] [--max-steps 62] [--only smoke|sweep|choose|held_out]

Run from the repo root on the VM by lab/l16v2/colab_job.sh. It runs lab/train.py --cuda once per grid point, in the
fixed order GRID, until the sweep budget is used; a run that fails is recorded and the next one starts. Then it
applies CHOICE_RULE, which is written here before any run starts and is copied into every file this leaves.

What it leaves under --out, each product with a .sha256 beside it (sha256, bytes, name) and a marker in markers/
that appears only after the product and its .sha256 are whole, so the driver on the workstation can pull each one
the moment it exists (Colab reclaims the VM about 20 minutes after it goes idle):

    status.txt                       one ASCII key=value line, rewritten about every 20 s: stage, run i of n with its
                                     learning rate, rank and dropout, optimizer step of total, latest training loss,
                                     latest validation loss, the best run so far, seconds elapsed and left
    finished.txt                     one ASCII key=value line per finished run with its final numbers
    products/run-NN-<name>.tar.gz    run.json, telemetry.jsonl, validation.jsonl, train.log, adapter_config.json
                                     (marker RUN_NN_DONE, or RUN_NN_FAILED with the log)
    products/best-adapter.tar.gz     the chosen run's adapter (the checkpoint of its best evaluation), chosen.json
                                     and its run.json (marker BEST_DONE)
    products/held-out-bf16.tar.gz    after the choice is written: base against chosen on the held-out side of
                                     tool-calling-v2 and of tool-calling-fixture, generated here (marker HELD_OUT_DONE)
    chosen.json, sweep.json          the choice with its rule and the table it was made from

Generation here is transformers greedy decoding of the bf16 weights with the adapter attached (not the packaged
Q4_K_M GGUF on llama-server that the pod scores): prompts of equal token length are batched together, so no prompt
is padded. The scoring is lab/tool_call_agreement.py's rule, unchanged: its prompt_for(), score_output() and run().
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
# (learning rate, rank, dropout); alpha is twice the rank. The order is the order of running, fixed here.
GRID = [(1e-4, 16, 0.05), (2e-4, 16, 0.05), (5e-5, 16, 0.05), (1e-4, 8, 0.05), (2e-4, 8, 0.05), (5e-5, 8, 0.05),
        (1e-4, 16, 0.1), (2e-4, 16, 0.1), (5e-5, 16, 0.1), (1e-4, 8, 0.1), (2e-4, 8, 0.1), (5e-5, 8, 0.1)]
WARMUP_RATIO = 0.06
EVAL_STEPS = 8
EARLY_STOP = 3
BATCH, GRAD_ACCUM = 4, 4          # the pod's pinned effective batch of 16, so the recipe is the same there
MAX_STEPS = 62                    # 62 x 16 = 992 example passes: what pod D trains in about 45 minutes
POD_PASSES = 1000
TOP = 3
CHOICE_RULE = (
    "Eligible: a grid run that trained and whose planned example passes (optimizer steps x 16) are at most 1,000, "
    "the pod D budget. Each run's score is its lowest validation loss over its evaluations (every 8 optimizer "
    "steps and after the last), and its adapter is the checkpoint of that evaluation. Take the 3 eligible runs "
    "with the lowest score (ties: fewer example passes at the best evaluation, then earlier in the grid order). "
    "For those 3, generate on every validation example with the kept adapter and score exact tool-call agreement "
    "by lab/tool_call_agreement.py's rule. Choose the highest validation agreement (ties: lower validation loss, "
    "then fewer example passes at the best evaluation, then earlier in the grid order). The base model's "
    "validation agreement is measured for reference and takes no part. The held-out side is not read until "
    "chosen.json is written, and nothing is chosen again after it is read."
)


def ascii_only(text: str) -> str:
    return text.encode("ascii", "replace").decode("ascii")


class Job:
    """Paths, the status line and the markers of one job."""

    def __init__(self, out: Path, budget_s: float) -> None:
        self.out, self.budget_s = out, budget_s
        self.products, self.markers, self.runs = out / "products", out / "markers", out / "runs"
        for folder in (self.products, self.markers, self.runs):
            folder.mkdir(parents=True, exist_ok=True)
        self.started = time.time()
        self.best = "none"
        self.fields: dict[str, Any] = {}

    def status(self, **fields: Any) -> None:
        self.fields.update(fields)
        shown = {key: value for key, value in self.fields.items() if value is not None}
        shown["best"] = self.best
        shown["elapsed_s"] = int(time.time() - self.started)
        line = " ".join(f"{key}={value}" for key, value in shown.items())
        temporary = self.out / "status.txt.tmp"
        temporary.write_text(ascii_only(line) + "\n", encoding="ascii")
        os.replace(temporary, self.out / "status.txt")

    def finished(self, line: str) -> None:
        with (self.out / "finished.txt").open("a", encoding="ascii") as out:
            out.write(ascii_only(line) + "\n")

    def product(self, name: str, members: list[tuple[Path, str]], marker: str) -> Path:
        """Tar the members (path, name in the archive) into products/<name>, write its .sha256, then its marker."""
        path = self.products / name
        with tarfile.open(path, "w:gz") as tar:
            for source, arcname in members:
                if source.exists():
                    tar.add(source, arcname=arcname)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        (self.products / (name + ".sha256")).write_text(f"{digest} {path.stat().st_size} {name}\n", encoding="ascii")
        (self.markers / marker).write_text(name + "\n", encoding="ascii")
        return path


def run_name(index: int, lr: float, rank: int, dropout: float) -> str:
    return f"r{index:02d}-lr{lr:g}-rank{rank}-drop{dropout:g}"


def _tail(path: Path, lines: int = 1) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError):
        return []
    return rows[-lines:]


def train_once(job: Job, args: argparse.Namespace, name: str, lr: float, rank: int, dropout: float, max_steps: int,
               label: str, torch_loss: bool, extra: list[str] | None = None) -> dict[str, Any]:
    """One lab/train.py --cuda run; the status line follows its telemetry while it runs. Returns its facts."""
    out = job.runs / name
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    dataset = Path(args.dataset)
    command = [sys.executable, "lab/train.py", "--cuda", "--no-install", "--dataset", str(dataset),
               "--rows", str(dataset / "rows-train.jsonl"), "--val-rows", str(dataset / "rows-validation.jsonl"),
               "--model", args.model, "--base-revision", BASE_REVISION, "--out", str(out), "--run-id", name,
               "--max-steps", str(max_steps), "--lr", f"{lr:g}", "--rank", str(rank), "--alpha", str(2 * rank),
               "--dropout", f"{dropout:g}", "--warmup-ratio", str(WARMUP_RATIO), "--batch", str(BATCH),
               "--grad-accum", str(GRAD_ACCUM), "--eval-steps", str(EVAL_STEPS), "--keep-best",
               "--early-stop", str(EARLY_STOP), "--seed", "42", *(extra or []),
               *(["--torch-loss-head"] if torch_loss else [])]
    began = time.time()
    with (out / "train.log").open("w", encoding="utf-8") as log, (out / "stdout.json").open("w", encoding="utf-8") as so:
        process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=so, stderr=log)
        while process.poll() is None:
            step = _tail(out / "telemetry.jsonl")
            check = _tail(out / "validation.jsonl")
            done = step[0]["step"] if step else 0
            per_step = (time.time() - began) / done if done else None
            job.status(stage=args.stage, run=label, lr=f"{lr:g}", rank=rank, drop=f"{dropout:g}",
                       step=f"{done}/{max_steps}", loss=f"{step[0]['loss']:.4f}" if step and step[0]["loss"] is not None else "na",
                       val=f"{check[0]['eval_loss']:.4f}" if check else "na",
                       eta_s=int(per_step * (max_steps - done)) if per_step else "na")
            time.sleep(15)
    wall = time.time() - began
    answer: dict[str, Any] = {}
    try:
        answer = json.loads((out / "stdout.json").read_text(encoding="utf-8").strip().splitlines()[-1])
    except (OSError, ValueError, IndexError):
        pass
    metrics = answer.get("metrics", {})
    fact = {"name": name, "lr": lr, "rank": rank, "alpha": 2 * rank, "dropout": dropout, "max_steps": max_steps,
            "warmup_ratio": WARMUP_RATIO, "outcome": answer.get("outcome", f"exit {process.returncode}"),
            "ok": bool(answer.get("ok")), "error": answer.get("error"), "wall_s": round(wall, 1),
            "torch_loss_head": torch_loss,
            "steps": metrics.get("trainer.steps"), "passes": metrics.get("trainer.examples_seen"),
            "planned_passes": max_steps * BATCH * GRAD_ACCUM,
            "best_step": metrics.get("trainer.best_step"), "best_passes": metrics.get("trainer.best_examples_seen"),
            "best_val": metrics.get("trainer.best_eval_loss"), "final_val": metrics.get("trainer.final_eval_loss"),
            "loss_first": metrics.get("trainer.loss_first"), "loss_last": metrics.get("trainer.loss_last"),
            "train_s": metrics.get("trainer.train_s"), "samples_per_s": metrics.get("trainer.samples_per_s"),
            "peak_mem_gb": metrics.get("trainer.peak_mem_gb"),
            "evaluations": _tail(out / "validation.jsonl", 100)}
    (out / "result.json").write_text(json.dumps(fact, indent=2) + "\n", encoding="utf-8")
    return fact


def pack_run(job: Job, index: int, fact: dict[str, Any]) -> None:
    out = job.runs / fact["name"]
    members = [(out / name, f"{fact['name']}/{name}") for name in
               ("run.json", "result.json", "telemetry.jsonl", "validation.jsonl", "train.log", "stdout.json")]
    members.append((out / "adapter" / "adapter_config.json", f"{fact['name']}/adapter_config.json"))
    job.product(f"run-{index:02d}-{fact['name']}.tar.gz", members,
                f"RUN_{index:02d}_{'DONE' if fact['ok'] else 'FAILED'}")


def fmt(value: Any, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "na"


def sweep(job: Job, args: argparse.Namespace) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    torch_loss = False
    total = len(GRID)
    began = time.time()
    for index, (lr, rank, dropout) in enumerate(GRID, 1):
        elapsed = time.time() - began
        longest = max((fact["wall_s"] for fact in facts), default=0.0)
        if facts and elapsed + longest > job.budget_s:
            job.finished(f"sweep_stopped_at_run={index} reason=budget elapsed_s={int(elapsed)} budget_s={int(job.budget_s)}")
            break
        name = run_name(index, lr, rank, dropout)
        fact = train_once(job, args, name, lr, rank, dropout, args.max_steps, f"{index}/{total}", torch_loss)
        if not fact["ok"] and not torch_loss and index == 1:
            # The first run decides the loss head for the whole sweep: TRL's Triton kernel, or plain torch if it cannot run.
            job.finished(f"run={index}/{total} name={name} outcome={fact['outcome']} retry=torch-loss-head")
            torch_loss = True
            fact = train_once(job, args, name, lr, rank, dropout, args.max_steps, f"{index}/{total}", torch_loss)
        fact["index"], fact["eligible"] = index, fact["planned_passes"] <= POD_PASSES
        facts.append(fact)
        pack_run(job, index, fact)
        good = [f for f in facts if f["ok"] and f["best_val"] is not None]
        if good:
            top = min(good, key=lambda f: f["best_val"])
            job.best = f"run{top['index']}:{top['best_val']:.4f}"
        job.finished(f"run={index}/{total} name={name} lr={lr:g} rank={rank} drop={dropout:g} outcome={fact['outcome']} "
                     f"steps={fact['steps']} passes={fact['passes']} best_step={fact['best_step']} "
                     f"best_passes={fact['best_passes']} best_val={fmt(fact['best_val'])} final_val={fmt(fact['final_val'])} "
                     f"loss_last={fmt(fact['loss_last'])} wall_s={fact['wall_s']} "
                     f"samples_per_s={fmt(fact['samples_per_s'], 2)} peak_mem_gb={fmt(fact['peak_mem_gb'], 1)}")
        (job.out / "sweep.json").write_text(json.dumps({"rule": CHOICE_RULE, "runs": facts}, indent=2) + "\n", encoding="utf-8")
    return facts


# --- generation and scoring ---------------------------------------------------------------------------------------

class Generator:
    """The base model in bf16 on the GPU, with one adapter attached at a time; greedy decoding."""

    def __init__(self, model_dir: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.base = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, attn_implementation="sdpa")
        self.base.to("cuda").eval()
        ends = [self.tokenizer.convert_tokens_to_ids("<|im_end|>"), self.tokenizer.convert_tokens_to_ids("<|endoftext|>")]
        self.ends = [token for token in ends if isinstance(token, int) and token >= 0]
        self.model = self.base
        self.attached = False

    def attach(self, adapter: Path | None) -> None:
        from peft import PeftModel

        if self.attached:
            self.base = self.model.unload()      # the LoRA layers come off; the base weights were never merged
            self.attached = False
        self.model = self.base
        if adapter is not None:
            self.model = PeftModel.from_pretrained(self.base, str(adapter)).eval()
            self.attached = True

    def generate(self, prompts: list[str], max_new_tokens: int, tick: Any = None) -> dict[str, str]:
        """{prompt: generated text through the first end token}. Prompts of one token length share a batch."""
        torch = self.torch
        by_length: dict[int, list[tuple[str, list[int]]]] = defaultdict(list)
        for prompt in dict.fromkeys(prompts):
            ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            by_length[len(ids)].append((prompt, ids))
        out: dict[str, str] = {}
        for length, group in sorted(by_length.items()):
            for start in range(0, len(group), 16):
                batch = group[start:start + 16]
                input_ids = torch.tensor([ids for _, ids in batch], device="cuda")
                with torch.inference_mode():
                    made = self.model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                               max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.0,
                                               eos_token_id=self.ends, pad_token_id=self.ends[-1])
                for (prompt, _), row in zip(batch, made.tolist()):
                    new = row[length:]
                    for position, token in enumerate(new):
                        if token in self.ends:
                            new = new[:position + 1]
                            break
                    out[prompt] = self.tokenizer.decode(new, skip_special_tokens=False)
                if tick:
                    tick(len(out), len(by_length) and sum(len(g) for g in by_length.values()))
        return out


class Replay:
    """A completer for lab/tool_call_agreement.py that returns what the generator already wrote for a prompt."""

    def __init__(self, outputs: dict[str, str]) -> None:
        self.outputs = outputs

    def complete(self, prompt: str) -> dict[str, Any]:
        return {"text": self.outputs[prompt], "text_source": "transformers generate, bf16"}


def prompts_of(manifest: Any, examples: list[Any]) -> list[str]:
    from lab import tool_call_agreement as tca
    from lab.fixture_dataset_v2 import system_for

    return [tca.prompt_for(system_for(manifest, example), example.context or "") for example in examples]


def validation_examples(dataset: Path) -> tuple[Any, list[Any]]:
    from lab.dataset import load_split
    from lab.fixture_dataset_v2 import load_validation

    split = load_split(dataset, require_images=False)
    wanted = set(load_validation(dataset)["validation"])
    return split, [example for example in split.train if example.id in wanted]


def choose(job: Job, args: argparse.Namespace, facts: list[dict[str, Any]]) -> dict[str, Any]:
    from lab import tool_call_agreement as tca
    from lab.fixture_dataset import load_expected
    from lab.fixture_dataset_v2 import system_for

    dataset = Path(args.dataset)
    eligible = [f for f in facts if f["ok"] and f.get("eligible") and f["best_val"] is not None]
    if not eligible:
        raise RuntimeError("no eligible run trained")
    ranked = sorted(eligible, key=lambda f: (f["best_val"], f["best_passes"], f["index"]))
    top = ranked[:TOP]
    split, examples = validation_examples(dataset)
    expected = load_expected(dataset)
    manifest = split.data.manifest
    prompts = prompts_of(manifest, examples)
    generator = Generator(args.model)
    folder = job.out / "validation-agreement"
    folder.mkdir(parents=True, exist_ok=True)
    scores: dict[str, dict[str, Any]] = {}
    arms = [("base", None)] + [(f["name"], job.runs / f["name"] / "adapter") for f in top]
    for number, (name, adapter) in enumerate(arms, 1):
        generator.attach(adapter)
        began = time.time()
        outputs = generator.generate(prompts, tca.MAX_TOKENS, lambda done, total: job.status(
            stage="val_agreement", run=f"{number}/{len(arms)}", arm=name, step=f"{done}/{total}", lr=None, rank=None,
            drop=None, loss=None, val=None, eta_s="na"))
        with (folder / f"validation.{name}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            rows = tca.run_arm(name, Replay(outputs), examples, expected, lambda example: system_for(manifest, example),
                               lambda row: handle.write(json.dumps(row, ensure_ascii=False) + "\n"))
        correct = sum(bool(row["correct"]) for row in rows.values())
        scores[name] = {"correct": correct, "n": len(rows), "rate": correct / len(rows),
                        "names": sum(bool(row["names"]) for row in rows.values()),
                        "parsed": sum(bool(row["parsed"]) for row in rows.values()),
                        "seconds": round(time.time() - began, 1)}
        job.finished(f"val_agreement arm={name} correct={correct}/{len(rows)} names={scores[name]['names']} "
                     f"parsed={scores[name]['parsed']} seconds={scores[name]['seconds']}")
    generator.attach(None)
    for fact in top:
        fact["val_agreement"] = scores[fact["name"]]
    best = sorted(top, key=lambda f: (-f["val_agreement"]["correct"], f["best_val"], f["best_passes"], f["index"]))[0]
    chosen = {
        "rule": CHOICE_RULE, "scoring_rule": tca.RULE, "decode": "transformers greedy, bf16 weights with the adapter "
        "attached, max 512 new tokens, prompts of equal token length batched (no padding)",
        "dataset_id": split.dataset_id, "split_sha256": split.sha256, "validation_examples": len(examples),
        "validation_groups": len({example.frame_ref for example in examples}),
        "base_validation_agreement": scores["base"],
        "ranking_by_validation_loss": [{"name": f["name"], "best_val": f["best_val"], "best_step": f["best_step"],
                                        "best_passes": f["best_passes"]} for f in ranked],
        "top": [{"name": f["name"], "best_val": f["best_val"], "val_agreement": f["val_agreement"]} for f in top],
        "chosen": {key: best[key] for key in ("name", "index", "lr", "rank", "alpha", "dropout", "warmup_ratio",
                                              "max_steps", "best_step", "best_passes", "best_val", "final_val",
                                              "passes", "wall_s", "torch_loss_head", "val_agreement")},
        "recipe_flags": ["--max-steps", str(best["max_steps"]), "--stop-step", str(best["best_step"]),
                         "--lr", f"{best['lr']:g}", "--rank", str(best["rank"]), "--alpha", str(best["alpha"]),
                         "--dropout", f"{best['dropout']:g}", "--warmup-ratio", str(WARMUP_RATIO),
                         "--batch", str(BATCH), "--grad-accum", str(GRAD_ACCUM), "--seed", "42"],
        "made_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (job.out / "chosen.json").write_text(json.dumps(chosen, indent=2) + "\n", encoding="utf-8")
    run_dir = job.runs / best["name"]
    members = [(run_dir / "adapter", "adapter"), (job.out / "chosen.json", "chosen.json"),
               (run_dir / "run.json", "run.json"), (job.out / "sweep.json", "sweep.json"),
               (folder, "validation-agreement")]
    job.product("best-adapter.tar.gz", members, "BEST_DONE")
    job.finished(f"chosen name={best['name']} lr={best['lr']:g} rank={best['rank']} drop={best['dropout']:g} "
                 f"best_step={best['best_step']} best_passes={best['best_passes']} best_val={fmt(best['best_val'])} "
                 f"val_agreement={best['val_agreement']['correct']}/{best['val_agreement']['n']} "
                 f"base_val_agreement={scores['base']['correct']}/{scores['base']['n']}")
    return chosen


def held_out(job: Job, args: argparse.Namespace, chosen: dict[str, Any]) -> None:
    """After the choice: base against the chosen adapter on the held-out side of both datasets, generated here."""
    from lab import tool_call_agreement as tca
    from lab.dataset import load_split

    generator = Generator(args.model)
    adapter = job.runs / chosen["chosen"]["name"] / "adapter"
    folder = job.out / "held-out-bf16"
    for label, dataset in (("v2", Path(args.dataset)), ("fixture", REPO_ROOT / "fixtures" / "learning" / "tool-calling-fixture")):
        split = load_split(dataset, require_images=False)
        examples = list(split.held_out)
        prompts = prompts_of(split.data.manifest, examples)
        outputs = {}
        for number, (arm, path) in enumerate((("before", None), ("after", adapter)), 1):
            generator.attach(path)
            outputs[arm] = generator.generate(prompts, tca.MAX_TOKENS, lambda done, total: job.status(
                stage="held_out_bf16", run=f"{label}:{number}/2", arm=arm, step=f"{done}/{total}", eta_s="na"))
        summary = tca.run(dataset=dataset, completers={arm: Replay(text) for arm, text in outputs.items()},
                          out=folder / label, run_id=f"a100-bf16-{chosen['chosen']['name']}", side="held_out",
                          before_runs=1, after_runs=1,
                          servers={"before": {"model": "Qwen3.5-4B bf16, transformers, A100"},
                                   "after": {"model": f"the same with adapter {chosen['chosen']['name']}"}})
        job.finished(f"held_out_bf16 dataset={split.dataset_id} n={summary['n_items']} groups={summary['n_groups']} "
                     f"before={summary['correct']['current']} after={summary['correct']['candidate']} "
                     f"gained={summary['mcnemar']['gained_0_to_1']} lost={summary['mcnemar']['lost_1_to_0']} "
                     f"p_mcnemar={summary['mcnemar']['p_value']:.4g} p_sign_flip={summary['p_value']:.4g}")
    generator.attach(None)
    job.product("held-out-bf16.tar.gz", [(folder, "held-out-bf16")], "HELD_OUT_DONE")


def smoke_generation(job: Job, args: argparse.Namespace) -> None:
    """The generation and scoring code paths on a handful of validation and train examples (never a held-out one),
    with the smoke adapter attached and taken off again, so a bug shows before the sweep and not after it."""
    from lab import tool_call_agreement as tca
    from lab.fixture_dataset import load_expected
    from lab.fixture_dataset_v2 import system_for

    dataset = Path(args.dataset)
    split, examples = validation_examples(dataset)
    examples = examples[:3] + examples[-3:]
    expected = load_expected(dataset)
    manifest = split.data.manifest
    prompts = prompts_of(manifest, examples)
    generator = Generator(args.model)
    seen: dict[str, dict[str, str]] = {}
    for name, adapter in (("base", None), ("smoke", job.runs / "smoke" / "adapter"), ("base-again", None)):
        generator.attach(adapter)
        began = time.time()
        seen[name] = generator.generate(prompts, tca.MAX_TOKENS)
        rows = tca.run_arm(name, Replay(seen[name]), examples, expected, lambda example: system_for(manifest, example))
        job.finished(f"smoke_generation arm={name} correct={sum(bool(r['correct']) for r in rows.values())}/{len(rows)} "
                     f"parsed={sum(bool(r['parsed']) for r in rows.values())} seconds={time.time() - began:.1f} "
                     f"first_output={ascii(seen[name][prompts[0]])[:150]}")
    job.finished(f"smoke_generation base_same_after_unload={seen['base'] == seen['base-again']}")
    train = list(split.train)[:4]
    outputs = generator.generate(prompts_of(manifest, train), tca.MAX_TOKENS)
    summary = tca.run(dataset=dataset, completers={"before": Replay(outputs), "after": Replay(outputs)},
                      out=job.out / "smoke-agreement", run_id="smoke", side="train", before_runs=1, after_runs=1, limit=4)
    job.finished(f"smoke_generation summary_path_ok n={summary['n_items']} correct={summary['correct']['current']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--budget-s", type=float, default=2400.0)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--only", choices=("smoke", "sweep", "choose", "held_out"))
    args = parser.parse_args()
    job = Job(Path(args.out), args.budget_s)
    (job.out / "choice_rule.txt").write_text(CHOICE_RULE + "\n", encoding="utf-8")
    if args.only == "smoke":   # two optimizer steps with an evaluation after each: the flags work on this GPU
        args.stage = "smoke"
        fact = train_once(job, args, "smoke", 1e-4, 16, 0.05, 2, "0/0", False, ["--eval-steps", "1"])
        if not fact["ok"]:
            fact = train_once(job, args, "smoke", 1e-4, 16, 0.05, 2, "0/0", True, ["--eval-steps", "1"])
        job.finished(f"smoke outcome={fact['outcome']} torch_loss_head={fact['torch_loss_head']} wall_s={fact['wall_s']} "
                     f"best_val={fmt(fact['best_val'])} samples_per_s={fmt(fact['samples_per_s'], 2)} "
                     f"error={str(fact['error'])[:160]}")
        if fact["ok"]:
            smoke_generation(job, args)
        return 0 if fact["ok"] else 1
    facts: list[dict[str, Any]] = []
    if args.only in (None, "sweep"):
        args.stage = "sweep"
        facts = sweep(job, args)
    else:
        facts = json.loads((job.out / "sweep.json").read_text(encoding="utf-8"))["runs"]
    if args.only in (None, "choose"):
        job.status(stage="val_agreement", run=None, lr=None, rank=None, drop=None, step=None, loss=None, val=None, eta_s="na")
        chosen = choose(job, args, facts)
    else:
        chosen = json.loads((job.out / "chosen.json").read_text(encoding="utf-8")) if (job.out / "chosen.json").exists() else {}
    if args.only in (None, "held_out") and chosen:
        held_out(job, args, chosen)
    job.status(stage="sweep_script_done", run=None, arm=None, step=None, eta_s=0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
