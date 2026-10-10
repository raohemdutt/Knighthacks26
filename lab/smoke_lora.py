"""Smoke LoRA SFT on one R9700: proves the training stack (PyTorch ROCm + PEFT + TRL) works end to end.

Trains a tiny LoRA on a synthetic dataset for a handful of steps and prints loss and samples/s.
Usage (on pod B):  python3 lab/smoke_lora.py --model ~/models/Qwen3.5-4B --steps 20
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run as a bare script on a pod, "lab" is one folder up
from lab.machine.record import recorded


def build_dataset(n: int) -> list[dict]:
    items = ["keys", "wallet", "mug", "phone", "notebook", "charger", "glasses", "remote"]
    places = ["on the desk", "by the door", "on the couch", "next to the laptop", "in the bowl", "on the shelf"]
    rows = []
    for i in range(n):
        item, place = items[i % len(items)], places[i % len(places)]
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": f"Where did I leave my {item}?"},
                    {"role": "assistant", "content": f"I last saw your {item} {place}."},
                ]
            }
        )
    return rows


@recorded("smoke-lora")
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path.home() / "models" / "Qwen3.5-4B"))
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--examples", type=int, default=48)
    ap.add_argument("--out", default=str(Path.home() / "runs" / "smoke-lora"))
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    assert torch.cuda.is_available(), "no HIP device visible"
    print("device", torch.cuda.get_device_name(0), "hip", torch.version.hip)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": 0})
    model.config.use_cache = False

    ds = Dataset.from_list(build_dataset(args.examples))
    lora = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM")
    cfg = SFTConfig(
        output_dir=args.out,
        max_steps=args.steps,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        dataloader_num_workers=2,
        dataloader_persistent_workers=False,
        max_length=256,
        torch_compile=False,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lora)
    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0
    seen = args.steps * cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    summary = {
        "steps": args.steps,
        "examples_seen": seen,
        "elapsed_s": round(elapsed, 1),
        "samples_per_s": round(seen / elapsed, 2),
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "train_loss": result.training_loss,
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "summary.json").write_text(json.dumps(summary, indent=2))
    print("SMOKE_LORA", json.dumps(summary))


if __name__ == "__main__":
    mp.set_start_method("forkserver", force=True)
    main()
