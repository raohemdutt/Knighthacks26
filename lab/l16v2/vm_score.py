"""Scoring on a Colab worker for the search driven from the workstation (lab/l16v2/bo_local.py).

    python3 /content/vm_score.py validation <label> <arm> [<arm> ...]     # an arm is "base" or a run under runs/
    python3 /content/vm_score.py held_out <chosen run>                      # only after the choice is written

Run from the repo clone on the VM. It reuses lab/l16v2/sweep.py unchanged: its Generator (transformers greedy on the
bf16 weights with the adapter attached, prompts of equal token length batched so nothing is padded), its Replay
completer, and lab/tool_call_agreement.py's own prompt, marks and summary. validation writes scores/<label>/ (one
jsonl of rows per arm and scores.json with the counts), packs it as products/score-<label>.tar.gz with its .sha256
and then writes scores/<label>.DONE (or .FAILED); the status line is scores/<label>.status. held_out is
sweep.held_out(): base against the chosen adapter on the held-out side of tool-calling-v2 and of
tool-calling-fixture, as products/held-out-bf16.tar.gz with the marker HELD_OUT_DONE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import time
import traceback
from pathlib import Path

REPO = Path("/content/Knighthacks26")
OUT = Path("/content/l16v2")
MODEL = "/content/models/Qwen3.5-4B"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lab" / "l16v2"))

import sweep  # noqa: E402


def validation(label: str, arms: list[str]) -> None:
    from lab import tool_call_agreement as tca
    from lab.fixture_dataset import load_expected
    from lab.fixture_dataset_v2 import system_for

    dataset = OUT / "data" / "tool-calling-v2"
    folder = OUT / "scores" / label
    folder.mkdir(parents=True, exist_ok=True)
    status = OUT / "scores" / f"{label}.status"
    split, examples = sweep.validation_examples(dataset)
    expected = load_expected(dataset)
    manifest = split.data.manifest
    prompts = sweep.prompts_of(manifest, examples)
    generator = sweep.Generator(MODEL)
    scores = {}
    for number, arm in enumerate(arms, 1):
        generator.attach(None if arm == "base" else OUT / "runs" / arm / "adapter")
        began = time.time()

        def tick(done: int, total: int, arm: str = arm, number: int = number) -> None:
            status.write_text(f"stage=val_agreement arm={arm} n={number}/{len(arms)} step={done}/{total} "
                              f"elapsed_s={int(time.time() - began)}\n", encoding="ascii")

        outputs = generator.generate(prompts, tca.MAX_TOKENS, tick)
        with (folder / f"validation.{arm}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            rows = tca.run_arm(arm, sweep.Replay(outputs), examples, expected, lambda example: system_for(manifest, example),
                               lambda row: handle.write(json.dumps(row, ensure_ascii=False) + "\n"))
        correct = sum(bool(row["correct"]) for row in rows.values())
        scores[arm] = {"correct": correct, "n": len(rows), "rate": correct / len(rows),
                       "names": sum(bool(row["names"]) for row in rows.values()),
                       "parsed": sum(bool(row["parsed"]) for row in rows.values()),
                       "seconds": round(time.time() - began, 1), "groups": len({example.frame_ref for example in examples}),
                       "rule_sha256": hashlib.sha256(tca.RULE.encode("utf-8")).hexdigest(), "split_sha256": split.sha256}
    generator.attach(None)
    (folder / "scores.json").write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    name = f"score-{label}.tar.gz"
    path = OUT / "products" / name
    with tarfile.open(path, "w:gz") as tar:
        tar.add(folder, arcname=label)
    (OUT / "products" / (name + ".sha256")).write_text(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()} {path.stat().st_size} {name}\n", encoding="ascii")
    (OUT / "scores" / f"{label}.DONE").write_text(name + "\n", encoding="ascii")


def held_out(name: str) -> None:
    job = sweep.Job(OUT, 0.0)
    args = argparse.Namespace(dataset=str(OUT / "data" / "tool-calling-v2"), model=MODEL)
    sweep.held_out(job, args, {"chosen": {"name": name}})


def main() -> int:
    what, rest = sys.argv[1], sys.argv[2:]
    label = rest[0] if what == "validation" else "held_out"
    (OUT / "scores").mkdir(parents=True, exist_ok=True)
    try:
        if what == "validation":
            validation(rest[0], rest[1:])
        else:
            held_out(rest[0])
    except Exception:
        traceback.print_exc()
        (OUT / "scores" / f"{label}.FAILED").write_text(traceback.format_exc()[-400:], encoding="utf-8")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
