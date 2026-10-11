"""A stand-in for lab/train.py in tests: the same arguments, the same files, no model and no GPU.

    python lab/train_fake.py --dataset D --rows R --model M --max-steps 20 --batch 4 --grad-accum 1 --out OUT \\
        --run-id ID [--step-s 0.05]

It writes what lab/interference.py reads from a real run: one telemetry.train.step event per optimizer step to
<out>/telemetry.jsonl, each --step-s apart (ts is the step's end, step_ms its length), and <out>/run.json with the
trainer's answer (outcome trained, trainer.samples_per_s over the steps' window). Like the real trainer on a pod, it
deletes its copy of the dataset folder and the rows file when it ends. Stdlib only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    for name in ("--dataset", "--rows", "--model", "--out", "--run-id"):
        p.add_argument(name, required=True)
    p.add_argument("--max-steps", type=int, required=True)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--step-s", type=float, default=0.05)
    args, _ = p.parse_known_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with open(out / "telemetry.jsonl", "w", encoding="utf-8") as handle:
        for step in range(1, args.max_steps + 1):
            begin = time.time()
            time.sleep(args.step_s)
            end = time.time()
            event = {"type": "telemetry.train.step", "node": "pod-b", "run_id": args.run_id, "step": step,
                     "ts": dt.datetime.fromtimestamp(end, dt.timezone.utc).isoformat(), "loss": 1.0 / step,
                     "step_ms": round((end - begin) * 1000.0, 3), "socket_w": None, "tokens": 512}
            handle.write(json.dumps(event) + "\n")
            handle.flush()
    seconds = time.time() - started
    shutil.rmtree(args.dataset, ignore_errors=True)
    Path(args.rows).unlink(missing_ok=True)
    answer = {"ok": True, "outcome": "trained", "metrics": {
        "trainer.samples_per_s": round(args.max_steps * args.batch * args.grad_accum / seconds, 4),
        "trainer.peak_mem_gb": 11.1}, "refs": {"run_id": args.run_id, "model": args.model}}
    (out / "run.json").write_text(json.dumps(answer, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(answer))
    return 0


if __name__ == "__main__":
    sys.exit(main())
