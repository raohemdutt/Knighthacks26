"""Writes fixtures/lab/train_profile/cpu/ from a recorded `python -m lab profile-train run --cpu` folder: the last two
steps of its torch trace, gzip'd, with the op arguments the parser reads (shapes and dtypes of the matmul ops only)
and every tenth allocator sample, plus the run's metrics and the refs the profile reads, and profile.json.

    python fixtures/lab/train_profile/trim_cpu_trace.py <run folder>

The run it was cut from: tests/test_lab_profile_train_cpu.py's tiny Qwen3.5 (one Gated DeltaNet layer, one attention
layer, LoRA r 16 on all-linear), three steps of batch 2, on the CPU with torch 2.9.1, transformers 5.19.0, PEFT 0.21.2,
TRL 1.15.0.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent / "cpu"
REFS = ("run_id", "mode", "node", "deltanet_kernel", "device", "torch", "rocm", "transformers", "peft", "trl",
        "layer_ranges", "model_class", "loss_head")
KEEP_ARGS = {"aten::mm", "aten::addmm", "aten::bmm", "aten::baddbmm"}


def trim(data: dict, keep_steps: int = 2) -> dict:
    events = data["traceEvents"]
    steps = sorted((e for e in events if str(e.get("name", "")).startswith("ProfilerStep#")), key=lambda e: e["ts"])
    chosen = steps[-keep_steps:]
    start, end = chosen[0]["ts"], chosen[-1]["ts"] + chosen[-1]["dur"]
    kept = []
    memory = 0
    for e in events:
        if e.get("ph") == "M":
            kept.append(e)
            continue
        ts = e.get("ts")
        if not isinstance(ts, (int, float)) or not start <= ts <= end:
            continue
        e = dict(e)
        if e.get("name") == "[memory]":
            memory += 1
            if memory % 10:
                continue
        elif e.get("cat") == "cpu_op":
            args = e.get("args") or {}
            e["args"] = ({k: args[k] for k in ("Input Dims", "Input type") if k in args}
                         if e.get("name") in KEEP_ARGS else {})
            if not e["args"]:
                e.pop("args")
        kept.append(e)
    out = {k: v for k, v in data.items() if k != "traceEvents"}
    out["traceEvents"] = kept
    return out


def main(run_dir: str) -> None:
    run = Path(run_dir)
    data = json.loads((run / "torch" / "trace.json").read_text(encoding="utf-8"))
    HERE.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(trim(data), separators=(",", ":")).encode("utf-8")
    (HERE / "trace.json.gz").write_bytes(gzip.compress(blob, mtime=0))
    answer = json.loads((run / "run.json").read_text(encoding="utf-8"))
    refs = {k: v for k, v in (answer.get("refs") or {}).items() if k in REFS}
    slim = {"ok": answer.get("ok"), "outcome": answer.get("outcome"), "metrics": answer.get("metrics"), "refs": refs}
    (HERE / "run.json").write_text(json.dumps(slim, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    meta = json.loads((run / "profile.json").read_text(encoding="utf-8"))
    (HERE / "profile.json").write_text(json.dumps({"cpu": True, "date": meta["date"], "steps": 2}, indent=2) + "\n",
                                       encoding="utf-8")
    print(f"wrote {HERE} ({len(blob)} bytes of JSON)")


if __name__ == "__main__":
    main(sys.argv[1])
