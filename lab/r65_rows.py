"""R65: the metrics rows of the two-pod work, rebuilt from the files in fixtures/lab/r65-2026-10-11/.

    python lab/r65_rows.py            print the rows (one JSON line each)
    python lab/r65_rows.py --write    replace R65's rows in lab/refine/metrics.jsonl with them

Rows of a two-pod measurement name both machines, node "pod-a+pod-b" (rank 0 ran on pod C, the teacher's pod-a role,
rank 1 on pod D, pod-b). The trainer rows of a split run are named trainer.split.* so the box's stats page, which shows
the newest trainer.* row of a single-pod run, never shows a pair's number as one pod's. Each run folder holds both ranks'
run.json and local_sgd.json as lab/r65_local_sgd.py wrote them; the link rows come from the transcribed probe outputs.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "lab" / "r65-2026-10-11"
REL = "fixtures/lab/r65-2026-10-11"
NODE = "pod-a+pod-b"
L16 = {"train_s": 784.541140422225, "samples_per_s": 0.367093559739906, "loss_last": 0.009453155100345612,
       "train_loss": 0.054358619886140026, "steps": 18}   # fixtures/demo/l16-run.json, the single-pod control

# Each run: its folder, what differs from the L16 recipe, and the slot rows it ran in.
RUNS = {
    "lsgd1": {"epochs": 3, "lr": 2e-4, "relay_in_train_s": True,
              "what": "same 252 examples as L16 split in two: 9 steps a rank (half L16's 18)",
              "slots": "GPU Queue rows of R65 on GPU A and GPU B, 08:08 to 08:18 UTC"},
    "lsgd2": {"epochs": 6, "lr": 2e-4, "relay_in_train_s": False,
              "what": "matched steps: 18 steps a rank as L16, the pair sees twice L16's examples",
              "slots": "GPU Queue rows of R65 on GPU A and GPU B, 08:33 to 08:50 UTC"},
    "lsgd3": {"epochs": 6, "lr": 2.83e-4, "relay_in_train_s": False,
              "what": "matched steps with the learning rate scaled by sqrt(2) for the doubled effective batch",
              "slots": "GPU Queue rows of R65 on GPU A and GPU B (sqrt-scaled lr)"},
}
SHARED = {
    "model": "Qwen/Qwen3.5-4B@851bf6e", "chip": "Strix Halo gfx1151 x2", "torch": "2.9.1+rocm7.13.0",
    "recipe": "L16: LoRA r16 alpha32 batch4 x accum4 seed42", "dataset": "tool-calling-fixture, L16 rows sha 628bd220 "
    "split by alternate rows into two halves of 42",
    "mode": "local SGD, 2 ranks, int8 deltas with error feedback every 4 optimizer steps, merged one step late, "
            "final synchronous round",
    "ranks": "rank 0 pod C (pod-a, beside the live teacher), rank 1 pod D (pod-b)",
    "control_path": "tailscale userspace via DERP sfo", "data_path": "hub in-cluster proxy proxy-public:80",
    "measure": "examples seen (steps x batch x accumulation) over trainer.train() wall time",
    "low_confidence_reason": "one run per setting",
}


def _row(metric: str, value: float, unit: str, conditions: dict[str, Any], source: str, ts: str) -> dict[str, Any]:
    return {"metric": metric, "value": round(float(value), 4), "unit": unit, "node": NODE, "conditions": conditions,
            "source": source, "ts": ts, "low_confidence": True}


def run_rows(run: str) -> list[dict[str, Any]]:
    folder = FIXTURES / run
    if not (folder / "rank0-run.json").exists():
        return []
    spec = RUNS[run]
    runs = {r: json.loads((folder / f"rank{r}-run.json").read_text(encoding="utf-8")) for r in (0, 1)}
    sgd = {r: json.loads((folder / f"rank{r}-local_sgd.json").read_text(encoding="utf-8")) for r in (0, 1)}
    m = {r: runs[r]["metrics"] for r in (0, 1)}
    ts = max(r["at"] for s in sgd.values() for r in s["rounds"])
    source = f"R65 run {run}, lab/r65_local_sgd.py over lab/train.py, {REL}/{run}/; {spec['slots']}"
    base = {**SHARED, "run": run, "epochs": spec["epochs"], "lr": spec["lr"], "what": spec["what"]}
    rows = []
    wall = max(m[r]["trainer.train_s"] for r in (0, 1))
    seen = sum(m[r]["trainer.examples_seen"] for r in (0, 1))
    rows.append(_row("trainer.split.samples_per_s", seen / wall, "samples/s",
                     {**base, "scope": "pair", "examples_seen": seen, "control_value": L16["samples_per_s"]}, source, ts))
    note = "includes the final round and one relay fetch" if spec["relay_in_train_s"] else "includes the final round"
    rows.append(_row("trainer.split.train_s", wall, "s", {**base, "scope": f"pair, the slower rank ({note})",
                                                           "control_value": L16["train_s"]}, source, ts))
    for r in (0, 1):
        scope = {**base, "scope": f"rank {r}", "steps": m[r]["trainer.steps"]}
        rows.append(_row("trainer.split.samples_per_s", m[r]["trainer.samples_per_s"], "samples/s", scope, source, ts))
        rows.append(_row("trainer.split.loss_last", m[r]["trainer.loss_last"], "loss",
                         {**scope, "control_value": L16["loss_last"]}, source, ts))
        rows.append(_row("trainer.split.train_loss", m[r]["trainer.train_loss"], "loss",
                         {**scope, "control_value": L16["train_loss"]}, source, ts))
        rows.append(_row("trainer.split.peak_mem_gb", m[r]["trainer.peak_mem_gb"], "GB", scope, source, ts))
    rounds = [x for r in (0, 1) for x in sgd[r]["rounds"]]
    rows.append(_row("link.exchange_s", statistics.median(x["fetch_s"] for x in rounds), "s",
                     {**base, "path": "hub proxy", "bytes": rounds[0]["bytes_fetched"], "chunks": rounds[0]["chunks"],
                      "parallel": sgd[0].get("parallel", 4), "rounds": len(rounds), "during": "training on both ranks"},
                     source, ts))
    relay = [sgd[r]["relay_exchange"]["seconds"] for r in (0, 1) if "seconds" in sgd[r].get("relay_exchange", {})]
    if relay:
        rows.append(_row("link.exchange_s", statistics.median(relay), "s",
                         {**base, "path": "tailscale relay (DERP)", "bytes": rounds[0]["bytes_fetched"],
                          "chunks": rounds[0]["chunks"], "rounds": len(relay)}, source, ts))
    rows.append(_row("link.exchange_bytes", rounds[0]["bytes_fetched"], "bytes",
                     {**base, "what_bytes": "one round's int8 code: one int8 a LoRA weight and one float32 scale per 4096"},
                     source, ts))
    rows.append(_row("link.exchange_pack_s", statistics.median(x["pack_s"] for x in rounds if "final" not in x["label"]),
                     "s", {**base, "what_s": "copy to the CPU, delta, int8 code and residual, before the barrier"},
                     source, ts))
    heldout = folder / "candidate-heldout.json"
    if heldout.exists():
        h = json.loads(heldout.read_text(encoding="utf-8"))
        rows.append(_row("trainer.split.heldout_passed", h["passed"], "items",
                         {**base, "scored": h["scored"], "split_sha256": h["split_sha256"], "control_value": 24,
                          "control": "L16's package, the gate's candidate arm: 24 of 24 (lab/L5_RESULT.md)",
                          "how": h["note"], "decode": "greedy (top_k 1), thinking off"},
                         f"R65 run {run}, {REL}/{run}/candidate-heldout.json and candidate-golden-eval.json", ts))
    rows.append(_row("link.exchange_reconstruction_rel_err", statistics.median(x["reconstruction_rel_err"] for x in rounds),
                     "ratio", {**base, "what_ratio": "norm of (delta - int8 code) / norm of delta, median over rounds"},
                     source, ts))
    return rows


def link_rows() -> list[dict[str, Any]]:
    """The link and role-switch numbers, from the probe outputs kept in the fixtures folder."""
    why = {"low_confidence_reason": "one probe run"}
    rows = [
        {"metric": "pod.teacher_start_s", "value": 451.0, "unit": "s", "node": "pod-b",
         "conditions": {**why, "what": "lab/pod_role.sh teacher D --apply: vllm_upgrade.sh serve from venv-vllm-new until "
                                       "/v1/models answers", "engine": "vllm 0.31.2.dev0+g5281e4990.gfx1151",
                        "checkpoint": "Qwen3.8-27B-INT4", "hub": "sjc", "account": "D", "max_model_len": 16384,
                        "gpu_memory_utilization": 0.9},
         "source": f"R65, {REL}/role-switch-pod-D.txt; GPU Queue row of R65 07:25 to 07:36 UTC",
         "ts": "2026-10-11T07:33:15Z", "low_confidence": True},
        {"metric": "link.rtt_ms", "value": 10.15, "unit": "ms", "node": NODE,
         "conditions": {"path": "tailscale userspace, DERP sfo, TCP through tailscaled SOCKS5", "samples": 200,
                        "direction": "pod C to pod D"},
         "source": f"R65, {REL}/tailscale-probe-C-to-D.txt", "ts": "2026-10-11T07:44:00Z", "low_confidence": False},
        {"metric": "link.mb_s", "value": 2.70, "unit": "MB/s", "node": NODE,
         "conditions": {**why, "path": "tailscale userspace, DERP sfo, TCP through tailscaled SOCKS5", "bytes": 1 << 30,
                        "direction": "pod C to pod D"},
         "source": f"R65, {REL}/tailscale-probe-C-to-D.txt", "ts": "2026-10-11T07:44:00Z", "low_confidence": True},
        {"metric": "link.rtt_ms", "value": 3.93, "unit": "ms", "node": NODE,
         "conditions": {**why, "path": "hub in-cluster proxy proxy-public:80, HTTP GET", "samples": 50,
                        "direction": "pod C to pod D"},
         "source": f"R65, {REL}/relay-and-tailscale-transcribed.txt section 1", "ts": "2026-10-11T07:20:00Z",
         "low_confidence": True},
        {"metric": "link.mb_s", "value": 98.0, "unit": "MB/s", "node": NODE,
         "conditions": {**why, "path": "hub in-cluster proxy proxy-public:80, HTTP GET of 32.5 MB in 8 MB chunks, "
                                      "4 connections", "direction": "both, median of 4"},
         "source": f"R65, {REL}/relay-and-tailscale-transcribed.txt section 4", "ts": "2026-10-11T08:06:00Z",
         "low_confidence": True},
    ]
    return rows


def all_rows() -> list[dict[str, Any]]:
    rows = []
    for run in RUNS:
        rows.extend(run_rows(run))
    return link_rows() + rows


def is_r65(row: dict[str, Any]) -> bool:
    return str(row.get("source", "")).startswith("R65")


def write(path: Path = ROOT / "lab" / "refine" / "metrics.jsonl") -> int:
    sys.path.insert(0, str(ROOT))
    from lab.machine import metrics

    kept = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not is_r65(json.loads(line))]
    rows = [metrics.check_row(row) for row in all_rows()]
    path.write_text("\n".join(kept + [json.dumps(dict(row)) for row in rows]) + "\n", encoding="utf-8", newline="\n")
    return len(rows)


if __name__ == "__main__":
    if "--write" in sys.argv:
        print(f"{write()} R65 rows written")
    else:
        for row in all_rows():
            print(json.dumps(row))
