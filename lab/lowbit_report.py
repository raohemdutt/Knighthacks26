"""The low-bit teacher's prediction and report: bytes per decode step, the bandwidth ceiling, and what was measured.

    python lab/lowbit_report.py predict                       # the ceilings per width at batch 1 and 4
    python lab/lowbit_report.py report --bench 4=DIR --bench 3=DIR --bench 2=DIR \\
        --eval 4=int4.json --eval 3=w3.json --eval 2=w2.json \\
        [--correctness 3=compare-w3.json --correctness 2=compare-w2.json] [--margin 0.05] [--date 2026-10-11]

`python -m lab lowbit-report ...` is the same.

Predict. Decode on the Strix Halo is weight-read bound: each step reads every quantized layer once, the bf16 lm_head,
and the Gated DeltaNet state of each sequence. From the serving checkpoint's shapes (w4a16_decode_gfx1201.md section
1.2: 12.60 GB of int4 values and bf16 group-128 scales, 2.54 GB of bf16 lm_head, about 0.30 GB of float32 state per
sequence) the quantized layers hold QUANT_PARAMS weights; at W bits with one bf16 scale per 128 weights they cost
QUANT_PARAMS x (W + 16/128) / 8 bytes. The bandwidth is the chip sheet's device copy rate above the 32 MiB cache
(the median over those sizes, about 210 GB/s), and the watts are the chip sheet's idle and the loaded band the
benchmarks saw (75 to 128 W). At batch B a step serves B tokens and reads the weights once and the state B times,
so the ceiling is B x bandwidth / (weights + lm_head + B x state) tokens per second, and joules per token is watts
over that rate. These are ceilings, never predictions of what a kernel reaches.

Report. Reads lab/bench.py sessions (DIR/rows.jsonl, rows with outcome ok; one session per width), the L4 scorer's
JSON for each width (`python -m contexts.agent.eval --runtime llama --base-url URL --json FILE`: its `score`), and
optionally a correctness compare per width against the INT4 baseline (`python -m lab correctness compare ...
--allow-condition-drift`: its match_rate and normalised_text_matches; a different checkpoint is expected to diverge,
so this is reported, not gated). It writes lab/refine/research/kernels/data/lowbit-gfx1151-<date>.json with, per width
and batch size, the median tok/s, watts and joules per token over the session's spawns beside the ceiling and the
fraction of it reached, the eval score and its drop from the 4-bit teacher, and the width it names: the lowest width
whose eval drop is within --margin (in score points), or none. Without --margin it names none and says the margin was
not given.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import date as _date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHIPSHEET = REPO / "lab" / "refine" / "research" / "kernels" / "data" / "chipsheet-gfx1151-2026-10-11.json"
OUT_DIR = REPO / "lab" / "refine" / "research" / "kernels" / "data"

GROUP = 128
INT4_LAYER_GB = 12.60                           # int4 values plus bf16 scales, RedHatAI/Qwen3.8-27B-INT4
QUANT_PARAMS = INT4_LAYER_GB * 1e9 * 8 / (4 + 16 / GROUP)   # weights in the quantized layers, about 24.4 billion
LM_HEAD_GB = 2.54                               # bf16 lm_head, kept unquantized at every width
STATE_GB = 0.30                                 # float32 Gated DeltaNet state per sequence, read and written
CACHE_BYTES = 32 * 1024 * 1024                  # above this the copy rate is DRAM's
LOADED_W = (75.0, 128.0)                        # the loaded band the gfx1151 benchmarks saw
WIDTHS = (4, 3, 2)
BATCHES = (1, 4)


class ReportError(RuntimeError):
    """An input the report cannot use; the message says which."""


def bits_per_weight(width: int) -> float:
    return width + 16 / GROUP


def step_gb(width: int, batch: int = 1) -> float:
    """Bytes read per decode step at this width and batch size, in GB."""
    return QUANT_PARAMS * bits_per_weight(width) / 8 / 1e9 + LM_HEAD_GB + batch * STATE_GB


def bandwidth_gbs(chipsheet: dict) -> float:
    """The median device copy rate over sizes above the cache, from a chip sheet (format atlas.chipsheet/1)."""
    rates = [m["value"] for m in chipsheet.get("measurements", [])
             if m.get("section") == "copy" and m.get("status") == "ok" and m.get("value")
             and (m.get("params") or {}).get("bytes", 0) > CACHE_BYTES]
    if not rates:
        raise ReportError("the chip sheet has no copy rate above 32 MiB")
    return statistics.median(rates)


def idle_w(chipsheet: dict) -> float | None:
    idle = ((chipsheet.get("power") or {}).get("idle") or {}).get("copy") or {}
    return idle.get("idle_w")


def predict(chipsheet: dict) -> dict:
    bw = bandwidth_gbs(chipsheet)
    rows = []
    for width in WIDTHS:
        for batch in BATCHES:
            gb = step_gb(width, batch)
            tok_s = batch * bw / gb
            rows.append({
                "width": width, "batch": batch, "bits_per_weight": round(bits_per_weight(width), 3),
                "step_gb": round(gb, 2), "ceiling_tok_s": round(tok_s, 1),
                "j_per_token": [round(LOADED_W[0] / tok_s, 1), round(LOADED_W[1] / tok_s, 1)],
            })
    return {"bandwidth_gbs": round(bw, 1), "idle_w": idle_w(chipsheet), "loaded_w": list(LOADED_W),
            "chipsheet": chipsheet.get("date"), "rows": rows}


def read_bench(path: Path) -> dict[int, dict]:
    """Per batch size, the medians of a lab/bench.py session's ok rows (a session folder or its rows.jsonl)."""
    rows_path = path / "rows.jsonl" if path.is_dir() else path
    if not rows_path.is_file():
        raise ReportError(f"{rows_path} not found")
    by_batch: dict[int, list[dict]] = {}
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("outcome") != "ok" or row.get("decode_tok_s") is None:
            continue
        by_batch.setdefault(int(row["batch_size"]), []).append(row)

    def med(values):
        values = [v for v in values if v is not None]
        return round(statistics.median(values), 3) if values else None

    out = {}
    for batch, rows in sorted(by_batch.items()):
        out[batch] = {
            "spawns": len(rows),
            "decode_tok_s": med(r["decode_tok_s"] for r in rows),
            "decode_tok_s_total": med(r.get("decode_tok_s_total") for r in rows),
            "watts_avg": med((r.get("watts") or {}).get("avg_w") for r in rows),
            "joules_per_token": med(r.get("joules_per_token") for r in rows),
            "checkpoint": ((rows[0].get("conditions") or {}).get("checkpoint")),
        }
    return out


def read_eval(path: Path) -> float:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("score"), (int, float)):
        raise ReportError(f"{path} has no numeric score")
    return float(data["score"])


def read_correctness(path: Path) -> dict:
    text = path.read_text(encoding="utf-8").strip()
    data = json.loads(text.splitlines()[-1]) if text.startswith("{") and "\n{" in text else json.loads(text)
    return {k: data.get(k) for k in ("prompts", "match_rate", "normalised_text_matches", "condition_drift")}


def choose(evals: dict[int, float], margin: float | None) -> dict:
    """The lowest width whose eval drop from the 4-bit teacher is within the margin, or none, with the reason."""
    if margin is None:
        return {"width": None, "reason": "no --margin given: the agreed margin is the owner's call"}
    if 4 not in evals:
        return {"width": None, "reason": "no 4-bit eval score to measure the drop against"}
    within = [w for w in sorted(evals) if w < 4 and evals[4] - evals[w] <= margin]
    if not within:
        return {"width": None, "reason": f"every lower width drops more than {margin} from the 4-bit score"}
    return {"width": within[0], "reason": f"lowest width within {margin} of the 4-bit score"}


def report(chipsheet: dict, bench: dict[int, Path], evals: dict[int, Path], correctness: dict[int, Path],
           margin: float | None, day: str) -> dict:
    pred = predict(chipsheet)
    ceiling = {(r["width"], r["batch"]): r for r in pred["rows"]}
    scores = {w: read_eval(p) for w, p in evals.items()}
    widths = []
    for width in sorted(set(bench) | set(scores) | set(correctness), reverse=True):
        measured = read_bench(bench[width]) if width in bench else {}
        per_batch = []
        for batch in BATCHES:
            c = ceiling.get((width, batch))
            m = measured.get(batch)
            per_batch.append({
                "batch": batch,
                "ceiling_tok_s": c["ceiling_tok_s"] if c else None,
                "step_gb": c["step_gb"] if c else None,
                **(m or {"spawns": 0, "decode_tok_s": None, "decode_tok_s_total": None, "watts_avg": None,
                         "joules_per_token": None, "checkpoint": None}),
                "fraction_of_ceiling": (round((m["decode_tok_s_total"] or m["decode_tok_s"]) / c["ceiling_tok_s"], 3)
                                        if m and c and (m["decode_tok_s_total"] or m["decode_tok_s"]) else None),
            })
        entry = {"width": width, "batches": per_batch, "eval_score": scores.get(width),
                 "eval_drop": round(scores[4] - scores[width], 4) if 4 in scores and width in scores else None}
        if width in correctness:
            entry["correctness_vs_int4"] = read_correctness(correctness[width])
        widths.append(entry)
    return {"format": "atlas.lowbit/1", "date": day, "chip": "gfx1151", "prediction": pred, "widths": widths,
            "margin": margin, "chosen": choose(scores, margin)}


def _pairs(values: list[str], flag: str) -> dict[int, Path]:
    out = {}
    for v in values:
        width, sep, path = v.partition("=")
        if not sep or not width.strip().isdigit() or int(width) not in WIDTHS:
            raise ReportError(f"{flag} takes WIDTH=PATH with WIDTH in {WIDTHS}, got {v!r}")
        out[int(width)] = Path(path)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lowbit_report", description=__doc__.splitlines()[0])
    ap.add_argument("--chipsheet", type=Path, default=CHIPSHEET)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("predict", help="the ceilings per width and batch size")
    rep = sub.add_parser("report", help="write the measured report")
    rep.add_argument("--bench", action="append", default=[], help="WIDTH=lab/bench.py session folder")
    rep.add_argument("--eval", action="append", default=[], help="WIDTH=contexts.agent.eval --json file")
    rep.add_argument("--correctness", action="append", default=[], help="WIDTH=lab correctness compare JSON")
    rep.add_argument("--margin", type=float, default=None, help="largest eval drop allowed, in score points")
    rep.add_argument("--date", default=_date.today().isoformat())
    rep.add_argument("--out", type=Path, default=None, help="default: data/lowbit-gfx1151-<date>.json")
    args = ap.parse_args(argv)
    try:
        chipsheet = json.loads(args.chipsheet.read_text(encoding="utf-8"))
        if args.cmd == "predict":
            print(json.dumps(predict(chipsheet), indent=1))
            return 0
        result = report(chipsheet, _pairs(args.bench, "--bench"), _pairs(args.eval, "--eval"),
                        _pairs(args.correctness, "--correctness"), args.margin, args.date)
        out = args.out or OUT_DIR / f"lowbit-gfx1151-{args.date}.json"
        out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        print(json.dumps({"ok": True, "out": str(out), "chosen": result["chosen"]}))
        return 0
    except (ReportError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
