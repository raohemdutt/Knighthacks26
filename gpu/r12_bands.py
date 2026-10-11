"""R12: every spawn's band for one arm, pooled across the per-round R11 sessions, and the change against stock.

    python gpu/r12_bands.py --arm stock --out gpu/results/bench/stock/bands.json ~/r12/bench/stock/*/rows.jsonl
    python gpu/r12_bands.py --arm stages --stock gpu/results/bench/stock/bands.json \\
        --out gpu/results/bench/stages/bands.json ~/r12/bench/stages/*/rows.jsonl

R12 runs one R11 spawn per arm per round (five rounds, arms interleaved), so each lab/bench.py session holds one
spawn and cannot band on its own (R11 bands from 4 spawns up). This pools an arm's rows.jsonl files, keys each spawn
by session and spawn index, and hands them to lab/bench.py's own summarize_runs: the same largest-gap split of each
spawn's probe decode rate, the same thresholds, the same per-band statistics. With --stock it adds, per batch size,
the change in the median of each metric against stock's, overall and within the fastest band of each arm; a slower
arm gets a positive joules-per-token change and is reported the same way.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if not __package__:  # run by path: the repo root, one folder up, holds lab/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.bench import METRIC_UNITS, summarize_runs  # noqa: E402

SCHEMA = "gpu.r12.bands/1"


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for p in paths:
        rows += [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]
    return rows


def _spawn_key(row: dict[str, Any]) -> str:
    return f"{row['session']}:{(row.get('spawn') or {}).get('index', 0)}"


def _pct(new: float | None, old: float | None) -> float | None:
    return None if new is None or not old else round((new - old) / old * 100.0, 6)


def bands(arm: str, paths: list[Path], *, stock: dict[str, Any] | None = None) -> dict[str, Any]:
    rows = load_rows(paths)
    spawns: dict[str, dict[str, Any]] = {}
    for row in rows:
        info = spawns.setdefault(_spawn_key(row), {"probe": None, "runs": []})
        info["runs"].append(row["run_id"])
        probe = row.get("probe") or {}
        if info["probe"] is None and probe.get("decode_tok_s") is not None:
            info["probe"] = probe
    summary = summarize_runs(rows, spawns)
    band_of = {s: c["band"] for c in summary["band_rule"]["clusters"] for s in c["spawns"]}
    report: dict[str, Any] = {
        "schema": SCHEMA, "arm": arm, "sources": [str(p) for p in paths], "spawns": len(spawns),
        "spawn_bands": [{"spawn": k, "session": k.rsplit(":", 1)[0],
                         "probe_decode_tok_s": (v["probe"] or {}).get("decode_tok_s"), "band": band_of.get(k)}
                        for k, v in spawns.items()],
        **summary}
    if stock is not None:
        report["vs_stock"] = {}
        for key, entry in summary["configurations"].items():
            base = stock["configurations"].get(key)
            if base is None:
                continue
            per_metric = {}
            for m in METRIC_UNITS:
                fast_new = entry["bands"].get("b1", {}).get("stats", {}).get(m, {}).get("median")
                fast_old = base["bands"].get("b1", {}).get("stats", {}).get(m, {}).get("median")
                per_metric[m] = {"stock_median": base["stats"][m]["median"], "arm_median": entry["stats"][m]["median"],
                                 "overall_change_pct": _pct(entry["stats"][m]["median"], base["stats"][m]["median"]),
                                 "fast_band_change_pct": _pct(fast_new, fast_old)}
            report["vs_stock"][key] = per_metric
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--stock", default=None, help="stock's bands.json, for the change against it")
    ap.add_argument("--out", required=True)
    ap.add_argument("rows", nargs="+")
    args = ap.parse_args(argv)
    stock = json.loads(Path(args.stock).read_text()) if args.stock else None
    rep = bands(args.arm, [Path(p) for p in args.rows], stock=stock)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({"arm": rep["arm"], "spawns": rep["spawns"],
                      "bands": {c["band"]: len(c["spawns"]) for c in rep["band_rule"]["clusters"]}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
