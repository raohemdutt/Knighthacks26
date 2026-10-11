"""Backfill the measured-row fields into lab/refine/metrics.jsonl.

    python -m lab.tools.backfill_row_fields [--path FILE] [--dry-run]

The telemetry rulings say every measured row names, in its conditions, the sampler, period_ms, samples and error, and
the hub, account and GPU (and, for a teacher metric, the engine) it was measured on. Rows written before that
ruling reached the people writing them lack some of these. This fills only what is missing, from what the repo
already knows, and changes nothing else: no value, unit, node, source, ts or low_confidence, and no key that is already
there. Node names stay as written (R31 decided R29's pod-c rows and R13's pod-d rows stay). A row that was touched gets
a `backfilled` list in its conditions naming the keys this added. Lines it does not touch are kept byte for byte, so
a rebase over rows other branches appended is a union. Running it twice changes nothing the second time.

What is filled, and from where:
  hub, account  pod-a and pod-c: sjc, C. pod-b and pod-d: sjc, D. a100: colab, colab (the account label is the host).
  gpu           "Radeon 8060S (gfx1151)" on the SJC pods; "NVIDIA A100" on a100; skipped when the row has gpu or chip.
  sampler       a power row on pod-a is read by lab/pod_power.py 1; else the reading path its source names (R29's
                script, the controller timer, the client clock, lab/correctness.py compare).
  period_ms     1000 when the row's sampler is lab/pod_power.py (T6's one-second period) or its power_source says once a
                second (R57's nvidia-smi), else null (not recorded).
  samples       what the row's own conditions say: stats.n for a repeated median or the spawns behind it, window_s over the period for a
                sampled idle window, matches plus divergences for a correctness compare, the ping count of a keep-alive
                run, one for a single launch or start; null where the number was not recorded.
  error         "unknown" (no reference run has measured it).
The engine is never guessed: a teacher row without one is reported, not filled.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lab.machine import metrics

SJC_LAUNCH_TS = "2026-10-10T19:07:00Z"  # both SJC pods were launched around 19:07 UTC; earlier pod-a/pod-b rows are DUB
HUB_ACCOUNT = {
    "pod-a": ("sjc", "C"), "pod-c": ("sjc", "C"), "pod-b": ("sjc", "D"), "pod-d": ("sjc", "D"),
    "a100": ("colab", "colab"),
}
GPU = {"a100": "NVIDIA A100"}
SJC_GPU = "Radeon 8060S (gfx1151)"
POWER_SUFFIXES = ("watts_avg", "watts_peak", "idle_watts", "joules_per_token", "idle_state_share")
REQUIRED = ("sampler", "period_ms", "samples", "error", "hub", "account")


def in_scope(row: dict) -> bool:
    """Rows on pod-c, pod-d or the A100 at any time, and on pod-a or pod-b from the SJC launch on."""
    node = row["node"]
    if node in ("a100", "pod-c", "pod-d"):
        return True
    return node in HUB_ACCOUNT and row["ts"] >= SJC_LAUNCH_TS


def is_power(row: dict) -> bool:
    return row["metric"].rsplit(".", 1)[-1] in POWER_SUFFIXES or "sensor" in row["conditions"]


def missing_fields(row: dict) -> list[str]:
    """The required condition keys this row lacks (gpu may be given as chip; the engine only for a teacher metric)."""
    c = row["conditions"]
    out = [k for k in REQUIRED if k not in c]
    if "gpu" not in c and "chip" not in c:
        out.append("gpu")
    if row["metric"].startswith("teacher") and "engine" not in c:
        out.append("engine")
    return out


def _sampler(row: dict) -> str | None:
    metric, source, c = row["metric"], row["source"], row["conditions"]
    if "r29_measure.py" in source or source.startswith("R29"):
        if is_power(row):
            return "amd-smi metric --power SOCKET_POWER polled by ~/serve/r29_measure.py"
        if "hub proxy" in str(c.get("route")):
            return "one client clock (workspace machine)"
        return "client clock in ~/serve/r29_measure.py"
    if is_power(row) and isinstance(c.get("power_source"), str):
        return c["power_source"]
    if "lab.controller" in source:
        return "lab.controller timer"
    if metric.endswith("output_match_rate"):
        return "lab/correctness.py compare"
    if is_power(row) and row["node"] == "pod-a":
        return "lab/pod_power.py 1"
    if isinstance(c.get("harness"), str):
        return c["harness"]
    return None


def _samples(row: dict, period):
    """The sample count the row's own conditions state, or None."""
    c, metric = row["conditions"], row["metric"]
    if metric.endswith("output_match_rate") and "matches" in c and "divergences" in c:
        return c["matches"] + c["divergences"]
    if "pings" in c:
        return c["pings"]
    if metric.endswith(("pod.start_s", "first_request_s", "engine_init_s")):
        return 1
    if "window_s" in c and period:
        return int(c["window_s"] * 1000 / period)
    if isinstance(c.get("per_spawn"), list) and not is_power(row):
        return len(c["per_spawn"])
    stats = c.get("stats")
    if isinstance(stats, dict) and not is_power(row) and isinstance(stats.get("n"), int):
        return stats["n"]
    return None


def backfill_row(row: dict) -> tuple[dict, list[str]]:
    """(the row with missing condition keys added, the keys added). Out-of-scope rows come back unchanged."""
    if not in_scope(row):
        return row, []
    c = dict(row["conditions"])
    added: list[str] = []

    def put(key, value):
        if key not in c:
            c[key] = value
            added.append(key)

    hub, account = HUB_ACCOUNT[row["node"]]
    put("hub", hub)
    put("account", account)
    if "gpu" not in c and "chip" not in c:
        put("gpu", GPU.get(row["node"], SJC_GPU))
    put("sampler", _sampler(row) or "unknown")
    period = c.get("period_ms")
    if "period_ms" not in c:
        once_a_second = "pod_power.py" in str(c.get("sampler")) or "once a second" in str(c.get("power_source"))
        period = 1000 if once_a_second else None
        put("period_ms", period)
    if "samples" not in c:
        put("samples", _samples({**row, "conditions": c}, period))
    put("error", "unknown")
    if not added:
        return row, []
    c["backfilled"] = sorted(set(c.get("backfilled", [])) | set(added))
    return {**row, "conditions": c}, added


def backfill_file(path: Path, *, dry_run: bool = False) -> dict:
    """Rewrite `path` with the fields filled in, validating before and after. Returns a summary."""
    path = Path(path)
    metrics.load(path)  # refuse a file that is not valid to begin with
    with metrics.FileLock(path):
        lines = path.read_bytes().decode("utf-8").split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        out, changed, groups, no_engine = [], 0, {}, []
        for n, line in enumerate(lines, 1):
            row = json.loads(line)
            new, added = backfill_row(row)
            if added:
                checked = metrics.check_row(new)
                if (checked["value"], checked["low_confidence"], checked["node"], checked["ts"]) != (
                        row["value"], row["low_confidence"], row["node"], row["ts"]):
                    raise metrics.MetricsFormatError(f"line {n}: backfill would change a value")
                changed += 1
                key = f"{new['node']} {new['metric'].split('.')[0]}"
                groups[key] = groups.get(key, 0) + 1
                line = json.dumps(new, ensure_ascii=False, allow_nan=False)
            if in_scope(new) and "engine" in missing_fields(new):
                no_engine.append(n)
            out.append(line)
        if changed and not dry_run:
            path.write_bytes(("\n".join(out) + "\n").encode("utf-8"))
    if changed and not dry_run:
        metrics.load(path)  # the whole file still validates
    return {"ok": True, "rows": len(lines), "changed": changed, "by_group": groups,
            "teacher_rows_without_engine": no_engine, "dry_run": dry_run}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lab.tools.backfill_row_fields",
                                 description="Fill the measured-row condition fields missing from metrics.jsonl.")
    ap.add_argument("--path", default=str(metrics.DEFAULT_PATH))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    print(json.dumps(backfill_file(Path(args.path), dry_run=args.dry_run)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
