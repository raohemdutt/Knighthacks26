"""R57: J/token from the nvidia-smi power.draw samples over each harness run's decode window, and the summary table.
Usage: python analyze.py fixtures/lab/bench/a100-2026-10-10   (writes summary_a100.json and table.md there)"""
import json, statistics, sys, datetime as dt
from pathlib import Path

d = Path(sys.argv[1])

def ts(s):
    s = s.strip()
    if "/" in s:   # nvidia-smi: 2026/10/11 00:56:01.234, the VM clock (UTC)
        return dt.datetime.strptime(s, "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp()
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()

pts = []
for line in (d / "nvidia_smi_power.csv").read_text().splitlines()[1:]:
    parts = [p.strip() for p in line.split(",")]
    try:
        pts.append((ts(parts[0]), float(parts[1].split()[0])))
    except (ValueError, IndexError):
        continue
pts.sort()

def at(x):
    if pts and pts[0][0] - 2.0 <= x < pts[0][0]:   # within a sample period before the first read: hold it
        return pts[0][1]
    if pts and pts[-1][0] < x <= pts[-1][0] + 2.0:
        return pts[-1][1]
    for (t1, w1), (t2, w2) in zip(pts, pts[1:]):
        if t1 <= x <= t2:
            return w1 if t2 == t1 else w1 + (w2 - w1) * (x - t1) / (t2 - t1)
    return None

def window(a, b):
    if a is None or b is None or b <= a or at(a) is None or at(b) is None:
        return None
    seq = [(a, at(a))] + [(t, w) for t, w in pts if a < t < b] + [(b, at(b))]
    e = sum((t2 - t1) * (w1 + w2) / 2 for (t1, w1), (t2, w2) in zip(seq, seq[1:]))
    return {"energy_j": e, "avg_w": e / (b - a), "peak_w": max(w for _, w in seq), "samples": len(seq) - 2,
            "duration_s": b - a}

iw = dict(l.split() for l in (d / "idle_window.txt").read_text().splitlines() if l.strip())
idle = window(ts(iw["idle_start"]), ts(iw["idle_end"]))
rows_path = next(d.glob("**/rows.jsonl"))
rows = [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]
per_run = []
for r in rows:
    out = {"run_id": r.get("run_id"), "batch": r.get("batch_size"), "outcome": r.get("outcome"),
           "reason": r.get("reason"), "decode_tok_s": r.get("decode_tok_s"),
           "decode_tok_s_total": r.get("decode_tok_s_total"), "ttft_s": r.get("ttft_s"),
           "tokens_generated": r.get("tokens_generated"), "window_s": r.get("window_s")}
    good = [q for q in r.get("requests") or [] if not q.get("error") and q.get("first_token") and q.get("last_token")]
    if r.get("outcome") == "ok" and good:
        a, b = min(ts(q["first_token"]) for q in good), max(ts(q["last_token"]) for q in good)
        w = window(a, b)
        out["power"] = w
        if w and r.get("tokens_generated"):
            out["joules_per_token"] = w["energy_j"] / r["tokens_generated"]
            out["joules_per_token_above_idle"] = (w["energy_j"] - idle["avg_w"] * w["duration_s"]) / r["tokens_generated"]
    per_run.append(out)

def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 4) if xs else None

table = {}
for b in sorted({p["batch"] for p in per_run}):
    runs = [p for p in per_run if p["batch"] == b]
    ok = [p for p in runs if p["outcome"] == "ok"]
    table[b] = {"runs_ok": len(ok), "runs": len(runs), "outcomes": [p["outcome"] for p in runs],
                "reasons": sorted({p["reason"] for p in runs if p["reason"]}),
                "decode_tok_s": med(p["decode_tok_s"] for p in ok), "decode_tok_s_total": med(p["decode_tok_s_total"] for p in ok),
                "ttft_s": med(p["ttft_s"] for p in ok),
                "watts_avg": med((p.get("power") or {}).get("avg_w") for p in ok),
                "watts_peak": max([(p.get("power") or {}).get("peak_w") or 0 for p in ok] or [0]) or None,
                "joules_per_token": med(p.get("joules_per_token") for p in ok),
                "joules_per_token_above_idle": med(p.get("joules_per_token_above_idle") for p in ok),
                "per_run_decode_tok_s": [p["decode_tok_s"] for p in ok],
                "per_run_joules_per_token": [round(p["joules_per_token"], 4) for p in ok if p.get("joules_per_token")]}
summary = {"idle_no_server": idle, "power_source": "nvidia-smi --query-gpu=timestamp,power.draw -l 1 (board power, W)",
           "method": "trapezoid over each run's decode window (first token of the first request to last token of the last), edges interpolated",
           "rows": rows_path.relative_to(d).as_posix(), "per_batch": table, "per_run": per_run}
(d / "summary_a100.json").write_text(json.dumps(summary, indent=1) + "\n", newline="\n")
lines = ["| batch | runs ok | tok/s per stream | tok/s total | TTFT | avg W | peak W | J/token gross | idle W (no server) |",
         "|---|---|---|---|---|---|---|---|---|"]
for b, t in table.items():
    f = lambda v, n=2: "-" if v is None else f"{v:.{n}f}"
    lines.append(f"| {b} | {t['runs_ok']}/{t['runs']} | {f(t['decode_tok_s'],1)} | {f(t['decode_tok_s_total'],1)} | {f(t['ttft_s'],3)} s | "
                 f"{f(t['watts_avg'],0)} | {f(t['watts_peak'],0)} | {f(t['joules_per_token'],3)} | {f(idle['avg_w'],1)} |")
(d / "table.md").write_text("\n".join(lines) + "\n", newline="\n")
print("\n".join(lines))
