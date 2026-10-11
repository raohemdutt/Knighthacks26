"""Joules above idle for each push and pull window of sync_energy.py, from pod C's sampler events.

    sync_windows.py OUT_DIR SAMPLES.jsonl
Windows are sampler sequence ranges: a transfer owns the samples with last_seq(start) < seq <= last_seq(end).
Each sample stands for one period (1000 ms), so a window's energy is the sum of its watts times the measured period.
"""
import json
import statistics
import sys
from pathlib import Path

out = Path(sys.argv[1])
marks = json.loads((out / "marks.json").read_text(encoding="utf-8"))
report = json.loads((out / "report.json").read_text(encoding="utf-8"))
events = [json.loads(l) for l in Path(sys.argv[2]).read_text(encoding="utf-8").splitlines() if l.strip()]
by_seq = {e["seq"]: e for e in events}
period_s = 1.0


def window(a: int, b: int) -> list[dict]:
    return [by_seq[s] for s in range(a + 1, b + 1) if s in by_seq]


def stats(ws: list[dict]) -> dict:
    w = [e["socket_w"] for e in ws if e.get("socket_w") is not None]
    act = [e.get("gfx_activity_pct") or 0.0 for e in ws]
    return {"samples": len(w), "mean_w": round(statistics.fmean(w), 3) if w else None,
            "min_w": min(w) if w else None, "max_w": max(w) if w else None,
            "gfx_activity_max_pct": max(act) if act else None,
            "first_ts": ws[0]["ts"] if ws else None, "last_ts": ws[-1]["ts"] if ws else None}


m = {}
for row in marks:
    m.setdefault(row["what"], []).append(row)
idle = stats(window(m["idle-start"][0]["last_seq"], m["idle-end"][0]["last_seq"]))
idle2 = stats(window(m["after-end"][0]["last_seq"], m["idle2-end"][0]["last_seq"]))
busy = (idle["gfx_activity_max_pct"] or 0) > 5
idle_w = idle2["mean_w"] if busy else idle["mean_w"]
runs = report["sizes"][0]["runs"]
result = {"idle_before": idle, "idle_after": idle2, "idle_w_used": idle_w, "idle_source": ("after: the window before had another client on the teacher (GFX activity up to " + str(idle["gfx_activity_max_pct"]) + " percent)") if busy else "before", "windows": []}
for direction, start, end in (("push", "push-start-payload", "push-end-payload"), ("pull", "pull-start", "pull-end")):
    for i, (s, e) in enumerate(zip(m.get(start, []), m.get(end, [])), 1):
        ws = window(s["last_seq"], e["last_seq"])
        st = stats(ws)
        gross = sum(x["socket_w"] for x in ws) * period_s
        above = sum(x["socket_w"] - idle_w for x in ws) * period_s
        run = next(r for r in runs if r["direction"] == direction and r["repeat"] == i)
        gb = run["bytes"] / 1e9
        result["windows"].append({
            "direction": direction, "repeat": i, "bytes": run["bytes"], "ok": run["ok"], "seconds": run["seconds"],
            "wall_seconds_local": e["seconds"], "mb_s": run["mb_s"], "gb_s": round(run["mb_s"] / 1000, 5) if run["mb_s"] else None,
            "window": st, "joules_gross": round(gross, 1), "joules_above_idle": round(above, 1),
            "joules_per_gb_above_idle": round(above / gb, 2) if gb else None,
            "joules_per_gb_gross": round(gross / gb, 2) if gb else None})
(out / "windows.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
print(json.dumps(result, indent=1))
