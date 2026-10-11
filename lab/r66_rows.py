"""R66 rows: the pod-to-pod tunnel and the cross-pod process group, from fixtures/lab/r66-2026-10-11/.

    python lab/r66_rows.py           print the rows
    python lab/r66_rows.py --write   append the ones not yet in lab/refine/metrics.jsonl

The vLLM pair's bench rows are written by `python -m lab bench attach` itself (prefix pair_pp122b); their watts_avg,
watts_peak and joules_per_token are pod C's meter only (the harness reads one sampler). pair_rows_both() adds the
pair as a whole from the same recorded windows: pod D's T6 sampler over each bench window (R66PWIN lines in
fixtures/lab/r66-2026-10-11/pair-122b.txt) summed with pod C's from the bench rows. These rows carry node
pod-a+pod-b like R65's split rows, so the single-pod figures stay single-pod.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
METRICS = REPO / "lab" / "refine" / "metrics.jsonl"
FIX = "fixtures/lab/r66-2026-10-11"
BASE = {
    "chip": "Strix Halo gfx1151 x2",
    "pods": "pod C (pod-a, 10.42.6.28) and pod D (pod-b, 10.42.1.131), SJC hub, no direct pod-to-pod network",
    "tunnel": "chisel 1.12.0, server on pod D behind Jupyter's /proxy/29999/, client on pod C via proxy-public:80, "
              "128 forward and 128 reverse remotes (P+200 reaches the peer's P, P 29600-29727)",
}
RELAY = "quick-ACK relay on pod D between Jupyter's proxy and chisel (lab/net/ackrelay.py)"
SHIM = "R66b podshim sha256 1f975c69, pods0 10.99.0.1 (C) / 10.99.0.2 (D)"
TORCH = "torch 2.12.0+rocm7.14.0 (~/venv-vllm-new), gloo, rank 0 pod C, rank 1 pod D, master 10.99.0.1:29600"


def row(metric, value, unit, conditions, source, ts, low=True):
    return {"metric": metric, "value": value, "unit": unit, "node": "pod-a+pod-b",
            "conditions": {**BASE, **conditions, "low_confidence_reason": "one run per setting"},
            "source": source, "ts": ts, "low_confidence": low}


def rows() -> list[dict]:
    t = f"{FIX}/tunnel-chisel-probe.txt"
    lat = f"{FIX}/latency-two-writes.txt"
    g0, g1 = f"{FIX}/gloo-gate-before-relay.txt", f"{FIX}/gloo-gate-after-relay.txt"
    return [
        row("link.pair.tunnel_send_mb_s", 105.2, "MB/s", {"direction": "C to D", "measure": "256 MB send, mean of 101.5 and 108.9",
            "relay": "none"}, f"R66 {t}", "2026-10-11T09:46:00Z"),
        row("link.pair.tunnel_send_mb_s", 106.0, "MB/s", {"direction": "D to C", "measure": "256 MB send", "relay": "none"},
            f"R66 {t}", "2026-10-11T09:46:30Z"),
        row("link.pair.tunnel_rtt_ms", 0.34, "ms", {"direction": "C to D", "measure": "median of 200 one-byte request and answer, "
            "one write each", "relay": "none"}, f"R66 {t}", "2026-10-11T09:46:00Z"),
        row("link.pair.two_write_rtt_ms", 1.67, "ms", {"direction": "C to D", "measure": "median of 200 round trips written as "
            "two writes (16 + 48 bytes), client TCP_NODELAY; p95 43.24 ms", "relay": "none"}, f"R66 {lat}", "2026-10-11T10:18:52Z"),
        row("link.pair.two_write_rtt_ms", 0.36, "ms", {"direction": "C to D", "measure": "median of 200 round trips written as "
            "two writes (16 + 48 bytes), client TCP_NODELAY; p95 0.57 ms", "relay": RELAY}, f"R66 {lat}", "2026-10-11T10:19:10Z"),
        row("dist.pair.gloo_small_allreduce_ms", 44.7, "ms", {"shim": SHIM, "torch": TORCH, "relay": "none",
            "measure": "median of 100 all-reduces of 4 floats; min 1.54, p95 92.5"}, f"R66 {g0}", "2026-10-11T10:00:00Z"),
        row("dist.pair.gloo_small_allreduce_ms", 1.38, "ms", {"shim": SHIM, "torch": TORCH, "relay": RELAY,
            "measure": "median of 100 all-reduces of 4 floats; min 1.07, p95 45.8 (rank 0)"}, f"R66 {g1}", "2026-10-11T10:21:30Z"),
        row("dist.pair.gloo_allreduce_mb_s", 47.7, "MB/s", {"shim": SHIM, "torch": TORCH, "relay": RELAY,
            "measure": "one 64 MB float32 all-reduce, 1.406 s"}, f"R66 {g1}", "2026-10-11T10:21:30Z"),
        row("dist.pair.gloo_init_s", 0.024, "s", {"shim": SHIM, "torch": TORCH, "relay": RELAY,
            "measure": "init_process_group on rank 0 (tcp:// rendezvous)"}, f"R66 {g1}", "2026-10-11T10:21:30Z"),
        row("dist.pair.p2p_roundtrip_16kb_ms", 1.1, "ms", {"shim": SHIM, "torch": TORCH, "relay": RELAY,
            "measure": "mean of 20 send and return of 16 KB (a pipeline stage's handoff size)"}, f"R66 {g1}", "2026-10-11T10:21:30Z"),
    ]


BENCH = [REPO / FIX / "bench-pp122b-1_4-rows.jsonl", REPO / FIX / "bench-pp122b-16-rows.jsonl"]
PWIN = REPO / FIX / "pair-122b.txt"
BOTH_NOTE = ("both pods summed: pod C's T6 sampler (the harness's own watts) plus pod D's T6 sampler over the same "
             "window (pod D's clock 4 s ahead of the workstation's request times); the pair_pp122b.watts_avg, "
             "watts_peak and joules_per_token rows of the same session are pod C's meter only")


def _pod_d_windows() -> dict[tuple[str, str], dict]:
    out = {}
    for line in PWIN.read_text(encoding="utf-8").splitlines():
        if line.startswith("R66PWIN "):
            w = json.loads(line[len("R66PWIN "):])
            if w["node"] == "gpu-b":
                out[(w["from"], w["to"])] = w
    return out


def pair_rows_both() -> list[dict]:
    """The pair's watts and joules per token for both pods, per bench session, computed from the recorded windows."""
    d_windows = _pod_d_windows()
    out = []
    for path in BENCH:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            measured = [q for q in r["requests"] if q["role"] == "measure"]
            window = (min(q["sent"] for q in measured), max(q["last_token"] for q in measured))
            d = d_windows[window]
            c = r["watts"]
            energy = c["energy_j"] + d["avg_w"] * r["window_s"]
            cond = {"batch_size": r["batch_size"], "session": r["session"], "window_s": r["window_s"],
                    "tokens_generated": r["tokens_generated"], "pod_d_samples": d["n"], "pod_c_samples": c["samples"],
                    "note": BOTH_NOTE}
            src, ts = f"lab/bench.py 1 {r['session']}", r["requests"][-1]["last_token"][:19] + "Z"
            def one(metric, value, unit, extra=None):
                return {"metric": metric, "value": round(value, 4), "unit": unit, "node": "pod-a+pod-b",
                        "conditions": {**cond, **(extra or {}), "low_confidence_reason": "one repeat per setting"},
                        "source": src, "ts": ts, "low_confidence": True}
            out += [one("pair_pp122b.watts_avg_pod_d", d["avg_w"], "W"),
                    one("pair_pp122b.watts_avg_both", c["avg_w"] + d["avg_w"], "W"),
                    one("pair_pp122b.watts_peak_both", c["peak_w"] + d["peak_w"], "W",
                        {"peak_note": "sum of each pod's peak in the window: an upper bound on the pair's peak"}),
                    one("pair_pp122b.joules_per_token_both", energy / r["tokens_generated"], "J/token",
                        {"energy_j_both": round(energy, 1)})]
    return out


def all_rows() -> list[dict]:
    return rows() + pair_rows_both()


def main() -> int:
    new = all_rows()
    if "--write" not in sys.argv:
        for r in new:
            print(json.dumps(r))
        return 0
    have = {(d.get("metric"), d.get("source"), d.get("value")) for d in
            (json.loads(l) for l in METRICS.read_text(encoding="utf-8").splitlines() if l.strip())}
    add = [r for r in new if (r["metric"], r["source"], r["value"]) not in have]
    with METRICS.open("a", encoding="utf-8", newline="\n") as f:
        for r in add:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({"ok": True, "written": len(add), "skipped": len(new) - len(add)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
