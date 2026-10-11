"""Writes the joules-per-answer replay fixtures of this folder: power samples, request windows, a clock offset.

    python fixtures/telemetry/joules/make_windows.py      # rewrites the three .jsonl files here

Synthetic, not measured: plausible shapes for the nodes, scripted so the output is the same on every run.
contexts/telemetry/tests/test_joules.py checks that the committed files are exactly what this writes.

overlap_pod_a.jsonl: pod-a (the teacher's pod, a Strix Halo) sampled once a second on its own clock, which runs
65.2 s ahead of the Jetson's, with whole-watt readings that repeat; one telemetry.clock.offset measured through the
hub proxy; six requests stamped on the Jetson's clock, four of them overlapping in a chain, one alone, and one with
no output tokens inside another; then 200 s of idle samples so every window settles.

conditions.jsonl: one window of each kind that sets low_confidence beyond the unmeasured error, plus a clean one,
each alone on its node: on the Jetson a window across a power-mode change (jetson-modes), one across an overcurrent
counter rise (jetson-oc) and one in steady MAXN_SUPER (jetson-clean); on pod-b, whose window clock is its own, one
with a throttled sample (pod-throttled) and one overlapping a training step (pod-train); on the Jetson one of four
seconds (jetson-short, fewer than 10 samples).

idle_sources.jsonl: pod-a and pod-b each answer twice, the second request starting 4.5 s after the first ended, so
the latest quiet reading before it is on the pod's power tail (13 W, deep idle is 10.3 W); the idle-source tests give
pod-a a calibrated idle and pod-b none.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
T0 = datetime(2026, 10, 10, 20, 0, tzinfo=UTC)  # the Jetson's clock
POD_AHEAD_S = 65.2


def iso(t: datetime) -> str:
    return t.isoformat().replace("+00:00", "Z")


def at(s: float) -> datetime:
    return T0 + timedelta(seconds=s)


def pod_sample(n: int, s: float, watts: float, node: str = "pod-a", throttled: bool = False, under_load=None,
               ahead: float = POD_AHEAD_S) -> dict:
    return {"event_id": f"{node}-p{n:04d}", "type": "telemetry.gpu.power", "version": 1, "ts": iso(at(s + ahead)),
            "node": node, "source": "amd-smi", "period_ms": 1000.0, "socket_w": watts, "gfx_mhz": None,
            "mem_mhz": None, "throttled": throttled, "power_field": "socket_power", "temps_c": {"edge": 40.0 + watts / 10},
            "sampler": "lab/pod_power.py 1", "under_load": watts >= 60 if under_load is None else under_load}


def jetson_sample(n: int, s: float, mw: float, mode: str = "MAXN_SUPER", oc: int = 0) -> dict:
    return {"event_id": f"jetson-p{n:04d}", "type": "telemetry.jetson.power", "version": 1, "ts": iso(at(s)),
            "node": "jetson", "source": "tegrastats", "period_ms": 1000.0,
            "rails": {"VDD_IN": mw, "VDD_CPU_GPU_CV": mw / 3, "VDD_SOC": 1900.0}, "nvpmodel": mode,
            "temps_c": {"cpu": 48.0, "gpu": 47.5}, "oc_events": {"oc1": oc, "oc2": 0, "oc3": 0}}


def request(rid: str, node: str, start: float, end: float, tokens_out: int | None, clock: str | None,
            target: str = "teacher") -> dict:
    return {"event_id": f"req-{rid}", "type": "telemetry.request", "version": 1, "ts": iso(at(end)), "node": node,
            "request_id": rid, "started": iso(at(start)), "ended": iso(at(end)), "tokens_in": 500,
            "tokens_out": tokens_out, "target": target, "clock": clock}


def overlap_pod_a() -> list[dict]:
    events = []
    # pod-a: idle 27 W, then busy 100 to 140 W in whole watts (each value held two seconds: repeats kept), idle again
    for n in range(0, 260):
        s = n * 1.0 - 0.4  # the pod reads at .6 past each Jetson second, on its own clock 65.2 s ahead
        busy = 10 <= n <= 40
        watts = float(100 + 10 * ((n // 2) % 5)) if busy else 27.0
        events.append(pod_sample(n, s, watts))
    events.append({"event_id": "offset-pod-a-1", "type": "telemetry.clock.offset", "version": 1, "ts": iso(at(5.2)),
                   "node": "pod-a", "reference": "jetson", "via": "hub proxy /samples",
                   "t1": iso(at(5.0)), "t2": iso(at(5.0 + 0.09 + POD_AHEAD_S)), "t3": iso(at(5.0 + 0.091 + POD_AHEAD_S)),
                   "t4": iso(at(5.2)), "offset_s": POD_AHEAD_S - 0.0095, "round_trip_s": 0.199, "error_s": 0.0995,
                   "exchanges": 10})
    for rid, start, end, tokens in (("a1", 11.3, 18.7, 120), ("a2", 15.1, 24.0, 40), ("a3", 22.5, 30.2, 300),
                                    ("a4", 29.0, 33.4, 80), ("a5", 36.2, 39.9, 64), ("a6", 16.0, 17.5, 0)):
        events.append(request(rid, "pod-a", start, end, tokens, "jetson"))
    return sorted(events, key=lambda e: e["ts"])


def conditions() -> list[dict]:
    events = []
    for n in range(0, 400):  # the Jetson, 1 Hz: 6.4 W idle, 12 W while it answers
        s = float(n)
        busy = any(a <= s <= b for a, b in ((21, 34), (61, 74), (101, 114), (140.2, 143.9)))
        mode = "15W" if 66 <= n <= 70 else "MAXN_SUPER"  # the power mode changes inside jetson-modes
        oc = 0 if n < 108 else 1  # an overcurrent event inside jetson-oc
        events.append(jetson_sample(n, s, 12000.0 if busy else 6400.0, mode=mode, oc=oc))
    for n in range(0, 400):  # pod-b, 1 Hz on its own clock (the window clock is the pod's own: no offset)
        s = float(n)
        busy = any(a <= s <= b for a, b in ((21, 34), (61, 74)))
        events.append(pod_sample(n, s, 110.0 if busy else 28.0, node="pod-b", throttled=(n == 25), ahead=0.0))
    events.append({"event_id": "train-pod-b-1", "type": "telemetry.train.step", "version": 1, "ts": iso(at(70.0)),
                   "node": "pod-b", "run_id": "run-1", "step": 1, "loss": 1.5, "step_ms": 4000.0, "socket_w": 110.0})
    for rid, node, start, end in (("jetson-clean", "jetson", 21.0, 34.0), ("jetson-modes", "jetson", 61.0, 74.0),
                                  ("jetson-oc", "jetson", 101.0, 114.0), ("jetson-short", "jetson", 140.2, 143.9),
                                  ("pod-throttled", "pod-b", 21.0, 34.0), ("pod-train", "pod-b", 61.0, 74.0)):
        events.append(request(rid, node, start, end, 50, None, target="student" if node == "jetson" else "teacher"))
    return sorted(events, key=lambda e: e["ts"])


def idle_sources() -> list[dict]:
    """Two nodes that each answer twice, the second request starting 4.5 s after the first one ended, while the pod
    is still on its power tail (57, 28, 17, 13 W after the 110 W): the latest quiet reading before the second
    request is the tail's 13 W, not the 10.3 W deep idle. pod-a is the node the test gives a calibrated idle (the
    calibrated path); pod-b has none (the fallback)."""
    events = []
    for node in ("pod-a", "pod-b"):
        for n in range(0, 300):  # 1 Hz on the pod's own clock (no offset)
            tail = {33: 57.0, 34: 28.0, 35: 17.0, 36: 13.0}
            if 20 <= n <= 32 or 37 <= n <= 49:
                watts = 110.0
            elif n in tail:
                watts = tail[n]
            else:
                watts = 10.0 + (n % 3) * 0.3
            events.append(pod_sample(n, float(n), watts, node=node, ahead=0.0))
        events.append(request(f"{node}-first", node, 20.0, 32.0, 60, None))
        events.append(request(f"{node}-second", node, 36.5, 48.5, 60, None))
    return sorted(events, key=lambda e: e["ts"])


def write(path: Path, events: list[dict]) -> None:
    path.write_text("".join(json.dumps(e, separators=(",", ":")) + "\n" for e in events), encoding="utf-8")


if __name__ == "__main__":
    write(HERE / "overlap_pod_a.jsonl", overlap_pod_a())
    write(HERE / "conditions.jsonl", conditions())
    write(HERE / "idle_sources.jsonl", idle_sources())
