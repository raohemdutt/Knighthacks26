"""T10: idle watts per observed clock state on the Strix Halo (pod C), computed from the recorded readings.

Inputs (fixtures/telemetry/pod_power/):
  strixhalo_2026-10-10_idle_baseline_sampler.jsonl         the T3 sampler's samples (1 Hz) from its first sample to the end
                                                           of the experiment: ts, socket_w, gfx_mhz, mem_mhz, gfx_activity_pct
                                                           (the clocks are the sampler's rocm-smi --showclocks fill)
  strixhalo_2026-10-10_idle_baseline_clock_readings.jsonl  the experiment's own reader, once a second, `rocm-smi
                                                           --showclocks` plus `amd-smi metric --power` plus the sysfs DPM
                                                           tables, beside the sampler
  strixhalo_2026-10-10_idle_baseline_requests.jsonl        the five single requests the experiment sent

The clock state of a sample comes from the sclk the chip reported. rocm-smi's "clock level" label is not usable as the
state: it reads level 1 both at 620 MHz and at 2900 MHz, so the state is the frequency band.

  deep_idle   sclk below 700 MHz (the 600 to 650 MHz level; rocm-smi prints no mclk line there)
  transition  sclk from 700 to 2599 MHz (the clock ramping up or down)
  busy        sclk 2600 MHz or more (the 2900 MHz level, mclk 1000 MHz)

Run `python idle_baseline.py` to print every number the metric rows in lab/refine/metrics.jsonl carry.
"""

from __future__ import annotations

import datetime as dt
import json
import statistics
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SAMPLER_FILE = HERE / "strixhalo_2026-10-10_idle_baseline_sampler.jsonl"
READINGS_FILE = HERE / "strixhalo_2026-10-10_idle_baseline_clock_readings.jsonl"
REQUESTS_FILE = HERE / "strixhalo_2026-10-10_idle_baseline_requests.jsonl"

DEEP_IDLE_BELOW_MHZ = 700.0
BUSY_FROM_MHZ = 2600.0
BUSY_WATTS = 60.0  # a sample at or above this is a request in flight
SETTLED_WATTS = 13.0  # idle plus 3 W; the power is settled when it stays at or under this
SETTLED_RUN = 5  # for this many samples in a row


def _ts(text: str) -> dt.datetime:
    return dt.datetime.strptime(text, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=dt.timezone.utc)


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def state_of(sclk_mhz: float | None) -> str | None:
    if sclk_mhz is None:
        return None
    if sclk_mhz < DEEP_IDLE_BELOW_MHZ:
        return "deep_idle"
    return "busy" if sclk_mhz >= BUSY_FROM_MHZ else "transition"


def by_state(samples: list[dict], *, key: str = "gfx_mhz") -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {"deep_idle": [], "transition": [], "busy": []}
    for s in samples:
        st = state_of(s.get(key))
        if st is not None and s.get("socket_w") is not None:
            out[st].append(s)
    return out


def watts_summary(samples: list[dict]) -> dict:
    w = [s["socket_w"] for s in samples]
    return {
        "n": len(w),
        "mean": round(statistics.fmean(w), 2),
        "stdev": round(statistics.pstdev(w), 2),
        "min": min(w),
        "max": max(w),
    }


def share(samples: list[dict]) -> dict[str, float]:
    groups = by_state(samples)
    total = sum(len(v) for v in groups.values())
    return {k: round(100.0 * len(v) / total, 2) for k, v in groups.items()}


def pure_idle_window(samples: list[dict], *, longest_of: str = "deep_idle") -> list[dict]:
    """The longest unbroken run of deep-idle samples (no gap above 1.5 periods): the pure-idle stretch."""
    best: list[dict] = []
    run: list[dict] = []
    prev = None
    for s in samples:
        t = _ts(s["ts"])
        ok = state_of(s.get("gfx_mhz")) == longest_of and s["socket_w"] is not None
        if ok and (prev is None or (t - prev).total_seconds() <= 1.5):
            run.append(s)
        elif ok:
            run = [s]
        else:
            run = []
        prev = t if ok else None
        if len(run) > len(best):
            best = list(run)
    return best


def settles(samples: list[dict], *, idle_w: float) -> list[dict]:
    """Every request end the sampler saw that was followed by a full settle (no new request before the power was
    back at idle for SETTLED_RUN samples). t0 is the last sample at or above BUSY_WATTS, so the request ended within
    the second after it; settle_s is the time from t0 to the first sample of the settled run. excess_j is the energy
    above idle over the samples between them, one period each."""
    out = []
    w = [s["socket_w"] for s in samples]
    for i in range(1, len(samples) - SETTLED_RUN - 1):
        if w[i - 1] is None or w[i] is None or not (w[i - 1] >= BUSY_WATTS > w[i]):
            continue
        j = i
        while j < len(samples) - SETTLED_RUN and w[j] is not None and w[j] > SETTLED_WATTS and w[j] < BUSY_WATTS:
            j += 1
        run = w[j : j + SETTLED_RUN]
        if len(run) < SETTLED_RUN or any(x is None or x > SETTLED_WATTS for x in run):
            continue
        t0 = _ts(samples[i - 1]["ts"])
        out.append(
            {
                "t0": samples[i - 1]["ts"],
                "settle_s": round((_ts(samples[j]["ts"]) - t0).total_seconds(), 1),
                "excess_j": round(sum(max(0.0, x - idle_w) for x in w[i:j]), 1),
                "watts": w[i - 1 : j + 1],
            }
        )
    return out


def report() -> dict:
    samples = load(SAMPLER_FILE)
    groups = by_state(samples)
    idle = pure_idle_window(samples)
    idle_w = watts_summary(idle)["mean"]
    ev = settles(samples, idle_w=idle_w)
    readings = [r for r in load(READINGS_FILE) if "marker" not in r]
    r_groups = by_state(readings, key="rocm_sclk_mhz")
    return {
        "samples": len(samples),
        "first": samples[0]["ts"],
        "last": samples[-1]["ts"],
        "states": {k: watts_summary(v) for k, v in groups.items() if v},
        "share_pct": share(samples),
        "pure_idle": {"first": idle[0]["ts"], "last": idle[-1]["ts"], **watts_summary(idle)},
        "deep_idle_mclk": dict(Counter("1000" if s["mem_mhz"] else "unreported" for s in groups["deep_idle"])),
        "deep_idle_by_mclk": {
            k: watts_summary(v)
            for k, v in (
                ("unreported", [s for s in groups["deep_idle"] if not s["mem_mhz"]]),
                ("1000", [s for s in groups["deep_idle"] if s["mem_mhz"]]),
            )
            if v
        },
        "settle_events": [e["settle_s"] for e in ev],
        "settle": {
            "events": len(ev),
            "median_s": statistics.median(e["settle_s"] for e in ev),
            "min_s": min(e["settle_s"] for e in ev),
            "max_s": max(e["settle_s"] for e in ev),
            "excess_j_median": statistics.median(e["excess_j"] for e in ev),
        },
        "reader": {k: watts_summary(v) for k, v in r_groups.items() if v},
    }


if __name__ == "__main__":
    print(json.dumps(report(), indent=1))
