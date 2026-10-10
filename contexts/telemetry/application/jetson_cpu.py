"""What the Jetson power sampler costs in CPU: the box measured for a stretch with the sampler off, then on.

    report = await measure_sampler_cpu(make_sampler, snapshot, seconds=600, log_dir=tmp)

The budget is at most 1 percent of one core at the default interval. The headline number (`value`, metric
`telemetry.sampler_cpu_pct`) is the whole box's busy CPU with the sampler on minus with it off, over equal
stretches, as percent of one core; `own_pct` is the sampler's own CPU time (this process, its finished children
such as nvpmodel, and the running tegrastats), which does not move with whatever else the box is doing. Both
must be within the budget. While on, every sample is published onto a bus that writes a throwaway log, so the
cost of writing each sample is part of what is measured.
"""

import asyncio
import contextlib
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from bus import EventBus
from contexts.telemetry.application.jetson_sampler import JetsonPowerSampler
from contexts.telemetry.application.sampling import SamplerRunner
from contexts.telemetry.domain.nodes import JETSON

METRIC = "telemetry.sampler_cpu_pct"
LIMIT_PCT = 1.0  # percent of one core


@dataclass(frozen=True)
class CpuSnapshot:
    wall_s: float  # a monotonic clock, seconds
    busy_s: float  # CPU seconds all cores together spent busy since boot
    own_s: float  # CPU seconds of this process, its finished children and the running tegrastats


def cpu_report(off: tuple[CpuSnapshot, CpuSnapshot], on: tuple[CpuSnapshot, CpuSnapshot], interval_ms: int,
               samples: int, limit_pct: float = LIMIT_PCT) -> dict:
    """The sampler's CPU cost from snapshots at the start and end of the off and on stretches."""
    off_s, on_s = off[1].wall_s - off[0].wall_s, on[1].wall_s - on[0].wall_s
    if off_s <= 0 or on_s <= 0:
        raise ValueError("both stretches must last longer than zero seconds")
    system_off = (off[1].busy_s - off[0].busy_s) / off_s * 100
    system_on = (on[1].busy_s - on[0].busy_s) / on_s * 100
    value = system_on - system_off
    own = ((on[1].own_s - on[0].own_s) / on_s - (off[1].own_s - off[0].own_s) / off_s) * 100
    within = value <= limit_pct and own <= limit_pct
    worst = max(value, own)
    return {
        "metric": METRIC, "node": JETSON, "value": round(value, 3), "unit": "percent of one core",
        "method": "busy CPU of the whole box (/proc/stat), sampler on minus sampler off",
        "own_pct": round(own, 3), "system_off_pct": round(system_off, 3), "system_on_pct": round(system_on, 3),
        "off_s": round(off_s, 1), "on_s": round(on_s, 1), "interval_ms": interval_ms, "samples": samples,
        "limit_pct": limit_pct, "within_limit": within,
        # if not: the cost grows with the sample rate, so a period this long should fit the budget
        "suggested_interval_ms": None if within else math.ceil(interval_ms * worst / limit_pct),
    }


async def measure_sampler_cpu(make_sampler: Callable[[], JetsonPowerSampler],
                              snapshot: Callable[[int | None], CpuSnapshot], seconds: float, log_dir: str | Path,
                              sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> dict:
    """Measure the box for `seconds` with no sampler, then for `seconds` with one sampling and publishing.

    `snapshot(pid)` reads the clocks; `pid` is the running tegrastats, whose CPU time it adds to `own_s`."""
    off_start = snapshot(None)
    await sleep(seconds)
    off_end = snapshot(None)
    sampler = make_sampler()
    runner = SamplerRunner(sampler, EventBus(log_dir).publish, nodes=[JETSON])
    on_start = snapshot(None)  # before tegrastats starts: all of its CPU time falls inside the stretch
    task = asyncio.create_task(runner.run())
    try:
        await sleep(seconds)
        on_end = snapshot(sampler.tegrastats.pid)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    if runner.published == 0:
        raise RuntimeError(f"the sampler published nothing while on, so its cost was not measured: {runner.status()}")
    report = cpu_report((off_start, off_end), (on_start, on_end), sampler.period_ms or sampler.interval_ms,
                        runner.published)
    report["sampler"] = runner.status()
    return report
