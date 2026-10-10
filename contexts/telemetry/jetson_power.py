"""The Jetson power sampler's first run on the box, in two read-only steps; neither writes the event log.

    python -m contexts.telemetry.jetson_power probe [--count 20] [--interval-ms 100]
    python -m contexts.telemetry.jetson_power cpu [--minutes 10] [--interval-ms 100]

probe  runs tegrastats for `count` samples and prints each `telemetry.jetson.power` event as a JSON line, then
       one summary line: the period it ran at, the INA3221 settings, the power mode, the measured gap between
       samples, and how many lines had no power rails, did not parse or arrived bunched together.
cpu    measures the box with the sampler off and then on (`minutes` each, 10 by default) and prints one JSON
       line: telemetry.sampler_cpu_pct with its method, both stretches and whether it is within 1 percent of
       one core (application/jetson_cpu.py). Stop the gateway's own sampler first (unset
       ATLAS_TELEMETRY_SOURCE and restart atlas-gateway): this refuses to start while another tegrastats runs.

Both run only on a Jetson and take the same settings as the gateway's sampler (wiring.py:
ATLAS_TELEMETRY_JETSON_INTERVAL_MS, ATLAS_TEGRASTATS, ATLAS_NVPMODEL); `--interval-ms` overrides the first.
"""

import argparse
import asyncio
import contextlib
import json
import os
import statistics
import sys
import tempfile
import time
from collections.abc import Callable, Sequence

from contexts.telemetry.application.jetson_cpu import CpuSnapshot, measure_sampler_cpu
from contexts.telemetry.application.jetson_sampler import JetsonPowerSampler
from contexts.telemetry.infrastructure import host
from contexts.telemetry.infrastructure.jetson_sysfs import JetsonSysfs
from contexts.telemetry.wiring import jetson_power_sampler


async def probe(sampler: JetsonPowerSampler, count: int) -> tuple[list[dict], dict]:
    """`count` samples as JSON-ready dicts, and a summary of the run."""
    events = []
    async with contextlib.aclosing(sampler.samples()) as samples:
        async for event in samples:
            events.append(event)
            if len(events) >= count:
                break
    gaps = [(b.ts - a.ts).total_seconds() * 1000 for a, b in zip(events, events[1:])]
    summary = {"summary": "probe", **sampler.status(),
               "measured_period_ms": round(statistics.median(gaps), 1) if gaps else None}
    return [e.model_dump(mode="json") for e in events], summary


def snapshot_reader(sysfs: JetsonSysfs) -> Callable[[int | None], CpuSnapshot]:
    def snapshot(pid: int | None) -> CpuSnapshot:
        t = os.times()
        own = t.user + t.system + t.children_user + t.children_system
        if pid is not None:
            own += sysfs.process_cpu_s(pid)
        return CpuSnapshot(wall_s=time.monotonic(), busy_s=sysfs.busy_cpu_s(), own_s=own)

    return snapshot


def running_tegrastats(sysfs: JetsonSysfs) -> list[int]:
    """Process ids whose command name is tegrastats."""
    pids = []
    for comm in sysfs.root.glob("proc/[0-9]*/comm"):
        with contextlib.suppress(OSError):
            if comm.read_text().strip() == "tegrastats":
                pids.append(int(comm.parent.name))
    return sorted(pids)


def main(argv: Sequence[str] | None = None, on_jetson: Callable[[], bool] = host.on_jetson,
         sysfs: JetsonSysfs | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m contexts.telemetry.jetson_power", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("probe", help="print a few samples and a summary")
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--interval-ms", type=int, default=None)
    c = sub.add_parser("cpu", help="measure the sampler's CPU cost, off then on")
    c.add_argument("--minutes", type=float, default=10.0)
    c.add_argument("--interval-ms", type=int, default=None)
    args = parser.parse_args(argv)

    if not on_jetson():
        print("refusing: this is not a Jetson (no /etc/nv_tegra_release); the Jetson power sampler runs only on the box",
              file=sys.stderr)
        return 2
    def sampler_factory() -> JetsonPowerSampler:
        return jetson_power_sampler(os.environ, interval_ms=args.interval_ms)

    sysfs = sysfs or JetsonSysfs()

    if args.command == "probe":
        events, summary = asyncio.run(probe(sampler_factory(), args.count))
        for event in events:
            print(json.dumps(event))
        print(json.dumps(summary))
        return 0

    others = running_tegrastats(sysfs)
    if others:
        print(f"refusing: tegrastats is already running (pid {', '.join(map(str, others))}); stop it, or the "
              "gateway's sampler, so the off stretch is really off", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory(prefix="atlas-jetson-cpu-") as log_dir:
        report = asyncio.run(measure_sampler_cpu(sampler_factory, snapshot_reader(sysfs), args.minutes * 60, log_dir))
    print(json.dumps(report))
    return 0 if report["within_limit"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
