"""What one answer on the box cost, for the Learning lab's Ask panel: "this answer cost about X J on the box" (G6).

    meter = AnswerMeter(atlas.telemetry)               # the telemetry service, or None when no sampler runs
    energy = await meter.measure(started, ended)       # the dict below, or None: "sampler off"

It reads the Jetson power sampler the telemetry context already runs in the gateway (ATLAS_TELEMETRY_SOURCE=
tegrastats, telemetry.jetson.power events folded by its projection) and applies Context: Telemetry's rule for joules
per answer (T4): the box's module input power (VDD_IN) integrated over the answer's window, gross, nothing
subtracted, with the box's calibrated idle stored beside it (config/telemetry_idle.yaml, 15.76 W, T13). The integral
is contexts/telemetry/domain/energy.integrate: the last sample at or before the start and the first at or after the
end bracket the window, the edges clipped by linear interpolation.

    {"joules": 41.3,          gross J over the window
     "idle_w": 15.76,         the box's calibrated idle, beside it (null when the box has none)
     "seconds": 2.4,          the window: from the request reaching the gateway to its answer
     "sampler": "tegrastats", the tool the samples came from ("fake replay of ..." on a laptop's fake source)
     "samples": 7,            samples inside the window (one every 350 ms)
     "above_idle_j": 3.5,     joules minus idle_w x seconds (null without idle_w)
     "low_confidence": true}  fewer than 10 samples inside the window, as contexts/telemetry/application/joules.py says

None, which the panel shows as "sampler off", when there is no telemetry service, when the box's node has no power
sample in the FRESH_S before the answer started (the sampler is not running), or when no sample after the end
arrives within WAIT_S. It never raises into the answer: the caller treats any failure as None.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from typing import Any

from contexts.telemetry.domain.energy import integrate
from contexts.telemetry.domain.nodes import JETSON

FRESH_S = 5.0  # a sampler that has not sampled for this long before the answer is not running
WAIT_S = 1.5  # how long to wait for the first sample after the answer's end (the period is 350 ms)
POLL_S = 0.1
MIN_SAMPLES = 10  # contexts/telemetry/application/joules.py MIN_SAMPLES


class AnswerMeter:
    def __init__(self, telemetry: Any | None, *, node: str = JETSON, idle_w: float | None = None,
                 wait_s: float = WAIT_S, poll_s: float = POLL_S,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                 monotonic: Callable[[], float] = time.monotonic) -> None:
        self.telemetry = telemetry
        self.node = node
        self._idle_w = idle_w
        self.wait_s = wait_s
        self.poll_s = poll_s
        self.sleep = sleep
        self.monotonic = monotonic

    def idle_w(self) -> float | None:
        """The node's calibrated idle: the one given, else the telemetry service's own (the figure it stores)."""
        if self._idle_w is not None:
            return self._idle_w
        idle = getattr(getattr(getattr(self.telemetry, "joules", None), "calibration", None), "idle", None) or {}
        figure = idle.get(self.node)
        return getattr(figure, "watts", None)

    def _trace(self, start: datetime, end: datetime) -> list[tuple[datetime, float, str]]:
        projection = getattr(self.telemetry, "projection", None)
        if projection is None:
            return []
        span = timedelta(seconds=FRESH_S)
        return [(r.ts, r.watts, r.source) for r in projection.readings(self.node, start - span, end + span)
                if r.watts is not None]

    async def measure(self, start: datetime, end: datetime) -> dict[str, Any] | None:
        if end < start:
            raise ValueError("the answer ends before it starts")
        trace = self._trace(start, end)
        if not any(t <= start for t, _, _ in trace):
            return None  # nothing sampled just before the answer: the sampler is off (or not on this node)
        deadline = self.monotonic() + self.wait_s
        while not any(t >= end for t, _, _ in trace) and self.monotonic() < deadline:
            await self.sleep(self.poll_s)
            trace = self._trace(start, end)
        joules = integrate([(t.timestamp(), w) for t, w, _ in trace], start.timestamp(), end.timestamp())
        if joules is None:
            return None
        seconds = (end - start).total_seconds()
        idle = self.idle_w()
        inside = sum(1 for t, _, _ in trace if start <= t <= end)
        tool = trace[-1][2]
        source = str(getattr(self.telemetry, "source", "") or "")
        return {"joules": round(joules, 1), "idle_w": idle, "seconds": round(seconds, 2),
                "sampler": f"fake replay of {tool}" if "fake" in source.split(",") and tool != "fake" else tool,
                "samples": inside,
                "above_idle_j": round(joules - idle * seconds, 1) if idle is not None else None,
                "low_confidence": inside < MIN_SAMPLES}
