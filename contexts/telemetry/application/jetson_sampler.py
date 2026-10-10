"""The Jetson power sampler: tegrastats lines in, one `telemetry.jetson.power` event out per line.

    sampler = JetsonPowerSampler(TegrastatsProcess(), JetsonSysfs(), NvpmodelQuery(), interval_ms=100)
    async for event in sampler.samples(): ...        # SamplerRunner publishes them onto the bus

Every event carries each rail's current mW (never tegrastats' running average), the temperatures, the power
mode, the overcurrent counters, the INA3221 averaging setting and update interval, and the period it was
sampled at; `ts` is when the line arrived. The period is the configured interval, raised to the INA3221 update
interval when that is longer: polling faster than the sensor updates only re-reads the same value and inflates
the sample count of every window.

The default of 100 ms gives a warm caption on the box (about 2.1 s, the shortest model request measured so
far) about 21 samples, where tegrastats' own default of 1000 ms gives about 2.

The power mode is read when sampling starts and again every `nvpmodel_every_s` in the background, so a mode
switch shows up in the samples within that time. The INA3221 settings are read again every
`settings_every_s`; if the update interval has grown past the period, `samples()` raises and the runner starts
tegrastats again at the new period.

A line without power rails is counted and skipped, never published as a sample with no power; so is a line
that is not tegrastats output. `status()` says how many of each, and how many lines arrived bunched together
(less than a quarter period after the previous one), which is what output held back in a buffer looks like.
"""

import asyncio
import contextlib
import logging
import math
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime
from typing import Protocol

from contexts.telemetry.domain.jetson_readings import Ina3221Settings, parse_tegrastats_line
from contexts.telemetry.domain.nodes import JETSON, check_node
from contracts.telemetry import JetsonPower

log = logging.getLogger("atlas.telemetry")

DEFAULT_INTERVAL_MS = 100
SOURCE = "tegrastats"


class TegrastatsLines(Protocol):
    pid: int | None

    def lines(self, interval_ms: int) -> AsyncIterator[tuple[datetime, str]]: ...


class JetsonSensors(Protocol):
    def forget(self) -> None: ...

    def ina3221(self) -> Ina3221Settings: ...

    def oc_events(self) -> dict[str, int]: ...


def sampling_period_ms(requested_ms: int, sensor_update_ms: float | None) -> int:
    """The period to run tegrastats at: the requested one, never shorter than the sensor's update interval."""
    if requested_ms < 1:
        raise ValueError(f"the sampling interval must be at least 1 ms, not {requested_ms}")
    if sensor_update_ms is None or sensor_update_ms <= requested_ms:
        return requested_ms
    return math.ceil(sensor_update_ms)


class JetsonPowerSampler:
    def __init__(self, tegrastats: TegrastatsLines, sensors: JetsonSensors,
                 nvpmodel: Callable[[], Awaitable[str | None]], interval_ms: int = DEFAULT_INTERVAL_MS,
                 node: str = JETSON, nvpmodel_every_s: float = 30.0, settings_every_s: float = 10.0) -> None:
        sampling_period_ms(interval_ms, None)  # refuses a period under 1 ms before anything starts
        self.tegrastats = tegrastats
        self.sensors = sensors
        self.nvpmodel = nvpmodel
        self.interval_ms = interval_ms
        self.node = check_node(node)
        self.nvpmodel_every_s = nvpmodel_every_s
        self.settings_every_s = settings_every_s
        self.period_ms: int | None = None  # the period tegrastats runs at, once started
        self.mode: str | None = None
        self.settings = Ina3221Settings(samples=None, update_ms=None)
        self.lines_read = 0
        self.samples_made = 0
        self.unparsed = 0
        self.without_rails = 0
        self.bunched = 0
        self._last_arrival: datetime | None = None

    async def samples(self) -> AsyncIterator[JetsonPower]:
        self.sensors.forget()
        self.settings = self.sensors.ina3221()
        period = sampling_period_ms(self.interval_ms, self.settings.update_ms)
        if period != self.interval_ms:
            log.warning("sampling every %d ms instead of %d ms: the INA3221 updates only every %g ms",
                        period, self.interval_ms, self.settings.update_ms)
        self.period_ms = period
        self.mode = await self.nvpmodel()
        refresher = asyncio.create_task(self._keep_mode_fresh()) if self.nvpmodel_every_s > 0 else None
        settings_every = max(1, round(self.settings_every_s * 1000 / period))
        self._last_arrival = None
        try:
            async with contextlib.aclosing(self.tegrastats.lines(period)) as lines:
                async for ts, text in lines:
                    self.lines_read += 1
                    line = parse_tegrastats_line(text)
                    if line is None:
                        self.unparsed += 1
                        if self.unparsed <= 3:
                            log.warning("not a tegrastats line: %r", text[:200])
                        continue
                    if not line.rails_mw:
                        self.without_rails += 1
                        if self.without_rails == 1:
                            log.warning("tegrastats printed no power rails; skipping such lines (%r)", text[:200])
                        continue
                    self._note_arrival(ts, period)
                    if self.lines_read % settings_every == 0:
                        self._check_settings(period)
                    self.samples_made += 1
                    yield JetsonPower(node=self.node, ts=ts, source=SOURCE, period_ms=period, rails=line.rails_mw,
                                      nvpmodel=self.mode, temps_c=line.temps_c, oc_events=self.sensors.oc_events(),
                                      sensor_avg_samples=self.settings.samples,
                                      sensor_update_ms=self.settings.update_ms)
        finally:
            if refresher is not None:
                refresher.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await refresher

    async def _keep_mode_fresh(self) -> None:
        while True:
            await asyncio.sleep(self.nvpmodel_every_s)
            self.mode = await self.nvpmodel()

    def _note_arrival(self, ts: datetime, period: int) -> None:
        if self._last_arrival is not None and (ts - self._last_arrival).total_seconds() * 1000 < period / 4:
            self.bunched += 1
            if self.bunched == 1:
                log.warning("tegrastats lines are arriving bunched together; sample times may be late")
        self._last_arrival = ts

    def _check_settings(self, period: int) -> None:
        self.settings = self.sensors.ina3221()
        if self.settings.update_ms is not None and self.settings.update_ms > period:
            raise RuntimeError(f"the INA3221 update interval is now {self.settings.update_ms:g} ms, longer than the "
                               f"{period} ms sampling period; restarting tegrastats at the new period")

    def status(self) -> dict:
        return {"requested_ms": self.interval_ms, "period_ms": self.period_ms, "nvpmodel": self.mode,
                "sensor_avg_samples": self.settings.samples, "sensor_update_ms": self.settings.update_ms,
                "lines": self.lines_read, "samples": self.samples_made, "unparsed": self.unparsed,
                "without_rails": self.without_rails, "bunched": self.bunched, "tegrastats_pid": self.tegrastats.pid}
