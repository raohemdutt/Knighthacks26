"""The telemetry context as the app runs it: the projections, the samplers that feed them, and joules per answer.

    service.subscribe(bus)     # the projection and joules per answer, both rebuilt from the log (replay=True)
    await service.run()        # every sampler until they end; meanwhile, every `flush_s`, the answers' values that
                               # can be computed are published (telemetry.joules_per_answer), and once more at the end
    service.stats()            # for the stats page: watts now, joules per answer and energy per day, per node
"""

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable, Sequence

from bus import EventBus
from contexts.telemetry.application.joules import JoulesPerAnswerProjection
from contexts.telemetry.application.projection import TelemetryProjection
from contexts.telemetry.application.sampling import SamplerRunner
from contracts.events import Event

log = logging.getLogger("atlas.telemetry")

FLUSH_S = 5.0


class TelemetryService:
    def __init__(self, projection: TelemetryProjection, runner: SamplerRunner | Sequence[SamplerRunner] | None = None,
                 source: str = "", joules: JoulesPerAnswerProjection | None = None,
                 publish: Callable[[Event], Awaitable[None]] | None = None, flush_s: float = FLUSH_S,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
        self.projection = projection
        self.runners: list[SamplerRunner] = ([] if runner is None else [runner] if isinstance(runner, SamplerRunner)
                                             else list(runner))
        self.source = source
        self.joules = joules
        self.publish = publish
        self.flush_s = flush_s
        self.sleep = sleep

    @property
    def runner(self) -> SamplerRunner | None:
        """The sampler runner when there is exactly one (one source), else None."""
        return self.runners[0] if len(self.runners) == 1 else None

    def subscribe(self, bus: EventBus) -> None:
        self.projection.subscribe(bus)
        if self.joules is not None:
            self.joules.subscribe(bus)
            if self.publish is None:
                self.publish = bus.publish

    async def flush(self) -> int:
        """Publish the joules-per-answer values that can be computed now; returns how many."""
        if self.joules is None or self.publish is None:
            return 0
        values = self.joules.ready()
        for value in values:
            await self.publish(value)
        return len(values)

    async def _flush_forever(self) -> None:
        while True:
            await self.sleep(self.flush_s)
            try:
                await self.flush()
            except Exception:
                log.exception("computing joules per answer failed; trying again in %.0f s", self.flush_s)

    async def run(self) -> None:
        """Run the samplers until they end or the task is cancelled; returns at once without one."""
        if not self.runners:
            return
        flusher = asyncio.create_task(self._flush_forever()) if self.joules is not None else None
        try:
            await asyncio.gather(*(r.run() for r in self.runners))
        finally:
            if flusher is not None:
                flusher.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await flusher
        await self.flush()

    def status(self) -> dict:
        if len(self.runners) <= 1:
            return {"source": self.source, "sampler": self.runner.status() if self.runner else None}
        names = self.source.split(",")
        return {"source": self.source, "samplers": {n: r.status() for n, r in zip(names, self.runners)}}

    def stats(self) -> dict:
        """What the stats page shows, per node: watts now, joules per answer, and energy per day."""
        if self.joules is None:
            return {"nodes": {}, "waiting": {}}
        return self.joules.stats()
