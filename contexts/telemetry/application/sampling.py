"""Running a sampler: every sample it yields is published on the bus, which appends it to the event log.

    runner = SamplerRunner(sampler, bus.publish, nodes=["jetson"])
    await runner.run()          # until the sampler ends or the task is cancelled

A sampler is anything with `samples()`, an async iterator of telemetry events; the fake and the
real samplers (tegrastats on the Jetson, the pod reader) all run through this one runner.

It fails closed: only `telemetry.*` events for the nodes the runner was started for are published,
and anything else is dropped and counted. A sampler that raises is logged and started again after
`retry_s`, so a flaky sensor never takes the app down; cancelling the task stops it.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from typing import Protocol

from contexts.telemetry.domain.nodes import check_node
from contracts.events import Event

log = logging.getLogger("atlas.telemetry")


class Sampler(Protocol):
    def samples(self) -> AsyncIterator[Event]: ...


class SamplerRunner:
    def __init__(self, sampler: Sampler, publish: Callable[[Event], Awaitable[None]], nodes: Iterable[str],
                 retry_s: float = 5.0, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
        self.sampler = sampler
        self.publish = publish
        self.nodes = frozenset(check_node(n) for n in nodes)
        self.retry_s = retry_s
        self.sleep = sleep
        self.published = 0
        self.dropped = 0
        self.errors = 0
        self.last_error: str | None = None

    def allowed(self, event: Event) -> bool:
        return event.type.startswith("telemetry.") and getattr(event, "node", None) in self.nodes

    async def run(self) -> None:
        while True:
            try:
                async for event in self.sampler.samples():
                    if not self.allowed(event):
                        self.dropped += 1
                        log.warning("dropped %s for node %r: this sampler speaks for %s", event.type,
                                    getattr(event, "node", None), ", ".join(sorted(self.nodes)))
                        continue
                    await self.publish(event)
                    self.published += 1
                return  # the sampler ran out (a finite replay)
            except Exception as exc:
                self.errors += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
                log.exception("telemetry sampler failed; starting it again in %.0f s", self.retry_s)
                await self.sleep(self.retry_s)

    def status(self) -> dict:
        return {"nodes": sorted(self.nodes), "published": self.published, "dropped": self.dropped,
                "errors": self.errors, "last_error": self.last_error}
