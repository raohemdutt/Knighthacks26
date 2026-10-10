"""Building the telemetry context from one setting, for the composition root.

    telemetry = build_telemetry("fake", bus)        # what ATLAS_TELEMETRY_SOURCE=fake does
    task = asyncio.create_task(telemetry.run())     # once the event loop is up, after replay

The projection is subscribed to the bus here (replay=True); the sampler publishes when `run()` runs.

Sources:
- "fake" replays recorded samples (fixtures/telemetry) as live ones, at the recorded pace and on a
  loop, for laptops and demos without hardware. It refuses to run on a Jetson, where its samples
  would land in the box's own event log as if they had been measured.
"""

from collections.abc import Callable
from pathlib import Path

from bus import EventBus
from contexts.telemetry.application.projection import TelemetryProjection
from contexts.telemetry.application.sampling import SamplerRunner
from contexts.telemetry.application.service import TelemetryService
from contexts.telemetry.domain.nodes import KNOWN_NODES
from contexts.telemetry.fakes.sampler import FakeSampler
from contexts.telemetry.infrastructure import host, recorded

SOURCES = ("fake",)


def build_telemetry(source: str, bus: EventBus, *, fixture: str | Path | None = None, speed: float = 1.0,
                    loop: bool = True, on_jetson: Callable[[], bool] | None = None) -> TelemetryService:
    if source not in SOURCES:
        raise ValueError(f"unknown telemetry source {source!r}; known sources: {', '.join(SOURCES)}")
    if (on_jetson or host.on_jetson)():
        raise RuntimeError("the fake telemetry source does not run on a Jetson: its samples would go into the "
                           "box's event log as if they had been measured")
    sampler = FakeSampler(recorded.load(fixture or recorded.DEFAULT), speed=speed, loop=loop)
    service = TelemetryService(TelemetryProjection(), SamplerRunner(sampler, bus.publish, nodes=KNOWN_NODES),
                               source=source)
    service.subscribe(bus)
    return service
