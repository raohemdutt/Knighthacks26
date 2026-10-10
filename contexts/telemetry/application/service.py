"""The telemetry context as the app runs it: the projection, and the sampler that feeds it when there is one."""

from bus import EventBus
from contexts.telemetry.application.projection import TelemetryProjection
from contexts.telemetry.application.sampling import SamplerRunner


class TelemetryService:
    def __init__(self, projection: TelemetryProjection, runner: SamplerRunner | None = None, source: str = "") -> None:
        self.projection = projection
        self.runner = runner
        self.source = source

    def subscribe(self, bus: EventBus) -> None:
        self.projection.subscribe(bus)

    async def run(self) -> None:
        """Run the sampler until it ends or the task is cancelled; returns at once without one."""
        if self.runner is not None:
            await self.runner.run()

    def status(self) -> dict:
        """The source, the runner's counts and, for a sampler that reports on itself (the pod reader), its own."""
        out = {"source": self.source, "sampler": self.runner.status() if self.runner else None}
        own = getattr(getattr(self.runner, "sampler", None), "status", None)
        if callable(own):
            out["sampler"] = {**out["sampler"], "own": own()}
        return out
