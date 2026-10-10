"""Boundary doubles for telemetry: stand-ins for the samplers that read real hardware."""

from contexts.telemetry.fakes.sampler import FakeSampler

__all__ = ["FakeSampler"]
