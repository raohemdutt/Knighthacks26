"""The fake sampler replays the recorded fixture as live samples: new ids, times moved to now, paced."""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from contexts.telemetry.fakes import FakeSampler
from contexts.telemetry.infrastructure import recorded
from contracts.telemetry import RequestWindow

START = datetime(2026, 10, 11, 9, 30, tzinfo=UTC)


class Clock:
    """Wall time that moves only when the sampler sleeps."""

    def __init__(self, now: datetime) -> None:
        self.now = now
        self.waits: list[float] = []

    def __call__(self) -> datetime:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.now += timedelta(seconds=seconds)


async def take(sampler: FakeSampler, n: int) -> list:
    out = []
    stream = sampler.samples()
    async for event in stream:
        out.append(event)
        if len(out) == n:
            break
    await stream.aclose()
    return out


def run(coro):
    return asyncio.run(coro)


def without_times_and_id(event) -> dict:
    return {k: v for k, v in event.model_dump().items() if k != "event_id" and not isinstance(v, datetime)}


def test_a_pass_is_the_recording_with_fresh_ids_and_every_time_moved_to_the_start():
    fixture = recorded.load()
    clock = Clock(START)
    replay = run(take(FakeSampler(fixture, speed=0, clock=clock, sleep=clock.sleep), len(fixture)))
    shift = START - fixture[0].ts
    assert [without_times_and_id(e) for e in replay] == [without_times_and_id(e) for e in fixture]
    assert [e.ts for e in replay] == [e.ts + shift for e in fixture] and replay[0].ts == START
    windows = [(e.started, e.first_token, e.ended) for e in replay if isinstance(e, RequestWindow)]
    assert windows == [(e.started + shift, e.first_token + shift, e.ended + shift)
                       for e in fixture if isinstance(e, RequestWindow)]
    assert len({e.event_id for e in replay}) == len(fixture)
    assert not {e.event_id for e in replay} & {e.event_id for e in fixture}
    assert clock.waits == []  # speed 0: no pacing


def test_it_keeps_the_recorded_pace_and_speed_divides_it():
    fixture = recorded.load()
    span = (fixture[-1].ts - fixture[0].ts).total_seconds()
    for speed in (1.0, 4.0):
        clock = Clock(START)
        run(take(FakeSampler(fixture, speed=speed, clock=clock, sleep=clock.sleep), len(fixture)))
        assert sum(clock.waits) == pytest.approx(span / speed)
        assert (clock.now - START).total_seconds() == pytest.approx(span / speed, abs=1e-3)


def test_on_a_loop_each_node_keeps_its_own_period_across_the_seam_between_passes():
    fixture = recorded.load()
    clock = Clock(START)
    two_passes = run(take(FakeSampler(fixture, speed=0, loop=True, clock=clock, sleep=clock.sleep), 2 * len(fixture)))
    first, second = two_passes[:len(fixture)], two_passes[len(fixture):]

    def seam(node: str, type_name: str) -> timedelta:
        last = [e.ts for e in first if e.node == node and e.type == type_name][-1]
        return [e.ts for e in second if e.node == node and e.type == type_name][0] - last

    assert seam("jetson", "telemetry.jetson.power") == timedelta(seconds=1)
    assert seam("gpu-a", "telemetry.gpu.power") == timedelta(seconds=1)
    assert seam("gpu-b", "telemetry.gpu.power") == timedelta(seconds=1)
    assert seam("gpu-b", "telemetry.train.step") == timedelta(seconds=2.5)
    assert second[0].ts - first[0].ts == timedelta(seconds=60)
    assert [e.ts for e in two_passes] == sorted(e.ts for e in two_passes)
    assert [without_times_and_id(e) for e in second] == [without_times_and_id(e) for e in first]
    assert len({e.event_id for e in two_passes}) == 2 * len(fixture)


def test_it_needs_something_to_replay_and_a_pace_that_is_not_negative():
    with pytest.raises(ValueError, match="at least one recorded event"):
        FakeSampler([])
    with pytest.raises(ValueError, match="speed"):
        FakeSampler(recorded.load(), speed=-1)
