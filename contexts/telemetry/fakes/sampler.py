"""A fake sampler: recorded telemetry replayed as if it were being sampled now.

    sampler = FakeSampler(recorded.load())               # fixtures/telemetry/synthetic_minute.jsonl
    async for event in sampler.samples(): ...

Each pass gives every event a fresh `event_id` and moves all its times (`ts`, `started`,
`ended`, `received_ts` ...) so the recording starts at the moment the sampler starts. With
`loop`, the next pass continues the timeline, and every periodic series (one node's samples of
one type) keeps its recorded period across the seam. `speed` paces the replay: 1 is the recorded
pace, 10 is ten times faster, 0 is as fast as it can go.

The default fixture is synthetic: plausible numbers shaped like the nodes', not measurements. Its
power samples say `source: "fake"`.
"""

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from contracts.events import Event


class FakeSampler:
    def __init__(self, events: Sequence[Event], speed: float = 1.0, loop: bool = False,
                 clock: Callable[[], datetime] = lambda: datetime.now(UTC),
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
        if not events:
            raise ValueError("a fake sampler needs at least one recorded event")
        if speed < 0:
            raise ValueError("speed must be 0 (no pacing) or more")
        self.events = sorted(events, key=lambda e: e.ts)
        self.speed = speed
        self.loop = loop
        self.clock = clock
        self.sleep = sleep
        self.pass_length = _pass_length(self.events)

    async def samples(self) -> AsyncIterator[Event]:
        first = self.events[0].ts
        anchor = self.clock()
        offset = timedelta(0)
        while True:
            for recorded in self.events:
                shift = anchor - first + offset
                due = recorded.ts + shift
                if self.speed:
                    wait = (due - anchor).total_seconds() / self.speed - (self.clock() - anchor).total_seconds()
                    if wait > 0:
                        await self.sleep(wait)
                yield _moved(recorded, shift)
            if not self.loop:
                return
            offset += self.pass_length


def _pass_length(events: Sequence[Event]) -> timedelta:
    """How far apart two passes start: each series of one node's samples of one type comes back one of its
    own periods (its median gap) after its last sample; never less than the recording plus its smallest gap."""
    series: dict[tuple, list[datetime]] = defaultdict(list)
    for e in events:
        series[(getattr(e, "node", None), e.type)].append(e.ts)
    lengths = []
    for times in series.values():
        gaps = sorted(b - a for a, b in zip(times, times[1:]) if b > a)
        if gaps:
            lengths.append(times[-1] - times[0] + gaps[len(gaps) // 2])
    every_gap = [b.ts - a.ts for a, b in zip(events, events[1:]) if b.ts > a.ts]
    floor = events[-1].ts - events[0].ts + (min(every_gap) if every_gap else timedelta(seconds=1))
    return max([floor, *lengths])


def _moved(event: Event, shift: timedelta) -> Event:
    """A copy of `event` with a fresh id and every datetime field moved by `shift`."""
    times = {name: value + shift for name, value in event if isinstance(value, datetime)}
    return event.model_copy(update={**times, "event_id": uuid4().hex})
