"""One or two sentences per closed episode (M3).

    memory.episode.closed -> runtime.summarize(captions) -> memory_episodes.summary + search index

A model call is slow and costs money, so this handler is live only and runs in
the background: `EpisodeSummarizer(store, runtime).subscribe(bus)`. It never
runs on replay. The summary is written straight to the store, because
contracts/memory.py has no event that carries it yet: it survives a restart
(the file stays), but `python -m contexts.memory rebuild` leaves summaries
empty until there is one.
"""

import asyncio

from bus import EventBus
from contexts.memory.infrastructure.store import MemoryStore
from contracts.memory import EpisodeClosed
from runtime.interface import Runtime


class EpisodeSummarizer:
    def __init__(self, store: MemoryStore, runtime: Runtime) -> None:
        self.store = store
        self.runtime = runtime

    def subscribe(self, bus: EventBus) -> None:
        """Live only (no replay=True), and off the bus's path: the model takes seconds."""
        bus.subscribe("memory.episode.closed", bus.in_background(self.on_episode_closed))

    async def on_episode_closed(self, e: EpisodeClosed) -> list:
        if not e.captions or e.summary:  # nothing to summarize, or it came with one
            return []
        ep = self.store.episode(e.episode_id)
        if ep is None or ep.summary:  # never pay for the same episode twice
            return []
        summary = (await asyncio.to_thread(self.runtime.summarize, e.captions)).strip()
        if summary:
            with self.store.db:  # back on the event loop, where every other write to this file happens
                self.store.set_summary(e.episode_id, summary)
        return []
