"""Vectors for the search index (M5).

    new caption, item, note or closed episode -> runtime.embed_text(texts without a vector) -> memory_vectors

Like the summarizer, this calls a model, so it is live only and runs in the
background: `VectorIndexer(store, embedder).subscribe(bus)`. It never runs on
replay. The event is only a nudge: each run embeds every indexed text that has
no vector from the current model, in batches, so the first live event after a
start also backfills whatever was there already (a fixture, a rebuilt file, a
summary written since the last run). Without an embedder, or while the model is
unavailable, it does nothing and search stays keyword-only.

Vectors are not in the event log, so `python -m contexts.memory rebuild` leaves
none; `python -m contexts.memory embed` makes them again.
"""

import asyncio

from bus import EventBus
from contexts.memory.application.embedder import TextEmbedder
from contexts.memory.infrastructure.store import MemoryStore

BATCH = 32
# Everything that adds text to memory_fts. Listed after the projection subscribes, so the text is there first.
NUDGES = ("perception.scene", "connector.item.synced", "agent.note.written", "memory.episode.closed",
          "workspace.file.added")


class VectorIndexer:
    def __init__(self, store: MemoryStore, embedder: TextEmbedder | None, batch: int = BATCH) -> None:
        self.store = store
        self.embedder = embedder
        self.batch = batch
        self._running = False

    def subscribe(self, bus: EventBus) -> None:
        """Live only (no replay=True), and off the bus's path: a batch of embeddings takes a while."""
        if self.embedder is None:
            return
        for prefix in NUDGES:
            bus.subscribe(prefix, bus.in_background(self.on_indexed))

    async def on_indexed(self, _event) -> list:
        await self.run()
        return []

    async def run(self) -> int:
        """Embed every indexed text that has no vector yet. Returns how many were embedded.

        One run at a time: a nudge that arrives during a run returns at once, and the run picks its text up,
        because it keeps asking the store for more until there is none."""
        if self.embedder is None or self._running:
            return 0
        self._running = True
        done = 0
        try:
            with self.store.db:
                self.store.prune_vectors()
            while texts := self.store.texts_without_vector(self.embedder.model, self.batch):
                vectors = await asyncio.to_thread(self.embedder.embed, texts)
                if vectors is None:  # no model, or it is down: try again on a later nudge
                    break
                with self.store.db:  # back on the event loop, where every other write to this file happens
                    self.store.put_vectors(self.embedder.model, dict(zip(texts, vectors)))
                done += len(texts)
        finally:
            self._running = False
        return done
