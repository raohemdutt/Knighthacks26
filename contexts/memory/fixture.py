"""Canned memory for demos and tests (M7): a hackathon table from Saturday midday to Sunday morning.

    fx = load()                      # fixtures/memory/memory.yaml
    memory = fake_memory(fx)         # the agent's FakeMemory
    fill_store(store, fx)            # or the real SQLite store, read through SqliteMemory

Same YAML shape as fixtures/golden/memory.yaml, plus `items`.
"""

from datetime import datetime
from pathlib import Path

import yaml
from pydantic import BaseModel

from contexts.memory.infrastructure.store import MemoryStore
from contracts.memory import Episode, Item, Sighting

FIXTURE = Path("fixtures/memory/memory.yaml")


class Fixture(BaseModel):
    now: datetime  # when the demo questions are asked
    episodes: list[Episode] = []
    sightings: list[Sighting] = []
    items: list[Item] = []


def load(path: str | Path = FIXTURE) -> Fixture:
    return Fixture.model_validate(yaml.safe_load(Path(path).read_text()))


def fake_memory(fx: Fixture):
    from contexts.agent.application.memory_port import FakeMemory  # the agent's test double, only needed here

    return FakeMemory(episodes=fx.episodes, sightings=fx.sightings, items=fx.items)


def fill_store(store: MemoryStore, fx: Fixture) -> None:
    """Write the fixture's rows and search text. Loading twice leaves the same rows."""
    with store.db:
        for ep in fx.episodes:
            store.put_episode(ep, ep.end or ep.start)
            store.unindex(ep.id)
            for caption in ep.captions:
                store.index("caption", ep.id, caption)
            if ep.summary:
                store.index("summary", ep.id, ep.summary)
        for s in fx.sightings:
            store.add_sighting(s)
        for item in fx.items:
            store.put_item(item, synced_at=item.when or fx.now)
