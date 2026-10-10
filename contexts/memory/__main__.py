"""Memory maintenance.

    python -m contexts.memory rebuild                 # data/memory.sqlite from data/events/*.jsonl
    python -m contexts.memory rebuild --data-dir /mnt/atlas --db /mnt/atlas/memory.sqlite
    python -m contexts.memory seed                    # load fixtures/memory/memory.yaml (demo without cameras)
    python -m contexts.memory embed                   # vectors for semantic search (M5); run it after a rebuild
    python -m contexts.memory embed --all             # throw the vectors away and embed everything again

`embed` needs a text embedding model: `embeddings.text` in config/models.yaml, and a llama-server runtime
(atlas_model.backend, or ATLAS_LLM_URL). It only adds what is missing, so it is safe to run again.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from bus import EventBus
from contexts.memory.application.embedder import TextEmbedder, embedder_from
from contexts.memory.application.indexer import VectorIndexer
from contexts.memory.application.projection import MemoryProjection
from contexts.memory.application.reader import SqliteMemory
from contexts.memory.fixture import FIXTURE, fill_store, load
from contexts.memory.infrastructure.store import MemoryStore


def rebuild(events_dir: Path, db: Path) -> int:
    """Drop the file and project the whole log again. Returns the number of events replayed.

    Episode summaries (M3) are not in the log, so a rebuilt file has none until episodes close again."""
    for leftover in (db, db.with_name(db.name + "-wal"), db.with_name(db.name + "-shm")):
        leftover.unlink(missing_ok=True)
    store = MemoryStore(db)
    bus = EventBus(events_dir, persist=False)
    MemoryProjection(store).subscribe(bus)
    try:
        return asyncio.run(bus.replay_into())
    finally:
        store.close()


def embed(db: Path, embedder: TextEmbedder, everything: bool = False) -> tuple[int, int]:
    """Embed the indexed texts that have no vector from this model. Returns (embedded, still missing).

    Something is still missing only when the model stopped answering part of the way; run it again."""
    store = MemoryStore(db)
    try:
        if everything:
            with store.db:
                store.clear_vectors()
        done = asyncio.run(VectorIndexer(store, embedder).run())
        return done, len(store.texts_without_vector(embedder.model, -1))
    finally:
        store.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["rebuild", "seed", "embed"])
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--db", type=Path, help="default: <data-dir>/memory.sqlite")
    p.add_argument("--fixture", type=Path, default=FIXTURE)
    p.add_argument("--config-dir", type=Path, default=Path("config"), help="embed: where models.yaml is")
    p.add_argument("--all", action="store_true", help="embed: redo every vector, not only the missing ones")
    args = p.parse_args(argv)
    db = args.db or args.data_dir / "memory.sqlite"

    if args.command == "rebuild":
        print(f"replayed {rebuild(args.data_dir / 'events', db)} events into {db}")
    elif args.command == "embed":
        from runtime.registry import build_runtime  # only this command needs a model

        models_yaml = args.config_dir / "models.yaml"
        embedder = embedder_from(build_runtime(models_yaml), models_yaml) if models_yaml.exists() else None
        if embedder is None:
            print(f"no text embedding model: set embeddings.text.backend in {models_yaml} and use a llama-server runtime")
            return 1
        done, missing = embed(db, embedder, everything=args.all)
        print(f"embedded {done} texts with {embedder.model} into {db}" + (f"; {missing} still missing (is the model up?)" if missing else ""))
        if missing:
            return 1
    else:
        store = MemoryStore(db)
        fill_store(store, load(args.fixture))
        store.close()
        print(f"loaded {args.fixture} into {db}")
    store = MemoryStore(db)
    stats = SqliteMemory(store).stats()
    store.close()
    print(f"{stats.episodes} episodes, {stats.captions} captions, {stats.hours_recorded} hours recorded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
