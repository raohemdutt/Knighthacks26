"""Memory maintenance.

    python -m contexts.memory rebuild                 # data/memory.sqlite from data/events/*.jsonl
    python -m contexts.memory rebuild --data-dir /mnt/atlas --db /mnt/atlas/memory.sqlite
    python -m contexts.memory seed                    # load fixtures/memory/memory.yaml (demo without cameras)
"""

import argparse
import asyncio
import sys
from pathlib import Path

from bus import EventBus
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["rebuild", "seed"])
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--db", type=Path, help="default: <data-dir>/memory.sqlite")
    p.add_argument("--fixture", type=Path, default=FIXTURE)
    args = p.parse_args(argv)
    db = args.db or args.data_dir / "memory.sqlite"

    if args.command == "rebuild":
        print(f"replayed {rebuild(args.data_dir / 'events', db)} events into {db}")
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
