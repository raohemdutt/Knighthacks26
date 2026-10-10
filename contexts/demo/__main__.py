"""The demo event feed.

    python -m contexts.demo seed [--data-dir data] [--now | --end ISO] [--force]
        The whole demo night, published through the real app: every projection fills from the log.
        Then run the app on it:
            ATLAS_DATA_DIR=data ATLAS_MEMORY_FIXTURE= ATLAS_CONNECTORS=0 python -m contexts.gateway

    python -m contexts.demo fake <context> [--data-dir data] [--now]
        A few realistic events of every type one context publishes, appended to the log.
        Contexts: perception connectors memory habits impact agent circle vault play
"""

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

from contexts.demo import fakes
from contexts.demo.seed import SeedRefused, seed, summary
from contexts.demo.story import load


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m contexts.demo", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("seed", help="publish the demo night through the real app")
    s.add_argument("--data-dir", type=Path, default=Path("data"))
    s.add_argument("--now", action="store_true", help="move the story so it ends at the current time")
    s.add_argument("--end", type=datetime.fromisoformat, help="move the story so it ends at this time (with a UTC offset)")
    s.add_argument("--force", action="store_true", help="add to a data directory that already has an event log")
    f = sub.add_parser("fake", help="publish a small sample of one context's events")
    f.add_argument("context", choices=fakes.CONTEXTS)
    f.add_argument("--data-dir", type=Path, default=Path("data"))
    f.add_argument("--now", action="store_true", help="start at the current time instead of the demo night's")
    args = p.parse_args(argv)

    if args.command == "seed":
        if args.end and args.end.tzinfo is None:
            p.error("--end needs a UTC offset, e.g. 2026-10-11T09:30:00-04:00")
        end = datetime.now(UTC).replace(microsecond=0) if args.now else args.end
        try:
            seeded = asyncio.run(seed(args.data_dir, now=end, force=args.force))
        except SeedRefused as refused:
            print(refused, file=sys.stderr)
            return 1
        print(summary(seeded))
        moved = seeded.now - load().now
        if moved.total_seconds() % 86400:
            print("\nNote: the story was moved by a part of a day, so its nights and meals no longer fall on the same local "
                  "days and the habit numbers differ from the script's. Use --end with a 09:30 local time to keep them.")
        return 0

    start = datetime.now(UTC).replace(microsecond=0) if args.now else load().now
    published = asyncio.run(fakes.publish(args.context, args.data_dir / "events", start))
    for e in published:
        print(f"{e.ts.isoformat()}  {e.type}")
    print(f"published {len(published)} {args.context} events into {args.data_dir / 'events'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
