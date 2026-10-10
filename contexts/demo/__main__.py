"""The demo event feed.

    python -m contexts.demo seed [--data-dir data] [--now | --end ISO] [--force]
        The whole demo night, published through the real app: every projection fills from the log.
        Then run the app on it:
            ATLAS_DATA_DIR=data ATLAS_MEMORY_FIXTURE= ATLAS_CONNECTORS=0 python -m contexts.gateway

    python -m contexts.demo fake <context> [--data-dir data] [--now]
        A few realistic events of every type one context publishes, appended to the log.

    python -m contexts.demo fake perception --gateway http://<box>:8000 [--token $ATLAS_TOKEN]
        The camera scene pushed into a running box (POST /api/events), timed from now: its own zone
        engine, rules, Sort It and Waste Watch react live, as if the camera saw it.
        Contexts: perception connectors memory habits impact agent circle vault play
"""

import argparse
import os
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
    f.add_argument("--gateway", help="push into a running box instead of writing the log, e.g. http://192.168.1.20:8000")
    f.add_argument("--token", default=os.environ.get("ATLAS_TOKEN"), help="the box's ATLAS_TOKEN, if it has one")
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

    if args.gateway:
        return push(args.context, args.gateway, args.token)
    start = datetime.now(UTC).replace(microsecond=0) if args.now else load().now
    published = asyncio.run(fakes.publish(args.context, args.data_dir / "events", start))
    for e in published:
        print(f"{e.ts.isoformat()}  {e.type}")
    print(f"published {len(published)} {args.context} events into {args.data_dir / 'events'}")
    return 0


def push(context: str, gateway: str, token: str | None) -> int:
    """The context's fake, timed from now, POSTed to a running gateway. Only input the box accepts is sent;
    zone events and the rest are what the box makes of it."""
    import tempfile

    import httpx

    from contexts.gateway.app import INPUT_EVENTS

    with tempfile.TemporaryDirectory() as tmp:
        events = asyncio.run(fakes.publish(context, tmp, datetime.now(UTC).replace(microsecond=0)))
    sent = [e for e in events if e.type.startswith(INPUT_EVENTS)]
    if not sent:
        print(f"{context} has no camera or connector input to push", file=sys.stderr)
        return 1
    body = [e.model_dump(mode="json", exclude={"event_id"}) for e in sent]
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = httpx.post(gateway.rstrip("/") + "/api/events", json=body, headers=headers, timeout=60)
    if r.status_code != 200:
        print(f"{r.status_code}: {r.text}", file=sys.stderr)
        return 1
    print(f"pushed {r.json()['published']} events into {gateway}: {', '.join(r.json()['types'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
