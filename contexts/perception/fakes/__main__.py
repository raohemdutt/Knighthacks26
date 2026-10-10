"""Publish the scripted scene into the event log, with the zone events it causes.

    python -m contexts.perception.fakes                  # starts now, all at once
    python -m contexts.perception.fakes --realtime       # one second per second
    python -m contexts.perception.fakes --start 2026-10-11T12:00:00+00:00 --log-dir data/events

The gateway replays the log when it starts, so run this first and the app
comes up with the scene already in it.
"""

import argparse
import asyncio
from datetime import UTC, datetime, timedelta

from bus import EventBus
from contexts.perception.domain.zones import ZoneEngine
from contexts.perception.fakes.script import LENGTH_S, play


def _show(e) -> None:
    dwell = f" {e.dwell_s:.0f}s" if e.dwell_s is not None else ""
    print(f"{e.ts:%H:%M:%S} {e.camera_id} {e.label}#{e.track_id} {e.kind} {e.zone_id}{dwell}")


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start", type=datetime.fromisoformat, default=None, help="event time of the first observation (default: now)")
    p.add_argument("--realtime", action="store_true", help="pace the script one second per second")
    p.add_argument("--log-dir", default="data/events")
    p.add_argument("--zones", default="config/zones.yaml")
    args = p.parse_args()

    bus = EventBus(args.log_dir)
    engine = ZoneEngine.load(args.zones)
    bus.subscribe("perception.observation", engine.on_observation)
    bus.subscribe("perception.zone", _show)
    start = args.start or datetime.now(UTC).replace(microsecond=0)
    n = await play(bus, start, realtime=args.realtime)
    await bus.publish_all(engine.flush(start + timedelta(seconds=LENGTH_S) + engine.exit_after))  # camera off: what is left exits
    print(f"published {n} observations from {start.isoformat()} into {args.log_dir}")


if __name__ == "__main__":
    asyncio.run(main())
