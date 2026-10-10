"""Perception maintenance.

    python -m contexts.perception backfill                # caption data/frames/**.jpg that have no caption in the log
    python -m contexts.perception backfill --limit 200    # oldest 200 only; run again for the rest
    python -m contexts.perception backfill --data-dir /mnt/atlas --config-dir config

The captions go into the event log. A running gateway does not see them until
it restarts (it replays the log); `python -m contexts.memory rebuild` also picks them up.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from bus import EventBus
from contexts.perception.application.backfill import backfill
from contexts.perception.infrastructure.frames import keyframe_loader, stored_frames
from runtime import FakeRuntime
from runtime.registry import build_runtime


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["backfill"])
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--config-dir", type=Path, default=Path("config"))
    p.add_argument("--limit", type=int, help="caption at most this many frames")
    p.add_argument("--allow-fake", action="store_true", help="write the fake runtime's canned caption (demos only)")
    args = p.parse_args(argv)

    runtime = build_runtime(args.config_dir / "models.yaml")
    if isinstance(runtime, FakeRuntime) and not args.allow_fake:
        print("no model configured (models.yaml backend: fake, ATLAS_LLM_URL unset): nothing written", file=sys.stderr)
        return 1
    bus = EventBus(args.data_dir / "events")
    n = asyncio.run(backfill(bus, runtime, stored_frames(args.data_dir), keyframe_loader(args.data_dir), limit=args.limit))
    print(f"captioned {n} keyframes with {runtime.model_info().version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
