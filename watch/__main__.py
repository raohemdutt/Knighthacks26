"""python -m watch <pull|summary|daily|sleep|workouts> [args]. Run from the repo root."""

from __future__ import annotations

import sys


def main() -> int:
    verb, *rest = sys.argv[1:] or ["--help"]
    if verb == "pull":
        from watch.pull import main as run
    elif verb in ("summary", "daily", "sleep", "workouts"):
        from watch.cli import main as cli

        return cli([verb, *rest])
    else:
        print(__doc__)
        return 0 if verb in ("-h", "--help") else 2
    return run(rest)


if __name__ == "__main__":
    sys.exit(main())
