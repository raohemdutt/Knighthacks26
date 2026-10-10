"""Stand-ins for the Jetson's tegrastats and nvpmodel, replaying recorded output, for tests on machines without one.

    python contexts/telemetry/fakes/jetson_tools.py tegrastats --interval 100 [--count N] [--exit-code K] [--file F]
    python contexts/telemetry/fakes/jetson_tools.py nvpmodel -q [--file F]

tegrastats prints the recorded lines (fixtures/telemetry/jetson_tegrastats_orin_nano.txt by default) one per
interval, in a loop, each with the current local time in place of the recorded one, flushing every line; with
--count it stops after that many lines and exits with --exit-code. nvpmodel prints the recorded `nvpmodel -q`
output (fixtures/telemetry/jetson_nvpmodel_q.txt). Lines starting with # in a fixture are provenance and are
not printed. Runs on its own (no imports from the repo), so a test can start it as a separate process.
"""

import argparse
import re
import sys
import time
from datetime import datetime
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "telemetry"
STAMP = re.compile(r"^\d\d-\d\d-\d{4} \d\d:\d\d:\d\d")


def recorded(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


def tegrastats(args: argparse.Namespace) -> int:
    lines = recorded(Path(args.file) if args.file else FIXTURES / "jetson_tegrastats_orin_nano.txt")
    printed = 0
    while args.count is None or printed < args.count:
        line = lines[printed % len(lines)]
        print(STAMP.sub(datetime.now().strftime("%m-%d-%Y %H:%M:%S"), line), flush=True)
        printed += 1
        if args.count is None or printed < args.count:
            time.sleep(args.interval / 1000)
    return args.exit_code


def nvpmodel(args: argparse.Namespace) -> int:
    for line in recorded(Path(args.file) if args.file else FIXTURES / "jetson_nvpmodel_q.txt"):
        print(line, flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="fake Jetson tools replaying recorded output")
    sub = parser.add_subparsers(dest="tool", required=True)
    t = sub.add_parser("tegrastats")
    t.add_argument("--interval", type=int, default=1000)
    t.add_argument("--count", type=int, default=None)
    t.add_argument("--exit-code", type=int, default=0)
    t.add_argument("--file", default=None)
    n = sub.add_parser("nvpmodel")
    n.add_argument("-q", action="store_true")
    n.add_argument("--file", default=None)
    args = parser.parse_args(argv)
    return tegrastats(args) if args.tool == "tegrastats" else nvpmodel(args)


if __name__ == "__main__":
    sys.exit(main())
