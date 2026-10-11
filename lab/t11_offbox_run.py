"""The T11 closed loop run from a workstation, with the outage rule of the real run and a record of progress.

    python -m lab.t11_offbox_run --progress lab/state/t11-progress.jsonl -- <routing-closed-loop arguments>

It runs `python -m lab routing-closed-loop` (lab/routing_closed_loop.py) unchanged, with two additions:

- Outages. When the teacher or the student gives no usable answer (a 5xx such as a 500, or no connection: the
  tunnel to the Jetson dropped), the request is paused and asked again every PAUSE_S (2 minutes), for at most
  GIVE_UP_S (20 minutes) in a row; then the run stops. Every pause is a line in the progress file, with the request
  it hit, so an answer whose window spans a pause can be read with that in mind (its latency and joules include the
  pause). A refusal (401, 403, 404, 424, a redirect to the login page) or any other 4xx (a request the server will
  not take, such as one past the student's context) is not an outage and is not retried here.
- Progress. Each answer's record is appended to the progress file as it is made, so a run that stops early still
  leaves every answer it got; `summarize --progress <file>` reads the arms from that file with the loop's own
  summary (the verdict math and the TT6 finding).

Nothing here touches the pod or the box: the closed loop only asks the two chat APIs and reads the power log.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab import label as label_mod
from lab import routing_closed_loop as loop

PAUSE_S = 120.0
GIVE_UP_S = 1200.0


class GaveUp(RuntimeError):
    """An outage outlasted GIVE_UP_S."""


class _Progress:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.current: str | None = None

    def write(self, row: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(row) + "\n")


def outage_retries(original: Callable[..., Any], progress: _Progress, *, pause_s: float = PAUSE_S,
                   give_up_s: float = GIVE_UP_S, sleep: Callable[[float], None] = time.sleep,
                   clock: Callable[[], float] = time.time) -> Callable[..., Any]:
    """Teacher._with_retries with the run's outage rule around it."""

    def with_pauses(self: label_mod.Teacher, once: Callable[[], Any]) -> Any:
        first = None
        while True:
            try:
                return original(self, once)
            except label_mod.TeacherError as err:
                if err.fatal or (err.status is not None and err.status < 500):
                    raise  # a refusal, or a request the server will not take (a 400): not an outage
                now = clock()
                first = now if first is None else first
                progress.write({"pause": True, "at": now, "request": progress.current, "node": self.node,
                                "status": err.status, "error": str(err)[:200], "outage_s": now - first})
                if now - first >= give_up_s:
                    raise GaveUp(f"no usable answer from {self.node} for {now - first:.0f} s") from None
                sleep(pause_s)

    return with_pauses


def install(progress: _Progress, **kwargs: Any) -> None:
    original_retries = label_mod.Teacher._with_retries
    label_mod.Teacher._with_retries = outage_retries(original_retries, progress, **kwargs)  # type: ignore[method-assign]
    original_ask = loop._ask

    def ask(index: int, arm: str, *args: Any, **kw: Any) -> loop.Record:
        progress.current = f"{index:03d}-{arm}"
        record = original_ask(index, arm, *args, **kw)
        progress.write(record.row())
        return record

    loop._ask = ask  # type: ignore[assignment]


def read_progress(path: Path) -> tuple[list[loop.Record], list[loop.Record], list[dict[str, Any]]]:
    records, probes, pauses = [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("pause"):
            pauses.append(row)
            continue
        record = loop.Record(**row)
        (probes if record.arm == "probe" else records).append(record)
    return records, probes, pauses


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(prog="python -m lab.t11_offbox_run")
    p.add_argument("--progress", required=True)
    p.add_argument("--summarize", action="store_true", help="only summarize the progress file")
    split = argv.index("--") if "--" in argv else len(argv)
    args = p.parse_args(argv[:split])
    progress = _Progress(Path(args.progress))
    if args.summarize:
        records, probes, pauses = read_progress(progress.path)
        print(json.dumps({"records": len(records), "probes": len(probes), "pauses": len(pauses),
                          **loop.summarize(records, probes)}, indent=2, default=str))
        return 0
    install(progress)
    try:
        answer = loop.main(argv[split + 1:])
    except GaveUp as err:
        print(json.dumps({"ok": False, "outcome": "gave_up", "error": str(err), "progress": str(progress.path)}))
        return 1
    print(json.dumps(answer))
    return 0 if answer.get("ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())
