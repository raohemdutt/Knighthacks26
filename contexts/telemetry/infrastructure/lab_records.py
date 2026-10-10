"""The lab's recorded numbers as stats figures, read through lab/machine: the metrics file and the ledger. Reads only.

    records = LabRecords()                       # lab/refine/metrics.jsonl and the ledger in LAB_STATE_DIR or lab/state
    found, problems = records.find([("teacher.decode_tok_s", None), ("detector.fps", "jetson")])
    found[("teacher.decode_tok_s", None)]        # [the metrics file's newest row, the ledger's newest row], as Figures

A metrics-file row is cited as `metrics.jsonl:<line>` and a ledger row as `ledger seq <n>`, each with its own measured
time and low-confidence flag. The newest row of a metric is the one `lab.machine.metrics.latest` and
`lab.machine.ledger.latest_metric` would give: latest ts, the later line winning a tie. A node narrows the rows to that
node; None takes any node (the teacher and the trainer have run on several pods).

Nothing here creates or writes a file: a missing metrics file or ledger is no rows. A file that cannot be read (corrupt,
or locked by a writer for longer than `lock_timeout`), and a checkout where `lab` cannot be imported (the gateway run
from outside the repo root), come back as problems, so the endpoint can say why a figure is null instead of failing.
"""

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from contexts.telemetry.domain.stats import Figure

Wanted = tuple[str, str | None]  # (metric, node or None for any node)


class LabRecords:
    def __init__(self, metrics_path: str | Path | None = None, state_dir: str | Path | None = None,
                 lock_timeout: float = 1.0) -> None:
        self.metrics_path = None if metrics_path is None else Path(metrics_path)  # None: the committed file
        self.state_dir = state_dir  # None: LAB_STATE_DIR when set, else lab/state
        self.lock_timeout = lock_timeout

    def find(self, wanted: Sequence[Wanted]) -> tuple[dict[Wanted, list[Figure]], list[str]]:
        """For each (metric, node): the metrics file's newest row, then the ledger's, as Figures (either may be
        absent); and what could not be read, in words."""
        found: dict[Wanted, list[Figure]] = {w: [] for w in wanted}
        try:
            from lab.machine import ledger, metrics
        except ImportError as exc:
            return found, [f"the lab records could not be imported here ({exc}); the gateway reads them when it runs "
                           "from the repo root"]
        problems = []
        path = self.metrics_path or metrics.DEFAULT_PATH
        try:
            rows = _numbered(metrics.load(path, lock_timeout=self.lock_timeout), path)
        except (OSError, ValueError) as exc:
            rows = []
            problems.append(f"the lab metrics file could not be read: {type(exc).__name__}: {exc}")
        for metric, node in wanted:
            row = _newest(rows, metric, node)
            if row is not None:
                line, r = row
                found[(metric, node)].append(Figure(value=r["value"], unit=r["unit"], node=r["node"], ts=_time(r["ts"]),
                                                    source=f"{path.name}:{line}", low_confidence=r["low_confidence"]))
        try:
            for metric, node in wanted:
                r = ledger.latest_metric(metric, node, state_dir=self.state_dir)
                if r is not None:
                    found[(metric, node)].append(Figure(value=r["value"], unit=r["unit"] or "", node=r["node"],
                                                        ts=_time(r["ts"]), source=f"ledger seq {r['seq']}",
                                                        low_confidence=r["low_confidence"]))
        except (OSError, ValueError) as exc:
            problems.append(f"the lab ledger could not be read: {type(exc).__name__}: {exc}")
        return found, problems


def _numbered(rows: list[dict], path: Path) -> list[tuple[int, dict]]:
    """Each loaded row with its line number in the file. load() skips blank lines and the file only grows at its end,
    so the n-th non-blank line holds the n-th row even if a row was appended after the load."""
    if not rows:
        return []
    lines = [n for n, raw in enumerate(path.read_bytes().split(b"\n"), start=1) if raw.strip()]
    return list(zip(lines, rows))


def _newest(rows: list[tuple[int, dict]], metric: str, node: str | None) -> tuple[int, dict] | None:
    best = None
    for line, r in rows:
        if r["metric"] == metric and node in (None, r["node"]):
            if best is None or _time(r["ts"]) >= _time(best[1]["ts"]):  # a later line wins a tie
                best = (line, r)
    return best


def _time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))  # Python 3.10 (the Jetson) does not read a bare Z
