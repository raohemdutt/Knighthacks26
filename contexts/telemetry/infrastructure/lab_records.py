"""The lab's recorded numbers as stats figures, read through lab/machine: the metrics file and the ledger. Reads only.

    records = LabRecords()                       # lab/refine/metrics.jsonl and the ledger in LAB_STATE_DIR or lab/state
    records = LabRecords(imported_dir=data_dir / "lab_state")   # + the ledger copy the lab pushed to the box
    found, problems = records.find([("teacher.decode_tok_s", None), ("detector.fps", "jetson")])
    found[("teacher.decode_tok_s", None)]        # [the metrics file's newest row, the ledger's newest row], as Figures

A metrics-file row is cited as `metrics.jsonl:<line>`, with the line number `lab.machine.metrics.load_numbered` gives,
and a ledger row as `ledger seq <n>`, each with its own measured time and low-confidence flag. The newest row of a
metric is the one `lab.machine.metrics.latest` and `lab.machine.ledger.latest_metric` would give: latest ts, the later
line winning a tie. A node narrows the rows to that node; None takes any node (the teacher and the trainer have run on
several pods), except a row marked `other_chip: true` (a reference on another chip, such as R57's A100 on Colab),
which answers only when its node is named.

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
                 lock_timeout: float = 1.0, imported_dir: str | Path | None = None) -> None:
        self.metrics_path = None if metrics_path is None else Path(metrics_path)  # None: the committed file
        self.state_dir = state_dir  # None: LAB_STATE_DIR when set, else lab/state
        self.lock_timeout = lock_timeout
        # The ledger copy the lab pushed to this box (POST /api/learning/lab/import?kind=ledger keeps it in
        # <data_dir>/lab_state); None reads only the state dir. Its rows are cited as `pushed ledger seq <n>`.
        self.imported_dir = None if imported_dir is None else Path(imported_dir)

    def find(self, wanted: Sequence[Wanted]) -> tuple[dict[Wanted, list[Figure]], list[str]]:
        """For each (metric, node): the metrics file's newest row, then the ledger's, then the pushed ledger's, as
        Figures (any may be absent); and what could not be read, in words."""
        found: dict[Wanted, list[Figure]] = {w: [] for w in wanted}
        try:
            from lab.machine import ledger, metrics
        except ImportError as exc:
            return found, [f"the lab records could not be imported here ({exc}); the gateway reads them when it runs "
                           "from the repo root"]
        problems = []
        path = self.metrics_path or metrics.DEFAULT_PATH
        try:
            rows = metrics.load_numbered(path, lock_timeout=self.lock_timeout)
        except (OSError, ValueError) as exc:
            rows = []
            problems.append(f"the lab metrics file could not be read: {type(exc).__name__}: {exc}")
        for metric, node in wanted:
            row = _newest(rows, metric, node)
            if row is not None:
                line, r = row
                found[(metric, node)].append(Figure(value=r["value"], unit=r["unit"], node=r["node"], ts=_time(r["ts"]),
                                                    source=f"{path.name}:{line}", low_confidence=r["low_confidence"]))
        ledgers = [(self.state_dir, "ledger", "the lab ledger")]
        if self.imported_dir is not None:
            ledgers.append((self.imported_dir, "pushed ledger", "the ledger the lab pushed to this box"))
        for state_dir, cite, named in ledgers:
            try:
                for metric, node in wanted:
                    r = ledger.latest_metric(metric, node, state_dir=state_dir)
                    if r is not None:
                        found[(metric, node)].append(Figure(
                            value=r["value"], unit=r["unit"] or "", node=r["node"], ts=_time(r["ts"]),
                            source=f"{cite} seq {r['seq']}", low_confidence=r["low_confidence"]))
            except (OSError, ValueError) as exc:
                problems.append(f"{named} could not be read: {type(exc).__name__}: {exc}")
        return found, problems


def _newest(rows: list[tuple[int, dict]], metric: str, node: str | None) -> tuple[int, dict] | None:
    best = None
    for line, r in rows:
        if r["metric"] == metric and node in (None, r["node"]):
            if node is None and (r.get("conditions") or {}).get("other_chip") is True:
                continue  # a reference on another chip (R57's A100) answers only when its node is asked for by name
            if best is None or _time(r["ts"]) >= _time(best[1]["ts"]):  # a later line wins a tie
                best = (line, r)
    return best


def _time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))  # Python 3.10 (the Jetson) does not read a bare Z
