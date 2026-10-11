"""The lab's recorded rehearsal of "alone, then with a friend", as a second and clearly named source.

    rehearsal_compare()      what GET /api/learning/lab/rehearsal serves, or None when the record is not in the checkout

GET /api/learning/lab/compare serves only what this box recorded (lab_compare.py, and tests/test_lab_no_fixtures.py
holds it to that): a box with no comparison of its own answers 404, and that rule is not touched here. The owner asked
that the Together tab then still show the one real measurement the team has, the lab's rehearsal of the loop on
Sun 2026-10-11 (fixtures/learning/live-loop-2026-10-11: two real runs on the pods with real power traces), as long as
the screen says where it is from. So this is a different route with a different name, its answer carries `record`
(`kind: "rehearsal"`, the label the screen prints, the file), and nothing else reads it: the run list, a run's story
and the compare route never see these runs.

The numbers are the comparison document's own, put through the same compare_runs as a box's record, so the two
sources cannot drift apart in shape or in arithmetic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from contexts.learning.application import lab_loops
from contexts.learning.application.lab_compare import CompareMissing, Loop, _jsonl, compare_runs

REHEARSAL_DIR = Path(lab_loops.FIXTURES) / "learning" / "live-loop-2026-10-11"
REHEARSAL_REL = "fixtures/learning/live-loop-2026-10-11"
LABEL = "from the lab's recorded run, not this box"


class RehearsalLoop(Loop):
    """The committed record, read as it is. Unlike a box's Loop it is allowed to live under fixtures/, and its runs
    keep their ids: that is the whole difference, and it is confined to this class."""

    def __init__(self, folder: Path) -> None:
        self.folder, self.root = folder, folder.parent
        self.source = f"{REHEARSAL_REL}/{lab_loops.COMPARISON}"
        self.comparison = json.loads((folder / lab_loops.COMPARISON).read_text(encoding="utf-8"))
        if not isinstance(self.comparison.get("acts"), dict):
            raise ValueError("no acts")
        self.stamp = str(self.comparison.get("stamp") or folder.name)
        self.stages = _jsonl(folder / "stages.jsonl")
        self.offsets = self.comparison.get("clock_offsets_s") or {}
        self.t0 = self._t0()

    def acts(self) -> dict[str, dict[str, Any]]:
        return {act: row for act, row in self.comparison["acts"].items() if isinstance(row, dict) and row.get("run_id")}


def rehearsal_compare(folder: str | Path | None = REHEARSAL_DIR) -> dict[str, Any] | None:
    """The rehearsal's two runs side by side, marked as the rehearsal; None when the record cannot be read."""
    path = Path(folder) if folder else None
    if path is None or not (path / lab_loops.COMPARISON).is_file():
        return None
    try:
        answer = compare_runs([RehearsalLoop(path)], None)
    except (CompareMissing, OSError, ValueError, KeyError, TypeError):
        return None
    return {**answer, "runs": [],  # nothing to pick between: this is one record, not the box's runs
            "record": {"kind": "rehearsal", "label": LABEL, "file": f"{REHEARSAL_REL}/{lab_loops.COMPARISON}",
                       "stamp": answer.get("stamp")}}
