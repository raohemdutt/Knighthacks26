"""The fake chip client for lab/capacity.py: answers with a chip sheet on disk and touches nothing else.

By default the sheet is the R33 sample, lab/refine/research/kernels/data/chipsheet-sample.json, whose memory
section also answers the memory read. Tests pass a sheet of their own (sample shape or the shape
lab/kernels/chipsheet/chipsheet.py writes) and, when that sheet has no memory section, the memory to report.
Every call is recorded in .calls so a test can check what the probe asked for.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lab import placement


class FakeChipClient:
    """A chip client with no chip behind it: the sheet is a file, the memory comes from it or from memory=."""

    def __init__(self, sheet: Path | str | None = None, memory: Mapping[str, Any] | None = None) -> None:
        self.path = Path(sheet) if sheet is not None else placement.SAMPLE_CHIPSHEET
        self.given_memory = dict(memory) if memory is not None else None
        self.calls: list[tuple[Any, ...]] = []

    @property
    def name(self) -> str:
        return f"fake: {placement.relative(self.path)}"

    def sheet(self, quick: bool = True) -> Path:
        self.calls.append(("sheet", quick))
        return self.path

    def memory(self) -> dict[str, Any]:
        """free_gib and visible_gib as numbers or {"value": ...}, and where they came from."""
        self.calls.append(("memory",))
        if self.given_memory is not None:
            return {"source": "fake: memory given by the test", **self.given_memory}
        sheet = json.loads(self.path.read_text(encoding="utf-8"))
        memory = sheet.get("memory") if isinstance(sheet.get("memory"), Mapping) else {}
        return {"free_gib": memory.get("free_gib"), "visible_gib": memory.get("visible_gib"),
                "source": f"{placement.relative(self.path)}: memory",
                "kind": "sample" if sheet.get("sample") is True else "recorded"}
