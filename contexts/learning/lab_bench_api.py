"""The Learning lab's other three tabs: speed and energy, the routing what-if, and the experiment timeline.

    app.include_router(lab_bench_router())      # BEFORE lab_router: its /api/learning/lab/{run_id} would take these paths

GET  /api/learning/lab/bench
    the lab's benchmark rows (lab/refine/metrics.jsonl), grouped: `sweeps` (a metric family measured over batch sizes),
    `student` (the Jetson's figures and what is derived from them), `power` (idle and serving watts per node),
    `series` (every row, grouped generically) and `not_charted` (what the screen draws nothing for).
    contexts/learning/application/lab_bench.py
POST /api/learning/lab/routing/simulate   {preset?, mix?, rpm?, minutes?, tokens?, policy?, student_basis?, seed?}
    the routing what-if: the real routing hook and its two policies run on a snapshot built from the recorded rows.
    400 with the reason when an input is out of range. contexts/learning/application/lab_routing.py
GET  /api/learning/lab/experiments
    experiments, refinement passes and benchmark sessions as cards, newest first.
    contexts/learning/application/lab_experiments.py

All three read files only: no model call, no pod, no hub. The metrics file is re-read when it changes on disk.
"""

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException

from contexts.learning.application.lab_bench import METRICS_FILE, bench, read_rows
from contexts.learning.application.lab_experiments import REFINE_DIR, timeline
from contexts.learning.application.lab_routing import SimError, simulate


def lab_bench_router(metrics_file: str | Path | None = METRICS_FILE, state_dir: str | Path | None = None,
                     refine_dir: str | Path = REFINE_DIR) -> APIRouter:
    router = APIRouter()
    cache: dict[str, Any] = {"stamp": None, "rows": [], "skipped": []}

    def rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        path = Path(metrics_file) if metrics_file else None
        try:
            stat = path.stat() if path else None
        except OSError:
            stat = None
        stamp = (stat.st_mtime, stat.st_size) if stat else None
        if stamp != cache["stamp"] or stamp is None:
            found, skipped = read_rows(path) if stamp else ([], [])
            cache.update(stamp=stamp, rows=found, skipped=skipped)
        return cache["rows"], cache["skipped"]

    @router.get("/api/learning/lab/bench")
    def lab_bench() -> dict:
        return bench(*rows())

    @router.post("/api/learning/lab/routing/simulate")
    def routing_simulate(params: dict[str, Any] | None = Body(default=None)) -> dict:
        try:
            return simulate(params, rows()[0])
        except SimError as err:
            raise HTTPException(400, str(err)) from None

    @router.get("/api/learning/lab/experiments")
    def lab_experiments() -> dict:
        return timeline(rows()[0], state_dir=state_dir, refine_dir=refine_dir)

    return router
