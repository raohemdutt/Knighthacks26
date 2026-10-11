"""The Learning lab's other three tabs: speed and energy, the routing what-if, and the experiment timeline.

    app.include_router(lab_bench_router())      # BEFORE lab_router: its /api/learning/lab/{run_id} would take these paths

GET  /api/learning/lab/bench
    by default the summary (lab_bench.summary): `summary: true`, every small part of the payload below as it is,
    `sessions` (per benchmark session its chip, engine, date, lever, row count and headline numbers per node and
    batch size: tok/s per stream, tok/s total, TTFT, watts, J per token, confidence), and the heavy keys (sweeps,
    series, not_charted, families) as empty lists named in `omitted`. Computed once per change of the file.
GET  /api/learning/lab/bench?all=1   and   GET /api/learning/lab/bench/rows?all=1
    the whole payload, the shape this route served before (also cached per change of the file):
    the lab's benchmark rows (lab/refine/metrics.jsonl), grouped: `sweeps` (a metric family measured over batch sizes),
    `student` (the Jetson's figures and what is derived from them), `power` (idle and serving watts per node),
    `series` (every row, grouped generically) and `not_charted` (what the screen draws nothing for).
    contexts/learning/application/lab_bench.py
GET  /api/learning/lab/bench/rows?session=<id>
    the whole payload's shape over one session's rows only (the id the summary's `sessions` gives). 404 for an
    unknown session; 400 when neither `session` nor `all=1` is given.
POST /api/learning/lab/routing/simulate   {preset?, mix?, rpm?, minutes?, tokens?, policy?, student_basis?, seed?}
    the routing what-if: the real routing hook and its two policies run on a snapshot built from the recorded rows.
    400 with the reason when an input is out of range. contexts/learning/application/lab_routing.py
GET  /api/learning/lab/experiments
    experiments, refinement passes and benchmark sessions as cards, newest first. `experiments.source` says where
    the ledger was read from (`from`: LAB_STATE_DIR, imported, checkout) and, for the copy the lab pushed
    (`imported_dir`, POST /api/learning/lab/import?kind=ledger), when it arrived and its row count.
    contexts/learning/application/lab_experiments.py
GET  /api/learning/lab/report
    the lab's "Under the hood" report pack (lab/refine/report/under-the-hood.json), read-only: per section its title,
    the sentence the lab wrote, its state, a few key numbers with file and line, and `stale` when the pack is older
    than the metrics file on disk. contexts/learning/application/lab_report.py

All four read files only: no model call, no pod, no hub. The metrics file is re-read when it changes on disk.
"""

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query

from contexts.learning.application.lab_bench import METRICS_FILE, bench, for_session, read_rows, summary
from contexts.learning.application.lab_experiments import REFINE_DIR, timeline
from contexts.learning.application.lab_report import PACK_FILE, under_the_hood
from contexts.learning.application.lab_routing import SimError, simulate


def lab_bench_router(metrics_file: str | Path | None = METRICS_FILE, state_dir: str | Path | None = None,
                     refine_dir: str | Path = REFINE_DIR, imported_dir: str | Path | None = None,
                     report_file: str | Path | None = PACK_FILE) -> APIRouter:
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
            cache.update(stamp=stamp, rows=found, skipped=skipped, made={})
        return cache["rows"], cache["skipped"]

    def made(key: str, build: Any) -> dict:
        """A payload built once per version of the file (the key includes nothing else that changes)."""
        found, skipped = rows()
        store = cache.setdefault("made", {})
        if key not in store:
            if len(store) > 64:  # one entry per session asked for; a long day's worth stays small
                store.clear()
            store[key] = build(found, skipped)
        return store[key]

    def whole() -> dict:
        return made("all", bench)

    @router.get("/api/learning/lab/bench")
    def lab_bench(all: str | None = Query(default=None)) -> dict:  # noqa: A002 (the query name the screen uses)
        if all and all not in ("0", "false"):
            return whole()
        return made("summary", summary)

    @router.get("/api/learning/lab/bench/rows")
    def lab_bench_rows(session: str | None = None, all: str | None = Query(default=None)) -> dict:  # noqa: A002
        if all and all not in ("0", "false"):
            return whole()
        if not session:
            raise HTTPException(400, "give ?session=<id> (an id from the summary's sessions) or ?all=1")
        payload = made("session:" + session, lambda found, skipped: bench(for_session(found, session), []))
        if not payload["rows"]:
            raise HTTPException(404, f"no rows for session {session}")
        return {**payload, "session": session}

    @router.post("/api/learning/lab/routing/simulate")
    def routing_simulate(params: dict[str, Any] | None = Body(default=None)) -> dict:
        try:
            return simulate(params, rows()[0])
        except SimError as err:
            raise HTTPException(400, str(err)) from None

    @router.get("/api/learning/lab/experiments")
    def lab_experiments() -> dict:
        return timeline(rows()[0], state_dir=state_dir, refine_dir=refine_dir, imported_dir=imported_dir)

    @router.get("/api/learning/lab/report")
    def lab_report() -> dict:
        return under_the_hood(report_file, metrics_file)

    return router
