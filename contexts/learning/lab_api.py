"""GET /api/learning/lab and /api/learning/lab/{run_id}: the Learning lab screen's data.

    app.include_router(lab_router(lab_view, current_model=lambda: {"version": ..., "source": ...}))

GET /api/learning/lab[?sample=1]
    {"runs": [{run_id, task, status, started, ended, scores, model_version, verdict}],   newest first
     "sample": bool, "sample_reason": str | null,       true when the runs are the committed sample run
     "real_runs": int,                                  runs in this box's own log, whatever is shown
     "current_model": {version, source} | null,         what the box serves now (never from the sample)
     "measured": [number],                              the trainer numbers measured on our hardware (lab metrics file)
     "suggested_events": [...],                         the additive events this view can use and the lab does not send yet
     "trainer_runs": [...]}                             only when the trainer's run folders were read from disk (lab_runs.py)

GET /api/learning/lab/{run_id}[?sample=1]
    the run's story (contexts/learning/application/lab_view.py), plus "sources": every number with where it came
    from, and "measured". 404 when no run has that id.

GET /api/learning/lab/compare[?a=<run_id>&b=<run_id>]
    two runs side by side (contexts/learning/application/lab_compare.py): the comparison document's shape (stamp, t0,
    total_s, acts, rule, idle, clock_offsets_s, pod_d_split, duo_over_solo, go_s) plus `series` (the three power
    traces on one time axis) and {a, b, ratios (b over a), ratio_keys, runs, asked, missing}; without a or b, the
    newest live loop's solo run against its pooled one. Only what this box holds (<data_dir>/lab_loops, posted by the
    relay) is served, never a fixture. 404 with a plain message when either run is not on the box yet: poll.

The sample run is served only when the log has no real run, or when `?sample=1` asks for it. A real run is never
mixed with the sample: a response is one or the other, and says which.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from contexts.learning.application.lab_bench import chip_of
from contexts.learning.application.lab_compare import CompareMissing, compare_runs, default_dirs, load_loops
from contexts.learning.application.lab_runs import sync as sync_trainer_runs
from contexts.learning.application.lab_sample import SAMPLE_DIR, cached_sample
from contexts.learning.application.lab_view import SUGGESTED_EVENTS, LabView, by_file, collect_numbers, num

METRICS_FILE = Path(__file__).resolve().parents[2] / "lab" / "refine" / "metrics.jsonl"
MEASURED = {  # the lab's metric name -> what the screen calls it
    "trainer.samples_per_s": "Trainer speed, 20-step smoke run",
    "trainer.smoke_loss_first": "First loss, 20-step smoke run",
    "trainer.smoke_loss_last": "Last loss, 20-step smoke run",
    "trainer.peak_mem_gb": "Trainer peak memory, 20-step smoke run",
}
NO_RUN = "this box's event log has no training run yet"


def measured(metrics_file: str | Path | None = METRICS_FILE) -> list[dict[str, Any]]:
    """The trainer numbers the lab measured on our hardware, each with its line in the metrics file. [] without it."""
    path = Path(metrics_file) if metrics_file else None
    if path is None or not path.is_file():
        return []
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("metric") in MEASURED and isinstance(row.get("value"), int | float):
            item = num(MEASURED[row["metric"]], row["value"], row.get("unit"), by_file("lab/refine/metrics.jsonl", n),
                       low_confidence=bool(row.get("low_confidence")))
            item.update(metric=row["metric"], node=row.get("node"), measured_at=row.get("ts"), sample=False)
            item["chip"] = chip_of(row)  # the chip the row was measured on, by the row's own fields
            item["other_chip"] = not item["chip"]["current"]  # not the current lab: the R9700s, or a reference machine
            out.append(item)
    return out


def lab_router(view: LabView, current_model: Callable[[], dict | None] | None = None,
               sample_dir: str | Path | None = SAMPLE_DIR, metrics_file: str | Path | None = METRICS_FILE,
               loop_dirs: list[Path] | None = None) -> APIRouter:
    router = APIRouter()
    loops_from = loop_dirs if loop_dirs is not None else default_dirs(None)

    def pick(sample: bool) -> tuple[LabView | None, str | None]:
        """The view to answer from and, when it is the sample, why."""
        if view.run_ids and not sample:
            return view, None
        fixture = cached_sample(str(sample_dir)) if sample_dir else None
        if fixture is None:
            return (None, None) if sample else (view, None)
        return fixture, "asked for with ?sample=1" if sample else NO_RUN

    @router.get("/api/learning/lab")
    def lab(sample: bool = False) -> dict:
        trainer_runs = sync_trainer_runs(view)  # the trainer's own files, when a run folder is attached (lab_runs.py)
        chosen, why = pick(sample)
        model = current_model() if current_model else None
        if model is None and view.deployments:  # the newest swap the log holds
            last = view.deployments[-1]
            model = {"version": last.model_version, "source": f"learning.model.deployed {last.event_id}"}
        return {"runs": chosen.runs() if chosen else [], "sample": bool(chosen and chosen.sample), "sample_reason": why,
                "real_runs": len(view.run_ids), "current_model": model, "measured": measured(metrics_file),
                "suggested_events": SUGGESTED_EVENTS,
                **({"trainer_runs": trainer_runs} if trainer_runs else {})}  # run folders read from disk, when there are any

    @router.get("/api/learning/lab/compare")  # before /{run_id}, which would take this path
    def compare(a: str | None = None, b: str | None = None) -> dict:
        sync_trainer_runs(view)
        try:
            return compare_runs(load_loops(loops_from), view, a, b)
        except CompareMissing as err:
            raise HTTPException(404, str(err)) from None

    @router.get("/api/learning/lab/{run_id}")
    def run(run_id: str, sample: bool = False) -> dict:
        sync_trainer_runs(view)
        chosen, why = pick(sample)
        story = chosen.story(run_id) if chosen else None
        if story is None:
            raise HTTPException(404, f"no training run {run_id!r}")
        return {**story, "sample_reason": why, "sources": collect_numbers(story), "measured": measured(metrics_file)}

    return router
