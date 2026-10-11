"""GET /api/learning/lab and /api/learning/lab/{run_id}: the Learning lab screen's data.

    app.include_router(lab_router(lab_view, current_model=lambda: {"version": ..., "source": ...}))

GET /api/learning/lab
    {"runs": [{run_id, task, status, started, ended, scores, model_version, verdict, run_status}],   newest first
     "sample": false, "sample_reason": null,            kept for the screen; the committed sample is never served
     "real_runs": int,                                  runs in this box's own log that are shown
     "current_model": {version, source} | null,         what the box serves now (never from the sample)
     "box_now": {model_version, since, source, reason}, the newest install the box's log holds, or why it holds none
     "measured": [number],                              the trainer numbers measured on our hardware (lab metrics file)
     "suggested_events": [...],                         the additive events this view can use and the lab does not send yet
     "trainer_runs": [...]}                             only when the trainer's run folders were read from disk (lab_runs.py)

GET /api/learning/lab/{run_id}
    the run's story (contexts/learning/application/lab_view.py), plus "sources": every number with where it came
    from, and "measured". 404 when no run has that id. While the run is going (L30): run_status "running" (then
    "finished" or "failed"), steps_so_far, and stage_strip, the four stages {labeling, training, packaging, offered}
    with their state, start, end and seconds, lit from the stage lines the relay posts (lab_import kind=stage).

GET /api/learning/lab/compare[?a=<run_id>&b=<run_id>]
    two runs side by side (contexts/learning/application/lab_compare.py): the comparison document's shape (stamp, t0,
    total_s, acts, rule, idle, clock_offsets_s, pod_d_split, duo_over_solo, go_s) plus `series` (the three power
    traces on one time axis) and {a, b, ratios (b over a), ratio_keys, runs, asked, missing}; without a or b, the
    newest live loop's solo run against its pooled one. Only what this box holds (<data_dir>/lab_loops, posted by the
    relay) is served, never a fixture. 404 with a plain message when either run is not on the box yet: poll.

Only what this box recorded is served. Nothing from fixtures/ ever reaches a screen: not the committed sample run
(fixtures/learning/sample_run; `?sample=1` is accepted and ignored), not a live-loop record kept under fixtures/ or
marked `fixture`/`sample`, and not a run a committed rehearsal names (lab_loops.fixture_run_ids), even when its files
were pushed into this box's log. With no recorded run the list is empty and a run's page is a 404.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from contexts.learning.application.lab_bench import chip_of
from contexts.learning.application.lab_compare import CompareMissing, compare_runs, env_dirs, load_loops
from contexts.learning.application.lab_runs import sync as sync_trainer_runs
from contexts.learning.application.lab_loops import is_fixture_run
from contexts.learning.application.lab_rehearsal import rehearsal_compare
from contexts.learning.application.lab_sample import SAMPLE_DIR  # the old default of sample_dir, now unused
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
    loops_from = loop_dirs if loop_dirs is not None else None   # None: the gateway's data dir, read per request

    def shown() -> list[str]:
        return [r for r in view.run_ids if not is_fixture_run(r)]

    @router.get("/api/learning/lab")
    def lab(sample: bool = False) -> dict:  # noqa: ARG001 - `sample` is accepted and ignored: never the sample run
        trainer_runs = sync_trainer_runs(view)  # the trainer's own files, when a run folder is attached (lab_runs.py)
        model = current_model() if current_model else None
        if model is None and view.deployments:  # the newest swap the log holds
            last = view.deployments[-1]
            model = {"version": last.model_version, "source": f"learning.model.deployed {last.event_id}"}
        runs = [r for r in view.runs() if not is_fixture_run(r["run_id"])]
        return {"runs": runs, "sample": False, "sample_reason": None,
                "real_runs": len(shown()), "current_model": model, "measured": measured(metrics_file),
                "box_now": view.box_now(),  # the newest install the box's log holds, or why it holds none
                "suggested_events": SUGGESTED_EVENTS,
                **({"trainer_runs": trainer_runs} if trainer_runs else {})}  # run folders read from disk, when there are any

    @router.get("/api/learning/lab/compare")  # before /{run_id}, which would take this path
    def compare(a: str | None = None, b: str | None = None) -> dict:
        sync_trainer_runs(view)
        try:
            return compare_runs(load_loops(loops_from if loops_from is not None else env_dirs()), view, a, b)
        except CompareMissing as err:
            raise HTTPException(404, str(err)) from None

    @router.get("/api/learning/lab/rehearsal")  # before /{run_id} too; a second source, never mixed into /compare
    def rehearsal() -> dict:
        """The lab's recorded rehearsal of alone against with a friend, labelled as such (application/lab_rehearsal.py)."""
        answer = rehearsal_compare()
        if answer is None:
            raise HTTPException(404, "the lab's recorded rehearsal is not in this checkout")
        return answer

    @router.get("/api/learning/lab/{run_id}")
    def run(run_id: str, sample: bool = False) -> dict:  # noqa: ARG001 - accepted and ignored
        sync_trainer_runs(view)
        story = None if is_fixture_run(run_id) else view.story(run_id)
        if story is None:
            raise HTTPException(404, f"no training run {run_id!r} on this box")
        return {**story, "sample_reason": None, "sources": collect_numbers(story), "measured": measured(metrics_file)}

    return router
