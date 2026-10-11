"""Building the stats GET /runtime/stats serves, for the composition root.

    stats = build_runtime_stats(atlas.bus, atlas.telemetry)      # before the bus replays the log
    app.include_router(stats_router(stats.snapshot))              # runtime/stats_api.py

It subscribes the eval-score fold and the training-run fold to the bus with replay=True, so it must be built before
`Atlas.start()` replays the log (the gateway builds it in create_app, as it does its other subscriptions). Both are on
whatever the sampler setting: training steps arrive from the lab through contexts/learning/lab_import.py. `telemetry`
is the gateway's TelemetryService, or None when ATLAS_TELEMETRY_SOURCE is not set; only its projection is read, for the
request-window figures. `lab` defaults to the committed lab/refine/metrics.jsonl and the ledger in LAB_STATE_DIR (else
lab/state); `imported_dir` (the gateway passes `<data_dir>/lab_state`) adds the ledger copy the lab pushed to the box,
and is ignored when `lab` is given.
"""

from pathlib import Path

from bus import EventBus
from contexts.telemetry.application.evals import EvalScores
from contexts.telemetry.application.stats import RuntimeStats
from contexts.telemetry.application.train_runs import TrainRuns
from contexts.telemetry.infrastructure.lab_records import LabRecords


def build_runtime_stats(bus: EventBus, telemetry: object | None = None, *,
                        lab: LabRecords | None = None, imported_dir: str | Path | None = None) -> RuntimeStats:
    evals = EvalScores()
    evals.subscribe(bus)
    runs = TrainRuns()
    runs.subscribe(bus)
    if lab is None:
        lab = LabRecords(imported_dir=imported_dir)
    return RuntimeStats(evals, projection=getattr(telemetry, "projection", None), lab=lab, runs=runs)
