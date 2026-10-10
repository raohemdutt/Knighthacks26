"""Before and after eval scores per deployed model, folded from the learning events and rebuilt by replaying the log.

    evals = EvalScores()
    evals.subscribe(bus)        # replay=True: learning.run.evaluated and learning.model.deployed
    evals.snapshot()            # {"models": [EvalRow, newest deployment first], "reason": None or why there are none}
    evals.evaluated_runs        # run ids that have an evaluation, so their training has ended

A deployed model is a version named by a `learning.model.deployed` (the learning loop, or jetson/atlas_swap.py through
POST /api/learning/deployed). Its scores come from the newest `learning.run.evaluated` of the same version: `current`
is the before (the model serving when the candidate was judged), `candidate` the after. A version deployed without an
evaluation in the log (a rollback to the base model) is listed with null scores and the reason. Candidates that were
evaluated but never deployed are not listed.
"""

from typing import Any

from bus import EventBus
from contexts.telemetry.domain.stats import EvalRow
from contracts.learning import ModelDeployed, RunEvaluated


class EvalScores:
    def __init__(self) -> None:
        self._evaluated: dict[str, RunEvaluated] = {}  # model_version -> its newest evaluation
        self._deployed: dict[str, ModelDeployed] = {}  # model_version -> its newest deployment
        self.evaluated_runs: set[str] = set()

    def subscribe(self, bus: EventBus) -> None:
        bus.subscribe(RunEvaluated.TYPE, self.on_evaluated, replay=True)
        bus.subscribe(ModelDeployed.TYPE, self.on_deployed, replay=True)

    def on_evaluated(self, e: RunEvaluated) -> None:
        self.evaluated_runs.add(e.run_id)
        have = self._evaluated.get(e.model_version)
        if have is None or e.ts >= have.ts:
            self._evaluated[e.model_version] = e

    def on_deployed(self, e: ModelDeployed) -> None:
        have = self._deployed.get(e.model_version)
        if have is None or e.ts >= have.ts:
            self._deployed[e.model_version] = e

    def rows(self) -> list[EvalRow]:
        out = []
        for d in sorted(self._deployed.values(), key=lambda d: d.ts, reverse=True):
            e = self._evaluated.get(d.model_version)
            row = EvalRow(model_version=d.model_version, deployed_ts=d.ts, previous_version=d.previous_version)
            if e is None:
                row.reason = f"no learning.run.evaluated for {d.model_version} in the event log"
            else:
                row = row.model_copy(update={
                    "task": e.task, "run_id": e.run_id, "decision": e.decision, "before": e.scores.get("current"),
                    "after": e.scores.get("candidate"), "teacher": e.scores.get("teacher"), "ts": e.ts,
                    "source": f"learning.run.evaluated {e.event_id}"})
                absent = [k for k in ("current", "candidate") if k not in e.scores]
                if absent:
                    row.reason = "the evaluation carries no " + " and no ".join(f"'{k}'" for k in absent) + " score"
            out.append(row)
        return out

    def snapshot(self) -> dict[str, Any]:
        rows = self.rows()
        return {"models": [r.model_dump(mode="json") for r in rows],
                "reason": None if rows else "no learning.model.deployed event in the event log yet"}
