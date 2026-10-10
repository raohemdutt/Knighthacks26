"""What the learning loop is doing, for the phone's Teach screen (X10): one entry per task, folded from `learning.*`.

    board = LearningBoard()
    board.subscribe(bus)          # replay=True: the log rebuilds it
    board.snapshot()              # {"tasks": [...], "deployments": [...]}

A task goes: waiting for approval -> exported -> labeling -> training -> eval -> packaging -> deploying -> done
(or failed). `scores` and `decision` come from the eval; deployments are the model versions the box switched to.
"""

from datetime import datetime
from typing import Any

from bus import EventBus
from contracts.agent import ActionProposed
from contracts.learning import DatasetExported, ModelDeployed, RunEvaluated, RunProgress, TaskCreated
from contracts.workspace import ActionCompleted


class LearningBoard:
    def __init__(self) -> None:
        self.tasks: dict[str, dict[str, Any]] = {}
        self.deployments: list[dict[str, Any]] = []
        self.export_actions: dict[str, str] = {}  # action_id of an export approval -> task slug

    def subscribe(self, bus: EventBus) -> None:
        bus.subscribe("learning.task", self.on_created, replay=True)
        bus.subscribe("learning.dataset", self.on_exported, replay=True)
        bus.subscribe("learning.run", self.on_run, replay=True)
        bus.subscribe("learning.model", self.on_deployed, replay=True)
        bus.subscribe("agent.action.proposed", self.on_proposed, replay=True)
        bus.subscribe("workspace.action.completed", self.on_completed, replay=True)

    def _task(self, slug: str, ts: datetime) -> dict[str, Any]:
        return self.tasks.setdefault(slug, {"task": slug, "instruction": "", "stage": "waiting for approval", "progress": 0.0,
                                            "created": ts, "updated": ts})

    def on_created(self, e: TaskCreated) -> None:
        t = self._task(e.task, e.ts)
        t.update(instruction=e.instruction, updated=e.ts)

    def on_proposed(self, e: ActionProposed) -> None:
        if e.kind == "export_training_data":
            self.export_actions[e.action_id] = e.target

    def on_completed(self, e: ActionCompleted) -> None:
        """A denied or expired export approval ends the task; without this it would wait for ever."""
        task = self.export_actions.get(e.action_id)
        t = self.tasks.get(task) if task else None
        if t and t["stage"] == "waiting for approval" and e.status in ("denied", "expired"):
            t.update(stage=e.status, progress=0.0, updated=e.ts)

    def on_exported(self, e: DatasetExported) -> None:
        t = self._task(e.task, e.ts)
        t.update(stage="exported", progress=1.0, dataset_id=e.dataset_id, examples=e.examples, updated=e.ts)

    def on_run(self, e: RunProgress | RunEvaluated) -> None:
        t = self._task(e.task, e.ts)
        if isinstance(e, RunEvaluated):
            t.update(scores=e.scores, decision=e.decision, model_version=e.model_version, run_id=e.run_id, updated=e.ts)
        else:
            t.update(stage=e.stage, progress=e.progress, loss=e.loss, message=e.message, run_id=e.run_id, updated=e.ts)

    def on_deployed(self, e: ModelDeployed) -> None:
        self.deployments.append({"model_version": e.model_version, "previous_version": e.previous_version, "ts": e.ts})
        for t in self.tasks.values():  # the evaluated candidate that got deployed
            if t.get("model_version") == e.model_version:
                t.update(stage="done", progress=1.0, deployed=True, updated=e.ts)

    def snapshot(self) -> dict[str, Any]:
        tasks = sorted(self.tasks.values(), key=lambda t: t["updated"], reverse=True)
        return {"tasks": tasks, "deployments": list(reversed(self.deployments))}
