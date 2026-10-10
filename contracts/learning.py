from typing import Literal

from contracts.events import TypedEvent, register


@register
class TaskCreated(TypedEvent):
    type: Literal["learning.task.created"] = "learning.task.created"
    task: str  # slug: "desk-objects"
    instruction: str
    output_schema: dict = {}


@register
class DatasetExported(TypedEvent):
    type: Literal["learning.dataset.exported"] = "learning.dataset.exported"
    task: str
    dataset_id: str
    examples: int
    action_id: str  # the approval that allowed it


@register
class RunProgress(TypedEvent):
    type: Literal["learning.run.progress"] = "learning.run.progress"
    task: str
    run_id: str
    stage: Literal["labeling", "training", "eval", "packaging", "deploying", "done", "failed"]
    progress: float  # 0..1 within the stage
    loss: float | None = None
    message: str | None = None


@register
class RunEvaluated(TypedEvent):
    type: Literal["learning.run.evaluated"] = "learning.run.evaluated"
    task: str
    run_id: str
    scores: dict[str, float]  # {"current": 0.41, "candidate": 0.78, "teacher": 0.86}
    decision: Literal["deploy", "reject"]
    model_version: str


@register
class ModelDeployed(TypedEvent):
    type: Literal["learning.model.deployed"] = "learning.model.deployed"
    model_version: str
    previous_version: str | None = None
