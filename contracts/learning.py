from datetime import datetime
from typing import Literal

from pydantic import BaseModel

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


# --- box <-> lab file formats (the only coupling between Atlas and lab/) ---
#
# Dataset export, written by the box after an approved export:
#   <dataset_id>/manifest.json   DatasetManifest
#   <dataset_id>/examples.jsonl  one DatasetExample per line
#   <dataset_id>/images/*.jpg    blurred keyframes only
# The lab's label.py fills `target` with the teacher's output.
#
# Model package, written by lab/package.sh, installed under data/models/<version>/:
#   manifest.json  ModelManifest
#   <gguf>, <mmproj>


class TeachExample(BaseModel):
    """A keyframe the person marked while teaching: thumbs up = like this, thumbs down = not this."""

    frame_ref: str
    verdict: Literal["up", "down"]


class TaskSpec(BaseModel):
    task: str  # slug: "desk-objects"
    instruction: str  # prompt given to teacher and student for each image
    output_schema: dict = {}  # JSON schema of the expected answer, {} for free text
    camera: str | None = None  # restrict examples to one camera
    label: str | None = None  # restrict to episodes with this detector label
    examples: list[TeachExample] = []  # a few thumbs up/down picks from the timeline


class DatasetExample(BaseModel):
    id: str
    image: str  # path relative to the dataset dir: "images/0001.jpg"
    frame_ref: str
    ts: datetime
    context: str | None = None  # the box's own caption, for the teacher to see
    target: str | None = None  # filled by the teacher in the lab
    verdict: Literal["up", "down"] | None = None  # user thumbs, if any


class DatasetManifest(BaseModel):
    dataset_id: str
    spec: TaskSpec
    created: datetime
    examples: int
    action_id: str  # the approval that allowed this export


class ModelManifest(BaseModel):
    version: str  # "atlas-v3-desk-objects"
    base: str  # "Qwen/Qwen3.5-4B"
    task: str | None = None
    gguf: str  # file name inside the package dir
    mmproj: str | None = None
    sha256: dict[str, str] = {}  # file name -> checksum
    scores: dict[str, float] = {}
    created: datetime
