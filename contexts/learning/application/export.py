"""Teach task + approved export (L1, box side).

Teaching a task proposes an `export_training_data` action; nothing leaves the
box until the user approves it on the phone. The approved export writes a
dataset in the format from contracts.learning (blurred keyframes + captions
only) under data/exports/, where the lab picks it up.
"""

import json
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

from contexts.agent.application.memory_port import MemoryReader
from contexts.workspace.domain.approvals import ActionRequest
from contracts.agent import ActionProposed
from contracts.learning import DatasetExample, DatasetExported, DatasetManifest, TaskCreated, TaskSpec

DEFAULT_LOOKBACK = timedelta(hours=24)
MAX_EXAMPLES = 300


def teach(spec: TaskSpec, now: datetime, lookback: timedelta = DEFAULT_LOOKBACK) -> tuple[TaskCreated, ActionProposed]:
    """A new task and the export request the user must approve."""
    created = TaskCreated(ts=now, task=spec.task, instruction=spec.instruction, output_schema=spec.output_schema)
    proposal = ActionProposed(
        ts=now,
        action_id=uuid4().hex,
        kind="export_training_data",
        target=spec.task,
        reason=f"Send blurred keyframes and captions from the last {lookback} to teach '{spec.task}'",
        params={
            "spec": spec.model_dump(),
            "start": (now - lookback).isoformat(),
            "end": now.isoformat(),
        },
    )
    return created, proposal


class DatasetExporter:
    """Workspace executor for `export_training_data`. Runs only after approval."""

    def __init__(
        self,
        memory: MemoryReader,
        load_keyframe: Callable[[str], bytes],
        publish: Callable[[DatasetExported], None],
        out_dir: Path = Path("data/exports"),
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.memory = memory
        self.load_keyframe = load_keyframe  # frame_ref -> blurred JPEG bytes
        self.publish = publish
        self.out_dir = out_dir
        self.clock = clock

    def __call__(self, req: ActionRequest) -> str:
        spec = TaskSpec.model_validate(req.params["spec"])
        start = datetime.fromisoformat(req.params["start"])
        end = datetime.fromisoformat(req.params["end"])
        dataset_id = f"{spec.task}-{uuid4().hex[:8]}"
        root = self.out_dir / dataset_id
        (root / "images").mkdir(parents=True)

        examples: list[DatasetExample] = []
        for ep in self.memory.episodes(start, end, camera=spec.camera, label=spec.label):
            for i, ref in enumerate(ep.keyframe_refs):
                if len(examples) >= MAX_EXAMPLES:
                    break
                image = f"images/{len(examples):04d}.jpg"
                (root / image).write_bytes(self.load_keyframe(ref))
                context = ep.captions[i] if i < len(ep.captions) else ep.summary
                examples.append(DatasetExample(id=f"{ep.id}-{i}", image=image, frame_ref=ref, ts=ep.start, context=context))

        with (root / "examples.jsonl").open("w") as f:
            for ex in examples:
                f.write(ex.model_dump_json() + "\n")
        now = self.clock() if self.clock else end
        manifest = DatasetManifest(dataset_id=dataset_id, spec=spec, created=now, examples=len(examples), action_id=req.id)
        (root / "manifest.json").write_text(json.dumps(manifest.model_dump(mode="json"), indent=2))

        self.publish(DatasetExported(ts=now, task=spec.task, dataset_id=dataset_id, examples=len(examples), action_id=req.id))
        return f"exported {len(examples)} examples as {dataset_id}"
