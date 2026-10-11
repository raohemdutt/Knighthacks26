"""Teach task + approved export (L1, box side).

Teaching a task proposes an `export_training_data` action; nothing leaves the
box until the user approves it on the phone. The approved export writes a
dataset in the format from contracts.learning (blurred keyframes + captions
only) under data/exports/, where the lab picks it up.
"""

import json
import logging
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

from contexts.agent.application.memory_port import MemoryReader
from contexts.workspace.domain.approvals import ActionRequest
from contracts.agent import ActionProposed
from contracts.learning import DatasetExample, DatasetExported, DatasetManifest, TaskCreated, TaskSpec

log = logging.getLogger("atlas.learning")

DEFAULT_LOOKBACK = timedelta(hours=24)
MAX_EXAMPLES = 300


def select_keyframes(spec: TaskSpec, episodes: list) -> list[tuple]:
    """Which keyframes to export, as (episode, index, frame_ref, verdict).

    With thumbs-up examples, keep the episodes that look like them: they share a
    detector label with an up-voted episode (a cheap, explainable match a small
    box can afford). The person's own picks always go in, carrying their verdict,
    so the lab can use them as gold labels; thumbs-down frames are kept as
    negatives but their look-alikes are not pulled in."""
    verdicts = {e.frame_ref: e.verdict for e in spec.examples}
    by_ref = {ref: ep for ep in episodes for ref in ep.keyframe_refs}
    liked_labels = {lab for ref, v in verdicts.items() if v == "up" and ref in by_ref for lab in by_ref[ref].labels}
    picked = []
    for ep in episodes:
        if liked_labels and not liked_labels & set(ep.labels) and not any(r in verdicts for r in ep.keyframe_refs):
            continue  # doesn't look like anything the person liked
        for i, ref in enumerate(ep.keyframe_refs):
            picked.append((ep, i, ref, verdicts.get(ref)))
    picked.sort(key=lambda t: t[3] is None)  # the person's own picks first, so the cap never drops them
    return picked


def span(d: timedelta) -> str:
    """A lookback in words: "24 hours", "3 days", "90 minutes"."""
    minutes = round(d.total_seconds() / 60)
    if minutes % (24 * 60) == 0 and minutes > 24 * 60:
        return f"{minutes // (24 * 60)} days"
    if minutes % 60 == 0:
        return f"{minutes // 60} hour" + ("s" if minutes != 60 else "")
    return f"{minutes} minutes"


def teach(spec: TaskSpec, now: datetime, lookback: timedelta = DEFAULT_LOOKBACK) -> tuple[TaskCreated, ActionProposed]:
    """A new task and the export request the user must approve."""
    created = TaskCreated(ts=now, task=spec.task, instruction=spec.instruction, output_schema=spec.output_schema)
    proposal = ActionProposed(
        ts=now,
        action_id=uuid4().hex,
        kind="export_training_data",
        target=spec.task,
        reason=f"Send blurred keyframes and captions from the last {span(lookback)} to teach '{spec.task}'",
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
        refused: type[Exception] | tuple[type[Exception], ...] = (),
    ) -> None:
        self.memory = memory
        self.load_keyframe = load_keyframe  # frame_ref -> blurred JPEG bytes
        self.publish = publish
        self.out_dir = out_dir
        self.clock = clock
        self.refused = refused  # what load_keyframe raises for a picture that may not leave: skipped, counted, never written

    def __call__(self, req: ActionRequest) -> str:
        spec = TaskSpec.model_validate(req.params["spec"])
        start = datetime.fromisoformat(req.params["start"])
        end = datetime.fromisoformat(req.params["end"])
        dataset_id = f"{spec.task}-{uuid4().hex[:8]}"
        root = self.out_dir / dataset_id
        (root / "images").mkdir(parents=True)

        examples: list[DatasetExample] = []
        held_back = 0
        for ep, i, ref, verdict in select_keyframes(spec, self.memory.episodes(start, end, camera=spec.camera, label=spec.label)):
            if len(examples) >= MAX_EXAMPLES:
                break
            image = f"images/{len(examples):04d}.jpg"
            try:
                jpeg = self.load_keyframe(ref)
            except self.refused as exc:
                held_back += 1
                log.warning("export %s: %s left out: %s", dataset_id, ref, exc)
                continue
            (root / image).write_bytes(jpeg)
            context = ep.captions[i] if i < len(ep.captions) else ep.summary
            examples.append(DatasetExample(id=f"{ep.id}-{i}", image=image, frame_ref=ref, ts=ep.start, context=context,
                                           verdict=verdict))

        with (root / "examples.jsonl").open("w") as f:
            for ex in examples:
                f.write(ex.model_dump_json() + "\n")
        now = self.clock() if self.clock else end
        manifest = DatasetManifest(dataset_id=dataset_id, spec=spec, created=now, examples=len(examples), action_id=req.id)
        (root / "manifest.json").write_text(json.dumps(manifest.model_dump(mode="json"), indent=2))

        self.publish(DatasetExported(ts=now, task=spec.task, dataset_id=dataset_id, examples=len(examples), action_id=req.id))
        result = f"exported {len(examples)} examples as {dataset_id}"
        if held_back:
            result += f"; {held_back} picture{'s' if held_back != 1 else ''} held back (faces could not be checked)"
        return result
