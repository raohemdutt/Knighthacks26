import asyncio
import json
from datetime import UTC, datetime, timedelta

from contexts.agent.application.memory_port import FakeMemory
from contexts.learning.application.export import DatasetExporter, teach
from contexts.workspace.domain.approvals import Approvals
from contracts.experience import ApprovalDecided
from contracts.learning import DatasetExample, DatasetManifest, TaskSpec
from contracts.memory import Episode

NOW = datetime(2026, 10, 11, 9, 0, tzinfo=UTC)
SPEC = TaskSpec(task="desk-objects", instruction="List small objects on the desk as JSON.", camera="cam-desk")


def episode(id, camera, hours_ago):
    start = NOW - timedelta(hours=hours_ago)
    return Episode(id=id, camera_id=camera, start=start, captions=[f"caption {id}"], keyframe_refs=[f"frames:{camera}:{id}"])


def setup(tmp_path):
    memory = FakeMemory([episode("e1", "cam-desk", 2), episode("e2", "cam-door", 2), episode("e3", "cam-desk", 30)])
    published = []
    exporter = DatasetExporter(memory, load_keyframe=lambda ref: b"jpeg", publish=published.append, out_dir=tmp_path)
    return Approvals({"export_training_data": exporter}), published


def test_nothing_is_written_until_approved(tmp_path):
    approvals, published = setup(tmp_path)
    _, proposal = teach(SPEC, NOW)
    approvals.on_proposed(proposal)
    assert list(tmp_path.iterdir()) == [] and published == []

    decision = ApprovalDecided(ts=NOW + timedelta(seconds=30), action_id=proposal.action_id, decision="deny")
    assert asyncio.run(approvals.on_decided(decision))[0].status == "denied"
    assert list(tmp_path.iterdir()) == []


def test_approved_export_writes_the_lab_format(tmp_path):
    approvals, published = setup(tmp_path)
    created, proposal = teach(SPEC, NOW)
    assert created.task == "desk-objects"
    approvals.on_proposed(proposal)
    [done] = asyncio.run(approvals.on_decided(ApprovalDecided(ts=NOW, action_id=proposal.action_id, decision="approve")))
    assert done.status == "done", done.result

    [exported] = published
    root = tmp_path / exported.dataset_id
    manifest = DatasetManifest.model_validate_json((root / "manifest.json").read_text())
    assert manifest.action_id == proposal.action_id and manifest.examples == 1
    [line] = (root / "examples.jsonl").read_text().splitlines()
    ex = DatasetExample.model_validate(json.loads(line))
    assert ex.frame_ref == "frames:cam-desk:e1" and ex.context == "caption e1"
    assert (root / ex.image).read_bytes() == b"jpeg"


def test_thumbs_up_examples_pick_look_alikes_and_keep_verdicts():
    from contexts.learning.application.export import select_keyframes
    from contracts.learning import TeachExample

    eps = [
        Episode(id="keys1", camera_id="cam-desk", start=NOW, labels=["person", "keys"], keyframe_refs=["k1"]),
        Episode(id="keys2", camera_id="cam-desk", start=NOW, labels=["keys", "cup"], keyframe_refs=["k2"]),
        Episode(id="empty", camera_id="cam-desk", start=NOW, labels=["chair"], keyframe_refs=["c1"]),
        Episode(id="bad", camera_id="cam-desk", start=NOW, labels=["laptop"], keyframe_refs=["b1"]),
    ]
    spec = SPEC.model_copy(update={"examples": [TeachExample(frame_ref="k1", verdict="up"), TeachExample(frame_ref="b1", verdict="down")]})
    picked = [(ref, verdict) for _, _, ref, verdict in select_keyframes(spec, eps)]
    assert picked[:2] == [("k1", "up"), ("b1", "down")]  # the person's picks first, with verdicts
    assert ("k2", None) in picked  # shares "keys" with the liked example
    assert ("c1", None) not in picked  # looks like nothing that was liked
    assert len(select_keyframes(SPEC, eps)) == 4  # no examples: export the whole window as before
