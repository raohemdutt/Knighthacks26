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


def test_the_approval_card_says_the_lookback_in_words():
    """Round 2 (L1): the card read "from the last 1 day, 0:00:00"."""
    from datetime import UTC, datetime, timedelta

    from contexts.learning.application.export import span, teach
    from contracts.learning import TaskSpec

    _, proposal = teach(TaskSpec(task="desk-objects", instruction="List objects."), datetime(2026, 10, 11, tzinfo=UTC))
    assert "from the last 24 hours to teach 'desk-objects'" in proposal.reason
    assert [span(timedelta(hours=1)), span(timedelta(days=3)), span(timedelta(minutes=90))] == ["1 hour", "3 days", "90 minutes"]


def test_a_watch_night_in_memory_is_never_part_of_the_export(tmp_path):
    """K9: the dataset is keyframes and captions of camera episodes. Connector items, health ones too, are not read."""
    from contracts.memory import Item

    night = Item(connector="health", kind="health", external_id="sleep:2026-10-11", title="Sleep 7 h 28 min",
                 when=NOW - timedelta(hours=2), summary="asleep 7 h 28 min · in bed 7 h 46 min · resting HR 53 bpm · HRV 67 ms")

    class EpisodesOnly(FakeMemory):
        def items(self, start, end, kind=None):
            raise AssertionError("the export must not read connector items")

    memory = EpisodesOnly([episode("e1", "cam-desk", 2)], items=[night])
    published = []
    approvals = Approvals({"export_training_data": DatasetExporter(memory, lambda ref: b"jpeg", published.append, out_dir=tmp_path)})
    _, proposal = teach(SPEC, NOW)
    assert "Sleep" not in proposal.model_dump_json()
    approvals.on_proposed(proposal)
    [done] = asyncio.run(approvals.on_decided(ApprovalDecided(ts=NOW, action_id=proposal.action_id, decision="approve")))
    assert done.status == "done", done.result
    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert sorted(p.name for p in files) == ["0000.jpg", "examples.jsonl", "manifest.json"]
    written = b"".join(p.read_bytes() for p in files).decode() + published[0].model_dump_json()
    assert not [word for word in ("Sleep", "asleep", "bpm", "HRV", "health", "53 ") if word in written]
