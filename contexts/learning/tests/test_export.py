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


def test_a_check_in_in_the_log_is_never_part_of_the_export(tmp_path):
    """The check-in (`day.checkin.raised`) is box-only: the export is built from Memory's camera episodes and reads
    neither the event log nor the Day loop, so a check-in sitting in the log leaves nothing in the dataset."""
    import re
    from pathlib import Path

    from bus import EventBus
    from contracts import parse_event
    from contracts.day import BOX_ONLY_EVENTS
    from tests.test_contract_samples import SAMPLES  # invented values: nothing measured on anybody

    events = [parse_event({"type": t, "ts": NOW.isoformat(), **SAMPLES[t]}) for t in sorted(BOX_ONLY_EVENTS)]
    raised = next(e for e in events if e.type == "day.checkin.raised")
    bus = EventBus(tmp_path / "events")
    asyncio.run(bus.publish_all(events))
    assert len(list(bus.read_log())) == len(BOX_ONLY_EVENTS) == 4  # the check-in, its answer, a yes to an optional event, a look forgotten: on the box

    out = tmp_path / "datasets"
    out.mkdir()
    published = []
    memory = FakeMemory([episode("e1", "cam-desk", 2)])
    approvals = Approvals({"export_training_data": DatasetExporter(memory, lambda ref: b"jpeg", published.append, out_dir=out)})
    _, proposal = teach(SPEC, NOW)
    approvals.on_proposed(proposal)
    [done] = asyncio.run(approvals.on_decided(ApprovalDecided(ts=NOW, action_id=proposal.action_id, decision="approve")))
    assert done.status == "done", done.result
    files = [p for p in out.rglob("*") if p.is_file()]
    written = b"".join(p.read_bytes() for p in files).decode() + published[0].model_dump_json() + proposal.model_dump_json()
    said = [s.summary for s in raised.signals] + [s.detail for s in raised.signals] + [s.text for s in raised.suggestions] + [raised.text]
    assert not [line for line in said if line in written]
    assert not [word for word in ("strained", "Heart rate", "heart rate", "slept", "drink", "checkin", "check-in") if word in written]
    assert "friends-dinner" not in written and "nearby:" not in written  # nor which optional event was said yes to

    # structural: nothing in the learning context imports the Day loop or names one of its events
    root = Path(__file__).resolve().parents[3]
    for path in (root / "contexts" / "learning").rglob("*.py"):
        if "tests" not in path.parts:
            text = path.read_text(encoding="utf-8")
            assert not re.search(r"contracts\.day|contexts\.day|[\"']day\.|checkin|Checkin", text), path


def test_an_accepted_optional_event_is_never_part_of_the_export(tmp_path):
    """An optional event the person said yes to is a block of the day's plan (in the log, on the box), and the friends
    its calendar item names are in Memory's items. The export reads camera episodes only: neither reaches a dataset."""
    from bus import EventBus
    from contexts.day.tests.test_optional_events import DINNER, FRIENDS, accepted
    from contexts.play.domain.places import PlaceSet

    b = accepted(PlaceSet.load("config/places"), tmp_path / "box")
    assert b.optional.accepted == {DINNER}
    logged = " ".join(e.model_dump_json() for e in EventBus(tmp_path / "box" / "events").read_log())
    assert "optional:friends-dinner" in logged and "day.optional.answered" in logged  # it is all in the log, on the box
    assert not [n for n in FRIENDS if n in logged]  # ... and even there no plan, card or narration names a friend
    items = list(b.loop.memory.items(NOW - timedelta(days=1), NOW + timedelta(days=5)))
    assert any("with Riley, Sam, Alex" in (i.summary or "") for i in items)  # Memory is where the names are

    class EpisodesOnly(FakeMemory):
        def items(self, start, end, kind=None):
            raise AssertionError("the export must not read connector items")

    out = tmp_path / "datasets"
    out.mkdir()
    published = []
    memory = EpisodesOnly([episode("e1", "cam-desk", 2)], items=items)
    approvals = Approvals({"export_training_data": DatasetExporter(memory, lambda ref: b"jpeg", published.append, out_dir=out)})
    _, proposal = teach(SPEC, NOW)
    approvals.on_proposed(proposal)
    [done] = asyncio.run(approvals.on_decided(ApprovalDecided(ts=NOW, action_id=proposal.action_id, decision="approve")))
    assert done.status == "done", done.result
    files = [p for p in out.rglob("*") if p.is_file()]
    written = b"".join(p.read_bytes() for p in files).decode() + published[0].model_dump_json() + proposal.model_dump_json()
    assert not [word for word in (*FRIENDS, "dinner", "nearby:", "optional:", "63 South", "GroupMe") if word in written]
