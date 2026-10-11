"""lab/fixture_dataset_v3.py: tool-calling v3, the scenes replayed from a box's own event log.

A small log is written here (eight desk episodes over about seven hours, three objects, and a scripted test run
that must be left out), so nothing reads the box. The agent's own code is the judge, as in the v2 tests: every
expected call is checked against the agent's pre-fetch at the example's own time against the memory as it stood.
"""
from __future__ import annotations

import gzip
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from contexts.agent.application.qa import SleepSchedule, time_window
from lab import dataset
from lab import fixture_dataset as fx
from lab import fixture_dataset_v2 as v2
from lab import fixture_dataset_v3 as v3
from lab.fixture_dataset import Trace

UTC = timezone.utc
START = datetime(2026, 10, 10, 14, 0, 7, 123456, tzinfo=UTC)
OBJECTS = ("laptop", "bottle", "keys")
SPLIT_OPTIONS = {"held_out_episodes": 2, "validation_episodes": 2, "held_out_objects": ("keys",),
                 "validation_objects": ("bottle",), "min_held_out": 1}


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LAB_ACTOR", "opus-test")
    monkeypatch.setenv("LAB_TIER", "opus")
    monkeypatch.delenv("LAB_TASK", raising=False)


def _iso(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def small_log() -> list[dict]:
    """Eight episodes 50 minutes apart, each about two minutes of observations with a caption naming an object, and
    a scripted run (whole-second times, a camera the box does not have, a fake frame) in the middle."""
    rows: list[dict] = []
    number = 0

    def add(row: dict) -> None:
        nonlocal number
        number += 1
        rows.append({"event_id": f"e{number:05d}", "version": 1, **row})

    for episode in range(8):
        begin = START + timedelta(minutes=50 * episode)
        thing = OBJECTS[episode % 3]
        for second in range(0, 120, 4):
            ts = begin + timedelta(seconds=second, microseconds=1000 * episode + 17)
            ref = f"frames:cam-desk:{int(ts.timestamp() * 1000)}"
            add({"type": "perception.observation.recorded", "ts": _iso(ts), "camera_id": "cam-desk",
                 "tracks": [{"label": "person", "confidence": 0.6, "box": {"x1": 0.1, "y1": 0.1, "x2": 0.5, "y2": 0.9},
                             "track_id": episode}], "frame_ref": ref if second == 0 else None})
            if second == 40:
                add({"type": "perception.scene.described", "ts": _iso(ts), "camera_id": "cam-desk",
                     "caption": f"A person at the desk with a {thing}. On the desk: {thing} (left of the person).",
                     "objects": [{"label": thing, "position": "left of the person"}], "zone_id": "desk",
                     "frame_ref": ref, "model_version": "atlas-base"})
        if episode == 3:   # the scripted run
            fake = begin + timedelta(minutes=20)
            for second in range(0, 20, 5):
                add({"type": "perception.observation.recorded", "ts": (fake + timedelta(seconds=second)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"), "camera_id": "cam-door", "tracks": [{"label": "person", "confidence": 0.9,
                    "box": {"x1": 0.7, "y1": 0.3, "x2": 0.9, "y2": 0.9}, "track_id": 99}], "frame_ref": None})
            add({"type": "perception.scene.described", "ts": (fake + timedelta(seconds=9)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "camera_id": "cam-desk", "caption": "On the desk: wallet (next to the mug).",
                 "objects": [{"label": "wallet", "position": "next to the mug"}], "frame_ref": "frames:cam-desk:12",
                 "model_version": "atlas-base"})
        add({"type": "perception.observation.recorded", "ts": _iso(begin + timedelta(minutes=5)),
             "camera_id": "cam-desk", "tracks": [], "frame_ref": None})
    rows.append({"event_id": "e-unrelated", "version": 1, "type": "telemetry.jetson.power", "ts": _iso(START)})
    return rows


@pytest.fixture(scope="module")
def log(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("box") / "events.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("".join(json.dumps(row) + "\n" for row in small_log()))
    return path


@pytest.fixture(scope="module")
def built(log, tmp_path_factory):
    return v3.build(tmp_path_factory.mktemp("v3"), log, **SPLIT_OPTIONS)


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_the_scripted_run_is_left_out_and_nothing_else(log):
    kept, dropped = v3.read_events(log)
    assert dropped == {"perception.observation.recorded": 4, "perception.scene.described": 1}
    assert all(row["camera_id"] == "cam-desk" for row in kept if "camera_id" in row)
    assert not any(row["type"] == "telemetry.jetson.power" for row in kept)
    assert v3.injected({"ts": "2026-10-10T18:36:21Z", "frame_ref": None})
    assert not v3.injected({"ts": "2026-10-10T21:50:10Z", "frame_ref": "frames:cam-desk:1791669010000"})
    assert not v3.injected({"ts": "2026-10-10T21:50:10.000004Z", "frame_ref": None})


def test_a_replay_lands_on_the_store_the_box_would_have(log, tmp_path):
    from contexts.memory.application.projection import MemoryProjection
    from contexts.memory.infrastructure.store import MemoryStore
    from contracts.events import parse_event
    box = MemoryStore(tmp_path / "memory.sqlite")       # the box: every event, live, in log order
    projection = MemoryProjection(box)
    handlers = {"perception.observation.recorded": projection.on_observation,
                "perception.scene.described": projection.on_scene}
    for row in small_log():
        if row["type"] in handlers:
            handlers[row["type"]](parse_event(row))
    box.db.commit()
    box.db.close()
    checked = v3.verify_snapshot(log, tmp_path / "memory.sqlite")
    assert all(checked["same"].values()), checked


def test_the_split_is_by_whole_group_and_by_time(built):
    expected = {row["id"]: row for row in jsonl(built.root / fx.EXPECTED)}
    examples = {row["id"]: row for row in jsonl(built.root / dataset.EXAMPLES)}
    listed = v2.load_validation(built.root)
    split = dataset.load_split(built.root, require_images=False, min_held_out=1)
    part_of: dict[str, set[str]] = defaultdict(set)
    clocks: dict[str, list[datetime]] = defaultdict(list)
    for part, ids in (("train", listed["train"]), ("validation", listed["validation"]),
                      ("held_out", [example.id for example in split.held_out])):
        for example_id in ids:
            part_of[expected[example_id]["group"]].add(part)
            clocks[part].append(datetime.fromisoformat(examples[example_id]["ts"]))
    assert all(len(parts) == 1 for parts in part_of.values())
    assert max(clocks["train"]) < min(clocks["validation"]) <= max(clocks["validation"]) < min(clocks["held_out"])
    assert built.parts["held_out"]["groups"] == 3 and built.parts["validation"]["groups"] == 3
    groups = {expected[i]["group"] for i in listed["train"]}
    assert "last_seen-laptop" in groups and not any("keys" in g or "bottle" in g for g in groups)
    held = {expected[example.id]["group"] for example in split.held_out}
    assert "last_seen-keys" in held


def test_every_call_is_the_agents_own_at_the_examples_clock_and_holds_its_scene(built):
    sleep = SleepSchedule.load(fx.AGENT_CONFIG)
    expected = {row["id"]: row for row in jsonl(built.root / fx.EXPECTED)}
    for example in jsonl(built.root / dataset.EXAMPLES):
        row = expected[example["id"]]
        now = datetime.fromisoformat(example["ts"])
        call = row["label"]["tool_calls"][0]
        assert len(row["label"]["tool_calls"]) == 1
        if row["group"].startswith("last_seen-"):
            assert call == {"name": "last_seen", "arguments": {"label": row["group"].split("-", 1)[1]}}
            continue
        start, end = time_window(example["context"].lower(), now, sleep)
        assert call == {"name": "episodes_between", "arguments": {"start": start.isoformat(), "end": end.isoformat()}}
        scene_ms = int(row["group"].rsplit("-", 1)[1])
        assert start.timestamp() * 1000 <= scene_ms < end.timestamp() * 1000
        assert now.timestamp() * 1000 > scene_ms                     # the box had the scene at that clock


def test_evidence_is_what_the_box_knew_at_the_clock(built):
    """A last_seen answer never names a sighting the box made after the example's clock."""
    expected = {row["id"]: row for row in jsonl(built.root / fx.EXPECTED)}
    for example in jsonl(built.root / dataset.EXAMPLES):
        row = expected[example["id"]]
        for ref in row["cited"]:
            if ref.startswith("frames:cam-desk:"):
                assert int(ref.rsplit(":", 1)[1]) <= datetime.fromisoformat(example["ts"]).timestamp() * 1000


def test_reserved_things_stay_off_train_and_validation(log, tmp_path):
    options = {**SPLIT_OPTIONS, "held_out_objects": (), "validation_objects": ()}
    with pytest.raises(fx.BuildError, match="earlier dataset holds out"):
        v3.build(tmp_path, log, **options)          # the keys would land on the train side


def test_clip_shortens_results_but_never_the_call():
    long = "\n".join(f"Sat {h}:00 PM hour: " + "word " * 120 for h in range(1, 12))
    trace = Trace(tool_calls=({"name": "episodes_between", "arguments": {"start": "a", "end": "b"}},),
                  tool_results=(long,), cited=("ep",), answer="x")
    clipped, changed = v3.clip(trace)
    assert changed and clipped.tool_calls == trace.tool_calls
    text = clipped.tool_results[0]
    assert len(text) <= v3.RESULT_CHARS + 40 and text.endswith("more lines)")
    assert all(len(line) <= v3.LINE_CHARS for line in text.split("\n"))
    short = Trace(tool_calls=trace.tool_calls, tool_results=("one line",), cited=(), answer="I didn't see that.")
    assert v3.clip(short) == (short, False)


def test_the_build_is_the_same_bytes_twice_and_frozen(log, built, tmp_path):
    again = v3.build(tmp_path, log, **SPLIT_OPTIONS)
    assert again.sha256 == built.sha256
    with pytest.raises(fx.BuildError, match="frozen"):
        v3.build(tmp_path, log, **SPLIT_OPTIONS)


def test_the_scorer_gives_v3_examples_their_own_clock(built):
    from lab import tool_call_agreement as tca  # noqa: F401  (the scorer imports system_for from v2)
    split = dataset.load_split(built.root, require_images=False, min_held_out=1)
    manifest = split.data.manifest
    assert manifest.action_id == v3.ACTION_ID
    for example in split.data.examples[:5]:
        assert v2.system_for(manifest, example) == fx.instruction(example.ts)


def test_teacher_lines_join_the_train_side_marked(log, tmp_path):
    line = {"id": "t-0000000001", "group": "teacher-laptop", "question": "When did I last see my laptop?",
            "ts": "2026-10-10T14:30:00-04:00", "outcome": "kept", "checkpoint": "x",
            "tool_calls": [{"name": "last_seen", "arguments": {"label": "laptop"}, "result": "laptop last seen"}],
            "answer": "Your laptop was last seen on the desk."}
    bad = {**line, "id": "t-0000000002", "outcome": "no_call", "tool_calls": []}
    path = tmp_path / "teacher.jsonl"
    path.write_text(json.dumps(line) + "\n" + json.dumps(bad) + "\n", encoding="utf-8")
    made = v3.build(tmp_path / "out", log, teacher=path, **SPLIT_OPTIONS)
    facts = json.loads((made.root / v2.BUILD).read_text(encoding="utf-8"))
    assert facts["teacher_labeled"] == 1 and facts["teacher_outcomes"] == {"kept": 1, "no_call": 1}
    assert facts["rule_labeled"] == made.examples - 1
    listed = v2.load_validation(made.root)
    assert "t-0000000001" in listed["train"] and "t-0000000002" not in listed["train"]
    source = json.loads((made.root / v3.SOURCE).read_text(encoding="utf-8"))
    assert source["teacher_ids"] == ["t-0000000001"]
