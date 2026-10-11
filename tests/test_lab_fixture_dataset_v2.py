"""lab/fixture_dataset_v2.py: tool-calling dataset v2 (reference labels, many objects and windows, a clock per example).

The agent's own code is the judge, as in test_lab_fixture_dataset.py: every expected call is cross-checked against the
agent's pre-fetch at the example's own time, and every row's prompt against the scorer's. No model is asked.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from contexts.agent.application.qa import SleepSchedule
from lab import dataset
from lab import fixture_dataset as fx
from lab import fixture_dataset_v2 as v2
from lab import format_traces as ft
from lab import tool_call_agreement as tca

REPO_ROOT = Path(__file__).resolve().parents[1]
OLD = REPO_ROOT / "fixtures" / "learning" / "tool-calling-fixture"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LAB_ACTOR", "opus-test")
    monkeypatch.setenv("LAB_TIER", "opus")
    monkeypatch.delenv("LAB_TASK", raising=False)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return v2.build(tmp_path_factory.mktemp("v2"))


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_the_build_meets_the_size_floor_and_splits_by_whole_group(built):
    assert built.groups >= 100
    assert 600 <= built.rows[v2.ROWS_TRAIN] <= 1500
    assert built.parts["held_out"]["groups"] >= 8 and built.parts["validation"]["groups"] >= 8
    expected = {row["id"]: row for row in jsonl(built.root / fx.EXPECTED)}
    listed = v2.load_validation(built.root)
    split = dataset.load_split(built.root, require_images=False)
    part_of: dict[str, set[str]] = defaultdict(set)
    for part, ids in (("train", listed["train"]), ("validation", listed["validation"]),
                      ("held_out", [example.id for example in split.held_out])):
        for example_id in ids:
            part_of[expected[example_id]["group"]].add(part)
    assert all(len(parts) == 1 for parts in part_of.values())            # no group is in two parts
    assert sorted(listed["train"] + listed["validation"]) == sorted(example.id for example in split.train)


def test_the_held_out_side_shows_objects_and_window_lengths_no_other_part_shows(built):
    expected = {row["id"]: row for row in jsonl(built.root / fx.EXPECTED)}
    split = dataset.load_split(built.root, require_images=False)

    def things(examples) -> tuple[set[str], set[timedelta]]:
        labels, lengths = set(), set()
        for example in examples:
            call = expected[example.id]["label"]["tool_calls"][0]
            if call["name"] == "last_seen":
                labels.add(call["arguments"]["label"])
            else:
                lengths.add(datetime.fromisoformat(call["arguments"]["end"])
                            - datetime.fromisoformat(call["arguments"]["start"]))
        return labels, lengths

    held_labels, held_lengths = things(split.held_out)
    train_labels, train_lengths = things(split.train)
    assert held_labels == set(v2.HELD_OUT_ITEMS) and not held_labels & train_labels
    assert held_lengths == {timedelta(minutes=m) for m in v2.HELD_OUT_WINDOWS} and not held_lengths & train_lengths


def test_nothing_of_the_first_fixtures_held_out_side_is_in_any_part(built):
    old_split = dataset.load_split(OLD, require_images=False)
    old_questions = {fx._normal(example.context) for example in old_split.held_out}
    old_calls = {json.dumps(row["label"]["tool_calls"], sort_keys=True) for row in jsonl(OLD / fx.EXPECTED)
                 if row["id"] in {example.id for example in old_split.held_out}}
    for row in jsonl(built.root / fx.EXPECTED):
        assert fx._normal(row["question"]) not in old_questions
        assert json.dumps(row["label"]["tool_calls"], sort_keys=True) not in old_calls
        call = row["label"]["tool_calls"][0]
        assert call["arguments"].get("label") not in v2.RESERVED["items"]
        if call["name"] == "episodes_between" and "minutes" in row["group"]:
            assert row["group"] != "episodes_between-60m"
    assert v2._reserved_problems([("train", "g", "Where are my keys?", {"name": "last_seen", "label": "keys"})])
    assert v2._reserved_problems([("train", "g", "What happened overnight?", {"name": "episodes_between"})])


def test_every_expected_call_is_what_the_agent_does_at_the_examples_own_time(built):
    memory, _ = v2.night_memory()
    sleep = SleepSchedule.load(fx.AGENT_CONFIG)
    data = dataset.load(built.root, require_images=False)
    expected = fx.load_expected(built.root)
    for example in data.examples[::7]:
        trace = fx.reference_trace(example.context, memory, example.ts, sleep)
        assert list(trace.tool_calls) == [{"name": call.name, "arguments": call.arguments}
                                          for call in expected[example.id].label.tool_calls]
        assert json.loads(example.target) == json.loads(expected[example.id].label.model_dump_json())


def test_each_example_has_its_own_clock_in_both_styles_and_its_row_carries_it(built):
    data = dataset.load(built.root, require_images=False)
    offsets = {example.ts.utcoffset() for example in data.examples}
    assert offsets == {timedelta(0), timedelta(hours=-4)}
    assert len({example.ts for example in data.examples}) > 0.8 * len(data.examples)
    by_id = {example.id: example for example in data.examples}
    rows = jsonl(built.root / v2.ROWS_TRAIN) + jsonl(built.root / v2.ROWS_VALIDATION)
    assert len(rows) == built.rows[v2.ROWS_TRAIN] + built.rows[v2.ROWS_VALIDATION]
    for row in rows[::5]:
        example = by_id[row["id"]]
        system = v2.system_for(data.manifest, example)
        assert f"The time now is {example.ts.isoformat()}." in system
        # the scorer's prompt is the row up to its first assistant span, with this example's own time
        assert tca.prompt_for(system, example.context) == row["text"][:row["spans"][0][0]]
        calls = ft.parse_assistant_turns(row["text"])[0].calls
        assert [{"name": call.name, "arguments": call.arguments} for call in calls] == \
            json.loads(example.target)["tool_calls"]


def test_validation_rows_are_never_training_rows_and_held_out_examples_get_no_row(built):
    train = {row["id"] for row in jsonl(built.root / v2.ROWS_TRAIN)}
    validation = {row["id"] for row in jsonl(built.root / v2.ROWS_VALIDATION)}
    split = dataset.load_split(built.root, require_images=False)
    held_out = {example.id for example in split.held_out}
    assert not train & validation and not (train | validation) & held_out
    listed = v2.load_validation(built.root)
    assert train == set(listed["train"]) and validation == set(listed["validation"])


def test_the_build_is_the_same_bytes_twice_and_refuses_a_folder_that_exists(built, tmp_path):
    again = v2.build(tmp_path / "again")
    assert again.sha256 == built.sha256
    with pytest.raises(fx.BuildError) as refused:
        v2.build(tmp_path / "again")
    assert refused.value.outcome == "refused"


def test_another_datasets_system_prompt_is_its_one_instruction():
    data = dataset.load(OLD, require_images=False)
    assert v2.system_for(data.manifest, data.examples[0]) == data.manifest.spec.instruction


def test_the_entry_point_writes_one_ledger_row_and_the_split_holds_up(tmp_path):
    answer = v2.main(["--out", str(tmp_path / "out")])
    assert (answer["ok"], answer["outcome"], answer["tokenizer"]) == (True, "built", "not checked")
    split = dataset.load_split(tmp_path / "out" / v2.DATASET_ID, require_images=False)
    assert split.sha256 == answer["refs"]["split_sha256"]
    assert split.meets_deploy_floor
