"""lab/fixture_dataset.py: the no-camera tool-calling dataset built from the golden questions and the fixture night.

The agent's own code is the judge of what is expected: tool calls and results are cross-checked against a run of
the real QA tool loop, and the golden answers against the golden eval's own scoring rules. Only the model is
faked, and the model is never asked. Every test builds into a temporary folder and uses a private ledger state dir.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import jsonschema
import pytest

from contexts.agent.application.qa import NO_EVIDENCE, QA, TOOLS, SleepSchedule
from contexts.agent.eval import load_fixture
from contracts.types import ChatResult, ToolCall
from lab import dataset
from lab import fixture_dataset as fx
from lab.machine import ledger, stop
from runtime import FakeRuntime

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = REPO_ROOT / "fixtures" / "golden"
COMMITTED = REPO_ROOT / "fixtures" / "learning" / fx.DATASET_ID
SCRIPT = REPO_ROOT / "lab" / "fixture_dataset.py"
FILES = ("manifest.json", "examples.jsonl", "expected.jsonl", "images/.gitkeep")

OVERNIGHT = "What happened at our table overnight?"
KEYS = "Where did I leave my keys?"
DOOR = "Did anyone come to the door in the last hour?"
LAST_LEFT = "When did someone last leave the table?"
WALLET = "Where is my wallet?"
DUE = "What is due this week?"
PACKAGE = "Did a package arrive?"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LAB_ACTOR", "sonnet-test")
    monkeypatch.setenv("LAB_TIER", "sonnet")
    monkeypatch.delenv("LAB_TASK", raising=False)


@pytest.fixture
def built(tmp_path):
    return fx.build(tmp_path / "out")


@pytest.fixture
def fixture_night():
    memory, now, golden = load_fixture(GOLDEN_DIR)
    return memory, now, golden, SleepSchedule.load(REPO_ROOT / "config" / "agent.yaml")


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_bytes().decode("utf-8").split("\n") if line.strip()]


def untargeted(rows: list[dict]) -> list[dict]:
    return [{k: v for k, v in row.items() if k != "target"} for row in rows]


def checked_against(rows: list[dict]) -> list[dict]:
    """The part of expected.jsonl that labels are checked against."""
    return [{"id": r["id"], "question": r["question"], "origin": r["origin"], "group": r["group"],
             "tool_calls": r["label"]["tool_calls"], "cited": r["cited"], "keywords": r["keywords"],
             "no_evidence": r["no_evidence"]} for r in rows]


def stale_parts(committed: Path, fresh: Path) -> list[str]:
    """What in a committed dataset folder no longer matches a fresh build: the task (manifest.json), the questions
    (examples.jsonl, whatever the teacher has put in `target`), and what labels are checked against in
    expected.jsonl: the tool calls, citations, groups and keywords. The evidence text and the answers are left out,
    so a new caption or time format in the fixtures does not fail a suite that is not about them."""
    parts = []
    if json.loads((committed / "manifest.json").read_text(encoding="utf-8")) != json.loads(
            (fresh / "manifest.json").read_text(encoding="utf-8")):
        parts.append("manifest.json")
    if untargeted(jsonl(committed / "examples.jsonl")) != untargeted(jsonl(fresh / "examples.jsonl")):
        parts.append("examples.jsonl")
    if checked_against(jsonl(committed / "expected.jsonl")) != checked_against(jsonl(fresh / "expected.jsonl")):
        parts.append("expected.jsonl")
    return parts


def only_row():
    found = ledger.read()
    assert len(found) == 1, found
    return found[0]


def call(name: str, **arguments: str) -> dict:
    return {"name": name, "arguments": arguments}


# --- what is in it ------------------------------------------------------------------------------

def test_the_build_is_an_export_the_reader_loads_as_text_only(built):
    data = dataset.load(built.root, require_images=False)
    assert built.root.name == fx.DATASET_ID and data.dataset_id == fx.DATASET_ID
    assert len(data.examples) == data.manifest.examples == built.examples == built.golden + built.variants
    assert (built.golden, len(built.skipped)) == (24, 18)
    assert data.manifest.action_id == fx.FIXTURE_ACTION_ID
    assert data.manifest.spec.task == fx.TASK
    assert data.labeled() == [] and [e.id for e in data.pending()] == [e.id for e in data.examples]
    refused = dataset.check(built.root)   # without the flag the reader refuses a dataset with no pictures
    assert len(refused) == built.examples and all("has no image" in problem for problem in refused)
    assert [p.name for p in (built.root / "images").iterdir()] == [".gitkeep"]


def test_every_example_is_a_text_question_with_the_fixture_time_and_no_target(built, fixture_night):
    _, now, _, _ = fixture_night
    examples = jsonl(built.root / "examples.jsonl")
    assert {(e["image"], e["ts"], e["target"], e["verdict"]) for e in examples} == {("", now.isoformat(), None, None)}
    expected = fx.load_expected(built.root)
    assert [e["id"] for e in examples] == list(expected)
    assert [e["context"] for e in examples] == [expected[e["id"]].question for e in examples]
    assert all(e["frame_ref"] == f"fixture:group:{expected[e['id']].group}" for e in examples)


def test_the_golden_questions_come_first_in_file_order_with_the_exact_tool_calls(built):
    expected = fx.load_expected(built.root)
    golden = [row for row in expected.values() if row.origin == "golden"]
    assert list(expected)[:len(golden)] == [r.id for r in golden] and len(golden) == built.golden == 24
    primary = [(r.id, r.question) for r in golden if r.id.count("-") == 1]
    assert primary == [("golden-01", OVERNIGHT), ("golden-02", KEYS), ("golden-03", DOOR), ("golden-06", WALLET)]
    calls = {r.question: [c.model_dump() for c in r.label.tool_calls] for r in golden}
    assert calls[OVERNIGHT] == [call("episodes_between", start="2026-10-10T20:00:00-04:00", end="2026-10-11T08:00:00-04:00")]
    assert calls[KEYS] == [call("last_seen", label="keys")]
    assert calls[DOOR] == [call("episodes_between", start="2026-10-11T08:30:00-04:00", end="2026-10-11T09:30:00-04:00")]
    assert calls[WALLET] == [call("last_seen", label="wallet")]


def test_the_golden_wordings_are_golden_examples_in_the_groups_of_their_question(built):
    expected = {row.question: row for row in fx.load_expected(built.root).values()}
    for wording, question in [("Where are my keys?", KEYS), ("Have you seen my keys?", KEYS), ("I can't find my keys", KEYS),
                              ("What happened last night?", OVERNIGHT), ("Was anyone at the door recently?", DOOR),
                              ("Where is the wallet now?", WALLET)]:
        assert (expected[wording].origin, expected[wording].group) == ("golden", expected[question].group), wording
        assert expected[wording].keywords == expected[question].keywords
    assert expected["Where'd I put my keys?"].id == "golden-02-2"


def test_the_golden_answers_are_the_evidence_and_an_abstention_is_the_agents_own_words(built):
    expected = {row.question: row for row in fx.load_expected(built.root).values()}
    assert expected[KEYS].label.answer == "Keys last seen Sat 7:41 PM in the desk zone on cam-desk, next to the coffee mug."
    assert expected[KEYS].tool_results == ["keys last seen Sat 7:41 PM in the desk zone on cam-desk, next to the coffee mug"]
    assert expected[WALLET].label.answer == NO_EVIDENCE == "I didn't see that."
    assert (expected[WALLET].tool_results, expected[WALLET].cited, expected[WALLET].no_evidence) == (
        ["nothing found"], [], True)
    assert expected[DOOR].label.answer == "Sun 9:05 AM on cam-door: A courier drops a cardboard box at the door."


def test_the_golden_answers_pass_the_golden_evals_own_scoring(built, fixture_night):
    _, _, golden, _ = fixture_night
    expected = {row.question: row for row in fx.load_expected(built.root).values()}
    skipped = {s["question"] for s in built.skipped}
    scored = {}
    for g in golden:
        for text in [g.q, *g.also]:
            if text in skipped:
                continue
            row = expected[text]
            if g.no_evidence:
                scored[text] = row.cited == [] and row.label.answer == NO_EVIDENCE
            else:
                scored[text] = (set(g.cite) <= set(row.cited)
                                and any(k.lower() in row.label.answer.lower() for k in g.keywords))
            assert row.keywords == g.keywords
    assert len(scored) == built.golden == 24 and all(scored.values())


def test_the_shapes_whose_evidence_no_tool_returns_are_skipped_with_their_reasons(built, fixture_night):
    _, _, golden, _ = fixture_night
    wordings = {g.q: [g.q, *g.also] for g in golden}
    reasons = {s["question"]: s["reason"] for s in built.skipped}
    assert set(reasons) == set(wordings[LAST_LEFT] + wordings[PACKAGE] + wordings[DUE]) and len(reasons) == 18
    assert all("Most recent" in reasons[t] for t in wordings[LAST_LEFT])
    assert all("48 hours" in reasons[t] for t in wordings[PACKAGE])
    assert all("TOOLS" in reasons[t] and "items" in reasons[t] for t in wordings[DUE])
    assert not set(reasons) & {row.question for row in fx.load_expected(built.root).values()}


def test_what_happened_to_an_item_has_no_example_because_no_tool_returns_its_evidence(built, fixture_night):
    memory, now, _, sleep = fixture_night
    assert "What happened to my keys?" not in {row.question for row in fx.load_expected(built.root).values()}
    for question, reason_has in [("What happened to my keys?", "48 hours"), ("Did a package arrive?", "48 hours"),
                                 ("When was the last time someone left the table?", "Most recent")]:
        with pytest.raises(fx.Unsupported) as why:
            fx.reference_trace(question, memory, now, sleep)
        assert reason_has in str(why.value), question


def test_no_wording_repeats_and_no_variant_is_a_golden_question(built):
    expected = fx.load_expected(built.root)
    normal = [" ".join(row.question.lower().split()).rstrip("?.! ") for row in expected.values()]
    assert len(set(normal)) == len(normal) == built.examples
    assert all(row.id.startswith("golden-") == (row.origin == "golden") for row in expected.values())
    assert all(row.id.startswith("v-") for row in expected.values() if row.origin == "variant")


def test_questions_with_the_same_tool_calls_share_a_group_and_no_other_does(built):
    expected = fx.load_expected(built.root)
    calls_of_group: dict[str, set[str]] = defaultdict(set)
    group_of_calls: dict[str, set[str]] = defaultdict(set)
    for row in expected.values():
        key = json.dumps([c.model_dump() for c in row.label.tool_calls], sort_keys=True)
        calls_of_group[row.group].add(key)
        group_of_calls[key].add(row.group)
    assert all(len(keys) == 1 for keys in calls_of_group.values())
    assert all(len(groups) == 1 for groups in group_of_calls.values())
    assert len(calls_of_group) == built.groups
    overnight = [r.question for r in expected.values() if r.group == expected["golden-01"].group]
    assert OVERNIGHT in overnight and "What happened last night?" in overnight and "What happened overnight?" in overnight
    assert "What happened in the last hour?" in [r.question for r in expected.values() if r.group == expected["golden-03"].group]


def test_holding_out_the_golden_groups_leaves_enough_held_out_examples_and_more_to_train_on(built):
    expected = fx.load_expected(built.root).values()
    golden_groups = {row.group for row in expected if row.origin == "golden"}
    held_out = [row for row in expected if row.group in golden_groups]
    train = [row for row in expected if row.group not in golden_groups]
    assert len(held_out) >= 30 and len(held_out) >= 0.2 * built.examples
    assert len(train) > len(held_out)
    assert {row.origin for row in train} == {"variant"}


def test_variants_cover_items_seen_and_unseen_and_windows_with_and_without_evidence(built):
    expected = {row.question: row for row in fx.load_expected(built.root).values()}
    assert expected["Where are my glasses right now?"].label.tool_calls[0].model_dump() == call("last_seen", label="glasses")
    assert expected["Where are my glasses right now?"].label.answer == NO_EVIDENCE
    assert expected["Where did I put my keys?"].cited == ["frames:cam-desk:k1941"]
    assert expected["Was there any activity in the last 10 minutes?"].label.answer == NO_EVIDENCE
    assert expected["Was there any activity in the last 10 minutes?"].label.tool_calls[0].model_dump() == call(
        "episodes_between", start="2026-10-11T09:20:00-04:00", end="2026-10-11T09:30:00-04:00")
    assert expected["What happened in the last 90 minutes?"].cited == ["e08", "frames:cam-door:e08"]


def test_a_question_that_names_a_camera_is_answered_from_that_camera_but_the_model_is_shown_every_camera(built):
    expected = {row.question: row for row in fx.load_expected(built.root).values()}
    door = expected["Did anyone come to the door overnight?"]
    assert [c.model_dump() for c in door.label.tool_calls] == [
        call("episodes_between", start="2026-10-10T20:00:00-04:00", end="2026-10-11T08:00:00-04:00")]
    assert door.label.answer == "Sun 2:20 AM on cam-door: A person in a hoodie walks past the door."
    assert door.cited == ["e05", "frames:cam-door:e05"]
    assert len(door.tool_results[0].split("\n")) == 5   # the model's call has no camera in it, so it gets the night
    assert door.group == expected[OVERNIGHT].group      # the same call, so the same group as the same window unnamed
    assert expected["Was anyone at the desk today?"].cited == ["e06", "e07", "frames:cam-desk:e06", "frames:cam-desk:e07"]


def test_only_the_golden_questions_when_variants_are_off(tmp_path):
    result = fx.build(tmp_path / "out", variants=False)
    assert (result.examples, result.golden, result.variants, result.groups) == (24, 24, 0, 4)
    assert all(r["id"].startswith("golden-") for r in jsonl(result.root / "examples.jsonl"))


# --- the task -----------------------------------------------------------------------------------

def test_the_instruction_and_every_tool_result_are_what_the_agents_real_tool_loop_does(built, fixture_night):
    memory, now, _, sleep = fixture_night
    manifest = json.loads((built.root / "manifest.json").read_text(encoding="utf-8"))
    mismatches = []
    for row in fx.load_expected(built.root).values():
        runtime = FakeRuntime(chat_replies=[
            ChatResult(content="", model_version="fake-0", tool_calls=[
                ToolCall(id=f"call-{n}", name=c.name, arguments=c.arguments) for n, c in enumerate(row.label.tool_calls)]),
            ChatResult(content="done", model_version="fake-0")])
        asyncio.run(QA(runtime, memory, clock=lambda: now, sleep=sleep).tool_loop(row.question))
        messages = runtime.calls[-1][1]
        shown = [m.content for m in messages if m.role == "tool"]
        if (messages[0].content, messages[1].content, shown) != (manifest["spec"]["instruction"], row.question, row.tool_results):
            mismatches.append(row.id)
    assert mismatches == []


def test_every_expected_label_validates_against_the_output_schema(built):
    manifest = json.loads((built.root / "manifest.json").read_text(encoding="utf-8"))
    schema = manifest["spec"]["output_schema"]
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    failures = [row.id for row in fx.load_expected(built.root).values()
                if list(validator.iter_errors(row.label.model_dump()))]
    assert failures == []
    assert schema == fx.output_schema()


@pytest.mark.parametrize("label", [
    {"tool_calls": [{"name": "last_seen", "arguments": {}}], "answer": "x"},                          # missing label
    {"tool_calls": [{"name": "last_seen", "arguments": {"label": "keys", "zone": "desk"}}], "answer": "x"},   # invented argument
    {"tool_calls": [{"name": "last_seen", "arguments": {"label": 5}}], "answer": "x"},                # wrong type
    {"tool_calls": [{"name": "open_door", "arguments": {"label": "keys"}}], "answer": "x"},           # no such tool
    {"tool_calls": [{"name": "episodes_between", "arguments": {"start": "a"}}], "answer": "x"},       # missing end
    {"tool_calls": [{"name": "last_seen", "arguments": {"label": "keys"}}], "answer": ""},            # empty answer
    {"tool_calls": [{"name": "last_seen", "arguments": {"label": "keys"}}]},                          # no answer
    {"answer": "x"},                                                                                  # no calls field
])
def test_a_label_that_misuses_a_tool_fails_the_output_schema(label):
    validator = jsonschema.Draft202012Validator(fx.output_schema())
    assert list(validator.iter_errors(label)) != []


def test_a_label_that_calls_each_tool_correctly_passes_the_output_schema():
    validator = jsonschema.Draft202012Validator(fx.output_schema())
    label = {"tool_calls": [call("search_memory", q="keys"), call("last_seen", label="keys"),
                            call("episodes_between", start="a", end="b")], "answer": "Done."}
    assert list(validator.iter_errors(label)) == []
    assert {t["function"]["name"] for t in TOOLS} == {"search_memory", "last_seen", "episodes_between"}


# --- the reference policy ---------------------------------------------------------------------------

def test_reference_trace_makes_the_one_tool_call_the_agent_makes(fixture_night):
    memory, now, _, sleep = fixture_night
    trace = fx.reference_trace("Where's my glasses?", memory, now, sleep)
    assert trace.tool_calls == (call("last_seen", label="glasses"),)   # the lookup fallback inside the tool is not a second call
    assert (trace.tool_results, trace.cited, trace.answer) == (("nothing found",), (), NO_EVIDENCE)


def test_reference_trace_follows_the_time_it_is_given(fixture_night):
    memory, now, _, sleep = fixture_night
    earlier = fx.reference_trace("What happened in the last 2 hours?", memory, now - timedelta(days=1), sleep)
    assert earlier.tool_calls == (call("episodes_between", start="2026-10-10T07:30:00-04:00", end="2026-10-10T09:30:00-04:00"),)
    assert earlier.answer == NO_EVIDENCE


@pytest.mark.parametrize("question, reason_has", [
    ("Tell me a joke", "open tool loop"),
    ("Why is the sky blue", "open tool loop"),
    (DUE, "TOOLS"),
])
def test_reference_trace_refuses_questions_the_agent_does_not_answer_with_a_tool(fixture_night, question, reason_has):
    memory, now, _, sleep = fixture_night
    with pytest.raises(fx.Unsupported) as why:
        fx.reference_trace(question, memory, now, sleep)
    assert reason_has in str(why.value)


# --- building -------------------------------------------------------------------------------------

def test_the_build_is_deterministic_to_the_byte(tmp_path):
    first, second = fx.build(tmp_path / "a"), fx.build(tmp_path / "b")
    assert (first.examples, first.groups) == (second.examples, second.groups)
    for name in FILES:
        assert (first.root / name).read_bytes() == (second.root / name).read_bytes(), name


def test_the_committed_fixture_matches_the_current_build(tmp_path):
    fresh = fx.build(tmp_path / "out")
    assert COMMITTED == fx.OUT_DIR / fx.DATASET_ID   # what the script writes by default is the committed folder
    assert COMMITTED.is_dir(), "fixtures/learning/tool-calling-fixture is missing: run python lab/fixture_dataset.py"
    assert stale_parts(COMMITTED, fresh.root) == [], (
        "the agent or the golden fixtures changed what a label is checked against: rebuild with "
        "python lab/fixture_dataset.py (it refuses while examples hold targets; --force drops them)")
    assert dataset.check(COMMITTED, require_images=False) == []


def test_labels_in_a_copy_of_the_committed_fixture_do_not_make_it_stale(tmp_path):
    fresh = fx.build(tmp_path / "out")
    labeled = tmp_path / "copy"
    shutil.copytree(COMMITTED, labeled)
    dataset.write_target(labeled, "golden-01", '{"tool_calls": [], "answer": "x"}')
    assert [e.id for e in dataset.load(labeled, require_images=False).labeled()] == ["golden-01"]
    assert stale_parts(labeled, fresh.root) == []


def test_a_committed_fixture_that_calls_a_different_tool_is_stale(tmp_path):
    fresh = fx.build(tmp_path / "out")
    changed = tmp_path / "copy"
    shutil.copytree(COMMITTED, changed)
    text = (changed / "expected.jsonl").read_bytes().decode("utf-8")
    assert '"label":"keys"' in text
    (changed / "expected.jsonl").write_bytes(text.replace('"label":"keys"', '"label":"key"').encode("utf-8"))
    (changed / "manifest.json").write_text(
        (changed / "manifest.json").read_text(encoding="utf-8").replace("tool-calling", "tool-callin"), encoding="utf-8")
    assert stale_parts(changed, fresh.root) == ["manifest.json", "expected.jsonl"]
    assert stale_parts(COMMITTED, fresh.root) == []


def test_a_new_caption_or_time_format_in_the_fixtures_does_not_make_the_committed_fixture_stale(tmp_path):
    fresh = fx.build(tmp_path / "out")
    reworded = tmp_path / "copy"
    shutil.copytree(COMMITTED, reworded)
    text = (reworded / "expected.jsonl").read_bytes().decode("utf-8")
    assert "A courier drops a cardboard box" in text
    (reworded / "expected.jsonl").write_bytes(text.replace("A courier drops a cardboard box",
                                                           "A courier leaves a parcel").encode("utf-8"))
    assert stale_parts(reworded, fresh.root) == []


def test_a_golden_question_the_agent_stops_citing_fails_the_build(tmp_path):
    golden = tmp_path / "golden"
    shutil.copytree(GOLDEN_DIR, golden)
    text = (golden / "questions.yaml").read_text(encoding="utf-8")
    (golden / "questions.yaml").write_text(text.replace('cite: ["frames:cam-desk:k1941"]',
                                                        'cite: ["frames:cam-desk:k0000"]'), encoding="utf-8")
    with pytest.raises(fx.BuildError, match=r"golden question 2: the agent's tool call does not cite \['frames:cam-desk:k0000'\]"):
        fx.build(tmp_path / "out", golden_dir=golden)
    assert not (tmp_path / "out").exists()


def test_a_golden_answer_without_its_keywords_fails_the_build(tmp_path):
    golden = tmp_path / "golden"
    shutil.copytree(GOLDEN_DIR, golden)
    text = (golden / "questions.yaml").read_text(encoding="utf-8")
    (golden / "questions.yaml").write_text(text.replace("keywords: [courier, box, package]", "keywords: [zebra]"),
                                           encoding="utf-8")
    with pytest.raises(fx.BuildError, match=r"golden question 3: the answer holds none of the keywords \['zebra'\]"):
        fx.build(tmp_path / "out", golden_dir=golden)


def test_a_question_marked_no_evidence_that_has_evidence_fails_the_build(tmp_path):
    golden = tmp_path / "golden"
    shutil.copytree(GOLDEN_DIR, golden)
    text = (golden / "questions.yaml").read_text(encoding="utf-8")
    marked = text.replace('  cite: ["frames:cam-desk:k1941"]\n  keywords: [mug, desk]', '  no_evidence: true')
    assert marked != text
    (golden / "questions.yaml").write_text(marked, encoding="utf-8")
    with pytest.raises(fx.BuildError, match="golden question 2 expects no evidence"):
        fx.build(tmp_path / "out", golden_dir=golden)


def test_golden_fixtures_that_cannot_be_read_fail_the_build_naming_only_the_folder(tmp_path):
    with pytest.raises(fx.BuildError, match="cannot read the golden fixtures in nowhere: FileNotFoundError"):
        fx.build(tmp_path / "out", golden_dir=tmp_path / "nowhere")
    broken = tmp_path / "broken"
    shutil.copytree(GOLDEN_DIR, broken)
    (broken / "questions.yaml").write_text("- q: 5\n  cite: not a list\n", encoding="utf-8")
    with pytest.raises(fx.BuildError, match="cannot read the golden fixtures in broken: ValidationError"):
        fx.build(tmp_path / "out", golden_dir=broken)
    assert not (tmp_path / "out").exists()


def test_a_variant_the_agent_would_not_answer_with_a_tool_fails_the_build(tmp_path, monkeypatch):
    monkeypatch.setattr(fx, "variant_questions", lambda: ["Tell me a joke"])
    with pytest.raises(fx.BuildError, match="variant question 'Tell me a joke' is not answered with a tool"):
        fx.build(tmp_path / "out")


def test_rebuilding_an_unlabeled_folder_is_fine_and_other_files_in_it_are_left_alone(tmp_path):
    first = fx.build(tmp_path / "out")
    (first.root / "split.json").write_text('{"held_out": []}', encoding="utf-8")
    before = (first.root / "examples.jsonl").read_bytes()
    again = fx.build(tmp_path / "out")
    assert (again.root / "examples.jsonl").read_bytes() == before
    assert (again.root / "split.json").read_text(encoding="utf-8") == '{"held_out": []}'


def test_a_folder_that_already_has_labels_is_not_rebuilt_without_force(tmp_path):
    first = fx.build(tmp_path / "out")
    dataset.write_target(first.root, "golden-02", '{"tool_calls": [], "answer": "x"}')
    with pytest.raises(fx.BuildError) as refused:
        fx.build(tmp_path / "out")
    assert refused.value.outcome == "refused" and "already has 1 labeled examples" in str(refused.value)
    assert dataset.load(first.root, require_images=False).labeled()[0].id == "golden-02"
    forced = fx.build(tmp_path / "out", force=True)
    assert dataset.load(forced.root, require_images=False).labeled() == []


def test_a_folder_that_cannot_be_read_is_not_rebuilt_without_force(tmp_path):
    first = fx.build(tmp_path / "out")
    (first.root / "examples.jsonl").write_text("{broken\n", encoding="utf-8")
    with pytest.raises(fx.BuildError) as refused:
        fx.build(tmp_path / "out")
    assert refused.value.outcome == "refused" and "cannot be read" in str(refused.value)
    assert fx.build(tmp_path / "out", force=True).examples == first.examples


# --- the entry point -------------------------------------------------------------------------------

def test_the_entry_point_builds_and_leaves_a_ledger_row(tmp_path):
    answer = fx.main(["--out", str(tmp_path / "out"), "--no-variants"])
    assert answer == {"ok": True, "outcome": "built", "dataset_id": fx.DATASET_ID,
                      "path": str(tmp_path / "out" / fx.DATASET_ID), "examples": 24, "golden": 24, "variants": 0,
                      "groups": 4, "skipped": answer["skipped"], "refs": {"dataset_id": fx.DATASET_ID}}
    assert len(answer["skipped"]) == 18 and DUE in {s["question"] for s in answer["skipped"]}
    row = only_row()
    assert (row["verb"], row["outcome"], row["actor"], row["tier"]) == ("fixture-dataset", "built", "sonnet-test", "sonnet")
    assert row["refs"]["dataset_id"] == fx.DATASET_ID
    assert dataset.check(tmp_path / "out" / fx.DATASET_ID, require_images=False) == []


def test_the_entry_point_takes_another_dataset_id_and_another_golden_folder(tmp_path):
    answer = fx.main(["--out", str(tmp_path / "out"), "--dataset-id", "other", "--golden", str(GOLDEN_DIR), "--no-variants"])
    assert answer["ok"] is True and (tmp_path / "out" / "other" / "manifest.json").is_file()
    assert json.loads((tmp_path / "out" / "other" / "manifest.json").read_text(encoding="utf-8"))["dataset_id"] == "other"


def test_the_entry_point_reports_a_refusal_and_the_row_says_refused(tmp_path):
    out = tmp_path / "out"
    fx.main(["--out", str(out)])
    dataset.write_target(out / fx.DATASET_ID, "golden-02", "label")
    answer = fx.main(["--out", str(out)])
    assert answer["ok"] is False and answer["outcome"] == "refused" and "labeled examples" in answer["error"]
    assert [r["outcome"] for r in ledger.read()] == ["built", "refused"]
    assert fx.main(["--out", str(out), "--force"])["ok"] is True


def test_the_entry_point_refuses_to_run_while_the_machine_is_stopped(tmp_path, capsys):
    state = tmp_path / "state"
    state.mkdir()
    (state / "STOP").write_text("maintenance\n", encoding="utf-8")
    with pytest.raises(stop.Stopped):
        fx.main(["--out", str(tmp_path / "out")])
    assert json.loads(capsys.readouterr().out)["outcome"] == "stopped"
    assert not (tmp_path / "out").exists()
    assert only_row()["outcome"] == "stopped"


def test_run_by_path_from_another_folder_the_script_builds_and_prints_one_json_line(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    done = subprocess.run([sys.executable, str(SCRIPT), "--out", str(tmp_path / "out"), "--no-variants"], cwd=elsewhere,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    answer = json.loads(done.stdout)
    assert (answer["ok"], answer["examples"]) == (True, 24)
    assert (tmp_path / "out" / fx.DATASET_ID / "examples.jsonl").is_file()
    assert only_row()["verb"] == "fixture-dataset"

