"""lab/dataset.py load_split(): a dataset's frozen split, and the committed split of the tool-calling fixture.

The reader is held to the real thing: the committed fixtures/learning/tool-calling-fixture folder with its
split.json, and small datasets written with the contract models for the cases that must be refused. Nothing is
faked. Every test works in a temporary folder or reads the committed one, and uses a private ledger state dir.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from contracts.learning import DatasetExample, DatasetManifest, TaskSpec
from lab import dataset
from lab import fixture_dataset as fx
from lab.dataset import DatasetError, load_split
from lab.machine import ledger

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "lab" / "dataset.py"
COMMITTED = REPO_ROOT / "fixtures" / "learning" / fx.DATASET_ID
NOW = datetime(2026, 10, 11, 9, 0, tzinfo=UTC)
SPEC = TaskSpec(task="tool-calling", instruction="Answer with tool calls.")
SECRET = "household secret"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """A private ledger state dir and a known writer for every test."""
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LAB_ACTOR", "sonnet-test")
    monkeypatch.setenv("LAB_TIER", "sonnet")
    monkeypatch.delenv("LAB_TASK", raising=False)


@pytest.fixture
def committed():
    return load_split(COMMITTED, require_images=False)


def make(root: Path, train_sizes=(5,) * 6, held_out_sizes=(5,) * 10, *, pictures=False):
    """A dataset folder with a split: one group of examples per size, a frame_ref and a question per group.

    Returns the folder and the ids of the train and held-out sides, in the order split.json lists them."""
    root.mkdir(parents=True)
    rows, sides = [], {"t": [], "h": []}
    for side, sizes in (("t", train_sizes), ("h", held_out_sizes)):
        for group, size in enumerate(sizes):
            for wording in range(size):
                name = f"{side}{group}-{wording}"
                rows.append((name, f"group:{side}{group}", f"Question {side}{group} wording {wording}?"))
                sides[side].append(name)
    examples = [DatasetExample(id=name, image=f"images/{name}.jpg" if pictures else "", frame_ref=ref, ts=NOW,
                               context=text) for name, ref, text in rows]
    if pictures:
        (root / "images").mkdir()
        for example in examples:
            (root / example.image).write_bytes(b"jpeg " + example.id.encode("utf-8"))
    manifest = DatasetManifest(dataset_id=root.name, spec=SPEC, created=NOW, examples=len(examples),
                               action_id="approval-1")
    (root / "manifest.json").write_text(json.dumps(manifest.model_dump(mode="json"), indent=2), encoding="utf-8")
    (root / "examples.jsonl").write_text("".join(e.model_dump_json() + "\n" for e in examples), encoding="utf-8",
                                         newline="\n")
    write_split(root, {"version": 1, "dataset_id": root.name, "rule": "a test split",
                       "train": sides["t"], "held_out": sides["h"]})
    return root, sides["t"], sides["h"]


def write_split(root: Path, body: dict) -> None:
    (root / "split.json").write_text(json.dumps(body, indent=2), encoding="utf-8", newline="\n")


def read_split(root: Path) -> dict:
    return json.loads((root / "split.json").read_text(encoding="utf-8"))


def rows(root: Path) -> list[dict]:
    text = (root / "examples.jsonl").read_bytes().decode("utf-8")
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def write_rows(root: Path, data: list[dict]) -> None:
    (root / "examples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in data), encoding="utf-8", newline="\n")


def change_examples(root: Path, changes: dict[str, dict]) -> None:
    """Set fields on examples: change_examples(root, {"t0-0": {"frame_ref": "x"}})."""
    data = rows(root)
    for row in data:
        row.update(changes.get(row["id"], {}))
    write_rows(root, data)


def problems_of(root: Path, **options) -> list[str]:
    options.setdefault("require_images", False)
    with pytest.raises(DatasetError) as err:
        load_split(root, **options)
    return err.value.problems


def only_row():
    found = ledger.read()
    assert len(found) == 1, found
    return found[0]


def words(question: str) -> str:
    """A question the way a person would say it is the same: case, spacing and closing punctuation do not matter."""
    return " ".join(question.lower().split()).rstrip("?.! ")


# --- the committed split of the tool-calling fixture -------------------------------------------------------------

def test_the_committed_split_lists_every_example_once_and_the_reader_accepts_it(committed):
    everyone = [e.id for e in committed.data.examples]
    train, held_out = [e.id for e in committed.train], [e.id for e in committed.held_out]
    assert len(everyone) == 142 == committed.data.manifest.examples
    assert (len(train), len(held_out)) == (84, 58)
    assert set(train).isdisjoint(held_out)
    assert sorted(train + held_out) == sorted(everyone)
    assert committed.dataset_id == fx.DATASET_ID == read_split(COMMITTED)["dataset_id"]
    assert dataset.check(COMMITTED, require_images=False) == []


def test_the_committed_held_out_side_holds_at_least_the_50_examples_the_owner_requires(committed):
    assert dataset.MIN_HELD_OUT == 50
    assert len(committed.held_out) >= 50
    assert len(committed.held_out) / len(committed.data.examples) >= 0.2


def test_no_question_group_is_on_both_sides_of_the_committed_split(committed):
    group = {row.id: row.group for row in fx.load_expected(COMMITTED).values()}
    train_groups = {group[e.id] for e in committed.train}
    held_out_groups = {group[e.id] for e in committed.held_out}
    assert train_groups.isdisjoint(held_out_groups)
    assert (len(train_groups), len(held_out_groups)) == (12, 4)
    assert committed.held_out_frame_refs == 4   # 58 examples, but only 4 distinct cases among them


def test_no_frame_ref_and_no_question_is_on_both_sides_of_the_committed_split(committed):
    question = {row.id: words(row.question) for row in fx.load_expected(COMMITTED).values()}
    assert {e.frame_ref for e in committed.train}.isdisjoint({e.frame_ref for e in committed.held_out})
    assert {question[e.id] for e in committed.train}.isdisjoint({question[e.id] for e in committed.held_out})
    assert len({question[e.id] for e in committed.held_out}) == 58   # every held-out example is its own wording


def test_the_committed_split_holds_out_exactly_the_groups_that_have_a_golden_question(committed):
    expected = fx.load_expected(COMMITTED)
    golden_groups = {row.group for row in expected.values() if row.origin == "golden"}
    held_out = {e.id for e in committed.held_out}
    assert len(golden_groups) == 4
    assert held_out == {row.id for row in expected.values() if row.group in golden_groups}
    assert {row.id for row in expected.values() if row.origin == "golden"} <= held_out
    assert {expected[e.id].origin for e in committed.train} == {"variant"}


def test_the_hash_is_the_sha256_of_the_file_with_its_line_endings_read_as_lf(tmp_path, committed):
    raw = (COMMITTED / "split.json").read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    assert committed.sha256 == hashlib.sha256(lf).hexdigest()
    crlf = tmp_path / "crlf"
    shutil.copytree(COMMITTED, crlf)
    (crlf / "split.json").write_bytes(lf.replace(b"\n", b"\r\n"))
    assert load_split(crlf, require_images=False).sha256 == committed.sha256


def test_a_text_only_dataset_is_read_with_require_images_false_as_load_demands():
    problems = problems_of(COMMITTED, require_images=True)
    assert len(problems) == 142 and all("has no image" in problem for problem in problems)


# --- what the reader refuses ----------------------------------------------------------------------------------

def test_a_missing_split_file_is_one_problem(tmp_path):
    root, _, _ = make(tmp_path / "d")
    (root / "split.json").unlink()
    assert problems_of(root) == ["split.json: missing"]


def test_a_split_file_that_is_not_json_or_not_a_split_names_the_field(tmp_path):
    root, train, held_out = make(tmp_path / "d")
    (root / "split.json").write_text("{not json", encoding="utf-8")
    [problem] = problems_of(root)
    assert problem.startswith("split.json: ") and "JSON" in problem

    write_split(root, {"version": 2, "dataset_id": "d", "train": train, "held_out": held_out})
    assert problems_of(root) == ["split.json: version: Input should be 1"]
    write_split(root, {"version": 1, "dataset_id": "d", "train": train})
    assert problems_of(root) == ["split.json: held_out: Field required"]
    write_split(root, {"version": 1, "dataset_id": "d", "train": train, "held_out": held_out, "heldout": []})
    assert problems_of(root) == ["split.json: heldout: Extra inputs are not permitted"]


def test_a_file_that_is_not_utf8_is_a_problem(tmp_path):
    root, _, _ = make(tmp_path / "d")
    (root / "split.json").write_bytes(b"\xff\xfe not utf-8")
    assert problems_of(root) == ["split.json: cannot be read (UnicodeDecodeError)"]


def test_a_dataset_that_does_not_hold_up_is_reported_before_its_split(tmp_path):
    root, _, _ = make(tmp_path / "d")
    (root / "manifest.json").unlink()
    assert problems_of(root) == ["manifest.json: missing"]


def test_a_split_made_for_another_dataset_is_refused(tmp_path):
    root, _, _ = make(tmp_path / "d")
    write_split(root, {**read_split(root), "dataset_id": "other"})
    assert problems_of(root) == ["split.json: is for dataset 'other', but the folder holds 'd'"]


def test_ids_the_dataset_does_not_have_are_named(tmp_path):
    root, train, held_out = make(tmp_path / "d")
    write_split(root, {**read_split(root), "train": [*train, "ghost-1"], "held_out": [*held_out, "ghost-2", "ghost-3"]})
    assert problems_of(root) == [
        "split.json: train ids that examples.jsonl does not have (1): 'ghost-1'",
        "split.json: held_out ids that examples.jsonl does not have (2): 'ghost-2', 'ghost-3'"]


def test_an_id_listed_twice_on_one_side_is_named(tmp_path):
    root, train, held_out = make(tmp_path / "d")
    write_split(root, {**read_split(root), "held_out": [*held_out, held_out[3]]})
    assert problems_of(root) == ["split.json: held_out ids listed more than once (1): 'h0-3'"]


def test_an_id_on_both_sides_is_named_once(tmp_path):
    root, train, held_out = make(tmp_path / "d")
    write_split(root, {**read_split(root), "held_out": [*held_out, train[0]]})
    assert problems_of(root) == ["split.json: ids listed in both train and held_out (1): 't0-0'"]


def test_an_example_on_neither_side_is_named_so_it_cannot_slip_into_training(tmp_path):
    root, train, held_out = make(tmp_path / "d")
    write_split(root, {**read_split(root), "train": train[:-2]})
    assert problems_of(root) == ["split.json: examples listed on neither side (2): 't5-3', 't5-4'"]


def test_a_long_list_of_ids_is_cut_to_five_and_a_count(tmp_path):
    root, train, held_out = make(tmp_path / "d")
    write_split(root, {**read_split(root), "train": train[:-8]})
    [problem] = problems_of(root)
    assert problem == ("split.json: examples listed on neither side (8): 't4-2', 't4-3', 't4-4', 't5-0', 't5-1' "
                       "and 3 more")


def test_an_example_added_to_the_dataset_after_the_freeze_joins_neither_side(tmp_path):
    root, train, held_out = make(tmp_path / "d")
    data = rows(root)
    data.append({**data[0], "id": "late-1", "frame_ref": "group:late", "context": "A question added later?"})
    write_rows(root, data)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["examples"] += 1
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert dataset.check(root, require_images=False) == []   # the dataset itself is fine
    assert problems_of(root) == ["split.json: examples listed on neither side (1): 'late-1'"]


def test_a_frame_ref_on_both_sides_is_refused_and_named_by_ids_not_by_value(tmp_path):
    root, train, held_out = make(tmp_path / "d", pictures=True)
    change_examples(root, {train[0]: {"frame_ref": "frames:cam-private:kitchen"},
                           held_out[0]: {"frame_ref": "frames:cam-private:kitchen"}})
    [problem] = problems_of(root, require_images=True)
    assert problem == "split.json: frame_refs on both sides (1), for example train 't0-0' and held_out 'h0-0'"
    assert "kitchen" not in problem


def test_a_group_split_across_the_sides_is_one_frame_ref_on_both_however_many_examples_it_has(tmp_path):
    root, train, held_out = make(tmp_path / "d")
    write_split(root, {**read_split(root), "train": [*train, *held_out[:2]], "held_out": held_out[2:]})
    assert problems_of(root, min_held_out=1) == [
        "split.json: frame_refs on both sides (1), for example train 'h0-0' and held_out 'h0-2'"]


def test_an_example_without_a_frame_ref_cannot_be_kept_on_one_side(tmp_path):
    root, train, held_out = make(tmp_path / "d")
    change_examples(root, {train[0]: {"frame_ref": " "}})
    assert problems_of(root) == [
        "split.json: examples with no frame_ref, which the split cannot keep on one side (1): 't0-0'"]


def test_the_same_question_on_both_sides_is_refused_however_it_is_spelled(tmp_path):
    root, train, held_out = make(tmp_path / "d")
    change_examples(root, {train[0]: {"context": f"Where is the {SECRET}?"},
                           held_out[0]: {"context": f"  WHERE is   the {SECRET.upper()} ?? "}})
    [problem] = problems_of(root)
    assert problem == "split.json: questions on both sides (1), for example train 't0-0' and held_out 'h0-0'"


def test_the_same_caption_beside_different_pictures_is_not_the_same_input(tmp_path):
    root, train, held_out = make(tmp_path / "d", pictures=True)
    change_examples(root, {train[0]: {"context": "A person at the desk."},
                           held_out[0]: {"context": "A person at the desk."}})
    split = load_split(root, require_images=True)
    assert (len(split.train), len(split.held_out)) == (30, 50)


def test_the_held_out_side_needs_50_examples_and_49_is_refused(tmp_path):
    enough, _, held_out = make(tmp_path / "enough", held_out_sizes=(5,) * 10)
    assert len(load_split(enough, require_images=False).held_out) == 50 == len(held_out)

    short, _, held_out = make(tmp_path / "short", held_out_sizes=(5,) * 9 + (4,))
    assert len(held_out) == 49
    assert problems_of(short) == ["split.json: held_out has 49 examples, below the 50 a gate needs before it may deploy"]


def test_a_short_held_out_set_is_read_only_when_asked_for_and_then_it_cannot_deploy(tmp_path, committed):
    short, _, _ = make(tmp_path / "short", train_sizes=(2,), held_out_sizes=(3, 4))
    assert problems_of(short) == ["split.json: held_out has 7 examples, below the 50 a gate needs before it may deploy"]
    reported = load_split(short, require_images=False, min_held_out=7)
    assert len(reported.held_out) == 7 and reported.meets_deploy_floor is False
    assert committed.meets_deploy_floor is True
    exact, _, _ = make(tmp_path / "exact")
    assert load_split(exact, require_images=False).meets_deploy_floor is True


def test_a_split_with_nothing_to_train_on_is_refused(tmp_path):
    root, _, _ = make(tmp_path / "d", train_sizes=())
    assert problems_of(root) == ["split.json: train is empty"]
    write_split(root, {**read_split(root), "train": ["ghost"]})   # a train list with no real example in it is empty too
    assert problems_of(root) == ["split.json: train ids that examples.jsonl does not have (1): 'ghost'",
                                 "split.json: train is empty"]


def test_an_unknown_id_and_a_repeated_id_do_not_count_toward_the_floor(tmp_path):
    root, train, held_out = make(tmp_path / "d")
    write_split(root, {**read_split(root), "held_out": [*held_out[:-1], "ghost-1"]})   # 49 real examples and a ghost
    assert problems_of(root) == [
        "split.json: held_out ids that examples.jsonl does not have (1): 'ghost-1'",
        "split.json: examples listed on neither side (1): 'h9-4'",
        "split.json: held_out has 49 examples, below the 50 a gate needs before it may deploy"]

    short, train, held_out = make(tmp_path / "short", held_out_sizes=(5,) * 9 + (4,))
    write_split(short, {**read_split(short), "held_out": [*held_out, held_out[0]]})   # 50 entries, 49 distinct
    assert problems_of(short) == [
        "split.json: held_out ids listed more than once (1): 'h0-0'",
        "split.json: held_out has 49 examples, below the 50 a gate needs before it may deploy"]


def test_every_problem_is_listed_at_once_and_none_quotes_a_caption_a_question_or_a_label(tmp_path):
    root, train, held_out = make(tmp_path / "d", held_out_sizes=(5,) * 9)
    change_examples(root, {
        train[0]: {"context": f"Where is the {SECRET}?", "target": f"label with a {SECRET}"},
        held_out[0]: {"context": f"where is the {SECRET}", "target": f"label with a {SECRET}"},
        train[1]: {"frame_ref": f"frames:cam-{SECRET}"}, held_out[1]: {"frame_ref": f"frames:cam-{SECRET}"}})
    write_split(root, {**read_split(root), "train": [*train, "ghost"]})
    problems = problems_of(root)
    assert problems == [
        "split.json: train ids that examples.jsonl does not have (1): 'ghost'",
        "split.json: frame_refs on both sides (1), for example train 't0-1' and held_out 'h0-1'",
        "split.json: questions on both sides (1), for example train 't0-0' and held_out 'h0-0'",
        "split.json: held_out has 45 examples, below the 50 a gate needs before it may deploy"]
    assert SECRET not in " ".join(problems)


# --- membership is what the file says, and nothing else -----------------------------------------------------

def test_membership_is_the_file_in_the_order_it_lists_whatever_the_origin_of_a_question(tmp_path, committed):
    swapped = tmp_path / "swapped"
    shutil.copytree(COMMITTED, swapped)
    body = read_split(swapped)
    write_split(swapped, {**body, "train": body["held_out"], "held_out": body["train"]})   # the golden groups now train
    split = load_split(swapped, require_images=False)
    assert [e.id for e in split.train] == body["held_out"] and [e.id for e in split.held_out] == body["train"]
    golden = {row.id for row in fx.load_expected(swapped).values() if row.origin == "golden"}
    assert golden <= {e.id for e in split.train}
    assert split.sha256 != committed.sha256


def test_membership_does_not_depend_on_the_expected_file_the_order_of_examples_or_the_labels(tmp_path, committed):
    copy = tmp_path / "copy"
    shutil.copytree(COMMITTED, copy)
    (copy / "expected.jsonl").unlink()
    write_rows(copy, rows(copy)[::-1])
    dataset.write_target(copy, "golden-01", '{"tool_calls": [], "answer": "x"}')
    dataset.write_target(copy, "v-4ec085e7", '{"tool_calls": [], "answer": "y"}')
    split = load_split(copy, require_images=False)
    assert [e.id for e in split.train] == [e.id for e in committed.train]
    assert [e.id for e in split.held_out] == [e.id for e in committed.held_out]
    assert split.sha256 == committed.sha256
    assert {e.id for e in split.held_out if e.target} == {"golden-01"}
    assert {e.id for e in split.train if e.target} == {"v-4ec085e7"}


def test_the_trainer_and_the_gate_get_the_same_split_from_separate_reads(committed):
    again = load_split(COMMITTED, require_images=False)
    assert ([e.id for e in again.train], [e.id for e in again.held_out], again.sha256) == (
        [e.id for e in committed.train], [e.id for e in committed.held_out], committed.sha256)


# --- recording the hash --------------------------------------------------------------------------------------

def test_the_entry_point_with_split_prints_the_hash_and_the_ledger_row_carries_it(committed):
    answer = dataset.main([str(COMMITTED), "--text-only", "--split"])
    assert answer == {
        "ok": True, "outcome": "ok", "dataset_id": fx.DATASET_ID, "examples": 142, "labeled": 0, "pending": 142,
        "problems": [], "refs": {"dataset_id": fx.DATASET_ID, "split_sha256": committed.sha256},
        "split": {"sha256": committed.sha256, "train": 84, "held_out": 58, "held_out_frame_refs": 4}}
    row = only_row()
    assert (row["verb"], row["outcome"], row["actor"], row["tier"]) == ("dataset-check", "ok", "sonnet-test", "sonnet")
    assert row["refs"] == {"pr": None, "commit": None, "slot": None, "run_id": None,
                           "dataset_id": fx.DATASET_ID, "split_sha256": committed.sha256}
    assert len(row["refs"]["split_sha256"]) == 64


def test_the_entry_point_without_the_flag_does_not_look_at_the_split(tmp_path):
    root, _, _ = make(tmp_path / "d")
    (root / "split.json").write_text("{not json", encoding="utf-8")
    answer = dataset.main([str(root), "--text-only"])
    assert answer["ok"] is True and "split" not in answer and answer["refs"] == {"dataset_id": "d"}
    assert only_row()["refs"].get("split_sha256") is None


def test_a_split_that_does_not_hold_up_fails_the_check_and_the_row_says_so(tmp_path):
    root, train, held_out = make(tmp_path / "d")
    write_split(root, {**read_split(root), "held_out": held_out[:-1]})
    answer = dataset.main([str(root), "--text-only", "--split"])
    problems = ["split.json: examples listed on neither side (1): 'h9-4'",
                "split.json: held_out has 49 examples, below the 50 a gate needs before it may deploy"]
    assert answer == {"ok": False, "outcome": "invalid", "problems": problems,
                      "error": f"2 problems: {problems[0]}; {problems[1]}"}
    row = only_row()
    assert (row["verb"], row["outcome"], row["note"]) == ("dataset-check", "invalid", answer["error"])
    assert "split_sha256" not in row["refs"]


def test_a_dataset_that_is_missing_its_split_fails_the_split_check(tmp_path):
    root, _, _ = make(tmp_path / "d")
    (root / "split.json").unlink()
    answer = dataset.main([str(root), "--text-only", "--split"])
    assert (answer["ok"], answer["problems"]) == (False, ["split.json: missing"])


def test_run_by_path_from_another_folder_the_script_prints_the_split_and_writes_the_row(tmp_path, committed):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    done = subprocess.run([sys.executable, str(SCRIPT), str(COMMITTED), "--text-only", "--split"], cwd=elsewhere,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    answer = json.loads(done.stdout)
    assert answer["split"] == {"sha256": committed.sha256, "train": 84, "held_out": 58, "held_out_frame_refs": 4}
    assert done.stderr == ""
    assert only_row()["refs"]["split_sha256"] == committed.sha256


def test_run_by_path_on_a_split_that_does_not_hold_up_exits_1(tmp_path):
    root, _, _ = make(tmp_path / "d", held_out_sizes=(5,) * 9)
    done = subprocess.run([sys.executable, str(SCRIPT), str(root), "--text-only", "--split"], cwd=tmp_path,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 1
    assert json.loads(done.stdout)["problems"] == [
        "split.json: held_out has 45 examples, below the 50 a gate needs before it may deploy"]
