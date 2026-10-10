"""lab/dataset.py: reading a dataset export against contracts/learning.py, and writing labels back into it.

The exports under test are written by the box's own exporter (contexts/learning/application/export.py) or, for
the failure cases, by editing such an export, so the reader is held to what the producer really writes. The
teacher is the only thing faked. Every test works in a temporary folder and a private ledger state dir.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from contexts.agent.application.memory_port import FakeMemory
from contexts.learning.application.export import DatasetExporter, teach
from contexts.workspace.domain.approvals import Approvals
from contracts.experience import ApprovalDecided
from contracts.learning import DatasetExample, DatasetManifest, TaskSpec
from contracts.memory import Episode
from lab import dataset
from lab.dataset import DatasetError, Written, check, has_target, load, write_target, write_targets
from lab.machine import ledger, stop
from lab.machine.lock import FileLock, LockTimeout

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "lab" / "dataset.py"
NOW = datetime(2026, 10, 11, 9, 0, tzinfo=UTC)
SPEC = TaskSpec(task="desk-objects", instruction="List small objects on the desk as JSON.", camera="cam-desk")
IDS = ["e1-0", "e2-0", "e3-0"]


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """A private ledger state dir and a known writer for every test."""
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LAB_ACTOR", "sonnet-test")
    monkeypatch.setenv("LAB_TIER", "sonnet")
    monkeypatch.delenv("LAB_TASK", raising=False)


def box_export(tmp_path: Path) -> Path:
    """An approved export of three desk keyframes, written by the box's exporter."""
    episodes = [Episode(id=f"e{n}", camera_id="cam-desk", start=NOW - timedelta(hours=n), captions=[f"caption {n}"],
                        keyframe_refs=[f"frames:cam-desk:e{n}"]) for n in (1, 2, 3)]
    out = tmp_path / "exports"
    exporter = DatasetExporter(FakeMemory(episodes), load_keyframe=lambda ref: b"jpeg " + ref.encode(),
                               publish=lambda event: None, out_dir=out)
    approvals = Approvals({"export_training_data": exporter})
    _, proposal = teach(SPEC, NOW)
    approvals.on_proposed(proposal)
    [done] = asyncio.run(approvals.on_decided(ApprovalDecided(ts=NOW, action_id=proposal.action_id,
                                                              decision="approve")))
    assert done.status == "done", done.result
    [root] = out.iterdir()
    return root


def text_only_export(tmp_path: Path, count: int = 3, ids: list[str] | None = None) -> Path:
    """A dataset folder whose examples carry no picture, written straight from the contract models."""
    root = tmp_path / "text-only"
    root.mkdir()
    names = ids or [f"q{n}" for n in range(count)]
    examples = [DatasetExample(id=name, image="", frame_ref=f"fixture:{name}", ts=NOW, context=f"question {name}")
                for name in names]
    manifest = DatasetManifest(dataset_id="text-only", spec=SPEC, created=NOW, examples=len(examples),
                               action_id="approval-1")
    (root / "manifest.json").write_text(json.dumps(manifest.model_dump(mode="json"), indent=2), encoding="utf-8")
    (root / "examples.jsonl").write_text("".join(e.model_dump_json() + "\n" for e in examples), encoding="utf-8",
                                         newline="\n")
    return root


def rows(root: Path) -> list[dict]:
    text = (root / "examples.jsonl").read_bytes().decode("utf-8")
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def rewrite(root: Path, edit) -> None:
    """Apply edit(rows) to the rows of examples.jsonl and save them."""
    data = rows(root)
    edit(data)
    (root / "examples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in data), encoding="utf-8", newline="\n")


def edit_manifest(root: Path, **changes) -> None:
    data = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    data.update(changes)
    (root / "manifest.json").write_text(json.dumps(data), encoding="utf-8")


def only_row():
    found = ledger.read()
    assert len(found) == 1, found
    return found[0]


def link_dir(link: Path, target: Path) -> None:
    """A directory link that needs no privilege: a junction on Windows, a symlink elsewhere."""
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)
    else:
        os.symlink(target, link, target_is_directory=True)


# --- reading what the box wrote -------------------------------------------------------------

def test_load_reads_an_export_the_boxs_exporter_wrote(tmp_path):
    root = box_export(tmp_path)
    data = load(root)
    assert data.root == root
    assert data.dataset_id == root.name
    assert data.manifest.spec == SPEC
    assert data.manifest.examples == 3
    assert [e.id for e in data.examples] == IDS
    assert [e.context for e in data.examples] == ["caption 1", "caption 2", "caption 3"]
    assert [e.frame_ref for e in data.examples] == [f"frames:cam-desk:e{n}" for n in (1, 2, 3)]
    assert [data.image_path(e).read_bytes() for e in data.examples] == [
        b"jpeg frames:cam-desk:e1", b"jpeg frames:cam-desk:e2", b"jpeg frames:cam-desk:e3"]
    assert [e.id for e in data.pending()] == IDS
    assert data.labeled() == []
    assert check(root) == []


def test_a_byte_order_mark_from_an_editor_is_tolerated(tmp_path):
    root = box_export(tmp_path)
    for name in ("manifest.json", "examples.jsonl"):
        path = root / name
        path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    assert [e.id for e in load(root).examples] == IDS


def test_windows_line_endings_and_blank_lines_are_tolerated(tmp_path):
    root = box_export(tmp_path)
    lines = (root / "examples.jsonl").read_bytes().decode("utf-8").splitlines()
    (root / "examples.jsonl").write_bytes(("\r\n\r\n".join(lines) + "\r\n").encode("utf-8"))
    assert [e.id for e in load(root).examples] == IDS


def test_a_file_that_is_not_utf8_is_a_problem_to_read_and_to_write(tmp_path):
    root = box_export(tmp_path)
    (root / "examples.jsonl").write_bytes(b"\xff\xfe not utf-8\n")
    assert check(root) == ["examples.jsonl: cannot be read (UnicodeDecodeError)"]
    with pytest.raises(DatasetError) as err:
        write_target(root, "e1-0", "label")
    assert err.value.problems == ["examples.jsonl: cannot be read (UnicodeDecodeError)"]
    assert (root / "examples.jsonl").read_bytes() == b"\xff\xfe not utf-8\n"


def test_a_folder_that_does_not_exist_is_one_problem(tmp_path):
    assert check(tmp_path / "nope") == ["nope: not a folder"]
    with pytest.raises(DatasetError) as err:
        load(tmp_path / "nope")
    assert err.value.problems == ["nope: not a folder"]
    assert str(err.value) == "1 problem: nope: not a folder"


def test_a_missing_manifest_and_a_missing_examples_file_are_both_reported(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    assert check(root) == ["manifest.json: missing", "examples.jsonl: missing"]


def test_a_manifest_that_is_not_json_or_lacks_a_field_names_the_file_and_the_field(tmp_path):
    root = box_export(tmp_path)
    (root / "manifest.json").write_text("{not json", encoding="utf-8")
    [problem] = check(root)
    assert problem.startswith("manifest.json: ") and "JSON" in problem

    data = json.loads(json.dumps({"dataset_id": "x", "spec": {"task": "t", "instruction": "i"},
                                  "created": NOW.isoformat(), "examples": 3}))
    (root / "manifest.json").write_text(json.dumps(data), encoding="utf-8")
    assert check(root) == ["manifest.json: action_id: Field required"]


def test_an_invalid_example_line_names_its_line_and_field_and_never_quotes_the_value(tmp_path):
    root = box_export(tmp_path)

    def edit(data):
        data[1]["verdict"] = "household secret"
        data[2]["ts"] = "a note about the kitchen"
    rewrite(root, edit)
    problems = check(root)
    assert len(problems) == 2
    assert problems[0].startswith("examples.jsonl line 2: verdict: ")
    assert problems[1].startswith("examples.jsonl line 3: ts: ")
    assert "secret" not in " ".join(problems) and "kitchen" not in " ".join(problems)


def test_a_line_that_is_not_json_is_reported_by_line_number(tmp_path):
    root = box_export(tmp_path)
    lines = (root / "examples.jsonl").read_text(encoding="utf-8").split("\n")
    lines[1] = "{broken"
    (root / "examples.jsonl").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    [problem] = check(root)
    assert problem.startswith("examples.jsonl line 2: ") and "JSON" in problem


def test_the_manifest_example_count_must_match_the_lines(tmp_path):
    root = box_export(tmp_path)
    edit_manifest(root, examples=5)
    assert check(root) == ["manifest.json: says 5 examples but examples.jsonl has 3"]


def test_an_export_without_the_approval_that_allowed_it_is_refused(tmp_path):
    root = box_export(tmp_path)
    edit_manifest(root, action_id="  ")
    assert check(root) == ["manifest.json: action_id is empty; an export is read only with the approval that "
                           "allowed it"]


def test_duplicate_and_empty_example_ids_are_reported(tmp_path):
    root = box_export(tmp_path)

    def edit(data):
        data[2]["id"] = data[0]["id"]
        data[1]["id"] = " "
    rewrite(root, edit)
    assert check(root) == ["examples.jsonl line 2 (id ' '): the id is empty",
                           "examples.jsonl line 3 (id 'e1-0'): the id repeats line 1"]


def test_a_missing_picture_is_reported(tmp_path):
    root = box_export(tmp_path)
    (root / "images" / "0001.jpg").unlink()
    assert check(root) == ["examples.jsonl line 2 (id 'e2-0'): image is not a file in the dataset folder"]


@pytest.mark.parametrize("image", ["../outside.jpg", "images/../../outside.jpg", "/etc/hosts", "C:/Windows/x.jpg",
                                   "images\\0000.jpg"])
def test_an_image_path_that_leaves_the_dataset_folder_is_refused(tmp_path, image):
    root = box_export(tmp_path)
    (tmp_path / "exports" / "outside.jpg").write_bytes(b"x")
    rewrite(root, lambda data: data[0].update(image=image))
    assert check(root) == ["examples.jsonl line 1 (id 'e1-0'): image is not a relative path inside the dataset "
                           "folder"]


def test_an_image_reached_through_a_link_out_of_the_folder_is_refused(tmp_path):
    root = box_export(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "pic.jpg").write_bytes(b"x")
    link_dir(root / "images" / "out", elsewhere)
    rewrite(root, lambda data: data[0].update(image="images/out/pic.jpg"))
    assert check(root) == ["examples.jsonl line 1 (id 'e1-0'): image leaves the dataset folder through a link"]


def test_every_problem_is_reported_at_once_and_the_error_lists_the_first_five(tmp_path):
    root = box_export(tmp_path)
    edit_manifest(root, examples=9, action_id="")
    (root / "images" / "0002.jpg").unlink()
    problems = check(root)
    assert problems == [
        "manifest.json: says 9 examples but examples.jsonl has 3",
        "manifest.json: action_id is empty; an export is read only with the approval that allowed it",
        "examples.jsonl line 3 (id 'e3-0'): image is not a file in the dataset folder"]
    rewrite(root, lambda data: [row.update(id="same") for row in data])
    with pytest.raises(DatasetError) as err:
        load(root)
    assert len(err.value.problems) == 5
    assert str(err.value).startswith("5 problems: ") and not str(err.value).endswith("...")
    (root / "images" / "0000.jpg").unlink()
    with pytest.raises(DatasetError) as err:
        load(root)
    assert str(err.value).endswith("; ...") and len(err.value.problems) == 6


def test_a_text_only_dataset_needs_the_flag_and_has_no_image_paths(tmp_path):
    root = text_only_export(tmp_path)
    problems = check(root)
    assert problems == [f"examples.jsonl line {n} (id 'q{n - 1}'): has no image (a text-only dataset is read with "
                        "require_images=False)" for n in (1, 2, 3)]
    data = load(root, require_images=False)
    assert [data.image_path(e) for e in data.examples] == [None, None, None]
    assert check(root, require_images=False) == []


def test_extra_files_in_the_folder_do_not_matter(tmp_path):
    root = box_export(tmp_path)
    (root / "split.json").write_text("{}", encoding="utf-8")
    (root / "expected.jsonl").write_text("not read\n", encoding="utf-8")
    assert check(root) == []


# --- labels written back ----------------------------------------------------------------------

def test_a_target_is_filled_and_every_other_line_is_left_byte_for_byte(tmp_path):
    root = box_export(tmp_path)
    before = (root / "examples.jsonl").read_bytes().split(b"\n")
    assert write_target(root, "e2-0", '{"objects": ["keys"]}') == Written(1, 0)
    after = (root / "examples.jsonl").read_bytes().split(b"\n")
    assert after[0] == before[0] and after[2] == before[2] and after[3] == before[3] == b""
    changed = json.loads(after[1])
    assert changed == {**json.loads(before[1]), "target": '{"objects": ["keys"]}'}
    data = load(root)
    assert [e.id for e in data.labeled()] == ["e2-0"]
    assert [e.id for e in data.pending()] == ["e1-0", "e3-0"]
    assert data.labeled()[0].target == '{"objects": ["keys"]}'


def test_fields_the_contract_does_not_know_yet_survive_a_write(tmp_path):
    root = box_export(tmp_path)
    rewrite(root, lambda data: data[0].update(future={"nested": [1, 2]}, note="kept"))
    write_target(root, "e1-0", "label")
    row = rows(root)[0]
    assert (row["future"], row["note"], row["target"]) == ({"nested": [1, 2]}, "kept", "label")


def test_several_targets_are_written_in_one_replace(tmp_path):
    root = box_export(tmp_path)
    assert write_targets(root, {"e1-0": "a", "e3-0": "b"}) == Written(2, 0)
    assert [(r["id"], r["target"]) for r in rows(root)] == [("e1-0", "a"), ("e2-0", None), ("e3-0", "b")]


def test_the_same_label_again_changes_nothing(tmp_path):
    root = box_export(tmp_path)
    write_target(root, "e1-0", "label")
    before = (root / "examples.jsonl").read_bytes()
    assert write_target(root, "e1-0", "label") == Written(0, 1)
    assert (root / "examples.jsonl").read_bytes() == before


def test_a_different_label_needs_overwrite(tmp_path):
    root = box_export(tmp_path)
    write_target(root, "e1-0", "first")
    before = (root / "examples.jsonl").read_bytes()
    with pytest.raises(DatasetError) as err:
        write_target(root, "e1-0", "second")
    assert err.value.problems == ["id 'e1-0': already has a different target (overwrite=True replaces it)"]
    assert (root / "examples.jsonl").read_bytes() == before
    assert write_target(root, "e1-0", "second", overwrite=True) == Written(1, 0)
    assert rows(root)[0]["target"] == "second"


def test_an_unknown_id_refuses_the_whole_call(tmp_path):
    root = box_export(tmp_path)
    before = (root / "examples.jsonl").read_bytes()
    with pytest.raises(DatasetError) as err:
        write_targets(root, {"e1-0": "a", "nope": "b"})
    assert err.value.problems == ["id 'nope': not in examples.jsonl"]
    assert (root / "examples.jsonl").read_bytes() == before


@pytest.mark.parametrize("label", ["", "   \n", None, 7, {"answer": "x"}])
def test_a_label_that_is_not_non_blank_text_is_refused(tmp_path, label):
    root = box_export(tmp_path)
    before = (root / "examples.jsonl").read_bytes()
    with pytest.raises(DatasetError) as err:
        write_targets(root, {"e1-0": label})
    assert err.value.problems == ["id 'e1-0': the target must be non-blank text"]
    assert (root / "examples.jsonl").read_bytes() == before


def test_a_blank_target_counts_as_not_yet_labeled(tmp_path):
    root = box_export(tmp_path)
    rewrite(root, lambda data: data[0].update(target="  "))
    data = load(root)
    assert not has_target(data.examples[0]) and [e.id for e in data.pending()] == IDS
    assert write_target(root, "e1-0", "label") == Written(1, 0)


def test_an_example_the_contract_would_refuse_is_not_written_but_the_others_can_be(tmp_path):
    root = box_export(tmp_path)
    rewrite(root, lambda data: data[1].update(ts="not a time"))
    before = (root / "examples.jsonl").read_bytes()
    with pytest.raises(DatasetError) as err:
        write_target(root, "e2-0", "label")
    assert len(err.value.problems) == 1 and err.value.problems[0].startswith("examples.jsonl line 2 (id 'e2-0'): ts: ")
    assert (root / "examples.jsonl").read_bytes() == before
    assert write_target(root, "e1-0", "label") == Written(1, 0)
    assert rows(root)[0]["target"] == "label" and rows(root)[1]["ts"] == "not a time"


def test_a_file_with_a_line_that_is_not_an_object_with_an_id_is_not_written(tmp_path):
    root = box_export(tmp_path)
    lines = (root / "examples.jsonl").read_text(encoding="utf-8").split("\n")
    lines[1], lines[2] = "{broken", '["no id"]'
    (root / "examples.jsonl").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    before = (root / "examples.jsonl").read_bytes()
    with pytest.raises(DatasetError) as err:
        write_target(root, "e1-0", "label")
    assert err.value.problems == ["examples.jsonl line 2: not JSON", "examples.jsonl line 3: not an object with a text id"]
    assert (root / "examples.jsonl").read_bytes() == before


def test_repeated_ids_in_the_file_refuse_a_write(tmp_path):
    root = box_export(tmp_path)
    rewrite(root, lambda data: data[2].update(id="e1-0"))
    with pytest.raises(DatasetError) as err:
        write_target(root, "e2-0", "label")
    assert err.value.problems == ["examples.jsonl line 3 (id 'e1-0'): the id repeats line 1"]


def test_a_folder_without_examples_or_that_does_not_exist_is_refused_and_not_created(tmp_path):
    with pytest.raises(DatasetError) as err:
        write_target(tmp_path / "nope", "e1-0", "label")
    assert err.value.problems == ["nope: not a folder"]
    assert not (tmp_path / "nope").exists()
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(DatasetError) as err:
        write_target(empty, "e1-0", "label")
    assert err.value.problems == ["examples.jsonl: missing"]


def test_writing_nothing_takes_no_lock_and_touches_nothing(tmp_path):
    root = box_export(tmp_path)
    before = sorted(p.name for p in root.iterdir())
    assert write_targets(root, {}) == Written(0, 0)
    assert sorted(p.name for p in root.iterdir()) == before


def test_non_ascii_labels_are_stored_as_utf8_text(tmp_path):
    root = box_export(tmp_path)
    write_target(root, "e3-0", "café 机")
    assert b"caf\xc3\xa9 \xe6\x9c\xba" in (root / "examples.jsonl").read_bytes()
    assert load(root).labeled()[0].target == "café 机"


def test_a_file_saved_with_windows_line_endings_keeps_them(tmp_path):
    root = box_export(tmp_path)
    lines = (root / "examples.jsonl").read_bytes().decode("utf-8").split("\n")
    (root / "examples.jsonl").write_bytes("\r\n".join(lines).encode("utf-8"))
    write_target(root, "e2-0", "label")
    raw = (root / "examples.jsonl").read_bytes()
    assert raw.count(b"\r\n") == 3 and raw.count(b"\n") == 3
    assert load(root).labeled()[0].target == "label"


def test_a_crashed_labeling_run_resumes_with_only_the_examples_left(tmp_path):
    root = box_export(tmp_path)
    asked: list[str] = []

    def label_all(crash_at: str | None) -> None:
        for example in load(root).pending():
            if example.id == crash_at:
                raise RuntimeError("the process was killed")
            asked.append(example.id)
            write_target(root, example.id, f"label of {example.id}")

    with pytest.raises(RuntimeError, match="killed"):
        label_all(crash_at="e3-0")
    assert asked == ["e1-0", "e2-0"]
    assert [e.id for e in load(root).pending()] == ["e3-0"]
    label_all(crash_at=None)
    assert asked == ["e1-0", "e2-0", "e3-0"]
    assert [(e.id, e.target) for e in load(root).examples] == [(i, f"label of {i}") for i in IDS]
    assert load(root).pending() == []


def test_a_failure_while_replacing_the_file_leaves_the_old_file_and_no_temporary_file(tmp_path, monkeypatch):
    root = box_export(tmp_path)
    before_bytes = (root / "examples.jsonl").read_bytes()
    before_names = sorted(p.name for p in root.iterdir())

    def disk_full(source, destination):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(dataset.os, "replace", disk_full)
    with pytest.raises(OSError, match="No space left"):
        write_target(root, "e1-0", "label")
    assert (root / "examples.jsonl").read_bytes() == before_bytes
    assert sorted(p.name for p in root.iterdir()) == before_names + [dataset.LOCK_NAME] or \
        sorted(p.name for p in root.iterdir()) == sorted(before_names + [dataset.LOCK_NAME])


def test_a_busy_file_is_retried_and_then_the_label_lands(tmp_path, monkeypatch):
    root = box_export(tmp_path)
    real_replace, attempts = os.replace, []

    def busy_twice(source, destination):
        attempts.append(source)
        if len(attempts) <= 2:
            raise PermissionError(13, "in use by another process")
        real_replace(source, destination)

    monkeypatch.setattr(dataset.os, "replace", busy_twice)
    assert write_target(root, "e1-0", "label") == Written(1, 0)
    assert len(attempts) == 3
    assert rows(root)[0]["target"] == "label"


def test_a_file_that_stays_busy_fails_after_the_last_try_and_loses_nothing(tmp_path, monkeypatch):
    root = box_export(tmp_path)
    before = (root / "examples.jsonl").read_bytes()
    attempts = []

    def always_busy(source, destination):
        attempts.append(source)
        raise PermissionError(13, "in use by another process")

    monkeypatch.setattr(dataset.os, "replace", always_busy)
    monkeypatch.setattr(dataset.time, "sleep", lambda seconds: None)
    with pytest.raises(PermissionError):
        write_target(root, "e1-0", "label")
    assert len(attempts) == dataset.REPLACE_TRIES
    assert (root / "examples.jsonl").read_bytes() == before
    assert not [p for p in root.iterdir() if p.suffix == ".tmp"]


def test_writers_at_the_same_moment_lose_no_labels(tmp_path):
    ids = [f"q{n:02d}" for n in range(80)]
    root = text_only_export(tmp_path, ids=ids)
    failures: list[BaseException] = []

    def label(chunk: list[str]) -> None:
        try:
            for name in chunk:
                write_target(root, name, f"label of {name}")
        except BaseException as exc:  # surfaced below, in the test thread
            failures.append(exc)

    threads = [threading.Thread(target=label, args=(ids[n::8],)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    assert failures == []
    assert [(e.id, e.target) for e in load(root, require_images=False).examples] == [
        (name, f"label of {name}") for name in ids]


def test_a_labeler_that_cannot_get_the_lock_times_out_and_writes_nothing(tmp_path):
    root = box_export(tmp_path)
    before = (root / "examples.jsonl").read_bytes()
    with FileLock(root / dataset.LOCK_NAME, timeout=1.0):
        with pytest.raises(LockTimeout):
            write_target(root, "e1-0", "label", timeout=0.2)
    assert (root / "examples.jsonl").read_bytes() == before
    assert write_target(root, "e1-0", "label") == Written(1, 0)


# --- the entry point -----------------------------------------------------------------------

def test_the_entry_point_counts_a_good_export_and_leaves_a_ledger_row(tmp_path):
    root = box_export(tmp_path)
    write_target(root, "e1-0", "label")
    answer = dataset.main([str(root)])
    assert answer == {"ok": True, "outcome": "ok", "dataset_id": root.name, "examples": 3, "labeled": 1,
                      "pending": 2, "problems": [], "refs": {"dataset_id": root.name}}
    row = only_row()
    assert (row["verb"], row["outcome"], row["actor"], row["tier"]) == ("dataset-check", "ok", "sonnet-test", "sonnet")
    assert row["refs"]["dataset_id"] == root.name and row["duration_s"] >= 0


def test_the_entry_point_reports_a_bad_export_as_invalid_and_the_row_says_so(tmp_path):
    root = box_export(tmp_path)
    edit_manifest(root, examples=7)
    answer = dataset.main([str(root)])
    assert answer["ok"] is False and answer["outcome"] == "invalid"
    assert answer["problems"] == ["manifest.json: says 7 examples but examples.jsonl has 3"]
    assert answer["error"] == "1 problem: manifest.json: says 7 examples but examples.jsonl has 3"
    row = only_row()
    assert (row["verb"], row["outcome"], row["note"]) == ("dataset-check", "invalid", answer["error"])


def test_the_entry_point_takes_text_only_datasets_with_the_flag(tmp_path):
    root = text_only_export(tmp_path)
    assert dataset.main([str(root)])["ok"] is False
    answer = dataset.main([str(root), "--text-only"])
    assert (answer["ok"], answer["examples"], answer["pending"]) == (True, 3, 3)


def test_the_entry_point_refuses_to_run_while_the_machine_is_stopped(tmp_path, capsys):
    root = box_export(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    (state / "STOP").write_text("maintenance\n", encoding="utf-8")
    with pytest.raises(stop.Stopped):
        dataset.main([str(root)])
    assert json.loads(capsys.readouterr().out)["outcome"] == "stopped"
    row = only_row()
    assert (row["verb"], row["outcome"], row["note"]) == ("dataset-check", "stopped", "maintenance")


def test_run_by_path_from_another_folder_the_script_finds_its_package_and_prints_one_json_line(tmp_path):
    root = box_export(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    done = subprocess.run([sys.executable, str(SCRIPT), str(root)], cwd=elsewhere, capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 0, done.stderr
    answer = json.loads(done.stdout)
    assert (answer["ok"], answer["dataset_id"], answer["examples"]) == (True, root.name, 3)
    assert done.stderr == ""
    assert only_row()["verb"] == "dataset-check"


def test_run_by_path_on_a_bad_folder_exits_1(tmp_path):
    done = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path / "nope")], cwd=tmp_path, capture_output=True,
                          text=True, timeout=120)
    assert done.returncode == 1
    assert json.loads(done.stdout)["problems"] == ["nope: not a folder"]
