"""lab/label_fallback.py: the reference traces turned into labels (the cut-off route of owner ruling 12), and a
teacher run published under the same layout. The fixture is always read, never written; every output goes to a
temporary folder."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import jsonschema
import pytest

from lab import format_traces, label, label_fallback
from lab.dataset import load, load_split
from lab.fixture_dataset import load_expected
from lab.machine import ledger
from lab.teacher_fake import FakeTeacher, reference_policy

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "fixtures" / "learning" / "tool-calling-fixture"
COMMITTED = REPO / "fixtures" / "learning" / "tool-calling-labels"
USER = "acct-c"
TOKEN = "hubtoken-" + "k" * 32


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LAB_ACTOR", "test-labeler")
    monkeypatch.setenv("LAB_ENV_FILES", "")
    for name in ("LAB_TASK", "LAB_TIER", "LAB_HUB_URL", "LAB_HUB_USER_AGENT", "JUPYTERHUB_USER",
                 "JUPYTERHUB_API_TOKEN"):
        monkeypatch.delenv(name, raising=False)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_copy(tmp_path, where="fixture") -> Path:
    target = tmp_path / where / FIXTURE.name
    shutil.copytree(FIXTURE, target)
    return target


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --- the reference traces as labels -------------------------------------------------------------

def test_the_reference_traces_become_a_label_for_every_example_that_follows_the_schema(tmp_path):
    out = tmp_path / "labels"
    before = {name: sha(FIXTURE / name) for name in ("expected.jsonl", "examples.jsonl", "manifest.json", "split.json")}

    manifest = label_fallback.convert(FIXTURE, out)

    assert {name: sha(FIXTURE / name) for name in before} == before    # the fixture, expected.jsonl included, is untouched
    assert not (out / "expected.jsonl").exists()
    data = load(out, require_images=False)
    assert (len(data.examples), len(data.labeled()), len(data.pending())) == (142, 142, 0)
    schema = data.manifest.spec.output_schema
    expected = load_expected(FIXTURE)
    for example in data.examples:
        value = json.loads(example.target)
        jsonschema.validate(value, schema)
        reference = expected[example.id]
        assert value["answer"] == reference.label.answer
        assert [call["name"] for call in value["tool_calls"]] == [call.name for call in reference.label.tool_calls]
    assert manifest["source"] == "reference traces, teacher unavailable"
    assert manifest["teacher"] is None
    assert manifest["counts"] == {"asked": 0, "accepted": 142, "rejected": {}, "failed": {}, "examples": 142,
                                  "labeled": 142, "unlabeled": 0}
    assert manifest["input"]["examples_sha256"] == before["examples.jsonl"]
    assert manifest["input"]["examples"] == 142 and manifest["input"]["groups"] == 16
    assert manifest["reference"]["sha256"] == before["expected.jsonl"]
    assert json.loads((out / "label_manifest.json").read_text(encoding="utf-8")) == manifest
    assert (out / "rejects.jsonl").read_text(encoding="utf-8") == ""
    for name, digest in manifest["files"].items():
        assert sha(out / name) == digest


def test_the_traces_hold_each_call_with_its_result_and_the_split_and_manifest_are_the_fixtures(tmp_path):
    out = tmp_path / "labels"
    label_fallback.convert(FIXTURE, out)

    traces = {row["id"]: row for row in lines(out / "traces.jsonl")}
    expected = load_expected(FIXTURE)
    assert set(traces) == set(expected)
    for example_id, row in expected.items():
        trace = traces[example_id]
        assert [(call["name"], call["arguments"], call["result"]) for call in trace["tool_calls"]] == [
            (call.name, call.arguments, result) for call, result in zip(row.label.tool_calls, row.tool_results)]
        assert trace["answer"] == row.label.answer
    for name in ("manifest.json", "split.json"):
        assert (out / name).read_bytes() == (FIXTURE / name).read_bytes()
    assert load_split(out, require_images=False).sha256 == load_split(FIXTURE, require_images=False).sha256


def test_the_converted_folder_goes_through_the_trace_formatter_like_a_teacher_labeled_one(tmp_path):
    out = tmp_path / "labels"
    label_fallback.convert(FIXTURE, out)

    answer = format_traces.main([str(out), "--out", str(tmp_path / "rows.jsonl"), "--split"])

    assert answer["ok"] is True
    assert answer["metrics"]["format.rejected"] == 0 and answer["metrics"]["format.failed"] == 0
    assert answer["metrics"]["format.rows"] == len(load_split(out, require_images=False).train)


def test_a_reference_row_that_breaks_the_schema_is_left_unlabeled_and_logged_never_repaired(tmp_path):
    source = fixture_copy(tmp_path)
    rows = lines(source / "expected.jsonl")
    rows[0]["label"]["tool_calls"][0]["name"] = "delete_memory"       # a tool the task does not allow
    rows[1]["tool_results"] = []                                      # a call with no result
    rows[2]["label"]["answer"] = "  "                                 # nothing said
    rows.pop(3)                                                       # no reference at all
    (source / "expected.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    out = tmp_path / "labels"

    manifest = label_fallback.convert(source, out)

    assert manifest["counts"]["accepted"] == 138
    assert manifest["counts"]["unlabeled"] == 4
    assert manifest["counts"]["rejected"] == {"empty": 1, "no_reference": 1, "results_mismatch": 1, "schema": 1}
    rejected = {row["id"]: row for row in lines(out / "rejects.jsonl")}
    assert set(rejected) == {rows[0]["id"], rows[1]["id"], rows[2]["id"], "golden-01-3"}
    assert rejected[rows[0]["id"]]["reason"] == "schema"
    stored = {example.id: example.target for example in load(out, require_images=False).examples}
    assert all(stored[key] is None for key in rejected)
    assert {row["id"] for row in lines(out / "traces.jsonl")}.isdisjoint(rejected)


def test_a_folder_that_is_not_empty_is_not_overwritten_and_a_fixture_without_references_is_refused(tmp_path):
    out = tmp_path / "labels"
    out.mkdir()
    (out / "keep.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(label_fallback.LabelsError, match="not empty"):
        label_fallback.convert(FIXTURE, out)
    assert (out / "keep.txt").read_text(encoding="utf-8") == "mine"

    source = fixture_copy(tmp_path)
    (source / "expected.jsonl").unlink()
    with pytest.raises(label_fallback.LabelsError, match="no reference traces"):
        label_fallback.convert(source, tmp_path / "other")


def test_the_command_line_prints_one_json_line_and_exits_clean(tmp_path, capsys):
    answer = label_fallback.main(["convert", str(FIXTURE), str(tmp_path / "labels")])

    assert answer["ok"] is True and answer["counts"]["labeled"] == 142
    assert answer["source"] == "reference traces, teacher unavailable"
    refused = label_fallback.main(["convert", str(FIXTURE), str(tmp_path / "labels")])
    assert refused["ok"] is False and refused["outcome"] == "refused"
    row = ledger.read(state_dir=tmp_path / "state", verb="label-fallback")[0]
    assert row["metrics"]["labels.labeled"] == 142 and row["outcome"] == "ok"


# --- a teacher run, published ----------------------------------------------------------------

def teacher_run(tmp_path):
    labeled = fixture_copy(tmp_path, "labeling")
    expected = load_expected(FIXTURE)
    fake = FakeTeacher(reference_policy(expected), user=USER, token=TOKEN)
    with label.Teacher.from_hub(fake.account("C"), transport=fake.transport, sleep=lambda seconds: None) as teacher:
        result = label.label_dataset(labeled, teacher, workers=4)
    return labeled, result


def test_a_teacher_run_is_published_with_its_ledger_rows_counts_and_means(tmp_path):
    labeled, result = teacher_run(tmp_path)
    assert result.accepted == 142
    out = tmp_path / "labels"

    manifest = label_fallback.package(labeled, FIXTURE, out, state_dir=tmp_path / "state", run_ids=[result.run_id],
                                      teacher_info={"flags_seen": {"tool_calls": "structured"}})

    assert manifest["source"] == "teacher"
    assert manifest["teacher"]["model"] == "teacher"
    assert manifest["teacher"]["checkpoint"] == result.checkpoint
    assert manifest["teacher"]["flags_seen"] == {"tool_calls": "structured"}
    assert manifest["decode"]["thinking"] is False and manifest["decode"]["temperature"] == 0.0
    assert manifest["runs"] == [result.run_id]
    assert manifest["counts"] == {"asked": 142, "accepted": 142, "rejected": {}, "failed": {}, "examples": 142,
                                  "labeled": 142, "unlabeled": 0}
    assert manifest["input"]["examples_sha256"] == sha(FIXTURE / "examples.jsonl")
    for key in ("label.seconds.mean", "label.prompt_tokens.mean", "label.completion_tokens.mean",
                "label.tokens.mean"):
        assert manifest["per_example"][key] > 0
    exported = lines(out / "label_ledger.jsonl")
    assert len(exported) == 142 and {row["verb"] for row in exported} == {"label-example"}
    assert len(lines(out / "traces.jsonl")) == 142
    assert "wall_seconds.total" not in manifest["per_example"]   # label_dataset writes no run row; main() does
    assert ".examples.lock" not in manifest["files"] and not (out / ".examples.lock").exists()
    assert (out / "rejects.jsonl").read_text(encoding="utf-8") == ""
    assert (out / "examples.jsonl").read_bytes() == (labeled / "examples.jsonl").read_bytes()
    assert (FIXTURE / "examples.jsonl").read_text(encoding="utf-8").count('"target":null') == 142


def test_an_example_the_teacher_did_not_label_is_listed_with_its_reason(tmp_path):
    labeled = fixture_copy(tmp_path, "labeling")
    fake = FakeTeacher(reference_policy(load_expected(FIXTURE)), user=USER, token=TOKEN, tool_parser=False)
    with label.Teacher.from_hub(fake.account("C"), transport=fake.transport, sleep=lambda seconds: None) as teacher:
        result = label.label_dataset(labeled, teacher, limit=3, max_failures=99)
    assert result.accepted == 0 and result.rejected["tool_markup"] == 3
    out = tmp_path / "labels"

    manifest = label_fallback.package(labeled, FIXTURE, out, state_dir=tmp_path / "state")

    assert manifest["counts"]["rejected"] == {"tool_markup": 3}
    assert manifest["counts"]["labeled"] == 0 and manifest["counts"]["unlabeled"] == 142
    reasons = [row["reason"] for row in lines(out / "rejects.jsonl")]
    assert reasons.count("tool_markup") == 3 and reasons.count("not_asked") == 139
    answer = label_fallback.main(["package", str(labeled), str(FIXTURE), str(tmp_path / "again"),
                                  "--state-dir", str(tmp_path / "state")])
    assert answer["ok"] is False and answer["outcome"] == "incomplete"


def test_a_target_that_breaks_the_schema_is_not_published(tmp_path):
    labeled, result = teacher_run(tmp_path)
    broken = lines(labeled / "examples.jsonl")
    value = json.loads(broken[0]["target"])
    value["tool_calls"][0]["name"] = "delete_memory"
    broken[0]["target"] = json.dumps(value)
    (labeled / "examples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in broken), encoding="utf-8")

    manifest = label_fallback.package(labeled, FIXTURE, tmp_path / "labels", state_dir=tmp_path / "state")

    assert manifest["counts"]["labeled"] == 141
    assert lines(tmp_path / "labels" / "rejects.jsonl")[0]["reason"] == "schema"


# --- the committed folder -----------------------------------------------------------------------

@pytest.mark.skipif(not COMMITTED.is_dir(), reason="the labels folder is not committed in this checkout")
def test_the_committed_labels_folder_is_whole_and_its_manifest_matches_the_files_and_the_input():
    manifest = json.loads((COMMITTED / "label_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"] in (label_fallback.SOURCE_TEACHER, label_fallback.SOURCE_REFERENCE)
    for name, digest in manifest["files"].items():
        assert sha(COMMITTED / name) == digest, name
    assert manifest["input"]["examples_sha256"] == sha(FIXTURE / "examples.jsonl")
    assert manifest["input"]["manifest_sha256"] == sha(FIXTURE / "manifest.json")
    data = load(COMMITTED, require_images=False)
    assert manifest["counts"]["labeled"] == len(data.labeled())
    assert manifest["counts"]["unlabeled"] == 0, "every example has a target"
    schema = data.manifest.spec.output_schema
    for example in data.examples:
        jsonschema.validate(json.loads(example.target), schema)
    assert (COMMITTED / "rejects.jsonl").is_file()
    if manifest["source"] == label_fallback.SOURCE_TEACHER:
        assert manifest["teacher"]["checkpoint"] and manifest["teacher"]["model"] == "teacher"
        assert manifest["per_example"]["label.seconds.mean"] > 0
    else:
        assert manifest["teacher"] is None
