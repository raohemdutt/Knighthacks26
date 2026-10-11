"""lab/label_agreement.py against the fake teacher: 30 labeled examples are asked again with another seed and a
reworded prompt, compared field by field, and the agreement rate, the disagreements and a verdict are reported and
recorded. The fixture is labeled once by the reference fake; every dataset here is a copy in a temporary folder."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from lab import __main__ as dispatcher
from lab import label, label_agreement
from lab.dataset import load
from lab.fixture_dataset import load_expected
from lab.label_agreement import DEFAULT_THRESHOLD, REWORD_PREFIX
from lab.machine import ledger
from lab.teacher_fake import FakeTeacher, reference_policy, swayed_policy

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "fixtures" / "learning" / "tool-calling-fixture"
RUBRIC = REPO / "lab" / "refine" / "learning" / "RUBRIC.md"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LAB_ACTOR", "test-agreement")
    monkeypatch.setenv("LAB_ENV_FILES", "")
    for name in ("LAB_TASK", "LAB_TIER", "LAB_HUB_URL", "JUPYTERHUB_USER", "JUPYTERHUB_API_TOKEN"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="module")
def labeled_master(tmp_path_factory):
    """The fixture labeled end to end by the reference fake, kept to copy from."""
    folder = tmp_path_factory.mktemp("master") / FIXTURE.name
    shutil.copytree(FIXTURE, folder)
    fake = FakeTeacher(reference_policy(load_expected(FIXTURE)))
    with label.Teacher(fake.base_url, transport=fake.transport) as teacher:
        run = label.label_dataset(folder, teacher, state_dir=folder.parent / "state")
    assert run.pending_after == 0
    return folder


@pytest.fixture()
def labeled(labeled_master, tmp_path):
    folder = tmp_path / "work" / FIXTURE.name
    shutil.copytree(labeled_master, folder)
    return folder


def questions_of(folder, ids):
    by_id = {example.id: example for example in load(folder, require_images=False).examples}
    return [by_id[i].context for i in ids]


def second_teacher(fake):
    return label.Teacher(fake.base_url, transport=fake.transport)


def agree(folder, fake, **kwargs):
    with second_teacher(fake) as teacher:
        return label_agreement.check_agreement(folder, teacher, **kwargs)


def reference_fake(policy=None):
    return FakeTeacher(policy or reference_policy(load_expected(FIXTURE)))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- clause: 30 labeled examples, a different seed and a reworded prompt --------------------

def test_takes_the_first_30_labeled_examples(labeled):
    report = agree(labeled, reference_fake())
    first = [e.id for e in load(labeled, require_images=False).labeled()][:30]
    assert report["sample"]["ids"] == first
    assert report["sample"]["size"] == 30 and report["sample"]["mode"] == "first"


def test_seeded_sample_is_30_distinct_labeled_ids_and_repeatable(labeled):
    one = agree(labeled, reference_fake(), sample="seeded", sample_seed=7)
    two = agree(labeled, reference_fake(), sample="seeded", sample_seed=7)
    other = agree(labeled, reference_fake(), sample="seeded", sample_seed=8)
    ids = one["sample"]["ids"]
    assert len(ids) == len(set(ids)) == 30
    assert ids == two["sample"]["ids"] != other["sample"]["ids"]
    assert set(ids) <= {e.id for e in load(labeled, require_images=False).labeled()}
    assert one["sample"]["seed"] == 7


def test_second_pass_uses_another_seed_and_reworded_prompt(labeled):
    fake = reference_fake()
    report = agree(labeled, fake, second_seed=5)
    chats = [seen.body for seen in fake.requests if seen.path.endswith("/v1/chat/completions")]
    assert chats and all(body["seed"] == 5 for body in chats)
    systems = [m["content"] for body in chats for m in body["messages"] if m["role"] == "system"]
    assert systems and all(text.startswith(REWORD_PREFIX) for text in systems)
    assert report["second_pass"] == {"seed": 5, "first_seed": 0, "rewording": REWORD_PREFIX.strip()}


def test_second_pass_with_the_same_seed_as_the_first_is_refused(labeled):
    with pytest.raises(label_agreement.AgreementRefused, match="seed"):
        agree(labeled, reference_fake(), second_seed=0, first_seed=0)


def test_source_labels_are_not_touched(labeled):
    before = digest(labeled / "examples.jsonl")
    agree(labeled, reference_fake())
    assert digest(labeled / "examples.jsonl") == before


def test_only_labeled_examples_are_asked(labeled_master, tmp_path):
    folder = tmp_path / "w" / FIXTURE.name
    shutil.copytree(labeled_master, folder)
    rows = [json.loads(line) for line in (folder / "examples.jsonl").read_text().splitlines()]
    for row in rows[:5]:
        row["target"] = None
    (folder / "examples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    fake = reference_fake()
    report = agree(folder, fake)
    assert not set(report["sample"]["ids"]) & {r["id"] for r in rows[:5]}


# --- clause: compared field by field, rate and disagreements listed ---------------------------

def test_identical_passes_agree_on_every_field(labeled):
    report = agree(labeled, reference_fake())
    assert report["overall"]["rate"] == 1.0 and report["disagreements"] == []
    assert set(report["fields"]) == {"tool_calls", "answer"}
    assert all(f["rate"] == 1.0 and f["total"] == 30 for f in report["fields"].values())


def test_disagreements_list_id_field_first_and_second_value(labeled):
    ids = [e.id for e in load(labeled, require_images=False).labeled()][:30]
    swayed = ids[:3]
    fake = reference_fake(swayed_policy(reference_policy(load_expected(FIXTURE)), questions_of(labeled, swayed)))
    report = agree(labeled, fake)
    got = report["disagreements"]
    assert {d["id"] for d in got} == set(swayed) and {d["field"] for d in got} == {"answer"}
    first = {e.id: json.loads(e.target) for e in load(labeled, require_images=False).labeled()}
    for d in got:
        assert d["first"] == first[d["id"]]["answer"]
        assert d["second"] == first[d["id"]]["answer"] + " (not sure)"
    assert report["fields"]["answer"]["agree"] == 27 and report["fields"]["answer"]["total"] == 30
    assert report["fields"]["tool_calls"]["rate"] == 1.0
    assert report["overall"] == {"agree": 57, "total": 60, "rate": pytest.approx(57 / 60)}


def test_a_changed_tool_call_is_a_disagreement_in_that_field(labeled):
    ids = [e.id for e in load(labeled, require_images=False).labeled()][:30]
    questions = questions_of(labeled, ids[:2])
    fake = reference_fake(swayed_policy(reference_policy(load_expected(FIXTURE)), questions, fields=("tool_calls",)))
    report = agree(labeled, fake)
    assert {d["field"] for d in report["disagreements"]} <= {"tool_calls"}
    assert report["disagreements"], "a changed call must show up (or be dropped as unanswered)"


def test_example_the_second_pass_could_not_label_counts_against_agreement(labeled):
    ids = [e.id for e in load(labeled, require_images=False).labeled()][:30]
    asked = questions_of(labeled, ids[:1])[0]

    def policy(body):
        from lab.teacher_fake import Turn, _question
        if _question(body) == asked:
            return Turn(content="")
        return reference_policy(load_expected(FIXTURE))(body)

    report = agree(labeled, reference_fake(policy))
    assert report["unanswered"] == [{"id": ids[0], "reason": "empty"}]
    assert {(d["id"], d["field"], d["second"]) for d in report["disagreements"]} == {
        (ids[0], "answer", None), (ids[0], "tool_calls", None)}


def test_free_text_and_list_labels_compare_as_one_field(tmp_path):
    first = {"a": '{"x": 1}', "b": "  A blue   mug. ", "c": '[{"label": "keys"}]'}
    second = {"a": '{"x": 2}', "b": "A blue mug.", "c": '[{"label": "keys"}]'}
    fields = label_agreement.compare_targets(first["b"], second["b"])
    assert fields == {"label": (True, "A blue mug.", "A blue mug.")}
    assert label_agreement.compare_targets(first["c"], second["c"]) == {
        "label": (True, [{"label": "keys"}], [{"label": "keys"}])}
    diff = label_agreement.compare_targets(first["a"], second["a"])
    assert diff == {"x": (False, 1, 2)}


# --- clause: verdict against a threshold (learning rubric test 2) ------------------------------

def test_threshold_defaults_to_09_and_says_the_rubric_names_none(labeled):
    assert DEFAULT_THRESHOLD == 0.9
    assert label_agreement.rubric_threshold(RUBRIC) is None   # the default holds only while the rubric names none
    report = agree(labeled, reference_fake())
    assert report["threshold"] == {"value": 0.9, "source": "default: lab/refine/learning/RUBRIC.md LT2 names no "
                                                           "number, so 0.9 is used"}
    assert report["verdict"] == "consistent"


def test_a_number_the_rubric_names_for_agreement_wins_over_the_default(labeled, tmp_path, monkeypatch):
    text = "### LT1. Other\n- at least 0.5 of x\n\n### LT2. Labels checked\n- the label agreement rate is at least 85%.\n\n### LT3. Y\n"
    rubric = tmp_path / "RUBRIC.md"
    rubric.write_text(text)
    assert label_agreement.rubric_threshold(rubric) == 0.85
    assert label_agreement.rubric_threshold(tmp_path / "missing.md") is None
    monkeypatch.setattr(label_agreement, "RUBRIC", rubric)
    report = agree(labeled, reference_fake())
    assert report["threshold"] == {"value": 0.85, "source": "lab/refine/learning/RUBRIC.md LT2"}


def test_verdict_inconsistent_below_threshold_and_threshold_is_changeable(labeled):
    ids = [e.id for e in load(labeled, require_images=False).labeled()][:30]
    fake = reference_fake(swayed_policy(reference_policy(load_expected(FIXTURE)), questions_of(labeled, ids[:10])))
    low = agree(labeled, fake)          # 50 of 60 fields: 0.833
    assert low["verdict"] == "inconsistent"
    lax = agree(labeled, fake, threshold=0.8)
    assert lax["verdict"] == "consistent" and lax["threshold"]["source"] == "--threshold"


def test_exactly_at_the_threshold_is_consistent(labeled):
    ids = [e.id for e in load(labeled, require_images=False).labeled()][:30]
    fake = reference_fake(swayed_policy(reference_policy(load_expected(FIXTURE)), questions_of(labeled, ids[:3])))
    report = agree(labeled, fake, threshold=57 / 60)
    assert report["verdict"] == "consistent"


def test_fewer_than_30_labeled_is_insufficient_not_a_verdict(labeled_master, tmp_path):
    folder = tmp_path / "w" / FIXTURE.name
    shutil.copytree(labeled_master, folder)
    rows = [json.loads(line) for line in (folder / "examples.jsonl").read_text().splitlines()]
    for row in rows[10:]:
        row["target"] = None
    (folder / "examples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = agree(folder, reference_fake())
    assert report["verdict"] == "insufficient" and report["sample"]["size"] == 10


# --- clause: threshold written into the rubric with a citation to this run ---------------------

def test_report_carries_the_proposed_rubric_line_with_citation_and_rubric_is_untouched(labeled, tmp_path):
    before = digest(RUBRIC)
    report = agree(labeled, reference_fake(), run_id="agr-0001")
    line = report["proposed_rubric_line"]
    assert "0.9" in line and "LT2" in line
    assert "agr-0001" in line and "label_agreement" in line
    assert report["run_id"] == "agr-0001"
    assert digest(RUBRIC) == before
    assert "merge pass" in report["proposed_rubric_note"]


# --- report files and the ledger row -----------------------------------------------------------

def test_reports_are_written_as_json_and_markdown(labeled, tmp_path):
    report = agree(labeled, reference_fake())
    paths = label_agreement.write_report(report, tmp_path / "out")
    assert json.loads(paths["json"].read_text())["verdict"] == "consistent"
    table = paths["markdown"].read_text()
    assert "| field |" in table and "| answer |" in table and "| overall |" in table and "consistent" in table
    assert "RUBRIC" in table


def test_main_writes_reports_and_one_ledger_row(labeled, tmp_path):
    ids = [e.id for e in load(labeled, require_images=False).labeled()][:30]
    fake = reference_fake(swayed_policy(reference_policy(load_expected(FIXTURE)), questions_of(labeled, ids[:3])))
    out = tmp_path / "reports"
    answer = label_agreement.main([str(labeled), "--url", fake.base_url, "--out", str(out), "--run-id", "agr-0003"],
                                  transport=fake.transport)
    assert answer["ok"] and answer["outcome"] == "consistent" and answer["refs"]["run_id"] == "agr-0003"
    assert answer["metrics"]["label.agreement_rate"] == pytest.approx(57 / 60)
    assert (out / "label_agreement.json").is_file() and (out / "label_agreement.md").is_file()
    rows = [r for r in ledger.read(state_dir=tmp_path / "state") if r["verb"] == "label-agreement"]
    assert len(rows) == 1 and rows[0]["outcome"] == "consistent"
    assert rows[0]["metrics"]["label.agreement_rate"] == pytest.approx(57 / 60)
    assert rows[0]["metrics"]["label.agreement_sample"] == 30
    assert rows[0]["refs"]["run_id"] == "agr-0003"
    assert rows[0]["conditions"]["verdict"] == "consistent"
    assert rows[0]["conditions"]["threshold"] == 0.9


def test_main_refuses_a_dataset_that_is_not_an_export_or_fixture(labeled, tmp_path):
    odd = tmp_path / "elsewhere" / "other-name"
    shutil.copytree(labeled, odd)
    fake = reference_fake()
    answer = label_agreement.main([str(odd), "--url", fake.base_url, "--out", str(tmp_path / "o")],
                                  transport=fake.transport)
    assert answer["ok"] is False and answer["outcome"] == "refused"
    assert not [s for s in fake.requests if s.path.endswith("/v1/chat/completions")]


# --- the verb ----------------------------------------------------------------------------------

def test_verb_is_registered_in_sorted_order():
    assert dispatcher.VERBS["label-agreement"] == "lab.label_agreement:main"
    assert list(dispatcher.VERBS) == sorted(dispatcher.VERBS)
    assert dispatcher.summary("lab.label_agreement:main")
