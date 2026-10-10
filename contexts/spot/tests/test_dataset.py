"""The fine-tune dataset built from the kit: the team's export format, a split by mission in time, grounded labels."""

import json
from pathlib import Path

import pytest

from contexts.spot import dataset
from contexts.spot.application.assistant import ID, Assistant, grounded
from contexts.spot.application.service import Spot
from contexts.spot.infrastructure.kit import load_kit
from contracts.learning import DatasetExample, DatasetManifest

SAMPLE = Path("fixtures/spot/dataset-sample")


@pytest.fixture(scope="module")
def kit():
    return load_kit(Path("fixtures/spot/kit"))


@pytest.fixture(scope="module")
def data(kit):
    return dataset.build(kit)


def test_sizes_and_the_split_by_mission_in_time_order(data, kit):
    train, held = data["train"], data["heldout"]
    assert len(train) >= 300 and len(held) >= 50  # the team's gate only deploys on 50 or more held-out items
    t_anchors, h_anchors = sorted({i["anchor_mission"] for i in train}), sorted({i["anchor_mission"] for i in held})
    assert h_anchors == ["M022", "M023", "M024", "M025", "M026", "M027"] and t_anchors[0] == "M006" and t_anchors[-1] == "M020"
    assert max(i["ts"] for i in train) < kit.mission["M021"].start_time.replace(tzinfo=train[0]["ts"].tzinfo)  # nothing newer than M020 in train
    assert not {i["group"] for i in train} & {i["group"] for i in held}
    assert len({i["id"] for i in train + held}) == len(train) + len(held)
    assert {i["intent"] for i in held} == set(dataset.Q)  # every kind of question is held out too
    assert len({i["question"] for i in train}) > 100  # varied wording


def test_every_label_is_grounded_in_real_rows_and_in_its_own_evidence(data, kit):
    known = set(kit.mission) | set(kit.action) | set(kit.anomaly) | set(kit.asset) | set(kit.location) | {r.reading_id for r in kit.readings}
    for item in data["train"][::7] + data["heldout"][::3]:
        label = json.loads(item["target"])
        assert set(label) == {"answer", "citations"} and label["answer"]
        assert set(label["citations"]) <= known, item["id"]
        assert set(ID.findall(label["answer"])) - {"V1", "V2"} <= set(label["citations"]), item["id"]  # every id the answer names is cited
        after = int(item["anchor_mission"][1:])
        for c in label["citations"]:  # and nothing from after the anchor mission
            if c in kit.mission:
                assert int(c[1:]) <= after, item["id"]
            if c in kit.action:
                assert int(kit.action[c].mission_id[1:]) <= after, item["id"]
        assert item["context"].startswith("Data to ") and "Question: " in item["context"] and "Evidence: {" in item["context"]
        json.loads(item["context"].split("Evidence: ", 1)[1])


def test_written_folders_match_the_export_contract_and_the_labs_reader(data, kit, tmp_path):
    info = dataset.write(tmp_path / "spot-qa-heldout", "spot-qa-heldout", data["heldout"], kit.now)
    folder = tmp_path / "spot-qa-heldout"
    manifest = DatasetManifest.model_validate_json((folder / "manifest.json").read_text())
    lines = (folder / "examples.jsonl").read_text().splitlines()
    assert manifest.examples == len(lines) == info["examples"] and manifest.spec.task == "spot-qa" and manifest.action_id
    first = DatasetExample.model_validate_json(lines[0])
    assert first.image == "" and first.frame_ref.startswith("spot:group:") and first.target
    expected = [json.loads(x) for x in (folder / "expected.jsonl").read_text().splitlines()]
    assert [e["id"] for e in expected] == [DatasetExample.model_validate_json(x).id for x in lines] and expected[0]["split"] == "heldout"
    lab_dataset = pytest.importorskip("lab.dataset")
    assert lab_dataset.check(folder, require_images=False) == []


def test_the_committed_sample_is_what_the_generator_writes(data, kit, tmp_path):
    dataset.write(tmp_path / "s", "spot-qa-sample", dataset.sample(data, 20), kit.now.replace(tzinfo=data["train"][0]["ts"].tzinfo))
    for name in ("examples.jsonl", "expected.jsonl", "manifest.json"):
        assert (tmp_path / "s" / name).read_text() == (SAMPLE / name).read_text(), name
    assert len((SAMPLE / "examples.jsonl").read_text().splitlines()) == 20


def test_cli_writes_both_splits(tmp_path, capsys):
    assert dataset.main(["--out", str(tmp_path), "--kit", str(tmp_path / "none")]) == 2
    assert dataset.main(["--out", str(tmp_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert [r["dataset_id"] for r in report] == ["spot-qa-train", "spot-qa-heldout"] and report[1]["examples"] >= 50
    assert (tmp_path / "spot-qa-train" / "images" / ".gitkeep").exists()


def test_truncate_cuts_the_kit_after_a_mission(kit):
    cut = dataset.truncate(kit, 10)
    assert len(cut.missions) == 10 and len(cut.actions) == 100 and cut.now == kit.mission["M010"].end_time
    assert all(a.mission_id <= "M010" for a in cut.anomalies) and len(cut.assets) == 20


def test_the_two_question_kinds_added_for_the_dataset(kit):
    a = Assistant(Spot(kit), None, rephrase=False)
    worse = a.draft("Which assets got worse between mission 26 and 27?")
    assert worse["intent"] == "worse" and "HX-202 (CRITICAL) was not visited in M026 and failed A0265 and raised AN005 in M027" in worse["text"]
    assert worse["citations"]["missions"] == ["M026", "M027"] and "AN015" in worse["citations"]["anomalies"]
    assert "No asset got worse" in a.draft("what got worse between mission 18 and 19")["text"] or a.draft("what got worse between mission 18 and 19")["intent"] == "worse"
    normal = a.draft("Is the temperature at HX-204 normal?")
    assert normal["intent"] == "reading_normal" and "41.4 C" in normal["text"] and "inside" in normal["text"]
    assert "AN014" in normal["text"] and "not the same as 'safe'" in normal["text"]  # in range, and yet a CRITICAL anomaly: said, not hidden
    assert grounded(normal["text"], normal["text"])[0]
