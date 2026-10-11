"""lab publish-adapter: every test runs on the recorded fake hub client; nothing opens a socket and
huggingface_hub is never imported."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from lab import __main__ as lab_cli
from lab import publish_adapter as pa
from lab.hf_fake import FakeHubClient, canned_card
from lab.machine import ledger

BASE = "Qwen/Qwen3.5-4B"
REPO = "IlanTree/atlas-desk-objects-lora"
METRICS = {"learning.heldout.current": 0.41, "learning.heldout.candidate": 0.78, "learning.heldout.teacher": 0.86,
           "learning.heldout.n_items": 58, "learning.heldout.n_groups": 12,
           "learning.heldout.noise": 0.0172, "learning.heldout.p_value": 0.0156}
SPEC = {"task": "desk-objects", "instruction": "List every object on the desk."}


@pytest.fixture
def state(tmp_path):
    d = tmp_path / "state"
    ledger.append({"actor": "gate", "tier": None, "task": "L4", "verb": "eval-gate", "outcome": "ok",
                   "metrics": METRICS, "refs": {"run_id": "run-7"}}, state_dir=d)
    return d


@pytest.fixture
def adapter(tmp_path):
    d = tmp_path / "adapter"
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": BASE, "peft_type": "LORA"}))
    (d / "adapter_model.safetensors").write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00{}      \x00\x01\x02")
    return d


@pytest.fixture
def spec(tmp_path):
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(SPEC))
    return p


def run(state, adapter, spec, client=None, **kw):
    client = client or FakeHubClient()
    kw.setdefault("run_id", "run-7")
    return client, pa.publish(repo=REPO, adapter_dir=adapter, spec_path=spec, state_dir=state, client=client, **kw)


def uploaded(client):
    assert len(client.uploads) == 1
    return client.uploads[0]


def front_matter(text):
    return yaml.safe_load(text.split("---")[1])


# --- the card's metadata
def test_metadata_sets_license_base_relation_library_and_no_datasets(state, adapter, spec):
    client, out = run(state, adapter, spec)
    assert out["ok"]
    meta = front_matter(uploaded(client).files["README.md"].decode())
    assert meta["license"] == "apache-2.0"
    assert meta["base_model"] == BASE
    assert meta["base_model_relation"] == "adapter"
    assert meta["library_name"] == "peft"
    assert "datasets" not in meta


# --- the card's text, filled from the ledger row
def test_text_names_task_instruction_and_every_number_from_the_row(state, adapter, spec):
    client, _ = run(state, adapter, spec)
    text = uploaded(client).files["README.md"].decode()
    assert "desk-objects" in text and "List every object on the desk." in text
    for needle in ("0.410", "0.780", "0.860", "58 items", "12 groups", "0.0172", "0.0156"):
        assert needle in text, needle


def test_a_different_row_gives_different_numbers_so_nothing_is_typed(tmp_path, adapter, spec):
    d = tmp_path / "s2"
    ledger.append({"actor": "gate", "tier": None, "task": "L4", "verb": "eval-gate", "outcome": "ok",
                   "metrics": {**METRICS, "learning.heldout.candidate": 0.6543, "learning.heldout.n_items": 77},
                   "refs": {"run_id": "run-9"}}, state_dir=d)
    client, _ = run(d, adapter, spec, run_id="run-9")
    text = uploaded(client).files["README.md"].decode()
    assert "0.654" in text and "77 items" in text and "0.780" not in text


def test_no_run_id_takes_the_newest_gate_row(state, adapter, spec):
    ledger.append({"actor": "gate", "tier": None, "task": "L4", "verb": "eval-gate", "outcome": "ok",
                   "metrics": {**METRICS, "learning.heldout.candidate": 0.9}, "refs": {"run_id": "run-8"}}, state_dir=state)
    client, _ = run(state, adapter, spec, run_id=None)
    assert "0.900" in uploaded(client).files["README.md"].decode()


def test_missing_row_or_metric_refuses_before_any_upload(state, adapter, spec):
    client, out = run(state, adapter, spec, run_id="nope")
    assert not out["ok"] and out["outcome"] == "no_ledger_row" and client.uploads == []
    d = state.parent / "s3"
    bad = dict(METRICS)
    del bad["learning.heldout.p_value"]
    ledger.append({"actor": "gate", "tier": None, "task": "L4", "verb": "eval-gate", "outcome": "ok",
                   "metrics": bad, "refs": {"run_id": "run-1"}}, state_dir=d)
    client, out = run(d, adapter, spec, run_id="run-1")
    assert out["outcome"] == "row_incomplete" and "learning.heldout.p_value" in out["error"] and client.uploads == []


# --- license text and the changes line
def test_upload_carries_apache_license_and_the_lora_line(state, adapter, spec):
    client, _ = run(state, adapter, spec)
    files = uploaded(client).files
    lic = files["LICENSE"].decode()
    assert "Apache License" in lic and "Version 2.0, January 2004" in lic and "END OF TERMS AND CONDITIONS" in lic
    text = files["README.md"].decode()
    assert f"This adapter is a LoRA fine-tune of {BASE}." in text
    assert {"adapter_config.json", "adapter_model.safetensors"} <= set(files)


def test_adapter_for_another_base_is_refused(state, adapter, spec):
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "other/model"}))
    client, out = run(state, adapter, spec)
    assert out["outcome"] == "wrong_base" and client.uploads == []


# --- the base's license is read again at publish time
def test_base_license_is_read_through_the_client_at_publish_time(state, adapter, spec):
    client, _ = run(state, adapter, spec)
    assert client.card_reads == [BASE]
    assert client.events.index(("read_card", BASE)) < client.events.index(("upload", REPO))


@pytest.mark.parametrize("card", [canned_card("mit"), canned_card(None), "no front matter at all", "---\n: [\n---\n"])
def test_a_base_that_is_not_apache_stops_the_publish(state, adapter, spec, card):
    client, out = run(state, adapter, spec, client=FakeHubClient(cards={BASE: card}))
    assert not out["ok"] and out["outcome"] == "base_license" and client.uploads == []


# --- the privacy check
TRIPS = {
    "image": ("pic.jpg", b"\xff\xd8\xff\xe0\x00\x10JFIF", "image"),
    "image_by_content": ("notes.txt", b"\x89PNG\r\n\x1a\n0000", "image"),
    "markdown_image": ("notes.md", b"see ![desk](x)", "image"),
    "caption": ("notes.txt", b"Caption: a mug on a desk", "caption"),
    "label": ("notes.txt", b"detector label = cup", "label"),
    "frame_ref": ("notes.json", b'{"frame_ref": "f1"}', "frame_ref"),
    "data_path": ("notes.txt", b"trained from data/exports/ds1/examples.jsonl", "data_path"),
    "camera_id": ("notes.json", b'{"camera_id": "cam-2"}', "camera_id"),
    "camera_video": ("notes.txt", b"opened /dev/video0", "camera_id"),
}


@pytest.mark.parametrize("name", sorted(TRIPS))
def test_each_privacy_pattern_has_a_file_that_trips_it(tmp_path, name):
    fname, content, kind = TRIPS[name]
    (tmp_path / fname).write_bytes(content)
    assert pa.privacy_findings(tmp_path) == [{"file": fname, "kind": kind}]


def test_clean_folder_has_no_findings(tmp_path):
    (tmp_path / "a.txt").write_text("a LoRA adapter for the desk task, data splits stay private")
    assert pa.privacy_findings(tmp_path) == []


@pytest.mark.parametrize("name", sorted(TRIPS))
def test_a_tripping_adapter_file_fails_the_publish_before_any_push(state, adapter, spec, name):
    fname, content, _ = TRIPS[name]
    (adapter / fname).write_bytes(content)
    client, out = run(state, adapter, spec)
    assert out["outcome"] == "privacy" and out["findings"] and client.uploads == []
    assert ("upload", REPO) not in client.events


def test_an_instruction_naming_a_label_is_caught_in_the_generated_card(state, adapter, tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"task": "t", "instruction": "Write a caption for the frame."}))
    client, out = run(state, adapter, p)
    assert out["outcome"] == "privacy" and client.uploads == []


def test_the_privacy_check_covers_the_license_and_card_text(state, adapter, spec):
    client, _ = run(state, adapter, spec)
    # what the real run would push is itself clean under the same patterns
    folder = client.uploads[0].folder_copy
    assert pa.privacy_findings(folder) == []


# --- token and the command line
def test_token_name_is_hf_token_and_a_value_is_never_printed(state, adapter, spec, monkeypatch, capsys):
    assert pa.TOKEN_ENV == "HF_TOKEN"
    secret = "hf_" + "A1b2C3d4E5" * 3 + "xy"
    monkeypatch.setenv("HF_TOKEN", secret)
    monkeypatch.setenv("LAB_STATE_DIR", str(state))
    code = lab_cli.main(["publish-adapter", "--fake", "--repo", REPO, "--adapter-dir", str(adapter),
                         "--task-spec", str(spec), "--run-id", "run-7"])
    out = capsys.readouterr()
    assert code == 0 and secret not in out.out + out.err
    assert json.loads(out.out)["ok"] is True


def test_real_client_without_the_token_fails_naming_only_the_variable(state, adapter, spec, monkeypatch, capsys):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("LAB_STATE_DIR", str(state))
    code = lab_cli.main(["publish-adapter", "--repo", REPO, "--adapter-dir", str(adapter), "--task-spec", str(spec)])
    out = json.loads(capsys.readouterr().out)
    assert code == 1 and out["outcome"] == "missing_token" and "HF_TOKEN" in out["error"]


def test_verb_is_registered_sorted_and_huggingface_hub_is_never_imported():
    names = list(lab_cli.VERBS)
    assert names == sorted(names) and lab_cli.VERBS["publish-adapter"] == "lab.publish_adapter:main"
    assert "huggingface_hub" not in sys.modules
