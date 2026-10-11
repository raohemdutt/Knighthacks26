"""lab/spot_score.py: the SPOT scorer, held to a set written by contexts.spot.dataset's own code and the fake server.

Nothing here touches the network: the model is lab/spot_fake.py's FakeSpotServer behind an httpx transport, and every
test uses a private ledger state dir. The set (574 train, 240 held out) is written once per session into a temporary
folder by `python -m contexts.spot.dataset`'s main; the small sample its --sample writes is what the label tests use.
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import pytest

import lab.__main__ as lab_cli
from contexts.agent.eval import Result
from contexts.spot import dataset as spot_dataset
from lab import dataset
from lab import eval_gate as gate
from lab import spot_fake
from lab import spot_score as ss
from lab.machine import ledger


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LAB_ACTOR", "spot-score-test")
    monkeypatch.delenv("LAB_TASK", raising=False)


@pytest.fixture(scope="session")
def written(tmp_path_factory) -> Path:
    """The set as contexts.spot.dataset writes it, with a 20-example sample."""
    root = tmp_path_factory.mktemp("spot")
    with contextlib.redirect_stdout(io.StringIO()):
        assert spot_dataset.main(["--out", str(root), "--sample", "20"]) == 0
    return root


@pytest.fixture(scope="session")
def joined(written, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("joined") / ss.DATASET_ID
    ss.join(written, out)
    return out


def sample_labels(written: Path) -> list[dict]:
    return [json.loads(e.target) for e in dataset.load(written / "spot-qa-sample", require_images=False).examples]


def reply(label: dict) -> str:
    return json.dumps(label, ensure_ascii=False)


# --- numbers --------------------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("1,234 of 10,000", {"1234", "10000"}),                      # thousands separators
    ("10.0% and 10.50 and 0.250", {"10", "10.5", "0.25"}),      # decimals and trailing zeros
    ("24min, 3.5V, 86%, 2nd", {"24", "3.5", "86", "2"}),          # a unit or percent glued to the number
    ("86 % or 86 percent", {"86"}),
    ("M022 WP12 A0213 AN036 R0001 PANEL-218 HX-104 V2 x2 abc123", set()),   # ids and words are not numbers
    ("-3.2, p=-0.0412, (-7), −5", {"-3.2", "-0.0412", "-7", "-5"}),  # negative
    ("+3.2", {"3.2"}),
    ("2-40, 5–10, 2% to 40%", {"2", "40", "5", "10"}),   # ranges: the minus between numbers is not a sign
    ("2026-09-08 18:00", {"2026", "9", "8", "18", "0"}),        # a date and a time are their parts
    ("-0 and 0.0", {"0"}),
    ("1,2 and 3, 4.", {"1", "2", "3", "4"}),                     # a comma before fewer than three digits separates
    ("ended in 10.", {"10"}),                                    # a full stop after a number is not a decimal point
])
def test_numbers_are_read_as_values_and_ids_are_not_numbers(text, expected):
    assert ss.numbers(text) == expected


def test_a_number_written_another_way_still_counts_and_a_different_one_does_not():
    assert ss.numbers("1234 and 10%") == ss.numbers("1,234 and 10.0 %")
    assert ss.numbers("3.2") != ss.numbers("-3.2")
    assert ss.numbers("M022") == set() and ss.numbers("22") == {"22"}


# --- judging a reply against the label ----------------------------------------------------------

def test_the_sample_is_the_datasets_own_and_every_label_has_numbers_to_check(written):
    labels = sample_labels(written)
    assert len(labels) == 20
    assert all(set(label) == {"answer", "citations"} for label in labels)
    assert sum(bool(ss.numbers(label["answer"])) for label in labels) >= 15


def test_the_label_itself_is_right(written):
    for label in sample_labels(written):
        verdict = ss.judge(reply(label), label["answer"], label["citations"])
        assert verdict["citations_ok"] and verdict["answer_ok"] and not verdict["malformed"], label


def test_a_right_reply_in_other_words_with_reordered_citations_is_right(written):
    label = next(lb for lb in sample_labels(written) if len(lb["citations"]) > 1 and ss.numbers(lb["answer"]))
    nums = " ".join(sorted(ss.numbers(label["answer"])))
    other = {"answer": f"In short: {nums}.", "citations": [*reversed(label["citations"]), label["citations"][0]]}
    fenced = "<think>\nok\n</think>\n```json\n" + reply(other) + "\n```"
    verdict = ss.judge(fenced, label["answer"], label["citations"])
    assert verdict["citations_ok"] and verdict["answer_ok"]


def test_a_wrong_citation_set_is_wrong(written):
    for label in sample_labels(written):
        cited = label["citations"]
        for wrong in (cited[1:], [*cited, "A9999"]):
            if set(wrong) == set(cited):
                continue
            verdict = ss.judge(reply({**label, "citations": wrong}), label["answer"], cited)
            assert not verdict["citations_ok"] and verdict["answer_ok"]


def test_a_missing_number_is_wrong_and_named(written):
    label = next(lb for lb in sample_labels(written) if ss.numbers(lb["answer"]))
    gone = max(ss.numbers(label["answer"]), key=len)
    answer = " ".join(n for n in sorted(ss.numbers(label["answer"])) if n != gone)
    verdict = ss.judge(reply({**label, "answer": answer}), label["answer"], label["citations"])
    assert verdict["citations_ok"] and not verdict["answer_ok"]
    assert verdict["missing_numbers"] == [gone]


def test_a_number_inside_an_id_does_not_stand_in_for_the_number(written):
    verdict = ss.judge(reply({"answer": "M022 and WP12", "citations": []}), "22 of 12", [])
    assert not verdict["answer_ok"] and verdict["missing_numbers"] == ["12", "22"]


@pytest.mark.parametrize("text", [
    "The mission went fine.",                                        # prose
    '{"answer": "fine", "citations": "M022"}',                       # citations not a list
    '{"answer": 3, "citations": []}',                                # answer not a string
    '{"answer": "fine", "citations": [1, 2]}',                       # citations not strings
    '{"answer": "M022 ran for 24 min", "citations": ["M02',          # cut off
    '["M022"]',                                                      # not an object
    "",
])
def test_a_malformed_reply_is_wrong_on_both(written, text):
    label = sample_labels(written)[0]
    verdict = ss.judge(text, label["answer"], label["citations"])
    assert verdict["malformed"] and not verdict["citations_ok"] and not verdict["answer_ok"]


# --- the joined set and its split ----------------------------------------------------------------

def test_split_joins_the_two_folders_by_anchor_mission(written, joined):
    split = dataset.load_split(joined, require_images=False)
    assert (len(split.train), len(split.held_out)) == (574, 240)
    assert dataset.check(joined, require_images=False) == []
    assert split.meets_deploy_floor
    held = {ss.group_of(e) for e in split.held_out}
    assert held == {f"mission:M0{n}" for n in range(22, 28)}
    assert "mission:M021" not in {ss.group_of(e) for e in split.train}
    assert all(e.id.startswith("spot-heldout-") for e in split.held_out)


def test_split_is_written_once_and_recorded(written, tmp_path):
    out = tmp_path / "again"
    first = ss.main(["split", str(written), "--out", str(out)])
    assert first["ok"] and first["held_out"] == 240 and first["groups"] == 6
    assert ss.main(["split", str(written), "--out", str(out)])["ok"] is False
    rows = ledger.read(state_dir=tmp_path / "state", verb="spot-split")
    assert [r["refs"]["split_sha256"] for r in rows] == [first["sha256"]]


def test_held_out_examples_carry_the_label_and_the_question(joined):
    examples = ss.held_out_examples(dataset.load_split(joined, require_images=False))
    assert len(examples) == 240
    first = examples[0]
    assert first.group == "mission:M022" and first.question and first.question in first.context
    assert first.citations and isinstance(first.answer, str)


# --- the client against the fake server -------------------------------------------------------------

def test_score_sends_the_instruction_the_context_and_the_gates_decode(joined):
    split = dataset.load_split(joined, require_images=False)
    examples = ss.held_out_examples(split)[:12]
    server = spot_fake.FakeSpotServer(spot_fake.role_policy(split))
    out = ss.score(spot_fake.CANDIDATE, examples, instruction=split.data.manifest.spec.instruction,
                   transport=server.transport)
    body = server.requests[0]["body"]
    assert body["messages"][0] == {"role": "system", "content": split.data.manifest.spec.instruction}
    assert body["messages"][1] == {"role": "user", "content": examples[0].context}
    assert body["max_tokens"] == gate.decode_for("spot-qa")["max_tokens"] == gate.SPOT_MAX_TOKENS
    assert body["temperature"] == 0
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["model"] == spot_fake.SERVED_MODEL
    assert out["model"] == spot_fake.SERVED_MODEL and out["spot.n_items"] == 12
    for row in out["results"]:
        passed = Result.model_validate({k: row[k] for k in Result.model_fields if k in row}).passed
        assert passed == (row["citations_ok"] and row["answer_ok"])
    assert out["score"] == sum(r["citations_ok"] and r["answer_ok"] for r in out["results"]) / 12


def test_the_teacher_scores_every_item_and_the_mistakes_are_caught(joined):
    split = dataset.load_split(joined, require_images=False)
    examples = ss.held_out_examples(split)
    server = spot_fake.FakeSpotServer(spot_fake.role_policy(split))
    teacher = ss.score(spot_fake.TEACHER, examples, instruction="x", transport=server.transport)
    current = ss.score(spot_fake.CURRENT, examples, instruction="x", transport=server.transport)
    assert teacher["score"] == 1.0
    assert 0 < current["score"] < 0.5 and current["spot.malformed"] > 0
    assert current["spot.citations_ok"] < 1 and current["spot.numbers_ok"] < 1


def test_a_server_error_stops_the_score(joined):
    split = dataset.load_split(joined, require_images=False)
    server = spot_fake.FakeSpotServer(spot_fake.role_policy(split))
    server.fail(503)
    with pytest.raises(ss.SpotError, match="503"):
        ss.score(spot_fake.TEACHER, ss.held_out_examples(split)[:2], instruction="x", transport=server.transport)


def test_the_verb_scores_and_records_a_row(joined, tmp_path):
    split = dataset.load_split(joined, require_images=False)
    server = spot_fake.FakeSpotServer(spot_fake.role_policy(split))
    out = ss.main(["score", str(joined), "--base-url", spot_fake.TEACHER, "--commit", "c0" * 20, "--runs", "2",
                   "--json", str(tmp_path / "out.json")], transport=server.transport)
    assert out["ok"] and out["scores"]["score"] == 1.0 and out["noise"] == 0
    assert json.loads((tmp_path / "out.json").read_text())["spot.n_groups"] == 6
    rows = ledger.read(state_dir=tmp_path / "state", verb="spot-score")
    assert len(rows) == 2 and rows[0]["task"] == ss.TASK and rows[0]["metrics"]["spot.score"] == 1.0


# --- the gate calls it by task name ----------------------------------------------------------------

def test_the_gate_has_the_scorer_registered_by_task_and_the_verb_is_listed():
    assert gate.SCORERS[ss.DATASET_TASK].client == "lab.spot_score:GateClient"
    assert isinstance(gate.client_for(ss.DATASET_TASK, "anywhere"), ss.GateClient)
    assert isinstance(gate.client_for("tool-calling", "anywhere"), gate.RealClient)
    assert lab_cli.VERBS["spot-score"] == "lab.spot_score:main"


def test_a_dry_run_of_the_gate_scores_before_after_and_teacher_on_the_held_out_set(tmp_path):
    out = spot_fake.dry_run(tmp_path / "dry")
    assert out["n_items"] == 240 and out["n_groups"] == 6          # paired on the anchor mission
    assert out["scores"]["teacher"] == 1.0
    assert out["scores"]["current"] < out["scores"]["candidate"] < 1.0
    assert out["requests"] == 4 * 240                                # current twice, candidate, teacher
    saved = json.loads(Path(out["scores_json"]).read_text())
    assert saved["task"] == "spot-qa" and saved["held_out"] == 240 and saved["unscored"] == 0
    assert saved["event"]["type"] == "learning.run.evaluated"
    # the run's one decode setting carries SPOT's own maximum output length, not the default 512
    assert saved["decode"] == {"thinking": "off", "sampling": "greedy", "max_tokens": gate.SPOT_MAX_TOKENS}


def test_spot_qa_has_its_own_maximum_output_length_and_every_other_task_keeps_512():
    # SPOT's longest held-out reference answer is 977 tokens (Qwen3.5-4B tokenizer); 512 would cut 78 of 240 short
    assert gate.SPOT_MAX_TOKENS > 977
    assert gate.decode_for("spot-qa") == {**gate.DECODE, "max_tokens": gate.SPOT_MAX_TOKENS}
    assert gate.decode_for("tool-calling") == gate.decode_for(None) == gate.DECODE
    assert gate.DECODE["max_tokens"] == 512                      # what runtime/llama.py sends for chat, unchanged
    assert ss.DECODE["max_tokens"] == gate.SPOT_MAX_TOKENS


def test_the_gate_verb_routes_a_spot_dataset_to_the_spot_client(joined, monkeypatch, tmp_path):
    seen = {}

    def run_gate(**kwargs):
        seen.update(kwargs)
        return {"ok": True}
    monkeypatch.setattr(gate, "run_gate", run_gate)
    gate.main(["--dataset", str(joined), "--current-url", "u", "--candidate-url", "u", "--teacher-url", "u",
               "--candidate-manifest", "m", "--run-id", "r", "--out", str(tmp_path / "g")])
    assert isinstance(seen["client"], ss.GateClient) and seen["client"].dataset_root == joined
