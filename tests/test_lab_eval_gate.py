"""lab/eval_gate.py: the eval gate that scores current, candidate and teacher on the frozen held-out set and decides.

Every test runs on lab/eval_gate_fake.py: no server, pod or network is touched. Datasets are written with the contract
models in a temporary folder (or the committed tool-calling fixture is read), and every test uses a private ledger
state dir.
"""
from __future__ import annotations

import hashlib
import inspect
import itertools
import json
import math
import re
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

import pytest

import lab.__main__ as dispatcher
from bus import EventBus
from contracts.events import parse_event
from contracts.learning import DatasetExample, DatasetManifest, ModelManifest, RunEvaluated, TaskSpec
from lab import eval_gate as gate
from lab import slide_numbers
from lab.dataset import load_split
from lab.eval_gate_fake import FakeClient
from lab.machine import ledger

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED = REPO_ROOT / "fixtures" / "learning" / "tool-calling-fixture"
NOW = datetime(2026, 10, 11, 9, 0, tzinfo=UTC)
SPEC = TaskSpec(task="tool-calling", instruction="Answer with tool calls.")
CUR, CAND, TEACH, PREV = "http://current.test", "http://candidate.test", "http://teacher.test", "http://preview.test"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LAB_ACTOR", "eval-gate-test")
    monkeypatch.delenv("LAB_TASK", raising=False)


def make(root: Path, held_out_sizes, train_sizes=(3, 3), *, pictures=False) -> Path:
    """A dataset with a frozen split: one frame_ref per group, one wording per example."""
    root.mkdir(parents=True)
    examples, sides = [], {"train": [], "held_out": []}
    for side, sizes in (("train", train_sizes), ("held_out", held_out_sizes)):
        for group, size in enumerate(sizes):
            for wording in range(size):
                name = f"{side[0]}{group}-{wording}"
                image = f"images/{name}.jpg" if pictures else ""
                examples.append(DatasetExample(id=name, image=image, frame_ref=f"frame:{side[0]}{group}", ts=NOW,
                                               context=f"Question {side[0]}{group} wording {wording}?"))
                sides[side].append(name)
    if pictures:
        (root / "images").mkdir()
        for example in examples:
            (root / example.image).write_bytes(b"jpeg")
    manifest = DatasetManifest(dataset_id=root.name, spec=SPEC, created=NOW, examples=len(examples),
                               action_id="approval-1")
    (root / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    (root / "examples.jsonl").write_text("".join(e.model_dump_json() + "\n" for e in examples), encoding="utf-8",
                                         newline="\n")
    (root / "split.json").write_text(json.dumps({"version": 1, "dataset_id": root.name, "rule": "test", **sides}),
                                     encoding="utf-8", newline="\n")
    return root


def candidate_manifest(tmp_path: Path, gguf="atlas-v3-tool-calling-Q4_K_M.gguf") -> Path:
    path = tmp_path / "pkg" / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    m = ModelManifest(version="atlas-v3-tool-calling", base="Qwen/Qwen3.5-4B", task="tool-calling", gguf=gguf,
                      sha256={gguf: "ab" * 32}, created=NOW)
    path.write_text(m.model_dump_json(), encoding="utf-8")
    return path


def held_out_questions(root: Path) -> list[str]:
    split = load_split(root, require_images=False, min_held_out=1)
    return [e.context for e in split.held_out]


def run(tmp_path, root, client, *, min_held_out=gate.MIN_ITEMS, **more):
    options = dict(dataset=root, current_url=CUR, candidate_url=CAND, teacher_url=TEACH, run_id="run-7",
                   out_dir=tmp_path / "out", current_version="atlas-v2-tool-calling", min_held_out=min_held_out)
    options.update(more)
    options.setdefault("candidate_manifest", candidate_manifest(tmp_path))
    return gate.run_gate(client=client, **options)


def first_n(questions, n):
    return set(questions[:n])


def rows(verb):
    return [r for r in ledger.read() if r["verb"] == verb]


# --- the paired test ------------------------------------------------------------------------------------------------

def mcnemar_exact(b: int, c: int) -> float:
    """The exact McNemar test, two-sided, written out independently of the gate."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


@pytest.mark.parametrize("b,c,ties", [(0, 0, 5), (6, 0, 3), (9, 2, 40), (5, 5, 1), (13, 4, 0), (1, 0, 0), (20, 7, 23)])
def test_the_sign_flip_test_is_exact_mcnemar_when_every_group_holds_one_item(b, c, ties):
    diffs = [Fraction(1)] * b + [Fraction(-1)] * c + [Fraction(0)] * ties
    assert float(gate.sign_flip_p(diffs)) == pytest.approx(mcnemar_exact(b, c), abs=1e-12)


def test_the_sign_flip_test_enumerates_every_sign_of_the_group_means():
    diffs = [Fraction(2, 3), Fraction(1, 2), Fraction(-1, 4), Fraction(1, 1), Fraction(0), Fraction(-1, 3)]
    observed = abs(sum(diffs))
    hits = sum(abs(sum(s * d for s, d in zip(signs, diffs))) >= observed
               for signs in itertools.product((1, -1), repeat=len(diffs)))
    assert gate.sign_flip_p(diffs) == Fraction(hits, 2 ** len(diffs))


def test_four_groups_all_in_favour_cannot_reach_p_below_0_05():
    assert gate.sign_flip_p([Fraction(1)] * 4) == Fraction(2, 16)


def test_the_test_unit_is_the_group_so_items_that_share_a_frame_are_not_counted_as_independent(tmp_path):
    root = make(tmp_path / "d", held_out_sizes=(15,) * 4)   # 60 items, 4 groups
    qs = held_out_questions(root)
    client = FakeClient(root, {CUR: [set(), set()], CAND: [set(qs)], TEACH: [set(qs)]})
    result = run(tmp_path, root, client)
    assert (result["n_items"], result["n_groups"]) == (60, 4)
    assert result["p_value"] == pytest.approx(0.125)   # item-level McNemar would give p near 1e-18
    assert result["decision"] == "reject"


# --- the held-out set: only from the frozen split file ------------------------------------------------------------

def test_only_the_held_out_ids_of_split_json_are_scored_and_its_sha256_is_in_every_score_row(tmp_path):
    root = make(tmp_path / "d", held_out_sizes=(1,) * 60)
    qs = held_out_questions(root)
    client = FakeClient(root, {CUR: [set(), set()], CAND: [set(qs)], TEACH: [set(qs)]},
                        extra_questions=["Something the split does not hold?"])
    result = run(tmp_path, root, client)
    digest = hashlib.sha256((root / "split.json").read_bytes()).hexdigest()
    assert result["split_sha256"] == digest
    assert result["n_items"] == 60 and result["outside_split"] == 1
    score_rows = rows("eval-gate.score")
    assert len(score_rows) == 4   # current twice, candidate, teacher
    assert all(r["conditions"]["split_sha256"] == digest and r["refs"]["split_sha256"] == digest for r in score_rows)


def test_a_scorer_that_asks_a_training_question_is_refused(tmp_path):
    root = make(tmp_path / "d", held_out_sizes=(1,) * 60)
    train_q = "Question t0 wording 1?"
    client = FakeClient(root, {CUR: [set()], CAND: [set()], TEACH: [set()]}, extra_questions=[train_q])
    with pytest.raises(gate.GateError, match="train"):
        run(tmp_path, root, client)


def test_a_held_out_item_the_scorer_did_not_score_is_reported_and_not_counted(tmp_path):
    root = make(tmp_path / "d", held_out_sizes=(1,) * 60)
    qs = held_out_questions(root)
    client = FakeClient(root, {CUR: [set(), set()], CAND: [set(qs)], TEACH: [set(qs)]}, questions=qs[:55])
    result = run(tmp_path, root, client)
    assert (result["n_items"], result["unscored"]) == (55, 5)


# --- one scorer commit, one decode setting ----------------------------------------------------------------------

def test_every_ledger_row_carries_the_one_scorer_commit_and_decode_setting(tmp_path):
    root = make(tmp_path / "d", held_out_sizes=(1,) * 60)
    client = FakeClient(root, {CUR: [set()], CAND: [set()], TEACH: [set()]}, commit="c0ffee1")
    run(tmp_path, root, client)
    gate_rows = rows("eval-gate.score") + rows("eval-gate")
    assert len(gate_rows) == 5
    for r in gate_rows:
        assert r["conditions"]["decode"] == {"thinking": "off", "sampling": "greedy", "max_tokens": 512}
        assert r["conditions"]["scorer_commit"] == "c0ffee1" and r["refs"]["commit"] == "c0ffee1"
    assert client.commit_calls == 1


def test_the_fixed_max_output_length_is_the_one_the_runtime_sends_for_chat():
    from runtime.llama import LlamaServerRuntime
    assert re.search(rf"max_tokens={gate.DECODE['max_tokens']}\b", inspect.getsource(LlamaServerRuntime.chat))


def test_a_server_that_reports_another_decode_setting_is_refused(tmp_path):
    root = make(tmp_path / "d", held_out_sizes=(1,) * 60)
    client = FakeClient(root, {CUR: [set()], CAND: [set()], TEACH: [set()]},
                        decode={TEACH: {"thinking": "on", "sampling": "greedy", "max_tokens": 512}})
    with pytest.raises(gate.GateError, match="decode"):
        run(tmp_path, root, client)


def test_a_server_whose_decode_setting_cannot_be_read_needs_an_attestation(tmp_path):
    root = make(tmp_path / "d", held_out_sizes=(1,) * 60)
    client = FakeClient(root, {CUR: [set()], CAND: [set()], TEACH: [set()]}, decode={CAND: None})
    with pytest.raises(gate.GateError, match="attest"):
        run(tmp_path, root, client)
    result = run(tmp_path, root, FakeClient(root, {CUR: [set()], CAND: [set()], TEACH: [set()]}, decode={CAND: None}),
                 decode_attested=True)
    assert result["decode_source"]["candidate"] == "attested"


# --- run-to-run noise --------------------------------------------------------------------------------------------

def test_the_current_model_is_scored_at_least_twice_and_the_spread_is_the_noise(tmp_path):
    root = make(tmp_path / "d", held_out_sizes=(1,) * 60)
    qs = held_out_questions(root)
    client = FakeClient(root, {CUR: [first_n(qs, 20), first_n(qs, 23), first_n(qs, 21)], CAND: [set()], TEACH: [set()]})
    result = run(tmp_path, root, client, current_runs=3)
    assert client.calls.count(CUR) == 3
    assert result["noise"] == pytest.approx(3 / 60)
    assert result["scores"]["current"] == pytest.approx(64 / 180)
    with pytest.raises(gate.GateError, match="twice"):
        run(tmp_path, root, client, current_runs=1)


# --- the decision ------------------------------------------------------------------------------------------------

def test_a_candidate_that_beats_current_beyond_noise_with_p_below_0_05_on_50_items_deploys(tmp_path):
    root = make(tmp_path / "d", held_out_sizes=(1,) * 60)
    qs = held_out_questions(root)
    client = FakeClient(root, {CUR: [first_n(qs, 20), first_n(qs, 21)], CAND: [first_n(qs, 40)], TEACH: [set(qs)]})
    result = run(tmp_path, root, client)
    assert result["decision"] == "deploy", result["reasons"]
    assert result["p_value"] < 0.05 and result["delta"] > result["noise"]


def test_a_gain_within_the_noise_is_rejected(tmp_path):
    root = make(tmp_path / "d", held_out_sizes=(1,) * 60)
    qs = held_out_questions(root)
    # current wobbles by 20 items between runs; candidate gains 15 over the mean
    client = FakeClient(root, {CUR: [first_n(qs, 10), first_n(qs, 30)], CAND: [first_n(qs, 35)], TEACH: [set(qs)]})
    result = run(tmp_path, root, client)
    assert result["decision"] == "reject"
    assert any("noise" in reason for reason in result["reasons"])


def test_a_gain_that_is_not_significant_is_rejected(tmp_path):
    root = make(tmp_path / "d", held_out_sizes=(1,) * 60)
    qs = held_out_questions(root)
    client = FakeClient(root, {CUR: [first_n(qs, 20)] * 2, CAND: [first_n(qs, 23)], TEACH: [set(qs)]})
    result = run(tmp_path, root, client)
    assert result["decision"] == "reject" and result["p_value"] >= 0.05


def test_a_set_under_50_items_is_scored_and_reported_but_never_deploys(tmp_path):
    root = make(tmp_path / "d", held_out_sizes=(1,) * 30)
    qs = held_out_questions(root)
    client = FakeClient(root, {CUR: [set()] * 2, CAND: [set(qs)], TEACH: [set(qs)]})
    result = run(tmp_path, root, client, min_held_out=10)
    assert result["scores"]["candidate"] == 1.0 and result["p_value"] < 0.05
    assert result["decision"] == "reject"
    assert any("50" in reason for reason in result["reasons"])


def test_a_bf16_preview_is_recorded_under_a_preview_key_and_never_decides(tmp_path):
    root = make(tmp_path / "d", held_out_sizes=(1,) * 60)
    qs = held_out_questions(root)
    client = FakeClient(root, {CUR: [set()] * 2, CAND: [set()], TEACH: [set(qs)], PREV: [set(qs)]})
    result = run(tmp_path, root, client, preview_url=PREV)
    assert result["scores"]["preview.candidate_bf16"] == 1.0
    assert result["decision"] == "reject"
    assert [r["conditions"]["role"] for r in rows("eval-gate.score")].count("preview") == 1


def test_the_candidate_must_be_the_packaged_q4_k_m_gguf(tmp_path):
    root = make(tmp_path / "d", held_out_sizes=(1,) * 60)
    client = FakeClient(root, {CUR: [set()], CAND: [set()], TEACH: [set()]})
    with pytest.raises(gate.GateError, match="Q4_K_M"):
        run(tmp_path, root, client, candidate_manifest=candidate_manifest(tmp_path / "bf16", gguf="atlas-v3-bf16.gguf"))


def test_the_ledger_records_n_as_items_and_groups_the_noise_and_p_beside_the_scores(tmp_path):
    root = make(tmp_path / "d", held_out_sizes=(2,) * 30)
    qs = held_out_questions(root)
    client = FakeClient(root, {CUR: [first_n(qs, 10)] * 2, CAND: [first_n(qs, 50)], TEACH: [set(qs)]})
    result = run(tmp_path, root, client)
    (row,) = rows("eval-gate")
    m = row["metrics"]
    assert (m["eval.n_items"], m["eval.n_groups"]) == (60, 30)
    assert m["eval.noise"] == 0 and m["eval.p_value"] == pytest.approx(result["p_value"])
    assert (m["eval.current"], m["eval.candidate"], m["eval.teacher"]) == (pytest.approx(10 / 60),
                                                                          pytest.approx(50 / 60), 1.0)
    assert row["outcome"] == result["decision"] == "deploy"


# --- image tasks: the episode is the unit -----------------------------------------------------------------------

def test_image_tasks_group_by_episode_and_refuse_without_an_episode_map(tmp_path):
    root = make(tmp_path / "d", held_out_sizes=(1,) * 60, pictures=True)
    split = load_split(root)
    ids = [e.id for e in split.held_out]
    client = FakeClient(root, {CUR: [set()] * 2, CAND: [set(ids)], TEACH: [set(ids)]}, by_id=True)
    with pytest.raises(gate.GateError, match="episode"):
        run(tmp_path, root, client)
    episodes = tmp_path / "episodes.json"
    episodes.write_text(json.dumps({e.frame_ref: f"ep-{i // 10}" for i, e in enumerate(split.held_out)}))
    client = FakeClient(root, {CUR: [set()] * 2, CAND: [set(ids)], TEACH: [set(ids)]}, by_id=True)
    result = run(tmp_path, root, client, episodes=episodes)
    assert (result["n_items"], result["n_groups"]) == (60, 6)
    assert result["p_value"] == pytest.approx(2 / 64)   # 6 episodes, all in favour
    assert result["decision"] == "deploy"


# --- the event and the slide -------------------------------------------------------------------------------------

def test_learning_run_evaluated_carries_the_three_scores_and_the_decision(tmp_path):
    root = make(tmp_path / "d", held_out_sizes=(1,) * 60)
    qs = held_out_questions(root)
    client = FakeClient(root, {CUR: [first_n(qs, 20)] * 2, CAND: [first_n(qs, 40)], TEACH: [set(qs)]})
    result = run(tmp_path, root, client, events_dir=tmp_path / "events")
    events = list(EventBus(log_dir=tmp_path / "events").read_log())
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, RunEvaluated)
    assert event.decision == "deploy" and event.run_id == "run-7" and event.task == "tool-calling"
    assert event.model_version == "atlas-v3-tool-calling"
    assert (event.scores["current"], event.scores["candidate"], event.scores["teacher"]) == (
        pytest.approx(20 / 60), pytest.approx(40 / 60), 1.0)
    assert "candidate.citations_ok" in event.scores and "gate.p_value" in event.scores
    assert all(isinstance(v, float) for v in event.scores.values())
    saved = (tmp_path / "out" / gate.EVENT_FILE).read_text().strip()
    assert parse_event(saved).event_id == event.event_id == result["event"]["event_id"]


def test_the_scores_are_saved_for_the_demo_slide(tmp_path):
    root = make(tmp_path / "d", held_out_sizes=(1,) * 60)
    qs = held_out_questions(root)
    client = FakeClient(root, {CUR: [first_n(qs, 20)] * 2, CAND: [first_n(qs, 40)], TEACH: [set(qs)]})
    run(tmp_path, root, client)
    out = tmp_path / "out"
    saved = json.loads((out / gate.SCORES_FILE).read_text())
    assert saved["decision"] == "deploy" and saved["scores"]["candidate"] == pytest.approx(40 / 60)
    assert saved["n_items"] == 60 and "p_value" in saved and "noise" in saved
    (row,) = slide_numbers.eval_rows(str(out / "current-1.json"), str(out / "candidate.json"))
    assert "0.33" in row.value and "0.67" in row.value


# --- the committed fixture ------------------------------------------------------------------------------------------

def test_the_committed_fixture_scores_58_items_in_4_groups_and_cannot_deploy(tmp_path):
    qs = held_out_questions(COMMITTED)
    client = FakeClient(COMMITTED, {CUR: [set()] * 2, CAND: [set(qs)], TEACH: [set(qs)]})
    result = run(tmp_path, COMMITTED, client)
    assert (result["n_items"], result["n_groups"]) == (58, 4)
    assert result["decision"] == "reject" and result["p_value"] == pytest.approx(0.125)


# --- the verb -----------------------------------------------------------------------------------------------------

def test_eval_gate_is_one_sorted_line_of_the_verb_registry():
    assert dispatcher.VERBS["eval-gate"] == "lab.eval_gate:main"
    assert list(dispatcher.VERBS) == sorted(dispatcher.VERBS)


def test_python_m_lab_eval_gate_prints_one_json_line(tmp_path, monkeypatch, capsys):
    root = make(tmp_path / "d", held_out_sizes=(1,) * 60)
    qs = held_out_questions(root)
    client = FakeClient(root, {CUR: [first_n(qs, 20)] * 2, CAND: [first_n(qs, 40)], TEACH: [set(qs)]})
    monkeypatch.setattr(gate, "make_client", lambda: client)
    code = dispatcher.main(["eval-gate", "--dataset", str(root), "--current-url", CUR, "--candidate-url", CAND,
                            "--teacher-url", TEACH, "--candidate-manifest", str(candidate_manifest(tmp_path)),
                            "--current-version", "atlas-v2", "--run-id", "run-9", "--out", str(tmp_path / "o")])
    answer = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == 0 and answer["ok"] and answer["decision"] == "deploy"


def test_a_refused_gate_answers_not_ok_and_writes_a_refused_row(tmp_path, monkeypatch, capsys):
    root = make(tmp_path / "d", held_out_sizes=(1,) * 60)
    monkeypatch.setattr(gate, "make_client", lambda: FakeClient(root, {CUR: [set()], CAND: [set()], TEACH: [set()]}))
    code = dispatcher.main(["eval-gate", "--dataset", str(root), "--current-url", CUR, "--candidate-url", CAND,
                            "--teacher-url", TEACH, "--candidate-manifest", str(candidate_manifest(tmp_path)),
                            "--current-runs", "1", "--run-id", "run-9", "--out", str(tmp_path / "o")])
    answer = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == 1 and answer["outcome"] == "refused"
    assert [r["outcome"] for r in rows("eval-gate")] == ["refused"]


def test_the_real_scorer_command_is_the_documented_one(tmp_path):
    command = gate.RealClient().command("http://127.0.0.1:8080", tmp_path / "o.json", model_version="atlas-v2")
    assert command[1:] == ["-m", "contexts.agent.eval", "--runtime", "llama", "--base-url", "http://127.0.0.1:8080",
                           "--model-version", "atlas-v2", "--json", str(tmp_path / "o.json")]


def test_the_scorer_asks_only_the_golden_wordings_so_24_of_the_58_held_out_items_are_scored(tmp_path):
    from contexts.agent.eval import load_fixture
    _, _, golden = load_fixture(REPO_ROOT / "fixtures" / "golden")
    asked = [q for g in golden for q in [g.q, *g.also]]   # what the scorer CLI asks, in its order
    client = FakeClient(COMMITTED, {CUR: [set()] * 2, CAND: [set(asked)], TEACH: [set(asked)]}, questions=asked)
    result = run(tmp_path, COMMITTED, client)
    assert (result["n_items"], result["unscored"], result["n_groups"]) == (24, 34, 4)
    assert result["outside_split"] == len(asked) - 24
    assert result["decision"] == "reject"
