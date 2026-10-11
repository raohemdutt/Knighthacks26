"""lab/desk_score.py: the desk-object scorer, held to the committed synthetic desk fixture and the fake server.

Nothing here touches the network: the model is lab/desk_fake.py's FakeDeskServer behind an httpx transport, the
export is fixtures/learning/desk-fixture (12 examples on 8 frame_refs, a 64x64 PNG per frame made by code), and every test that
writes a ledger row uses a private state dir.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from lab import __main__ as lab_cli
from lab import dataset
from lab import desk_fake
from lab import desk_score as ds
from lab.machine import ledger
from lab.machine.metrics import _METRIC_RE

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "fixtures" / "learning" / "desk-fixture"
COMMIT = "0" * 40
URL = "http://desk.invalid:8080"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LAB_ACTOR", "desk-test")
    monkeypatch.delenv("LAB_TASK", raising=False)


@pytest.fixture
def export(tmp_path):
    """A copy of the committed fixture without its split, as an export arrives from the box."""
    root = tmp_path / desk_fake.DATASET_ID
    shutil.copytree(FIXTURE, root)
    (root / dataset.SPLIT).unlink()
    return root


@pytest.fixture
def held_out():
    return ds.held_out_examples(ds.load_desk_split(FIXTURE, min_held_out=1))


def fake(policy=None):
    return desk_fake.FakeDeskServer(policy or desk_fake.fixture_policy(FIXTURE))


# --- normalisation ------------------------------------------------------------------------

@pytest.mark.parametrize("raw, normal", [
    ("A mug", "mug"),
    ("mugs", "mug"),
    ("the Keys", "key"),
    ("stapler.", "stapler"),
    ("An apple!", "apple"),
    ("  Phone   Chargers ", "phone charger"),
    ("Batteries", "battery"),
    ("glasses", "glass"),
    ("glass", "glass"),
    ("boxes", "box"),
    ("Watches.", "watch"),
    ("\"Mug\"", "mug"),
    ("a pair of scissors", "pair of scissor"),
    ("THE", ""),
    ("", ""),
])
def test_the_normalisation_table(raw, normal):
    assert ds.normalize(raw) == normal


def test_answer_and_label_go_through_the_same_normalisation():
    assert ds.is_match("The mugs.", "mug")
    assert ds.is_match("keys", "A key")
    assert not ds.is_match("cup", "mug")
    assert not ds.is_match("a red mug", "mug")      # exact match, not containment
    assert not ds.is_match("", "")                   # an empty label never matches


def test_a_thinking_block_is_not_part_of_the_answer():
    assert ds.is_match("<think>\n\n</think>\n\nA mug.", "mug")


# --- the frozen split ---------------------------------------------------------------------

def test_the_split_is_written_once_by_frame_ref_group_and_its_sha256_is_the_readers(export):
    written = ds.write_split(export, held_out=5)
    split = dataset.load_split(export, min_held_out=1)
    assert written["sha256"] == split.sha256
    train_refs = {e.frame_ref for e in split.train}
    held_refs = {e.frame_ref for e in split.held_out}
    assert not train_refs & held_refs
    assert len(split.held_out) >= 5 and len(split.train) + len(split.held_out) == 12
    with pytest.raises(ds.DeskError, match="already"):
        ds.write_split(export, held_out=5)
    assert dataset.load_split(export, min_held_out=1).sha256 == split.sha256


def test_the_split_does_not_depend_on_the_order_of_the_export(export, tmp_path):
    other = tmp_path / "shuffled" / desk_fake.DATASET_ID
    shutil.copytree(export, other)
    lines = (other / dataset.EXAMPLES).read_text().splitlines()
    (other / dataset.EXAMPLES).write_text("\n".join(reversed(lines)) + "\n")
    first, second = ds.write_split(export, held_out=5), ds.write_split(other, held_out=5)
    assert sorted(first["held_out"]) == sorted(second["held_out"])


def test_the_committed_fixture_split_is_what_write_split_writes(export):
    ds.write_split(export, held_out=desk_fake.FIXTURE_HELD_OUT)
    assert (export / dataset.SPLIT).read_bytes() == (FIXTURE / dataset.SPLIT).read_bytes()


def test_the_committed_fixture_is_what_the_builder_makes(tmp_path):
    built = desk_fake.build_fixture(tmp_path / desk_fake.DATASET_ID)
    for path in sorted(p for p in FIXTURE.rglob("*") if p.is_file()):
        rel = path.relative_to(FIXTURE)
        assert (built / rel).read_bytes() == path.read_bytes(), rel
    assert len(list((FIXTURE / "images").glob("*.png"))) == 8   # one picture per frame_ref


def test_the_held_out_set_is_read_only_from_the_split_file(export):
    with pytest.raises(dataset.DatasetError, match="split.json: missing"):
        ds.load_desk_split(export, min_held_out=1)


def test_a_held_out_example_without_a_label_is_refused(export):
    ds.write_split(export, held_out=5)
    split = ds.load_desk_split(export, min_held_out=1)
    rows = [json.loads(line) for line in (export / dataset.EXAMPLES).read_text().splitlines()]
    for row in rows:
        if row["id"] == split.held_out[0].id:
            row["target"] = " "
    (export / dataset.EXAMPLES).write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ds.DeskError, match="no label"):
        ds.held_out_examples(ds.load_desk_split(export, min_held_out=1))


def test_desk_examples_carry_the_export_fields(held_out):
    first = held_out[0]
    assert first.frame_ref.startswith("frames:desk:")
    assert first.image.is_file() and first.image.suffix == ".png"
    assert first.label and first.caption


# --- the request: served the way the box serves it ---------------------------------------

def test_each_example_is_one_chat_request_with_the_image_the_fixed_prompt_and_the_fixed_decode(held_out):
    server = fake()
    ds.score(URL, held_out, prompt="Name it.", transport=server.transport)
    chats = [r for r in server.requests if r["path"] == "/v1/chat/completions"]
    assert len(chats) == len(held_out)
    for body in (r["body"] for r in chats):
        content = body["messages"][0]["content"]
        assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert content[1] == {"type": "text", "text": "Name it."}
        assert len(body["messages"]) == 1
        assert body["temperature"] == 0 and body["top_k"] == 1 and body["seed"] == ds.SEED
        assert body["max_tokens"] == ds.MAX_TOKENS
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert body["stream"] is False


def test_the_decode_setting_is_one_constant_and_goes_into_the_row(held_out):
    assert ds.DECODE == {"temperature": 0, "top_k": 1, "seed": ds.SEED, "max_tokens": ds.MAX_TOKENS,
                         "chat_template_kwargs": {"enable_thinking": False}}


# --- the score: exact match, per-object detail, the gate's shape -------------------------

def test_the_top_line_is_the_exact_match_rate_and_per_object_detail_is_dotted(held_out):
    out = ds.score(URL, held_out, prompt="Name it.", transport=fake().transport)
    expected = desk_fake.fixture_answers(FIXTURE)
    hits = [ds.is_match(expected[e.id], e.label) for e in held_out]
    assert out["score"] == pytest.approx(sum(hits) / len(hits))
    assert 0 < out["score"] < 1          # the fixture's fake answers miss on purpose
    labels = {ds.normalize(e.label) for e in held_out}
    for label in labels:
        mine = [h for h, e in zip(hits, held_out) if ds.normalize(e.label) == label]
        assert out[f"desk.{ds.slug(label)}"] == pytest.approx(sum(mine) / len(mine))
    assert out["desk.n_items"] == len(held_out)
    assert out["desk.n_groups"] == len({e.frame_ref for e in held_out})


def test_the_output_is_the_eval_json_shape_the_gate_reads(held_out):
    out = ds.score(URL, held_out, prompt="Name it.", transport=fake().transport)
    # contexts/agent/eval.py writes {"score", "model", "results"}; the dotted keys sit beside them
    assert {"score", "model", "results"} <= set(out)
    dotted = {k: v for k, v in out.items() if k not in ("score", "model", "results")}
    assert dotted and all(_METRIC_RE.fullmatch(k) for k in dotted)
    assert all(isinstance(v, (int, float)) for v in dotted.values())
    assert out["model"] == desk_fake.SERVED_MODEL
    # every item carries its group, so a paired test can be taken over groups, never single items
    assert [r["id"] for r in out["results"]] == [e.id for e in held_out]
    assert all({"id", "frame_ref", "label", "answer", "match"} <= set(r) for r in out["results"])
    json.dumps(out)


def test_a_model_that_thinks_or_rambles_is_scored_on_what_it_says():
    server = desk_fake.FakeDeskServer(lambda body, image: "The object is a mug. " * 8)
    examples = [ds.DeskExample("x-0", "frames:desk:1", FIXTURE / "images" / "desk-00.png", "", "mug")]
    out = ds.score(URL, examples, prompt="Name it.", transport=server.transport)
    assert out["score"] == 0.0
    assert server.requests[-1]["finish_reason"] == "length"


def test_a_server_error_stops_the_run_rather_than_counting_a_miss(held_out):
    server = fake()
    server.fail(500)
    with pytest.raises(ds.DeskError, match="500"):
        ds.score(URL, held_out, prompt="Name it.", transport=server.transport)


def test_an_object_name_that_would_hide_a_reserved_key_is_refused():
    examples = [ds.DeskExample("x-0", "frames:desk:1", FIXTURE / "images" / "desk-00.png", "", "Score")]
    with pytest.raises(ds.DeskError, match="reserved"):
        ds.score(URL, examples, prompt="Name it.", transport=fake(lambda b, i: "x").transport)


# --- the verb ------------------------------------------------------------------------------

def run(argv, transport):
    return ds.main(argv, transport=transport)


def test_the_verb_is_registered_once_and_the_registry_stays_sorted():
    assert lab_cli.VERBS["desk-score"] == "lab.desk_score:main"
    assert list(lab_cli.VERBS) == sorted(lab_cli.VERBS)


def test_score_records_the_split_sha256_the_commit_and_the_decode_in_a_ledger_row(tmp_path):
    out = run(["score", str(FIXTURE), "--base-url", URL, "--commit", COMMIT, "--min-held-out", "1",
               "--json", str(tmp_path / "out.json"), "--node", "jetson"], fake().transport)
    assert out["ok"] is True
    split = ds.load_desk_split(FIXTURE, min_held_out=1)
    rows = ledger.read(verb="desk-score")
    assert len(rows) == 1
    row = rows[0]
    assert row["task"] == "L24" and row["outcome"] == "ok" and row["node"] == "jetson"
    assert row["refs"]["split_sha256"] == split.sha256
    assert row["refs"]["commit"] == COMMIT
    assert row["refs"]["model"] == desk_fake.SERVED_MODEL
    assert row["conditions"]["decode"] == ds.DECODE
    assert row["metrics"]["desk.score"] == out["scores"]["score"]
    assert row["metrics"]["desk.n_items"] == len(split.held_out)
    assert row["metrics"]["desk.n_groups"] == split.held_out_frame_refs
    assert row["conditions"]["meets_deploy_floor"] is False
    saved = json.loads((tmp_path / "out.json").read_text())
    assert saved["score"] == out["scores"]["score"] and saved["results"]


def test_runs_2_records_the_score_twice_for_noise_on_the_same_split_and_commit():
    out = run(["score", str(FIXTURE), "--base-url", URL, "--commit", COMMIT, "--min-held-out", "1",
               "--runs", "2"], fake().transport)
    rows = ledger.read(verb="desk-score")
    assert len(rows) == 2 and len(out["runs"]) == 2
    assert {r["refs"]["split_sha256"] for r in rows} == {ds.load_desk_split(FIXTURE, min_held_out=1).sha256}
    assert {r["refs"]["commit"] for r in rows} == {COMMIT}
    assert len({r["refs"]["run_id"] for r in rows}) == 2
    assert out["noise"] == pytest.approx(abs(rows[0]["metrics"]["desk.score"] - rows[1]["metrics"]["desk.score"]))


def test_score_without_a_known_commit_is_refused_and_records_nothing(monkeypatch):
    monkeypatch.setattr(ds, "repo_commit", lambda: None)
    out = run(["score", str(FIXTURE), "--base-url", URL, "--min-held-out", "1"], fake().transport)
    assert out["ok"] is False and out["outcome"] == "no_commit"
    assert ledger.read(verb="desk-score") == []


def test_the_commit_is_read_from_git_when_not_given():
    commit = ds.repo_commit()
    assert commit is None or (len(commit) == 40 and all(c in "0123456789abcdef" for c in commit))


def test_the_split_verb_writes_split_json_once_and_records_its_sha256(export):
    out = run(["split", str(export), "--held-out", "5"], None)
    assert out["ok"] is True and out["sha256"] == dataset.load_split(export, min_held_out=1).sha256
    row = ledger.read(verb="desk-split")[0]
    assert row["refs"]["split_sha256"] == out["sha256"]
    again = run(["split", str(export), "--held-out", "5"], None)
    assert again["ok"] is False and again["outcome"] == "exists"


def test_the_dispatcher_runs_the_verb(capsys, export):
    code = lab_cli.main(["desk-score", "split", str(export), "--held-out", "5"])
    line = json.loads(capsys.readouterr().out)
    assert code == 0 and line["ok"] is True
