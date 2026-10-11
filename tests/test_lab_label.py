"""lab/label.py against the fake teacher (lab/teacher_fake.py): what it may label, what it asks, which labels it keeps,
what it records, and how it fails. The fake stands in for vLLM on the pod and the hub proxy in front of it; the
dataset reader, the agent's tool code, the ledger and the label checks are the real ones. Every dataset is a copy in
a temporary folder, so the repository's fixture is never written."""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

from contexts.agent.application.qa import TOOLS
from contracts.learning import DatasetExample, DatasetManifest, TaskSpec
from lab import label
from lab.dataset import load
from lab.fixture_dataset import load_expected
from lab.hub_fake import FakeClock
from lab.machine import ledger, stop
from lab.teacher_fake import TTFT_S, DECODE_TOK_S, FakeTeacher, Turn, reference_policy, text_policy

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "fixtures" / "learning" / "tool-calling-fixture"
USER = "acct-c"
TOKEN = "hubtoken-" + "k" * 32
OBJECTS = {"type": "array", "items": {"type": "object",
                                      "properties": {"label": {"type": "string"}, "position": {"type": "string"}},
                                      "required": ["label", "position"], "additionalProperties": False}}
GOOD_OBJECTS = '[{"label": "keys", "position": "left of the laptop"}]'
CAPTION = "a desk with a laptop and a blue mug"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LAB_ACTOR", "test-labeler")
    monkeypatch.setenv("LAB_ENV_FILES", "")
    for name in ("LAB_TASK", "LAB_TIER", "LAB_HUB_URL", "LAB_HUB_USER_AGENT", "JUPYTERHUB_USER",
                 "JUPYTERHUB_API_TOKEN"):
        monkeypatch.delenv(name, raising=False)


# --- helpers ----------------------------------------------------------------------------------

def expected_rows():
    return load_expected(FIXTURE)


def by_question(question):
    return next(row for row in expected_rows().values() if row.question == question)


def fixture_copy(tmp_path, where="labeling"):
    target = tmp_path / where / FIXTURE.name
    shutil.copytree(FIXTURE, target)
    return target


def image_export(tmp_path, *, schema=None, count=2, dataset_id="desk-objects-3fa91c2b", parent=("data", "exports"),
                 action_id="act-0042"):
    root = tmp_path.joinpath(*parent, dataset_id)
    (root / "images").mkdir(parents=True)
    examples = []
    for number in range(1, count + 1):
        image = f"images/{number:04d}.jpg"
        (root / image).write_bytes(b"\xff\xd8\xff\xe0blurred keyframe %d" % number)
        examples.append(DatasetExample(id=f"ex-{number}", image=image, frame_ref=f"frames:cam-desk:e{number:02d}",
                                       ts="2026-10-10T21:00:00-04:00", context=f"{CAPTION} {number}"))
    manifest = DatasetManifest(
        dataset_id=dataset_id,
        spec=TaskSpec(task="desk-objects", instruction="List the small objects on the desk with their positions.",
                      output_schema=OBJECTS if schema is None else schema),
        created="2026-10-10T21:30:00-04:00", examples=count, action_id=action_id)
    (root / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    (root / "examples.jsonl").write_text("".join(e.model_dump_json() + "\n" for e in examples), encoding="utf-8")
    return root


def proxied(policy=None, **kwargs):
    return FakeTeacher(policy or reference_policy(expected_rows()), user=USER, token=TOKEN, **kwargs)


def teacher_for(fake, sleeps=None, **kwargs):
    kwargs.setdefault("sleep", (sleeps.append if sleeps is not None else (lambda seconds: None)))
    if fake.proxied:
        return label.Teacher.from_hub(fake.account("C"), transport=fake.transport, **kwargs)
    return label.Teacher(fake.base_url, transport=fake.transport, **kwargs)


def run(folder, fake, **kwargs):
    teacher_kwargs = {key: kwargs.pop(key) for key in ("stream", "sleeps", "clock_for_teacher") if key in kwargs}
    clock = teacher_kwargs.pop("clock_for_teacher", None)
    if clock is not None:
        teacher_kwargs["clock"] = clock
    with teacher_for(fake, **teacher_kwargs) as teacher:
        return label.label_dataset(folder, teacher, **kwargs)


def targets(folder):
    return {example.id: example.target for example in load(folder, require_images=False).examples}


def rows(tmp_path, **filters):
    return ledger.read(state_dir=tmp_path / "state", **filters)


def chat_requests(fake):
    return [seen for seen in fake.requests if seen.path.endswith("/v1/chat/completions")]


def answer_after_calls(calls, answer):
    """A teacher that makes these calls first and gives this answer once tool results are in."""
    def policy(body):
        if any(message.get("role") == "tool" for message in body["messages"]):
            return Turn(content=answer)
        return Turn(tool_calls=tuple(calls))
    return policy


# --- the fixture, labeled ------------------------------------------------------------------

def test_the_whole_fixture_is_labeled_with_the_reference_traces_and_every_label_follows_the_schema(tmp_path):
    folder = fixture_copy(tmp_path)
    fake = proxied()

    result = run(folder, fake, workers=4)

    expected = expected_rows()
    assert (result.outcome, result.asked, result.accepted, dict(result.rejected), dict(result.failed)) == (
        "ok", 142, 142, {}, {})
    assert result.pending_after == 0
    schema = load(folder, require_images=False).manifest.spec.output_schema
    stored = targets(folder)
    for example_id, row in expected.items():
        value = json.loads(stored[example_id])
        jsonschema.validate(value, schema)
        assert value == {"tool_calls": [call.model_dump() for call in row.label.tool_calls], "answer": row.label.answer}
    traces = [json.loads(line) for line in (folder / label.TRACES).read_text(encoding="utf-8").splitlines()]
    assert sorted(trace["id"] for trace in traces) == sorted(expected)
    for trace in traces:
        row = expected[trace["id"]]
        assert [call["result"] for call in trace["tool_calls"]] == row.tool_results
        assert {"tool_calls": [{"name": c["name"], "arguments": c["arguments"]} for c in trace["tool_calls"]],
                "answer": trace["answer"]} == json.loads(stored[trace["id"]])
        assert trace["run_id"] == result.run_id


def test_a_second_run_asks_nothing_about_examples_that_already_have_labels(tmp_path):
    folder = fixture_copy(tmp_path)
    fake = proxied()
    first = run(folder, fake, ids=["golden-01", "golden-02"])
    requests_before = len(fake.requests)

    second = run(folder, fake, ids=["golden-01", "golden-02"])

    assert (first.accepted, second.asked, second.accepted) == (2, 0, 0)
    assert len(fake.requests) == requests_before
    assert (second.outcome, second.pending_after) == ("ok", 140)


# --- what is asked -------------------------------------------------------------------------

def test_every_request_turns_thinking_off_decodes_greedily_and_streams_with_the_exact_accept_header(tmp_path):
    folder = fixture_copy(tmp_path)
    fake = proxied()
    row = expected_rows()["golden-01"]
    manifest = load(folder, require_images=False).manifest

    result = run(folder, fake, ids=["golden-01"])

    assert result.accepted == 1
    assert [(seen.method, seen.path) for seen in fake.requests] == [
        ("GET", f"/user/{USER}/proxy/8000/v1/models"),
        ("POST", f"/user/{USER}/proxy/8000/v1/chat/completions"),
        ("POST", f"/user/{USER}/proxy/8000/v1/chat/completions")]
    first, second = chat_requests(fake)
    for seen in (first, second):
        assert seen.headers["accept"] == "text/event-stream"
        assert seen.headers["authorization"] == f"token {TOKEN}"
        assert seen.headers["user-agent"].startswith("Mozilla/5.0")
        body = dict(seen.body)
        messages = body.pop("messages")
        assert body == {"model": "teacher", "temperature": 0.0, "top_p": 1.0, "max_tokens": 512, "seed": 0,
                        "chat_template_kwargs": {"enable_thinking": False}, "tools": TOOLS, "tool_choice": "auto",
                        "stream": True, "stream_options": {"include_usage": True}}
        assert messages[:2] == [{"role": "system", "content": manifest.spec.instruction},
                                {"role": "user", "content": row.question}]
    assert len(first.body["messages"]) == 2
    assistant, tool = second.body["messages"][2:]
    call = row.label.tool_calls[0]
    assert assistant["role"] == "assistant"
    assert [(c["type"], c["function"]["name"], json.loads(c["function"]["arguments"])) for c in assistant["tool_calls"]] \
        == [("function", call.name, call.arguments)]
    assert tool == {"role": "tool", "tool_call_id": assistant["tool_calls"][0]["id"], "content": row.tool_results[0]}


def test_without_streaming_each_request_asks_for_one_json_answer(tmp_path):
    folder = fixture_copy(tmp_path)
    fake = proxied()

    result = run(folder, fake, ids=["golden-01"], stream=False)

    assert result.accepted == 1
    for seen in chat_requests(fake):
        assert seen.headers["accept"] == "application/json"
        assert "stream" not in seen.body and "stream_options" not in seen.body
    expected = expected_rows()["golden-01"].label
    assert json.loads(targets(folder)["golden-01"]) == {
        "tool_calls": [call.model_dump() for call in expected.tool_calls], "answer": expected.answer}


def test_a_long_answer_streamed_arrives_where_the_same_answer_buffered_is_cut_by_the_proxy(tmp_path):
    long_answer = json.dumps([{"label": f"object {n}", "position": "somewhere on the desk"} for n in range(12)])
    streamed_folder = image_export(tmp_path / "a", count=1)
    buffered_folder = image_export(tmp_path / "b", count=1)
    streamed_fake = proxied(text_policy(long_answer), decode_tok_s=0.5)
    buffered_fake = proxied(text_policy(long_answer), decode_tok_s=0.5)
    sleeps: list[float] = []

    streamed = run(streamed_folder, streamed_fake)
    buffered = run(buffered_folder, buffered_fake, stream=False, sleeps=sleeps)

    assert (streamed.accepted, streamed.outcome) == (1, "ok")
    assert json.loads(targets(streamed_folder)["ex-1"]) == json.loads(long_answer)
    assert dict(buffered.failed) == {"request": 1}
    assert targets(buffered_folder)["ex-1"] is None
    assert len(chat_requests(buffered_fake)) == 4 and sleeps == [1.0, 2.0, 4.0]
    [example_row] = rows(tmp_path, verb="label-example", refs={"run_id": buffered.run_id})
    assert example_row["outcome"] == "failed" and example_row["note"].startswith("request: 524")


def test_an_image_goes_inline_with_the_instruction_caption_and_schema_and_a_fenced_json_answer_is_kept(tmp_path):
    folder = image_export(tmp_path, count=1)
    fake = proxied(text_policy(f"```json\n{GOOD_OBJECTS}\n```"))

    result = run(folder, fake)

    assert result.accepted == 1
    assert json.loads(targets(folder)["ex-1"]) == json.loads(GOOD_OBJECTS)
    [seen] = chat_requests(fake)
    assert "tools" not in seen.body and "tool_choice" not in seen.body
    [message] = seen.body["messages"]
    image_part, text_part = message["content"]
    picture = (folder / "images" / "0001.jpg").read_bytes()
    assert image_part == {"type": "image_url",
                          "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(picture).decode()}}
    assert text_part["type"] == "text"
    assert text_part["text"] == ("List the small objects on the desk with their positions.\n\n"
                                 f"What the box noted: {CAPTION} 1\n\n"
                                 "Reply with only JSON that follows this JSON schema, and nothing else:\n"
                                 + json.dumps(OBJECTS))


# --- which labels are kept ------------------------------------------------------------------

@pytest.mark.parametrize("answer, reason", [
    ('{"label": "keys", "position": "left of the laptop"}', "schema"),
    ('[{"label": "keys"}]', "schema"),
    ("keys, left of the laptop", "not_json"),
    ("[NaN]", "not_json"),
    ("<think>\nlet me look\n</think>\n\n" + GOOD_OBJECTS, "thinking"),
    ("", "empty"),
])
def test_a_label_that_breaks_the_rules_is_dropped_counted_and_left_pending(tmp_path, answer, reason):
    folder = image_export(tmp_path, count=2)
    fake = proxied(text_policy(answer))

    result = run(folder, fake)

    assert (result.outcome, result.asked, result.accepted, dict(result.rejected)) == ("ok", 2, 0, {reason: 2})
    assert targets(folder) == {"ex-1": None, "ex-2": None}
    assert [example.id for example in load(folder).pending()] == ["ex-1", "ex-2"]
    example_rows = rows(tmp_path, verb="label-example")
    assert [(row["outcome"], row["refs"]["example_id"]) for row in example_rows] == [("rejected", "ex-1"),
                                                                                    ("rejected", "ex-2")]
    assert all(row["note"].split(":")[0] == reason for row in example_rows)
    assert result.metrics()[f"labels.rejected.{reason}"] == 2


def test_a_schema_reject_names_where_and_which_rule_never_the_label(tmp_path):
    folder = image_export(tmp_path, count=1)
    fake = proxied(text_policy('[{"label": "keys", "position": "left of the laptop", "colour": "silver"}]'))

    run(folder, fake)

    [row] = rows(tmp_path, verb="label-example")
    assert row["note"] == "schema: 0 breaks additionalProperties"


def test_an_answer_cut_off_at_the_token_limit_is_rejected_as_truncated(tmp_path):
    folder = image_export(tmp_path, count=1)
    fake = proxied(text_policy(GOOD_OBJECTS))

    result = run(folder, fake, decode=label.Decode(max_tokens=4))

    assert dict(result.rejected) == {"truncated": 1}
    assert chat_requests(fake)[0].body["max_tokens"] == 4
    assert targets(folder)["ex-1"] is None


def test_a_free_text_task_keeps_the_answer_and_drops_what_its_own_check_refuses(tmp_path):
    kept_folder = image_export(tmp_path / "kept", count=1, schema={})
    checked_folder = image_export(tmp_path / "checked", count=1, schema={})
    sentence = "Keys and a blue mug sit to the left of the laptop."

    def short(text):
        return "fewer than 80 characters" if len(text) < 80 else None

    kept = run(kept_folder, proxied(text_policy(sentence)))
    checked = run(checked_folder, proxied(text_policy(sentence)), check=short)

    assert kept.accepted == 1 and targets(kept_folder)["ex-1"] == sentence
    assert dict(checked.rejected) == {"check": 1} and targets(checked_folder)["ex-1"] is None
    [row] = rows(tmp_path, verb="label-example", refs={"run_id": checked.run_id})
    assert row["note"] == "check: fewer than 80 characters"


KEYS = "Where did I leave my keys?"


@pytest.mark.parametrize("policy, reason", [
    (answer_after_calls([("last_seen", {"label": "keys", "room": "kitchen"})], "On the desk at 9 PM."), "schema"),
    (answer_after_calls([("web_search", {"query": "keys"})], "On the desk at 9 PM."), "schema"),
    (answer_after_calls([("last_seen", "{label: keys")], "On the desk at 9 PM."), "bad_arguments"),
    (answer_after_calls([("last_seen", "[\"keys\"]")], "On the desk at 9 PM."), "bad_arguments"),
    (answer_after_calls([("last_seen", {"label": "keys"})], "   "), "empty"),
    (answer_after_calls([("last_seen", {"label": "keys"})], "<think>\nhmm\n</think>\n\nOn the desk."), "thinking"),
])
def test_a_tool_trace_that_breaks_the_rules_is_dropped(tmp_path, policy, reason):
    folder = fixture_copy(tmp_path)
    example_id = by_question(KEYS).id
    fake = proxied(policy)

    result = run(folder, fake, ids=[example_id])

    assert dict(result.rejected) == {reason: 1}
    assert targets(folder)[example_id] is None
    assert not (folder / label.TRACES).exists()


def test_a_teacher_that_keeps_calling_tools_is_stopped_after_the_agents_rounds(tmp_path):
    folder = fixture_copy(tmp_path)
    example_id = by_question(KEYS).id
    fake = proxied(lambda body: Turn(tool_calls=(("last_seen", {"label": "keys"}),)))

    result = run(folder, fake, ids=[example_id])

    assert dict(result.rejected) == {"tool_rounds": 1}
    assert len(chat_requests(fake)) == 4
    [row] = rows(tmp_path, verb="label-example")
    assert (row["metrics"]["label.requests"], row["metrics"]["label.tool_calls"]) == (4, 3)


def test_a_server_without_a_tool_parser_gets_its_traces_rejected_and_the_run_stops_after_five(tmp_path):
    folder = fixture_copy(tmp_path)
    fake = proxied(tool_parser=False)

    result = run(folder, fake, limit=8)

    assert (result.outcome, result.asked, dict(result.rejected)) == ("fail", 5, {"tool_markup": 5})
    assert "tool_markup" in result.error
    assert all(target is None for target in targets(folder).values())
    assert len(chat_requests(fake)) == 5


# --- what may be labeled --------------------------------------------------------------------

def test_an_export_outside_data_exports_is_refused_before_the_teacher_is_asked(tmp_path):
    folder = image_export(tmp_path, parent=("downloads",))
    fake = proxied(text_policy(GOOD_OBJECTS))

    with pytest.raises(label.LabelRefused, match="data/exports"):
        run(folder, fake)
    assert fake.requests == []
    assert targets(folder) == {"ex-1": None, "ex-2": None}


def test_a_folder_not_named_for_its_dataset_id_is_refused(tmp_path):
    folder = image_export(tmp_path)
    renamed = folder.rename(folder.parent / "desk-objects-copy")
    fake = proxied(text_policy(GOOD_OBJECTS))

    with pytest.raises(label.LabelRefused, match="desk-objects-3fa91c2b"):
        run(renamed, fake)
    assert fake.requests == []


def test_on_a_pod_an_export_is_refused_and_the_fixture_is_still_labeled(tmp_path):
    export = image_export(tmp_path)
    folder = fixture_copy(tmp_path)
    export_fake = proxied(text_policy(GOOD_OBJECTS))
    fixture_fake = proxied()
    pod = {"JUPYTERHUB_USER": "jovyan"}

    with pytest.raises(label.LabelRefused, match="pod"):
        run(export, export_fake, environ=pod)
    fixture_run = run(folder, fixture_fake, environ=pod, ids=["golden-01"])

    assert export_fake.requests == []
    assert (fixture_run.outcome, fixture_run.accepted) == ("ok", 1)


def test_an_image_task_whose_picture_is_missing_is_refused_by_the_reader(tmp_path):
    folder = image_export(tmp_path)
    (folder / "images" / "0002.jpg").unlink()
    fake = proxied(text_policy(GOOD_OBJECTS))

    with pytest.raises(label.DatasetError, match="ex-2"):
        run(folder, fake)
    assert fake.requests == []


def test_a_schema_that_is_not_a_json_schema_is_refused_before_the_teacher_is_asked(tmp_path):
    folder = image_export(tmp_path, schema={"type": "list-of-things"})
    fake = proxied(text_policy(GOOD_OBJECTS))

    with pytest.raises(label.LabelRefused, match="not a valid JSON schema"):
        run(folder, fake)
    assert fake.requests == []


# --- what is recorded -----------------------------------------------------------------------

def test_each_example_writes_a_ledger_row_with_its_seconds_tokens_and_conditions(tmp_path):
    folder = fixture_copy(tmp_path)
    clock = FakeClock()
    fake = proxied(clock=clock)

    result = run(folder, fake, ids=["golden-01"], clock=clock.time, clock_for_teacher=clock.time)

    [row] = rows(tmp_path, verb="label-example")
    metrics = row["metrics"]
    assert (row["outcome"], row["actor"], row["node"], row["note"]) == ("accepted", "test-labeler", "pod-a", None)
    assert row["refs"] == {"pr": None, "commit": None, "slot": None, "run_id": result.run_id,
                           "dataset_id": "tool-calling-fixture", "example_id": "golden-01"}
    assert (metrics["label.requests"], metrics["label.tool_calls"]) == (2, 1)
    assert metrics["label.tokens"] == metrics["label.prompt_tokens"] + metrics["label.completion_tokens"]
    assert metrics["label.prompt_tokens"] > 0
    assert metrics["label.seconds"] == pytest.approx(2 * TTFT_S + metrics["label.completion_tokens"] / DECODE_TOK_S,
                                                     abs=1e-3)
    assert row["units"] == {"label.seconds": "s", "label.requests": "requests", "label.tool_calls": "calls",
                            "label.prompt_tokens": "tokens", "label.completion_tokens": "tokens",
                            "label.tokens": "tokens"}
    assert row["conditions"] == {"model": "teacher", "checkpoint": "RedHatAI/Qwen3.8-27B-INT4", "host": "teacher.test",
                                 "stream": True, "thinking": False, "temperature": 0.0, "top_p": 1.0,
                                 "max_tokens": 512, "seed": 0}
    totals = result.metrics()
    assert totals["labels.seconds_per_example"] == metrics["label.seconds"]
    assert totals["labels.tokens_per_example"] == metrics["label.tokens"]
    assert totals["labels.usage_missing"] == 0


def test_no_question_caption_label_or_token_reaches_the_ledger(tmp_path):
    export = image_export(tmp_path, count=2)
    folder = fixture_copy(tmp_path)
    row = expected_rows()["golden-01"]
    answers = iter(['[{"label": "silver keys", "position": "left of the laptop", "extra": 1}]', GOOD_OBJECTS])
    run(export, proxied(lambda body: Turn(content=next(answers))))
    run(folder, proxied(), ids=["golden-01"])

    text = (tmp_path / "state" / "ledger.jsonl").read_text(encoding="utf-8")

    assert len(rows(tmp_path, verb="label-example")) == 3
    for secret in (CAPTION, "silver keys", "left of the laptop", row.question, row.label.answer[:40], TOKEN):
        assert secret not in text


def test_an_error_message_that_quotes_the_token_is_scrubbed_before_it_is_recorded(tmp_path):
    folder = image_export(tmp_path, count=1)
    fake = proxied(text_policy(GOOD_OBJECTS))
    fake.fail(400, message=f"header Authorization: token {TOKEN} is malformed")

    result = run(folder, fake)

    assert dict(result.failed) == {"request": 1}
    [row] = rows(tmp_path, verb="label-example")
    assert TOKEN not in row["note"] and "***" in row["note"]


# --- how it fails ---------------------------------------------------------------------------

def test_gateway_errors_dropped_connections_and_cut_streams_are_tried_again(tmp_path):
    folder = image_export(tmp_path, count=1)
    fake = proxied(text_policy(GOOD_OBJECTS))
    fake.fail(502, page="<html><body>502 Bad Gateway</body></html>")
    fake.drop()
    fake.cut()
    sleeps: list[float] = []

    result = run(folder, fake, sleeps=sleeps)

    assert (result.accepted, sleeps) == (1, [1.0, 2.0, 4.0])
    assert len(chat_requests(fake)) == 4
    assert json.loads(targets(folder)["ex-1"]) == json.loads(GOOD_OBJECTS)


def test_an_error_the_server_sends_mid_answer_is_tried_again(tmp_path):
    folder = fixture_copy(tmp_path)
    fake = proxied()
    fake.break_stream()
    sleeps: list[float] = []

    result = run(folder, fake, ids=["golden-01"], sleeps=sleeps)

    assert (result.accepted, sleeps) == (1, [1.0])
    assert len(chat_requests(fake)) == 3
    [row] = rows(tmp_path, verb="label-example")
    assert (row["outcome"], row["metrics"]["label.requests"]) == ("accepted", 2)


def test_a_refused_token_stops_the_run_before_any_example(tmp_path):
    folder = fixture_copy(tmp_path)
    fake = proxied()
    wrong = label.Teacher.from_hub(label.hub.Account("C", USER, "wrong-" + "t" * 30, "https://teacher.test",
                                                   role="pod-a"),
                                   transport=fake.transport, sleep=lambda seconds: None)

    with wrong:
        result = label.label_dataset(folder, wrong)

    assert (result.outcome, result.asked) == ("fail", 0)
    assert "login page" in result.error and "wrong-" not in result.error
    assert chat_requests(fake) == []
    assert rows(tmp_path, verb="label-example") == []


def test_a_pod_that_is_not_running_stops_the_run(tmp_path):
    folder = fixture_copy(tmp_path)
    fake = proxied()
    fake.stop_server()

    result = run(folder, fake)

    assert (result.outcome, result.asked) == ("fail", 0)
    assert result.error.startswith("424")


@pytest.mark.parametrize("model, root, why", [
    ("teacher", "Qwen/Qwen3.5-4B", "not Qwen3.8-27B"),
    ("student", "RedHatAI/Qwen3.8-27B-INT4", "does not serve 'teacher'"),
])
def test_a_server_that_does_not_serve_the_teacher_is_not_used(tmp_path, model, root, why):
    folder = fixture_copy(tmp_path)
    fake = proxied(model=model, root=root)

    result = run(folder, fake)

    assert (result.outcome, result.asked) == ("fail", 0)
    assert why in result.error
    assert chat_requests(fake) == []


def test_the_checkpoint_check_can_be_switched_off(tmp_path):
    folder = fixture_copy(tmp_path)
    fake = proxied(root="Qwen/Qwen3.5-4B")

    result = run(folder, fake, ids=["golden-01"], expect_model=None)

    assert (result.outcome, result.accepted, result.checkpoint) == ("ok", 1, "Qwen/Qwen3.5-4B")


@pytest.mark.parametrize("root, expect, ok", [
    ("Qwen/Qwen3.5-122B-A10B-W4A16", "Qwen3.5-122B", True),            # the two-pod pair (L29's default)
    ("RedHatAI/Qwen3.8-27B-INT4", "Qwen3.5-122B", False),              # the 27B is not the pair
    ("RedHatAI/Qwen3.8-27B-INT4", "Qwen3.5-122B|Qwen3.8-27B", True),   # either one, the documented fallback
    ("Qwen/Qwen3.5-122B-A10B-W4A16", "Qwen3.5-122B|Qwen3.8-27B", True),
    ("Qwen/Qwen3.5-4B", "Qwen3.5-122B|Qwen3.8-27B", False),
])
def test_the_checkpoint_check_takes_the_pairs_name_or_either_of_two(tmp_path, root, expect, ok):
    folder = fixture_copy(tmp_path)
    fake = proxied(root=root)

    result = run(folder, fake, ids=["golden-01"], expect_model=expect)

    assert (result.outcome == "ok") is ok, result.error
    if not ok:
        assert chat_requests(fake) == [] and f"not {expect}" in result.error


def test_five_examples_in_a_row_without_an_answer_stop_the_run(tmp_path):
    folder = image_export(tmp_path, count=8)
    fake = proxied(text_policy(GOOD_OBJECTS))
    fake.fail(500, times=99, message="EngineCore encountered an issue")

    result = run(folder, fake)

    assert (result.outcome, result.asked, dict(result.failed)) == ("fail", 5, {"request": 5})
    assert "5 examples in a row" in result.error and "EngineCore" in result.error
    assert len(chat_requests(fake)) == 5


def test_the_stop_file_halts_the_run_between_examples_and_keeps_what_was_labeled(tmp_path):
    folder = fixture_copy(tmp_path)
    reference = reference_policy(expected_rows())

    def policy(body):
        stop.stop_file().parent.mkdir(parents=True, exist_ok=True)
        stop.stop_file().write_text("halt the labeling for the test", encoding="utf-8")
        return reference(body)

    result = run(folder, proxied(policy), ids=["golden-01", "golden-02", "golden-03"])

    assert (result.outcome, result.asked, result.accepted) == ("stopped", 1, 1)
    assert result.error == "halt the labeling for the test"
    stored = targets(folder)
    assert stored["golden-01"] is not None and stored["golden-02"] is None and stored["golden-03"] is None


def test_a_rejected_example_is_asked_again_by_the_next_run(tmp_path):
    folder = image_export(tmp_path, count=1)

    first = run(folder, proxied(text_policy("keys, left of the laptop")))
    second = run(folder, proxied(text_policy(GOOD_OBJECTS)))

    assert (dict(first.rejected), second.accepted) == ({"not_json": 1}, 1)
    assert json.loads(targets(folder)["ex-1"]) == json.loads(GOOD_OBJECTS)


# --- the entry point ------------------------------------------------------------------------

def test_main_labels_through_a_real_http_server_and_records_the_run(tmp_path):
    folder = fixture_copy(tmp_path)
    fake = FakeTeacher(reference_policy(expected_rows()))

    with fake.serve() as url:
        answer = label.main([str(folder), "--url", url, "--limit", "2", "--workers", "2"])

    assert (answer["ok"], answer["outcome"], answer["asked"], answer["accepted"], answer["pending"]) == (
        True, "ok", 2, 2, 140)
    [run_row] = rows(tmp_path, verb="label")
    assert run_row["outcome"] == "ok"
    assert run_row["refs"]["run_id"] == answer["run_id"]
    assert (run_row["metrics"]["labels.accepted"], run_row["metrics"]["labels.pending_after"]) == (2, 140)
    assert len(rows(tmp_path, verb="label-example", refs={"run_id": answer["run_id"]})) == 2
    assert [example_id for example_id, target in targets(folder).items() if target][:2] == ["golden-01", "golden-01-1"]


def test_the_script_runs_by_path_as_on_a_machine_without_the_package(tmp_path):
    folder = fixture_copy(tmp_path)
    fake = FakeTeacher(reference_policy(expected_rows()))
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    env.update({"LAB_STATE_DIR": str(tmp_path / "state"), "LAB_ENV_FILES": ""})

    with fake.serve() as url:
        done = subprocess.run([sys.executable, str(REPO / "lab" / "label.py"), str(folder), "--url", url,
                               "--id", "golden-01"], cwd=tmp_path, env=env, capture_output=True, text=True,
                              timeout=120)

    assert done.returncode == 0, done.stderr
    answer = json.loads(done.stdout.strip().splitlines()[-1])
    assert (answer["outcome"], answer["accepted"]) == ("ok", 1)
    assert targets(folder)["golden-01"] is not None


def test_main_reaches_the_teacher_through_the_hub_with_the_accounts_keys(tmp_path, monkeypatch):
    folder = fixture_copy(tmp_path)
    fake = proxied()
    env_file = tmp_path / "keys.env"
    env_file.write_text(f"AUP_HUB_USER_Z={USER}\nAUP_HUB_TOKEN_Z={TOKEN}\nAUP_HUB_URL_Z=https://teacher.test\nLAB_POD_ROLES=Z:pod-a\n",
                        encoding="utf-8")
    monkeypatch.setenv("LAB_ENV_FILES", str(env_file))

    answer = label.main([str(folder), "--account", "z", "--id", "golden-01"], transport=fake.transport)

    assert (answer["ok"], answer["accepted"]) == (True, 1)
    assert {seen.path.rsplit("/v1/", 1)[0] for seen in fake.requests} == {f"/user/{USER}/proxy/8000"}
    [row] = rows(tmp_path, verb="label-example")
    assert row["node"] == "pod-a"


def test_main_without_the_accounts_keys_refuses_and_names_only_the_keys(tmp_path):
    folder = fixture_copy(tmp_path)

    answer = label.main([str(folder), "--account", "Q"])

    assert (answer["ok"], answer["outcome"]) == (False, "refused")
    assert "AUP_HUB_USER_Q" in answer["error"] and "AUP_HUB_TOKEN_Q" in answer["error"]
    [run_row] = rows(tmp_path, verb="label")
    assert run_row["outcome"] == "refused"


def test_main_on_an_export_without_its_approval_answers_invalid(tmp_path):
    folder = image_export(tmp_path, action_id="")
    fake = proxied(text_policy(GOOD_OBJECTS))

    answer = label.main([str(folder), "--url", fake.base_url], transport=fake.transport)

    assert (answer["ok"], answer["outcome"]) == (False, "invalid")
    assert any("action_id is empty" in problem for problem in answer["problems"])
    assert fake.requests == []


def test_main_refuses_to_start_while_the_stop_file_exists(tmp_path, capsys):
    folder = fixture_copy(tmp_path)
    stop.stop_file().parent.mkdir(parents=True, exist_ok=True)
    stop.stop_file().write_text("maintenance", encoding="utf-8")

    with pytest.raises(stop.Stopped):
        label.main([str(folder), "--url", "http://teacher.test"])

    assert json.loads(capsys.readouterr().out.strip())["outcome"] == "stopped"
    assert rows(tmp_path, verb="label")[0]["outcome"] == "stopped"
