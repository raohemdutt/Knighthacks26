"""lab/format_traces.py: labeled tool-calling examples become training rows in the student's own chat format.

The reference for the format is what the real Qwen3.5-4B chat template rendered (fixtures/lab/qwen35-template/,
recorded through transformers at the revision named in source.json): the renderer must write each recorded
conversation exactly as recorded. The tool results and the labels come from the committed no-camera fixture
dataset, and the labels are written into a copy of it from expected.jsonl, the stand-in for the teacher. Only the
student's tokenizer is replaced, by a small byte-level one that keeps special tokens whole, because the real one is
a 12 MB download and the `tokenizers` package is not a requirement of the repo.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import unicodedata
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import pytest

from contexts.agent.application.qa import QA, SYSTEM, TOOLS, Evidence, SleepSchedule
from contexts.agent.eval import load_fixture
from lab import dataset
from lab import fixture_dataset as fx
from lab import format_traces as ft
from lab.machine import ledger, stop
from runtime import FakeRuntime

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = REPO_ROOT / "fixtures" / "golden"
FIXTURE = REPO_ROOT / "fixtures" / "learning" / fx.DATASET_ID
TEMPLATE_DIR = REPO_ROOT / "fixtures" / "lab" / "qwen35-template"
SCRIPT = REPO_ROOT / "lab" / "format_traces.py"
KEYS = "Where did I leave my keys?"
NOW_PROMPT = f"{SYSTEM} The time now is 2026-10-11T09:30:00-04:00. Use tools to find evidence."


def recorded_cases() -> list[dict]:
    text = (TEMPLATE_DIR / "rendered.jsonl").read_bytes().decode("utf-8")
    return [json.loads(line) for line in text.split("\n") if line.strip()]


RECORDED = {case["name"]: case for case in recorded_cases()}
EXPECTED = fx.load_expected(FIXTURE)   # what a correct label is, by example id, in file order
IDS = list(EXPECTED)
TOTAL = len(IDS)
KEYS_ID = next(name for name, row in EXPECTED.items() if row.question == KEYS)
OVERNIGHT_ID = next(name for name, row in EXPECTED.items() if row.label.tool_calls[0].name == "episodes_between")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LAB_ACTOR", "sonnet-test")
    monkeypatch.setenv("LAB_TIER", "sonnet")
    monkeypatch.delenv("LAB_TASK", raising=False)


# --- the student's tokenizer, in miniature ---------------------------------------------------

class ByteTokenizer:
    """Every byte is a token and each special token of the template is one token: text comes back exactly."""

    SPECIAL = ("<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>",
               "</tool_response>")

    def __init__(self) -> None:
        self.ids = {name: 256 + number for number, name in enumerate(self.SPECIAL)}
        self.names = {number: name for name, number in self.ids.items()}
        self.split = re.compile("(" + "|".join(re.escape(name) for name in self.SPECIAL) + ")")

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for part in self.split.split(text):
            ids.extend([self.ids[part]] if part in self.ids else part.encode("utf-8"))
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        pieces, pending = [], bytearray()
        for number in ids:
            if number < 256:
                pending.append(number)
            else:
                pieces += [pending.decode("utf-8"), self.names[number]]
                pending = bytearray()
        return "".join(pieces + [pending.decode("utf-8")])


class NormalizingTokenizer(ByteTokenizer):
    """Like the student's tokenizer, which normalizes text to NFC before it splits it."""

    def encode(self, text: str) -> list[int]:
        return super().encode(unicodedata.normalize("NFC", text))


# --- helpers -----------------------------------------------------------------------------------

def reference_target(row: fx.Expected) -> str:
    return json.dumps({"tool_calls": [c.model_dump() for c in row.label.tool_calls], "answer": row.label.answer})


def labeled_copy(destination: Path, only: Iterable[str] | None = None) -> Path:
    """A copy of the committed fixture dataset with the reference labels written in, for all examples or `only`."""
    shutil.copytree(FIXTURE, destination)
    rows = fx.load_expected(destination)
    dataset.write_targets(destination, {name: reference_target(rows[name]) for name in (rows if only is None else only)})
    return destination


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_bytes().decode("utf-8").split("\n") if line.strip()]


def only_row() -> dict:
    found = ledger.read()
    assert len(found) == 1, found
    return found[0]


def one_call(name: str, **arguments: object) -> ft.Round:
    return ft.Round((ft.Call(name, dict(arguments)),), ("result",))


def make_trace(*rounds: ft.Round, answer: str = "On the desk.", question: str = KEYS, system: str = NOW_PROMPT) -> ft.Trace:
    return ft.Trace("t", system, question, tuple(rounds), answer)


def check(trace: ft.Trace, **kwargs) -> ft.Checked:
    return ft.check_trace(trace, TOOLS, **kwargs)


@pytest.fixture(scope="module")
def expected():
    return EXPECTED


@pytest.fixture(scope="module")
def results():
    return ft.fixture_tool_results(GOLDEN_DIR)


@pytest.fixture(scope="module")
def formatted(tmp_path_factory, results):
    """The whole pipeline once over the fixture dataset labeled with the reference labels."""
    base = tmp_path_factory.mktemp("formatted")
    root = labeled_copy(base / fx.DATASET_ID)
    out = base / "rows.jsonl"
    return ft.format_dataset(root, results, out, tokenizer=ByteTokenizer()), out


# --- the renderer is the real template -----------------------------------------------------------

@pytest.mark.parametrize("name", sorted(RECORDED))
def test_the_renderer_writes_each_recorded_conversation_exactly_as_the_real_template_did(name):
    case = RECORDED[name]
    assert ft.render(case["messages"], case["tools"]) == case["text"]


def test_the_generation_prompt_is_the_text_up_to_where_the_model_starts_writing_with_thinking_off():
    checked = 0
    for case in RECORDED.values():
        for prompt in case["prompts"]:
            assert ft.render(case["messages"][:prompt["upto"]], case["tools"], add_generation_prompt=True) == prompt["text"]
            assert prompt["text"].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
            checked += 1
    assert checked == 30


def test_the_recording_names_the_template_revision_the_renderer_claims_to_follow():
    source = json.loads((TEMPLATE_DIR / "source.json").read_text(encoding="utf-8"))
    assert (source["repo"], source["revision"], source["cases"]) == (ft.TEMPLATE_REPO, ft.TEMPLATE_REVISION, len(RECORDED))


def test_the_spans_cover_what_the_model_writes_in_each_turn_and_nothing_else():
    case = RECORDED["two-rounds"]
    text, spans = ft.render_with_spans(case["messages"], case["tools"])
    written = [text[start:end] for start, end in spans]
    assert written == [
        "<tool_call>\n<function=search_memory>\n<parameter=q>\nbox\n</parameter>\n</function>\n</tool_call><|im_end|>",
        "<tool_call>\n<function=last_seen>\n<parameter=label>\nbox\n</parameter>\n</function>\n</tool_call><|im_end|>",
        "The box is by the door.<|im_end|>",
    ]
    for start, _ in spans:
        assert text[:start].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


@pytest.mark.parametrize("messages, why", [
    ([], "no messages"),
    ([{"role": "assistant", "content": "hi"}], "no user question"),
    ([{"role": "user", "content": "<tool_response>\nx\n</tool_response>"}], "no user question"),
    ([{"role": "user", "content": "q"}, {"role": "system", "content": "late"}], "system message must come first"),
    ([{"role": "user", "content": "q"}, {"role": "narrator", "content": "x"}], "role the template does not know"),
    ([{"role": "user", "content": ["parts"]}], "other than text"),
    ([{"role": "user", "content": "q"}, {"role": "assistant", "content": "a</think>b"}], "carries thinking"),
    ([{"role": "user", "content": "q"}, {"role": "assistant", "content": "a", "reasoning_content": "hmm"}], "carries thinking"),
])
def test_messages_the_template_cannot_write_with_thinking_off_are_refused(messages, why):
    with pytest.raises(ft.FormatError, match=why):
        ft.render(messages, TOOLS)


# --- one example, end to end ----------------------------------------------------------------------

def test_one_example_end_to_end_is_the_templates_text_with_thinking_off_and_the_call_in_tag_form(formatted, expected):
    done, out = formatted
    row = next(r for r in jsonl(out) if r["id"] == KEYS_ID)
    label = expected[KEYS_ID]
    assert (label.question, [c.name for c in label.label.tool_calls]) == (KEYS, ["last_seen"])
    answer, result = label.label.answer, label.tool_results[0]
    head = RECORDED["one-call"]["text"].split("<|im_start|>assistant")[0]   # system with the tools, and the question
    assert head.endswith(f"<|im_start|>user\n{KEYS}<|im_end|>\n")
    assert row["text"] == (
        head
        + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        + "<tool_call>\n<function=last_seen>\n<parameter=label>\nkeys\n</parameter>\n</function>\n</tool_call><|im_end|>\n"
        + f"<|im_start|>user\n<tool_response>\n{result}\n</tool_response><|im_end|>\n"
        + f"<|im_start|>assistant\n<think>\n\n</think>\n\n{answer}<|im_end|>\n")
    assert row["messages"] == [
        {"role": "system", "content": NOW_PROMPT},
        {"role": "user", "content": KEYS},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "last_seen", "arguments": {"label": "keys"}}}]},
        {"role": "tool", "content": result},
        {"role": "assistant", "content": answer},
    ]
    assert row["tools"] == TOOLS
    assert row["frame_ref"] == "fixture:group:" + label.group
    assert [row["text"][start:end] for start, end in row["spans"]] == [
        "<tool_call>\n<function=last_seen>\n<parameter=label>\nkeys\n</parameter>\n</function>\n</tool_call><|im_end|>",
        f"{answer}<|im_end|>",
    ]


# --- the check over the fixture's traces -------------------------------------------------------

def test_every_fixture_trace_renders_tokenizes_detokenizes_and_reads_back_as_the_same_calls_and_answer(formatted, expected):
    done, out = formatted
    tokenizer = ByteTokenizer()
    rows = jsonl(out)
    assert len(rows) == len(done.rows) == len(expected) == TOTAL >= 20
    for row in rows:
        text = tokenizer.decode(tokenizer.encode(row["text"]))
        assert text == row["text"]
        label = expected[row["id"]].label
        turns = ft.parse_assistant_turns(text, TOOLS)
        assert [[(c.name, c.arguments) for c in turn.calls] for turn in turns] == [
            [(c.name, c.arguments) for c in label.tool_calls], []]
        assert turns[-1].content == label.answer.strip()


def test_nothing_is_rejected_or_failed_when_the_labels_are_the_reference_labels(formatted):
    done, out = formatted
    assert (done.labeled, done.pending, done.rejected, done.failed) == (TOTAL, 0, (), ())
    tokenizer = ByteTokenizer()
    assert list(done.tokens) == [len(tokenizer.encode(row["text"])) for row in jsonl(out)]


def test_each_assistant_turn_starts_with_the_empty_thinking_block_and_no_call_is_written_as_json(formatted):
    _, out = formatted
    turns = []
    for row in jsonl(out):
        found = re.findall(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", row["text"], re.DOTALL)
        assert len(found) == 2
        assert sum(turn.count("<tool_call>") for turn in found) == sum(turn.count("</tool_call>") for turn in found) == 1
        assert row["text"].count("<tool_response>") == row["text"].count("</tool_response>") == 1
        turns += found
    assert len(turns) == 2 * TOTAL
    for turn in turns:
        assert turn.startswith("<think>\n\n</think>\n\n")
        assert '"arguments"' not in turn and '"name"' not in turn and "{" not in turn


def test_the_text_before_each_turn_is_what_the_box_sends_the_model_before_that_turn(formatted):
    _, out = formatted
    for row in jsonl(out):
        turns = [i for i, message in enumerate(row["messages"]) if message["role"] == "assistant"]
        assert len(turns) == len(row["spans"]) == 2
        for index, (start, end) in zip(turns, row["spans"]):
            assert row["text"][:start] == ft.render(row["messages"][:index], row["tools"], add_generation_prompt=True)
            assert row["text"][end:end + 1] == "\n"


def test_a_lossy_tokenizer_is_caught_by_the_round_trip():
    decomposed = "Where is my cafe" + chr(0x301) + " card?"
    trace = make_trace(one_call("search_memory", q="card"), question=decomposed)
    assert check(trace, tokenizer=ByteTokenizer()).problems == ()
    problems = check(trace, tokenizer=NormalizingTokenizer()).problems
    assert len(problems) == 1 and problems[0].startswith("tokenizer: the round trip changes the text at character ")


def test_the_template_rendering_is_a_reference_the_trace_must_equal():
    trace = make_trace(ft.Round((ft.Call("last_seen", {"label": "keys"}),), ("keys last seen Sat 9:00 PM in the desk zone on cam-desk",)),
                       answer="Your keys were on the desk at 9:00 PM.")
    recorded = RECORDED["one-call"]
    assert ft.messages_of(trace) == recorded["messages"]
    assert check(trace, reference=lambda messages, tools: recorded["text"]).problems == ()
    off = check(trace, reference=lambda messages, tools: recorded["text"].replace("<think>\n\n</think>\n\n", "", 1))
    assert len(off.problems) == 1 and off.problems[0].startswith("reference: differs from the chat template's own rendering")


def test_a_template_or_tokenizer_that_fails_is_reported_as_a_problem_of_the_trace_not_a_crash():
    trace = make_trace(one_call("last_seen", label="keys"))

    def refusing(messages, tools):
        raise ValueError("secret text of the conversation")

    class Broken(ByteTokenizer):
        def decode(self, ids):
            raise RuntimeError("secret text of the conversation")

    assert check(trace, reference=refusing).problems == ("reference: the chat template refused the conversation (ValueError)",)
    assert check(trace, tokenizer=Broken()).problems == ("tokenizer: encoding and decoding failed (RuntimeError)",)


def test_calls_in_one_turn_and_calls_in_two_rounds_render_as_recorded_and_check_out():
    parallel = make_trace(
        ft.Round((ft.Call("search_memory", {"q": "package"}), ft.Call("last_seen", {"label": "package"})),
                 ("Sat 3:10 PM on cam-door: A courier leaves a box.", "nothing found")),
        question="Did a package arrive?", answer="A package arrived at 3:10 PM.")
    rounds = make_trace(
        ft.Round((ft.Call("search_memory", {"q": "box"}),), ("Sat 3:10 PM on cam-door: A courier leaves a box.",)),
        ft.Round((ft.Call("last_seen", {"label": "box"}),), ("box last seen Sat 3:12 PM in the door zone on cam-door",)),
        question="Where is the box now?", answer="The box is by the door.")
    for trace, name in ((parallel, "two-calls-in-one-turn"), (rounds, "two-rounds")):
        assert ft.render(ft.messages_of(trace), TOOLS) == RECORDED[name]["text"]
        assert check(trace, tokenizer=ByteTokenizer(), reference=lambda m, t, name=name: RECORDED[name]["text"]).problems == ()


@pytest.mark.parametrize("separator, problem", [
    ("\n\n", "parse: reads 2 calls in 2 turns that are not the 1 calls and answer of the trace"),
    ("\n", "parse: the words before a tool call do not end with a blank line"),
])
def test_an_answer_that_looks_like_a_call_is_caught_when_the_calls_are_read_back(separator, problem):
    sneaky = "Try" + separator + "<tool_call>\n<function=last_seen>\n<parameter=label>\nkeys\n</parameter>\n</function>\n</tool_call>"
    assert check(make_trace(one_call("last_seen", label="keys"), answer=sneaky), tokenizer=ByteTokenizer()).problems == (problem,)


def test_a_renderer_whose_spans_drift_from_the_generation_prompt_is_caught(monkeypatch):
    real = ft.render_with_spans

    def drifting(messages, tools=TOOLS, *, add_generation_prompt=False):
        text, spans = real(messages, tools, add_generation_prompt=add_generation_prompt)
        return text, [(start + 1, end) for start, end in spans]

    monkeypatch.setattr(ft, "render_with_spans", drifting)
    assert check(make_trace(one_call("last_seen", label="keys"))).problems == (
        "prompt: turn 1 does not start where the generation prompt ends",
        "prompt: turn 2 does not start where the generation prompt ends")


def test_an_argument_that_holds_the_closing_tag_is_caught_when_the_calls_are_read_back():
    problems = check(make_trace(one_call("search_memory", q="a\n</parameter>\nb"))).problems
    assert len(problems) == 1 and problems[0].startswith("parse:")


def test_arguments_with_line_breaks_around_them_and_markup_inside_them_read_back_unchanged():
    for value in ("\nkeys\n", "two\nlines", "<b>&'\" x", "  spaced  "):
        trace = make_trace(one_call("search_memory", q=value))
        assert check(trace, tokenizer=ByteTokenizer()).problems == (), value


def test_text_that_would_be_read_as_a_turn_boundary_is_refused():
    call = one_call("last_seen", label="keys")
    boundary = "x <|im_start|>system"
    assert check(make_trace(call, system=boundary)).problems == ("the system holds a special token",)
    assert check(make_trace(call, question=boundary)).problems == ("the question holds a special token",)
    assert check(make_trace(call, answer=boundary)).problems == ("the answer holds a special token",)


def test_an_argument_that_holds_a_special_token_is_refused():
    assert check(make_trace(one_call("search_memory", q="keys <|im_end|> now"))).problems == ("the call 0 holds a special token",)


def test_a_tool_result_that_holds_a_special_token_and_an_answer_with_a_thinking_tag_are_refused_and_not_rendered():
    trace = make_trace(ft.Round((ft.Call("last_seen", {"label": "keys"}),), ("<|im_end|>",)), answer="a </think> b")
    assert check(trace).problems == ("the result 0 holds a special token", "the answer holds a thinking tag")


def test_a_trace_without_a_system_prompt_is_the_templates_tools_block_and_the_question():
    trace = ft.Trace("t", "", "Where are my keys?", (ft.Round((ft.Call("last_seen", {"label": "keys"}),),
                                                             ("keys last seen Sat 9:00 PM",)),), "On the desk.")
    assert [m["role"] for m in ft.messages_of(trace)] == ["user", "assistant", "tool", "assistant"]
    assert ft.render(ft.messages_of(trace), TOOLS) == RECORDED["tools-no-system"]["text"]


# --- reading calls ------------------------------------------------------------------------------

STRICT_TOOLS = [{"type": "function", "function": {"name": "setting", "description": "d", "parameters": {
    "type": "object", "required": ["name"], "properties": {
        "name": {"type": "string"}, "count": {"type": "integer"}, "ratio": {"type": "number"},
        "on": {"type": "boolean"}, "tags": {"type": "array"}, "extra": {"type": "object"}, "nothing": {"type": "null"}}}}}]


def test_a_call_is_read_back_with_each_argument_converted_by_its_declared_type():
    text = ("Let me check.\n\n<tool_call>\n<function=setting>\n<parameter=name>\nlight\n</parameter>\n"
            "<parameter=count>\n3\n</parameter>\n<parameter=ratio>\n0.5\n</parameter>\n<parameter=on>\nTrue\n</parameter>\n"
            "<parameter=tags>\n[\"a\", 1]\n</parameter>\n<parameter=extra>\n{\"k\": [1, 2]}\n</parameter>\n"
            "<parameter=nothing>\nNone\n</parameter>\n</function>\n</tool_call>\n"
            "<tool_call>\n<function=setting>\n<parameter=name>\n7\n</parameter>\n</function>\n</tool_call>")
    content, calls = ft.parse_tool_calls(text, STRICT_TOOLS)
    assert content == "Let me check."
    assert calls == (
        ft.Call("setting", {"name": "light", "count": 3, "ratio": 0.5, "on": True, "tags": ["a", 1], "extra": {"k": [1, 2]},
                            "nothing": None}),
        ft.Call("setting", {"name": "7"}))


def test_a_turn_with_no_call_is_all_content():
    assert ft.parse_tool_calls("On the desk.", TOOLS) == ("On the desk.", ())


@pytest.mark.parametrize("text, why", [
    ("<tool_call>\n<function=last_seen>\n<parameter=label>\nkeys\n</parameter>\n</function>", "is not closed"),
    ("<tool_call>\n<function=last_seen>\n<parameter=label>\nkeys\n</parameter>\n</function>\n</tool_call>\nand then", "something other"),
    ("<tool_call>\n<function=last_seen>\n<parameter=label>\nkeys</parameter>\n</function>\n</tool_call>", "not written in the template's form"),
    ("Look<tool_call>\n<function=last_seen>\n</function>\n</tool_call>", "do not end with a blank line"),
    ("<tool_call>\n<function=last_seen>\n</function>\n</tool_call><tool_call>\n<function=last_seen>\n</function>\n</tool_call>",
     "not separated by a line break"),
])
def test_a_call_that_is_not_in_the_templates_form_is_refused(text, why):
    with pytest.raises(ft.FormatError, match=why):
        ft.parse_tool_calls(text, TOOLS)


def test_a_value_its_type_cannot_hold_is_refused():
    with pytest.raises(ft.FormatError, match="argument count does not parse as its type"):
        ft.parse_tool_calls("<tool_call>\n<function=setting>\n<parameter=count>\nmany\n</parameter>\n</function>\n</tool_call>",
                            STRICT_TOOLS)


def test_an_assistant_turn_without_the_empty_thinking_block_is_refused():
    with pytest.raises(ft.FormatError, match="does not start with the empty thinking block"):
        ft.parse_assistant_turns("<|im_start|>assistant\nOn the desk.<|im_end|>\n", TOOLS)


# --- calls from the teacher ----------------------------------------------------------------------------

def test_the_teachers_openai_call_is_converted_and_never_copied_as_json():
    wire = {"id": "call_1", "type": "function", "function": {"name": "episodes_between",
                                                             "arguments": "{\"end\": \"2026-10-11T08:00:00-04:00\", \"start\": \"2026-10-10T20:00:00-04:00\"}"}}
    label = {"name": "episodes_between", "arguments": {"end": "2026-10-11T08:00:00-04:00", "start": "2026-10-10T20:00:00-04:00"}}
    assert ft.as_call(wire) == ft.as_call(label) == ft.Call("episodes_between", label["arguments"])
    trace = make_trace(ft.Round((ft.as_call(wire),), ("Sat 9:00 PM hour: A person packs a laptop.",)), question="What happened overnight?")
    text, spans = ft.render_with_spans(ft.messages_of(trace), TOOLS)
    assert [text[start:end] for start, end in spans] == [
        "<tool_call>\n<function=episodes_between>\n<parameter=end>\n2026-10-11T08:00:00-04:00\n</parameter>\n"
        "<parameter=start>\n2026-10-10T20:00:00-04:00\n</parameter>\n</function>\n</tool_call><|im_end|>",
        "On the desk.<|im_end|>"]
    assert check(trace, tokenizer=ByteTokenizer()).problems == ()


def test_the_arguments_of_a_wire_call_with_no_arguments_are_empty():
    assert ft.as_call({"function": {"name": "last_seen", "arguments": ""}}) == ft.Call("last_seen", {})
    assert ft.as_call({"name": "last_seen"}) == ft.Call("last_seen", {})


@pytest.mark.parametrize("raw, why", [
    ("last_seen", "not an object"),
    ({"arguments": {}}, "has no name"),
    ({"name": "  "}, "has no name"),
    ({"name": "last_seen", "arguments": "{oops"}, "are not JSON"),
    ({"name": "last_seen", "arguments": "[1]"}, "are not an object"),
    ({"name": "last_seen", "arguments": ["keys"]}, "are not an object"),
])
def test_a_call_that_is_not_a_name_and_an_object_of_arguments_is_refused(raw, why):
    with pytest.raises(ft.FormatError, match=why):
        ft.as_call(raw)


@pytest.mark.parametrize("call, problem", [
    (ft.Call("web_search", {"query": "x"}), "web_search: not one of the tools"),
    (ft.Call("last_seen", {}), "last_seen: argument label is missing"),
    (ft.Call("last_seen", {"label": "keys", "colour": "red"}), "last_seen: argument colour is not one of its arguments"),
    (ft.Call("last_seen", {"label": 3}), "last_seen: argument label is not a string"),
    (ft.Call("episodes_between", {"start": "a"}), "episodes_between: argument end is missing"),
])
def test_a_call_that_misuses_a_tool_is_named_with_the_tool_and_the_argument(call, problem):
    assert ft.call_problems(call, TOOLS) == [problem]


def test_a_correct_call_to_each_tool_has_no_problem():
    for call in (ft.Call("search_memory", {"q": "keys"}), ft.Call("last_seen", {"label": "keys"}),
                 ft.Call("episodes_between", {"start": "a", "end": "b"})):
        assert ft.call_problems(call, TOOLS) == []


# --- reading a dataset ------------------------------------------------------------------------------------

def test_the_results_are_what_the_agents_own_tool_loop_shows_the_model_for_each_reference_call(results, expected):
    for row in expected.values():
        calls = [ft.Call(c.name, c.arguments) for c in row.label.tool_calls]
        assert list(results(row.question, calls)) == row.tool_results, row.id


def test_golden_fixtures_that_cannot_be_read_are_refused_naming_only_the_folder(tmp_path):
    with pytest.raises(ft.FormatError) as refused:
        ft.fixture_tool_results(tmp_path / "nowhere")
    assert str(refused.value) == "cannot read the golden fixtures in nowhere (FileNotFoundError)"
    root = labeled_copy(tmp_path / "d", only=[IDS[1]])
    answer = ft.main([str(root), "--golden", str(tmp_path / "nowhere")])
    assert (answer["ok"], answer["outcome"]) == (False, "refused")


def test_a_call_the_reference_did_not_make_gets_what_the_agents_own_tool_returns_for_it(results):
    memory, now, _ = load_fixture(GOLDEN_DIR)
    qa = QA(FakeRuntime(), memory, clock=lambda: now, sleep=SleepSchedule.load(REPO_ROOT / "config" / "agent.yaml"))
    start, end = "2026-10-10T20:00:00-04:00", "2026-10-10T22:00:00-04:00"
    window = Evidence()
    qa.episodes_between(datetime.fromisoformat(start), datetime.fromisoformat(end), window)
    reference_window = EXPECTED[OVERNIGHT_ID].tool_results
    assert window.lines and "\n".join(window.lines) not in reference_window
    assert results("x", [ft.Call("episodes_between", {"start": start, "end": end})]) == ["\n".join(window.lines)]
    seen = Evidence()
    qa.last_seen("keys", seen)
    assert results("x", [ft.Call("last_seen", {"label": "wallet"}), ft.Call("last_seen", {"label": "keys"})]) == [
        "nothing found", "\n".join(seen.lines)]


def test_examples_with_no_target_are_pending_and_the_labeled_ones_are_read_in_file_order(tmp_path, results):
    root = labeled_copy(tmp_path / "d", only=[IDS[2], IDS[0]])
    loaded = ft.read_traces(dataset.load(root, require_images=False), results)
    assert [t.id for t in loaded.traces] == [IDS[0], IDS[2]]
    assert (loaded.rejected, loaded.pending, loaded.dataset_id) == ((), TOTAL - 2, fx.DATASET_ID)
    first = loaded.traces[0]
    assert (first.question, first.system, first.frame_ref) == (
        EXPECTED[IDS[0]].question, NOW_PROMPT, "fixture:group:" + EXPECTED[IDS[0]].group)


def test_arguments_are_written_in_the_order_the_tool_declares_them_whatever_order_the_teacher_used(tmp_path, results):
    root = labeled_copy(tmp_path / "d", only=[])
    reversed_call = {"name": "episodes_between", "arguments": {"end": "2026-10-11T08:00:00-04:00", "start": "2026-10-10T20:00:00-04:00"}}
    dataset.write_target(root, OVERNIGHT_ID, json.dumps({"tool_calls": [reversed_call], "answer": "Nothing."}))
    trace = ft.read_traces(dataset.load(root, require_images=False), results).traces[0]
    assert list(trace.rounds[0].calls[0].arguments) == ["start", "end"]
    text = ft.render(ft.messages_of(trace), TOOLS)
    assert text.index("<parameter=start>") < text.index("<parameter=end>")


def test_a_label_that_makes_no_valid_trace_is_rejected_with_a_reason_that_quotes_nothing(tmp_path, results):
    root = labeled_copy(tmp_path / "d", only=[])
    secret = "PRIVATE-WORDS"
    labels = {
        IDS[0]: "not json " + secret,
        IDS[1]: json.dumps({"tool_calls": [], "answers": secret}),
        IDS[2]: json.dumps({"tool_calls": [], "answer": "  "}),
        IDS[3]: json.dumps({"tool_calls": [{"name": "web_search", "arguments": {"query": secret}}], "answer": "x"}),
        IDS[4]: json.dumps({"tool_calls": [{"name": "last_seen", "arguments": {}}], "answer": secret}),
        IDS[5]: json.dumps({"tool_calls": [{"name": "last_seen", "arguments": "{" + secret}], "answer": "x"}),
        IDS[6]: json.dumps({"tool_calls": "last_seen", "answer": secret}),
    }
    dataset.write_targets(root, labels)
    loaded = ft.read_traces(dataset.load(root, require_images=False), results)
    assert loaded.traces == ()
    assert [(r.id, r.reason) for r in loaded.rejected] == [
        (IDS[0], "the target is not JSON"),
        (IDS[1], "the target is not an object with tool_calls and answer"),
        (IDS[2], "the answer is empty"),
        (IDS[3], "web_search: not one of the tools"),
        (IDS[4], "last_seen: argument label is missing"),
        (IDS[5], "the arguments of last_seen are not JSON"),
        (IDS[6], "the target is not an object with tool_calls and answer"),
    ]
    assert not any(secret in r.reason for r in loaded.rejected)


def test_a_label_with_no_calls_is_an_answer_given_straight_away(tmp_path, results):
    root = labeled_copy(tmp_path / "d", only=[])
    dataset.write_target(root, IDS[1], json.dumps({"tool_calls": [], "answer": "In the drawer."}))
    trace = ft.read_traces(dataset.load(root, require_images=False), results).traces[0]
    assert trace.rounds == ()
    assert [m["role"] for m in ft.messages_of(trace)] == ["system", "user", "assistant"]
    assert check(trace, tokenizer=ByteTokenizer()).problems == ()


def test_a_tool_that_cannot_be_run_rejects_the_example_and_says_which_kind_of_failure(tmp_path):
    root = labeled_copy(tmp_path / "d", only=[IDS[1]])

    def broken(question, calls):
        raise OSError("disk path C:\\private")

    def short(question, calls):
        return []

    for runner, reason in ((broken, "the tool results could not be produced (OSError)"),
                           (short, "the tool results are not one text per call")):
        loaded = ft.read_traces(dataset.load(root, require_images=False), runner)
        assert [(r.id, r.reason) for r in loaded.rejected] == [(IDS[1], reason)]


def test_a_dataset_of_another_task_is_refused(tmp_path, results):
    root = labeled_copy(tmp_path / "d", only=[IDS[1]])
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["spec"]["task"] = "desk-objects"
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ft.FormatError, match="task is not tool-calling"):
        ft.read_traces(dataset.load(root, require_images=False), results)


def test_a_trace_whose_rendering_does_not_check_out_is_left_out_and_counted_not_repaired(tmp_path, results):
    root = labeled_copy(tmp_path / "d", only=[IDS[1], IDS[2]])
    dataset.write_target(root, IDS[2], json.dumps({"tool_calls": [{"name": "last_seen", "arguments": {"label": "keys"}}],
                                                        "answer": "It is </think> here"}), overwrite=True)
    done = ft.format_dataset(root, results, tmp_path / "rows.jsonl", tokenizer=ByteTokenizer())
    assert [row["id"] for row in done.rows] == [IDS[1]]
    assert [(f.id, f.reason) for f in done.failed] == [(IDS[2], "the answer holds a thinking tag")]
    assert [row["id"] for row in jsonl(tmp_path / "rows.jsonl")] == [IDS[1]]


# --- the rows file --------------------------------------------------------------------------------

def test_the_rows_file_is_one_json_object_per_line_in_file_order_with_the_documented_keys(formatted, expected):
    done, out = formatted
    rows = jsonl(out)
    assert [row["id"] for row in rows] == list(expected)
    assert all(list(row) == ["id", "frame_ref", "messages", "tools", "text", "spans"] for row in rows)
    assert out.read_bytes().endswith(b"\n") and b"\r" not in out.read_bytes()
    assert done.sha256 == hashlib.sha256(out.read_bytes()).hexdigest()
    assert done.path == out
    assert sorted(path.name for path in out.parent.iterdir() if path.is_file()) == ["rows.jsonl"]


def test_the_same_labels_give_the_same_bytes_twice(tmp_path, results):
    root = labeled_copy(tmp_path / "d", only=[IDS[0], IDS[1], IDS[3]])
    first = ft.format_dataset(root, results, tmp_path / "a" / "rows.jsonl")
    second = ft.format_dataset(root, results, tmp_path / "b" / "rows.jsonl", tokenizer=ByteTokenizer())
    assert first.sha256 == second.sha256
    assert (tmp_path / "a" / "rows.jsonl").read_bytes() == (tmp_path / "b" / "rows.jsonl").read_bytes()


def test_a_run_with_no_out_checks_and_writes_nothing(tmp_path, results):
    root = labeled_copy(tmp_path / "d", only=[IDS[1]])
    done = ft.format_dataset(root, results, None, tokenizer=ByteTokenizer())
    assert (len(done.rows), done.path, done.sha256) == (1, None, None)


def test_nothing_is_written_when_no_trace_passes(tmp_path, results):
    root = labeled_copy(tmp_path / "d", only=[])
    dataset.write_target(root, IDS[1], "not json")
    done = ft.format_dataset(root, results, tmp_path / "out" / "rows.jsonl")
    assert (done.rows, done.path, [r.id for r in done.rejected]) == ((), None, [IDS[1]])
    assert not (tmp_path / "out").exists()


def test_a_folder_that_does_not_hold_up_as_a_dataset_raises_dataset_error(tmp_path, results):
    with pytest.raises(dataset.DatasetError):
        ft.format_dataset(tmp_path / "missing", results, None)


# --- the entry point ----------------------------------------------------------------------------------

def test_the_entry_point_formats_the_dataset_and_leaves_a_ledger_row(tmp_path):
    root = labeled_copy(tmp_path / "d")
    out = tmp_path / "rows.jsonl"
    answer = ft.main([str(root), "--out", str(out), "--golden", str(GOLDEN_DIR)])
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    assert answer == {
        "ok": True, "outcome": "formatted", "dataset_id": fx.DATASET_ID, "labeled": TOTAL, "pending": 0, "rows": TOTAL,
        "rejected": 0, "failed": 0, "template": f"Qwen/Qwen3.5-4B@{ft.TEMPLATE_REVISION}", "tokenizer": "not checked",
        "reference": "not checked", "path": str(out), "sha256": digest, "rejects": [], "failures": [],
        "metrics": {"format.traces": TOTAL, "format.rows": TOTAL, "format.rejected": 0, "format.failed": 0},
        "refs": {"dataset_id": fx.DATASET_ID, "rows_sha256": digest}}
    row = only_row()
    assert (row["verb"], row["outcome"], row["actor"], row["tier"]) == ("format-traces", "formatted", "sonnet-test", "sonnet")
    assert row["metrics"] == answer["metrics"]
    assert (row["refs"]["dataset_id"], row["refs"]["rows_sha256"]) == (fx.DATASET_ID, digest)


def test_the_entry_point_without_out_only_checks(tmp_path):
    root = labeled_copy(tmp_path / "d", only=[IDS[0], IDS[1]])
    answer = ft.main([str(root)])
    assert (answer["ok"], answer["outcome"], answer["rows"], answer["pending"], answer["path"]) == (True, "checked", 2, TOTAL - 2, None)
    assert only_row()["outcome"] == "checked"


def test_the_entry_point_reports_rejected_labels_by_id_and_reason_and_still_succeeds(tmp_path):
    root = labeled_copy(tmp_path / "d", only=[IDS[0]])
    dataset.write_target(root, IDS[1], json.dumps({"tool_calls": [{"name": "nope", "arguments": {}}], "answer": "x"}))
    answer = ft.main([str(root), "--out", str(tmp_path / "rows.jsonl")])
    assert (answer["ok"], answer["rows"], answer["rejected"], answer["labeled"]) == (True, 1, 1, 2)
    assert answer["rejects"] == [{"id": IDS[1], "reason": "nope: not one of the tools"}]
    assert answer["metrics"]["format.traces"] == 1


def test_the_entry_point_fails_when_no_example_has_a_target_yet(tmp_path):
    root = labeled_copy(tmp_path / "d", only=[])
    answer = ft.main([str(root), "--out", str(tmp_path / "rows.jsonl")])
    assert (answer["ok"], answer["outcome"], answer["error"]) == (False, "nothing-passed", "no example has a target yet")
    assert not (tmp_path / "rows.jsonl").exists()
    assert only_row()["outcome"] == "nothing-passed"


def test_the_entry_point_names_the_problems_of_a_folder_that_is_not_a_dataset(tmp_path):
    answer = ft.main([str(tmp_path / "missing")])
    assert (answer["ok"], answer["outcome"], answer["problems"]) == (False, "invalid", ["missing: not a folder"])


def test_the_entry_point_refuses_a_dataset_whose_memory_is_not_in_the_repo(tmp_path):
    root = labeled_copy(tmp_path / "d", only=[IDS[1]])
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["action_id"] = "act-3fa91c2b"
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    answer = ft.main([str(root), "--out", str(tmp_path / "rows.jsonl")])
    assert (answer["ok"], answer["outcome"]) == (False, "refused") and "only available for the no-camera fixture" in answer["error"]
    assert not (tmp_path / "rows.jsonl").exists()


def test_a_tokenizer_that_is_not_there_is_refused_before_anything_is_written(tmp_path):
    root = labeled_copy(tmp_path / "d", only=[IDS[1]])
    answer = ft.main([str(root), "--out", str(tmp_path / "rows.jsonl"), "--tokenizer", str(tmp_path / "nowhere")])
    assert (answer["ok"], answer["outcome"], answer["error"]) == (False, "refused", "no tokenizer.json at nowhere")
    assert not (tmp_path / "rows.jsonl").exists()


def test_the_entry_point_refuses_to_run_while_the_machine_is_stopped(tmp_path, capsys):
    state = tmp_path / "state"
    state.mkdir()
    (state / "STOP").write_text("maintenance\n", encoding="utf-8")
    root = labeled_copy(tmp_path / "d", only=[IDS[1]])
    with pytest.raises(stop.Stopped):
        ft.main([str(root), "--out", str(tmp_path / "rows.jsonl")])
    assert json.loads(capsys.readouterr().out)["outcome"] == "stopped"
    assert not (tmp_path / "rows.jsonl").exists()
    assert only_row()["outcome"] == "stopped"


def test_run_by_path_from_another_folder_the_script_prints_one_json_line_and_writes_the_rows(tmp_path):
    root = labeled_copy(tmp_path / "d", only=[IDS[0], IDS[1]])
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    done = subprocess.run([sys.executable, str(SCRIPT), str(root), "--out", str(tmp_path / "rows.jsonl")], cwd=elsewhere,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    answer = json.loads(done.stdout)
    assert (answer["ok"], answer["rows"]) == (True, 2)
    assert [row["id"] for row in jsonl(tmp_path / "rows.jsonl")] == [IDS[0], IDS[1]]
    assert only_row()["verb"] == "format-traces"


def test_the_script_exits_with_one_when_nothing_could_be_formatted(tmp_path):
    root = labeled_copy(tmp_path / "d", only=[])
    done = subprocess.run([sys.executable, str(SCRIPT), str(root)], cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert done.returncode == 1
    assert json.loads(done.stdout)["outcome"] == "nothing-passed"
