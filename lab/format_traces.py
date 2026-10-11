"""Turn labeled tool-calling examples into training rows in the student's own chat format.

    python lab/format_traces.py <dataset folder> --out rows.jsonl --split
    python lab/format_traces.py <dataset folder> --out rows.jsonl --split --tokenizer ~/models/Qwen3.5-4B
    python lab/format_traces.py <dataset folder> --tokenizer ~/models/Qwen3.5-4B       # check only, write nothing

The Jetson serves Qwen/Qwen3.5-4B with `llama-server --jinja --reasoning-budget 0`: the model's own chat template
renders every prompt with thinking off, so each assistant turn is generated right after an empty thinking block,
and the model answers a tool question by writing the template's call form

    <tool_call>
    <function=last_seen>
    <parameter=label>
    keys
    </parameter>
    </function>
    </tool_call>

The teacher (vLLM) hands its calls back as OpenAI JSON. Training on that JSON, or on a thinking block the box never
produces, would teach a format the box does not parse, so this module re-renders every trace the way the template
does and refuses to write a row it cannot prove:

    * as_call() takes a call in either shape the pipeline holds (a label's {"name", "arguments"}, or the OpenAI wire
      form with its arguments as JSON text) and the renderer writes it only in the template's form;
    * render() is the Qwen3.5 template for the messages a tool question produces (system and tools, user, assistant
      calls, tool results, answer), without Jinja, so it runs where only the repo's own requirements are installed.
      Tests compare it with outputs the real template produced (fixtures/lab/qwen35-template/), and a check with the
      student's tokenizer folder compares it with the template itself;
    * each assistant turn starts, in the training text, exactly where the generation prompt the box sends ends
      (`<|im_start|>assistant`, newline, the empty thinking block), and tool results sit in a user turn as
      `<tool_response>` blocks;
    * parse_assistant_turns() reads the calls back out of the rendered text (detokenized, when a tokenizer is given),
      and a trace is kept only when the function names, arguments and answer come back equal and the tokenizer round
      trip changed nothing.

A row of the output JSONL is one trace:

    id          the example id
    frame_ref   the example's frame_ref (a split by frame_ref or id keeps its groups together)
    messages    the conversation, OpenAI-style: system, user, assistant with tool_calls (arguments as objects), tool
                results, the final answer
    tools       the TOOLS schemas of contexts/agent/application/qa.py, as llama-server receives them
    text        messages and tools rendered through the template with thinking off: what the student is trained on
    spans       [start, end) character offsets into text of what the model generates in each assistant turn, from just
                after the empty thinking block to the end of <|im_end|>; loss belongs on these and nowhere else.
                text[:start] is what the box feeds the model before that turn.

Train on `text` and `spans`. If you render again from `messages` and `tools`, load the rows with the json module and
pass enable_thinking=False: the datasets library gives the argument objects of different tools one shared set of
keys, and the template would then print the missing ones as `None`.

With --split the dataset's frozen split.json is read as lab/dataset.py load_split() does (a split that leaks or is
too short is refused) and only the examples it lists under train get a row, so a held-out example cannot reach the
trainer through this file; the ledger row records the split's sha256. Training rows should always be made this way.

Where the tool results come from. A label holds the calls and the answer, not what the calls returned, so each call
is run again through the agent's own tool loop against the fixture night (fixtures/golden/memory.yaml): the text the
box would have shown the model. That is only possible for the no-camera fixture dataset (action_id
`fixture:golden-questions`), whose memory is in the repo. A label whose calls do not validate against the TOOLS
schemas, whose text could forge a turn boundary, or whose rendering does not check out, is dropped and counted in
`rejected` or `failed`, never repaired.

The entry point prints one JSON line (ok, outcome, counts, rejects, the sha256 of the rows file) and exits 0 when
at least one row passed. With --tokenizer (a folder holding tokenizer.json, or the file; needs the `tokenizers`
package, and `transformers` too to compare against the template's own rendering when the folder has chat_template.jinja)
every trace is also encoded and decoded. It writes a ledger row (verb format-traces) and refuses to run while
lab/state/STOP exists.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

if not __package__:  # run by path: the repo root, one folder up, is what holds contexts/, contracts/ and lab/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contexts.agent.application.qa import TOOLS
from lab.dataset import Dataset, DatasetError, load, load_split
from lab.fixture_dataset import FIXTURE_ACTION_ID, GOLDEN_DIR, TASK
from lab.machine import stop
from lab.machine.record import recorded

__all__ = ["TEMPLATE_REPO", "TEMPLATE_REVISION", "EMPTY_THINK", "GENERATION_PROMPT", "FormatError", "Call", "Round",
           "Trace", "Turn", "Rejected", "Checked", "Loaded", "Formatted", "Tokenizer", "Reference", "ToolResults",
           "as_call", "call_problems", "messages_of", "render", "render_with_spans", "parse_tool_calls",
           "parse_assistant_turns", "check_trace", "read_traces", "tool_loop_results", "fixture_tool_results", "load_tokenizer", "build_row",
           "format_dataset", "main"]

TEMPLATE_REPO = "Qwen/Qwen3.5-4B"
TEMPLATE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"   # the revision whose chat template this follows

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_CONFIG = REPO_ROOT / "config" / "agent.yaml"

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
EMPTY_THINK = "<think>\n\n</think>\n\n"   # what the template puts first in an assistant turn when thinking is off
GENERATION_PROMPT = IM_START + "assistant\n" + EMPTY_THINK
SHOWN_LIMIT = 10                           # entries of a list the entry point prints in full
NAME_SHOWN = 40                            # characters of a tool or argument name a reason repeats

# The template's system block when tools are given, verbatim: the heading, one JSON schema per line, the call format.
TOOLS_HEAD = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
TOOLS_TAIL = (
    "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested "
    "within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE the function call, "
    "but NOT after\n- If there is no function call available, answer the question like normal with your current "
    "knowledge and do not tell the user about function calls\n</IMPORTANT>"
)

_SPECIAL = re.compile(r"<\|[A-Za-z_]+\|>")   # a special token written in running text is read as the token itself
_THINK = re.compile(r"</?think>")             # the template reads these in an assistant turn as the model's reasoning
_TURN = re.compile(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", re.DOTALL)
_CALL = re.compile(r"<tool_call>\n<function=([^>\n]*)>\n(.*?)</function>\n</tool_call>", re.DOTALL)
_PARAM = re.compile(r"<parameter=([^>\n]*)>\n(.*?)\n</parameter>\n", re.DOTALL)


class FormatError(ValueError):
    """Something cannot be formatted. str() says why and never quotes the text of a trace, so it can go in a log."""


# --- the pieces of a trace ---------------------------------------------------------------

@dataclass(frozen=True)
class Call:
    """One tool call: the tool's name and its arguments as an object."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Round:
    """The calls the model made in one assistant turn and, in the same order, the text each tool returned."""

    calls: tuple[Call, ...]
    results: tuple[str, ...]


@dataclass(frozen=True)
class Trace:
    """A labeled question: the system prompt, the question, the rounds of calls and results, the final answer."""

    id: str
    system: str
    question: str
    rounds: tuple[Round, ...]
    answer: str
    frame_ref: str | None = None


@dataclass(frozen=True)
class Turn:
    """An assistant turn read back out of rendered text: the words before its calls, and its calls."""

    content: str
    calls: tuple[Call, ...]


@dataclass(frozen=True)
class Rejected:
    """A labeled example that was not turned into a trace, and why (no text of the example in the reason)."""

    id: str
    reason: str


# --- calls ---------------------------------------------------------------------------------

def as_call(raw: Any) -> Call:
    """A tool call in either shape the pipeline holds: a label's {"name", "arguments"}, or the OpenAI wire form
    {"id", "type": "function", "function": {"name", "arguments"}} whose arguments may be JSON text."""
    if not isinstance(raw, Mapping):
        raise FormatError("a tool call is not an object")
    body = raw["function"] if isinstance(raw.get("function"), Mapping) else raw
    name, arguments = body.get("name"), body.get("arguments", {})
    if not isinstance(name, str) or not name.strip():
        raise FormatError("a tool call has no name")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except ValueError:
            raise FormatError(f"the arguments of {name[:NAME_SHOWN]} are not JSON") from None
    if not isinstance(arguments, Mapping):
        raise FormatError(f"the arguments of {name[:NAME_SHOWN]} are not an object")
    return Call(name, dict(arguments))


def _functions(tools: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {tool["function"]["name"]: tool["function"].get("parameters") or {} for tool in tools}


def _is_type(value: Any, kind: str) -> bool:
    if kind == "string":
        return isinstance(value, str)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "array":
        return isinstance(value, list)
    if kind == "object":
        return isinstance(value, dict)
    if kind == "null":
        return value is None
    return True


def call_problems(call: Call, tools: Sequence[Mapping[str, Any]] = TOOLS) -> list[str]:
    """Why the call is not a valid use of the tools: an unknown tool, a missing or unknown argument, a wrong type."""
    schema = _functions(tools).get(call.name)
    name = call.name[:NAME_SHOWN]
    if schema is None:
        return [f"{name}: not one of the tools"]
    properties = schema.get("properties", {})
    problems = [f"{name}: argument {key} is missing" for key in schema.get("required", []) if key not in call.arguments]
    for key, value in call.arguments.items():
        if key not in properties:
            problems.append(f"{name}: argument {key[:NAME_SHOWN]} is not one of its arguments")
        elif isinstance(properties[key].get("type"), str) and not _is_type(value, properties[key]["type"]):
            problems.append(f"{name}: argument {key} is not a {properties[key]['type']}")
    return problems


def _in_schema_order(call: Call, tools: Sequence[Mapping[str, Any]]) -> Call:
    """The call with its arguments in the order the tool declares them, so the student sees one order whatever
    order the teacher wrote them in."""
    properties = list((_functions(tools).get(call.name) or {}).get("properties", {}))
    ordered = {key: call.arguments[key] for key in properties if key in call.arguments}
    ordered.update({key: value for key, value in call.arguments.items() if key not in ordered})
    return Call(call.name, ordered)


# --- the Qwen3.5 chat template -----------------------------------------------------------------

def messages_of(trace: Trace) -> list[dict[str, Any]]:
    """The trace as OpenAI-style messages: system (when there is one), the question, an assistant message with its
    tool_calls and the tool messages after it for each round, then the answer."""
    messages: list[dict[str, Any]] = []
    if trace.system:
        messages.append({"role": "system", "content": trace.system})
    messages.append({"role": "user", "content": trace.question})
    for round_ in trace.rounds:
        messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": call.name, "arguments": call.arguments}} for call in round_.calls]})
        messages.extend({"role": "tool", "content": result} for result in round_.results)
    messages.append({"role": "assistant", "content": trace.answer})
    return messages


def _text(value: Any) -> str:
    """The template's render_content for text: stripped; no content is empty text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    raise FormatError("a message holds something other than text")


def _argument_text(value: Any) -> str:
    """How the template writes an argument: objects and lists as JSON, anything else as Python prints it."""
    if isinstance(value, Mapping) or (isinstance(value, Sequence) and not isinstance(value, (str, bytes))):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _last_query(messages: Sequence[Mapping[str, Any]]) -> int:
    """Index of the last user message that is not itself a tool result: the assistant turns after it are the ones
    the template writes with the thinking block."""
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            text = _text(messages[index].get("content"))
            if not (text.startswith("<tool_response>") and text.endswith("</tool_response>")):
                return index
    raise FormatError("there is no user question in the messages")


def render_with_spans(messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]] = TOOLS, *,
                      add_generation_prompt: bool = False) -> tuple[str, list[tuple[int, int]]]:
    """The conversation as the Qwen3.5 chat template renders it with thinking off, and the [start, end) offsets of
    what the model generates in each assistant turn after the question (from the end of the empty thinking block
    to the end of <|im_end|>). add_generation_prompt ends the text where the model starts writing."""
    if not messages:
        raise FormatError("there are no messages")
    pieces: list[str] = []
    size = 0
    spans: list[tuple[int, int]] = []

    def put(piece: str) -> None:
        nonlocal size
        pieces.append(piece)
        size += len(piece)

    system = _text(messages[0].get("content")) if messages[0].get("role") == "system" else None
    if tools:
        put(IM_START + "system\n" + TOOLS_HEAD)
        for tool in tools:
            put("\n" + json.dumps(tool, ensure_ascii=False))
        put(TOOLS_TAIL)
        if system:
            put("\n\n" + system)
        put(IM_END + "\n")
    elif system is not None:
        put(IM_START + "system\n" + system + IM_END + "\n")

    last_query = _last_query(messages)
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "system":
            if index:
                raise FormatError("a system message must come first")
        elif role == "user":
            put(IM_START + "user\n" + _text(message.get("content")) + IM_END + "\n")
        elif role == "assistant":
            content = _text(message.get("content"))
            if message.get("reasoning_content") or _THINK.search(content):
                raise FormatError("an assistant message carries thinking; this formats thinking-off turns only")
            start = None
            if index > last_query:
                put(IM_START + "assistant\n" + EMPTY_THINK)
                start = size
                put(content)
            else:
                put(IM_START + "assistant\n" + content)
            for number, raw in enumerate(message.get("tool_calls") or []):
                call = as_call(raw)
                lead = ("\n\n" if content else "") if number == 0 else "\n"
                put(lead + "<tool_call>\n<function=" + call.name + ">\n")
                for key, value in call.arguments.items():
                    put("<parameter=" + key + ">\n" + _argument_text(value) + "\n</parameter>\n")
                put("</function>\n</tool_call>")
            put(IM_END)
            if start is not None:
                spans.append((start, size))
            put("\n")
        elif role == "tool":
            if index and messages[index - 1].get("role") != "tool":
                put(IM_START + "user")
            put("\n<tool_response>\n" + _text(message.get("content")) + "\n</tool_response>")
            if index + 1 == len(messages) or messages[index + 1].get("role") != "tool":
                put(IM_END + "\n")
        else:
            raise FormatError("a message has a role the template does not know")
    if add_generation_prompt:
        put(GENERATION_PROMPT)
    return "".join(pieces), spans


def render(messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]] = TOOLS, *,
           add_generation_prompt: bool = False) -> str:
    """render_with_spans() without the spans."""
    return render_with_spans(messages, tools, add_generation_prompt=add_generation_prompt)[0]


# --- reading calls back out of rendered text ---------------------------------------------------------

def _convert(text: str, kind: Any) -> Any:
    """An argument's text as the value its schema type asks for (a string stays a string)."""
    if kind == "integer":
        return int(text)
    if kind == "number":
        return float(text) if any(mark in text for mark in ".eE") else int(text)
    if kind == "boolean":
        if text.lower() not in ("true", "false"):
            raise ValueError(text)
        return text.lower() == "true"
    if kind in ("array", "object"):
        return json.loads(text)
    if kind == "null":
        return None
    return text


def parse_tool_calls(text: str, tools: Sequence[Mapping[str, Any]] = TOOLS) -> tuple[str, tuple[Call, ...]]:
    """What an assistant turn says (after its empty thinking block): the words before the calls and the calls.

    Reads the template's call form the way a qwen3_coder parser does: a value loses the one newline the template
    puts on each side of it and is converted by its schema type. FormatError for anything else: a call that is not
    closed, text after the calls, a value its type cannot hold."""
    first = text.find("<tool_call>")
    if first < 0:
        return text, ()
    head = text[:first]
    if head and not head.endswith("\n\n"):
        raise FormatError("the words before a tool call do not end with a blank line")
    functions = _functions(tools)
    calls: list[Call] = []
    position = first
    while position < len(text):
        found = _CALL.match(text, position)
        if found is None:
            raise FormatError("a tool call is not closed, or something other than a call follows one")
        name, body = found.group(1), found.group(2)
        properties = (functions.get(name) or {}).get("properties", {})
        arguments: dict[str, Any] = {}
        offset = 0
        while offset < len(body):
            parameter = _PARAM.match(body, offset)
            if parameter is None:
                raise FormatError(f"{name}: an argument is not written in the template's form")
            key, value = parameter.group(1), parameter.group(2)
            try:
                arguments[key] = _convert(value, (properties.get(key) or {}).get("type"))
            except ValueError:
                raise FormatError(f"{name}: argument {key} does not parse as its type") from None
            offset = parameter.end()
        calls.append(Call(name, arguments))
        position = found.end()
        if position < len(text):
            if text[position] != "\n":
                raise FormatError("two tool calls are not separated by a line break")
            position += 1
    return head[:-2] if head else "", tuple(calls)


def parse_assistant_turns(text: str, tools: Sequence[Mapping[str, Any]] = TOOLS) -> list[Turn]:
    """The assistant turns of a rendered conversation, each with the empty thinking block taken off and its calls
    read. FormatError for a turn that does not start with the empty thinking block."""
    turns = []
    for found in _TURN.finditer(text):
        body = found.group(1)
        if not body.startswith(EMPTY_THINK):
            raise FormatError("an assistant turn does not start with the empty thinking block")
        content, calls = parse_tool_calls(body[len(EMPTY_THINK):], tools)
        turns.append(Turn(content, calls))
    return turns


# --- the check -----------------------------------------------------------------------------

class Tokenizer(Protocol):
    """What the check needs of a tokenizer: ids for a text, and the exact text back (special tokens kept)."""

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: Sequence[int]) -> str: ...


Reference = Callable[[list[dict[str, Any]], list[dict[str, Any]]], str]   # (messages, tools) -> the template's own text


@dataclass(frozen=True)
class Checked:
    """The outcome of checking one trace: what is wrong (empty when nothing is) and its length in tokens."""

    problems: tuple[str, ...]
    tokens: int | None = None


def _first_difference(left: str, right: str) -> int:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))


def _forged(trace: Trace) -> list[str]:
    """Text that would be read as structure instead of as words."""
    problems = []
    texts = {"system": trace.system, "question": trace.question, "answer": trace.answer}
    texts.update({f"result {n}": result for n, round_ in enumerate(trace.rounds) for result in round_.results})
    texts.update({f"call {n}": json.dumps([call.name, call.arguments], ensure_ascii=False)
                  for n, call in enumerate(call for round_ in trace.rounds for call in round_.calls)})
    for where, text in texts.items():
        if _SPECIAL.search(text):
            problems.append(f"the {where} holds a special token")
    if _THINK.search(trace.answer):
        problems.append("the answer holds a thinking tag")
    return problems


def check_trace(trace: Trace, tools: Sequence[Mapping[str, Any]] = TOOLS, tokenizer: Tokenizer | None = None,
                reference: Reference | None = None) -> Checked:
    """Render the trace and prove the rendering: every assistant turn starts where the generation prompt the box
    sends ends, the tokenizer (when given) gives the text back unchanged, the template's own rendering (when given)
    is the same text, and the calls and the answer read back from the detokenized text equal the trace's."""
    problems = _forged(trace)
    if problems:   # such text is not rendered: the template would read it as structure
        return Checked(tuple(problems))
    messages = messages_of(trace)
    try:
        text, spans = render_with_spans(messages, tools)
    except FormatError as why:
        return Checked((f"render: {why}",))
    plain_tools = [dict(tool) for tool in tools]

    if reference is not None:
        try:
            official = reference(messages, plain_tools)
        except Exception as exc:  # the template itself refuses the conversation
            problems.append(f"reference: the chat template refused the conversation ({type(exc).__name__})")
        else:
            if official != text:
                problems.append(f"reference: differs from the chat template's own rendering at character "
                                f"{_first_difference(official, text)}")

    turns = [index for index, message in enumerate(messages) if message["role"] == "assistant"]
    for index, (start, _) in zip(turns, spans):
        prompt = render(messages[:index], tools, add_generation_prompt=True)
        if not text.startswith(prompt) or start != len(prompt):
            problems.append(f"prompt: turn {turns.index(index) + 1} does not start where the generation prompt ends")

    back, count = text, None
    if tokenizer is not None:
        try:
            ids = tokenizer.encode(text)
            back, count = tokenizer.decode(ids), len(ids)
        except Exception as exc:  # a tokenizer that cannot take the text, or cannot give it back
            problems.append(f"tokenizer: encoding and decoding failed ({type(exc).__name__})")
            back = text
        else:
            if back != text:
                problems.append(f"tokenizer: the round trip changes the text at character {_first_difference(text, back)}")

    try:
        parsed = parse_assistant_turns(back, tools)
    except FormatError as why:
        problems.append(f"parse: {why}")
    else:
        expected = [Turn("", tuple(round_.calls)) for round_ in trace.rounds] + [Turn(trace.answer.strip(), ())]
        if [(t.content, t.calls) for t in parsed] != [(t.content, t.calls) for t in expected]:
            problems.append(f"parse: reads {sum(len(t.calls) for t in parsed)} calls in {len(parsed)} turns "
                            f"that are not the {sum(len(r.calls) for r in trace.rounds)} calls and answer of the trace")
    return Checked(tuple(problems), count)


# --- reading a dataset -----------------------------------------------------------------------

ToolResults = Callable[[str, Sequence[Call]], Sequence[str]]   # (question, calls of one round) -> one text per call


@dataclass(frozen=True)
class Loaded:
    """The labeled examples of a dataset: those that became traces, those that did not, and how many still wait."""

    dataset_id: str
    traces: tuple[Trace, ...]
    rejected: tuple[Rejected, ...]
    pending: int


def _trace(example: Any, system: str, results: ToolResults, tools: Sequence[Mapping[str, Any]]) -> Trace:
    try:
        label = json.loads(example.target)
    except ValueError:
        raise FormatError("the target is not JSON") from None
    if not isinstance(label, dict) or set(label) != {"tool_calls", "answer"} or not isinstance(label["tool_calls"], list):
        raise FormatError("the target is not an object with tool_calls and answer")
    answer = label["answer"]
    if not isinstance(answer, str) or not answer.strip():
        raise FormatError("the answer is empty")
    question = example.context
    if not isinstance(question, str) or not question.strip():
        raise FormatError("the example has no question in context")
    calls = [_in_schema_order(as_call(raw), tools) for raw in label["tool_calls"]]
    problems = [problem for call in calls for problem in call_problems(call, tools)]
    if problems:
        raise FormatError("; ".join(problems))
    rounds: tuple[Round, ...] = ()
    if calls:
        try:
            shown = tuple(results(question, calls))
        except FormatError:
            raise
        except Exception as exc:  # whatever runs the tools: say which kind of failure, quote nothing
            raise FormatError(f"the tool results could not be produced ({type(exc).__name__})") from None
        if len(shown) != len(calls) or not all(isinstance(text, str) for text in shown):
            raise FormatError("the tool results are not one text per call")
        rounds = (Round(tuple(calls), shown),)
    return Trace(example.id, system, question, rounds, answer, example.frame_ref)


def read_traces(data: Dataset, results: ToolResults, tools: Sequence[Mapping[str, Any]] = TOOLS) -> Loaded:
    """The labeled examples of a loaded dataset as traces, in file order. An example whose target does not make a
    valid trace is listed in `rejected` with the reason and left out; examples with no target are `pending`."""
    if data.manifest.spec.task != TASK:
        raise FormatError(f"the dataset's task is not {TASK}")
    system = data.manifest.spec.instruction
    traces, rejected = [], []
    for example in data.labeled():
        try:
            traces.append(_trace(example, system, results, tools))
        except FormatError as why:
            rejected.append(Rejected(example.id, str(why)))
    return Loaded(data.dataset_id, tuple(traces), tuple(rejected), len(data.pending()))


def tool_loop_results(memory: Any, now: Any, sleep: Any) -> ToolResults:
    """Tool results against one memory at one time: the calls are run through the agent's own tool loop, and what
    the loop shows the model for each call is the result."""
    from contexts.agent.application.qa import QA
    from contracts.types import ChatResult, ToolCall
    from runtime import FakeRuntime

    def results(question: str, calls: Sequence[Call]) -> list[str]:
        runtime = FakeRuntime(chat_replies=[
            ChatResult(content="", model_version="fake-0",
                       tool_calls=[ToolCall(id=f"call-{n}", name=call.name, arguments=dict(call.arguments))
                                   for n, call in enumerate(calls)]),
            ChatResult(content="done", model_version="fake-0")])
        asyncio.run(QA(runtime, memory, clock=lambda: now, sleep=sleep).tool_loop(question))
        return [message.content for message in runtime.calls[-1][1] if message.role == "tool"]

    return results


def fixture_tool_results(golden_dir: str | Path = GOLDEN_DIR) -> ToolResults:
    """Tool results for the no-camera fixture: tool_loop_results() against the fixture night at the fixture's now."""
    from contexts.agent.application.qa import SleepSchedule
    from contexts.agent.eval import load_fixture

    try:
        memory, now, _ = load_fixture(Path(golden_dir))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise FormatError(f"cannot read the golden fixtures in {Path(golden_dir).name} ({type(exc).__name__})") from None
    return tool_loop_results(memory, now, SleepSchedule.load(AGENT_CONFIG))


# --- the student's tokenizer ---------------------------------------------------------------------------

class _FastTokenizer:
    """tokenizer.json through the `tokenizers` package: special tokens are matched in the text and kept on decode."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def decode(self, ids: Sequence[int]) -> str:
        return self._tokenizer.decode(list(ids), skip_special_tokens=False)


def load_tokenizer(path: str | Path) -> tuple[Tokenizer, Reference | None]:
    """The student's tokenizer from a folder holding tokenizer.json (or from the file), and, when the folder also
    has chat_template.jinja and `transformers` is installed, a function that renders (messages, tools) with the
    template itself, thinking off. FormatError when the file is missing or `tokenizers` is not installed."""
    target = Path(path).expanduser()
    file = target / "tokenizer.json" if target.is_dir() else target
    if not file.is_file():
        raise FormatError(f"no tokenizer.json at {target.name or str(target)}")
    try:
        from tokenizers import Tokenizer as FastTokenizer
    except ImportError:
        raise FormatError("the tokenizer check needs the tokenizers package (pip install tokenizers)") from None
    reference: Reference | None = None
    if target.is_dir() and (target / "chat_template.jinja").is_file():
        try:
            from transformers import AutoTokenizer
        except ImportError:
            pass
        else:
            official = AutoTokenizer.from_pretrained(str(target))

            def render_officially(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
                return official.apply_chat_template(messages, tools=tools, tokenize=False, enable_thinking=False)

            reference = render_officially
    return _FastTokenizer(FastTokenizer.from_file(str(file))), reference


# --- rows -----------------------------------------------------------------------------------

def build_row(trace: Trace, tools: Sequence[Mapping[str, Any]] = TOOLS) -> dict[str, Any]:
    """The JSONL row of a trace (see the module docstring)."""
    messages = messages_of(trace)
    text, spans = render_with_spans(messages, tools)
    return {"id": trace.id, "frame_ref": trace.frame_ref, "messages": messages, "tools": [dict(tool) for tool in tools],
            "text": text, "spans": [[start, end] for start, end in spans]}


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    """Replace the file with the rows (a temporary file beside it, then os.replace); the sha256 of what was written."""
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class Formatted:
    """What format_dataset() did."""

    dataset_id: str
    labeled: int
    pending: int
    rows: tuple[dict[str, Any], ...]
    rejected: tuple[Rejected, ...]                 # labels that were not valid traces
    failed: tuple[Rejected, ...]                   # traces whose rendering did not check out
    path: Path | None
    sha256: str | None
    tokens: tuple[int, ...] = ()                   # per row, when a tokenizer was given
    held_out: int = 0                              # examples the frozen split keeps out of training (split only)
    split_sha256: str | None = None                # of split.json (split only)


def format_dataset(root: str | Path, results: ToolResults, out: str | Path | None = None, *,
                   tokenizer: Tokenizer | None = None, reference: Reference | None = None,
                   tools: Sequence[Mapping[str, Any]] = TOOLS, split: bool = False) -> Formatted:
    """Read the labeled examples of the dataset folder, check every trace, and write the rows of those that passed
    to `out` (nothing is written when out is None or no row passed). With split, only the examples the folder's
    frozen split.json lists under train are read: a held-out example gets no row, and `labeled` and `pending` count
    the train side. DatasetError when the folder, or its split, does not hold up."""
    held_out, split_digest = 0, None
    if split:
        frozen = load_split(root, require_images=False)
        train = {example.id for example in frozen.train}
        data = replace(frozen.data, examples=tuple(example for example in frozen.data.examples if example.id in train))
        held_out, split_digest = len(frozen.held_out), frozen.sha256
    else:
        data = load(root, require_images=False)
    loaded = read_traces(data, results, tools)
    rows, failed, tokens = [], [], []
    for trace in loaded.traces:
        checked = check_trace(trace, tools, tokenizer, reference)
        if checked.problems:
            failed.append(Rejected(trace.id, "; ".join(checked.problems)))
            continue
        rows.append(build_row(trace, tools))
        if checked.tokens is not None:
            tokens.append(checked.tokens)
    path, digest = None, None
    if out is not None and rows:
        path = Path(out)
        digest = _write_rows(path, rows)
    return Formatted(loaded.dataset_id, len(data.labeled()), loaded.pending, tuple(rows), loaded.rejected,
                     tuple(failed), path, digest, tuple(tokens), held_out, split_digest)


# --- the entry point -----------------------------------------------------------------------

def _listed(items: Sequence[Rejected]) -> list[dict[str, str]]:
    return [{"id": item.id, "reason": item.reason} for item in items[:SHOWN_LIMIT]]


@recorded("format-traces")
@stop.guarded
def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Format the labeled tool-calling examples of a dataset folder into training rows for the student."""
    parser = argparse.ArgumentParser(prog="python lab/format_traces.py", description=main.__doc__, allow_abbrev=False)
    parser.add_argument("dataset", help="the <dataset_id> folder: manifest.json, examples.jsonl with targets")
    parser.add_argument("--out", help="the JSONL file to write; without it the traces are only checked")
    parser.add_argument("--golden", default=str(GOLDEN_DIR),
                        help="questions.yaml and memory.yaml the tool results come from (default: fixtures/golden)")
    parser.add_argument("--tokenizer", help="a folder with tokenizer.json (and chat_template.jinja), or the file: "
                                            "encode and decode every trace, and compare with the template's rendering")
    parser.add_argument("--split", action="store_true",
                        help="only the examples the dataset's frozen split.json lists under train get a row")
    args = parser.parse_args(argv)
    try:
        data = load(args.dataset, require_images=False)
        if data.manifest.action_id != FIXTURE_ACTION_ID:
            raise FormatError("tool results are only available for the no-camera fixture dataset, "
                              "whose memory is in the repo")
        tokenizer, reference = load_tokenizer(args.tokenizer) if args.tokenizer else (None, None)
        done = format_dataset(args.dataset, fixture_tool_results(args.golden), args.out,
                              tokenizer=tokenizer, reference=reference, split=args.split)
    except DatasetError as err:
        return {"ok": False, "outcome": "invalid", "problems": err.problems, "error": str(err)}
    except FormatError as err:
        return {"ok": False, "outcome": "refused", "error": str(err)}
    metrics: dict[str, float] = {"format.traces": done.labeled - len(done.rejected), "format.rows": len(done.rows),
                                 "format.rejected": len(done.rejected), "format.failed": len(done.failed)}
    if done.tokens:
        metrics.update({"format.tokens_total": sum(done.tokens), "format.tokens_max": max(done.tokens)})
    if done.split_sha256:
        metrics["format.held_out_skipped"] = done.held_out
    answer: dict[str, Any] = {
        "ok": bool(done.rows), "outcome": "formatted" if done.path else "checked" if done.rows else "nothing-passed",
        "dataset_id": done.dataset_id, "labeled": done.labeled, "pending": done.pending, "rows": len(done.rows),
        "rejected": len(done.rejected), "failed": len(done.failed),
        "template": f"{TEMPLATE_REPO}@{TEMPLATE_REVISION}",
        "tokenizer": "checked" if tokenizer else "not checked",
        "reference": "checked" if reference else "not checked",
        "path": str(done.path) if done.path else None, "sha256": done.sha256,
        "rejects": _listed(done.rejected), "failures": _listed(done.failed),
        "metrics": metrics, "refs": {"dataset_id": done.dataset_id, "rows_sha256": done.sha256},
    }
    if done.tokens:
        answer["tokens"] = {"total": sum(done.tokens), "max": max(done.tokens), "min": min(done.tokens)}
    if done.split_sha256:
        answer["split"] = {"sha256": done.split_sha256, "held_out_skipped": done.held_out}
        answer["refs"]["split_sha256"] = done.split_sha256
    if not done.rows:
        answer["error"] = "no trace passed the check" if done.labeled else "no example has a target yet"
    return answer


if __name__ == "__main__":
    result = main()
    print(json.dumps(result))
    sys.exit(0 if result.get("ok", True) else 1)
