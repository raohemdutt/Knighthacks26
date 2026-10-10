"""Label a dataset with the teacher: each example still waiting for a label gets one, checked before it is kept.

    python lab/label.py data/exports/desk-objects-3fa91c2b --account C
    python lab/label.py fixtures/learning/tool-calling-fixture --account C --workers 4
    python lab/label.py <dataset folder> --url http://127.0.0.1:8000        # a teacher reached directly

What it reads. A dataset folder in the box's export layout, through lab/dataset.py and nothing else, and only one of
two kinds: an approved export, which is a data/exports/<dataset_id>/ folder named for the dataset id in its manifest
(lab/dataset.py already refuses a manifest without the approval's action id), or the no-camera fixture that
lab/fixture_dataset.py builds, wherever it sits (its action id says it is fixture data). Any other folder is refused
before the teacher is asked anything.

Who answers. The teacher is the vLLM server that serves Qwen3.8-27B under the model name "teacher" on port 8000 of a
pod. --account reaches it through the hub proxy, https://<hub>/user/<account>/proxy/8000/v1/chat/completions, with
the account's hub token in the Authorization header: lab/hub.py finds the hub's address and the token, and the
token is cut out of every message. --url names a server directly. Before the first example the run asks the server
what it serves (GET /v1/models) and stops unless it serves "teacher" from a Qwen3.8-27B checkpoint.

How it asks. Every request turns thinking off (chat_template_kwargs enable_thinking false) and decodes greedily by
default (temperature 0, top_p 1, seed 0, at most 512 new tokens), so a label is the teacher's most likely answer.
Answers are streamed with an Accept header of exactly text/event-stream, because the hub proxy passes a stream
through only for that header and cuts a buffered answer after 125 seconds; --no-stream asks for one JSON answer
instead. A gateway error (502, 503, 504, 524), a rate limit (429), a dropped connection, a stream cut short and an
error the server sends in the middle of a stream are tried again after 1, 2 and 4 seconds. A redirect to the login page, a refusal (401, 403), a missing model or path (404) and a pod that is
not running (424) stop the run at once, as do five examples in a row that got no usable answer.

Two kinds of task.

    A tool-trace task, whose output schema has a top-level "tool_calls" property, like the fixture's tool-calling
    task: the instruction is the system prompt and the example's context is the question. The teacher gets the
    agent's TOOLS and runs the agent's tool loop: each call it makes is run by the agent's own tool code against the
    fixture memory (fixtures/golden, at the fixture's time) and the result goes back to it, for at most the agent's
    MAX_TOOL_CALLS rounds of calls. The label is the JSON text
    {"tool_calls": [{"name": ..., "arguments": {...}}, ...], "answer": ...}, every call in the order made.

    Any other task: one request holding the example's image (inline, as a data URL), the instruction and the box's
    caption; with an output schema the teacher is asked for JSON that follows it.

What is kept. A label is checked before it is stored, and dropped, never repaired, when it is cut off at the token
limit (truncated), carries a thinking block (thinking), says nothing (empty), is not JSON while the task has a schema
(not_json), does not follow the task's TaskSpec.output_schema (schema), passes tool arguments that are not a JSON
object (bad_arguments), still calls tools after the last round (tool_rounds), answers with tool-call markup the
server did not parse (tool_markup: vLLM does that when it is served without --enable-auto-tool-choice and
--tool-call-parser), or, for a free-text task (an empty schema), fails the check the task brings (check). A dropped
example stays pending, so the next run asks again. A kept label goes into examples.jsonl through lab/dataset.py's
write_target, which replaces the file whole under a lock: a run can be stopped at any point and started again,
with nothing lost and no example asked twice once its label is in. A tool-trace task also appends one row per kept
label to traces.jsonl beside it, {"id", "run_id", "tool_calls": [{"name", "arguments", "result"}], "answer"}: the
calls with what each one returned, which the label leaves out. When an interrupted run leaves two rows for one id,
the last one counts.

What is recorded. Every example asked writes a label-example row to the lab ledger: outcome accepted, rejected or
failed, the reason as a word (never the text of a label, a caption or a question), the wall-clock seconds the
example took (label.seconds), the prompt and completion tokens the server counted (label.prompt_tokens,
label.completion_tokens, label.tokens), the requests made (label.requests) and the tool calls run (label.tool_calls),
with the decode settings and the served checkpoint as its conditions. The entry point's own row (verb label) holds
the run's counts, its rejects and failures by reason, and the per-example means.

Pods. Nothing is copied to a pod: label.py runs where the export is and sends the teacher requests, each holding one
example, and vLLM writes no request to disk unless it is started with --enable-log-requests, which
lab/pod_a_teacher.sh does not pass. Run on a pod (JUPYTERHUB_USER or JUPYTERHUB_API_TOKEN is set), it refuses an
export, because the copy it would label there would outlive the run; the fixture holds nothing personal and may be
labeled anywhere.

The entry point prints one JSON line (ok, outcome, counts, metrics) and exits 0 when every pending example was asked
and 1 otherwise. It refuses to start while lab/state/STOP exists, and stops between examples when the file appears.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import concurrent.futures
import datetime as dt
import json
import mimetypes
import os
import re
import secrets
import sys
import time
import urllib.parse
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if not __package__:  # run by path: the repo root, one folder up, is what holds contracts/ and lab/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from contracts.learning import DatasetExample, DatasetManifest, TaskSpec
from lab import hub
from lab.dataset import Dataset, DatasetError, load, write_target
from lab.machine import stop
from lab.machine.lock import FileLock, LockTimeout
from lab.machine.record import record, recorded

__all__ = ["MODEL", "EXPECTED_MODEL", "TEACHER_PORT", "STREAM_ACCEPT", "TRACES", "REJECT_REASONS", "Decode",
           "Completion", "Call", "TeacherError", "LabelRefused", "Teacher", "Toolbox", "Run", "golden_toolbox",
           "is_trace_task", "running_on_pod", "source_problem", "label_dataset", "main"]

MODEL = "teacher"                 # the name vLLM serves the teacher under (lab/pod_a_teacher.sh, --served-model-name)
EXPECTED_MODEL = "Qwen3.8-27B"    # what the served checkpoint's name must hold
TEACHER_PORT = 8000
CHAT_PATH = "/v1/chat/completions"
MODELS_PATH = "/v1/models"
STREAM_ACCEPT = "text/event-stream"   # the one Accept value the hub proxy streams for; anything else is buffered
TRACES = "traces.jsonl"
TRACES_LOCK = ".traces.lock"
EXPORTS_PARENT = ("data", "exports")  # an approved export sits at <...>/data/exports/<dataset_id>/
POD_ENV = ("JUPYTERHUB_USER", "JUPYTERHUB_API_TOKEN")   # set in every JupyterHub single-user server and its terminals

DEFAULT_MAX_TOKENS = 512
RETRIES = 3                        # further tries of a request that met a gateway error or a dropped connection
BACKOFF_FIRST_S = 1.0
BACKOFF_MAX_S = 8.0
CONNECT_TIMEOUT_S = 30.0
STREAM_READ_TIMEOUT_S = 120.0      # longest silence between two pieces of a streamed answer
BUFFERED_TIMEOUT_S = 130.0         # a buffered answer: the proxy cuts one at 125 s and answers 524 first
MAX_FAILURES = 5                   # examples in a row with no usable answer before the run stops
DETAIL_CHARS = 200

REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
FATAL_STATUSES = frozenset({401, 403, 404, 424})
RETRY_STATUSES = frozenset({429, 502, 503, 504, 524})

REJECT_REASONS = ("truncated", "thinking", "empty", "not_json", "schema", "bad_arguments", "tool_rounds",
                  "tool_markup", "check")
UNITS = {"label.seconds": "s", "label.prompt_tokens": "tokens", "label.completion_tokens": "tokens",
         "label.tokens": "tokens", "label.requests": "requests", "label.tool_calls": "calls"}

FENCED = re.compile(r"^```[A-Za-z]*[ \t]*\n(?P<body>.*?)\n?```$", re.DOTALL)
TAG = re.compile(r"<[^>]+>")


# --- errors ---------------------------------------------------------------------------------

class TeacherError(Exception):
    """A call that got no usable answer. fatal: nothing later can succeed (a refused token, a missing model, a pod
    that is not running), so the run stops. retry: the same request may succeed when sent again."""

    def __init__(self, message: str, status: int | None = None, *, fatal: bool = False, retry: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.fatal = fatal
        self.retry = retry


class LabelRefused(Exception):
    """The run was refused before the teacher was asked anything; str() says why."""


# --- the teacher ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Decode:
    """The sampling settings of every request. Thinking is always off."""

    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = DEFAULT_MAX_TOKENS
    seed: int | None = 0

    def payload(self) -> dict[str, Any]:
        body: dict[str, Any] = {"temperature": self.temperature, "top_p": self.top_p, "max_tokens": self.max_tokens,
                                "chat_template_kwargs": {"enable_thinking": False}}
        if self.seed is not None:
            body["seed"] = self.seed
        return body


@dataclass(frozen=True)
class Call:
    """One tool call as the teacher made it: arguments is the JSON text it wrote."""

    id: str
    name: str
    arguments: str


@dataclass
class Completion:
    """One answer: its text, the tool calls it made, why it stopped, the tokens the server counted, and how long the
    request took by the caller's clock."""

    content: str
    calls: list[Call]
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    seconds: float = 0.0


def _one_line(text: str, limit: int = DETAIL_CHARS) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[:limit - 3] + "..."


def _usage(body: Any) -> tuple[int | None, int | None]:
    usage = body.get("usage") if isinstance(body, Mapping) else None
    if not isinstance(usage, Mapping):
        return None, None
    prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
    return (prompt if isinstance(prompt, int) and not isinstance(prompt, bool) else None,
            completion if isinstance(completion, int) and not isinstance(completion, bool) else None)


def _error_message(body: Any) -> str | None:
    """The message of a server error body: vLLM's {"error": {"message"}}, or a bare message or detail."""
    if not isinstance(body, Mapping):
        return None
    nested = body.get("error")
    if isinstance(nested, Mapping) and isinstance(nested.get("message"), str):
        return nested["message"]
    if isinstance(nested, str):
        return nested
    for key in ("message", "detail"):
        if isinstance(body.get(key), str):
            return body[key]
    return None


class Teacher:
    """The teacher's OpenAI-compatible API at base_url (the server's address, without /v1).

    headers go with every request (the hub token, the browser agent the hub's front end wants); scrub cuts secrets
    out of every message; transport replaces the network in tests; clock and sleep time the requests and pace the
    retries. node names the machine behind the address, for the ledger."""

    def __init__(self, base_url: str, *, headers: Mapping[str, str] | None = None,
                 transport: httpx.BaseTransport | None = None, stream: bool = True, model: str = MODEL,
                 retries: int = RETRIES, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.perf_counter, scrub: Callable[[str], str] | None = None,
                 node: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.host = urllib.parse.urlsplit(self.base_url).netloc
        self.stream = stream
        self.model = model
        self.retries = retries
        self.node = node
        self._sleep = sleep
        self._clock = clock
        self._scrub = scrub or (lambda text: text)
        self._client = httpx.Client(
            base_url=self.base_url, transport=transport, follow_redirects=False,
            headers={"User-Agent": hub.USER_AGENT, **(headers or {})},
            timeout=httpx.Timeout(STREAM_READ_TIMEOUT_S if stream else BUFFERED_TIMEOUT_S, connect=CONNECT_TIMEOUT_S))

    @classmethod
    def from_hub(cls, account: hub.Account, *, port: int = TEACHER_PORT, transport: httpx.BaseTransport | None = None,
                 environ: Mapping[str, str] | None = None, **kwargs: Any) -> "Teacher":
        """The teacher behind an account's pod, through the hub proxy: lab/hub.py decides the hub's address (the
        account's AUP_HUB_URL_<letter>, else LAB_HUB_URL, else the default), the token header and the agent."""
        with hub.Hub(account, transport=transport, environ=environ) as client:
            base = client.base_url + client.pod_url(f"/proxy/{port}")
            headers = {**client.ws_headers(), "User-Agent": client.user_agent}
            scrub = client.scrub
        return cls(base, headers=headers, transport=transport, scrub=scrub, node=account.node, **kwargs)

    def __repr__(self) -> str:
        return f"Teacher(host={self.host!r}, model={self.model!r}, stream={self.stream})"

    def __enter__(self) -> "Teacher":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # what the server serves

    def models(self) -> list[dict[str, Any]]:
        """GET /v1/models: the models the server serves, each with its id and, from vLLM, root (the checkpoint)."""
        def once() -> list[dict[str, Any]]:
            try:
                response = self._client.get(MODELS_PATH, headers={"Accept": "application/json"})
            except httpx.TransportError as exc:
                raise TeacherError(self._scrub(f"GET {MODELS_PATH}: no answer ({type(exc).__name__})"),
                                   retry=True) from None
            if response.status_code != 200:
                raise self._status_error(response)
            try:
                data = response.json().get("data")
            except (ValueError, AttributeError):
                data = None
            if not isinstance(data, list):
                raise TeacherError(f"GET {MODELS_PATH}: the answer is not a model list", response.status_code,
                                   fatal=True)
            return [entry for entry in data if isinstance(entry, Mapping)]
        return self._with_retries(once)

    def served_checkpoint(self, expect: str | None = EXPECTED_MODEL) -> str | None:
        """The checkpoint the server serves under this client's model name (vLLM's root). TeacherError, fatal, when
        it serves no such model, or when expect is given and the checkpoint's name does not hold it."""
        entries = {entry.get("id"): entry for entry in self.models()}
        entry = entries.get(self.model)
        if entry is None:
            served = ", ".join(sorted(str(key) for key in entries)) or "nothing"
            raise TeacherError(f"the server does not serve {self.model!r} (it serves {served})", fatal=True)
        root = entry.get("root") if isinstance(entry.get("root"), str) else None
        if expect and (root is None or expect.lower() not in root.lower()):
            raise TeacherError(f"{self.model!r} is served from {root or 'an unnamed checkpoint'}, not {expect}",
                               fatal=True)
        return root

    # answers

    def chat(self, messages: Sequence[Mapping[str, Any]], *, decode: Decode = Decode(),
             tools: Sequence[Mapping[str, Any]] | None = None) -> Completion:
        """One chat completion, thinking off. TeacherError when no usable answer came, after the retries."""
        payload: dict[str, Any] = {"model": self.model, "messages": list(messages), **decode.payload()}
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"
        if self.stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        return self._with_retries(lambda: self._chat_once(payload))

    def _with_retries(self, once: Callable[[], Any]) -> Any:
        for attempt in range(1, self.retries + 2):
            try:
                return once()
            except TeacherError as err:
                if err.fatal or not err.retry or attempt > self.retries:
                    raise
            self._sleep(min(BACKOFF_FIRST_S * 2 ** (attempt - 1), BACKOFF_MAX_S))
        raise AssertionError("unreachable")

    def _chat_once(self, payload: Mapping[str, Any]) -> Completion:
        started = self._clock()
        try:
            if self.stream:
                with self._client.stream("POST", CHAT_PATH, json=payload, headers={"Accept": STREAM_ACCEPT}) as reply:
                    if reply.status_code != 200:
                        reply.read()
                        raise self._status_error(reply)
                    completion = self._read_stream(reply)
            else:
                reply = self._client.post(CHAT_PATH, json=payload, headers={"Accept": "application/json"})
                if reply.status_code != 200:
                    raise self._status_error(reply)
                completion = self._read_body(reply)
        except httpx.TransportError as exc:
            raise TeacherError(self._scrub(f"POST {CHAT_PATH}: no answer ({type(exc).__name__})"), retry=True) from None
        completion.seconds = self._clock() - started
        return completion

    def _read_body(self, reply: httpx.Response) -> Completion:
        try:
            body = reply.json()
            choice = body["choices"][0]
            message = choice.get("message") or {}
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise TeacherError("the answer is not a chat completion", reply.status_code, retry=True) from None
        calls = []
        for number, item in enumerate(message.get("tool_calls") or []):
            if not isinstance(item, Mapping):
                continue
            function = item.get("function") if isinstance(item.get("function"), Mapping) else {}
            arguments = function.get("arguments")
            calls.append(Call(id=str(item.get("id") or f"call-{number}"), name=str(function.get("name") or ""),
                              arguments=arguments if isinstance(arguments, str) else json.dumps(arguments)))
        prompt, completion = _usage(body)
        content = message.get("content")
        return Completion(content=content if isinstance(content, str) else "", calls=calls,
                          finish_reason=choice.get("finish_reason"), prompt_tokens=prompt, completion_tokens=completion)

    def _read_stream(self, reply: httpx.Response) -> Completion:
        """Put a streamed answer back together: the text, each call from its pieces, the finish reason, the usage."""
        text: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        finish: str | None = None
        prompt = completion = None
        done = False
        for line in reply.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                done = True
                break
            try:
                chunk = json.loads(data)
            except ValueError:
                raise TeacherError("a piece of the streamed answer is not JSON", retry=True) from None
            if not isinstance(chunk, Mapping):
                continue
            if "error" in chunk or chunk.get("object") == "error":
                message = _error_message(chunk) or "no message"
                raise TeacherError(self._scrub(f"the server failed mid-answer: {_one_line(message)}"), retry=True)
            if chunk.get("usage"):
                prompt, completion = _usage(chunk)
            try:
                finish = self._take_choices(chunk.get("choices") or [], text, calls) or finish
            except (AttributeError, TypeError):
                raise TeacherError("a piece of the streamed answer is not a chat chunk", retry=True) from None
        if not done or finish is None:
            raise TeacherError("the streamed answer ended before it was finished", retry=True)
        made = [Call(id=str(slot["id"] or f"call-{index}"), name=slot["name"], arguments="".join(slot["arguments"]))
                for index, slot in sorted(calls.items())]
        return Completion(content="".join(text), calls=made, finish_reason=finish, prompt_tokens=prompt,
                          completion_tokens=completion)

    @staticmethod
    def _take_choices(choices: Sequence[Any], text: list[str], calls: dict[int, dict[str, Any]]) -> str | None:
        """Add one chunk's text and tool-call pieces to what came before; its finish reason, if it has one."""
        finish = None
        for choice in choices:
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                text.append(delta["content"])
            for piece in delta.get("tool_calls") or []:
                slot = calls.setdefault(piece.get("index", len(calls)), {"id": None, "name": "", "arguments": []})
                if piece.get("id"):
                    slot["id"] = piece["id"]
                function = piece.get("function") or {}
                if isinstance(function.get("name"), str):
                    slot["name"] += function["name"]
                if isinstance(function.get("arguments"), str):
                    slot["arguments"].append(function["arguments"])
            finish = choice.get("finish_reason") or finish
        return finish

    def _status_error(self, reply: httpx.Response) -> TeacherError:
        status = reply.status_code
        if status in REDIRECT_STATUSES:
            return TeacherError(f"{status}: redirected to a login page: the hub refused the token, or the address "
                                "needs one (use --account)", status, fatal=True)
        return TeacherError(self._scrub(f"{status}: {self._detail(reply)}"), status, fatal=status in FATAL_STATUSES,
                            retry=status in RETRY_STATUSES)

    @staticmethod
    def _detail(reply: httpx.Response) -> str:
        try:
            message = _error_message(reply.json())
        except ValueError:
            message = None
        if message is None:
            message = TAG.sub(" ", reply.text)
        return _one_line(message) or reply.reason_phrase


# --- tools ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Toolbox:
    """What a tool-trace task's teacher may call: the tool schemas it is given, the code that runs a call and returns
    the text the model sees, and how many rounds of calls it may make."""

    specs: tuple[dict[str, Any], ...]
    run: Callable[[str, dict[str, Any]], str]
    max_rounds: int


def golden_toolbox(golden_dir: str | os.PathLike[str] | None = None) -> Toolbox:
    """The agent's own memory tools over the golden fixture memory at the fixture's time: the same TOOLS, the same
    tool code (QA's tool loop) and the same "nothing found" the agent's tool loop hands its model."""
    from contexts.agent.application.qa import MAX_TOOL_CALLS, QA, TOOLS, Evidence, SleepSchedule
    from contexts.agent.eval import load_fixture
    from contracts.types import ToolCall
    from lab.fixture_dataset import AGENT_CONFIG, GOLDEN_DIR, NOTHING_FOUND
    from runtime import FakeRuntime

    memory, now, _ = load_fixture(Path(golden_dir) if golden_dir is not None else GOLDEN_DIR)
    qa = QA(FakeRuntime(), memory, clock=lambda: now, sleep=SleepSchedule.load(AGENT_CONFIG))

    def run(name: str, arguments: dict[str, Any]) -> str:
        found = Evidence()
        asyncio.run(qa._run_tool(ToolCall(id="", name=name, arguments=arguments), found))
        return "\n".join(found.lines) or NOTHING_FOUND

    return Toolbox(specs=tuple(TOOLS), run=run, max_rounds=MAX_TOOL_CALLS)


# --- what may be labeled --------------------------------------------------------------------

def is_trace_task(spec: TaskSpec) -> bool:
    """A task whose label is a tool-call trace: its output schema has a top-level tool_calls property."""
    properties = spec.output_schema.get("properties") if isinstance(spec.output_schema, Mapping) else None
    return isinstance(properties, Mapping) and "tool_calls" in properties


def running_on_pod(environ: Mapping[str, str] | None = None) -> bool:
    """Whether this process runs on a hub pod: JupyterHub sets these in every single-user server and its terminals."""
    environ = os.environ if environ is None else environ
    return any(environ.get(name) for name in POD_ENV)


def _is_fixture(manifest: DatasetManifest) -> bool:
    try:
        from lab.fixture_dataset import FIXTURE_ACTION_ID
    except ImportError:   # a machine without the agent's code holds no fixture
        return False
    return manifest.action_id == FIXTURE_ACTION_ID


def source_problem(root: Path, manifest: DatasetManifest, *, environ: Mapping[str, str] | None = None) -> str | None:
    """Why this folder may not be labeled, or None when it may: it must be named for its dataset id and be either
    an approved export under data/exports/ or the no-camera fixture; on a pod, only the fixture."""
    folder = root.resolve()
    if folder.name != manifest.dataset_id:
        return (f"the folder is named {folder.name!r} but its manifest is dataset {manifest.dataset_id!r}; an export "
                "is read only from data/exports/<dataset_id>/")
    fixture = _is_fixture(manifest)
    parent = folder.parent
    if not fixture and (parent.name, parent.parent.name) != EXPORTS_PARENT[::-1]:
        return ("not an approved export: an export is read only from a data/exports/<dataset_id>/ folder, and the "
                "only other dataset labeled is the no-camera fixture")
    if not fixture and running_on_pod(environ):
        return ("this runs on a pod, and an export labeled here would leave a copy on the pod after the run: label it "
                "where it lives, with --account to reach the teacher through the hub proxy")
    return None


def _validator(schema: Mapping[str, Any]) -> Any:
    """A JSON-schema validator for the task's output schema; LabelRefused when it cannot check labels."""
    try:
        import jsonschema
    except ImportError:
        raise LabelRefused("the task has an output schema and jsonschema is not installed, so no label could be "
                           "checked") from None
    validator_class = jsonschema.validators.validator_for(schema)
    try:
        validator_class.check_schema(schema)
    except jsonschema.SchemaError as err:
        where = "/".join(str(part) for part in err.absolute_path) or "the top"
        raise LabelRefused(f"the task's output_schema is not a valid JSON schema ({err.validator} at {where})") from None
    return validator_class(schema)


def _schema_problem(validator: Any, value: Any) -> str | None:
    """Where the value breaks the schema and which rule it breaks, never the value itself; None when it holds."""
    import jsonschema

    error = jsonschema.exceptions.best_match(validator.iter_errors(value))
    if error is None:
        return None
    where = "/".join(str(part) for part in error.absolute_path) or "the top"
    return f"{where} breaks {error.validator}"


def _parse_json(text: str) -> Any:
    """The JSON a label holds: the whole text, or the inside of one fenced block. ValueError when it is not JSON."""
    body = text.strip()
    fenced = FENCED.match(body)
    if fenced:
        body = fenced.group("body").strip()

    def refuse(constant: str) -> Any:
        raise ValueError(f"{constant} is not JSON")

    return json.loads(body, parse_constant=refuse)


# --- one example ----------------------------------------------------------------------------

@dataclass
class Outcome:
    """What asking the teacher about one example came to."""

    id: str
    status: str                       # accepted, rejected or failed
    reason: str | None = None         # a reject reason or a failure kind, one word
    detail: str | None = None         # where a label broke the schema, or what a failed request said
    target: str | None = None
    trace: list[dict[str, Any]] | None = None
    answer: str | None = None
    seconds: float = 0.0
    prompt_tokens: int | None = 0
    completion_tokens: int | None = 0
    requests: int = 0
    tool_calls: int = 0
    fatal: bool = False
    counts_as_failure: bool = False   # toward the run's limit of examples in a row with no usable answer

    def count(self, completion: Completion) -> None:
        self.requests += 1
        self.prompt_tokens = (None if self.prompt_tokens is None or completion.prompt_tokens is None
                              else self.prompt_tokens + completion.prompt_tokens)
        self.completion_tokens = (None if self.completion_tokens is None or completion.completion_tokens is None
                                  else self.completion_tokens + completion.completion_tokens)

    def reject(self, reason: str, detail: str | None = None) -> "Outcome":
        self.status, self.reason, self.detail = "rejected", reason, detail
        self.counts_as_failure = reason == "tool_markup"
        return self


def _text_problem(text: str) -> str | None:
    if "<think>" in text or "</think>" in text:
        return "thinking"
    if "<tool_call>" in text:
        return "tool_markup"
    if not text.strip():
        return "empty"
    return None


class _Labeler:
    """Asks the teacher about one example at a time and checks what comes back; shared by the run's workers."""

    def __init__(self, data: Dataset, teacher: Teacher, decode: Decode, toolbox: Toolbox | None,
                 check: Callable[[str], str | None] | None, clock: Callable[[], float]) -> None:
        self.data = data
        self.spec = data.manifest.spec
        self.teacher = teacher
        self.decode = decode
        self.toolbox = toolbox
        self.check = check
        self.clock = clock
        self.validator = _validator(self.spec.output_schema) if self.spec.output_schema else None

    def label(self, example: DatasetExample) -> Outcome:
        outcome = Outcome(example.id, "accepted")
        started = self.clock()
        try:
            if self.toolbox is not None:
                self._trace(example, outcome)
            else:
                self._answer(example, outcome)
        except TeacherError as err:
            outcome.status, outcome.reason, outcome.detail = "failed", "request", str(err)
            outcome.fatal, outcome.counts_as_failure = err.fatal, True
        outcome.seconds = self.clock() - started
        return outcome

    def _keep(self, outcome: Outcome, value: Any) -> None:
        if self.validator is not None:
            problem = _schema_problem(self.validator, value)
            if problem is not None:
                outcome.reject("schema", problem)
                return
        outcome.target = json.dumps(value, ensure_ascii=False)

    def _trace(self, example: DatasetExample, outcome: Outcome) -> None:
        assert self.toolbox is not None
        question = (example.context or "").strip()
        if not question:
            outcome.status, outcome.reason = "failed", "no_question"
            return
        messages: list[dict[str, Any]] = [{"role": "system", "content": self.spec.instruction},
                                          {"role": "user", "content": question}]
        made: list[dict[str, Any]] = []
        for round_number in range(self.toolbox.max_rounds + 1):
            completion = self.teacher.chat(messages, decode=self.decode, tools=self.toolbox.specs)
            outcome.count(completion)
            if completion.finish_reason == "length":
                outcome.reject("truncated")
                return
            if not completion.calls:
                break
            if round_number == self.toolbox.max_rounds:
                outcome.reject("tool_rounds")
                return
            messages.append({"role": "assistant", "content": completion.content or "",
                             "tool_calls": [{"id": call.id, "type": "function",
                                             "function": {"name": call.name, "arguments": call.arguments}}
                                            for call in completion.calls]})
            for call in completion.calls:
                try:
                    arguments = json.loads(call.arguments) if call.arguments.strip() else {}
                except ValueError:
                    arguments = None
                if not isinstance(arguments, dict):
                    outcome.reject("bad_arguments")
                    return
                result = self.toolbox.run(call.name, arguments)
                made.append({"name": call.name, "arguments": arguments, "result": result})
                outcome.tool_calls += 1
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
        answer = completion.content.strip()
        problem = _text_problem(answer)
        if problem is not None:
            outcome.reject(problem)
            return
        outcome.trace, outcome.answer = made, answer
        self._keep(outcome, {"tool_calls": [{"name": call["name"], "arguments": call["arguments"]} for call in made],
                             "answer": answer})

    def _prompt(self, example: DatasetExample) -> str:
        parts = [self.spec.instruction.strip()]
        if example.context and example.context.strip():
            parts.append(f"What the box noted: {example.context.strip()}")
        if self.spec.output_schema:
            parts.append("Reply with only JSON that follows this JSON schema, and nothing else:\n"
                         + json.dumps(self.spec.output_schema, ensure_ascii=False))
        return "\n\n".join(parts)

    def _answer(self, example: DatasetExample, outcome: Outcome) -> None:
        prompt = self._prompt(example)
        image = self.data.image_path(example)
        content: Any = prompt
        if image is not None:
            try:
                encoded = base64.b64encode(image.read_bytes()).decode("ascii")
            except OSError:
                outcome.status, outcome.reason = "failed", "image"
                return
            kind = mimetypes.guess_type(image.name)[0] or "image/jpeg"
            content = [{"type": "image_url", "image_url": {"url": f"data:{kind};base64,{encoded}"}},
                       {"type": "text", "text": prompt}]
        completion = self.teacher.chat([{"role": "user", "content": content}], decode=self.decode)
        outcome.count(completion)
        if completion.finish_reason == "length":
            outcome.reject("truncated")
            return
        text = completion.content.strip()
        problem = _text_problem(text)
        if problem is not None:
            outcome.reject(problem)
            return
        if self.spec.output_schema:
            try:
                value = _parse_json(text)
            except ValueError:
                outcome.reject("not_json")
                return
            self._keep(outcome, value)
            return
        if self.check is not None:
            problem = self.check(text)
            if problem:
                outcome.reject("check", _one_line(problem))
                return
        outcome.target = text


# --- the run --------------------------------------------------------------------------------

@dataclass
class Run:
    """What a labeling run did. outcome: ok (every pending example was asked), stopped (the STOP file appeared),
    fail (the teacher could not be used)."""

    run_id: str
    dataset_id: str
    pending_before: int
    outcome: str = "ok"
    error: str | None = None
    checkpoint: str | None = None
    asked: int = 0
    accepted: int = 0
    rejected: Counter = field(default_factory=Counter)
    failed: Counter = field(default_factory=Counter)
    seconds: float = 0.0
    wall_seconds: float = 0.0
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    with_requests: int = 0
    with_usage: int = 0
    pending_after: int = 0

    def metrics(self) -> dict[str, int | float]:
        """The run's numbers for its ledger row: counts, rejects and failures by reason, totals and means."""
        out: dict[str, int | float] = {
            "labels.pending_before": self.pending_before, "labels.asked": self.asked,
            "labels.accepted": self.accepted, "labels.rejected": sum(self.rejected.values()),
            "labels.failed": sum(self.failed.values()), "labels.pending_after": self.pending_after,
            "labels.requests": self.requests, "labels.seconds_total": round(self.seconds, 3),
            "labels.wall_seconds": round(self.wall_seconds, 3),
            "labels.usage_missing": self.with_requests - self.with_usage}
        out.update({f"labels.rejected.{reason}": count for reason, count in sorted(self.rejected.items())})
        out.update({f"labels.failed.{kind}": count for kind, count in sorted(self.failed.items())})
        if self.asked:
            out["labels.seconds_per_example"] = round(self.seconds / self.asked, 3)
        if self.with_usage:
            tokens = self.prompt_tokens + self.completion_tokens
            out.update({"labels.tokens_total": tokens,
                        "labels.tokens_per_example": round(tokens / self.with_usage, 1),
                        "labels.prompt_tokens_per_example": round(self.prompt_tokens / self.with_usage, 1),
                        "labels.completion_tokens_per_example": round(self.completion_tokens / self.with_usage, 1)})
        return out


def _new_run_id() -> str:
    return f"label-{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}-{secrets.token_hex(2)}"


def _append_trace(root: Path, row: Mapping[str, Any]) -> None:
    """One row at the end of traces.jsonl, under its own lock, fsynced; a line an earlier crash left unfinished is
    closed first so the new row stays a line of its own."""
    with FileLock(root / TRACES_LOCK, timeout=30.0):
        with open(root / TRACES, "ab+") as out:
            if out.seek(0, os.SEEK_END) > 0:
                out.seek(-1, os.SEEK_END)
                if out.read(1) != b"\n":
                    out.write(b"\n")
            out.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
            out.flush()
            os.fsync(out.fileno())


class _Settler:
    """Stores what the workers bring back, one example at a time: the trace, the target, the ledger row, the counts."""

    def __init__(self, run: Run, root: Path, teacher: Teacher, decode: Decode, state_dir: Any) -> None:
        self.run = run
        self.root = root
        self.node = teacher.node
        self.state_dir = state_dir
        self.conditions = {"model": teacher.model, "checkpoint": run.checkpoint, "host": teacher.host,
                           "stream": teacher.stream, "thinking": False, "temperature": decode.temperature,
                           "top_p": decode.top_p, "max_tokens": decode.max_tokens, "seed": decode.seed}
        self.failures_in_a_row = 0

    def settle(self, outcome: Outcome) -> None:
        if outcome.status == "accepted":
            try:
                if outcome.trace is not None:
                    _append_trace(self.root, {"id": outcome.id, "run_id": self.run.run_id,
                                              "tool_calls": outcome.trace, "answer": outcome.answer})
                write_target(self.root, outcome.id, outcome.target or "")
            except (DatasetError, LockTimeout, OSError) as err:
                outcome.status, outcome.reason = "failed", "write"
                outcome.detail = _one_line(str(err))
        self._count(outcome)
        self._row(outcome)

    def _count(self, outcome: Outcome) -> None:
        run = self.run
        run.asked += 1
        run.requests += outcome.requests
        run.seconds += outcome.seconds
        run.with_requests += 1 if outcome.requests else 0
        if outcome.prompt_tokens is not None and outcome.completion_tokens is not None and outcome.requests:
            run.with_usage += 1
            run.prompt_tokens += outcome.prompt_tokens
            run.completion_tokens += outcome.completion_tokens
        if outcome.status == "accepted":
            run.accepted += 1
        elif outcome.status == "rejected":
            run.rejected[outcome.reason or "unknown"] += 1
        else:
            run.failed[outcome.reason or "unknown"] += 1
        self.failures_in_a_row = self.failures_in_a_row + 1 if outcome.counts_as_failure else 0

    def _row(self, outcome: Outcome) -> None:
        metrics: dict[str, int | float] = {"label.seconds": round(outcome.seconds, 3),
                                           "label.requests": outcome.requests, "label.tool_calls": outcome.tool_calls}
        if outcome.requests and outcome.prompt_tokens is not None and outcome.completion_tokens is not None:
            metrics.update({"label.prompt_tokens": outcome.prompt_tokens,
                            "label.completion_tokens": outcome.completion_tokens,
                            "label.tokens": outcome.prompt_tokens + outcome.completion_tokens})
        note = None
        if outcome.reason:
            note = outcome.reason + (f": {outcome.detail}" if outcome.detail else "")
        record("label-example", outcome.status, metrics,
               {"run_id": self.run.run_id, "dataset_id": self.run.dataset_id, "example_id": outcome.id},
               node=self.node, units={name: UNITS[name] for name in metrics}, conditions=self.conditions, note=note,
               state_dir=self.state_dir)


def label_dataset(root: str | os.PathLike[str], teacher: Teacher, *, decode: Decode = Decode(),
                  toolbox: Toolbox | None = None, golden_dir: str | os.PathLike[str] | None = None,
                  check: Callable[[str], str | None] | None = None, text_only: bool | None = None,
                  workers: int = 1, limit: int | None = None, ids: Iterable[str] | None = None,
                  expect_model: str | None = EXPECTED_MODEL, max_failures: int = MAX_FAILURES,
                  run_id: str | None = None, environ: Mapping[str, str] | None = None,
                  state_dir: str | os.PathLike[str] | None = None,
                  clock: Callable[[], float] = time.perf_counter) -> Run:
    """Ask the teacher for every pending example of the dataset (or the ones ids names, at most limit of them) and
    store each label that passes its checks (see the module docstring).

    A tool-trace task runs its calls through toolbox, by default the agent's tools over the golden fixture memory in
    golden_dir. text_only None means a tool-trace task carries no pictures and any other task needs them all. check
    is a free-text task's own test of a label: the reason it fails, or None. DatasetError when the folder does not
    load; LabelRefused when it may not be labeled or its labels could not be checked; a teacher that cannot be used
    ends the run with outcome fail."""
    if workers < 1:
        raise LabelRefused("workers must be at least 1")
    root = Path(root)
    data = load(root, require_images=False)
    spec = data.manifest.spec
    trace = is_trace_task(spec)
    if not (trace if text_only is None else text_only):
        data = load(root, require_images=True)
    problem = source_problem(root, data.manifest, environ=environ)
    if problem is not None:
        raise LabelRefused(problem)
    if trace and toolbox is None:
        toolbox = golden_toolbox(golden_dir)
    labeler = _Labeler(data, teacher, decode, toolbox if trace else None, check, clock)

    todo = data.pending()
    if ids is not None:
        wanted = set(ids)
        todo = [example for example in todo if example.id in wanted]
    if limit is not None:
        todo = todo[:max(0, limit)]
    run = Run(run_id or _new_run_id(), data.dataset_id, pending_before=len(data.pending()))
    if not todo:   # nothing to ask: the pod is not called at all
        run.pending_after = run.pending_before
        return run
    started = clock()
    try:
        run.checkpoint = teacher.served_checkpoint(expect_model)
    except TeacherError as err:
        run.outcome, run.error = "fail", str(err)
        run.pending_after = run.pending_before
        return run
    settler = _Settler(run, root, teacher, decode, state_dir)

    queue = iter(todo)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="label") as pool:
        running: dict[concurrent.futures.Future[Outcome], str] = {}

        def fill() -> None:
            while len(running) < workers:
                if stop.stopped():
                    run.outcome, run.error = "stopped", stop.stop_reason() or stop.NO_REASON
                    return
                example = next(queue, None)
                if example is None:
                    return
                running[pool.submit(labeler.label, example)] = example.id

        fill()
        while running:
            finished, _ = concurrent.futures.wait(running, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in finished:
                running.pop(future)
                outcome = future.result()
                settler.settle(outcome)
                if run.outcome == "ok" and outcome.fatal:
                    run.outcome, run.error = "fail", outcome.detail
                elif run.outcome == "ok" and settler.failures_in_a_row >= max_failures:
                    run.outcome = "fail"
                    run.error = (f"{settler.failures_in_a_row} examples in a row got no usable answer; the last: "
                                 f"{outcome.reason}" + (f" ({outcome.detail})" if outcome.detail else ""))
            if run.outcome == "ok":
                fill()
    run.wall_seconds = clock() - started
    run.pending_after = len(load(root, require_images=False).pending())
    return run


# --- the entry point ------------------------------------------------------------------------

def _teacher_from_args(args: argparse.Namespace, transport: httpx.BaseTransport | None) -> Teacher:
    if args.url:
        return Teacher(args.url, transport=transport, stream=not args.no_stream)
    accounts = hub.load_accounts([args.account], require_role=True)
    return Teacher.from_hub(accounts[args.account], port=args.port, transport=transport, stream=not args.no_stream)


def _answer(run: Run) -> dict[str, Any]:
    return {"ok": run.outcome == "ok", "outcome": run.outcome, "dataset_id": run.dataset_id, "run_id": run.run_id,
            "checkpoint": run.checkpoint, "asked": run.asked, "accepted": run.accepted,
            "rejected": dict(sorted(run.rejected.items())), "failed": dict(sorted(run.failed.items())),
            "pending": run.pending_after, "metrics": run.metrics(),
            "refs": {"run_id": run.run_id, "dataset_id": run.dataset_id},
            **({"error": run.error} if run.error else {})}


@recorded("label")
@stop.guarded
def main(argv: Sequence[str] | None = None, *, transport: httpx.BaseTransport | None = None) -> dict[str, Any]:
    """Label the pending examples of a dataset with the teacher, checking every label before it is kept."""
    parser = argparse.ArgumentParser(prog="python lab/label.py", description=main.__doc__, allow_abbrev=False)
    parser.add_argument("dataset", help="an approved export (data/exports/<dataset_id>/) or the no-camera fixture")
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--account", type=str.upper, help="the hub account whose pod serves the teacher (C, D, ...)")
    where.add_argument("--url", help="the teacher server's address, without /v1 (a server reached directly)")
    parser.add_argument("--port", type=int, default=TEACHER_PORT, help="the teacher's port on the pod (default 8000)")
    parser.add_argument("--workers", type=int, default=1, help="examples asked at the same time (default 1)")
    parser.add_argument("--limit", type=int, help="ask about at most this many pending examples")
    parser.add_argument("--id", action="append", dest="ids", metavar="ID", help="ask only about this example (repeat)")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-stream", action="store_true", help="one buffered JSON answer per request")
    parser.add_argument("--golden", help="the fixture memory a tool-trace task's calls run against "
                                         "(default: fixtures/golden)")
    parser.add_argument("--text-only", action="store_true", help="the examples carry no picture")
    parser.add_argument("--expect-model", default=EXPECTED_MODEL,
                        help=f"what the served checkpoint's name must hold (default {EXPECTED_MODEL}; '' accepts any)")
    parser.add_argument("--max-failures", type=int, default=MAX_FAILURES,
                        help="examples in a row with no usable answer before the run stops")
    args = parser.parse_args(argv)
    decode = Decode(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, seed=args.seed)
    try:
        teacher = _teacher_from_args(args, transport)
    except hub.MissingCredentials as err:
        return {"ok": False, "outcome": "refused", "error": f"the account's keys are missing: {err}"}
    except hub.PodRoleError as err:
        return {"ok": False, "outcome": "refused", "error": str(err)}
    with teacher:
        try:
            run = label_dataset(args.dataset, teacher, decode=decode, golden_dir=args.golden,
                                text_only=True if args.text_only else None, workers=args.workers, limit=args.limit,
                                ids=args.ids, expect_model=args.expect_model or None, max_failures=args.max_failures)
        except DatasetError as err:
            return {"ok": False, "outcome": "invalid", "problems": err.problems, "error": str(err)}
        except LabelRefused as err:
            return {"ok": False, "outcome": "refused", "error": str(err)}
    return _answer(run)


if __name__ == "__main__":
    answer = main()
    print(json.dumps(answer))
    sys.exit(0 if answer.get("ok", True) else 1)
