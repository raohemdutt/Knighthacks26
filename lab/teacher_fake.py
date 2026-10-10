"""A stand-in for the teacher that tests and rehearsals can drive: vLLM's OpenAI-compatible server and, when it is given
an account, the hub proxy in front of it. No GPU, no pod, no network, no real waiting.

    fake = FakeTeacher(reference_policy(load_expected(folder)))           # the server alone, as on the pod's port
    teacher = Teacher(fake.base_url, transport=fake.transport)

    fake = FakeTeacher(policy, user="acct-c", token="...")                # behind the hub proxy
    teacher = Teacher.from_hub(fake.account("C"), transport=fake.transport)

    with fake.serve() as url:                                             # a real HTTP server on 127.0.0.1
        ...

    python -m lab.teacher_fake --port 8000      # serves the fixture's reference answers until stopped

What it answers, as vLLM does at the commit the pod builds (b1388b1): GET /v1/models lists the served model with its
checkpoint as root, and each of its aliases after it (vLLM's --served-model-name takes several names, answers to any
of them and names the first in every reply); POST /v1/chat/completions answers one choice, as one JSON body or, for
"stream": true, as server-sent events ("data: {chunk}" lines, the text and each tool call's arguments in pieces, a
last chunk with the finish reason, a usage chunk when stream_options.include_usage is set, then "data: [DONE]"); an
unknown model is a 404 whose body is {"error": {"message", "type", "param", "code"}}. A request whose
chat_template_kwargs does not set enable_thinking to false gets a thinking block in front of its answer. A server
started without a tool-call parser (tool_parser=False, the flags lab/pod_a_teacher.sh leaves out today) puts the
model's call markup in the text and returns no tool_calls, as vLLM does when "auto" tool choice is asked of it without
--enable-auto-tool-choice and --tool-call-parser. Tokens are counted as one per four characters, and an answer longer
than max_tokens is cut there with finish reason "length".

What the policy decides: what the model says, given the request body. reference_policy() answers like the fixture's
reference (expected.jsonl): the reference's tool calls first, its answer once tool results are in. A policy may also
return tool arguments as raw text, to stand for arguments the model wrote badly.

Behind the proxy (an account given), as the hub does: a User-Agent that is not a browser's gets the front end's 403
page, a wrong or missing token a 302 to the login page, a stopped server (stop_server()) a 424, and a path outside
/user/<account>/proxy/<port>/ a 404. The proxy streams an answer only when the request's Accept header is exactly
text/event-stream and buffers it otherwise, and a buffered answer that takes longer than 125 seconds is cut with
Cloudflare's 524.

Time: each answer takes the time-to-first-token plus its completion tokens over the decode rate, which decides the 524;
with a FakeClock (lab/hub_fake.py) that time also passes on the clock, so a caller timing its requests with that clock
sees it. The default rates, 0.55 s and 7.7 tok/s, are the morning's R9700 numbers in lab/README.md, another chip than
the Strix Halo pods the teacher now runs on, whose rates are not measured yet: they set the fake's pace, nothing more.

Faults for a test to queue: fail() answers the next requests with a status, drop() drops the next connections, cut()
ends the next streamed answers before they finish, and break_stream() ends them with the error chunk vLLM sends when
the engine fails mid-answer ({"error": {...}}, then [DONE]). Every request is kept in requests, in order.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

if not __package__:  # run by path: the repo root, one folder up, is what holds lab/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from lab.hub import Account
from lab.hub_fake import FAKE_ROLES

__all__ = ["Turn", "Policy", "Seen", "FakeTeacher", "reference_policy", "text_policy", "main"]

ROOT = "RedHatAI/Qwen3.8-27B-INT4"     # the checkpoint the teacher serves (lab/pod_a_teacher.sh)
TTFT_S = 0.55                          # time to first token, text, thinking off: the R9700 number in lab/README.md
DECODE_TOK_S = 7.7                     # decode rate: the R9700 number in lab/README.md, not the Strix Halo pods'
BUFFERED_LIMIT_S = 125.0               # Cloudflare cuts a buffered answer after this long and answers 524
MAX_MODEL_LEN = 16384
NO_EVIDENCE = "I didn't see that."     # what the agent says when nothing answers the question
CLOUDFLARE_403 = "<html><head><title>Access denied</title></head><body>Error 1010: browser signature banned</body></html>"
CLOUDFLARE_524 = ("<html><head><title>524: A timeout occurred</title></head><body>The origin web server timed out "
                  "responding to this request.</body></html>")
THINKING = "<think>\nThe user is asking a question. Let me work out what to call and what to say.\n</think>\n\n"
PIECE_CHARS = 12                        # how much text one streamed chunk carries


@dataclass(frozen=True)
class Turn:
    """What the model says: text, tool calls, or both. A call is (name, arguments): arguments a dict, or raw text
    that stands for what a model wrote when it got the JSON wrong."""

    content: str = ""
    tool_calls: tuple[tuple[str, Any], ...] = ()


Policy = Callable[[Mapping[str, Any]], Turn]


@dataclass
class Seen:
    """One request the fake received: method, path, headers (lowercased names) and JSON body."""

    method: str
    path: str
    headers: dict[str, str]
    body: Any


@dataclass
class _Reply:
    status: int
    headers: dict[str, str]
    chunks: list[bytes]
    streamed: bool = False


@dataclass
class _Fault:
    kind: str                     # status, drop, cut or error
    remaining: int
    status: int = 0
    body: Any = None
    text: str | None = None


def text_policy(text: str) -> Policy:
    """A teacher that always says the same thing."""
    return lambda body: Turn(content=text)


def _question(body: Mapping[str, Any]) -> str:
    for message in body.get("messages") or []:
        if isinstance(message, Mapping) and message.get("role") == "user" and isinstance(message.get("content"), str):
            return message["content"]
    return ""


def reference_policy(expected: Mapping[str, Any]) -> Policy:
    """A teacher that answers like the reference: expected is lab.fixture_dataset.load_expected() of a fixture folder.
    For a question it knows it first makes the reference's tool calls, then, once tool results are in, gives the
    reference's answer; a question it does not know gets "I didn't see that." and no call."""
    by_question = {row.question: row.label for row in expected.values()}

    def answer(body: Mapping[str, Any]) -> Turn:
        label = by_question.get(_question(body))
        if label is None:
            return Turn(content=NO_EVIDENCE)
        answered = any(isinstance(m, Mapping) and m.get("role") == "tool" for m in body.get("messages") or [])
        if label.tool_calls and not answered:
            return Turn(tool_calls=tuple((call.name, call.arguments) for call in label.tool_calls))
        return Turn(content=label.answer)

    return answer


def _tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


def _arguments_text(arguments: Any) -> str:
    return arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)


def _markup(name: str, arguments: Any) -> str:
    """A call the way the model writes it when nothing parses it (the Qwen3.5 template's form)."""
    if not isinstance(arguments, Mapping):
        return f"<tool_call>\n<function={name}>\n{arguments}\n</function>\n</tool_call>"
    parameters = "".join(f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in arguments.items())
    return f"<tool_call>\n<function={name}>\n{parameters}</function>\n</tool_call>"


def _pieces(text: str) -> list[str]:
    return [text[start:start + PIECE_CHARS] for start in range(0, len(text), PIECE_CHARS)]


def _json_reply(status: int, body: Any) -> _Reply:
    return _Reply(status, {"content-type": "application/json"}, [json.dumps(body).encode("utf-8")])


def _html_reply(status: int, page: str) -> _Reply:
    return _Reply(status, {"content-type": "text/html"}, [page.encode("utf-8")])


def _vllm_error(status: int, kind: str, message: str, param: str | None = None) -> _Reply:
    return _json_reply(status, {"error": {"message": message, "type": kind, "param": param, "code": status}})


class FakeTeacher:
    """The teacher server, and the hub proxy in front of it when user and token are given (see the module docstring).

    policy decides what the model says; model is the served name, aliases the further names it answers to, and root
    its checkpoint; tool_parser False stands for a server started without a tool-call parser; clock, a FakeClock, is
    moved by the time each answer takes."""

    def __init__(self, policy: Policy, *, model: str = "teacher", aliases: Sequence[str] = (), root: str | None = ROOT,
                 user: str | None = None, token: str | None = None, host: str = "teacher.test", port: int = 8000,
                 tool_parser: bool = True, ttft_s: float = TTFT_S, decode_tok_s: float = DECODE_TOK_S,
                 buffered_limit_s: float = BUFFERED_LIMIT_S, clock: Any = None) -> None:
        if (user is None) != (token is None):
            raise ValueError("give both user and token for the proxy, or neither for the server alone")
        self.policy = policy
        self.model = model
        self.aliases = tuple(aliases)
        self.root = root
        self.user = user
        self.token = token
        self.host = host
        self.port = port
        self.tool_parser = tool_parser
        self.ttft_s = ttft_s
        self.decode_tok_s = decode_tok_s
        self.buffered_limit_s = buffered_limit_s
        self.clock = clock
        self.running = True
        self.requests: list[Seen] = []
        self._faults: list[_Fault] = []
        self._lock = threading.Lock()
        self._calls = 0

    # --- addresses ---------------------------------------------------------------------------

    @property
    def proxied(self) -> bool:
        return self.token is not None

    @property
    def prefix(self) -> str:
        """The path in front of the server's own paths: /user/<account>/proxy/<port> behind the proxy, else none."""
        return f"/user/{self.user}/proxy/{self.port}" if self.proxied else ""

    @property
    def base_url(self) -> str:
        """The address a Teacher on transport uses: the proxy's path on the fake hub, or the server itself."""
        return f"https://{self.host}{self.prefix}" if self.proxied else f"http://{self.host}"

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def account(self, letter: str = "C", url: str | None = None) -> Account:
        """The hub account whose pod this fake stands behind, with the fake hub's address (or url)."""
        if not self.proxied:
            raise ValueError("this fake has no account: give it user and token")
        return Account(letter, self.user or "", self.token or "", url or f"https://{self.host}",
                       role=FAKE_ROLES.get(letter))

    # --- faults and state ----------------------------------------------------------------------

    def fail(self, status: int, *, times: int = 1, message: str | None = None, page: str | None = None) -> None:
        """The next chat requests get this status: a vLLM error body with message, else the page as HTML."""
        body = None if page is not None else {"error": {"message": message or "fake failure", "type": "Error",
                                                         "param": None, "code": status}}
        self._faults.append(_Fault("status", times, status, body, page))

    def drop(self, times: int = 1) -> None:
        """The next chat requests get no answer at all: the connection drops."""
        self._faults.append(_Fault("drop", times))

    def cut(self, times: int = 1) -> None:
        """The next streamed answers stop before their last chunk and [DONE]."""
        self._faults.append(_Fault("cut", times))

    def break_stream(self, *, times: int = 1, message: str = "EngineCore encountered an issue") -> None:
        """The next streamed answers stop with vLLM's mid-stream error chunk instead of their last chunk."""
        self._faults.append(_Fault("error", times, 500, {"error": {"message": message, "type": "InternalServerError",
                                                                     "param": None, "code": 500}}))

    def stop_server(self) -> None:
        self.running = False

    def start_server(self) -> None:
        self.running = True

    def _take_fault(self, kinds: Sequence[str]) -> _Fault | None:
        with self._lock:
            for fault in self._faults:
                if fault.kind in kinds and fault.remaining > 0:
                    fault.remaining -= 1
                    return fault
        return None

    # --- answering -----------------------------------------------------------------------------

    def _handle(self, request: httpx.Request) -> httpx.Response:
        reply = self._respond(request.method, request.url.raw_path.decode("ascii").split("?")[0],
                              {key.lower(): value for key, value in request.headers.items()}, request.content)
        if reply is None:
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.", request=request)
        if reply.streamed:
            return httpx.Response(reply.status, headers=reply.headers, content=iter(reply.chunks))
        return httpx.Response(reply.status, headers=reply.headers, content=b"".join(reply.chunks))

    def _respond(self, method: str, path: str, headers: Mapping[str, str], content: bytes) -> _Reply | None:
        try:
            body = json.loads(content) if content else None
        except ValueError:
            body = None
        with self._lock:
            self.requests.append(Seen(method, path, dict(headers), body))
        if self.proxied:
            if not headers.get("user-agent", "").startswith("Mozilla/5.0"):
                return _html_reply(403, CLOUDFLARE_403)
            if headers.get("authorization") != f"token {self.token}":
                return _Reply(302, {"location": f"/hub/api/oauth2/authorize?client_id=jupyterhub-user-{self.user}"},
                              [b""])
            if not self.running:
                return _json_reply(424, {"status": 424, "message": f"Server {self.user} is not running"})
            if not path.startswith(self.prefix + "/"):
                return _json_reply(404, {"message": "Not Found"})
            path = path[len(self.prefix):]
        if method == "GET" and path == "/v1/models":
            return _json_reply(200, {"object": "list", "data": [{
                "id": name, "object": "model", "created": 1760000000, "owned_by": "vllm", "root": self.root,
                "parent": None, "max_model_len": MAX_MODEL_LEN} for name in (self.model, *self.aliases)]})
        if method == "GET" and path == "/health":
            return _Reply(200, {}, [b""])
        if method == "POST" and path == "/v1/chat/completions":
            fault = self._take_fault(("status", "drop"))
            if fault is not None and fault.kind == "drop":
                return None
            if fault is not None:
                if fault.body is not None:
                    return _json_reply(fault.status, fault.body)
                return _html_reply(fault.status, fault.text or "")
            return self._chat(body if isinstance(body, Mapping) else {}, headers)
        return _json_reply(404, {"detail": "Not Found"})

    def _chat(self, body: Mapping[str, Any], headers: Mapping[str, str]) -> _Reply:
        if body.get("model") not in (self.model, *self.aliases):
            return _vllm_error(404, "NotFoundError", f"The model `{body.get('model')}` does not exist.", "model")
        turn = self.policy(body)
        content = turn.content
        calls = [(name, _arguments_text(arguments), arguments) for name, arguments in turn.tool_calls]
        if (body.get("chat_template_kwargs") or {}).get("enable_thinking") is not False:
            content = THINKING + content
        tools_on = bool(body.get("tools")) and body.get("tool_choice", "auto") != "none"
        if calls and not (tools_on and self.tool_parser):
            content += "".join(_markup(name, arguments) for name, _, arguments in calls)
            calls = []
        finish = "tool_calls" if calls else "stop"
        completion_tokens = max(1, _tokens(content) + sum(_tokens(name + text) + 4 for name, text, _ in calls))
        limit = body.get("max_tokens") if isinstance(body.get("max_tokens"), int) else MAX_MODEL_LEN
        if completion_tokens > limit:
            content = content[:limit * 4]
            calls = [(name, text[:max(0, len(text) // 2)], None) for name, text, _ in calls]
            completion_tokens, finish = limit, "length"
        prompt_tokens = _tokens(json.dumps(body.get("messages") or [])) + _tokens(json.dumps(body.get("tools") or []))
        seconds = self.ttft_s + completion_tokens / self.decode_tok_s
        if self.clock is not None:
            self.clock.sleep(seconds)
        if self.proxied and headers.get("accept") != "text/event-stream" and seconds > self.buffered_limit_s:
            return _html_reply(524, CLOUDFLARE_524)
        with self._lock:
            self._calls += 1
            number = self._calls
        ids = [f"chatcmpl-tool-{number:04d}{index}" for index in range(len(calls))]
        usage = {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens + completion_tokens,
                 "completion_tokens": completion_tokens}
        if body.get("stream") is True:
            return self._stream(number, content, calls, ids, finish, usage,
                                bool((body.get("stream_options") or {}).get("include_usage")))
        message: dict[str, Any] = {"role": "assistant", "content": content or None, "reasoning_content": None,
                                   "tool_calls": [{"id": ids[index], "type": "function",
                                                   "function": {"name": name, "arguments": text}}
                                                  for index, (name, text, _) in enumerate(calls)]}
        return _json_reply(200, {"id": f"chatcmpl-{number:04d}", "object": "chat.completion", "created": 1760000000,
                                 "model": self.model, "choices": [{"index": 0, "message": message, "logprobs": None,
                                                                   "finish_reason": finish, "stop_reason": None}],
                                 "usage": usage})

    def _stream(self, number: int, content: str, calls: list[tuple[str, str, Any]], ids: list[str], finish: str,
                usage: Mapping[str, int], include_usage: bool) -> _Reply:
        base = {"id": f"chatcmpl-{number:04d}", "object": "chat.completion.chunk", "created": 1760000000,
                "model": self.model}

        def chunk(delta: Mapping[str, Any], reason: str | None = None) -> bytes:
            data = {**base, "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": reason}]}
            return f"data: {json.dumps(data)}\n\n".encode("utf-8")

        chunks = [chunk({"role": "assistant", "content": ""})]
        chunks += [chunk({"content": piece}) for piece in _pieces(content)]
        for index, (name, text, _) in enumerate(calls):
            chunks.append(chunk({"tool_calls": [{"id": ids[index], "type": "function", "index": index,
                                                 "function": {"name": name, "arguments": ""}}]}))
            half = len(text) // 2
            chunks += [chunk({"tool_calls": [{"index": index, "function": {"arguments": part}}]})
                       for part in (text[:half], text[half:]) if part]
        broken = self._take_fault(("cut", "error"))
        if broken is not None and broken.kind == "error":
            chunks.append(f"data: {json.dumps(broken.body)}\n\n".encode("utf-8"))
            chunks.append(b"data: [DONE]\n\n")
        elif broken is None:
            chunks.append(chunk({"content": ""}, finish))
            if include_usage:
                chunks.append(f"data: {json.dumps({**base, 'choices': [], 'usage': dict(usage)})}\n\n".encode("utf-8"))
            chunks.append(b"data: [DONE]\n\n")
        return _Reply(200, {"content-type": "text/event-stream"}, chunks, streamed=True)

    # --- a real server -------------------------------------------------------------------------

    @contextlib.contextmanager
    def serve(self, port: int = 0) -> Iterator[str]:
        """Answer on 127.0.0.1 over real HTTP while the block runs; yields the base address (with the proxy's path
        in front when the fake stands behind the proxy). port 0 takes any free port."""
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def _handle(self) -> None:
                length = int(self.headers.get("content-length") or 0)
                content = self.rfile.read(length) if length else b""
                reply = fake._respond(self.command, self.path.split("?")[0],
                                      {key.lower(): value for key, value in self.headers.items()}, content)
                if reply is None:
                    self.close_connection = True
                    return
                self.send_response(reply.status)
                for key, value in reply.headers.items():
                    self.send_header(key, value)
                if not reply.streamed:
                    self.send_header("content-length", str(sum(len(part) for part in reply.chunks)))
                self.end_headers()
                for part in reply.chunks:
                    self.wfile.write(part)
                    self.wfile.flush()

            do_GET = do_POST = _handle

            def log_message(self, *args: Any) -> None:
                pass

        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, name="teacher-fake", daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}{self.prefix}"
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


def main(argv: Sequence[str] | None = None) -> int:
    """Serve the fixture's reference answers as the teacher on 127.0.0.1, for a rehearsal of lab/label.py."""
    from lab.fixture_dataset import DATASET_ID, OUT_DIR, load_expected

    parser = argparse.ArgumentParser(prog="python -m lab.teacher_fake", description=main.__doc__, allow_abbrev=False)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--fixture", default=str(OUT_DIR / DATASET_ID),
                        help="the fixture folder whose expected.jsonl the answers come from")
    parser.add_argument("--no-tool-parser", action="store_true",
                        help="answer as a server started without a tool-call parser")
    args = parser.parse_args(argv)
    fake = FakeTeacher(reference_policy(load_expected(args.fixture)), tool_parser=not args.no_tool_parser)
    with fake.serve(args.port) as url:
        print(json.dumps({"ok": True, "url": url, "model": fake.model, "root": fake.root}), flush=True)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
