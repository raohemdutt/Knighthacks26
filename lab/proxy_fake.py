"""A stand-in for everything between the box and a pod's ports: no network, no real pod, no real time.

    clock = FakeClock()
    fake = FakePodProxy(clock)
    account = fake.add_account("C", user="roscon01", token="...")
    teacher = fake.serve_model("C", port=8000, models=("teacher",), tokens_per_s=7.7, ttft_s=0.55, reply="...")
    link = PodLink(Hub(account, base_url=fake.url, transport=fake.transport), transport=fake.transport,
                   sleep=clock.sleep, clock=clock.time)

The fake is an httpx transport, so the real client code runs against it unchanged: the same URLs, the same headers, the
same JSON and event streams. Time moves only when something calls clock.sleep(); the fake model server spends its
generation time that way, so an answer that takes 150 seconds takes none. It shares lab.hub_fake.FakeClock, so a scenario
can run it beside a FakeHub on one clock. It does not launch, bill or cull pods: that is FakeHub's.

What it models, in the order a request meets it, and where each shape comes from (lab/refine/research/plumbing/
proxy-path.md and transfers.md, and the measurements of 2026-10-10 they cite). Not recorded from a live call, and so the
least certain, are marked (research).

  Cloudflare. Every answer could carry `server: cloudflare`. A client whose User-Agent is not a browser's gets 403 with
  the 1010 page (as FakeHub does). A request body over `edge_body_cap` (100 MB) gets 413 (research). With `hub_down` every
  request gets the front door's 502, which is what dub.aupcloud.io answered all day on 2026-10-10. A request whose origin
  sends nothing for `cloudflare_read_timeout_s` (125 s) gets 524.

  The hub. A token that is missing or wrong gets 302 to the login page on a GET or HEAD and 403 on any other method
  (measured for GET; the 403 is research). A valid token on another account's address gets 302 to /hub/user/<name>/....
  A pod that is not running gets 424 with the hub's JSON message (research); one that has just stopped (`stop_pod` with
  route_lingers_s) gets 503 until its route goes; one that is starting gets 303 to the spawn-pending page on a GET.

  The pod's Jupyter server with jupyter-server-proxy. /proxy/<port>/<path> reaches what listens on the port; a port with
  nothing on it gets 500 with a Jupyter HTML page (measured); /proxy/absolute/<port>/... reaches the model server with
  the prefix still on and so gets its 404 (measured). The proxy streams a reply only when the request's Accept header is
  exactly `text/event-stream`; otherwise it holds the whole reply until the model server has finished (research). A held
  reply over `proxy_response_cap` (100 MiB) comes back as 599 (research).
  /api/contents/<path> is the contents API: PUT of a file or a directory, chunked upload (`chunk` 1 creates, 2 and up
  append, -1 appends the last), GET with content=0 for the size, DELETE; a file whose directory does not exist is a 500,
  as Jupyter's is (research). /files/<path> answers GET with byte ranges, 206 with Content-Range, 416 past the end
  (measured for a small file). The Jupyter root is `pod.files`; it is not the persistent home, and `root_has_home` makes
  its listing show a `home` folder the way a newer image's would.

  With `bandwidth` set, every request and every held answer takes its size over that many bytes a second on the clock,
  which gives a push or a pull a speed to measure.

  The model server (FakeModelServer) answers /v1/models, /health, /metrics and /v1/chat/completions, with the
  OpenAI-style chunks vLLM streams, a usage chunk, tool-call fragments and a final [DONE]. It takes `ttft_s` to a first
  token and then writes `tokens_per_s` tokens a second. FakeFileServer stands in for a pod-side file server on a port.

A test sees the traffic in `fake.calls` and what reached each listener in its own lists (the proxy forwards every request
header to the model server, so `authorizations` shows the hub token arriving there).
"""
from __future__ import annotations

import base64
import binascii
import json
import re
import urllib.parse
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from lab.hub import Account
from lab.hub_fake import FakeClock

__all__ = ["FakePodProxy", "FakeModelServer", "FakeFileServer", "FakeCall"]

HTML = {"content-type": "text/html"}
BAN_PAGE = "<html><head><title>Access denied</title></head><body>Error 1010: browser signature banned</body></html>"
TIMEOUT_PAGE = "<html><head><title>524: A timeout occurred</title></head><body>Error 524: A timeout occurred</body></html>"
TOO_BIG_PAGE = "<html><head><title>413 Request Entity Too Large</title></head><body>413</body></html>"
BAD_GATEWAY_PAGE = "<html><head><title>502 Bad gateway</title></head><body>Error 502: Bad gateway</body></html>"
JUPYTER_ERROR_PAGE = "<html><body><h1>500 : Internal Server Error</h1></body></html>"
SSE_HEADERS = {"content-type": "text/event-stream; charset=utf-8", "cache-control": "no-cache"}


@dataclass
class FakeCall:
    """One request the fake received, as the hub's front end saw it. body is the JSON the request carried, with a
    contents-API file's content cut to its length; t is the seconds since the clock started."""

    method: str
    path: str
    query: str
    headers: dict[str, str]
    body: Any
    body_bytes: int
    t: float
    status: int | None = None


@dataclass
class _Pod:
    letter: str
    user: str
    token: str
    state: str = "running"                  # running, stopped, starting
    route_until: float = 0.0                # a stopped pod's route lingers (503) until this clock time
    listeners: dict[int, Any] = field(default_factory=dict)
    files: dict[str, bytearray] = field(default_factory=dict)      # the Jupyter root
    dirs: set[str] = field(default_factory=lambda: {""})
    root_has_home: bool = False


@dataclass
class _Canned:
    method: str
    contains: str
    status: int
    body: Any
    text: str | None
    remaining: int
    apply: bool
    raises: Exception | None = None
    headers: dict[str, str] = field(default_factory=dict)
    after: int = 0                          # matching requests to let through before this one takes effect
    content: bytes | None = None            # a binary body, instead of body or text


class _Timed(httpx.SyncByteStream):
    """A body that arrives in pieces, each after its delay on the fake clock, and can break part-way."""

    def __init__(self, clock: FakeClock, pieces: Sequence[tuple[float, bytes]], break_after: int | None = None) -> None:
        self._clock = clock
        self._pieces = list(pieces)
        self._break_after = break_after

    def __iter__(self) -> Iterator[bytes]:
        for index, (delay, data) in enumerate(self._pieces):
            if self._break_after is not None and index >= self._break_after:
                raise httpx.RemoteProtocolError("peer closed connection without sending complete message body")
            self._clock.sleep(delay)
            yield data

    def close(self) -> None:
        pass


def _sse(payload: Any) -> bytes:
    return b"data: " + json.dumps(payload).encode("utf-8") + b"\n\n"


class FakeModelServer:
    """vLLM (or llama-server) listening on a pod port. reply is the text it writes, a token to a word (or reply_tokens
    words w0 w1 ... when given); a request's max_tokens cuts it short with finish reason `length`. The default speed
    (0.55 s to a first token, 7.7 tokens a second) is what the R9700 teacher measured on 2026-10-10; the Strix Halo
    pods have not been measured, and a test that depends on timing sets its own."""

    def __init__(self, fake: "FakePodProxy", *, models: Sequence[str] = ("teacher",), tokens_per_s: float = 7.7,
                 ttft_s: float = 0.55, reply: str = "Hello from the fake teacher.", reply_tokens: int | None = None,
                 healthy: bool = True) -> None:
        self.fake = fake
        self.models = tuple(models)
        self.tokens_per_s = tokens_per_s
        self.ttft_s = ttft_s
        self.reply = reply
        self.reply_tokens = reply_tokens
        self.healthy = healthy
        self.tool_calls: list[tuple[str, dict[str, Any]]] = []     # answered when a request carries tools
        self.fail_next: list[tuple[int, Any]] = []                 # (status, JSON body) for the next chat requests
        self.break_stream_after: int | None = None                 # events a stream delivers before it breaks
        self.requests: list[dict[str, Any]] = []                   # the JSON bodies of chat requests
        self.accepts: list[str | None] = []                        # the Accept header of every request
        self.authorizations: list[str | None] = []                 # the Authorization header of every request

    def _pieces(self, max_tokens: int | None) -> tuple[list[str], bool]:
        words = [f"w{i}" for i in range(self.reply_tokens)] if self.reply_tokens is not None else self.reply.split(" ")
        pieces = [word if i == 0 else " " + word for i, word in enumerate(words)]
        if max_tokens is not None and max_tokens < len(pieces):
            return pieces[:max_tokens], True
        return pieces, False

    def handle(self, request: httpx.Request, tail: str) -> httpx.Response:
        self.accepts.append(request.headers.get("accept"))
        self.authorizations.append(request.headers.get("authorization"))
        if request.method == "GET" and tail == "v1/models":
            return httpx.Response(200, json={"object": "list", "data": [{"id": name, "object": "model"} for name in self.models]})
        if request.method == "GET" and tail == "health":
            return httpx.Response(200 if self.healthy else 503, text="")
        if request.method == "GET" and tail == "metrics":
            return httpx.Response(200, text="vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n",
                                  headers={"content-type": "text/plain"})
        if request.method == "POST" and tail == "v1/chat/completions":
            return self._chat(request)
        return httpx.Response(404, json={"detail": "Not Found"})

    def _chat(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        self.requests.append(body)
        if self.fail_next:
            status, payload = self.fail_next.pop(0)
            return httpx.Response(status, json=payload)
        name = body.get("model") or self.models[0]
        if name not in self.models:
            return httpx.Response(404, json={"object": "error", "message": f"The model `{name}` does not exist.",
                                             "type": "NotFoundError", "code": 404})
        pieces, cut = self._pieces(body.get("max_tokens"))
        calls = self.tool_calls if body.get("tools") and self.tool_calls else []
        if calls:
            pieces, cut = [], False             # a model that calls a tool writes no text
        finish = "tool_calls" if calls else ("length" if cut else "stop")
        usage = {"prompt_tokens": sum(len(str(m.get("content", "")).split()) for m in body.get("messages", [])),
                 "completion_tokens": len(pieces), "total_tokens": 0}
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        steps = max(len(pieces) + len(calls) - 1, 0)
        duration = self.ttft_s + steps / self.tokens_per_s
        if body.get("stream"):
            events = self._events(name, pieces, calls, finish, usage if (body.get("stream_options") or {}).get("include_usage") else None)
            if request.headers.get("accept") == "text/event-stream":
                return httpx.Response(200, headers=SSE_HEADERS,
                                      stream=_Timed(self.fake.clock, events, self.break_stream_after))
            return self.fake.held(duration, lambda: httpx.Response(200, headers=SSE_HEADERS,
                                                                    content=b"".join(data for _, data in events)))
        message: dict[str, Any] = {"role": "assistant", "content": "".join(pieces)}
        if calls:
            message["content"] = None
            message["tool_calls"] = [{"id": f"call_{i}", "type": "function",
                                      "function": {"name": call, "arguments": json.dumps(args)}}
                                     for i, (call, args) in enumerate(calls)]
        return self.fake.held(duration, lambda: httpx.Response(200, json={
            "id": "chatcmpl-fake", "object": "chat.completion", "model": name,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage}))

    def _events(self, name: str, pieces: list[str], calls: list[tuple[str, dict[str, Any]]], finish: str,
                usage: dict[str, Any] | None) -> list[tuple[float, bytes]]:
        def chunk(delta: dict[str, Any], reason: str | None = None) -> bytes:
            return _sse({"id": "chatcmpl-fake", "object": "chat.completion.chunk", "model": name,
                         "choices": [{"index": 0, "delta": delta, "finish_reason": reason}]})

        events: list[tuple[float, bytes]] = [(0.0, chunk({"role": "assistant", "content": ""}))]
        first = True
        for piece in pieces:
            events.append((self.ttft_s if first else 1.0 / self.tokens_per_s, chunk({"content": piece})))
            first = False
        for index, (call, args) in enumerate(calls):
            text = json.dumps(args)
            half = max(1, len(text) // 2)
            delay = self.ttft_s if first else 1.0 / self.tokens_per_s
            first = False
            events.append((delay, chunk({"tool_calls": [{"index": index, "id": f"call_{index}", "type": "function",
                                                         "function": {"name": call, "arguments": ""}}]})))
            events.append((0.0, chunk({"tool_calls": [{"index": index, "function": {"arguments": text[:half]}}]})))
            events.append((0.0, chunk({"tool_calls": [{"index": index, "function": {"arguments": text[half:]}}]})))
        events.append((self.ttft_s if first else 0.0, chunk({}, finish)))
        if usage is not None:
            events.append((0.0, _sse({"id": "chatcmpl-fake", "object": "chat.completion.chunk", "model": name,
                                      "choices": [], "usage": usage})))
        events.append((0.0, b"data: [DONE]\n\n"))
        return events


class FakeFileServer:
    """A file server a task puts on a pod port (R10's): serves pod.files below `root`. With ranges=False it is
    `python3 -m http.server`, which ignores a Range header and always sends the whole file."""

    def __init__(self, fake: "FakePodProxy", pod: _Pod, *, root: str = "", ranges: bool = True) -> None:
        self.fake = fake
        self.pod = pod
        self.root = root.strip("/")
        self.ranges = ranges
        self.requests: list[tuple[str, str | None]] = []           # (path, Range header)
        self.authorizations: list[str | None] = []

    def handle(self, request: httpx.Request, tail: str) -> httpx.Response:
        path = urllib.parse.unquote(tail)
        self.requests.append((path, request.headers.get("range")))
        self.authorizations.append(request.headers.get("authorization"))
        if request.method != "GET":
            return httpx.Response(501, text="Unsupported method", headers={"content-type": "text/plain"})
        data = self.pod.files.get(f"{self.root}/{path}".strip("/"))
        if data is None:
            return httpx.Response(404, text="File not found", headers={"content-type": "text/plain"})
        if not self.ranges:
            return httpx.Response(200, content=bytes(data), headers={"content-type": "application/octet-stream"})
        return serve_range(data, request.headers.get("range"))


def serve_range(data: bytes | bytearray, range_header: str | None) -> httpx.Response:
    """What Tornado's StaticFileHandler (the Jupyter server's /files/) answers for a Range header."""
    size = len(data)
    headers = {"accept-ranges": "bytes", "content-type": "application/octet-stream"}
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", (range_header or "").strip())
    if range_header is None or match is None or (match.group(1) == "" and match.group(2) == ""):
        return httpx.Response(200, content=bytes(data), headers=headers)
    first, last = match.group(1), match.group(2)
    if first == "":
        count = int(last)
        if count == 0 or size == 0:
            return httpx.Response(416, headers={**headers, "content-range": f"bytes */{size}"})
        start, end = max(0, size - count), size - 1
    else:
        start = int(first)
        end = min(int(last), size - 1) if last else size - 1
        if start >= size or end < start:
            return httpx.Response(416, headers={**headers, "content-range": f"bytes */{size}"})
    return httpx.Response(206, content=bytes(data[start:end + 1]),
                          headers={**headers, "content-range": f"bytes {start}-{end}/{size}"})


class FakePodProxy:
    """The path to the pods, as the callers' code sees it. See the module docstring."""

    def __init__(self, clock: FakeClock | None = None, *, host: str = "hub.test", edge_body_cap: int = 100_000_000,
                 proxy_response_cap: int = 100 * 1024 * 1024, cloudflare_read_timeout_s: float = 125.0,
                 bandwidth: float | None = None) -> None:
        self.clock = clock or FakeClock()
        self.url = f"https://{host}"
        self.bandwidth = bandwidth                             # bytes a second each way; None: instant
        self.edge_body_cap = edge_body_cap
        self.proxy_response_cap = proxy_response_cap
        self.cloudflare_read_timeout_s = cloudflare_read_timeout_s
        self.hub_down = False
        self.break_next_download_after: int | None = None      # the next file body breaks off after this many bytes
        self.calls: list[FakeCall] = []
        self._pods: dict[str, _Pod] = {}
        self._canned: list[_Canned] = []
        self.transport = httpx.MockTransport(self._handle)

    # --- scenario ---------------------------------------------------------------------------

    def add_account(self, letter: str, *, user: str, token: str) -> Account:
        """A hub account whose pod is running, returned in the form the callers take."""
        self._pods[letter] = _Pod(letter, user, token)
        return Account(letter, user, token)

    def pod(self, letter: str) -> _Pod:
        return self._pods[letter]

    def stop_pod(self, letter: str, *, route_lingers_s: float = 0.0) -> None:
        """The pod goes away. Its route stays for route_lingers_s seconds (503), then the hub answers 424."""
        pod = self._pods[letter]
        pod.state = "stopped"
        pod.route_until = self.clock.time() + route_lingers_s

    def start_pod(self, letter: str, *, ready: bool = True) -> None:
        self._pods[letter].state = "running" if ready else "starting"

    def serve_model(self, letter: str, port: int = 8000, **options: Any) -> FakeModelServer:
        listener = FakeModelServer(self, **options)
        self._pods[letter].listeners[port] = listener
        return listener

    def serve_files(self, letter: str, port: int, **options: Any) -> FakeFileServer:
        pod = self._pods[letter]
        listener = FakeFileServer(self, pod, **options)
        pod.listeners[port] = listener
        return listener

    def put_file(self, letter: str, path: str, data: bytes) -> None:
        """A file already in the pod's Jupyter root (its folders are made on the way)."""
        pod = self._pods[letter]
        parts = path.split("/")
        for i in range(1, len(parts)):
            pod.dirs.add("/".join(parts[:i]))
        pod.files[path] = bytearray(data)

    def file(self, letter: str, path: str) -> bytes | None:
        data = self._pods[letter].files.get(path)
        return None if data is None else bytes(data)

    def queue(self, method: str, contains: str, status: int, *, body: Any = None, text: str | None = None,
              content: bytes | None = None, headers: dict[str, str] | None = None, times: int = 1, apply: bool = False,
              after: int = 0) -> None:
        """Answer the next `times` requests of that method whose path contains `contains` with this status, body and
        headers, once `after` of them have gone through untouched. With apply=True the request is first carried out as
        usual and only its answer is replaced: an answer lost on its way back."""
        self._canned.append(_Canned(method.upper(), contains, status, body, text, times, apply, None, dict(headers or {}),
                                    after, content))

    def queue_dropped(self, method: str, contains: str, *, times: int = 1, after: int = 0) -> None:
        """Drop the connection of the next `times` requests of that method whose path contains `contains`."""
        error = httpx.ReadError("the connection was reset")
        self._canned.append(_Canned(method.upper(), contains, 0, None, None, times, False, error, {}, after))

    def calls_matching(self, method: str | None = None, contains: str = "") -> list[FakeCall]:
        return [call for call in self.calls
                if (method is None or call.method == method.upper()) and contains in call.path]

    # --- time -------------------------------------------------------------------------------

    def held(self, seconds: float, build: Callable[[], httpx.Response]) -> httpx.Response:
        """A reply the proxy holds until the model server has finished: nothing reaches Cloudflare meanwhile, so one
        that takes longer than its read timeout is cut with 524, and a body over the proxy's cap comes back as 599."""
        if seconds > self.cloudflare_read_timeout_s:
            self.clock.sleep(self.cloudflare_read_timeout_s)
            return httpx.Response(524, text=TIMEOUT_PAGE, headers=HTML)
        self.clock.sleep(seconds)
        response = build()
        if len(response.content) > self.proxy_response_cap:
            return httpx.Response(599, text="HTTP 599: Response body too large", headers={"content-type": "text/plain"})
        return response

    # --- the requests -----------------------------------------------------------------------

    def _handle(self, request: httpx.Request) -> httpx.Response:
        body: Any = None
        if request.content and "json" in request.headers.get("content-type", ""):
            try:
                body = json.loads(request.content)
            except ValueError:
                body = None
            if isinstance(body, dict) and isinstance(body.get("content"), str):
                body = {**body, "content": len(body["content"])}
        call = FakeCall(request.method, request.url.path, request.url.query.decode(),
                        {name.lower(): value for name, value in request.headers.items()}, body, len(request.content),
                        self.clock.elapsed)
        self.calls.append(call)
        response = self._edge(request, call)
        response = self._break_download(request, response)
        if self.bandwidth:
            moved = len(request.content)
            if not isinstance(response.stream, _Timed):
                moved += len(response.content)
            self.clock.sleep(moved / self.bandwidth)
        call.status = response.status_code
        response.headers["server"] = "cloudflare"
        return response

    def _break_download(self, request: httpx.Request, response: httpx.Response) -> httpx.Response:
        """If asked to, let a file body start and then break off, as a link that drops mid-download does."""
        cut = self.break_next_download_after
        if (cut is None or request.method != "GET" or response.status_code not in (200, 206) or "/api/" in request.url.path
                or isinstance(response.stream, _Timed) or len(response.content) <= cut):
            return response
        self.break_next_download_after = None
        headers = {name: value for name, value in response.headers.items() if name.lower() != "content-length"}
        return httpx.Response(response.status_code, headers=headers,
                              stream=_Timed(self.clock, [(0.0, response.content[:cut]), (0.0, b"")], break_after=1))

    def _edge(self, request: httpx.Request, call: FakeCall) -> httpx.Response:
        if self.hub_down:
            return httpx.Response(502, text=BAD_GATEWAY_PAGE, headers=HTML)
        if not call.headers.get("user-agent", "").startswith("Mozilla/5.0"):
            return httpx.Response(403, text=BAN_PAGE, headers=HTML)
        if call.body_bytes > self.edge_body_cap:
            return httpx.Response(413, text=TOO_BIG_PAGE, headers=HTML)
        canned = self._take_canned(call)
        if canned is not None and canned.raises is not None:
            raise canned.raises
        if canned is not None and not canned.apply:
            return self._canned_response(canned)
        response = self._hub(request, call)
        return self._canned_response(canned) if canned is not None else response

    def _take_canned(self, call: FakeCall) -> _Canned | None:
        for entry in self._canned:
            if entry.remaining > 0 and entry.method == call.method and entry.contains in call.path:
                if entry.after > 0:
                    entry.after -= 1
                    continue
                entry.remaining -= 1
                return entry
        return None

    @staticmethod
    def _canned_response(entry: _Canned) -> httpx.Response:
        if entry.content is not None:
            return httpx.Response(entry.status, content=entry.content, headers=entry.headers)
        if entry.body is not None:
            return httpx.Response(entry.status, json=entry.body, headers=entry.headers)
        return httpx.Response(entry.status, text=entry.text or "", headers={"content-type": "text/plain", **entry.headers})

    def _hub(self, request: httpx.Request, call: FakeCall) -> httpx.Response:
        match = re.match(r"^/user/([^/]+)(/.*)?$", call.path)
        if match is None:
            return httpx.Response(404, json={"status": 404, "message": "Not Found"})
        user = urllib.parse.unquote(match.group(1))
        rest = match.group(2) or "/"
        scheme, _, token = call.headers.get("authorization", "").partition(" ")
        holder = next((pod for pod in self._pods.values() if scheme.lower() == "token" and pod.token == token), None)
        if holder is None:
            if request.method in ("GET", "HEAD"):
                return httpx.Response(302, headers={"location": f"/hub/api/oauth2/authorize?client_id=jupyterhub-user-{user}"})
            return httpx.Response(403, json={"message": "Forbidden", "reason": None})
        if holder.user != user:
            return httpx.Response(302, headers={"location": f"/hub/user/{user}{rest}"})
        if holder.state == "starting":
            if request.method == "GET":
                return httpx.Response(303, headers={"location": f"/hub/spawn-pending/{user}"})
            return httpx.Response(503, text="", headers={"content-type": "text/plain"})
        if holder.state == "stopped":
            if self.clock.time() < holder.route_until:
                return httpx.Response(503, text="Service Unavailable", headers={"content-type": "text/plain"})
            wants_json = request.method != "GET" or "json" in call.headers.get("accept", "") or "/api/" in rest
            if wants_json:
                return httpx.Response(424, json={"status": 424, "message": (
                    f"JupyterHub server no longer running at /user/{user}/. Restart the server at /hub/home")})
            return httpx.Response(424, text="<html><body>Server not running</body></html>", headers=HTML)
        return self._pod(request, call, holder, rest)

    # --- the pod ----------------------------------------------------------------------------

    def _pod(self, request: httpx.Request, call: FakeCall, pod: _Pod, rest: str) -> httpx.Response:
        proxied = re.match(r"^/proxy/(\d+)/(.*)$", rest, re.DOTALL)
        if proxied:
            return self._proxy(request, pod, int(proxied.group(1)), proxied.group(2))
        if re.match(r"^/proxy/absolute/\d+/", rest):
            return httpx.Response(404, json={"detail": "Not Found"})
        contents = re.match(r"^/api/contents/?(.*)$", rest, re.DOTALL)
        if contents:
            return self._contents(request, pod, urllib.parse.unquote(contents.group(1)).strip("/"))
        files = re.match(r"^/files/(.+)$", rest, re.DOTALL)
        if files and request.method in ("GET", "HEAD"):
            data = pod.files.get(urllib.parse.unquote(files.group(1)))
            if data is None:
                return httpx.Response(404, text=JUPYTER_ERROR_PAGE.replace("500", "404"), headers=HTML)
            return serve_range(data, request.headers.get("range"))
        if rest == "/api/me":
            return httpx.Response(200, json={"identity": {"username": pod.user}, "permissions": {}})
        return httpx.Response(404, json={"message": "Not Found", "reason": None})

    def _proxy(self, request: httpx.Request, pod: _Pod, port: int, tail: str) -> httpx.Response:
        listener = pod.listeners.get(port)
        if listener is None:
            return httpx.Response(500, text=JUPYTER_ERROR_PAGE, headers=HTML)
        response = listener.handle(request, tail)
        if isinstance(response.stream, _Timed):
            return response                       # a progressive stream: nothing is held
        if len(response.content) > self.proxy_response_cap:
            return httpx.Response(599, text="HTTP 599: Response body too large", headers={"content-type": "text/plain"})
        return response

    def _contents(self, request: httpx.Request, pod: _Pod, path: str) -> httpx.Response:
        query = dict(request.url.params)
        if request.method == "GET":
            return self._contents_get(pod, path, query)
        if request.method == "PUT":
            return self._contents_put(request, pod, path)
        if request.method == "DELETE":
            if path in pod.files:
                del pod.files[path]
            elif path in pod.dirs and path:
                pod.dirs.discard(path)
            else:
                return httpx.Response(404, json={"message": f"No such file or directory: {path}", "reason": None})
            return httpx.Response(204)
        return httpx.Response(405, json={"message": "Method Not Allowed", "reason": None})

    @staticmethod
    def _model(path: str, kind: str, size: int | None) -> dict[str, Any]:
        return {"name": path.rsplit("/", 1)[-1], "path": path, "type": kind, "size": size, "writable": True,
                "created": "2026-10-10T14:00:00Z", "last_modified": "2026-10-10T14:00:00Z", "mimetype": None,
                "content": None, "format": None}

    def _contents_get(self, pod: _Pod, path: str, query: dict[str, str]) -> httpx.Response:
        wants_content = query.get("content", "1") != "0"
        if path in pod.dirs:
            model = self._model(path, "directory", None)
            if wants_content:
                prefix = f"{path}/" if path else ""
                names = {name[len(prefix):].split("/", 1)[0] for name in list(pod.files) + list(pod.dirs)
                         if name and name.startswith(prefix) and name != path}
                entries = []
                for name in sorted(names):
                    full = prefix + name
                    entries.append(self._model(full, "directory" if full in pod.dirs else "file",
                                               None if full in pod.dirs else len(pod.files[full])))
                if path == "" and pod.root_has_home and "home" not in names:
                    entries.append(self._model("home", "directory", None))
                model.update(content=entries, format="json")
            return httpx.Response(200, json=model)
        data = pod.files.get(path)
        if data is None:
            return httpx.Response(404, json={"message": f"No such file or directory: {path}", "reason": None})
        model = self._model(path, "file", len(data))
        if wants_content:
            model.update(content=base64.b64encode(bytes(data)).decode("ascii"), format="base64")
        return httpx.Response(200, json=model)

    def _contents_put(self, request: httpx.Request, pod: _Pod, path: str) -> httpx.Response:
        try:
            model = json.loads(request.content or b"{}")
        except ValueError:
            return httpx.Response(400, json={"message": "Invalid JSON in body of request", "reason": None})
        parent = path.rsplit("/", 1)[0] if "/" in path else ""
        existed = path in pod.files or path in pod.dirs
        if model.get("type") == "directory":
            if path in pod.files:
                return httpx.Response(400, json={"message": f"Not a directory: {path}", "reason": None})
            if parent not in pod.dirs:
                return httpx.Response(500, json={"message": f"Unexpected error while saving file: {path} [Errno 2]", "reason": None})
            pod.dirs.add(path)
            return httpx.Response(200 if existed else 201, json=self._model(path, "directory", None))
        if model.get("type") != "file":
            return httpx.Response(400, json={"message": "No file type provided", "reason": None})
        if "content" not in model:
            return httpx.Response(400, json={"message": "No file content provided", "reason": None})
        fmt = model.get("format")
        try:
            if fmt == "base64":
                raw = base64.b64decode(str(model["content"]).encode("ascii"), validate=True)
            elif fmt == "text":
                raw = str(model["content"]).encode("utf-8")
            else:
                return httpx.Response(400, json={"message": "Must specify format of file contents as 'text' or 'base64'",
                                                 "reason": None})
        except (binascii.Error, ValueError):
            return httpx.Response(400, json={"message": f"Encoding error saving {path}", "reason": None})
        if parent not in pod.dirs or path in pod.dirs:
            return httpx.Response(500, json={"message": f"Unexpected error while saving file: {path} [Errno 2]", "reason": None})
        chunk = model.get("chunk")
        if chunk is None or chunk == 1:
            pod.files[path] = bytearray(raw)
        else:
            pod.files.setdefault(path, bytearray()).extend(raw)
        return httpx.Response(200 if existed else 201, json=self._model(path, "file", len(pod.files[path])))
