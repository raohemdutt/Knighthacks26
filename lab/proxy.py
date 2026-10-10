"""proxy: the box's way to a pod's model server, through the hub, with the account's hub token.

    from lab import hub, proxy

    account = hub.load_accounts(["C"])["C"]        # AUP_HUB_USER_C, AUP_HUB_TOKEN_C and, if set, AUP_HUB_URL_C
    with proxy.PodLink.for_account(account) as link:
        teacher = link.server(8000)                # https://<hub>/user/<account>/proxy/8000/
        teacher.models()                           # ["teacher"]
        done = teacher.chat([{"role": "user", "content": "Hello"}], thinking=False, on_text=print)
        done.content, done.first_token_s, done.usage

    runtime = proxy.proxied_runtime(model_version, link, 8000, model_name="teacher")  # the stock LlamaServerRuntime

    python -m lab.proxy probe --account C [--port 8000] [--wrong-token]
    python -m lab.proxy chat --account C --prompt "Say hello" [--port 8000] [--max-tokens 64] [--no-thinking]

What sits between the box and a model server on a pod: Cloudflare, the hub's proxy, the pod's Jupyter server running
jupyter-server-proxy, then the model server. Every fact below is from lab/refine/research/plumbing/proxy-path.md (its
sources and the measurements of 2026-10-10 are indexed there). Those were taken on the DUB hub and its pods; the SJC hub
(accounts C and D) has not been probed for Cloudflare, for the 125 s and 100 MB limits or for jupyter-server-proxy, and
none of it has been tried from the Jetson. `probe` prints the answer's `server` and `cf-ray` headers so the first live
call settles whether SJC sits behind Cloudflare; streaming is the right way to send a long reply either way.

The address. https://<hub>/user/<account>/proxy/<port>/<path>. The hub is the one lab/hub.py settles on for the
account (its AUP_HUB_URL_<letter>, else $LAB_HUB_URL, else DUB); this module never names one. The
/proxy/absolute/<port>/ form is never used: the model server would see the prefix and answer 404.

The token. Every request carries `Authorization: token <hub token>` and a browser's User-Agent, because the hub's front
end refuses a script's. A redirect is never followed: a wrong or missing token answers 302 to the hub's login page on a
GET and 403 on any other method, and a client that followed it would take the login page for an answer. The token is cut
out of every message this module raises or returns, and the account's name is left out of them too. The proxy forwards
every request header to the model server, so a server that logs headers writes the token down.

Streaming. jupyter-server-proxy streams a reply only when the request's Accept header is exactly `text/event-stream`;
with any other value it holds the whole reply until the model server has finished. Cloudflare closes a request whose
origin has sent nothing for 125 seconds (524), and a held reply sends nothing, so a long answer fails unless it is
streamed. chat() therefore always sends stream true with that exact header and gathers the pieces itself; the answer can
take minutes because bytes flow from the first token. Streams are read with `Accept-Encoding: identity` so nothing
compresses and holds them back.

Failures. Every answer that is not a success becomes one ProxyError (a subclass of httpx.HTTPError, so the callers that
already catch that keep working). Its kind says what happened, `retryable` whether sending the same call again later can
work with nothing changed, and `hint` what to do:

    302/303/307 to a login page, 403    BadToken           the token is wrong, revoked or another account's
    303 to the spawn-pending page       PodStarting        retryable
    401                                 Unauthorized       the model server itself wants a key
    403 with Cloudflare's 1010 page     Blocked            the front end refused the client's User-Agent
    404                                 NotFound           no access to that pod, a dead route, or no such path
    413                                 TooLarge           Cloudflare refuses a request body over 100 MB
    424                                 PodStopped         the pod's server is not running
    500, an HTML page                   ModelServerDown    nothing listens on that port
    500, JSON; other 4xx and 5xx        ModelServerError   the model server's own error
    502, 504, 520-523, 525-530          HubDown            the front door or the hub is down; retryable
    503                                 PodUnresponsive    a route to a pod that no longer answers; retryable
    524                                 GatewayTimeout     125 s without a byte from the origin: stream
    599                                 ProxyTimeout       the proxy gave up (over 300 s buffered, or over 100 MiB)
    no answer at all                    Unreachable        retryable
    a stream that broke after it began  StreamInterrupted  retryable; what arrived is in .partial

A GET, HEAD, PUT or DELETE that meets 502, 503, 504 or a dropped connection is tried again after 1, 2 and 4 seconds, as
lab/hub.py does; a POST never is. Nothing here starts, stops or restarts a pod: a PodStopped is for the controller, and
launching is the owner's call.

PodLink takes the address, the token, the user agent and the scrubber from a lab.hub.Hub and sends with the same rules.
It does not use Hub.request because that reads a whole answer into text cut at 2000 characters, which cannot carry an
event stream or a byte range.

For the stock runtime. proxied_runtime() builds runtime.llama.LlamaServerRuntime over a ProxyTransport. The transport
adds the token, sends the exact Accept header on a stream, refuses to send the token to any other host, raises the
mapped errors, and turns a non-streamed chat completion into a streamed one that it gathers back into the ordinary
JSON reply, so no reply the runtime asks for can run into the 125 second cut. runtime/ itself is not changed. Two
options serve a caller that wants more than the runtime writes: extra sets fields on every chat completion body (vLLM's
priority, chat_template_kwargs, a smaller max_tokens), replacing the runtime's own value of the same name; text_only
refuses, before a byte is sent, any request whose messages hold anything but plain text or whose body carries a data
URL, so a picture cannot leave through that transport whatever the runtime is asked to do.

Time. Durations come from `clock` and pauses from `sleep`, both handed in by tests.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import logging
import re
import sys
import time
import urllib.parse
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TextIO

import httpx

if __package__ in (None, ""):          # started as `python lab/proxy.py`: put the repo root on the path
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.hub import Account, Hub, MissingCredentials, load_accounts

__all__ = [
    "EVENT_STREAM", "ProxyError", "BadToken", "Unauthorized", "Blocked", "NotFound", "TooLarge", "PodStopped",
    "PodStarting", "PodUnresponsive", "HubDown", "ModelServerDown", "ModelServerError", "GatewayTimeout",
    "ProxyTimeout", "Unreachable", "StreamInterrupted", "error_for", "ProxyReply", "CallRecord", "SseEvent",
    "parse_sse", "StreamHandle", "ChatCompletion", "collect_chat", "PodLink", "ModelServer", "ProxyTransport",
    "text_only_problem", "proxied_runtime", "main", "build_parser"]

log = logging.getLogger("lab.proxy")

EVENT_STREAM = "text/event-stream"
JSON_TYPE = "application/json"
IDEMPOTENT_METHODS = frozenset({"GET", "HEAD", "PUT", "DELETE", "OPTIONS"})
RETRY_STATUSES = frozenset({502, 503, 504})      # the gateway errors lab/hub.py also tries again
RETRY_PAUSES_S = (1.0, 2.0, 4.0)
CONNECT_TIMEOUT_S = 10.0
READ_TIMEOUT_S = 130.0       # just past Cloudflare's 125 s, so the edge answers (524) before the client gives up
DETAIL_CHARS = 200
MIN_NAME_CHARS = 4           # a shorter account name would mangle the text it is cut out of
TEACHER_PORT = 8000


# --- what went wrong --------------------------------------------------------------------------

class ProxyError(httpx.HTTPError):
    """A call through the proxy that did not succeed. kind names the cause, retryable says whether sending the same
    call again later can work with nothing changed, hint says what to do, status is the HTTP status (None when there
    was no answer), where is the call without the account ("GET /proxy/8000/v1/models") and detail is the server's own
    message when it sent one. The message never holds the token or the account's name."""

    kind = "error"
    retryable = False
    hint = "the call through the proxy failed"

    def __init__(self, message: str, *, status: int | None = None, where: str = "", detail: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.where = where
        self.detail = detail


class BadToken(ProxyError):
    kind = "bad_token"
    hint = ("the hub did not accept the token for this pod (a redirect to its login page, or 403): the token is wrong, "
            "revoked or another account's, and sending the same call again cannot help")


class Unauthorized(ProxyError):
    kind = "unauthorized"
    hint = ("the model server itself refused the call: it wants a key of its own, and the hub token cannot share its "
            "Authorization header")


class Blocked(ProxyError):
    kind = "blocked"
    hint = ("the hub's front end refused this client (Cloudflare error 1010: the User-Agent is not a browser's); check "
            "$LAB_HUB_USER_AGENT")


class NotFound(ProxyError):
    kind = "not_found"
    hint = "the token has no access to that pod, the route is gone, or the server has no such path"


class TooLarge(ProxyError):
    kind = "too_large"
    hint = "Cloudflare refuses a request body over 100 MB; send it in smaller parts"


class PodStopped(ProxyError):
    kind = "pod_stopped"
    hint = ("the pod's server is not running; launch it again (python -m lab.controller launch, with the owner's "
            "go) and call again")


class PodStarting(ProxyError):
    kind = "pod_starting"
    retryable = True
    hint = "the pod is still starting (the hub sent the caller to its spawn-pending page); wait and call again"


class PodUnresponsive(ProxyError):
    kind = "pod_unresponsive"
    retryable = True
    hint = ("the hub still routes to the pod but nothing answers there; a pod that just died shows this for up to "
            "30 s and then 424")


class HubDown(ProxyError):
    kind = "hub_down"
    retryable = True
    hint = ("the hub's front door is not answering (502, 504 or a Cloudflare 52x); it is down or restarting, wait and "
            "call again")


class ModelServerDown(ProxyError):
    kind = "model_server_down"
    hint = ("the pod's Jupyter server found nothing listening on that port (the model server is not started, or is "
            "restarting)")


class ModelServerError(ProxyError):
    kind = "model_server_error"
    hint = "the server on the pod (the model server, or Jupyter for the contents API) answered with an error of its own"


class GatewayTimeout(ProxyError):
    kind = "gateway_timeout"
    hint = ("Cloudflare gives up on a request its origin has not answered for 125 s, and a reply that is not "
            "streamed sends nothing until it is whole; send stream true with Accept text/event-stream")


class ProxyTimeout(ProxyError):
    kind = "proxy_timeout"
    hint = ("the pod's proxy gave up (a buffered request over 300 s, or a response body over 100 MiB); stream the "
            "answer, or fetch a big file in Range parts")


class Unreachable(ProxyError):
    kind = "unreachable"
    retryable = True
    hint = "no answer at all (no connection, a timeout, a dropped link)"


class StreamInterrupted(ProxyError):
    kind = "stream_interrupted"
    retryable = True
    hint = "the stream broke after it had begun; what arrived before is in .partial"
    partial: str = ""


def _one_line(text: Any, limit: int = DETAIL_CHARS) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[:max(0, limit - 3)] + "..."


def _detail(body: Any, text: str) -> str:
    """The server's own message: a JSON message, error or detail field (also one level down, as vLLM nests it), else
    its page with the markup taken out."""
    if isinstance(body, Mapping):
        for key in ("message", "error", "detail"):
            value = body.get(key)
            if isinstance(value, str) and value.strip():
                return _one_line(value)
            if isinstance(value, Mapping):
                inner = value.get("message")
                if isinstance(inner, str) and inner.strip():
                    return _one_line(inner)
    return _one_line(re.sub(r"<[^>]+>", " ", text))


def _looks_like_html(headers: Mapping[str, str], text: str) -> bool:
    if "html" in headers.get("content-type", "").lower():
        return True
    return text.lstrip()[:1] == "<"


def _classify(status: int, headers: Mapping[str, str], text: str) -> type[ProxyError] | None:
    if status < 300:
        return None
    if status < 400:
        return PodStarting if "spawn-pending" in headers.get("location", "") else BadToken
    if status == 401:
        return Unauthorized
    if status == 403:
        lowered = text.lower()
        return Blocked if ("1010" in lowered or "browser signature" in lowered) else BadToken
    if status == 404:
        return NotFound
    if status == 413:
        return TooLarge
    if status == 424:
        return PodStopped
    if status == 500:
        return ModelServerDown if _looks_like_html(headers, text) else ModelServerError
    if status == 503:
        return PodUnresponsive
    if status == 524:
        return GatewayTimeout
    if status == 599:
        return ProxyTimeout
    if status in (502, 504) or 520 <= status <= 530:
        return HubDown
    return ModelServerError


def error_for(status: int, *, headers: Mapping[str, str], text: str, body: Any = None, where: str = "",
              label: str = "pod", scrub: Callable[[str], str] = lambda s: s) -> ProxyError | None:
    """The ProxyError an answer stands for, or None when the status is a success. headers are lower-cased names; text
    and body (the parsed JSON, when there was any) give the server's own message. A redirect is told apart by where it
    points and is never described by its address, which names the account."""
    cls = _classify(status, headers, text)
    if cls is None:
        return None
    detail = _detail(body, text) if status >= 400 else ""
    message = f"{label}: {where} answered {status} ({cls.kind}): {cls.hint}" + (f" [{detail}]" if detail else "")
    return cls(scrub(message), status=status, where=where, detail=scrub(detail))


# --- answers, calls and events -------------------------------------------------------------------

@dataclass(frozen=True)
class ProxyReply:
    """A finished answer: the status, the lower-cased headers, the body, the seconds the call took and how many tries
    it needed. It carries no request, so nothing in it can show the token."""

    status: int
    headers: Mapping[str, str]
    content: bytes = field(repr=False)
    elapsed_s: float = 0.0
    attempts: int = 1

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return json.loads(self.content)


@dataclass(frozen=True)
class CallRecord:
    """One call as an observer sees it, for tables like the failure-mode map and for the failed-call rate. where is the
    path below the account; status is 0 when there was no answer; kind is "ok" or the ProxyError kind;
    first_byte_s is when the answer's headers arrived."""

    method: str
    where: str
    status: int
    kind: str
    elapsed_s: float
    first_byte_s: float | None
    attempts: int


@dataclass(frozen=True)
class SseEvent:
    """One server-sent event: its data (joined lines), event name and id when it had them, and t, the seconds after the
    request was sent at which it was complete."""

    data: str
    event: str | None = None
    id: str | None = None
    t: float = 0.0


def parse_sse(lines: Iterable[str], stamp: Callable[[], float] = lambda: 0.0) -> Iterator[SseEvent]:
    """Events out of the lines of a text/event-stream body: `data:` lines join with newlines, a blank line ends an
    event, `:` lines are comments, and an event still open when the lines run out is delivered."""
    data: list[str] = []
    event: str | None = None
    last_id: str | None = None
    for line in lines:
        if line == "":
            if data:
                yield SseEvent("\n".join(data), event, last_id, stamp())
            data, event = [], None
            continue
        if line.startswith(":"):
            continue
        name, sep, value = line.partition(":")
        if sep and value.startswith(" "):
            value = value[1:]
        if name == "data":
            data.append(value)
        elif name == "event":
            event = value
        elif name == "id":
            last_id = value
    if data:
        yield SseEvent("\n".join(data), event, last_id, stamp())


# --- the link -----------------------------------------------------------------------------------

def _port(port: Any) -> int:
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError(f"not a port: {port!r}")
    return port


def _below_account(path: str) -> str:
    """The path with its /user/<account> prefix taken off, so a message never names the account."""
    return re.sub(r"^/user/[^/]+", "", path) or "/"


class StreamHandle:
    """An open answer whose body is still arriving: status and headers are known (a failure status has already been
    raised as its ProxyError), the body is read with iter_bytes() or events(). Close it, or use PodLink.stream()."""

    def __init__(self, link: "PodLink", response: httpx.Response, started: float, method: str, where: str) -> None:
        self.link = link
        self.response = response
        self.status = response.status_code
        self.headers = {name.lower(): value for name, value in response.headers.items()}
        self.method = method
        self.where = where
        self._started = started

    def elapsed(self) -> float:
        """Seconds since the request was sent."""
        return self.link.clock() - self._started

    def iter_bytes(self) -> Iterator[bytes]:
        """The body as it arrives, piece by piece as the transport delivers it (httpx would hold a short piece back to
        fill a chunk of a fixed size, and lose it when the stream breaks)."""
        return self.response.iter_bytes()

    def events(self) -> Iterator[SseEvent]:
        return parse_sse(self.response.iter_lines(), self.elapsed)

    def interrupted(self, exc: BaseException | str, *, partial: str = "") -> StreamInterrupted:
        """The error for a stream that broke after it began, with what had arrived."""
        reason = exc if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
        message = self.link._scrub(f"{self.link.label}: {self.method} {self.where} broke after it began: {reason}")
        error = StreamInterrupted(message, status=self.status, where=self.where)
        error.partial = partial
        return error

    def close(self) -> None:
        self.response.close()


class PodLink:
    """One account's pod as the box reaches it: the address, the token and the rules of the module docstring. hub
    supplies them; transport replaces the network (an httpx transport); observer is called with a CallRecord after each
    call. Use it as a context manager, or close() it."""

    def __init__(self, hub: Hub, *, transport: httpx.BaseTransport | None = None,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.perf_counter,
                 retries: int = len(RETRY_PAUSES_S), connect_timeout_s: float = CONNECT_TIMEOUT_S,
                 read_timeout_s: float = READ_TIMEOUT_S,
                 observer: Callable[[CallRecord], None] | None = None) -> None:
        self.hub = hub
        self.letter = hub.account.letter
        self.label = f"pod {self.letter}"
        self.base_url = hub.base_url
        self.retries = max(0, min(retries, len(RETRY_PAUSES_S)))
        self.sleep = sleep
        self.clock = clock
        self._observer = observer
        self._owns_hub = False
        self._client = httpx.Client(
            base_url=hub.base_url, transport=transport, follow_redirects=False,
            timeout=httpx.Timeout(read_timeout_s, connect=connect_timeout_s),
            headers={"Authorization": f"token {hub.account.token}", "User-Agent": hub.user_agent, "Accept": JSON_TYPE})

    @classmethod
    def for_account(cls, account: Account, *, base_url: str | None = None, transport: httpx.BaseTransport | None = None,
                    environ: Mapping[str, str] | None = None, **kwargs: Any) -> "PodLink":
        """A link with a Hub of its own for the account (closed with the link)."""
        hub = Hub(account, base_url=base_url, transport=transport, environ=environ)
        link = cls(hub, transport=transport, **kwargs)
        link._owns_hub = True
        return link

    def __repr__(self) -> str:
        return f"PodLink(account={self.letter!r}, base_url={self.base_url!r})"

    def __enter__(self) -> "PodLink":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()
        if self._owns_hub:
            self.hub.close()

    # addresses

    def url_for(self, rel: str) -> str:
        """The full address of a path below the account: /proxy/8000/v1/models, /api/contents/x, /files/x."""
        return self.base_url + self.hub.pod_url(rel)

    def proxy_rel(self, port: int, path: str = "") -> str:
        """The path of a model server's address below the account, always the plain /proxy/<port>/ form."""
        return f"/proxy/{_port(port)}/{path.lstrip('/')}"

    def proxy_url(self, port: int, path: str = "") -> str:
        return self.url_for(self.proxy_rel(port, path))

    def server(self, port: int = TEACHER_PORT) -> "ModelServer":
        return ModelServer(self, port)

    # secrets

    def _scrub(self, text: str) -> str:
        """text with the token and the account's name cut out (a hub message can quote its own address)."""
        text = self.hub.scrub(text)
        user = self.hub.account.user
        if len(user) >= MIN_NAME_CHARS:
            text = text.replace(urllib.parse.quote(user, safe=""), "<account>").replace(user, "<account>")
        return text

    def auth_headers(self) -> dict[str, str]:
        """The two headers every request carries."""
        return {"Authorization": f"token {self.hub.account.token}", "User-Agent": self.hub.user_agent}

    # calls

    def _record(self, method: str, where: str, status: int, kind: str, started: float, first_byte: float | None,
                attempts: int) -> None:
        if self._observer is None:
            return
        try:
            self._observer(CallRecord(method, where, status, kind, round(self.clock() - started, 3),
                                      None if first_byte is None else round(first_byte, 3), attempts))
        except Exception:      # an observer must not get in the way of the call it watches
            log.warning("a call observer raised; ignored", exc_info=True)

    def _error(self, method: str, where: str, status: int, headers: Mapping[str, str], content: bytes) -> ProxyError | None:
        text = content.decode("utf-8", errors="replace")
        body: Any = None
        if text.lstrip()[:1] in ("{", "["):
            try:
                body = json.loads(text)
            except ValueError:
                body = None
        return error_for(status, headers=headers, text=text, body=body, where=f"{method} {where}", label=self.label,
                         scrub=self._scrub)

    def request(self, method: str, rel: str, *, json: Any = None, content: bytes | None = None,
                params: Mapping[str, Any] | None = None, headers: Mapping[str, str] | None = None,
                accept: str = JSON_TYPE, timeout: float | None = None, retry: bool | None = None,
                also_ok: Collection[int] = ()) -> ProxyReply:
        """One call whose whole answer is read, for a path below the account. Raises the ProxyError for any status that is
        not a success and not in also_ok. retry says whether a gateway error or a dropped connection may be tried again;
        it defaults to yes for the methods that are safe to repeat and no for POST."""
        method = method.upper()
        again = (method in IDEMPOTENT_METHODS) if retry is None else retry
        pauses = RETRY_PAUSES_S[:self.retries] if again else ()
        path = self.hub.pod_url(rel)
        extra: dict[str, Any] = {} if timeout is None else {"timeout": timeout}
        sent_headers = {"Accept": accept, **(headers or {})}
        attempt = 0
        while True:
            attempt += 1
            started = self.clock()
            error: ProxyError
            try:
                response = self._client.request(method, path, json=json, content=content, params=params,
                                                headers=sent_headers, **extra)
            except httpx.TransportError as exc:
                error = Unreachable(self._scrub(f"{self.label}: {method} {rel}: {type(exc).__name__}: {exc}"),
                                    where=f"{method} {rel}")
                self._record(method, rel, 0, error.kind, started, None, attempt)
            else:
                lowered = {name.lower(): value for name, value in response.headers.items()}
                elapsed = self.clock() - started
                reply = ProxyReply(response.status_code, lowered, response.content, round(elapsed, 3), attempt)
                found = None if reply.status in also_ok else self._error(method, rel, reply.status, lowered, reply.content)
                self._record(method, rel, reply.status, "ok" if found is None else found.kind, started, elapsed, attempt)
                if found is None:
                    return reply
                error = found
            if attempt <= len(pauses) and (isinstance(error, Unreachable) or error.status in RETRY_STATUSES):
                self.sleep(pauses[attempt - 1])
                continue
            raise error

    def open(self, request: httpx.Request, *, also_ok: Collection[int] = ()) -> StreamHandle:
        """Send a request that is already built and hand back the answer with its body unread. A failure status is
        raised as its ProxyError after its body is read; the caller closes the handle."""
        started = self.clock()
        where = _below_account(request.url.path)
        try:
            response = self._client.send(request, stream=True)
        except httpx.TransportError as exc:
            error = Unreachable(self._scrub(f"{self.label}: {request.method} {where}: {type(exc).__name__}: {exc}"),
                                where=f"{request.method} {where}")
            self._record(request.method, where, 0, error.kind, started, None, 1)
            raise error from exc
        headers = {name.lower(): value for name, value in response.headers.items()}
        first_byte = self.clock() - started
        found = None
        if response.status_code not in also_ok and not 200 <= response.status_code < 300:
            try:
                response.read()
            except httpx.TransportError as exc:
                response.close()
                error = Unreachable(self._scrub(f"{self.label}: {request.method} {where}: {type(exc).__name__}: {exc}"),
                                    where=f"{request.method} {where}")
                self._record(request.method, where, response.status_code, error.kind, started, first_byte, 1)
                raise error from exc
            found = self._error(request.method, where, response.status_code, headers, response.content)
        self._record(request.method, where, response.status_code, "ok" if found is None else found.kind, started,
                     first_byte, 1)
        if found is not None:
            response.close()
            raise found
        return StreamHandle(self, response, started, request.method, where)

    @contextlib.contextmanager
    def stream(self, method: str, rel: str, *, json: Any = None, content: bytes | None = None,
               params: Mapping[str, Any] | None = None, headers: Mapping[str, str] | None = None,
               accept: str = EVENT_STREAM, timeout: float | None = None,
               also_ok: Collection[int] = ()) -> Iterator[StreamHandle]:
        """One call whose answer is read as it arrives. The body is never retried: a caller that can resume does it.
        A transport failure while the body is being read comes out as StreamInterrupted."""
        extra: dict[str, Any] = {} if timeout is None else {"timeout": timeout}
        request = self._client.build_request(
            method.upper(), self.hub.pod_url(rel), json=json, content=content, params=params,
            headers={"Accept": accept, "Accept-Encoding": "identity", **(headers or {})}, **extra)
        handle = self.open(request, also_ok=also_ok)
        try:
            yield handle
        except httpx.TransportError as exc:
            raise handle.interrupted(exc) from exc
        finally:
            handle.close()


# --- chat ---------------------------------------------------------------------------------------

@dataclass
class ChatCompletion:
    """A gathered streamed chat completion: the text, any tool calls (OpenAI shape), why it stopped and the server's
    usage figures when it sent them; first_token_s is the seconds from sending the request to the first piece, elapsed_s
    the whole call and chunk_times when each piece that carried text or a tool call arrived (a reply that was held and
    delivered at once shows as many pieces at one moment)."""

    content: str = ""
    reasoning: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = None
    model: str | None = None
    usage: dict[str, Any] | None = None
    first_token_s: float | None = None
    elapsed_s: float = 0.0
    chunk_times: tuple[float, ...] = ()

    def message(self) -> dict[str, Any]:
        """The assistant message of an ordinary (non-streamed) chat completion."""
        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        return message

    def as_openai(self) -> dict[str, Any]:
        """The whole body an ordinary chat completion would have had."""
        body: dict[str, Any] = {"object": "chat.completion", "model": self.model or "",
                                "choices": [{"index": 0, "message": self.message(), "finish_reason": self.finish_reason}]}
        if self.usage:
            body["usage"] = self.usage
        return body


def collect_chat(handle: StreamHandle, *, on_text: Callable[[str], None] | None = None) -> ChatCompletion:
    """Read a chat completion stream to its end. on_text is called with each piece of text as it arrives. A stream
    that ends without [DONE] or a finish reason, or that breaks, raises StreamInterrupted with the text so far."""
    parts: list[str] = []
    reasoning: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    done = ChatCompletion()
    times: list[float] = []
    finished = False
    try:
        for event in handle.events():
            if event.data.strip() == "[DONE]":
                finished = True
                break
            try:
                chunk = json.loads(event.data)
            except ValueError:
                raise handle.interrupted("an event that is not JSON", partial="".join(parts)) from None
            if not isinstance(chunk, dict):
                continue
            if "error" in chunk and not chunk.get("choices"):
                message = _detail(chunk, "") or "the model server reported an error in the stream"
                raise ModelServerError(handle.link._scrub(f"{handle.link.label}: {handle.method} {handle.where} "
                                                          f"streamed an error: {message}"),
                                       status=handle.status, where=f"{handle.method} {handle.where}", detail=message)
            done.model = done.model or chunk.get("model")
            if isinstance(chunk.get("usage"), dict):
                done.usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                    continue
                delta = choice.get("delta") or {}
                carried = False
                piece = delta.get("content")
                if piece:
                    parts.append(piece)
                    carried = True
                    if on_text is not None:
                        on_text(piece)
                thought = delta.get("reasoning_content") or delta.get("reasoning")
                if thought:
                    reasoning.append(thought)
                    carried = True
                for fragment in delta.get("tool_calls") or []:
                    slot = calls.setdefault(int(fragment.get("index", 0)), {
                        "id": None, "type": "function", "function": {"name": "", "arguments": ""}})
                    if fragment.get("id"):
                        slot["id"] = fragment["id"]
                    function = fragment.get("function") or {}
                    slot["function"]["name"] += function.get("name") or ""
                    slot["function"]["arguments"] += function.get("arguments") or ""
                    carried = True
                if carried:
                    times.append(event.t)
                    if done.first_token_s is None:
                        done.first_token_s = event.t
                if choice.get("finish_reason"):
                    done.finish_reason = choice["finish_reason"]
    except httpx.TransportError as exc:
        raise handle.interrupted(exc, partial="".join(parts)) from exc
    if not finished and done.finish_reason is None:
        raise handle.interrupted("the stream ended before [DONE] or a finish reason", partial="".join(parts))
    done.content = "".join(parts)
    done.reasoning = "".join(reasoning)
    done.tool_calls = [{**call, "id": call["id"] or f"call_{index}"} for index, call in sorted(calls.items())]
    done.chunk_times = tuple(times)
    done.elapsed_s = handle.elapsed()
    return done


class ModelServer:
    """A model server on one port of the pod: vLLM or llama-server behind /proxy/<port>/."""

    def __init__(self, link: PodLink, port: int = TEACHER_PORT) -> None:
        self.link = link
        self.port = _port(port)

    def __repr__(self) -> str:
        return f"ModelServer({self.link!r}, port={self.port})"

    def rel(self, path: str = "") -> str:
        return self.link.proxy_rel(self.port, path)

    def url(self, path: str = "") -> str:
        return self.link.proxy_url(self.port, path)

    def get(self, path: str, **kwargs: Any) -> ProxyReply:
        return self.link.request("GET", self.rel(path), **kwargs)

    def post_json(self, path: str, body: Any, **kwargs: Any) -> ProxyReply:
        """A POST whose reply is read whole: keep it short, a reply held for 125 s is cut (use chat() for generation)."""
        return self.link.request("POST", self.rel(path), json=body, **kwargs)

    def models(self) -> list[str]:
        """The model names the server serves (GET /v1/models)."""
        return self.model_names(self.get("v1/models"))

    def model_names(self, reply: ProxyReply) -> list[str]:
        """The names in an answer to GET /v1/models; a ModelServerError when it is not a model list."""
        try:
            return [str(item["id"]) for item in reply.json()["data"]]
        except (ValueError, KeyError, TypeError):
            where = f"GET {self.rel('v1/models')}"
            raise ModelServerError(self.link._scrub(f"{self.link.label}: {where} answered {reply.status} with something "
                                                    "that is not a model list"), status=reply.status, where=where) from None

    def chat(self, messages: Sequence[Mapping[str, Any]], *, model: str | None = None, max_tokens: int | None = None,
             temperature: float | None = None, tools: Sequence[Mapping[str, Any]] | None = None,
             json_schema: Mapping[str, Any] | None = None, thinking: bool | None = None,
             extra: Mapping[str, Any] | None = None, on_text: Callable[[str], None] | None = None,
             timeout: float | None = None) -> ChatCompletion:
        """A chat completion, streamed and gathered (see the module docstring). messages are OpenAI chat messages.
        thinking=False asks a Qwen3 model not to spend its budget reasoning (chat_template_kwargs.enable_thinking);
        extra is merged into the request body last. (Measured on the R9700 teacher of 2026-10-10: 0.55 s to a first token
        and 7.7 tokens a second; the Strix Halo pods that replace it have not been measured.)"""
        body: dict[str, Any] = {"messages": list(messages), "stream": True, "stream_options": {"include_usage": True}}
        if model is not None:
            body["model"] = model
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
        if tools:
            body["tools"] = list(tools)
        if json_schema:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "reply", "schema": dict(json_schema), "strict": True}}
        if thinking is not None:
            body["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
        if extra:
            body.update(extra)
        with self.link.stream("POST", self.rel("v1/chat/completions"), json=body, timeout=timeout) as handle:
            return collect_chat(handle, on_text=on_text)


# --- the stock runtime over the proxy -------------------------------------------------------------

DATA_URL = re.compile(r"data:[a-z0-9.+-]+/[a-z0-9.+-]+(;[a-z0-9=.+-]+)*;base64,", re.IGNORECASE)


def text_only_problem(body: Any) -> str | None:
    """Why a request body is not text only, or None when it is: every message's content must be a string (no list of
    parts, so no image_url, input_audio or video part), and nothing in the body may be a data URL."""
    if not isinstance(body, Mapping):
        return "the body is not a JSON object"
    messages = body.get("messages")
    if messages is not None:
        if not isinstance(messages, list):
            return "messages is not a list"
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                return f"message {index} is not an object"
            content = message.get("content")
            if content is not None and not isinstance(content, str):
                return f"message {index} has content that is not a string (a picture or a file part)"
    if DATA_URL.search(json.dumps(body, ensure_ascii=False)):
        return "the body holds a data URL"
    return None


class ProxyTransport(httpx.BaseTransport):
    """An httpx transport that sends what the stock LlamaServerRuntime asks for through a PodLink. It adds the token and
    the browser User-Agent, refuses to send them to any other host, sends the exact Accept header on a stream, raises
    ProxyErrors for failure statuses, fills in `model` when model_name is given and the request names none, and (with
    upgrade) sends a chat completion that was not asked to stream as a stream and hands back the gathered reply as the
    ordinary JSON, so that no reply can run into Cloudflare's 125 second cut. extra is set on every chat completion
    body, over the runtime's own values; text_only refuses (ValueError) any request that is not plain text before it
    is sent."""

    def __init__(self, link: PodLink, *, model_name: str | None = None, upgrade: bool = True,
                 extra: Mapping[str, Any] | None = None, text_only: bool = False) -> None:
        self.link = link
        self.model_name = model_name
        self.upgrade = upgrade
        self.extra = dict(extra or {})
        self.text_only = text_only
        self._destination = self._where(httpx.URL(link.base_url))

    @staticmethod
    def _where(url: httpx.URL) -> tuple[str, int]:
        return url.host, url.port or {"https": 443, "http": 80}.get(url.scheme, 0)

    def close(self) -> None:
        pass

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if self._where(request.url) != self._destination:
            raise ValueError("this transport carries the hub token and talks only to the hub it was built for")
        request.read()
        body = self._json_body(request)
        is_chat = (request.method == "POST" and body is not None
                   and request.url.path.rstrip("/").endswith("/chat/completions"))
        if is_chat and body is not None:
            if self.model_name and not body.get("model"):
                body["model"] = self.model_name
            body.update(self.extra)
        if self.text_only and request.content:
            problem = text_only_problem(body) if body is not None else "the body is not JSON"
            if problem is not None:
                raise ValueError(f"text only: {problem}; nothing was sent")
        if is_chat and body is not None:
            if not body.get("stream") and self.upgrade:
                return self._gathered(request, body)
            content = json.dumps(body).encode("utf-8")
            return self._forward(request, content, accept=EVENT_STREAM if body.get("stream") else None).response
        return self._forward(request, request.content, accept=None).response

    @staticmethod
    def _json_body(request: httpx.Request) -> dict[str, Any] | None:
        if not request.content or "json" not in request.headers.get("content-type", ""):
            return None
        try:
            body = json.loads(request.content)
        except ValueError:
            return None
        return body if isinstance(body, dict) else None

    def _forward(self, request: httpx.Request, content: bytes, *, accept: str | None) -> StreamHandle:
        headers = httpx.Headers(request.headers)
        for name in ("host", "content-length", "authorization"):
            headers.pop(name, None)
        headers.update(self.link.auth_headers())
        if accept is not None:
            headers["Accept"] = accept
            headers["Accept-Encoding"] = "identity"
        rebuilt = httpx.Request(request.method, request.url, headers=headers, content=content,
                                extensions=request.extensions)
        return self.link.open(rebuilt)

    def _gathered(self, request: httpx.Request, body: dict[str, Any]) -> httpx.Response:
        body = {**body, "stream": True}
        body.setdefault("stream_options", {"include_usage": True})
        handle = self._forward(request, json.dumps(body).encode("utf-8"), accept=EVENT_STREAM)
        try:
            done = collect_chat(handle)
        finally:
            handle.close()
        return httpx.Response(200, json=done.as_openai(), request=request)


def proxied_runtime(model: Any, link: PodLink, port: int = TEACHER_PORT, *, model_name: str | None = None,
                    upgrade: bool = True, extra: Mapping[str, Any] | None = None, text_only: bool = False,
                    **kwargs: Any) -> Any:
    """The stock runtime.llama.LlamaServerRuntime over the proxy to the model server on `port`. model is its
    ModelVersion; model_name is the name the server serves it under (the teacher is called "teacher"); extra and
    text_only are the ProxyTransport's; kwargs go to LlamaServerRuntime (detector, embedders, timeout_s ...), which
    already takes the transport used here."""
    from runtime.llama import LlamaServerRuntime

    transport = ProxyTransport(link, model_name=model_name, upgrade=upgrade, extra=extra, text_only=text_only)
    return LlamaServerRuntime(model, base_url=link.proxy_url(port), transport=transport, **kwargs)


# --- the command line ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m lab.proxy", description=(__doc__ or "").split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    probe = commands.add_parser("probe", help="ask the model server what it serves, and whether it is healthy")
    chat = commands.add_parser("chat", help="stream one chat completion and time it")
    for command in (probe, chat):
        command.add_argument("--account", required=True, help="the account letter, e.g. C")
        command.add_argument("--port", type=int, default=TEACHER_PORT)
    probe.add_argument("--wrong-token", action="store_true",
                       help="send a made-up token instead of the account's, to see the failure the hub answers with")
    chat.add_argument("--prompt", required=True)
    chat.add_argument("--max-tokens", type=int, default=64)
    chat.add_argument("--model", default=None)
    chat.add_argument("--no-thinking", action="store_true")
    return parser


def _emit(out: TextIO, payload: Mapping[str, Any]) -> None:
    out.write(json.dumps(payload, sort_keys=True) + "\n")
    out.flush()


def _failure(exc: ProxyError) -> dict[str, Any]:
    return {"ok": False, "kind": exc.kind, "status": exc.status, "retryable": exc.retryable, "error": str(exc)}


def main(argv: Sequence[str] | None = None, *, environ: Mapping[str, str] | None = None,
         env_files: Sequence[Any] | None = None, transport: httpx.BaseTransport | None = None,
         sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.perf_counter,
         stdout: TextIO | None = None, stderr: TextIO | None = None) -> int:
    """Exit 0 when the call worked, 1 when it did not, 2 for bad arguments and for missing keys. One JSON object on
    stdout; a chat also writes the text to stderr as it arrives. No token and no account name is printed."""
    out, err = stdout or sys.stdout, stderr or sys.stderr
    args = build_parser().parse_args(argv)
    if not re.fullmatch(r"[A-Za-z]", args.account):
        _emit(out, {"ok": False, "kind": "usage", "error": "--account is one letter"})
        return 2
    letter = args.account.upper()
    try:
        account = load_accounts([letter], environ=environ, env_files=env_files)[letter]
    except MissingCredentials as exc:
        _emit(out, {"ok": False, "kind": "missing_keys", "error": str(exc), "missing": exc.missing})
        return 2
    if getattr(args, "wrong_token", False):
        account = dataclasses.replace(account, token="0" * len(account.token))
    try:
        with PodLink.for_account(account, transport=transport, environ=environ, sleep=sleep, clock=clock) as link:
            server = link.server(args.port)
            if args.command == "probe":
                reply = server.get("v1/models")
                health = server.get("health", also_ok=(503,)).status
                _emit(out, {"ok": True, "models": server.model_names(reply), "models_s": reply.elapsed_s,
                            "health_status": health, "port": args.port,
                            "edge": {"server": reply.headers.get("server"), "cf_ray": reply.headers.get("cf-ray")}})
                return 0
            done = server.chat([{"role": "user", "content": args.prompt}], model=args.model, max_tokens=args.max_tokens,
                               thinking=False if args.no_thinking else None, on_text=lambda piece: (err.write(piece), err.flush()))
            err.write("\n")
            _emit(out, {"ok": True, "text": done.content, "first_token_s": done.first_token_s,
                        "elapsed_s": round(done.elapsed_s, 3), "pieces": len(done.chunk_times),
                        "finish_reason": done.finish_reason, "usage": done.usage, "port": args.port})
            return 0
    except ProxyError as exc:
        _emit(out, _failure(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
