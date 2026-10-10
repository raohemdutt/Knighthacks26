"""A stand-in for the pod's side of the hub proxy: a real package server behind a fake hub, for tests and rehearsals.

    clock = FakeClock()
    hub = FakeHub(clock)
    account = hub.add_account("B", user="acct-b", token="...")
    hub.start_server("B")
    server = package_server.make_server(packages_dir)           # the real file server, on 127.0.0.1
    threading.Thread(target=server.serve_forever, daemon=True).start()
    proxy = FakePodProxy(hub, account, server.server_address[1], response_limit=8192)
    client = PackageClient(account, proxy.port, base_url="https://hub.test", transport=proxy.transport, sleep=clock.sleep)

A request to /user/<account>/proxy/<port>/... goes through the fake hub first (lab/hub_fake.py), which decides what the
hub decides: a wrong token is a 302 to the login page, a stopped pod is a 424, a request counts as activity, and a
failure queued with hub.queue("GET", "/proxy/8300/", 502) is answered here. A request the hub lets through is sent to
the real package server over a real socket, with every header the box sent, the hub token included: that is what
jupyter-server-proxy 4.6.0 does (proxy_request_headers copies self.request.headers), and it is what makes a test that
reads the server's log mean something.

What the proxy does with the answer, as jupyter-server-proxy 4.6.0 does it for a request that is not
`Accept: text/event-stream` (handlers.py _proxy_buffered): it fetches the whole answer before sending any of it, and
its HTTP client refuses a response body larger than 100 MiB (Tornado's default max_body_size), which the proxy turns
into an error status; here that status is `refusal_status` (599 by default, the code jupyter-server-proxy raises for
a client error) and the size is `response_limit`, so a test can use a few KB to stand for the 100 MiB. Nothing here
was recorded from the live proxy: the status it really answers with is what `python -m lab.package_pull probe`
measures on a pod. Cloudflare and the hub's own proxy are not modeled.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import re
import urllib.parse
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import httpx

from lab import package_server
from lab.hub import Account
from lab.hub_fake import FakeHub

__all__ = ["FakePodProxy", "Seen", "PROXY_LIMIT", "write_package"]

PROXY_LIMIT = 100 * 1024 * 1024
STRIPPED_REQUEST = frozenset({"host", "connection", "keep-alive", "proxy-connection", "transfer-encoding", "te", "trailer",
                              "upgrade", "proxy-authorization", "proxy-authenticate", "content-length"})
STRIPPED_ANSWER = frozenset({"connection", "keep-alive", "transfer-encoding", "content-length", "content-encoding",
                             "server", "date"})
NOTHING_LISTENING = ("<html><head><title>500 : Internal Server Error</title></head>"
                     "<body><h1>500 : Internal Server Error</h1></body></html>")


def write_package(root: str | Path, version: str, files: Mapping[str, bytes], *, gguf: str = "model.gguf",
                  mmproj: str | None = "mmproj-model.gguf", base: str = "Qwen/Qwen3.5-4B",
                  checksums: Mapping[str, str] | None = None) -> dict[str, Any]:
    """A package folder as the packaging step leaves it on the pod: root/<version>/ with the given files and a
    manifest.json (ModelManifest) whose sha256 are the real ones, or `checksums` where a test plants a wrong one.
    Returns the manifest."""
    folder = Path(root) / version
    folder.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (folder / name).write_bytes(data)
    manifest = {"version": version, "base": base, "task": None, "gguf": gguf, "mmproj": mmproj,
                "sha256": {name: hashlib.sha256(files[name]).hexdigest() for name in (gguf, mmproj) if name},
                "scores": {}, "created": dt.datetime(2026, 10, 10, 12, 0, tzinfo=dt.timezone.utc).isoformat()}
    manifest["sha256"].update(checksums or {})
    (folder / package_server.MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


@dataclasses.dataclass
class Seen:
    """One request that reached the proxy's port, as the package server was asked for it."""

    method: str
    path: str                        # the address behind /proxy/<port>, percent-encoded, without its query
    query: str
    headers: dict[str, str]          # what the box sent, lower-case names: the token included, as the proxy forwards it
    status: int | None = None
    body_bytes: int = 0              # bytes of the answer's body that were sent on

    @property
    def range(self) -> str | None:
        return self.headers.get("range")


@dataclasses.dataclass
class _Fault:
    status: int | None               # an error answer; None for a body that is cut off
    remaining: int
    contains: str
    text: str = ""
    cut_after: int = 0


class _CutStream(httpx.SyncByteStream):
    """A body that stops after some bytes with a reset connection, as a proxy that loses its client does."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def __iter__(self) -> Iterator[bytes]:
        if self._data:
            yield self._data
        raise httpx.ReadError("connection reset by peer")


class FakePodProxy:
    """The proxy route of one pod port, in front of a package server listening on 127.0.0.1:backend_port."""

    def __init__(self, hub: FakeHub, account: Account, backend_port: int, *, port: int = package_server.DEFAULT_PORT,
                 response_limit: int = PROXY_LIMIT, refusal_status: int = 599) -> None:
        self.hub = hub
        self.account = account
        self.port = port
        self.response_limit = response_limit
        self.refusal_status = refusal_status
        self.listening = True                 # False: the port has nothing on it, and the proxy answers 500
        self.seen: list[Seen] = []
        self._faults: list[_Fault] = []
        self._user = urllib.parse.quote(account.user, safe="")
        self._route = re.compile(rf"^/user/{re.escape(self._user)}/proxy/{port}(/.*)?$")
        hub.serve_model(account.letter, port=port)        # the hub treats the port as listening and counts requests to it
        self._backend = httpx.Client(base_url=f"http://127.0.0.1:{backend_port}", timeout=30.0)
        self.transport = httpx.MockTransport(self._handle)

    def close(self) -> None:
        self._backend.close()

    # --- faults a test can queue (the hub's own go through hub.queue) -----------------------------------------------

    def fail(self, status: int, *, times: int = 1, contains: str = "", text: str = "") -> None:
        """Answer the next `times` requests whose address contains `contains` with this status, once the hub has let
        them through and before the package server is asked."""
        self._faults.append(_Fault(status, times, contains, text))

    def cut_body(self, after_bytes: int, *, times: int = 1, contains: str = "") -> None:
        """Send the next `times` answers whose address contains `contains` with their status and headers and only the
        first `after_bytes` bytes of their body, then drop the connection."""
        self._faults.append(_Fault(None, times, contains, cut_after=after_bytes))

    def requests(self, contains: str = "") -> list[Seen]:
        return [item for item in self.seen if contains in item.path]

    # --- the route ----------------------------------------------------------------------------------------------

    def _handle(self, request: httpx.Request) -> httpx.Response:
        raw_path, _, query = request.url.raw_path.decode("ascii").partition("?")
        found = self._route.match(raw_path)
        if found is None:
            return self.hub.transport.handle_request(request)
        rest = found.group(1) or "/"
        gate = httpx.Request("GET", request.url.copy_with(path=f"/user/{self._user}/proxy/{self.port}/health", query=b""),
                             headers=request.headers)
        gated = self.hub.transport.handle_request(gate)
        gated.read()
        if gated.status_code != 200:
            return gated                       # the hub's answer: a login redirect, 424, a queued 502 ...
        seen = Seen(request.method, rest, query, {name.lower(): value for name, value in request.headers.items()})
        self.seen.append(seen)
        response = self._answer(request, rest, query, seen, gated.headers.get("date"))
        seen.status = response.status_code
        return response

    def _answer(self, request: httpx.Request, rest: str, query: str, seen: Seen, date: str | None) -> httpx.Response:
        if not self.listening:
            return httpx.Response(500, text=NOTHING_LISTENING, headers={"content-type": "text/html"})
        fault = self._take(rest)
        if fault is not None and fault.status is not None:
            return httpx.Response(fault.status, text=fault.text or f"fault {fault.status}", headers={"content-type": "text/plain"})
        sent = {name: value for name, value in request.headers.items() if name.lower() not in STRIPPED_REQUEST}
        target = rest + (f"?{query}" if query else "")
        with self._backend.stream(request.method, target, headers=sent) as upstream:
            declared = upstream.headers.get("content-length", "")
            if declared.isdigit() and int(declared) > self.response_limit:
                return self._refuse(int(declared))
            body = upstream.read()
            status, headers = upstream.status_code, upstream.headers
        if len(body) > self.response_limit:
            return self._refuse(len(body))
        kept: dict[str, str] = {name: value for name, value in headers.items() if name.lower() not in STRIPPED_ANSWER}
        if date:
            kept["date"] = date
        seen.body_bytes = len(body)
        if fault is not None:
            seen.body_bytes = min(len(body), fault.cut_after)
            return httpx.Response(status, headers={**kept, "content-length": str(len(body))},
                                  stream=_CutStream(body[:fault.cut_after]))
        return httpx.Response(status, headers=kept, content=body)

    def _take(self, rest: str) -> _Fault | None:
        for fault in self._faults:
            if fault.remaining > 0 and fault.contains in rest:
                fault.remaining -= 1
                return fault
        return None

    def _refuse(self, size: int) -> httpx.Response:
        """What the proxy does with an answer its client will not take: an error status and a short text."""
        return httpx.Response(self.refusal_status, headers={"content-type": "text/html"},
                              text=f"<html><body>HTTP {self.refusal_status}: response of {size} bytes is over the limit "
                                   f"of {self.response_limit}</body></html>")
