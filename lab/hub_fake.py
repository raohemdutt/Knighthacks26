"""A stand-in for the AUP hub that tests and rehearsals can drive: no network, no real pods, no real time.

    clock = FakeClock("2026-10-10T09:27:00Z")
    fake = FakeHub(clock)
    account = fake.add_account("A", user="acct-a", token="...", balance=59960)
    controller = Controller({"A": account}, transport=fake.transport, sleep=clock.sleep, clock=clock.time)

The fake is an httpx transport, so the real client code runs against it unchanged: the same URLs, the same
headers, the same JSON. Time moves only when something calls clock.sleep(), which is how a test lets 130
minutes pass in a millisecond.

What it answers, and where the shapes come from. The hub user model (`GET /hub/api/users/<user>`), the quota
answer (`GET /hub/api/quota/me`) and its transactions (`GET /hub/admin/api/quota/<user>`), the launch (`POST
/hub/api/users/<user>/server`: 202 and a pending spawn, 400 "is already running" or "is pending stop", 403 for
a quota refusal) and the stop (`DELETE`: 204, or 202 while a stop is pending) are the shapes the hub returned to
the read-only probes and hand calls of 2026-10-10 (lab/refine/research/plumbing/measurements-2026-10-10.md,
M3 to M5, M9, M14) and the ones its source defines (JupyterHub 5.4.4 apihandlers/users.py; AUP
runtime/hub/core/handlers.py and quota/manager.py at 9b4af19). Not recorded from a live call, and so the least
certain: the body of a request to a stopped pod's address (424 here) and what the hub does with a launch the
quota refuses (here the POST itself answers 403).

What it does over time, as the hub does: a launch is pending for `spawn_s` seconds and then ready; a stop is
pending for `stop_s` seconds; a pod runs until `started + runtime_minutes`; the idle culler checks every
`cull_every_s` seconds from the start and stops a pod whose last activity is `cull_after_s` old; a session is
charged when it ends, whole elapsed minutes times the rate, and never takes the balance below zero. Requests
through the pod's address count as activity and hub API reads do not. The hub's clock runs
`hub_clock_ahead_s` ahead of the caller's and says so in the Date header of every answer, which is how the
hub's clock differed on the day of the probes.

Like the real front end it answers 403 to a request whose User-Agent is not a browser's, and like the hub it
answers a wrong token with 403 on its API and a redirect to a login page on a pod's address.

Two hubs, as with DUB and SJC: one FakeHub per hub, and network() makes one transport out of them, sending each
call to the hub whose host it names (a host no hub has is a dropped connection). An account added with a quota that
is off (`quota_enabled=False`) answers as the SJC hub does: balance 0, unlimited, enabled false, and a launch is
never refused for credits.

    transport = network({"dub.test": dub, "sjc.test": sjc})
    c = sjc.add_account("C", user="acct-c", token="...", quota_enabled=False, url="https://sjc.test")

The web side of the hub, for the spawn form. Recorded live on SJC on 2026-10-10: `GET /hub/native/login` sets the
`_xsrf` cookie and carries the same token in a hidden input; `POST /hub/native/login` (username, password, _xsrf)
answers 302 with the cookies `jupyterhub-hub-login` and `jupyterhub-session-id` when the hub accepts the account
and 200 with the login page and a `login_error` message when it does not; `GET /hub/spawn/<user>` answers 200 with
a session and 302 to the login page without one; `POST /hub/spawn/<user>?_xsrf=<cookie>` with the spawn-ui fields
answers 302 to `/hub/spawn-pending/<user>` when the spawn starts. An account added with a `password` can log in; the
web calls carry cookies and no token. The pod a form launch starts runs the real image (`AUP_IMAGE`); as on SJC,
`rest_launch_ignored = True` makes a REST launch answer as usual, echo its options, and start a pod that runs the
stock singleuser image (`STOCK_IMAGE`) with no GPU. `pod_image()` and `pod_has_gpu()` say what a pod runs, which
is what a terminal on it would tell. `omit_xsrf = True` serves the login page without any `_xsrf` token.
"""
from __future__ import annotations

import datetime as dt
import email.utils
import heapq
import http.cookies
import itertools
import json
import math
import re
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from lab.hub import Account

__all__ = ["FakeClock", "FakeHub", "Call", "network", "STOCK_IMAGE", "AUP_IMAGE"]

STOCK_IMAGE = "quay.io/jupyterhub/singleuser:latest"
AUP_IMAGE = "ghcr.io/amdresearch/auplc-base:latest"
SESSION_COOKIE = "jupyterhub-hub-login"

UTC = dt.timezone.utc
HUB_VERSION = "5.4.4"
DEFAULT_RATES = {"cpu": 1, "r9700": 4}
CLOUDFLARE_PAGE = "<html><head><title>Access denied</title></head><body>Error 1010: browser signature banned</body></html>"


def _epoch(text: str) -> float:
    return dt.datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def _iso(epoch: float) -> str:
    return dt.datetime.fromtimestamp(epoch, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class FakeClock:
    """Seconds since the epoch that move only when someone sleeps. A callback set with call_later() runs at its
    moment, in the middle of the sleep that carries the clock past it."""

    def __init__(self, start: str | float = "2026-10-10T09:27:00Z") -> None:
        self.start = _epoch(start) if isinstance(start, str) else float(start)
        self.now = self.start
        self._timers: list[tuple[float, int, Callable[[], None]]] = []
        self._order = itertools.count()

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        target = self.now + max(0.0, float(seconds))
        while self._timers and self._timers[0][0] <= target:
            when, _, callback = heapq.heappop(self._timers)
            self.now = max(self.now, when)
            callback()
        self.now = target

    def call_later(self, seconds: float, callback: Callable[[], None]) -> None:
        heapq.heappush(self._timers, (self.now + seconds, next(self._order), callback))

    @property
    def elapsed(self) -> float:
        return self.now - self.start


@dataclass
class Call:
    """One request the fake received, as the hub would have seen it. t is the seconds since the clock started."""

    method: str
    path: str
    query: str
    host: str
    headers: dict[str, str]
    json: Any
    t: float
    account: str | None = None
    status: int | None = None
    form: dict[str, str] | None = None     # the fields of a form-encoded body (the web login and spawn)


@dataclass
class _Server:
    started: float                     # hub clock
    runtime_minutes: int
    options: dict[str, Any]
    last_activity: float
    ready_at: float | None             # the hub clock at which the spawn ends; None once it has
    stop_done_at: float | None = None  # the hub clock at which a pending stop ends
    failed: bool = False               # listed by the hub, but its spawn failed: not ready, nothing pending
    image: str = AUP_IMAGE             # what the pod runs; a REST launch the hub ignores starts STOCK_IMAGE


@dataclass
class _Account:
    letter: str
    user: str
    token: str
    balance: int
    rate: int
    unlimited: bool
    quota_enabled: bool
    password: str | None = None
    server: _Server | None = None
    transactions: list[dict[str, Any]] = field(default_factory=list)
    sessions: int = 0
    terminals: int = 0
    user_activity: float = 0.0
    models: dict[int, tuple[tuple[str, ...], bool]] = field(default_factory=dict)


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


FAKE_ROLES = {"A": "pod-a", "B": "pod-b", "C": "pod-a", "D": "pod-b"}


class FakeHub:
    """The hub, as the callers' code sees it. See the module docstring."""

    def __init__(self, clock: FakeClock, *, spawn_s: float = 15.0, stop_s: float = 0.0,
                 hub_clock_ahead_s: float = 46.0, cull_after_s: float = 3600.0, cull_every_s: float = 600.0) -> None:
        self.clock = clock
        self.spawn_s = spawn_s
        self.stop_s = stop_s
        self.spawn_fails = False
        self.rest_launch_ignored = False      # as on SJC: a REST launch is accepted, but its pod is the stock image
        self.omit_xsrf = False                # serve the login page without any _xsrf token
        self.hub_clock_ahead_s = hub_clock_ahead_s
        self.cull_after_s = cull_after_s
        self.cull_every_s = cull_every_s
        self.calls: list[Call] = []
        self.files: dict[tuple[str, str], str] = {}
        self._accounts: dict[str, _Account] = {}
        self._canned: list[_Canned] = []
        self._sessions: dict[str, str] = {}   # a web session cookie to the letter of the account that logged in
        self._tokens = itertools.count(1)
        self.transport = httpx.MockTransport(self._handle)

    # --- scenario ---------------------------------------------------------------------------

    def add_account(self, letter: str, *, user: str, token: str, balance: int = 59960, rate: int = 4,
                    unlimited: bool = False, quota_enabled: bool = True, url: str | None = None,
                    password: str | None = None, role: str | None = None) -> Account:
        """A hub account, returned in the form the callers take. url is the address it is given as its own
        (AUP_HUB_URL_<letter>); the fake itself answers on whatever address it is reached at. password lets the
        account log in on the web side (the spawn form); without one the fake refuses every login as that user.
        role is the pod role the account fills (LAB_POD_ROLES in real use); unset, the letter's role in FAKE_ROLES
        (A and C the teacher's pod-a, B and D the trainer's pod-b), and none for any other letter."""
        self._accounts[letter] = _Account(letter, user, token, balance, rate, unlimited, quota_enabled, password,
                                          user_activity=self._hub_now())
        return Account(letter, user, token, url, password, role=role or FAKE_ROLES.get(letter))

    def start_server(self, letter: str, *, runtime_minutes: int = 600, started_ago_s: float = 0.0, idle_s: float = 0.0,
                     ready: bool = True, accelerator: str = "r9700", image: str = AUP_IMAGE) -> None:
        """A pod that is already there: ready and running for started_ago_s, or still launching (ready=False)."""
        account = self._accounts[letter]
        now = self._hub_now()
        options = {"runtime_minutes": runtime_minutes, "resource_type": "gpu", "gpu_selection": accelerator}
        account.sessions += 1
        if ready:
            account.server = _Server(now - started_ago_s, runtime_minutes, options, now - idle_s, None, image=image)
        else:
            account.server = _Server(now, runtime_minutes, options, now, now + self.spawn_s, image=image)

    def start_failed_server(self, letter: str) -> None:
        """A server the hub still lists after its spawn failed: not ready and with nothing pending."""
        account = self._accounts[letter]
        now = self._hub_now()
        options = {"runtime_minutes": 600, "resource_type": "gpu", "gpu_selection": "r9700"}
        account.sessions += 1
        account.server = _Server(now, 600, options, now, None, failed=True)

    def drop_server(self, letter: str) -> None:
        """The pod goes away on its own, as when it is culled or its runtime ends: charged, with no stop issued."""
        account = self._accounts[letter]
        self._end_session(account, self._hub_now())

    def schedule(self, after_s: float, callback: Callable[[], None]) -> None:
        """Run callback when the clock reaches the moment after_s seconds from now."""
        self.clock.call_later(after_s, callback)

    def queue(self, method: str, contains: str, status: int, *, body: Any = None, text: str | None = None,
              times: int = 1, apply: bool = False) -> None:
        """Answer the next `times` requests of that method whose path contains `contains` with this status and body.
        With apply=True the request is first carried out as usual and only its answer is replaced, which is how an
        answer is lost on its way back."""
        self._canned.append(_Canned(method.upper(), contains, status, body, text, times, apply))

    def queue_dropped(self, method: str, contains: str, *, times: int = 1) -> None:
        """Drop the connection of the next `times` requests of that method whose path contains `contains`: the
        caller sees a transport error, and the error text names the address, as a real one does."""
        error = httpx.ConnectError(f"connection to https://hub.test{contains} was reset")
        self._canned.append(_Canned(method.upper(), contains, 0, None, None, times, False, error))

    def set_balance(self, letter: str, amount: int) -> None:
        self._accounts[letter].balance = amount

    def set_quota_enabled(self, letter: str, enabled: bool) -> None:
        self._accounts[letter].quota_enabled = enabled

    def serve_model(self, letter: str, *, port: int = 8000, models: Sequence[str] = ("teacher",),
                    healthy: bool = True) -> None:
        """A model server listening on a pod port, as vLLM answers /v1/models and /health."""
        self._accounts[letter].models[port] = (tuple(models), healthy)

    # --- what a test looks at ---------------------------------------------------------------

    def running(self, letter: str) -> bool:
        """Whether the account has a server (launching, ready or stopping)."""
        account = self._accounts[letter]
        self._advance(account)
        return account.server is not None

    def balance(self, letter: str) -> int:
        account = self._accounts[letter]
        self._advance(account)
        return account.balance

    def transactions(self, letter: str) -> list[dict[str, Any]]:
        """The account's transactions, newest first, as the hub lists them."""
        account = self._accounts[letter]
        self._advance(account)
        return [dict(item) for item in reversed(account.transactions)]

    def last_activity(self, letter: str) -> float | None:
        """The server's last activity on the hub's clock; None when there is no server."""
        account = self._accounts[letter]
        self._advance(account)
        return None if account.server is None else account.server.last_activity

    def pod_image(self, letter: str) -> str | None:
        """The image the account's pod runs (what its JUPYTER_IMAGE_SPEC says); None when there is no server."""
        account = self._accounts[letter]
        self._advance(account)
        return None if account.server is None else account.server.image

    def pod_has_gpu(self, letter: str) -> bool | None:
        """Whether the pod has a GPU device (/dev/kfd): the stock image has none; None when there is no server."""
        image = self.pod_image(letter)
        return None if image is None else image != STOCK_IMAGE

    def calls_matching(self, method: str | None = None, contains: str = "") -> list[Call]:
        return [call for call in self.calls
                if (method is None or call.method == method.upper()) and contains in call.path]

    # --- time -------------------------------------------------------------------------------

    def _hub_now(self) -> float:
        return self.clock.time() + self.hub_clock_ahead_s

    def _advance(self, account: _Account) -> None:
        """Carry out, in order, everything the hub would have done to this account's server up to now."""
        now = self._hub_now()
        while account.server is not None:
            event = self._next_event(account, now)
            if event is None:
                return
            when, kind = event
            server = account.server
            if kind == "spawned":
                if self.spawn_fails:
                    account.server = None
                    self._charge(account, server, when)
                else:
                    server.ready_at = None
            elif kind == "stopped":
                account.server = None
            else:                                  # expired or culled
                self._end_session(account, when)

    def _next_event(self, account: _Account, now: float) -> tuple[float, str] | None:
        server = account.server
        assert server is not None
        found: list[tuple[float, str]] = []
        if server.failed:
            return None
        if server.ready_at is not None and server.ready_at <= now:
            found.append((server.ready_at, "spawned"))
        if server.stop_done_at is not None:
            if server.stop_done_at <= now:
                found.append((server.stop_done_at, "stopped"))
        else:
            expires = server.started + server.runtime_minutes * 60
            if expires <= now:
                found.append((expires, "expired"))
            if server.ready_at is None and self.cull_after_s:
                ticks = max(1, math.ceil((server.last_activity + self.cull_after_s - server.started) / self.cull_every_s))
                culled = server.started + ticks * self.cull_every_s
                if culled <= now:
                    found.append((culled, "culled"))
        return min(found) if found else None

    def _end_session(self, account: _Account, when: float) -> None:
        server = account.server
        if server is not None:
            account.server = None
            self._charge(account, server, when)

    def _charge(self, account: _Account, server: _Server, when: float) -> None:
        """Settle a session the way the hub does when it ends: whole minutes at the rate, never below zero."""
        minutes = int((when - server.started) / 60)
        amount = minutes * account.rate
        if not account.quota_enabled or account.unlimited or amount <= 0:
            return
        before = account.balance
        account.balance = max(0, before - amount)
        accelerator = server.options.get("gpu_selection", "r9700")
        account.transactions.append({
            "id": len(account.transactions) + 1, "username": account.user, "amount": -amount,
            "transaction_type": "usage", "resource_type": server.options.get("resource_type", "gpu"),
            "description": f"Session {account.sessions}: {minutes} min @ {accelerator} ({account.rate}/min)",
            "balance_before": before, "balance_after": account.balance, "created_at": _iso(when), "created_by": None})

    # --- the requests -----------------------------------------------------------------------

    def _handle(self, request: httpx.Request) -> httpx.Response:
        headers = {k.lower(): v for k, v in request.headers.items()}
        kind = headers.get("content-type", "")
        call = Call(method=request.method, path=request.url.path, query=request.url.query.decode(),
                    host=request.url.host, headers=headers,
                    json=json.loads(request.content) if request.content and "json" in kind else None,
                    t=self.clock.elapsed,
                    form=dict(urllib.parse.parse_qsl(request.content.decode(), keep_blank_values=True))
                    if request.content and "x-www-form-urlencoded" in kind else None)
        self.calls.append(call)
        response = self._respond(request, call)
        call.status = response.status_code
        response.headers["date"] = email.utils.format_datetime(dt.datetime.fromtimestamp(self._hub_now(), tz=UTC), usegmt=True)
        return response

    def _respond(self, request: httpx.Request, call: Call) -> httpx.Response:
        if not call.headers.get("user-agent", "").startswith("Mozilla/5.0"):
            return httpx.Response(403, text=CLOUDFLARE_PAGE, headers={"content-type": "text/html"})
        canned = self._take_canned(call)
        if canned is not None and canned.raises is not None:
            raise canned.raises
        if canned is not None and not canned.apply:
            return self._canned_response(canned)
        response = self._route(request, call)
        return self._canned_response(canned) if canned is not None else response

    def _take_canned(self, call: Call) -> _Canned | None:
        for entry in self._canned:
            if entry.remaining > 0 and entry.method == call.method and entry.contains in call.path:
                entry.remaining -= 1
                return entry
        return None

    @staticmethod
    def _canned_response(entry: _Canned) -> httpx.Response:
        if entry.body is not None:
            return httpx.Response(entry.status, json=entry.body)
        return httpx.Response(entry.status, text=entry.text or "", headers={"content-type": "text/plain"})

    def _route(self, request: httpx.Request, call: Call) -> httpx.Response:
        path = call.path
        if path.rstrip("/") == "/hub/api":
            return httpx.Response(200, json={"version": HUB_VERSION})
        if path == "/hub/native/login":
            return self._login(call)
        spawn_page = re.match(r"^/hub/spawn/([^/]+)$", path)
        if spawn_page:
            return self._spawn_page(call, urllib.parse.unquote(spawn_page.group(1)))
        account = self._by_token(call.headers.get("authorization", ""))
        call.account = account.letter if account else None
        is_pod = path.startswith("/user/")
        owner = self._owner(path)
        if account is None or (owner is not None and owner != account.user):
            if is_pod:
                target = urllib.parse.quote(owner or "unknown")
                if account is None:
                    return httpx.Response(302, headers={"location": f"/hub/api/oauth2/authorize?client_id=jupyterhub-user-{target}"})
                return httpx.Response(302, headers={"location": f"/hub/user/{target}{path[len('/user/' + target):]}"})
            return httpx.Response(403, json={"status": 403, "message": "Missing or invalid credentials."})
        self._advance(account)
        if is_pod:
            return self._pod(request, call, account)
        account.user_activity = self._hub_now()
        return self._hub_api(request, call, account)

    def _by_token(self, header: str) -> _Account | None:
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "token":
            return None
        return next((account for account in self._accounts.values() if account.token == token), None)

    @staticmethod
    def _owner(path: str) -> str | None:
        match = re.match(r"^/(?:hub/api/users|user)/([^/]+)", path)
        return urllib.parse.unquote(match.group(1)) if match else None

    # --- the hub's API ----------------------------------------------------------------------

    def _hub_api(self, request: httpx.Request, call: Call, account: _Account) -> httpx.Response:
        path, method = call.path, call.method
        if path == f"/hub/api/users/{urllib.parse.quote(account.user, safe='')}" and method == "GET":
            return httpx.Response(200, json=self._user_model(account))
        if path.endswith("/server") and method == "POST":
            return self._launch(account, call.json or {}, STOCK_IMAGE if self.rest_launch_ignored else AUP_IMAGE)
        if path.endswith("/server") and method == "DELETE":
            return self._stop(account)
        if path == "/hub/api/quota/me" and method == "GET":
            return httpx.Response(200, json=self._quota(account))
        if path == f"/hub/admin/api/quota/{urllib.parse.quote(account.user, safe='')}" and method == "GET":
            return httpx.Response(200, json={
                "username": account.user, "balance": account.balance, "unlimited": account.unlimited,
                "recent_transactions": list(reversed(account.transactions))[:20]})
        return httpx.Response(404, json={"status": 404, "message": "Not Found"})

    def _user_model(self, account: _Account) -> dict[str, Any]:
        server = account.server
        servers: dict[str, Any] = {}
        pending = None
        if server is not None:
            pending = "stop" if server.stop_done_at is not None else ("spawn" if server.ready_at is not None else None)
            servers[""] = {
                "name": "", "full_name": f"{account.user}/", "last_activity": _iso(server.last_activity),
                "started": _iso(server.started), "pending": pending, "ready": pending is None and not server.failed, "stopped": False,
                "url": f"/user/{account.user}/", "user_options": dict(server.options),
                "progress_url": f"/hub/api/users/{account.user}/server/progress", "state": {}}
        return {"kind": "user", "name": account.user, "admin": False, "roles": ["user"], "groups": [],
                "server": f"/user/{account.user}/" if server is not None and pending is None and not server.failed else None,
                "pending": None, "created": _iso(self.clock.start), "last_activity": _iso(account.user_activity),
                "servers": servers}

    def _quota(self, account: _Account) -> dict[str, Any]:
        if not account.quota_enabled:
            return {"username": account.user, "balance": 0, "unlimited": True, "rates": dict(DEFAULT_RATES),
                    "enabled": False}
        return {"username": account.user, "balance": account.balance, "unlimited": account.unlimited,
                "rates": {**DEFAULT_RATES, "r9700": account.rate}, "enabled": True}

    def _launch(self, account: _Account, options: Mapping[str, Any], image: str) -> httpx.Response:
        server = account.server
        text_plain = {"content-type": "text/plain"}
        if server is not None and server.failed:
            account.server = server = None
        if server is not None:
            if server.stop_done_at is not None:
                return httpx.Response(400, json={"status": 400, "message": f"{account.user} is pending stop"})
            if server.ready_at is not None:
                return httpx.Response(202, headers=text_plain)
            return httpx.Response(400, json={"status": 400, "message": f"{account.user} is already running"})
        minutes = int(options.get("runtime_minutes", 20))
        if account.quota_enabled and not account.unlimited:
            cost = minutes * account.rate
            if account.balance <= 0:
                reason = f"Insufficient quota (balance: {account.balance})"
            elif account.balance < cost:
                reason = (f"Insufficient quota for {minutes} min (balance: {account.balance}, need: {cost}, "
                          f"max: {account.balance // account.rate} min)")
            else:
                reason = ""
            if reason:
                return httpx.Response(403, json={
                    "status": 403, "message": f"Cannot start container: {reason}. Please contact administrator to add quota."})
        now = self._hub_now()
        account.sessions += 1
        ready_at = now + self.spawn_s if self.spawn_s > 0 else None
        account.server = _Server(now, minutes, dict(options), now, ready_at, image=image)
        return httpx.Response(202 if ready_at is not None else 201, headers=text_plain)

    def _stop(self, account: _Account) -> httpx.Response:
        server = account.server
        text_plain = {"content-type": "text/plain"}
        if server is None:
            return httpx.Response(204)
        if server.stop_done_at is not None:
            return httpx.Response(202, headers=text_plain)
        now = self._hub_now()
        self._charge(account, server, now)
        if self.stop_s <= 0:
            account.server = None
            return httpx.Response(204)
        server.stop_done_at = now + self.stop_s
        return httpx.Response(202, headers=text_plain)

    # --- the hub's web side: login and the spawn form ---------------------------------------

    @staticmethod
    def _cookies(call: Call) -> dict[str, str]:
        jar: http.cookies.SimpleCookie = http.cookies.SimpleCookie()
        try:
            jar.load(call.headers.get("cookie", ""))
        except http.cookies.CookieError:
            return {}
        return {name: morsel.value for name, morsel in jar.items()}

    @staticmethod
    def _page(body: str, *, status: int = 200, cookies: Sequence[str] = ()) -> httpx.Response:
        headers = [("content-type", "text/html; charset=utf-8"), *(("set-cookie", cookie) for cookie in cookies)]
        return httpx.Response(status, headers=headers, text=f"<!DOCTYPE HTML><html><body>{body}</body></html>")

    def _login(self, call: Call) -> httpx.Response:
        """GET: the login page, which sets the _xsrf cookie and repeats it in a hidden input far down the page (the
        real one has pages of style rules first). POST: the user name, the password and _xsrf."""
        if call.method == "GET":
            return self._login_page()
        if call.method != "POST":
            return httpx.Response(405, text="Method Not Allowed")
        form = call.form or {}
        cookie = self._cookies(call).get("_xsrf")
        if not cookie or form.get("_xsrf") != cookie:
            return httpx.Response(403, text="<html><body>403 : Forbidden '_xsrf' argument missing from POST</body></html>",
                                  headers={"content-type": "text/html"})
        account = next((a for a in self._accounts.values()
                        if a.password is not None and a.user == form.get("username") and a.password == form.get("password")),
                       None)
        if account is None:
            return self._login_page(error="Invalid username or password")
        session = f"session-{account.letter}-{next(self._tokens)}"
        self._sessions[session] = account.letter
        return httpx.Response(302, headers=[("location", "/hub/"), ("set-cookie", f"{SESSION_COOKIE}={session}; Path=/hub/"),
                                            ("set-cookie", f"jupyterhub-session-id=id-{session}; Path=/")])

    def _login_page(self, error: str | None = None) -> httpx.Response:
        padding = "<style>" + "/* the hub's style rules */ " * 200 + "</style>"
        message = f'<p class="login_error">{error}</p>' if error else ""
        if self.omit_xsrf:
            return self._page(f'{padding}<form action="/hub/native/login">{message}<input name="username"/></form>')
        token = f"xsrf-{next(self._tokens)}"
        return self._page(
            f'{padding}<form action="/hub/native/login" method="post">{message}<input name="username"/>'
            f'<input type="hidden" name="_xsrf" value="{token}"/></form>', cookies=[f"_xsrf={token}; Path=/hub/"])

    def _spawn_page(self, call: Call, user: str) -> httpx.Response:
        """GET: the spawn page for a session of that user. POST: the spawn-ui fields, with _xsrf in the query."""
        cookies = self._cookies(call)
        letter = self._sessions.get(cookies.get(SESSION_COOKIE, ""))
        account = self._accounts.get(letter) if letter else None
        if account is None or account.user != user:
            return httpx.Response(302, headers={"location": f"/hub/login?next=%2Fhub%2Fspawn%2F{urllib.parse.quote(user)}"})
        self._advance(account)
        if call.method == "GET":
            if account.server is not None:
                return httpx.Response(302, headers={"location": f"/hub/user/{urllib.parse.quote(user)}/"})
            return self._page('<div id="spawn-root"></div>')
        if call.method != "POST":
            return httpx.Response(405, text="Method Not Allowed")
        asked = urllib.parse.parse_qs(call.query).get("_xsrf", [""])[0]
        if not asked or asked != cookies.get("_xsrf"):
            return httpx.Response(403, text="<html><body>403 : Forbidden '_xsrf' argument missing from POST</body></html>",
                                  headers={"content-type": "text/html"})
        form = call.form or {}
        resource = form.get("resource_type", "")
        accelerator = form.get(f"gpu_selection_{resource}", "")
        minutes = form.get("runtime", "")
        if not resource or not accelerator or not minutes.isdigit():
            return self._page('<p class="login_error">The spawn form is incomplete</p>', status=400)
        options = {"gpu_selection": accelerator, "resource_type": resource, "runtime_minutes": int(minutes)}
        launched = self._launch(account, options, AUP_IMAGE)
        if launched.status_code in (201, 202):
            return httpx.Response(302, headers={"location": f"/hub/spawn-pending/{urllib.parse.quote(user)}"})
        return launched

    # --- the pod behind the proxy -----------------------------------------------------------

    def _pod(self, request: httpx.Request, call: Call, account: _Account) -> httpx.Response:
        server = account.server
        if server is None or server.failed or server.ready_at is not None or server.stop_done_at is not None:
            return httpx.Response(424, json={"status": 424, "message": f"Server {account.user} is not running"})
        server.last_activity = self._hub_now()
        rest = call.path[len(f"/user/{urllib.parse.quote(account.user, safe='')}"):]
        method = call.method
        if rest == "/api/me" and method == "GET":
            return httpx.Response(200, json={"identity": {"username": account.user, "name": account.user}, "permissions": {}})
        if rest == "/api/status" and method == "GET":
            return httpx.Response(200, json={"connections": 0, "kernels": 0, "last_activity": _iso(server.last_activity),
                                             "started": _iso(server.started)})
        proxied = re.match(r"^/proxy/(\d+)(/.*)$", rest)
        if proxied and method == "GET":
            return self._model_server(account, int(proxied.group(1)), proxied.group(2))
        contents = re.match(r"^/api/contents/(.+)$", rest)
        if contents:
            return self._contents(account, urllib.parse.unquote(contents.group(1)), call)
        if rest == "/api/terminals" and method == "POST":
            account.terminals += 1
            return httpx.Response(200, json={"name": str(account.terminals), "last_activity": _iso(server.last_activity)})
        return httpx.Response(404, json={"message": "Not Found"})

    @staticmethod
    def _model_server(account: _Account, port: int, rest: str) -> httpx.Response:
        listening = account.models.get(port)
        if listening is None:
            return httpx.Response(500, text="<html><body><h1>500 : Internal Server Error</h1></body></html>",
                                  headers={"content-type": "text/html"})
        models, healthy = listening
        if rest == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": [{"id": name, "object": "model"} for name in models]})
        if rest == "/health":
            return httpx.Response(200 if healthy else 503, text="")
        return httpx.Response(404, json={"detail": "Not Found"})

    def _contents(self, account: _Account, path: str, call: Call) -> httpx.Response:
        key = (account.letter, path)
        if call.method == "PUT":
            created = key not in self.files
            self.files[key] = str((call.json or {}).get("content") or "")
            return httpx.Response(201 if created else 200, json={"name": path.rsplit("/", 1)[-1], "path": path, "type": "file",
                                                                  "format": None, "content": None})
        if call.method == "GET":
            if key not in self.files:
                return httpx.Response(404, json={"message": f"No such file or directory: {path}", "reason": None})
            return httpx.Response(200, json={"name": path.rsplit("/", 1)[-1], "path": path, "type": "file",
                                             "format": "text", "content": self.files[key]})
        return httpx.Response(405, json={"message": "Method Not Allowed"})


def network(hubs: Mapping[str, FakeHub]) -> httpx.MockTransport:
    """One transport for several fake hubs, told apart by the host of the address a call names. A call to a host
    that none of them has fails as a connection to nowhere does."""
    def route(request: httpx.Request) -> httpx.Response:
        hub = hubs.get(request.url.host)
        if hub is None:
            raise httpx.ConnectError(f"no hub answers at {request.url.host}")
        return hub._handle(request)

    return httpx.MockTransport(route)
