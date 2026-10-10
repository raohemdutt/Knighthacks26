"""The AUP Learning Cloud hub over HTTP, one account at a time.

    from lab import hub

    accounts = hub.load_accounts(["C"])                  # AUP_HUB_USER_C and AUP_HUB_TOKEN_C
    with hub.Hub(accounts["C"]) as client:
        reply = client.get(client.user_url())            # GET /hub/api/users/<account>
        reply.ok, reply.status, reply.body, reply.hub_time

An account is any single letter whose AUP_HUB_USER_<letter> and AUP_HUB_TOKEN_<letter> keys exist; A and B are
the two on the DUB hub, C and D the two on the SJC hub.

This is the one place the hub is spoken to, so the rules for talking to it hold for every caller:

- Every request carries the account's token as `Authorization: token ...` and a browser-like
  User-Agent (the hub's front end answers 403 to a bare script agent). Nothing else about the token is
  ever sent, logged or shown: a Hub's repr, an exception message and a reply's text have it cut out.
- A redirect is handed back as the reply and never followed, so the token cannot be sent anywhere the
  caller did not name. A wrong or revoked token on a pod's address answers 302 to a login page.
- A GET, HEAD, PUT or DELETE that meets a gateway error (502, 503, 504) or a dropped connection is tried
  again with growing pauses (1, 2, 4 seconds) before the last answer is handed back or HubUnreachable is
  raised. A POST is never repeated: its first try may have landed, and the caller reads the state back.
- Whatever the hub answers is a Reply, a failure status included. Only a connection that never gets an
  answer raises HubUnreachable.

The address of an account is its own AUP_HUB_URL_<letter> when that key is set, else $LAB_HUB_URL when that is
set, else https://dub.aupcloud.io; an address handed to Hub outranks all three. The keys come from the process
environment, then from the .env files lab/machine/ready.py searches (the clone's, then the two folders
above it); $LAB_ENV_FILES replaces that search. A test hands Hub an httpx transport instead of a network.
"""
from __future__ import annotations

import datetime as dt
import email.utils
import json
import os
import re
import time
import urllib.parse
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from lab.machine import ready

__all__ = ["USER_AGENT", "ENV_USER_AGENT", "ENV_URL", "DEFAULT_URL", "LETTERS", "NODES", "TRANSIENT_STATUSES",
           "Account", "MissingCredentials", "HubUnreachable", "Reply", "Hub", "load_accounts", "node_of", "url_key",
           "one_line", "clip"]

DEFAULT_URL = ready.HUB_URL
ENV_URL = ready.HUB_URL_ENV
ENV_USER_AGENT = "LAB_HUB_USER_AGENT"
# A real browser's agent string: the hub's front end refuses python's and httpx's own.
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/130.0 Safari/537.36")
LETTERS = ("A", "B")                          # what `both` stands for: the pair on the DUB hub
NODES = {"A": "pod-a", "B": "pod-b"}          # the ledger's name for the machine behind the first two accounts
TRANSIENT_STATUSES = frozenset({502, 503, 504})
IDEMPOTENT_METHODS = frozenset({"GET", "HEAD", "PUT", "DELETE", "OPTIONS"})
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
DEFAULT_RETRIES = 3
BACKOFF_FIRST_S = 1.0
BACKOFF_MAX_S = 8.0
TIMEOUT_S = 30.0
TEXT_CHARS = 2000        # how much of an answer's text a Reply keeps
DETAIL_CHARS = 200       # how much of it detail() shows
MIN_HIDDEN_CHARS = ready.MIN_HIDDEN_CHARS


def node_of(letter: str) -> str:
    """The ledger node of an account letter: pod-a for A, pod-b for B, pod-c for C and so on."""
    return NODES.get(letter) or f"pod-{letter.lower()}"


def one_line(text: Any, limit: int | None = None) -> str:
    """text with every run of white space made a single space, cut at limit characters when one is given."""
    flat = " ".join(str(text).split())
    if limit is not None and len(flat) > limit:
        return flat[:max(0, limit - 3)] + "..."
    return flat


def clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit]


# --- accounts -------------------------------------------------------------------------------

@dataclass(frozen=True)
class Account:
    """One hub account: its letter, the user name the hub knows it by, its API token and, when AUP_HUB_URL_<letter>
    is set, the address of the hub it lives on (None: the shared default, see Hub)."""

    letter: str
    user: str
    token: str = field(repr=False)
    url: str | None = None

    @property
    def node(self) -> str:
        return node_of(self.letter)


class MissingCredentials(Exception):
    """Keys the accounts need that no environment or .env file defines. missing holds key NAMES only."""

    def __init__(self, missing: Sequence[str]) -> None:
        self.missing = list(missing)
        super().__init__("missing keys: " + ", ".join(self.missing))


class HubUnreachable(Exception):
    """A call that never got an answer (no connection, a timeout, a dropped stream). The message names the
    call and the kind of failure and carries no secret."""


def key_names(letter: str) -> tuple[str, str]:
    return f"AUP_HUB_USER_{letter}", f"AUP_HUB_TOKEN_{letter}"


def url_key(letter: str) -> str:
    """The optional key holding the address of the hub an account lives on."""
    return f"AUP_HUB_URL_{letter}"


def load_accounts(letters: Iterable[str], *, environ: Mapping[str, str] | None = None,
                  env_files: Sequence[Path] | None = None) -> dict[str, Account]:
    """The accounts of those letters, from the process environment first and then the .env files.

    Raises MissingCredentials naming every key that is absent or blank, in letter order. The address key
    AUP_HUB_URL_<letter> is optional and read the same way. env_files replaces the search of
    lab/machine/ready.py (the clone's .env and the two folders above it)."""
    environ = os.environ if environ is None else environ
    files = ready.default_env_files(environ) if env_files is None else list(env_files)
    wanted = [(letter, *key_names(letter)) for letter in letters]
    names = [name for _, user_key, token_key in wanted for name in (user_key, token_key)]
    url_names = [url_key(letter) for letter, _, _ in wanted]
    given = {name: environ[name].strip() for name in names + url_names
             if isinstance(environ.get(name), str) and environ[name].strip()}
    found, _ = ready.collect_env(names + url_names, given, files)
    missing = [name for name in names if name not in found]
    if missing:
        raise MissingCredentials(missing)
    return {letter: Account(letter, found[user_key], found[token_key], found.get(url_key(letter)))
            for letter, user_key, token_key in wanted}


# --- replies --------------------------------------------------------------------------------

@dataclass(frozen=True)
class Reply:
    """What the hub answered: the status, the JSON body when there was one, the text, a few headers, the hub's
    own clock from its Date header, the seconds the call took and how many tries it needed."""

    status: int
    body: Any = None
    text: str = ""
    headers: Mapping[str, str] = field(default_factory=dict)
    hub_time: dt.datetime | None = None
    elapsed_s: float = 0.0
    attempts: int = 1

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def detail(self) -> str:
        """Why the call failed, on one line: the hub's own message when it sent one, else its page with the
        markup taken out. A redirect says only that it was one: its address names the account."""
        if isinstance(self.body, Mapping):
            for key in ("message", "error", "detail"):
                value = self.body.get(key)
                if isinstance(value, str) and value.strip():
                    return one_line(value, DETAIL_CHARS)
        if self.status in REDIRECT_STATUSES:
            return "redirected to a login page: the token is wrong or revoked, or the path needs one"
        return one_line(re.sub(r"<[^>]+>", " ", self.text), DETAIL_CHARS)


def _hub_time(headers: Mapping[str, str]) -> dt.datetime | None:
    text = headers.get("date")
    if not text:
        return None
    try:
        moment = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=dt.timezone.utc)


# --- the client -----------------------------------------------------------------------------

class Hub:
    """The calls of one account. transport replaces the network (an httpx transport); sleep paces the retries.

    The hub's address is base_url when given, else the account's own (AUP_HUB_URL_<letter>), else $LAB_HUB_URL,
    else the DUB default."""

    def __init__(self, account: Account, *, base_url: str | None = None, transport: httpx.BaseTransport | None = None,
                 sleep: Callable[[float], None] = time.sleep, retries: int = DEFAULT_RETRIES,
                 timeout_s: float = TIMEOUT_S, user_agent: str | None = None,
                 environ: Mapping[str, str] | None = None) -> None:
        environ = os.environ if environ is None else environ
        self.account = account
        self.base_url = (base_url or account.url or environ.get(ENV_URL) or DEFAULT_URL).rstrip("/")
        self.user_agent = user_agent or environ.get(ENV_USER_AGENT) or USER_AGENT
        self.retries = retries
        self.timeout_s = timeout_s
        self._sleep = sleep
        self._hide = [account.token] if len(account.token) >= MIN_HIDDEN_CHARS else []
        self._client = httpx.Client(
            base_url=self.base_url, transport=transport, timeout=timeout_s, follow_redirects=False,
            headers={"Authorization": f"token {account.token}", "User-Agent": self.user_agent,
                     "Accept": "application/json"})

    def __repr__(self) -> str:
        return f"Hub(account={self.account.letter!r}, base_url={self.base_url!r})"

    def __enter__(self) -> "Hub":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # addresses

    def user_url(self, suffix: str = "") -> str:
        """The hub API address of this account: /hub/api/users/<user><suffix>."""
        return f"/hub/api/users/{urllib.parse.quote(self.account.user, safe='')}{suffix}"

    def pod_url(self, suffix: str = "") -> str:
        """The address of this account's server behind the hub's proxy: /user/<user><suffix>."""
        return f"/user/{urllib.parse.quote(self.account.user, safe='')}{suffix}"

    def ws_url(self, suffix: str) -> str:
        """pod_url(suffix) as a websocket address: wss for an https hub, ws for an http one."""
        parts = urllib.parse.urlsplit(self.base_url)
        scheme = "wss" if parts.scheme == "https" else "ws"
        return urllib.parse.urlunsplit((scheme, parts.netloc, parts.path.rstrip("/") + self.pod_url(suffix), "", ""))

    def ws_headers(self) -> dict[str, str]:
        """The headers a websocket handshake needs besides the user agent (see user_agent)."""
        return {"Authorization": f"token {self.account.token}"}

    # secrets

    def scrub(self, text: str) -> str:
        """text with this account's token cut out."""
        for secret in self._hide:
            text = text.replace(secret, "***")
        return text

    # calls

    def request(self, method: str, path: str, *, json: Any = None, params: Mapping[str, Any] | None = None,
                timeout: float | None = None, retry: bool | None = None,
                headers: Mapping[str, str] | None = None) -> Reply:
        """One call to the hub. retry says whether a gateway error or dropped connection may be tried again;
        it defaults to yes for methods that are safe to repeat and no for POST."""
        method = method.upper()
        again = (method in IDEMPOTENT_METHODS) if retry is None else retry
        tries = 1 + (self.retries if again else 0)
        reply: Reply | None = None
        failure: HubUnreachable | None = None
        for attempt in range(1, tries + 1):
            started = time.perf_counter()
            try:
                response = self._client.request(method, path, json=json, params=params, headers=headers,
                                                timeout=timeout if timeout is not None else self.timeout_s)
            except httpx.TransportError as exc:
                failure = HubUnreachable(self.scrub(f"{method} {path}: {type(exc).__name__}: {exc}"))
                reply = None
            else:
                failure = None
                reply = self._reply(response, time.perf_counter() - started, attempt)
                if reply.status not in TRANSIENT_STATUSES:
                    return reply
            if attempt < tries:
                self._sleep(min(BACKOFF_FIRST_S * 2 ** (attempt - 1), BACKOFF_MAX_S))
        if reply is not None:
            return reply
        assert failure is not None
        raise failure

    def get(self, path: str, **kwargs: Any) -> Reply:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> Reply:
        return self.request("POST", path, **kwargs)

    def put(self, path: str, **kwargs: Any) -> Reply:
        return self.request("PUT", path, **kwargs)

    def delete(self, path: str, **kwargs: Any) -> Reply:
        return self.request("DELETE", path, **kwargs)

    def _reply(self, response: httpx.Response, seconds: float, attempts: int) -> Reply:
        text = response.text
        body: Any = None
        if text.strip()[:1] in ("{", "["):
            try:
                body = json.loads(text)
            except ValueError:
                body = None
        headers = {name.lower(): value for name, value in response.headers.items()}
        return Reply(status=response.status_code, body=body, text=clip(self.scrub(text), TEXT_CHARS), headers=headers,
                     hub_time=_hub_time(headers), elapsed_s=round(seconds, 3), attempts=attempts)
