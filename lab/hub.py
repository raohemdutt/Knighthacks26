"""The AUP Learning Cloud hub over HTTP, one account at a time.

    from lab import hub

    accounts = hub.load_accounts(["C"])                  # AUP_HUB_USER_C and AUP_HUB_TOKEN_C
    with hub.Hub(accounts["C"]) as client:
        reply = client.get(client.user_url())            # GET /hub/api/users/<account>
        reply.ok, reply.status, reply.body, reply.hub_time

An account is any single letter whose AUP_HUB_USER_<letter> and AUP_HUB_TOKEN_<letter> keys exist; A and B are
the two on the DUB hub, C and D the two on the SJC hub. Four more keys are optional per account:
AUP_HUB_PASS_<letter> (the web login password, used only by the spawn form below), AUP_HUB_LAUNCH_<letter> (form,
rest or auto: how lab/controller.py launches the pod), and AUP_HUB_GPU_<letter> and AUP_HUB_RESOURCE_<letter> (the
accelerator and the resource the account launches by default).

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
- The web spawn form (Hub.spawn_form) is the one call that does not use the token: it logs in as the user with
  AUP_HUB_PASS_<letter> and holds the session cookies in a client of its own, so the token never rides with a cookie.
  The password is cut out of every reply text and error message exactly as the token is.

The address of an account is its own AUP_HUB_URL_<letter> when that key is set, else $LAB_HUB_URL when that is
set, else https://dub.aupcloud.io; an address handed to Hub outranks all three. The keys come from the process
environment, then from the .env files lab/machine/ready.py searches (the clone's, then the two folders
above it); $LAB_ENV_FILES replaces that search.

Rows are named by pod role, not by account: LAB_POD_ROLES=C:pod-a,D:pod-b says which account fills the teacher's
(pod-a) and the trainer's (pod-b) role, read the same way (environment, then the .env files); unset it is
A:pod-a,B:pod-b. An account with no role is refused (PodRoleError), never named pod-<letter>.

Rows are named by pod role, not by account: LAB_POD_ROLES=C:pod-a,D:pod-b says which account fills the teacher's
(pod-a) and the trainer's (pod-b) role, read the same way (environment, then the .env files); unset it is
A:pod-a,B:pod-b. An account with no role is refused (PodRoleError), never named pod-<letter>. A test hands Hub an httpx transport instead of a network.
"""
from __future__ import annotations

import datetime as dt
import email.utils
import html
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

__all__ = ["USER_AGENT", "ENV_USER_AGENT", "ENV_URL", "DEFAULT_URL", "LETTERS", "NODES", "ENV_POD_ROLES", "DEFAULT_POD_ROLES",
           "ROLES", "TRANSIENT_STATUSES", "PodRoleError", "pod_roles", "hub_of",
           "LAUNCH_MODES", "Account", "MissingCredentials", "HubUnreachable", "Reply", "FormResult", "Hub",
           "load_accounts", "node_of", "url_key", "password_key", "spawn_fields", "one_line", "clip"]

DEFAULT_URL = ready.HUB_URL
ENV_URL = ready.HUB_URL_ENV
ENV_USER_AGENT = "LAB_HUB_USER_AGENT"
# A real browser's agent string: the hub's front end refuses python's and httpx's own.
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/130.0 Safari/537.36")
LETTERS = ("A", "B")                          # what `both` stands for: the pair on the DUB hub
NODES = {"A": "pod-a", "B": "pod-b"}          # the roles of the DUB hub's pair: what LAB_POD_ROLES defaults to
ENV_POD_ROLES = "LAB_POD_ROLES"
DEFAULT_POD_ROLES = "A:pod-a,B:pod-b"
ROLES = ("pod-a", "pod-b")                    # the machines a row can be about: the teacher's pod and the trainer's
TRANSIENT_STATUSES = frozenset({502, 503, 504})
IDEMPOTENT_METHODS = frozenset({"GET", "HEAD", "PUT", "DELETE", "OPTIONS"})
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
DEFAULT_RETRIES = 3
BACKOFF_FIRST_S = 1.0
BACKOFF_MAX_S = 8.0
TIMEOUT_S = 30.0
TEXT_CHARS = 2000        # how much of an answer's text a Reply keeps
PAGE_CHARS = 400_000     # how much of a web page the spawn form keeps: the login page's _xsrf field sits far down
DETAIL_CHARS = 200       # how much of it detail() shows
MIN_HIDDEN_CHARS = ready.MIN_HIDDEN_CHARS
LAUNCH_MODES = ("form", "rest", "auto")      # AUP_HUB_LAUNCH_<letter>: see lab/controller.py
LOGIN_PATH = "/hub/native/login"
LOGIN_COOKIE = "jupyterhub-hub-login"        # set by a login the hub accepted
XSRF_COOKIE = "_xsrf"
HTML_ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"


class PodRoleError(ValueError):
    """An account that fills no pod role in LAB_POD_ROLES, or a setting that cannot be read. The message says
    which and how to fix it; a node is never made up from the letter."""


def _parse_roles(text: str) -> dict[str, str]:
    roles: dict[str, str] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        letter, sep, role = (piece.strip() for piece in part.partition(":"))
        if not sep or len(letter) != 1 or not letter.isalpha() or role.lower() not in ROLES:
            raise PodRoleError(f"{ENV_POD_ROLES} entry {part!r} is not <letter>:<role> with the role one of "
                               f"{' or '.join(ROLES)} (for example {DEFAULT_POD_ROLES})")
        if letter.upper() in roles:
            raise PodRoleError(f"{ENV_POD_ROLES} names account {letter.upper()} twice")
        roles[letter.upper()] = role.lower()
    return roles


def pod_roles(environ: Mapping[str, str] | None = None, env_files: Sequence[Path] | None = None) -> dict[str, str]:
    """The pod role each account letter fills, from LAB_POD_ROLES: the process environment, then the .env files
    lab/machine/ready.py searches. Unset, it is A:pod-a,B:pod-b (the DUB hub). On SJC, C:pod-a,D:pod-b."""
    environ = os.environ if environ is None else environ
    files = ready.default_env_files(environ) if env_files is None else list(env_files)
    found, _ = ready.collect_env([ENV_POD_ROLES], {k: v for k, v in environ.items() if k == ENV_POD_ROLES}, files)
    return _parse_roles(found.get(ENV_POD_ROLES, DEFAULT_POD_ROLES))


def node_of(letter: str, roles: Mapping[str, str] | None = None) -> str:
    """The ledger node of an account letter: the role it fills (pod-a for the teacher's account, pod-b for the
    trainer's). roles is pod_roles(); without it, the default pair. Raises PodRoleError for a letter with no role."""
    roles = _parse_roles(DEFAULT_POD_ROLES) if roles is None else roles
    role = roles.get(letter.upper())
    if role is None:
        have = ", ".join(f"{k}:{v}" for k, v in sorted(roles.items())) or "none"
        raise PodRoleError(f"account {letter} fills no pod role: {ENV_POD_ROLES} names {have}; set "
                           f"{ENV_POD_ROLES}=<letter>:pod-a,<letter>:pod-b (SJC: C:pod-a,D:pod-b) in the environment "
                           "or the workspace .env")
    return role


def hub_of(url: str) -> str:
    """The name of a hub in a row's conditions: its host (dub.aupcloud.io, sjc.aupcloud.io)."""
    return urllib.parse.urlsplit(url if "//" in url else "//" + url).netloc or url


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
    is set, the address of the hub it lives on (None: the shared default, see Hub).

    The rest is optional and comes from keys of the same letter: the web login password (AUP_HUB_PASS_<letter>,
    never shown by repr), how the controller launches the pod (AUP_HUB_LAUNCH_<letter>: form, rest or auto) and
    the accelerator and resource it launches by default (AUP_HUB_GPU_<letter>, AUP_HUB_RESOURCE_<letter>)."""

    letter: str
    user: str
    token: str = field(repr=False)
    url: str | None = None
    password: str | None = field(default=None, repr=False)
    launch: str | None = None
    gpu: str | None = None
    resource: str | None = None
    role: str | None = None     # the pod role this account fills (LAB_POD_ROLES); None: the default pair's

    @property
    def node(self) -> str:
        """The ledger node: the role the account fills, never pod-<letter>. PodRoleError when it fills none."""
        return self.role or node_of(self.letter)


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


def password_key(letter: str) -> str:
    """The optional key holding the password of the account's web login (the spawn form)."""
    return f"AUP_HUB_PASS_{letter}"


def option_keys(letter: str) -> tuple[str, str, str, str, str]:
    """Every optional key of an account: address, password, launch mode, accelerator, resource."""
    return (url_key(letter), password_key(letter), f"AUP_HUB_LAUNCH_{letter}", f"AUP_HUB_GPU_{letter}",
            f"AUP_HUB_RESOURCE_{letter}")


def load_accounts(letters: Iterable[str], *, environ: Mapping[str, str] | None = None,
                  env_files: Sequence[Path] | None = None, require_role: bool = False) -> dict[str, Account]:
    """The accounts of those letters, from the process environment first and then the .env files.

    Raises MissingCredentials naming every key that is absent or blank, in letter order. The optional keys
    (address, web password, launch mode, accelerator, resource: see option_keys) are read the same way and never
    required. require_role refuses (PodRoleError) an account that fills no pod role in LAB_POD_ROLES, as every verb
    that writes a row must; a caller that only talks to the pod (the proxy, the transfer) leaves it off. env_files replaces the search of lab/machine/ready.py (the clone's .env and the two folders above it)."""
    environ = os.environ if environ is None else environ
    files = ready.default_env_files(environ) if env_files is None else list(env_files)
    roles = pod_roles(environ, files)
    letters = list(letters)
    wanted = [(letter, *key_names(letter)) for letter in letters]
    names = [name for _, user_key, token_key in wanted for name in (user_key, token_key)]
    optional = [name for letter, _, _ in wanted for name in option_keys(letter)] + [ENV_URL]
    given = {name: environ[name].strip() for name in names + optional
             if isinstance(environ.get(name), str) and environ[name].strip()}
    found, _ = ready.collect_env(names + optional, given, files)
    missing = [name for name in names if name not in found]
    if missing:
        raise MissingCredentials(missing)
    for letter in letters if require_role else ():
        node_of(letter, roles)      # an account with no role is refused here
    accounts = {}
    for letter, user_key, token_key in wanted:
        url, password, launch, gpu, resource = (found.get(name) for name in option_keys(letter))
        accounts[letter] = Account(letter, found[user_key], found[token_key], url or found.get(ENV_URL), password,
                                   launch, gpu, resource, roles.get(letter.upper()))
    return accounts


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


@dataclass(frozen=True)
class FormResult:
    """How a launch through the hub's web spawn form went.

    ok says the hub accepted the spawn (it answered with a redirect that is not back to the login page). stage is
    the step it reached: login_page, login, spawn_page or spawn. status is the last status the hub answered with
    (0: it did not answer). kind names what went wrong when ok is False: form_unavailable (a page that should be
    there is not), xsrf_missing (no _xsrf token to send), login_failed (the hub did not accept the user name and
    password), spawn_refused (the hub did not start the server), unexpected_status or no_answer. error is the hub's
    own words when it had some, with the password and the token cut out. location is the path the hub sent the
    browser to after an accepted spawn."""

    ok: bool
    stage: str
    status: int
    kind: str | None = None
    error: str | None = None
    location: str | None = None


def spawn_fields(minutes: int, resource_type: str, accelerator: str) -> dict[str, str]:
    """The fields the hub's spawn page posts: the resource, the accelerator under gpu_selection_<resource>, the
    runtime in minutes and the repository fields left empty (no repository is cloned)."""
    return {"resource_type": resource_type, f"gpu_selection_{resource_type}": accelerator, "runtime": str(minutes),
            "repo_url": "", "repo_persist": "true", "repo_branch": ""}


_INPUT_TAG = re.compile(r"<input\b[^>]*>", re.IGNORECASE)
_ATTRIBUTE = re.compile(r"""([\w:.-]+)\s*=\s*(?:"([^"]*)"|'([^']*)')""")
_PAGE_ERROR = re.compile(r"""(?is)<[^>]*\bclass\s*=\s*["'][^"']*(?:login_error|alert)[^"']*["'][^>]*>(.*?)</""")


def _hidden_input(page: str, name: str) -> str | None:
    """The value of the hidden input of that name in an HTML page; None when it is absent or empty."""
    for tag in _INPUT_TAG.findall(page):
        attributes = {m.group(1).lower(): html.unescape(m.group(2) if m.group(2) is not None else m.group(3))
                      for m in _ATTRIBUTE.finditer(tag)}
        if attributes.get("name") == name and attributes.get("type", "hidden").lower() == "hidden":
            return attributes.get("value") or None
    return None


def _page_error(page: str) -> str | None:
    """The message a hub page shows for a refused form (its login_error or alert element), as one short line."""
    match = _PAGE_ERROR.search(page)
    text = one_line(html.unescape(re.sub(r"<[^>]+>", " ", match.group(1))), DETAIL_CHARS) if match else ""
    return text or None


def _cookie(client: httpx.Client, name: str) -> str | None:
    """The newest cookie of that name the client holds, whatever its path; None when it has none."""
    found = [cookie.value for cookie in client.cookies.jar if cookie.name == name and cookie.value]
    return found[-1] if found else None


def _login_redirect(location: str) -> bool:
    """Whether a redirect leads back to the hub's login page: the session or the _xsrf token was not accepted."""
    return urllib.parse.urlsplit(location).path.rstrip("/").endswith("/login")


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
        self._transport = transport
        self._hide = [secret for secret in (account.token, account.password)
                      if secret and len(secret) >= MIN_HIDDEN_CHARS]
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

    def spawn_url(self) -> str:
        """The address of this account's web spawn page: /hub/spawn/<user>."""
        return f"/hub/spawn/{urllib.parse.quote(self.account.user, safe='')}"

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
        """text with this account's token and web password cut out."""
        for secret in self._hide:
            text = text.replace(secret, "***")
        return text

    # calls

    def request(self, method: str, path: str, *, json: Any = None, params: Mapping[str, Any] | None = None,
                timeout: float | None = None, retry: bool | None = None,
                headers: Mapping[str, str] | None = None, data: Mapping[str, str] | None = None,
                client: httpx.Client | None = None, text_chars: int = TEXT_CHARS) -> Reply:
        """One call to the hub. retry says whether a gateway error or dropped connection may be tried again;
        it defaults to yes for methods that are safe to repeat and no for POST. data is a form body. client is
        another session to send through (the spawn form's, which holds cookies and no token) and text_chars how
        much of the answer's text the Reply keeps."""
        method = method.upper()
        again = (method in IDEMPOTENT_METHODS) if retry is None else retry
        tries = 1 + (self.retries if again else 0)
        reply: Reply | None = None
        failure: HubUnreachable | None = None
        for attempt in range(1, tries + 1):
            started = time.perf_counter()
            try:
                response = (client or self._client).request(
                    method, path, json=json, data=data, params=params, headers=headers,
                    timeout=timeout if timeout is not None else self.timeout_s)
            except httpx.TransportError as exc:
                failure = HubUnreachable(self.scrub(f"{method} {path}: {type(exc).__name__}: {exc}"))
                reply = None
            else:
                failure = None
                reply = self._reply(response, time.perf_counter() - started, attempt, text_chars)
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

    def _reply(self, response: httpx.Response, seconds: float, attempts: int, text_chars: int = TEXT_CHARS) -> Reply:
        text = response.text
        body: Any = None
        if text.strip()[:1] in ("{", "["):
            try:
                body = json.loads(text)
            except ValueError:
                body = None
        headers = {name.lower(): value for name, value in response.headers.items()}
        return Reply(status=response.status_code, body=body, text=clip(self.scrub(text), text_chars), headers=headers,
                     hub_time=_hub_time(headers), elapsed_s=round(seconds, 3), attempts=attempts)

    # the web spawn form

    def spawn_form(self, minutes: int, resource_type: str, accelerator: str) -> FormResult:
        """Start this account's server through the hub's web spawn form, as the browser does.

        Needed where the REST launch (POST /hub/api/users/<user>/server) is accepted but its options are ignored, as
        on the SJC hub: the pod then comes up as the stock singleuser image, and only the form gets the real one.
        The steps: GET /hub/native/login (the _xsrf cookie and hidden field), POST it with the user name, the
        password and _xsrf (a redirect and the jupyterhub-hub-login cookie mean the hub accepted it), GET
        /hub/spawn/<user> (200), then POST /hub/spawn/<user>?_xsrf=<the _xsrf cookie> with spawn_fields (a redirect
        that is not back to the login page means the spawn started). The caller polls the API for readiness.

        The session lives in a client of its own that holds the cookies and never the token; the password is read
        from the account (MissingCredentials names AUP_HUB_PASS_<letter> when it has none) and is cut out of
        every message. A POST is sent once. Whatever goes wrong comes back as a FormResult, a hub that does not
        answer included (kind no_answer)."""
        password = self.account.password
        if not password:
            raise MissingCredentials([password_key(self.account.letter)])
        spawn_path = self.spawn_url()
        stage = "login_page"

        def fail(kind: str, error: str, status: int = 0) -> FormResult:
            return FormResult(False, stage, status, kind, one_line(self.scrub(error), DETAIL_CHARS))

        def refused(reply: Reply, what: str) -> str:
            if isinstance(reply.body, Mapping) and reply.detail():
                return reply.detail()
            return _page_error(reply.text) or f"the hub answered {reply.status} to {what}"

        session = httpx.Client(base_url=self.base_url, transport=self._transport, timeout=self.timeout_s,
                               follow_redirects=False, headers={"User-Agent": self.user_agent, "Accept": HTML_ACCEPT})
        try:
            page = self.request("GET", LOGIN_PATH, client=session, text_chars=PAGE_CHARS)
            if page.status != 200:
                return fail("form_unavailable", f"the login page answered {page.status}", page.status)
            xsrf = _cookie(session, XSRF_COOKIE) or _hidden_input(page.text, XSRF_COOKIE)
            if not xsrf:
                return fail("xsrf_missing", "the login page carried no _xsrf token: no cookie and no hidden field",
                            page.status)

            stage = "login"
            login = self.request("POST", LOGIN_PATH, client=session, text_chars=PAGE_CHARS,
                                 data={"username": self.account.user, "password": password, XSRF_COOKIE: xsrf},
                                 headers={"Referer": self.base_url + LOGIN_PATH})
            if login.status not in REDIRECT_STATUSES or not _cookie(session, LOGIN_COOKIE):
                if login.status in (200, *REDIRECT_STATUSES):
                    return fail("login_failed", _page_error(login.text) or "the hub did not accept the user name "
                                "and password", login.status)
                return fail("unexpected_status", refused(login, "the login"), login.status)

            stage = "spawn_page"
            form = self.request("GET", spawn_path, client=session)
            if form.status != 200:
                where = " (a server may already be running)" if form.status in REDIRECT_STATUSES else ""
                return fail("form_unavailable", f"the spawn page answered {form.status}{where}", form.status)
            xsrf = _cookie(session, XSRF_COOKIE)
            if not xsrf:
                return fail("xsrf_missing", "the hub set no _xsrf cookie for the spawn form", form.status)

            stage = "spawn"
            spawn = self.request("POST", spawn_path, client=session, params={XSRF_COOKIE: xsrf}, text_chars=PAGE_CHARS,
                                 data=spawn_fields(minutes, resource_type, accelerator),
                                 headers={"Referer": self.base_url + spawn_path})
            location = spawn.headers.get("location")
            if spawn.status in REDIRECT_STATUSES:
                if location and not _login_redirect(location):
                    return FormResult(True, stage, spawn.status, location=urllib.parse.urlsplit(location).path)
                return fail("spawn_refused", "the hub sent the spawn back to its login page: the session or the "
                            "_xsrf token was not accepted" if location else
                            "the hub redirected the spawn without saying where", spawn.status)
            if spawn.status >= 500:
                return fail("unexpected_status", refused(spawn, "the spawn"), spawn.status)
            return fail("spawn_refused", refused(spawn, "the spawn"), spawn.status)
        except HubUnreachable as exc:
            return fail("no_answer", str(exc))
        finally:
            session.close()
