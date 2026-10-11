"""ready: may a sitting start right now? One JSON line; exit 0 when yes, 1 when not.

    python -m lab ready

Every failing reason is listed at once. The check changes nothing a sitting would notice:
its only write is a throwaway file in the state dir, and its only network traffic is one
read-only GET per hub account.

Gates (GATE_CHECKS): any one failing makes ready false, lists it under "reasons", and exits 1.

    stop       no STOP file (lab/machine/stop.py); this verb runs while stopped so it can say so
    state_dir  lab/state (or $LAB_STATE_DIR) exists, is created when missing, and takes a file
    env        the .env keys a sitting needs are present: key NAMES only, a value is never printed. The accounts
               a sitting needs are the letters LAB_POD_ROLES names (C and D on SJC; unset, A and B on DUB)

Advisory (every other check): still run and reported under "checks", but a failure is listed
under "warnings" and never changes ready or the exit code. A hub outage must not halt GPU-free
work; a sitting that needs a pod reads checks.hub and the warnings for itself.

    hub        GET <hub>/hub/api/users/<account> answers 200 with a JSON object, once per account that
               LAB_POD_ROLES names, through lab.hub.Hub, the one hub client (token, agent, no redirects,
               scrubbing), at the account's own AUP_HUB_URL_<letter> (else $LAB_HUB_URL, else DUB). Each
               account reports reachable, running (the hub lists a server for it) and ready (that server is
               ready), plus its role and the hub asked. An account with no keys, or with keys but no role,
               is reported and not probed.

Keys come from the process environment first, then from .env files: the clone's, then the one
in each of the two folders above it (the Atlas workspace keeps its .env one level above atlas/
and two above work/<ID>/). The first definition wins and a blank value counts as missing.
$LAB_ENV_FILES (paths separated by os.pathsep) replaces that search; $LAB_HUB_URL replaces the
hub address.
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
import urllib.request
from typing import Any

import httpx

from lab.machine import stop

CLI_VERB = "ready"   # registers python -m lab ready
REPO_ROOT = Path(__file__).resolve().parents[2]
ACCOUNTS = ("A", "B")
REQUIRED_ENV = ("AUP_HUB_TOKEN_A", "AUP_HUB_TOKEN_B", "AUP_HUB_USER_A", "AUP_HUB_USER_B")   # the DUB default pair
CREDENTIAL_KEY = re.compile(r"^AUP_HUB_(USER|TOKEN)_([A-Za-z])$")
ENV_FILES_ENV = "LAB_ENV_FILES"
HUB_URL_ENV = "LAB_HUB_URL"
HUB_URL = "https://dub.aupcloud.io"
HUB_TIMEOUT_S = 8.0
# ready reports what the hub answers right now; it never waits out the hub client's gateway backoff.
HUB_PROBE_RETRIES = 0
HTTP_OK = 200
# Values shorter than this are not worth hiding in an error message; they would mangle it.
MIN_HIDDEN_CHARS = 6


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never follow a redirect: a token must only go to the address it was meant for. Kept here for
    lab.machine.slots_notion, which builds its opener with it; ready's own hub probe goes through lab.hub.Hub."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None
# The checks that decide ready and the exit code; every other check (the hub) only warns.
GATE_CHECKS = ("stop", "state_dir", "env")

# A test's stand-in for the network under lab.hub.Hub: (url, headers, timeout) -> (status, parsed JSON or None).
Fetch = Callable[[str, Mapping[str, str], float], tuple[int, Any]]


def read_env_file(path: Path) -> dict[str, str]:
    """KEY=VALUE pairs of a dotenv file; {} when it cannot be read.

    Blank lines and # comments are skipped, `export ` is allowed, one pair of quotes is
    stripped, and a trailing ` # comment` is cut from an unquoted value.
    """
    pairs: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return pairs
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()
        pairs[key] = value
    return pairs


def default_env_files(environ: Mapping[str, str] | None = None) -> list[Path]:
    """$LAB_ENV_FILES when set (empty means none), else the clone's .env and the two folders above it."""
    environ = os.environ if environ is None else environ
    override = environ.get(ENV_FILES_ENV)
    if override is not None:
        return [Path(part) for part in override.split(os.pathsep) if part]
    return [REPO_ROOT / ".env", REPO_ROOT.parent / ".env", REPO_ROOT.parent.parent / ".env"]


def collect_env(
    wanted: Iterable[str], environ: Mapping[str, str], files: Sequence[Path]
) -> tuple[dict[str, str], list[Path]]:
    """The wanted keys that have a value, and the .env files that exist.

    The process environment is read first, then each file in order; the first definition wins.
    """
    wanted = tuple(wanted)
    found = {key: environ[key] for key in wanted if environ.get(key)}
    used: list[Path] = []
    for path in files:
        if not path.is_file():
            continue
        used.append(path)
        pairs = read_env_file(path)
        for key in wanted:
            if key not in found and pairs.get(key):
                found[key] = pairs[key]
    return found, used


def scrub(text: str, hide: Sequence[str]) -> str:
    """text with every secret in hide replaced by ***, both as given and with its surrounding whitespace stripped.

    A token sourced from a CRLF .env arrives with a CR or LF on its end, yet the messages that carry it spell it
    bare, or with the line break written as an escape the way http.client's "Invalid header value" does; only the
    stripped form matches those. A secret that is blank once stripped hides nothing.
    """
    for secret in hide:
        bare = secret.strip()
        if bare:
            text = text.replace(secret, "***").replace(bare, "***")
    return text


def check_stop() -> dict[str, Any]:
    is_stopped = stop.stopped()
    return {"ok": not is_stopped, "stopped": is_stopped, "reason": stop.stop_reason(), "file": str(stop.stop_file())}


def check_state_dir() -> dict[str, Any]:
    path = stop.state_dir()
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path, prefix=".ready-"):
            pass
    except OSError as exc:
        return {"ok": False, "path": str(path), "writable": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "path": str(path), "writable": True}


def required_env(letters: Iterable[str]) -> tuple[str, ...]:
    """The keys a sitting needs for those account letters: every token, then every user name."""
    letters = tuple(letters)
    return tuple(f"AUP_HUB_TOKEN_{x}" for x in letters) + tuple(f"AUP_HUB_USER_{x}" for x in letters)


def check_env(
    values: Mapping[str, str], files: Sequence[Path], required: Sequence[str] = REQUIRED_ENV, roles_error: str = ""
) -> dict[str, Any]:
    missing = [key for key in required if key not in values]
    result: dict[str, Any] = {
        "ok": not missing and not roles_error,
        "present": [key for key in required if key in values],
        "missing": missing,
        "files": [str(path) for path in files],
    }
    if roles_error:
        result["roles_error"] = roles_error
    return result


def credential_letters(environ: Mapping[str, str], files: Sequence[Path]) -> list[str]:
    """Every account letter that has both a user and a token somewhere in the environment or the .env files."""
    have: dict[str, set[str]] = {}
    sources = [{k: v for k, v in environ.items() if isinstance(v, str)}] + [read_env_file(p) for p in files]
    for source in sources:
        for key, value in source.items():
            match = CREDENTIAL_KEY.match(key)
            if match and value.strip():
                have.setdefault(match.group(2).upper(), set()).add(match.group(1))
    return sorted(letter for letter, kinds in have.items() if kinds == {"USER", "TOKEN"})


def fetch_transport(fetch: Fetch, timeout: float) -> httpx.BaseTransport:
    """An httpx transport for lab.hub.Hub that hands each request to fetch(url, headers, timeout).

    fetch answers (status, parsed JSON or None); whatever it raises reaches the probe unchanged.
    It is the seam tests use in place of a network.
    """

    def handle(request: httpx.Request) -> httpx.Response:
        status, body = fetch(str(request.url), dict(request.headers), timeout)
        if body is None:
            return httpx.Response(status, request=request)
        return httpx.Response(status, json=body, request=request)

    return httpx.MockTransport(handle)


def probe_account(
    letter: str,
    user: str,
    token: str,
    *,
    base_url: str,
    transport: httpx.BaseTransport | None,
    environ: Mapping[str, str],
    hide: Sequence[str],
) -> dict[str, Any]:
    """One read-only GET of the account's hub user endpoint. Any failure is a verdict, never a crash.

    reachable: the hub answered 200 with a JSON object. running: that object lists a server that is not stopping.
    ready: that server is ready. ok is reachable.
    """
    # Imported here, not at the top: lab.hub reads this module's constants when it loads.
    from lab import hub as lab_hub

    account = lab_hub.Account(letter, user, token)
    down = {"ok": False, "reachable": False, "running": False, "ready": False}
    try:
        with lab_hub.Hub(
            account,
            base_url=base_url,
            transport=transport,
            retries=HUB_PROBE_RETRIES,
            timeout_s=HUB_TIMEOUT_S,
            environ=environ,
        ) as client:
            reply = client.get(client.user_url())
    except Exception as exc:
        # HubUnreachable for no answer at all; anything else a stand-in transport raised. Both are reported.
        return {**down, "error": scrub(f"{type(exc).__name__}: {exc}", hide)}
    if reply.status != HTTP_OK:
        return {**down, "status": reply.status}
    if not isinstance(reply.body, dict):
        return {**down, "status": reply.status, "error": "response is not a JSON object"}
    servers = reply.body.get("servers")
    server = servers.get("") if isinstance(servers, dict) else None
    running = isinstance(server, dict) and not server.get("stopped") and server.get("pending") != "stop"
    return {
        "ok": True,
        "reachable": True,
        "running": running,
        "ready": bool(running and server.get("ready") is True),
        "status": reply.status,
    }


def check_hub(
    values: Mapping[str, str],
    roles: Mapping[str, str],
    base_url: str | None,
    transport: httpx.BaseTransport | None,
    environ: Mapping[str, str],
    unroled: Sequence[str] = (),
) -> dict[str, Any]:
    """Probe each account that fills a pod role, at its own hub: base_url when given, else its
    AUP_HUB_URL_<letter>, else $LAB_HUB_URL, else DUB. unroled lists letters that have keys and no role: reported only."""
    from lab import hub as lab_hub

    hide = sorted((value for value in values.values() if len(value) >= MIN_HIDDEN_CHARS), key=len, reverse=True)
    accounts: dict[str, dict[str, Any]] = {}
    urls: list[str] = []
    for letter in sorted(roles, key=lambda x: (roles[x], x)):
        url = (base_url or values.get(lab_hub.url_key(letter)) or environ.get(HUB_URL_ENV) or HUB_URL).rstrip("/")
        user, token = values.get(f"AUP_HUB_USER_{letter}"), values.get(f"AUP_HUB_TOKEN_{letter}")
        if user and token:
            probe = probe_account(letter, user, token, base_url=url, transport=transport, environ=environ, hide=hide)
            probe.update(role=roles[letter], hub=lab_hub.hub_of(url), url=url)
            urls.append(url)
        else:
            probe = {"ok": False, "reachable": False, "running": False, "ready": False,
                     "skipped": "credentials missing", "role": roles[letter]}
        accounts[letter] = probe
    ok = all(probe["ok"] for probe in accounts.values())
    for letter in unroled:
        accounts[letter] = {"ok": False, "reachable": False, "running": False, "ready": False,
                            "skipped": "fills no pod role in LAB_POD_ROLES"}
    urls = list(dict.fromkeys(urls))
    return {
        "ok": ok,
        "url": urls[0] if len(urls) == 1 else None,
        "hubs": [lab_hub.hub_of(url) for url in urls],
        "accounts": accounts,
    }


def reasons_for(checks: Mapping[str, Any]) -> list[str]:
    """Why a sitting may not start: one line per failing gate check (GATE_CHECKS)."""
    reasons: list[str] = []
    if not checks["stop"]["ok"]:
        reasons.append(f"STOP file present: {checks['stop']['reason']}")
    if not checks["state_dir"]["ok"]:
        reasons.append(f"state dir not writable: {checks['state_dir']['path']}")
    if checks["env"].get("roles_error"):
        reasons.append("LAB_POD_ROLES unreadable: " + checks["env"]["roles_error"])
    if checks["env"]["missing"]:
        reasons.append("env keys missing: " + ", ".join(checks["env"]["missing"]))
    return reasons


def warnings_for(checks: Mapping[str, Any]) -> list[str]:
    """What failed among the advisory checks: one line per hub account that did not answer.

    An account skipped for missing credentials is not repeated here; the env gate already names its keys.
    """
    warnings: list[str] = []
    for letter, probe in checks["hub"]["accounts"].items():
        if probe["ok"] or "skipped" in probe:
            continue
        if "status" not in probe:
            warnings.append(f"hub account {letter} unreachable")
        elif "error" in probe:
            warnings.append(f"hub account {letter} answered {probe['status']} ({probe['error']})")
        else:
            warnings.append(f"hub account {letter} answered {probe['status']}")
    return warnings


def check(
    *,
    environ: Mapping[str, str] | None = None,
    env_files: Sequence[Path] | None = None,
    fetch: Fetch | None = None,
    base_url: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """Run every check and return the verdict; only GATE_CHECKS decide ready. The arguments exist for tests.

    fetch or transport, when given, stand in for the network under lab.hub.Hub (see fetch_transport); base_url
    sends every account to one address instead of its own.
    """
    from lab import hub as lab_hub

    environ = os.environ if environ is None else environ
    files = default_env_files(environ) if env_files is None else list(env_files)
    roles_error = ""
    try:
        roles = lab_hub.pod_roles(environ, files)
    except lab_hub.PodRoleError as exc:
        roles, roles_error = {}, str(exc)
    required = required_env(sorted(roles, key=lambda x: (roles[x], x)))
    wanted = required + tuple(lab_hub.url_key(letter) for letter in roles)
    values, used = collect_env(wanted, environ, files)
    unroled = [letter for letter in credential_letters(environ, files) if letter not in roles] if roles else []
    if transport is None and fetch is not None:
        transport = fetch_transport(fetch, HUB_TIMEOUT_S)
    checks = {
        "stop": check_stop(),
        "state_dir": check_state_dir(),
        "env": check_env(values, used, required, roles_error),
        "hub": check_hub(values, roles, base_url, transport, environ, unroled),
    }
    ready = all(checks[name]["ok"] for name in GATE_CHECKS)
    return {
        "ok": ready,
        "ready": ready,
        "outcome": "ready" if ready else "not_ready",
        "reasons": reasons_for(checks),
        "warnings": warnings_for(checks),
        "checks": checks,
    }


@stop.exempt
def main(argv: Sequence[str]) -> dict[str, Any]:
    """Check what a sitting needs: no STOP, a writable state dir, .env key names; the hub only warns."""
    argparse.ArgumentParser(prog="python -m lab ready", description=main.__doc__).parse_args(argv)
    return check()
