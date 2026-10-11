"""ready: may a sitting start right now? One JSON line; exit 0 when yes, 1 when not.

    python -m lab ready

Every failing reason is listed at once. The check changes nothing a sitting would notice:
its only write is a throwaway file in the state dir, and its only network traffic is one
read-only GET per hub account.

Gates (GATE_CHECKS): any one failing makes ready false, lists it under "reasons", and exits 1.

    stop       no STOP file (lab/machine/stop.py); this verb runs while stopped so it can say so
    state_dir  lab/state (or $LAB_STATE_DIR) exists, is created when missing, and takes a file
    env        the .env keys a sitting needs are present: key NAMES only, a value is never printed

Advisory (every other check): still run and reported under "checks", but a failure is listed
under "warnings" and never changes ready or the exit code. A hub outage must not halt GPU-free
work; a sitting that needs a pod reads checks.hub and the warnings for itself.

    hub        GET <hub>/hub/api/users/<account> answers 200 with a JSON object, per pod account,
               asked once through lab.hub.Hub, the one hub client (token, agent, no redirects, scrubbing)

Keys come from the process environment first, then from .env files: the clone's, then the one
in each of the two folders above it (the Atlas workspace keeps its .env one level above atlas/
and two above work/<ID>/). The first definition wins and a blank value counts as missing.
$LAB_ENV_FILES (paths separated by os.pathsep) replaces that search; $LAB_HUB_URL replaces the
hub address.
"""

from __future__ import annotations

import argparse
import os
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
REQUIRED_ENV = ("AUP_HUB_TOKEN_A", "AUP_HUB_TOKEN_B", "AUP_HUB_USER_A", "AUP_HUB_USER_B")
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


def check_env(values: Mapping[str, str], files: Sequence[Path]) -> dict[str, Any]:
    missing = [key for key in REQUIRED_ENV if key not in values]
    return {
        "ok": not missing,
        "present": [key for key in REQUIRED_ENV if key in values],
        "missing": missing,
        "files": [str(path) for path in files],
    }


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
    """One read-only GET of the account's hub user endpoint. Any failure is a verdict, never a crash."""
    # Imported here, not at the top: lab.hub reads this module's constants when it loads.
    from lab import hub as lab_hub

    account = lab_hub.Account(letter, user, token)
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
        return {"ok": False, "error": scrub(f"{type(exc).__name__}: {exc}", hide)}
    if reply.status != HTTP_OK:
        return {"ok": False, "status": reply.status}
    if not isinstance(reply.body, dict):
        return {"ok": False, "status": reply.status, "error": "response is not a JSON object"}
    return {"ok": True, "status": reply.status}


def check_hub(
    values: Mapping[str, str],
    base_url: str,
    transport: httpx.BaseTransport | None,
    environ: Mapping[str, str],
) -> dict[str, Any]:
    hide = sorted((value for value in values.values() if len(value) >= MIN_HIDDEN_CHARS), key=len, reverse=True)
    accounts: dict[str, dict[str, Any]] = {}
    for letter in ACCOUNTS:
        user, token = values.get(f"AUP_HUB_USER_{letter}"), values.get(f"AUP_HUB_TOKEN_{letter}")
        if user and token:
            accounts[letter] = probe_account(
                letter, user, token, base_url=base_url, transport=transport, environ=environ, hide=hide
            )
        else:
            accounts[letter] = {"ok": False, "skipped": "credentials missing"}
    return {"ok": all(probe["ok"] for probe in accounts.values()), "url": base_url, "accounts": accounts}


def reasons_for(checks: Mapping[str, Any]) -> list[str]:
    """Why a sitting may not start: one line per failing gate check (GATE_CHECKS)."""
    reasons: list[str] = []
    if not checks["stop"]["ok"]:
        reasons.append(f"STOP file present: {checks['stop']['reason']}")
    if not checks["state_dir"]["ok"]:
        reasons.append(f"state dir not writable: {checks['state_dir']['path']}")
    if not checks["env"]["ok"]:
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
) -> dict[str, Any]:
    """Run every check and return the verdict; only GATE_CHECKS decide ready. The arguments exist for tests.

    fetch, when given, stands in for the network under lab.hub.Hub (see fetch_transport).
    """
    environ = os.environ if environ is None else environ
    files = default_env_files(environ) if env_files is None else list(env_files)
    base_url = base_url or environ.get(HUB_URL_ENV) or HUB_URL
    values, used = collect_env(REQUIRED_ENV, environ, files)
    checks = {
        "stop": check_stop(),
        "state_dir": check_state_dir(),
        "env": check_env(values, used),
        "hub": check_hub(values, base_url, None if fetch is None else fetch_transport(fetch, HUB_TIMEOUT_S), environ),
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
