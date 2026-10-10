"""controller: launch, stop, watch and keep alive the AUP pods (one GPU per hub account).

    python -m lab.controller status [--account LETTER|both] [--probe-port PORT]
    python -m lab.controller quota [--account LETTER|both] [--transactions]
    python -m lab.controller launch --account LETTER --minutes N [--accelerator NAME] [--resource NAME] [--no-wait]
                                    [--timeout S] [--dry-run] [--reason TEXT]
    python -m lab.controller stop --account LETTER [--no-wait] [--timeout S] [--reason TEXT]
    python -m lab.controller wait --account LETTER [--timeout S]
    python -m lab.controller keep-alive [--account LETTER|both] [--interval S] [--minutes N]
    python -m lab.controller run-script --account LETTER SCRIPT [ARG ...] [--timeout S]

An account is any single letter whose AUP_HUB_USER_<LETTER> and AUP_HUB_TOKEN_<LETTER> keys exist: A and B on the
DUB hub, C and D on the SJC hub. `both` means A and B (the default of status, quota and keep-alive); the SJC pair
is asked for by letter.

Run it from the repo root (`python lab/controller.py ...` works from any folder). Every call answers with one JSON
object on stdout and an exit code, as the verbs of `python -m lab` do: 0 when it worked, 1 when it did not (a
refused launch, a hub that does not answer, a pod that was lost, the kill switch), 2 for bad arguments and for
missing keys. The same commands are methods of Controller, for the scripts that drive pods (benchmarks, the
recovery drill, the queue): they get the same ledger rows and the same checks.

The hub. Launch is `POST /hub/api/users/<account>/server` with {runtime_minutes, resource_type: gpu,
gpu_selection: r9700}; stop is `DELETE` on the same path; state is `GET /hub/api/users/<account>` and the balance
`GET /hub/api/quota/me`. The hub enforces no ceiling on runtime_minutes, so --minutes takes any whole number; the
session is charged when it ends, 4 credits for each whole minute, so a launch whose cost the balance cannot cover
is refused here, with the longest runtime it can, before anything is sent. A hub that keeps no accounts (quota
disabled: balance 0, unlimited, enabled false, as the SJC hub answers) refuses nothing: the launch goes ahead and
the answer says "quota disabled on this hub". The keys are AUP_HUB_USER_<LETTER> and AUP_HUB_TOKEN_<LETTER>, found
as lab/machine/ready.py finds them, and each account lives on the hub at its AUP_HUB_URL_<LETTER> when that is set,
else at $LAB_HUB_URL, else on the DUB hub (lab/hub.py); no key, token or account name is ever printed, logged or
written to the ledger.

Two ways to launch. AUP_HUB_LAUNCH_<LETTER> picks one per account. `rest` is the call above. `form` is the hub's own
web page (Hub.spawn_form in lab/hub.py): log in at /hub/native/login with the user name and AUP_HUB_PASS_<LETTER>
(cookies and the _xsrf token), POST /hub/spawn/<account>?_xsrf=<cookie> with the page's fields (resource_type,
gpu_selection_<resource_type>, runtime in minutes, repo_url, repo_persist, repo_branch), then the same poll of the API
until the pod is ready. The SJC hub accepts the REST call and echoes its options but ignores them: the pod comes up as
the stock singleuser image with no GPU and no ROCm, and only the form gets ghcr.io/amdresearch/auplc-base. `auto`, the
default, launches by REST as before and, when the account has a password, asks the ready pod which image it runs (a
small script in a terminal reads JUPYTER_IMAGE_SPEC and /dev/kfd); a pod found to be the stock image is stopped and
launched again through the form, and a pod that cannot be asked is left as it is and marked image_check unknown.
--accelerator and --resource name the card and the catalog entry; their defaults are AUP_HUB_GPU_<LETTER> and
AUP_HUB_RESOURCE_<LETTER>, else r9700 and gpu. The answer and the pod row of a launch say which path was used
(launch_path) and the accelerator. The password is read like the other keys and is cut out of every answer, error
and ledger row, as the token is.

Safe to repeat. A launch of a pod that is ready answers already_running and sends nothing; a launch while a
spawn is pending joins it; a launch while a stop is pending waits for the stop and then launches; the hub's own
400 "already running" counts as success. A stop of an account with no server answers already_stopped and sends
nothing. A POST the hub answered with a gateway error is never sent again: the state is read back, and a launch
that did not start is reported as launch_unconfirmed.

The idle culler. The hub stops a server whose last activity is an hour old and checks every ten minutes. Any request
through the pod's address counts as activity and any read of the hub API does not, so `status` is free to run as
often as wanted and `keep-alive` pings `GET /user/<account>/api/me` every 10 minutes (--interval, 30 to 1800
seconds) for --minutes, or until the pod is gone. That call passes the proxy, which records route activity, and is
one the Jupyter server records as well (its `/api/status` is the one it deliberately does not). A keep-alive on an
idle pod still bills 240 credits an hour: bound it with --minutes. It re-reads the hub each cycle, so a pod that
vanished is reported as lost within one interval; it relaunches nothing. `status --probe-port` asks the model
server through the proxy and so counts as activity.

Time. Durations and the scheduled end (start plus runtime_minutes) are computed on the hub's clock, read from
the Date header of its answers, because the hub's clock is not the caller's; hub_clock_offset_s says by how much.

The kill switch. While lab/state/STOP exists (lab/machine/stop.py) launch, stop, keep-alive and run-script are
refused with outcome stopped, and a keep-alive that is running ends at its next cycle. status, quota and wait only
read, and run on.

The ledger. A command writes a row for the call (verbs `controller launch`, `controller stop`, ... through
@recorded: outcome, start, end, duration) and, where it learned or changed something about a pod, a pod row in the
words the alert rules read (lab/machine/alerts.py): verbs `pod launch`, `pod stop`, `pod status`, `pod wait`,
`pod keep-alive`, `pod quota` and `pod run-script`, with node pod-a or pod-b and refs.account. The metrics on them
are pod.start_s (the launch request to a ready pod) and pod.balance_credits. A script run on a pod also leaves its
row here, because a pod's own ledger is never read. A ledger that cannot be written never undoes what the hub
did: the answer then carries ledger_error.

Outcomes. launch: running, already_running, launching (--no-wait), refused_quota, refused, timeout, failed, error,
dry_run. stop: ok, already_stopped, stopping (--no-wait), timeout, error. wait: ready, timeout, not_running, error.
keep-alive: completed, lost, not_running, unreachable, stopped. run-script: ok, fail, timeout, not_running, error.
An error carries kind (hub_unreachable, launch_unconfirmed, unexpected_status, terminal_unreachable, ...). A launch
through the form also has login_failed, no_credentials (exit 2), xsrf_missing, form_unavailable and spawn_refused.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import re
import secrets
import shlex
import sys
import time
import traceback
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):          # started as `python lab/controller.py`: put the repo root on the path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab import hub as hubs
from lab.hub import Account, Hub, HubUnreachable, MissingCredentials, Reply, load_accounts, one_line
from lab.machine import record
from lab.machine import stop as kill_switch
from lab.machine.record import recorded

__all__ = ["Controller", "Server", "Quota", "run", "main", "build_parser"]

log = logging.getLogger("lab.controller")

RESOURCE_TYPE = "gpu"
ACCELERATOR = "r9700"
NO_ANSWER = 0                       # the status of a Reply for a call that never got one
POLL_S = 1.0                        # seconds between reads while a pod launches: the start time is only as fine as this
STOP_POLL_S = 2.0                   # seconds between reads while a pod stops
POST_TIMEOUT_S = 60.0
LAUNCH_TIMEOUT_S = 420.0            # the hub itself gives a spawn 300 s
STOP_TIMEOUT_S = 180.0
WAIT_TIMEOUT_S = 420.0
VANISH_GRACE_S = 15.0               # how long a launching server may be missing from the hub's list before it counts as gone
CONFIRM_READS = 3                   # reads that settle whether a POST the hub answered badly started a server
CONFIRM_PAUSE_S = 3.0               # seconds between those reads
KEEP_ALIVE_INTERVAL_S = 600
KEEP_ALIVE_MIN_S = 30
KEEP_ALIVE_MAX_S = 1800             # the culler waits an hour and checks every ten minutes: stay well inside
PING_PATH = "/api/me"                # a tracked API call; /api/status is the one the Jupyter server does not count
SCRIPT_TIMEOUT_S = 900.0
SCRIPT_POLL_S = 5.0
END_MARKER = "### END"
TAIL_CHARS = 4000
TERMINAL_OPEN_S = 15.0
TERMINAL_SETTLE_S = 3.0             # real seconds to wait for the shell to echo a command before moving on
NOTE_CHARS = 300
MIN_NAME_CHARS = 4                  # a shorter account name would mangle the text it is cut out of
IMAGE_CHECK_S = 60.0                # how long a ready pod has to answer the question which image it runs
STOCK_IMAGE = "jupyterhub/singleuser"   # the name the hub's default image carries: the pod a REST launch gets on SJC
# What a ready pod is asked: the image KubeSpawner records in its environment and whether the GPU device is there.
IMAGE_SCRIPT = ('echo "image=${JUPYTER_IMAGE_SPEC:-${JUPYTER_IMAGE:-}}"\n'
                'if [ -e /dev/kfd ]; then echo "gpu_device=yes"; else echo "gpu_device=no"; fi\n')

QUOTA_DISABLED = "quota disabled on this hub"
POD_LAUNCH, POD_STOP, POD_STATUS = "pod launch", "pod stop", "pod status"
POD_WAIT, POD_KEEP_ALIVE, POD_QUOTA, POD_RUN_SCRIPT = "pod wait", "pod keep-alive", "pod quota", "pod run-script"
BALANCE = "pod.balance_credits"
START = "pod.start_s"


# --- what the hub says about a server ---------------------------------------------------------

def parse_time(value: Any) -> dt.datetime | None:
    """A hub timestamp as an aware UTC time; None when it is missing or unreadable."""
    if not isinstance(value, str) or not value:
        return None
    try:
        moment = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=dt.timezone.utc)


def iso_z(moment: dt.datetime | None) -> str | None:
    """A time as UTC text to the second, ending in Z."""
    if moment is None:
        return None
    return moment.astimezone(dt.timezone.utc).replace(microsecond=0, tzinfo=None).isoformat() + "Z"


def _whole(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return None


@dataclass(frozen=True)
class Server:
    """An account's default server as the hub lists it."""

    ready: bool
    pending: str | None
    started: dt.datetime | None
    last_activity: dt.datetime | None
    runtime_minutes: int | None
    resource_type: str | None
    accelerator: str | None

    @classmethod
    def from_model(cls, raw: Mapping[str, Any]) -> "Server":
        options = raw.get("user_options") if isinstance(raw.get("user_options"), Mapping) else {}
        pending = raw.get("pending")
        return cls(ready=bool(raw.get("ready")), pending=pending if isinstance(pending, str) and pending else None,
                   started=parse_time(raw.get("started")), last_activity=parse_time(raw.get("last_activity")),
                   runtime_minutes=_whole(options.get("runtime_minutes")),
                   resource_type=options.get("resource_type") if isinstance(options.get("resource_type"), str) else None,
                   accelerator=options.get("gpu_selection") if isinstance(options.get("gpu_selection"), str) else None)

    @property
    def up(self) -> bool:
        """Ready and not in the middle of anything."""
        return self.ready and self.pending is None

    @property
    def scheduled_end(self) -> dt.datetime | None:
        if self.started is None or self.runtime_minutes is None:
            return None
        return self.started + dt.timedelta(minutes=self.runtime_minutes)

    def minutes_run(self, now: dt.datetime) -> float | None:
        return None if self.started is None else max(0.0, (now - self.started).total_seconds() / 60)

    def state(self) -> str:
        """The word the ledger rows use: running, launching or stopping."""
        if self.pending == "stop":
            return "stopping"
        return "running" if self.up else "launching"

    def view(self, now: dt.datetime) -> dict[str, Any]:
        """The server as the answers show it. Times are the hub's."""
        run = self.minutes_run(now)
        end = self.scheduled_end
        idle = None if self.last_activity is None else max(0.0, (now - self.last_activity).total_seconds() / 60)
        left = None if end is None else max(0.0, (end - now).total_seconds() / 60)
        return {"running": True, "ready": self.ready, "pending": self.pending, "started": iso_z(self.started),
                "runtime_minutes": self.runtime_minutes, "scheduled_end": iso_z(end),
                "minutes_run": None if run is None else round(run, 1),
                "minutes_left": None if left is None else round(left, 1),
                "last_activity": iso_z(self.last_activity), "idle_minutes": None if idle is None else round(idle, 1),
                "resource_type": self.resource_type, "accelerator": self.accelerator}


def view_of(server: Server | None, now: dt.datetime) -> dict[str, Any]:
    return {"running": False} if server is None else server.view(now)


@dataclass(frozen=True)
class Quota:
    """What `GET /hub/api/quota/me` says. enabled False means the hub does no accounting (and says balance 0)."""

    balance: int
    unlimited: bool
    enabled: bool
    rates: Mapping[str, Any]

    @property
    def billed(self) -> bool:
        return self.enabled and not self.unlimited

    def rate(self, accelerator: str | None) -> int | None:
        value = self.rates.get(accelerator or ACCELERATOR)
        return value if isinstance(value, int) and not isinstance(value, bool) else None


def credit_figures(server: Server | None, quota: Quota, rate: int | None, now: dt.datetime) -> dict[str, Any]:
    """What a running session has cost and will cost. The hub charges when a session ends, whole minutes at the rate,
    so the balance does not move while the pod runs and these are worked out from the start time."""
    if server is None or not quota.billed or rate is None:
        return {}
    figures: dict[str, Any] = {}
    run = server.minutes_run(now)
    if run is not None:
        so_far = int(run) * rate
        figures["credits_so_far"] = so_far
        figures["balance_if_stopped_now"] = max(0, quota.balance - so_far)
    if server.runtime_minutes:
        whole = server.runtime_minutes * rate
        figures["credits_if_run_to_end"] = whole
        figures["balance_at_scheduled_end"] = max(0, quota.balance - whole)
        figures["exceeds_balance"] = whole > quota.balance
    return figures


@dataclass
class Seen:
    """One read of an account's state. ok False means the hub gave no usable answer (status 0: none at all)."""

    ok: bool
    server: Server | None = None
    now: dt.datetime | None = None
    status: int | None = None
    error: str | None = None


@dataclass
class Watch:
    """One account in a keep-alive."""

    letter: str
    account: Account
    state: str = "waiting"          # alive, failing, launching, lost, not_running
    pings: int = 0
    failures: int = 0
    streak: int = 0
    last_ping: dt.datetime | None = None
    reported: str | None = None
    scheduled_end: dt.datetime | None = None
    seen_up: bool = False
    active: bool = True
    note: str | None = None

    def summary(self) -> dict[str, Any]:
        entry: dict[str, Any] = {"node": self.account.node, "state": self.state, "pings": self.pings,
                                 "failures": self.failures, "last_ping": iso_z(self.last_ping)}
        if self.note:
            entry["note"] = self.note
        return entry


# --- which image a pod runs ---------------------------------------------------------------------

def parse_image_answer(output: str) -> dict[str, Any]:
    """The image and the GPU device a pod reported when it ran IMAGE_SCRIPT: {image: text or None, gpu_device:
    True, False or None}. A line that is not one of the two is ignored."""
    values = {key: value.strip() for key, _, value in (line.partition("=") for line in output.splitlines())
              if key in ("image", "gpu_device")}
    return {"image": values.get("image") or None,
            "gpu_device": {"yes": True, "no": False}.get(values.get("gpu_device", ""))}


def image_verdict(image: str | None, gpu_device: bool | None, resource_type: str) -> str:
    """Whether a pod is the stock image the hub falls back to (stock), the one that was asked for (ok) or cannot
    be told (unknown). The image name settles it; with no name, a pod without the GPU device is the stock image
    when the resource is not a cpu one."""
    if image:
        return "stock" if STOCK_IMAGE in image.lower() else "ok"
    if gpu_device is True:
        return "ok"
    if gpu_device is False and not resource_type.strip().lower().endswith("cpu"):
        return "stock"
    return "unknown"


# --- the controller ---------------------------------------------------------------------------

class Controller:
    """The pods of the given accounts. transport replaces the network (an httpx transport), sleep and clock the time:
    a rehearsal passes a fake hub and a fake clock, so nothing sleeps or sends for real."""

    def __init__(self, accounts: Mapping[str, Account], *, base_url: str | None = None, transport: Any = None,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.time,
                 poll_s: float = POLL_S, environ: Mapping[str, str] | None = None) -> None:
        self.accounts = dict(accounts)
        self.sleep = sleep
        self.clock = clock
        self.poll_s = poll_s
        self._base_url = base_url
        self._transport = transport
        self._environ = environ
        self._hubs: dict[str, Hub] = {}
        self._offset_s: float | None = None

    def __enter__(self) -> "Controller":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        for client in self._hubs.values():
            client.close()
        self._hubs.clear()

    # --- plumbing ---------------------------------------------------------------------------

    @property
    def base_url(self) -> str:
        """The address of the hub of the first account (the accounts may live on different hubs, see hub_urls)."""
        return self._hub(next(iter(self.accounts))).base_url

    def hub_urls(self, letters: Sequence[str] | None = None) -> dict[str, str]:
        """The address of the hub each of those accounts lives on."""
        return {letter: self._hub(letter).base_url for letter in self._letters(letters)}

    def _account(self, letter: str) -> Account:
        try:
            return self.accounts[letter]
        except KeyError:
            raise KeyError(f"unknown account {letter!r}; the accounts here are {', '.join(sorted(self.accounts))}") from None

    def _letters(self, letters: Sequence[str] | None) -> list[str]:
        if letters is None:
            return sorted(self.accounts)
        for letter in letters:
            self._account(letter)
        return list(letters)

    def client(self, letter: str) -> Hub:
        """The Hub client of an account, for the calls the controller has no command for."""
        return self._hub(letter)

    def _hub(self, letter: str) -> Hub:
        client = self._hubs.get(letter)
        if client is None:
            client = Hub(self._account(letter), base_url=self._base_url, transport=self._transport, sleep=self.sleep,
                         environ=self._environ)
            self._hubs[letter] = client
        return client

    def _send(self, letter: str, method: str, path: str, **kwargs: Any) -> Reply:
        """One call to the hub; a call that gets no answer comes back as a Reply with status NO_ANSWER."""
        try:
            reply = self._hub(letter).request(method, path, **kwargs)
        except HubUnreachable as exc:
            return Reply(status=NO_ANSWER, text=self.scrub(str(exc)))
        if reply.hub_time is not None:
            self._offset_s = round((reply.hub_time - self._now()).total_seconds(), 1)
        return reply

    def _now(self) -> dt.datetime:
        return dt.datetime.fromtimestamp(self.clock(), tz=dt.timezone.utc)

    def scrub(self, text: str) -> str:
        """text with every token of these accounts cut out and every account name made <account A> or <account B>."""
        for account in self.accounts.values():
            if len(account.token) >= hubs.MIN_HIDDEN_CHARS:
                text = text.replace(account.token, "***")
            if account.password and len(account.password) >= hubs.MIN_HIDDEN_CHARS:
                text = text.replace(account.password, "***")
            if len(account.user) >= MIN_NAME_CHARS:
                text = text.replace(account.user, f"<account {account.letter}>")
        return text

    @staticmethod
    def _guard() -> None:
        """The kill switch: nothing that changes a pod runs while lab/state/STOP exists."""
        if kill_switch.stopped():
            raise kill_switch.Stopped(kill_switch.stop_reason() or kill_switch.NO_REASON)

    def _look(self, letter: str) -> Seen:
        """Read the account's default server from the hub API (which does not count as activity)."""
        client = self._hub(letter)
        reply = self._send(letter, "GET", client.user_url())
        if not reply.ok or not isinstance(reply.body, Mapping):
            return Seen(False, status=reply.status, error=self._failure_text(reply))
        raw = (reply.body.get("servers") or {}).get("")
        server = Server.from_model(raw) if isinstance(raw, Mapping) else None
        return Seen(True, server, reply.hub_time or self._now(), reply.status)

    def _failure_text(self, reply: Reply) -> str:
        if reply.status == NO_ANSWER:
            return f"the hub did not answer: {reply.text}"
        return self.scrub(f"the hub answered {reply.status}: {reply.detail()}")

    def _quota(self, letter: str) -> Quota | None:
        return self._quota_from(self._send(letter, "GET", "/hub/api/quota/me"))

    @staticmethod
    def _quota_from(reply: Reply) -> Quota | None:
        """The Quota in a `GET /hub/api/quota/me` answer; None when the call failed or the answer is not one."""
        body = reply.body
        if not reply.ok or not isinstance(body, Mapping):
            return None
        balance = _whole(body.get("balance"))
        if balance is None:
            return None
        rates = body.get("rates") if isinstance(body.get("rates"), Mapping) else {}
        return Quota(balance, bool(body.get("unlimited")), bool(body.get("enabled", True)), rates)

    def _trouble(self, seen: Seen) -> dict[str, Any]:
        """The part of an answer for a hub that could not be read."""
        answer: dict[str, Any] = {"ok": False, "outcome": "error", "kind": "hub_unreachable", "error": seen.error}
        if seen.status:
            answer["status"] = seen.status
        return answer

    def _pod_row(self, verb: str, outcome: str, account: Account, *, metrics: Mapping[str, Any] | None = None,
                 units: Mapping[str, str] | None = None, conditions: Mapping[str, Any] | None = None,
                 note: str | None = None, refs: Mapping[str, Any] | None = None) -> str | None:
        """Append the ledger row about a pod, with its node. Returns why the row could not be written, else None."""
        fields: dict[str, Any] = {"node": account.node}
        if metrics and units:
            fields["units"] = {name: unit for name, unit in units.items() if name in metrics}
        if conditions:
            fields["conditions"] = dict(conditions)
        if note:
            fields["note"] = one_line(self.scrub(str(note)), NOTE_CHARS)
        try:
            record.record(verb, outcome, dict(metrics) if metrics else None, {"account": account.letter, **(refs or {})}, **fields)
        except Exception as exc:
            log.warning("no ledger row written for %s (%s): %s: %s", verb, outcome, type(exc).__name__, exc)
            return f"{type(exc).__name__}: {exc}"
        return None

    def _row(self, answer: dict[str, Any], verb: str, account: Account, *, metrics: Mapping[str, Any] | None = None,
             units: Mapping[str, str] | None = None, conditions: Mapping[str, Any] | None = None,
             note: str | None = None, refs: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Write the pod row for a finished command and give the call's own row its refs."""
        answer["refs"] = {"account": account.letter, **(refs or {})}
        error = self._pod_row(verb, answer["outcome"], account, metrics=metrics, units=units, conditions=conditions,
                              note=note, refs=refs)
        if error:
            answer["ledger_error"] = error
        return answer

    @staticmethod
    def _whole_minutes(value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"the runtime is a whole number of minutes, 1 or more, not {value!r}")
        return value

    # --- polling ----------------------------------------------------------------------------

    def _await_ready(self, letter: str, timeout_s: float, since: float) -> tuple[str, Seen]:
        """Read the account until its server is ready. Returns ready, timeout, vanished (the server left the list
        while launching) or cancelled (someone stopped it), with the last read."""
        deadline = since + timeout_s
        while True:
            seen = self._look(letter)
            if seen.ok:
                if seen.server is None:
                    if self.clock() - since > VANISH_GRACE_S:
                        return "vanished", seen
                elif seen.server.pending == "stop":
                    return "cancelled", seen
                elif seen.server.up:
                    return "ready", seen
            if self.clock() >= deadline:
                return "timeout", seen
            self.sleep(self.poll_s)

    def _await_gone(self, letter: str, timeout_s: float) -> tuple[str, Seen]:
        """Read the account until it has no server. Returns gone or timeout, with the last read."""
        deadline = self.clock() + timeout_s
        while True:
            seen = self._look(letter)
            if seen.ok and seen.server is None:
                return "gone", seen
            if self.clock() >= deadline:
                return "timeout", seen
            self.sleep(STOP_POLL_S)

    def _confirm_started(self, letter: str) -> Seen:
        """After a POST whose answer cannot be trusted: read the account a few times to see whether a server appeared."""
        seen = self._look(letter)
        for _ in range(CONFIRM_READS - 1):
            if seen.ok and seen.server is not None:
                break
            self.sleep(CONFIRM_PAUSE_S)
            seen = self._look(letter)
        return seen

    # --- launch -----------------------------------------------------------------------------

    @recorded("controller launch")
    def launch(self, letter: str, minutes: int, *, wait: bool = True, timeout_s: float = LAUNCH_TIMEOUT_S,
               dry_run: bool = False, resource_type: str | None = None, accelerator: str | None = None,
               reason: str | None = None) -> dict[str, Any]:
        """Launch the account's pod for `minutes` and wait until it is ready (a no-op when it already runs).

        resource_type and accelerator default to the account's AUP_HUB_RESOURCE_<letter> and AUP_HUB_GPU_<letter>,
        else gpu and r9700. The account's AUP_HUB_LAUNCH_<letter> says how the hub is asked: rest (the API call),
        form (the hub's web spawn form) or auto, the default (the API call, then a look at the image the pod runs
        when the account has a password, and the form if it is the stock one; --no-wait skips the look)."""
        self._guard()
        account = self._account(letter)
        minutes = self._whole_minutes(minutes)
        resource_type = (account.resource or RESOURCE_TYPE) if resource_type is None else resource_type
        accelerator = (account.gpu or ACCELERATOR) if accelerator is None else accelerator
        if not accelerator.strip() or accelerator.strip().lower() == "auto":
            raise ValueError("the accelerator is named: auto picks the cheapest card the hub has free, not the R9700")
        if not resource_type.strip():
            raise ValueError("the resource is named: gpu is the Basic GPU Environment")
        mode = self._launch_mode(account)
        via_form = mode == "form"
        payload = {"runtime_minutes": minutes, "resource_type": resource_type, "gpu_selection": accelerator}
        client = self._hub(letter)
        base: dict[str, Any] = {"command": "launch", "account": letter, "node": account.node,
                                "runtime_minutes": minutes, "resource_type": resource_type, "accelerator": accelerator}
        conditions = {"resource": resource_type, "accelerator": accelerator}
        quota: Quota | None = None

        def finish(answer: dict[str, Any], *, start_s: float | None = None, write: bool = True) -> dict[str, Any]:
            answer = {**base, **answer}
            if quota is not None and quota.enabled:
                answer.setdefault("balance", quota.balance)
            if quota is not None and not quota.enabled:
                answer.setdefault("note", QUOTA_DISABLED)
            if not write:
                return answer
            metrics: dict[str, Any] = {}
            if start_s is not None:
                metrics[START] = start_s
            if quota is not None and quota.enabled:
                metrics[BALANCE] = quota.balance
            note = " | ".join(part for part in (reason, answer.get("error")) if part)
            return self._row(answer, POD_LAUNCH, account, metrics=metrics, units={START: "s", BALANCE: "credits"},
                             conditions=conditions, note=note or None)

        if via_form and not account.password:
            key = hubs.password_key(letter)
            return finish({"ok": False, "outcome": "refused", "kind": "no_credentials", "missing": [key],
                           "error": f"the form launch logs in with the account's password and {key} is not set"})
        seen = self._look(letter)
        if not seen.ok:
            return finish(self._trouble(seen))
        server = seen.server
        joined = False
        if server is not None and server.pending == "stop":
            outcome, seen = self._await_gone(letter, STOP_TIMEOUT_S)
            if outcome != "gone":
                return finish({"ok": False, "outcome": "timeout", "kind": "stop_pending",
                               "error": "the server is still stopping; nothing was launched",
                               "server": view_of(seen.server, seen.now or self._now())})
            server = seen.server
        if server is not None and server.up:
            return finish({"ok": True, "outcome": "already_running", "server": server.view(seen.now or self._now())})
        if server is not None and server.pending == "spawn":
            joined = True                       # a spawn is already under way: wait for it instead of starting another

        rate: int | None = None
        cost: int | None = None
        if not joined:
            quota = self._quota(letter)
            if quota is not None:
                rate = quota.rate(accelerator)
                if quota.billed and rate:
                    cost = minutes * rate
                    if quota.balance < cost:
                        most = quota.balance // rate
                        return finish({"ok": False, "outcome": "refused_quota", "rate": rate, "cost_credits": cost,
                                       "max_minutes": most,
                                       "error": (f"the balance of {quota.balance} credits cannot cover {minutes} minutes "
                                                 f"at {rate} credits a minute ({cost}); it covers {most}")})
            if dry_run:
                plan = ({"launch_path": "form", "login": hubs.LOGIN_PATH, "method": "POST", "path": client.spawn_url(),
                         "form": hubs.spawn_fields(minutes, resource_type, accelerator)} if via_form else
                        {"method": "POST", "path": client.user_url("/server"), "json": payload})
                return finish({"ok": True, "outcome": "dry_run", "rate": rate, "cost_credits": cost, "plan": plan},
                              write=False)

        posted_at = self.clock()
        post_status: int | None = None
        if not joined and via_form:
            base["launch_path"] = conditions["launch_path"] = "form"
            failure = self._submit_form(letter, minutes, resource_type, accelerator)
            if failure is not None:
                return finish(failure)
        elif not joined:
            base["launch_path"] = conditions["launch_path"] = "rest"
            reply = self._send(letter, "POST", client.user_url("/server"), json=payload, timeout=POST_TIMEOUT_S)
            if reply.status == 400 and "pending stop" in reply.detail().lower():
                outcome, seen = self._await_gone(letter, STOP_TIMEOUT_S)
                if outcome != "gone":
                    return finish({"ok": False, "outcome": "timeout", "kind": "stop_pending",
                                   "error": "the server is still stopping; nothing was launched"})
                posted_at = self.clock()
                reply = self._send(letter, "POST", client.user_url("/server"), json=payload, timeout=POST_TIMEOUT_S)
            post_status = reply.status
            if reply.status == 400 and "already running" in reply.detail().lower():
                seen = self._look(letter)
                return finish({"ok": True, "outcome": "already_running",
                               "server": view_of(seen.server, seen.now or self._now())})
            if reply.status == 403:
                text = self.scrub(reply.detail())
                return finish({"ok": False, "outcome": "refused_quota" if "quota" in text.lower() else "refused",
                               "kind": "launch_refused", "error": text, "status": 403, "rate": rate, "cost_credits": cost})
            if reply.status == NO_ANSWER or reply.status >= 500:
                seen = self._confirm_started(letter)
                if not (seen.ok and seen.server is not None):
                    return finish({"ok": False, "outcome": "error", "kind": "launch_unconfirmed",
                                   "status": reply.status, "post_status": reply.status,
                                   "error": (f"the launch got no usable answer ({self._failure_text(reply)}) and no server "
                                             "appeared; it was not sent again")})
            elif reply.status not in (200, 201, 202):
                return finish({"ok": False, "outcome": "error", "kind": "unexpected_status", "status": reply.status,
                               "error": self._failure_text(reply)})

        extra = {"rate": rate, "cost_credits": cost}
        if post_status not in (None, 200, 201, 202):
            extra["post_status"] = post_status
        if not wait:
            seen = self._look(letter)
            return finish({"ok": True, "outcome": "launching", **extra,
                           "server": view_of(seen.server, seen.now or self._now())})
        outcome, seen = self._await_ready(letter, timeout_s, posted_at)
        ready_s = self.clock()                  # the start time stops here, before the pod is asked which image it runs
        if outcome == "ready" and mode == "auto" and account.password and base.get("launch_path") == "rest":
            checked, failure, outcome, seen = self._vet_rest_pod(letter, minutes, resource_type, accelerator, timeout_s,
                                                                 seen)
            base.update(checked)
            if "fell_back_from" in checked:
                base["launch_path"] = conditions["launch_path"] = "form"
                conditions["fell_back_from"] = checked["fell_back_from"]
                ready_s = self.clock()          # the pod that counts is the one the form started
            if failure is not None:
                return finish(failure)
        view = view_of(seen.server, seen.now or self._now())
        if outcome == "ready":
            start_s = None if joined else round(ready_s - posted_at, 3)
            answer = {"ok": True, "outcome": "running", **extra, "server": view}
            if start_s is not None:
                answer["start_s"] = start_s
            return finish(answer, start_s=start_s)
        if outcome == "timeout":
            return finish({"ok": False, "outcome": "timeout", **extra, "server": view,
                           "error": f"the pod was not ready after {timeout_s:g} s; it was left as it is"})
        reason_text = ("the server left the hub's list while launching: the spawn failed or was cancelled"
                       if outcome == "vanished" else "the server was stopped while launching")
        return finish({"ok": False, "outcome": "failed", "kind": outcome, **extra, "server": view, "error": reason_text})

    @staticmethod
    def _launch_mode(account: Account) -> str:
        """How the account is launched: form, rest or auto (AUP_HUB_LAUNCH_<letter>; auto when it is not set)."""
        mode = (account.launch or "auto").strip().lower()
        if mode not in hubs.LAUNCH_MODES:
            raise ValueError(f"AUP_HUB_LAUNCH_{account.letter} is {account.launch!r}: use one of "
                             f"{', '.join(hubs.LAUNCH_MODES)}")
        return mode

    def _submit_form(self, letter: str, minutes: int, resource_type: str, accelerator: str) -> dict[str, Any] | None:
        """Ask the hub for the pod through its web spawn form. None when the hub accepted the spawn (the caller then
        waits for the pod); else the answer that says why not. A spawn the hub answered badly is never sent again:
        the state is read back, as for the REST call."""
        result = self._hub(letter).spawn_form(minutes, resource_type, accelerator)
        if result.ok:
            return None
        error = self.scrub(result.error or f"the {result.stage} step failed")
        answer: dict[str, Any] = {
            "ok": False, "outcome": "refused" if result.kind in ("login_failed", "spawn_refused") else "error",
            "kind": "hub_unreachable" if result.kind == "no_answer" else result.kind, "stage": result.stage,
            "status": result.status, "error": error}
        if result.stage == "spawn" and (result.status == NO_ANSWER or result.status >= 500):
            seen = self._confirm_started(letter)
            if seen.ok and seen.server is not None:
                return None
            answer.update(outcome="error", kind="launch_unconfirmed",
                          error=f"the form launch got no usable answer ({error}) and no server appeared; it was not "
                                "sent again")
        return answer

    def _shell_on_pod(self, letter: str, stem: str, text: str, timeout_s: float) -> tuple[str | None, str | None]:
        """Run a shell script on the ready pod in a terminal, the way run-script does, and wait for it to end.

        The script and its output are files in the Jupyter root (lab_<stem>_<id>.sh and .out); the output is read
        back until the line `### END rc=<code>` that the command appends. Returns (the output, None), or (None, why
        it could not be run or did not end)."""
        client = self._hub(letter)
        run_id = f"{self._now():%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"
        remote, out = f"lab_{stem}_{run_id}.sh", f"lab_{stem}_{run_id}.out"
        for name, content in ((remote, text), (out, "")):
            reply = self._send(letter, "PUT", client.pod_url(f"/api/contents/{name}"),
                               json={"type": "file", "format": "text", "content": content}, timeout=120)
            if reply.status not in (200, 201):
                return None, f"could not upload {name}: {self._failure_text(reply)}"
        terminal = self._send(letter, "POST", client.pod_url("/api/terminals"), json={})
        name = terminal.body.get("name") if terminal.ok and isinstance(terminal.body, Mapping) else None
        if not isinstance(name, str):
            return None, f"could not open a terminal: {self._failure_text(terminal)}"
        command = f"bash ./{remote} > ./{out} 2>&1; echo \"{END_MARKER} rc=$?\" >> ./{out}"
        try:
            self._type_into_terminal(client, name, command, remote)
        except HubUnreachable as exc:
            return None, self.scrub(str(exc))
        deadline = self.clock() + timeout_s
        output = ""
        while True:
            reply = self._send(letter, "GET", client.pod_url(f"/api/contents/{out}"),
                               params={"content": "1", "format": "text"})
            if reply.ok and isinstance(reply.body, Mapping) and isinstance(reply.body.get("content"), str):
                output = reply.body["content"]
            if re.search(rf"^{re.escape(END_MARKER)} rc=-?\d+[ \t]*$", output, re.MULTILINE):
                return self.scrub(output), None
            if self.clock() >= deadline:
                return None, f"the script did not end within {timeout_s:g} s"
            self.sleep(self.poll_s)

    def _vet_rest_pod(self, letter: str, minutes: int, resource_type: str, accelerator: str, timeout_s: float,
                      seen: Seen) -> tuple[dict[str, Any], dict[str, Any] | None, str, Seen]:
        """Ask a pod the REST launch brought up which image it runs, and replace it by a form launch if it is the
        stock one, which has no GPU and is no use.

        Returns what to add to the answer (image_check ok, stock or unknown with the image or the reason; for a
        stock pod rest_pod_image and, once the form is tried, fell_back_from), the failed answer when the swap
        failed (None otherwise), and the outcome and last read of the wait for the pod: those it was given when the
        pod stays, those of the form launch when it was replaced. A pod that cannot be asked is left alone."""
        output, error = self._shell_on_pod(letter, "image", IMAGE_SCRIPT, IMAGE_CHECK_S)
        asked = parse_image_answer(output) if output is not None else {}
        verdict = image_verdict(asked.get("image"), asked.get("gpu_device"), resource_type)
        checked: dict[str, Any] = {"image_check": verdict}
        if asked.get("image"):
            checked["rest_pod_image" if verdict == "stock" else "pod_image"] = asked["image"]
        if error:
            checked["image_check_note"] = one_line(self.scrub(error), NOTE_CHARS)
        elif verdict == "unknown":
            checked["image_check_note"] = "the pod named no image and showed no GPU device either way"
        if verdict != "stock":
            return checked, None, "ready", seen
        reply = self._send(letter, "DELETE", self._hub(letter).user_url("/server"))
        if reply.status not in (202, 204):
            return checked, {"ok": False, "outcome": "error", "kind": "stock_pod_not_stopped", "status": reply.status,
                             "error": ("the pod runs the stock image and could not be stopped to launch it again: "
                                       + self._failure_text(reply))}, "ready", seen
        gone, seen = self._await_gone(letter, STOP_TIMEOUT_S)
        if gone != "gone":
            return checked, {"ok": False, "outcome": "timeout", "kind": "stop_pending",
                             "error": "the stock pod is still stopping; the form launch was not sent"}, "ready", seen
        checked["fell_back_from"] = "rest"
        failure = self._submit_form(letter, minutes, resource_type, accelerator)
        if failure is not None:
            failure["error"] = ("the REST launch gave the stock image, which was stopped; the launch through the form "
                                f"then failed: {failure['error']}")
            return checked, failure, "failed", seen
        outcome, seen = self._await_ready(letter, timeout_s, self.clock())
        return checked, None, outcome, seen

    # --- stop -------------------------------------------------------------------------------

    @recorded("controller stop")
    def stop(self, letter: str, *, wait: bool = True, timeout_s: float = STOP_TIMEOUT_S,
             reason: str | None = None) -> dict[str, Any]:
        """Stop the account's pod and report what the session cost (a no-op when nothing runs)."""
        self._guard()
        account = self._account(letter)
        client = self._hub(letter)
        base: dict[str, Any] = {"command": "stop", "account": letter, "node": account.node}
        before: Quota | None = None
        after: Quota | None = None

        def finish(answer: dict[str, Any]) -> dict[str, Any]:
            answer = {**base, **answer}
            shown = after if after is not None else before
            metrics = {BALANCE: shown.balance} if shown is not None and shown.enabled else {}
            note = " | ".join(part for part in (reason, answer.get("error")) if part)
            return self._row(answer, POD_STOP, account, metrics=metrics, units={BALANCE: "credits"}, note=note or None)

        seen = self._look(letter)
        if not seen.ok:
            return finish(self._trouble(seen))
        before = self._quota(letter)
        server = seen.server
        view_now = seen.now or self._now()
        if server is None:
            return finish({"ok": True, "outcome": "already_stopped", "state": "stopped", "minutes_run": None})
        minutes_run = server.minutes_run(view_now)
        run = None if minutes_run is None else round(minutes_run, 1)
        reply = self._send(letter, "DELETE", client.user_url("/server"))
        state = "stopped" if reply.status == 204 else "stopping" if reply.status == 202 else None
        if state is None and (reply.status == NO_ANSWER or reply.status >= 500):
            again = self._look(letter)
            if again.ok and again.server is None:
                state = "stopped"
        if state is None:
            return finish({"ok": False, "outcome": "error", "kind": "unexpected_status" if reply.status else "hub_unreachable",
                           "status": reply.status, "error": self._failure_text(reply), "minutes_run": run})
        if state == "stopping":
            if not wait:
                return finish({"ok": True, "outcome": "stopping", "state": "stopping", "minutes_run": run})
            outcome, _ = self._await_gone(letter, timeout_s)
            if outcome != "gone":
                return finish({"ok": False, "outcome": "timeout", "state": "stopping", "minutes_run": run,
                               "error": f"the pod was still stopping after {timeout_s:g} s"})
        after = self._quota(letter)
        answer: dict[str, Any] = {"ok": True, "outcome": "ok", "state": "stopped", "minutes_run": run}
        if before is not None and after is not None and before.billed:
            answer.update(balance_before=before.balance, balance_after=after.balance,
                          credits_charged=before.balance - after.balance)
        return finish(answer)

    # --- wait -------------------------------------------------------------------------------

    @recorded("controller wait")
    def wait_ready(self, letter: str, *, timeout_s: float = WAIT_TIMEOUT_S) -> dict[str, Any]:
        """Wait until the account's pod is ready; an account with no server is not waited for."""
        account = self._account(letter)
        base: dict[str, Any] = {"command": "wait", "account": letter, "node": account.node}

        def finish(answer: dict[str, Any], *, waited: float = 0.0) -> dict[str, Any]:
            answer = {**base, **answer}
            if "waited_s" not in answer and answer["outcome"] != "error":
                answer["waited_s"] = round(waited, 3)
            return self._row(answer, POD_WAIT, account, note=answer.get("error"))

        started = self.clock()
        seen = self._look(letter)
        if not seen.ok:
            return finish(self._trouble(seen))
        if seen.server is None:
            return finish({"ok": False, "outcome": "not_running", "server": {"running": False}})
        if seen.server.up:
            return finish({"ok": True, "outcome": "ready", "server": seen.server.view(seen.now or self._now())})
        outcome, seen = self._await_ready(letter, timeout_s, started)
        view = view_of(seen.server, seen.now or self._now())
        waited = self.clock() - started
        if outcome == "ready":
            return finish({"ok": True, "outcome": "ready", "server": view}, waited=waited)
        if outcome == "timeout":
            return finish({"ok": False, "outcome": "timeout", "server": view,
                           "error": f"the pod was not ready after {timeout_s:g} s"}, waited=waited)
        return finish({"ok": False, "outcome": "not_running", "server": view,
                       "error": "the server left the hub's list or was stopped while waiting"}, waited=waited)

    # --- status and quota -------------------------------------------------------------------

    @recorded("controller status")
    def status(self, letters: Sequence[str] | None = None, *, probe_port: int | None = None) -> dict[str, Any]:
        """The accounts at a glance: running, ready, scheduled end, spend, balance (hub API reads only). The answer
        names the hub they live on (`hub`), or one hub per account (`hubs`) when they do not share one."""
        asked = self._letters(letters)
        entries: dict[str, Any] = {}
        errors: list[str] = []
        ledger_errors: list[str] = []
        for letter in asked:
            account = self._account(letter)
            seen = self._look(letter)
            if not seen.ok:
                entries[letter] = {"node": account.node, "reachable": False, "error": seen.error}
                errors.append(f"{letter}: {seen.error}")
                problem = self._pod_row(POD_STATUS, "error", account, note=seen.error)
            else:
                now = seen.now or self._now()
                entry: dict[str, Any] = {"node": account.node, "reachable": True, **view_of(seen.server, now)}
                quota = self._quota(letter)
                metrics: dict[str, Any] = {}
                if quota is not None:
                    rate = quota.rate(seen.server.accelerator if seen.server else None)
                    entry.update(balance=quota.balance, unlimited=quota.unlimited, quota_enabled=quota.enabled, rate=rate)
                    entry.update(credit_figures(seen.server, quota, rate, now))
                    if quota.enabled:
                        metrics[BALANCE] = quota.balance
                if probe_port is not None and seen.server is not None and seen.server.up:
                    entry["serving"] = self._probe(letter, probe_port)
                entries[letter] = entry
                state = "not_running" if seen.server is None else seen.server.state()
                problem = self._pod_row(POD_STATUS, state, account, metrics=metrics, units={BALANCE: "credits"})
            if problem:
                ledger_errors.append(problem)
        where = self.hub_urls(asked)
        location = {"hub": next(iter(where.values()))} if len(set(where.values())) == 1 else {"hubs": where}
        answer: dict[str, Any] = {"ok": not errors, "command": "status", "outcome": "error" if errors else "ok",
                                  **location, "hub_clock_offset_s": self._offset_s, "accounts": entries,
                                  "refs": {"accounts": ",".join(asked)}}
        if errors:
            answer.update(kind="hub_unreachable", error="; ".join(errors))
        if ledger_errors:
            answer["ledger_error"] = "; ".join(ledger_errors)
        return answer

    def _probe(self, letter: str, port: int) -> dict[str, Any]:
        """Ask the model server on a pod port through the proxy (this counts as activity on the pod)."""
        client = self._hub(letter)
        models = self._send(letter, "GET", client.pod_url(f"/proxy/{port}/v1/models"))
        health = self._send(letter, "GET", client.pod_url(f"/proxy/{port}/health"))
        names: list[str] = []
        if models.ok and isinstance(models.body, Mapping):
            names = [item["id"] for item in models.body.get("data") or []
                     if isinstance(item, Mapping) and isinstance(item.get("id"), str)]
        return {"port": port, "models": names, "health": health.status}

    @recorded("controller quota")
    def quota(self, letters: Sequence[str] | None = None, *, transactions: bool = False) -> dict[str, Any]:
        """The balance and rates of the accounts, and with transactions=True their recent sessions."""
        asked = self._letters(letters)
        entries: dict[str, Any] = {}
        errors: list[str] = []
        ledger_errors: list[str] = []
        for letter in asked:
            account = self._account(letter)
            reply = self._send(letter, "GET", "/hub/api/quota/me")
            quota = self._quota_from(reply)
            if quota is None:
                text = self._failure_text(reply)
                entries[letter] = {"node": account.node, "reachable": False, "error": text}
                errors.append(f"{letter}: {text}")
                continue
            entry: dict[str, Any] = {"node": account.node, "reachable": True, "balance": quota.balance,
                                     "unlimited": quota.unlimited, "enabled": quota.enabled,
                                     "rate": quota.rate(ACCELERATOR), "rates": dict(quota.rates)}
            if not quota.enabled:
                entry["note"] = QUOTA_DISABLED
            if transactions:
                user = urllib.parse.quote(account.user, safe="")
                extra = self._send(letter, "GET", f"/hub/admin/api/quota/{user}")
                items = extra.body.get("recent_transactions") if extra.ok and isinstance(extra.body, Mapping) else None
                entry["transactions"] = [{k: v for k, v in item.items() if k != "username"}
                                         for item in items or [] if isinstance(item, Mapping)]
            entries[letter] = entry
            problem = self._pod_row(POD_QUOTA, "ok", account, metrics={BALANCE: quota.balance} if quota.enabled else None,
                                    units={BALANCE: "credits"})
            if problem:
                ledger_errors.append(problem)
        answer: dict[str, Any] = {"ok": not errors, "command": "quota", "outcome": "error" if errors else "ok",
                                  "accounts": entries, "refs": {"accounts": ",".join(asked)}}
        if errors:
            answer.update(kind="hub_unreachable", error="; ".join(errors))
        if ledger_errors:
            answer["ledger_error"] = "; ".join(ledger_errors)
        return answer

    # --- keep-alive -------------------------------------------------------------------------

    @recorded("controller keep-alive")
    def keep_alive(self, letters: Sequence[str] | None = None, *, interval_s: float = KEEP_ALIVE_INTERVAL_S,
                   minutes: float | None = None, path: str = PING_PATH) -> dict[str, Any]:
        """Ping each running pod every interval_s seconds so the idle culler never stops it, for `minutes` minutes or
        until no pod is left. Ends early when the kill switch is set."""
        self._guard()
        if not KEEP_ALIVE_MIN_S <= interval_s <= KEEP_ALIVE_MAX_S:
            raise ValueError(f"the interval is {KEEP_ALIVE_MIN_S} to {KEEP_ALIVE_MAX_S} seconds, not {interval_s!r}: "
                             "the culler waits an hour and looks every ten minutes")
        if minutes is not None and not minutes > 0:
            raise ValueError(f"the time to keep alive is more than 0 minutes, not {minutes!r}")
        asked = self._letters(letters)
        watches = [Watch(letter, self._account(letter)) for letter in asked]
        problems: list[str] = []
        started = self.clock()
        deadline = None if minutes is None else started + minutes * 60
        next_at = started
        halted = False
        while True:
            if kill_switch.stopped():
                halted = True
                break
            for watch in watches:
                if watch.active:
                    self._cycle(watch, path, problems)
            if not any(watch.active for watch in watches):
                break
            next_at += interval_s
            target = next_at if deadline is None else min(next_at, deadline)
            self.sleep(max(0.0, target - self.clock()))
            if deadline is not None and self.clock() >= deadline:
                break
        states = [watch.state for watch in watches]
        if halted:
            outcome = "stopped"
        elif "lost" in states:
            outcome = "lost"
        elif all(state == "not_running" for state in states):
            outcome = "not_running"
        elif "failing" in states:
            outcome = "unreachable"
        else:
            outcome = "completed"
        answer: dict[str, Any] = {"ok": outcome == "completed", "command": "keep-alive", "outcome": outcome,
                                  "interval_s": interval_s,
                                  "minutes": round((self.clock() - started) / 60, 1),
                                  "accounts": {watch.letter: watch.summary() for watch in watches},
                                  "refs": {"accounts": ",".join(asked)}}
        if halted:
            answer["reason"] = kill_switch.stop_reason()
        if problems:
            answer["ledger_error"] = "; ".join(sorted(set(problems)))
        return answer

    def _row_for(self, problems: list[str], *args: Any, **kwargs: Any) -> None:
        """Write a pod row; a row that could not be written is added to problems."""
        problem = self._pod_row(*args, **kwargs)
        if problem:
            problems.append(problem)

    def _cycle(self, watch: Watch, path: str, problems: list[str]) -> None:
        """One look at a pod and, when it is up (or the hub cannot say), one ping."""
        seen = self._look(watch.letter)
        if seen.ok:
            server = seen.server
            if server is None or server.pending == "stop":
                self._gone(watch, seen, problems)
                return
            watch.scheduled_end = server.scheduled_end or watch.scheduled_end
            if not server.ready:
                watch.state = "launching"
                return
            watch.seen_up = True
        client = self._hub(watch.letter)
        reply = self._send(watch.letter, "GET", client.pod_url(path))
        if reply.ok:
            watch.pings += 1
            watch.seen_up = True
            watch.last_ping = reply.hub_time or self._now()
            watch.streak = 0
            watch.state = "alive"
            if watch.reported != "alive":
                watch.reported = "alive"
                self._row_for(problems, POD_KEEP_ALIVE, "alive", watch.account)
            return
        watch.failures += 1
        watch.streak += 1
        watch.state = "failing"
        if watch.streak == 1:
            watch.reported = "failed_ping"
            self._row_for(problems, POD_KEEP_ALIVE, "failed_ping", watch.account, note=self._failure_text(reply))

    def _gone(self, watch: Watch, seen: Seen, problems: list[str]) -> None:
        watch.active = False
        if not watch.seen_up:
            watch.state = "not_running"
            self._row_for(problems, POD_KEEP_ALIVE, "not_running", watch.account)
            return
        watch.state = "lost"
        now = seen.now or self._now()
        if watch.scheduled_end is not None and now >= watch.scheduled_end:
            watch.note = "the scheduled end passed"
        self._row_for(problems, POD_KEEP_ALIVE, "lost", watch.account, note=watch.note)

    # --- run-script -------------------------------------------------------------------------

    @recorded("controller run-script")
    def run_script(self, letter: str, script: str | Path, *, args: Sequence[str] = (), timeout_s: float = SCRIPT_TIMEOUT_S,
                   run_id: str | None = None, poll_s: float = SCRIPT_POLL_S) -> dict[str, Any]:
        """Upload a shell script to the pod, run it in a terminal there and wait for its exit code.

        The pod must be ready. The script and its output live in the Jupyter root (`lab_run_<id>.sh`, `.out`); the
        output is read back through the contents API until the line `### END rc=<code>` that this command appends. A
        script that starts something that must outlive its terminal starts it with nohup, as the pod scripts do."""
        self._guard()
        account = self._account(letter)
        path = Path(script)
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        run_id = run_id or f"{self._now():%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"
        remote, out = f"lab_run_{run_id}.sh", f"lab_run_{run_id}.out"
        client = self._hub(letter)
        base: dict[str, Any] = {"command": "run-script", "account": letter, "node": account.node, "run_id": run_id,
                                "script": path.name, "remote": remote, "exit_code": None}
        started = self.clock()

        def finish(answer: dict[str, Any]) -> dict[str, Any]:
            answer = {**base, "seconds": round(self.clock() - started, 3), **answer}
            conditions = {"script": path.name, "exit_code": answer["exit_code"], "seconds": answer["seconds"]}
            return self._row(answer, POD_RUN_SCRIPT, account, conditions=conditions, note=answer.get("error"),
                             refs={"run_id": run_id})

        seen = self._look(letter)
        if not seen.ok:
            return finish(self._trouble(seen))
        if seen.server is None or not seen.server.up:
            return finish({"ok": False, "outcome": "not_running", "error": "the pod is not ready"})
        for name, content in ((remote, text), (out, "")):
            reply = self._send(letter, "PUT", client.pod_url(f"/api/contents/{name}"),
                               json={"type": "file", "format": "text", "content": content}, timeout=120)
            if reply.status not in (200, 201):
                return finish({"ok": False, "outcome": "error", "kind": "upload_failed", "status": reply.status,
                               "error": f"could not upload {name}: {self._failure_text(reply)}"})
        terminal = self._send(letter, "POST", client.pod_url("/api/terminals"), json={})
        name = terminal.body.get("name") if terminal.ok and isinstance(terminal.body, Mapping) else None
        if not isinstance(name, str):
            return finish({"ok": False, "outcome": "error", "kind": "terminal_unreachable", "status": terminal.status,
                           "error": f"could not open a terminal: {self._failure_text(terminal)}"})
        command = (f"bash ./{remote}{''.join(' ' + shlex.quote(str(arg)) for arg in args)} > ./{out} 2>&1; "
                   f"echo \"{END_MARKER} rc=$?\" >> ./{out}")
        try:
            self._type_into_terminal(client, name, command, remote)
        except HubUnreachable as exc:
            return finish({"ok": False, "outcome": "error", "kind": "terminal_unreachable", "error": self.scrub(str(exc))})
        deadline = self.clock() + timeout_s
        output = ""
        done = None
        while True:
            reply = self._send(letter, "GET", client.pod_url(f"/api/contents/{out}"), params={"content": "1", "format": "text"})
            if reply.ok and isinstance(reply.body, Mapping) and isinstance(reply.body.get("content"), str):
                output = reply.body["content"]
            done = re.search(rf"^{re.escape(END_MARKER)} rc=(-?\d+)[ \t]*$", output, re.MULTILINE)
            if done or self.clock() >= deadline:
                break
            self.sleep(poll_s)
        printed = output[:done.start()] if done else output
        tail = self.scrub(printed.rstrip())[-TAIL_CHARS:]
        base["output_tail"] = tail
        if done is None:
            return finish({"ok": False, "outcome": "timeout", "error": f"no exit code after {timeout_s:g} s; the script may still run"})
        code = int(done.group(1))
        base["exit_code"] = code
        return finish({"ok": code == 0, "outcome": "ok" if code == 0 else "fail"})

    def _type_into_terminal(self, client: Hub, name: str, command: str, echo: str) -> None:
        """Send a command line to a Jupyter terminal over its websocket and wait briefly for the shell to echo it."""
        import websockets.exceptions
        from websockets.sync.client import connect

        url = client.ws_url(f"/terminals/websocket/{name}")
        try:
            with connect(url, additional_headers=client.ws_headers(), user_agent_header=client.user_agent,
                         open_timeout=TERMINAL_OPEN_S) as terminal:
                terminal.send(json.dumps(["stdin", command + "\n"]))
                give_up = time.monotonic() + TERMINAL_SETTLE_S
                while time.monotonic() < give_up:
                    try:
                        message = terminal.recv(timeout=max(0.0, give_up - time.monotonic()))
                    except (TimeoutError, websockets.exceptions.ConnectionClosed):
                        break
                    try:
                        kind, payload = json.loads(message)
                    except (TypeError, ValueError):
                        continue
                    if kind == "stdout" and isinstance(payload, str) and echo in payload:
                        break
        except (OSError, websockets.exceptions.WebSocketException) as exc:
            raise HubUnreachable(client.scrub(f"terminal {name}: {type(exc).__name__}: {exc}")) from None


# --- the command line -------------------------------------------------------------------------

def _account_letter(text: str) -> str:
    """An account is one letter; whether its keys exist is settled when they are loaded."""
    letter = text.strip()
    if not re.fullmatch(r"[A-Za-z]", letter):
        raise argparse.ArgumentTypeError(
            f"{text!r} is not an account: use one letter (A, B, C, D, ...) whose AUP_HUB_USER_<letter> and "
            "AUP_HUB_TOKEN_<letter> keys exist")
    return letter.upper()


def _account_or_both(text: str) -> str:
    return "both" if text.strip().lower() == "both" else _account_letter(text)


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return value


def _catalog_name(text: str) -> str:
    """A name from the hub's catalog (a resource or an accelerator): not empty, and not auto, which picks a card."""
    name = text.strip()
    if not name or name.lower() == "auto":
        raise argparse.ArgumentTypeError("name one: auto picks the cheapest card the hub has free, not the one you want")
    return name


def _seconds(low: float, high: float | None = None) -> Callable[[str], float]:
    def parse(text: str) -> float:
        try:
            value = float(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
        if value < low or (high is not None and value > high):
            raise argparse.ArgumentTypeError(f"must be {low:g} to {high:g}" if high is not None else f"must be at least {low:g}")
        return value
    return parse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m lab.controller", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    def with_accounts(sub: argparse.ArgumentParser, *, required: bool) -> None:
        if required:
            sub.add_argument("--account", required=True, type=_account_letter, help="an account letter: A, B, C or D")
        else:
            sub.add_argument("--account", default="both", type=_account_or_both,
                             help="an account letter (A, B, C or D) or both (A and B, default)")

    status = commands.add_parser("status", help="both accounts: running, ready, scheduled end, spend, balance")
    with_accounts(status, required=False)
    status.add_argument("--probe-port", type=_positive_int, help="also ask the model server on this pod port (counts as activity)")

    quota = commands.add_parser("quota", help="the balance and rates of the accounts")
    with_accounts(quota, required=False)
    quota.add_argument("--transactions", action="store_true", help="add the account's recent sessions")

    launch = commands.add_parser("launch", help="launch the pod and wait until it is ready")
    with_accounts(launch, required=True)
    launch.add_argument("--minutes", required=True, type=_positive_int, help="runtime in minutes; the hub sets no ceiling")
    launch.add_argument("--accelerator", type=_catalog_name,
                        help=f"the card to ask for (default: AUP_HUB_GPU_<account>, else {ACCELERATOR})")
    launch.add_argument("--resource", type=_catalog_name,
                        help=f"the catalog resource (default: AUP_HUB_RESOURCE_<account>, else {RESOURCE_TYPE})")
    launch.add_argument("--no-wait", action="store_true", help="return when the hub accepts the launch")
    launch.add_argument("--timeout", type=_seconds(1), default=LAUNCH_TIMEOUT_S, help="seconds to wait for ready")
    launch.add_argument("--dry-run", action="store_true", help="check the balance and show the request; send nothing")
    launch.add_argument("--reason", help="why, kept in the ledger row's note")

    stop = commands.add_parser("stop", help="stop the pod and report what the session cost")
    with_accounts(stop, required=True)
    stop.add_argument("--no-wait", action="store_true", help="return when the hub accepts the stop")
    stop.add_argument("--timeout", type=_seconds(1), default=STOP_TIMEOUT_S, help="seconds to wait for the stop")
    stop.add_argument("--reason", help="why, kept in the ledger row's note")

    wait = commands.add_parser("wait", help="wait until the pod is ready")
    with_accounts(wait, required=True)
    wait.add_argument("--timeout", type=_seconds(1), default=WAIT_TIMEOUT_S, help="seconds to wait")

    keep = commands.add_parser("keep-alive", help="ping the pods so the idle culler never stops them")
    with_accounts(keep, required=False)
    keep.add_argument("--interval", type=_seconds(KEEP_ALIVE_MIN_S, KEEP_ALIVE_MAX_S), default=KEEP_ALIVE_INTERVAL_S,
                      help=f"seconds between pings ({KEEP_ALIVE_MIN_S} to {KEEP_ALIVE_MAX_S}, default {KEEP_ALIVE_INTERVAL_S})")
    keep.add_argument("--minutes", type=_seconds(0.1), help="stop after this long (default: until the pods are gone)")

    script = commands.add_parser("run-script", help="run a shell script on the pod and wait for its exit code")
    with_accounts(script, required=True)
    script.add_argument("script", help="a local shell script")
    script.add_argument("script_args", nargs=argparse.REMAINDER, help="arguments for the script")
    script.add_argument("--timeout", type=_seconds(1), default=SCRIPT_TIMEOUT_S, help="seconds to wait for the exit code")
    return parser


def _letters_of(args: argparse.Namespace) -> list[str]:
    return list(hubs.LETTERS) if args.account == "both" else [args.account]


HANDLERS: dict[str, Callable[[Controller, argparse.Namespace], dict[str, Any]]] = {
    "status": lambda c, a: c.status(_letters_of(a), probe_port=a.probe_port),
    "quota": lambda c, a: c.quota(_letters_of(a), transactions=a.transactions),
    "launch": lambda c, a: c.launch(a.account, a.minutes, wait=not a.no_wait, timeout_s=a.timeout, dry_run=a.dry_run,
                                    resource_type=a.resource, accelerator=a.accelerator, reason=a.reason),
    "stop": lambda c, a: c.stop(a.account, wait=not a.no_wait, timeout_s=a.timeout, reason=a.reason),
    "wait": lambda c, a: c.wait_ready(a.account, timeout_s=a.timeout),
    "keep-alive": lambda c, a: c.keep_alive(_letters_of(a), interval_s=a.interval, minutes=a.minutes),
    "run-script": lambda c, a: c.run_script(a.account, a.script, args=a.script_args, timeout_s=a.timeout),
}


def run(argv: Sequence[str], *, environ: Mapping[str, str] | None = None, env_files: Sequence[Path] | None = None,
        transport: Any = None, sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None) -> tuple[dict[str, Any], int]:
    """One command: (the answer, the exit code). environ and env_files say where the keys are looked up, and
    transport, sleep and clock replace the network and the time, for rehearsals."""
    args = build_parser().parse_args(list(argv))
    environ = os.environ if environ is None else environ
    try:
        accounts = load_accounts(_letters_of(args), environ=environ, env_files=env_files)
    except MissingCredentials as exc:
        return ({"ok": False, "command": args.command, "outcome": "no_credentials", "missing": exc.missing,
                 "error": "keys missing from the environment and the .env files",
                 "hint": "set them in the workspace .env (see config/.env.example); the values are never printed"}, 2)
    controller = Controller(accounts, transport=transport, sleep=sleep or time.sleep, clock=clock or time.time,
                            environ=environ)
    with controller:
        try:
            answer = HANDLERS[args.command](controller, args)
        except kill_switch.Stopped as exc:
            answer = {"ok": False, "command": args.command, "outcome": "stopped", "reason": exc.reason,
                      "file": str(kill_switch.stop_file())}
        answer = {key: value for key, value in answer.items() if key != "refs"}
    if answer.get("kind") == "no_credentials":
        return answer, 2
    return answer, 0 if answer.get("ok", True) else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Launch, stop, watch and keep alive the two AUP pods: status, quota, launch, stop, wait, keep-alive, run-script."""
    try:
        answer, code = run(list(sys.argv[1:] if argv is None else argv))
    except KeyboardInterrupt:
        answer, code = {"ok": False, "outcome": "interrupted"}, 1
    except FileNotFoundError as exc:
        answer, code = {"ok": False, "outcome": "error", "kind": "file_not_found", "error": f"{exc.strerror}: {exc.filename}"}, 1
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        answer, code = {"ok": False, "outcome": "error", "error": f"{type(exc).__name__}: {exc}"}, 1
    print(json.dumps(answer))
    return code


if __name__ == "__main__":
    sys.exit(main())
