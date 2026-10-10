"""credits: what the pod accounts have left, how fast it is going, and whether it lasts until judging.

    python -m lab credits [--account C,D|all] [--window-hours H] [--until TIME] [--no-ledger]

Run it from the repo root. One call asks the hub of each account what it knows (GET requests only, nothing is
launched, stopped or changed) and answers with one JSON object: per account the figures below, the alerts, and
`lines`, one plain sentence per account. Exit code 0 when every account was read, 1 when one could not be
(outcome error), 2 for bad arguments and for missing keys. An alert is a result, not a failure: the outcome is
`alert` and the exit code stays 0. The verb only reads, so it runs while lab/state/STOP exists.

The accounts. --account takes letters separated by commas (C,D is the default, the pair on the SJC hub), or `all`
for every letter whose AUP_HUB_USER_<letter> and AUP_HUB_TOKEN_<letter> keys exist. Each account is asked at its
own AUP_HUB_URL_<letter> when that key is set, else $LAB_HUB_URL, else the DUB hub; the keys are found the way
lab/machine/ready.py finds them. No token and no account name is ever printed or written to the ledger.

Two kinds of hub. GET /hub/api/quota/me says which.

  Metered (quota enabled, account not unlimited; DUB: 4 credits a minute on the R9700 from a 60,000 quota). The
  hub deducts a session only when it ends, whole elapsed minutes times the rate, so the balance does not move
  while a pod runs and a spend worked out from balances would read 0 until a lump at the stop. The figures come
  from the sessions instead: the settled ones from the usage transactions (GET /hub/admin/api/quota/<account>,
  each charge spread evenly over the minutes the session ran) and the running one from the server's `started`
  time in the user model (GET /hub/api/users/<account>) at the posted rate. Then:

    credits_left        the balance less what the running sessions have already accrued (whole minutes x rate)
    credits_per_hour    credits spent inside the last --window-hours (default 3), settled plus running, over the
                        window: 0 when nothing ran in it
    burn_per_hour       the higher of what the running servers burn now (rate x 60 each) and credits_per_hour
    hours_left          credits_left / burn_per_hour; absent when nothing burns; runs_out_at is now plus that
    session             for a running server: minutes run, credits so far, scheduled end, credits if it runs to
                        its end and the balance then

  An alert is raised for an account whose credits are already gone, or whose hours_left is shorter than the time
  to judging (--until, default Sunday 2026-10-11 12:00 Eastern, the start of judging and demos). Once judging has
  started nothing is raised.

  Unmetered (quota disabled, as on the SJC hub, or an unlimited account: balance 0 means no accounting, not an
  empty account). There is no balance to run out of, so nothing is projected and nothing alerts. The hub's usage
  statistics (GET /hub/api/stats/me) give the line `unmetered: usage N minutes today, M sessions`, counting the
  sessions that are still running, with today the hub's UTC day, plus the pod-minutes in the window.

Hub time. Elapsed time and the day are read from the hub's clock (the Date header of its answers), because the
workstation's clock is not the hub's.

The ledger. Unless --no-ledger is given, one row per account is appended with verb `credits`, the node (pod-c for
C), refs.account, the window in conditions, and the outcome ok, alert, unmetered or error. The metrics on a
metered row are pod.balance_credits (as the hub says it), pod.credits_left, pod.credits_per_hour,
pod.credits_burn_per_hour and, when something burns, pod.credits_hours_left; on an unmetered row
pod.usage_min_today, pod.sessions_today and pod.usage_min_window. A row built on an answer that was only partly
read is marked low_confidence. A ledger that cannot be written never undoes the reading: the answer then carries
ledger_error.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import re
import sys
import time
import urllib.parse
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from lab import hub as hubs
from lab.controller import Quota, Server, iso_z, parse_time
from lab.hub import Account, Hub, HubUnreachable, MissingCredentials, Reply, one_line
from lab.machine import ledger, ready, record, stop

__all__ = ["VERB", "DEFAULT_ACCOUNTS", "JUDGING", "DEFAULT_WINDOW_HOURS", "Charge", "Facts", "Assessed",
           "parse_charges", "overlap_minutes", "settled_credits", "assess", "read_account", "run", "main",
           "build_parser"]

VERB = "credits"
DEFAULT_ACCOUNTS = ("C", "D")
# Sunday 2026-10-11 12:00 Eastern (UTC-4): judging and demos start.
JUDGING = dt.datetime(2026, 10, 11, 16, 0, tzinfo=dt.timezone.utc)
DEFAULT_WINDOW_HOURS = 3.0
RECENT_SHOWN = 5               # recent transactions listed in an account's answer
HUB_TRANSACTIONS = 20          # how many recent transactions the hub lists
NOTE_CHARS = 300
MIN_NAME_CHARS = 4             # a shorter account name would mangle the text it is cut out of
NO_ANSWER = 0                  # the status of a Reply for a call that never got one
ALL = "all"

QUOTA_ME = "/hub/api/quota/me"
STATS_ME = "/hub/api/stats/me"

BALANCE = "pod.balance_credits"
LEFT = "pod.credits_left"
PER_HOUR = "pod.credits_per_hour"
BURN = "pod.credits_burn_per_hour"
HOURS_LEFT = "pod.credits_hours_left"
USAGE_TODAY = "pod.usage_min_today"
SESSIONS_TODAY = "pod.sessions_today"
USAGE_WINDOW = "pod.usage_min_window"
UNITS = {BALANCE: "credits", LEFT: "credits", PER_HOUR: "credits/h", BURN: "credits/h", HOURS_LEFT: "h",
         USAGE_TODAY: "min", SESSIONS_TODAY: "sessions", USAGE_WINDOW: "min"}

OK, ALERT, UNMETERED, ERROR = "ok", "alert", "unmetered", "error"
OUT_OF_CREDITS, RUNS_OUT = "out_of_credits", "run_out_before_judging"

_MINUTES = re.compile(r"(\d+)\s*min\b")


# --- small helpers ----------------------------------------------------------------------------

def node_of(letter: str) -> str:
    """The ledger's name for the machine behind an account: pod-a, pod-b, pod-c, ..."""
    return hubs.NODES.get(letter) or f"pod-{letter.lower()}"


def url_key(letter: str) -> str:
    """The key holding the address of the hub an account lives on."""
    return f"AUP_HUB_URL_{letter}"


def _number(value: Any) -> float | None:
    """value as a finite number; None for anything else (a bool is not a number here)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


def _whole(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None and number == int(number) else None


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def _round(value: float, places: int = 2) -> float:
    return round(value + 0.0, places)


def _hours(delta: dt.timedelta) -> float:
    return delta.total_seconds() / 3600


# --- what the hub's answers say -----------------------------------------------------------------

def quota_from(reply: Reply) -> Quota | None:
    """The Quota in a GET /hub/api/quota/me answer; None when the call failed or the answer is not one."""
    body = reply.body
    if not reply.ok or not isinstance(body, Mapping):
        return None
    balance = _whole(body.get("balance"))
    if balance is None:
        return None
    rates = body.get("rates") if isinstance(body.get("rates"), Mapping) else {}
    return Quota(balance, bool(body.get("unlimited")), bool(body.get("enabled", True)), rates)


@dataclass(frozen=True)
class Charge:
    """A session the hub has settled: what it cost, when it was charged (its end) and, when the description says
    how many minutes it ran, when it started (to the minute: the hub bills whole minutes)."""

    end: dt.datetime
    credits: int
    start: dt.datetime | None = None


def parse_charges(items: Iterable[Any]) -> list[Charge]:
    """The usage charges among the hub's transactions, oldest first. A top-up, a balance set or anything that is
    not a deduction for a session is not spend and is left out."""
    found = []
    for item in items:
        if not isinstance(item, Mapping) or item.get("transaction_type") != "usage":
            continue
        amount = _whole(item.get("amount"))
        end = parse_time(item.get("created_at"))
        if amount is None or amount >= 0 or end is None:
            continue
        match = _MINUTES.search(str(item.get("description") or ""))
        start = end - dt.timedelta(minutes=int(match.group(1))) if match and int(match.group(1)) > 0 else None
        found.append(Charge(end, -amount, start))
    found.sort(key=lambda charge: charge.end)
    return found


def overlap_minutes(start: dt.datetime, end: dt.datetime, lo: dt.datetime, hi: dt.datetime) -> float:
    """The minutes that [start, end] and [lo, hi] have in common."""
    return max(0.0, (min(end, hi) - max(start, lo)).total_seconds() / 60)


def settled_credits(charges: Iterable[Charge], lo: dt.datetime, hi: dt.datetime) -> float:
    """The credits of settled sessions that fall inside [lo, hi]. A charge is spread evenly over the minutes its
    session ran; one with no known start counts whole when its end is inside."""
    total = 0.0
    for charge in charges:
        if charge.start is None or charge.end <= charge.start:
            total += charge.credits if lo <= charge.end <= hi else 0.0
        else:
            ran = (charge.end - charge.start).total_seconds() / 60
            total += charge.credits * overlap_minutes(charge.start, charge.end, lo, hi) / ran
    return total


# --- what one account's hub told us ---------------------------------------------------------------

@dataclass
class Facts:
    """Everything read about one account, before any arithmetic. now is the hub's clock when it said so."""

    letter: str
    node: str
    hub: str
    now: dt.datetime
    quota: Quota | None = None
    servers: list[Server] = field(default_factory=list)
    transactions: list[Mapping[str, Any]] | None = None     # None: the hub's list could not be read
    stats: Mapping[str, Any] | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def scrubbed(account: Account, text: str) -> str:
    """text with the token cut out and the account's name made <account C>."""
    if len(account.token) >= hubs.MIN_HIDDEN_CHARS:
        text = text.replace(account.token, "***")
    if len(account.user) >= MIN_NAME_CHARS:
        text = text.replace(account.user, f"<account {account.letter}>")
    return text


def _failure(reply: Reply) -> str:
    if reply.status == NO_ANSWER:
        return f"the hub did not answer: {reply.text}"
    return f"the hub answered {reply.status}: {reply.detail()}"


def read_account(client: Hub, clock: Callable[[], dt.datetime]) -> Facts:
    """Ask the hub about one account: GET requests only. Metered, it reads the user model (the running servers) and
    the recent transactions; unmetered, the usage statistics. A read that fails is in errors (the account cannot
    be judged) or in warnings (the figures are partial)."""
    account = client.account
    facts = Facts(account.letter, node_of(account.letter), client.base_url, clock())
    times: list[dt.datetime] = []

    def get(path: str) -> Reply:
        try:
            reply = client.get(path)
        except HubUnreachable as exc:
            return Reply(status=NO_ANSWER, text=str(exc))
        if reply.hub_time is not None:
            times.append(reply.hub_time)
        return reply

    def clean(text: str) -> str:
        return scrubbed(account, one_line(text))

    reply = get(QUOTA_ME)
    facts.quota = quota_from(reply)
    if facts.quota is None:
        facts.errors.append(clean(f"GET {QUOTA_ME}: {_failure(reply)}" if not reply.ok
                                  else f"GET {QUOTA_ME}: the answer has no balance"))
    elif facts.quota.billed:
        user = get(client.user_url())
        servers = user.body.get("servers") if user.ok and isinstance(user.body, Mapping) else None
        if not isinstance(servers, Mapping):
            facts.errors.append(clean(f"GET the user model: {_failure(user) if not user.ok else 'no servers listed'}"))
        else:
            facts.servers = [server for server in (Server.from_model(raw) for raw in servers.values()
                                                    if isinstance(raw, Mapping)) if server.started is not None]
        quota_user = urllib.parse.quote(account.user, safe="")
        listed = get(f"/hub/admin/api/quota/{quota_user}")
        items = listed.body.get("recent_transactions") if listed.ok and isinstance(listed.body, Mapping) else None
        if isinstance(items, list):
            facts.transactions = [item for item in items if isinstance(item, Mapping)]
        else:
            facts.warnings.append(clean("the recent transactions could not be read, so settled sessions are missing "
                                        f"from credits per hour ({_failure(listed)})"))
    else:
        stats = get(STATS_ME)
        if stats.ok and isinstance(stats.body, Mapping):
            facts.stats = stats.body
        else:
            facts.errors.append(clean(f"GET {STATS_ME}: {_failure(stats)}"))
    if times:
        facts.now = max(times)
    return facts


# --- the arithmetic -------------------------------------------------------------------------------

@dataclass
class Assessed:
    """One account's figures: the answer's entry, the alerts it raises, and what its ledger row holds."""

    letter: str
    node: str
    outcome: str
    entry: dict[str, Any]
    line: str
    alerts: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    conditions: dict[str, Any] = field(default_factory=dict)
    note: str | None = None
    low_confidence: bool = False

    @property
    def ok(self) -> bool:
        return self.outcome != ERROR


def _failed(facts: Facts, window_hours: float) -> Assessed:
    text = "; ".join(facts.errors)
    entry = {"node": facts.node, "hub": facts.hub, "ok": False, "reachable": facts.quota is not None, "error": text}
    return Assessed(facts.letter, facts.node, ERROR, entry, f"{facts.letter} ({facts.node}): could not be read - {text}",
                    note=text[:NOTE_CHARS], conditions={"window_hours": window_hours})


def _running_view(server: Server, rate: int | None, now: dt.datetime, balance: int) -> dict[str, Any]:
    minutes = server.minutes_run(now) or 0.0
    view: dict[str, Any] = {"started": iso_z(server.started), "ready": server.up, "pending": server.pending,
                            "accelerator": server.accelerator, "rate": rate, "minutes_run": _round(minutes, 1),
                            "runtime_minutes": server.runtime_minutes, "scheduled_end": iso_z(server.scheduled_end)}
    if rate is not None:
        view["credits_so_far"] = int(minutes) * rate
        if server.runtime_minutes:
            whole = server.runtime_minutes * rate
            view.update(credits_if_run_to_end=whole, balance_at_scheduled_end=max(0, balance - whole),
                        exceeds_balance=whole > balance)
    return view


def _assess_metered(facts: Facts, window_hours: float, judging: dt.datetime) -> Assessed:
    quota = facts.quota
    assert quota is not None
    now = facts.now
    lo = now - dt.timedelta(hours=window_hours)
    warnings = list(facts.warnings)
    running: list[dict[str, Any]] = []
    accrued = 0
    burn_now = 0.0
    running_in_window = 0.0
    for server in facts.servers:
        rate = quota.rate(server.accelerator)
        running.append(_running_view(server, rate, now, quota.balance))
        if rate is None:
            warnings.append(f"no posted rate for {server.accelerator or 'the default accelerator'}: "
                            "its running session is not counted")
            continue
        accrued += int(server.minutes_run(now) or 0.0) * rate
        burn_now += rate * 60
        running_in_window += rate * overlap_minutes(server.started, now, lo, now)  # type: ignore[arg-type]
    charges = parse_charges(facts.transactions or [])
    if facts.transactions is not None and len(facts.transactions) >= HUB_TRANSACTIONS:
        oldest = min((charge.end for charge in charges), default=None)
        if oldest is not None and oldest > lo:
            warnings.append(f"the hub lists only its last {HUB_TRANSACTIONS} transactions and the window reaches "
                            "back past the oldest of them")
    left = max(0, quota.balance - accrued)
    per_hour = (settled_credits(charges, lo, now) + running_in_window) / window_hours
    burn = max(burn_now, per_hour)
    basis = "running" if burn_now > 0 and burn_now >= per_hour else ("window" if per_hour > 0 else "none")
    hours_left = left / burn if burn > 0 else None
    to_judging = _hours(judging - now)
    entry: dict[str, Any] = {
        "node": facts.node, "hub": facts.hub, "ok": True, "reachable": True, "metered": True,
        "balance": quota.balance, "rate": quota.rate("r9700"), "credits_left": left, "accrued_running": accrued,
        "window_hours": window_hours, "credits_per_hour": _round(per_hour), "burn_per_hour": _round(burn),
        "burn_basis": basis, "hours_left": None if hours_left is None else _round(hours_left),
        "runs_out_at": None if hours_left is None else iso_z(now + dt.timedelta(hours=hours_left)),
        "running": running,
        "recent_transactions": _recent(facts.transactions)}
    if warnings:
        entry["warnings"] = warnings
    alerts: list[dict[str, Any]] = []
    if to_judging > 0:
        kind = OUT_OF_CREDITS if left <= 0 else (RUNS_OUT if hours_left is not None and hours_left < to_judging else None)
        if kind is not None:
            alerts.append(_alert(facts, kind, left, burn, hours_left, to_judging, judging, now))
    else:
        entry["judging_started"] = True
    metrics: dict[str, float] = {BALANCE: quota.balance, LEFT: left, PER_HOUR: _round(per_hour), BURN: _round(burn)}
    if hours_left is not None:
        metrics[HOURS_LEFT] = _round(hours_left)
    outcome = ALERT if alerts else OK
    projection = ("nothing is burning" if hours_left is None else
                  f"{burn:,.0f} credits/h ({basis}) leaves {hours_left:.1f} h")
    line = (f"{facts.letter} ({facts.node}): {left:,} credits left of {quota.balance:,}, {per_hour:,.0f} credits/h over "
            f"{window_hours:g} h, {projection}; judging in {max(0.0, to_judging):.1f} h"
            f"{' - ' + alerts[0]['message'] if alerts else ''}")
    note = alerts[0]["message"] if alerts else line
    return Assessed(facts.letter, facts.node, outcome, entry, line, alerts, metrics,
                    {"window_hours": window_hours, "metered": True}, note[:NOTE_CHARS],
                    low_confidence=bool(warnings))


def _alert(facts: Facts, kind: str, left: int, burn: float, hours_left: float | None, to_judging: float,
           judging: dt.datetime, now: dt.datetime) -> dict[str, Any]:
    if kind == OUT_OF_CREDITS:
        message = f"{facts.letter} ({facts.node}) has no credits left; judging starts in {to_judging:.1f} h"
        runs_out = None
    else:
        assert hours_left is not None
        runs_out = iso_z(now + dt.timedelta(hours=hours_left))
        message = (f"{facts.letter} ({facts.node}) is projected to run out of credits at {runs_out}, in "
                   f"{hours_left:.1f} h and {to_judging - hours_left:.1f} h before judging at {iso_z(judging)}: "
                   f"{left:,} credits left at {burn:,.0f} credits/h")
    return {"account": facts.letter, "node": facts.node, "kind": kind, "credits_left": left,
            "burn_per_hour": _round(burn), "hours_left": None if hours_left is None else _round(hours_left),
            "runs_out_at": runs_out, "judging": iso_z(judging), "hours_to_judging": _round(to_judging),
            "message": message}


def _recent(items: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]] | None:
    if items is None:
        return None
    ordered = sorted(items, key=lambda item: parse_time(item.get("created_at")) or dt.datetime.min.replace(
        tzinfo=dt.timezone.utc), reverse=True)
    return [{"at": iso_z(parse_time(item.get("created_at"))), "type": item.get("transaction_type"),
             "amount": item.get("amount"), "description": one_line(item.get("description") or "", 120)}
            for item in ordered[:RECENT_SHOWN]]


def _assess_unmetered(facts: Facts, window_hours: float) -> Assessed:
    quota = facts.quota
    assert quota is not None and facts.stats is not None
    now = facts.now
    day = now.date().isoformat()
    day_start = dt.datetime(now.year, now.month, now.day, tzinfo=dt.timezone.utc)
    lo = now - dt.timedelta(hours=window_hours)
    done_minutes = done_sessions = 0
    for item in facts.stats.get("usage") or []:
        if isinstance(item, Mapping) and item.get("date") == day:
            done_minutes += _whole(item.get("minutes")) or 0
            done_sessions += _whole(item.get("sessions")) or 0
    running: list[dict[str, Any]] = []
    running_today = 0.0
    in_window = 0.0
    for item in facts.stats.get("recent_sessions") or []:
        if not isinstance(item, Mapping):
            continue
        start, end = parse_time(item.get("start_time")), parse_time(item.get("end_time"))
        if start is None:
            continue
        active = end is None and item.get("status") == "active"
        stop_at = now if end is None else end
        in_window += overlap_minutes(start, stop_at, lo, now)
        if active:
            running_today += overlap_minutes(start, now, day_start, now)
            running.append({"started": iso_z(start), "accelerator": item.get("accelerator_type"),
                            "minutes_run": _round((now - start).total_seconds() / 60, 1)})
    minutes_today = done_minutes + int(running_today)
    sessions_today = done_sessions + len(running)
    reason = "quota disabled on this hub" if not quota.enabled else "unlimited quota"
    summary = f"unmetered: usage {_plural(minutes_today, 'minute')} today, {_plural(sessions_today, 'session')}"
    entry: dict[str, Any] = {
        "node": facts.node, "hub": facts.hub, "ok": True, "reachable": True, "metered": False, "reason": reason,
        "summary": summary, "day": day, "usage_minutes_today": minutes_today, "sessions_today": sessions_today,
        "window_hours": window_hours, "usage_minutes_window": _round(in_window, 1),
        "total_minutes": _whole(facts.stats.get("total_minutes")),
        "total_sessions": _whole(facts.stats.get("total_sessions")), "running": running}
    if facts.warnings:
        entry["warnings"] = list(facts.warnings)
    metrics = {USAGE_TODAY: minutes_today, SESSIONS_TODAY: sessions_today, USAGE_WINDOW: _round(in_window, 1)}
    live = f"; {_plural(len(running), 'session')} running" if running else ""
    line = f"{facts.letter} ({facts.node}): {summary}{live}"
    return Assessed(facts.letter, facts.node, UNMETERED, entry, line, [], metrics,
                    {"window_hours": window_hours, "metered": False, "day": day}, f"{summary} ({reason})"[:NOTE_CHARS],
                    low_confidence=bool(facts.warnings))


def assess(facts: Facts, window_hours: float, judging: dt.datetime) -> Assessed:
    """The figures of one account from what was read about it."""
    if facts.errors or facts.quota is None:
        return _failed(facts, window_hours)
    if facts.quota.billed:
        return _assess_metered(facts, window_hours, judging)
    return _assess_unmetered(facts, window_hours)


# --- the ledger -------------------------------------------------------------------------------------

def write_row(item: Assessed) -> str | None:
    """Append the account's row. Returns why it could not be written, else None."""
    fields: dict[str, Any] = {"node": item.node, "conditions": item.conditions}
    if item.metrics:
        fields["units"] = {name: UNITS[name] for name in item.metrics}
    if item.note:
        fields["note"] = item.note
    if item.low_confidence:
        fields["low_confidence"] = True
    try:
        record.record(VERB, item.outcome, item.metrics, {"account": item.letter}, **fields)
    except (ledger.RowError, ledger.LockTimeout, OSError, ValueError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


# --- the command line -------------------------------------------------------------------------------

def _letters(text: str) -> list[str]:
    parts = [part for part in re.split(r"[,\s]+", text.strip()) if part]
    if [part.lower() for part in parts] == [ALL]:
        return [ALL]
    letters: list[str] = []
    for part in parts:
        if not re.fullmatch(r"[A-Za-z]", part):
            raise argparse.ArgumentTypeError(
                f"{part!r} is not an account: use letters separated by commas (C,D), or all on its own")
        if part.upper() not in letters:
            letters.append(part.upper())
    if not letters:
        raise argparse.ArgumentTypeError("name at least one account letter, or all")
    return letters


def _positive(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number of hours") from None
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"the window is a positive number of hours, not {text}")
    return value


def _aware(text: str) -> dt.datetime:
    moment = parse_time(text.strip())
    if moment is None or (text.strip()[-1:].upper() != "Z" and not re.search(r"[+-]\d\d:?\d\d$", text.strip())):
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a time with a UTC offset, such as 2026-10-11T16:00:00Z or 2026-10-11T12:00:00-04:00")
    return moment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"python -m lab {VERB}", description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--account", type=_letters, default=list(DEFAULT_ACCOUNTS), metavar="LETTERS",
                        help="account letters separated by commas, or all (default: %s)" % ",".join(DEFAULT_ACCOUNTS))
    parser.add_argument("--window-hours", type=_positive, default=DEFAULT_WINDOW_HOURS, metavar="H",
                        help="the window credits per hour is taken over (default: %g)" % DEFAULT_WINDOW_HOURS)
    parser.add_argument("--until", type=_aware, default=JUDGING, metavar="TIME",
                        help="the moment the credits must last until (default: judging, %s)" % iso_z(JUDGING))
    parser.add_argument("--no-ledger", action="store_true", help="read and report; write no ledger rows")
    return parser


def _all_letters(environ: Mapping[str, str], files: Sequence[Any]) -> list[str]:
    names = [name for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for name in hubs.key_names(letter)]
    found, _ = ready.collect_env(names, environ, files)
    return [letter for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if all(name in found for name in hubs.key_names(letter))]


def run(argv: Sequence[str], *, environ: Mapping[str, str] | None = None, env_files: Sequence[Any] | None = None,
        transport: httpx.BaseTransport | None = None, sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], dt.datetime] | None = None) -> dict[str, Any]:
    """The verb with its doubles: environ and env_files replace the keys' search, transport replaces the network,
    sleep paces the hub client's retries and clock is the fallback for the hub's own clock."""
    args = build_parser().parse_args(list(argv))
    env = os.environ if environ is None else environ
    files = ready.default_env_files(env) if env_files is None else list(env_files)
    now_of = clock or (lambda: dt.datetime.now(dt.timezone.utc))
    letters = _all_letters(env, files) if args.account == [ALL] else args.account
    if not letters:
        return {"ok": False, "outcome": "missing_credentials", "error": "no account has its AUP_HUB_USER_<letter> "
                "and AUP_HUB_TOKEN_<letter> keys", "missing": []}
    try:
        accounts = hubs.load_accounts(letters, environ=env, env_files=files)
    except MissingCredentials as exc:
        return {"ok": False, "outcome": "missing_credentials", "error": str(exc), "missing": exc.missing}
    urls, _ = ready.collect_env([url_key(letter) for letter in letters], env, files)
    assessed: list[Assessed] = []
    for letter in letters:
        with Hub(accounts[letter], base_url=urls.get(url_key(letter)), transport=transport, sleep=sleep,
                 environ=env) as client:
            assessed.append(assess(read_account(client, now_of), args.window_hours, args.until))
    ledger_errors: list[str] = []
    if not args.no_ledger:
        ledger_errors = [problem for item in assessed if (problem := write_row(item))]
    failed = [item for item in assessed if not item.ok]
    alerts = [alert for item in assessed for alert in item.alerts]
    answer: dict[str, Any] = {
        "ok": not failed, "command": VERB, "outcome": ERROR if failed else (ALERT if alerts else OK),
        "window_hours": args.window_hours, "judging": iso_z(args.until),
        "accounts": {item.letter: item.entry for item in assessed}, "alerts": alerts,
        "lines": [item.line for item in assessed], "refs": {"accounts": ",".join(letters)}}
    if failed:
        answer.update(kind="hub_unreachable", error="; ".join(f"{item.letter}: {item.entry['error']}" for item in failed))
    if ledger_errors:
        answer["ledger_error"] = "; ".join(ledger_errors)
    return answer


@stop.exempt
def main(argv: Sequence[str]) -> dict[str, Any]:
    """Credits left on the pod accounts, credits per hour, and whether they last until judging."""
    return run(argv)


if __name__ == "__main__":      # pragma: no cover
    from lab.__main__ import main as dispatch
    sys.exit(dispatch([VERB, *sys.argv[1:]]))
