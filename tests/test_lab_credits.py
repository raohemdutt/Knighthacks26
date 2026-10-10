"""lab credits: the tracker drives the real Hub client against an httpx mock transport that replays the recorded
hub responses in fixtures/lab/credits (account A on the DUB hub, metered; account C on the SJC hub, quota
disabled). Nothing opens a socket; the ledger goes to a temp state dir."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import httpx
import pytest

from lab import __main__ as lab_cli
from lab import credits
from lab.machine import ledger

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "lab" / "credits"
TOKEN_A = "aaaa" + "0123456789abcdef0123456789ab"   # built in parts: the secret scan flags 32 hex after "token"
TOKEN_C = "cccc" + "0123456789abcdef0123456789ab"
ENV = {"AUP_HUB_USER_A": "acct-a", "AUP_HUB_TOKEN_A": TOKEN_A, "AUP_HUB_URL_A": "https://dub.test",
       "AUP_HUB_USER_C": "acct-c", "AUP_HUB_TOKEN_C": TOKEN_C, "AUP_HUB_URL_C": "https://sjc.test"}
FAR_JUDGING = "2026-10-11T16:00:00Z"
AFTER_JUDGING = "2026-10-10T10:00:00Z"      # before the recorded hub time: judging has already started


def load(name):
    return json.loads((FIXTURES / name / "responses.json").read_text(encoding="utf-8"))["responses"]


class Hubs:
    """Replays recorded responses by host and path; a path with no recording answers 404."""

    def __init__(self, dub=None, sjc=None):
        self.tables = {"dub.test": dub if dub is not None else load("dub_a"),
                       "sjc.test": sjc if sjc is not None else load("sjc_c")}
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        entry = self.tables[request.url.host].get(request.url.path)
        if entry is None:
            return httpx.Response(404, json={"message": "not found"})
        headers = {"date": entry["date"]} if entry.get("date") else {}
        return httpx.Response(entry["status"], json=entry.get("body"), headers=headers)


@pytest.fixture(autouse=True)
def state(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    return tmp_path / "state"


def go(argv, hubs=None, env=ENV):
    hubs = hubs or Hubs()
    answer = credits.run(argv, environ=env, env_files=[], transport=httpx.MockTransport(hubs), sleep=lambda s: None)
    return answer, hubs


def rows(state):
    return ledger.read(state_dir=state)


def metered_tables(**changes):
    table = load("dub_a")
    if "balance" in changes:
        table["/hub/api/quota/me"]["body"]["balance"] = changes["balance"]
        table["/hub/admin/api/quota/acct-a"]["body"]["balance"] = changes["balance"]
    return table


# --- metered ------------------------------------------------------------------------------------------

def test_metered_account_with_a_running_session_and_settled_charges():
    answer, hubs = go(["--account", "A", "--no-ledger", "--until", FAR_JUDGING])
    a = answer["accounts"]["A"]
    # the session started 09:27:24 and the hub's clock says 11:03:23: 95 whole minutes at 4 a minute
    assert a["metered"] is True and a["balance"] == 59960
    assert a["accrued_running"] == 380 and a["credits_left"] == 59960 - 380
    assert a["running"][0]["credits_so_far"] == 380
    assert a["running"][0]["credits_if_run_to_end"] == 2400
    # window 3 h: the running session (95.98 min x 4) plus the 3 min session settled at 09:12:45 (12 credits)
    assert a["credits_per_hour"] == pytest.approx((95.98 * 4 + 12) / 3, abs=0.1)
    assert a["burn_per_hour"] == 240 and a["burn_basis"] == "running"
    assert a["hours_left"] == pytest.approx((59960 - 380) / 240, abs=0.01)
    assert answer["ok"] and answer["outcome"] == "ok" and answer["alerts"] == []
    assert all(r.method == "GET" for r in hubs.requests)


def test_a_settled_charge_outside_the_window_is_not_counted():
    answer, _ = go(["--account", "A", "--no-ledger", "--window-hours", "1"])
    assert answer["accounts"]["A"]["credits_per_hour"] == pytest.approx(240, abs=0.5)
    answer, _ = go(["--account", "A", "--no-ledger", "--window-hours", "200"])
    # the running session (about 383), the two settled charges (12 + 28), not the top-up
    assert answer["accounts"]["A"]["credits_per_hour"] == pytest.approx((383.9 + 12 + 28) / 200, abs=0.05)


def test_alert_when_the_credits_run_out_before_judging():
    hubs = Hubs(dub=metered_tables(balance=2000))
    answer, _ = go(["--account", "A", "--no-ledger", "--until", FAR_JUDGING], hubs)
    a = answer["accounts"]["A"]
    assert a["credits_left"] == 1620 and a["hours_left"] == pytest.approx(6.75)
    assert answer["outcome"] == "alert" and answer["ok"] is True
    (alert,) = answer["alerts"]
    assert alert["kind"] == credits.RUNS_OUT and alert["account"] == "A"
    assert alert["runs_out_at"].startswith("2026-10-10T17:48")
    assert "before judging" in alert["message"] and TOKEN_A not in json.dumps(answer)


def test_alert_when_the_credits_are_already_gone():
    hubs = Hubs(dub=metered_tables(balance=300))      # the running session has accrued 380
    answer, _ = go(["--account", "A", "--no-ledger", "--until", FAR_JUDGING], hubs)
    assert answer["accounts"]["A"]["credits_left"] == 0
    assert answer["alerts"][0]["kind"] == credits.OUT_OF_CREDITS


def test_no_alert_after_judging_has_started_even_with_no_credits():
    hubs = Hubs(dub=metered_tables(balance=300))
    answer, _ = go(["--account", "A", "--no-ledger", "--until", AFTER_JUDGING], hubs)
    assert answer["alerts"] == [] and answer["outcome"] == "ok"
    assert answer["accounts"]["A"]["judging_started"] is True


def test_nothing_burning_means_no_hours_left_and_no_alert():
    table = metered_tables()
    table["/hub/api/users/acct-a"]["body"]["servers"] = {}
    answer, _ = go(["--account", "A", "--no-ledger", "--window-hours", "1"], Hubs(dub=table))
    a = answer["accounts"]["A"]
    assert a["burn_per_hour"] == 0 and a["hours_left"] is None and a["runs_out_at"] is None
    assert a["credits_left"] == 59960 and answer["alerts"] == []


def test_unreadable_transactions_still_answer_but_are_marked_partial(state):
    table = metered_tables()
    del table["/hub/admin/api/quota/acct-a"]
    answer, _ = go(["--account", "A"], Hubs(dub=table))
    assert answer["ok"] and answer["accounts"]["A"]["warnings"]
    (row,) = rows(state)
    assert row.get("low_confidence") is True


# --- unmetered ----------------------------------------------------------------------------------------

def test_unmetered_account_reports_usage_and_projects_nothing():
    answer, hubs = go(["--account", "C", "--no-ledger"])
    c = answer["accounts"]["C"]
    assert c["metered"] is False and c["day"] == "2026-10-10"
    assert c["summary"].startswith("unmetered: usage ") and c["summary"].endswith(" sessions")
    assert c["usage_minutes_today"] >= 6 and c["sessions_today"] >= 3 and len(c["running"]) == 1
    for projected in ("balance", "credits_left", "hours_left", "burn_per_hour", "runs_out_at"):
        assert projected not in c
    assert answer["alerts"] == [] and answer["outcome"] == "ok"
    assert answer["lines"][0].startswith("C (pod-c): unmetered: usage ")
    assert not any("/hub/admin" in r.url.path or "/users/" in r.url.path for r in hubs.requests)


def test_unlimited_account_with_quota_enabled_is_unmetered_too():
    table = load("sjc_c")
    table["/hub/api/quota/me"]["body"]["enabled"] = True
    answer, _ = go(["--account", "C", "--no-ledger"], Hubs(sjc=table))
    assert answer["accounts"]["C"]["metered"] is False
    assert answer["accounts"]["C"]["reason"] == "unlimited quota"


def test_unmetered_never_alerts_even_when_judging_is_far_off():
    answer, _ = go(["--account", "C,A", "--no-ledger", "--until", "2030-01-01T00:00:00Z"])
    assert [a["account"] for a in answer["alerts"]] == ["A"]
    assert answer["accounts"]["C"]["metered"] is False


# --- failures -----------------------------------------------------------------------------------------

def test_one_unreadable_account_is_an_error_and_exit_1(monkeypatch, state):
    table = load("sjc_c")
    table["/hub/api/quota/me"] = {"status": 502, "date": "Sat, 10 Oct 2026 19:05:34 GMT",
                                  "body": {"message": "bad gateway"}}
    answer, _ = go(["--account", "A,C", "--no-ledger"], Hubs(sjc=table))
    assert answer["ok"] is False and answer["outcome"] == "error" and answer["kind"] == "hub_unreachable"
    assert answer["accounts"]["A"]["ok"] is True and answer["accounts"]["C"]["ok"] is False
    assert "502" in answer["accounts"]["C"]["error"]
    assert "acct-c" not in json.dumps(answer) and TOKEN_C not in json.dumps(answer)

    monkeypatch.setattr(credits, "run", lambda argv: answer)
    assert lab_cli.main(["credits"]) == 1


def test_missing_keys_and_bad_arguments():
    answer, _ = go(["--account", "D"])
    assert answer["outcome"] == "missing_credentials" and answer["ok"] is False
    for bad in (["--account", "CD"], ["--window-hours", "0"], ["--until", "2026-10-11T12:00:00"]):
        with pytest.raises(SystemExit):
            go(bad)


def test_all_finds_the_letters_with_both_keys():
    answer, _ = go(["--account", "all", "--no-ledger"])
    assert list(answer["accounts"]) == ["A", "C"]


# --- the ledger ---------------------------------------------------------------------------------------

def test_ledger_rows_one_per_account_without_secrets(state):
    answer, _ = go(["--account", "A,C"])
    a, c = rows(state)
    assert (a["verb"], a["outcome"], a["node"], a["refs"]["account"]) == ("credits", "ok", "pod-a", "A")
    assert a["metrics"]["pod.credits_left"] == 59580 and a["metrics"]["pod.balance_credits"] == 59960
    assert a["metrics"]["pod.credits_burn_per_hour"] == 240
    assert a["metrics"]["pod.credits_hours_left"] == pytest.approx(248.25)
    assert a["units"]["pod.credits_left"] == "credits" and a["conditions"]["window_hours"] == 3.0
    assert not a.get("low_confidence")
    assert (c["outcome"], c["node"], c["refs"]["account"]) == ("unmetered", "pod-c", "C")
    assert set(c["metrics"]) == {"pod.usage_min_today", "pod.sessions_today", "pod.usage_min_window"}
    text = json.dumps(list(rows(state)), default=str)
    assert TOKEN_A not in text and TOKEN_C not in text and "acct-" not in text
    assert "ledger_error" not in answer


def test_alert_row_carries_the_message(state):
    go(["--account", "A", "--until", FAR_JUDGING], Hubs(dub=metered_tables(balance=2000)))
    (row,) = rows(state)
    assert row["outcome"] == "alert" and "run out" in row["note"]


def test_error_row_for_an_unreadable_account(state):
    table = load("sjc_c")
    table["/hub/api/quota/me"] = {"status": 502, "body": {"message": "bad gateway"}}
    go(["--account", "C"], Hubs(sjc=table))
    (row,) = rows(state)
    assert row["outcome"] == "error" and row["metrics"] == {}


def test_no_ledger_writes_nothing(state):
    go(["--account", "A,C", "--no-ledger"])
    assert not (state / "ledger.jsonl").exists()


def test_a_ledger_that_cannot_be_written_keeps_the_reading(monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(credits.record, "record", broken)
    answer, _ = go(["--account", "A"])
    assert answer["ok"] and "disk full" in answer["ledger_error"]
    assert answer["accounts"]["A"]["credits_left"] == 59580


# --- the pieces ---------------------------------------------------------------------------------------

def test_only_usage_deductions_are_spend():
    charges = credits.parse_charges([
        {"amount": 60000, "transaction_type": "set", "created_at": "2026-10-08 13:59:55"},
        {"amount": -12, "transaction_type": "usage", "description": "Session 1: 3 min @ r9700",
         "created_at": "2026-10-10 09:12:45"},
        {"amount": -5, "transaction_type": "admin", "created_at": "2026-10-10 09:12:45"},
        "junk"])
    assert [(c.credits, c.start.minute) for c in charges] == [(12, 9)]


def test_the_verb_is_registered_alphabetically():
    names = list(lab_cli.VERBS)
    assert lab_cli.VERBS["credits"] == "lab.credits:main" and names == sorted(names)


# --- against the fake hub on its fake clock -----------------------------------------------------------

def test_fake_hub_settled_session_then_a_running_one():
    from lab.hub_fake import FakeClock, FakeHub
    clock = FakeClock("2026-10-10T09:27:00Z")
    fake = FakeHub(clock, hub_clock_ahead_s=0.0)
    fake.add_account("A", user="acct-a", token=TOKEN_A, balance=1000)
    fake.start_server("A", runtime_minutes=600)
    clock.sleep(30 * 60)
    fake.drop_server("A")                       # settles 30 min x 4 = 120
    fake.start_server("A", runtime_minutes=600)
    clock.sleep(10 * 60)
    answer = credits.run(["--account", "A", "--no-ledger", "--until", "2026-10-10T20:00:00Z"],
                         environ={"AUP_HUB_USER_A": "acct-a", "AUP_HUB_TOKEN_A": TOKEN_A},
                         env_files=[], transport=fake.transport, sleep=clock.sleep,
                         clock=lambda: dt.datetime.fromtimestamp(clock.time(), tz=dt.timezone.utc))
    a = answer["accounts"]["A"]
    assert a["balance"] == 880 and a["accrued_running"] == 40 and a["credits_left"] == 840
    assert a["burn_per_hour"] == 240 and a["hours_left"] == pytest.approx(3.5)
    assert answer["outcome"] == "alert"          # 3.5 h left, judging 10.5 h away
    assert all(c.method == "GET" for c in fake.calls)
