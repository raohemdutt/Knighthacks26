"""lab.controller: launching, stopping, watching and keeping alive the two AUP pods.

Every test drives the real Controller (and the real Hub client under it). The only double is the hub itself,
lab.hub_fake.FakeHub, which answers with the shapes the hub was recorded to use and moves on a fake clock, so
nothing here touches the network, sleeps for real, or launches or stops a pod. Each test works in its own
state dir; the ledger rows it checks are the ones the real ledger wrote."""
from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
import re
import shlex
import subprocess
import sys
import threading
from pathlib import Path
from unittest import mock

import httpx
import pytest

from lab import controller as ctl_module
from lab.controller import Controller
from lab.hub import load_accounts
from lab.hub_fake import FakeClock, FakeHub, network
from lab.machine import alerts, allowed, ledger, stop

REPO = Path(__file__).resolve().parents[1]
TOKEN_A = "aaaa1111bbbb2222cccc3333dddd4444"
TOKEN_B = "eeee5555ffff6666aaaa7777bbbb8888"
ENV = {"AUP_HUB_USER_A": "acct-a", "AUP_HUB_TOKEN_A": TOKEN_A, "AUP_HUB_USER_B": "acct-b", "AUP_HUB_TOKEN_B": TOKEN_B}
LAUNCH_KEYS = {"runtime_minutes": 600, "resource_type": "gpu", "gpu_selection": "strix-halo"}


class World:
    """Both accounts on one fake hub, a controller wired to it, and a private state dir."""

    def __init__(self, state: Path):
        self.state = state
        self.clock = FakeClock("2026-10-10T09:27:00Z")
        self.hub = FakeHub(self.clock)
        self.a = self.hub.add_account("A", user="acct-a", token=TOKEN_A, balance=59960)
        self.b = self.hub.add_account("B", user="acct-b", token=TOKEN_B, balance=59628)
        self.controller = self.make_controller()

    def make_controller(self, base_url="https://hub.test"):
        return Controller({"A": self.a, "B": self.b}, base_url=base_url, transport=self.hub.transport,
                          sleep=self.clock.sleep, clock=self.clock.time)

    def rows(self, verb=None):
        return ledger.read(verb=verb) if verb else ledger.read()

    def only_row(self, verb):
        rows = self.rows(verb)
        assert len(rows) == 1, rows
        return rows[0]


@pytest.fixture
def world(tmp_path, monkeypatch):
    state = tmp_path / "state"
    monkeypatch.setenv("LAB_STATE_DIR", str(state))
    monkeypatch.setenv("LAB_ACTOR", "tester")
    for name in ("LAB_TIER", "LAB_TASK"):
        monkeypatch.delenv(name, raising=False)
    return World(state)


def aware_now():
    return dt.datetime.now(dt.timezone.utc)


# --- launch ---------------------------------------------------------------------------------

def test_launch_posts_the_launch_keys_polls_until_ready_and_times_the_start(world):
    answer = world.controller.launch("A", 600)

    posts = world.hub.calls_matching("POST", "/server")
    assert [(call.path, call.json) for call in posts] == [("/hub/api/users/acct-a/server", LAUNCH_KEYS)]
    assert answer["ok"] is True
    assert answer["outcome"] == "running"
    assert (answer["command"], answer["account"], answer["node"]) == ("launch", "A", "pod-a")
    assert answer["start_s"] == 15.0
    assert answer["balance"] == 59960
    assert answer["rate"] == 4
    assert answer["cost_credits"] == 2400
    assert answer["server"]["ready"] is True
    assert answer["server"]["runtime_minutes"] == 600
    assert answer["server"]["scheduled_end"] == "2026-10-10T19:27:46Z"
    assert world.hub.running("A") is True
    assert world.hub.running("B") is False


def test_launch_leaves_a_pod_row_with_its_node_and_a_row_for_the_call(world):
    world.controller.launch("A", 600)

    pod = world.only_row("pod launch")
    assert (pod["node"], pod["outcome"], pod["actor"]) == ("pod-a", "running", "tester")
    assert pod["metrics"] == {"pod.start_s": 15.0, "pod.balance_credits": 59960}
    assert pod["units"] == {"pod.start_s": "s", "pod.balance_credits": "credits"}
    assert pod["conditions"] == {"resource": "gpu", "accelerator": "strix-halo", "launch_path": "rest", "hub": "hub.test",
                                 "account": "A"}
    assert pod["refs"]["account"] == "A"

    call = world.only_row("controller launch")
    assert call["outcome"] == "running"
    assert call["refs"]["account"] == "A"
    assert call["metrics"] == {}
    assert call["duration_s"] >= 0
    assert "node" not in call


def test_launch_takes_any_runtime_and_does_not_cap_it(world):
    world.controller.launch("B", 2160)
    assert world.hub.calls_matching("POST", "/server")[0].json["runtime_minutes"] == 2160
    assert world.hub.calls_matching("POST", "/server")[0].path == "/hub/api/users/acct-b/server"


def test_launch_on_a_ready_pod_is_a_no_op_and_can_be_repeated(world):
    world.hub.start_server("A")

    first = world.controller.launch("A", 600)
    second = world.controller.launch("A", 600)

    assert [first["outcome"], second["outcome"]] == ["already_running", "already_running"]
    assert first["ok"] is True and second["ok"] is True
    assert world.hub.calls_matching("POST", "/server") == []
    assert first["server"]["running"] is True
    assert [row["outcome"] for row in world.rows("pod launch")] == ["already_running", "already_running"]


def test_launch_joins_a_spawn_that_is_already_pending_instead_of_posting_again(world):
    world.hub.start_server("A", ready=False)

    answer = world.controller.launch("A", 600)

    assert answer["outcome"] == "running"
    assert answer["ok"] is True
    assert world.hub.calls_matching("POST", "/server") == []


def test_launch_waits_out_a_pending_stop_and_then_posts_once(world):
    world.hub.start_server("A")
    world.hub.stop_s = 20.0
    assert world.controller.stop("A", wait=False)["outcome"] == "stopping"

    answer = world.controller.launch("A", 600)

    assert answer["outcome"] == "running"
    posts = world.hub.calls_matching("POST", "/server")
    deletes = world.hub.calls_matching("DELETE", "/server")
    assert len(posts) == 1 and len(deletes) == 1
    assert posts[0].t - deletes[0].t >= 20.0


def test_launch_replaces_a_server_the_hub_still_lists_after_its_spawn_failed(world):
    world.hub.start_failed_server("A")

    answer = world.controller.launch("A", 600)

    assert answer["outcome"] == "running"
    assert len(world.hub.calls_matching("POST", "/server")) == 1


def test_a_400_already_running_from_the_hub_counts_as_success(world):
    world.hub.queue("POST", "/server", 400, body={"status": 400, "message": "acct-a is already running"})

    answer = world.controller.launch("A", 600)

    assert answer["outcome"] == "already_running"
    assert answer["ok"] is True
    assert len(world.hub.calls_matching("POST", "/server")) == 1


def test_launch_is_refused_before_any_post_when_the_balance_cannot_cover_the_runtime(world):
    world.hub.set_balance("A", 100)

    answer = world.controller.launch("A", 600)

    assert answer["ok"] is False
    assert answer["outcome"] == "refused_quota"
    assert (answer["balance"], answer["cost_credits"], answer["max_minutes"]) == (100, 2400, 25)
    assert world.hub.calls_matching("POST", "/server") == []
    assert world.only_row("pod launch")["outcome"] == "refused_quota"
    assert world.only_row("controller launch")["outcome"] == "refused_quota"


def test_launch_reports_a_quota_refusal_the_hub_makes_itself(world):
    message = ("Cannot start container: Insufficient quota for 600 min (balance: 100, need: 2400, max: 25 min). "
               "Please contact administrator to add quota.")
    world.hub.queue("POST", "/server", 403, body={"status": 403, "message": message})

    answer = world.controller.launch("A", 600)

    assert answer["ok"] is False
    assert answer["outcome"] == "refused_quota"
    assert answer["error"] == message
    assert len(world.hub.calls_matching("POST", "/server")) == 1


def test_a_post_the_hub_answered_with_502_but_acted_on_is_found_by_reading_the_server_back(world):
    world.hub.queue("POST", "/server", 502, text="Bad gateway", apply=True)

    answer = world.controller.launch("A", 600)

    assert answer["outcome"] == "running"
    assert answer["ok"] is True
    assert answer["post_status"] == 502
    assert len(world.hub.calls_matching("POST", "/server")) == 1


def test_a_post_the_hub_never_acted_on_is_reported_as_unconfirmed_and_not_repeated(world):
    world.hub.queue("POST", "/server", 502, text="Bad gateway")

    answer = world.controller.launch("A", 600)

    assert answer["ok"] is False
    assert (answer["outcome"], answer["kind"]) == ("error", "launch_unconfirmed")
    assert len(world.hub.calls_matching("POST", "/server")) == 1


def test_launch_stops_before_posting_when_the_hub_cannot_be_read(world):
    world.hub.queue("GET", "/hub/api/users/acct-a", 502, text="Bad gateway", times=100)

    answer = world.controller.launch("A", 600)

    assert answer["ok"] is False
    assert (answer["outcome"], answer["kind"], answer["status"]) == ("error", "hub_unreachable", 502)
    assert world.hub.calls_matching("POST", "/server") == []
    assert world.only_row("pod launch")["outcome"] == "error"


def test_launch_times_out_without_touching_the_pod_it_started(world):
    world.hub.spawn_s = 10_000.0

    answer = world.controller.launch("A", 600, timeout_s=60)

    assert answer["ok"] is False
    assert answer["outcome"] == "timeout"
    assert world.hub.calls_matching("DELETE", "/server") == []
    assert world.hub.running("A") is True


def test_launch_notices_a_spawn_that_disappears(world):
    world.hub.spawn_s = 5.0
    world.hub.spawn_fails = True

    answer = world.controller.launch("A", 600)

    assert answer["ok"] is False
    assert answer["outcome"] == "failed"


def test_dry_run_checks_the_balance_and_posts_nothing(world):
    answer = world.controller.launch("A", 600, dry_run=True)

    assert answer["ok"] is True
    assert answer["outcome"] == "dry_run"
    assert answer["plan"] == {"method": "POST", "path": "/hub/api/users/acct-a/server", "json": LAUNCH_KEYS}
    assert (answer["balance"], answer["cost_credits"]) == (59960, 2400)
    assert world.hub.calls_matching("POST") == []
    assert world.rows("pod launch") == []


def test_launch_without_waiting_returns_when_the_hub_accepts(world):
    answer = world.controller.launch("A", 600, wait=False)

    assert answer["ok"] is True
    assert answer["outcome"] == "launching"
    assert "start_s" not in answer
    assert world.only_row("pod launch")["outcome"] == "launching"
    assert world.clock.elapsed == 0.0


@pytest.mark.parametrize("minutes", [0, -5, 2.5, "600", True])
def test_a_runtime_that_is_not_a_positive_whole_number_is_refused(world, minutes):
    with pytest.raises(ValueError):
        world.controller.launch("A", minutes)
    assert world.hub.calls == []
    assert world.only_row("controller launch")["outcome"] == "error"


def test_the_automatic_accelerator_is_refused_because_the_card_must_be_named(world):
    for accelerator in ("auto", "AUTO ", ""):
        with pytest.raises(ValueError):
            world.controller.launch("A", 10, accelerator=accelerator)
    assert world.hub.calls == []


def test_the_r9700_is_still_a_card_a_launch_can_name(world):
    world.controller.launch("A", 10, accelerator="r9700")

    assert [call.json["gpu_selection"] for call in world.hub.calls_matching("POST", "/server")] == ["r9700"]
    assert world.only_row("pod launch")["conditions"]["accelerator"] == "r9700"


def test_an_unknown_account_is_refused(world):
    with pytest.raises(KeyError):
        world.controller.launch("C", 10)


# --- stop -----------------------------------------------------------------------------------

def test_stop_deletes_once_and_reads_the_charge_from_the_balance(world):
    world.hub.start_server("A", started_ago_s=3690)

    answer = world.controller.stop("A")

    assert [call.path for call in world.hub.calls_matching("DELETE")] == ["/hub/api/users/acct-a/server"]
    assert answer["ok"] is True
    assert (answer["outcome"], answer["state"]) == ("ok", "stopped")
    assert answer["minutes_run"] == 61.5
    assert (answer["balance_before"], answer["balance_after"], answer["credits_charged"]) == (59960, 59716, 244)
    assert world.hub.transactions("A")[0]["description"] == "Session 1: 61 min @ strix-halo (4/min)"
    assert world.hub.running("A") is False

    pod = world.only_row("pod stop")
    assert (pod["node"], pod["outcome"]) == ("pod-a", "ok")
    assert pod["metrics"] == {"pod.balance_credits": 59716}
    assert world.only_row("controller stop")["outcome"] == "ok"


def test_stopping_a_stopped_account_is_a_no_op_and_sends_no_delete(world):
    world.hub.start_server("A")
    first = world.controller.stop("A")
    second = world.controller.stop("A")

    assert (first["outcome"], second["outcome"]) == ("ok", "already_stopped")
    assert second["ok"] is True
    assert len(world.hub.calls_matching("DELETE")) == 1
    assert [row["outcome"] for row in world.rows("pod stop")] == ["ok", "already_stopped"]


def test_a_stop_the_hub_answers_with_202_is_waited_out_and_the_charge_still_read(world):
    world.hub.start_server("A", started_ago_s=3690)
    world.hub.stop_s = 30.0

    answer = world.controller.stop("A")

    assert answer["ok"] is True
    assert answer["outcome"] == "ok"
    assert answer["credits_charged"] == 244
    assert world.hub.running("A") is False


def test_stop_times_out_when_the_hub_keeps_the_pod_stopping(world):
    world.hub.start_server("A")
    world.hub.stop_s = 10_000.0

    answer = world.controller.stop("A", timeout_s=30)

    assert answer["ok"] is False
    assert answer["outcome"] == "timeout"


def test_stop_without_waiting_reports_stopping(world):
    world.hub.start_server("A")
    world.hub.stop_s = 30.0

    answer = world.controller.stop("A", wait=False)

    assert (answer["outcome"], answer["state"], answer["ok"]) == ("stopping", "stopping", True)
    assert world.hub.running("A") is True


def test_stop_does_nothing_when_the_hub_cannot_be_read(world):
    world.hub.start_server("A")
    world.hub.queue("GET", "/hub/api/users/acct-a", 502, text="Bad gateway", times=100)

    answer = world.controller.stop("A")

    assert (answer["ok"], answer["outcome"], answer["kind"]) == (False, "error", "hub_unreachable")
    assert world.hub.calls_matching("DELETE") == []


# --- wait -----------------------------------------------------------------------------------

def test_wait_polls_a_launching_pod_until_it_is_ready(world):
    world.hub.start_server("B", ready=False)

    answer = world.controller.wait_ready("B", timeout_s=120)

    assert (answer["ok"], answer["outcome"], answer["waited_s"]) == (True, "ready", 15.0)
    assert answer["server"]["ready"] is True


def test_wait_on_an_account_with_no_server_does_not_wait(world):
    answer = world.controller.wait_ready("B", timeout_s=120)

    assert (answer["ok"], answer["outcome"], answer["waited_s"]) == (False, "not_running", 0.0)
    assert world.clock.elapsed == 0.0


def test_wait_gives_up_at_the_timeout(world):
    world.hub.spawn_s = 10_000.0
    world.hub.start_server("B", ready=False)

    answer = world.controller.wait_ready("B", timeout_s=30)

    assert (answer["ok"], answer["outcome"]) == (False, "timeout")


# --- status ---------------------------------------------------------------------------------

def test_status_shows_both_accounts_with_balance_spend_and_the_scheduled_end(world):
    world.hub.start_server("A", started_ago_s=3690, idle_s=450)

    answer = world.controller.status()

    assert answer["ok"] is True
    assert (answer["command"], answer["outcome"], answer["hub"]) == ("status", "ok", "https://hub.test")
    assert answer["hub_clock_offset_s"] == 46.0
    assert answer["accounts"]["A"] == {
        "node": "pod-a", "reachable": True, "running": True, "ready": True, "pending": None,
        "started": "2026-10-10T08:26:16Z", "runtime_minutes": 600, "scheduled_end": "2026-10-10T18:26:16Z",
        "minutes_run": 61.5, "minutes_left": 538.5, "last_activity": "2026-10-10T09:20:16Z", "idle_minutes": 7.5,
        "resource_type": "gpu", "accelerator": "strix-halo",
        "balance": 59960, "unlimited": False, "quota_enabled": True, "rate": 4,
        "credits_so_far": 244, "credits_if_run_to_end": 2400, "balance_if_stopped_now": 59716,
        "balance_at_scheduled_end": 57560, "exceeds_balance": False}
    assert answer["accounts"]["B"] == {
        "node": "pod-b", "reachable": True, "running": False,
        "balance": 59628, "unlimited": False, "quota_enabled": True, "rate": 4}


def test_status_reads_only_the_hub_api_so_it_never_resets_the_idle_clock(world):
    world.hub.start_server("A", idle_s=450)
    before = world.hub.last_activity("A")

    world.controller.status()

    assert world.hub.calls_matching(contains="/user/") == []
    assert world.hub.last_activity("A") == before


def test_status_leaves_one_pod_row_per_account_with_the_balance(world):
    world.hub.start_server("A")

    world.controller.status()

    rows = world.rows("pod status")
    assert [(row["node"], row["outcome"], row["metrics"]) for row in rows] == [
        ("pod-a", "running", {"pod.balance_credits": 59960}),
        ("pod-b", "not_running", {"pod.balance_credits": 59628})]
    assert rows[0]["units"] == {"pod.balance_credits": "credits"}
    assert rows[0]["refs"]["account"] == "A"
    assert world.only_row("controller status")["outcome"] == "ok"


def test_status_says_so_when_the_hub_does_not_answer_for_one_account(world):
    world.hub.start_server("A")
    world.hub.queue("GET", "/hub/api/users/acct-b", 502, text="Bad gateway", times=100)

    answer = world.controller.status()

    assert answer["ok"] is False
    assert (answer["outcome"], answer["kind"]) == ("error", "hub_unreachable")
    assert answer["accounts"]["A"]["running"] is True
    assert answer["accounts"]["B"]["reachable"] is False
    assert "502" in answer["accounts"]["B"]["error"]
    assert [row["outcome"] for row in world.rows("pod status")] == ["running", "error"]


def test_status_of_an_account_without_quota_accounting_shows_no_credit_figures(world):
    world.hub.set_quota_enabled("A", False)
    world.hub.start_server("A")

    entry = world.controller.status(["A"])["accounts"]["A"]

    assert entry["quota_enabled"] is False
    assert entry["unlimited"] is True
    for key in ("credits_so_far", "credits_if_run_to_end", "balance_if_stopped_now", "balance_at_scheduled_end"):
        assert key not in entry
    assert world.only_row("pod status")["metrics"] == {}


def test_status_can_probe_the_model_server_and_says_that_it_counts_as_activity(world):
    world.hub.start_server("A", idle_s=450)
    world.hub.serve_model("A", port=8000, models=("teacher",))
    before = world.hub.last_activity("A")

    answer = world.controller.status(["A"], probe_port=8000)

    assert answer["accounts"]["A"]["serving"] == {"port": 8000, "models": ["teacher"], "health": 200}
    assert [call.path for call in world.hub.calls_matching(contains="/proxy/")] == [
        "/user/acct-a/proxy/8000/v1/models", "/user/acct-a/proxy/8000/health"]
    assert world.hub.last_activity("A") > before


def test_a_probe_of_a_model_server_that_is_down_reports_the_health_code(world):
    world.hub.start_server("A")
    world.hub.serve_model("A", port=8000, models=("teacher",), healthy=False)

    entry = world.controller.status(["A"], probe_port=8000)["accounts"]["A"]

    assert entry["serving"]["health"] == 503


def test_status_asks_only_for_the_accounts_it_is_given(world):
    answer = world.controller.status(["B"])

    assert list(answer["accounts"]) == ["B"]
    assert world.hub.calls_matching(contains="acct-a") == []


# --- quota ----------------------------------------------------------------------------------

def test_quota_reads_the_balance_and_rates_for_each_account(world):
    answer = world.controller.quota()

    assert answer["ok"] is True
    assert answer["accounts"]["A"] == {"node": "pod-a", "reachable": True, "balance": 59960, "unlimited": False,
                                       "enabled": True, "rate": 4, "rates": {"cpu": 1, "strix-halo": 4, "r9700": 4}}
    assert answer["accounts"]["B"]["balance"] == 59628
    assert [(row["node"], row["metrics"]) for row in world.rows("pod quota")] == [
        ("pod-a", {"pod.balance_credits": 59960}), ("pod-b", {"pod.balance_credits": 59628})]


def test_quota_can_add_the_recent_transactions_of_the_account(world):
    world.hub.start_server("A", started_ago_s=3690)
    world.controller.stop("A")

    answer = world.controller.quota(["A"], transactions=True)

    transactions = answer["accounts"]["A"]["transactions"]
    assert [(t["amount"], t["transaction_type"], t["balance_before"], t["balance_after"]) for t in transactions] == [
        (-244, "usage", 59960, 59716)]
    assert transactions[0]["description"] == "Session 1: 61 min @ strix-halo (4/min)"
    assert "/hub/admin/api/quota/acct-a" in [call.path for call in world.hub.calls]


# --- keep-alive -----------------------------------------------------------------------------

def test_keep_alive_pings_every_ten_minutes_until_the_time_is_up(world):
    world.hub.start_server("A")

    answer = world.controller.keep_alive(["A"], minutes=35)

    pings = world.hub.calls_matching("GET", "/user/acct-a/api/me")
    assert [call.t for call in pings] == [0.0, 600.0, 1200.0, 1800.0]
    assert answer["ok"] is True
    assert (answer["outcome"], answer["interval_s"], answer["minutes"]) == ("completed", 600, 35.0)
    assert answer["accounts"]["A"]["state"] == "alive"
    assert (answer["accounts"]["A"]["pings"], answer["accounts"]["A"]["failures"]) == (4, 0)
    assert answer["accounts"]["A"]["last_ping"] == "2026-10-10T09:57:46Z"
    assert [row["outcome"] for row in world.rows("pod keep-alive")] == ["alive"]
    assert world.only_row("pod keep-alive")["node"] == "pod-a"
    assert world.only_row("controller keep-alive")["outcome"] == "completed"


def test_the_idle_culler_stops_the_pod_nobody_pings_and_spares_the_one_that_is_pinged(world):
    world.hub.start_server("A")
    world.hub.start_server("B")

    world.controller.keep_alive(["A"], minutes=130)

    assert world.hub.running("A") is True
    assert world.hub.running("B") is False
    assert world.hub.transactions("B")[0]["description"] == "Session 1: 60 min @ strix-halo (4/min)"


def test_keep_alive_reports_a_pod_that_disappears_and_stops_when_none_is_left(world):
    world.hub.start_server("A")
    world.hub.schedule(25 * 60, lambda: world.hub.drop_server("A"))

    answer = world.controller.keep_alive(["A"], minutes=120)

    assert answer["ok"] is False
    assert answer["outcome"] == "lost"
    assert answer["accounts"]["A"]["state"] == "lost"
    assert answer["accounts"]["A"]["pings"] == 3
    assert answer["minutes"] == 30.0
    assert [row["outcome"] for row in world.rows("pod keep-alive")] == ["alive", "lost"]


def test_keep_alive_keeps_the_other_pod_going_when_one_is_lost(world):
    world.hub.start_server("A")
    world.hub.start_server("B")
    world.hub.schedule(25 * 60, lambda: world.hub.drop_server("B"))

    answer = world.controller.keep_alive(["A", "B"], minutes=65)

    assert answer["outcome"] == "lost"
    assert answer["accounts"]["A"]["state"] == "alive"
    assert answer["accounts"]["B"]["state"] == "lost"
    assert answer["accounts"]["A"]["pings"] == 7
    assert answer["accounts"]["B"]["pings"] == 3
    assert world.hub.running("A") is True


def test_keep_alive_survives_a_failed_ping_and_notes_the_recovery(world):
    world.hub.start_server("A")
    world.hub.queue("GET", "/user/acct-a/api/me", 502, text="Bad gateway", times=4)

    answer = world.controller.keep_alive(["A"], minutes=25)

    assert answer["ok"] is True
    assert answer["outcome"] == "completed"
    assert answer["accounts"]["A"]["failures"] == 1
    assert answer["accounts"]["A"]["state"] == "alive"
    assert [row["outcome"] for row in world.rows("pod keep-alive")] == ["failed_ping", "alive"]


def test_keep_alive_that_cannot_reach_the_pod_at_the_end_says_unreachable(world):
    world.hub.start_server("A")
    world.hub.queue("GET", "/user/acct-a/api/me", 502, text="Bad gateway", times=1000)

    answer = world.controller.keep_alive(["A"], minutes=25)

    assert answer["ok"] is False
    assert answer["outcome"] == "unreachable"
    assert answer["accounts"]["A"]["state"] == "failing"
    assert answer["accounts"]["A"]["pings"] == 0
    assert [row["outcome"] for row in world.rows("pod keep-alive")] == ["failed_ping"]


def test_keep_alive_does_not_ping_a_pod_that_is_still_launching(world):
    world.hub.spawn_s = 10_000.0
    world.hub.start_server("A", ready=False)

    answer = world.controller.keep_alive(["A"], minutes=15)

    assert world.hub.calls_matching("GET", "/user/acct-a/api/me") == []
    assert answer["accounts"]["A"]["state"] == "launching"
    assert answer["ok"] is True


def test_keep_alive_with_nothing_running_ends_at_once(world):
    answer = world.controller.keep_alive(["A"], minutes=60)

    assert answer["ok"] is False
    assert answer["outcome"] == "not_running"
    assert answer["accounts"]["A"]["state"] == "not_running"
    assert world.clock.elapsed == 0.0
    assert [(row["node"], row["outcome"]) for row in world.rows("pod keep-alive")] == [("pod-a", "not_running")]


def test_the_kill_switch_ends_a_keep_alive_at_the_next_cycle(world):
    world.hub.start_server("A")

    def raise_the_kill_switch():
        world.state.mkdir(parents=True, exist_ok=True)
        (world.state / "STOP").write_text("pause\n", encoding="utf-8")

    world.hub.schedule(15 * 60, raise_the_kill_switch)

    answer = world.controller.keep_alive(["A"], minutes=120)

    assert answer["ok"] is False
    assert answer["outcome"] == "stopped"
    assert answer["accounts"]["A"]["pings"] == 2
    assert world.only_row("controller keep-alive")["outcome"] == "stopped"


def test_keep_alive_calls_a_pod_lost_when_pings_worked_though_the_hub_could_not_be_read(world):
    world.hub.start_server("A")
    world.hub.queue("GET", "/hub/api/users/acct-a", 502, text="Bad gateway", times=4)
    world.hub.schedule(300, lambda: world.hub.drop_server("A"))

    answer = world.controller.keep_alive(["A"], minutes=30)

    assert answer["outcome"] == "lost"
    assert answer["accounts"]["A"]["pings"] == 1
    assert [row["outcome"] for row in world.rows("pod keep-alive")] == ["alive", "lost"]


@pytest.mark.parametrize("interval", [0, 29, 1201, 1800, -600])
def test_keep_alive_refuses_an_interval_the_culler_would_make_useless_or_wasteful(world, interval):
    world.hub.start_server("A")
    with pytest.raises(ValueError):
        world.controller.keep_alive(["A"], interval_s=interval, minutes=10)
    assert world.hub.calls == []


def test_keep_alive_pings_without_a_bound_until_the_pod_is_gone(world):
    world.hub.start_server("A", runtime_minutes=45)

    answer = world.controller.keep_alive(["A"])

    assert answer["outcome"] == "lost"
    assert answer["accounts"]["A"]["pings"] == 5
    assert answer["accounts"]["A"]["note"] == "the scheduled end passed"


# --- no account name in anything that is shown ------------------------------------------------

def test_an_error_the_hub_words_with_the_account_name_is_shown_without_it(world):
    world.hub.queue("GET", "/hub/api/users/acct-a", 404, body={"status": 404, "message": "No such user acct-a"}, times=100)

    answer = world.controller.status(["A"])

    assert answer["accounts"]["A"]["error"] == "the hub answered 404: No such user <account A>"
    assert "acct-a" not in json.dumps(answer) + ledger.ledger_path().read_text(encoding="utf-8")


def test_a_dropped_connection_is_shown_without_the_account_name_in_its_address(world):
    world.hub.queue_dropped("GET", "/hub/api/users/acct-a", times=100)

    answer = world.controller.status(["A"])

    assert answer["ok"] is False
    assert answer["accounts"]["A"]["error"].startswith("the hub did not answer: GET /hub/api/users/<account A>")
    assert "acct-a" not in json.dumps(answer) + ledger.ledger_path().read_text(encoding="utf-8")


def test_a_dropped_connection_on_the_launch_read_is_a_hub_that_cannot_be_read(world):
    world.hub.queue_dropped("GET", "/hub/api/users/acct-a", times=100)

    answer = world.controller.launch("A", 600)

    assert (answer["ok"], answer["outcome"], answer["kind"]) == (False, "error", "hub_unreachable")
    assert "status" not in answer
    assert world.hub.calls_matching("POST") == []


# --- the rows are what the alert rules read -------------------------------------------------

def alerts_for(world):
    now = aware_now() + dt.timedelta(minutes=1)
    return alerts.scan(ledger.read(), now=now, since=now - dt.timedelta(hours=1))


def test_a_pod_the_controller_saw_disappear_raises_the_stopped_unexpectedly_alert(world):
    world.controller.launch("A", 600)
    world.hub.schedule(25 * 60, lambda: world.hub.drop_server("A"))
    world.controller.keep_alive(["A"], minutes=120)

    found = alerts_for(world)

    assert [(alert.rule, alert.title) for alert in found] == [("pod_stopped", "Pod pod-a stopped unexpectedly")]


def test_a_pod_the_controller_stopped_raises_no_alert_even_when_status_sees_it_gone(world):
    world.controller.launch("A", 600)
    world.controller.stop("A")
    world.controller.status()

    assert alerts_for(world) == []


def test_a_pod_that_was_up_and_is_gone_at_the_next_status_raises_the_alert_once(world):
    world.controller.launch("A", 600)
    world.hub.drop_server("A")
    world.controller.status()
    world.controller.status()

    assert [alert.rule for alert in alerts_for(world)] == ["pod_stopped"]


# --- run-script -----------------------------------------------------------------------------

@contextlib.contextmanager
def fake_terminal(world, letter, *, output="hello from the pod\n", exit_code=0):
    """A Jupyter terminal on localhost. The pod that runs a command is this handler writing the command's
    output file into the fake hub's file store, the way the shell on a real pod would. It echoes the command
    only after the file is written, and the controller waits for that echo, so a slow machine cannot make the
    controller read the file too early."""
    from websockets.sync.server import serve

    received = []

    def handler(connection):
        connection.send(json.dumps(["setup", {}]))
        for raw in connection:
            kind, payload = json.loads(raw)
            received.append({"kind": kind, "payload": payload, "headers": connection.request.headers})
            if kind == "stdin":
                match = re.search(r"> \./(\S+\.out)", payload)
                if match and exit_code is not None:
                    world.hub.files[(letter, match.group(1))] = output + f"### END rc={exit_code}\n"
                elif match:
                    world.hub.files[(letter, match.group(1))] = output
                connection.send(json.dumps(["stdout", payload]))

    with serve(handler, "127.0.0.1", 0) as server, mock.patch.object(ctl_module, "TERMINAL_SETTLE_S", 60.0):
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server.socket.getsockname()[1], received
        finally:
            server.shutdown()
            thread.join(timeout=5)


def script_file(tmp_path, text="echo hello\r\nexit 0\r\n"):
    path = tmp_path / "job.sh"
    path.write_bytes(text.encode("utf-8"))
    return path


def test_run_script_uploads_it_starts_it_in_a_terminal_and_reads_the_exit_code(world, tmp_path):
    world.hub.start_server("B")
    script = script_file(tmp_path)

    with fake_terminal(world, "B") as (port, received):
        controller = world.make_controller(base_url=f"http://127.0.0.1:{port}")
        answer = controller.run_script("B", script, args=["smoke", "two words"], timeout_s=60, run_id="t1")

    assert answer["ok"] is True
    assert answer["outcome"] == "ok"
    assert (answer["account"], answer["node"], answer["run_id"], answer["exit_code"]) == ("B", "pod-b", "t1", 0)
    assert answer["output_tail"] == "hello from the pod"
    assert answer["remote"] == "lab_run_t1.sh"
    assert world.hub.files[("B", "lab_run_t1.sh")] == "echo hello\nexit 0\n"

    stdin = [message for message in received if message["kind"] == "stdin"]
    assert [message["payload"] for message in stdin] == [
        "bash ./lab_run_t1.sh smoke 'two words' > ./lab_run_t1.out 2>&1; echo \"### END rc=$?\" >> ./lab_run_t1.out\n"]
    assert stdin[0]["headers"]["Authorization"] == f"token {TOKEN_B}"
    assert stdin[0]["headers"]["User-Agent"].startswith("Mozilla/5.0")

    uploads = [call.path for call in world.hub.calls_matching("PUT")]
    assert uploads == ["/user/acct-b/api/contents/lab_run_t1.sh", "/user/acct-b/api/contents/lab_run_t1.out"]
    assert [call.path for call in world.hub.calls_matching("POST", "/terminals")] == ["/user/acct-b/api/terminals"]


def test_run_script_uploads_the_files_it_is_given_first_next_to_the_script_and_names_them_in_its_row(world, tmp_path):
    world.hub.start_server("A")
    script = script_file(tmp_path)
    helper = tmp_path / "pod_power.py"
    helper.write_bytes(b"print('sampling')\r\n")

    with fake_terminal(world, "A") as (port, received):
        answer = world.make_controller(base_url=f"http://127.0.0.1:{port}").run_script(
            "A", script, args=["start", "gpu-a"], timeout_s=60, run_id="t7", files=[helper])

    assert (answer["ok"], answer["exit_code"]) == (True, 0)
    uploads = [call.path for call in world.hub.calls_matching("PUT")]
    assert uploads == ["/user/acct-a/api/contents/pod_power.py", "/user/acct-a/api/contents/lab_run_t7.sh",
                       "/user/acct-a/api/contents/lab_run_t7.out"]
    assert world.hub.files[("A", "pod_power.py")] == "print('sampling')\n"
    assert [m["payload"] for m in received if m["kind"] == "stdin"] == [
        "bash ./lab_run_t7.sh start gpu-a > ./lab_run_t7.out 2>&1; echo \"### END rc=$?\" >> ./lab_run_t7.out\n"]
    assert world.only_row("pod run-script")["conditions"]["with"] == ["pod_power.py"]


def test_run_script_takes_with_files_before_the_script_and_refuses_two_files_of_one_name(world, tmp_path):
    args = ctl_module.build_parser().parse_args(
        ["run-script", "--account", "A", "--with", "lab/pod_power.py", "lab/pod_power.sh", "start", "gpu-a"])
    assert (args.files, args.script, args.script_args) == (["lab/pod_power.py"], "lab/pod_power.sh", ["start", "gpu-a"])

    world.hub.start_server("A")
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    for folder in ("one", "two"):
        (tmp_path / folder / "same.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="two files to upload have the same name"):
        world.controller.run_script("A", script_file(tmp_path), timeout_s=60, run_id="t8",
                                    files=[tmp_path / "one" / "same.py", tmp_path / "two" / "same.py"])
    assert world.hub.calls_matching("PUT") == []


def test_run_script_leaves_a_local_row_because_the_pods_own_ledger_is_never_read(world, tmp_path):
    world.hub.start_server("B")
    script = script_file(tmp_path)

    with fake_terminal(world, "B", exit_code=3) as (port, _):
        answer = world.make_controller(base_url=f"http://127.0.0.1:{port}").run_script(
            "B", script, timeout_s=60, run_id="t2")

    assert answer["ok"] is False
    assert (answer["outcome"], answer["exit_code"]) == ("fail", 3)
    row = world.only_row("pod run-script")
    assert (row["node"], row["outcome"], row["actor"]) == ("pod-b", "fail", "tester")
    assert row["refs"]["run_id"] == "t2"
    assert row["conditions"]["script"] == "job.sh"
    assert row["conditions"]["exit_code"] == 3
    assert world.only_row("controller run-script")["outcome"] == "fail"


def test_run_script_copies_the_scripts_step_lines_into_its_row_but_not_the_end_line(world, tmp_path):
    world.hub.start_server("A")
    output = ("teacher up on 8000\n### serve wall_s=312 rc=0 kernel=RDNAHybridW4A16LinearKernel\n"
              "### build wall_s=4 rc=1\n### build wall_s=290 rc=0\n")

    with fake_terminal(world, "A", output=output) as (port, _):
        answer = world.make_controller(base_url=f"http://127.0.0.1:{port}").run_script(
            "A", script_file(tmp_path), timeout_s=60, run_id="t9")

    want = {"serve": {"wall_s": "312", "rc": "0", "kernel": "RDNAHybridW4A16LinearKernel"},
            "build": {"wall_s": "290", "rc": "0"}}
    assert answer["steps"] == want
    assert world.only_row("pod run-script")["conditions"]["steps"] == want


def test_step_marks_reads_only_step_lines():
    assert ctl_module.step_marks("### END rc=0\n### serve\nnot ### a step=1\n### verify note=two words\n") == {}


def test_run_script_times_out_when_the_end_marker_never_appears(world, tmp_path):
    world.hub.start_server("B")
    script = script_file(tmp_path)

    with fake_terminal(world, "B", output="still going\n", exit_code=None) as (port, _):
        answer = world.make_controller(base_url=f"http://127.0.0.1:{port}").run_script(
            "B", script, timeout_s=30, run_id="t3")

    assert answer["ok"] is False
    assert answer["outcome"] == "timeout"
    assert answer["output_tail"] == "still going"
    assert answer["exit_code"] is None


def test_run_script_never_returns_a_token_the_script_printed(world, tmp_path):
    world.hub.start_server("B")
    script = script_file(tmp_path)

    with fake_terminal(world, "B", output=f"token is {TOKEN_B}\n") as (port, _):
        answer = world.make_controller(base_url=f"http://127.0.0.1:{port}").run_script(
            "B", script, timeout_s=60, run_id="t4")

    assert TOKEN_B not in json.dumps(answer)
    assert "token is ***" in answer["output_tail"]


def test_run_script_on_a_stopped_account_uploads_nothing(world, tmp_path):
    answer = world.controller.run_script("B", script_file(tmp_path), timeout_s=60, run_id="t5")

    assert (answer["ok"], answer["outcome"]) == (False, "not_running")
    assert world.hub.calls_matching("PUT") == []
    assert world.only_row("pod run-script")["outcome"] == "not_running"


def test_run_script_reports_a_terminal_it_cannot_reach(world, tmp_path):
    world.hub.start_server("B")
    with socket_closed_port() as port:
        answer = world.make_controller(base_url=f"http://127.0.0.1:{port}").run_script(
            "B", script_file(tmp_path), timeout_s=60, run_id="t6")

    assert (answer["ok"], answer["outcome"], answer["kind"]) == (False, "error", "terminal_unreachable")


@contextlib.contextmanager
def socket_closed_port():
    import socket
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    yield port


# --- accounts C and D on the SJC hub -----------------------------------------------------------

TOKEN_C = "cccc9999dddd0000eeee1111ffff2222"
TOKEN_D = "dddd3333eeee4444ffff5555aaaa6666"
DUB = "https://dub.test"
SJC = "https://sjc.test"
SJC_ENV = {"LAB_POD_ROLES": "C:pod-a,D:pod-b", "AUP_HUB_USER_C": "acct-c", "AUP_HUB_TOKEN_C": TOKEN_C, "AUP_HUB_URL_C": SJC,
           "AUP_HUB_USER_D": "acct-d", "AUP_HUB_TOKEN_D": TOKEN_D, "AUP_HUB_URL_D": SJC}
BOTH_HUBS_ENV = {**ENV, **SJC_ENV, "LAB_HUB_URL": DUB, "LAB_POD_ROLES": "A:pod-a,B:pod-b,C:pod-a,D:pod-b"}
DISABLED = "quota disabled on this hub"


class TwoHubs:
    """DUB with accounts A and B (credits are charged) and SJC with C and D (quota disabled), two fake hubs on one
    fake network and one clock: a call goes to the hub whose host it names."""

    def __init__(self):
        self.clock = FakeClock("2026-10-10T09:27:00Z")
        self.dub = FakeHub(self.clock)
        self.sjc = FakeHub(self.clock)
        self.dub.add_account("A", user="acct-a", token=TOKEN_A, balance=59960)
        self.dub.add_account("B", user="acct-b", token=TOKEN_B, balance=59628)
        self.sjc.add_account("C", user="acct-c", token=TOKEN_C, quota_enabled=False)
        self.sjc.add_account("D", user="acct-d", token=TOKEN_D, quota_enabled=False)
        self.transport = network({"dub.test": self.dub, "sjc.test": self.sjc})

    def rows(self, verb, node=None):
        return [row for row in ledger.read(verb=verb) if node is None or row.get("node") == node]

    def only_row(self, verb, node=None):
        rows = self.rows(verb, node)
        assert len(rows) == 1, rows
        return rows[0]


@pytest.fixture
def two_hubs(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LAB_ACTOR", "tester")
    for name in ("LAB_TIER", "LAB_TASK"):
        monkeypatch.delenv(name, raising=False)
    return TwoHubs()


def sjc_cli(world, *argv, environ=None):
    """The command line against both fake hubs; the keys come from environ alone, never from a .env file."""
    return ctl_module.run(list(argv), environ=BOTH_HUBS_ENV if environ is None else environ, env_files=[],
                          transport=world.transport, sleep=world.clock.sleep, clock=world.clock.time)


def test_status_of_c_asks_the_sjc_hub_with_the_token_of_c_and_nothing_else(two_hubs):
    answer, code = sjc_cli(two_hubs, "status", "--account", "C")

    assert code == 0
    assert (answer["command"], answer["outcome"], answer["hub"]) == ("status", "ok", SJC)
    assert answer["accounts"] == {"C": {"node": "pod-a", "reachable": True, "running": False, "balance": 0,
                                        "unlimited": True, "quota_enabled": False, "rate": 3}}
    assert two_hubs.dub.calls == []
    assert {call.host for call in two_hubs.sjc.calls} == {"sjc.test"}
    assert {call.headers["authorization"] for call in two_hubs.sjc.calls} == {f"token {TOKEN_C}"}
    assert {call.account for call in two_hubs.sjc.calls} == {"C"}
    assert two_hubs.only_row("pod status")["node"] == "pod-a"


def test_quota_of_c_says_the_quota_is_disabled_instead_of_showing_an_empty_balance(two_hubs):
    answer, code = sjc_cli(two_hubs, "quota", "--account", "C")

    assert code == 0
    assert answer["accounts"]["C"] == {"node": "pod-a", "reachable": True, "balance": 0, "unlimited": True,
                                       "enabled": False, "rate": 3, "rates": {"cpu": 1, "strix-halo": 3, "r9700": 4}, "note": DISABLED}
    row = two_hubs.only_row("pod quota")
    assert (row["node"], row["outcome"], row["metrics"]) == ("pod-a", "ok", {})


def test_launch_on_a_hub_without_quota_goes_ahead_where_the_same_zero_balance_is_refused_on_a_metered_one(two_hubs):
    two_hubs.dub.set_balance("A", 0)
    refused, refused_code = sjc_cli(two_hubs, "launch", "--account", "A", "--minutes", "30")
    assert (refused_code, refused["outcome"]) == (1, "refused_quota")
    assert two_hubs.dub.calls_matching("POST", "/server") == []

    answer, code = sjc_cli(two_hubs, "launch", "--account", "C", "--minutes", "30")

    assert (code, answer["ok"], answer["outcome"]) == (0, True, "running")
    assert answer["note"] == DISABLED
    assert "balance" not in answer and answer["cost_credits"] is None
    assert [(call.host, call.path, call.json) for call in two_hubs.sjc.calls_matching("POST", "/server")] == [
        ("sjc.test", "/hub/api/users/acct-c/server", {"runtime_minutes": 30, "resource_type": "gpu", "gpu_selection": "strix-halo"})]
    pod = [row for row in two_hubs.rows("pod launch") if row["conditions"]["hub"] == "sjc.test"][0]
    assert (pod["outcome"], pod["metrics"]) == ("running", {"pod.start_s": 15.0})


def test_a_dry_run_on_a_hub_without_quota_plans_the_launch_says_so_and_sends_nothing(two_hubs):
    answer, code = sjc_cli(two_hubs, "launch", "--account", "D", "--minutes", "600", "--dry-run")

    assert (code, answer["outcome"], answer["note"]) == (0, "dry_run", DISABLED)
    assert answer["plan"]["path"] == "/hub/api/users/acct-d/server"
    assert two_hubs.sjc.calls_matching("POST", "/server") == []


def test_a_pod_on_d_lives_its_whole_life_through_the_command_line_with_no_credit_figures(two_hubs):
    answers = {}
    for argv in (["launch", "--account", "D", "--minutes", "30"], ["wait", "--account", "d", "--timeout", "5"],
                 ["status", "--account", "D"], ["keep-alive", "--account", "D", "--minutes", "1"],
                 ["stop", "--account", "D"], ["stop", "--account", "D"]):
        answer, code = sjc_cli(two_hubs, *argv)
        assert code == 0, (argv, answer)
        answers.setdefault(argv[0], []).append(answer)

    assert answers["launch"][0]["outcome"] == "running"
    assert answers["wait"][0]["outcome"] == "ready"
    entry = answers["status"][0]["accounts"]["D"]
    assert (entry["node"], entry["running"], entry["ready"], entry["quota_enabled"]) == ("pod-b", True, True, False)
    assert not [key for key in entry if key.startswith(("credits_", "balance_")) or key == "exceeds_balance"]
    assert answers["keep-alive"][0]["outcome"] == "completed"
    assert answers["keep-alive"][0]["accounts"]["D"]["pings"] >= 1
    first_stop, second_stop = answers["stop"]
    assert (first_stop["outcome"], first_stop["state"]) == ("ok", "stopped")
    assert not {"credits_charged", "balance_before", "balance_after"} & set(first_stop)
    assert second_stop["outcome"] == "already_stopped"
    assert two_hubs.dub.calls == []
    assert {call.headers["authorization"] for call in two_hubs.sjc.calls} == {f"token {TOKEN_D}"}
    assert all("acct-c" not in call.path for call in two_hubs.sjc.calls)


def test_run_script_takes_c_and_asks_the_sjc_hub_whether_the_pod_is_there(two_hubs, tmp_path):
    script = tmp_path / "job.sh"
    script.write_text("echo hello\n", encoding="utf-8")

    answer, code = sjc_cli(two_hubs, "run-script", "--account", "C", str(script))

    assert (code, answer["outcome"], answer["node"]) == (1, "not_running", "pod-a")
    assert [call.path for call in two_hubs.sjc.calls] == ["/hub/api/users/acct-c"]
    assert two_hubs.dub.calls == []


@pytest.mark.parametrize("letter", ["C", "D", "c", "d"])
@pytest.mark.parametrize("command", ["status", "quota", "launch --minutes 5", "stop", "wait", "keep-alive", "run-script x.sh"])
def test_every_command_takes_the_account_letters_c_and_d_in_either_case(command, letter):
    name, _, rest = command.partition(" ")

    args = ctl_module.build_parser().parse_args(shlex.split(f"{name} --account {letter} {rest}"))

    assert args.account == letter.upper()


def test_both_still_means_accounts_a_and_b_even_when_c_and_d_have_keys(two_hubs):
    status, _ = sjc_cli(two_hubs, "status")
    quota, _ = sjc_cli(two_hubs, "quota", "--account", "both")

    assert list(status["accounts"]) == ["A", "B"] and status["hub"] == DUB
    assert list(quota["accounts"]) == ["A", "B"]
    assert two_hubs.sjc.calls == []


def test_each_account_goes_to_its_own_address_when_accounts_of_two_hubs_are_asked_together(two_hubs):
    accounts = load_accounts(["A", "C"], environ=BOTH_HUBS_ENV, env_files=[])

    with Controller(accounts, transport=two_hubs.transport, sleep=two_hubs.clock.sleep, clock=two_hubs.clock.time,
                    environ=BOTH_HUBS_ENV) as controller:
        answer = controller.status()

    assert answer["ok"] is True
    assert "hub" not in answer and answer["hubs"] == {"A": DUB, "C": SJC}
    assert {call.account for call in two_hubs.dub.calls} == {"A"}
    assert {call.account for call in two_hubs.sjc.calls} == {"C"}


def test_an_account_without_its_own_address_uses_lab_hub_url_and_then_the_dub_default(two_hubs):
    own_url_missing = {key: value for key, value in BOTH_HUBS_ENV.items() if key != "AUP_HUB_URL_D"}
    answer, code = sjc_cli(two_hubs, "status", "--account", "D", environ={**own_url_missing, "LAB_HUB_URL": SJC})
    assert (code, answer["hub"]) == (0, SJC)
    assert {call.host for call in two_hubs.sjc.calls} == {"sjc.test"}

    bare = {key: value for key, value in own_url_missing.items() if key != "LAB_HUB_URL"}
    answer, code = sjc_cli(two_hubs, "status", "--account", "D", environ=bare)
    assert (code, answer["hub"], answer["accounts"]["D"]["reachable"]) == (1, "https://dub.aupcloud.io", False)


def test_an_account_without_keys_is_named_by_its_missing_keys_and_nothing_is_sent(two_hubs):
    answer, code = sjc_cli(two_hubs, "launch", "--account", "c", "--minutes", "5",
                        environ={**ENV, "LAB_POD_ROLES": "A:pod-a,B:pod-b,C:pod-a"})

    assert code == 2
    assert (answer["ok"], answer["outcome"]) == (False, "no_credentials")
    assert answer["missing"] == ["AUP_HUB_USER_C", "AUP_HUB_TOKEN_C"]
    assert two_hubs.dub.calls == [] and two_hubs.sjc.calls == []


def test_nothing_of_account_c_is_printed_or_written_anywhere_even_when_its_hub_fails(two_hubs, capsys):
    answers = [sjc_cli(two_hubs, "status", "--account", "C")[0], sjc_cli(two_hubs, "quota", "--account", "C")[0],
               sjc_cli(two_hubs, "launch", "--account", "C", "--minutes", "5")[0]]
    two_hubs.sjc.queue_dropped("GET", "/hub/api/users/acct-c", times=10)
    failed, code = sjc_cli(two_hubs, "status", "--account", "C")
    answers.append(failed)

    assert code == 1 and failed["accounts"]["C"]["reachable"] is False
    captured = capsys.readouterr()
    everything = json.dumps(answers) + captured.out + captured.err + ledger.ledger_path().read_text(encoding="utf-8")
    assert TOKEN_C not in everything and "acct-c" not in everything
    assert "<account C>" in failed["accounts"]["C"]["error"]


# --- what a sitting may run -------------------------------------------------------------------

COMMANDS = ["status", "quota --transactions", "launch --account B --minutes 600", "stop --account B",
            "wait --account B", "keep-alive --minutes 90", "run-script --account B lab/pod_b_trainer.sh smoke"]


def spellings(command):
    """The ways to type a command, taken from the module's real name and file."""
    name, script = ctl_module.__name__, f"lab/{Path(ctl_module.__file__).name}"
    return [f"python -m {name} {command}", f"python {script} {command}", f".venv/Scripts/python -m {name} {command}"]


@pytest.mark.parametrize("tier", ["fable", "opus", "sonnet"])
@pytest.mark.parametrize("command", COMMANDS)
def test_no_sitting_may_run_a_controller_command_in_any_spelling(tier, command):
    ctl_module.build_parser().parse_args(shlex.split(command))          # it is a command the controller takes
    for line in spellings(command):
        assert allowed.allows(tier, "Bash", line) is False, line


@pytest.mark.parametrize("command", ["launch --account B --minutes 600", "stop --account B"])
def test_launching_and_stopping_are_denied_by_name_in_every_spelling(command):
    ctl_module.build_parser().parse_args(shlex.split(command))
    for line in spellings(command):
        assert allowed.denies("Bash", line) is True, line


# --- the command line -----------------------------------------------------------------------

def cli(world, *argv, environ=None, env_files=()):
    return ctl_module.run(list(argv), environ=ENV if environ is None else environ, env_files=list(env_files),
                          transport=world.hub.transport, sleep=world.clock.sleep, clock=world.clock.time)


def test_status_through_the_command_line_prints_no_secret_anywhere(world, capsys):
    world.hub.start_server("A")

    answer, code = cli(world, "status")

    assert code == 0
    assert answer["accounts"]["A"]["running"] is True
    captured = capsys.readouterr()
    everything = json.dumps(answer) + captured.out + captured.err + ledger.ledger_path().read_text(encoding="utf-8")
    assert TOKEN_A not in everything and TOKEN_B not in everything
    assert "acct-a" not in everything and "acct-b" not in everything


def test_every_call_the_command_line_makes_carries_the_token_of_the_account_it_is_about(world):
    cli(world, "status")

    with_a = [call for call in world.hub.calls if call.headers["authorization"] == f"token {TOKEN_A}"]
    with_b = [call for call in world.hub.calls if call.headers["authorization"] == f"token {TOKEN_B}"]
    assert with_a and with_b
    assert len(with_a) + len(with_b) == len(world.hub.calls)
    assert all("acct-b" not in call.path for call in with_a)
    assert all("acct-a" not in call.path for call in with_b)
    assert {call.account for call in with_a} == {"A"}
    assert {call.account for call in with_b} == {"B"}


def test_the_command_line_launches_and_stops_with_its_arguments(world):
    answer, code = cli(world, "launch", "--account", "b", "--minutes", "90", "--reason", "rehearsal")
    assert (code, answer["outcome"]) == (0, "running")
    assert world.hub.calls_matching("POST", "/server")[0].json["runtime_minutes"] == 90
    assert world.only_row("pod launch")["note"] == "rehearsal"

    answer, code = cli(world, "stop", "--account", "B")
    assert (code, answer["outcome"]) == (0, "ok")


def test_the_command_line_exits_1_for_an_answer_that_is_not_ok(world):
    world.hub.set_balance("A", 10)
    answer, code = cli(world, "launch", "--account", "A", "--minutes", "600")
    assert (code, answer["outcome"]) == (1, "refused_quota")


def test_missing_keys_are_named_and_nothing_is_sent(world):
    answer, code = cli(world, "status", environ={"AUP_HUB_USER_A": "acct-a"})

    assert code == 2
    assert (answer["ok"], answer["outcome"]) == (False, "no_credentials")
    assert answer["missing"] == ["AUP_HUB_TOKEN_A", "AUP_HUB_USER_B", "AUP_HUB_TOKEN_B"]
    assert world.hub.calls == []


def test_one_account_is_enough_when_only_that_account_is_asked_for(world):
    only_b = {key: value for key, value in ENV.items() if key.endswith("_B")}

    answer, code = cli(world, "wait", "--account", "B", "--timeout", "5", environ=only_b)

    assert (code, answer["outcome"]) == (1, "not_running")


def test_the_keys_can_come_from_a_dotenv_file(world, tmp_path):
    dotenv = tmp_path / "keys.env"
    dotenv.write_text("\n".join(f"{key}={value}" for key, value in ENV.items()) + "\n", encoding="utf-8")

    answer, code = cli(world, "quota", environ={}, env_files=[dotenv])

    assert code == 0
    assert answer["accounts"]["B"]["balance"] == 59628


def test_the_hub_address_comes_from_lab_hub_url(world):
    cli(world, "status", environ={**ENV, "LAB_HUB_URL": "https://hub.example"})
    assert {call.host for call in world.hub.calls} == {"hub.example"}


def test_while_stop_exists_nothing_that_changes_a_pod_runs_but_reading_still_works(world):
    world.state.mkdir(parents=True)
    (world.state / "STOP").write_text("pause the machine\n", encoding="utf-8")
    world.hub.start_server("A")

    for argv in (["launch", "--account", "B", "--minutes", "30"], ["stop", "--account", "A"],
                 ["keep-alive", "--account", "A", "--minutes", "5"], ["run-script", "--account", "A", "x.sh"]):
        answer, code = cli(world, *argv)
        assert (code, answer["ok"], answer["outcome"], answer["reason"]) == (1, False, "stopped", "pause the machine")
    assert world.hub.calls == []
    assert stop.stopped()

    for argv in (["status"], ["quota"], ["wait", "--account", "A", "--timeout", "5"]):
        _, code = cli(world, *argv)
        assert code == 0
    assert {row["outcome"] for row in world.rows("controller launch")} == {"stopped"}


def test_bad_arguments_exit_2(world, capsys):
    for argv in (["launch"], ["launch", "--account", "AB", "--minutes", "5"], ["launch", "--account", "A", "--minutes", "x"],
                 ["status", "--account", "7"], ["status", "--account", "C,D"], ["quota", "--account", ""],
                 ["keep-alive", "--interval", "5"], ["nothing"], []):
        with pytest.raises(SystemExit) as caught:
            cli(world, *argv)
        assert caught.value.code == 2
    assert world.hub.calls == []


def test_main_prints_exactly_one_json_line(world, capsys, monkeypatch):
    monkeypatch.setattr(ctl_module, "run", lambda argv: ({"ok": True, "outcome": "ok", "command": "status"}, 0))

    code = ctl_module.main(["status"])

    assert code == 0
    assert capsys.readouterr().out.splitlines() == ['{"ok": true, "outcome": "ok", "command": "status"}']


def test_main_turns_a_keyboard_interrupt_into_an_interrupted_answer(capsys, monkeypatch):
    def interrupted(argv):
        raise KeyboardInterrupt

    monkeypatch.setattr(ctl_module, "run", interrupted)

    assert ctl_module.main(["keep-alive"]) == 1
    assert json.loads(capsys.readouterr().out)["outcome"] == "interrupted"


@pytest.mark.parametrize("argv", [[sys.executable, "-m", "lab.controller", "--help"],
                                  [sys.executable, str(REPO / "lab" / "controller.py"), "--help"]])
def test_the_controller_runs_as_a_module_and_as_a_script_from_any_folder(argv, tmp_path):
    done = subprocess.run(argv, cwd=REPO if argv[1] == "-m" else tmp_path, capture_output=True, text=True, timeout=60,
                          env={**os.environ, "PYTHONPATH": str(REPO) if argv[1] == "-m" else ""})
    assert done.returncode == 0, done.stderr
    for command in ("status", "launch", "stop", "wait", "keep-alive", "quota", "run-script"):
        assert command in done.stdout


# --- rows are named by the pod role an account fills (LAB_POD_ROLES), R31 -----------------------

@pytest.mark.parametrize("letter, host, node", [("A", "dub.test", "pod-a"), ("B", "dub.test", "pod-b"),
                                                ("C", "sjc.test", "pod-a"), ("D", "sjc.test", "pod-b")])
def test_every_pod_row_of_accounts_a_to_d_is_named_by_its_role_and_records_the_hub_and_the_letter(
        two_hubs, letter, host, node):
    for argv in (("status",), ("quota",), ("launch", "--minutes", "30"), ("stop",)):
        answer, code = sjc_cli(two_hubs, argv[0], "--account", letter, *argv[1:])
        assert code == 0, answer
        assert node in json.dumps(answer) and "pod-c" not in json.dumps(answer) and "pod-d" not in json.dumps(answer)

    rows = [row for verb in ("pod status", "pod quota", "pod launch", "pod stop") for row in two_hubs.rows(verb)]
    assert len(rows) == 4
    assert {row["node"] for row in rows} == {node}
    assert all(row["conditions"]["hub"] == host and row["conditions"]["account"] == letter for row in rows)
    assert {row["refs"]["account"] for row in rows} == {letter}
    assert all(row.get("node") not in ("pod-c", "pod-d") for row in ledger.read())


def test_the_setting_is_read_from_the_workspace_dotenv_when_the_environment_has_none(two_hubs, tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text("LAB_POD_ROLES=C:pod-b,D:pod-a\n", encoding="utf-8")
    env = {key: value for key, value in BOTH_HUBS_ENV.items() if key != "LAB_POD_ROLES"}

    answer, code = ctl_module.run(["status", "--account", "C"], environ=env, env_files=[dotenv],
                                  transport=two_hubs.transport, sleep=two_hubs.clock.sleep, clock=two_hubs.clock.time)

    assert code == 0 and answer["accounts"]["C"]["node"] == "pod-b"
    assert two_hubs.only_row("pod status")["node"] == "pod-b"


@pytest.mark.parametrize("letter", ["C", "D"])
def test_an_account_with_no_role_in_the_setting_is_refused_with_a_clear_message_and_nothing_is_sent(two_hubs, letter):
    env = {key: value for key, value in BOTH_HUBS_ENV.items() if key != "LAB_POD_ROLES"}    # the default A:pod-a,B:pod-b

    answer, code = sjc_cli(two_hubs, "launch", "--account", letter, "--minutes", "5", environ=env)

    assert (code, answer["ok"], answer["outcome"]) == (2, False, "no_pod_role")
    assert f"account {letter} fills no pod role" in answer["error"] and "LAB_POD_ROLES" in answer["error"]
    assert "pod-" + letter.lower() not in answer["error"].replace("pod-a", "").replace("pod-b", "")
    assert two_hubs.dub.calls == [] and two_hubs.sjc.calls == []
    assert ledger.read() == []


@pytest.mark.parametrize("setting", ["C:pod-c,D:pod-d", "C-pod-a", "C:pod-a,C:pod-b", "CD:pod-a"])
def test_a_setting_that_names_a_machine_other_than_the_two_roles_is_refused(two_hubs, setting):
    env = {**BOTH_HUBS_ENV, "LAB_POD_ROLES": setting}

    answer, code = sjc_cli(two_hubs, "status", "--account", "C", environ=env)

    assert (code, answer["outcome"]) == (2, "no_pod_role")
    assert "LAB_POD_ROLES" in answer["error"]
    assert two_hubs.sjc.calls == [] and ledger.read() == []


# --- O51 deferred findings ------------------------------------------------------------------

def test_the_readme_smoke_command_uploads_smoke_lora_py_beside_the_script():
    # pods-run-script-pod-scripts-report-ok-on-failure: run-script uploads only the script unless --with names the
    # sibling, so the documented command for the smoke step must carry --with lab/smoke_lora.py.
    lines = (REPO / "lab" / "README.md").read_text(encoding="utf-8").splitlines()
    commands = [line for line in lines if "run-script" in line and "lab/pod_b_trainer.sh smoke" in line]
    assert commands
    assert all("--with lab/smoke_lora.py" in line for line in commands), commands


def test_the_ledger_verb_names_are_the_ones_alerts_reads():
    # modul-dup-ledger-verb-names: one definition, in alerts, restated nowhere.
    assert ctl_module.BALANCE is alerts.BALANCE_METRIC
    assert ctl_module.POD_LAUNCH is alerts.POD_LAUNCH
    assert ctl_module.POD_STOP is alerts.POD_STOP


def test_keep_alive_interval_leaves_room_for_one_failed_ping_inside_the_culler_hour(world, capsys):
    # pods-keep-alive-max-interval-no-margin: _cycle does not retry a ping, so at 1800 s one failure is a gap of
    # 3600 s, the culler's idle threshold. The longest interval is 1200 s.
    world.hub.start_server("A")
    with pytest.raises(ValueError):
        world.controller.keep_alive(["A"], interval_s=1201, minutes=10)
    assert world.hub.calls == []
    with pytest.raises(SystemExit) as caught:
        ctl_module.run(["keep-alive", "--interval", "1500"], environ=ENV, env_files=[], transport=world.hub.transport,
                       sleep=world.clock.sleep, clock=world.clock.time)
    assert caught.value.code == 2
    assert "1200" in capsys.readouterr().err
    world.controller.keep_alive(["A"], interval_s=1200, minutes=10)


def test_an_account_name_that_the_address_quotes_is_scrubbed_in_both_forms(world):
    # pods-scrub-misses-url-quoted-account-name: the transport error names the address with the user URL-quoted
    # (amd%20user%401), and scrub replaced only the raw name.
    user = "amd user@1"
    account = world.hub.add_account("A", user=user, token=TOKEN_A)

    def dropped(request):
        raise httpx.ConnectError(f"connection to {request.url} was reset")

    controller = Controller({"A": account}, base_url="https://hub.test", transport=httpx.MockTransport(dropped),
                            sleep=world.clock.sleep, clock=world.clock.time)

    answer = controller.status(["A"])

    shown = json.dumps(answer) + ledger.ledger_path().read_text(encoding="utf-8")
    assert "amd%20user%401" not in shown and user not in shown
    assert "<account A>" in answer["accounts"]["A"]["error"]
    assert "<account A>" in ledger.ledger_path().read_text(encoding="utf-8")


def test_a_keep_alive_cycle_between_a_stops_delete_and_its_row_raises_no_stopped_alert(world):
    # xcall-keepalive-lost-row-races-concurrent-stop: a server with pending == "stop" is an intended stop, not a
    # lost pod; the keep-alive row must not be a "lost" row that alerts replays before the stop's own row.
    world.hub.start_server("A")
    world.hub.stop_s = 10_000.0
    world.hub.schedule(300, lambda: world.controller._hub("A").delete(world.controller._hub("A").user_url("/server")))

    answer = world.controller.keep_alive(["A"], minutes=30)
    world.controller.stop("A", wait=False)          # the concurrent stop writes its row after the cycle saw it

    assert world.hub.calls_matching("DELETE", "/server")
    assert [row["outcome"] for row in world.rows("pod keep-alive")] == ["alive", "stopping"]
    assert answer["accounts"]["A"]["state"] == "stopping"
    assert answer["outcome"] == "completed"
    assert [alert.rule for alert in alerts_for(world)] == []
