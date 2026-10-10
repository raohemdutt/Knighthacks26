"""Tests for the GPU queue verbs (python -m lab slot) and the Notion client under them.

Nothing here reaches the real Notion. The wire is replaced by FakeNotion (lab/machine/slots_fake.py), an
in-memory stand-in that answers in the shapes api.notion.com documents and refuses what it refuses, so the
real NotionClient, the real queue rules and the real verbs run. Every test gets its own state dir and sees no
.env file and no NOTION_TOKEN, so none of them can write the real ledger or call the real queue.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from lab import __main__ as lab_cli
from lab.machine import allowed, ledger, slots, slots_fake, slots_notion, stop
from lab.machine.slots_fake import FakeNotion
from lab.machine.slots_notion import NotionClient, NotionError

REPO_ROOT = Path(__file__).resolve().parents[1]
UTC = dt.timezone.utc
T0 = dt.datetime(2026, 10, 10, 14, 0, 0, tzinfo=UTC)
TOKEN = slots_fake.TOKEN
B = "B exclusive"
A = "A teacher shared service"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """A private state dir, no .env files, no Notion settings, loopback exempt from proxies."""
    state = tmp_path / "state"
    monkeypatch.setenv("LAB_STATE_DIR", str(state))
    monkeypatch.setenv("LAB_ENV_FILES", "")
    for name in ("NOTION_TOKEN", "LAB_QUEUE_DATA_SOURCE", "LAB_NOTION_URL", "LAB_ACTOR", "LAB_TIER", "LAB_TASK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    return state


class World:
    """A fake Notion, a clock the test moves, and the real client and verbs wired to them."""

    def __init__(self, *, page_limit: int = 100) -> None:
        self.now = T0
        self.sleeps: list[float] = []
        self.fake = FakeNotion(clock=lambda: self.now, page_limit=page_limit)
        self.client = NotionClient(TOKEN, transport=self.fake.transport, sleep=self.sleeps.append)

    def advance(self, **delta: float) -> None:
        self.now += dt.timedelta(**delta)

    def ago(self, **delta: float) -> dt.datetime:
        return self.now - dt.timedelta(**delta)

    def call(self, *argv: str) -> tuple[dict, int]:
        return slots.run(list(argv), client=self.client, now=lambda: self.now)

    def seed(self, request: str, **fields) -> str:
        fields.setdefault("task_id", request.split(":")[0])
        fields.setdefault("requested", self.now)
        return self.fake.add_row(request, **fields)

    def running(self, request: str, *, started_ago: float = 5, **fields) -> str:
        return self.seed(request, status="Running", started=self.ago(minutes=started_ago), **fields)

    def stored(self, row: str) -> dict:
        return self.fake.values(row)

    def queue(self) -> slots.SlotQueue:
        return slots.SlotQueue(self.client, now=lambda: self.now)


@pytest.fixture
def world() -> World:
    return World()


def ledger_rows() -> list[dict]:
    return ledger.read()


def one_json_line(text: str) -> dict:
    lines = text.splitlines()
    assert len(lines) == 1, f"expected exactly one stdout line, got {lines!r}"
    return json.loads(lines[0])


def make_slot(**overrides) -> slots.Slot:
    base = dict(id="a" * 32, url="https://www.notion.so/x", request="L3: x", task_id="L3", gpu=B, minutes=15.0,
                status="Queued", agent="Ilan", requested=T0, created=T0, started=None, ended=None, extended=False,
                progress_note="", complaint="", result="")
    base.update(overrides)
    return slots.Slot(**base)


# --- the Notion client ------------------------------------------------------------------

def test_every_request_carries_the_bearer_token_the_api_version_and_a_json_body():
    fake = FakeNotion()
    seen = []

    def spy(method, url, headers, body, timeout):
        seen.append((method, url, dict(headers), body))
        return fake.transport(method, url, headers, body, timeout)

    client = NotionClient(TOKEN, transport=spy)
    created = client.create_page(slots_fake.DATA_SOURCE_ID, slots.rest_properties({"Request": "L3: x", "Status": "Queued"}))
    client.query_data_source(slots_fake.DATA_SOURCE_ID)
    client.retrieve_page(created["id"])
    client.update_page(created["id"], slots.rest_properties({"Result": "r"}))
    assert [(m, u.removeprefix("https://api.notion.com")) for m, u, _, _ in seen] == [
        ("POST", "/v1/pages"),
        ("POST", f"/v1/data_sources/{slots_fake.DATA_SOURCE_ID}/query"),
        ("GET", f"/v1/pages/{created['id']}"),
        ("PATCH", f"/v1/pages/{created['id']}"),
    ]
    for method, _, headers, body in seen:
        assert headers["Authorization"] == f"Bearer {TOKEN}"
        assert headers["Notion-Version"] == "2025-09-03"
        assert ("Content-Type" in headers) == (body is not None)
    assert json.loads(seen[0][3])["parent"] == {"type": "data_source_id", "data_source_id": slots_fake.DATA_SOURCE_ID}


def test_a_query_follows_the_cursor_to_the_end_of_the_list():
    world = World(page_limit=2)
    ids = [world.seed(f"L{n}: job", minutes=15) for n in range(1, 6)]
    rows = world.client.query_data_source(slots_fake.DATA_SOURCE_ID)
    assert [slots_notion.page_id_of(row["id"]) for row in rows] == ids
    assert [r for r in world.fake.requests if r[0] == "POST"][-1][2]["start_cursor"] == "cursor-4"
    assert len([r for r in world.fake.requests if r[0] == "POST"]) == 3


def test_a_429_is_retried_after_the_wait_notion_asks_for(world):
    row = world.seed("L3: job")
    world.fake.fail_next(2, status=429, code="rate_limited", retry_after="2")
    page = world.client.retrieve_page(row)
    assert slots_notion.page_id_of(page["id"]) == row
    assert world.sleeps == [2.0, 2.0]


def test_without_a_retry_after_the_wait_doubles(world):
    row = world.seed("L3: job")
    world.fake.fail_next(3, status=503, code="service_unavailable")
    world.client.retrieve_page(row)
    assert world.sleeps == [1.0, 2.0, 4.0]


def test_retries_stop_after_max_retries_and_the_error_says_what_happened(world):
    row = world.seed("L3: job")
    world.fake.fail_next(10, status=503, code="service_unavailable")
    with pytest.raises(NotionError) as caught:
        world.client.retrieve_page(row)
    assert caught.value.kind == "unavailable" and caught.value.status == 503 and caught.value.code == "service_unavailable"
    assert len(world.fake.requests) == slots_notion.MAX_RETRIES + 1
    assert len(world.sleeps) == slots_notion.MAX_RETRIES


def test_a_wait_longer_than_the_cap_is_not_waited_out(world):
    row = world.seed("L3: job")
    world.fake.fail_next(1, status=429, code="rate_limited", retry_after="60")
    with pytest.raises(NotionError) as caught:
        world.client.retrieve_page(row)
    assert caught.value.kind == "rate_limited" and caught.value.retry_after == 60.0
    assert world.sleeps == []
    assert len(world.fake.requests) == 1


def test_a_refused_request_is_not_retried_and_carries_notions_reason(world):
    row = world.seed("L3: job")
    with pytest.raises(NotionError) as caught:
        world.client.update_page(row, {"Nope": slots_notion.checkbox(True)})
    assert caught.value.kind == "validation" and caught.value.status == 400 and caught.value.code == "validation_error"
    assert "Nope is not a property that exists" in str(caught.value)
    assert world.sleeps == [] and len(world.fake.requests) == 1


def test_a_wrong_token_and_a_missing_page_are_told_apart(world):
    wrong = NotionClient("some-other-token-0123456789", transport=world.fake.transport)
    with pytest.raises(NotionError) as unauthorized:
        wrong.retrieve_page("a" * 32)
    assert unauthorized.value.kind == "unauthorized" and unauthorized.value.status == 401
    with pytest.raises(NotionError) as missing:
        world.client.retrieve_page("a" * 32)
    assert missing.value.kind == "not_found" and missing.value.status == 404


def test_a_text_longer_than_notions_limit_is_refused_by_notion_not_cut_short(world):
    row = world.seed("L3: job")
    with pytest.raises(NotionError) as caught:
        world.client.update_page(row, slots.rest_properties({"Complaint": "x" * 2001}))
    assert caught.value.kind == "validation"
    assert world.stored(row)["Complaint"] == ""


def test_the_token_is_scrubbed_from_every_error():
    def leaking_transport(method, url, headers, body, timeout):
        raise OSError(f"tls failure while sending {TOKEN} to the host")

    def echoing_transport(method, url, headers, body, timeout):
        answer = {"object": "error", "status": 400, "code": "validation_error", "message": f"bad {TOKEN} here"}
        return 400, json.dumps(answer).encode(), {}

    for transport in (leaking_transport, echoing_transport):
        with pytest.raises(NotionError) as caught:
            NotionClient(TOKEN, transport=transport).retrieve_page("a" * 32)
        assert TOKEN not in str(caught.value)
        assert "***" in str(caught.value)
    assert TOKEN not in repr(NotionClient(TOKEN))


def test_a_transport_failure_is_one_error_and_the_request_is_not_sent_twice():
    calls = []

    def dropped(method, url, headers, body, timeout):
        calls.append(method)
        raise ConnectionResetError("connection reset by peer")

    with pytest.raises(NotionError) as caught:
        NotionClient(TOKEN, transport=dropped).create_page(slots_fake.DATA_SOURCE_ID, {})
    assert caught.value.kind == "transport" and "ConnectionResetError" in str(caught.value)
    assert calls == ["POST"]


def test_an_answer_that_is_not_json_is_a_bad_response():
    def html(method, url, headers, body, timeout):
        return 200, b"<html>maintenance</html>", {}

    with pytest.raises(NotionError) as caught:
        NotionClient(TOKEN, transport=html).retrieve_page("a" * 32)
    assert caught.value.kind == "bad_response"


@pytest.mark.parametrize("text,expected", [
    ("3f596ef5fec581f096aac6afa10b7b12", "3f596ef5fec581f096aac6afa10b7b12"),
    ("3F596EF5-FEC5-81F0-96AA-C6AFA10B7B12", "3f596ef5fec581f096aac6afa10b7b12"),
    ("https://app.notion.com/p/3f596ef5fec581f096aac6afa10b7b12?pvs=204", "3f596ef5fec581f096aac6afa10b7b12"),
    ("https://www.notion.so/ws/GPU-slot-3f596ef5fec581f096aac6afa10b7b12", "3f596ef5fec581f096aac6afa10b7b12"),
    ("https://www.notion.so/3f596ef5fec581f096aac6afa10b7b12?v=c69762a396874e049a4af6819f762615",
     "3f596ef5fec581f096aac6afa10b7b12"),
    ("  3f596ef5-fec5-81f0-96aa-c6afa10b7b12  ", "3f596ef5fec581f096aac6afa10b7b12"),
])
def test_a_page_id_is_found_in_every_form_notion_shows_it(text, expected):
    assert slots_notion.page_id_of(text) == expected


@pytest.mark.parametrize("text", ["", "L3", "not a page", "3f596ef5fec581f096aac6afa10b7b1", "3f596ef5fec581f096aac6afa10b7b123"])
def test_text_without_a_page_id_has_none(text):
    assert slots_notion.page_id_of(text) is None


def test_the_real_transport_talks_to_a_server_over_http():
    fake = FakeNotion()
    with fake.serve() as base:
        client = NotionClient(TOKEN, base_url=base)
        created = client.create_page(slots_fake.DATA_SOURCE_ID, slots.rest_properties({"Request": "L3: x", "Status": "Queued"}))
        rows = client.query_data_source(slots_fake.DATA_SOURCE_ID, filter=slots.open_rows_filter(B))
        assert rows == [] and created["properties"]["Status"]["select"]["name"] == "Queued"
        stored = client.update_page(created["id"], slots.rest_properties({"GPU": B}))
        assert [p["id"] for p in client.query_data_source(slots_fake.DATA_SOURCE_ID, filter=slots.open_rows_filter(B))] \
            == [stored["id"]]
        with pytest.raises(NotionError) as caught:
            NotionClient("wrong-token-0123456789", base_url=base).retrieve_page(created["id"])
        assert caught.value.kind == "unauthorized"


def test_a_redirect_is_not_followed_so_the_token_stays_where_it_was_sent():
    hits = []

    class Target(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(dict(self.headers))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    target = ThreadingHTTPServer(("127.0.0.1", 0), Target)
    threading.Thread(target=target.serve_forever, daemon=True).start()

    class Redirect(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{target.server_address[1]}/stolen")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    redirector = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
    threading.Thread(target=redirector.serve_forever, daemon=True).start()
    try:
        client = NotionClient(TOKEN, base_url=f"http://127.0.0.1:{redirector.server_address[1]}")
        with pytest.raises(NotionError) as caught:
            client.retrieve_page("a" * 32)
        assert caught.value.kind == "bad_response" and caught.value.status == 302
        assert hits == []
    finally:
        for server in (redirector, target):
            server.shutdown()
            server.server_close()


def test_a_server_that_is_not_there_is_a_transport_error():
    probe = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    port = probe.server_address[1]
    probe.server_close()
    with pytest.raises(NotionError) as caught:
        NotionClient(TOKEN, base_url=f"http://127.0.0.1:{port}", timeout=3).retrieve_page("a" * 32)
    assert caught.value.kind == "transport"
    assert TOKEN not in str(caught.value)


# --- the rules ---------------------------------------------------------------------------

def test_a_shorter_request_goes_first_even_when_it_was_asked_for_later():
    long_early = make_slot(id="1" * 32, minutes=60.0, requested=T0)
    short_late = make_slot(id="2" * 32, minutes=15.0, requested=T0 + dt.timedelta(minutes=30))
    assert slots.queue_order([long_early, short_late]) == [short_late, long_early]


def test_equal_minutes_go_to_the_earlier_request():
    first = make_slot(id="2" * 32, requested=T0)
    second = make_slot(id="1" * 32, requested=T0 + dt.timedelta(seconds=1))
    assert slots.queue_order([second, first]) == [first, second]


def test_a_request_with_no_time_of_day_is_placed_by_when_notion_made_the_row():
    early = make_slot(id="2" * 32, requested=None, created=T0)
    late = make_slot(id="1" * 32, requested=None, created=T0 + dt.timedelta(minutes=2))
    assert slots.queue_order([late, early]) == [early, late]


def test_rows_that_tie_on_everything_are_ordered_by_id_so_the_order_is_total():
    a, b = make_slot(id="a" * 32), make_slot(id="b" * 32)
    assert slots.queue_order([b, a]) == [a, b]
    assert slots.ahead_of(b, [a, b]) == [a]
    assert slots.ahead_of(a, [a, b]) == []


def test_only_queued_rows_are_in_the_queue_and_only_those_on_the_same_gpu_are_ahead():
    mine = make_slot(id="c" * 32, minutes=60.0)
    rows = [make_slot(id="1" * 32, status="Running"), make_slot(id="2" * 32, status="Done"),
            make_slot(id="3" * 32, gpu=A), make_slot(id="4" * 32), mine]
    assert [row.id for row in slots.queue_order(rows)] == ["3" * 32, "4" * 32, "c" * 32]
    assert [row.id for row in slots.ahead_of(mine, rows)] == ["4" * 32]


def test_a_slot_ends_at_started_plus_its_minutes_and_fifteen_more_once_extended():
    row = make_slot(status="Running", started=T0, minutes=15.0)
    assert slots.cap_minutes(row) == 15 and slots.ends_at(row) == T0 + dt.timedelta(minutes=15)
    extended = make_slot(status="Running", started=T0, minutes=60.0, extended=True)
    assert slots.cap_minutes(extended) == 75 and slots.ends_at(extended) == T0 + dt.timedelta(minutes=75)
    assert slots.ends_at(make_slot(status="Queued")) is None


def test_a_running_row_is_overrun_only_after_its_cap():
    row = make_slot(status="Running", started=T0, minutes=15.0)
    assert not slots.is_overrun(row, T0 + dt.timedelta(minutes=15))
    assert slots.is_overrun(row, T0 + dt.timedelta(minutes=15, seconds=1))
    assert not slots.is_overrun(make_slot(status="Queued", started=T0), T0 + dt.timedelta(hours=9))
    assert not slots.is_overrun(make_slot(status="Running", started=None), T0 + dt.timedelta(hours=9))


@pytest.mark.parametrize("text,expected", [
    ("2026-10-10T14:00:00.000Z", T0),
    ("2026-10-10T14:00:00+00:00", T0),
    ("2026-10-10T10:00:00.000-04:00", T0),
    ("2026-10-10T16:00:00+02:00", T0),
    ("2026-10-10", None),
    ("", None),
    (None, None),
    ("yesterday afternoon", None),
])
def test_a_notion_date_time_is_read_as_utc_and_a_bare_date_is_not_a_time(text, expected):
    assert slots.parse_moment(text) == expected


def test_a_time_with_no_offset_takes_the_dates_time_zone_else_utc():
    assert slots.parse_moment("2026-10-10T10:00:00", "America/New_York") == T0
    assert slots.parse_moment("2026-10-10T14:00:00") == T0
    assert slots.parse_moment("2026-10-10T14:00:00", "Not/AZone") == T0


def test_a_page_read_back_from_notion_becomes_the_row_that_was_written(world):
    row = world.seed("L3: smoke", task_id="L3", minutes=60, status="Running", agent="Hemdutt", requested=world.ago(minutes=40),
                     started=world.ago(minutes=10), extended=True, progress_note="half way", complaint="long wait",
                     result="r", gpu=A)
    slot = slots.slot_from_page(world.client.retrieve_page(row))
    assert (slot.id, slot.request, slot.task_id, slot.gpu, slot.minutes, slot.status, slot.agent) == (
        row, "L3: smoke", "L3", A, 60.0, "Running", "Hemdutt")
    assert (slot.requested, slot.started, slot.ended) == (world.ago(minutes=40), world.ago(minutes=10), None)
    assert (slot.extended, slot.progress_note, slot.complaint, slot.result) == (True, "half way", "long wait", "r")
    assert slot.created == T0 and slot.data_source == slots_notion.page_id_of(slots_fake.DATA_SOURCE_ID)


def test_the_same_change_is_written_to_notion_and_printed_for_the_mcp_tools_alike(world):
    change = {"Request": "L3: x", "Task ID": "L3", "GPU": B, "Minutes": 60, "Status": "Running", "Agent": "Ilan",
              "Requested": T0, "Started": T0, "Ended": None, "Extended": True, "Progress note": "p", "Result": "r"}
    page = world.client.create_page(slots_fake.DATA_SOURCE_ID, slots.rest_properties(change))
    stored = world.stored(slots_notion.page_id_of(page["id"]))
    assert {name: stored[name] for name in change if name not in ("Requested", "Started", "Ended")} \
        == {name: value for name, value in change.items() if name not in ("Requested", "Started", "Ended")}
    assert stored["Requested"] == "2026-10-10T14:00:00.000+00:00" and stored["Ended"] is None
    assert slots.mcp_properties(change) == {
        "Request": "L3: x", "Task ID": "L3", "GPU": B, "Minutes": 60, "Status": "Running", "Agent": "Ilan",
        "date:Requested:start": "2026-10-10T14:00:00.000Z", "date:Requested:is_datetime": 1,
        "date:Started:start": "2026-10-10T14:00:00.000Z", "date:Started:is_datetime": 1,
        "date:Ended:start": None, "Extended": "__YES__", "Progress note": "p", "Result": "r"}
    assert slots.mcp_properties({"Extended": False}) == {"Extended": "__NO__"}


def test_a_complaint_is_added_after_the_earlier_ones_and_the_oldest_are_dropped_to_fit():
    assert slots.add_complaint("", "first") == "first"
    assert slots.add_complaint("first", "second") == "first | second"
    old = " | ".join(f"entry {n:03d} " + "x" * 90 for n in range(20))
    fitted = slots.add_complaint(old, "newest")
    assert len(fitted) <= slots.MAX_COMPLAINT_CHARS and fitted.endswith(" | newest") and "entry 019" in fitted
    assert "entry 000" not in fitted


def test_the_request_names_the_task_and_what_runs_and_a_long_slot_says_why():
    assert slots.request_title("L3", "smoke LoRA", 15, None) == "L3: smoke LoRA"
    assert slots.request_title("L3", "L3: smoke LoRA", 15, None) == "L3: smoke LoRA"
    assert slots.request_title("L3", "train the adapter", 60, "training run") == "L3: train the adapter (60 min: training run)"


# --- request -----------------------------------------------------------------------------

def test_a_request_adds_one_row_with_the_columns_the_queue_rules_name(world):
    answer, code = world.call("request", "--task", "L3", "--what", "smoke LoRA 200 steps")
    assert code == 0 and answer["ok"] is True and answer["outcome"] == "queued" and answer["verb"] == "slot request"
    [row] = world.fake.row_ids()
    assert answer["row"] == row
    assert world.stored(row) == {
        "Request": "L3: smoke LoRA 200 steps", "Task ID": "L3", "GPU": B, "Minutes": 15, "Status": "Queued",
        "Agent": "Ilan", "Requested": "2026-10-10T14:00:00.000+00:00", "Started": None, "Ended": None,
        "Extended": False, "Progress note": "", "Complaint": "", "Result": ""}
    [(method, path, body)] = world.fake.edits()
    assert (method, path, body["parent"]) == ("POST", "/v1/pages", {"type": "data_source_id", "data_source_id": slots.DATA_SOURCE_ID})
    assert answer["position"] == 1 and answer["may_start"] is True and answer["ahead"] == [] and answer["running"] == []
    assert answer["next"] == f"python -m lab slot start --row {row}"


def test_a_long_slot_needs_a_reason_and_the_reason_is_written_into_the_request(world):
    answer, code = world.call("request", "--task", "l3", "--what", "train the adapter", "--minutes", "60",
                              "--why", "training run")
    assert code == 0
    assert world.stored(answer["row"])["Request"] == "L3: train the adapter (60 min: training run)"
    assert world.stored(answer["row"])["Minutes"] == 60 and world.stored(answer["row"])["Task ID"] == "L3"


@pytest.mark.parametrize("argv", [
    ["request", "--task", "L3", "--what", "x", "--minutes", "60"],
    ["request", "--task", "L3", "--what", "x", "--why", "because"],
    ["request", "--task", "L3", "--what", "x", "--minutes", "30", "--why", "long"],
    ["request", "--task", "ticket", "--what", "x"],
    ["request", "--task", "L3", "--what", "   "],
    ["request", "--task", "L3", "--what", "x" * 201],
    ["request", "--what", "x"],
    ["request", "--task", "L3", "--what", "x", "--gpu", "C"],
    ["start"],
    ["start", "--row", "L3"],
    ["extend", "--row", "a" * 32],
    ["extend", "--row", "a" * 32, "--note", " "],
    ["done", "--row", "a" * 32],
    [],
    ["reserve"],
])
def test_bad_arguments_exit_2_and_touch_nothing(world, argv, capsys):
    with pytest.raises(SystemExit) as caught:
        world.call(*argv)
    assert caught.value.code == 2
    assert world.fake.requests == [] and ledger_rows() == []
    assert "usage" in capsys.readouterr().err.lower()


def test_asking_again_in_the_same_words_returns_the_queued_row_instead_of_adding_a_second(world):
    first, _ = world.call("request", "--task", "L3", "--what", "smoke LoRA")
    world.advance(minutes=1)
    again, code = world.call("request", "--task", "L3", "--what", "smoke LoRA")
    assert code == 0 and again["outcome"] == "already_queued" and again["row"] == first["row"]
    assert len(world.fake.row_ids()) == 1 and len(world.fake.edits()) == 1


def test_a_different_job_of_the_same_task_is_a_second_row(world):
    world.call("request", "--task", "L3", "--what", "smoke LoRA")
    again, _ = world.call("request", "--task", "L3", "--what", "eval the adapter")
    assert again["outcome"] == "queued" and len(world.fake.row_ids()) == 2


def test_a_short_request_made_later_is_first_in_line_and_the_long_one_is_behind_it(world):
    long_row, _ = world.call("request", "--task", "L3", "--what", "train", "--minutes", "60", "--why", "training run")
    assert long_row["position"] == 1 and long_row["may_start"] is True
    world.advance(minutes=3)
    short, _ = world.call("request", "--task", "O4", "--what", "bench")
    assert short["position"] == 1 and short["may_start"] is True and short["ahead"] == []
    status, _ = world.call("status")
    assert [row["row"] for row in status["queued"]] == [short["row"], long_row["row"]]
    assert [row["may_start"] for row in status["queued"]] == [True, False]


def test_a_request_behind_a_running_slot_waits_and_says_what_it_waits_behind(world):
    world.running("L4: eval", started_ago=5)
    answer, code = world.call("request", "--task", "O4", "--what", "bench")
    assert code == 0 and answer["may_start"] is False and answer["position"] == 1
    assert [row["request"] for row in answer["running"]] == ["L4: eval"]
    assert answer["running"][0]["remaining_min"] == 10
    assert "slot status" in answer["next"] and "slot complain" in answer["next"]


def test_a_request_for_the_shared_teacher_pod_is_queued_on_its_own_card_with_a_warning(world):
    world.running("L4: eval", started_ago=5)
    answer, code = world.call("request", "--task", "R2", "--what", "restart the teacher server", "--gpu", "A")
    assert code == 0 and answer["may_start"] is True and "shared teacher" in answer["warning"]
    assert world.stored(answer["row"])["GPU"] == A
    assert ledger_rows()[0]["node"] == "pod-a"


def test_a_request_names_the_agent_it_is_for(world):
    answer, _ = world.call("request", "--task", "L3", "--what", "x", "--agent", "Hemdutt")
    assert world.stored(answer["row"])["Agent"] == "Hemdutt"


# --- start -------------------------------------------------------------------------------

def test_a_row_that_is_first_and_alone_starts_with_status_and_started_set_in_one_update(world):
    row = world.seed("L3: smoke")
    world.advance(minutes=2)
    answer, code = world.call("start", "--row", row)
    assert code == 0 and answer["outcome"] == "started" and answer["row"] == row
    assert answer["slot"]["started"] == "2026-10-10T14:02:00.000Z" and answer["slot"]["ends_at"] == "2026-10-10T14:17:00.000Z"
    assert answer["release"] == f'python -m lab slot done --row {row} --result "<one line with the numbers>"'
    stored = world.stored(row)
    assert stored["Status"] == "Running" and stored["Started"] == "2026-10-10T14:02:00.000+00:00"
    updates = [e for e in world.fake.edits() if e[0] == "PATCH"]
    assert updates[0][2] == {"properties": {"Status": {"select": {"name": "Running"}},
                                            "Started": {"date": {"start": "2026-10-10T14:02:00.000Z"}}}}


def test_a_row_can_start_from_the_url_notion_shows_for_it(world):
    row = world.seed("L3: smoke")
    answer, code = world.call("start", "--row", f"https://app.notion.com/p/{row}?pvs=204")
    assert code == 0 and answer["outcome"] == "started"


def test_nothing_starts_while_another_row_is_running_and_nothing_is_written(world):
    world.running("L4: eval", started_ago=5)
    row = world.seed("L3: smoke")
    answer, code = world.call("start", "--row", row)
    assert code == 1 and answer["ok"] is False and answer["outcome"] == "busy"
    assert [r["request"] for r in answer["running"]] == ["L4: eval"] and "past its cap" not in answer["reason"]
    assert world.fake.edits() == [] and world.stored(row)["Status"] == "Queued"


def test_a_running_row_blocks_even_when_it_looks_stale_and_the_answer_says_so(world):
    world.running("L4: eval", started_ago=120)
    row = world.seed("L3: smoke")
    answer, code = world.call("start", "--row", row)
    assert code == 1 and answer["outcome"] == "busy" and "past its cap" in answer["reason"]
    assert answer["running"][0]["overrun"] is True and answer["running"][0]["remaining_min"] == -105
    assert world.fake.edits() == []


def test_a_running_row_on_the_other_gpu_does_not_block(world):
    world.running("R2: restart", gpu=A)
    row = world.seed("L3: smoke")
    answer, code = world.call("start", "--row", row)
    assert code == 0 and answer["outcome"] == "started"


def test_a_row_does_not_start_while_a_shorter_one_is_queued_ahead_of_it(world):
    long_row = world.seed("L3: train", minutes=60, requested=world.ago(minutes=30))
    short_row = world.seed("O4: bench", minutes=15, requested=world.now)
    refused, code = world.call("start", "--row", long_row)
    assert code == 1 and refused["outcome"] == "not_next" and [r["row"] for r in refused["ahead"]] == [short_row]
    assert world.fake.edits() == [] and world.stored(long_row)["Status"] == "Queued"
    started, code = world.call("start", "--row", short_row)
    assert code == 0 and started["outcome"] == "started"
    assert world.stored(short_row)["Status"] == "Running" and world.stored(long_row)["Status"] == "Queued"


def test_equal_minutes_start_in_the_order_they_were_requested(world):
    early = world.seed("L3: a", requested=world.ago(minutes=5))
    late = world.seed("O4: b", requested=world.ago(minutes=1))
    refused, code = world.call("start", "--row", late)
    assert code == 1 and refused["outcome"] == "not_next" and [r["row"] for r in refused["ahead"]] == [early]
    assert world.call("start", "--row", early)[1] == 0


def test_a_queued_row_on_the_other_gpu_is_not_ahead(world):
    world.seed("R2: restart", gpu=A, minutes=15, requested=world.ago(minutes=9))
    row = world.seed("L3: smoke")
    assert world.call("start", "--row", row)[0]["outcome"] == "started"


def test_a_row_that_is_not_queued_cannot_be_started(world):
    running = world.running("L3: smoke")
    done = world.seed("L4: done", status="Done")
    for row, status in ((running, "Running"), (done, "Done")):
        answer, code = world.call("start", "--row", row)
        assert code == 1 and answer["outcome"] == "not_queued" and status in answer["reason"]
    assert world.fake.edits() == []


def test_another_agents_row_is_never_written(world):
    row = world.seed("L3: smoke", agent="Hemdutt")
    for argv in (["start"], ["extend", "--note", "n"], ["done", "--result", "r"], ["complain"]):
        answer, code = world.call(argv[0], "--row", row, *argv[1:])
        assert code == 1 and answer["outcome"] == "not_your_row"
    assert world.fake.edits() == []
    mine, code = world.call("start", "--row", row, "--agent", "hemdutt")
    assert code == 0 and mine["outcome"] == "started"


def test_a_row_without_an_agent_belongs_to_whoever_asks(world):
    row = world.seed("L3: smoke", agent="")
    assert world.call("start", "--row", row)[0]["outcome"] == "started"


def test_a_page_that_is_not_in_the_queue_is_refused_without_writing(world, monkeypatch):
    row = world.seed("L3: smoke")
    monkeypatch.setenv("LAB_QUEUE_DATA_SOURCE", "11111111-2222-4333-8444-555555555555")
    answer, code = slots.run(["start", "--row", row], client=world.client, now=lambda: world.now)
    assert code == 1 and answer["outcome"] == "not_a_queue_row"
    assert world.fake.edits() == []


def test_a_row_notion_cannot_find_is_row_not_found(world):
    answer, code = world.call("start", "--row", "b" * 32)
    assert code == 1 and answer["outcome"] == "row_not_found" and answer["row"] == "b" * 32


def test_a_row_with_no_gpu_is_in_no_queue(world):
    row = world.seed("L3: smoke", gpu=None)
    assert world.stored(row)["GPU"] is None
    answer, code = world.call("start", "--row", row)
    assert code == 1 and answer["outcome"] == "no_gpu" and world.fake.edits() == []


def test_a_start_that_finds_another_row_running_after_its_own_update_steps_back(world):
    mine = world.seed("L3: train", minutes=60)
    fired = []

    def rival_starts_first(fake, method, path, body):
        """Another agent asks for a shorter slot and gets the card between this row's check and its update."""
        if method == "PATCH" and not fired:
            fired.append(True)
            rival = world.seed("O4: bench", minutes=15)
            started, code = world.call("start", "--row", rival)
            assert code == 0 and started["outcome"] == "started"
            fired.append(rival)

    world.fake.on_request(rival_starts_first)
    answer, code = world.call("start", "--row", mine)
    rival = fired[1]
    assert code == 1 and answer["ok"] is False and answer["outcome"] == "lost_race"
    assert [r["row"] for r in answer["running"]] == [rival]
    assert world.stored(mine)["Status"] == "Queued" and world.stored(mine)["Started"] is None
    assert world.stored(rival)["Status"] == "Running"
    assert [world.stored(row)["Status"] for row in world.fake.row_ids()].count("Running") == 1


def test_a_start_that_cannot_put_itself_back_says_so_and_what_to_do(world):
    mine = world.seed("L3: train", minutes=60)
    my_page = f"/v1/pages/{slots_notion.dashed_id(mine)}"
    updates_of_mine = []

    def rival_then_failure(fake, method, path, body):
        if method != "PATCH" or path != my_page:
            return
        updates_of_mine.append(path)
        if len(updates_of_mine) == 1:
            rival = world.seed("O4: bench", minutes=15)
            assert world.call("start", "--row", rival)[1] == 0
        else:
            fake.fail_next(1, status=500, code="internal_server_error")

    world.fake.on_request(rival_then_failure)
    answer, code = world.call("start", "--row", mine)
    assert code == 1 and answer["outcome"] == "error_lost_race" and "internal_server_error" in answer["revert_error"]
    assert "set its Status to Queued" in answer["reason"]


def test_a_start_whose_second_look_at_the_queue_fails_still_reports_the_card_as_started(world):
    row = world.seed("L3: smoke")
    reads = []

    def second_read_fails(fake, method, path, body):
        if method == "POST" and path.endswith("/query"):
            reads.append(path)
            if len(reads) == 2:
                fake.fail_next(1, status=500, code="internal_server_error")

    world.fake.on_request(second_read_fails)
    answer, code = world.call("start", "--row", row)
    assert code == 0 and answer["ok"] is True and answer["outcome"] == "started"
    assert "could not be read again" in answer["warning"]
    assert world.stored(row)["Status"] == "Running"


# --- extend ------------------------------------------------------------------------------

def test_a_running_slot_is_extended_once_with_a_progress_note(world):
    row = world.running("L3: smoke", started_ago=10)
    answer, code = world.call("extend", "--row", row, "--note", "loss fell from 2.1 to 1.4 in 100 steps")
    assert code == 0 and answer["outcome"] == "extended"
    assert answer["slot"]["cap_min"] == 30 and answer["slot"]["ends_at"] == "2026-10-10T14:20:00.000Z"
    assert answer["slot"]["extended"] is True
    stored = world.stored(row)
    assert stored["Extended"] is True and stored["Progress note"] == "loss fell from 2.1 to 1.4 in 100 steps"
    assert stored["Minutes"] == 15 and stored["Status"] == "Running"


def test_a_second_extension_is_refused_and_changes_nothing(world):
    row = world.running("L3: smoke", started_ago=10, extended=True, progress_note="first")
    answer, code = world.call("extend", "--row", row, "--note", "again")
    assert code == 1 and answer["outcome"] == "already_extended"
    assert world.fake.edits() == [] and world.stored(row)["Progress note"] == "first"


def test_only_a_running_row_can_be_extended(world):
    queued = world.seed("L3: smoke")
    answer, code = world.call("extend", "--row", queued, "--note", "n")
    assert code == 1 and answer["outcome"] == "not_running" and world.fake.edits() == []


def test_an_extension_asked_for_after_the_cap_is_refused_but_at_the_cap_it_is_allowed(world):
    late = world.running("L3: smoke", started_ago=15.5)
    answer, code = world.call("extend", "--row", late, "--note", "n")
    assert code == 1 and answer["outcome"] == "past_cap" and world.stored(late)["Extended"] is False
    other = World()
    exact = other.running("L3: smoke", started_ago=15)
    assert other.call("extend", "--row", exact, "--note", "n")[0]["outcome"] == "extended"


# --- done --------------------------------------------------------------------------------

def test_done_finishes_the_slot_records_the_result_and_releases_the_card(world):
    row = world.running("L3: smoke", started_ago=12)
    answer, code = world.call("done", "--row", row, "--result", "smoke LoRA 200 steps, 14.2 samples/s, peak 9.8 GB")
    assert code == 0 and answer["outcome"] == "done" and answer["card_released"] is True
    assert answer["ran_min"] == 12 and answer["overrun_min"] == 0
    stored = world.stored(row)
    assert (stored["Status"], stored["Ended"], stored["Result"]) == (
        "Done", "2026-10-10T14:00:00.000+00:00", "smoke LoRA 200 steps, 14.2 samples/s, peak 9.8 GB")
    follower = world.seed("O4: bench")
    assert world.call("start", "--row", follower)[0]["outcome"] == "started"


def test_done_after_the_cap_still_records_the_slot_and_says_by_how_much_it_overran(world):
    row = world.running("L3: smoke", started_ago=20)
    answer, code = world.call("done", "--row", row, "--result", "ran long, 3 of 5 runs")
    assert code == 0 and answer["ran_min"] == 20 and answer["overrun_min"] == 5
    assert world.stored(row)["Status"] == "Done"


def test_an_extended_slot_overruns_only_past_the_longer_cap(world):
    row = world.running("L3: smoke", started_ago=20, extended=True, progress_note="p")
    assert world.call("done", "--row", row, "--result", "r")[0]["overrun_min"] == 0


def test_the_result_is_one_line_and_cannot_be_left_out(world):
    row = world.running("L3: smoke")
    answer, _ = world.call("done", "--row", row, "--result", "  tok/s 7.7\n  peak 9 GB \r\n ok  ")
    assert answer["outcome"] == "done" and world.stored(row)["Result"] == "tok/s 7.7 peak 9 GB ok"
    with pytest.raises(SystemExit):
        world.call("done", "--row", row)
    other = World()
    running = other.running("L3: smoke")
    assert other.queue().done(running, None)["outcome"] == "result_required"
    assert other.fake.edits() == []


def test_a_queued_row_is_not_finished_with_done_but_can_be_cancelled(world):
    row = world.seed("L3: smoke")
    answer, code = world.call("done", "--row", row, "--result", "r")
    assert code == 1 and answer["outcome"] == "not_running" and "--cancel" in answer["reason"]
    assert world.fake.edits() == []
    cancelled, code = world.call("done", "--row", row, "--cancel", "--result", "no longer needed")
    assert code == 0 and cancelled["outcome"] == "cancelled" and cancelled["card_released"] is True
    stored = world.stored(row)
    assert (stored["Status"], stored["Result"], stored["Ended"]) == ("Cancelled", "no longer needed", "2026-10-10T14:00:00.000+00:00")
    other = world.seed("O4: bench")
    assert world.call("start", "--row", other)[0]["outcome"] == "started"


def test_a_cancelled_row_needs_no_reason_and_a_running_row_cannot_be_cancelled(world):
    queued = world.seed("L3: smoke")
    assert world.call("done", "--row", queued, "--cancel")[0]["outcome"] == "cancelled"
    assert world.stored(queued)["Result"] == ""
    running = world.running("O4: bench")
    answer, code = world.call("done", "--row", running, "--cancel")
    assert code == 1 and answer["outcome"] == "not_queued" and world.stored(running)["Status"] == "Running"


def test_a_finished_row_cannot_be_finished_again(world):
    row = world.seed("L3: smoke", status="Done", started=world.ago(minutes=9), ended=world.ago(minutes=1), result="r")
    answer, code = world.call("done", "--row", row, "--result", "second")
    assert code == 1 and answer["outcome"] == "not_running" and world.stored(row)["Result"] == "r"


# --- complain ----------------------------------------------------------------------------

def test_a_complaint_before_the_row_has_waited_twice_its_minutes_is_refused(world):
    blocker = world.running("L4: eval", started_ago=5)
    row = world.seed("L3: smoke", requested=world.ago(minutes=29))
    answer, code = world.call("complain", "--row", row)
    assert code == 1 and answer["outcome"] == "too_early" and answer["complain_after_min"] == 30
    assert answer["earliest"] == "2026-10-10T14:01:00.000Z" and world.fake.edits() == []
    assert world.stored(blocker)["Status"] == "Running" and world.stored(row)["Complaint"] == ""


def test_waiting_exactly_twice_the_minutes_is_not_yet_more_than_twice(world):
    world.running("L4: eval", started_ago=5)
    row = world.seed("L3: smoke", requested=world.ago(minutes=30))
    assert world.call("complain", "--row", row)[0]["outcome"] == "too_early"
    world.advance(seconds=1)
    assert world.call("complain", "--row", row)[0]["outcome"] == "complaint_recorded"


def test_a_complaint_after_the_wait_says_how_long_and_what_it_waits_behind(world):
    world.running("L4: eval", started_ago=10)
    row = world.seed("L3: smoke", requested=world.ago(minutes=40))
    answer, code = world.call("complain", "--row", row, "--text", "the eval has run 10 minutes of its 15")
    assert code == 0 and answer["outcome"] == "complaint_recorded"
    text = world.stored(row)["Complaint"]
    assert text.startswith("2026-10-10T14:00Z waited 40 minutes, more than twice the 15 asked for; waiting behind: ")
    assert "Running 'L4: eval' (ends 2026-10-10T14:05:00.000Z)" in text
    assert text.endswith(". the eval has run 10 minutes of its 15")
    assert world.stored(row)["Status"] == "Queued"


def test_a_complaint_is_allowed_at_once_when_the_running_row_is_past_its_cap(world):
    world.running("L4: eval", started_ago=45)
    row = world.seed("L3: smoke", requested=world.ago(minutes=1))
    answer, code = world.call("complain", "--row", row)
    assert code == 0 and "is past its cap" in answer["reason"]
    assert "Running 'L4: eval'" in world.stored(row)["Complaint"]


def test_a_later_complaint_is_added_after_the_earlier_one(world):
    world.running("L4: eval", started_ago=10)
    row = world.seed("L3: smoke", requested=world.ago(minutes=40))
    world.call("complain", "--row", row)
    world.advance(minutes=20)
    world.call("complain", "--row", row, "--text", "still waiting")
    first, second = world.stored(row)["Complaint"].split(" | ")
    assert first.startswith("2026-10-10T14:00Z waited 40 minutes") and second.startswith("2026-10-10T14:20Z waited 60 minutes")
    assert second.endswith(". still waiting")


def test_only_a_waiting_row_complains(world):
    running = world.running("L3: smoke")
    answer, code = world.call("complain", "--row", running)
    assert code == 1 and answer["outcome"] == "not_waiting" and world.fake.edits() == []


# --- status ------------------------------------------------------------------------------

def test_status_lists_the_running_slot_and_the_queue_in_the_order_they_will_start(world):
    running = world.running("L4: eval", started_ago=10)
    late_short = world.seed("O4: bench", minutes=15, requested=world.ago(minutes=2))
    early_short = world.seed("L5: pack", minutes=15, requested=world.ago(minutes=9))
    long_row = world.seed("L3: train", minutes=60, requested=world.ago(minutes=30))
    world.seed("R2: other card", gpu=A)
    answer, code = world.call("status")
    assert code == 0 and answer["outcome"] == "status" and answer["gpu"] == B and answer["card_free"] is False
    assert [r["row"] for r in answer["running"]] == [running]
    assert [r["row"] for r in answer["queued"]] == [early_short, late_short, long_row]
    assert [r["position"] for r in answer["queued"]] == [1, 2, 3]
    assert all(r["may_start"] is False for r in answer["queued"]) and answer["next_row"] == early_short
    assert answer["running"][0]["remaining_min"] == 5 and answer["running"][0]["overrun"] is False
    assert answer["queued"][2]["waited_min"] == 30


def test_with_the_card_free_the_first_in_line_may_start_and_no_other(world):
    first = world.seed("O4: bench", minutes=15, requested=world.ago(minutes=2))
    world.seed("L3: train", minutes=60, requested=world.ago(minutes=30))
    answer, _ = world.call("status")
    assert answer["card_free"] is True and [r["may_start"] for r in answer["queued"]] == [True, False]
    assert answer["queued"][0]["row"] == first


def test_status_flags_a_running_slot_that_is_past_its_cap(world):
    world.running("L4: eval", started_ago=20, extended=False)
    answer, _ = world.call("status")
    assert answer["running"][0]["overrun"] is True and answer["running"][0]["remaining_min"] == -5


def test_status_changes_nothing_and_asks_only_for_the_queue(world):
    world.running("L4: eval")
    world.seed("L3: smoke")
    world.call("status")
    assert world.fake.edits() == []
    assert {(m, p.rsplit("/", 1)[-1]) for m, p, _ in world.fake.requests} == {("POST", "query")}


def test_status_of_the_other_gpu_and_of_one_task(world):
    world.seed("R2: restart", gpu=A, task_id="R2")
    world.seed("L3: smoke", task_id="L3")
    world.seed("L3: eval", task_id="L3", minutes=60)
    world.seed("O4: bench", task_id="O4")
    on_a, _ = world.call("status", "--gpu", "A")
    assert [r["request"] for r in on_a["queued"]] == ["R2: restart"] and on_a["gpu"] == A
    only_l3, _ = world.call("status", "--task", "L3")
    assert [r["request"] for r in only_l3["queued"]] == ["L3: smoke", "L3: eval"]
    assert [r["position"] for r in only_l3["queued"]] == [1, 3]


def test_status_of_one_row_tells_its_place_and_may_be_asked_about_another_agents_row(world):
    world.running("L4: eval", started_ago=1)
    ahead = world.seed("O4: bench", minutes=15, requested=world.ago(minutes=5), agent="Hemdutt")
    mine = world.seed("L3: smoke", minutes=15, requested=world.ago(minutes=1))
    answer, code = world.call("status", "--row", mine)
    assert code == 0 and answer["position"] == 2 and answer["may_start"] is False
    assert [r["row"] for r in answer["ahead"]] == [ahead] and [r["request"] for r in answer["running"]] == ["L4: eval"]
    theirs, code = world.call("status", "--row", ahead)
    assert code == 0 and theirs["slot"]["agent"] == "Hemdutt" and theirs["position"] == 1


def test_status_of_a_finished_row_reports_how_long_it_ran(world):
    row = world.seed("L3: smoke", status="Done", started=world.ago(minutes=12), ended=world.ago(minutes=1), result="r")
    answer, _ = world.call("status", "--row", row)
    assert answer["slot"]["status"] == "Done" and answer["slot"]["ran_min"] == 11 and "position" not in answer


def test_status_reads_every_page_of_a_long_queue():
    world = World(page_limit=2)
    for n in range(1, 8):
        world.seed(f"L{n}: job", requested=world.ago(minutes=10 - n))
    answer, _ = world.call("status")
    assert [r["request"] for r in answer["queued"]] == [f"L{n}: job" for n in range(1, 8)]


# --- Notion trouble ----------------------------------------------------------------------

def test_a_notion_failure_is_an_answer_with_the_kind_and_nothing_half_done(world):
    row = world.seed("L3: smoke")
    world.fake.fail_next(10, status=401, code="unauthorized")
    answer, code = world.call("start", "--row", row)
    assert code == 1 and answer["ok"] is False and answer["outcome"] == "error" and answer["kind"] == "unauthorized"
    assert "rejected NOTION_TOKEN" in answer["reason"]
    assert world.fake.edits() == [] and world.stored(row)["Status"] == "Queued"
    assert ledger_rows()[-1]["outcome"] == "error" and ledger_rows()[-1]["kind"] == "unauthorized"


def test_a_database_the_token_cannot_see_is_explained(world):
    world.fake.fail_next(1, status=404, code="object_not_found")
    answer, code = world.call("status")
    assert code == 1 and answer["outcome"] == "error" and answer["kind"] == "not_found"
    assert "share it with the integration" in answer["reason"]


# --- the ledger --------------------------------------------------------------------------

def test_every_call_writes_one_ledger_row_with_the_verb_the_outcome_the_task_and_the_slot(world, monkeypatch):
    monkeypatch.setenv("LAB_ACTOR", "sitting-7")
    monkeypatch.setenv("LAB_TIER", "Sonnet")
    request, _ = world.call("request", "--task", "L3", "--what", "smoke LoRA")
    row = request["row"]
    world.advance(minutes=1)
    world.call("status", "--row", row)
    world.call("start", "--row", row)
    world.advance(minutes=10)
    world.call("extend", "--row", row, "--note", "half done")
    world.advance(minutes=15)
    world.call("done", "--row", row, "--result", "7.7 tok/s")
    other, _ = world.call("request", "--task", "O4", "--what", "bench")
    world.advance(minutes=1)
    world.call("done", "--row", other["row"], "--cancel")
    rows = ledger_rows()
    assert [(r["verb"], r["outcome"], r["task"]) for r in rows] == [
        ("slot request", "queued", "L3"), ("slot status", "status", "L3"), ("slot start", "started", "L3"),
        ("slot extend", "extended", "L3"), ("slot done", "done", "L3"), ("slot request", "queued", "O4"),
        ("slot done", "cancelled", "O4")]
    assert [r["seq"] for r in rows] == list(range(1, 8))
    for r in rows:
        assert (r["actor"], r["tier"], r["node"], r["metrics"]) == ("sitting-7", "sonnet", "pod-b", {})
    assert [r["refs"]["slot"] for r in rows] == [row] * 5 + [other["row"]] * 2
    assert (rows[0]["minutes"], rows[0]["position"], rows[0]["gpu"]) == (15, 1, B)
    assert rows[4]["ran_min"] == 25 and rows[4]["overrun_min"] == 0


def test_the_task_of_a_row_comes_from_the_call_then_the_row_then_the_sittings_task(world, monkeypatch):
    monkeypatch.setenv("LAB_TASK", "T2")
    row = world.seed("L3: smoke", task_id="L3")
    world.call("status", "--gpu", "A")
    world.call("start", "--row", row)
    world.call("request", "--task", "O4", "--what", "bench")
    assert [r["task"] for r in ledger_rows()] == ["T2", "L3", "O4"]
    slots.run(["status"], client=world.client, now=lambda: world.now, environ={"LAB_TASK": "T9"})
    assert ledger_rows()[-1]["task"] == "T9"


def test_a_refusal_is_a_ledger_row_too(world):
    world.running("L4: eval")
    row = world.seed("L3: smoke")
    world.call("start", "--row", row)
    [entry] = ledger_rows()
    assert (entry["verb"], entry["outcome"], entry["task"], entry["refs"]["slot"]) == ("slot start", "busy", "L3", row)
    assert entry["actor"] == "unknown" and entry["tier"] is None
    assert entry["reason"].startswith("a row is Running on this GPU")


def test_a_ledger_that_cannot_be_written_never_undoes_what_notion_holds(world, tmp_path, monkeypatch):
    blocker = tmp_path / "not-a-folder"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setenv("LAB_STATE_DIR", str(blocker))
    row = world.seed("L3: smoke")
    answer, code = world.call("start", "--row", row)
    assert code == 0 and answer["ok"] is True and answer["outcome"] == "started"
    assert answer["ledger_seq"] is None and answer["ledger_error"]
    assert world.stored(row)["Status"] == "Running"


def test_the_ledger_row_of_the_answer_is_reported_in_it(world):
    answer, _ = world.call("request", "--task", "L3", "--what", "smoke")
    assert answer["ledger_seq"] == 1 == ledger_rows()[0]["seq"]


# --- no token ----------------------------------------------------------------------------

def run_offline(*argv: str) -> tuple[dict, int]:
    return slots.run(list(argv), environ={}, env_files=[], now=lambda: T0)


def test_without_a_token_a_request_prints_the_exact_row_to_create_and_exits_2(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("a verb without a token must not touch the network")

    monkeypatch.setattr(slots_notion, "urllib_transport", no_network)
    answer, code = run_offline("request", "--task", "L3", "--what", "smoke LoRA 200 steps")
    assert code == 2 and answer["ok"] is False and answer["outcome"] == "no_token" and answer["verb"] == "slot request"
    row = {"Request": "L3: smoke LoRA 200 steps", "Task ID": "L3", "GPU": B, "Minutes": 15, "Status": "Queued",
           "Agent": "Ilan", "date:Requested:start": "2026-10-10T14:00:00.000Z", "date:Requested:is_datetime": 1}
    assert answer["write"] == row
    create, query = answer["steps"]
    assert create["tool"] == f"mcp__{allowed.NOTION_SERVER}__notion-create-pages"
    assert create["arguments"] == {"parent": {"type": "data_source_id", "data_source_id": slots.DATA_SOURCE_ID},
                                   "pages": [{"properties": row}]}
    assert query["tool"] == f"mcp__{allowed.NOTION_SERVER}__notion-query-data-sources"
    assert query["arguments"] == {"data": {
        "mode": "rows", "data_source_url": f"collection://{slots.DATA_SOURCE_ID}", "limit": 100,
        "filter": {"type": "group", "operator": "and", "filters": [
            {"type": "property", "property": "GPU", "propertyType": "select", "operator": "enum_is",
             "value": {"type": "exact", "value": B}},
            {"type": "group", "operator": "or", "filters": [
                {"type": "property", "property": "Status", "propertyType": "select", "operator": "enum_is",
                 "value": {"type": "exact", "value": "Queued"}},
                {"type": "property", "property": "Status", "propertyType": "select", "operator": "enum_is",
                 "value": {"type": "exact", "value": "Running"}}]}]}}}
    assert "asks for more minutes than yours" in answer["rule"]


def test_without_a_token_a_long_request_prints_its_reason_in_the_title():
    answer, code = run_offline("request", "--task", "L3", "--what", "train", "--minutes", "60", "--why", "training run",
                               "--gpu", "A")
    assert code == 2
    assert answer["write"]["Request"] == "L3: train (60 min: training run)" and answer["write"]["Minutes"] == 60
    assert answer["write"]["GPU"] == A


ROW = "3f596ef5fec581f096aac6afa10b7b12"
UPDATE = f"mcp__{allowed.NOTION_SERVER}__notion-update-page"


@pytest.mark.parametrize("argv,props,tools", [
    (["start", "--row", ROW],
     {"Status": "Running", "date:Started:start": "2026-10-10T14:00:00.000Z", "date:Started:is_datetime": 1},
     ["fetch", "query-data-sources", "update-page", "query-data-sources"]),
    (["extend", "--row", ROW, "--note", "loss fell"], {"Extended": "__YES__", "Progress note": "loss fell"},
     ["fetch", "update-page"]),
    (["done", "--row", ROW, "--result", "7.7 tok/s"],
     {"Status": "Done", "date:Ended:start": "2026-10-10T14:00:00.000Z", "date:Ended:is_datetime": 1, "Result": "7.7 tok/s"},
     ["update-page"]),
    (["done", "--row", ROW, "--cancel"],
     {"Status": "Cancelled", "date:Ended:start": "2026-10-10T14:00:00.000Z", "date:Ended:is_datetime": 1},
     ["update-page"]),
    (["done", "--row", ROW, "--cancel", "--result", "not needed"],
     {"Status": "Cancelled", "date:Ended:start": "2026-10-10T14:00:00.000Z", "date:Ended:is_datetime": 1,
      "Result": "not needed"}, ["update-page"]),
    (["complain", "--row", ROW, "--text", "behind the eval for 40 minutes"],
     {"Complaint": "2026-10-10T14:00Z behind the eval for 40 minutes"}, ["fetch", "update-page"]),
])
def test_without_a_token_each_verb_prints_the_properties_to_set_on_the_row(argv, props, tools):
    answer, code = run_offline(*argv)
    assert code == 2 and answer["outcome"] == "no_token" and answer["write"] == props
    assert [step["tool"].rsplit("notion-", 1)[1] for step in answer["steps"]] == tools
    [update] = [step for step in answer["steps"] if step["tool"] == UPDATE]
    assert update["arguments"] == {"page_id": ROW, "command": "update_properties", "properties": props}


def test_without_a_token_status_prints_the_query_to_run():
    answer, code = run_offline("status", "--gpu", "A")
    assert code == 2 and answer["outcome"] == "no_token"
    [query] = answer["steps"]
    assert query["arguments"]["data"]["filter"]["filters"][0]["value"]["value"] == A
    assert "write" not in answer


def decode_mcp(write: dict) -> dict:
    """The flat properties of a Notion MCP call as the Python values they mean."""
    out = {}
    for key, value in write.items():
        if key.startswith("date:") and key.endswith(":start"):
            out[key.split(":")[1]] = slots.parse_moment(value)
        elif key.startswith("date:"):
            assert value == 1
        elif value in ("__YES__", "__NO__"):
            out[key] = value == "__YES__"
        else:
            out[key] = value
    return out


def stored_as_python(world: World, row: str, names) -> dict:
    values = world.stored(row)
    return {name: slots.parse_moment(values[name]) if name in ("Requested", "Started", "Ended") else values[name]
            for name in names}


def test_the_row_printed_without_a_token_is_the_row_the_verb_writes_with_one():
    online = World()
    argv = ["request", "--task", "L3", "--what", "train the adapter", "--minutes", "60", "--why", "training run",
            "--gpu", "A"]
    printed, code = slots.run(argv, environ={}, env_files=[], now=lambda: online.now)
    assert code == 2
    written, _ = online.call(*argv)
    wanted = decode_mcp(printed["write"])
    assert stored_as_python(online, written["row"], wanted) == wanted
    stored = online.stored(written["row"])
    assert {name for name, value in stored.items() if value not in (None, "", False)} == set(wanted)


@pytest.mark.parametrize("make,argv", [
    (lambda w: w.seed("L3: smoke"), ["start", "--row", "{row}"]),
    (lambda w: w.running("L3: smoke", started_ago=3), ["extend", "--row", "{row}", "--note", "loss fell"]),
    (lambda w: w.running("L3: smoke", started_ago=3), ["done", "--row", "{row}", "--result", "7.7 tok/s"]),
    (lambda w: w.seed("L3: smoke"), ["done", "--row", "{row}", "--cancel", "--result", "not needed"]),
    (lambda w: w.seed("L3: smoke"), ["done", "--row", "{row}", "--cancel"]),
])
def test_the_properties_printed_without_a_token_are_the_ones_the_verb_sets_with_one(make, argv):
    world = World()
    row = make(world)
    argv = [arg.format(row=row) for arg in argv]
    printed, code = slots.run(argv, environ={}, env_files=[], now=lambda: world.now)
    assert code == 2
    before = dict(world.stored(row))
    answer, code = world.call(*argv)
    assert code == 0, answer
    wanted = decode_mcp(printed["write"])
    assert stored_as_python(world, row, wanted) == wanted
    after = world.stored(row)
    changed = {name for name in after if after[name] != before[name]}
    assert changed == {name for name in wanted if after[name] != before[name]}


@pytest.mark.parametrize("argv", [
    ["request", "--task", "L3", "--what", "x"], ["start", "--row", ROW], ["extend", "--row", ROW, "--note", "n"],
    ["done", "--row", ROW, "--result", "r"], ["complain", "--row", ROW], ["status"]])
def test_without_a_token_each_call_still_writes_its_ledger_row(argv):
    answer, code = run_offline(*argv)
    [entry] = ledger_rows()
    assert code == 2 and answer["ledger_seq"] == entry["seq"] == 1
    assert (entry["verb"], entry["outcome"]) == (f"slot {argv[0]}", "no_token")


def test_a_blank_token_counts_as_missing(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("NOTION_TOKEN=   \n", encoding="utf-8")
    answer, code = slots.run(["status"], environ={"NOTION_TOKEN": ""}, env_files=[env_file], now=lambda: T0)
    assert code == 2 and answer["outcome"] == "no_token"


def test_the_token_is_found_in_the_environment_then_in_env_files_and_only_its_presence_is_ever_used(tmp_path, capsys):
    env_file = tmp_path / ".env"
    env_file.write_text(f"OTHER=1\nNOTION_TOKEN={TOKEN}\n", encoding="utf-8")
    fake = FakeNotion()
    with fake.serve() as base:
        environ = {"LAB_NOTION_URL": base}
        answer, code = slots.run(["request", "--task", "L3", "--what", "x"], environ=environ, env_files=[env_file],
                                 now=lambda: T0)
        assert code == 0 and answer["outcome"] == "queued"
        environ = {"LAB_NOTION_URL": base, "NOTION_TOKEN": "a-different-token-0123456789"}
        answer, code = slots.run(["status"], environ=environ, env_files=[env_file], now=lambda: T0)
        assert code == 1 and answer["kind"] == "unauthorized"
    assert len(fake.row_ids()) == 1
    assert TOKEN not in json.dumps(answer) and "a-different-token-0123456789" not in json.dumps(answer)
    assert TOKEN not in (ledger.ledger_path()).read_text(encoding="utf-8")
    assert "a-different-token-0123456789" not in (ledger.ledger_path()).read_text(encoding="utf-8")
    captured = capsys.readouterr()
    assert TOKEN not in captured.out + captured.err


# --- the verb in the dispatcher ----------------------------------------------------------

def test_slot_is_a_registered_verb_with_a_line_in_help(capsys):
    assert lab_cli.VERBS["slot"] == "lab.machine.slots:main"
    assert lab_cli.main(["--help"]) == 0
    help_lines = [line for line in capsys.readouterr().out.splitlines() if line.strip().startswith("slot ")]
    assert len(help_lines) == 1 and "request, start, extend, done, complain, status" in help_lines[0]


def test_the_dispatcher_prints_one_json_line_and_exits_2_without_a_token(capsys):
    code = lab_cli.main(["slot", "request", "--task", "L3", "--what", "smoke LoRA 200 steps"])
    out, err = capsys.readouterr()
    answer = one_json_line(out)
    assert code == 2 and answer["outcome"] == "no_token" and answer["ledger_seq"] == 1 and err == ""


def test_the_dispatcher_exits_0_or_1_for_the_queues_answer(monkeypatch, capsys):
    fake = FakeNotion()
    with fake.serve() as base:
        monkeypatch.setenv("NOTION_TOKEN", TOKEN)
        monkeypatch.setenv("LAB_NOTION_URL", base)
        assert lab_cli.main(["slot", "request", "--task", "L3", "--what", "smoke LoRA"]) == 0
        first = one_json_line(capsys.readouterr().out)
        assert lab_cli.main(["slot", "start", "--row", first["row"]]) == 0
        assert one_json_line(capsys.readouterr().out)["outcome"] == "started"
        assert lab_cli.main(["slot", "start", "--row", first["row"]]) == 1
        refused = one_json_line(capsys.readouterr().out)
        assert refused["ok"] is False and refused["outcome"] == "not_queued"
        assert lab_cli.main(["slot", "done", "--row", first["row"], "--result", "ok, 3 runs"]) == 0
        assert one_json_line(capsys.readouterr().out)["card_released"] is True
    assert [r["verb"] for r in ledger_rows()] == ["slot request", "slot start", "slot start", "slot done"]


def test_a_bad_sub_verb_is_bad_arguments(capsys):
    assert lab_cli.main(["slot", "reserve"]) == 2
    out, err = capsys.readouterr()
    assert out == "" and "invalid choice" in err
    assert ledger_rows() == []


def test_while_the_stop_file_exists_every_slot_verb_is_refused_before_it_runs(tmp_path, monkeypatch, capsys):
    state = Path(os.environ["LAB_STATE_DIR"])
    state.mkdir(parents=True)
    (state / "STOP").write_text("hands off\n", encoding="utf-8")

    def never(*args, **kwargs):
        raise AssertionError("a stopped verb must not build a Notion client")

    monkeypatch.setattr(slots_notion, "NotionClient", never)
    monkeypatch.setenv("NOTION_TOKEN", TOKEN)
    for argv in (["request", "--task", "L3", "--what", "x"], ["start", "--row", ROW],
                 ["extend", "--row", ROW, "--note", "n"], ["done", "--row", ROW, "--result", "r"],
                 ["complain", "--row", ROW], ["status"]):
        assert lab_cli.main(["slot", *argv]) == 1
        refused = one_json_line(capsys.readouterr().out)
        assert refused["ok"] is False and refused["outcome"] == "stopped" and refused["reason"] == "hands off"
    assert not ledger.ledger_path().exists()


def test_called_without_the_dispatcher_the_verb_still_honors_the_stop_file(capsys):
    state = Path(os.environ["LAB_STATE_DIR"])
    state.mkdir(parents=True)
    (state / "STOP").write_text("hands off\n", encoding="utf-8")
    with pytest.raises(SystemExit) as caught:
        slots.main(["status"])
    assert isinstance(caught.value, stop.Stopped) and caught.value.code == 1
    assert one_json_line(capsys.readouterr().out)["outcome"] == "stopped"


def test_the_real_command_line_exits_2_with_the_row_to_write_when_there_is_no_token(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("NOTION_") and not k.startswith("LAB_")}
    env.update(LAB_STATE_DIR=str(tmp_path / "state"), LAB_ENV_FILES="", PYTHONIOENCODING="utf-8")
    done = subprocess.run([sys.executable, "-m", "lab", "slot", "request", "--task", "L3", "--what", "smoke LoRA"],
                          cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=120)
    answer = one_json_line(done.stdout)
    assert done.returncode == 2 and answer["outcome"] == "no_token" and answer["write"]["Task ID"] == "L3"
    assert done.stderr == ""
    rows = ledger.read(state_dir=tmp_path / "state")
    assert [(r["verb"], r["outcome"]) for r in rows] == [("slot request", "no_token")]


@pytest.mark.parametrize("call", [
    'slot request --task L3 --what "smoke LoRA 200 steps"',
    'slot request --task L3 --what "train the adapter" --minutes 60 --why "training run" --gpu B',
    f"slot start --row {ROW}",
    f'slot extend --row {ROW} --note "loss fell from 2.1 to 1.4"',
    f'slot done --row {ROW} --result "7.7 tok/s"',
    f"slot done --row {ROW} --cancel",
    f"slot complain --row {ROW}",
    "slot status",
    f"slot status --gpu B --task L3 --row {ROW}",
])
def test_every_call_a_sitting_is_allowed_to_make_is_one_the_verb_accepts(call):
    for python in allowed.PYTHONS:
        for tier in ("fable", "opus", "sonnet"):
            assert allowed.allows(tier, "Bash", f"{python} -m lab {call}"), (tier, call)
    args = slots.parse_args(shlex.split(call)[1:])
    assert args.sub == shlex.split(call)[1]
