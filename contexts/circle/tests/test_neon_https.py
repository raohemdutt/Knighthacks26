"""The Circle's database over HTTPS (port 443): the second transport behind NeonDb, for networks that refuse 5432.

No test here reaches the internet. The endpoint is a local `http.server` that speaks the shape neon_wire.py
assumes and records what it was sent, or the fake endpoint inside FakeNeon. What these tests cannot show is that
the real service speaks that shape: `ATLAS_NEON_TRANSPORT=https python -m contexts.circle.schema --check` does."""

import json
import logging
import socket
import sys
import threading
import traceback
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from contexts.circle import schema
from contexts.circle.application.protocol import AgentMessage
from contexts.circle.application.sharing import CircleOffline, OutboxRow
from contexts.circle.infrastructure import neon, neon_https, neon_wire
from contexts.circle.infrastructure.neon import (
    CircleBoard,
    NeonCircleStore,
    NeonCleanup,
    NeonDb,
    NeonPresence,
    NeonShares,
    NeonTransport,
)
from contexts.circle.infrastructure.neon_https import (
    HttpsConnection,
    NeonHttpsDropped,
    NeonHttpsError,
    NeonHttpsTimeout,
    from_text,
    to_dollar,
    to_text,
)
from contexts.circle.infrastructure.neon_sealed import NeonSealedStore
from contexts.circle.tests.neon_helpers import FAKE_URL, FAKE_URL_PARTS, FakeNeon, wire_result
from contexts.play.league.infrastructure.neon import NeonLeague
from contracts.circle import Proposal, SealedMessage
from contracts.play import LeagueChallengeClosed, LeagueChallengeStarted, LeagueStakeSet, LeagueWeekClosed, Stake

NOW = datetime(2026, 10, 10, 23, tzinfo=UTC)
NAMES_ITS_TRANSPORT = True  # conftest.py: these tests say which transport they mean, so they are not run once per transport


def clean(text: str) -> bool:
    """True when `text` holds no part of the connection string."""
    return not any(part in text for part in (*FAKE_URL_PARTS, FAKE_URL))


# --- a local endpoint ---


class Endpoint:
    """`http.server` on 127.0.0.1 in a thread. `answer(request_json, headers)` returns (status, body) for each POST;
    the default answers every statement with no rows."""

    def __init__(self):
        self.requests: list[tuple[dict, dict]] = []  # (headers, body) in the order received
        self.answer = lambda body, headers: (200, reply(body, lambda q: wire_result([], [], q["query"])))
        self.hold = threading.Event()  # a handler told to wait (the timeout test) is let go when this is set
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                headers = {k.lower(): v for k, v in self.headers.items()}
                outer.requests.append((headers, body))
                status, out = outer.answer(body, headers)
                if status == "wait":
                    outer.hold.wait(5)
                    return
                data = out if isinstance(out, bytes) else json.dumps(out).encode()
                self.send_response(status)
                if status in (301, 302, 307, 308):
                    self.send_header("Location", f"{outer.url}/elsewhere")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()

    def connect(self, timeout: float = 5) -> HttpsConnection:
        return HttpsConnection(FAKE_URL, timeout=timeout, endpoint=self.url + "/sql")

    def close(self):
        self.hold.set()
        self.server.shutdown()
        self.server.server_close()


def reply(body: dict, one) -> dict:
    """The endpoint's reply to `body`, with `one(query)` answering each statement."""
    if "queries" in body:
        return {"results": [one(q) for q in body["queries"]]}
    return one(body)


def table(fields: list[tuple[str, int]], rows: list[list], command: str = "SELECT") -> dict:
    return {"command": command, "rowCount": len(rows), "rowAsArray": True,
            "fields": [{"name": n, "dataTypeID": t} for n, t in fields], "rows": rows}


@pytest.fixture
def endpoint():
    e = Endpoint()
    yield e
    e.close()


def closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --- the wire shape (neon_wire.py) ---


def test_the_request_is_one_statement_or_one_transaction(endpoint):
    assert neon_wire.endpoint(FAKE_URL) == "https://ep-fake-name-123456-pooler.c-9.us-east-9.aws.neon.invalid/sql"
    conn = endpoint.connect()
    cur = conn.cursor()
    cur.execute("SELECT id FROM circles WHERE id = %s", ("team",))
    assert cur.fetchall() == []
    headers, body = endpoint.requests[0]
    assert body == {"query": "SELECT id FROM circles WHERE id = $1", "params": ["team"]}
    assert headers["neon-connection-string"] == FAKE_URL
    assert (headers["neon-raw-text-output"], headers["neon-array-mode"]) == ("true", "true")
    assert headers["content-type"] == "application/json" and "neon-batch-read-only" not in headers

    cur.execute("INSERT INTO circles (id, name) VALUES (%s, %s)", ("team", "team"))
    cur.execute("INSERT INTO members (id, circle_id) VALUES (%s, %s)", ("dom", "team"))
    conn.commit()
    headers, body = endpoint.requests[1]
    assert body == {"queries": [{"query": "INSERT INTO circles (id, name) VALUES ($1, $2)", "params": ["team", "team"]},
                                {"query": "INSERT INTO members (id, circle_id) VALUES ($1, $2)", "params": ["dom", "team"]}]}
    assert "neon-batch-read-only" not in headers


def test_a_read_only_connection_sends_every_statement_as_a_read_only_transaction(endpoint):
    conn = endpoint.connect()
    conn.read_only = True
    cur = conn.cursor()
    cur.execute("SELECT 1")
    cur.fetchall()
    headers, body = endpoint.requests[0]
    assert body == {"queries": [{"query": "SELECT 1", "params": []}]} and headers["neon-batch-read-only"] == "true"


def test_a_reply_as_objects_instead_of_arrays_is_read_too():
    r = neon_wire.results(200, json.dumps({"command": "SELECT", "rowCount": 1, "fields": [
        {"name": "id", "dataTypeID": 25}, {"name": "n", "dataTypeID": 20}], "rows": [{"n": "2", "id": "a"}]}).encode(), 1, False)
    assert (r[0].columns, r[0].rows, r[0].rowcount) == (["id", "n"], [["a", "2"]], 1)
    with pytest.raises(neon_wire.Refused):  # one result for two statements: not a reply to trust
        neon_wire.results(200, json.dumps({"results": [{"fields": [], "rows": []}]}).encode(), 2, True)


# --- placeholders ---


def test_placeholders_become_dollar_numbers_the_way_psycopg_reads_them():
    assert to_dollar("SELECT %s, %s::timestamptz + (x - now())", ["a", "b"]) == ("SELECT $1, $2::timestamptz + (x - now())", ["a", "b"])
    assert to_dollar("SELECT 'half is 50%%', %s", (1,)) == ("SELECT 'half is 50%', $1", ["1"])
    assert to_dollar("SELECT '100%' AS literal", None) == ("SELECT '100%' AS literal", [])  # no parameters: untouched
    assert to_dollar("SELECT 1", ()) == ("SELECT 1", [])
    # an interval has to say so: `now() - $3` alone would be read as a timestamp
    sql, params = to_dollar("DELETE FROM t WHERE created_at < now() - %s LIMIT %s", (timedelta(days=30), 500))
    assert sql == "DELETE FROM t WHERE created_at < now() - $1::interval LIMIT $2"
    assert params == ["30 days 0 seconds 0 microseconds", "500"]
    for bad_sql, bad_params in (("SELECT %s", ()), ("SELECT 1", (1,)), ("SELECT %d", (1,)), ("SELECT %s", "text")):
        with pytest.raises((ValueError, TypeError)):
            to_dollar(bad_sql, bad_params)


# --- values in and out ---


@pytest.mark.parametrize("value, text", [
    (None, None), ("Library", "Library"), (True, "true"), (False, "false"), (7, "7"), (7.5, "7.5"), (Decimal("1.20"), "1.20"),
    (date(2026, 10, 9), "2026-10-9".replace("-9", "-09")), (NOW, "2026-10-10T23:00:00+00:00"),
    (datetime(2026, 10, 10, 19, tzinfo=timezone(timedelta(hours=-4))), "2026-10-10T19:00:00-04:00"),
    (timedelta(days=30), "30 days 0 seconds 0 microseconds"),
    (["s1", "s2"], '{"s1","s2"}'), (['a "b"', "c\\d", None], '{"a \\"b\\"","c\\\\d",NULL}'), ([], "{}"), ([1, 2], '{"1","2"}'),
    ('{"answer": "free"}', '{"answer": "free"}'),  # json goes in as the text the callers already make
])
def test_a_python_value_goes_in_as_postgres_text(value, text):
    assert to_text(value) == text


def test_a_value_it_cannot_send_is_an_error_not_a_guess():
    with pytest.raises(TypeError, match="json.dumps"):
        to_text({"answer": "free"})


@pytest.mark.parametrize("type_id, text, value", [
    (25, "Library", "Library"), (1043, "x", "x"), (16, "t", True), (16, "f", False), (20, "42", 42), (23, "-3", -3),
    (701, "7.5", 7.5), (1700, "0.4", Decimal("0.4")), (3802, '{"answer": "free", "n": [1]}', {"answer": "free", "n": [1]}),
    (114, "[1, 2]", [1, 2]), (1082, "2026-10-09", date(2026, 10, 9)),
    (1184, "2026-10-10 23:00:00+00", NOW), (1184, "2026-10-10 19:00:00.5-04", NOW + timedelta(microseconds=500000)),
    (1184, "2026-10-11 04:30:00.123456+05:30", NOW + timedelta(microseconds=123456)),
    (1114, "2026-10-10 23:00:00", NOW.replace(tzinfo=None)), (25, None, None), (1184, None, None),
    (2950, "0b2f0c1e-0000-0000-0000-000000000000", "0b2f0c1e-0000-0000-0000-000000000000"),  # a type it does not know: its text
    (20, 42, 42), (3802, {"a": 1}, {"a": 1}), (1700, 0.5, Decimal("0.5")),  # a server that answered in JSON types after all
])
def test_postgres_text_comes_out_as_the_value_psycopg_gives(type_id, text, value):
    got = from_text(type_id, text)
    assert got == value and type(got) is type(value)
    if type_id == 1184 and text is not None:
        assert got.utcoffset() == timedelta(0)  # tz-aware, in UTC


def test_rows_come_back_typed_with_rowcount_and_description(endpoint):
    endpoint.answer = lambda body, headers: (200, table(
        [("member_id", 25), ("streak", 23), ("met", 16), ("day", 1082), ("shared_at", 1184), ("body", 3802)],
        [["dom", "4", "t", "2026-10-09", "2026-10-10 23:00:00+00", '{"k": 1}'], ["ana", None, "f", None, None, None]]))
    cur = endpoint.connect().cursor()
    cur.execute("SELECT 1")
    assert cur.rowcount == 2 and [d[0] for d in cur.description] == ["member_id", "streak", "met", "day", "shared_at", "body"]
    assert cur.fetchone() == ("dom", 4, True, date(2026, 10, 9), NOW, {"k": 1})
    assert cur.fetchall() == [("ana", None, False, None, None, None)]
    assert cur.fetchone() is None and cur.fetchall() == []
    assert len(endpoint.requests) == 1


def test_a_write_has_a_rowcount_and_no_rows(endpoint):
    endpoint.answer = lambda body, headers: (200, {"command": "DELETE", "rowCount": 3, "rows": [], "fields": [], "rowAsArray": True})
    cur = endpoint.connect().cursor()
    cur.execute("DELETE FROM presence WHERE member_id = %s", ("dom",))
    assert cur.rowcount == 3 and cur.description is None
    with pytest.raises(NeonHttpsError, match="did not return rows"):
        cur.fetchall()


# --- transactions: what is one, and what is not ---


def test_writes_wait_and_go_together_at_commit(endpoint):
    conn = endpoint.connect()
    cur = conn.cursor()
    cur.execute("INSERT INTO circles (id, name) VALUES (%s, %s)", ("team", "team"))
    cur.execute("INSERT INTO members (id) VALUES (%s)", ("dom",))
    assert endpoint.requests == []  # nothing has left the box yet
    conn.commit()
    assert [len(body["queries"]) for _, body in endpoint.requests] == [2]
    conn.commit()  # nothing new: nothing sent
    assert len(endpoint.requests) == 1


def test_rollback_forgets_what_was_not_sent_and_cannot_undo_what_was(endpoint):
    conn = endpoint.connect()
    cur = conn.cursor()
    cur.execute("INSERT INTO circles (id, name) VALUES (%s, %s)", ("team", "team"))
    assert cur.rowcount == 1  # asking for the answer sends it: this insert is committed now
    cur.execute("INSERT INTO members (id) VALUES (%s)", ("dom",))
    conn.rollback()
    conn.commit()
    assert [body["query"].split()[2] for _, body in endpoint.requests] == ["circles"]  # the member row was never sent
    assert cur.rowcount == -1  # the forgotten statement has no answer


def test_a_read_between_writes_splits_the_work_into_two_transactions(endpoint):
    conn = endpoint.connect()
    cur = conn.cursor()
    cur.execute("INSERT INTO circles (id, name) VALUES (%s, %s)", ("team", "team"))
    cur.execute("INSERT INTO events (id) VALUES (%s)", ("t1",))
    assert cur.rowcount == 1
    cur.execute("INSERT INTO rsvps (event_id) VALUES (%s)", ("t1",))
    cur.execute("INSERT INTO rsvps (event_id) VALUES (%s)", ("t2",))
    conn.commit()
    assert [len(body["queries"]) for _, body in endpoint.requests] == [2, 2]


def test_a_transaction_the_database_refuses_leaves_nothing_waiting(endpoint):
    endpoint.answer = lambda body, headers: (400, {"message": 'relation "nope" does not exist', "code": "42P01"})
    conn = endpoint.connect()
    cur = conn.cursor()
    cur.execute("INSERT INTO circles (id, name) VALUES (%s, %s)", ("team", "team"))
    cur.execute("INSERT INTO nope (id) VALUES (%s)", ("x",))
    with pytest.raises(NeonHttpsError) as e:
        conn.commit()
    assert e.value.sqlstate == "42P01" and "nope" in str(e.value)
    conn.commit()  # the refused statements are gone, not sent again
    assert len(endpoint.requests) == 1


EVENT = Proposal(title="Study session", start=NOW + timedelta(days=1), end=NOW + timedelta(days=1, hours=2),
                 location="Library", invitees=["hemdutt"])


def circle_work(db: NeonDb) -> dict:
    """Every piece of `work` the app hands to NeonDb.run, by name."""
    store, shares, board = NeonCircleStore(db), NeonShares(db), CircleBoard(db)
    presence, sealed, league = NeonPresence(db), NeonSealedStore(db), NeonLeague(db, board)
    message = AgentMessage(id="m1", circle_id="team", thread_id="t1", from_member="dom", kind="proposal", proposal=EVENT,
                           created_at=NOW)
    sealed_row = SealedMessage(id="s1", circle_id="team", from_member="ana", to_member="bob", kind="face_share",
                               body={"header": {"v": 1}, "ciphertext": "AAAA"}, created_at=NOW)
    place = {"place": "Library", "source": "phone", "shared_at": NOW.isoformat(),
             "expires_at": (NOW + timedelta(hours=2)).isoformat()}
    habit = {"habit_id": "sleep", "day": "2026-10-09", "met": True, "value": 7.5, "streak": 3}
    count = {"items_seen": 9, "recyclable_seen": 5, "recycled": 4, "contamination": 1, "diversion_rate": 0.8}
    score = {"round_id": "r1", "session_id": "s1", "item": "can", "players": 2, "correct": True, "seconds": 6.5,
             "path_len": 7, "score": 88.0}
    stake = LeagueStakeSet(ts=NOW, circle_id="team", week_start=date(2026, 10, 5), stake=Stake(kind="pays_for_food"), set_by="dom")
    week = LeagueWeekClosed(ts=NOW, circle_id="team", week_start=date(2026, 10, 5), week_end=date(2026, 10, 11),
                            standings=[], stake=Stake(kind="pays_for_food"), summary="a quiet week")
    dare = LeagueChallengeStarted(ts=NOW, challenge_id="c1", circle_id="team", title="Sleep", habit_id="sleep",
                                  direction="at_least", target=7.0, starts=date(2026, 10, 5), ends=date(2026, 10, 11),
                                  members=["dom", "ana"], started_by="dom")
    done = LeagueChallengeClosed(ts=NOW, challenge_id="c1", circle_id="team", title="Sleep", results=[],
                                 winners=["dom"], summary="Dom")
    return {
        "store.post": lambda: store.post(message),
        "store.since": lambda: store.since("team", NOW),
        "store.thread": lambda: store.thread("team", "t1"),
        "store.record_event": lambda: store.record_event("team", "t1", EVENT, "dom"),
        "shares.push habit_summary": lambda: shares.push(OutboxRow("e1", "team", "dom", "habit_summary", habit, NOW)),
        "shares.push impact_count": lambda: shares.push(OutboxRow("e2", "team", "dom", "impact_count", count, NOW)),
        "shares.push game_score": lambda: shares.push(OutboxRow("e3", "team", "dom", "game_score", score, NOW)),
        "board.habit_leaderboard": lambda: board.habit_leaderboard("team"),
        "board.impact_leaderboard": lambda: board.impact_leaderboard("team"),
        "board.top_scores": lambda: board.top_scores("team"),
        "board.habit_summaries": lambda: board.habit_summaries("team", date(2026, 10, 1)),
        "board.impact_counters": lambda: board.impact_counters("team"),
        "board.upcoming_events": lambda: board.upcoming_events("team", NOW),
        "presence.push": lambda: presence.push(OutboxRow("e4", "team", "dom", "presence", place, NOW)),
        "presence.active": lambda: presence.active("team", NOW),
        "cleanup.sweep": lambda: NeonCleanup(db).sweep("team", "dom", timedelta(days=30)),
        "sealed.post": lambda: sealed.post(sealed_row),
        "sealed.inbox": lambda: sealed.inbox("team", "bob"),
        "sealed.delete": lambda: sealed.delete("team", ["s1", "s2"]),
        "league.push_stake": lambda: league.push_stake(stake),
        "league.push_week": lambda: league.push_week(week),
        "league.push_challenge": lambda: league.push_challenge(dare, done),
    }


# Statements per request for each `work` over https. One number: one request, so one transaction, as over tcp.
# More than one number: the work reads an answer in between, and each request commits by itself.
REQUESTS = {
    "store.post": [3], "store.since": [1], "store.thread": [1],
    "store.record_event": [4, 2],  # circle, two members and the event; then, after `rowcount`, the two RSVPs
    "shares.push habit_summary": [4], "shares.push impact_count": [4], "shares.push game_score": [4],
    "board.habit_leaderboard": [1], "board.impact_leaderboard": [1], "board.top_scores": [1],
    "board.habit_summaries": [1], "board.impact_counters": [1], "board.upcoming_events": [1],
    "presence.push": [3], "presence.active": [1],
    "cleanup.sweep": [1, 1, 1],  # `rowcount` after each delete
    "sealed.post": [3], "sealed.inbox": [1], "sealed.delete": [1],
    "league.push_stake": [3], "league.push_week": [2], "league.push_challenge": [4],
}


@pytest.mark.parametrize("name", sorted(REQUESTS))
def test_every_piece_of_work_is_sent_as_the_transactions_the_code_says(name):
    """Also the check that every statement the app has converts: the fake endpoint refuses a `%s`, a parameter
    that is not text, and `$n` that do not match the parameters."""
    neon = FakeNeon(transport="https")
    work = circle_work(NeonDb(neon.connect))
    assert set(work) == set(REQUESTS)
    work[name]()
    assert [len(body.get("queries", [body])) for _, body in neon.requests] == REQUESTS[name]
    params = [p for _, body in neon.requests for q in body.get("queries", [body]) for p in q["params"]]
    assert all(p is None or isinstance(p, str) for p in params)


def test_the_awkward_parameters_arrive_as_the_server_needs_them():
    neon = FakeNeon(transport="https")
    work = circle_work(NeonDb(neon.connect))
    work["sealed.delete"]()
    assert neon.requests[-1][1] == {"query": "DELETE FROM sealed_messages WHERE circle_id = $1 AND id = ANY($2)",
                                    "params": ["team", '{"s1","s2"}']}
    work["cleanup.sweep"]()
    body = neon.requests[-1][1]
    assert "expires_at < now() - $3::interval" in body["query"] and body["params"][2] == "30 days 0 seconds 0 microseconds"
    work["store.post"]()
    insert = neon.requests[-1][1]["queries"][-1]
    assert json.loads(insert["params"][5])["title"] == "Study session" and insert["params"][6] is None  # jsonb, and a null
    assert insert["params"][7] == "2026-10-10T23:00:00+00:00"  # timestamptz
    work["presence.push"]()
    assert "now() + ($6::timestamptz - $7::timestamptz)" in neon.requests[-1][1]["queries"][-1]["query"]
    work["shares.push habit_summary"]()
    assert neon.requests[-1][1]["queries"][-1]["params"][2:] == ["sleep", "2026-10-09", "true", "7.5", "3"]


# --- failures: offline is CircleOffline, a refused statement is an error, and neither hangs ---


def test_a_refused_connection_is_offline(endpoint):
    conn = HttpsConnection(FAKE_URL, timeout=2, endpoint=f"http://127.0.0.1:{closed_port()}/sql")
    db = NeonDb(lambda: conn)
    with pytest.raises(CircleOffline, match="unreachable") as e:
        db.run(lambda cur: cur.execute("SELECT 1"))
    assert isinstance(e.value.__cause__, NeonHttpsDropped) and str(e.value.__cause__) == "connection refused"
    assert isinstance(e.value.__cause__, OSError) and db.offline  # the class NeonDb already treats as a lost connection


def test_no_answer_in_time_is_offline_and_is_not_tried_twice(endpoint):
    endpoint.answer = lambda body, headers: ("wait", None)
    db = NeonDb(lambda: endpoint.connect(timeout=0.2))
    db._conn = endpoint.connect(timeout=0.2)  # a connection that has worked before: NeonDb would normally retry once
    with pytest.raises(CircleOffline) as e:
        db.run(lambda cur: cur.execute("SELECT 1"))
    assert isinstance(e.value.__cause__, NeonHttpsTimeout) and len(endpoint.requests) == 1


def test_a_request_that_fails_at_once_is_tried_once_more(endpoint):
    answers = [(502, b"bad gateway"), (200, table([("n", 23)], [["1"]]))]
    endpoint.answer = lambda body, headers: answers.pop(0)
    db = NeonDb(endpoint.connect, retry_s=0)
    db._conn = endpoint.connect()  # used before, as in a running box
    assert db.run(lambda cur: cur.execute("SELECT 1").fetchall()) == [(1,)]
    assert len(endpoint.requests) == 2 and db.offline is None


@pytest.mark.parametrize("status, body, reason", [
    (500, b"internal error", "HTTP 500"), (502, b"<html>bad gateway</html>", "HTTP 502"), (429, b"slow down", "HTTP 429"),
    (401, b"{}", "HTTP 401"), (404, b"not found", "HTTP 404"), (307, b"", "HTTP 307"),
    (400, {"message": "password authentication failed for user 'box_owner'", "code": "28P01"}, "HTTP 400, SQLSTATE 28P01"),
    (400, {"message": "Control plane request failed: endpoint is disabled"}, "HTTP 400"),
    (400, {"message": "terminating connection due to administrator command", "code": "57P01"}, "HTTP 400, SQLSTATE 57P01"),
    (200, b"<html>Sign in to the venue Wi-Fi</html>", "the answer was not from the database"),
    (200, {"unexpected": True}, "the answer was not from the database"),
])
def test_an_answer_that_is_not_about_the_statement_is_offline(endpoint, status, body, reason):
    endpoint.answer = lambda b, h: (status, body)
    db = NeonDb(endpoint.connect)
    with pytest.raises(CircleOffline) as e:
        db.run(lambda cur: cur.execute("SELECT 1"))
    assert isinstance(e.value.__cause__, NeonHttpsDropped) and str(e.value.__cause__) == reason
    assert clean(str(e.value) + repr(e.value.__cause__))


def test_a_redirect_is_not_followed(endpoint):
    endpoint.answer = lambda b, h: (307, b"")
    with pytest.raises(NeonHttpsDropped):
        endpoint.connect().cursor().execute("SELECT 1").fetchall()
    assert len(endpoint.requests) == 1  # the connection string did not go to the address it was pointed at


@pytest.mark.parametrize("code, message", [
    ("42P01", 'relation "sealed_messages" does not exist'), ("23503", "insert or update violates foreign key constraint"),
    ("25006", "cannot execute INSERT in a read-only transaction"), (None, "unsupported parameter type"),
])
def test_a_statement_postgres_refuses_is_an_error_and_the_box_stays_online(endpoint, code, message):
    endpoint.answer = lambda b, h: (400, {"message": message, **({"code": code} if code else {})})
    db = NeonDb(endpoint.connect)
    with pytest.raises(NeonHttpsError) as e:
        db.run(lambda cur: cur.execute("SELECT 1"))
    assert (e.value.sqlstate, str(e.value)) == (code, message) and not isinstance(e.value, OSError)
    assert db.offline is None
    endpoint.answer = lambda b, h: (200, table([("n", 23)], [["1"]]))
    assert db.run(lambda cur: cur.execute("SELECT 1").fetchall()) == [(1,)]  # the next call simply works


def test_offline_over_https_queues_like_offline_over_tcp(tmp_path):
    """The outbox does not know which transport failed: the rows wait, and go when the database is back."""
    from contexts.circle.application.sharing import CircleSharing
    from contexts.circle.infrastructure.outbox import CircleOutbox
    from contracts.habits import DayClosed

    neon = FakeNeon(transport="https")
    neon.down = True
    sharing = CircleSharing(CircleOutbox(tmp_path / "circle.sqlite"), NeonShares(NeonDb(neon.connect, retry_s=0)),
                            "team", "dom", True)
    sharing.on_day_closed(DayClosed(ts=NOW, habit_id="sleep", day=date(2026, 10, 9), met=True, value=7.5, streak=3))
    assert sharing.sync(NOW) == [] and len(sharing.outbox.pending()) == 1 and neon.requests == []
    neon.down = False
    assert len(sharing.sync(NOW)) == 1 and sharing.outbox.pending() == []
    assert [len(body["queries"]) for _, body in neon.requests] == [4]


# --- which transport: ATLAS_NEON_TRANSPORT ---


def refuses(url):
    raise ConnectionRefusedError(f"connection to {url} refused")  # the driver's message names the host


def no_driver(url):
    raise ImportError("No module named 'psycopg'")


def transport_with(tcp, neon: FakeNeon) -> tuple[NeonTransport, list]:
    tried = []

    def counted(url):
        tried.append(url)
        return tcp(url)

    return NeonTransport(tcp=counted, https=lambda url: neon.connect()), tried


def test_auto_falls_back_to_https_remembers_it_and_says_so_once(caplog):
    neon = FakeNeon(transport="https", rows=[("team",)], columns=["id"])
    transport, tried = transport_with(refuses, neon)
    db = NeonDb(lambda: transport.connect(FAKE_URL, "auto"), retry_s=0)
    with caplog.at_level(logging.INFO, logger="atlas.circle"):
        assert db.run(lambda cur: cur.execute("SELECT id FROM circles").fetchall()) == [("team",)]
        db.close()  # the next statement needs a new connection: port 5432 is not tried again
        assert db.run(lambda cur: cur.execute("SELECT id FROM circles").fetchall()) == [("team",)]
        db.close()
        transport.connect(FAKE_URL, "auto")
    assert len(tried) == 1 and transport.chosen == "https" and len(neon.requests) == 2
    assert [r.getMessage() for r in caplog.records] == [
        "Circle database: using https (port 443) until this process restarts; port 5432 could not be reached: ConnectionRefusedError"]
    assert caplog.records[0].levelno == logging.WARNING and clean(caplog.text)


def test_auto_uses_https_when_the_driver_is_not_installed(caplog):
    transport, tried = transport_with(no_driver, FakeNeon(transport="https"))
    with caplog.at_level(logging.INFO, logger="atlas.circle"):
        assert isinstance(transport.connect(FAKE_URL, "auto"), HttpsConnection)
    assert transport.chosen == "https" and "the Postgres driver is not installed" in caplog.text


def test_auto_stays_on_tcp_while_it_connects_and_says_so_once(caplog):
    neon = FakeNeon(transport="tcp")
    transport, tried = transport_with(lambda url: neon.connect(), FakeNeon(transport="https"))
    with caplog.at_level(logging.INFO, logger="atlas.circle"):
        transport.connect(FAKE_URL, "auto"), transport.connect(FAKE_URL, "auto")
    assert transport.chosen == "tcp" and len(tried) == 2
    assert [r.getMessage() for r in caplog.records] == ["Circle database: using tcp (psycopg, port 5432)"]


def test_tcp_only_never_falls_back():
    neon = FakeNeon(transport="https")
    transport, _ = transport_with(refuses, neon)
    db = NeonDb(lambda: transport.connect(FAKE_URL, "tcp"))
    with pytest.raises(CircleOffline, match="unreachable"):
        db.run(lambda cur: cur.execute("SELECT 1"))
    assert neon.requests == [] and transport.chosen is None
    transport, _ = transport_with(no_driver, neon)
    with pytest.raises(CircleOffline, match="driver is not installed"):
        transport.connect(FAKE_URL, "tcp")


def test_https_only_never_tries_the_driver(caplog):
    neon = FakeNeon(transport="https")
    transport, tried = transport_with(refuses, neon)
    with caplog.at_level(logging.INFO, logger="atlas.circle"):
        NeonDb(lambda: transport.connect(FAKE_URL, "https")).run(lambda cur: cur.execute("SELECT 1"))
    assert tried == [] and len(neon.requests) == 1
    assert [r.getMessage() for r in caplog.records] == ["Circle database: using https (port 443), set by ATLAS_NEON_TRANSPORT"]


def test_a_setting_that_is_not_one_of_the_three_is_offline_with_the_reason():
    transport, tried = transport_with(refuses, FakeNeon(transport="https"))
    db = NeonDb(lambda: transport.connect(FAKE_URL, "quic"))
    with pytest.raises(CircleOffline, match="ATLAS_NEON_TRANSPORT must be auto, tcp or https"):
        db.run(lambda cur: cur.execute("SELECT 1"))
    assert tried == []


@pytest.fixture
def no_psycopg(monkeypatch):
    monkeypatch.setitem(sys.modules, "psycopg", None)  # `import psycopg` now raises ImportError
    with pytest.raises(ImportError):
        import psycopg  # noqa: F401


def test_the_app_reads_the_setting_from_the_environment_and_https_needs_no_driver(monkeypatch, no_psycopg, endpoint):
    endpoint.answer = lambda body, headers: (200, reply(body, lambda q: table([("id", 25)], [["team"]])))
    monkeypatch.setenv("NEON_DATABASE_URL", FAKE_URL)
    monkeypatch.setenv("ATLAS_NEON_TRANSPORT", "https")
    monkeypatch.setattr(neon_wire, "endpoint", lambda url: endpoint.url + "/sql")
    monkeypatch.setattr(neon, "TRANSPORT", NeonTransport())
    db = NeonDb()
    assert db.run(lambda cur: cur.execute("SELECT id FROM circles WHERE id = %s", ("team",)).fetchall()) == [("team",)]
    assert endpoint.requests[0][1] == {"query": "SELECT id FROM circles WHERE id = $1", "params": ["team"]}
    assert neon.TRANSPORT.chosen == "https"

    monkeypatch.setenv("ATLAS_NEON_TRANSPORT", "auto")  # no driver: auto ends up on https as well
    monkeypatch.setattr(neon, "TRANSPORT", NeonTransport())
    assert NeonDb().run(lambda cur: cur.execute("SELECT 1").fetchall()) == [("team",)] and neon.TRANSPORT.chosen == "https"


# --- the connection string stays out of everything a person or a log sees ---


def test_the_connection_string_is_in_no_log_error_or_repr(endpoint, caplog):
    said = []

    def note(fn):
        try:
            fn()
        except Exception as e:
            said.append("".join(traceback.format_exception(e)) + repr(e) + str(e))

    with caplog.at_level(logging.DEBUG):
        conn = endpoint.connect()
        cur = conn.cursor()
        cur.execute("SELECT %s", (FAKE_URL,))  # even as a parameter that is waiting
        said.extend([repr(conn), str(conn), repr(cur), repr(conn.__dict__.keys())])
        conn.rollback()
        # the server's own wording names the user, the database and the endpoint: none of it is passed on
        for message in ("password authentication failed for user 'box_owner'",
                        'database "circledb" does not exist',
                        "Couldn't connect to compute node ep-fake-name-123456 at ep-fake-name-123456-pooler.c-9.us-east-9.aws.neon.invalid",
                        f'syntax error at or near "{FAKE_URL}"',
                        'permission denied for table presence for role "box_owner"'):
            endpoint.answer = lambda b, h, m=message: (400, {"message": m, "code": "42501" if "permission" in m or "syntax" in m else None})
            note(lambda: endpoint.connect().cursor().execute("SELECT 1").fetchall())
            note(lambda: NeonDb(endpoint.connect).run(lambda c: c.execute("SELECT 1")))
        endpoint.answer = lambda b, h: (503, FAKE_URL.encode())
        note(lambda: NeonDb(endpoint.connect).run(lambda c: c.execute("SELECT 1")))
        refused = HttpsConnection(FAKE_URL, timeout=2, endpoint=f"http://127.0.0.1:{closed_port()}/sql")
        note(lambda: NeonDb(lambda: refused).run(lambda c: c.execute("SELECT 1")))
        db = NeonDb(lambda: NeonTransport(tcp=refuses, https=lambda url: refused).connect(FAKE_URL, "auto"))
        note(lambda: db.run(lambda c: c.execute("SELECT 1")))
        said.extend([repr(db), str(db.offline)])
    assert len(said) >= 16 and clean("\n".join(said)) and clean(caplog.text)
    assert "permission denied for table presence" in "\n".join(said)  # what is safe to say is still said


def test_a_connection_string_without_a_host_is_offline_not_a_crash():
    db = NeonDb(lambda: NeonTransport(tcp=refuses).connect("postgresql:///nothing", "https"))
    with pytest.raises(CircleOffline):
        db.run(lambda cur: cur.execute("SELECT 1"))


# --- python -m contexts.circle.schema over https ---


class FakeDatabase:
    """Answers the schema tool's own questions for a database that holds `tables` (name -> columns), every index
    and view of schema.sql unless `missing`, and two members in one circle."""

    def __init__(self, missing: tuple[str, ...] = ()):
        want = schema.expected(schema.SCHEMA.read_text())
        self.tables = {t: cols for t, cols in want["tables"].items() if t not in missing}
        self.indexes = {i: t for i, t in want["indexes"].items() if t in self.tables and i not in missing}
        self.views = [v for v in want["views"] if v not in missing]
        self.created: list[str] = []

    def __call__(self, body: dict, headers: dict):
        if any(q["query"].startswith("CREATE") for q in body.get("queries", [body])) and headers.get("neon-batch-read-only"):
            return 400, {"message": "cannot execute CREATE TABLE in a read-only transaction", "code": "25006"}
        return 200, reply(body, self.one)

    def one(self, q: dict) -> dict:
        sql = q["query"]
        text = lambda *rows: table([("x", 25)], [list(r) for r in rows])  # noqa: E731
        if sql.startswith("CREATE"):
            self.created.append(" ".join(sql.split()[:6]))
            return {"command": "CREATE", "rowCount": None, "rows": [], "fields": [], "rowAsArray": True}
        if sql == "SHOW server_version":
            return text(["17.5"])
        if "pg_database_size" in sql:
            return text(["31 MB"])
        if "extract(epoch" in sql:
            assert float(q["params"][0]) > 1e9
            return table([("?column?", 1700)], [["0.4"]])
        if "information_schema.tables" in sql or "pg_class" in sql:
            names = sorted(self.tables) + (sorted({*self.indexes, *self.views}) if "pg_class" in sql else [])
            return text(*[[n] for n in names])
        if "information_schema.columns" in sql:
            return table([("t", 25), ("c", 25), ("k", 25), ("n", 25)],
                         [[t, c, "text", "NO"] for t, cols in sorted(self.tables.items()) for c in cols])
        if "pg_indexes" in sql:
            return table([("i", 25), ("t", 25)], [[i, t] for i, t in self.indexes.items()])
        if "information_schema.views" in sql:
            return text(*[[v] for v in self.views])
        if "pg_total_relation_size" in sql:
            assert q["params"][0].startswith('public."') and "$1::regclass" in sql
            return text(["16 kB"])
        if sql.startswith("SELECT count(*)"):
            return table([("count", 20)], [["2" if '"members"' in sql else "0"]])
        if sql.startswith("SELECT circle_id, count(*)"):
            return table([("circle_id", 25), ("count", 20)], [["team-secret-circle", "2"]])
        if sql.startswith("SELECT min("):
            return table([("min", 1184), ("max", 1184)], [["2026-10-01 12:00:00.123456+00", "2026-10-10 23:00:00+00"]])
        raise AssertionError(f"the fake database has no answer for: {sql}")


@pytest.fixture
def schema_over_https(monkeypatch, endpoint, no_psycopg):
    """The schema tool pointed at the local endpoint, with the driver not installed."""
    monkeypatch.setenv("NEON_DATABASE_URL", FAKE_URL)
    monkeypatch.setenv("ATLAS_NEON_TRANSPORT", "https")
    monkeypatch.setattr(neon_wire, "endpoint", lambda url: endpoint.url + "/sql")
    endpoint.answer = FakeDatabase()
    return endpoint


def test_schema_check_and_counts_run_over_https_read_only_without_the_driver(schema_over_https, capsys):
    assert schema.main(["--check"]) == 0
    said = capsys.readouterr()
    assert said.out == "transport: https (port 443)\nagainst schema.sql: matches\n" and clean(said.out + said.err)
    assert all(h["neon-batch-read-only"] == "true" and len(b["queries"]) == 1 for h, b in schema_over_https.requests)

    assert schema.main(["--counts"]) == 0
    said = capsys.readouterr()
    lines = said.out.splitlines()
    assert lines[0] == "transport: https (port 443)" and lines[-1] == "against schema.sql: matches"
    assert lines[1].startswith("server 17.5, database size 31 MB, database clock minus this machine's: +0.4 s")
    assert "members: 2 rows, 16 kB" in lines and "  per circle: te…(18)=2" in lines
    assert "  joined_at: 2026-10-01 12:00:00 .. 2026-10-10 23:00:00" in lines  # timestamps print as over tcp
    assert "team-secret-circle" not in said.out and clean(said.out + said.err)


def test_schema_check_over_https_reports_what_is_missing(schema_over_https, capsys):
    schema_over_https.answer = FakeDatabase(missing=("sealed_messages", "presence_circle_expiry", "habit_leaderboard"))
    assert schema.main(["--check"]) == 1
    out = capsys.readouterr().out
    assert "against schema.sql: 4 difference(s)" in out and "table sealed_messages is in schema.sql but not in the database" in out
    assert not schema_over_https.answer.created


def test_schema_apply_over_https_creates_what_is_missing_in_one_transaction(schema_over_https, capsys):
    db = schema_over_https.answer = FakeDatabase(missing=("sealed_messages", "presence_circle_expiry"))
    schema.main(["--apply"])
    out = capsys.readouterr().out
    assert "created: index presence_circle_expiry, table sealed_messages, index sealed_messages_inbox" in out
    looked, created = schema_over_https.requests[0], schema_over_https.requests[1]
    assert "query" in looked[1] and "pg_class" in looked[1]["query"]  # the read: its own request
    assert [q["query"].split()[1] for q in created[1]["queries"]] == ["INDEX", "TABLE", "INDEX"]  # then one transaction
    assert all(q["params"] == [] for q in created[1]["queries"]) and "neon-batch-read-only" not in created[0]
    assert len(db.created) == 3 and all("neon-batch-read-only" in h for h, _ in schema_over_https.requests[2:])


@pytest.mark.parametrize("transport", ["https", "auto", "tcp"])
def test_the_schema_tool_that_cannot_connect_says_what_kind_of_failure_never_where(monkeypatch, capsys, no_psycopg, transport):
    monkeypatch.setenv("NEON_DATABASE_URL", FAKE_URL)
    monkeypatch.setenv("ATLAS_NEON_TRANSPORT", transport)
    monkeypatch.setattr(neon_wire, "endpoint", lambda url: f"http://127.0.0.1:{closed_port()}/sql")
    code = schema.main(["--counts"])
    said = capsys.readouterr()
    if transport == "tcp":  # the driver is not installed and nothing else may be used
        assert code == 2 and "driver is not installed" in said.out
    else:
        assert code == 1 and "could not connect over https, port 443 (connection refused)" in said.out
    assert clean(said.out + said.err) and "127.0.0.1" not in said.out + said.err
