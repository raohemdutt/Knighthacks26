"""Helpers for the Neon tests: a fake connection for unit tests, and a throwaway
`test-…` circle for the tests that run against the real database.

The fake speaks both transports. As `tcp` it stands in for the psycopg connection. As `https` it hands out the real
HttpsConnection (neon_https.py) with a fake endpoint behind it, so the statements go through the placeholder
conversion, the wire shape (neon_wire.py) and the reply parsing on their way. conftest.py runs every test that
uses FakeNeon once per transport."""

import json
import os
import re
from datetime import UTC, date, datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from contexts.circle.infrastructure.neon import ENV_URL, NeonDb
from contexts.circle.infrastructure.neon_https import HttpsConnection

TRANSPORT = "tcp"  # what FakeNeon() speaks unless told; conftest.py's neon_transport fixture sets it per test
# Not a real database. The tests check that none of these parts ever shows up in a log, an error or a repr.
FAKE_URL = ("postgresql://box_owner:n0t-a-real-s3cret@ep-fake-name-123456-pooler.c-9.us-east-9.aws.neon.invalid"
            "/circledb?sslmode=require")
FAKE_URL_PARTS = ("n0t-a-real-s3cret", "box_owner", "ep-fake-name-123456", "neon.invalid", "circledb")

needs_neon = pytest.mark.skipif(not os.environ.get(ENV_URL), reason=f"{ENV_URL} is not set")

OWNED_TABLES = ["members", "shares", "agent_messages", "events", "habit_summaries", "impact_counters",
                "game_scores", "presence", "league_stakes", "league_weeks", "league_challenges", "sealed_messages"]


class FakeNeon:
    """Stands in for the driver: records statements, returns canned rows, can drop or refuse connections."""

    def __init__(self, rows=(), columns=(), transport: str | None = None):
        self.rows, self.columns = list(rows), list(columns)
        self.statements: list[tuple[str, tuple]] = []
        self.connects = self.commits = self.drops = 0
        self.down = False
        self.transport = transport or TRANSPORT
        self.requests: list[tuple[dict, dict]] = []  # https only: (headers, body) of each request that got through
        self.lacks: set[str] = set()  # "table.column": a statement naming both is refused, as by a database without it
        self.refused = 0  # statements refused that way

    def refuses(self, sql: str) -> str | None:
        """Postgres's message for a statement that names a column this database does not have, else None."""
        for table, column in (name.split(".") for name in self.lacks):
            if re.search(rf"\b{table}\b", sql) and re.search(rf"\b{column}\b", sql):
                self.refused += 1
                return f'column "{column}" of relation "{table}" does not exist'
        return None

    def connect(self):
        if self.transport == "https":
            return _FakeHttps(self)  # no network at connect: `down` and `drops` show at the first request
        if self.down:
            raise ConnectionError("no route to host")
        self.connects += 1
        return _FakeConnection(self)

    def tables(self) -> list[str]:
        """Tables written to, in order."""
        return [sql.split()[2] for sql, _ in self.statements if sql.startswith("INSERT INTO")]


class UndefinedColumn(Exception):
    """What the driver raises for a column that is not there (psycopg.errors.UndefinedColumn has the same code)."""

    sqlstate = "42703"


class _FakeConnection:
    rowcount = 1

    def __init__(self, neon: FakeNeon):
        self.neon = neon
        self._committed = len(neon.statements)

    def cursor(self):
        return self

    def execute(self, sql, params=()):
        if self.neon.drops:
            self.neon.drops -= 1
            raise ConnectionError("server closed the connection unexpectedly")
        why = self.neon.refuses(sql)
        if why:
            raise UndefinedColumn(why)
        self.neon.statements.append((sql, tuple(params)))

    @property
    def description(self):
        return [(c,) for c in self.neon.columns]

    def fetchall(self):
        return self.neon.rows

    def commit(self):
        self.neon.commits += 1
        self._committed = len(self.neon.statements)

    def rollback(self):
        del self.neon.statements[self._committed:]  # as over https: `statements` holds what was committed, or is about to be

    def close(self):
        pass


class _FakeHttps(HttpsConnection):
    """The real HTTPS connection with a fake endpoint. `statements` gets what the caller passed (so the same
    assertions hold on both transports) once the request carrying it got through; `connects` counts the
    connections that reached the endpoint."""

    def __init__(self, neon: FakeNeon):
        super().__init__(FAKE_URL, post=self._endpoint_answers)
        self.neon, self._reached = neon, False

    def flush(self):
        sent = [(sql, tuple(params or ())) for _, sql, params in self._noted]
        super().flush()
        self.neon.statements += sent

    def commit(self):
        self.flush()
        self.neon.commits += 1

    def _endpoint_answers(self, endpoint, headers, body, timeout):
        neon = self.neon
        if neon.down:
            raise ConnectionRefusedError("no route to host")
        if neon.drops:
            neon.drops -= 1
            raise ConnectionResetError("server closed the connection unexpectedly")
        if not self._reached:
            self._reached = True
            neon.connects += 1
        request = json.loads(body)
        neon.requests.append((dict(headers), request))
        assert endpoint.startswith("https://") and headers["Neon-Connection-String"] == FAKE_URL
        queries = request["queries"] if "queries" in request else [request]
        for q in queries:  # what a real server would refuse
            assert "%s" not in q["query"] and all(p is None or isinstance(p, str) for p in q["params"])
            assert {int(n) for n in re.findall(r"\$(\d+)", q["query"])} == set(range(1, len(q["params"]) + 1))
        for q in queries:  # one refused statement refuses the whole request: nothing of it is kept
            why = neon.refuses(q["query"])
            if why:
                return 400, json.dumps({"message": why, "code": "42703"}).encode()
        answers = [wire_result(neon.rows, neon.columns, q["query"]) for q in queries]
        return 200, json.dumps({"results": answers} if "queries" in request else answers[0]).encode()


def wire_text(value) -> str | None:
    """A Python value the way Postgres prints it (what the endpoint sends with Neon-Raw-Text-Output)."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "t" if value else "f"
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.isoformat(sep=" ")
        return value.astimezone(UTC).isoformat(sep=" ").replace("+00:00", "+00")
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


TYPE_IDS = [(bool, 16), (int, 20), (float, 701), (Decimal, 1700), (datetime, 1184), (date, 1082), (dict, 3802), (list, 3802)]


def wire_type(values) -> int:
    """The column's Postgres type id, from the Python values in it. Text unless a value says otherwise (a json
    column may hold text in one row and a dict in another: psycopg hands back either)."""
    found = {next((oid for kind, oid in TYPE_IDS if isinstance(v, kind)), 25) for v in values if v is not None}
    if 3802 in found:
        return 3802
    if any(isinstance(v, datetime) and v.tzinfo is None for v in values):
        return 1114
    return found.pop() if len(found) == 1 else 25


def wire_result(rows, columns, query: str = "SELECT") -> dict:
    """One statement's reply in the endpoint's shape (neon_wire.py), for canned Python rows."""
    command = query.split()[0].upper()
    if command not in ("SELECT", "SHOW") and " RETURNING " not in query.upper():
        return {"command": command, "rowCount": 1, "rows": [], "fields": [], "rowAsArray": True}
    width = len(rows[0]) if rows else len(columns)
    names = list(columns) if len(columns) == width else [f"c{i}" for i in range(width)]
    types = [wire_type([r[i] for r in rows]) for i in range(width)]
    return {"command": command, "rowCount": 1, "rowAsArray": True,
            "fields": [{"name": n, "dataTypeID": t} for n, t in zip(names, types)] or [{"name": "?column?", "dataTypeID": 25}],
            "rows": [[wire_text(v) for v in r] for r in rows]}


def open_test_circle() -> tuple[NeonDb, str]:
    return NeonDb(), f"test-{uuid4().hex[:12]}"


def drop_test_circle(db: NeonDb, circle_id: str) -> int:
    """Delete the throwaway circle (everything in it cascades) and return how many of its rows are left."""

    def work(cur):
        cur.execute("DELETE FROM circles WHERE id = %s", (circle_id,))
        left = 0
        for table in OWNED_TABLES:
            cur.execute(f"SELECT count(*) FROM {table} WHERE circle_id = %s", (circle_id,))
            left += cur.fetchall()[0][0]
        return left

    try:
        return db.run(work)
    finally:
        db.close()
