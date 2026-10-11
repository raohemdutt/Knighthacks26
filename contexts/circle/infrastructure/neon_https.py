"""The Circle's database over HTTPS (port 443), for networks that refuse Postgres's own port 5432.

    conn = HttpsConnection(url)            # nothing is sent yet; standard library only, no driver needed
    cur = conn.cursor()
    cur.execute("SELECT id FROM circles WHERE id = %s", ("team",))
    cur.fetchall(); conn.commit(); conn.close()

It stands in for the psycopg connection NeonDb and the schema tool use: the same `%s` placeholders, the same
Python values in and out, `rowcount`, `description`, `fetchone`, `fetchall`, `commit`, `rollback`, `read_only`.
The request and reply shapes are in neon_wire.py.

TRANSACTIONS ARE NOT THE SAME AS OVER TCP. HTTP has no open transaction to talk to, so:
- `execute` only notes the statement down. Nothing is sent until the caller needs an answer (`rowcount`,
  `description`, `fetchone`, `fetchall`) or commits.
- Whatever is noted down at that moment goes in ONE request and is ONE transaction: all of it commits or none.
- So work that only writes (make sure the circle and member rows exist, then insert) is as atomic as over TCP.
  Work that reads an answer and then writes more is two transactions: the first part is already committed when
  the second is sent, and a failure in between leaves the first part in place. `rollback` can only forget what
  has not been sent. Each such caller says in a comment what that means for it.

Failures come in the two kinds NeonDb already knows:
- NeonHttpsDropped (an OSError): the database could not be reached or would not let us in. NeonDb turns it into
  CircleOffline, so the outbox, the presence queue and the tick carry on exactly as with a dropped TCP link.
- NeonHttpsError: Postgres ran the statement and refused it (a missing table, a broken constraint).
Neither ever carries the connection string, the host or the server's own wording about them.

What is supported is what the Circle's SQL uses: None, bool, int, float, Decimal, str, date, datetime, timedelta
and lists of those going in; text, numbers, booleans, json/jsonb, date, timestamp and timestamptz coming out
(tz-aware, in UTC). Other column types come back as their Postgres text.
"""

import http.client
import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from urllib.parse import unquote, urlparse

from contexts.circle.infrastructure import neon_wire

TIMEOUT_S = 10  # to connect, and again for each read of the reply


class NeonHttpsDropped(OSError):
    """Could not reach the database, or it would not let us in. `retry`: worth one more try straight away."""

    retry = True


class NeonHttpsTimeout(NeonHttpsDropped):
    retry = False  # it already waited out the timeout once: a second wait would hold the caller up twice as long


class NeonHttpsError(Exception):
    """Postgres refused the statement. `sqlstate` is its five-character code when it gave one."""

    def __init__(self, message: str, sqlstate: str | None = None) -> None:
        super().__init__(message)
        self.sqlstate = sqlstate


# SQLSTATE classes psycopg raises OperationalError for (connection, login, resources, shutdown), plus 3D: a missing
# database fails at connect over TCP. Over HTTPS they arrive as an answer to the statement, and mean the same.
UNREACHABLE_CLASSES = {"08", "28", "3D", "40", "53", "54", "55", "57", "58"}
# A 400 without a SQLSTATE that is about getting in rather than about the statement (the proxy's own wording).
UNREACHABLE_WORDS = re.compile(r"authenticat|password|endpoint|connection|compute|quota|suspend|project", re.I)

# --- going in: %s -> $n, Python values -> Postgres text -------------------------------------------------------------

_PLACEHOLDER = re.compile(r"%(.?)", re.S)


def to_dollar(sql: str, params) -> tuple[str, list[str | None]]:
    """psycopg's placeholders as the server's: `%s` becomes `$1..$n` and `%%` a literal `%`, everywhere in the
    text, string literals included, which is exactly what psycopg does. `params=None` leaves the text alone."""
    if params is None:
        return sql, []
    if isinstance(params, (str, bytes, dict)):
        raise TypeError("query parameters must be a list or a tuple")
    values, used = list(params), 0

    def swap(m: re.Match) -> str:
        nonlocal used
        if m.group(1) == "%":
            return "%"
        if m.group(1) != "s":
            raise ValueError("only %s placeholders and %% are supported")
        used += 1
        if used > len(values):
            raise ValueError("the query has more placeholders than parameters")
        return f"${used}{_cast(values[used - 1])}"

    text = _PLACEHOLDER.sub(swap, sql)
    if used != len(values):
        raise ValueError(f"the query has {used} placeholders but {len(values)} parameters were passed")
    return text, [to_text(v) for v in values]


def _cast(value) -> str:
    """Text parameters have no type of their own. The server works it out from the column or operator, except
    where two readings are possible: `now() - $1` would take $1 for a timestamp. psycopg says interval; so do we."""
    return "::interval" if isinstance(value, timedelta) else ""


def to_text(value) -> str | None:
    """A Python value in Postgres text form."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, Decimal)):
        return str(value)
    if isinstance(value, float):
        return {"nan": "NaN", "inf": "Infinity", "-inf": "-Infinity"}.get(repr(value), repr(value))
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return f"{value.days} days {value.seconds} seconds {value.microseconds} microseconds"
    if isinstance(value, (list, tuple)):
        return "{" + ",".join(_array_item(v) for v in value) + "}"
    raise TypeError(f"cannot send a {type(value).__name__} to the database (pass JSON as json.dumps text)")


def _array_item(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, (list, tuple)):
        return to_text(value)
    return '"' + to_text(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


# --- coming out: Postgres text -> the Python values psycopg returns ---------------------------------------------------

_STAMP = re.compile(r"(\d{4})-(\d\d)-(\d\d)[ T](\d\d):(\d\d):(\d\d)(?:\.(\d{1,6}))?(?:([+-])(\d\d)(?::?(\d\d))?(?::?(\d\d))?)?$")


def _stamp(text: str) -> datetime:
    m = _STAMP.match(text)
    if m is None:
        raise ValueError(f"not a timestamp this adapter reads: {text[:40]!r}")
    y, mo, d, h, mi, s, frac, sign, oh, om, os_ = m.groups()
    out = datetime(int(y), int(mo), int(d), int(h), int(mi), int(s), int((frac or "0").ljust(6, "0")))
    if sign is None:
        return out
    offset = timedelta(hours=int(oh), minutes=int(om or 0), seconds=int(os_ or 0))
    return out.replace(tzinfo=timezone(offset if sign == "+" else -offset)).astimezone(UTC)


def _bool(text: str) -> bool:
    return text in ("t", "true")


READERS: dict[int, Callable[[str], object]] = {
    16: _bool,
    20: int, 21: int, 23: int, 26: int,  # int8, int2, int4, oid
    700: float, 701: float,
    1700: Decimal,  # numeric
    114: json.loads, 3802: json.loads,  # json, jsonb
    1082: date.fromisoformat,
    1114: _stamp, 1184: _stamp,  # timestamp (naive), timestamptz (aware, UTC)
}


def from_text(type_id: int, value):
    """One value of a reply. Text is read by its column type; anything else (null, or a server that answered in
    JSON types after all) is passed on as it is, except a numeric, which psycopg gives as a Decimal."""
    reader = READERS.get(type_id)
    if value is None or reader is None:
        return value
    if isinstance(value, str):
        return reader(value)
    return Decimal(str(value)) if reader is Decimal and not isinstance(value, bool) else value


# --- the request ------------------------------------------------------------------------------------------------------

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect is never followed: the connection string is in a header and must not go to another address."""

    def redirect_request(self, *args, **kwargs):
        return None


def send(endpoint: str, headers: dict[str, str], body: bytes, timeout: float) -> tuple[int, bytes]:
    """POST and return (status, body). An error status is returned, not raised; no network raises OSError."""
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(urllib.request.Request(endpoint, data=body, headers=headers, method="POST"), timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        try:
            return e.code, e.read()
        finally:
            e.close()


def _why(e: BaseException) -> NeonHttpsDropped:
    """A network failure as ours, described by its kind only: the original's message can name the host."""
    reason = e.reason if isinstance(e, urllib.error.URLError) and isinstance(e.reason, BaseException) else e
    if isinstance(reason, TimeoutError):
        return NeonHttpsTimeout("no answer in time")
    if isinstance(reason, ConnectionRefusedError):
        return NeonHttpsDropped("connection refused")
    return NeonHttpsDropped(f"no connection ({type(reason).__name__})")


def scrub(text: str, url: str) -> str:
    """`text` without anything the connection string holds (the server's wording can name the user or database)."""
    parts = urlparse(url)
    host = parts.hostname or ""
    secrets = {url, parts.netloc, host, host.split(".")[0], host.split(".")[0].removesuffix("-pooler"),
               parts.username or "", unquote(parts.username or ""), parts.password or "", unquote(parts.password or ""),
               parts.path.lstrip("/")}
    for secret in sorted((s for s in secrets if len(s) >= 3), key=len, reverse=True):
        text = text.replace(secret, "…")
    return text[:300]


class HttpsCursor:
    def __init__(self, conn: "HttpsConnection") -> None:
        self._conn = conn
        self._result: neon_wire.Result | None = None
        self._rows: list[tuple] = []
        self._at = 0
        self._waiting = False  # noted down, not sent yet

    def execute(self, sql: str, params=None) -> "HttpsCursor":
        self._result, self._rows, self._at = None, [], 0
        self._conn._note(self, sql, params)
        return self

    def _answer(self) -> neon_wire.Result | None:
        if self._waiting:
            self._conn.flush()  # the caller wants an answer: everything noted so far is sent, and committed, now
        return self._result

    def _set(self, result: neon_wire.Result) -> None:
        self._result, self._at = result, 0
        self._rows = [tuple(from_text(t, v) for t, v in zip(result.type_ids, row)) for row in result.rows]

    @property
    def rowcount(self) -> int:
        result = self._answer()
        return -1 if result is None else result.rowcount

    @property
    def description(self) -> list[tuple] | None:
        result = self._answer()
        if result is None or not result.columns:
            return None
        return [(name, type_id, None, None, None, None, None) for name, type_id in zip(result.columns, result.type_ids)]

    def fetchall(self) -> list[tuple]:
        self._must_have_rows()
        rows, self._at = self._rows[self._at:], len(self._rows)
        return rows

    def fetchone(self) -> tuple | None:
        self._must_have_rows()
        if self._at >= len(self._rows):
            return None
        self._at += 1
        return self._rows[self._at - 1]

    def _must_have_rows(self) -> None:
        result = self._answer()
        if result is None or not result.columns:
            raise NeonHttpsError("the last statement did not return rows")

    def close(self) -> None:
        pass


class HttpsConnection:
    """What NeonDb and the schema tool need of a connection. `stateless`: there is no link to keep or to lose."""

    stateless = True

    def __init__(self, url: str, timeout: float = TIMEOUT_S, post: Callable = send, endpoint: str | None = None) -> None:
        self._url, self._timeout, self._post = url, timeout, post
        try:
            self._endpoint = endpoint or neon_wire.endpoint(url)
        except ValueError as e:
            raise NeonHttpsDropped(str(e)) from None
        self._noted: list[tuple[HttpsCursor, str, object]] = []
        self.read_only = False  # True: every request is a read-only transaction, enforced by the server

    def __repr__(self) -> str:  # never the connection string
        return f"<HttpsConnection {len(self._noted)} statement(s) waiting>"

    def cursor(self) -> HttpsCursor:
        return HttpsCursor(self)

    def _note(self, cur: HttpsCursor, sql: str, params) -> None:
        self._noted.append((cur, sql, params))
        cur._waiting = True

    def flush(self) -> None:
        """Send what is noted down as one request: one statement, or one transaction of several."""
        noted, self._noted = self._noted, []
        for cur, _, _ in noted:
            cur._waiting = False
        if not noted:
            return
        statements = [to_dollar(sql, params) for _, sql, params in noted]
        headers, body = neon_wire.request(self._url, statements, self.read_only)
        try:
            status, reply = self._post(self._endpoint, headers, body, self._timeout)
        except NeonHttpsDropped:
            raise
        except OSError as e:
            raise _why(e) from None
        except http.client.HTTPException as e:  # a reply cut short, a line that is not HTTP
            raise NeonHttpsDropped(f"no connection ({type(e).__name__})") from None
        try:
            results = neon_wire.results(status, reply, len(statements), neon_wire.is_batch(statements, self.read_only))
        except neon_wire.Refused as e:
            raise self._refusal(e) from None
        for (cur, _, _), result in zip(noted, results):
            cur._set(result)

    def commit(self) -> None:
        self.flush()

    def _refusal(self, e: neon_wire.Refused) -> Exception:
        about_statement = e.status == 400 and (
            e.sqlstate[:2] not in UNREACHABLE_CLASSES if e.sqlstate else not UNREACHABLE_WORDS.search(e.message))
        if about_statement:
            return NeonHttpsError(scrub(e.message, self._url) or "the database refused the statement", e.sqlstate)
        if e.status == 200:
            return NeonHttpsDropped("the answer was not from the database")  # a sign-in page, a proxy's own reply
        return NeonHttpsDropped(f"HTTP {e.status}" + (f", SQLSTATE {e.sqlstate}" if e.sqlstate else ""))

    def rollback(self) -> None:
        """Forget what has not been sent. What was sent is committed already and stays."""
        for cur, _, _ in self._noted:
            cur._waiting = False
        self._noted = []

    def close(self) -> None:
        self.rollback()
