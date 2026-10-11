"""The wire shape of Neon's SQL-over-HTTP endpoint, and nothing else. If a live run shows a detail is wrong,
this is the one file to correct: neon_https.py builds every request and reads every reply through it.

ASSUMED, from the documented protocol of the `@neondatabase/serverless` driver (its `neon()` HTTP function).
Not checked against the real service when this was written (the tests use a local fake of this shape).

Request
    POST https://<host of the connection string>/sql                      (port 443)
    Neon-Connection-String: <the postgres:// URL, unchanged>
    Neon-Raw-Text-Output: true      every value comes back as Postgres text (or null); we parse by type id
    Neon-Array-Mode: true           each row is a JSON array in column order, not an object
    Content-Type: application/json

    one statement, committed by itself:
        {"query": "SELECT $1", "params": ["a string, or null"]}
    several statements as ONE transaction (all commit, or none do); nothing can be read back in between:
        {"queries": [{"query": "...", "params": [...]}, ...]}
        Neon-Batch-Read-Only: true          only when the caller asked for a read-only transaction

    Placeholders are $1..$n. Every parameter is sent as a string in Postgres text form (or null), the way the
    driver sends them, so the server works each type out from where the parameter is used.

Reply, HTTP 200
    one statement:   {"command": "SELECT", "rowCount": 1, "rowAsArray": true,
                      "fields": [{"name": "id", "dataTypeID": 25, ...}], "rows": [["abc"]]}
    a transaction:   {"results": [<one such object per statement, in order>]}

Refusal
    HTTP 400 with a JSON object: {"message": "...", "code": "<SQLSTATE, when Postgres raised it>", ...}
    Any other status (401, 404, 429, 5xx, a redirect) is not about the statement: the body is not relied on.

Things the driver can do that are deliberately not used: the `api.<region>` host newer drivers post to (we post to
the connection string's own host, which is the address the box's egress allowlist already names),
Neon-Batch-Isolation-Level and Neon-Batch-Deferrable (the server default, READ COMMITTED, is what psycopg uses too).
"""

import json
from dataclasses import dataclass
from urllib.parse import urlparse

PATH = "/sql"
CONNECTION_HEADER = "Neon-Connection-String"


@dataclass
class Result:
    columns: list[str]  # empty for a statement that returns no rows
    type_ids: list[int]  # Postgres type OID per column
    rows: list[list]  # raw values: Postgres text or None
    rowcount: int  # -1 when the server did not say
    command: str = ""


class Refused(Exception):
    """Anything but a readable 200. `sqlstate` is set when Postgres itself raised the error."""

    def __init__(self, status: int, message: str = "", sqlstate: str | None = None) -> None:
        super().__init__(f"HTTP {status}")
        self.status, self.message, self.sqlstate = status, message, sqlstate


def endpoint(url: str) -> str:
    """Where to POST for this connection string. Raises ValueError when the string names no host."""
    host = urlparse(url).hostname
    if not host:
        raise ValueError("the connection string names no host")
    return f"https://{host}{PATH}"


def request(url: str, statements: list[tuple[str, list]], read_only: bool = False) -> tuple[dict[str, str], bytes]:
    """Headers and body for `statements` ((sql with $n, params) pairs). More than one, or read-only: a transaction."""
    headers = {CONNECTION_HEADER: url, "Neon-Raw-Text-Output": "true", "Neon-Array-Mode": "true",
               "Content-Type": "application/json"}
    if len(statements) == 1 and not read_only:
        body = {"query": statements[0][0], "params": statements[0][1]}
    else:
        body = {"queries": [{"query": sql, "params": params} for sql, params in statements]}
        if read_only:
            headers["Neon-Batch-Read-Only"] = "true"
    return headers, json.dumps(body).encode()


def results(status: int, body: bytes, statements: int, batch: bool) -> list[Result]:
    """One Result per statement sent. Raises Refused for an error status or a reply that is not this shape."""
    try:
        reply = json.loads(body)
    except ValueError:
        reply = None
    if status != 200:
        if status == 400 and isinstance(reply, dict):
            code = reply.get("code")
            raise Refused(status, str(reply.get("message") or ""), str(code) if code else None)
        raise Refused(status)
    try:
        found = reply["results"] if batch else [reply]
        out = [_result(r) for r in found]
    except (KeyError, TypeError, ValueError):
        raise Refused(status, "the reply was not the expected JSON") from None
    if len(out) != statements:
        raise Refused(status, "the reply does not have one result per statement")
    return out


def is_batch(statements: list, read_only: bool) -> bool:
    return len(statements) != 1 or read_only


def _result(r: dict) -> Result:
    if not isinstance(r, dict) or not {"rows", "fields", "command", "rowCount"} & set(r):
        raise ValueError("not a statement's result")
    fields = r.get("fields") or []
    names = [f["name"] for f in fields]
    rows = [[row[n] for n in names] if isinstance(row, dict) else list(row) for row in r.get("rows") or []]
    count = r.get("rowCount")
    return Result(names, [int(f["dataTypeID"]) for f in fields], rows, -1 if count is None else int(count),
                  str(r.get("command") or ""))
