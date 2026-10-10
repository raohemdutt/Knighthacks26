"""An in-memory Notion for the GPU queue's tests and rehearsals: the wire, not the logic.

FakeNotion stands where api.notion.com stands. NotionClient(token, transport=fake.transport)
sends it the same requests it would send Notion and gets back answers in the shapes Notion
documents, so everything above the wire runs for real: the request building, the retries, the
property encoding and decoding, the queue rules and the verbs. Nothing here knows the queue
rules; it only keeps pages and checks what Notion checks.

What it answers (a page lives in one data source, the GPU Queue's):

    POST  /v1/data_sources/<id>/query    select-equals filters, and / or groups, page_size and cursors
    GET   /v1/pages/<id>                 one page, or 404 object_not_found
    POST  /v1/pages                      a page whose parent is the data source
    PATCH /v1/pages/<id>                 changed properties only; the rest keep their values

What it checks, answering 400 validation_error like Notion does: the bearer token (401 otherwise),
the Notion-Version header (2025-09-03 or later), that every property named exists in SCHEMA, that
each value has the shape of its property's type, that no text is longer than Notion's 2000
characters, and that a date-time carries a UTC offset. It is stricter than Notion in one way that
protects the real database: a select value that is not an existing option is refused, where Notion
would quietly add the option.

A test can also make it misbehave:

    fake.fail_next(2, status=429, retry_after="1")    the next two requests are refused that way
    fake.on_request(hook)                             hook(fake, method, path, body) runs before every
                                                      request is handled: another writer getting in first
    fake.requests                                     every request received, in order, as
                                                      (method, path, body) with the token never kept
    fake.serve()                                      a loopback HTTP server over the same logic, for the
                                                      tests of the real urllib transport

Rows are seeded with add_row() and read back with values() in plain Python, so a test checks what
Notion holds without going through the client it is testing.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import re
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

__all__ = ["DATA_SOURCE_ID", "DATABASE_ID", "SCHEMA", "TOKEN", "FakeNotion"]

DATA_SOURCE_ID = "dc46dcb7-79d4-4727-bc04-40426b5d9114"
DATABASE_ID = "c69762a3-9687-4e04-9a4a-f6819f762615"
TOKEN = "fake-notion-token-0123456789abcdef"
MIN_VERSION = "2025-09-03"
MAX_TEXT = 2000
MAX_PAGE_SIZE = 100

#: The GPU Queue database's columns, as the live data source reports them.
SCHEMA: dict[str, dict[str, Any]] = {
    "Request": {"type": "title"},
    "Task ID": {"type": "rich_text"},
    "GPU": {"type": "select", "options": ("A teacher shared service", "B exclusive")},
    "Minutes": {"type": "number"},
    "Status": {"type": "select", "options": ("Queued", "Running", "Done", "Cancelled", "Ended by referee")},
    "Agent": {"type": "rich_text"},
    "Requested": {"type": "date"},
    "Started": {"type": "date"},
    "Ended": {"type": "date"},
    "Extended": {"type": "checkbox"},
    "Progress note": {"type": "rich_text"},
    "Complaint": {"type": "rich_text"},
    "Result": {"type": "rich_text"},
}

_ANNOTATIONS = {"bold": False, "italic": False, "strikethrough": False, "underline": False, "code": False,
                "color": "default"}


class _Refusal(Exception):
    def __init__(self, status: int, code: str, message: str, headers: Mapping[str, str] | None = None) -> None:
        super().__init__(message)
        self.status, self.code, self.message, self.headers = status, code, message, dict(headers or {})


def _stamp(moment: dt.datetime) -> str:
    """A time as Notion returns created_time: UTC to the millisecond, ending in Z."""
    return moment.astimezone(dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _parse_start(text: Any, name: str) -> str:
    """A date value's start as Notion stores it: a date as is, a date-time to the millisecond with its offset."""
    if not isinstance(text, str):
        raise _Refusal(400, "validation_error", f"body.properties.{name}.date.start should be a string")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        try:
            dt.date.fromisoformat(text)
        except ValueError:
            raise _Refusal(400, "validation_error", f"{name}: {text!r} is not a valid date") from None
        return text
    try:
        moment = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        raise _Refusal(400, "validation_error", f"{name}: {text!r} is not an ISO 8601 date or date-time") from None
    if moment.utcoffset() is None:
        raise _Refusal(400, "validation_error", f"{name}: a date-time needs a UTC offset or a time_zone")
    return moment.isoformat(timespec="milliseconds")


class FakeNotion:
    """See the module docstring."""

    def __init__(self, *, token: str = TOKEN, data_source_id: str = DATA_SOURCE_ID,
                 clock: Callable[[], dt.datetime] | None = None, page_limit: int = MAX_PAGE_SIZE) -> None:
        self.token = token
        self.data_source_id = data_source_id
        self.page_limit = min(page_limit, MAX_PAGE_SIZE)
        self._clock = clock or (lambda: dt.datetime.now(dt.timezone.utc))
        self._pages: dict[str, dict[str, Any]] = {}
        self._order: list[str] = []
        self._counter = 0
        self._failures: list[tuple[int, str, dict[str, str]]] = []
        self._hooks: list[Callable[[FakeNotion, str, str, Any], None]] = []
        self.requests: list[tuple[str, str, Any]] = []
        self._lock = threading.RLock()

    # --- seeding and reading, in plain Python ---------------------------------------------

    def add_row(self, request: str, *, task_id: str = "", gpu: str = "B exclusive", minutes: float | None = 15,
                status: str | None = "Queued", agent: str = "Ilan", requested: dt.datetime | None = None,
                started: dt.datetime | None = None, ended: dt.datetime | None = None, extended: bool = False,
                progress_note: str = "", complaint: str = "", result: str = "",
                created: dt.datetime | None = None) -> str:
        """Put a row straight into the data source, as a person or another tool would have; the row's 32-digit id."""
        def when(moment: dt.datetime | None) -> dict[str, Any] | None:
            return None if moment is None else {"start": _parse_start(moment.isoformat(), "seed")}

        values = {"Request": request, "Task ID": task_id, "GPU": gpu, "Minutes": minutes, "Status": status,
                  "Agent": agent, "Requested": when(requested), "Started": when(started), "Ended": when(ended),
                  "Extended": extended, "Progress note": progress_note, "Complaint": complaint, "Result": result}
        with self._lock:
            page = self._new_page(created or self._clock())
            for name, value in values.items():
                kind = SCHEMA[name]["type"]
                if kind in ("title", "rich_text"):
                    wire: Any = {kind: [{"text": {"content": value}}]} if value else {kind: []}
                elif kind == "number":
                    wire = {"number": value}
                elif kind == "select":
                    wire = {"select": None if value is None else {"name": value}}
                elif kind == "checkbox":
                    wire = {"checkbox": value}
                else:
                    wire = {"date": value}
                page["properties"][name] = self._store(name, wire)
            return page["id"].replace("-", "")

    def values(self, row_id: str) -> dict[str, Any]:
        """The row's columns as plain Python: text, number, option name, bool, date text or None."""
        page = self._pages[self._dashed(row_id)]
        out: dict[str, Any] = {}
        for name, prop in page["properties"].items():
            kind = prop["type"]
            if kind in ("title", "rich_text"):
                out[name] = "".join(item["plain_text"] for item in prop[kind])
            elif kind == "select":
                out[name] = prop["select"]["name"] if prop["select"] else None
            elif kind == "date":
                out[name] = prop["date"]["start"] if prop["date"] else None
            else:
                out[name] = prop[kind]
        return out

    def row_ids(self) -> list[str]:
        """Every row's 32-digit id, oldest first."""
        return [page_id.replace("-", "") for page_id in self._order]

    def edits(self) -> list[tuple[str, str, Any]]:
        """The requests that changed something: (method, path, body) of every POST /v1/pages and PATCH."""
        return [r for r in self.requests if r[0] == "PATCH" or (r[0] == "POST" and r[1] == "/v1/pages")]

    # --- misbehaving on purpose ------------------------------------------------------------

    def fail_next(self, count: int = 1, *, status: int = 429, code: str = "rate_limited",
                  retry_after: str | None = None) -> None:
        """Refuse the next count requests with this status and code (a Retry-After header when given)."""
        headers = {"Retry-After": retry_after} if retry_after is not None else {}
        self._failures.extend([(status, code, headers)] * count)

    def on_request(self, hook: Callable[[FakeNotion, str, str, Any], None]) -> None:
        """Run hook(fake, method, path, body) before each request is handled, so a test can make another
        writer get in first. A hook runs once per request and its own writes call add_row or the page methods."""
        self._hooks.append(hook)

    # --- the wire --------------------------------------------------------------------------

    def transport(self, method: str, url: str, headers: Mapping[str, str], body: bytes | None,
                  timeout: float) -> tuple[int, bytes, Mapping[str, str]]:
        """The Transport of a NotionClient: (status, body bytes, headers)."""
        path = "/" + url.split("://", 1)[-1].split("/", 1)[-1] if "://" in url else url
        try:
            parsed = json.loads(body) if body else None
        except ValueError:
            parsed = None
        with self._lock:
            self.requests.append((method, path, parsed))
            for hook in list(self._hooks):
                hook(self, method, path, parsed)
            try:
                if self._failures:
                    status, code, extra = self._failures.pop(0)
                    raise _Refusal(status, code, f"simulated {code}", extra)
                result = self._handle(method, path, {k.lower(): v for k, v in headers.items()}, parsed, body)
                return 200, json.dumps(result).encode("utf-8"), {"Content-Type": "application/json"}
            except _Refusal as refusal:
                answer = {"object": "error", "status": refusal.status, "code": refusal.code,
                          "message": refusal.message}
                return refusal.status, json.dumps(answer).encode("utf-8"), {"Content-Type": "application/json",
                                                                              **refusal.headers}

    @contextlib.contextmanager
    def serve(self) -> Iterator[str]:
        """A loopback HTTP server over the same logic; yields its base URL (http://127.0.0.1:<port>)."""
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else None
                status, payload, headers = fake.transport(self.command, self.path, dict(self.headers), raw, 5.0)
                self.send_response(status)
                for name, value in headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            do_GET = do_POST = do_PATCH = _answer

            def log_message(self, *args: Any) -> None:  # keep the test output clean
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}"
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    # --- what Notion does ------------------------------------------------------------------

    def _handle(self, method: str, path: str, headers: Mapping[str, str], body: Any, raw: bytes | None) -> Any:
        if headers.get("authorization") != f"Bearer {self.token}":
            raise _Refusal(401, "unauthorized", "API token is invalid.")
        version = headers.get("notion-version")
        if not version or version < MIN_VERSION:
            raise _Refusal(400, "validation_error", f"Notion-Version {version!r} does not have data sources; "
                                                    f"send {MIN_VERSION} or later")
        if raw and body is None:
            raise _Refusal(400, "invalid_json", "Error parsing JSON body.")
        query = re.fullmatch(r"/v1/data_sources/([0-9a-fA-F-]+)/query", path)
        if method == "POST" and query:
            return self._query(query.group(1), body or {})
        page = re.fullmatch(r"/v1/pages/([0-9a-fA-F-]+)", path)
        if method == "GET" and page:
            return self._page(self._existing(page.group(1)))
        if method == "PATCH" and page:
            return self._update(self._existing(page.group(1)), body or {})
        if method == "POST" and path == "/v1/pages":
            return self._create(body or {})
        raise _Refusal(404, "invalid_request_url", f"Invalid request URL: {method} {path}")

    def _dashed(self, row_id: str) -> str:
        plain = row_id.replace("-", "").lower()
        return f"{plain[:8]}-{plain[8:12]}-{plain[12:16]}-{plain[16:20]}-{plain[20:]}"

    def _existing(self, page_id: str) -> dict[str, Any]:
        found = self._pages.get(self._dashed(page_id))
        if found is None:
            raise _Refusal(404, "object_not_found", f"Could not find page with ID: {self._dashed(page_id)}. "
                                                    "Make sure the relevant pages and databases are shared with your integration.")
        return found

    def _new_page(self, created: dt.datetime) -> dict[str, Any]:
        self._counter += 1
        page_id = f"{0xFA4E0000 + self._counter:08x}-0000-4000-8000-{self._counter:012x}"
        page = {"object": "page", "id": page_id, "created_time": _stamp(created), "last_edited_time": _stamp(created),
                "in_trash": False, "is_archived": False, "is_locked": False,
                "url": f"https://www.notion.so/{page_id.replace('-', '')}", "public_url": None,
                "parent": {"type": "data_source_id", "data_source_id": self.data_source_id,
                           "database_id": DATABASE_ID},
                "properties": {}}
        self._pages[page_id] = page
        self._order.append(page_id)
        return page

    @staticmethod
    def _page(page: dict[str, Any]) -> dict[str, Any]:
        return json.loads(json.dumps(page))

    def _create(self, body: Mapping[str, Any]) -> dict[str, Any]:
        parent = body.get("parent")
        if not isinstance(parent, Mapping) or parent.get("type") != "data_source_id" \
                or parent.get("data_source_id") != self.data_source_id:
            raise _Refusal(404, "object_not_found", "Could not find the data source this page is meant for.")
        properties = self._validated(body.get("properties"))
        page = self._new_page(self._clock())
        page["properties"] = {name: self._blank(name) for name in SCHEMA}
        page["properties"].update(properties)
        return self._page(page)

    def _update(self, page: dict[str, Any], body: Mapping[str, Any]) -> dict[str, Any]:
        unknown = sorted(set(body) - {"properties"})
        if unknown:
            raise _Refusal(400, "validation_error", f"body failed validation: unknown key {unknown[0]!r}")
        page["properties"].update(self._validated(body.get("properties")))
        page["last_edited_time"] = _stamp(self._clock())
        return self._page(page)

    def _validated(self, properties: Any) -> dict[str, Any]:
        if not isinstance(properties, Mapping):
            raise _Refusal(400, "validation_error", "body.properties should be an object")
        stored: dict[str, Any] = {}
        for name, value in properties.items():
            if name not in SCHEMA:
                raise _Refusal(400, "validation_error", f"{name} is not a property that exists.")
            stored[name] = self._store(name, value)
        return stored

    def _blank(self, name: str) -> dict[str, Any]:
        """The value of a column nobody has written: no text, no number, no option, no date, unchecked."""
        kind = SCHEMA[name]["type"]
        return self._store(name, {kind: [] if kind in ("title", "rich_text") else (False if kind == "checkbox" else None)})

    def _store(self, name: str, value: Any) -> dict[str, Any]:
        """A written property value checked against the column's type and turned into what a read returns."""
        kind = SCHEMA[name]["type"]
        bad = _Refusal(400, "validation_error", f"body.properties.{name} should be a {kind} value, instead got "
                                                f"{json.dumps(value)[:80]}.")
        if not isinstance(value, Mapping) or set(value) != {kind}:
            raise bad
        prop: dict[str, Any] = {"id": re.sub(r"\W", "", name)[:4].lower() or "x", "type": kind}
        raw = value[kind]
        if kind in ("title", "rich_text"):
            if not isinstance(raw, list):
                raise bad
            items = []
            for item in raw:
                content = item.get("text", {}).get("content") if isinstance(item, Mapping) else None
                if not isinstance(content, str):
                    raise bad
                if len(content) > MAX_TEXT:
                    raise _Refusal(400, "validation_error", f"body.properties.{name}[0].text.content.length should "
                                                            f"be <= {MAX_TEXT}, instead was {len(content)}.")
                items.append({"type": "text", "text": {"content": content, "link": None},
                              "annotations": dict(_ANNOTATIONS), "plain_text": content, "href": None})
            prop[kind] = items
        elif kind == "number":
            if raw is not None and (isinstance(raw, bool) or not isinstance(raw, (int, float))):
                raise bad
            prop[kind] = raw
        elif kind == "select":
            if raw is None:
                prop[kind] = None
            else:
                option = raw.get("name") if isinstance(raw, Mapping) else None
                if option not in SCHEMA[name]["options"]:
                    raise _Refusal(400, "validation_error", f"{name}: {option!r} is not an existing select option "
                                                            "(Notion would add it; this fake refuses so a typo is caught)")
                prop[kind] = {"id": uuid.uuid5(uuid.NAMESPACE_URL, f"{name}/{option}").hex, "name": option,
                              "color": "default"}
        elif kind == "checkbox":
            if not isinstance(raw, bool):
                raise bad
            prop[kind] = raw
        else:
            if raw is None:
                prop[kind] = None
            else:
                if not isinstance(raw, Mapping) or "start" not in raw:
                    raise bad
                if set(raw) - {"start", "end", "time_zone"}:
                    raise bad
                if raw.get("time_zone") is not None:
                    raise _Refusal(400, "validation_error", "the fake takes UTC offsets, not time_zone names")
                prop[kind] = {"start": _parse_start(raw["start"], name),
                              "end": None if raw.get("end") is None else _parse_start(raw["end"], name),
                              "time_zone": None}
        return prop

    def _query(self, data_source_id: str, body: Mapping[str, Any]) -> dict[str, Any]:
        if self._dashed(data_source_id) != self.data_source_id:
            raise _Refusal(404, "object_not_found", f"Could not find data source with ID: {data_source_id}.")
        unsupported = sorted(set(body) - {"filter", "page_size", "start_cursor"})
        if unsupported:
            raise _Refusal(400, "validation_error", f"the fake does not support {unsupported[0]!r} in a query")
        size = body.get("page_size", MAX_PAGE_SIZE)
        if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= MAX_PAGE_SIZE:
            raise _Refusal(400, "validation_error", f"body.page_size should be between 1 and {MAX_PAGE_SIZE}")
        size = min(size, self.page_limit)
        cursor = body.get("start_cursor")
        offset = 0
        if cursor is not None:
            if not isinstance(cursor, str) or not re.fullmatch(r"cursor-\d+", cursor):
                raise _Refusal(400, "validation_error", "body.start_cursor is not a valid cursor")
            offset = int(cursor.split("-")[1])
        matches = [self._pages[i] for i in self._order if self._matches(body.get("filter"), self._pages[i])]
        chunk = matches[offset: offset + size]
        more = offset + size < len(matches)
        return {"object": "list", "results": [self._page(page) for page in chunk], "next_cursor":
                f"cursor-{offset + size}" if more else None, "has_more": more, "type": "page_or_data_source",
                "page_or_data_source": {}}

    def _matches(self, filter_: Any, page: Mapping[str, Any]) -> bool:
        if filter_ is None:
            return True
        if not isinstance(filter_, Mapping):
            raise _Refusal(400, "validation_error", "body.filter should be an object")
        if set(filter_) in ({"and"}, {"or"}):
            kind = next(iter(filter_))
            parts = filter_[kind]
            if not isinstance(parts, list) or not parts:
                raise _Refusal(400, "validation_error", f"body.filter.{kind} should be a non-empty array")
            results = [self._matches(part, page) for part in parts]
            return all(results) if kind == "and" else any(results)
        name = filter_.get("property")
        condition = filter_.get("select")
        if set(filter_) != {"property", "select"} or name not in SCHEMA or SCHEMA[name]["type"] != "select" \
                or not isinstance(condition, Mapping) or set(condition) != {"equals"}:
            raise _Refusal(400, "validation_error", "the fake supports only select-equals filters and and / or groups")
        chosen = page["properties"][name]["select"]
        return bool(chosen) and chosen["name"] == condition["equals"]
