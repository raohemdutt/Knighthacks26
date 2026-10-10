"""A small Notion REST client: query a data source, read, create and update a page.

It is what the GPU queue verbs (lab/machine/slots.py) talk to the GPU Queue database
with. It speaks only the four calls the queue needs, over the standard library's urllib,
so it adds no dependency:

    POST  /v1/data_sources/<id>/query    list the rows, following the cursor to the end
    GET   /v1/pages/<id>                 read one row
    POST  /v1/pages                      add a row (the parent is the data source)
    PATCH /v1/pages/<id>                 change some properties of a row

Every request carries the token as a bearer credential and Notion-Version 2025-09-03, the
first version whose pages hang from data sources. The token is read by the caller (the
NOTION_TOKEN key of the environment or a .env file) and handed to NotionClient; the client
keeps it in a private attribute and scrubs it from every error it raises, so it appears in
no message, log line or answer.

What goes wrong is one exception type, NotionError, whose kind says what to do about it:

    unauthorized   401: the token is wrong or revoked
    restricted     403: the integration lacks a capability
    not_found      404: no such page, or it is not shared with the integration
    validation     400: Notion refused the request (a property name or value it does not accept)
    rate_limited   429 that outlasted the retries
    unavailable    5xx that outlasted the retries
    transport      the request never got an answer (DNS, TLS, timeout, reset)
    bad_response   an answer that is not the JSON Notion documents, or a redirect (never followed)

A 429 (Notion's rate limit, an average of 3 requests a second on most plans), 503 and 529 are
retried, up to MAX_RETRIES times, waiting what Retry-After says (at most MAX_WAIT_S a time;
a longer ask is not waited out). Nothing else is retried: a request that may have reached
Notion is never sent twice, so a timed-out create cannot add a second row.

The wire is the one boundary a test replaces: NotionClient(token, transport=fn) calls
fn(method, url, headers, body, timeout) -> (status, body bytes, headers), which
lab/machine/slots_fake.py implements in memory. Redirects are never followed, so the token
only ever goes to the address it was meant for. Verified against developers.notion.com
(query a data source, create a page, update a page, the parent object, request limits).
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any

__all__ = [
    "API_URL", "API_VERSION", "MAX_RETRIES", "MAX_WAIT_S", "NotionClient", "NotionError", "Transport",
    "checkbox", "date", "dashed_id", "number", "page_id_of", "plain_text", "rich_text", "select", "title",
    "urllib_transport",
]

API_URL = "https://api.notion.com"
API_VERSION = "2025-09-03"
TIMEOUT_S = 15.0
MAX_BODY_BYTES = 5_000_000
MAX_RETRIES = 3
MAX_WAIT_S = 15.0
RETRY_STATUSES = frozenset({429, 503, 529})
PAGE_SIZE = 100
MAX_PAGES = 20            # a list longer than this many pages of PAGE_SIZE rows is cut short and refused
USER_AGENT = "atlas-lab-slots"
# Text shorter than this is not worth scrubbing out of a message; it would mangle the message.
MIN_HIDDEN_CHARS = 6
MAX_MESSAGE_CHARS = 300

Transport = Callable[[str, str, Mapping[str, str], "bytes | None", float], "tuple[int, bytes, Mapping[str, str]]"]


class NotionError(RuntimeError):
    """A call to Notion that did not give back what was asked. kind says which sort of failure (module
    docstring); status and code are Notion's HTTP status and error code when it answered at all."""

    def __init__(self, message: str, *, kind: str, status: int | None = None, code: str | None = None,
                 retry_after: float | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.status = status
        self.code = code
        self.retry_after = retry_after


# --- the wire -------------------------------------------------------------------------

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never follow a redirect: the token must only go to the address it was meant for."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


def urllib_transport(method: str, url: str, headers: Mapping[str, str], body: bytes | None,
                     timeout: float) -> tuple[int, bytes, Mapping[str, str]]:
    """One HTTP exchange: (status, body, headers). An HTTP error status is an answer, not an exception;
    a failure to get any answer raises OSError (URLError, a timeout and a TLS failure are all OSErrors)."""
    request = urllib.request.Request(url, data=body, headers=dict(headers), method=method)
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, response.read(MAX_BODY_BYTES), dict(response.headers)
    except urllib.error.HTTPError as exc:
        with exc:
            return exc.code, exc.read(MAX_BODY_BYTES), dict(exc.headers)


# --- page ids -------------------------------------------------------------------------

_HEX32 = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{32}(?![0-9a-fA-F])")
_DASHED = re.compile(r"(?<![0-9a-fA-F])([0-9a-fA-F]{8})-([0-9a-fA-F]{4})-([0-9a-fA-F]{4})-([0-9a-fA-F]{4})-([0-9a-fA-F]{12})(?![0-9a-fA-F])")


def page_id_of(text: str) -> str | None:
    """The 32 lowercase hex digits of the page id that a Notion page id or URL names, else None.

    A bare id, a dashed id and a URL are all accepted (https://app.notion.com/p/<id>?pvs=204,
    https://www.notion.so/Some-title-<id>). In a URL the id is searched for in the path only, so the
    view id of a ?v=<id> query is never mistaken for the page."""
    candidate = str(text).strip()
    candidate = re.split(r"[?#]", candidate, maxsplit=1)[0]
    dashed = _DASHED.findall(candidate)
    if dashed:
        return "".join(dashed[-1]).lower()
    found = _HEX32.findall(candidate)
    return found[-1].lower() if found else None


def dashed_id(page_id: str) -> str:
    """A 32-digit id as the 8-4-4-4-12 text the API uses."""
    plain = page_id.replace("-", "").lower()
    return f"{plain[:8]}-{plain[8:12]}-{plain[12:16]}-{plain[16:20]}-{plain[20:]}"


# --- property values, as the API writes them -------------------------------------------

def _text_objects(text: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": {"content": text}}] if text else []


def title(text: str) -> dict[str, Any]:
    return {"title": _text_objects(text)}


def rich_text(text: str) -> dict[str, Any]:
    """Text for a rich-text property; "" clears it."""
    return {"rich_text": _text_objects(text)}


def number(value: float | int | None) -> dict[str, Any]:
    return {"number": value}


def select(name: str | None) -> dict[str, Any]:
    """A select option by name; None clears it."""
    return {"select": None if name is None else {"name": name}}


def checkbox(value: bool) -> dict[str, Any]:
    return {"checkbox": bool(value)}


def date(start: str | None) -> dict[str, Any]:
    """A date or date-time (ISO 8601 text, with a UTC offset when it has a time); None clears it."""
    return {"date": None if start is None else {"start": start}}


# --- property values, as the API reads them --------------------------------------------

def plain_text(prop: Mapping[str, Any] | None) -> str:
    """The text of a title or rich-text property value as the API returns it; "" for anything else."""
    if not isinstance(prop, Mapping):
        return ""
    items = prop.get(str(prop.get("type")))
    if not isinstance(items, list):
        return ""
    parts: list[str] = []
    for item in items:
        if isinstance(item, Mapping):
            text = item.get("plain_text")
            if text is None and isinstance(item.get("text"), Mapping):
                text = item["text"].get("content")
            parts.append(text if isinstance(text, str) else "")
    return "".join(parts)


# --- the client -----------------------------------------------------------------------

def _scrub(text: str, hide: str) -> str:
    return text.replace(hide, "***") if len(hide) >= MIN_HIDDEN_CHARS else text


def _one_line(text: str) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= MAX_MESSAGE_CHARS else flat[: MAX_MESSAGE_CHARS - 3] + "..."


_KINDS = {400: "validation", 401: "unauthorized", 403: "restricted", 404: "not_found", 429: "rate_limited"}


def _kind_of(status: int) -> str:
    if status in _KINDS:
        return _KINDS[status]
    if status >= 500:
        return "unavailable"
    return "bad_response" if status < 400 else "validation"


def _retry_after(headers: Mapping[str, str]) -> float | None:
    for name, value in headers.items():
        if name.lower() == "retry-after":
            try:
                seconds = float(str(value).strip())
            except ValueError:
                return None
            return seconds if seconds >= 0 else None
    return None


class NotionClient:
    """The four Notion calls the GPU queue needs. See the module docstring."""

    def __init__(self, token: str, *, base_url: str = API_URL, transport: Transport | None = None,
                 sleep: Callable[[float], None] = time.sleep, timeout: float = TIMEOUT_S) -> None:
        if not isinstance(token, str) or not token.strip():
            raise ValueError("a Notion token must be a non-empty string")
        self._token = token.strip()
        self.base_url = base_url.rstrip("/")
        self._transport: Transport = transport or urllib_transport
        self._sleep = sleep
        self.timeout = timeout

    def __repr__(self) -> str:
        return f"NotionClient(base_url={self.base_url!r})"

    def _wait(self, attempt: int, hinted: float | None) -> float | None:
        """Seconds to wait before retry number attempt (1-based); None when the wait would be too long."""
        seconds = hinted if hinted is not None else float(2 ** (attempt - 1))
        return seconds if seconds <= MAX_WAIT_S else None

    def request(self, method: str, path: str, body: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """One call, retried as the module docstring says; the parsed JSON object, or NotionError."""
        url = self.base_url + path
        payload = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {"Authorization": f"Bearer {self._token}", "Notion-Version": API_VERSION,
                   "Accept": "application/json", "User-Agent": USER_AGENT}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        attempt = 0
        while True:
            try:
                status, raw, response_headers = self._transport(method, url, headers, payload, self.timeout)
            except Exception as exc:
                reason = getattr(exc, "reason", None) or exc
                raise NotionError(_scrub(_one_line(f"cannot reach Notion: {type(exc).__name__}: {reason}"), self._token),
                                  kind="transport") from None
            if status in RETRY_STATUSES and attempt < MAX_RETRIES:
                attempt += 1
                hinted = _retry_after(response_headers)
                seconds = self._wait(attempt, hinted)
                if seconds is None:
                    break
                self._sleep(seconds)
                continue
            break
        return self._answer(status, raw, response_headers)

    def _answer(self, status: int, raw: bytes, headers: Mapping[str, str]) -> dict[str, Any]:
        try:
            data = json.loads(raw) if raw else {}
        except ValueError:
            data = None
        if 200 <= status < 300:
            if isinstance(data, dict):
                return data
            raise NotionError(f"Notion answered {status} with something that is not a JSON object",
                              kind="bad_response", status=status)
        code = data.get("code") if isinstance(data, dict) else None
        message = data.get("message") if isinstance(data, dict) else None
        detail = _scrub(_one_line(message or f"HTTP {status}"), self._token)
        code = code if isinstance(code, str) else None
        raise NotionError(f"Notion answered {status}{' ' + code if code else ''}: {detail}", kind=_kind_of(status),
                          status=status, code=code, retry_after=_retry_after(headers))

    def query_data_source(self, data_source_id: str, *,
                          filter: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        """Every row of the data source that passes filter, in Notion's order, following the cursor to the end.
        More than MAX_PAGES pages is a NotionError (bad_response): a queue that long is not a queue."""
        rows: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(MAX_PAGES):
            body: dict[str, Any] = {"page_size": PAGE_SIZE}
            if filter is not None:
                body["filter"] = filter
            if cursor:
                body["start_cursor"] = cursor
            data = self.request("POST", f"/v1/data_sources/{data_source_id}/query", body)
            results = data.get("results")
            if not isinstance(results, list):
                raise NotionError("Notion's query answer has no results list", kind="bad_response")
            rows.extend(item for item in results if isinstance(item, dict))
            cursor = data.get("next_cursor") if data.get("has_more") else None
            if not cursor:
                return rows
        raise NotionError(f"the query is still not finished after {MAX_PAGES * PAGE_SIZE} rows", kind="bad_response")

    def retrieve_page(self, page_id: str) -> dict[str, Any]:
        return self.request("GET", f"/v1/pages/{dashed_id(page_id)}")

    def create_page(self, data_source_id: str, properties: Mapping[str, Any]) -> dict[str, Any]:
        body = {"parent": {"type": "data_source_id", "data_source_id": data_source_id}, "properties": dict(properties)}
        return self.request("POST", "/v1/pages", body)

    def update_page(self, page_id: str, properties: Mapping[str, Any]) -> dict[str, Any]:
        return self.request("PATCH", f"/v1/pages/{dashed_id(page_id)}", {"properties": dict(properties)})
