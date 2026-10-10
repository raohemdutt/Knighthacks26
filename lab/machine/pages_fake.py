"""An in-memory Notion for the tests and rehearsals of the flows that edit a page's text.

FakePages stands where Notion's page-markdown endpoints stand (GET and PATCH /v1/pages/<id>/markdown),
the calls lab/machine/rulings.py speaks. It serves two boundaries from the same pages:

    the NotionPages protocol of rulings.py     read_page(), replace_text()
    the wire under rulings.RestNotionClient    RestNotionClient(token, transport=fake.transport)

so a test can run a flow against the plain protocol, or run the real REST client against the fake wire and
see the requests it would send. A page is its markdown text, kept exactly as given.

What it enforces, as the API reference says Notion does: a content update replaces `old_str` with `new_str`
and is refused (400 validation_error) when `old_str` is not found or is found more than once and
`replace_all_matches` is not true; a page that is not known answers 404 object_not_found; a request needs
the bearer token (401 otherwise) and the Notion-Version the markdown endpoints require (400 otherwise). The
protocol raises rulings.NotionError for the same refusals.

A test can also make it misbehave:

    fake.fail_next(2, status=500)       the next two calls (either boundary) fail that way
    fake.set_truncated(page_id)         the page reads as cut short, with unknown blocks listed
    fake.on_call(hook)                  hook(fake, kind, page_id) runs before every call: another writer
                                        getting in first

and look back at what happened: `edits` (page, old, new, in order), `reads` (page ids), `requests` (the wire
calls as (method, path, body); the token is never kept) and `text(page_id)` for the page as it stands now.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from typing import Any

from lab.machine.rulings import NotionError

__all__ = ["API_VERSION", "TOKEN", "FakePages"]

TOKEN = "fake-notion-token-0123456789abcdef"
API_VERSION = "2026-03-11"
_URL = re.compile(r"https://api\.notion\.com/v1/pages/(?P<id>[^/]+)/markdown")


def _key(page_id: str) -> str:
    """The id as it is compared: dashes dropped, lowercase."""
    return page_id.replace("-", "").lower()


class FakePages:
    """See the module docstring."""

    def __init__(self, pages: Mapping[str, str] | None = None, *, token: str = TOKEN) -> None:
        self.token = token
        self._pages: dict[str, str] = {_key(page_id): text for page_id, text in (pages or {}).items()}
        self._truncated: set[str] = set()
        self._failures: list[tuple[int, str]] = []
        self._hooks: list[Callable[[FakePages, str, str], None]] = []
        self.edits: list[tuple[str, str, str]] = []
        self.reads: list[str] = []
        self.requests: list[tuple[str, str, Any]] = []

    # --- seeding and looking back ----------------------------------------------------------

    def add_page(self, page_id: str, text: str) -> None:
        self._pages[_key(page_id)] = text

    def text(self, page_id: str) -> str:
        """The page as it stands now."""
        return self._pages[_key(page_id)]

    def fail_next(self, count: int = 1, *, status: int = 500, code: str = "internal_server_error") -> None:
        self._failures.extend([(status, code)] * count)

    def set_truncated(self, page_id: str) -> None:
        self._truncated.add(_key(page_id))

    def on_call(self, hook: Callable[[FakePages, str, str], None]) -> None:
        self._hooks.append(hook)

    # --- the NotionPages protocol ------------------------------------------------------------

    def read_page(self, page_id: str) -> str:
        self.reads.append(page_id)
        refusal = self._before("read", page_id)
        if refusal:
            raise NotionError(f"Notion answered {refusal[0]} to GET on page {page_id}: {refusal[1]}")
        return self._page(page_id)

    def replace_text(self, page_id: str, old: str, new: str) -> None:
        refusal = self._before("replace", page_id)
        if refusal:
            raise NotionError(f"Notion answered {refusal[0]} to PATCH on page {page_id}: {refusal[1]}")
        problem = self._apply(page_id, old, new, replace_all=False)
        if problem:
            raise NotionError(f"Notion answered 400 to PATCH on page {page_id}: {problem}")

    # --- the wire under RestNotionClient -----------------------------------------------------

    def transport(self, method: str, url: str, headers: Mapping[str, str], body: bytes | None,
                  timeout: float) -> tuple[int, Any]:
        """(status, decoded JSON) for one request, as rulings.urllib_transport would hand it back."""
        found = _URL.fullmatch(url)
        if not found:
            return 404, _error(404, "invalid_request_url", f"unexpected address {url}")
        page_id = found["id"]
        self.requests.append((method, f"/v1/pages/{page_id}/markdown", json.loads(body) if body else None))
        if headers.get("Authorization") != f"Bearer {self.token}":
            return 401, _error(401, "unauthorized", "API token is invalid.")
        if headers.get("Notion-Version") != API_VERSION:
            return 400, _error(400, "validation_error", f"the markdown endpoints need Notion-Version {API_VERSION}")
        refusal = self._before("read" if method == "GET" else "replace", page_id)
        if refusal:
            return refusal[0], _error(refusal[0], refusal[1], "the fake was told to fail")
        if _key(page_id) not in self._pages:
            return 404, _error(404, "object_not_found", f"no page {page_id}")
        if method == "GET":
            self.reads.append(page_id)
            return 200, self._markdown(page_id)
        if method != "PATCH":
            return 405, _error(405, "invalid_request", f"{method} is not served")
        update = json.loads(body) if body else {}
        if update.get("type") != "update_content":
            return 400, _error(400, "validation_error", f"operation {update.get('type')!r} is not served by this fake")
        for item in update["update_content"]["content_updates"]:
            problem = self._apply(page_id, item["old_str"], item["new_str"],
                                  replace_all=bool(item.get("replace_all_matches")))
            if problem:
                return 400, _error(400, "validation_error", problem)
        return 200, self._markdown(page_id)

    # --- inside --------------------------------------------------------------------------------

    def _before(self, kind: str, page_id: str) -> tuple[int, str] | None:
        for hook in list(self._hooks):
            hook(self, kind, page_id)
        return self._failures.pop(0) if self._failures else None

    def _page(self, page_id: str) -> str:
        try:
            return self._pages[_key(page_id)]
        except KeyError:
            raise NotionError(f"Notion answered 404 to a call on page {page_id}: object_not_found") from None

    def _apply(self, page_id: str, old: str, new: str, *, replace_all: bool) -> str | None:
        """Replace in the page; the reason it was refused, or None when it was done."""
        if not old:
            return "old_str is an empty string"
        text = self._page(page_id)
        found = text.count(old)
        if found == 0:
            return "old_str matched nothing"
        if found > 1 and not replace_all:
            return f"old_str matched {found} places and replace_all_matches is not true"
        self._pages[_key(page_id)] = text.replace(old, new)
        self.edits.append((page_id, old, new))
        return None

    def _markdown(self, page_id: str) -> dict[str, Any]:
        cut = _key(page_id) in self._truncated
        return {"object": "page_markdown", "id": page_id, "markdown": self._pages[_key(page_id)], "truncated": cut,
                "unknown_block_ids": ["00000000-0000-0000-0000-000000000001"] if cut else []}


def _error(status: int, code: str, message: str) -> dict[str, Any]:
    return {"object": "error", "status": status, "code": code, "message": message}
