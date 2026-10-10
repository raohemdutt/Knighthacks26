"""Site watcher (K3) as an MCP server: python -m contexts.connectors.servers.site

Fetches each page in `options.pages`, keeps the visible text of the last copy
under `options.state_dir`, and returns one item per page whose text changed.
The first fetch of a page only records a baseline.
"""

import difflib
import hashlib
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP

from contexts.connectors.infrastructure.mcp_client import server_options

SKIP_TAGS = {"script", "style", "noscript", "svg", "head"}
SUMMARY_LEN = 160


class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        self._skip += tag in SKIP_TAGS

    def handle_endtag(self, tag):
        self._skip -= tag in SKIP_TAGS and self._skip > 0

    def handle_data(self, data):
        text = " ".join(data.split())
        if text and not self._skip:
            self.lines.append(text)


def visible_text(html: str) -> list[str]:
    p = _Text()
    p.feed(html)
    return p.lines


def summarize_change(old: list[str], new: list[str]) -> str | None:
    added = [line for line in difflib.ndiff(old, new) if line.startswith("+ ")]
    removed = [line for line in difflib.ndiff(old, new) if line.startswith("- ")]
    if not added and not removed:
        return None
    lead = added[0][2:] if added else f"removed: {removed[0][2:]}"
    more = len(added) + len(removed) - 1
    summary = lead if more <= 0 else f"{lead} (+{more} more changes)"
    return summary[:SUMMARY_LEN]


def check(pages: list[dict], state_dir: Path, fetch) -> list[dict]:
    state_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for page in pages:
        url = page["url"]
        lines = visible_text(fetch(url))
        copy = state_dir / (hashlib.sha256(url.encode()).hexdigest()[:16] + ".txt")
        old = copy.read_text(encoding="utf-8").splitlines() if copy.exists() else None
        copy.write_text("\n".join(lines), encoding="utf-8")
        if old is None:
            continue  # baseline
        summary = summarize_change(old, lines)
        if summary:
            items.append({
                "external_id": f"{url}#{hashlib.sha256(chr(10).join(lines).encode()).hexdigest()[:12]}",
                "title": f"{page.get('title', url)} changed",
                "url": url,
                "summary": summary,
                "when": datetime.now(UTC).isoformat(),
            })
    return items


def _fetch(url: str) -> str:
    r = httpx.get(url, timeout=20, follow_redirects=True, headers={"User-Agent": "Atlas site watcher"})
    r.raise_for_status()
    return r.text


mcp = FastMCP("atlas-site-watcher")


@mcp.tool()
def check_pages() -> list[dict]:
    """Check watched pages and return one item per page that changed since the last check."""
    opts = server_options()
    return check(opts.get("pages", []), Path(opts.get("state_dir", "data/connectors/site")), _fetch)


if __name__ == "__main__":
    mcp.run()
