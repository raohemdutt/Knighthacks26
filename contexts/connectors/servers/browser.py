"""Browser connector (K7): web search, Reddit and page reading through patchright.

    python -m contexts.connectors.servers.browser

patchright is a Playwright fork that sites detect less often. Findings from
probing on 2026-10-10:
- Headless gets DuckDuckGo's captcha and Reddit's "blocked by network
  security". A visible (headful) browser with a persistent profile passes
  both, so `headless` defaults to false. On the Jetson, run it under Xvfb
  (`xvfb-run -a python -m ...`) or set headless: true and accept the misses.
- Reddit's JSON endpoints and old.reddit are blocked; the normal search page
  works and is parsed instead.
- DuckDuckGo's HTML endpoint is the primary search; Bing is the fallback.

Read tools only. Anything that clicks, submits or logs in would be a write
tool and must go through an approval card.

options:
  headless: false
  channel: null          # "chrome" uses installed Chrome; null uses patchright's Chromium
  profile_dir: data/browser/profile
  allowed_hosts: ["*"]   # browse() refuses other hosts; the Vault allowlist (H4) still applies
  max_chars: 6000
"""

import asyncio
import base64
import re
from urllib.parse import parse_qs, quote_plus, urlparse

from mcp.server.fastmcp import FastMCP

from contexts.connectors.infrastructure.mcp_client import server_options

DDG = "https://html.duckduckgo.com/html/?q={q}"
BING = "https://www.bing.com/search?q={q}"
REDDIT = "https://www.reddit.com/r/{sub}/search/?q={q}"

DDG_JS = """() => [...document.querySelectorAll('.result')].map(e => ({
  title: e.querySelector('.result__a')?.innerText?.trim(),
  url: e.querySelector('.result__a')?.href,
  snippet: e.querySelector('.result__snippet')?.innerText?.trim()}))"""
BING_JS = """() => [...document.querySelectorAll('li.b_algo')].map(e => ({
  title: e.querySelector('h2 a')?.innerText?.trim(),
  url: e.querySelector('h2 a')?.href,
  snippet: (e.querySelector('.b_caption p') || e.querySelector('p'))?.innerText?.trim()}))"""
REDDIT_JS = """() => [...document.querySelectorAll('a[data-testid="post-title-text"], a[data-testid="post-title"]')].map(a => ({
  title: a.innerText.trim(), url: a.href}))"""
READABLE_JS = """() => {
  const root = document.querySelector('article') || document.querySelector('main') || document.body;
  return root ? root.innerText : '';
}"""


def real_url(href: str) -> str:
    """Unwrap DuckDuckGo (uddg=) and Bing (ck/a?u=a1<base64>) redirect links."""
    parsed = urlparse(href)
    query = parse_qs(parsed.query)
    if "duckduckgo.com" in parsed.netloc and "uddg" in query:
        return query["uddg"][0]
    if "bing.com" in parsed.netloc and parsed.path.startswith("/ck/") and query.get("u", [""])[0].startswith("a1"):
        b64 = query["u"][0][2:]
        try:
            return base64.urlsafe_b64decode(b64 + "=" * (-len(b64) % 4)).decode()
        except ValueError:
            return href
    return href


def clean_results(raw: list[dict], limit: int) -> list[dict]:
    seen, out = set(), []
    for r in raw:
        if not r.get("title") or not r.get("url"):
            continue
        url = real_url(r["url"])
        if url in seen:
            continue
        seen.add(url)
        out.append({"external_id": url, "kind": "search_result", "title": r["title"], "url": url, "summary": r.get("snippet")})
        if len(out) == limit:
            break
    return out


def readable(text: str, max_chars: int) -> str:
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if len(line) > 1)[:max_chars]


def focus(text: str, goal: str | None, n: int = 3) -> str | None:
    """The few lines that mention the goal's words, as a quick extractive summary."""
    if not goal:
        return None
    words = {w for w in re.findall(r"[a-z0-9]+", goal.lower()) if len(w) > 2}
    scored = sorted(
        ((sum(w in line.lower() for w in words), i, line) for i, line in enumerate(text.splitlines())),
        key=lambda t: (-t[0], t[1]),
    )
    picked = sorted((i, line) for score, i, line in scored[:n] if score)
    return " … ".join(line[:200] for _, line in picked) or None


def host_allowed(url: str, allowed: list[str]) -> bool:
    host = urlparse(url).hostname or ""
    return "*" in allowed or any(host == h or host.endswith("." + h) for h in allowed)


class Browser:
    def __init__(self, options: dict) -> None:
        self.options = options
        self._pw = None
        self._ctx = None
        self._lock = asyncio.Lock()

    async def context(self):
        async with self._lock:
            if self._ctx is None:
                from patchright.async_api import async_playwright

                self._pw = await async_playwright().start()
                self._ctx = await self._pw.chromium.launch_persistent_context(
                    user_data_dir=self.options.get("profile_dir", "data/browser/profile"),
                    channel=self.options.get("channel"),
                    headless=self.options.get("headless", False),
                    no_viewport=True,
                )
            return self._ctx

    async def run(self, url: str, script: str, settle_ms: int = 0, wait_for: str | None = None):
        page = await (await self.context()).new_page()
        try:
            await page.goto(url, wait_until="load", timeout=30_000)
            if wait_for:  # e.g. Reddit's JS challenge redirects once before results render
                try:
                    await page.wait_for_selector(wait_for, timeout=10_000)
                except Exception:
                    pass  # no results or blocked: the script returns what is there
            if settle_ms:
                await page.wait_for_timeout(settle_ms)
            return await page.title(), page.url, await page.evaluate(script)
        finally:
            await page.close()

    async def search(self, query: str, limit: int = 5) -> list[dict]:
        q = quote_plus(query)
        _, _, raw = await self.run(DDG.format(q=q), DDG_JS)
        results = clean_results(raw, limit)
        if not results:  # captcha or layout change: try Bing
            _, _, raw = await self.run(BING.format(q=q), BING_JS, settle_ms=1000)
            results = clean_results(raw, limit)
        return results

    async def reddit(self, subreddit: str, query: str, limit: int = 5) -> list[dict]:
        url = REDDIT.format(sub=quote_plus(subreddit), q=quote_plus(query))
        posts: list[dict] = []
        for _ in range(2):  # a fresh profile gets a JS challenge first; the retry carries its cookie
            _, _, raw = await self.run(url, REDDIT_JS, wait_for='a[data-testid="post-title-text"]')
            if posts := clean_results(raw, limit):
                break
        for p in posts:
            p.update(kind="post", summary=f"r/{subreddit}")
        return posts

    async def browse(self, url: str, goal: str | None = None) -> dict:
        if not host_allowed(url, self.options.get("allowed_hosts", ["*"])):
            raise ValueError(f"{urlparse(url).hostname} is not an allowed host")
        title, final_url, text = await self.run(url, READABLE_JS, settle_ms=500)
        body = readable(text, self.options.get("max_chars", 6000))
        return {"url": final_url, "title": title, "text": body, "summary": focus(body, goal)}

    async def close(self) -> None:
        if self._ctx:
            await self._ctx.close()
        if self._pw:
            await self._pw.stop()
        self._ctx = self._pw = None


mcp = FastMCP("atlas-browser")
_browser: Browser | None = None


def browser() -> Browser:
    global _browser
    if _browser is None:
        _browser = Browser(server_options())
    return _browser


@mcp.tool()
async def search(query: str, limit: int = 5) -> list[dict]:
    """Web search. Returns title, url and snippet for the top results."""
    return await browser().search(query, limit)


@mcp.tool()
async def reddit(subreddit: str, query: str, limit: int = 5) -> list[dict]:
    """Search one subreddit (e.g. "ucf"). Returns post titles and links."""
    return await browser().reddit(subreddit, query, limit)


@mcp.tool()
async def browse(url: str, goal: str | None = None) -> dict:
    """Open a page and return its readable text, plus the lines most relevant to `goal`."""
    return await browser().browse(url, goal)


if __name__ == "__main__":
    mcp.run()
