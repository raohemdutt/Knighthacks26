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

search, reddit and browse are read tools. submit_form clicks and submits, so
it is a write tool: it only runs after an approval card, and it refuses
credential or payment fields.

options:
  headless: false
  channel: auto          # installed Chrome if present, else patchright's Chromium (Jetson)
  locale: en-US
  timezone: America/New_York
  min_gap_s: 2           # at least this long between pages on one host
  profile_dir: data/browser/profile
  allowed_hosts: ["*"]   # browse() refuses other hosts; the Vault allowlist (H4) still applies
  max_chars: 6000
"""

import asyncio
import base64
import random
import re
import time
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


CHALLENGE_MARKERS = {
    # text that only shows on a bot check: kind -> phrases (lowercase)
    "js": ["just a moment", "checking your browser", "verifying you are human", "please wait while we verify"],
    "captcha": [
        "bots use duckduckgo too", "select all squares", "confirm this search was made by a human",
        "unusual traffic from your computer", "verify you are human", "i'm not a robot", "press & hold",
    ],
    "blocked": ["blocked by network security", "access denied", "you have been blocked", "request blocked"],
}


def challenge_kind(url: str, text: str) -> str | None:
    """None for a normal page, else "js" (solves itself), "captcha" (needs a person) or "blocked"."""
    if "js_challenge=1" in url or "/cdn-cgi/challenge" in url:
        return "js"
    head = text[:1500].lower()
    for kind in ("js", "captcha", "blocked"):
        if any(m in head for m in CHALLENGE_MARKERS[kind]):
            return kind
    return None


class ChallengeError(RuntimeError):
    def __init__(self, kind: str, url: str) -> None:
        super().__init__(f"{kind} bot check at {url}; open it on your phone to get past it")
        self.kind, self.url = kind, url


class Browser:
    """One persistent, visible browser that looks and paces like the person it works for.

    Best effort against bot checks, for sites the user would visit themselves:
    - patchright (no automation flags), the user's real Chrome when installed,
      a persistent profile so cookies and passed challenges carry over, the
      household's locale and time zone, and human-ish pauses between pages.
    - JS challenges (Cloudflare, Reddit) clear on their own in a real browser:
      wait for them, then retry once with the cookie they set.
    - A real CAPTCHA is not solved automatically. Search falls back to another
      engine; otherwise the tool reports the URL so the person can clear it
      once on their phone, and the shared profile keeps it cleared.
    """

    def __init__(self, options: dict) -> None:
        self.options = options
        self._pw = None
        self._ctx = None
        self._lock = asyncio.Lock()
        self._last_visit: dict[str, float] = {}

    async def context(self):
        async with self._lock:
            if self._ctx is None:
                from patchright.async_api import async_playwright

                self._pw = await async_playwright().start()
                channel = self.options.get("channel", "auto")
                for ch in (["chrome", None] if channel == "auto" else [channel]):
                    try:
                        self._ctx = await self._pw.chromium.launch_persistent_context(
                            user_data_dir=self.options.get("profile_dir", "data/browser/profile"),
                            channel=ch,
                            headless=self.options.get("headless", False),
                            no_viewport=True,
                            locale=self.options.get("locale", "en-US"),
                            timezone_id=self.options.get("timezone", "America/New_York"),
                        )
                        break
                    except Exception:
                        if ch is None or channel != "auto":
                            raise  # bundled Chromium was the last try (e.g. Jetson has no Chrome)
            return self._ctx

    async def _pace(self, url: str) -> None:
        """A short random pause, and at least `min_gap_s` between hits on the same host."""
        host = urlparse(url).hostname or ""
        gap = self.options.get("min_gap_s", 2.0) - (time.monotonic() - self._last_visit.get(host, 0))
        await asyncio.sleep(max(gap, 0) + random.uniform(0.3, 1.0))
        self._last_visit[host] = time.monotonic()

    async def _check(self, page) -> str | None:
        return challenge_kind(page.url, await page.evaluate("() => document.body ? document.body.innerText : ''"))

    async def run(self, url: str, script: str, settle_ms: int = 0, wait_for: str | None = None):
        await self._pace(url)
        page = await (await self.context()).new_page()
        try:
            await page.goto(url, wait_until="load", timeout=30_000)
            kind = await self._check(page)
            for _ in range(15 if kind == "js" else 0):  # let a JS challenge finish and redirect
                await page.wait_for_timeout(1000)
                kind = await self._check(page)
                if kind != "js":
                    break
            if kind:
                raise ChallengeError(kind, page.url)
            if wait_for:
                try:
                    await page.wait_for_selector(wait_for, timeout=10_000)
                except Exception:
                    pass  # no results: the script returns what is there
            if settle_ms:
                await page.wait_for_timeout(settle_ms)
            await page.mouse.wheel(0, random.randint(200, 600))  # people scroll; some checks look for it
            return await page.title(), page.url, await page.evaluate(script)
        finally:
            await page.close()

    async def _retrying(self, url: str, script: str, **kw):
        """One retry after a challenge: the first visit often only sets the cookie."""
        try:
            return await self.run(url, script, **kw)
        except ChallengeError as first:
            if first.kind == "captcha":
                raise
            return await self.run(url, script, **kw)

    async def search(self, query: str, limit: int = 5) -> list[dict]:
        q = quote_plus(query)
        blocked: ChallengeError | None = None
        for template, script, settle in ((DDG, DDG_JS, 0), (BING, BING_JS, 1000)):
            try:
                _, _, raw = await self._retrying(template.format(q=q), script, settle_ms=settle)
            except ChallengeError as exc:
                blocked = exc
                continue
            if results := clean_results(raw, limit):
                return results
        if blocked:
            raise blocked
        return []

    async def reddit(self, subreddit: str, query: str, limit: int = 5) -> list[dict]:
        url = REDDIT.format(sub=quote_plus(subreddit), q=quote_plus(query))
        posts: list[dict] = []
        for _ in range(2):  # results can render late after the challenge redirect
            _, _, raw = await self._retrying(url, REDDIT_JS, wait_for='a[data-testid="post-title-text"]')
            if posts := clean_results(raw, limit):
                break
        for p in posts:
            p.update(kind="post", summary=f"r/{subreddit}")
        return posts

    async def browse(self, url: str, goal: str | None = None) -> dict:
        if not host_allowed(url, self.options.get("allowed_hosts", ["*"])):
            raise ValueError(f"{urlparse(url).hostname} is not an allowed host")
        try:
            title, final_url, text = await self._retrying(url, READABLE_JS, settle_ms=500)
        except ChallengeError as exc:
            # Say so plainly, so the agent hands the link to the person instead of guessing.
            return {"url": exc.url, "title": None, "text": "", "summary": None, "blocked": exc.kind}
        body = readable(text, self.options.get("max_chars", 6000))
        return {"url": final_url, "title": title, "text": body, "summary": focus(body, goal)}

    async def submit_form(self, url: str, fields: dict[str, str], submit: str, wait_for: str | None = None) -> dict:
        """WRITE: fill fields (CSS selector -> value), click `submit`, and report where it landed.

        Declared under `writes`, so it only ever runs after the person approves the
        card that shows exactly these fields. Never used for logins or payments."""
        if not host_allowed(url, self.options.get("allowed_hosts", ["*"])):
            raise ValueError(f"{urlparse(url).hostname} is not an allowed host")
        if any(t in (k + v).lower() for k, v in fields.items() for t in ("password", "card", "cvv", "ssn")):
            raise ValueError("refusing to fill credentials or payment fields")
        await self._pace(url)
        page = await (await self.context()).new_page()
        try:
            await page.goto(url, wait_until="load", timeout=30_000)
            if kind := await self._check(page):
                raise ChallengeError(kind, page.url)
            for selector, value in fields.items():
                await page.fill(selector, value)
                await page.wait_for_timeout(random.randint(150, 400))  # type like a person, not a script
            await page.click(submit)
            await page.wait_for_load_state("load")
            if wait_for:
                await page.wait_for_selector(wait_for, timeout=10_000)
            text = readable(await page.evaluate(READABLE_JS), 1500)
            return {"url": page.url, "title": await page.title(), "text": text}
        finally:
            await page.close()

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


@mcp.tool()
async def submit_form(url: str, fields: dict[str, str], submit: str, wait_for: str | None = None) -> dict:
    """WRITE (approval only): fill a form and submit it, e.g. an RSVP or a room booking."""
    return await browser().submit_form(url, fields, submit, wait_for)


if __name__ == "__main__":
    mcp.run()
