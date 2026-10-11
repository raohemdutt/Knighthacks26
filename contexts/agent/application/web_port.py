"""What the Agent needs from the web (K5, K7): search, Reddit, and reading a page.

The real adapter goes through ConnectorHub to the patchright browser connector;
FakeWeb serves canned results for tests.
"""

import re
from typing import Protocol

import httpx

from contexts.connectors.application.hub import ConnectorHub


class WebTools(Protocol):
    async def search(self, query: str) -> list[dict]: ...  # [{title, url, summary}]

    async def reddit(self, subreddit: str, query: str) -> list[dict]: ...

    async def browse(self, url: str, goal: str | None = None) -> dict: ...  # {url, title, text, summary, blocked?}


TAVILY_URL = "https://api.tavily.com/search"
# "the most recent game", "today's score": search the last week's news, where results are about the latest event
RECENT = re.compile(r"\b(?:most recent|latest|last|today|tonight|yesterday|this (?:week|weekend)|right now|scores?|news)\b",
                    re.IGNORECASE)


# Tavily's summary saying the results don't answer the question
NOT_FOUND = re.compile(r"\b(?:don['\u2019]?t|do not|doesn['\u2019]?t|does not) (?:provide|include|mention|contain|say)"
                       r"|\bno (?:information|details|mention)\b", re.IGNORECASE)


class TavilyWebTools:
    """Search through Tavily's API (about 2 s, no browser); Reddit and page reading stay with `fallback` (the browser
    connector) when there is one. The key comes from TAVILY_API_KEY in .env."""

    def __init__(self, api_key: str, fallback: "WebTools | None" = None, limit: int = 4, timeout_s: float = 8.0,
                 post=None) -> None:
        self.api_key = api_key
        self.fallback = fallback
        self.limit = limit
        self.timeout_s = timeout_s
        self._post = post  # tests: an async (url, json, headers) -> dict

    async def _call(self, body: dict) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if self._post:
            return await self._post(TAVILY_URL, body, headers)
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.post(TAVILY_URL, json=body, headers=headers)
            r.raise_for_status()
            return r.json()

    async def search(self, query: str) -> list[dict]:
        body = {"query": query, "max_results": self.limit, "include_answer": True, "search_depth": "basic"}
        data = None
        if RECENT.search(query):
            data = await self._call(body | {"topic": "news", "time_range": "week"})
            if not data.get("results") or NOT_FOUND.search(data.get("answer") or ""):
                data = None  # nothing this week ("the last UCF vs UF game" was in 2024): search everything instead
        data = data or await self._call(body)
        results = [{"title": r.get("title") or r.get("url", ""), "url": r.get("url", ""),
                    "summary": " ".join((r.get("content") or "").split())[:300]}
                   for r in data.get("results") or [] if r.get("url")]
        if data.get("answer") and results:  # Tavily's own summary of these results, cited to the top one
            results.insert(0, {"title": "Search summary", "url": results[0]["url"], "summary": data["answer"][:400]})
        return results

    async def reddit(self, subreddit: str, query: str) -> list[dict]:
        return await self.fallback.reddit(subreddit, query) if self.fallback else []

    async def browse(self, url: str, goal: str | None = None) -> dict:
        if self.fallback:
            return await self.fallback.browse(url, goal)
        return {"url": url, "title": None, "text": "", "summary": None}


class HubWebTools:
    def __init__(self, hub: ConnectorHub, connector: str = "browser", limit: int = 5) -> None:
        self.hub = hub
        self.connector = connector
        self.limit = limit

    async def search(self, query: str) -> list[dict]:
        return await self.hub.call_tool(f"{self.connector}.search", {"query": query, "limit": self.limit})

    async def reddit(self, subreddit: str, query: str) -> list[dict]:
        args = {"subreddit": subreddit, "query": query, "limit": self.limit}
        return await self.hub.call_tool(f"{self.connector}.reddit", args)

    async def browse(self, url: str, goal: str | None = None) -> dict:
        return await self.hub.call_tool(f"{self.connector}.browse", {"url": url, "goal": goal})


class FakeWeb:
    def __init__(self, results: list[dict] | None = None, posts: list[dict] | None = None, pages: dict | None = None):
        self.results = results or []
        self.posts = posts or []
        self.pages = pages or {}
        self.calls: list[tuple] = []

    async def search(self, query):
        self.calls.append(("search", query))
        return self.results

    async def reddit(self, subreddit, query):
        self.calls.append(("reddit", subreddit, query))
        return self.posts

    async def browse(self, url, goal=None):
        self.calls.append(("browse", url, goal))
        return self.pages.get(url, {"url": url, "title": None, "text": "", "summary": None})
