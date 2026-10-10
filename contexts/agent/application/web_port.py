"""What the Agent needs from the web (K5, K7): search, Reddit, and reading a page.

The real adapter goes through ConnectorHub to the patchright browser connector;
FakeWeb serves canned results for tests.
"""

from typing import Protocol

from contexts.connectors.application.hub import ConnectorHub


class WebTools(Protocol):
    async def search(self, query: str) -> list[dict]: ...  # [{title, url, summary}]

    async def reddit(self, subreddit: str, query: str) -> list[dict]: ...

    async def browse(self, url: str, goal: str | None = None) -> dict: ...  # {url, title, text, summary, blocked?}


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
