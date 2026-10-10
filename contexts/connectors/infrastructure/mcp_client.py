"""MCP client side: run a connector as a stdio MCP server and call its tools."""

import json
import logging
import os
import sys
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from contexts.connectors.domain.spec import ConnectorSpec

OPTIONS_ENV = "ATLAS_CONNECTOR_OPTIONS"


class McpConnector:
    def __init__(self, spec: ConnectorSpec) -> None:
        self.spec = spec
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def _connect(self) -> ClientSession:
        if self._session:
            return self._session
        env = {k: os.environ[k] for k in self.spec.env if k in os.environ}
        env[OPTIONS_ENV] = json.dumps(self.spec.options)
        env["PATH"] = os.environ.get("PATH", "")
        params = StdioServerParameters(command=sys.executable, args=["-m", self.spec.module], env=env)
        self._stack = AsyncExitStack()
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        return self._session

    async def call(self, tool: str, arguments: dict[str, Any] | None = None) -> Any:
        session = await self._connect()
        result = await session.call_tool(tool, arguments or {})
        if result.isError:
            raise RuntimeError(" ".join(getattr(c, "text", "") for c in result.content) or f"{tool} failed")
        if result.structuredContent is not None:
            # FastMCP wraps non-object returns as {"result": ...}
            sc = result.structuredContent
            return sc["result"] if set(sc) == {"result"} else sc
        texts = [c.text for c in result.content if getattr(c, "type", None) == "text"]
        if len(texts) == 1:
            try:
                return json.loads(texts[0])
            except json.JSONDecodeError:
                return texts[0]
        return [json.loads(t) for t in texts]

    async def close(self) -> None:
        if self._stack:
            await self._stack.aclose()
        self._stack = self._session = None


def server_options() -> dict:
    """For connector servers: the `options` block from connectors.yaml."""
    # httpx logs every request with its full URL at INFO, and a server's stderr is the box's service log:
    # a published feed link or a ?token= query would be written there on every sync
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return json.loads(os.environ.get(OPTIONS_ENV) or "{}")
