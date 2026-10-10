from typing import Any, Protocol


class Connector(Protocol):
    """One connected source. Real ones are MCP servers; FakeConnector is for tests."""

    async def call(self, tool: str, arguments: dict[str, Any] | None = None) -> Any:
        """Call a tool by its short name ("check_pages"). Returns parsed JSON."""
        ...

    async def close(self) -> None: ...
