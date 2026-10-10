"""A connector to build against without any external service (K1)."""

from typing import Any


class FakeConnector:
    def __init__(self, reads: dict[str, list[dict]] | None = None) -> None:
        self.reads = reads or {}  # tool -> items it returns
        self.calls: list[tuple[str, dict]] = []
        self.fail: str | None = None  # set to make every call raise

    async def call(self, tool: str, arguments: dict[str, Any] | None = None) -> Any:
        self.calls.append((tool, arguments or {}))
        if self.fail:
            raise RuntimeError(self.fail)
        if tool in self.reads:
            return self.reads[tool]
        return {"ok": True, "tool": tool, "arguments": arguments or {}}

    async def close(self) -> None:
        pass
