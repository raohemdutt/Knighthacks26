"""A tiny MCP connector for tests and for building against: python -m contexts.connectors.servers.fake_server

Reads return `options.items`; the `post` write echoes what it would have sent.
"""

from mcp.server.fastmcp import FastMCP

from contexts.connectors.infrastructure.mcp_client import server_options

mcp = FastMCP("atlas-fake-connector")


@mcp.tool()
def list_items() -> list[dict]:
    """Items this fake source holds."""
    return server_options().get("items", [])


@mcp.tool()
def post(text: str) -> str:
    """Pretend to send a message (a write tool: only runs after approval)."""
    return f"posted: {text}"


if __name__ == "__main__":
    mcp.run()
