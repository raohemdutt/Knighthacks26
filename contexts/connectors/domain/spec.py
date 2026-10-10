"""Connector registry from config/connectors.yaml (K1).

Each connector is an MCP server. Its read tools run on a schedule and return
items; its write tools only ever run after an approval card.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel

from contracts.connectors import ItemKind


class ConnectorSpec(BaseModel):
    name: str  # "site", "canvas", "calendar", "groupme", "search"
    enabled: bool = True
    module: str  # python module run as an MCP stdio server: python -m <module>
    every_s: int = 300  # sync period for read tools
    hosts: list[str] = []  # egress the Vault opens for this connector (H4)
    reads: dict[str, ItemKind] = {}  # read tool -> kind of items it returns
    writes: list[str] = []  # tools with an outside effect; always proposed, never run directly
    options: dict = {}  # passed to the server as ATLAS_CONNECTOR_OPTIONS (JSON); no secrets here
    env: list[str] = []  # names of .env variables the server may read (tokens)

    def tool(self, name: str) -> str:
        return f"{self.name}.{name}"


def load_specs(path: str | Path = "config/connectors.yaml") -> list[ConnectorSpec]:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return [ConnectorSpec(name=name, **body) for name, body in raw.items()]
