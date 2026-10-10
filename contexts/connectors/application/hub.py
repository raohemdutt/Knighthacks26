"""Connector hub (K1): reads sync on a timer, writes ask first.

- `sync_due(now)` runs every read tool whose connector is due and turns new or
  changed items into `connector.item.synced`; failures become
  `connector.sync.failed`. Offline, syncs pause and say so.
- `propose_write(...)` turns a write tool call into `agent.action.proposed`;
  `execute_write` is the Workspace executor that runs it after approval.
"""

import hashlib
import json
from collections.abc import Callable
from datetime import datetime, timedelta
from uuid import uuid4

from contexts.connectors.application.port import Connector
from contexts.connectors.domain.spec import ConnectorSpec
from contexts.workspace.domain.approvals import ActionRequest
from contracts.agent import ActionProposed
from contracts.connectors import ItemSynced, SyncFailed
from contracts.events import Event


class ConnectorHub:
    def __init__(
        self,
        specs: list[ConnectorSpec],
        factory: Callable[[ConnectorSpec], Connector],
        offline: Callable[[], bool] = lambda: False,
    ) -> None:
        self.specs = {s.name: s for s in specs if s.enabled}
        self.factory = factory
        self.offline = offline
        self._connectors: dict[str, Connector] = {}
        self._last_sync: dict[str, datetime] = {}
        self._fingerprints: dict[tuple[str, str], str] = {}

    def connector(self, name: str) -> Connector:
        if name not in self._connectors:
            self._connectors[name] = self.factory(self.specs[name])
        return self._connectors[name]

    # --- reads ---

    def due(self, now: datetime) -> list[str]:
        return [
            name
            for name, spec in self.specs.items()
            if spec.reads and (name not in self._last_sync or now - self._last_sync[name] >= timedelta(seconds=spec.every_s))
        ]

    async def sync_due(self, now: datetime) -> list[Event]:
        out: list[Event] = []
        for name in self.due(now):
            out += await self.sync(name, now)
        return out

    async def sync(self, name: str, now: datetime) -> list[Event]:
        spec = self.specs[name]
        self._last_sync[name] = now
        if self.offline():
            return [SyncFailed(ts=now, connector=name, error="offline", offline=True)]
        out: list[Event] = []
        for tool, kind in spec.reads.items():
            try:
                items = await self.connector(name).call(tool)
            except Exception as exc:  # one broken source must not stop the others
                out.append(SyncFailed(ts=now, connector=name, error=f"{tool}: {exc}"))
                continue
            for raw in items or []:
                item = ItemSynced(ts=now, connector=name, kind=raw.get("kind", kind), **_item_fields(raw))
                key = (name, item.external_id)
                fingerprint = _fingerprint(item)
                if self._fingerprints.get(key) != fingerprint:
                    self._fingerprints[key] = fingerprint
                    out.append(item)
        return out

    # --- writes ---

    def is_write(self, tool: str) -> bool:
        name, _, short = tool.partition(".")
        return name in self.specs and short in self.specs[name].writes

    def propose_write(self, tool: str, arguments: dict, reason: str, now: datetime) -> ActionProposed:
        if not self.is_write(tool):
            raise ValueError(f"{tool} is not a declared write tool")
        return ActionProposed(
            ts=now, action_id=uuid4().hex, kind="connector_write", target=tool, reason=reason, params=arguments
        )

    async def execute_write(self, req: ActionRequest) -> str:
        """Workspace executor for `connector_write`. Only reached after an approve."""
        if not self.is_write(req.target):
            raise ValueError(f"{req.target} is not a declared write tool")
        if self.offline():
            raise RuntimeError("offline: connector writes are paused")
        name, _, short = req.target.partition(".")
        result = await self.connector(name).call(short, req.params)
        return result if isinstance(result, str) else json.dumps(result)

    async def close(self) -> None:
        for c in self._connectors.values():
            await c.close()
        self._connectors.clear()


def _item_fields(raw: dict) -> dict:
    fields = {k: raw[k] for k in ("title", "when", "url", "summary") if raw.get(k) is not None}
    return {"external_id": str(raw["external_id"]), **fields}


def _fingerprint(item: ItemSynced) -> str:
    body = item.model_dump(mode="json", include={"kind", "title", "when", "url", "summary"})
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
