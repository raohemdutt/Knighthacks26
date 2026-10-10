from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

from bus import EventBus
from contexts.vault import Policy, Wiper
from contexts.vault.api import vault_router

T0 = datetime(2026, 10, 11, 12, tzinfo=UTC)


class Clock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        return self.now

    def tick(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class Box:
    """The vault wired the way the gateway wires it, on a data dir under tmp_path."""

    def __init__(self, data_dir: Path, token: str | None = None) -> None:
        self.bus = EventBus(data_dir / "events")
        self.policy = Policy(data_dir)
        self.policy.subscribe(self.bus)  # replayed
        self.wiper = Wiper(data_dir)
        self.bus.subscribe("vault.wipe.requested", self.wiper.on_wipe_requested)  # live only
        self.clock = Clock()
        self.seen: list = []
        self.bus.subscribe("vault.", self.seen.append)

        def guard(authorization: str | None = Header(default=None)) -> None:
            if token and authorization != f"Bearer {token}":
                raise HTTPException(401, "missing or wrong token")

        app = FastAPI()
        app.include_router(vault_router(self.policy, self.bus, wiper=self.wiper, guard=guard, clock=self.clock))
        self.http = TestClient(app)
