"""Privacy Vault API for the phone's Privacy screen (V3, V4).

    app.include_router(vault_router(atlas.policy, atlas.bus, wiper=atlas.wiper, guard=authorized))

- GET  /vault/policy          {blur_faces, paused_cameras, offline, keep_days}
- PUT  /vault/policy          any of those fields; the rest stay as they are
- POST /vault/wipe/request    {scope: all|keyframes|memory} -> {token, scope, expires_at}
- POST /vault/wipe            {token} -> publishes vault.wipe.requested; the Wiper does the deleting
- GET  /vault/egress          allowed bytes per group (hub, each connector, LAN, NTP, DNS), blocked count and recent blocks
                               from the firewall's snapshot; `enforced` is false where no firewall runs

`guard` is the gateway's token check; it goes on the PUT and both POSTs. The
endpoints only publish events: the policy changes because `policy.subscribe(bus)`
applies them, the same way it does on replay.
"""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from bus import EventBus
from contexts.vault.application.policy import Policy
from contexts.vault.application.wipe import WipeTokens, Wiper
from contracts.types import CameraId
from contracts.vault import PolicyChanged, RetentionChanged, WipeRequested


class PolicyBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    blur_faces: bool | None = None
    paused_cameras: list[CameraId] | None = None
    offline: bool | None = None
    keep_days: int | None = Field(default=None, ge=1)  # sent as null = keep forever; left out = unchanged


class WipeRequestBody(BaseModel):
    scope: Literal["all", "keyframes", "memory"] = "all"


class WipeBody(BaseModel):
    token: str


def vault_router(
    policy: Policy,
    bus: EventBus,
    wiper: Wiper | None = None,
    guard: Callable | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    tokens: WipeTokens | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/vault", tags=["vault"])
    writes = [Depends(guard)] if guard else []
    tokens = tokens or WipeTokens()

    @router.get("/policy")
    async def get_policy():
        return policy.snapshot()

    @router.put("/policy", dependencies=writes)
    async def put_policy(body: PolicyBody):
        now = policy.state
        events = [PolicyChanged(
            blur_faces=now.blur_faces if body.blur_faces is None else body.blur_faces,
            paused_cameras=now.paused_cameras if body.paused_cameras is None else sorted(set(body.paused_cameras)),
            offline=now.offline if body.offline is None else body.offline,
        )]
        if "keep_days" in body.model_fields_set and body.keep_days != now.keep_days:
            events.append(RetentionChanged(keep_days=body.keep_days))
        await bus.publish_all(events)
        return policy.snapshot()

    @router.post("/wipe/request", dependencies=writes)
    async def request_wipe(body: WipeRequestBody):
        token, expires_at = tokens.request(body.scope, clock())
        return {"token": token, "scope": body.scope, "expires_at": expires_at.isoformat()}

    @router.post("/wipe", dependencies=writes)
    async def wipe(body: WipeBody):
        scope = tokens.redeem(body.token, clock())
        if scope is None:
            raise HTTPException(403, "confirmation token is wrong, already used or expired")
        before = wiper.last if wiper else None
        await bus.publish(WipeRequested(scope=scope))  # a subscribed Wiper runs before this returns
        report = wiper.last if wiper and wiper.last is not before else None
        out = {"scope": scope, "done": report is not None}
        if report:
            out |= {"removed": report.removed, "refused": report.refused}
        return out

    @router.get("/egress")
    async def egress():
        """What the egress firewall (V6, jetson/egress.py) let out per group and blocked, from its snapshot file
        next to policy.json. Without one (a laptop, or the service is not running) `enforced: false` says so."""
        snap = policy.path.parent / "egress.json"
        try:
            data = json.loads(snap.read_text())
        except (OSError, ValueError):
            return {"allowed": [], "blocked": 0, "enforced": False}
        updated = datetime.strptime(data["updated"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        data["stale"] = (clock() - updated).total_seconds() > 120  # the service stopped writing: the rules still hold
        return data

    return router
