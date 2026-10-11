"""Privacy Vault API for the phone's Privacy screen (V3, V4).

    app.include_router(vault_router(atlas.policy, atlas.bus, wiper=atlas.wiper, guard=authorized))

- GET  /vault/policy          {blur_faces, paused_cameras, offline, keep_days, live_faces, train_faces}; `live_faces` is
                               "clear" unless the owner chose "owner" (only their own scanned face is clear on their
                               screen) or "blurred", also on a box that never recorded it; `train_faces` is "blurred"
                               unless a scanned owner allowed their own face unblurred in a training export. Where the
                               box knows who its owner is (`owner` below) the answer also holds, read-only:
                               `owner` {scanned, ready, reason, demo}, `live_faces_now` (what a ticket holder gets
                               right now: "owner" falls back to "blurred" until the face scan is done and recognition
                               is on) and `faces_note` (one line saying why, or null)
- PUT  /vault/policy          any of the first six fields; the rest stay as they are. 409 with the one-line reason when
                               `live_faces: "owner"` or `train_faces: "owner_clear"` is asked for and nobody has done
                               the face scan (or recognition is off)
- POST /vault/view-ticket     -> {ticket, expires_at, expires_in}: a short-lived ticket the owner's signed-in screen puts
                               on the live picture's address (`?ticket=`) to see faces there while `live_faces` is
                               "clear". Everyone without one gets the blurred live picture, as before.
- POST /vault/wipe/request    {scope: all|keyframes|memory} -> {token, scope, expires_at}
- POST /vault/wipe            {token} -> publishes vault.wipe.requested; the Wiper does the deleting
- GET  /vault/egress          allowed bytes per group (hub, each connector, LAN, NTP, DNS), blocked count and recent blocks
                               from the firewall's snapshot; `enforced` is false where no firewall runs; `pictures`
                               says how many camera pictures were cleared to leave, and by which approvals (V7)

`guard` is the gateway's token check; it goes on the PUT and every POST. The gateway lets every write through
when no ATLAS_TOKEN is set, and "no token" must not mean "everyone on the Wi-Fi sees faces": so while
`signed_in_possible()` is false (no token configured) a view ticket is only given to a request that comes from
the box itself (loopback, and not passed on by a proxy), and refused (403) for everyone else. Set ATLAS_TOKEN to
get the clear live view on a phone or a laptop.

The endpoints only publish events: the policy changes because `policy.subscribe(bus)`
applies them, the same way it does on replay.
"""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from bus import EventBus
from contexts.vault.application.pictures import PictureLedger
from contexts.vault.application.policy import Policy
from contexts.vault.application.tickets import ViewTickets
from contexts.vault.application.wipe import WipeTokens, Wiper
from contracts.types import CameraId
from contracts.vault import PolicyChanged, RetentionChanged, WipeRequested


class PolicyBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    blur_faces: bool | None = None
    paused_cameras: list[CameraId] | None = None
    offline: bool | None = None
    keep_days: int | None = Field(default=None, ge=1)  # sent as null = keep forever; left out = unchanged
    live_faces: Literal["blurred", "clear", "owner"] | None = None  # the owner's own signed-in live view; left out = unchanged
    train_faces: Literal["blurred", "owner_clear"] | None = None  # the owner's own face in a training export; left out = unchanged


class WipeRequestBody(BaseModel):
    scope: Literal["all", "keyframes", "memory"] = "all"


class WipeBody(BaseModel):
    token: str


SCAN_FIRST = "Do the face scan first."

LOOPBACK = {"127.0.0.1", "::1", "::ffff:127.0.0.1", "localhost"}


def from_this_box(request: Request) -> bool:
    """The request was made on the box itself: a loopback peer, and not one a proxy passed on for someone else."""
    peer = request.client.host if request.client else ""
    forwarded = [h.strip() for h in request.headers.get("x-forwarded-for", "").split(",") if h.strip()]
    return peer in LOOPBACK and "forwarded" not in request.headers and all(h in LOOPBACK for h in forwarded)


def vault_router(
    policy: Policy,
    bus: EventBus,
    wiper: Wiper | None = None,
    guard: Callable | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    tokens: WipeTokens | None = None,
    pictures: PictureLedger | None = None,
    view_tickets: ViewTickets | None = None,
    signed_in_possible: Callable[[], bool] = lambda: True,
    owner: Callable[[], dict | None] | None = None,
) -> APIRouter:
    """`owner` is the box's answer to "has the owner done the face scan, and can the box tell them now?"
    ({scanned, ready, reason, demo}; the gateway hands in the identity service's). Without it the two owner-only
    choices cannot be switched on, and GET /vault/policy is the policy alone."""
    router = APIRouter(prefix="/vault", tags=["vault"])
    writes = [Depends(guard)] if guard else []
    tokens = tokens or WipeTokens()

    def who() -> dict:
        try:
            found = owner() if owner else None
        except Exception:  # the box cannot tell: as if nobody had scanned
            found = None
        return found or {"scanned": False, "ready": False, "reason": SCAN_FIRST, "demo": False}

    def answer() -> dict:
        out = policy.snapshot()
        if owner is None:
            return out
        me = who()
        wants_owner = out["live_faces"] == "owner" or out["train_faces"] == "owner_clear"
        return out | {"owner": me, "live_faces_now": "blurred" if out["live_faces"] == "owner" and not me["ready"] else out["live_faces"],
                      "faces_note": me["reason"] if wants_owner and not me["ready"] else None}

    @router.get("/policy")
    async def get_policy():
        return answer()

    @router.put("/policy", dependencies=writes)
    async def put_policy(body: PolicyBody):
        now = policy.state
        asks_owner = (body.live_faces == "owner" and now.live_faces != "owner") or (
            body.train_faces == "owner_clear" and now.train_faces != "owner_clear")
        if asks_owner and not who()["ready"]:  # only someone the box can tell apart can ask to be the one left clear
            raise HTTPException(409, who()["reason"] or SCAN_FIRST)
        events = [PolicyChanged(
            blur_faces=now.blur_faces if body.blur_faces is None else body.blur_faces,
            paused_cameras=now.paused_cameras if body.paused_cameras is None else sorted(set(body.paused_cameras)),
            offline=now.offline if body.offline is None else body.offline,
            live_faces=now.live_faces if body.live_faces is None else body.live_faces,  # carried, or a pause would reset it
            train_faces=now.train_faces if body.train_faces is None else body.train_faces,
        )]
        if "keep_days" in body.model_fields_set and body.keep_days != now.keep_days:
            events.append(RetentionChanged(keep_days=body.keep_days))
        await bus.publish_all(events)
        return answer()

    @router.post("/view-ticket", dependencies=writes)
    async def view_ticket(request: Request):
        """A ticket for the clear live view on the screen that asked. The module docstring says what a box with no token does."""
        if view_tickets is None:
            raise HTTPException(404, "this box gives no view tickets")
        if not signed_in_possible() and not from_this_box(request):
            raise HTTPException(403, "set ATLAS_TOKEN on the box and sign in to see faces in the live view")
        ticket, expires_at = view_tickets.issue(clock())
        return {"ticket": ticket, "expires_at": expires_at.isoformat(), "expires_in": view_tickets.lifetime_s}

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
            out |= {"removed": report.removed, "refused": report.refused, "identity_removed": report.identity_removed,
                    "live_memory_cleared": report.live_memory_cleared, "summary": report.summary()}
        return out

    @router.get("/egress")
    async def egress():
        """What the egress firewall (V6, jetson/egress.py) let out per group and blocked, from its snapshot file
        next to policy.json. Without one (a laptop, or the service is not running) `enforced: false` says so."""
        snap = policy.path.parent / "egress.json"
        extra = {"pictures": pictures.snapshot()} if pictures else {}
        try:
            data = json.loads(snap.read_text())
            updated = datetime.strptime(data["updated"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        except (OSError, ValueError, KeyError, TypeError):  # no file, or a half-written or foreign one: as good as none
            return {"allowed": [], "blocked": 0, "enforced": False, **extra}
        data["stale"] = (clock() - updated).total_seconds() > 120  # the service stopped writing: the rules still hold
        return data | extra

    return router
