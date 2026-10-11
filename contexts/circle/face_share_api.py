"""Face share endpoints for the phone's People screen and the Circle's People tab.

    app.include_router(face_share_router(atlas.face_share, atlas.bus, guard=authorized))

- GET  /api/circle/face-share                     who can recognise whom, both ways:
    {"enabled": true, "circle_id", "member_id", "expiry_days", "recognition_enabled", "recognition_available",
     "explanation", "me": {"fingerprint", "member": {"member_id", "name"}|null, "can_share", "why_not"},
     "friends": [{"member_id", "display_name", "renewal_offered",
                  "key": {"known", "fingerprint", "changed", "new_fingerprint"},
                  "outgoing": {"state": "none"|"waiting_approval"|"shared", "share_id", "shared_at", "expires_at", "action_id"},
                  "incoming": {"state": "none"|"offered"|"accepted", "received_at", "expires_at", "offered_at", "usable"}}]}
- POST /api/circle/face-share/{friend}/share      "Let <friend>'s Atlas recognise me": an approval card on this box,
                                                  {"action_id"}. Nothing is sent until the card is approved.
- POST /api/circle/face-share/{friend}/revoke     take it back now: {"revoked": friend}
- POST /api/circle/face-share/{friend}/review     raise the Accept card again for an offer that is waiting: {"action_id"}
- POST /api/circle/face-share/{friend}/forget     delete what the friend shared, or turn their offer down: {"forgotten": friend}
- POST /api/circle/face-share/{friend}/trust-key  {"fingerprint"}: after comparing it with the friend, take their
                                                  changed box key: {"trusted": friend}

Accepting or declining an incoming share is the approval card itself (POST /api/approvals/...).

While face sharing is switched off, or the box has no Circle, EVERY endpoint here answers 409 with the plain reason
as `detail`. 404: no such friend, or nothing to revoke, review or forget. 409 also when it cannot be done now (the
friend's box has no key yet, nobody is enrolled here as this box's own person, ...). 503 when the Circle is offline.
No answer ever holds a template, a vector, a key or ciphertext: ids, names from the config, states, times and key
fingerprints only. The POSTs need the token like other writes.
"""

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, StrictStr

from contexts.circle.application.face_share import NO_CIRCLE, OFF, FaceShare, FaceShareError, FaceShareOff
from contexts.circle.application.sharing import CircleOffline


class TrustBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fingerprint: StrictStr


def face_share_router(share: FaceShare | None, bus: Any, *, guard: Callable | None = None, reason: str | None = None) -> APIRouter:
    """`share` is None when the box has no Circle (or no recognition service): every endpoint then answers 409."""
    router = APIRouter(prefix="/api/circle/face-share")
    writes = [Depends(guard)] if guard else []

    async def run(work: Callable, *args: Any) -> Any:
        """The service call off the event loop (it may talk to the Circle), with its refusals as status codes."""
        if share is None:
            raise HTTPException(409, reason or NO_CIRCLE)
        if not share.enabled:
            raise HTTPException(409, OFF)
        try:
            return await run_in_threadpool(work, *args)
        except FaceShareOff as exc:
            raise HTTPException(409, str(exc)) from None
        except FaceShareError as exc:
            raise HTTPException(409, str(exc)) from None
        except KeyError:
            raise HTTPException(404, "no such friend, or nothing there to do this to") from None
        except CircleOffline as exc:
            raise HTTPException(503, f"the Circle is offline: {exc}") from None

    @router.get("")
    async def status() -> dict:
        return await run(share.status if share else None)

    @router.post("/{friend}/share", dependencies=writes)
    async def propose(friend: str) -> dict:
        card = await run(share.propose_share if share else None, friend)
        await bus.publish(card)
        return {"action_id": card.action_id}

    @router.post("/{friend}/revoke", dependencies=writes)
    async def revoke(friend: str) -> dict:
        events = await run(share.revoke if share else None, friend)
        await bus.publish_all(events)
        return {"revoked": friend}

    @router.post("/{friend}/review", dependencies=writes)
    async def review(friend: str) -> dict:
        card = await run(share.review if share else None, friend)
        await bus.publish(card)
        return {"action_id": card.action_id}

    @router.post("/{friend}/forget", dependencies=writes)
    async def forget(friend: str) -> dict:
        events = await run(share.forget if share else None, friend)
        await bus.publish_all(events)
        return {"forgotten": friend}

    @router.post("/{friend}/trust-key", dependencies=writes)
    async def trust(friend: str, body: TrustBody) -> dict:
        await run(share.trust_key if share else None, friend, body.fingerprint)
        return {"trusted": friend}

    return router
