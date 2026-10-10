"""Group brain endpoints (G4) for the phone: the member's opt-in switch, and one planning question.

    app.include_router(group_brain_router(atlas.group_brain, guard=authorized))

- GET  /api/circle/brain          {"configured", "circle_id", "member_id", "opted_in", "opt_in_ref", "changed_at",
                                   "node", "host", "badge"}: whether this member asks the group brain and where the text
                                   would go. 200 in every state (no Circle: configured false); calls nothing.
- POST /api/circle/brain/consent  {"on": true|false} -> the same status after the change, which is an event in the log
                                  (circle.brain.consent.changed, source "phone").
- POST /api/circle/brain/ask      {"question", "facts": [text, ...], "purpose"?} -> {"answer", "badge", "ask_id",
                                  "opt_in_ref", "model", "latency_ms"}. 403 not opted in, 409 another question is in
                                  flight, 422 not plain text, too long or empty, 503 the group brain did not answer
                                  ({"kind", "message", "ask_id"}; answer with the box's own model instead).
Both POSTs need the token like other writes, and answer 404 when the Circle is not configured. Every field is checked
as text as it arrives (strict strings), and the group brain checks again before anything leaves.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, StrictBool, StrictStr

from contexts.circle.application.group_brain import BADGE, BrainRefused, BrainUnavailable, GroupBrain

REFUSED_STATUS = {"not_opted_in": 403, "busy": 409, "not_text": 422, "too_long": 422, "empty": 422}


class ConsentBody(BaseModel):
    on: StrictBool


class AskBody(BaseModel):
    question: StrictStr
    facts: list[StrictStr] = Field(default_factory=list)
    purpose: StrictStr = "plan"


def group_brain_router(brain: GroupBrain | None, *, guard: Callable | None = None) -> APIRouter:
    """brain is None when the box has no Circle; the GET still answers so the screen can say so."""
    router = APIRouter()
    writes = [Depends(guard)] if guard else []

    def configured() -> GroupBrain:
        if brain is None:
            raise HTTPException(404, "the Circle is not configured")
        return brain

    @router.get("/api/circle/brain")
    def status() -> dict[str, Any]:
        if brain is None:
            return {"configured": False, "circle_id": None, "member_id": None, "opted_in": False, "opt_in_ref": None,
                    "changed_at": None, "node": None, "host": None, "badge": BADGE}
        return {"configured": True, **brain.status()}

    @router.post("/api/circle/brain/consent", dependencies=writes)
    async def consent(body: ConsentBody) -> dict[str, Any]:
        found = configured()
        await found.set_consent(body.on, "phone")
        return {"configured": True, **found.status()}

    @router.post("/api/circle/brain/ask", dependencies=writes)
    async def ask(body: AskBody) -> dict[str, Any]:
        found = configured()
        try:
            answer = await found.ask(body.question, body.facts, purpose=body.purpose)
        except BrainRefused as exc:
            raise HTTPException(REFUSED_STATUS[exc.reason], {"reason": exc.reason, "message": str(exc)}) from None
        except BrainUnavailable as exc:
            raise HTTPException(503, {"kind": exc.kind, "message": str(exc), "ask_id": exc.ask_id}) from None
        except ValueError as exc:  # a purpose that is not a short word
            raise HTTPException(422, {"reason": "purpose", "message": str(exc)}) from None
        return {"answer": answer.text, "badge": answer.badge, "ask_id": answer.ask_id, "opt_in_ref": answer.opt_in_ref,
                "model": answer.model, "latency_ms": answer.latency_ms}

    return router
