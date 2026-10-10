"""Owner recognition API for the phone's People screen (P9).

    app.include_router(identity_router(atlas.identity, atlas.bus, guard=authorized))

- GET    /identity/settings         {identity_enabled, available, recognizer[, reason]}
- PUT    /identity/settings         {identity_enabled: bool}; 409 when the recognition model is not on this box
- POST   /identity/enroll/start     {name[, member_id][, camera_id]} -> the enrolment status; 409 while switched off
- GET    /identity/enroll/status    {state: idle|collecting|done|failed, frames_done, frames_needed, hint, error, ...}
- GET    /identity/members          [{member_id, name, enrolled_at}]: the only place a name is returned, never the template
- DELETE /identity/members/{id}     removes the template

`guard` is the gateway's token check; it goes on the PUT, the POST and the DELETE.
"""

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict

from contexts.perception.application.identity import IdentityError, IdentityService


class SettingsBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    identity_enabled: bool


class EnrollBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    member_id: str | None = None  # left out: the box makes one up, so the id never has to be the name
    camera_id: str | None = None  # left out: whichever camera sees one face first


def identity_router(identity: IdentityService, bus: Any, guard: Callable | None = None) -> APIRouter:
    router = APIRouter(prefix="/identity", tags=["identity"])
    writes = [Depends(guard)] if guard else []

    @router.get("/settings")
    async def get_settings():
        return identity.settings()

    @router.put("/settings", dependencies=writes)
    async def put_settings(body: SettingsBody):
        try:
            events = identity.set_enabled(body.identity_enabled)
        except IdentityError as exc:
            raise HTTPException(409, str(exc)) from None
        await bus.publish_all(events)
        return identity.settings()

    @router.post("/enroll/start", dependencies=writes)
    async def enroll_start(body: EnrollBody):
        try:
            return identity.start_enrollment(body.member_id, body.name, body.camera_id)
        except IdentityError as exc:
            raise HTTPException(409, str(exc)) from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    @router.get("/enroll/status")
    async def enroll_status():
        return identity.enrollment_status()

    @router.get("/members")
    async def members():
        return identity.members()

    @router.delete("/members/{member_id}", dependencies=writes)
    async def delete_member(member_id: str):
        try:
            events = identity.remove(member_id)
        except KeyError:
            raise HTTPException(404, "no such member") from None
        await bus.publish_all(events)
        return {"removed": member_id}

    return router
