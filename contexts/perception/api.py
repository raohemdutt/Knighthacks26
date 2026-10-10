"""Owner recognition and face profile API for the phone's People screen (P9).

    app.include_router(identity_router(atlas.identity, atlas.bus, guard=authorized))

- GET    /identity/settings         {identity_enabled, available, recognizer, everyone, expressions_available,
                                     expression_reader, photos_available[, reason]}
- PUT    /identity/settings         {identity_enabled?: bool, everyone?: bool, confirm?: bool}; 409 when the model for it
                                    is not on this box or recognition is off; 422 when `everyone: true` comes without
                                    `confirm: true`
- POST   /identity/enroll/start     {name, consent_ack: true[, member_id][, camera_id]} -> the enrolment status; 409 while
                                    switched off; 422 without `consent_ack: true`
- GET    /identity/enroll/status    {state: idle|collecting|done|failed, frames_done, frames_needed, hint, error, ...}
- POST   /identity/enroll/photos    enrol from 3 to 12 photos, or add 1 to 12 photos to a member (`member_id`). The body is
                                    JSON {name?, member_id?, consent_ack: true, photos: [base64 or data: URL, ...]} or
                                    multipart/form-data with the same fields and one `photos` file part per picture.
                                    -> {state: done|failed, member_id, accepted, rejected, needed, exemplars, error,
                                        photos: [{index, accepted, reason, message}]}
                                    409 while switched off (or without OpenCV), 413 when too large, 422 for a bad body
- GET    /identity/members          [{member_id, name, enrolled_at, updated_at, photos, source, expressions}]: the only
                                    place a name is returned, never the template
- PUT    /identity/members/{id}     {expressions?: bool, name?: str}: "read my expressions" for this member; 409 when the
                                    expression model is not on this box
- DELETE /identity/members/{id}     removes the template, the exemplars and the member's settings
- GET    /identity/expressions      ?member_id=&hours=: the stretches of apparent expression for one enrolled member, or,
                                    without member_id, the anonymous room aggregate of the "everyone" mode

`guard` is the gateway's token check; it goes on every PUT, POST and DELETE.

The photos of an upload exist only in this process's memory while the request is answered: the body is read
and split here (not by a form parser that may spool large parts to a temporary file), handed to the service,
and dropped. They are never written, logged or echoed back, also not in an error.
"""

import base64
import binascii
import json
import re
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict

from contexts.perception.application.identity import MAX_PHOTO_BYTES, MAX_PHOTOS, IdentityError, IdentityService

MAX_BODY_BYTES = 64 * 1024 * 1024  # twelve phone photos as base64 fit; anything larger is refused before it is read in
PHOTO_FIELDS = {"name", "member_id", "consent_ack", "photos"}


class SettingsBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    identity_enabled: bool | None = None
    everyone: bool | None = None  # the anonymous room mode; switching it on needs `confirm`
    confirm: bool = False


class EnrollBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    member_id: str | None = None  # left out: the box makes one up, so the id never has to be the name
    camera_id: str | None = None  # left out: whichever camera sees one face first
    consent_ack: bool = False  # the person being enrolled agrees; must be true


class MemberBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expressions: bool | None = None
    name: str | None = None


def parse_photos(content_type: str, body: bytes) -> tuple[dict[str, Any], list[bytes]]:
    """({name, member_id, consent_ack}, the photos' bytes) from a JSON or multipart body. A ValueError's
    message never holds anything from the body."""
    kind = content_type.split(";", 1)[0].strip().lower()
    if kind == "multipart/form-data":
        fields, photos = _multipart(content_type, body)
    elif kind in ("application/json", ""):
        try:
            raw = json.loads(body)
        except ValueError:
            raise ValueError("the body is not JSON") from None
        if not isinstance(raw, dict) or not isinstance(raw.get("photos"), list):
            raise ValueError("send {name, consent_ack, photos: [...]}")
        if set(raw) - PHOTO_FIELDS:
            raise ValueError("unknown field; send name, member_id, consent_ack and photos")
        fields, photos = {k: v for k, v in raw.items() if k != "photos"}, [_decode(p) for p in raw["photos"]]
    else:
        raise ValueError("send JSON or multipart/form-data")
    for key in ("name", "member_id"):
        if fields.get(key) is not None and not isinstance(fields[key], str):
            raise ValueError(f"{key} must be text")
    if len(photos) > MAX_PHOTOS:
        raise ValueError(f"send at most {MAX_PHOTOS} photos")
    if any(len(p) > MAX_PHOTO_BYTES for p in photos):
        raise ValueError(f"a photo is larger than {MAX_PHOTO_BYTES // (1024 * 1024)} MB")
    consent = fields.get("consent_ack")
    return {"name": fields.get("name") or None, "member_id": fields.get("member_id") or None,
            "consent_ack": consent is True or (isinstance(consent, str) and consent.strip().lower() in ("true", "1", "on", "yes"))}, photos


def _decode(photo: Any) -> bytes:
    if not isinstance(photo, str):
        raise ValueError("each photo must be base64 text")
    text = photo.split(",", 1)[1] if photo.startswith("data:") and "," in photo else photo
    try:
        return base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("a photo is not valid base64") from None


def _multipart(content_type: str, body: bytes) -> tuple[dict[str, Any], list[bytes]]:
    """Split a multipart/form-data body in memory. Parts with a file name are photos, the rest are fields."""
    found = re.search(r'boundary="?([^";,\s]+)"?', content_type)
    if not found:
        raise ValueError("the multipart body has no boundary")
    fields: dict[str, Any] = {}
    photos: list[bytes] = []
    for part in (b"\r\n" + body).split(b"\r\n--" + found.group(1).encode())[1:]:
        if part.startswith(b"--"):
            break  # the closing boundary
        head, sep, content = part.partition(b"\r\n\r\n")
        disposition = re.search(rb'content-disposition:[^\r\n]*', head, re.IGNORECASE)
        name = re.search(rb'[; ]name="([^"]*)"', disposition.group(0)) if disposition and sep else None
        if name is None:
            continue
        if re.search(rb'filename\*?=', disposition.group(0)):
            photos.append(content)
        elif name.group(1).decode("utf-8", "replace") in PHOTO_FIELDS:
            fields[name.group(1).decode()] = content.decode("utf-8", "replace")
    return fields, photos


def identity_router(identity: IdentityService, bus: Any, guard: Callable | None = None) -> APIRouter:
    router = APIRouter(prefix="/identity", tags=["identity"])
    writes = [Depends(guard)] if guard else []

    @router.get("/settings")
    async def get_settings():
        return identity.settings()

    @router.put("/settings", dependencies=writes)
    async def put_settings(body: SettingsBody):
        if body.identity_enabled is None and body.everyone is None:
            raise HTTPException(422, "send identity_enabled or everyone")
        events, refused = [], None
        try:
            if body.identity_enabled is not None:
                events = identity.set_enabled(body.identity_enabled)
            if body.everyone is not None:
                events = identity.set_everyone(body.everyone, body.confirm)
        except IdentityError as exc:
            refused = HTTPException(409, str(exc))
        except ValueError as exc:
            refused = HTTPException(422, str(exc))
        await bus.publish_all(events[-1:])  # what did change is in the log, also when the second half was refused
        if refused:
            raise refused
        return identity.settings()

    @router.post("/enroll/start", dependencies=writes)
    async def enroll_start(body: EnrollBody):
        try:
            return identity.start_enrollment(body.member_id, body.name, body.camera_id, consent_ack=body.consent_ack)
        except IdentityError as exc:
            raise HTTPException(409, str(exc)) from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    @router.get("/enroll/status")
    async def enroll_status():
        return identity.enrollment_status()

    @router.post("/enroll/photos", dependencies=writes)
    async def enroll_photos(request: Request):
        """Enrol from uploaded photos, or add photos to someone enrolled. The photos are never stored."""
        if not identity.enabled:  # before the pictures are even read in
            raise HTTPException(409, "recognition is switched off")
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > MAX_BODY_BYTES:
            raise HTTPException(413, "too large: send fewer or smaller photos")
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_BODY_BYTES:
                raise HTTPException(413, "too large: send fewer or smaller photos")
            chunks.append(chunk)
        try:
            fields, photos = parse_photos(request.headers.get("content-type", ""), b"".join(chunks))
            del chunks
            result, events = await run_in_threadpool(identity.enroll_photos, fields["member_id"], fields["name"], photos,
                                                     consent_ack=fields["consent_ack"])
        except IdentityError as exc:
            raise HTTPException(409, str(exc)) from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        await bus.publish_all(events)
        return result

    @router.get("/members")
    async def members():
        return identity.members()

    @router.put("/members/{member_id}", dependencies=writes)
    async def put_member(member_id: str, body: MemberBody):
        try:
            events = identity.update_member(member_id, expressions=body.expressions, name=body.name)
        except KeyError:
            raise HTTPException(404, "no such member") from None
        except IdentityError as exc:
            raise HTTPException(409, str(exc)) from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        await bus.publish_all(events)
        return next(m for m in identity.members() if m["member_id"] == member_id)

    @router.delete("/members/{member_id}", dependencies=writes)
    async def delete_member(member_id: str):
        try:
            events = identity.remove(member_id)
        except KeyError:
            raise HTTPException(404, "no such member") from None
        await bus.publish_all(events)
        return {"removed": member_id}

    @router.get("/expressions")
    async def expressions(member_id: str | None = None, hours: float = Query(6.0, gt=0, le=48)):
        try:
            return identity.expressions(member_id or None, hours)
        except KeyError:
            raise HTTPException(404, "no such member") from None

    return router
