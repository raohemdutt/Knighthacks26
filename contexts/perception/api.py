"""Owner recognition and face profile API for the phone's People screen (P9).

    app.include_router(identity_router(atlas.identity, atlas.bus, guard=authorized, circle_members=lambda: [...]))

- GET    /identity/settings         {identity_enabled, available, recognizer, everyone, expressions_available,
                                     expression_reader, photos_available[, reason]}
- PUT    /identity/settings         {identity_enabled?: bool, everyone?: bool, confirm?: bool}; 409 when the model for it
                                    is not on this box. `everyone: true` is always refused (409): the room mode read
                                    people who never did the face scan and is retired; `everyone` in the answer is
                                    always false
- POST   /identity/enroll/start     {name, consent_ack: true[, member_id][, camera_id][, circle_member_id]} -> the enrolment
                                    status; 409 while switched off; 422 without `consent_ack: true`
- POST   /identity/enroll/scan/start   {name?, member_id?, consent_ack: true[, circle_member_id]} -> {session, count, needed,
                                    poses, want, prompt, done}: a guided scan with the phone's own camera (the usual way)
- POST   /identity/enroll/scan/frame   {session, photo: base64 or data: URL} -> the same, plus {accepted, reason, message,
                                    pose} for this still; 404 for a scan that is not open, 409 when it timed out
- POST   /identity/enroll/scan/finish  {session} -> what /identity/enroll/photos returns (the same checks and rules)
- POST   /identity/enroll/scan/cancel  {session} -> {cancelled: true}: what was collected is dropped
- GET    /identity/enroll/status    {state: idle|collecting|done|failed, frames_done, frames_needed, hint, error, ...}
- POST   /identity/enroll/photos    add 1 to 12 photos to a member (`member_id`), or keep 3 to 12 photos for someone new,
                                    who is then "not yet scanned" (`scanned: false` and a `message` in the answer): kept,
                                    but not recognised and not read until a scan of that member finishes. The body is
                                    JSON {name?, member_id?, circle_member_id?, consent_ack: true, photos: [base64 or data: URL, ...]} or
                                    multipart/form-data with the same fields and one `photos` file part per picture.
                                    -> {state: done|failed, member_id, accepted, rejected, needed, exemplars, error,
                                        scanned, message, photos: [{index, accepted, reason, message}]}
                                    409 while switched off (or without OpenCV), 413 when too large, 422 for a bad body
- GET    /identity/members          [{member_id, name, enrolled_at, updated_at, photos, source, expressions,
                                    circle_member_id, scanned, recognised}]: the only place a name is returned, never the template. A row a
                                    friend's box shared (source "circle", face share) also has {shared_by, received_at,
                                    expires_at}; it can be deleted here and not changed (409)
- PUT    /identity/members/{id}     {expressions?: bool, name?: str, circle_member_id?: str}: "read my expressions" for this
                                    member (409 when the expression model is not on this box, and 409 "Finish the face
                                    scan first." for a member who has not scanned); which Circle member they
                                    are ("" unlinks; 422 when the Circle has no such member)
- GET    /identity/present          {present: [{member_id, circle_member_id, cameras}]}: who a camera recognises right now
- DELETE /identity/members/{id}     removes the template, the exemplars and the member's settings
- GET    /identity/expressions      ?member_id=&hours=: the stretches of apparent expression for one enrolled member, or,
                                    without member_id, the anonymous room aggregates an older log still holds (none
                                    are made any more)

`guard` is the gateway's token check; it goes on every PUT, POST and DELETE.

`circle_members` gives the configured Circle's member ids (None or empty: no Circle here, any well-formed id is
taken). The link is an id next to the template; nothing in this router sends face data anywhere or fetches any.
(Face share, off by default, is the one way a template crosses boxes: contexts/circle/face_share_api.py.)

The photos of an upload and the stills of a scan exist only in this process's memory while the request is answered: the body is read
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
MAX_FRAME_BYTES = MAX_PHOTO_BYTES * 4 // 3 + 4096  # one still of a scan, as base64 in JSON
PHOTO_FIELDS = {"name", "member_id", "consent_ack", "circle_member_id", "photos"}


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
    circle_member_id: str | None = None  # which Circle member this is (optional): an id, nothing else


class ScanStartBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None  # needed for someone new; left out when adding to `member_id`
    member_id: str | None = None
    consent_ack: bool = False
    circle_member_id: str | None = None


class ScanBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session: str


class MemberBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expressions: bool | None = None
    name: str | None = None
    circle_member_id: str | None = None  # "" takes the link away


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
    for key in ("name", "member_id", "circle_member_id"):
        if fields.get(key) is not None and not isinstance(fields[key], str):
            raise ValueError(f"{key} must be text")
    if len(photos) > MAX_PHOTOS:
        raise ValueError(f"send at most {MAX_PHOTOS} photos")
    if any(len(p) > MAX_PHOTO_BYTES for p in photos):
        raise ValueError(f"a photo is larger than {MAX_PHOTO_BYTES // (1024 * 1024)} MB")
    consent = fields.get("consent_ack")
    return {"name": fields.get("name") or None, "member_id": fields.get("member_id") or None,
            "circle_member_id": fields.get("circle_member_id") or None,
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


def identity_router(identity: IdentityService, bus: Any, guard: Callable | None = None,
                    circle_members: Callable[[], list[str] | None] | None = None) -> APIRouter:
    router = APIRouter(prefix="/identity", tags=["identity"])
    writes = [Depends(guard)] if guard else []

    def linked(circle_member_id: str | None) -> str | None:
        """The id as given, once the configured Circle is known to have that member."""
        roster = circle_members() if circle_members and circle_member_id else None
        if roster and circle_member_id not in roster:
            raise ValueError("the Circle has no such member")
        return circle_member_id

    async def body_of(request: Request, limit: int) -> bytes:
        """The request's body, in memory, refused once it is larger than `limit`."""
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > limit:
            raise HTTPException(413, "too large: send fewer or smaller photos")
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > limit:
                raise HTTPException(413, "too large: send fewer or smaller photos")
            chunks.append(chunk)
        return b"".join(chunks)

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
            return identity.start_enrollment(body.member_id, body.name, body.camera_id, consent_ack=body.consent_ack,
                                             circle_member_id=linked(body.circle_member_id))
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
        raw = await body_of(request, MAX_BODY_BYTES)
        try:
            fields, photos = parse_photos(request.headers.get("content-type", ""), raw)
            del raw
            result, events = await run_in_threadpool(identity.enroll_photos, fields["member_id"], fields["name"], photos,
                                                     consent_ack=fields["consent_ack"],
                                                     circle_member_id=linked(fields["circle_member_id"]))
        except IdentityError as exc:
            raise HTTPException(409, str(exc)) from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        await bus.publish_all(events)
        return result

    @router.post("/enroll/scan/start", dependencies=writes)
    async def scan_start(body: ScanStartBody):
        """Open a guided scan: the phone then sends stills from its own camera, one at a time."""
        try:
            return identity.scan_start(body.member_id, body.name, consent_ack=body.consent_ack,
                                       circle_member_id=linked(body.circle_member_id))
        except IdentityError as exc:
            raise HTTPException(409, str(exc)) from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    @router.post("/enroll/scan/frame", dependencies=writes)
    async def scan_frame(request: Request):
        """One still of a scan: accepted or not, why, and what to do next. The still is never stored."""
        if not identity.enabled:
            raise HTTPException(409, "recognition is switched off")
        raw = await body_of(request, MAX_FRAME_BYTES)
        try:  # read by hand, so a refusal can never quote the picture back
            sent = json.loads(raw)
            del raw
            if not isinstance(sent, dict) or set(sent) != {"session", "photo"} or not isinstance(sent["session"], str):
                raise ValueError("send {session, photo}")
            still = _decode(sent["photo"])
            if len(still) > MAX_PHOTO_BYTES:
                raise HTTPException(413, "too large: send a smaller still")
            return await run_in_threadpool(identity.scan_frame, sent["session"], still)
        except KeyError:
            raise HTTPException(404, "no such scan: start again") from None
        except IdentityError as exc:
            raise HTTPException(409, str(exc)) from None
        except ValueError as exc:
            raise HTTPException(422, str(exc) if not isinstance(exc, json.JSONDecodeError) else "the body is not JSON") from None

    @router.post("/enroll/scan/finish", dependencies=writes)
    async def scan_finish(body: ScanBody):
        try:
            result, events = await run_in_threadpool(identity.scan_finish, body.session)
        except KeyError:
            raise HTTPException(404, "no such scan: start again") from None
        except IdentityError as exc:
            raise HTTPException(409, str(exc)) from None
        await bus.publish_all(events)
        return result

    @router.post("/enroll/scan/cancel", dependencies=writes)
    async def scan_cancel(body: ScanBody):
        identity.scan_cancel(body.session)
        return {"cancelled": True}

    @router.get("/present")
    async def present():
        return {"present": identity.present()}

    @router.get("/members")
    async def members():
        return identity.members()

    @router.put("/members/{member_id}", dependencies=writes)
    async def put_member(member_id: str, body: MemberBody):
        try:
            events = identity.update_member(member_id, expressions=body.expressions, name=body.name,
                                            circle_member_id=linked(body.circle_member_id) if body.circle_member_id else body.circle_member_id)
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
