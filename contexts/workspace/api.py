"""The file inbox over HTTP (W1).

POST /api/files                 the file as the raw body; its name in ?name= or the X-File-Name header (URL-encoded),
                                its type in Content-Type -> the record (200, or 200 with "duplicate": true)
POST /api/files/share           multipart/form-data from the phone's share sheet (the PWA's share_target): every file
                                part is added; shared text or a link with no file becomes a small .txt note. Answers
                                303 to /files/ so the phone lands on the list.
GET  /api/files                 the inbox, newest first
GET  /api/files/{file_id}       the file itself, decrypted on the way out

Writes need the bearer token when ATLAS_TOKEN is set. The share sheet cannot send one, so with a token set,
share from the Files screen instead.
"""

from collections.abc import Callable
from email.parser import BytesParser
from email.policy import HTTP
from typing import Any
from urllib.parse import parse_qs, quote, unquote

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import RedirectResponse, Response

from contexts.workspace.application.inbox import FileInbox, FileTooLarge
from contexts.workspace.domain.files import MAX_BYTES, FileRecord

SHARE_LIMIT = MAX_BYTES + 1024 * 1024  # one file plus the form around it


async def body_of(request: Request, limit: int) -> bytes:
    """The request's body, in memory, refused once it is larger than `limit`."""
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise HTTPException(413, f"too large: files are limited to {MAX_BYTES // (1024 * 1024)} MB")
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise HTTPException(413, f"too large: files are limited to {MAX_BYTES // (1024 * 1024)} MB")
        chunks.append(chunk)
    return b"".join(chunks)


def form_parts(content_type: str, body: bytes) -> tuple[dict[str, str], list[tuple[str | None, str, bytes]]]:
    """(text fields, [(filename, content type, bytes)]) of a multipart/form-data body, with the standard library.
    A url-encoded form (a share of only text or a link) has fields and no files."""
    if content_type.lower().startswith("application/x-www-form-urlencoded"):
        return {k: v[0] for k, v in parse_qs(body.decode("utf-8", "replace"), keep_blank_values=True).items()}, []
    if "multipart/form-data" not in content_type.lower() or "boundary=" not in content_type.lower():
        raise HTTPException(415, "send multipart/form-data")
    msg = BytesParser(policy=HTTP).parsebytes(b"Content-Type: " + content_type.encode("latin-1") + b"\r\n\r\n" + body)
    if not msg.is_multipart():
        raise HTTPException(400, "the multipart body could not be read")
    fields: dict[str, str] = {}
    files: list[tuple[str | None, str, bytes]] = []
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        data = part.get_payload(decode=True) or b""
        filename = part.get_filename()
        if filename is not None:
            if data:
                files.append((filename, part.get_content_type(), data))
        elif name:
            fields[str(name)] = data.decode(part.get_content_charset() or "utf-8", "replace")
    return fields, files


def files_router(inbox: FileInbox, bus: Any, guard: Callable | None = None) -> APIRouter:
    router = APIRouter(prefix="/api/files", tags=["workspace"])
    writes = [Depends(guard)] if guard else []

    async def add(name: str | None, content_type: str | None, data: bytes) -> tuple[FileRecord, bool]:
        try:
            rec, event = inbox.add(name, content_type, data)
        except FileTooLarge as exc:
            raise HTTPException(413, str(exc)) from None
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        if event:
            await bus.publish(event)
        return rec, event is None

    @router.get("", response_model=list[FileRecord])
    async def list_files():
        return inbox.list()

    @router.post("", dependencies=writes)
    async def upload(request: Request, name: str | None = Query(default=None)):
        data = await body_of(request, MAX_BYTES)
        filename = name or unquote(request.headers.get("x-file-name", "")) or None
        rec, duplicate = await add(filename, request.headers.get("content-type"), data)
        return {**rec.model_dump(mode="json"), "duplicate": duplicate}

    @router.post("/share", dependencies=writes)
    async def share(request: Request):
        fields, files = form_parts(request.headers.get("content-type", ""), await body_of(request, SHARE_LIMIT))
        added = 0
        for filename, ctype, data in files:
            await add(filename, ctype, data)
            added += 1
        note = "\n".join(v.strip() for k in ("title", "text", "url") if (v := fields.get(k, "")).strip())
        if not files and note:  # a shared link or a bit of text: kept as a note, searchable like a file
            title = fields.get("title", "").strip() or "Shared note"
            await add(f"{title[:80]}.txt", "text/plain", note.encode("utf-8"))
            added += 1
        if not added:
            raise HTTPException(400, "nothing was shared: no file, text or link")
        return RedirectResponse(f"/files/?shared={added}", status_code=303)

    @router.get("/{file_id}")
    async def download(file_id: str):
        rec = inbox.get(file_id)
        if rec is None:
            raise HTTPException(404, "no such file")
        try:
            data = inbox.read(file_id)
        except KeyError:
            raise HTTPException(404, "no such file") from None
        disposition = f"attachment; filename*=UTF-8''{quote(rec.name)}"
        return Response(data, media_type=rec.content_type, headers={"Content-Disposition": disposition,
                                                                     "X-Content-Type-Options": "nosniff"})

    return router
