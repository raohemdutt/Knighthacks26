"""Memory's read API (M4), mounted by the gateway.

    app.include_router(build_router(memory, data_dir / "frames"))

- GET /memory/episodes?from&to&camera&label   list[Episode]
- GET /memory/window?from&to                  EpisodeWindow (hour buckets)
- GET /memory/objects/{label}/last-seen       Sighting, 404 if never seen
- GET /memory/last-seen/{label}               the same, for marking it in the picture: zone, position, frame_ref, box or null,
                                              seen_at, and the other boxed things in that frame (anchors); 404 if never seen
- GET /memory/search?q=&limit=                list[SearchHit], best first (keywords, plus meaning when vectors exist: M5)
- GET /memory/items?from&to&kind              list[Item]
- GET /memory/keyframes/{ref}                 the JPEG behind frames:<camera>:<ts_ms>, as stored
- GET /memory/keyframes/{ref}?view=out        the copy that may leave the box (a phone alert's photo): through the vault's
                                              outbound gate, faces scrambled whatever "Blur faces before saving" says;
                                              503 and no picture when the gate cannot check it
- GET /memory/stats                           MemoryStats

`from` and `to` are ISO datetimes (no offset means UTC); leave them out for everything.
"""

import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response

from contexts.memory.application.reader import SqliteMemory
from contracts.memory import Episode, EpisodeWindow, Item, MemoryStats, SearchHit, Sighting

EARLIEST = datetime(1970, 1, 1, tzinfo=UTC)
LATEST = datetime(9999, 1, 1, tzinfo=UTC)
FRAME_REF = re.compile(r"frames:([\w.-]+):([\w-]+)")  # nothing that can climb out of data/frames

From = Query(EARLIEST, alias="from")
To = Query(LATEST, alias="to")


def _span(start: datetime, end: datetime) -> tuple[datetime, datetime]:
    """`from`/`to` as UTC. A date so far out that it has no UTC form (year 1 with a +14:00 offset) is a 422, not a 500."""
    try:
        return tuple(t.astimezone(UTC) if t.tzinfo else t.replace(tzinfo=UTC) for t in (start, end))
    except (OverflowError, ValueError):
        raise HTTPException(422, "from and to must be dates between years 1 and 9999 in UTC") from None


def _inside(path: Path, root: Path) -> bool:
    """True when `path` really is a file under `root`: a symlink in data/frames must not serve what it points at."""
    if path.is_symlink() or path.parent.is_symlink():
        return False
    return root.resolve() in path.resolve().parents


def build_router(memory: SqliteMemory, frames_dir: str | Path = "data/frames",
                 outbound: Callable[[bytes], bytes] | None = None) -> APIRouter:
    """`outbound(jpeg) -> jpeg` is the vault's gate for the phone's copy; it raises when the picture may not leave."""
    router = APIRouter(prefix="/memory", tags=["memory"])
    frames = Path(frames_dir)

    @router.get("/episodes")
    async def episodes(start: datetime = From, end: datetime = To, camera: str | None = None, label: str | None = None) -> list[Episode]:
        return memory.episodes(*_span(start, end), camera=camera, label=label)

    @router.get("/window")
    async def window(start: datetime = From, end: datetime = To) -> EpisodeWindow:
        return memory.window(*_span(start, end))

    @router.get("/objects/{label}/last-seen")
    async def last_seen(label: str) -> Sighting:
        sighting = memory.last_seen(label)
        if sighting is None:
            raise HTTPException(404, f"{label} has not been seen")
        return sighting

    @router.get("/last-seen/{label}")
    async def where(label: str) -> dict:
        """Zone, position in words, frame, box (or null) and seen_at; `anchors` are the other boxed things in that frame."""
        found = memory.where(label)
        if found is None:
            raise HTTPException(404, f"{label} has not been seen")
        return found

    @router.get("/search")
    async def search(q: str, limit: int = Query(5, ge=1, le=50)) -> list[SearchHit]:
        return await memory.search_hybrid(q, limit)  # embeds the query off the event loop (M5)

    @router.get("/items")
    async def items(start: datetime = From, end: datetime = To, kind: str | None = None) -> list[Item]:
        return memory.items(*_span(start, end), kind=kind)

    @router.get("/keyframes/{ref}")
    async def keyframe(ref: str, view: Literal["out"] | None = None) -> Response:
        m = FRAME_REF.fullmatch(ref)
        path = frames / m[1] / f"{m[2]}.jpg" if m and ".." not in m[1] else None
        if path is None or not path.is_file() or not _inside(path, frames):
            raise HTTPException(404, "no such keyframe")
        if view is None:
            return FileResponse(path, media_type="image/jpeg")
        if outbound is None:
            raise HTTPException(503, "no outbound gate on this box: the picture stays here")
        try:
            jpeg = await run_in_threadpool(outbound, path.read_bytes())  # decode, find faces, scramble: off the event loop
        except Exception:  # never the stored file instead
            raise HTTPException(503, "this picture could not be checked for faces, so it stays on the box") from None
        return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @router.get("/stats")
    async def stats() -> MemoryStats:
        return memory.stats()

    return router
