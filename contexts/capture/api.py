"""GET /capture/status: is each camera delivering frames, and is the detector keeping up.

    app.include_router(capture_router(atlas.capture))

{"cameras": {"cam-desk": {"status": "ok", "fps_actual": 15.0, "dropped": 0, "detected": 912, "errors": 0,
                          "detector_fps": 41.2, "detector_ms": 24.3}}}

`detector_fps` is what the detector itself can do (1000 / its mean milliseconds over the newest 100 calls), not the
rate the pipeline paces it at; null until it has run.

Also for the Live screen (X3): GET /capture/<camera>/snapshot.jpg (a blurred JPEG) and GET /capture/<camera>/tracks,
and GET /capture/live/<camera>[?fps=5] as a motion-JPEG stream (C5).
`status` is the reader's: ok, degraded (under 70% of the target rate) or down (no frame for 3 s).

Both pictures take `?ticket=`: with a view ticket the Vault gave the owner's signed-in screen (POST /vault/view-ticket)
the faces are not blurred, unless the owner chose otherwise (`live_faces` "owner": only the scanned owner's own face is
left clear, every other face is blurred; "blurred": everyone). Without one, or with one that is wrong or expired, the
answer is the blurred picture exactly as before. `clear` is the Vault's check of a ticket.

Both also take `?blur=1`, for a picture that will be kept or shared (jetson/record_clip.py): the faces are blurred
with the strict blur whatever the policy says, and a ticket on the same request is ignored.
"""

import asyncio
from collections.abc import Callable

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse

from contexts.capture.application.service import CaptureService


def capture_router(capture: CaptureService, clear: Callable[[str], bool] = lambda ticket: False) -> APIRouter:
    router = APIRouter()

    @router.get("/capture/status")
    def status() -> dict:
        return {"cameras": capture.stats()}

    @router.get("/capture/{camera_id}/snapshot.jpg")
    def snapshot(camera_id: str, ticket: str | None = None, blur: bool = False) -> Response:
        """The newest frame (faces blurred unless the Vault policy turns blur off, and always with ?blur=1); 404 if
        paused or no frame yet."""
        if blur:  # asked for a picture to keep or share: blurred, and a ticket does not change that
            jpeg = capture.snapshot(camera_id, blur=True)
        else:
            jpeg = capture.snapshot(camera_id, clear=True) if ticket and clear(ticket) else capture.snapshot(camera_id)
        if jpeg is None:
            raise HTTPException(404, "no picture from this camera right now")
        return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @router.get("/capture/live/{camera_id}")
    async def live(camera_id: str, request: Request, fps: float = 5.0, frames_max: int | None = Query(None, alias="max"),
                   ticket: str | None = None, blur: bool = False) -> StreamingResponse:
        """A motion-JPEG stream (multipart/x-mixed-replace) of the camera, faces blurred like the snapshot (`blur=1`
        included); `fps` up to 15; `max` ends it after that many frames (for a test or a one-off `curl`)."""
        if capture.snapshot(camera_id) is None:
            raise HTTPException(404, "no picture from this camera right now")
        period = 1.0 / min(max(fps, 0.5), 15.0)

        async def frames():
            last, sent = None, 0
            while (frames_max is None or sent < frames_max) and not await request.is_disconnected():
                if blur:
                    jpeg = await asyncio.to_thread(capture.snapshot, camera_id, False, True)
                elif ticket and clear(ticket):  # asked per frame: the stream is blurred again once the ticket runs out
                    jpeg = await asyncio.to_thread(capture.snapshot, camera_id, True)
                else:
                    jpeg = await asyncio.to_thread(capture.snapshot, camera_id)
                if jpeg is None:  # paused or the camera went away: end the stream, the client reconnects
                    return
                if jpeg is not last:
                    last = jpeg
                    head = b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                    sent += 1
                    yield head + jpeg + b"\r\n"
                await asyncio.sleep(period)

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame", headers={"Cache-Control": "no-store"})

    @router.get("/capture/{camera_id}/tracks")
    def tracks(camera_id: str) -> dict:
        """The newest detection: normalized boxes with labels and track ids, for drawing over the snapshot."""
        found = capture.tracks(camera_id)
        if found is None:
            raise HTTPException(404, "unknown camera")
        return found

    return router
