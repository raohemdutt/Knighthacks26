"""GET /capture/status: is each camera delivering frames, and is the detector keeping up.

    app.include_router(capture_router(atlas.capture))

{"cameras": {"cam-desk": {"status": "ok", "fps_actual": 15.0, "dropped": 0, "detected": 912, "errors": 0}}}

Also for the Live screen (X3): GET /capture/<camera>/snapshot.jpg (a blurred JPEG) and GET /capture/<camera>/tracks.
`status` is the reader's: ok, degraded (under 70% of the target rate) or down (no frame for 3 s).
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from contexts.capture.application.service import CaptureService


def capture_router(capture: CaptureService) -> APIRouter:
    router = APIRouter()

    @router.get("/capture/status")
    def status() -> dict:
        return {"cameras": capture.stats()}

    @router.get("/capture/{camera_id}/snapshot.jpg")
    def snapshot(camera_id: str) -> Response:
        """The newest frame (faces blurred unless the Vault policy turns blur off); 404 if paused or no frame yet."""
        jpeg = capture.snapshot(camera_id)
        if jpeg is None:
            raise HTTPException(404, "no picture from this camera right now")
        return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @router.get("/capture/{camera_id}/tracks")
    def tracks(camera_id: str) -> dict:
        """The newest detection: normalized boxes with labels and track ids, for drawing over the snapshot."""
        found = capture.tracks(camera_id)
        if found is None:
            raise HTTPException(404, "unknown camera")
        return found

    return router
