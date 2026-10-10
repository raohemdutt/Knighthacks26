"""GET /capture/status: is each camera delivering frames, and is the detector keeping up.

    app.include_router(capture_router(atlas.capture))

{"cameras": {"cam-desk": {"status": "ok", "fps_actual": 15.0, "dropped": 0, "detected": 912, "errors": 0}}}
`status` is the reader's: ok, degraded (under 70% of the target rate) or down (no frame for 3 s).
"""

from fastapi import APIRouter

from contexts.capture.application.service import CaptureService


def capture_router(capture: CaptureService) -> APIRouter:
    router = APIRouter()

    @router.get("/capture/status")
    def status() -> dict:
        return {"cameras": capture.stats()}

    return router
