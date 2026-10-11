"""The Agent's small HTTP surface: the daily digest (A6).

GET /api/digest[?day=YYYY-MM-DD]  the day's episodes summarized, written from the data (no model call):
                                  text to read out, counts, what was seen most, the busiest hour, cited moments.
                                  No day: today (the household's local day, up to now).
"""

from datetime import date

from fastapi import APIRouter, HTTPException, Query

from contexts.agent.application.digest import DailyDigest, Digester


def digest_router(digester: Digester) -> APIRouter:
    router = APIRouter(tags=["agent"])

    @router.get("/api/digest", response_model=DailyDigest)
    def get_digest(day: str | None = Query(default=None, description="YYYY-MM-DD, local; default today")) -> DailyDigest:
        try:
            wanted = date.fromisoformat(day) if day else None
        except ValueError:
            raise HTTPException(422, "day must be YYYY-MM-DD") from None
        return digester.build(wanted)

    return router
