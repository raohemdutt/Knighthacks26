"""Habits read API for the phone's Today screen.

    app.include_router(habits_router(atlas.habits), prefix="/api")

- GET /habits/today                      every habit: value, target, met so far, streak, latest nudge
- GET /habits/{habit_id}/history?days=7  closed days, oldest first, ending with today so far
"""

from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Query

from contexts.habits.domain.tracker import HabitTracker


def habits_router(tracker: HabitTracker) -> APIRouter:
    router = APIRouter(prefix="/habits", tags=["habits"])

    @router.get("/today")
    async def today():
        return tracker.digest(now=tracker.now or datetime.now(UTC))  # event time; the ticker keeps it current

    @router.get("/{habit_id}/history")
    async def history(habit_id: str, days: int = Query(7, ge=1, le=90)):
        if tracker.cfg.by_id(habit_id) is None:
            raise HTTPException(404, "no such habit")
        return tracker.history_for(habit_id, days)

    return router
