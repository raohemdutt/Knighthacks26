"""Habits read API for the phone's Today screen.

    app.include_router(habits_router(atlas.habits), prefix="/api")

- GET /habits/today                      every habit: value, target, met so far, streak, latest nudge
- GET /habits/{habit_id}/history?days=7  closed days, oldest first, ending with today so far

With owner recognition on and somebody enrolled (P9, `MemberHabits`), both take `?member=<member_id>` for one
person's habits, and `today` also says whose it returned (`member`) and who can be asked for (`members`, ids
only). Without `member` it is the only enrolled person's, or the unnamed set when several are enrolled.
While recognition is off nothing changes: no extra fields, and `?member=` is a 404.
"""

from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Query

from contexts.habits.domain.tracker import HabitTracker


def habits_router(tracker: HabitTracker) -> APIRouter:
    router = APIRouter(prefix="/habits", tags=["habits"])

    def view(member: str | None) -> HabitTracker:
        """The tracker to answer from: one member's, or the default one."""
        pick = getattr(tracker, "view", None)  # MemberHabits has it; the plain tracker is its own only view
        if pick is None or not tracker.attributing:
            if member is not None:
                raise HTTPException(404, "no such member")
            return tracker
        try:
            return pick(member)
        except KeyError:
            raise HTTPException(404, "no such member") from None

    @router.get("/today")
    async def today(member: str | None = None):
        t = view(member)
        out = t.digest(now=tracker.now or datetime.now(UTC))  # event time; the ticker keeps it current
        if getattr(tracker, "attributing", False):
            ids = list(tracker.members)
            out |= {"member": next((m for m in ids if tracker.members[m] is t), None), "members": ids}
        return out

    @router.get("/{habit_id}/history")
    async def history(habit_id: str, days: int = Query(7, ge=1, le=90), member: str | None = None):
        if tracker.cfg.by_id(habit_id) is None:
            raise HTTPException(404, "no such habit")
        return view(member).history_for(habit_id, days)

    return router
