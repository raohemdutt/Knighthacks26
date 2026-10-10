"""The Impact screen's endpoints (GET /api/impact itself stays in the gateway, unchanged).

    app.include_router(impact_router(atlas.impact, publish=atlas.bus.publish, guard=authorized))

- GET  /api/impact/story?member=     everything the Impact screen shows: the week in real units (estimated ranges),
                                     the disposal feed with the rule for each wrong one, the weekly goal, clean
                                     days, best day, projection, before and after nudges with n, milestones, the
                                     practice set, and which rules file is active. 404 when Waste Watch is not set up.
- GET  /api/impact/practice?member=  the Sort It practice set: {"items": [{"item", "mistakes", "asked", "went_to",
                                     "belongs_in", "zone", "rule", "frame_ref", "last_wrong_at"}], "size", ...},
                                     the most mistakes first. `item` is a label Sort It's GET /api/play/route takes.
- POST /api/impact/practice          {label, item_id?, member?}: the "teach me" tap. Publishes `impact.practice.added`
                                     and answers with the new practice set. Nothing leaves the box.
"""

from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from contexts.impact.application.story import ImpactStory


class PracticeBody(BaseModel):
    label: str = Field(min_length=1, max_length=60)
    item_id: str | None = Field(default=None, max_length=120)
    member: str | None = Field(default=None, max_length=80)


def impact_router(story: ImpactStory | None, publish: Callable[[object], Awaitable] | None = None, guard: Callable | None = None,
                  clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> APIRouter:
    router = APIRouter(prefix="/api/impact", tags=["impact"])
    writes = [Depends(guard)] if guard else []

    def need() -> ImpactStory:
        if story is None:
            raise HTTPException(404, "Waste Watch is not configured")
        return story

    @router.get("/story")
    async def get_story(member: str | None = None) -> dict:
        return need().view(clock(), member or None)

    @router.get("/practice")
    async def get_practice(member: str | None = None) -> dict:
        return need().practice(clock(), member or None)

    @router.post("/practice", dependencies=writes)
    async def add_practice(body: PracticeBody) -> dict:
        s = need()
        now = clock()
        try:
            added = s.add_practice(body.label.strip().lower(), now, item_id=body.item_id, member=body.member or None)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        if publish is None:
            s.on_practice(added)
        else:
            await publish(added)
        return {"added": added.label, "practice": s.practice(now, body.member or None)}

    return router
