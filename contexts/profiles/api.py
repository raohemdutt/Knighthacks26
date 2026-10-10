"""GET /api/profile: which profile this box runs, its words, and which config files it overrides.

    app.include_router(profile_router(atlas.profile), prefix="/api")

The phone reads it once and uses `words` for its labels (Today, Day Run, Ask chips), so a shift
worker's box says "shift" and "station" where a student's says "class" and "desk".
"""

from fastapi import APIRouter

from contexts.profiles.loader import Profile


def profile_router(profile: Profile) -> APIRouter:
    router = APIRouter(prefix="/profile", tags=["profile"])

    @router.get("")
    async def current():
        return profile.summary()

    return router
