"""League endpoints for the phone's Circle screen.

    app.include_router(league_router(atlas.bus, league=atlas.league, guard=authorized))

- GET  /api/league            standings for the running week, worst and best with reasons, the stake, the dares,
                              the Waste Watch team line, the week window and when it closes. 200 in every state:
                              {"configured": false, "note": ...} when there is no league at all.
- GET  /api/league/me         this member's own breakdown by target
- POST /api/league/stake      {kind, text?} -> {"action_id"}: an approval card; the stake changes only once approved
- POST /api/league/challenges {kind, days, invitees, target?} -> {"action_id"}: an approval card that invites friends

Both POSTs raise `agent.action.proposed` and nothing else: nothing leaves the box and nothing changes
until the card is approved on the phone.
"""

from collections.abc import Callable
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from contracts.play import Stake

HOW_TO = ("Add config/league.yaml to turn the League on. To play with friends, also set circle.member_id and "
          "circle.circle_id in config/agent.yaml.")


class StakeBody(BaseModel):
    kind: str
    text: str | None = Field(default=None, max_length=200)


class ChallengeBody(BaseModel):
    kind: str
    days: int = Field(3, ge=1, le=60)
    invitees: list[str] = []
    target: float | None = None


def league_router(bus, *, league=None, clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                  guard: Callable | None = None) -> APIRouter:
    router = APIRouter()
    writes = [Depends(guard)] if guard else []

    @router.get("/api/league")
    def standings() -> dict:
        if league is None:
            return {"configured": False, "local": True, "circle_id": None, "member_id": None, "note": HOW_TO, "members": [],
                    "week": None, "targets": [], "standings": [], "worst": None, "best": None, "summary": None,
                    "stake": None, "team_waste": None, "challenges": [], "presets": [], "last_closed": None}
        return league.view(clock())

    @router.get("/api/league/me")
    def me() -> dict:
        if league is None:
            raise HTTPException(404, "the League is not configured")
        return league.me_view(clock())

    @router.post("/api/league/stake", dependencies=writes)
    async def set_stake(body: StakeBody) -> dict:
        if league is None:
            raise HTTPException(404, "the League is not configured")
        try:
            action = league.propose_stake(Stake(kind=body.kind, text=body.text), clock())
        except ValueError as exc:
            raise HTTPException(422, str(exc)[:300]) from None
        await bus.publish(action)
        return {"action_id": action.action_id, "reason": action.reason}

    @router.post("/api/league/challenges", dependencies=writes)
    async def start_challenge(body: ChallengeBody) -> dict:
        if league is None:
            raise HTTPException(404, "the League is not configured")
        try:
            action = league.propose_challenge(body.kind, body.days, body.invitees, clock(), target=body.target)
        except ValueError as exc:
            raise HTTPException(422, str(exc)[:300]) from None
        await bus.publish(action)
        return {"action_id": action.action_id, "challenge_id": action.params["challenge"]["challenge_id"],
                "reason": action.reason}

    return router
