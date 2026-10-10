"""Day Run's API (S4, S6) for the phone.

    app.include_router(dayrun_router(atlas.dayrun, publish=atlas.bus.publish, guard=authorized))

- GET  /api/dayrun/board?start=      the next 24 hours: fixed blocks, deadlines, open slots and their moves
- POST /api/dayrun/simulate          {start, moves} -> meter timeline, route legs, outcome, stars, rejected moves
- POST /api/dayrun/commit            {start, moves} -> one approval card for the calendar blocks (winning plans only)
- GET  /api/dayrun/review?day=       the committed plan against what Habits measured that day
- POST /api/dayrun/replay            {day, moves} -> a past day again: what happened vs a new plan
- POST /api/dayrun/hard              {start, moves, seed} -> seeded disruptions, re-validated and re-routed
- POST /api/dayrun/campaign          {start, days: [moves, ...]} -> consecutive days with carry-over (at most 31)
- POST /api/dayrun/forecast          {estimates, measured | day} -> how well you estimate your own time

Every route takes an optional `scenario` (a file in fixtures/play/, e.g. "demo") to run on a
scripted day instead of live sources. Place packs and routes are served by the gateway
(GET/POST /api/places, GET /api/play/campus-route).
"""

from collections.abc import Awaitable, Callable
from datetime import date, datetime

from fastapi import APIRouter, Depends, HTTPException

from contexts.play.dayrun.models import (
    Board,
    CampaignBody,
    CampaignResult,
    CommitResult,
    ForecastBody,
    ForecastResult,
    HardBody,
    HardResult,
    PlanBody,
    ReplayBody,
    ReplayResult,
    Review,
    SimResult,
)
from contexts.play.dayrun.service import DayRun, PlanNotWinning


def dayrun_router(service: DayRun, publish: Callable[[object], Awaitable] | None = None, guard: Callable | None = None) -> APIRouter:
    router = APIRouter(prefix="/api/dayrun", tags=["dayrun"])
    writes = [Depends(guard)] if guard else []

    def run(call: Callable, *args, **kw):
        try:
            return call(*args, **kw)
        except PlanNotWinning as exc:
            raise HTTPException(422, {"message": "only a winning plan can be committed", "reasons": exc.result.outcome.reasons}) from None
        except LookupError as exc:
            raise HTTPException(404, str(exc).strip("'\"")) from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        except OverflowError:  # a date at the edge of the calendar, or minutes too large to add to one
            raise HTTPException(422, "a date or a number of minutes in the request is out of range") from None
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from None

    @router.get("/board")
    async def board(start: datetime | None = None, scenario: str | None = None) -> Board:
        return run(service.board, start, None, scenario)

    @router.post("/simulate")
    async def simulate(body: PlanBody) -> SimResult:
        return run(service.simulate, body.moves, body.start, body.initial, body.scenario)

    @router.post("/commit", dependencies=writes)
    async def commit(body: PlanBody) -> CommitResult:
        proposal, result = run(service.commit, body.moves, body.start, body.initial, body.scenario)
        if publish:
            await publish(proposal)  # agent.action.proposed: the approval card; Workspace writes only after a yes
        return result

    @router.get("/review")
    async def review(day: date, scenario: str | None = None) -> Review:
        return run(service.review, day, None, scenario)

    @router.post("/replay")
    async def replay(body: ReplayBody) -> ReplayResult:
        return run(service.replay, body.day, body.moves, body.baseline, body.actual, body.scenario)

    @router.post("/hard")
    async def hard(body: HardBody) -> HardResult:
        return run(service.hard, body.moves, body.start, body.initial, body.scenario, body.seed, body.count, body.disruptions)

    @router.post("/campaign")
    async def campaign(body: CampaignBody) -> CampaignResult:
        return run(service.campaign, body.days, body.start, body.initial, body.scenario)

    @router.post("/forecast")
    async def forecast(body: ForecastBody) -> ForecastResult:
        return run(service.forecast, body.estimates, body.measured, body.day, body.scenario)

    return router
