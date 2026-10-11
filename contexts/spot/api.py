"""Inspection (SPOT) endpoints for the dashboard at /spot/.

    app.include_router(spot_router(atlas.spot, atlas.bus, guard=authorized))

Read (every one answers 200 without a kit: {"configured": false, "note": how to add it}):
- GET  /api/spot/overview            KPIs with n and intervals, the three things to investigate next, the headline findings
- GET  /api/spot/findings            every finding: number, n, interval, sentence, caveat, breakdown rows, evidence ids
- GET  /api/spot/triage              the ranked "needs attention" list with score parts, and the uninspected-risk list
- GET  /api/spot/missions            one row per mission; GET /api/spot/missions/{id} adds the timeline with readings and anomalies
- GET  /api/spot/locations           waypoints: coordinates, floor, health with n, failed actions, anomalies; the floor plan if the kit has one
- GET  /api/spot/assets              one row per asset; GET /api/spot/assets/{id} adds readings over time, anomalies, last good inspection
- GET  /api/spot/replay/state        the live picture: mission in progress, robot position, counts, alerts, scrubber marks
- GET  /api/spot/shift               the operator's first screen in plain words: status line, do-this-first, watch-out, every asset with a
                                     traffic-light level and a sentence, the rounds, the floor plan (application/plain.py). No statistics.
- GET  /api/spot/shift/rounds/{id}   one round as steps, each a sentence with its time, result, reading and alarm
- GET  /api/spot/handoff             five to eight lines for the next shift, each with its evidence
- GET  /api/spot/assumptions         every assumption the analysis makes
- GET  /api/spot/validation          our numbers next to the kit's validation_results.json, and every data surprise
- GET  /api/spot/model               the small anomaly scorer trained on the kit: honest evaluation, weights, what it flags
- GET  /api/spot/work_orders         work orders written on this box (after approval)
- GET  /api/spot/kit/assets/{path}   an image from the kit's assets folder (read-only; only files the loader listed)
- GET  /api/spot/plan/image          the floor plan re-encoded small, when fixtures/spot/floor_plan.jpg exists

Write (token-guarded like every other write):
- POST /api/spot/replay/start | pause | seek {index | mission_id | fraction} | speed {speed}
- POST /api/spot/ask {text, rephrase?, plain?}  a cited answer built from the analysis, returned at once (no model call).
                                     plain=true words it for an operator (`text`); `draft` keeps the exact wording.
                                     rephrase=true also offers a short draft to the on-device model, bounded and checked
- GET  /api/spot/shiftrun/board | /api/spot/shiftrun/recommended?hard=   Shift Run: what can be planned, and Atlas's plan
- POST /api/spot/shiftrun/simulate {route, turn_back_at, seed?, hard?}     a plan run through the toy model (domain/shiftrun.py): score,
                                     stars, steps, reasons, and Atlas's plan beside it. Pure: nothing is written, so it is not guarded.
- POST /api/spot/actions/plan {route, turn_back_at}   the plan as a mission request: an approval card, like the two below
- POST /api/spot/actions/reinspect {asset_id} | /api/spot/actions/work_order {anomaly_id}
                                     -> {"action_id"}: an approval card. Nothing is written until it is approved.
"""

from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from contexts.spot.application import plain
from contexts.spot.application.assistant import SUGGESTED
from contexts.spot.application.service import not_configured
from contexts.spot.wiring import SpotContext

CACHE = {"Cache-Control": "public, max-age=3600"}


class AskBody(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    rephrase: bool = False  # false: the code-built answer at once, no model call
    plain: bool = False  # true: the same answer in operator words; `draft` keeps the exact wording


class SeekBody(BaseModel):
    index: int | None = Field(default=None, ge=0)
    mission_id: str | None = None
    fraction: float | None = Field(default=None, ge=0, le=1)


class SpeedBody(BaseModel):
    speed: float = Field(gt=0, le=36_000)


class ReinspectBody(BaseModel):
    asset_id: str


class WorkOrderBody(BaseModel):
    anomaly_id: str


class PlanBody(BaseModel):
    route: list[str] = Field(default_factory=list, max_length=24)  # asset ids in visiting order
    turn_back_at: float = Field(default=40, ge=5, le=95)  # battery % at which SPOT stops and walks back
    seed: int = 0  # for the "play" outcomes only; the score is an expected value
    hard: bool = False  # the weakest waypoint is blocked


def spot_router(spot: SpotContext, bus, guard: Callable | None = None) -> APIRouter:
    router = APIRouter(prefix="/api/spot")
    writes = [Depends(guard)] if guard else []
    read = spot.read

    def model_line() -> dict:
        info = spot.runtime.model_info() if spot.runtime is not None else None
        fake = info is None or info.base == "fake"
        return {"version": info.version if info else None, "real": not fake,
                "role": ("Answers are built by code from the tables. " + ("The on-device model only rewords the short ones afterwards, and its wording is "
                         "dropped if it changes a number or an id." if not fake else "No model is configured, so nothing is rephrased."))}

    @router.get("/overview")
    def overview() -> dict:
        return {**read.overview(), "model": model_line()}

    @router.get("/findings")
    def findings() -> dict:
        if not read.configured:
            return {**not_configured(), "findings": []}
        return {"configured": True, "findings": [f.model_dump() for f in read.findings.values()]}

    @router.get("/triage")
    def triage() -> dict:
        if not read.configured:
            return {**not_configured(), "items": [], "uninspected": [], "next": []}
        return {"configured": True, "items": read.triage, "uninspected": read.uninspected, "next": read.next_up,
                "score": "severity of the worst anomaly (CRITICAL 40, HIGH 25, MEDIUM 10, LOW 3) times recency (1 within a day, 0.75 within "
                         "three, 0.5 within seven, else 0.3) + asset criticality (CRITICAL 25, HIGH 15, MEDIUM 6) + 4 per other anomaly on "
                         "the asset (up to 12) + 15 if the latest inspection attempt failed + 2 per day since the last good inspection "
                         "(up to 10) - up to 10 if a later inspection succeeded and raised nothing. Waypoints: 60 x failure rate, halved "
                         "when nothing failed there under the latest version."}

    @router.get("/missions")
    def missions() -> dict:
        return {"configured": True, "missions": read.missions()} if read.configured else {**not_configured(), "missions": []}

    @router.get("/missions/{mission_id}")
    def mission(mission_id: str) -> dict:
        if not read.configured:
            return not_configured()
        found = read.mission(mission_id.upper())
        if found is None:
            raise HTTPException(404, f"no mission {mission_id}")
        return {"configured": True, **found}

    @router.get("/locations")
    def locations() -> dict:
        return read.locations()

    @router.get("/assets")
    def assets() -> dict:
        return {"configured": True, "assets": read.assets()} if read.configured else {**not_configured(), "assets": []}

    @router.get("/assets/{asset_id}")
    def asset(asset_id: str) -> dict:
        if not read.configured:
            return not_configured()
        found = read.asset(asset_id.upper())
        if found is None:
            raise HTTPException(404, f"no asset {asset_id}")
        return {"configured": True, **found}

    shift_view: dict = {}  # the kit never changes while the box runs: worded once

    @router.get("/shift")
    def shift() -> dict:
        """The operator's first screen: what to do, what to watch, what is fine, in sentences (no statistics)."""
        if "view" not in shift_view:
            shift_view["view"] = plain.shift(read)
        return shift_view["view"]

    @router.get("/shift/rounds/{mission_id}")
    def shift_round(mission_id: str) -> dict:
        if not read.configured:
            return not_configured()
        found = plain.round_view(read, mission_id.upper())
        if found is None:
            raise HTTPException(404, f"no round {mission_id}")
        return {"configured": True, **found}

    @router.get("/handoff")
    def handoff() -> dict:
        return read.handoff()

    @router.get("/assumptions")
    def assumptions() -> dict:
        return {"configured": True, "assumptions": read.assumptions} if read.configured else {**not_configured(), "assumptions": []}

    @router.get("/validation")
    def validation() -> dict:
        return {"configured": True, "kit_path": read.kit.path, **read.validation} if read.configured else {**not_configured(), "rows": [], "issues": []}

    @router.get("/model")
    def model() -> dict:
        """The anomaly scorer trained on the kit: how it did on later missions against two plain rules, its weights, its flags."""
        return read.model_view()

    @router.get("/work_orders")
    def work_orders() -> dict:
        return {"configured": read.configured, "work_orders": spot.orders.view(), "folder": str(spot.orders.folder)}

    @router.get("/kit/assets/{path:path}")
    def kit_asset(path: str):
        kit = read.kit
        if kit is None or not kit.assets_dir or path not in kit.asset_files:  # only names the loader listed: no path can escape
            raise HTTPException(404, "no such kit image")
        return FileResponse(Path(kit.assets_dir) / path, headers=CACHE)

    @router.get("/plan/image")
    def plan_image():
        small = Path(read.kit.path).parent / "floor_plan.jpg" if read.kit else None
        if small is None or not small.is_file() or not read.plan:
            raise HTTPException(404, "no floor plan")
        return FileResponse(small, media_type="image/jpeg", headers=CACHE)

    # --- replay ---

    def state() -> dict:
        if not read.configured:
            return {**not_configured(), "index": 0, "total": 0, "playing": False}
        return {"configured": True, **spot.player.view()}

    @router.get("/replay/state")
    def replay_state() -> dict:
        return state()

    @router.post("/replay/start", dependencies=writes)
    async def replay_start() -> dict:
        await spot.player.start()
        return state()

    @router.post("/replay/pause", dependencies=writes)
    async def replay_pause() -> dict:
        await spot.player.pause()
        return state()

    @router.post("/replay/seek", dependencies=writes)
    async def replay_seek(body: SeekBody) -> dict:
        if not read.configured:
            return state()
        total = len(spot.player.stream)
        if body.mission_id is not None:
            index = spot.player.index_of_mission(body.mission_id.upper())
            if index is None:
                raise HTTPException(404, f"no mission {body.mission_id} in the replay")
        elif body.index is not None:
            index = body.index
        elif body.fraction is not None:
            index = round(body.fraction * total)
        else:
            raise HTTPException(422, "give index, mission_id or fraction")
        await spot.player.seek(index)
        return state()

    @router.post("/replay/speed", dependencies=writes)
    async def replay_speed(body: SpeedBody) -> dict:
        spot.player.set_speed(body.speed)
        return state()

    # --- ask, and the two operator actions ---

    @router.get("/ask/suggested")
    def suggested() -> dict:
        return {"suggested": SUGGESTED}

    @router.post("/ask", dependencies=writes)
    async def ask(body: AskBody) -> dict:
        out = {"configured": read.configured, **await spot.assistant.ask(body.text, rephrase=body.rephrase and not body.plain)}
        if body.plain:
            words = plain.answer(read, out)
            out.update(plain=words is not None, text=words or out["text"])
        return out

    # --- Shift Run: plan a round, see how it would probably go (a toy model from the recorded rounds) ---

    @router.get("/shiftrun/board")
    def shiftrun_board() -> dict:
        return spot.shiftrun.board()

    @router.get("/shiftrun/recommended")
    def shiftrun_recommended(hard: bool = False) -> dict:
        return spot.shiftrun.recommended(hard=hard)

    @router.post("/shiftrun/simulate")
    def shiftrun_simulate(body: PlanBody) -> dict:
        return spot.shiftrun.simulate(body.route, body.turn_back_at, seed=body.seed, hard=body.hard)

    @router.post("/actions/plan", dependencies=writes)
    async def plan(body: PlanBody) -> dict:
        try:
            result = spot.shiftrun.simulate(body.route, body.turn_back_at) if read.configured else {}
            action = spot.orders.propose_plan(body.route, body.turn_back_at, result.get("summary", ""))
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from None
        await bus.publish(action)
        return {"action_id": action.action_id, "reason": action.reason, "target": action.target}

    @router.post("/actions/reinspect", dependencies=writes)
    async def reinspect(body: ReinspectBody) -> dict:
        try:
            action = spot.orders.propose_reinspect(body.asset_id.upper())
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from None
        await bus.publish(action)
        return {"action_id": action.action_id, "reason": action.reason, "target": action.target}

    @router.post("/actions/work_order", dependencies=writes)
    async def work_order(body: WorkOrderBody) -> dict:
        try:
            action = spot.orders.propose_work_order(body.anomaly_id.upper())
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from None
        await bus.publish(action)
        return {"action_id": action.action_id, "reason": action.reason, "target": action.target}

    return router
