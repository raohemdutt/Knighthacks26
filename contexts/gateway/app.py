"""Gateway (X2): the phone's one door into Atlas.

    python -m contexts.gateway            # serves on :8000

- WS  /ws                      every bus event as JSON, live (the PWA's feed)
- POST /api/ask                {text} -> the answer with citations (also on /ws)
- GET  /api/approvals          pending approval cards
- POST /api/approvals/{id}     {decision: approve|deny} (ntfy buttons call this)
- POST /api/teach              {task, instruction, ...} -> approval card for the export
- POST /api/rules              {text} plain English -> a new alert rule (A3)
- GET  /api/play/route         ?item=can&x=&y= -> A* route to the right bin (S2)
- GET  /api/play/accuracy      Sort It round 1 vs round 5 accuracy (S1)
- GET  /api/impact             Waste Watch counts, diversion rate, before vs after nudges (E2)
- POST /api/presence           {place, hours} share a named place with the Circle (G5, opt-in)
- GET  /api/presence           friends' current places
- GET  /api/circle/leaderboard Waste Watch, habit and Sort It leaderboards from Neon (E2)
- GET  /api/invites[/{name}]   calendar invites (.ics) created after approval, to open or share (K2)
- GET  /health

Writes need `Authorization: Bearer $ATLAS_TOKEN` when ATLAS_TOKEN is set; the
tailnet keeps strangers out, the token keeps a guest on the Wi-Fi from
approving things.
"""

import asyncio
import contextlib
import hmac
from datetime import UTC, datetime, timedelta
from typing import Literal
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from contexts.gateway.bootstrap import Atlas, Settings, build
from contexts.agent.application.rule_compiler import RuleCompileError, compile_rule
from contexts.habits.api import habits_router
from contexts.learning.application.export import teach
from contexts.memory.api import build_router as memory_router
from contracts import Event
from contracts.agent import AnswerGenerated, RuleAdded
from contracts.experience import ApprovalDecided, QuestionAsked
from contracts.learning import TaskSpec

ASK_TIMEOUT_S = 90


class AskBody(BaseModel):
    text: str


class DecisionBody(BaseModel):
    decision: Literal["approve", "deny"]


def create_app(settings: Settings | None = None, atlas: Atlas | None = None) -> FastAPI:
    atlas = atlas or build(settings)
    sockets: set[WebSocket] = set()
    waiting: dict[str, asyncio.Future] = {}

    async def fan_out(e: Event) -> None:
        if e.type == "agent.answer.generated" and (f := waiting.pop(e.question_id, None)) and not f.done():
            f.set_result(e)
        data = e.model_dump_json()
        for ws in list(sockets):
            try:
                await ws.send_text(data)
            except Exception:
                sockets.discard(ws)

    atlas.bus.subscribe("", fan_out)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        await atlas.start()
        yield
        await atlas.stop()

    app = FastAPI(title="Atlas", lifespan=lifespan)
    app.state.atlas = atlas
    if atlas.habits:  # GET /api/habits/today, /api/habits/{id}/history (H2)
        app.include_router(habits_router(atlas.habits), prefix="/api")
    if atlas.memory:  # GET /memory/... (M4)
        app.include_router(memory_router(atlas.memory, atlas.settings.data_dir / "frames"))

    def authorized(authorization: str | None = Header(default=None)) -> None:
        token = atlas.settings.token
        if token and not hmac.compare_digest(authorization or "", f"Bearer {token}"):
            raise HTTPException(401, "missing or wrong token")

    @app.get("/health")
    async def health():
        return {"ok": True, "model": atlas.qa.runtime.model_info().version}

    @app.post("/api/ask", dependencies=[Depends(authorized)])
    async def ask(body: AskBody) -> AnswerGenerated:
        qid = uuid4().hex
        answer = waiting[qid] = asyncio.get_running_loop().create_future()
        await atlas.bus.publish(QuestionAsked(question_id=qid, text=body.text))  # may answer before returning
        try:
            return await asyncio.wait_for(answer, ASK_TIMEOUT_S)
        except TimeoutError:
            raise HTTPException(504, "the model took too long") from None
        finally:
            waiting.pop(qid, None)

    @app.get("/api/approvals")
    async def approvals():
        return [r.model_dump(mode="json") for r in atlas.approvals.pending()]

    @app.post("/api/approvals/{action_id}", dependencies=[Depends(authorized)])
    async def decide(action_id: str, body: DecisionBody):
        req = atlas.approvals.requests.get(action_id)
        if req is None:
            raise HTTPException(404, "no such request")
        await atlas.bus.publish(ApprovalDecided(action_id=action_id, decision=body.decision))
        return req.model_dump(mode="json")

    @app.post("/api/teach", dependencies=[Depends(authorized)])
    async def start_teaching(spec: TaskSpec):
        created, proposal = teach(spec, datetime.now(UTC))
        await atlas.bus.publish_all([created, proposal])
        return {"task": created.task, "action_id": proposal.action_id}

    @app.post("/api/rules", dependencies=[Depends(authorized)])
    async def add_rule(body: AskBody):
        try:
            rule = await compile_rule(body.text, atlas.qa.runtime, atlas.zones)
        except RuleCompileError as exc:
            raise HTTPException(422, str(exc)) from None
        await atlas.bus.publish(RuleAdded(rule=rule.model_dump(), source_text=body.text))
        return rule.model_dump()

    @app.get("/api/play/route")
    async def play_route(item: str, x: float = 0.75, y: float = 0.75):
        """Route from a normalized point to the right bin for `item`, as grid cells for the phone to draw."""
        if atlas.game is None:
            raise HTTPException(404, "Sort It is not configured")
        from contexts.play.domain.pathfinding import route

        zone, path = route(atlas.game.map, atlas.game.rules, item, (x, y))
        return {"item": item, "bin": zone, "rows": atlas.game.map.rows, "cols": atlas.game.map.cols,
                "path": [c.model_dump() for c in path]}

    @app.get("/api/play/accuracy")
    async def play_accuracy():
        g = atlas.game
        if g is None:
            raise HTTPException(404, "Sort It is not configured")
        acc = g.accuracy(g.session_id)
        return {"session_id": g.session_id, "rounds": len(g.results.get(g.session_id, [])),
                "first_round": acc[0] if acc else None, "round_5_on": acc[1] if acc else None}

    @app.get("/api/impact")
    async def impact():
        """Waste Watch numbers for the Today screen and the slide: measured counts only."""
        if atlas.waste is None:
            raise HTTPException(404, "Waste Watch is not configured")
        return {"total": atlas.waste.metrics(), "before_after_nudges": atlas.waste.before_after()}

    class PlaceBody(BaseModel):
        place: str
        hours: float = 2

    @app.post("/api/presence", dependencies=[Depends(authorized)])
    async def share_place(body: PlaceBody):
        """Phone opt-in: share a named place with the Circle until it expires (G5)."""
        if atlas.presence is None:
            raise HTTPException(404, "the Circle is not configured")
        try:
            e = atlas.presence.share(body.place, "phone", datetime.now(UTC), timedelta(hours=body.hours))
        except (PermissionError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from None
        await atlas.bus.publish(e)
        return e.model_dump(mode="json")

    @app.get("/api/presence")
    async def friends_places():
        if atlas.presence is None:
            raise HTTPException(404, "the Circle is not configured")
        return [p.model_dump(mode="json") for p in atlas.presence.friends(datetime.now(UTC))]

    @app.get("/api/circle/leaderboard")
    async def leaderboard():
        """Circle leaderboards from Neon (E2, G3): Waste Watch, habit streaks, Sort It scores."""
        if atlas.board is None or atlas.circle is None:
            raise HTTPException(404, "the Circle is not connected to Neon")
        cid = atlas.circle.circle_id
        impact, habits, scores = await asyncio.gather(
            asyncio.to_thread(atlas.board.impact_leaderboard, cid),
            asyncio.to_thread(atlas.board.habit_leaderboard, cid),
            asyncio.to_thread(atlas.board.top_scores, cid),
        )
        return {"circle_id": cid, "impact": impact, "habits": habits, "scores": scores}

    invites = atlas.settings.data_dir / "invites"

    @app.get("/api/invites")
    async def list_invites():
        """Calendar invites created after approval (K2), newest first, for the phone or the group chat link."""
        files = sorted(invites.glob("invite-*.ics"), reverse=True) if invites.exists() else []
        return [{"name": f.name, "url": f"/api/invites/{f.name}"} for f in files[:20]]

    @app.get("/api/invites/{name}")
    async def get_invite(name: str):
        path = invites / name
        if not name.startswith("invite-") or not name.endswith(".ics") or "/" in name or "\\" in name or not path.is_file():
            raise HTTPException(404, "no such invite")
        return FileResponse(path, media_type="text/calendar", filename=name)

    @app.websocket("/ws")
    async def events(ws: WebSocket):
        await ws.accept()
        sockets.add(ws)
        try:
            while True:
                await ws.receive_text()  # the phone talks over HTTP; this just keeps the socket open
        except WebSocketDisconnect:
            sockets.discard(ws)

    return app
