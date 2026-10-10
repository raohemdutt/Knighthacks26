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
- GET  /api/circle/events      the Circle's shared calendar: agreed events, soonest first
- POST /api/circle/proposals   {title, start, end, location?, invitees?} -> approval card for a plan with friends (G1)
- GET  /api/circle/sharing     what leaves this box for the Circle: settings, outbox counts, Neon up or down
- GET  /api/invites[/{name}]   calendar invites (.ics) created after approval, to open or share (K2)
- GET  /api/places             place packs; POST one as YAML/JSON (validated) (S5)
- GET  /api/play/campus-route  ?from=ucf.union&to=ucf.hec[&depart=&due=] or to=friend:<member> (S5)
- GET  /api/learning        taught tasks: stage, progress, scores, deployed models (X10)
- GET  /capture/status      camera status and detector counts, where cameras run (C6)
- GET  /health
- GET  /runtime/models      installed model versions, the current one and what llama-server really serves (R7)

Writes need `Authorization: Bearer $ATLAS_TOKEN` when ATLAS_TOKEN is set; the
tailnet keeps strangers out, the token keeps a guest on the Wi-Fi from
approving things.
"""

import asyncio
import contextlib
import hmac
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from contexts.gateway.bootstrap import Atlas, Settings, build
from contexts.agent.application.rule_compiler import RuleCompileError, compile_rule
from contexts.circle.api import circle_router
from contexts.habits.api import habits_router
from contexts.learning.application.export import teach
from contexts.vault.api import vault_router
from contexts.memory.api import build_router as memory_router
from contexts.perception.api import identity_router
from contexts.play.dayrun_api import dayrun_router
from contracts import Event
from contracts.agent import AnswerGenerated, RuleAdded
from contracts.experience import ApprovalDecided, QuestionAsked
from contracts.learning import TaskSpec
from contexts.capture.api import capture_router
from contexts.learning.api import learning_router
from runtime.api import runtime_router
from runtime.modelstore import ModelStore

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

    if atlas.dayrun:  # /api/dayrun/* (S4, S6)
        app.include_router(dayrun_router(atlas.dayrun, publish=atlas.bus.publish, guard=authorized))
    if atlas.policy:  # GET/PUT /vault/policy, POST /vault/wipe/request, POST /vault/wipe, GET /vault/egress
        app.include_router(vault_router(atlas.policy, atlas.bus, wiper=atlas.wiper, guard=authorized))
    if atlas.identity:  # GET/PUT /identity/settings, POST /identity/enroll/start, GET /identity/enroll/status, GET/DELETE /identity/members (P9)
        app.include_router(identity_router(atlas.identity, atlas.bus, guard=authorized))
    if atlas.capture:  # GET /capture/status (C6)
        app.include_router(capture_router(atlas.capture))
    if atlas.learning:  # GET /api/learning (X10)
        app.include_router(learning_router(atlas.learning, publish=atlas.bus.publish, guard=authorized))
    # GET /api/circle/events, /api/circle/sharing: read-only, and they answer in every state (no Circle, no Neon)
    # POST /api/circle/proposals: an approval card for a plan with friends (G1)
    app.include_router(circle_router(atlas.bus, circle=atlas.circle, sharing=atlas.sharing, presence=atlas.presence,
                                     board=atlas.board, config_dir=atlas.settings.config_dir, guard=authorized))
    app.include_router(runtime_router(ModelStore(atlas.settings.data_dir / "models"), llm_url=atlas.settings.llm_url))  # GET /runtime/models (R7)

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
        out = req.model_dump(mode="json")
        if out["status"] == "proposed":  # the decision is handled in the background; say what was decided, not the old state
            out["status"] = "denied" if body.decision == "deny" else "approved"
        return out

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

        if atlas.game.rules.canonical(item) is None:
            raise HTTPException(404, f"unknown item {item!r}; sortable: {', '.join(sorted(atlas.game.rules.items))}")
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

    @app.get("/api/places")
    async def list_places():
        """Place packs (S5): campus, home and room maps with their places."""
        if atlas.places is None:
            raise HTTPException(404, "no place packs loaded")
        return [{"name": p.name, "kind": p.kind, "entry": p.entry,
                 "places": {pid: pl.model_dump(exclude_none=True) for pid, pl in p.places.items()}}
                for p in atlas.places.packs.values()]

    @app.post("/api/places", dependencies=[Depends(authorized)])
    async def upload_place_pack(request: Request):
        """Upload a place pack as YAML or JSON. It is validated (known places, all reachable, nesting) before use."""
        from contexts.play.domain.places import PlaceSet, parse_pack

        text = (await request.body()).decode("utf-8")
        try:
            pack = parse_pack(text)
            if atlas.places is None:
                atlas.places = PlaceSet({})
            atlas.places.add(pack)
        except ValueError as exc:  # pydantic's ValidationError is a ValueError
            raise HTTPException(422, str(exc)) from None
        folder = atlas.settings.data_dir / "places"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{pack.name}.yaml").write_text(text, encoding="utf-8")
        return {"name": pack.name, "places": len(pack.places), "edges": len(pack.edges)}

    @app.get("/api/play/campus-route")
    async def campus_route(src: str = Query(alias="from"), dst: str = Query(alias="to"),
                           depart: datetime | None = None, due: datetime | None = None):
        """Walking route and minutes between two places ("ucf.union"), or to a friend ("friend:hemdutt").
        With `depart` and `due`, also whether you make it and by how much."""
        if atlas.places is None:
            raise HTTPException(404, "no place packs loaded")
        if dst.startswith("friend:"):
            member = dst.split(":", 1)[1]
            shared = {p.member_id: p for p in atlas.presence.friends(datetime.now(UTC))} if atlas.presence else {}
            if member not in shared:
                raise HTTPException(404, f"{member} isn't sharing a place right now")
            dst = atlas.places.find_shared(shared[member].place) or ""
            if not dst:
                raise HTTPException(404, f"{shared[member].place!r} isn't on any map")
        try:
            if depart and due:
                return atlas.places.feasible(src, dst, depart, due)
            path, minutes = atlas.places.route(src, dst)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        return {"path": path, "minutes": minutes}

    @app.websocket("/ws")
    async def events(ws: WebSocket):
        await ws.accept()
        sockets.add(ws)
        try:
            while True:
                await ws.receive_text()  # the phone talks over HTTP; this just keeps the socket open
        except WebSocketDisconnect:
            sockets.discard(ws)

    # The phone app (web/, a Next.js static export). Mounted last so every API route and /ws wins.
    web_dir = Path(os.environ.get("ATLAS_WEB_DIR", "web/out"))
    if web_dir.is_dir():
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
    return app
