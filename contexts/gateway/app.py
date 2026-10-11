"""Gateway (X2): the phone's one door into Atlas.

    python -m contexts.gateway            # serves on :8000

- WS  /ws                      every bus event as JSON, live (the PWA's feed); while an answer is being written,
                               also {"type": "agent.answer.delta", "question_id", "text"} pieces (not logged; A11)
- POST /api/ask                {text} -> the answer with citations (also on /ws)
- POST /api/ask/stream         the same as lines of JSON as the answer is written: {"delta"}..., then {"final"} (the box's voice)
- GET  /api/approvals          pending approval cards
- POST /api/approvals/{id}     {decision: approve|deny} (ntfy buttons call this)
- POST /api/teach              {task, instruction, ...} -> approval card for the export
- POST /api/rules              {text} plain English -> a new alert rule (A3)
- GET  /api/rules              every rule: id, description, on/off, where it came from (X6)
- POST /api/rules/{id}/enabled {enabled} switch a rule off or back on (agent.rule.toggled, replays)
- GET  /api/alerts?limit=      recent alerts, newest first: message, rule, time, evidence
- GET  /api/preflight          the demo's T-15 checklist, checked by the box itself (D9); POST .../reset clears the stage
- POST /api/events             one event or a list: camera and connector input pushed into a running box (demo, T7)
- GET  /api/day/now            the Day loop: live status, today's plan, a pending re-plan, the review (D1)
- GET  /api/day/plan?day=      a day's plan with its moves in Day Run's editor shape; POST {moves?, from_here?} remakes it
- GET  /api/play/route         ?item=can&x=&y= -> A* route to the right bin (S2)
- GET  /api/play/room-route    ?item=can[&from_track=|&x=&y=] -> the same walk on the floor plan, in metres (X18)
- GET  /api/room, /api/room/live    the room twin: floor plan + calibration + zones on the floor; live tracks on it
- GET  /api/play/accuracy      Sort It round 1 vs round 5 accuracy (S1)
- GET  /api/impact             Waste Watch counts, diversion rate, before vs after nudges (E2)
- GET  /api/impact/story       the Impact screen: the week in estimated real units, the disposal feed with each rule,
                               the weekly goal, milestones, which rules file is active (contexts/impact/api.py)
- GET  /api/impact/practice    the Sort It practice set: this week's mistakes, most first; POST {label} is "teach me"
- GET  /api/day/coach/feedback recent sessions' breaks with "Was that right?", per-trigger precision, what Atlas is learning;
                               POST /api/day/coach/corrections {day, block_id, break_index, right, answer?, note?}
- POST /api/presence           {place, hours} share a named place with the Circle (G5, opt-in), up to 12 hours
- DELETE /api/presence         stop sharing my place now
- GET  /api/presence           friends' current places
- GET  /api/circle/leaderboard Waste Watch, habit and Sort It leaderboards from Neon (E2)
- GET  /api/circle/events      the Circle's shared calendar: agreed events, soonest first
- POST /api/circle/proposals   {title, start, end, location?, invitees?} -> approval card for a plan with friends (G1)
- GET  /api/circle/sharing     what leaves this box for the Circle: settings, outbox counts, Neon up or down
- GET  /api/circle/face-share  friends whose boxes can recognise you and the other way round; POST .../{friend}/share,
                               /revoke, /review, /forget, /trust-key. 409 while it is switched off (the default)
- GET  /api/invites[/{name}]   calendar invites (.ics) created after approval, to open or share (K2)
- GET  /api/places             place packs; POST one as YAML/JSON (validated) (S5)
- GET  /api/play/campus-route  ?from=ucf.union&to=ucf.hec[&depart=&due=] or to=friend:<member> (S5)
- GET  /api/places/route      ?from=&to=&arrive_by=|depart=  "when to leave": times, minutes, metres, the line walked
- GET  /api/places/map/{pack} the pack's campus map (buildings, paths; © OpenStreetMap contributors) for the phone to draw
- GET  /api/learning        taught tasks: stage, progress, scores, deployed models (X10)
- GET  /api/learning/lab    training runs as stories for the Learning lab screen; /api/learning/lab/{run_id} is one
- GET  /api/learning/lab/bench    the lab's benchmark and energy rows; /experiments is its timeline; POST /routing/simulate
- GET  /api/profile         who this box serves: name, description, vocabulary, overridden config files
- GET  /api/league          the League: this week's standings, worst and best with reasons, stake, dares, team waste line
- GET  /api/league/me       this member's breakdown by target
- POST /api/league/stake    {kind, text?} -> approval card to set the stake; POST /api/league/challenges {kind, days, invitees}
- GET  /api/voice/status    the box's own voice: off|idle|listening|thinking|speaking, mic and speaker health (X16)
- POST /api/voice/talk      push to talk (also /stop, /enabled): the microphone and speaker are on the box, not the phone
- GET  /api/zones           the zones per camera, read fresh from config/zones.yaml; PUT /api/zones/{camera} sets them live,
                               POST /api/zones/demo applies the forgiving demo zones (the Zones page, /zones/)
- GET  /capture/status      camera status and detector counts, where cameras run (C6)
- GET  /health
- GET  /runtime/models      installed model versions, the current one and what llama-server really serves (R7)
- GET  /runtime/stats       detector FPS, on-device tok/s and caption latency, trainer and teacher speed, eval scores
                               per deployed model; each with its record and measured time, or null and why (R5)

Writes need `Authorization: Bearer $ATLAS_TOKEN` when ATLAS_TOKEN is set; the
tailnet keeps strangers out, the token keeps a guest on the Wi-Fi from
approving things.
"""

import asyncio
import contextlib
import json
import hmac
import math
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal
from uuid import uuid4

import yaml
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from contexts.gateway.bootstrap import Atlas, Settings, build
from contexts.agent.application.rule_compiler import RuleCompileError, compile_rule
from contexts.circle.api import circle_router
from contexts.circle.face_share_api import face_share_router
from contexts.circle.application.sharing import CircleOffline
from contexts.day.api import day_router
from contexts.day.application.coach_feedback import coach_feedback_router
from contexts.impact.api import impact_router
from contexts.habits.api import habits_router
from contexts.learning.application.export import teach
from contexts.vault.api import vault_router
from contexts.memory.api import build_router as memory_router
from contexts.perception.api import identity_router
from contexts.perception.zones_api import zones_router
from contexts.play.dayrun_api import dayrun_router
from contexts.play.room.api import room_router
from contexts.profiles.api import profile_router
from contexts.play.league.api import league_router
from contexts.voice.proxy import voice_router
from contracts import REGISTRY, Event, parse_event
from contracts.agent import AnswerGenerated, RuleAdded, RuleToggled
from contracts.experience import ApprovalDecided, QuestionAsked
from contracts.learning import TaskSpec
from contexts.capture.api import capture_router
from contexts.learning.api import learning_router
from contexts.learning.application.lab_runs import attach as attach_trainer_runs
from contexts.learning.application.lab_runs import default_runs_dir
from contexts.learning.lab_api import lab_router
from contexts.learning.lab_bench_api import lab_bench_router
from contexts.learning.lab_import import lab_import_router
from contexts.telemetry.runtime_stats import build_runtime_stats
from runtime.api import runtime_router
from runtime.stats_api import stats_router
from runtime.modelstore import ModelStore

ASK_TIMEOUT_S = 90


MAX_SHARE_HOURS = 12  # a shared place never outlives a day (G5)
# What may be pushed in from outside (POST /api/events): what a camera or a connector would have produced.
# Never decisions, approvals or agent output: those must come from the box itself. Zone events are derived
# by the box's own zone engine from the observations, so they are not accepted either.
INPUT_EVENTS = ("perception.observation", "perception.faces.", "perception.scene.", "capture.", "connector.item.")
MAX_PUSHED = 500


class AskBody(BaseModel):
    text: str


class EnabledBody(BaseModel):
    enabled: bool


class DecisionBody(BaseModel):
    decision: Literal["approve", "deny"]


def create_app(settings: Settings | None = None, atlas: Atlas | None = None) -> FastAPI:
    atlas = atlas or build(settings)
    sockets: set[WebSocket] = set()
    waiting: dict[str, asyncio.Future] = {}
    streams: dict[str, asyncio.Queue] = {}  # question_id -> pieces for a /api/ask/stream caller

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

    async def send_all(data: str) -> None:
        for ws in list(sockets):
            try:
                await ws.send_text(data)
            except Exception:
                sockets.discard(ws)

    def stream_answer(question_id: str, text: str) -> None:
        """Called from the model's worker thread with each new piece of an answer (A11): straight to the phones,
        never to the log. The final agent.answer.generated is unchanged."""
        loop = app.state.loop
        if loop is None:
            return
        if sockets:
            data = json.dumps({"type": "agent.answer.delta", "question_id": question_id, "text": text})
            loop.call_soon_threadsafe(lambda: loop.create_task(send_all(data)))
        if (pieces := streams.get(question_id)) is not None:
            loop.call_soon_threadsafe(pieces.put_nowait, text)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.loop = asyncio.get_running_loop()
        atlas.qa.on_delta = stream_answer
        await atlas.start()
        yield
        await atlas.stop()

    app = FastAPI(title="Atlas", lifespan=lifespan)
    app.state.atlas = atlas
    app.state.loop = None
    if atlas.habits:  # GET /api/habits/today, /api/habits/{id}/history (H2)
        app.include_router(habits_router(atlas.habits), prefix="/api")
    if atlas.profile:  # GET /api/profile: the words the screens use
        app.include_router(profile_router(atlas.profile), prefix="/api")
    if atlas.memory:  # GET /memory/... (M4)
        app.include_router(memory_router(atlas.memory, atlas.settings.data_dir / "frames"))

    def authorized(authorization: str | None = Header(default=None)) -> None:
        token = atlas.settings.token
        if token and not hmac.compare_digest(authorization or "", f"Bearer {token}"):
            raise HTTPException(401, "missing or wrong token")

    if atlas.dayrun:  # /api/dayrun/* (S4, S6)
        app.include_router(dayrun_router(atlas.dayrun, publish=atlas.bus.publish, guard=authorized))
    if atlas.day:  # GET /api/day/now, GET/POST /api/day/plan, GET /api/day/patterns (D1)
        app.include_router(day_router(atlas.day, publish=atlas.bus.publish, guard=authorized, memory=atlas.memory,
                                      feedback=atlas.coach_feedback))
    if atlas.day and atlas.coach_feedback:  # GET /api/day/coach/feedback, POST /api/day/coach/corrections ("Was that right?")
        app.include_router(coach_feedback_router(atlas.day, atlas.coach_feedback, publish=atlas.bus.publish, guard=authorized))
    # GET /api/impact/story, GET/POST /api/impact/practice: the Impact screen (404 when Waste Watch is not set up)
    app.include_router(impact_router(atlas.impact, publish=atlas.bus.publish, guard=authorized))
    if atlas.policy:  # GET/PUT /vault/policy, POST /vault/wipe/request, POST /vault/wipe, GET /vault/egress
        app.include_router(vault_router(atlas.policy, atlas.bus, wiper=atlas.wiper, guard=authorized))
    if atlas.identity:  # /identity/settings, /identity/enroll/start, status, photos, scan/*; /identity/members, present, expressions (P9)
        # A person can be linked to a Circle member by id; the ids come from the configured Circle, nothing else crosses over.
        app.include_router(identity_router(atlas.identity, atlas.bus, guard=authorized, circle_members=lambda: (
            list(dict.fromkeys([atlas.circle.member_id, *(atlas.circle.members or [])])) if atlas.circle else None)))
    if atlas.capture:  # GET /capture/status (C6)
        app.include_router(capture_router(atlas.capture))
    if atlas.learning:  # GET /api/learning (X10)
        app.include_router(learning_router(atlas.learning, publish=atlas.bus.publish, guard=authorized))
    # GET /api/learning/lab/bench, /experiments, POST /routing/simulate: the lab's recorded numbers (before {run_id})
    app.include_router(lab_bench_router())
    if atlas.lab_view:  # the trainer's own step files, folded into the same stories as they are written
        attach_trainer_runs(atlas.lab_view, default_runs_dir(atlas.settings.data_dir))
        # POST /api/learning/lab/import: a run's telemetry.jsonl (or eval scores, or manifest) pushed from the lab's side
        app.include_router(lab_import_router(atlas.lab_view, atlas.bus.publish, guard=authorized))
    if atlas.lab_view:  # GET /api/learning/lab, /api/learning/lab/{run_id}: the Learning lab screen
        app.include_router(lab_router(atlas.lab_view, current_model=lambda: {
            "version": atlas.qa.runtime.model_info().version, "source": "the runtime on this box (the model that answers)"}))
    # GET /api/circle/events, /api/circle/sharing: read-only, and they answer in every state (no Circle, no Neon)
    # POST /api/circle/proposals: an approval card for a plan with friends (G1)
    app.include_router(circle_router(atlas.bus, circle=atlas.circle, sharing=atlas.sharing, presence=atlas.presence,
                                     board=atlas.board, config_dir=atlas.settings.config_dir, guard=authorized))
    # /api/circle/face-share: every endpoint answers 409 while face sharing is off (the default) or there is no Circle
    app.include_router(face_share_router(atlas.face_share, atlas.bus, guard=authorized))
    # GET /api/league, /api/league/me (200 in every state); POST /api/league/stake, /api/league/challenges -> approval cards
    app.include_router(league_router(atlas.bus, league=atlas.league, guard=authorized))
    if atlas.spot:  # /api/spot/*: the inspection (SPOT) dashboard; 200 in every state, also without a kit
        from contexts.spot.api import spot_router

        app.include_router(spot_router(atlas.spot, atlas.bus, guard=authorized))
    app.include_router(zones_router(atlas.settings.config_dir, lambda: atlas.zone_engine, guard=authorized), prefix="/api")  # set the zones from a browser
    # GET /api/room, /api/room/live, /api/play/room-route; PUT /api/room/calibration/{camera} (the room twin, X18)
    app.include_router(room_router(lambda: atlas.room, atlas.settings.config_dir, guard=authorized))
    app.include_router(voice_router(guard=authorized), prefix="/api")  # GET /api/voice/status, POST /api/voice/talk|stop|enabled (X16)
    app.include_router(runtime_router(ModelStore(atlas.settings.data_dir / "models"), llm_url=atlas.settings.llm_url))  # GET /runtime/models (R7)
    app.include_router(stats_router(build_runtime_stats(atlas.bus, atlas.telemetry).snapshot))  # GET /runtime/stats (R5)

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
        except (TimeoutError, asyncio.TimeoutError):  # distinct classes on Python 3.10, which the Jetson runs
            raise HTTPException(504, "the model took too long") from None
        finally:
            waiting.pop(qid, None)

    @app.post("/api/ask/stream", dependencies=[Depends(authorized)])
    async def ask_stream(body: AskBody):
        """/api/ask for a caller that wants to start using the answer before it is finished (the box's voice speaks the
        first sentence while the rest is still being written). Lines of JSON: {"delta": text} for each piece as it comes,
        then one {"final": <the agent.answer.generated event>}, or {"error": why}."""
        loop = asyncio.get_running_loop()
        qid = uuid4().hex
        answer = waiting[qid] = loop.create_future()
        pieces = streams[qid] = asyncio.Queue()

        async def lines():
            try:
                await atlas.bus.publish(QuestionAsked(question_id=qid, text=body.text))  # may answer before returning
                deadline = loop.time() + ASK_TIMEOUT_S
                while True:
                    next_piece = asyncio.ensure_future(pieces.get())
                    done, _ = await asyncio.wait({next_piece, answer}, timeout=max(0.0, deadline - loop.time()),
                                                 return_when=asyncio.FIRST_COMPLETED)
                    if next_piece in done:
                        yield json.dumps({"delta": next_piece.result()}) + "\n"
                        continue
                    next_piece.cancel()
                    if answer in done:
                        while not pieces.empty():
                            yield json.dumps({"delta": pieces.get_nowait()}) + "\n"
                        yield json.dumps({"final": answer.result().model_dump(mode="json")}) + "\n"
                    else:
                        yield json.dumps({"error": "the model took too long"}) + "\n"
                    return
            finally:
                waiting.pop(qid, None)
                streams.pop(qid, None)

        return StreamingResponse(lines(), media_type="application/x-ndjson")

    @app.get("/api/approvals")
    async def approvals():
        return [r.model_dump(mode="json") for r in atlas.approvals.pending()]

    @app.post("/api/approvals/{action_id}", dependencies=[Depends(authorized)])
    async def decide(action_id: str, body: DecisionBody):
        req = atlas.approvals.requests.get(action_id)
        if req is None:
            raise HTTPException(404, "no such request")
        if req.status != "proposed":  # a second tap on the phone: say so, and log nothing
            raise HTTPException(409, f"already {req.status}")
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

    @app.get("/api/rules")
    async def list_rules():
        return atlas.rules.listing() if atlas.rules else []

    @app.post("/api/rules/{rule_id}/enabled", dependencies=[Depends(authorized)])
    async def toggle_rule(rule_id: str, body: EnabledBody):
        if atlas.rules is None or not atlas.rules.known(rule_id):
            raise HTTPException(404, "no such rule")
        await atlas.bus.publish(RuleToggled(rule_id=rule_id, enabled=body.enabled))
        return next(r for r in atlas.rules.listing() if r["id"] == rule_id)

    @app.get("/api/preflight")
    async def preflight_checks():
        from contexts.gateway.preflight import run_checks

        checks = await asyncio.to_thread(run_checks, atlas)
        return {"ok": all(c["ok"] for c in checks), "checks": checks}

    @app.post("/api/preflight/reset", dependencies=[Depends(authorized)])
    async def preflight_reset():
        from contexts.gateway.preflight import reset

        done = reset(atlas)
        await atlas.bus.publish_all(done.pop("events"))
        return done

    @app.post("/api/events", dependencies=[Depends(authorized)])
    async def push_events(body: dict | list[dict]):
        """Push camera or connector input into the running box, as if a camera saw it (`python -m contexts.demo
        fake perception --gateway ...`). Every event is checked against its contract before any is published."""
        raw = body if isinstance(body, list) else [body]
        if len(raw) > MAX_PUSHED:
            raise HTTPException(413, f"at most {MAX_PUSHED} events per request")
        events = []
        for i, item in enumerate(raw):
            kind = str(item.get("type", ""))
            if not kind.startswith(INPUT_EVENTS):
                raise HTTPException(403, f"event {i}: {kind or 'no type'} cannot be pushed in; only camera and connector "
                                         f"input ({', '.join(INPUT_EVENTS)})")
            if kind not in REGISTRY:
                raise HTTPException(422, f"event {i}: unknown type {kind}")
            try:
                events.append(parse_event({"event_id": uuid4().hex, **item}))
            except (ValueError, KeyError) as exc:
                raise HTTPException(422, f"event {i} ({kind}): {exc}") from None
        await atlas.bus.publish_all(events)
        return {"published": len(events), "types": sorted({e.type for e in events})}

    @app.get("/api/alerts")
    async def recent_alerts(limit: int = Query(50, ge=1, le=200)):
        return atlas.alerts.recent(limit) if atlas.alerts else []

    @app.post("/api/rules", dependencies=[Depends(authorized)])
    async def add_rule(body: AskBody):
        try:
            rule = await compile_rule(body.text, atlas.qa.runtime, atlas.zones)
        except RuleCompileError as exc:
            raise HTTPException(422, str(exc)) from None
        await atlas.bus.publish(RuleAdded(rule=rule.model_dump(), source_text=body.text))
        return rule.model_dump()

    @app.get("/api/play/route")
    async def play_route(item: str, x: float = Query(0.75, ge=0, le=1), y: float = Query(0.75, ge=0, le=1)):
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
        hours: float = Field(2, gt=0, le=MAX_SHARE_HOURS)

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

    @app.delete("/api/presence", dependencies=[Depends(authorized)])
    async def stop_sharing():
        """Stop sharing my place now, before it expires: a share that ends now replaces the current one."""
        if atlas.presence is None:
            raise HTTPException(404, "the Circle is not configured")
        mine = atlas.presence.current.get(atlas.presence.me)
        now = datetime.now(UTC)
        if mine is None or mine.expires_at <= now:
            return {"sharing": False}
        await atlas.bus.publish(mine.model_copy(update={"event_id": uuid4().hex, "ts": now, "expires_at": now}))
        return {"sharing": False}

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
        try:
            impact, habits, scores = await asyncio.gather(
                asyncio.to_thread(atlas.board.impact_leaderboard, cid),
                asyncio.to_thread(atlas.board.habit_leaderboard, cid),
                asyncio.to_thread(atlas.board.top_scores, cid),
            )
        except CircleOffline as exc:
            raise HTTPException(503, f"the Circle is offline: {exc}") from None
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
                 "places": {pid: pl.model_dump(exclude_none=True) for pid, pl in p.places.items()},
                 **({"map": f"/api/places/map/{p.name}"} if p.map_file else {})}
                for p in atlas.places.packs.values()]

    @app.get("/api/places/map/{pack}")
    async def place_map(pack: str):
        """The campus as drawn and walked: building footprints, greens, water, roads and the footpath network
        (config/places/<pack>.map.json, built by scripts/build_campus_pack.py). Map data © OpenStreetMap contributors."""
        found = atlas.places.packs.get(pack) if atlas.places else None
        if found is None or found.map_file is None or not found.map_file.is_file():
            raise HTTPException(404, f"no map for the {pack!r} pack")
        return FileResponse(found.map_file, media_type="application/json", headers={"Cache-Control": "public, max-age=3600"})

    def place_ref(text: str, what: str) -> str:
        """A place from a pack id ("ucf.eng2"), a name or alias ("Engineering II", "HEC"), a friend ("friend:maya")
        or "here" (where the box last saw you)."""
        text = text.strip()
        if text.startswith("friend:"):
            member = text.split(":", 1)[1]
            shared = {p.member_id.lower(): p for p in atlas.presence.friends(datetime.now(UTC))} if atlas.presence else {}
            if member.lower() not in shared:
                raise HTTPException(404, f"{member} isn't sharing a place right now")
            found = atlas.places.find_shared(shared[member.lower()].place)
            if not found:
                raise HTTPException(404, f"{shared[member.lower()].place!r} isn't on any map")
            return found
        if text.lower() == "here":
            tracker = getattr(atlas.day, "tracker", None)
            seen = (tracker.status.seen_at if tracker and tracker.status and tracker.status.seen_at
                    else tracker.seen[0] if tracker and tracker.seen else None)
            if seen is None and atlas.dayrun is not None:
                seen = atlas.dayrun.walks.ref(atlas.dayrun.cfg.start_place)
            if seen is None:
                raise HTTPException(404, "the box has not seen where you are yet")
            return seen
        pack, _, place = text.partition(".")
        if place and pack in atlas.places.packs and (pid := atlas.places.packs[pack].find(place)):
            return f"{pack}.{pid}"
        found = atlas.places.find_shared(text) or atlas.places.locate(text)
        if not found:
            raise HTTPException(422, f"unknown {what} place {text!r} (use pack.place such as ucf.library, a place's name, or friend:<member>)")
        return found

    @app.get("/api/places/route")
    async def place_route(src: str = Query("here", alias="from"), dst: str = Query(alias="to"),
                          arrive_by: datetime | None = None, depart: datetime | None = None):
        """When to leave. The walk from one place to another along the real footpaths, with the time to set off
        for `arrive_by`, or the time you get there if you leave at `depart` (now, when neither is given)."""
        from contexts.play.domain.walknet import leave_sentence

        if atlas.places is None:
            raise HTTPException(404, "no place packs loaded")
        a, b = place_ref(src, "starting"), place_ref(dst, "destination")
        try:
            path, minutes = atlas.places.route(a, b)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        line, meters = atlas.places.trace(path)
        whole = 0 if a == b else max(1, math.ceil(minutes - 1e-9))  # whole minutes, rounded up, as the day plan counts walks
        tz = getattr(getattr(atlas.day, "clock", None), "tz", None) or UTC
        if arrive_by is not None:
            arrive = arrive_by if arrive_by.tzinfo else arrive_by.replace(tzinfo=tz)
            leave, mode = arrive - timedelta(minutes=whole), "arrive_by"
        else:
            leave = (depart if depart.tzinfo else depart.replace(tzinfo=tz)) if depart is not None else datetime.now(UTC).astimezone(tz)
            arrive, mode = leave + timedelta(minutes=whole), "depart"

        def get(ref: str):
            pack, place = ref.split(".", 1)
            return atlas.places.packs[pack].places[place]

        def clock12(t: datetime) -> str:
            return t.strftime("%I:%M %p").lstrip("0")

        to_name = get(b).name.split(" (")[0]
        if a == b:
            sentence = f"You are already at {to_name}"
        elif mode == "arrive_by":
            sentence = leave_sentence(clock12(leave), whole, to_name, meters)
        else:
            sentence = leave_sentence(clock12(leave), whole, to_name, meters).replace("Leave by", "Leave at", 1) + f": there by {clock12(arrive)}"
        return {"from": a, "to": b, "from_name": get(a).name.split(" (")[0], "to_name": to_name, "mode": mode,
                "depart": leave.isoformat(), "arrive": arrive.isoformat(), "minutes": minutes, "walk_minutes": whole,
                "meters": meters, "path": path, "polyline": line, "sentence": sentence,
                "from_xy": [get(a).lat, get(a).lon] if get(a).lat is not None else None,
                "to_xy": [get(b).lat, get(b).lon] if get(b).lat is not None else None,
                "attribution": "© OpenStreetMap contributors" if line else None}

    @app.post("/api/places", dependencies=[Depends(authorized)])
    async def upload_place_pack(request: Request):
        """Upload a place pack as YAML or JSON. It is validated (known places, all reachable, nesting) before use."""
        from contexts.play.domain.places import PlaceSet, parse_pack

        from contexts.play.dayrun.service import SCENARIO_NAME

        try:
            text = (await request.body()).decode("utf-8")
            pack = parse_pack(text)
            if not SCENARIO_NAME.match(pack.name):  # the name becomes a file name under data/places
                raise ValueError("pack name must be 1 to 40 of a-z, 0-9, _ or -")
            if atlas.places is None:
                atlas.places = PlaceSet({})
            atlas.places.add(pack)
        except (ValueError, yaml.YAMLError, RecursionError) as exc:  # pydantic's ValidationError is a ValueError
            raise HTTPException(422, str(exc)[:500]) from None
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
        line, meters = atlas.places.trace(path)
        return {"path": path, "minutes": minutes, **({"meters": meters, "polyline": line} if line else {})}

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
