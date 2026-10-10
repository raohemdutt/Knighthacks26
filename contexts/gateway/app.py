"""Gateway (X2): the phone's one door into Atlas.

    python -m contexts.gateway            # serves on :8000

- WS  /ws                      every bus event as JSON, live (the PWA's feed)
- POST /api/ask                {text} -> the answer with citations (also on /ws)
- GET  /api/approvals          pending approval cards
- POST /api/approvals/{id}     {decision: approve|deny} (ntfy buttons call this)
- POST /api/teach              {task, instruction, ...} -> approval card for the export
- GET  /health

Writes need `Authorization: Bearer $ATLAS_TOKEN` when ATLAS_TOKEN is set; the
tailnet keeps strangers out, the token keeps a guest on the Wi-Fi from
approving things.
"""

import asyncio
import contextlib
import hmac
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from contexts.gateway.bootstrap import Atlas, Settings, build
from contexts.learning.application.export import teach
from contracts import Event
from contracts.agent import AnswerGenerated
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
