"""GET /api/learning: where each taught task is, for the Teach screen.

    app.include_router(learning_router(atlas.learning))

POST /api/learning/deployed {model_version, previous_version} publishes learning.model.deployed (after a model swap).

{"tasks": [{"task", "instruction", "stage", "progress", "loss", "message", "scores", "decision", "model_version",
            "dataset_id", "examples", "deployed", "created", "updated"}], "deployments": [{"model_version", ...}]}
"""

from collections.abc import Awaitable, Callable

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from contexts.learning.application.board import LearningBoard
from contracts.learning import ModelDeployed


class DeployedBody(BaseModel):
    model_version: str
    previous_version: str | None = None


def learning_router(board: LearningBoard, publish: Callable[[ModelDeployed], Awaitable[None]] | None = None,
                    guard: Callable | None = None) -> APIRouter:
    router = APIRouter()

    @router.get("/api/learning")
    def learning() -> dict:
        return board.snapshot()

    if publish is not None:

        @router.post("/api/learning/deployed", dependencies=[Depends(guard)] if guard else [])
        async def deployed(body: DeployedBody) -> dict:
            """The box switched models (jetson/atlas_swap.py calls this after a swap that answered its test question):
            publishes `learning.model.deployed`, which the Teach screen shows."""
            await publish(ModelDeployed(model_version=body.model_version, previous_version=body.previous_version))
            return {"ok": True}

    return router
