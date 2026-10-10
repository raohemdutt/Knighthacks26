"""GET /api/learning: where each taught task is, for the Teach screen.

    app.include_router(learning_router(atlas.learning))

{"tasks": [{"task", "instruction", "stage", "progress", "loss", "message", "scores", "decision", "model_version",
            "dataset_id", "examples", "deployed", "created", "updated"}], "deployments": [{"model_version", ...}]}
"""

from fastapi import APIRouter

from contexts.learning.application.board import LearningBoard


def learning_router(board: LearningBoard) -> APIRouter:
    router = APIRouter()

    @router.get("/api/learning")
    def learning() -> dict:
        return board.snapshot()

    return router
