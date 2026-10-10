"""GET /runtime/stats: the box's and the lab's measured numbers, each with the record it came from.

    app.include_router(stats_router(stats.snapshot))

The numbers are assembled by the Telemetry context (contexts/telemetry/application/stats.py) from what is already
recorded: the lab's metrics file and ledger, the telemetry projection and the learning events in the log. This module
only serves them; it measures nothing and calls no model server, pod or hub.
A figure nothing has recorded is served as null with its reason; which figures have no producer yet is listed in
that module's docstring.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import APIRouter


def stats_router(snapshot: Callable[[], Awaitable[dict[str, Any]]]) -> APIRouter:
    router = APIRouter()

    @router.get("/runtime/stats")
    async def stats() -> dict[str, Any]:
        return await snapshot()

    return router
