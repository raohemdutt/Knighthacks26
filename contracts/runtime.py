from typing import Literal

from contracts.events import TypedEvent, register
from contracts.types import ModelVersion


@register
class ModelChanged(TypedEvent):
    type: Literal["runtime.model.changed"] = "runtime.model.changed"
    model: ModelVersion
    previous_version: str | None = None


@register
class BackendHealth(TypedEvent):
    type: Literal["runtime.backend.health"] = "runtime.backend.health"
    backend: Literal["detector", "atlas_model", "face_blur"]
    status: Literal["ok", "loading", "down"]
    detail: str | None = None
