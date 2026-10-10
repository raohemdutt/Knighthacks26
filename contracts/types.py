"""Value objects shared across contexts (no behavior, no I/O)."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

CameraId = str  # "cam-door"
ZoneId = str  # "door"
FrameRef = str  # "frames:<camera_id>:<ts_ms>" -> data/frames/<camera_id>/<ts_ms>.jpg


def frame_ref(camera_id: CameraId, ts_ms: int) -> FrameRef:
    return f"frames:{camera_id}:{ts_ms}"


class Box(BaseModel):
    """Normalized [0, 1] coordinates, origin top-left."""

    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    label: str
    confidence: float
    box: Box


class Track(Detection):
    track_id: int


class ObjectHit(BaseModel):
    """A small object found by the Atlas model in a crop, e.g. keys left of the laptop."""

    label: str
    position: str  # free text from the model
    box: Box | None = None
    confidence: float | None = None


class ModelVersion(BaseModel):
    version: str  # "atlas-v3-desk-objects"
    base: str  # "Qwen/Qwen3.5-4B"
    task: str | None = None  # LoRA task, None for the base model
    scores: dict[str, float] = {}


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    images: list[FrameRef] = []
    tool_calls: list[ToolCall] = []
    tool_call_id: str | None = None


class ChatResult(BaseModel):
    content: str
    tool_calls: list[ToolCall] = []
    model_version: str
    latency_ms: int | None = None


class Citation(BaseModel):
    episode_id: str | None = None
    frame_ref: FrameRef | None = None
    ts: datetime | None = None
