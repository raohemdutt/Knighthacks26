"""The only way contexts touch models (anti-corruption layer, R1).

Contexts depend on `Runtime`, never on llama.cpp or ultralytics directly, so a
new model version is a registry change, not a code change. Everything here
runs on the Jetson; the lab GPUs are never in the request path.

Priority on the device (enforced by the scheduler, R4):
detection > alerts > chat > captions > summaries > backfill.
"""

from typing import Any, Literal, Protocol

from contracts.types import ChatMessage, ChatResult, ModelVersion, ObjectHit, Track

Priority = Literal["chat", "caption", "summary", "backfill"]


class Runtime(Protocol):
    def detect_track(self, frame: Any) -> list[Track]:
        """Detect + track on one BGR frame (numpy array). Real-time path."""
        ...

    def caption(self, jpeg: bytes, prompt: str = "Describe the scene in one sentence.") -> str: ...

    def find_objects(self, jpeg: bytes, items: list[str] | None = None) -> list[ObjectHit]:
        """Small objects in a full-resolution crop; `items` narrows to watch items."""
        ...

    def summarize(self, captions: list[str]) -> str: ...

    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict] | None = None,
        priority: Priority = "chat",
    ) -> ChatResult:
        """`tools` are OpenAI-style function schemas."""
        ...

    def model_info(self) -> ModelVersion: ...
