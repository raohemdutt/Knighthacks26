"""The only way contexts touch models (anti-corruption layer, R1).

Contexts depend on `Runtime`, never on llama.cpp or ultralytics directly, so a
new model version is a registry change, not a code change. Everything here
runs on the Jetson; the lab GPUs are never in the request path.

Priority on the device (enforced by the scheduler, R4):
detection > alerts > chat > captions > summaries > backfill.
"""

from typing import Any, Literal, Protocol

from contracts.types import ChatMessage, ChatResult, Detection, ModelVersion, ObjectHit, Track

Priority = Literal["chat", "caption", "summary", "backfill"]


class Runtime(Protocol):
    def detect_track(self, frame: Any) -> list[Track]:
        """Detect + track on one BGR frame (numpy array). Real-time path."""
        ...

    def detect_open(self, jpeg: bytes, labels: list[str]) -> list[Detection]:
        """Open-vocabulary detection: find things the detector has no class for ("keys", "charger").

        Boxes are normalized [0, 1]. Slower than detect_track; use on keyframes and crops."""
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
        json_schema: dict | None = None,
    ) -> ChatResult:
        """`tools` are OpenAI-style function schemas. `json_schema` constrains the reply to JSON
        matching that schema (llama-server grammar-constrained decoding), for structured output."""
        ...

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        """One vector per text, for semantic search over captions and items (M5)."""
        ...

    def embed_image(self, jpeg: bytes) -> list[float]:
        """One vector per image, in the same space as embed_text when the backend supports it."""
        ...

    def model_info(self) -> ModelVersion: ...
