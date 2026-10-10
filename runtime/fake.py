"""Deterministic Runtime for laptops and tests. No models, no GPU."""

import hashlib
from collections.abc import Callable
from typing import Any

from contracts.types import ChatMessage, ChatResult, Detection, ModelVersion, ObjectHit, Track
from runtime.interface import Priority


class FakeRuntime:
    def __init__(
        self,
        chat_replies: list[ChatResult] | None = None,
        objects: list[ObjectHit] | None = None,
        caption_text: str = "A person sits at a table with a laptop.",
        detections: list[Detection] | None = None,
    ) -> None:
        self.detections = detections or []
        self.chat_replies = list(chat_replies or [])
        self.objects = objects or []
        self.caption_text = caption_text
        self.calls: list[tuple[str, Any]] = []

    def detect_track(self, frame: Any) -> list[Track]:
        self.calls.append(("detect_track", None))
        return []

    def detect_open(self, jpeg: bytes, labels: list[str]) -> list[Detection]:
        self.calls.append(("detect_open", labels))
        return [d for d in self.detections if d.label in labels]

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(("embed_text", texts))
        return [_vector(t.encode()) for t in texts]

    def embed_image(self, jpeg: bytes) -> list[float]:
        self.calls.append(("embed_image", None))
        return _vector(jpeg)

    def caption(self, jpeg: bytes, prompt: str = "Describe the scene in one sentence.") -> str:
        self.calls.append(("caption", prompt))
        return self.caption_text

    def find_objects(self, jpeg: bytes, items: list[str] | None = None) -> list[ObjectHit]:
        self.calls.append(("find_objects", items))
        return [o for o in self.objects if items is None or o.label in items]

    def summarize(self, captions: list[str]) -> str:
        self.calls.append(("summarize", captions))
        return " ".join(captions[:2])

    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict] | None = None,
        priority: Priority = "chat",
        json_schema: dict | None = None,
        on_text: Callable[[str], None] | None = None,
    ) -> ChatResult:
        self.calls.append(("chat", messages))
        self.last_json_schema = json_schema
        reply = self.chat_replies.pop(0) if self.chat_replies else ChatResult(content=f"(fake) {messages[-1].content}",
                                                                               model_version="fake-0")
        if on_text and not tools and not json_schema and reply.content:
            for i, word in enumerate(reply.content.split(" ")):  # streams like a model: word by word
                on_text(word if i == 0 else " " + word)
        return reply

    def model_info(self) -> ModelVersion:
        return ModelVersion(version="fake-0", base="fake")


def _vector(data: bytes, dim: int = 8) -> list[float]:
    """Deterministic unit-ish vector from bytes, so tests can compare embeddings."""
    digest = hashlib.sha256(data).digest()
    return [b / 255 for b in digest[:dim]]
