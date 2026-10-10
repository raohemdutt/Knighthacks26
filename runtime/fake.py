"""Deterministic Runtime for laptops and tests. No models, no GPU."""

from typing import Any

from contracts.types import ChatMessage, ChatResult, ModelVersion, ObjectHit, Track
from runtime.interface import Priority


class FakeRuntime:
    def __init__(
        self,
        chat_replies: list[ChatResult] | None = None,
        objects: list[ObjectHit] | None = None,
        caption_text: str = "A person sits at a table with a laptop.",
    ) -> None:
        self.chat_replies = list(chat_replies or [])
        self.objects = objects or []
        self.caption_text = caption_text
        self.calls: list[tuple[str, Any]] = []

    def detect_track(self, frame: Any) -> list[Track]:
        self.calls.append(("detect_track", None))
        return []

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
    ) -> ChatResult:
        self.calls.append(("chat", messages))
        if self.chat_replies:
            return self.chat_replies.pop(0)
        return ChatResult(content=f"(fake) {messages[-1].content}", model_version="fake-0")

    def model_info(self) -> ModelVersion:
        return ModelVersion(version="fake-0", base="fake")
