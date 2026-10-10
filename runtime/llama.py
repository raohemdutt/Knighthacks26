"""Runtime backed by llama-server (llama.cpp, OpenAI-compatible API) on the Jetson.

Start the server with the Atlas model and its vision projector, e.g.
    llama-server -m atlas.gguf --mmproj mmproj.gguf --jinja --port 8080
`--jinja` is needed for tool calls. Detection is a separate engine (P1) and is
plugged in as a callable so this module never imports ultralytics.
"""

import base64
import json
import re
import time
from collections.abc import Callable
from typing import Any

import httpx

from contracts.types import ChatMessage, ChatResult, ModelVersion, ObjectHit, ToolCall, Track
from runtime.interface import Priority

FIND_OBJECTS_PROMPT = (
    "List the small objects you can see. Reply with only a JSON array like "
    '[{"label": "keys", "position": "left of the laptop"}].'
)


def _image_part(jpeg: bytes) -> dict:
    url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
    return {"type": "image_url", "image_url": {"url": url}}


def _parse_json_array(text: str) -> list[dict]:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [i for i in items if isinstance(i, dict) and "label" in i]


class LlamaServerRuntime:
    def __init__(
        self,
        model: ModelVersion,
        base_url: str = "http://127.0.0.1:8080",
        detector: Callable[[Any], list[Track]] | None = None,
        load_image: Callable[[str], bytes] | None = None,
        timeout_s: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.detector = detector
        self.load_image = load_image  # frame_ref -> jpeg bytes, for ChatMessage.images
        self.http = httpx.Client(base_url=base_url, timeout=timeout_s, transport=transport)

    def detect_track(self, frame: Any) -> list[Track]:
        if self.detector is None:
            raise RuntimeError("no detector configured")
        return self.detector(frame)

    def caption(self, jpeg: bytes, prompt: str = "Describe the scene in one sentence.") -> str:
        content = [_image_part(jpeg), {"type": "text", "text": prompt}]
        return self._complete([{"role": "user", "content": content}], max_tokens=80)["content"].strip()

    def find_objects(self, jpeg: bytes, items: list[str] | None = None) -> list[ObjectHit]:
        prompt = FIND_OBJECTS_PROMPT
        if items:
            prompt += " Only include: " + ", ".join(items) + "."
        content = [_image_part(jpeg), {"type": "text", "text": prompt}]
        text = self._complete([{"role": "user", "content": content}], max_tokens=256)["content"]
        hits = [ObjectHit(label=i["label"], position=str(i.get("position", ""))) for i in _parse_json_array(text)]
        return [h for h in hits if items is None or h.label in items]

    def summarize(self, captions: list[str]) -> str:
        prompt = "Summarize what happened in one or two sentences:\n" + "\n".join(f"- {c}" for c in captions)
        return self._complete([{"role": "user", "content": prompt}], max_tokens=120)["content"].strip()

    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict] | None = None,
        priority: Priority = "chat",
    ) -> ChatResult:
        start = time.monotonic()
        msg = self._complete([self._to_openai(m) for m in messages], tools=tools, max_tokens=512)
        calls = [
            ToolCall(id=c["id"], name=c["function"]["name"], arguments=json.loads(c["function"]["arguments"] or "{}"))
            for c in msg.get("tool_calls") or []
        ]
        return ChatResult(
            content=msg.get("content") or "",
            tool_calls=calls,
            model_version=self.model.version,
            latency_ms=int((time.monotonic() - start) * 1000),
        )

    def model_info(self) -> ModelVersion:
        return self.model

    def _to_openai(self, m: ChatMessage) -> dict:
        out: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.images and self.load_image:
            out["content"] = [_image_part(self.load_image(r)) for r in m.images] + [{"type": "text", "text": m.content}]
        if m.tool_calls:
            out["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                for c in m.tool_calls
            ]
        if m.tool_call_id:
            out["tool_call_id"] = m.tool_call_id
        return out

    def _complete(self, messages: list[dict], tools: list[dict] | None = None, max_tokens: int = 256) -> dict:
        body: dict[str, Any] = {"messages": messages, "max_tokens": max_tokens, "temperature": 0.2}
        if tools:
            body["tools"] = tools
        r = self.http.post("/v1/chat/completions", json=body)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]
