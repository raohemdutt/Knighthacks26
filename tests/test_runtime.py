import json

import httpx

from contracts.types import ChatMessage, ModelVersion
from runtime import FakeRuntime, Runtime
from runtime.llama import LlamaServerRuntime


def _server(reply: dict):
    seen: list[dict] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": reply}]})

    return httpx.MockTransport(handle), seen


def _runtime(reply: dict):
    transport, seen = _server(reply)
    rt = LlamaServerRuntime(ModelVersion(version="atlas-v1", base="Qwen/Qwen3.5-4B"), transport=transport)
    return rt, seen


def test_both_runtimes_satisfy_the_protocol():
    rt: Runtime = FakeRuntime()
    rt, _ = _runtime({"content": ""})
    assert isinstance(rt.model_info(), ModelVersion)


def test_find_objects_parses_json_inside_chatter():
    rt, seen = _runtime({"content": 'Sure: [{"label": "keys", "position": "by the mug"}, {"label": "pen"}]'})
    hits = rt.find_objects(b"jpeg", items=["keys"])
    assert [(h.label, h.position) for h in hits] == [("keys", "by the mug")]
    assert seen[0]["messages"][0]["content"][0]["type"] == "image_url"


def test_find_objects_tolerates_garbage():
    rt, _ = _runtime({"content": "I see nothing"})
    assert rt.find_objects(b"jpeg") == []


def test_chat_returns_tool_calls():
    reply = {
        "content": None,
        "tool_calls": [{"id": "c1", "function": {"name": "last_seen", "arguments": '{"label": "keys"}'}}],
    }
    rt, seen = _runtime(reply)
    result = rt.chat([ChatMessage(role="user", content="where are my keys")], tools=[{"type": "function"}])
    assert result.tool_calls[0].name == "last_seen"
    assert result.tool_calls[0].arguments == {"label": "keys"}
    assert result.model_version == "atlas-v1"
    assert seen[0]["tools"] == [{"type": "function"}]
