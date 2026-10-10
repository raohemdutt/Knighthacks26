import json
from datetime import UTC, datetime

import httpx
import pytest

from contracts.learning import ModelManifest
from contracts.types import ModelVersion
from runtime import FakeRuntime
from runtime.embeddings import HttpTextEmbedder
from runtime.llama import LlamaServerRuntime
from runtime.registry import build_runtime, model_version


def llama(reply: str, **kw) -> LlamaServerRuntime:
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"choices": [{"message": {"content": reply}}]}))
    return LlamaServerRuntime(ModelVersion(version="v", base="b"), transport=transport, **kw)


def test_detect_open_normalizes_1000_scale_boxes_and_drops_junk():
    reply = ('[{"label": "keys", "box": [100, 200, 300, 400]}, {"label": "keys", "box": [5, 5, 1, 1]}, '
             '{"label": "cat", "box": [0, 0, 10, 10]}, {"label": "charger", "bbox_2d": [0, 900, 100, 1200]}]')
    hits = llama(reply).detect_open(b"jpeg", ["keys", "charger"])
    assert [(h.label, h.box.x1, h.box.y1, h.box.x2, h.box.y2) for h in hits] == [
        ("keys", 0.1, 0.2, 0.3, 0.4),
        ("charger", 0.0, 0.9, 0.1, 1.0),  # clamped to the image
    ]


def test_text_embeddings_over_openai_http():
    def handle(request):
        body = json.loads(request.content)
        assert body == {"input": ["a", "b"], "model": "bge-m3"}
        return httpx.Response(200, json={"data": [{"index": 1, "embedding": [0.0, 1.0]}, {"index": 0, "embedding": [1.0, 0.0]}]})

    emb = HttpTextEmbedder("http://emb.test", "bge-m3", transport=httpx.MockTransport(handle))
    rt = llama("", text_embedder=emb)
    assert rt.embed_text(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]
    with pytest.raises(RuntimeError, match="image embedding"):
        rt.embed_image(b"jpeg")


def test_fake_runtime_is_deterministic():
    rt = FakeRuntime()
    assert rt.embed_text(["x"]) == rt.embed_text(["x"]) != rt.embed_text(["y"])
    assert rt.detect_open(b"", ["keys"]) == []


def test_models_yaml_drives_the_choice(tmp_path, monkeypatch):
    monkeypatch.delenv("ATLAS_LLM_URL", raising=False)
    assert isinstance(build_runtime("config/models.yaml"), FakeRuntime)  # the checked-in default
    cfg = tmp_path / "models.yaml"
    cfg.write_text("atlas_model: {backend: llama-server, url: 'http://jetson:8080', version: atlas-v2}\n"
                   "embeddings: {text: {backend: openai-http, url: 'http://jetson:8081'}}\n")
    rt = build_runtime(cfg)
    assert isinstance(rt, LlamaServerRuntime) and rt.model_info().version == "atlas-v2"
    assert str(rt.http.base_url) == "http://jetson:8080" and rt.text_embedder is not None
    assert isinstance(build_runtime("config/models.yaml", llm_url="http://tunnel:8080"), LlamaServerRuntime)


def test_installed_manifest_names_the_model(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(ModelManifest(version="atlas-v3-desk-objects", base="Qwen/Qwen3.5-4B", task="desk-objects",
                                      gguf="m.gguf", scores={"candidate": 0.78}, created=datetime.now(UTC)).model_dump_json())
    v = model_version({"manifest": str(manifest), "version": "ignored"})
    assert (v.version, v.task, v.scores) == ("atlas-v3-desk-objects", "desk-objects", {"candidate": 0.78})
