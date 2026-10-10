"""Embedding backends behind Runtime.embed_text / embed_image (R1).

- text: any OpenAI-compatible /v1/embeddings server, e.g. a second llama-server
  started with `--embedding` and a small model such as bge-m3 or
  nomic-embed-text (GGUF). Runs on the Jetson.
- image: CLIP through sentence-transformers, loaded lazily so the import cost
  is paid only if image search is turned on (Jetson memory is tight: P2).
"""

from collections.abc import Callable

import httpx


class HttpTextEmbedder:
    def __init__(self, base_url: str, model: str | None = None, timeout_s: float = 30.0,
                 transport: httpx.BaseTransport | None = None) -> None:
        self.model = model
        self.http = httpx.Client(base_url=base_url, timeout=timeout_s, transport=transport)

    def __call__(self, texts: list[str]) -> list[list[float]]:
        body: dict = {"input": texts}
        if self.model:
            body["model"] = self.model
        r = self.http.post("/v1/embeddings", json=body)
        r.raise_for_status()
        data = sorted(r.json()["data"], key=lambda d: d.get("index", 0))
        return [d["embedding"] for d in data]


class ClipImageEmbedder:
    """CLIP image vectors; the matching text side is ClipTextEmbedder for image search by words."""

    def __init__(self, model_name: str = "clip-ViT-B-32") -> None:
        self.model_name = model_name
        self._model = None

    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # optional dependency

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def __call__(self, jpeg: bytes) -> list[float]:
        import io

        from PIL import Image

        return self.model().encode(Image.open(io.BytesIO(jpeg))).tolist()


def text_embedder_from(cfg: dict | None) -> Callable[[list[str]], list[list[float]]] | None:
    if not cfg or cfg.get("backend") in (None, "none"):
        return None
    if cfg["backend"] == "openai-http":
        return HttpTextEmbedder(cfg["url"], cfg.get("model"))
    raise ValueError(f"unknown text embedding backend {cfg['backend']!r}")


def image_embedder_from(cfg: dict | None) -> Callable[[bytes], list[float]] | None:
    if not cfg or cfg.get("backend") in (None, "none"):
        return None
    if cfg["backend"] == "clip":
        return ClipImageEmbedder(cfg.get("model", "clip-ViT-B-32"))
    raise ValueError(f"unknown image embedding backend {cfg['backend']!r}")
