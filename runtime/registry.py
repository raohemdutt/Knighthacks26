"""Build the Runtime from config/models.yaml (R1): config picks the models, not code.

The deployed Atlas model's version comes from its package manifest when one is
installed (`data/models/<version>/manifest.json`, written by lab/package.sh and
switched by the hot-swap in R7), so answers always name the model that gave them.
"""

import os
from pathlib import Path

import yaml

from contracts.learning import ModelManifest
from contracts.types import ModelVersion
from runtime.embeddings import image_embedder_from, text_embedder_from
from runtime.fake import FakeRuntime
from runtime.interface import Runtime


def load_models_config(path: str | Path = "config/models.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text()) or {}


def model_version(cfg: dict) -> ModelVersion:
    """From the installed manifest if there is one, else from models.yaml."""
    manifest = cfg.get("manifest")
    if manifest and Path(manifest).exists():
        m = ModelManifest.model_validate_json(Path(manifest).read_text())
        return ModelVersion(version=m.version, base=m.base, task=m.task, scores=m.scores)
    return ModelVersion(version=cfg.get("version", "atlas-base"), base=cfg.get("base", "Qwen/Qwen3.5-4B"))


def build_runtime(path: str | Path = "config/models.yaml", llm_url: str | None = None) -> Runtime:
    """`llm_url` (or ATLAS_LLM_URL) overrides the Atlas model's URL, e.g. on a laptop tunnelled to the Jetson."""
    cfg = load_models_config(path)
    atlas = cfg.get("atlas_model", {})
    backend = "llama-server" if (llm_url or os.environ.get("ATLAS_LLM_URL")) else atlas.get("backend", "fake")
    if backend == "fake":
        return FakeRuntime()
    if backend != "llama-server":
        raise ValueError(f"unknown atlas_model backend {backend!r}")

    from runtime.llama import LlamaServerRuntime
    from runtime.scheduler import GpuScheduler

    emb = cfg.get("embeddings", {})
    return LlamaServerRuntime(
        model_version(atlas),
        base_url=llm_url or os.environ.get("ATLAS_LLM_URL") or atlas.get("url", "http://127.0.0.1:8080"),
        text_embedder=text_embedder_from(emb.get("text")),
        image_embedder=image_embedder_from(emb.get("image")),
        timeout_s=float(atlas.get("timeout_s", 120)),
        scheduler=GpuScheduler.from_config(cfg.get("scheduler")),
    )
