"""GET /runtime/models: which model versions are installed, which one the box is serving.

    app.include_router(runtime_router(ModelStore(data_dir / "models"), llm_url=settings.llm_url))

`serving` is what llama-server itself reports (its /props), so if a switch half-failed the phone
shows the mismatch (`in_sync: false`) instead of the version the link claims.
"""

from pathlib import Path

import httpx
from fastapi import APIRouter

from runtime.modelstore import ModelStore


def serving_model(llm_url: str | None, timeout_s: float = 2.0) -> dict:
    """What the running llama-server says it loaded; {} if it does not answer."""
    if not llm_url:
        return {}
    try:
        props = httpx.get(llm_url.rstrip("/") + "/props", timeout=timeout_s).json()
    except (httpx.HTTPError, ValueError):
        return {}
    path = props.get("model_path") or ""
    return {"model_path": path, "folder": Path(path).parent.name if path else None}


def runtime_router(store: ModelStore, llm_url: str | None = None) -> APIRouter:
    router = APIRouter()

    @router.get("/runtime/models")
    def models() -> dict:
        current = store.current()
        serving = serving_model(llm_url)
        installed = []
        for m in store.versions():
            installed.append({"version": m.version, "base": m.base, "task": m.task, "scores": m.scores,
                              "created": m.created, "current": m.version == current})
        folder = serving.get("folder")
        return {
            "current": current,
            "previous": store.previous(),
            "serving": folder,
            # None when the server was not asked or does not answer
            "in_sync": (folder == current) if folder else None,
            "versions": installed,
        }

    return router
