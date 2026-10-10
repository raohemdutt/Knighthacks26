"""Text embeddings for semantic search (M5), behind one small object that never raises.

    embedder = embedder_from(runtime, "config/models.yaml")   # None unless models.yaml turns a text model on
    embedder.embed(["energy drink can"])                       # blocking: call it from a thread
    embedder.query("something to drink")                       # cached; waits at most query_timeout_s

`Runtime.embed_text` is a blocking call (HTTP to a second llama-server) and it
raises when no embedding model is configured or the server is down. Search must
keep working either way, so a failure only pauses embedding for `retry_s`: every
call in that time returns None at once and search stays keyword-only.

`SqliteMemory.search` is synchronous and the agent calls it on the event loop,
so `query` runs the model call on a worker thread and waits a bounded time for
it. A slow answer still lands in the cache for the next question.
"""

import asyncio
import concurrent.futures
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

import yaml

from runtime.fake import FakeRuntime
from runtime.interface import Runtime

log = logging.getLogger(__name__)

MIN_SIMILARITY = 0.5  # cosine; below this a text is not "about" the query. Model-specific: set it in models.yaml.
QUERY_CACHE = 256


class TextEmbedder:
    def __init__(self, runtime: Runtime, model: str, min_similarity: float = MIN_SIMILARITY, query_timeout_s: float = 2.0,
                 retry_s: float = 60.0, clock: Callable[[], float] = time.monotonic) -> None:
        self.runtime = runtime
        self.model = model  # stored next to every vector, so a change of model re-embeds
        self.min_similarity = min_similarity
        self.query_timeout_s = query_timeout_s
        self.retry_s = retry_s
        self.clock = clock  # monotonic, and never stored: it only spaces out retries
        self._paused_until = 0.0
        self._lock = threading.Lock()
        self._cache: dict[str, list[float]] = {}
        self._pending: dict[str, concurrent.futures.Future] = {}
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="memory-embed")

    @property
    def paused(self) -> bool:
        return self.clock() < self._paused_until

    def embed(self, texts: list[str]) -> list[list[float]] | None:
        """One vector per text, or None when the model is unavailable. Blocking."""
        if self.paused:
            return None
        try:
            vectors = [[float(x) for x in v] for v in self.runtime.embed_text(texts)]
            if len(vectors) != len(texts) or any(len(v) != len(vectors[0]) or not v for v in vectors):
                raise ValueError(f"asked for {len(texts)} vectors, got {len(vectors)} of uneven or zero size")
        except Exception as e:  # no model configured, server down, bad reply: all the same to search
            self._pause(f"{type(e).__name__}: {e}")
            return None
        return vectors

    def query(self, q: str) -> list[float] | None:
        """The query's vector, from the cache or within `query_timeout_s`; None means keyword search only."""
        key = q.strip()
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        if self.paused:
            return None
        try:
            return self._submit(key).result(timeout=self.query_timeout_s)
        except concurrent.futures.TimeoutError:  # not the builtin TimeoutError on Python 3.10
            self._pause(f"no answer in {self.query_timeout_s:g} s")
            return None

    async def aquery(self, q: str) -> list[float] | None:
        """`query` without holding the event loop, and without the time limit."""
        key = q.strip()
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        if self.paused:
            return None
        return await asyncio.wrap_future(self._submit(key))

    def _submit(self, key: str) -> concurrent.futures.Future:
        with self._lock:
            if key not in self._pending:  # two askers of the same question share one model call
                self._pending[key] = self._pool.submit(self._embed_query, key)
            return self._pending[key]

    def _embed_query(self, key: str) -> list[float] | None:
        vectors = self.embed([key])
        with self._lock:
            self._pending.pop(key, None)
            if vectors:
                self._cache[key] = vectors[0]
                while len(self._cache) > QUERY_CACHE:
                    del self._cache[next(iter(self._cache))]  # oldest first
        return vectors[0] if vectors else None

    def _pause(self, why: str) -> None:
        log.warning("text embeddings unavailable (%s); search is keyword-only for %g s", why, self.retry_s)
        self._paused_until = self.clock() + self.retry_s


def embedder_from(runtime: Runtime, models_yaml: str | Path = "config/models.yaml") -> TextEmbedder | None:
    """None unless models.yaml has `embeddings.text.backend` set to something other than `none`.

    Also None on FakeRuntime: its vectors are hashes of the text, so they would rank noise.
    `embeddings.text.min_similarity` (optional) overrides the similarity floor for that model."""
    path = Path(models_yaml)
    if not path.exists() or isinstance(runtime, FakeRuntime):
        return None
    cfg = ((yaml.safe_load(path.read_text()) or {}).get("embeddings") or {}).get("text") or {}
    if cfg.get("backend") in (None, "none"):
        return None
    return TextEmbedder(runtime, model=str(cfg.get("model") or cfg.get("url") or cfg["backend"]),
                        min_similarity=float(cfg.get("min_similarity", MIN_SIMILARITY)))
