"""Hybrid search (M5): vectors next to the FTS index, made in the background, fused with the keyword ranking."""

import asyncio
import threading
from datetime import timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from bus import EventBus
from contexts.memory.__main__ import embed, main, rebuild
from contexts.memory.api import build_router
from contexts.memory.application.embedder import TextEmbedder, embedder_from
from contexts.memory.application.indexer import VectorIndexer
from contexts.memory.application.projection import MemoryProjection
from contexts.memory.application.reader import SqliteMemory, fuse
from contexts.memory.infrastructure.store import MemoryStore, pack_vector, unpack_vector
from contexts.memory.tests.test_projection import at, script
from contracts.memory import Episode, Item
from runtime.fake import FakeRuntime

# A hand-made embedding space: one axis per idea, a text's vector counts its words on each axis.
AXES = [
    {"drink", "drinks", "beverage", "soda", "thirsty", "sip", "coffee"},
    {"dog", "cat", "pet", "animal"},
    {"keys", "key", "keyring", "unlock"},
    {"assignment", "milestone", "homework", "coursework", "due"},
    {"pizza", "food", "lunch", "eat", "hungry"},
]

DRINK = "An energy drink can sits by the laptop."
SODA = "Two soda cans and a coffee cup are on the table."
KEYS = "A person puts a set of keys next to the mug."
EMPTY = "Nobody is at the table."


class Axes(FakeRuntime):
    """Deterministic and tiny: words that mean the same thing land on the same axis."""

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(("embed_text", texts))
        return [[float(sum(w.strip(".,?!:()") in axis for w in t.lower().split())) for axis in AXES] for t in texts]


class NoModel(FakeRuntime):
    def embed_text(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(("embed_text", texts))
        raise RuntimeError("no text embedding model configured (models.yaml: embeddings.text)")


def embed_calls(runtime: FakeRuntime) -> list:
    return [args for name, args in runtime.calls if name == "embed_text"]


def filled(tmp_path, name: str = "memory.sqlite") -> MemoryStore:
    """Three desk episodes (one has two drink captions and a summary) and two chat messages."""
    store = MemoryStore(tmp_path / name)
    for n, captions in enumerate(([DRINK, SODA], [KEYS], [EMPTY])):
        ep = Episode(id=f"ep{n}", camera_id="cam-desk", start=at(n * 3600), end=at(n * 3600 + 120), captions=captions,
                     keyframe_refs=[f"frames:cam-desk:{n}"])
        store.put_episode(ep, ep.end)
        for caption in captions:
            store.index("caption", ep.id, caption)
    store.set_summary("ep0", "Someone left a drink and a soda at the desk.")
    store.put_item(Item(connector="groupme", kind="message", external_id="m1", title="Sam: who wants pizza", when=at(0)), synced_at=at(0))
    store.put_item(Item(connector="canvas", kind="assignment", external_id="a1", title="Milestone1 [COP4533]", when=at(0),
                        summary="Due Tuesday night."), synced_at=at(0))
    store.db.commit()
    return store


def indexed(tmp_path, runtime: FakeRuntime | None = None, model: str = "axes-1", **kw) -> tuple[MemoryStore, TextEmbedder]:
    store = filled(tmp_path)
    embedder = TextEmbedder(runtime or Axes(), model, **kw)
    assert asyncio.run(VectorIndexer(store, embedder, batch=2).run()) == 7  # every distinct text, four batches
    return store, embedder


def results(memory: SqliteMemory, q: str, limit: int = 5) -> list[tuple]:
    return [(h.episode_id, h.text, h.kind, h.frame_ref) for h in memory.search(q, limit)]


def test_vectors_are_unit_length_float32():
    blob = pack_vector([3.0, 0.0, 4.0])
    assert len(blob) == 12 and unpack_vector(blob) == pytest.approx((0.6, 0.0, 0.8))
    assert unpack_vector(pack_vector([0.0, 0.0])) == (0.0, 0.0)  # nothing to scale


def test_a_result_found_only_by_meaning_comes_back(tmp_path):
    store, embedder = indexed(tmp_path)
    assert SqliteMemory(store).search("thirsty for a beverage") == []  # no word in common with any caption

    hits = SqliteMemory(store, embedder).search("thirsty for a beverage")
    assert [h.episode_id for h in hits] == ["ep0"]
    assert (hits[0].text, hits[0].camera_id, hits[0].ts, hits[0].frame_ref) == (DRINK, "cam-desk", at(0), "frames:cam-desk:0")
    # an item is found by meaning as well, and comes back as an item
    [canvas] = SqliteMemory(store, embedder).search("homework coursework")
    assert (canvas.episode_id, canvas.kind, canvas.camera_id) == ("item:canvas:a1", "assignment", "canvas")


def test_keyword_and_meaning_are_fused(tmp_path):
    store, embedder = indexed(tmp_path)
    memory = SqliteMemory(store, embedder)

    # "mug" is only in the keys caption (keyword), "thirsty" only means the drinks (vector): both come back
    assert {h.episode_id for h in memory.search("thirsty mug")} == {"ep0", "ep1"}
    # a result in both rankings beats a result in one: ep0 says "drink" and means it, the pizza chat only says "wants"
    hits = memory.search("wants a drink", limit=10)
    assert [h.episode_id for h in hits] == ["ep0", "item:groupme:m1"]
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)

    a, b, c = ({"ref": r, "source": "caption", "text": r} for r in "abc")
    assert [r["ref"] for r in fuse([a, b], [b, c])] == ["b", "a", "c"]  # in both, then keyword before vector on a tie


def test_without_an_embedder_search_is_the_keyword_search(tmp_path):
    store = filled(tmp_path)
    questions = ["energy drink", "keys mug", "milestone due", "pizza", "table", "thirsty for a beverage", "is it?"]
    before = [SqliteMemory(store).search(q, limit=10) for q in questions]
    assert before[0] and before[5] == []

    # An embedder with nothing indexed yet: the same answers, and the model is not asked.
    runtime = Axes()
    waiting = SqliteMemory(store, TextEmbedder(runtime, "axes-1"))
    assert [waiting.search(q, limit=10) for q in questions] == before and runtime.calls == []

    # Vectors in the file, no embedder to read them with: still the same answers.
    assert asyncio.run(VectorIndexer(store, TextEmbedder(Axes(), "axes-1")).run()) == 7
    assert [SqliteMemory(store).search(q, limit=10) for q in questions] == before


def test_another_models_vectors_are_ignored_and_remade(tmp_path):
    store, old = indexed(tmp_path, model="axes-1")
    runtime = Axes()
    new = TextEmbedder(runtime, "axes-2")
    memory = SqliteMemory(store, new)

    assert memory.search("thirsty for a beverage") == [] and runtime.calls == []  # nothing from axes-2 to compare with
    assert SqliteMemory(store, old).search("thirsty for a beverage") != []

    assert asyncio.run(VectorIndexer(store, new).run()) == 7
    assert (store.vector_count("axes-2"), store.vector_count("axes-1"), store.vector_count()) == (7, 0, 7)
    assert [h.episode_id for h in memory.search("thirsty for a beverage")] == ["ep0"]

    # Same name, different size (the model behind the name was swapped): those rows are skipped, not compared.
    with store.db:
        store.put_vectors("axes-2", {DRINK: [1.0, 0.0], SODA: [1.0, 0.0]})
    assert memory.search("thirsty for a beverage")[0].text != DRINK


def test_an_unrelated_question_finds_nothing(tmp_path):
    store, embedder = indexed(tmp_path)
    memory = SqliteMemory(store, embedder)

    assert memory.search("Did a dog walk in?") == []  # a real direction in the space, and no text near it
    assert memory.search("zxqv flurble") == []  # no direction at all
    # "pet food" is half about food: the pizza chat is at cosine 0.71, over the default floor and under a stricter one
    assert [h.episode_id for h in memory.search("pet food")] == ["item:groupme:m1"]
    strict = SqliteMemory(store, TextEmbedder(Axes(), "axes-1", min_similarity=0.9))
    assert strict.search("pet food") == [] and [h.episode_id for h in strict.search("thirsty")] == ["ep0"]


def test_an_episode_or_item_still_comes_back_once(tmp_path):
    store, embedder = indexed(tmp_path)
    memory = SqliteMemory(store, embedder)

    hits = memory.search("beverage", limit=20)  # ep0 has three texts about drinks
    assert [h.episode_id for h in hits] == ["ep0"] and hits[0].text == DRINK  # its closest text; a tie keeps the first indexed
    hits = memory.search("drink soda pizza keys table", limit=20)  # everything, by keyword and by meaning
    ids = [h.episode_id for h in hits]
    assert sorted(ids) == ["ep0", "ep1", "ep2", "item:groupme:m1"] and len(set(ids)) == len(ids)
    assert [h.episode_id for h in memory.search("drink soda pizza keys table", limit=2)] == ids[:2]  # the cut is still best first


def live(tmp_path, events, embedder, db: str = "memory.sqlite") -> MemoryStore:
    """The app's wiring: projection on replay, indexer live only. Waits for the background model calls."""
    store = MemoryStore(tmp_path / db)
    bus = EventBus(tmp_path / "events")
    MemoryProjection(store).subscribe(bus)
    VectorIndexer(store, embedder).subscribe(bus)

    async def main() -> None:
        await bus.publish_all(events)
        while pending := asyncio.all_tasks() - {asyncio.current_task()}:
            await asyncio.gather(*pending)

    asyncio.run(main())
    return store


def test_live_events_are_embedded_in_the_background_and_replay_embeds_nothing(tmp_path):
    runtime = Axes()
    embedder = TextEmbedder(runtime, "axes-1")
    store = live(tmp_path, script(), embedder)
    texts = {r["text"] for r in store.db.execute("SELECT text FROM memory_fts")}

    assert len(texts) == 3 and store.vector_count("axes-1") == 3  # two captions and the Canvas item
    assert {t for batch in embed_calls(runtime) for t in batch} == texts
    assert sum(len(batch) for batch in embed_calls(runtime)) == 3  # each text once
    assert SqliteMemory(store, embedder).search("unlock the keyring")[0].camera_id == "cam-desk"  # meaning only
    calls = len(runtime.calls)

    # A restart: the log replays into the same file, and into an empty one. No model call either way.
    for target in (store, MemoryStore(tmp_path / "fresh.sqlite")):
        bus = EventBus(tmp_path / "events")
        MemoryProjection(target).subscribe(bus)
        VectorIndexer(target, embedder).subscribe(bus)
        assert asyncio.run(bus.replay_into()) > len(script())
        assert len(runtime.calls) == calls
    assert target.vector_count() == 0 and store.vector_count() == 3

    # Vectors are not in the log, so a rebuild has none; `embed` makes them again, and only the missing ones.
    rebuild(tmp_path / "events", tmp_path / "rebuilt.sqlite")
    assert embed(tmp_path / "rebuilt.sqlite", embedder) == (3, 0)
    assert embed(tmp_path / "rebuilt.sqlite", embedder) == (0, 0)
    assert embed(tmp_path / "rebuilt.sqlite", embedder, everything=True) == (3, 0)
    rebuilt = MemoryStore(tmp_path / "rebuilt.sqlite")
    assert rebuilt.dump() == target.dump()  # the projected tables are untouched
    assert SqliteMemory(rebuilt, embedder).search("unlock the keyring")[0].camera_id == "cam-desk"


def test_no_embedder_wires_nothing(tmp_path):
    store = live(tmp_path, script(), None)
    assert store.vector_count() == 0 and asyncio.run(VectorIndexer(store, None).run()) == 0
    assert SqliteMemory(store).search("keys")[0].camera_id == "cam-desk"


def test_a_runtime_without_an_embedding_model_leaves_search_keyword_only(tmp_path):
    runtime, now = NoModel(), [0.0]
    embedder = TextEmbedder(runtime, "none", retry_s=60, clock=lambda: now[0])
    store = live(tmp_path, script(), embedder)  # three nudges, and the bus is fine

    assert store.vector_count() == 0
    assert len(embed_calls(runtime)) == 1  # asked once, then left alone for retry_s
    assert SqliteMemory(store, embedder).search("keys")[0].camera_id == "cam-desk"
    assert asyncio.run(VectorIndexer(store, embedder).run()) == 0 and len(embed_calls(runtime)) == 1

    now[0] = 61.0  # the pause is over: one more try, and the same quiet result
    assert asyncio.run(VectorIndexer(store, embedder).run()) == 0 and len(embed_calls(runtime)) == 2


def test_a_model_that_fails_at_query_time_falls_back_to_keywords(tmp_path):
    store, _ = indexed(tmp_path)
    now = [0.0]
    down = TextEmbedder(NoModel(), "axes-1", clock=lambda: now[0])
    memory = SqliteMemory(store, down)
    keyword = SqliteMemory(store)

    assert memory.search("energy drink") == keyword.search("energy drink") != []
    assert memory.search("thirsty for a beverage") == []
    assert len(embed_calls(down.runtime)) == 1 and down.paused  # the second question did not wait on the model

    class Short(Axes):  # a reply that does not line up with what was asked is a failure too
        def embed_text(self, texts):
            return super().embed_text(texts)[:-1]

    assert TextEmbedder(Short(), "axes-1").embed(["a", "b"]) is None


def test_a_slow_model_does_not_hold_the_sync_search(tmp_path):
    store, _ = indexed(tmp_path)
    release, now = threading.Event(), [0.0]

    class Slow(Axes):
        def embed_text(self, texts):
            release.wait(5)
            return super().embed_text(texts)

    slow = TextEmbedder(Slow(), "axes-1", query_timeout_s=0.05, clock=lambda: now[0])
    memory = SqliteMemory(store, slow)

    assert memory.search("thirsty for a beverage") == []  # gave up after 50 ms: keyword only this time
    release.set()
    slow._pool.shutdown(wait=True)  # the late answer still lands in the cache
    assert [h.episode_id for h in memory.search("thirsty for a beverage")] == ["ep0"]


def test_vectors_follow_the_index(tmp_path):
    store, embedder = indexed(tmp_path)
    indexer = VectorIndexer(store, embedder)
    with store.db:  # the Canvas item is renamed: its old text leaves the index
        store.put_item(Item(connector="canvas", kind="assignment", external_id="a1", title="Homework 3 [COP4533]", when=at(0)),
                       synced_at=at(0) + timedelta(hours=1))

    assert asyncio.run(indexer.run()) == 1 and store.vector_count() == 7  # one new text in, the orphan out
    stored = {r["text"] for r in store.db.execute("SELECT text FROM memory_vectors")}
    assert "Homework 3 [COP4533]" in stored and not any("Milestone1" in t for t in stored)


def test_the_search_endpoint_is_hybrid(tmp_path):
    store, embedder = indexed(tmp_path)
    app = FastAPI()
    app.include_router(build_router(SqliteMemory(store, embedder)))
    c = TestClient(app)

    assert [h["episode_id"] for h in c.get("/memory/search", params={"q": "thirsty for a beverage"}).json()] == ["ep0"]
    assert c.get("/memory/search", params={"q": "Did a dog walk in?"}).json() == []
    assert c.get("/memory/search", params={"q": "keys mug", "limit": 1}).json()[0]["episode_id"] == "ep1"

    plain = FastAPI()
    plain.include_router(build_router(SqliteMemory(store)))
    assert TestClient(plain).get("/memory/search", params={"q": "thirsty for a beverage"}).json() == []


def test_models_yaml_turns_it_on(tmp_path, capsys):
    from runtime.llama import LlamaServerRuntime
    from runtime.registry import model_version

    off, on = tmp_path / "off.yaml", tmp_path / "models.yaml"
    off.write_text("embeddings:\n  text:\n    backend: none\n    model: bge-m3\n")
    on.write_text("embeddings:\n  text:\n    backend: openai-http\n    url: http://127.0.0.1:8081\n    model: bge-m3\n    min_similarity: 0.6\n")
    real = LlamaServerRuntime(model_version({}))

    assert embedder_from(real, off) is None and embedder_from(real, tmp_path / "missing.yaml") is None
    assert embedder_from(FakeRuntime(), on) is None  # the fake's vectors are hashes: they would rank noise
    embedder = embedder_from(real, on)
    assert (embedder.model, embedder.min_similarity) == ("bge-m3", 0.6)

    # The command says so and fails when there is no model to embed with (the default models.yaml is the fake runtime).
    filled(tmp_path, "cli.sqlite").close()
    assert main(["embed", "--db", str(tmp_path / "cli.sqlite"), "--config-dir", str(tmp_path)]) == 1
    assert "no text embedding model" in capsys.readouterr().out
