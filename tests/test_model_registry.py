import json
import os
from datetime import UTC, datetime

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from contracts.learning import ModelManifest
from runtime.api import runtime_router
from runtime.modelstore import ModelStore, ModelStoreError, sha256_file


def make_version(root, name, *, created=None, gguf=b"weights", mmproj=b"projector", checksum=True):
    d = root / name
    d.mkdir(parents=True)
    (d / "m.gguf").write_bytes(gguf)
    (d / "mm.gguf").write_bytes(mmproj)
    sha = {"m.gguf": sha256_file(d / "m.gguf"), "mm.gguf": sha256_file(d / "mm.gguf")} if checksum else {}
    m = ModelManifest(version=name, base="Qwen/Qwen3.5-4B", task="t", gguf="m.gguf", mmproj="mm.gguf", sha256=sha,
                      scores={"acc": 0.5}, created=created or datetime(2026, 10, 10, tzinfo=UTC))
    (d / "manifest.json").write_text(m.model_dump_json())
    return d


@pytest.fixture
def store(tmp_path):
    try:  # creating symlinks needs a privilege on Windows
        os.symlink("x", tmp_path / "probe")
    except (OSError, NotImplementedError):
        pytest.skip("this machine cannot create symlinks")
    (tmp_path / "probe").unlink()
    return ModelStore(tmp_path)


def test_activate_switches_atomically_and_remembers_the_previous_version(store, tmp_path):
    make_version(tmp_path, "v1")
    make_version(tmp_path, "v2", created=datetime(2026, 10, 11, tzinfo=UTC))
    assert store.current() is None
    store.activate("v1")
    assert store.current() == "v1" and store.previous() is None
    store.activate("v2")
    assert (store.current(), store.previous()) == ("v2", "v1")
    assert (tmp_path / "current" / "manifest.json").exists() and not (tmp_path / "current.tmp").exists()
    assert store.rollback() == "v1" and store.current() == "v1" and store.previous() == "v2"


def test_a_damaged_or_incomplete_package_is_never_switched_to(store, tmp_path):
    make_version(tmp_path, "good")
    store.activate("good")
    d = make_version(tmp_path, "bad")
    (d / "m.gguf").write_bytes(b"truncated")
    assert "m.gguf does not match its sha256" in store.verify("bad")[0]
    with pytest.raises(ModelStoreError, match="not safe"):
        store.activate("bad")
    assert store.current() == "good"  # untouched
    make_version(tmp_path, "nosum", checksum=False)
    assert "no checksum" in store.verify("nosum")[0]
    (tmp_path / "good" / "mm.gguf").unlink()
    assert store.verify("good") == ["mm.gguf is missing"]


def test_versions_lists_newest_first_and_skips_the_alias_and_junk(store, tmp_path):
    make_version(tmp_path, "old", created=datetime(2026, 10, 9, tzinfo=UTC))
    make_version(tmp_path, "new", created=datetime(2026, 10, 11, tzinfo=UTC))
    (tmp_path / "junk").mkdir()
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "manifest.json").write_text("{not json")
    store.activate("old")
    assert [m.version for m in store.versions()] == ["new", "old"]
    with pytest.raises(ModelStoreError):
        store.rollback()  # nothing earlier yet


def test_the_endpoint_reports_what_the_server_really_serves(store, tmp_path, monkeypatch):
    make_version(tmp_path, "v1")
    make_version(tmp_path, "v2", created=datetime(2026, 10, 11, tzinfo=UTC))
    store.activate("v1")
    store.activate("v2")
    app = FastAPI()
    app.include_router(runtime_router(store, llm_url="http://llm.test"))
    client = TestClient(app)

    def props(path):
        monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(200, json={"model_path": path}))

    props(str(tmp_path / "v2" / "m.gguf"))
    body = client.get("/runtime/models").json()
    assert (body["current"], body["previous"], body["serving"], body["in_sync"]) == ("v2", "v1", "v2", True)
    assert [(v["version"], v["current"]) for v in body["versions"]] == [("v2", True), ("v1", False)]
    props(str(tmp_path / "v1" / "m.gguf"))  # a switch that did not take
    assert client.get("/runtime/models").json()["in_sync"] is False

    def down(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", down)
    body = client.get("/runtime/models").json()
    assert body["serving"] is None and body["in_sync"] is None and body["current"] == "v2"
    json.dumps(body)
