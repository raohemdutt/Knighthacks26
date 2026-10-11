"""Where a thing is inside its frame: the box is kept when the model or the detector gave one, and never invented."""

import sqlite3

from fastapi import FastAPI
from fastapi.testclient import TestClient

from contexts.memory.api import build_router
from contexts.memory.application.reader import SqliteMemory
from contexts.memory.infrastructure.store import MemoryStore
from contexts.memory.tests.test_projection import BOX, at, ref, run, seen
from contracts.memory import Sighting
from contracts.perception import SceneDescribed
from contracts.types import Box, ObjectHit

KEYS = Box(x1=0.40, y1=0.55, x2=0.48, y2=0.62)
LAPTOP = Box(x1=0.55, y1=0.30, x2=0.90, y2=0.70)


def scene_with_boxes(s: float) -> SceneDescribed:
    return SceneDescribed(ts=at(s), camera_id="cam-desk", caption="A desk.", zone_id="desk", frame_ref=ref("cam-desk", s), model_version="fake-0",
                          objects=[ObjectHit(label="keys", position="left of the laptop", box=KEYS), ObjectHit(label="laptop", position="on the desk", box=LAPTOP),
                                   ObjectHit(label="wallet", position="behind the mug")])


def test_a_box_from_the_object_list_is_kept_with_the_position(tmp_path):
    store, _, _ = run(tmp_path, [scene_with_boxes(5)])
    memory = SqliteMemory(store)
    keys = memory.last_seen("keys")
    assert keys.position == "left of the laptop" and keys.box == KEYS
    wallet = memory.last_seen("wallet")
    assert wallet.position == "behind the mug" and wallet.box is None  # words only: no box is made up
    where = memory.where("wallet")
    assert where["box"] is None and where["zone"] == "desk" and where["frame_ref"] == ref("cam-desk", 5) and where["seen_at"] == at(5)
    assert {a["label"]: a["box"] for a in where["anchors"]} == {"keys": KEYS, "laptop": LAPTOP}  # the boxed things in the same frame
    assert [a["label"] for a in memory.where("keys")["anchors"]] == ["laptop"]
    assert memory.where("umbrella") is None


def test_the_detectors_latest_track_gives_a_box_for_its_own_frame_only(tmp_path):
    store, _, _ = run(tmp_path, [seen(0, "laptop"), seen(1, "laptop")])
    memory = SqliteMemory(store)
    laptop = memory.last_seen("laptop")
    assert laptop.frame_ref == ref("cam-desk", 1) and laptop.box == BOX
    assert store.box_of(ref("cam-desk", 0), "laptop") is None  # one detector box per label: the newest
    assert memory.last_seen("person") is None and store.boxes_in(ref("cam-desk", 1)) == [("laptop", BOX)]


def test_a_store_written_before_boxes_existed_still_loads(tmp_path):
    path = tmp_path / "old.sqlite"
    old = sqlite3.connect(path)
    old.executescript("""CREATE TABLE memory_sightings (id TEXT PRIMARY KEY, label TEXT NOT NULL, camera_id TEXT NOT NULL, zone_id TEXT,
                         ts TEXT NOT NULL, frame_ref TEXT NOT NULL, position TEXT);
                         INSERT INTO memory_sightings VALUES ('a', 'keys', 'cam-desk', 'desk', '2026-10-10T12:00:00.000000+00:00', 'frames:cam-desk:k', 'by the mug');""")
    old.commit()
    old.close()
    memory = SqliteMemory(MemoryStore(path))
    keys = memory.last_seen("keys")
    assert keys.position == "by the mug" and keys.box is None
    assert memory.where("keys")["anchors"] == []
    assert Sighting.model_validate({"label": "keys", "camera_id": "cam-desk", "ts": "2026-10-10T12:00:00Z", "frame_ref": "f"}).box is None  # old payloads too


def test_the_read_endpoint(tmp_path):
    store, _, _ = run(tmp_path, [scene_with_boxes(5)])
    app = FastAPI()
    app.include_router(build_router(SqliteMemory(store)))
    with TestClient(app) as client:
        got = client.get("/memory/last-seen/keys").json()
        assert got["box"] == {"x1": 0.4, "y1": 0.55, "x2": 0.48, "y2": 0.62} and got["position"] == "left of the laptop" and got["zone"] == "desk"
        assert got["anchors"] == [{"label": "laptop", "box": {"x1": 0.55, "y1": 0.3, "x2": 0.9, "y2": 0.7}}]
        assert client.get("/memory/last-seen/wallet").json()["box"] is None
        assert client.get("/memory/last-seen/umbrella").status_code == 404
        assert client.get("/memory/objects/keys/last-seen").json()["box"] == got["box"]  # the old endpoint carries it too, additively
