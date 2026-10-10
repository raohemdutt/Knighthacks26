import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

from contexts.perception.domain.zones import ZoneEngine, load_zones
from contexts.perception.zones_api import area, zones_router
from contracts.perception import ObservationRecorded
from contracts.types import Box, Track

REPO = Path(__file__).resolve().parents[3]
RECT = [[0.1, 0.2], [0.6, 0.2], [0.6, 0.9], [0.1, 0.9]]


def setup(tmp_path, token=None):
    shutil.copy(REPO / "config" / "zones.yaml", tmp_path / "zones.yaml")
    shutil.copy(REPO / "config" / "zones.demo.yaml", tmp_path / "zones.demo.yaml")
    engine = ZoneEngine.load(tmp_path / "zones.yaml")

    def guard(authorization: str | None = Header(default=None)):
        if authorization != f"Bearer {token}":
            raise HTTPException(401, "missing or wrong token")

    app = FastAPI()
    app.include_router(zones_router(tmp_path, lambda: engine, guard=guard if token else None), prefix="/api")
    return TestClient(app), engine, tmp_path / "zones.yaml"


def observe(engine, camera, x, y, ts):
    box = Box(x1=x - 0.02, y1=y - 0.1, x2=x + 0.02, y2=y)  # anchored at its bottom centre (x, y)
    return engine.on_observation(ObservationRecorded(camera_id=camera, ts=ts, frame_ref=f"frames:{camera}:1",
                                                      tracks=[Track(track_id=1, label="person", confidence=0.9, box=box)]))


def test_get_reads_the_file_fresh(tmp_path):
    c, _, path = setup(tmp_path)
    assert set(c.get("/api/zones").json()) == {"cam-door", "cam-desk"}
    path.write_text("cam-desk:\n  desk: [[0, 0], [1, 0], [1, 1], [0, 1]]\n")  # edited by hand on the box
    assert c.get("/api/zones").json() == {"cam-desk": {"desk": [[0, 0], [1, 0], [1, 1], [0, 1]]}}


def test_put_rewrites_the_file_and_the_running_engine_without_a_restart(tmp_path):
    c, engine, path = setup(tmp_path)
    t0 = datetime(2026, 10, 10, 20, 0, tzinfo=UTC)
    assert observe(engine, "cam-desk", 0.05, 0.5, t0) == []  # left of the old desk zone (x from 0.10)
    r = c.put("/api/zones/cam-desk", json={"zones": {"desk": [[0.0, 0.3], [0.5, 0.3], [0.5, 1.0], [0.0, 1.0]]}})
    assert r.status_code == 200
    assert [e.kind for e in observe(engine, "cam-desk", 0.25, 0.5, t0)] == ["enter"]  # live now
    on_disk = load_zones(path)
    assert on_disk["cam-desk"]["desk"] == [(0.0, 0.3), (0.5, 0.3), (0.5, 1.0), (0.0, 1.0)]
    assert on_disk["cam-door"] == load_zones(REPO / "config" / "zones.yaml")["cam-door"]  # other cameras untouched
    text = path.read_text()
    assert text.startswith("# Zones per camera") and "  desk: [[0, 0.3], [0.5, 0.3]" in text  # header and one-line style kept
    assert engine.zones["cam-desk"]["desk"] == on_disk["cam-desk"]["desk"]


def test_a_stay_in_the_old_polygon_is_dropped_for_that_camera_only(tmp_path):
    c, engine, _ = setup(tmp_path)
    t0 = datetime(2026, 10, 10, 20, 0, tzinfo=UTC)
    observe(engine, "cam-desk", 0.5, 0.6, t0)
    observe(engine, "cam-door", 0.7, 0.8, t0)
    assert {k[0] for k in engine.stays} == {"cam-desk", "cam-door"}
    c.put("/api/zones/cam-desk", json={"zones": {"desk": RECT}})
    assert {k[0] for k in engine.stays} == {"cam-door"}


@pytest.mark.parametrize("camera,zones,status", [
    ("cam-nowhere", {"desk": RECT}, 404),
    ("cam-desk", {"kitchen": RECT}, 404),  # a new id is not known to rules until a restart
    ("cam-desk", {"desk": [[0.1, 0.1], [0.2, 0.2]]}, 422),  # two points
    ("cam-desk", {"desk": [[0.1, 0.1], [1.5, 0.1], [1.5, 0.9]]}, 422),  # outside the picture
    ("cam-desk", {"desk": [[0.1, 0.1], [0.11, 0.1], [0.11, 0.11], [0.1, 0.11]]}, 422),  # a stray click
])
def test_bad_requests_change_nothing(tmp_path, camera, zones, status):
    c, engine, path = setup(tmp_path)
    before = path.read_text()
    assert c.put(f"/api/zones/{camera}", json={"zones": zones}).status_code == status
    assert path.read_text() == before


def test_the_buttons_need_the_token_but_reading_does_not(tmp_path):
    c, *_ = setup(tmp_path, token="secret")
    assert c.get("/api/zones").status_code == 200
    assert c.put("/api/zones/cam-desk", json={"zones": {"desk": RECT}}).status_code == 401
    assert c.post("/api/zones/demo").status_code == 401
    assert c.put("/api/zones/cam-desk", json={"zones": {"desk": RECT}}, headers={"Authorization": "Bearer secret"}).status_code == 200


def test_the_demo_zones_are_big_and_apply_to_the_same_cameras_and_ids(tmp_path):
    c, engine, path = setup(tmp_path)
    r = c.post("/api/zones/demo")
    assert r.status_code == 200
    zones = load_zones(path)
    assert area(zones["cam-desk"]["desk"]) > 0.6  # most of the frame
    assert set(zones["cam-door"]) == {"door", "recycling", "trash"}
    assert area(zones["cam-door"]["door"]) > 0.3
    assert engine.zones == zones
    # the bins stay apart, so Waste Watch can still tell them apart
    rec, trash = zones["cam-door"]["recycling"], zones["cam-door"]["trash"]
    assert max(x for x, _ in rec) < min(x for x, _ in trash)


def test_the_shipped_demo_zones_only_use_ids_that_zones_yaml_has():
    real, demo = load_zones(REPO / "config" / "zones.yaml"), load_zones(REPO / "config" / "zones.demo.yaml")
    for camera, named in demo.items():
        assert camera in real and set(named) <= set(real[camera])


def test_no_demo_file_is_a_clear_404(tmp_path):
    c, _, _ = setup(tmp_path)
    (tmp_path / "zones.demo.yaml").unlink()
    assert c.post("/api/zones/demo").status_code == 404
