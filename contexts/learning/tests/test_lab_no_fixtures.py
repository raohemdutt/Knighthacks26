"""Ilan's rule: nothing from fixtures/ is ever served to a screen. A box whose only records are fixtures answers the
Compare tab with a 404 and the run list with no runs, however the fixture got there."""

import json
import shutil
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from bus import EventBus
from contexts.learning.application import lab_loops
from contexts.learning.application.lab_compare import default_dirs
from contexts.learning.application.lab_sample import SAMPLE_DIR
from contexts.learning.application.lab_view import LabView
from contexts.learning.lab_api import lab_router
from contexts.learning.lab_import import lab_import_router
from contexts.learning.tests import rehearsal
from contexts.learning.tests.rehearsal import FIXTURE_DUO, FIXTURE_SOLO, REHEARSAL

FIXTURES_LEARNING = Path(lab_loops.FIXTURES) / "learning"


def box(tmp_path, loop_dirs):
    view, bus = LabView(), EventBus(log_dir=tmp_path / "events")
    view.subscribe(bus)
    app = FastAPI()
    app.include_router(lab_import_router(view, bus.publish, data_dir=tmp_path / "data"))
    app.include_router(lab_router(view, sample_dir=SAMPLE_DIR, loop_dirs=loop_dirs))   # the sample is on disk
    return TestClient(app)


def assert_nothing_served(client):
    compare = client.get("/api/learning/lab/compare")
    assert compare.status_code == 404, compare.text
    lab = client.get("/api/learning/lab").json()
    assert lab["runs"] == [] and lab["real_runs"] == 0 and lab["sample"] is False
    for run_id in (FIXTURE_SOLO, FIXTURE_DUO):
        assert client.get(f"/api/learning/lab/{run_id}").status_code == 404
        assert client.get(f"/api/learning/lab/compare?a={run_id}").status_code == 404
    assert client.get(f"/api/learning/lab/compare?a={FIXTURE_SOLO}&b={FIXTURE_DUO}").status_code == 404


def test_the_rehearsal_run_ids_are_known_as_fixtures():
    ids = lab_loops.fixture_run_ids()
    assert {FIXTURE_SOLO, FIXTURE_DUO} <= ids and "live-1011-035827-solo" in ids   # the first rehearsal too
    assert not lab_loops.is_fixture_run(rehearsal.SOLO)


def test_a_store_pointed_at_fixtures_serves_nothing(tmp_path):
    """Main's old default read fixtures/learning straight: a store that is only that is a 404 and no runs."""
    assert_nothing_served(box(tmp_path, [FIXTURES_LEARNING]))


def test_a_fixture_copied_onto_the_box_serves_nothing(tmp_path):
    shutil.copytree(REHEARSAL, tmp_path / "data" / "lab_loops" / "1011-040342")
    assert_nothing_served(box(tmp_path, default_dirs(tmp_path / "data")))


def test_a_fixture_posted_through_the_import_route_serves_nothing(tmp_path):
    """What --fallback would post (the rehearsal's steps, its stage lines, its comparison and windows) stays off screen."""
    client = box(tmp_path, default_dirs(tmp_path / "data"))
    doc = (REHEARSAL / "comparison.json").read_bytes()
    assert client.post("/api/learning/lab/import?kind=compare", content=doc).status_code == 200
    for act, run_id in (("solo", FIXTURE_SOLO), ("duo", FIXTURE_DUO)):
        steps = (REHEARSAL / act / "telemetry.jsonl").read_text(encoding="utf-8")
        assert client.post(f"/api/learning/lab/import?run_id={run_id}", content=steps).status_code == 200
        lines = [x for x in (REHEARSAL / "stages.jsonl").read_text(encoding="utf-8").splitlines() if f'"act": "{act}"' in x]
        assert client.post(f"/api/learning/lab/import?kind=stage&run_id={run_id}&task=desk-objects",
                           content="\n".join(lines)).status_code == 200
    assert_nothing_served(client)


def test_a_comparison_marked_as_a_fixture_is_not_served(tmp_path):
    client = box(tmp_path, default_dirs(tmp_path / "data"))
    doc = {**json.loads(rehearsal.text("comparison.json")), "fixture": True}
    assert client.post("/api/learning/lab/import?kind=compare", json=doc).status_code == 200
    assert client.get("/api/learning/lab/compare").status_code == 404
    unmarked = {k: v for k, v in doc.items() if k != "fixture"}
    assert client.post("/api/learning/lab/import?kind=compare", json=unmarked).status_code == 200
    assert client.get("/api/learning/lab/compare").json()["asked"] == {"a": rehearsal.SOLO, "b": rehearsal.DUO}
