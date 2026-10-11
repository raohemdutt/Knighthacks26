"""The lab's recorded rehearsal is a second, named source for the Together tab. It is labelled, it never leaks into the
routes that serve only what the box recorded, and it is the committed record's own numbers."""

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from contexts.learning.application.lab_rehearsal import LABEL, REHEARSAL_DIR, rehearsal_compare
from contexts.learning.application.lab_view import LabView
from contexts.learning.lab_api import lab_router
from contexts.learning.tests.rehearsal import FIXTURE_DUO, FIXTURE_SOLO


def client(tmp_path) -> TestClient:
    app = FastAPI()
    app.include_router(lab_router(LabView(), loop_dirs=[tmp_path / "lab_loops"]))
    return TestClient(app)


def test_the_rehearsal_is_served_under_its_own_name_with_its_label_and_the_records_own_numbers(tmp_path):
    body = client(tmp_path).get("/api/learning/lab/rehearsal").json()
    assert body["record"]["kind"] == "rehearsal" and body["record"]["label"] == LABEL
    assert body["record"]["file"].startswith("fixtures/learning/")
    doc = json.loads((REHEARSAL_DIR / "comparison.json").read_text(encoding="utf-8"))
    assert (body["a"]["run_id"], body["b"]["run_id"]) == (FIXTURE_SOLO, FIXTURE_DUO)
    for side, act in (("a", "solo"), ("b", "duo")):  # copied, not recomputed
        assert body[side]["examples"] == doc["acts"][act]["examples"]
        assert body[side]["j_per_example"] == doc["acts"][act]["j_per_example"]
        assert body[side]["traces"], "the power traces come with it"
    assert body["ratios"]["j_per_example"] == round(doc["acts"]["duo"]["j_per_example"] / doc["acts"]["solo"]["j_per_example"], 3)
    assert body["runs"] == []  # one record, nothing to pick between


def test_the_routes_that_serve_only_what_the_box_recorded_still_serve_none_of_it(tmp_path):
    c = client(tmp_path)
    assert c.get("/api/learning/lab/rehearsal").status_code == 200
    assert c.get("/api/learning/lab/compare").status_code == 404
    assert c.get(f"/api/learning/lab/compare?a={FIXTURE_SOLO}&b={FIXTURE_DUO}").status_code == 404
    assert c.get(f"/api/learning/lab/{FIXTURE_SOLO}").status_code == 404
    lab = c.get("/api/learning/lab").json()
    assert lab["runs"] == [] and lab["real_runs"] == 0


def test_a_checkout_without_the_record_has_no_rehearsal(tmp_path):
    assert rehearsal_compare(tmp_path / "nothing") is None
    assert rehearsal_compare(None) is None
