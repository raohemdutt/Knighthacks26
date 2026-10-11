"""GET /api/learning/lab/{run_id} while the live loop runs (L30's back-end half): run_status running with the steps so
far, and the stage strip (labeling, training, packaging, offered) lit by the stage lines the relay posts."""

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from bus import EventBus
from contexts.learning.application.lab_view import LabView
from contexts.learning.lab_api import lab_router
from contexts.learning.lab_import import lab_import_router
from contexts.learning.tests import rehearsal

SOLO = "live-1011-990001-solo"
TASK = "desk-objects"


def box(tmp_path):
    view, bus = LabView(), EventBus(log_dir=tmp_path / "events")
    view.subscribe(bus)
    app = FastAPI()
    app.include_router(lab_import_router(view, bus.publish, data_dir=tmp_path / "data"))
    app.include_router(lab_router(view, sample_dir=None, metrics_file=None))
    return TestClient(app)


def solo_lines():
    rows = [json.loads(x) for x in rehearsal.text("stages.jsonl").splitlines() if x.strip()]
    return {r["stage"]: r for r in rows if r.get("act") == "solo"}


def steps():
    return rehearsal.text("solo/telemetry.jsonl").splitlines()


def post_stage(client, line):
    r = client.post(f"/api/learning/lab/import?kind=stage&run_id={SOLO}&task={TASK}", content=json.dumps(line))
    assert r.status_code == 200, r.text


def strip(story):
    return {s["stage"]: s["state"] for s in story["stage_strip"]}


def test_the_story_redraws_live_from_labeling_to_offered(tmp_path):
    client = box(tmp_path)
    lines = solo_lines()
    post_stage(client, lines["start"])
    story = client.get(f"/api/learning/lab/{SOLO}").json()
    assert story["run_status"] == "running" and story["steps_so_far"] == 0
    assert strip(story) == {"labeling": "now", "training": "waiting", "packaging": "waiting", "offered": "waiting"}
    assert story["stage_strip"][0]["started"].startswith("2026-10-11T08:03:48")

    post_stage(client, lines["labeled"])
    post_stage(client, lines["train"])
    client.post(f"/api/learning/lab/import?run_id={SOLO}", content="\n".join(steps()[:3]) + "\n")
    story = client.get(f"/api/learning/lab/{SOLO}").json()
    assert story["run_status"] == "running" and story["steps_so_far"] == 3 and story["status"] == "training"
    assert strip(story) == {"labeling": "done", "training": "now", "packaging": "waiting", "offered": "waiting"}
    assert story["stage_strip"][0]["seconds"] == 32.055
    assert len(story["training"]["points"]) == 3   # the loss curve so far

    client.post(f"/api/learning/lab/import?run_id={SOLO}", content="\n".join(steps()) + "\n")
    post_stage(client, lines["trained"])
    story = client.get(f"/api/learning/lab/{SOLO}").json()
    assert story["steps_so_far"] == len(steps()) and story["run_status"] == "running"
    assert strip(story) == {"labeling": "done", "training": "done", "packaging": "now", "offered": "waiting"}
    assert round(story["stage_strip"][1]["seconds"], 1) == 53.9

    post_stage(client, lines["done"])
    story = client.get(f"/api/learning/lab/{SOLO}").json()
    assert story["run_status"] == "finished"
    assert strip(story) == {"labeling": "done", "training": "done", "packaging": "done", "offered": "done"}
    assert round(story["stage_strip"][2]["seconds"], 1) == 71.1
    listed = client.get("/api/learning/lab").json()["runs"]
    assert listed[0]["run_id"] == SOLO and listed[0]["run_status"] == "finished"


def test_a_failed_act_lights_the_stage_it_stopped_in(tmp_path):
    client = box(tmp_path)
    lines = solo_lines()
    for name in ("start", "labeled", "train"):
        post_stage(client, lines[name])
    post_stage(client, {"act": "solo", "stage": "failed", "ts": "2026-10-11T08:04:40Z", "error": "trainer exited 3"})
    story = client.get(f"/api/learning/lab/{SOLO}").json()
    assert story["run_status"] == "failed"
    assert strip(story) == {"labeling": "done", "training": "failed", "packaging": "waiting", "offered": "waiting"}
    assert story["stage_strip"][1]["message"] == "trainer exited 3"


def test_a_run_from_files_without_stage_lines_still_lights_training_from_its_steps(tmp_path):
    client = box(tmp_path)
    client.post(f"/api/learning/lab/import?run_id={SOLO}", content="\n".join(steps()) + "\n")
    story = client.get(f"/api/learning/lab/{SOLO}").json()
    assert story["run_status"] == "running" and story["steps_so_far"] == len(steps())
    assert strip(story) == {"labeling": "not reported", "training": "now", "packaging": "waiting", "offered": "waiting"}
