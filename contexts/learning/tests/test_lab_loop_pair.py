"""Act 2 on two pods as the box shows it (L29): "2 members", both pods on the stage strip, pod C's rank inside act 2's
story (never a run of its own in the list), and the per-pod training rows on the Compare tab."""

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from bus import EventBus
from contexts.learning.application.lab_compare import default_dirs
from contexts.learning.application.lab_view import LabView
from contexts.learning.lab_api import lab_router
from contexts.learning.lab_import import lab_import_router
from contexts.learning.tests import rehearsal

DUO, SOLO = rehearsal.DUO, rehearsal.SOLO
PART = f"{DUO}.pod-c"
TASK = "desk-objects"


def box(tmp_path):
    view, bus = LabView(), EventBus(log_dir=tmp_path / "events")
    view.subscribe(bus)
    app = FastAPI()
    app.include_router(lab_import_router(view, bus.publish, data_dir=tmp_path / "data"))
    app.include_router(lab_router(view, sample_dir=None, metrics_file=None, loop_dirs=default_dirs(tmp_path / "data")))
    return TestClient(app)


def duo_lines():
    """Act 2's stage lines as pod D writes them now: the start line names the members, the pods and the part."""
    lines = [x for x in rehearsal.jsonl("stages.jsonl") if x.get("act") == "duo"]
    lines[0] = {**lines[0], "members": 2, "pods": ["pod-d", "pod-c"], "parts": {"pod-c": PART}}
    return lines


def rank1_lines():
    return [{"act": "duo", "stage": "rank1_train", "ts": "2026-10-11T08:06:24.000Z", "node": "pod-c", "run_id": PART},
            {"act": "duo", "stage": "rank1_trained", "ts": "2026-10-11T08:07:30.000Z", "node": "pod-c", "run_id": PART,
             "rc": 0}]


def post(client, lines, run_id):
    r = client.post(f"/api/learning/lab/import?kind=stage&run_id={run_id}&task={TASK}",
                    content="".join(json.dumps(x) + "\n" for x in lines))
    assert r.status_code == 200, r.text
    return r.json()


def test_act_two_on_two_pods_reads_two_members_and_both_pods_with_pod_c_inside_it(tmp_path):
    client = box(tmp_path)
    post(client, duo_lines()[:3], DUO)          # start, labeled, train (trained is still to come)
    steps = rehearsal.text("duo/telemetry.jsonl")
    assert client.post(f"/api/learning/lab/import?run_id={DUO}", content=steps).status_code == 200
    part_steps = steps.replace(f'"run_id":"{DUO}"', f'"run_id":"{PART}"').replace('"event_id":"', '"event_id":"c')
    assert client.post(f"/api/learning/lab/import?run_id={PART}", content=part_steps).status_code == 200
    post(client, rank1_lines(), PART)

    story = client.get(f"/api/learning/lab/{DUO}").json()
    assert story["members"] == 2 and story["pods"] == ["pod-d", "pod-c"] and story["part_of"] is None
    assert story["loop"]["source"]["ref"].endswith(f"import:{DUO}:loop.act")
    training = next(s for s in story["stage_strip"] if s["stage"] == "training")
    assert training["pods"] == ["pod-d", "pod-c"] and training["state"] == "now"
    (part,) = story["parts"]
    assert part["node"] == "pod-c" and part["run_id"] == PART and part["steps"] == len(steps.splitlines())
    assert part["loss_first"] is not None and len(part["points"]) == part["steps"]

    runs = [r["run_id"] for r in client.get("/api/learning/lab").json()["runs"]]
    assert DUO in runs and PART not in runs                     # pod C's rank is not a run of its own
    assert client.get(f"/api/learning/lab/{PART}").json()["part_of"] == DUO   # still reachable by its id

    again = post(client, duo_lines()[:1], DUO)
    assert again["published"] == 0                               # the act event, like the rest, lands once


def test_a_start_line_with_bad_parts_is_refused(tmp_path):
    client = box(tmp_path)
    bad = {**duo_lines()[0], "parts": {"pod-c": "someone-else"}}
    r = client.post(f"/api/learning/lab/import?kind=stage&run_id={DUO}&task={TASK}", content=json.dumps(bad))
    assert r.status_code == 400 and "parts" in r.text
    r = client.post(f"/api/learning/lab/import?kind=stage&run_id={DUO}&task={TASK}",
                    content=json.dumps({**duo_lines()[0], "members": 0}))
    assert r.status_code == 400 and "members" in r.text


def test_the_compare_tab_gets_both_pods_training_rows_and_the_members(tmp_path):
    client = box(tmp_path)
    doc = json.loads(rehearsal.text("comparison.json"))
    duo = doc["acts"]["duo"]
    d, c = 3000.0, 2500.0
    duo["training"]["joules"] = d + c
    duo.update(members=2, pods=["pod-d", "pod-c"], teacher_nodes=["pod-c", "pod-d"],
               training_pods={"pod-d": {"joules": d, "steps": 8, "node": "pod D (rank 0)"},
                              "pod-c": {"joules": c, "steps": 8, "node": "pod C (rank 1)", "run_id": PART}})
    doc["acts"]["solo"].update(members=1, pods=["pod-d"], teacher_nodes=["pod-c", "pod-d"])
    assert client.post("/api/learning/lab/import?kind=compare", json=doc).status_code == 200
    out = client.get("/api/learning/lab/compare").json()
    assert out["b"]["members"] == 2 and out["b"]["pods"] == ["pod-d", "pod-c"]
    assert out["b"]["training_pods"]["pod-c"]["joules"] == c and out["a"]["pods"] == ["pod-d"]
    assert out["b"]["teacher_nodes"] == ["pod-c", "pod-d"]
    assert out["ratios"]["training_j"] == round((d + c) / doc["acts"]["solo"]["training"]["joules"], 3)
    assert {r["run_id"]: r["pods"] for r in out["runs"]}[DUO] == ["pod-d", "pod-c"]
