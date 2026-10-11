"""A run that stopped talking is stalled, not training; a stand-in gate says it is one.

Stalled is judged on event time (the repo rule): a running run whose last step or stage event is more than five
minutes older than the newest event this box holds. The stand-in gate is marked additively, by the relay's
`gate.stand_in` or by its canned answers, and a real gate's fields are left as they were."""

import json

from contexts.learning.application.lab_view import STAND_IN_NOTE
from contexts.learning.tests import rehearsal
from contexts.learning.tests.test_lab_live_story import SOLO, TASK, box, post_stage, solo_lines

OTHER = "live-1011-990002-solo"


def at(line, ts, run_id=None):
    return {**line, "ts": ts, **({"run_id": run_id} if run_id else {})}


def test_a_running_run_quiet_for_more_than_five_minutes_of_event_time_is_stalled_and_keeps_its_status(tmp_path):
    client = box(tmp_path)
    lines = solo_lines()
    post_stage(client, at(lines["start"], "2026-10-11T08:12:15Z"))
    story = client.get(f"/api/learning/lab/{SOLO}").json()
    assert story["run_status"] == "running" and story["stalled"] is None
    assert story["last_event_ts"].startswith("2026-10-11T08:12:15")

    # another run's event four minutes later: the box's newest event, not the wall clock, is "now"
    r = client.post(f"/api/learning/lab/import?kind=stage&run_id={OTHER}&task={TASK}",
                    content=json.dumps(at(lines["start"], "2026-10-11T08:16:15Z", OTHER)))
    assert r.status_code == 200, r.text
    assert client.get(f"/api/learning/lab/{SOLO}").json()["run_status"] == "running"

    r = client.post(f"/api/learning/lab/import?kind=stage&run_id={OTHER}&task={TASK}",
                    content=json.dumps(at(lines["labeled"], "2026-10-11T08:17:16Z", OTHER)))
    assert r.status_code == 200, r.text
    story = client.get(f"/api/learning/lab/{SOLO}").json()
    assert story["run_status"] == "stalled" and story["status"] == "training"
    assert story["stalled"]["quiet_s"] == 301.0 and story["stalled"]["after_s"] == 300
    listed = {x["run_id"]: x for x in client.get("/api/learning/lab").json()["runs"]}
    assert listed[SOLO]["run_status"] == "stalled" and listed[SOLO]["last_event_ts"].startswith("2026-10-11T08:12:15")
    assert listed[OTHER]["run_status"] == "running" and listed[OTHER]["stalled"] is None


def test_a_finished_run_is_never_stalled_however_old(tmp_path):
    client = box(tmp_path)
    lines = solo_lines()
    for name in ("start", "labeled", "train", "trained", "done"):
        post_stage(client, at(lines[name], "2026-10-11T08:00:00Z"))
    client.post(f"/api/learning/lab/import?kind=stage&run_id={OTHER}&task={TASK}",
                content=json.dumps(at(lines["start"], "2026-10-11T10:00:00Z", OTHER)))
    story = client.get(f"/api/learning/lab/{SOLO}").json()
    assert story["run_status"] == "finished" and story["stalled"] is None


def scores(**extra):
    return json.dumps({**json.loads(rehearsal.text("solo/scores.json")), **extra})


def test_the_stand_in_gate_is_marked_on_the_story_and_the_run_list_from_its_canned_answers(tmp_path):
    client = box(tmp_path)
    post_stage(client, solo_lines()["start"])
    r = client.post(f"/api/learning/lab/import?kind=eval&run_id={SOLO}", content=scores())
    assert r.status_code == 200, r.text
    story = client.get(f"/api/learning/lab/{SOLO}").json()
    assert story["gate"] == {"stand_in": True, "note": STAND_IN_NOTE, "marked_by": "answers"} == story["eval"]["gate"]
    original = json.loads(rehearsal.text("solo/scores.json"))
    assert story["eval"]["p_value"]["value"] == original["p_value"]  # the gate's own fields are untouched
    listed = client.get("/api/learning/lab").json()["runs"]
    assert listed[0]["gate"]["stand_in"] is True


def test_the_relays_marker_is_carried_and_a_real_gate_has_none(tmp_path):
    client = box(tmp_path)
    post_stage(client, solo_lines()["start"])
    note = "a stand-in gate: the live loop did not run the real held-out check"
    client.post(f"/api/learning/lab/import?kind=eval&run_id={SOLO}", content=scores(gate={"stand_in": True, "note": note}))
    assert client.get(f"/api/learning/lab/{SOLO}").json()["gate"] == {"stand_in": True, "note": note, "marked_by": "relay"}

    real = json.loads(rehearsal.text("solo/scores.json"))
    for i, it in enumerate(real["items"]):
        it["before"]["answer"], it["after"]["answer"] = f"a mug {i}", f"a red mug {i}"
    client = box(tmp_path / "real")
    post_stage(client, solo_lines()["start"])
    client.post(f"/api/learning/lab/import?kind=eval&run_id={SOLO}", content=json.dumps(real))
    story = client.get(f"/api/learning/lab/{SOLO}").json()
    assert story["gate"] is None and story["eval"]["n_items"]["value"] == real["n_items"]
