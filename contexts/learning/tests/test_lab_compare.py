"""The Learning lab's Compare tab (L31): two runs side by side, from what the relay posted to this box and the box's
own log, never from a fixture. The rehearsal under fixtures/learning/live-loop-2026-10-11 is posted here through the
import route, the way the relay posts a live run, so the data shapes are the real ones."""

import json
import shutil
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from bus import EventBus
from contexts.learning.application.lab_compare import RATIO_KEYS, default_dirs
from contexts.learning.application.lab_view import LabView
from contexts.learning.application.loop_energy import iso_s, window
from contexts.learning.lab_api import lab_router
from contexts.learning.lab_import import lab_import_router
from contexts.learning.tests.test_lab_runs import LOSSES, write_steps

REHEARSAL = Path(__file__).resolve().parents[3] / "fixtures" / "learning" / "live-loop-2026-10-11"
SOLO, DUO = "live-1011-040342-solo", "live-1011-040342-duo"
COMPARISON_KEYS = {"stamp", "total_s", "acts", "rule", "idle", "clock_offsets_s", "pod_d_split", "duo_over_solo", "go_s"}


def recorded():
    return json.loads((REHEARSAL / "comparison.json").read_text(encoding="utf-8"))


def stages():
    return [json.loads(x) for x in (REHEARSAL / "stages.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


def with_t0(doc):
    """The relay's document carries its stopwatch zero; the committed copy left it out, so put it back as --fallback
    works it out (the go line on pod D's clock, moved onto the relay's, minus go_s)."""
    go = next(s for s in stages() if s["stage"] == "go")
    return {**doc, "t0": iso_s(go["ts"]) - doc["clock_offsets_s"]["pod-d"] - doc["go_s"]}


def power_rows(node):
    return [json.loads(x) for x in (REHEARSAL / f"power-{node}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


def windows_for(act, run_id):
    """The relay's power windows for one act: labeling on pod C, training and packaging on pod D, and the whole act on
    each of the three nodes, each the samples a few seconds either side of the window on the node's own clock."""
    offsets = recorded()["clock_offsets_s"]
    out = []
    for stage, first, last, node in (("labeling", "start", "labeled", "pod-c"), ("training", "train", "trained", "pod-d"),
                                     ("packaging", "trained", "done", "pod-d"), ("act", "start", "done", "pod-c"),
                                     ("act", "start", "done", "pod-d"), ("act", "start", "done", "box")):
        a, b = window(stages(), act, first, last)
        d = offsets[node] - offsets["pod-d"]
        samples = [{"ts": r["ts"], "w": r["w"]} for r in power_rows(node) if a + d - 5 <= iso_s(r["ts"]) <= b + d + 5]
        out.append((stage, {"run_id": run_id, "node": node, "sampler": "test", "period_ms": 1000, "samples": samples,
                            "idle_w": recorded()["idle"][node]["calibrated_w"], "clock_offset_s": offsets[node]}))
    return out


def box(tmp_path, view=None):
    view = view or LabView()
    bus = EventBus(log_dir=tmp_path / "events")
    view.subscribe(bus)
    data = tmp_path / "data"
    app = FastAPI()
    app.include_router(lab_import_router(view, bus.publish, data_dir=data))
    app.include_router(lab_router(view, sample_dir=None, metrics_file=None, loop_dirs=default_dirs(data)))
    return TestClient(app), view


def post_loop(client, doc=None, power=True):
    assert client.post("/api/learning/lab/import?kind=compare", json=doc or with_t0(recorded())).status_code == 200
    if power:
        for act, run_id in (("solo", SOLO), ("duo", DUO)):
            for stage, win in windows_for(act, run_id):
                r = client.post(f"/api/learning/lab/import?kind=power&run_id={run_id}&stage={stage}", json=win)
                assert r.status_code == 200, r.text


def test_a_box_that_holds_no_loop_answers_404_in_plain_words_even_with_the_rehearsal_in_the_checkout(tmp_path, monkeypatch):
    monkeypatch.setenv("ATLAS_DATA_DIR", str(tmp_path / "empty"))
    assert (REHEARSAL / "comparison.json").is_file()   # the fixture is right there, and is not served
    client, _ = box(tmp_path)
    r = client.get("/api/learning/lab/compare")
    assert r.status_code == 404 and "no live-loop comparison yet" in r.json()["detail"]
    assert default_dirs(None) == []
    bare = FastAPI()
    bare.include_router(lab_router(LabView(), sample_dir=None, metrics_file=None))   # no loop folders: the data dir
    assert TestClient(bare).get("/api/learning/lab/compare").status_code == 404


def test_the_defaults_are_the_newest_loops_solo_run_against_its_pooled_one(tmp_path):
    client, _ = box(tmp_path)
    post_loop(client)
    out = client.get("/api/learning/lab/compare").json()
    assert out["asked"] == {"a": SOLO, "b": DUO} and out["same_loop"] is True
    assert out["a"]["members"] == 1 and out["b"]["members"] == 2
    assert out["a"]["examples"] == 100 and out["b"]["examples"] == 200 and out["missing"] == []


def test_the_answer_is_the_comparison_documents_shape_with_the_two_runs_as_its_acts(tmp_path):
    client, _ = box(tmp_path)
    post_loop(client)
    out = client.get("/api/learning/lab/compare").json()
    doc = recorded()
    assert COMPARISON_KEYS <= set(out)
    assert out["acts"] == doc["acts"] and out["idle"] == doc["idle"] and out["stamp"] == doc["stamp"]
    assert out["duo_over_solo"] == doc["duo_over_solo"]   # recomputed by loop_energy.ratios: the relay's own figures


def test_the_ratios_are_the_relays_own_duo_over_solo_plus_members_and_stage_seconds(tmp_path):
    client, _ = box(tmp_path)
    post_loop(client)
    ratios = client.get("/api/learning/lab/compare").json()["ratios"]
    for key, value in recorded()["duo_over_solo"].items():
        assert ratios[key] == value, key
    assert ratios["members"] == 2.0 and ratios["labeling_s"] == round(61.8 / 32.1, 3)
    assert set(ratios) == set(RATIO_KEYS)


def test_each_run_carries_its_stage_numbers_with_the_idle_floor_beside(tmp_path):
    client, _ = box(tmp_path)
    post_loop(client)
    a = client.get("/api/learning/lab/compare").json()["a"]
    assert a["labeling"]["joules"] == 3723.0 and a["labeling"]["idle_j"] == 329.7
    assert a["training"]["j_per_step"] == 340.9 and a["packaging"]["joules"] == 3333.9
    assert a["seconds"] == {"labeling": 32.1, "waiting": 0.0, "training": 53.9, "packaging": 71.1, "total": 157.0}
    assert a["source"] == {"kind": "loop", "ref": "lab_loops/1011-040342/comparison.json", "stamp": "1011-040342",
                           "rule": recorded()["rule"], "split": recorded()["pod_d_split"]}
    assert a["idle"]["pod-c"]["calibrated_w"] == 10.27


def test_the_three_power_traces_share_one_time_axis_from_the_loops_start(tmp_path):
    client, _ = box(tmp_path)
    post_loop(client)
    s = client.get("/api/learning/lab/compare").json()["series"]
    assert set(s["nodes"]) == {"pod-c", "pod-d", "box"} and s["zero"].startswith("2026-10-11T08:03:4")
    runs = {r["run_id"]: r for r in s["runs"]}
    assert runs[SOLO]["start_s"] == 2.7 and runs[DUO]["start_s"] == 35.0   # the relay's own start_s, on the same axis
    assert runs[SOLO]["end_s"] == 159.7 and runs[SOLO]["stages"]["training"] == [34.8, 88.7]
    for node, body in s["nodes"].items():
        ts = [t for t, _ in body["points"]]
        assert ts and ts == sorted(ts), node
        assert -10 <= ts[0] <= 40 and ts[-1] <= 236, (node, ts[0], ts[-1])
        assert {w["run_id"] for w in body["windows"]} == {SOLO, DUO}
    assert s["nodes"]["pod-c"]["idle_w"] == 10.27 and s["nodes"]["box"]["idle_w"] == 15.76
    labeling = [w for t, w in s["nodes"]["pod-c"]["points"] if 5 <= t <= 30]
    assert labeling and min(labeling) > 60   # the teacher is busy labeling act 1 in those seconds


def test_each_run_also_has_its_traces_from_its_own_start(tmp_path):
    client, _ = box(tmp_path)
    post_loop(client)
    out = client.get("/api/learning/lab/compare").json()
    for run in (out["a"], out["b"]):
        assert set(run["traces"]) == {"pod-c", "pod-d", "box"}
        total = run["seconds"]["total"]
        for trace in run["traces"].values():
            ts = [t for t, _ in trace["points"]]
            assert ts and -5.5 <= ts[0] <= 5 and ts[-1] <= total + 5.5 and ts == sorted(ts), trace["name"]


def test_a_comparison_with_no_power_posted_yet_still_answers_with_empty_traces(tmp_path):
    client, _ = box(tmp_path)
    post_loop(client, power=False)
    out = client.get("/api/learning/lab/compare").json()
    assert out["acts"]["solo"]["labeling"]["joules"] == 3723.0
    assert all(body["points"] == [] for body in out["series"]["nodes"].values())


def test_after_act_one_only_the_defaults_answer_404_until_act_two_lands(tmp_path):
    client, _ = box(tmp_path)
    doc = with_t0(recorded())
    part = {**doc, "acts": {"solo": doc["acts"]["solo"]}}
    part.pop("duo_over_solo")
    post_loop(client, part, power=False)
    r = client.get("/api/learning/lab/compare")
    assert r.status_code == 404 and "does not have the pooled run (act 2) yet" in r.json()["detail"]
    post_loop(client, doc, power=False)
    assert client.get("/api/learning/lab/compare").status_code == 200


def test_a_run_the_box_does_not_hold_is_a_404_naming_it(tmp_path):
    client, _ = box(tmp_path)
    post_loop(client, power=False)
    r = client.get(f"/api/learning/lab/compare?a={SOLO}&b=live-9999-solo")
    assert r.status_code == 404 and "'live-9999-solo'" in r.json()["detail"]


def test_a_run_from_the_log_alone_compares_with_empty_joules_and_a_reason(tmp_path):
    view = LabView()
    client, view = box(tmp_path, view)
    post_loop(client, power=False)
    write_steps(tmp_path / "run", LOSSES)
    assert client.post("/api/learning/lab/import?run_id=run-l16", content=(tmp_path / "run" / "telemetry.jsonl").read_text()).status_code == 200
    out = client.get(f"/api/learning/lab/compare?a={SOLO}&b=run-l16").json()
    assert out["b"]["source"]["kind"] == "log" and out["b"]["labeling"] is None and out["ratios"]["labeling_j"] is None
    assert out["b"]["training"]["steps"] == len(LOSSES)   # the steps' own power readings, from the box's log
    assert set(out["acts"]) == {"solo", "b"} and out["same_loop"] is False and out["stamp"] is None
    assert {r["run_id"] for r in out["runs"]} >= {SOLO, DUO, "run-l16"}


def test_a_relay_out_folder_copied_onto_the_box_is_read_with_its_own_traces(tmp_path):
    client, _ = box(tmp_path)
    shutil.copytree(REHEARSAL, tmp_path / "data" / "lab_loops" / "1011-040342")   # copied by hand, power-*.jsonl beside it
    out = client.get("/api/learning/lab/compare").json()
    assert out["asked"] == {"a": SOLO, "b": DUO}
    assert all(len(body["points"]) > 50 for body in out["series"]["nodes"].values())
    assert out["series"]["runs"][0]["start_s"] == 2.7
