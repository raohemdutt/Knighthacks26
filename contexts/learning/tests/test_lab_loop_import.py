"""The live loop's own imports (L30, L31): the comparison document and the power windows kept as files on the box,
the stage lines as the run's progress, and the status every answer now carries. The rehearsal under
fixtures/learning/live-loop-2026-10-11 is the data shape here only; the box never reads that folder."""

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from bus import EventBus
from contexts.learning.application import lab_loops
from contexts.learning.application.lab_view import LabView
from contexts.learning.lab_api import lab_router
from contexts.learning.lab_import import lab_import_router
from contracts.learning import RunProgress

REHEARSAL = Path(__file__).resolve().parents[3] / "fixtures" / "learning" / "live-loop-2026-10-11"
SOLO = "live-1011-040342-solo"


def client_for(tmp_path, data_dir=True):
    view, bus = LabView(), EventBus(log_dir=tmp_path / "events")
    view.subscribe(bus)
    app = FastAPI()
    app.include_router(lab_import_router(view, bus.publish, data_dir=tmp_path / "data" if data_dir else None))
    app.include_router(lab_router(view, sample_dir=None, metrics_file=None))
    return TestClient(app), view, bus


def stage_lines(act="solo"):
    rows = [json.loads(x) for x in (REHEARSAL / "stages.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    return [r for r in rows if r.get("act") == act]


def window(run_id=SOLO, node="pod-c", n=5):
    return {"run_id": run_id, "node": node, "sampler": "lab/pod_power.py serve :8790", "period_ms": 1000,
            "idle_w": 10.27, "quiet_w": 11.0, "clock_offset_s": 63.88, "hub": "sjc.aupcloud.io", "account": "C",
            "error": None, "samples": [{"ts": f"2026-10-11T08:04:2{i}.000Z", "w": 100.0 + i} for i in range(n)]}


def test_a_comparison_is_kept_by_its_stamp_and_the_same_bytes_again_write_nothing(tmp_path):
    client, _, _ = client_for(tmp_path)
    raw = (REHEARSAL / "comparison.json").read_bytes()
    first = client.post("/api/learning/lab/import?kind=compare", content=raw)
    assert first.status_code == 200, first.text
    assert first.json() == {"kind": "compare", "changed": True, "stamp": "1011-040342",
                            "runs": {"solo": SOLO, "duo": "live-1011-040342-duo"}, "complete": True,
                            "stored": "lab_loops/1011-040342/comparison.json"}
    assert (tmp_path / "data" / "lab_loops" / "1011-040342" / "comparison.json").read_bytes() == raw
    assert client.post("/api/learning/lab/import?kind=compare", content=raw).json()["changed"] is False


def test_a_comparison_after_act_one_only_is_kept_and_then_replaced_by_the_whole_one(tmp_path):
    client, _, _ = client_for(tmp_path)
    whole = json.loads((REHEARSAL / "comparison.json").read_text(encoding="utf-8"))
    part = {**whole, "acts": {"solo": whole["acts"]["solo"]}}
    part.pop("duo_over_solo")
    assert client.post("/api/learning/lab/import?kind=compare", json=part).json()["complete"] is False
    answer = client.post("/api/learning/lab/import?kind=compare", json=whole).json()
    assert answer["complete"] is True and answer["changed"] is True


def test_a_comparison_without_a_plain_stamp_or_run_ids_is_refused_and_nothing_is_written(tmp_path):
    client, _, _ = client_for(tmp_path)
    for doc, why in (({"acts": {"solo": {"run_id": SOLO}}}, "stamp"),
                     ({"stamp": "../etc", "acts": {"solo": {"run_id": SOLO}}}, "stamp"),
                     ({"stamp": "s1", "acts": {"solo": {"run_id": "a/b"}}}, "acts.solo.run_id"),
                     ({"stamp": "s1", "acts": {}}, "acts"),
                     ([1, 2], "JSON object")):
        r = client.post("/api/learning/lab/import?kind=compare", json=doc)
        assert r.status_code == 400 and why in json.dumps(r.json()), (doc, r.text)
    assert client.post("/api/learning/lab/import?kind=compare", content=b"not json").status_code == 400
    assert not (tmp_path / "data" / "lab_loops").exists()


def test_a_gateway_that_passes_no_data_dir_keeps_them_in_its_own_data_dir(tmp_path, monkeypatch):
    """An older app.py on the box mounts the import route without a data dir: the files go where the gateway keeps
    its data anyway (ATLAS_DATA_DIR, else ./data), and the compare route reads them from there."""
    monkeypatch.setenv("ATLAS_DATA_DIR", str(tmp_path / "gateway-data"))
    client, _, _ = client_for(tmp_path, data_dir=False)
    assert client.post("/api/learning/lab/import?kind=compare", content=(REHEARSAL / "comparison.json").read_bytes()).status_code == 200
    assert client.post(f"/api/learning/lab/import?kind=power&run_id={SOLO}&stage=labeling", json=window()).status_code == 200
    assert (tmp_path / "gateway-data" / "lab_loops" / "1011-040342" / "comparison.json").is_file()
    assert (tmp_path / "gateway-data" / "lab_loops" / "_power" / SOLO / "labeling.pod-c.json").is_file()
    assert client.get("/api/learning/lab/compare").json()["asked"]["a"] == SOLO


def test_a_power_window_is_kept_by_run_stage_and_node_with_its_sampler_fields(tmp_path):
    client, _, _ = client_for(tmp_path)
    r = client.post(f"/api/learning/lab/import?kind=power&run_id={SOLO}&stage=labeling", json=window())
    assert r.status_code == 200, r.text
    assert r.json() == {"kind": "power", "changed": True, "run_id": SOLO, "stage": "labeling", "node": "pod-c", "samples": 5,
                        "stored": f"lab_loops/_power/{SOLO}/labeling.pod-c.json"}
    kept = lab_loops.power_windows(tmp_path / "data" / "lab_loops", SOLO)
    assert len(kept) == 1 and kept[0]["sampler"] == "lab/pod_power.py serve :8790" and kept[0]["period_ms"] == 1000
    assert kept[0]["idle_w"] == 10.27 and kept[0]["hub"] == "sjc.aupcloud.io" and kept[0]["account"] == "C"
    assert kept[0]["samples"][0] == {"ts": "2026-10-11T08:04:20.000Z", "w": 100.0}
    again = client.post(f"/api/learning/lab/import?kind=power&run_id={SOLO}&stage=labeling", json=window())
    assert again.json()["changed"] is False


def test_a_power_window_takes_watts_under_either_name_and_refuses_what_is_not_a_reading(tmp_path):
    client, _, _ = client_for(tmp_path)
    w = window()
    w["samples"] = [{"ts": "2026-10-11T08:04:20Z", "watts": 50}, {"ts": "2026-10-11T08:04:21Z", "w": 51}]
    assert client.post(f"/api/learning/lab/import?kind=power&run_id={SOLO}&stage=training", json=w).status_code == 200
    for bad, why in (({**window(), "samples": []}, "samples"),
                     ({**window(), "samples": [{"ts": "yesterday", "w": 1}]}, "sample 1: ts"),
                     ({**window(), "samples": [{"ts": "2026-10-11T08:04:20Z", "w": "lots"}]}, "sample 1: w"),
                     ({**window(), "node": "../x"}, "node"),
                     ({**window(), "run_id": "another-run"}, "not"),
                     ({**window(), "period_ms": "fast"}, "period_ms")):
        r = client.post(f"/api/learning/lab/import?kind=power&run_id={SOLO}&stage=training", json=bad)
        assert r.status_code == 400 and why in json.dumps(r.json()), (bad, r.text)
    assert client.post("/api/learning/lab/import?kind=power&stage=training", json={**window(), "run_id": None}).status_code == 400


def test_the_stage_lines_become_the_runs_progress_once(tmp_path):
    client, _, bus = client_for(tmp_path)
    body = "\n".join(json.dumps(r) for r in stage_lines()) + "\n"
    r = client.post(f"/api/learning/lab/import?kind=stage&run_id={SOLO}&task=desk-objects", content=body)
    assert r.status_code == 200, r.text
    answer = r.json()
    assert answer["published"] == 6 and answer["skipped"] == [] and answer["status"] == "finished"
    progress = [e for e in bus.read_log() if isinstance(e, RunProgress)]
    assert [(p.stage, p.progress, p.message) for p in progress] == [
        ("labeling", 0.0, "labeling at the teacher"), ("labeling", 1.0, "labeled"), ("training", 0.0, "training started"),
        ("training", 1.0, "trained"), ("packaging", 0.0, "packaging"), ("packaging", 1.0, "offered")]
    assert all(p.task == "desk-objects" and p.event_id.startswith(f"import:{SOLO}:stage.") for p in progress)
    again = client.post(f"/api/learning/lab/import?kind=stage&run_id={SOLO}&task=desk-objects", content=body).json()
    assert again["published"] == 0 and again["already_there"] == 6


def test_stage_lines_post_as_they_come_and_the_answer_says_running_until_offered(tmp_path):
    client, _, _ = client_for(tmp_path)
    lines = stage_lines()
    url = f"/api/learning/lab/import?kind=stage&run_id={SOLO}"
    first = client.post(url + "&task=desk-objects", content=json.dumps(lines[0]))   # start: the task rides on the query once
    assert first.json()["status"] == "running" and first.json()["steps_so_far"] == 0
    for line in lines[1:-1]:
        assert client.post(url, content=json.dumps(line)).json()["status"] == "running"   # the log knows the task now
    assert client.post(url, content=json.dumps(lines[-1])).json()["status"] == "finished"


def test_a_stage_body_is_refused_without_a_run_id_a_task_or_with_another_runs_line(tmp_path):
    client, _, _ = client_for(tmp_path)
    start = json.dumps(stage_lines()[0])
    assert "run id" in client.post("/api/learning/lab/import?kind=stage&task=t", content=start).text
    assert "task is not known" in client.post(f"/api/learning/lab/import?kind=stage&run_id={SOLO}", content=start).text
    other = client.post("/api/learning/lab/import?kind=stage&run_id=another&task=t", content=start)
    assert other.status_code == 400 and f"run '{SOLO}', not 'another'" in other.text
    assert client.post(f"/api/learning/lab/import?kind=stage&run_id={SOLO}&task=t", content="{}").status_code == 400


def test_a_task_on_the_lines_is_enough_and_unknown_stages_are_counted_not_guessed(tmp_path):
    client, _, _ = client_for(tmp_path)
    go = {"act": "all", "stage": "go", "ts": "2026-10-11T08:03:48.850Z"}
    start = {**stage_lines()[0], "task": "desk-objects"}
    r = client.post(f"/api/learning/lab/import?kind=stage&run_id={SOLO}", content=f"{json.dumps(go)}\n{json.dumps(start)}\n")
    assert r.status_code == 200 and r.json()["published"] == 1 and r.json()["skipped"] == ["go"]


def test_an_act_that_ends_badly_reads_failed(tmp_path):
    client, _, bus = client_for(tmp_path)
    lines = stage_lines()[:3] + [{"act": "solo", "stage": "done", "ts": "2026-10-11T08:05:00Z", "rc": 3}]
    r = client.post(f"/api/learning/lab/import?kind=stage&run_id={SOLO}&task=desk-objects",
                    content="\n".join(json.dumps(x) for x in lines))
    assert r.json()["status"] == "failed"
    last = [e for e in bus.read_log() if isinstance(e, RunProgress)][-1]
    assert last.stage == "failed" and "rc 3" in last.message


def test_a_partial_telemetry_file_is_taken_and_the_answer_says_running_with_the_steps_so_far(tmp_path):
    client, _, _ = client_for(tmp_path)
    steps = (REHEARSAL / "solo" / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()
    first = client.post(f"/api/learning/lab/import?run_id={SOLO}", content="\n".join(steps[:3]) + "\n").json()
    assert first["status"] == "running" and first["steps_so_far"] == 3
    grown = client.post(f"/api/learning/lab/import?run_id={SOLO}", content="\n".join(steps) + "\n").json()
    assert grown["published"] == len(steps) - 3 and grown["steps_so_far"] == len(steps) and grown["status"] == "running"
