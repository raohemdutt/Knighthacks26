"""GET /runtime/stats on the running app: mounted by the gateway, rebuilt from the event log on start, and reading the
lab's committed metrics file through lab/machine."""

import asyncio
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from bus import EventBus
from contexts.gateway.app import create_app
from contexts.gateway.bootstrap import Settings
from contracts.learning import ModelDeployed, RunEvaluated
from contracts.telemetry import TrainStep
from lab.machine import metrics

T0 = datetime(2026, 10, 10, 14, 0, tzinfo=UTC)


def test_the_gateway_serves_every_figure_and_rebuilds_eval_scores_from_the_log(tmp_path, monkeypatch):
    monkeypatch.delenv("ATLAS_TELEMETRY_SOURCE", raising=False)
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "lab-state"))  # an empty ledger, whatever this machine holds
    evaluated = RunEvaluated(ts=T0, task="desk-objects", run_id="r1", scores={"current": 0.41, "candidate": 0.78},
                             decision="deploy", model_version="atlas-v2-desk")
    deployed = ModelDeployed(ts=T0.replace(minute=5), model_version="atlas-v2-desk", previous_version="atlas-base")
    asyncio.run(EventBus(tmp_path / "events").publish_all([evaluated, deployed]))  # recorded before this start
    settings = Settings(data_dir=tmp_path, connectors=False, ntfy_url=None, llm_url=None)

    with TestClient(create_app(settings)) as c:
        r = c.get("/runtime/stats")

    assert r.status_code == 200
    body = r.json()
    assert list(body) == ["detector_fps", "student_tok_s", "caption_latency_s", "trainer_samples_per_s",
                          "train_minutes", "train_loss", "teacher_tok_s", "eval_scores"]
    assert body["eval_scores"]["models"] == [
        {"model_version": "atlas-v2-desk", "deployed_ts": "2026-10-10T14:05:00Z", "previous_version": "atlas-base",
         "task": "desk-objects", "run_id": "r1", "decision": "deploy", "before": 0.41, "after": 0.78, "teacher": None,
         "ts": "2026-10-10T14:00:00Z", "source": f"learning.run.evaluated {evaluated.event_id}", "reason": None,
         "low_confidence": False}]
    # the committed lab/refine/metrics.jsonl's newest row, leaving out a reference on another chip (R57's A100)
    teacher = [r for r in metrics.history("teacher.decode_tok_s") if r["conditions"].get("other_chip") is not True][-1]
    assert (body["teacher_tok_s"]["value"], body["teacher_tok_s"]["node"], body["teacher_tok_s"]["low_confidence"]) == (
        teacher["value"], teacher["node"], teacher["low_confidence"])
    assert body["teacher_tok_s"]["ts"] == teacher["ts"]
    assert body["teacher_tok_s"]["source"].startswith("metrics.jsonl:")
    assert body["detector_fps"]["value"] is None
    assert body["detector_fps"]["reason"].startswith("not recorded yet: no detector.fps row for node jetson")
    assert body["student_tok_s"]["reason"].endswith("telemetry is off on this box (ATLAS_TELEMETRY_SOURCE is not set)")


def test_a_training_runs_loss_pushed_from_the_lab_reaches_runtime_stats_and_survives_a_restart(tmp_path, monkeypatch):
    """L3's 'logs loss to the stats page': lab/train.py's telemetry.jsonl, pushed to POST /api/learning/lab/import on a
    box with no sampler configured, is served as train_loss, and again after the gateway restarts from its log."""
    monkeypatch.delenv("ATLAS_TELEMETRY_SOURCE", raising=False)
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "lab-state"))
    steps = [TrainStep(ts=T0.replace(second=10 * k), node="pod-b", run_id="run-l16", step=k, loss=loss, step_ms=9000.0,
                       socket_w=61.0) for k, loss in ((1, 2.31), (2, 1.88), (3, 1.52))]
    telemetry_jsonl = "".join(s.model_dump_json() + "\n" for s in steps)  # exactly what lab/train.py's StepLog writes
    settings = Settings(data_dir=tmp_path, connectors=False, ntfy_url=None, llm_url=None, token=None)

    with TestClient(create_app(settings)) as c:
        before = c.get("/runtime/stats").json()["train_loss"]
        pushed = c.post("/api/learning/lab/import?run_id=run-l16", content=telemetry_jsonl)
        live = c.get("/runtime/stats").json()["train_loss"]
    assert before["value"] is None
    assert pushed.status_code == 200 and pushed.json()["published"] == 3
    assert (live["value"], live["node"], live["ts"]) == (1.52, "pod-b", "2026-10-10T14:00:30Z")
    assert live["source"].startswith("telemetry.train.step: run run-l16, step 3 of 3 (first loss 2.31)")

    with TestClient(create_app(settings)) as c:  # a restart: rebuilt from the event log on disk
        again = c.get("/runtime/stats").json()
    assert again["train_loss"]["value"] == 1.52
    assert again["train_minutes"]["value"] == (20 + 9) / 60
