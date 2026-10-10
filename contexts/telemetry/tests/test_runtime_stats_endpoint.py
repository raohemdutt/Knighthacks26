"""GET /runtime/stats on the running app: mounted by the gateway, rebuilt from the event log on start, and reading the
lab's committed metrics file through lab/machine."""

import asyncio
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from bus import EventBus
from contexts.gateway.app import create_app
from contexts.gateway.bootstrap import Settings
from contracts.learning import ModelDeployed, RunEvaluated
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
                          "train_minutes", "teacher_tok_s", "eval_scores"]
    assert body["eval_scores"]["models"] == [
        {"model_version": "atlas-v2-desk", "deployed_ts": "2026-10-10T14:05:00Z", "previous_version": "atlas-base",
         "task": "desk-objects", "run_id": "r1", "decision": "deploy", "before": 0.41, "after": 0.78, "teacher": None,
         "ts": "2026-10-10T14:00:00Z", "source": f"learning.run.evaluated {evaluated.event_id}", "reason": None,
         "low_confidence": False}]
    teacher = metrics.latest("teacher.decode_tok_s")  # the committed lab/refine/metrics.jsonl
    assert (body["teacher_tok_s"]["value"], body["teacher_tok_s"]["node"], body["teacher_tok_s"]["low_confidence"]) == (
        teacher["value"], teacher["node"], teacher["low_confidence"])
    assert body["teacher_tok_s"]["ts"] == teacher["ts"]
    assert body["teacher_tok_s"]["source"].startswith("metrics.jsonl:")
    assert body["detector_fps"]["value"] is None
    assert body["detector_fps"]["reason"].startswith("not recorded yet: no detector.fps row for node jetson")
    assert body["student_tok_s"]["reason"].endswith("telemetry is off on this box (ATLAS_TELEMETRY_SOURCE is not set)")
