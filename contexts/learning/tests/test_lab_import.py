"""Importing a run's files: the trainer's own lines go into the log as they are, twice adds nothing, a bad line
refuses the file, and the lab view then serves the real run instead of the sample."""

import json
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

from bus import EventBus
from contexts.learning import lab_import
from contexts.learning.application.lab_view import LabView
from contexts.learning.lab_api import lab_router
from contexts.learning.lab_import import ImportRefused, eval_events, lab_import_router, manifest_events, parse_steps
from contexts.learning.tests.test_lab_runs import LOSSES, write_steps
from contracts.learning import RunEvaluated
from contracts.telemetry import TrainStep


def telemetry(tmp_path, losses=LOSSES) -> str:
    """A telemetry.jsonl written by lab/train.py's own StepLog."""
    write_steps(tmp_path / "run", losses)
    return (tmp_path / "run" / "telemetry.jsonl").read_text(encoding="utf-8")


def scores_json(run_id="run-l16") -> str:
    """scores.json as lab/eval_gate.py writes it (the keys of its `result`, without ok, outcome and out)."""
    event = RunEvaluated(event_id=f"gate-{run_id}", ts=datetime(2026, 10, 11, 6, 0, tzinfo=UTC), task="desk-objects", run_id=run_id, decision="deploy",
                         scores={"current": 0.4, "candidate": 0.8, "teacher": 0.9, "gate.noise": 0.02}, model_version="atlas-v3-desk-objects")
    return json.dumps({"decision": "deploy", "reasons": [], "run_id": run_id, "task": "desk-objects", "model_version": event.model_version,
                       "scores": event.scores, "noise": 0.02, "delta": 0.4, "p_value": 0.001, "n_items": 60, "n_groups": 20, "held_out": 60,
                       "unscored": 0, "split_sha256": "ab" * 32, "decode": {"thinking": "off", "temperature": 0}, "current_runs": [0.4, 0.42],
                       "event": json.loads(event.model_dump_json())})


MANIFEST = json.dumps({"version": "atlas-v3-desk-objects", "base": "Qwen/Qwen3.5-4B", "task": "desk-objects", "gguf": "atlas-v3.gguf",
                       "sha256": {"atlas-v3.gguf": "cd" * 32}, "created": "2026-10-11T06:10:00Z"})


def app_with(view, bus, token=None):
    def guard(authorization: str | None = Header(default=None)) -> None:
        if token and authorization != f"Bearer {token}":
            raise HTTPException(401, "token needed")

    app = FastAPI()
    app.include_router(lab_import_router(view, bus.publish, guard=guard))
    app.include_router(lab_router(view))
    return app


def test_the_lines_are_published_exactly_as_the_trainer_wrote_them(tmp_path):
    text = telemetry(tmp_path)
    steps = parse_steps(text, "run-l16")
    assert [json.loads(s.model_dump_json()) for s in steps] == [json.loads(line) for line in text.splitlines()]
    assert all(type(s) is TrainStep for s in steps) and [s.step for s in steps] == [1, 2, 3, 4, 5]


def test_a_line_that_does_not_parse_refuses_the_whole_file_and_names_every_line(tmp_path):
    good = telemetry(tmp_path).splitlines()
    missing = {k: v for k, v in json.loads(good[1]).items() if k != "step_ms"}
    extra = {**json.loads(good[2]), "watts": 80}
    bad = "\n".join([good[0], json.dumps(missing), json.dumps(extra), "{half", json.dumps({"type": "telemetry.gpu.power", "node": "gpu-b"})])
    with pytest.raises(ImportRefused) as err:
        parse_steps(bad)
    assert [p.split(":")[0] for p in err.value.problems] == ["line 2", "line 3", "line 4", "line 5"]
    assert "step_ms" in err.value.problems[0] and "watts" in err.value.problems[1]
    with pytest.raises(ImportRefused, match="run 'run-l16', not 'other'"):
        parse_steps(good[0], "other")
    with pytest.raises(ImportRefused, match="no step lines"):
        parse_steps("\n\n")


def test_posting_the_file_shows_the_real_run_live_and_posting_it_again_adds_nothing(tmp_path):
    view, bus = LabView(), EventBus(log_dir=tmp_path / "events")
    view.subscribe(bus)
    with TestClient(app_with(view, bus)) as client:
        assert client.get("/api/learning/lab").json()["sample"] is True
        first = client.post("/api/learning/lab/import?run_id=run-l16", content=telemetry(tmp_path, LOSSES[:3]))
        assert first.status_code == 200 and first.json() == {"kind": "telemetry", "published": 3, "already_there": 0, "events": 3,
                                                             "runs": ["run-l16"], "last_step": 3}
        lab = client.get("/api/learning/lab").json()
        assert lab["sample"] is False and lab["real_runs"] == 1 and lab["runs"][0]["status"] == "training"  # live: no eval yet
        grown = telemetry(tmp_path / "later")  # the same run, two steps further on
        assert client.post("/api/learning/lab/import", content=grown).json()["published"] == 2
        assert client.post("/api/learning/lab/import", content=grown).json() == {"kind": "telemetry", "published": 0, "already_there": 5,
                                                                              "events": 5, "runs": ["run-l16"], "last_step": 5}
        story = client.get("/api/learning/lab/run-l16").json()
        assert [p["loss"] for p in story["training"]["points"]] == LOSSES and story["sample"] is False
    assert sum(isinstance(e, TrainStep) for e in bus.read_log()) == 5  # the log holds each step once


def test_the_eval_and_the_manifest_map_onto_the_events_the_view_already_reads(tmp_path):
    view, bus = LabView(), EventBus(log_dir=tmp_path / "events")
    view.subscribe(bus)
    with TestClient(app_with(view, bus)) as client:
        client.post("/api/learning/lab/import", content=telemetry(tmp_path))
        assert client.post("/api/learning/lab/import?kind=eval", content=scores_json()).json()["published"] == 2
        assert client.post("/api/learning/lab/import?kind=eval", content=scores_json()).json()["published"] == 0
        assert client.post("/api/learning/lab/import?kind=manifest", content=MANIFEST).status_code == 400  # no run id
        assert client.post("/api/learning/lab/import?kind=manifest&run_id=run-l16", content=MANIFEST).json()["published"] == 1
        story = client.get("/api/learning/lab/run-l16").json()
    ev = story["eval"]
    assert (ev["decision"], ev["scores"]["candidate"]["value"], ev["n_items"]["value"], ev["n_groups"]["value"]) == ("deploy", 0.8, 60, 20)
    assert (ev["noise"]["value"], ev["p_value"]["value"]) == (0.02, 0.001) and ev["n_items"]["source"]["ref"].endswith("import:run-l16:eval.reported")
    assert ev["items"] is None and "no per-question answers" in ev["items_reason"]  # the gate's file has none, so none are shown
    assert story["status"] == "gated" and story["package"]["gguf"] == "atlas-v3.gguf" and story["package"]["sha256"] == "cd" * 32
    evaluated, reported = eval_events(scores_json())
    assert evaluated.event_id == json.loads(scores_json())["event"]["event_id"] and "correct" not in reported.model_dump()
    with pytest.raises(ImportRefused):
        eval_events("{}")
    with pytest.raises(ImportRefused):
        manifest_events('{"version": "x"}', "run-l16")


def test_the_post_is_guarded_and_capped(tmp_path, monkeypatch):
    view, bus = LabView(), EventBus(log_dir=tmp_path / "events")
    with TestClient(app_with(view, bus, token="secret")) as client:
        text = telemetry(tmp_path)
        assert client.post("/api/learning/lab/import", content=text).status_code == 401
        ok = {"Authorization": "Bearer secret"}
        bad = client.post("/api/learning/lab/import", content="{nope\n", headers=ok)
        assert bad.status_code == 400 and bad.json()["detail"]["refused"] == ["line 1: not JSON"]
        assert client.post("/api/learning/lab/import?kind=other", content=text, headers=ok).status_code == 400
        monkeypatch.setattr(lab_import, "MAX_BYTES", 100)
        assert client.post("/api/learning/lab/import", content=text, headers=ok).status_code == 413
    assert list(bus.read_log()) == []


def test_the_command_imports_into_the_log_on_disk_once(tmp_path, capsys):
    telemetry(tmp_path)
    (tmp_path / "scores.json").write_text(scores_json())
    args = ["--telemetry", str(tmp_path / "run" / "telemetry.jsonl"), "--eval", str(tmp_path / "scores.json"), "--data-dir", str(tmp_path / "data")]
    assert lab_import.main(args) == 0
    out = json.loads(capsys.readouterr().out)
    assert (out["telemetry"]["published"], out["eval"]["published"]) == (5, 2)
    assert lab_import.main(args) == 0
    again = json.loads(capsys.readouterr().out)
    assert (again["telemetry"]["published"], again["telemetry"]["already_there"], again["eval"]["published"]) == (0, 5, 0)
    log = list(EventBus(log_dir=tmp_path / "data" / "events").read_log())
    assert len(log) == 7 and LabView().story("x") is None
    view = LabView()
    for e in log:
        view.on_event(e)
    assert view.story("run-l16")["training"]["steps"]["value"] == 5
    (tmp_path / "bad.jsonl").write_text("{nope\n")
    assert lab_import.main(["--telemetry", str(tmp_path / "bad.jsonl"), "--data-dir", str(tmp_path / "data")]) == 1
    assert "line 1: not JSON" in capsys.readouterr().err and len(list(EventBus(log_dir=tmp_path / "data" / "events").read_log())) == 7
