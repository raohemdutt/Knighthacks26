"""The trainer's own files become the run story the Training tab already draws. The fixture is written by the
trainer's own step logger (lab/train.py StepLog) and its own answer shape, so it is the shape a real run leaves."""

import json
from datetime import UTC, datetime, timedelta

from fastapi import FastAPI
from fastapi.testclient import TestClient

from contexts.learning.application.lab_runs import TrainerRuns, attach, default_runs_dir, trainer_events
from contexts.learning.application.lab_view import LabView
from contexts.learning.lab_api import lab_router
from contracts.telemetry import TrainStep
from lab import train

T0 = datetime(2026, 10, 11, 5, 30, tzinfo=UTC)
LOSSES = [2.31, 1.74, 1.02, 0.61, 0.33]


class Clock:
    def __init__(self, start: float) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def power_sample(ts: datetime, watts: float) -> str:
    return json.dumps({"type": "telemetry.gpu.power", "ts": ts.isoformat(), "node": "gpu-b", "source": "amd-smi", "period_ms": 1000.0,
                       "socket_w": watts, "gfx_mhz": None, "mem_mhz": None, "throttled": None, "sampler": "lab/pod_power.py 1"})


def write_steps(folder, losses=LOSSES, run_id="run-l16", power=True):
    """Drive the trainer's StepLog exactly as its callback does: begin, end, logged, one step every 4 s."""
    folder.mkdir(parents=True, exist_ok=True)
    feed = None
    if power:
        (folder / "power").mkdir(exist_ok=True)
        (folder / "power" / "samples-2026-10-11.jsonl").write_text(
            "\n".join(power_sample(T0 + timedelta(seconds=s), 80.0 + s % 3) for s in range(0, 4 * len(losses) + 1)) + "\n", encoding="utf-8")
        feed = train.PowerFeed(folder / "power")
    wall, timer = Clock(T0.timestamp()), Clock(10.0)
    log = train.StepLog(folder / "telemetry.jsonl", run_id, train.POD_NODE, feed, clock=wall, timer=timer)
    for step, loss in enumerate(losses, 1):
        log.begin()
        wall.now, timer.now = wall.now + 4.0, timer.now + 4.0
        log.end(step)
        log.logged(step, {"loss": loss, "num_tokens": 900.0 * step})
    return log


def write_run_json(folder, steps=len(LOSSES), outcome="trained"):
    """run.json as lab/train.py main() writes it: the answer of _answer(outcome, refs, metrics)."""
    answer = train._answer(outcome, {"run_id": "run-l16", "mode": "pod", "node": "pod-b", "task": "desk-objects", "dataset_id": "desk-objects-1",
                                     "action_id": "a1", "base_model": "Qwen/Qwen3.5-4B", "base_revision": "0" * 40, "out": str(folder)},
                           {"trainer.steps": steps, "trainer.batch": 4, "trainer.grad_accum": 2, "trainer.lora_rank": 16, "trainer.examples": 40,
                            "trainer.epochs": 1.0, "trainer.samples_per_s": 2.0, "trainer.loss_first": LOSSES[0], "trainer.loss_last": LOSSES[-1]})
    (folder / "run.json").write_text(json.dumps(answer, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return answer


def test_the_fixture_is_the_trainers_own_shape(tmp_path):
    write_steps(tmp_path / "run-l16")
    line = json.loads((tmp_path / "run-l16" / "telemetry.jsonl").read_text().splitlines()[0])
    assert set(line) == set(TrainStep.model_fields) and line["type"] == "telemetry.train.step" and line["node"] == "pod-b"
    assert set(write_run_json(tmp_path / "run-l16")) == {"ok", "outcome", "refs", "metrics"}


def test_a_run_still_training_shows_its_loss_per_step_and_power_from_the_step_file_alone(tmp_path):
    write_steps(tmp_path / "run-l16", LOSSES[:3])
    view = LabView()
    info = TrainerRuns(tmp_path).sync(view)
    assert info == [{"folder": "run-l16", "run_id": "run-l16", "steps": 3, "skipped_lines": 0, "outcome": None, "error": None,
                     "ended": False, "source": "run-l16/telemetry.jsonl"}]
    story = view.story("run-l16")
    assert story["sample"] is False and story["status"] == "training"
    assert [(p["step"], p["loss"]) for p in story["training"]["points"]] == [(1, 2.31), (2, 1.74), (3, 1.02)]
    assert story["training"]["steps"]["value"] == 3 and story["training"]["node"] == "pod-b"
    assert story["training"]["loss_first"]["value"] == 2.31 and "telemetry.train.step" in story["training"]["points_source"]["ref"]
    assert len(story["power"]["points"]) == 3 and story["power"]["energy_j"]["value"] > 0  # each step's mean watts x its length
    # what only run.json can say is missing with the reason, not guessed
    assert story["training"]["samples_per_s"]["value"] is None and "learning.training.started" in story["training"]["of"]["reason"]


def test_run_json_adds_the_settings_so_speed_is_computed_and_its_numbers_point_at_the_file(tmp_path):
    folder = tmp_path / "run-l16"
    write_steps(folder)
    write_run_json(folder)
    events, info = trainer_events(folder)
    assert events[0].type == "learning.training.started" and events[0].event_id == "run-l16/run.json"
    assert events[0].ts == T0  # one step before the first step ended
    assert (info["outcome"], info["ended"], info["steps"]) == ("trained", True, 5)
    view = LabView()
    TrainerRuns(tmp_path).sync(view)
    t = view.story("run-l16")["training"]
    assert (t["base"], t["lora_rank"], t["batch"], t["of"]["value"], t["samples"]["value"]) == ("Qwen/Qwen3.5-4B", 16, 8, 5, 40)
    assert t["samples_per_s"]["value"] == 2.0  # 5 steps x (4 x 2) samples over 20 s
    assert t["of"]["source"]["ref"] == "learning.training.started run-l16/run.json"
    assert view.story("run-l16")["task"]["slug"] == "desk-objects"


def test_a_growing_file_is_reread_and_a_step_is_never_counted_twice(tmp_path):
    folder = tmp_path / "run-l16"
    write_steps(folder, LOSSES[:2])
    view, runs = LabView(), TrainerRuns(tmp_path)
    runs.sync(view)
    runs.sync(view)  # unchanged: nothing is read again
    assert view.story("run-l16")["training"]["steps"]["value"] == 2
    (folder / "telemetry.jsonl").unlink()
    write_steps(folder, LOSSES)
    with (folder / "telemetry.jsonl").open("a") as f:
        f.write('{"type": "telemetry.train.step", "run_id": "run-l16", "step"\n')  # a half-written last line
        f.write(json.dumps({"type": "telemetry.gpu.power", "node": "gpu-b", "source": "amd-smi", "period_ms": 1000.0, "socket_w": 80.0,
                            "gfx_mhz": None, "mem_mhz": None, "throttled": None}) + "\n")
    info = runs.sync(view)
    assert view.story("run-l16")["training"]["steps"]["value"] == 5 and info[0]["skipped_lines"] == 2
    assert view.run_ids == ["run-l16"]


def test_no_folder_no_runs_and_the_env_names_the_folder(tmp_path, monkeypatch):
    view = LabView()
    assert TrainerRuns(tmp_path / "missing").sync(view) == [] and TrainerRuns(None).sync(view) == [] and view.run_ids == []
    assert default_runs_dir(tmp_path) == tmp_path / "lab_runs"
    monkeypatch.setenv("ATLAS_LAB_RUNS_DIR", str(tmp_path / "elsewhere"))
    assert default_runs_dir(tmp_path) == tmp_path / "elsewhere"
    write_steps(tmp_path / "flat")  # the run's files directly in the folder also count
    assert [r["folder"] for r in TrainerRuns(tmp_path / "flat").sync(view)] == ["flat"]


def test_the_api_serves_the_real_run_instead_of_the_sample_as_soon_as_its_files_exist(tmp_path):
    view = LabView()
    attach(view, tmp_path / "lab_runs")
    app = FastAPI()
    app.include_router(lab_router(view))
    with TestClient(app) as client:
        before = client.get("/api/learning/lab").json()
        assert before["sample"] is False and before["runs"] == [] and before["real_runs"] == 0 and "trainer_runs" not in before
        write_steps(tmp_path / "lab_runs" / "run-l16")
        after = client.get("/api/learning/lab").json()
        assert after["sample"] is False and after["real_runs"] == 1 and [r["run_id"] for r in after["runs"]] == ["run-l16"]
        assert after["trainer_runs"][0]["steps"] == 5
        story = client.get("/api/learning/lab/run-l16").json()
        assert story["sample"] is False and len(story["training"]["points"]) == 5
        assert all(n["source"] or n["reason"] for n in story["sources"])
