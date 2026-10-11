"""L29: a real run, imported file by file, moves the Teach screen's board (application/board.py) and the lab view's
stage, not only the lab view's loss curve. The files are the trainer's own (lab/train.py StepLog and _answer) and the
gate's scores.json shape; the board and the view are the ones the gateway wires."""

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from bus import EventBus
from contexts.learning import lab_import
from contexts.learning.application.board import LearningBoard
from contexts.learning.application.lab_view import LabView
from contexts.learning.lab_api import lab_router
from contexts.learning.lab_import import lab_import_router
from contexts.learning.tests.test_lab_import import MANIFEST, scores_json, telemetry
from contexts.learning.tests.test_lab_runs import LOSSES
from contracts.learning import RunProgress
from lab import train

TASK = "desk-objects"
IMPORT = "/api/learning/lab/import"


def run_json(run_id="run-l16", ok=True, steps=5) -> str:
    """run.json exactly as lab/train.py writes it: its _answer of the run's refs and metrics."""
    refs = {"run_id": run_id, "mode": "pod", "node": "pod-b", "task": TASK, "dataset_id": "desk-objects-14e303e5",
            "base_model": "Qwen/Qwen3.5-4B", "base_revision": "abc123", "seed": 42}
    metrics = {"trainer.steps": steps, "trainer.batch": 4, "trainer.grad_accum": 4, "trainer.examples": 136,
               "trainer.epochs": 3.0, "trainer.lora_rank": 16, "trainer.loss_first": LOSSES[0], "trainer.loss_last": LOSSES[-1],
               "trainer.train_s": 716.7, "trainer.samples_per_s": 0.11}
    answer = train._answer("trained", refs, metrics) if ok else train._answer("failed", refs, error="RuntimeError: HIP out of memory")
    return json.dumps(answer, indent=2, sort_keys=True)


def wired(tmp_path):
    bus = EventBus(log_dir=tmp_path / "events")
    view, board = LabView(), LearningBoard()
    view.subscribe(bus)
    board.subscribe(bus)
    app = FastAPI()
    app.include_router(lab_import_router(view, bus.publish))
    app.include_router(lab_router(view))
    return bus, view, board, app


def stage_of(board, run_id="run-l16"):
    task = board.snapshot()["tasks"][0]
    assert task["task"] == TASK and task["run_id"] == run_id
    return task


def test_a_real_run_moves_the_teach_board_from_training_to_eval_to_packaging(tmp_path):
    bus, view, board, app = wired(tmp_path)
    with TestClient(app) as client:
        # live, three steps in, with the task named: the board shows training, the step and the loss
        live = client.post(f"{IMPORT}?run_id=run-l16&task={TASK}", content=telemetry(tmp_path / "a", LOSSES[:3])).json()
        assert live["published"] == 4 and "note" not in live            # 3 steps + 1 progress
        t = stage_of(board)
        assert (t["stage"], t["progress"], t["loss"]) == ("training", 0.0, LOSSES[2])
        assert t["message"] == "step 3, planned steps not known yet"

        # the trainer's run.json: the settings the lab view needs, and the training stage closed at 1.0
        assert client.post(f"{IMPORT}?kind=run&run_id=run-l16", content=run_json()).json()["published"] == 2
        t = stage_of(board)
        assert (t["stage"], t["progress"], t["loss"]) == ("training", 1.0, LOSSES[-1]) and "5 steps" in t["message"]

        # the rest of the steps, now without &task=: the log knows the task and the planned steps
        rest = client.post(IMPORT, content=telemetry(tmp_path / "b")).json()
        assert rest["published"] == 3                                       # steps 4 and 5 + progress at step 5
        assert stage_of(board)["message"] == "step 5 of 5" and stage_of(board)["progress"] == 1.0

        # the gate: stage eval with the before and after
        assert client.post(f"{IMPORT}?kind=eval", content=scores_json()).json()["published"] == 3
        t = stage_of(board)
        assert t["stage"] == "eval" and t["scores"]["candidate"] == 0.8 and t["decision"] == "deploy"

        # a late repost of the steps or of run.json adds nothing and does not move the stage back
        late = client.post(IMPORT, content=telemetry(tmp_path / "c")).json()
        assert late["published"] == 0 and "past training" in late["note"]
        assert client.post(f"{IMPORT}?kind=run&run_id=run-l16", content=run_json()).json()["published"] == 0
        assert stage_of(board)["stage"] == "eval"

        # the package: stage packaging
        assert client.post(f"{IMPORT}?kind=manifest&run_id=run-l16", content=MANIFEST).json()["published"] == 2
        assert stage_of(board)["stage"] == "packaging"

        story = client.get("/api/learning/lab/run-l16").json()
    training = story["training"]
    assert [p["loss"] for p in training["points"]] == LOSSES                # the curve is the steps', not doubled
    assert training["points_source"]["ref"].startswith("one point per telemetry.train.step")
    assert (training["batch"], training["lora_rank"], training["of"]["value"]) == (16, 16, 5)
    assert training["samples"]["value"] == 5 * 16 and training["samples_per_s"]["value"] > 0
    stages = [e.stage for e in bus.read_log() if isinstance(e, RunProgress)]
    assert stages == ["training", "training", "training", "eval", "packaging"]


def test_run_json_checks_its_run_and_a_failed_run_says_failed(tmp_path):
    bus, view, board, app = wired(tmp_path)
    with TestClient(app) as client:
        wrong = client.post(f"{IMPORT}?kind=run&run_id=other", content=run_json())
        assert wrong.status_code == 400 and "not 'other'" in wrong.json()["detail"]["refused"][0]
        assert client.post(f"{IMPORT}?kind=run", content='{"ok": true}').status_code == 400
        assert client.post(f"{IMPORT}?kind=run", content=run_json("run-oom", ok=False)).json()["published"] == 2
    t = stage_of(board, "run-oom")
    assert t["stage"] == "failed" and "out of memory" in t["message"]
    assert list(bus.read_log())[0].type == lab_import.STARTED       # the settings still land for a failed run


def test_the_command_reads_run_json_first_so_the_steps_know_their_task(tmp_path, capsys):
    telemetry(tmp_path)
    (tmp_path / "run.json").write_text(run_json(), encoding="utf-8")
    args = ["--telemetry", str(tmp_path / "run" / "telemetry.jsonl"), "--run", str(tmp_path / "run.json"),
            "--data-dir", str(tmp_path / "data")]
    assert lab_import.main(args) == 0
    out = json.loads(capsys.readouterr().out)
    assert (out["run"]["published"], out["telemetry"]["published"]) == (2, 6)   # started + trained; 5 steps + progress
    assert lab_import.main(args) == 0
    again = json.loads(capsys.readouterr().out)
    assert (again["run"]["published"], again["telemetry"]["published"]) == (0, 0)
    board = LearningBoard()
    for e in EventBus(log_dir=tmp_path / "data" / "events").read_log():
        if isinstance(e, RunProgress):
            board.on_run(e)
    assert stage_of(board)["message"] == "step 5 of 5"
