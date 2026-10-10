import asyncio
from datetime import UTC, datetime, timedelta

from fastapi import FastAPI
from fastapi.testclient import TestClient

from bus import EventBus
from contexts.learning.api import learning_router
from contexts.learning.application.board import LearningBoard
from contracts.learning import DatasetExported, ModelDeployed, RunEvaluated, RunProgress, TaskCreated

T0 = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)


def story():
    return [
        TaskCreated(ts=T0, task="desk-objects", instruction="Name the small things on the desk."),
        DatasetExported(ts=T0 + timedelta(minutes=1), task="desk-objects", dataset_id="ds1", examples=40, action_id="a1"),
        RunProgress(ts=T0 + timedelta(minutes=5), task="desk-objects", run_id="r1", stage="training", progress=0.4, loss=1.2),
        RunProgress(ts=T0 + timedelta(minutes=9), task="desk-objects", run_id="r1", stage="eval", progress=0.9),
        RunEvaluated(ts=T0 + timedelta(minutes=10), task="desk-objects", run_id="r1", scores={"current": 0.41, "candidate": 0.78},
                     decision="deploy", model_version="atlas-v2"),
        ModelDeployed(ts=T0 + timedelta(minutes=12), model_version="atlas-v2", previous_version="atlas-base"),
    ]


def test_the_log_folds_into_one_entry_per_task_with_scores_and_the_deployment():
    async def run():
        bus = EventBus(persist=False)
        board = LearningBoard()
        board.subscribe(bus)
        events = story()
        await bus.publish(events[0])
        assert board.snapshot()["tasks"][0]["stage"] == "waiting for approval"
        await bus.publish_all(events[1:4])
        mid = board.snapshot()["tasks"][0]
        assert (mid["stage"], mid["progress"], mid["examples"]) == ("eval", 0.9, 40) and mid["loss"] is None
        await bus.publish_all(events[4:])
        return board.snapshot()

    snap = asyncio.run(run())
    (task,) = snap["tasks"]
    assert task["stage"] == "done" and task["deployed"] is True and task["decision"] == "deploy"
    assert task["scores"] == {"current": 0.41, "candidate": 0.78} and task["model_version"] == "atlas-v2"
    assert snap["deployments"][0]["model_version"] == "atlas-v2" and snap["deployments"][0]["previous_version"] == "atlas-base"


def test_a_rejected_candidate_is_not_marked_deployed_and_newest_task_comes_first():
    board = LearningBoard()
    board.on_created(TaskCreated(ts=T0, task="old", instruction="x"))
    board.on_created(TaskCreated(ts=T0 + timedelta(hours=1), task="new", instruction="y"))
    board.on_run(RunEvaluated(ts=T0 + timedelta(hours=2), task="old", run_id="r", scores={"current": 0.5, "candidate": 0.4},
                              decision="reject", model_version="atlas-v9"))
    snap = board.snapshot()
    assert [t["task"] for t in snap["tasks"]] == ["old", "new"]  # most recently changed first
    assert not snap["tasks"][0].get("deployed") and snap["tasks"][0]["decision"] == "reject"


def test_the_endpoint_serves_the_snapshot():
    board = LearningBoard()
    for e in story():
        {"learning.task.created": board.on_created, "learning.dataset.exported": board.on_exported,
         "learning.run.progress": board.on_run, "learning.run.evaluated": board.on_run,
         "learning.model.deployed": board.on_deployed}[e.type](e)
    app = FastAPI()
    app.include_router(learning_router(board))
    body = TestClient(app).get("/api/learning").json()
    assert body["tasks"][0]["task"] == "desk-objects" and body["tasks"][0]["scores"]["candidate"] == 0.78


def test_the_box_can_announce_a_deployed_model_through_the_api():
    published = []

    async def publish(event):
        published.append(event)

    board = LearningBoard()
    app = FastAPI()
    app.include_router(learning_router(board, publish=publish))
    r = TestClient(app).post("/api/learning/deployed", json={"model_version": "atlas-v2", "previous_version": "atlas-base"})
    assert r.status_code == 200 and (published[0].model_version, published[0].previous_version) == ("atlas-v2", "atlas-base")
    assert published[0].type == "learning.model.deployed"


def test_a_denied_or_expired_export_approval_ends_the_task_instead_of_waiting_for_ever():
    from contracts.agent import ActionProposed
    from contracts.workspace import ActionCompleted

    async def run(status):
        bus = EventBus(persist=False)
        board = LearningBoard()
        board.subscribe(bus)
        await bus.publish(TaskCreated(ts=T0, task="desk-objects", instruction="Name the small things."))
        await bus.publish(ActionProposed(ts=T0, action_id="a9", kind="export_training_data", target="desk-objects", reason="teach"))
        await bus.publish(ActionCompleted(ts=T0 + timedelta(minutes=1), action_id="a9", status=status))
        assert board.snapshot()["tasks"] == []  # hidden from the Teach screen by default (test cards)
        return board.snapshot(include_closed=True)["tasks"][0]["stage"]

    assert asyncio.run(run("denied")) == "denied"
    assert asyncio.run(run("expired")) == "expired"
