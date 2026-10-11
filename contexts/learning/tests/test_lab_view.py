"""The Learning lab's read model: a run's story is assembled from the log, says where each number came from, and
says what is missing instead of inventing it."""

import asyncio
import json
from datetime import UTC, datetime, timedelta

from fastapi.encoders import jsonable_encoder

from bus import EventBus
from contexts.learning.application.board import LearningBoard
from contexts.learning.application.lab_view import LabView, collect_numbers, fold
from contracts.agent import ActionProposed
from contracts.events import Event
from contracts.learning import DatasetExample, DatasetExported, ModelDeployed, RunEvaluated, RunProgress, TaskCreated
from contracts.telemetry import GpuPower, JoulesPerAnswer, RequestWindow, TrainStep
from contracts.workspace import ActionCompleted

T0 = datetime(2026, 10, 10, 14, 0, tzinfo=UTC)


def at(minutes: float = 0, seconds: float = 0) -> datetime:
    return T0 + timedelta(minutes=minutes, seconds=seconds)


def todays_events() -> list[Event]:
    """What the contracts let a run say today: stages with a loss, the scores and decision, the swap."""
    return [
        TaskCreated(ts=at(0), task="desk-objects", instruction="List the objects on the desk."),
        ActionProposed(ts=at(0), action_id="a1", kind="export_training_data", target="desk-objects", reason="Send blurred keyframes"),
        ActionCompleted(ts=at(1), action_id="a1", status="done", result="exported 3 examples as desk-objects-1"),
        DatasetExported(ts=at(1), task="desk-objects", dataset_id="desk-objects-1", examples=3, action_id="a1"),
        RunProgress(ts=at(5), task="desk-objects", run_id="r1", stage="training", progress=0.0, loss=2.0),
        RunProgress(ts=at(7), task="desk-objects", run_id="r1", stage="training", progress=0.5, loss=1.1),
        RunProgress(ts=at(9), task="desk-objects", run_id="r1", stage="training", progress=1.0, loss=0.6),
        RunProgress(ts=at(10), task="desk-objects", run_id="r1", stage="eval", progress=0.5),
        RunEvaluated(ts=at(12), task="desk-objects", run_id="r1", scores={"current": 0.41, "candidate": 0.78},
                     decision="deploy", model_version="atlas-v2"),
        ModelDeployed(ts=at(14), model_version="atlas-v2", previous_version="atlas-base"),
    ]


def extra(type_: str, ts: datetime, **fields) -> Event:
    return Event(type=type_, ts=ts, **fields)


def item(i: int, group: str, before: bool, after: bool) -> dict:
    return {"id": f"q{i}", "group": group, "question": f"question {i}?", "expected": "last_seen(label=\"keys\")",
            "before": {"answer": "x", "correct": before}, "after": {"answer": "y", "correct": after}}


def full_run(run_id: str = "r1") -> list[Event]:
    """The same run with step telemetry and the four optional events."""
    steps = [TrainStep(ts=at(5, 10 * k), node="pod-b", run_id=run_id, step=k, loss=2.0 - 0.3 * k, step_ms=10_000.0,
                       socket_w=200.0 + 10 * k, energy_j=None) for k in (1, 2, 3, 4)]
    items = [item(1, "g1", False, True), item(2, "g1", True, True), item(3, "g2", True, False), item(4, "g2", False, False)]
    return [
        *todays_events()[:4],
        extra("learning.training.started", at(5), task="desk-objects", run_id=run_id, dataset_id="desk-objects-1", node="pod-b",
              base="Qwen/Qwen3.5-4B", steps=4, batch=4, train_examples=2, epochs=8, lora_rank=16),
        RunProgress(ts=at(5), task="desk-objects", run_id=run_id, stage="training", progress=0.0),
        *steps,
        extra("learning.package.built", at(8), task="desk-objects", run_id=run_id, model_version="atlas-v2", base="Qwen/Qwen3.5-4B",
              gguf="atlas-v2-Q4_K_M.gguf", bytes=2_700_000_000, sha256="ab" * 32, quantization="Q4_K_M"),
        extra("learning.eval.reported", at(12), task="desk-objects", run_id=run_id, n_items=4, n_groups=2,
              correct={"current": 2, "candidate": 2, "teacher": 4}, noise=0.02, noise_runs=[2, 2], p_value=0.5, alpha=0.05,
              test="sign-flip", items=items, low_confidence=True),
        RunEvaluated(ts=at(12), task="desk-objects", run_id=run_id, scores={"current": 0.5, "candidate": 0.5, "teacher": 1.0},
                     decision="deploy", model_version="atlas-v2"),
        extra("learning.swap.completed", at(14), model_version="atlas-v2", previous_version="atlas-base", started=at(13).isoformat(),
              swap_s=41.5, outcome="serving"),
        ModelDeployed(ts=at(14), model_version="atlas-v2", previous_version="atlas-base"),
    ]


def test_a_run_made_of_todays_events_shows_what_the_log_holds_and_names_what_is_missing():
    story = fold(LabView(), todays_events()).story("r1")
    assert story["status"] == "deployed" and story["sample"] is False
    assert story["task"]["instruction"] == "List the objects on the desk."
    assert story["eval"]["scores"]["candidate"]["value"] == 0.78
    assert story["eval"]["scores"]["candidate"]["source"]["ref"].startswith("learning.run.evaluated ")
    # the teacher's score is not in the event: null with a reason, never a guess
    assert story["eval"]["scores"]["teacher"]["value"] is None and "teacher" in story["eval"]["scores"]["teacher"]["reason"]
    # the loss curve falls back to the stage events, and says so
    assert [p["loss"] for p in story["training"]["points"]] == [2.0, 1.1, 0.6]
    assert "learning.run.progress" in story["training"]["points_source"]["ref"]
    assert story["training"]["wall_s"]["value"] == 300.0
    for missing in (story["training"]["steps"], story["training"]["samples_per_s"], story["power"]["energy_j"],
                    story["eval"]["n_items"], story["eval"]["p_value"], story["eval"]["noise"], story["package"]["bytes"],
                    story["deploy"]["swap_s"], story["box"]["before"]["j_per_answer"], story["dataset"]["train"]):
        assert missing["value"] is None and missing["reason"] and missing["source"] is None
    assert story["eval"]["items"] is None and "learning.eval.reported" in story["eval"]["items_reason"]
    assert story["eval"]["verdict"] == "Better by more than chance: deployed."
    assert any("cannot show why" in line for line in story["eval"]["verdict_detail"])
    assert story["dataset"]["approval"]["approved"] == at(1) and story["dataset"]["approval"]["approved_by"] is None
    assert {g["label"] for g in story["gaps"]} >= {"Hot-swap time", "p-value", "Energy used by training"}


def test_every_number_has_a_source_or_a_reason_never_neither():
    for events in (todays_events(), full_run()):
        numbers = collect_numbers(fold(LabView(), events).story("r1"))
        assert len(numbers) > 20
        for n in numbers:
            assert (n["value"] is None) == (n["source"] is None) == (n["reason"] is not None), n["label"]
            if n["source"]:
                assert n["source"]["kind"] in ("event", "file", "derived") and n["source"]["ref"]


def test_step_telemetry_gives_the_curve_the_speed_and_the_energy():
    story = fold(LabView(), full_run()).story("r1")
    t, p = story["training"], story["power"]
    assert [x["step"] for x in t["points"]] == [1, 2, 3, 4] and t["points"][0]["t"] == 10.0
    assert (t["steps"]["value"], t["of"]["value"], t["samples"]["value"], t["wall_s"]["value"]) == (4, 4, 16, 40.0)
    assert t["samples_per_s"]["value"] == 0.4
    # no energy counter on the steps: mean watts x step time, summed
    assert p["energy_j"]["value"] == (210 + 220 + 230 + 240) * 10.0
    assert p["j_per_sample"]["value"] == 9000 / 16 and p["peak_w"]["value"] == 240.0 and p["mean_w"]["value"] == 225.0
    assert p["energy_j"]["source"]["events"] == 4 and not p["energy_j"]["low_confidence"]


def test_a_step_without_a_power_reading_makes_the_energy_low_confidence():
    events = full_run()
    blind = next(e for e in events if isinstance(e, TrainStep) and e.step == 2)
    events[events.index(blind)] = blind.model_copy(update={"socket_w": None})
    p = fold(LabView(), events).story("r1")["power"]
    assert p["energy_j"]["value"] == (210 + 230 + 240) * 10.0 and p["energy_j"]["low_confidence"]


def test_pod_power_samples_across_the_window_are_integrated_instead_of_the_steps():
    samples = [GpuPower(ts=at(5, s), node="pod-b", source="fake", period_ms=10_000.0, socket_w=w, gfx_mhz=None, mem_mhz=None,
                        throttled=False) for s, w in ((0, 100.0), (20, 300.0), (40, 300.0))]
    p = fold(LabView(), [*full_run(), *samples]).story("r1")["power"]
    assert p["energy_j"]["value"] == 200.0 * 20 + 300.0 * 20  # trapezoid rule
    assert "telemetry.gpu.power" in p["energy_j"]["source"]["ref"] and len(p["points"]) == 3


def test_the_eval_report_gives_n_noise_p_and_every_question_with_regressions_counted():
    ev = fold(LabView(), full_run()).story("r1")["eval"]
    assert (ev["n_items"]["value"], ev["n_groups"]["value"], ev["noise"]["value"], ev["p_value"]["value"]) == (4, 2, 0.02, 0.5)
    assert ev["n_items"]["low_confidence"]  # the report said so; it is carried, not dropped
    assert [i["change"] for i in ev["items"]] == ["fixed", "still_right", "broke", "still_wrong"]
    counts = {k: v["value"] for k, v in ev["counts"].items()}
    assert counts == {"n": 4, "before_right": 2, "after_right": 2, "fixed": 1, "broke": 1, "still_right": 1, "still_wrong": 1}
    assert ev["groups"] == [{"group": "g1", "n": 2, "before": 1, "after": 2, "diff": 0.5},
                            {"group": "g2", "n": 2, "before": 1, "after": 0, "diff": -0.5}]
    assert any("at least 50" in line for line in ev["verdict_detail"])


def test_a_rejected_run_says_not_deployed_and_has_no_box_numbers():
    events = [e for e in todays_events() if not isinstance(e, ModelDeployed)]
    events[-1] = events[-1].model_copy(update={"decision": "reject"})
    story = fold(LabView(), events).story("r1")
    assert story["status"] == "not deployed" and story["eval"]["verdict"] == "Not proven: not deployed."
    assert story["deploy"]["deployed"] is None and "did not clear" in story["deploy"]["deployed_reason"]
    assert story["box"]["before"] is None and "not deployed" in story["box"]["reason"]


def test_a_run_cleared_by_the_gate_but_not_swapped_yet_is_gated_and_a_rollback_is_shown():
    events = todays_events()
    assert fold(LabView(), events[:-1]).story("r1")["status"] == "gated"
    assert fold(LabView(), events[:5]).story("r1")["status"] == "training"
    back = ModelDeployed(ts=at(30), model_version="atlas-base", previous_version="atlas-v2")
    story = fold(LabView(), [*events, back]).story("r1")
    assert story["status"] == "rolled back" and story["deploy"]["rolled_back"]["to"] == "atlas-base"
    assert "went back to the old model" in story["eval"]["verdict"]


def test_energy_and_speed_per_answer_are_split_at_the_swap_by_model():
    def answer(i: int, model: str, minute: float, joules: float) -> list[Event]:
        t0 = at(minute)
        req = RequestWindow(ts=t0 + timedelta(seconds=6), node="jetson", request_id=f"q{i}", started=t0, ended=t0 + timedelta(seconds=6),
                            tokens_in=900, tokens_out=50, target="student", first_token=t0 + timedelta(seconds=1), model=model)
        j = JoulesPerAnswer(ts=req.ts, node="jetson", request_id=f"q{i}", model=model, started=t0, ended=req.ended, clock="jetson",
                            joules=joules, node_energy_j=joules, idle_w=None, idle_ts=None, power_field="VDD_IN", boundary="module input",
                            accelerator="x", sampler="fake", period_ms=1000.0, samples=7, error=None, low_confidence=i == 1)
        return [req, j]

    events = [*answer(1, "atlas-base", -30, 80.0), *answer(2, "atlas-base", -20, 90.0), *full_run(), *answer(3, "atlas-v2", 20, 70.0)]
    box = fold(LabView(), events).story("r1")["box"]
    assert (box["before"]["j_per_answer"]["value"], box["before"]["answers"], box["before"]["version"]) == (85.0, 2, "atlas-base")
    assert box["before"]["j_per_answer"]["low_confidence"] and not box["after"]["j_per_answer"]["low_confidence"]
    assert box["after"]["j_per_answer"]["value"] == 70.0 and box["after"]["tok_s"]["value"] == 10.0


def test_example_rows_come_from_the_export_folder_without_pictures_or_held_out_rows(tmp_path):
    root = tmp_path / "desk-objects-1"
    (root / "images").mkdir(parents=True)
    rows = [DatasetExample(id=f"e{i}", image=f"images/{i:04d}.jpg", frame_ref=f"frames:cam-desk:{i}", ts=at(0), context=f"caption {i}")
            for i in range(3)]
    (root / "examples.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in rows))
    (root / "split.json").write_text(json.dumps({"dataset_id": "desk-objects-1", "train": ["e0", "e1"], "held_out": ["e2"]}))
    d = fold(LabView(exports_dir=tmp_path), full_run()).story("r1")["dataset"]
    assert [r["id"] for r in d["rows"]] == ["e0", "e1"] and d["rows"][0]["frame_ref"] == "frames:cam-desk:0"
    assert all("image" not in r for r in d["rows"])  # a frame_ref for the blurred keyframe; never a picture or its path
    assert (d["train"]["value"], d["held_out"]["value"]) == (2, 1) and d["train"]["source"]["ref"] == "desk-objects-1/split.json"
    # without the folder: counts from the events, rows null with the reason
    d = fold(LabView(), full_run()).story("r1")["dataset"]
    assert d["rows"] is None and d["rows_reason"] and (d["train"]["value"], d["held_out"]["value"]) == (2, 4)


def test_an_optional_event_with_the_wrong_shape_is_listed_not_guessed_at():
    bad = extra("learning.eval.reported", at(12), run_id="r1", n_items="many")
    story = fold(LabView(), [*todays_events(), bad]).story("r1")
    assert story["eval"]["n_items"]["value"] is None
    assert story["ignored_events"][0]["type"] == "learning.eval.reported" and "n_items" in story["ignored_events"][0]["problem"]


def test_replaying_the_log_gives_the_same_story_and_the_teach_board_is_untouched_by_the_new_events(tmp_path):
    async def live() -> tuple[dict, dict]:
        bus = EventBus(tmp_path / "events")
        view, board = LabView(), LearningBoard()
        view.subscribe(bus)
        board.subscribe(bus)
        await bus.publish_all(full_run())
        return view.story("r1"), board.snapshot()

    async def replayed() -> tuple[dict, list]:
        bus = EventBus(tmp_path / "events")
        view = LabView()
        view.subscribe(bus)
        await bus.replay_into()
        return view.story("r1"), view.runs()

    story, snapshot = asyncio.run(live())
    again, runs = asyncio.run(replayed())
    assert jsonable_encoder(again) == jsonable_encoder(story) and story["status"] == "deployed"
    assert [r["run_id"] for r in runs] == ["r1"] and runs[0]["status"] == "deployed"
    # the optional events sit outside the prefixes the Teach board folds, so it still reads the run as before
    assert snapshot["tasks"][0]["stage"] == "done" and snapshot["tasks"][0]["scores"]["candidate"] == 0.5
    assert len(snapshot["deployments"]) == 1


def test_runs_are_listed_newest_first_and_an_unknown_run_is_none():
    later = [e.model_copy(update={"run_id": "r2", "ts": e.ts + timedelta(hours=2)}) for e in todays_events()
             if isinstance(e, RunProgress)]
    view = fold(LabView(), [*todays_events(), *later])
    assert [r["run_id"] for r in view.runs()] == ["r2", "r1"] and view.runs()[0]["status"] == "training"
    assert view.story("nope") is None
