"""GET /runtime/stats assembles recorded numbers: the lab's metrics file and ledger, the telemetry projection and the
learning events. Every figure is present; a figure nothing has recorded is null with the reason why."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from bus import EventBus
from contexts.telemetry.application.projection import TelemetryProjection
from contexts.telemetry.infrastructure.lab_records import LabRecords
from contexts.telemetry.runtime_stats import build_runtime_stats
from contracts.events import Event
from contracts.learning import ModelDeployed, RunEvaluated
from contracts.telemetry import RequestWindow, TrainStep
from lab.machine import ledger

T0 = datetime(2026, 10, 10, 14, 0, tzinfo=UTC)
FIGURES = ["detector_fps", "student_tok_s", "caption_latency_s", "trainer_samples_per_s", "train_minutes",
           "teacher_tok_s"]


def at(s: float) -> datetime:
    return T0 + timedelta(seconds=s)


def metrics_row(metric: str, value: float, unit: str, node: str, ts: str, low: bool = False) -> str:
    return json.dumps({"metric": metric, "value": value, "unit": unit, "node": node, "conditions": {},
                       "source": "a write-up", "ts": ts, "low_confidence": low})


def write_metrics(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def ledger_row(metric: str, value: float, unit: str, node: str, ts: str, low: bool | None = None) -> dict:
    row = {"ts": ts, "actor": "bench", "tier": None, "task": None, "verb": "bench.run", "outcome": "ok",
           "metrics": {metric: value}, "refs": {}, "node": node, "units": {metric: unit}}
    if low is not None:
        row["low_confidence"] = low
    return row


def window(node: str, request_id: str, start_s: float, first_s: float | None, end_s: float, tokens_out: int,
           target: str = "student") -> RequestWindow:
    return RequestWindow(ts=at(end_s), node=node, request_id=request_id, started=at(start_s),
                         first_token=None if first_s is None else at(first_s), ended=at(end_s), tokens_in=100,
                         tokens_out=tokens_out, target=target)


def step(run_id: str, n: int, s: float, step_ms: float = 2000.0) -> TrainStep:
    """A step from a pod sampler that still says gpu-b, the old name the projection reads as pod-b."""
    return TrainStep(ts=at(s), node="gpu-b", run_id=run_id, step=n, loss=1.0, step_ms=step_ms, socket_w=250.0)


def evaluated(run_id: str, version: str, s: float, scores: dict, decision: str = "deploy",
              task: str = "desk-objects") -> RunEvaluated:
    return RunEvaluated(ts=at(s), task=task, run_id=run_id, scores=scores, decision=decision, model_version=version)


def snapshot(events: list[Event] = (), *, telemetry: bool = True, metrics: Path | None = None,
             state_dir: Path | None = None) -> dict:
    """The stats as the endpoint serves them, after `events` went through the real bus."""

    class Telemetry:  # the gateway's TelemetryService: only its projection is read
        projection = TelemetryProjection()

    async def run() -> dict:
        bus = EventBus(persist=False)
        service = Telemetry() if telemetry else None
        if service is not None:
            service.projection.subscribe(bus)
        stats = build_runtime_stats(bus, service, lab=LabRecords(metrics_path=metrics, state_dir=state_dir))
        await bus.publish_all(list(events))
        return await stats.snapshot()

    return asyncio.run(run())


def nothing(tmp_path: Path) -> dict:
    return {"metrics": tmp_path / "metrics.jsonl", "state_dir": tmp_path / "state"}  # neither file exists


def test_a_lab_number_comes_with_its_metrics_line_measurement_time_and_low_confidence_flag(tmp_path):
    metrics = write_metrics(tmp_path / "metrics.jsonl", [
        metrics_row("teacher.decode_tok_s", 6.6, "tok/s", "pod-a", "2026-10-10T10:22:13Z"),
        "",  # a blank line still counts as a line of the file
        metrics_row("teacher.decode_tok_s", 7.7, "tok/s", "pod-a", "2026-10-10T10:22:13Z"),  # same time, later line
        metrics_row("teacher.decode_tok_s", 9.0, "tok/s", "pod-a", "2026-10-10T09:00:00Z"),  # later line, older time
        metrics_row("trainer.samples_per_s", 0.51, "samples/s", "pod-b", "2026-10-10T10:23:23Z", low=True),
    ])
    stats = snapshot(metrics=metrics, state_dir=tmp_path / "state")
    assert stats["teacher_tok_s"] == {"value": 7.7, "unit": "tok/s", "node": "pod-a", "ts": "2026-10-10T10:22:13Z",
                                      "source": "metrics.jsonl:3", "low_confidence": False, "reason": None}
    assert stats["trainer_samples_per_s"] == {"value": 0.51, "unit": "samples/s", "node": "pod-b",
                                              "ts": "2026-10-10T10:23:23Z", "source": "metrics.jsonl:5",
                                              "low_confidence": True, "reason": None}


def test_a_reference_row_on_another_chip_is_never_the_teachers_figure(tmp_path):
    a100 = json.loads(metrics_row("teacher.decode_tok_s", 64.956, "tok/s", "a100", "2026-10-11T01:52:01Z", low=True))
    a100["conditions"] = {"hub": "colab", "other_chip": True}  # R57's A100 reference: newer, but not the lab's teacher
    metrics = write_metrics(tmp_path / "metrics.jsonl", [
        metrics_row("teacher.decode_tok_s", 2.82, "tok/s", "pod-c", "2026-10-10T19:59:08Z"), json.dumps(a100)])
    stats = snapshot(metrics=metrics, state_dir=tmp_path / "state")
    assert (stats["teacher_tok_s"]["value"], stats["teacher_tok_s"]["source"]) == (2.82, "metrics.jsonl:1")
    found, _ = LabRecords(metrics_path=metrics, state_dir=tmp_path / "state").find([("teacher.decode_tok_s", "a100")])
    assert [f.value for f in found[("teacher.decode_tok_s", "a100")]] == [64.956]  # named, it still answers


def test_a_jetson_figure_reads_only_jetson_rows(tmp_path):
    metrics = write_metrics(tmp_path / "metrics.jsonl", [
        metrics_row("student.decode_tok_s", 15.7, "tok/s", "jetson", "2026-10-10T10:00:00Z"),
        metrics_row("student.decode_tok_s", 40.0, "tok/s", "pod-b", "2026-10-10T11:00:00Z"),  # on the lab card
        metrics_row("detector.fps", 41.0, "frames/s", "jetson", "2026-10-10T11:05:00Z"),
        metrics_row("student.caption_s", 2.1, "s", "jetson", "2026-10-10T11:06:00Z"),
    ])
    stats = snapshot(metrics=metrics, state_dir=tmp_path / "state")
    assert (stats["student_tok_s"]["value"], stats["student_tok_s"]["source"]) == (15.7, "metrics.jsonl:1")
    assert (stats["detector_fps"]["value"], stats["detector_fps"]["unit"], stats["detector_fps"]["source"]) == (
        41.0, "frames/s", "metrics.jsonl:3")
    assert (stats["caption_latency_s"]["value"], stats["caption_latency_s"]["source"]) == (2.1, "metrics.jsonl:4")


def test_a_newer_ledger_measurement_wins_and_cites_its_seq(tmp_path):
    metrics = write_metrics(tmp_path / "metrics.jsonl", [
        metrics_row("teacher.decode_tok_s", 7.7, "tok/s", "pod-a", "2026-10-10T10:22:13Z")])
    state = tmp_path / "state"
    ledger.append(ledger_row("teacher.decode_tok_s", 9.4, "tok/s", "pod-c", "2026-10-10T18:00:00+00:00", low=True),
                  state_dir=state)
    stats = snapshot(metrics=metrics, state_dir=state)
    assert stats["teacher_tok_s"] == {"value": 9.4, "unit": "tok/s", "node": "pod-c", "ts": "2026-10-10T18:00:00Z",
                                      "source": "ledger seq 1", "low_confidence": True, "reason": None}


def test_on_a_tie_the_metrics_file_row_is_cited_rather_than_its_ledger_copy(tmp_path):
    metrics = write_metrics(tmp_path / "metrics.jsonl", [
        metrics_row("teacher.decode_tok_s", 7.7, "tok/s", "pod-a", "2026-10-10T10:22:13Z")])
    state = tmp_path / "state"
    ledger.append(ledger_row("teacher.decode_tok_s", 7.7, "tok/s", "pod-a", "2026-10-10T10:22:13+00:00"),
                  state_dir=state)  # what `python -m lab ledger backfill` makes of the same row
    assert snapshot(metrics=metrics, state_dir=state)["teacher_tok_s"]["source"] == "metrics.jsonl:1"


def test_tok_s_from_request_windows_is_output_after_the_first_token_over_the_time_since_it(tmp_path):
    stats = snapshot([
        window("jetson", "q1", 0, 0.5, 5.5, tokens_out=51),  # 50 tokens after the first in 5 s
        window("jetson", "q2", 10, 10.5, 11.5, tokens_out=21),  # 20 tokens in 1 s
        window("jetson", "q3", 20, None, 22, tokens_out=40),  # no first-token time: prefill and decode mixed, left out
        window("jetson", "q4", 30, 30.2, 30.2, tokens_out=1),  # one token: no decode to time, left out
        window("gpu-a", "t1", 0, 0.55, 10.55, tokens_out=78, target="teacher"),
    ], **nothing(tmp_path))
    assert stats["student_tok_s"] == {"value": 70 / 6, "unit": "tok/s", "node": "jetson",
                                      "ts": "2026-10-10T14:00:11.500000Z",
                                      "source": "telemetry.request: 2 windows from node jetson",
                                      "low_confidence": False, "reason": None}
    assert (stats["teacher_tok_s"]["value"], stats["teacher_tok_s"]["node"], stats["teacher_tok_s"]["source"]) == (
        7.7, "pod-a", "telemetry.request: 1 window from node pod-a")


def test_the_newest_measurement_wins_between_telemetry_and_the_lab_records(tmp_path):
    metrics = write_metrics(tmp_path / "metrics.jsonl", [
        metrics_row("teacher.decode_tok_s", 7.7, "tok/s", "pod-a", "2026-10-10T10:22:13Z"),  # before T0
        metrics_row("student.decode_tok_s", 15.7, "tok/s", "jetson", "2026-10-10T15:00:00Z"),  # after T0
    ])
    stats = snapshot([window("jetson", "q1", 0, 0.5, 5.5, tokens_out=51),
                      window("gpu-a", "t1", 0, 0.5, 10.5, tokens_out=91, target="teacher")],
                     metrics=metrics, state_dir=tmp_path / "state")
    assert (stats["teacher_tok_s"]["value"], stats["teacher_tok_s"]["source"]) == (
        9.0, "telemetry.request: 1 window from node pod-a")
    assert (stats["student_tok_s"]["value"], stats["student_tok_s"]["source"]) == (15.7, "metrics.jsonl:2")


def test_train_minutes_is_the_newest_evaluated_run_from_its_first_step_to_its_last_plus_one_step(tmp_path):
    stats = snapshot([
        step("r1", 1, 0), step("r1", 2, 2), step("r1", 3, 600),  # 600 s from first to last step, plus one 2 s step
        evaluated("r1", "atlas-v2", 700, {"current": 0.4, "candidate": 0.8}),
        step("r2", 1, 1000), step("r2", 2, 1060),  # a later run with no evaluation yet: may still be training
    ], **nothing(tmp_path))
    assert stats["train_minutes"] == {"value": 602 / 60, "unit": "min", "node": "pod-b", "ts": "2026-10-10T14:10:00Z",
                                      "source": "telemetry.train.step: run r1, 3 steps", "low_confidence": False,
                                      "reason": None}


def test_a_run_not_evaluated_yet_is_served_as_low_confidence_when_it_is_the_only_one(tmp_path):
    stats = snapshot([step("r2", 1, 1000, step_ms=3000), step("r2", 2, 1060, step_ms=3000)], **nothing(tmp_path))
    assert stats["train_minutes"] == {"value": 63 / 60, "unit": "min", "node": "pod-b", "ts": "2026-10-10T14:17:40Z",
                                      "source": "telemetry.train.step: run r2, 2 steps (no learning.run.evaluated "
                                                "for it yet, so it may still be training)",
                                      "low_confidence": True, "reason": None}


def test_with_nothing_recorded_every_figure_is_present_null_with_its_reason(tmp_path):
    stats = snapshot(telemetry=False, **nothing(tmp_path))
    assert list(stats) == [*FIGURES, "eval_scores"]
    off = "telemetry is off on this box (ATLAS_TELEMETRY_SOURCE is not set)"
    assert {k: (stats[k]["value"], stats[k]["ts"], stats[k]["source"]) for k in FIGURES} == {
        k: (None, None, None) for k in FIGURES}
    assert {k: stats[k]["unit"] for k in FIGURES} == {
        "detector_fps": "frames/s", "student_tok_s": "tok/s", "caption_latency_s": "s",
        "trainer_samples_per_s": "samples/s", "train_minutes": "min", "teacher_tok_s": "tok/s"}
    assert stats["detector_fps"]["reason"] == (
        "not recorded yet: no detector.fps row for node jetson in the lab metrics file or the lab ledger; the gateway "
        "keeps no detector timing (GET /capture/status counts detections without timing them)")
    assert stats["student_tok_s"]["reason"] == (
        "not recorded yet: no student.decode_tok_s row for node jetson in the lab metrics file or the lab ledger; "
        f"{off}")
    assert stats["caption_latency_s"]["reason"] == (
        "not recorded yet: no student.caption_s row for node jetson in the lab metrics file or the lab ledger; "
        "telemetry.request windows do not say which requests were captions")
    assert stats["trainer_samples_per_s"]["reason"] == (
        "not recorded yet: no trainer.samples_per_s row in the lab metrics file or the lab ledger; "
        "telemetry.train.step carries no sample count")
    assert stats["train_minutes"]["reason"] == (
        f"not recorded yet: no trainer.run_minutes row in the lab metrics file or the lab ledger; {off}")
    assert stats["teacher_tok_s"]["reason"] == (
        f"not recorded yet: no teacher.decode_tok_s row in the lab metrics file or the lab ledger; {off}")
    assert stats["eval_scores"] == {"models": [], "reason": "no learning.model.deployed event in the event log yet"}


def test_with_telemetry_on_but_empty_the_reason_names_the_events_it_looked_for(tmp_path):
    stats = snapshot(**nothing(tmp_path))
    assert stats["student_tok_s"]["reason"] == (
        "not recorded yet: no student.decode_tok_s row for node jetson in the lab metrics file or the lab ledger; "
        "no telemetry.request window from node jetson with a first-token time and two or more output tokens")
    assert stats["train_minutes"]["reason"] == (
        "not recorded yet: no trainer.run_minutes row in the lab metrics file or the lab ledger; "
        "no telemetry.train.step from node pod-b")


def test_an_unreadable_metrics_file_is_named_in_the_reason_instead_of_failing_the_endpoint(tmp_path):
    metrics = write_metrics(tmp_path / "metrics.jsonl", ["{not json"])
    stats = snapshot(telemetry=False, metrics=metrics, state_dir=tmp_path / "state")
    reason = stats["teacher_tok_s"]["reason"]
    assert stats["teacher_tok_s"]["value"] is None
    assert reason.startswith("not recorded yet: no teacher.decode_tok_s row in the lab metrics file or the lab ledger; "
                             "telemetry is off on this box (ATLAS_TELEMETRY_SOURCE is not set); "
                             "the lab metrics file could not be read: MetricsFormatError: ")
    assert "is not valid JSON" in reason


def test_eval_scores_list_each_deployed_model_with_its_before_and_after_rebuilt_from_the_log(tmp_path):
    log = tmp_path / "events"
    v2 = evaluated("r1", "atlas-v2-desk", 100, {"current": 0.41, "candidate": 0.78, "teacher": 0.86})
    v2_again = evaluated("r1b", "atlas-v2-desk", 150, {"current": 0.41, "candidate": 0.8, "teacher": 0.86})
    v3 = evaluated("r2", "atlas-v3-desk", 300, {"current": 0.8, "candidate": 0.7}, decision="reject")
    deployed_v2 = ModelDeployed(ts=at(200), model_version="atlas-v2-desk", previous_version="atlas-base")
    rollback = ModelDeployed(ts=at(400), model_version="atlas-base", previous_version="atlas-v2-desk")

    async def run() -> dict:
        await EventBus(log).publish_all([v2, v2_again, deployed_v2, v3, rollback])  # yesterday's run, on disk
        bus = EventBus(log)  # a restart: the stats are rebuilt by replaying the log
        stats = build_runtime_stats(bus, None, lab=LabRecords(**{"metrics_path": tmp_path / "m.jsonl",
                                                                 "state_dir": tmp_path / "state"}))
        await bus.replay_into()
        return await stats.snapshot()

    scores = asyncio.run(run())["eval_scores"]
    assert scores == {"reason": None, "models": [
        {"model_version": "atlas-base", "deployed_ts": "2026-10-10T14:06:40Z", "previous_version": "atlas-v2-desk",
         "task": None, "run_id": None, "decision": None, "before": None, "after": None, "teacher": None, "ts": None,
         "source": None, "reason": "no learning.run.evaluated for atlas-base in the event log", "low_confidence": True},
        {"model_version": "atlas-v2-desk", "deployed_ts": "2026-10-10T14:03:20Z", "previous_version": "atlas-base",
         "task": "desk-objects", "run_id": "r1b", "decision": "deploy", "before": 0.41, "after": 0.8, "teacher": 0.86,
         "ts": "2026-10-10T14:02:30Z", "source": f"learning.run.evaluated {v2_again.event_id}", "reason": None, "low_confidence": False},
    ]}


def test_an_evaluation_without_a_before_or_after_score_says_which_is_missing(tmp_path):
    stats = snapshot([evaluated("r1", "atlas-v2", 100, {"candidate": 0.8}),
                      ModelDeployed(ts=at(200), model_version="atlas-v2")], **nothing(tmp_path))
    (model,) = stats["eval_scores"]["models"]
    assert (model["before"], model["after"], model["teacher"], model["reason"]) == (
        None, 0.8, None, "the evaluation carries no 'current' score")
    assert model["low_confidence"] is True and model["ts"] and model["source"]


def test_the_pod_figures_read_the_telemetry_of_the_nodes_main_names_pod_a_and_pod_b(tmp_path):
    stats = snapshot([
        window("pod-a", "t1", 0, 0.5, 10.5, tokens_out=91, target="teacher"),
        TrainStep(ts=at(0), node="pod-b", run_id="r1", step=1, loss=1.0, step_ms=2000.0, socket_w=250.0),
        TrainStep(ts=at(60), node="pod-b", run_id="r1", step=2, loss=1.0, step_ms=2000.0, socket_w=250.0),
        evaluated("r1", "atlas-v2", 120, {"current": 0.4, "candidate": 0.8}),
    ], **nothing(tmp_path))
    assert (stats["teacher_tok_s"]["value"], stats["teacher_tok_s"]["node"]) == (9.0, "pod-a")
    assert (stats["train_minutes"]["value"], stats["train_minutes"]["node"]) == (62 / 60, "pod-b")
