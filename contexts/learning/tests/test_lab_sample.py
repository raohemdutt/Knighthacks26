"""The sample run is labelled, kept apart from real runs, and its made-up numbers agree with each other."""

import importlib.util
import json
from fractions import Fraction
from itertools import product
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from contexts.learning.application.lab_sample import SAMPLE_DIR, SampleError, load_sample, read_sample_events
from contexts.learning.application.lab_view import LabView, collect_numbers, fold
from contexts.learning.lab_api import lab_router
from contexts.learning.tests.test_lab_view import todays_events

DEPLOYED, NOT_PROVEN = "sample-run-2-deployed", "sample-run-1-not-proven"
RAW = [json.loads(line) for line in (SAMPLE_DIR / "events.jsonl").read_text().splitlines()]


def of(type_: str, run_id: str | None = None) -> list[dict]:
    return [e for e in RAW if e["type"] == type_ and (run_id is None or e.get("run_id") == run_id)]


def every_dict(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from every_dict(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from every_dict(v)


def test_the_committed_files_are_exactly_what_the_generator_writes():
    spec = importlib.util.spec_from_file_location("make_sample_run", SAMPLE_DIR.parent / "make_sample_run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, files = module.build()
    for name, text in files.items():
        assert (SAMPLE_DIR / name).read_text(encoding="utf-8") == text, f"{name}: run python fixtures/learning/make_sample_run.py"


def test_every_line_and_file_says_it_is_a_sample_and_a_line_that_does_not_is_refused(tmp_path):
    assert RAW and all(e["sample"] is True and e["event_id"].startswith("sample-") for e in RAW)
    data = SAMPLE_DIR / "dataset" / "tool-calling-sample"
    assert json.loads((data / "manifest.json").read_text())["sample"] is True
    assert json.loads((data / "split.json").read_text())["sample"] is True
    assert all(json.loads(line)["sample"] is True for line in (data / "examples.jsonl").read_text().splitlines())
    real = {k: v for k, v in RAW[0].items() if k != "sample"}
    (tmp_path / "events.jsonl").write_text(json.dumps(real) + "\n")
    with pytest.raises(SampleError):
        read_sample_events(tmp_path / "events.jsonl")


def test_everything_the_sample_view_returns_is_marked_sample():
    view = load_sample()
    for payload in (view.runs(), view.story(DEPLOYED), view.story(NOT_PROVEN)):
        dicts = list(every_dict(payload))
        assert len(dicts) > 10 and all(d.get("sample") is True for d in dicts)


@pytest.mark.parametrize("run_id", [DEPLOYED, NOT_PROVEN])
def test_scores_are_counts_over_n_and_the_p_value_is_the_exact_group_test(run_id):
    report, evaluated = of("learning.eval.reported", run_id)[0], of("learning.run.evaluated", run_id)[0]
    items, n = report["items"], report["n_items"]
    assert n == len(items) == 52 >= 50 and report["n_groups"] == len({i["group"] for i in items}) == 8
    right = {"current": sum(i["before"]["correct"] for i in items), "candidate": sum(i["after"]["correct"] for i in items)}
    assert {k: report["correct"][k] for k in right} == right
    for key in ("current", "candidate", "teacher"):
        assert evaluated["scores"][key] == round(report["correct"][key] / n, 4)
    # a right answer is the expected call, a wrong one is something else
    assert all((i[side]["answer"] == i["expected"]) == i[side]["correct"] for i in items for side in ("before", "after"))
    assert report["noise"] == round((max(report["noise_runs"]) - min(report["noise_runs"])) / n, 4)
    assert report["noise_runs"][0] == right["current"]

    groups: dict[str, list] = {}
    for i in items:
        groups.setdefault(i["group"], []).append(i)
    diffs = [Fraction(sum(i["after"]["correct"] for i in g) - sum(i["before"]["correct"] for i in g), len(g)) for g in groups.values()]
    seen = abs(sum(diffs))
    p = sum(abs(sum(s * d for s, d in zip(signs, diffs))) >= seen for signs in product((1, -1), repeat=len(diffs))) / 2 ** len(diffs)
    assert report["p_value"] == p
    gain = evaluated["scores"]["candidate"] - evaluated["scores"]["current"]
    assert evaluated["decision"] == ("deploy" if n >= 50 and gain > report["noise"] and p < report["alpha"] else "reject")


def test_one_run_is_deployed_and_the_other_is_not_and_only_the_deployed_one_reaches_the_box():
    assert of("learning.run.evaluated", DEPLOYED)[0]["decision"] == "deploy"
    assert of("learning.run.evaluated", NOT_PROVEN)[0]["decision"] == "reject"
    assert [e["model_version"] for e in of("learning.model.deployed")] == ["sample-atlas-v2-tool-calling"]
    view = load_sample()
    lost, won = view.story(NOT_PROVEN), view.story(DEPLOYED)
    assert (lost["status"], lost["eval"]["verdict"]) == ("not deployed", "Not proven: not deployed.")
    assert lost["deploy"]["deployed"] is None and lost["box"]["before"] is None
    assert (won["status"], won["eval"]["verdict"]) == ("deployed", "Better by more than chance: deployed.")
    assert won["eval"]["counts"]["broke"]["value"] == 3  # the deployed run has regressions too, and they are in the data
    # the gate scored the packaged model: packaging ends before the eval does
    assert won["marks"]["packaged"] < won["marks"]["evaluated"] < won["marks"]["deployed"]


@pytest.mark.parametrize("run_id", [DEPLOYED, NOT_PROVEN])
def test_energy_is_the_sum_of_watts_times_time_and_the_story_adds_up(run_id):
    steps, cfg = of("telemetry.train.step", run_id), of("learning.training.started", run_id)[0]
    assert len(steps) == cfg["steps"] == 60 == cfg["train_examples"] * cfg["epochs"] // cfg["batch"]
    assert all(s["energy_j"] == round(s["socket_w"] * s["step_ms"] / 1000, 1) for s in steps)
    story = load_sample().story(run_id)
    t, p = story["training"], story["power"]
    energy = round(sum(s["energy_j"] for s in steps), 1)
    wall = sum(s["step_ms"] for s in steps) / 1000
    assert p["energy_j"]["value"] == energy and t["wall_s"]["value"] == pytest.approx(wall, abs=0.01)
    assert t["samples"]["value"] == 240 and t["samples_per_s"]["value"] == round(240 / wall, 3)
    assert p["j_per_sample"]["value"] == round(energy / 240, 1) and p["mean_w"]["value"] == round(energy / wall, 1)
    assert [x["loss"] for x in t["points"]] == [s["loss"] for s in steps] and t["points"][0]["loss"] > t["points"][-1]["loss"]


def test_the_split_keeps_groups_on_one_side_and_matches_the_counts():
    data = SAMPLE_DIR / "dataset" / "tool-calling-sample"
    split, rows = json.loads((data / "split.json").read_text()), [json.loads(x) for x in (data / "examples.jsonl").read_text().splitlines()]
    group = {r["id"]: r["frame_ref"] for r in rows}
    assert sorted(split["train"] + split["held_out"]) == sorted(group) and not set(split["train"]) & set(split["held_out"])
    assert not {group[i] for i in split["train"]} & {group[i] for i in split["held_out"]}
    assert of("learning.dataset.exported")[0]["examples"] == len(rows) == 112
    d = load_sample().story(DEPLOYED)["dataset"]
    assert (d["examples"]["value"], d["train"]["value"], d["held_out"]["value"], d["held_out_groups"]["value"]) == (112, 60, 52, 8)
    assert {i["id"] for i in of("learning.eval.reported", DEPLOYED)[0]["items"]} == set(split["held_out"])
    assert d["rows"] and all(r["side"] == "train" and r["target"] for r in d["rows"])  # held-out questions are never shown as taught


def test_box_answers_are_plausible_and_every_sample_number_has_a_source():
    for j in of("telemetry.joules_per_answer"):
        seconds = (__import__("datetime").datetime.fromisoformat(j["ended"]) - __import__("datetime").datetime.fromisoformat(j["started"])).total_seconds()
        assert 12 < j["joules"] / seconds < 15 and j["low_confidence"] and j["joules"] == j["node_energy_j"]
    story = load_sample().story(DEPLOYED)
    assert story["gaps"] == []  # the sample run is complete: nothing is null
    assert all(n["source"] and n["sample"] for n in collect_numbers(story))
    assert story["box"]["before"]["answers"] == story["box"]["after"]["answers"] == 8
    assert story["box"]["after"]["j_per_answer"]["low_confidence"]


def client(view: LabView, **kw) -> TestClient:
    app = FastAPI()
    app.include_router(lab_router(view, **kw))
    return TestClient(app)


def test_with_no_real_run_the_api_serves_the_sample_and_says_why():
    c = client(LabView(), current_model=lambda: {"version": "atlas-base", "source": "the runtime"})
    body = c.get("/api/learning/lab").json()
    assert body["sample"] is True and body["real_runs"] == 0 and "no training run" in body["sample_reason"]
    assert [r["run_id"] for r in body["runs"]] == [DEPLOYED, NOT_PROVEN] and all(r["sample"] for r in body["runs"])
    assert body["current_model"] == {"version": "atlas-base", "source": "the runtime"}  # the box's own model, not the sample's
    story = c.get(f"/api/learning/lab/{DEPLOYED}").json()
    assert story["sample"] is True and story["status"] == "deployed" and len(story["sources"]) > 30
    assert c.get("/api/learning/lab/nope").status_code == 404


def test_a_real_run_is_never_mixed_with_the_sample():
    c = client(fold(LabView(), todays_events()))
    body = c.get("/api/learning/lab").json()
    assert body["sample"] is False and body["sample_reason"] is None and [r["run_id"] for r in body["runs"]] == ["r1"]
    assert not any(d.get("sample") for d in every_dict(body["runs"]))
    story = c.get("/api/learning/lab/r1").json()
    assert story["sample"] is False and not any(d.get("sample") for d in every_dict({k: v for k, v in story.items() if k != "measured"}))
    assert c.get(f"/api/learning/lab/{DEPLOYED}").status_code == 404  # the sample is not reachable unless asked for
    asked = c.get("/api/learning/lab?sample=1").json()
    assert asked["sample"] is True and asked["real_runs"] == 1 and "r1" not in [r["run_id"] for r in asked["runs"]]
    assert c.get(f"/api/learning/lab/{DEPLOYED}?sample=1").json()["sample"] is True
    assert c.get("/api/learning/lab/r1?sample=1").status_code == 404
    # the model the box serves comes from the real log, whichever view is shown
    assert asked["current_model"]["version"] == "atlas-v2"


def test_measured_numbers_come_from_the_labs_metrics_file_with_their_line_and_are_not_sample():
    body = client(LabView()).get("/api/learning/lab").json()
    speed = next(m for m in body["measured"] if m["metric"] == "trainer.samples_per_s")
    assert speed["value"] == 0.51 and speed["sample"] is False and speed["source"]["ref"].startswith("lab/refine/metrics.jsonl:")
    line = int(speed["source"]["ref"].rsplit(":", 1)[1])
    assert json.loads(Path("lab/refine/metrics.jsonl").read_text().splitlines()[line - 1])["metric"] == "trainer.samples_per_s"
    assert client(LabView(), metrics_file=None, sample_dir=None).get("/api/learning/lab").json() == {
        "runs": [], "sample": False, "sample_reason": None, "real_runs": 0, "current_model": None, "measured": [],
        "suggested_events": body["suggested_events"]}


def test_the_gateway_serves_the_lab_from_the_log(tmp_path, monkeypatch):
    import asyncio

    from bus import EventBus

    async def run() -> dict:
        bus = EventBus(tmp_path / "events")
        await bus.publish_all(todays_events())
        fresh = EventBus(tmp_path / "events")
        view = LabView(exports_dir=tmp_path / "exports", models_dir=tmp_path / "models")
        view.subscribe(fresh)
        await fresh.replay_into()
        return client(view).get("/api/learning/lab/r1").json()

    story = asyncio.run(run())
    assert story["status"] == "deployed" and story["sample"] is False and story["eval"]["scores"]["current"]["value"] == 0.41
