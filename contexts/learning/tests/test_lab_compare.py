"""The Learning lab's Compare tab (L31): two runs side by side, from the live-loop records and the box's own log."""

import json
import shutil

from fastapi import FastAPI
from fastapi.testclient import TestClient

from contexts.learning.application.lab_compare import COMMITTED, RATIO_KEYS, compare_runs, load_loops
from contexts.learning.application.lab_view import LabView
from contexts.learning.lab_api import lab_router

REHEARSAL = COMMITTED / "live-loop-2026-10-11"
SOLO, DUO = "live-1011-040342-solo", "live-1011-040342-duo"


def rehearsal_only(tmp_path):
    """A loops folder holding just the committed rehearsal, so other fixtures added later cannot move the defaults."""
    shutil.copytree(REHEARSAL, tmp_path / "live-loop-2026-10-11")
    return [tmp_path]


def test_the_defaults_are_the_latest_solo_run_against_the_latest_pooled_one(tmp_path):
    out = compare_runs(load_loops(rehearsal_only(tmp_path)), None)
    assert out["asked"] == {"a": SOLO, "b": DUO}
    assert out["a"]["members"] == 1 and out["b"]["members"] == 2
    assert out["a"]["examples"] == 100 and out["b"]["examples"] == 200
    assert out["missing"] == []


def test_the_ratios_are_the_relays_own_duo_over_solo(tmp_path):
    """Same arithmetic as tooling/demo/live_loop.py: every ratio the relay wrote comes out identical here."""
    recorded = json.loads((REHEARSAL / "comparison.json").read_text(encoding="utf-8"))["duo_over_solo"]
    ratios = compare_runs(load_loops(rehearsal_only(tmp_path)), None)["ratios"]
    for key, value in recorded.items():
        assert ratios[key] == value, key
    assert ratios["members"] == 2.0
    assert set(ratios) == set(RATIO_KEYS)


def test_each_run_carries_its_stage_numbers_with_the_idle_floor_beside(tmp_path):
    a = compare_runs(load_loops(rehearsal_only(tmp_path)), None)["a"]
    assert a["labeling"]["joules"] == 3723.0 and a["labeling"]["idle_j"] == 329.7
    assert a["training"]["j_per_step"] == 340.9 and a["packaging"]["joules"] == 3333.9
    assert a["seconds"] == {"labeling": 32.1, "waiting": 0.0, "training": 53.9, "packaging": 71.1, "total": 157.0}
    assert a["source"]["kind"] == "loop" and a["source"]["ref"].endswith("live-loop-2026-10-11/comparison.json")
    assert a["idle"]["pod-c"]["calibrated_w"] == 10.27


def test_the_three_traces_share_one_time_axis_from_each_runs_start(tmp_path):
    out = compare_runs(load_loops(rehearsal_only(tmp_path)), None)
    for run in (out["a"], out["b"]):
        assert set(run["traces"]) == {"pod-c", "pod-d", "box"}
        total = run["seconds"]["total"]
        for trace in run["traces"].values():
            ts = [t for t, _ in trace["points"]]
            assert ts, trace["name"]
            assert -5.5 <= ts[0] <= 5 and ts[-1] <= total + 5.5
            assert ts == sorted(ts)


class OneRunView:
    """The two things lab_compare asks of LabView, for a run the box logged with no power sampler."""

    run_ids = {"l16-solo"}

    def runs(self):
        return [{"run_id": "l16-solo", "started": "2026-10-11T03:00:00Z"}]

    def story(self, run_id):
        n = lambda v: {"value": v}  # noqa: E731
        return {"run_id": run_id, "status": "not deployed", "wall_s": n(None),
                "dataset": {"examples": n(None), "train": n(None), "held_out": n(None)},
                "training": {"steps": n(18), "wall_s": n(785.0), "loss_first": n(4.1), "loss_last": n(0.6), "node": "pod-b"},
                "power": {"energy_j": n(None)}}


def test_a_run_no_sampler_measured_shows_its_counts_and_says_why_it_has_no_joules(tmp_path):
    out = compare_runs(load_loops(rehearsal_only(tmp_path)), OneRunView(), "l16-solo", DUO)
    a = out["a"]
    assert a["source"]["kind"] == "log" and a["training"]["steps"] == 18 and a["seconds"]["training"] == 785.0
    assert a["training"]["joules"] is None and a["labeling"] is None and "no power sampler" in a["reason"]
    assert out["ratios"]["training_j"] is None  # never a ratio against a missing figure
    assert out["ratios"]["training_s"] == round(68.8 / 785.0, 3)
    assert [c["run_id"] for c in out["runs"]][-1] == "l16-solo"


def test_the_route_answers_before_the_run_route_and_names_an_unknown_run(tmp_path):
    app = FastAPI()
    app.include_router(lab_router(LabView(), sample_dir=None, metrics_file=None, loop_dirs=rehearsal_only(tmp_path)))
    c = TestClient(app)
    body = c.get("/api/learning/lab/compare").json()
    assert body["asked"] == {"a": SOLO, "b": DUO} and body["ratios"]["examples"] == 2.0
    body = c.get("/api/learning/lab/compare", params={"a": SOLO, "b": "no-such-run"}).json()
    assert body["b"] is None and body["ratios"] is None and body["missing"] == ["no-such-run"]


def test_no_records_and_no_runs_is_an_empty_answer_not_an_error(tmp_path):
    out = compare_runs(load_loops([tmp_path / "nothing"]), None)
    assert out["a"] is None and out["b"] is None and out["runs"] == [] and out["ratios"] is None
