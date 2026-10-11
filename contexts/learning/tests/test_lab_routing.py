"""The routing what-if: it runs the real hook on recorded numbers, is repeatable, and never moves locked work.

The recorded numbers are the frozen rows of contexts/learning/tests/lab_rows.py, not the lab's committed file: the
lab keeps appending to that and re-measuring, and what these tests check is the model, not this week's figures. The
committed file is read by the one test named for it, which pins none of its values."""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import contexts.telemetry.application.routing as policies
import runtime.routing as hook
from contexts.learning.application import lab_routing
from contexts.learning.application.lab_bench import METRICS_FILE, read_rows
from contexts.learning.application.lab_routing import KINDS, MODEL_STATEMENT, NO_MEASUREMENT, SimError, interp, recorded_inputs, simulate
from contexts.learning.lab_bench_api import lab_bench_router
from contexts.learning.tests.lab_rows import POD_A_CURVE, POD_A_SESSION, pod_a_sweep, recorded, write

ROWS = [{**r, "_line": n} for n, r in enumerate(recorded(), 1)]
NAMES = [k["kind"] for k in KINDS]


def test_the_teachers_curve_is_the_teacher_pods_own_sweep_even_when_the_other_pod_ran_a_newer_one():
    cheaper = {name: (unit, tuple(v / 2 for v in values)) for name, (unit, values) in POD_A_CURVE.items()}
    later = pod_a_sweep(session="pod-b-night", node="pod-b", family="teacher", ts="2026-10-11T04:13:45Z")
    for r in later:  # a different curve, so a mix-up would show in the values
        r["value"] = cheaper[r["metric"].split(".", 1)[1]][1][[1, 4, 8, 16, 64].index(r["conditions"]["batch_size"])]
    rows = ROWS + [{**r, "_line": len(ROWS) + n} for n, r in enumerate(later, 1)]
    teacher = recorded_inputs(rows)["teacher"]
    assert (teacher["node"], teacher["session"]) == (policies.TEACHER_NODE, POD_A_SESSION) == ("pod-a", POD_A_SESSION)
    assert [p["value"] for p in teacher["j_per_token"]] == list(POD_A_CURVE["joules_per_token"][1])
    assert all(p["node"] == "pod-a" and p["line"] <= len(ROWS) for name in ("j_per_token", "tok_s", "ttft_s") for p in teacher[name])
    only_b = [r for r in rows if not r["metric"].startswith("teacher_batching.")]  # a sweep on the trainer's pod is not the teacher's
    assert recorded_inputs(only_b)["teacher"] is None and "batching sweep on pod-a" in recorded_inputs(only_b)["missing"][0]


def test_the_same_inputs_give_the_same_output_and_another_seed_gives_another_stream():
    a, b = simulate({"preset": "study", "seed": 3}, ROWS), simulate({"preset": "study", "seed": 3}, ROWS)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert simulate({"preset": "study", "seed": 4}, ROWS)["requests"]["kind"] != a["requests"]["kind"]
    assert a["requests"]["n"] == 24 * 5 == len(a["requests"]["t"]) and a["requests"]["t"] == sorted(a["requests"]["t"])


@pytest.mark.parametrize("preset", ["quiet", "study", "burst"])
@pytest.mark.parametrize("basis", ["request", "decode"])
def test_each_total_is_the_sum_of_its_requests_and_locked_kinds_never_leave_the_box(preset, basis):
    out = simulate({"preset": preset, "student_basis": basis}, ROWS)
    locked = {k["kind"] for k in out["kinds"] if k["locked"]}
    assert locked == {"chat", "caption"} == {x["kind"] for x in out["privacy"]["locked"]}
    for run in out["policies"].values():
        energy = wait = 0.0
        for kind, target in zip(out["requests"]["kind"], run["targets"], strict=True):
            row = run["by_kind"][NAMES[kind]]
            assert {"p": "student", "s": "student", "t": "teacher"}[target] == row["target"]
            assert (target == "p") == (NAMES[kind] in locked)  # the privacy line, and only it, locks a request
            energy += row["joules"]
            wait += row["latency_s"]
        assert run["energy_j"] == pytest.approx(energy, abs=0.01)
        assert run["mean_latency_s"] == pytest.approx(wait / out["requests"]["n"], abs=0.01)
        assert run["to_teacher"] == run["targets"].count("t") and run["to_teacher"] + run["to_student"] == out["requests"]["n"]
        assert run["locked"] == sum(NAMES[k] in locked for k in out["requests"]["kind"])
    assert out["compare"]["saved_j"] == pytest.approx(out["policies"]["capability"]["energy_j"] - out["policies"]["energy"]["energy_j"], abs=0.01)
    assert out["compare"]["saved_j"] >= -0.01  # the energy policy only moves work that its snapshot says is cheaper


def test_locked_work_stays_on_the_box_even_when_the_teacher_is_priced_at_nothing():
    line = lab_routing.privacy_line()
    assert [x["kind"] for x in line["locked"]] == ["chat", "caption"] and line["policy_name"] == hook.PRIVACY
    assert line["lab_sources"] == sorted(hook.LAB_SOURCES)


def test_every_decision_is_made_by_the_real_hook_and_the_real_policies(monkeypatch):
    seen = {"route": 0, "capability": 0, "energy": 0, "snapshot": 0}
    real_route, real_snapshot = hook.route, policies.routing_snapshot

    def counted(name, fn):
        def wrapper(*args, **kwargs):
            seen[name] += 1
            return fn(*args, **kwargs)
        return wrapper

    monkeypatch.setattr(lab_routing, "route", counted("route", real_route))
    monkeypatch.setattr(lab_routing, "routing_snapshot", counted("snapshot", real_snapshot))
    monkeypatch.setitem(lab_routing.POLICIES, "capability", counted("capability", policies.capability))
    monkeypatch.setitem(lab_routing.POLICIES, "energy", counted("energy", policies.energy))
    out = simulate({"preset": "burst"}, ROWS)
    assert all(v > 0 for v in seen.values()), seen
    assert lab_routing.route is not real_route and hook.route is real_route  # the simulator imports the hook; it has no copy
    assert lab_routing.POLICIES is policies.POLICIES
    # and the result is the hook's own: its policy names and its reasons, word for word
    for name, run in out["policies"].items():
        snap = run["snapshot"]
        assert set(snap) == {"nodes"} and set(snap["nodes"]) <= {policies.STUDENT_NODE, policies.TEACHER_NODE}
        for kind in KINDS:
            direct = real_route(lab_routing.as_request(kind), snap, policies.POLICIES[name])
            assert (run["by_kind"][kind["kind"]]["target"], run["by_kind"][kind["kind"]]["decided_by"], run["by_kind"][kind["kind"]]["reason"]) == \
                (direct.target, direct.policy, direct.reason)


def test_the_teachers_cost_is_read_off_the_recorded_curve_and_never_invented_beyond_it():
    out = simulate({"preset": "burst"}, ROWS)
    curve = {p["x"]: p["value"] for p in out["inputs"]["teacher"]["j_per_token"]}
    assert all("lab/refine/metrics.jsonl:" in p["source"]["ref"] for p in out["inputs"]["teacher"]["j_per_token"])
    pts = out["inputs"]["teacher"]["j_per_token"]
    assert interp(pts, 1) == curve[1] and interp(pts, 64) == curve[64] and interp(pts, 500) == curve[64] and interp(pts, 0.2) == curve[1]
    assert curve[8] > interp(pts, 11) > curve[16]
    recorded = [p for p in out["break_even"]["curve"] if p["recorded"]]
    assert [(p["batch"], p["teacher_j_per_token"]) for p in recorded] == [(b, round(v, 4)) for b, v in sorted(curve.items())]
    for run in out["policies"].values():
        assert run["teacher_batch"] is None or 1 <= run["teacher_batch"] <= 64


def test_the_two_policies_differ_only_where_the_recorded_numbers_let_them():
    # whole-request basis: the teacher's best recorded figure (batch 64) is under the student's, so a busy teacher wins eval work
    busy = simulate({"preset": "burst", "student_basis": "request"}, ROWS)
    assert busy["policies"]["capability"]["by_kind"]["eval"]["target"] == "student"
    assert busy["policies"]["energy"]["by_kind"]["eval"]["target"] == "teacher"
    assert busy["compare"]["moved"] == busy["kinds"][NAMES.index("eval")]["n"] > 0 and busy["compare"]["saved_j"] > 0
    assert busy["compare"]["sentence"].startswith("Energy-aware routing sends ") and "% less" in busy["compare"]["sentence"]
    assert 16 < busy["break_even"]["batch"] < 64 and busy["break_even"]["rpm_to_teacher"] > 0
    # decode-only basis: the box is cheaper per token at every recorded batch size, so nothing moves and it says why
    never = simulate({"preset": "burst", "student_basis": "decode"}, ROWS)
    assert never["compare"]["moved"] == 0 and never["break_even"]["batch"] is None and "never wins" in never["break_even"]["reason"]
    assert "same choices" in never["compare"]["sentence"]
    # a quiet teacher answers one at a time and costs the most per token: label work still has to go to it
    quiet = simulate({"preset": "quiet", "mix": {"label": 10}}, ROWS)
    assert quiet["policies"]["energy"]["by_kind"]["label"]["target"] == "teacher" and quiet["policies"]["energy"]["teacher_batch"] < 2


def test_it_is_labelled_a_simulation_until_closed_loop_rows_exist_and_then_shows_the_measurement_above_it():
    # the committed file holds T11's closed loop since 2026-10-11 (100 requests per policy, three metrics): it is shown
    # above the model, which is then a what-if
    real = simulate(None, ROWS)
    assert real["label"] == "what-if" and real["measured_reason"] is None and real["model_statement"] == MODEL_STATEMENT
    assert {m["metric"] for m in real["measured"]["rows"]} == {"label.seconds", "learning.label_agreement", "telemetry.joules_per_answer"}
    joules = next(m for m in real["measured"]["rows"] if m["metric"] == "telemetry.joules_per_answer")
    assert (joules["capability"]["value"], joules["energy"]["value"], joules["change_pct"], joules["low_confidence"]) == (235.2903, 247.4741, 5.2, True)
    # without closed-loop rows it is a simulation and says so
    before = [r for r in ROWS if "policy" not in r["conditions"]]
    out = simulate(None, before)
    assert (out["label"], out["measured"], out["measured_reason"], out["model_statement"]) == ("simulation", None, NO_MEASUREMENT, MODEL_STATEMENT)
    base = {"unit": "J", "node": "pod-a", "source": "T11 closed loop", "ts": "2026-10-11T09:00:00Z", "low_confidence": False}
    extra = [{**base, "metric": "routing.joules_per_answer", "value": 180.0, "conditions": {"policy": "capability", "requests": 120}, "_line": 201},
             {**base, "metric": "routing.joules_per_answer", "value": 150.0, "conditions": {"policy": "energy", "requests": 120}, "_line": 202},
             {**base, "metric": "routing.other", "value": 1.0, "conditions": {"policy": "energy", "requests": 40}, "_line": 203}]  # under 100 requests
    assert simulate(None, before + extra[:1])["measured"] is None  # one side only is not a comparison
    measured = simulate(None, before + extra)
    assert measured["label"] == "what-if" and measured["measured_reason"] is None
    (found,) = measured["measured"]["rows"]
    assert (found["capability"]["value"], found["energy"]["value"], found["change_pct"]) == (180.0, 150.0, -16.7)
    assert found["energy"]["source"]["ref"] == "lab/refine/metrics.jsonl:202"


def test_without_the_recorded_rows_it_refuses_to_model_and_bad_inputs_are_named():
    out = simulate(None, [r for r in ROWS if not r["metric"].startswith("teacher_batching.")])
    assert out["available"] is False and "batching sweep on pod-a" in out["reason"] and "policies" not in out
    for bad in ({"rpm": 0}, {"preset": "nope"}, {"mix": {"chat": -1}}, {"mix": {k: 0 for k in NAMES}}, {"tokens": {"eval": 2}},
                {"policy": "cheapest"}, {"student_basis": "x"}, {"seed": 1.5}, {"minutes": 99}):
        with pytest.raises(SimError):
            simulate(bad, ROWS)
    edited = simulate({"preset": "burst", "tokens": {"eval": 200}, "rpm": 30}, ROWS)
    assert edited["params"]["preset"] is None and edited["params"]["tokens"]["eval"] == 200 and edited["params"]["rpm"] == 30


def test_the_committed_file_gives_a_simulation_or_says_what_it_lacks_and_cites_only_rows_it_holds():
    rows, _ = read_rows(METRICS_FILE)
    by_line = {r["_line"]: r for r in rows}
    out = simulate({"preset": "burst"}, rows)
    if not out["available"]:
        assert out["reason"] and "policies" not in out
        return
    teacher = out["inputs"]["teacher"]
    assert teacher["node"] == policies.TEACHER_NODE and len(teacher["batches"]) >= 2
    for name in ("j_per_token", "tok_s", "ttft_s"):
        for p in teacher[name]:
            row = by_line[p["line"]]
            assert (row["node"], row["value"]) == (policies.TEACHER_NODE, p["value"]) and not p["other_chip"]  # never another chip's curve
    assert out["compare"]["saved_j"] >= -0.01 and set(out["policies"]) == {"capability", "energy"}


def test_the_endpoint_runs_the_simulation_and_answers_400_with_the_reason(tmp_path):
    app = FastAPI()
    app.include_router(lab_bench_router(metrics_file=write(tmp_path, recorded())))
    with TestClient(app) as client:
        ok = client.post("/api/learning/lab/routing/simulate", json={"preset": "quiet"})
        assert ok.status_code == 200 and ok.json()["params"]["preset"] == "quiet" and ok.json()["available"] is True
        assert client.post("/api/learning/lab/routing/simulate").json()["params"]["preset"] == "burst"
        bad = client.post("/api/learning/lab/routing/simulate", json={"rpm": 9999})
        assert bad.status_code == 400 and "rpm" in bad.json()["detail"]
