"""The experiment timeline: the lab's own records as cards, with nothing added. The ledger rows in these tests are
written by the lab's own verbs (lab.machine.experiments), so they are the shape the lab writes."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from contexts.learning.application.lab_bench import METRICS_FILE, read_rows
from contexts.learning.application.lab_experiments import pass_cards, session_cards, timeline
from contexts.learning.lab_bench_api import lab_bench_router
from lab.machine import experiments as lab_experiments

ROWS, _ = read_rows(METRICS_FILE)
E11 = {"id": "E11", "title": "Batch the teacher", "track": "kernels", "hypothesis": "Batching cuts joules per token.", "lever": "batch size",
       "arms": {"A": "batch 1", "B": "batch 16"}, "metric": "teacher.joules_per_token", "better": "lower", "citation": "kernels pass 1",
       "expected_delta": -0.8, "confidence": 0.8}


def by_id(payload):
    return {c["id"]: c for c in payload["cards"]}


def test_without_a_ledger_the_cards_are_the_seeded_queue_and_say_so(tmp_path):
    payload = timeline(ROWS, state_dir=tmp_path)
    cards = [c for c in payload["cards"] if c["kind"] == "experiment"]
    assert [c["id"] for c in cards] and {c["id"] for c in cards} == {s["id"] for s in lab_experiments.SEEDS}
    assert all(c["status"] == "queued" and c["ts"] is None and c["verdict"] is None for c in cards)
    assert all(c["source"] == {"kind": "file", "ref": "lab/machine/experiments.py SEEDS", "ids": []} for c in cards)
    assert payload["experiments"]["source"]["kind"] == "seeds" and "no lab ledger" in payload["experiments"]["source"]["note"]
    assert payload["summary"] == "5 experiments in the queue the lab's code starts from: 5 queued, 0 running, 0 with a verdict."
    assert payload["experiments"] | {"source": None} == {"total": 5, "queued": 5, "running": 0, "run": 0, "dropped": 0, "source": None}
    assert not (tmp_path / "ledger.jsonl").exists()  # reading never creates the lab's ledger


def test_ledger_rows_give_queued_running_won_lost_and_inconclusive_with_the_verdict_maths_own_words(tmp_path):
    lab_experiments.seed(state_dir=tmp_path)
    lab_experiments.add(E11, state_dir=tmp_path)
    lab_experiments.add({**E11, "id": "E12", "title": "Fewer queues", "expected_delta": -0.2, "confidence": 0.5}, state_dir=tmp_path)
    lab_experiments.add({**E11, "id": "E13", "title": "Hybrid kernel"}, state_dir=tmp_path)
    lab_experiments.add({**E11, "id": "E14", "title": "Speculative decoding"}, state_dir=tmp_path)
    lab_experiments.record_result("E11", {"A": [36.6, 36.5, 36.7, 36.6, 36.7], "B": [3.5, 3.6, 3.56, 3.55, 3.57]}, state_dir=tmp_path)
    lab_experiments.record_result("E12", {"A": [3.5, 3.6, 3.56, 3.55, 3.57], "B": [36.6, 36.5, 36.7, 36.6, 36.7]}, state_dir=tmp_path)
    lab_experiments.record_result("E13", {"A": [3.5, 3.6, 3.4, 3.7, 3.5], "B": [3.6, 3.5, 3.5, 3.6, 3.5]}, state_dir=tmp_path)
    lab_experiments.record_result("E14", {"A": [3.5, 3.6], "B": [3.1, 3.2]}, state_dir=tmp_path)  # under the floor of 5 runs
    lab_experiments.amend("E1", {"status": "dropped"}, reason="superseded", state_dir=tmp_path)
    payload = timeline(ROWS, state_dir=tmp_path)
    cards = by_id(payload)
    assert [cards[i]["status"] for i in ("E11", "E12", "E13", "E14", "E1", "E2")] == ["won", "lost", "inconclusive", "running", "dropped", "queued"]
    won = cards["E11"]
    assert won["verdict"] == "Arm B beat the control" and "p < 0.0001" in won["detail"] and won["source"]["ref"].startswith("lab ledger seq ")
    assert [(n["label"], n["value"], n["unit"]) for n in won["numbers"][-2:]] == [("Arm A: mean of 5 runs", 36.62, "J/token"), ("Arm B: mean of 5 runs", 3.556, "J/token")]
    assert (won["numbers"][0]["value"], won["numbers"][2]["value"]) == (-80.0, 0.64)  # the estimate written before the run, and its score
    assert cards["E14"]["verdict"] == "Too early to call" and all(n["low_confidence"] for n in cards["E14"]["numbers"][-2:])
    assert payload["experiments"] | {"source": None} == {"total": 9, "queued": 4, "running": 1, "run": 3, "dropped": 1, "source": None}
    assert "9 experiments in the lab ledger: 4 queued, 1 running, 3 with a verdict." == payload["summary"]
    dated = [c for c in payload["cards"] if c["ts"]]
    assert [str(c["ts"])[:10] for c in dated] == sorted((str(c["ts"])[:10] for c in dated), reverse=True)  # newest first


def test_refinement_passes_and_benchmark_sessions_come_from_the_committed_records():
    passes = pass_cards()
    assert {c["track"] for c in passes} == {"telemetry", "learning", "plumbing", "kernels", "all tracks"} and len(passes) >= 10
    for c in passes:
        assert c["status"] in ("open", "converged") and c["day_only"] and c["source"]["ref"].startswith("lab/refine/")
        assert f"{c['ts']}\"" in c["source"]["ref"] and [n["unit"] for n in c["numbers"]] == ["changes", "findings", "questions"]
    (session,) = session_cards(ROWS)
    assert session["id"] == "bench-20261010T215122Z-2c65" and session["status"] == "measured" and session["node"] == "pod-a"
    assert session["lever"] == "batch size 1, 4, 8, 16, 64" and [n["value"] for n in session["numbers"]] == [30, 30]
    assert session["source"]["ref"] == "lab/refine/metrics.jsonl:67-96" and "3 spawn(s)" in session["verdict"]


def test_the_endpoint_serves_the_timeline(tmp_path):
    app = FastAPI()
    app.include_router(lab_bench_router(state_dir=tmp_path))
    with TestClient(app) as client:
        body = client.get("/api/learning/lab/experiments").json()
        assert body["experiments"]["total"] == 5 and body["passes"] >= 10 and body["sessions"] == 1
        assert all(c["source"]["ref"] for c in body["cards"])
