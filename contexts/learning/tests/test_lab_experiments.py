"""The experiment timeline: the lab's own records as cards, with nothing added. The ledger rows in these tests are
written by the lab's own verbs (lab.machine.experiments), so they are the shape the lab writes.

The metrics rows are a small file each test writes itself (contexts/learning/tests/lab_rows.py). The lab's committed
records are read only by the tests named `..._committed_...`, which pin no count, node or value: the lab keeps adding
sessions, passes and seeded experiments."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from contexts.learning.application.lab_bench import METRICS_FILE, read_rows
from contexts.learning.application.lab_experiments import line_runs, pass_cards, session_cards, timeline
from contexts.learning.lab_bench_api import lab_bench_router
from contexts.learning.tests.lab_rows import A100_SESSION, POD_A_SESSION, check_sessions, pod_a_sweep, recorded, row, write
from lab.machine import experiments as lab_experiments
from lab.machine import refine_formats

SEEDED = len(lab_experiments.SEEDS)  # the queue the lab's code starts from; the lab adds to it
E11 = {"id": "E11", "title": "Batch the teacher", "track": "kernels", "hypothesis": "Batching cuts joules per token.", "lever": "batch size",
       "arms": {"A": "batch 1", "B": "batch 16"}, "metric": "teacher.joules_per_token", "better": "lower", "citation": "kernels pass 1",
       "expected_delta": -0.8, "confidence": 0.8}


def by_id(payload):
    return {c["id"]: c for c in payload["cards"]}


def rows_of(tmp_path, rows=None):
    return read_rows(write(tmp_path, recorded() if rows is None else rows))[0]


def test_without_a_ledger_the_cards_are_the_seeded_queue_and_say_so(tmp_path):
    payload = timeline(rows_of(tmp_path), state_dir=tmp_path)
    cards = [c for c in payload["cards"] if c["kind"] == "experiment"]
    assert [c["id"] for c in cards] and {c["id"] for c in cards} == {s["id"] for s in lab_experiments.SEEDS}
    assert all(c["status"] == "queued" and c["ts"] is None and c["verdict"] is None for c in cards)
    assert all(c["source"] == {"kind": "file", "ref": "lab/machine/experiments.py SEEDS", "ids": []} for c in cards)
    assert payload["experiments"]["source"]["kind"] == "seeds" and "no lab ledger" in payload["experiments"]["source"]["note"]
    assert payload["summary"] == f"{SEEDED} experiments in the queue the lab's code starts from: {SEEDED} queued, 0 running, 0 with a verdict."
    assert payload["experiments"] | {"source": None} == {"total": SEEDED, "queued": SEEDED, "running": 0, "run": 0, "dropped": 0, "source": None}
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
    payload = timeline(rows_of(tmp_path), state_dir=tmp_path)
    cards = by_id(payload)
    assert [cards[i]["status"] for i in ("E11", "E12", "E13", "E14", "E1", "E2")] == ["won", "lost", "inconclusive", "running", "dropped", "queued"]
    won = cards["E11"]
    assert won["verdict"] == "Arm B beat the control" and "p < 0.0001" in won["detail"] and won["source"]["ref"].startswith("lab ledger seq ")
    assert [(n["label"], n["value"], n["unit"]) for n in won["numbers"][-2:]] == [("Arm A: mean of 5 runs", 36.62, "J/token"), ("Arm B: mean of 5 runs", 3.556, "J/token")]
    assert (won["numbers"][0]["value"], won["numbers"][2]["value"]) == (-80.0, 0.64)  # the estimate written before the run, and its score
    assert cards["E14"]["verdict"] == "Too early to call" and all(n["low_confidence"] for n in cards["E14"]["numbers"][-2:])
    total, queued = SEEDED + 4, SEEDED - 1  # the seeds and our four; one seed dropped, our four run or running
    assert payload["experiments"] | {"source": None} == {"total": total, "queued": queued, "running": 1, "run": 3, "dropped": 1, "source": None}
    assert f"{total} experiments in the lab ledger: {queued} queued, 1 running, 3 with a verdict." == payload["summary"]
    dated = [c for c in payload["cards"] if c["ts"]]
    assert [str(c["ts"])[:10] for c in dated] == sorted((str(c["ts"])[:10] for c in dated), reverse=True)  # newest first


def test_refinement_passes_come_from_the_committed_logs():
    passes = pass_cards()
    assert passes and {c["track"] for c in passes} <= {*refine_formats.TRACKS, "all tracks"}
    assert len({c["id"] for c in passes}) == len(passes)
    for c in passes:
        assert c["status"] in ("open", "converged") and c["day_only"] and c["source"]["ref"].startswith("lab/refine/")
        assert f"{c['ts']}\"" in c["source"]["ref"] and [n["unit"] for n in c["numbers"]] == ["changes", "findings", "questions"]


def test_benchmark_sessions_are_one_card_each_with_their_own_rows(tmp_path):
    rows = rows_of(tmp_path)
    session, a100 = session_cards(rows)  # the batching sweep on the teacher's pod, then the A100 reference on Colab
    check_sessions([session, a100], rows)
    assert (a100["id"], a100["node"]) == (A100_SESSION, "a100") and [n["value"] for n in a100["numbers"]] == [30, 30]
    assert session["id"] == POD_A_SESSION and session["status"] == "measured" and session["node"] == "pod-a"
    assert session["title"] == "Benchmark: teacher_batching on pod-a"
    assert session["lever"] == "batch size 1, 4, 8, 16, 64" and [n["value"] for n in session["numbers"]] == [30, 30]
    assert session["source"]["ref"] == "lab/refine/metrics.jsonl:12-41" and "3 spawn(s)" in session["verdict"]
    assert a100["source"]["ref"] == "lab/refine/metrics.jsonl:43-72"  # the idle row at 73 carries no session


def test_a_session_card_cites_only_its_own_lines_counts_a_row_written_twice_once_and_gives_every_reason(tmp_path):
    base = recorded()
    later = pod_a_sweep(session="pod-b-night", node="pod-b", family="teacher", ts="2026-10-11T04:13:45Z")
    for r in later[:6]:  # the batch-1 rows stand on fewer spawns, and say so
        r["conditions"]["low_confidence_reason"] = "2 spawn(s), under the 5 that verdict() needs to call a difference"
    copies = [{**r, "conditions": {k: v for k, v in r["conditions"].items() if k != "sampler"}} for r in base[11:41]]  # the pod-a sweep, again
    loose = row("teacher.decode_tok_s", 2.9, node="pod-c", batch_size=1, session="pod-b-night")  # the same session, another node
    rows = rows_of(tmp_path, [*base, *later, *copies, loose])
    cards = {c["id"]: c for c in session_cards(rows)}
    check_sessions(list(cards.values()), rows)
    assert list(cards) == [POD_A_SESSION, A100_SESSION, "pod-b-night"]
    twice = cards[POD_A_SESSION]
    # lines 42 to 103 belong to other sessions: the card names its two runs of lines, not one span across them
    assert twice["source"]["ref"] == "lab/refine/metrics.jsonl:12-41, 104-133"
    assert [(n["label"], n["value"]) for n in twice["numbers"]] == [
        ("Rows recorded", 30), ("Rows marked low confidence", 30), ("Rows the file holds twice, counted once here", 30)]
    night = cards["pod-b-night"]
    assert night["source"]["ref"] == "lab/refine/metrics.jsonl:74-103, 134" and [n["value"] for n in night["numbers"]] == [31, 30]
    assert night["node"] is None and night["title"] == "Benchmark: teacher on pod-b, pod-c"  # two nodes: neither is claimed
    assert night["verdict"] == ("Measured, no verdict: 3 spawn(s), under the 5 that verdict() needs to call a difference (24 rows); "
                                "2 spawn(s), under the 5 that verdict() needs to call a difference (6 rows)")
    assert line_runs([3, 1, 2, 7, 9, 10]) == "1-3, 7, 9-10" and line_runs([5]) == "5"


def test_the_committed_sessions_are_one_card_each_whatever_the_lab_has_added():
    rows, _ = read_rows(METRICS_FILE)
    check_sessions(session_cards(rows), rows)


def test_the_endpoint_serves_the_timeline(tmp_path):
    app = FastAPI()
    app.include_router(lab_bench_router(metrics_file=write(tmp_path, recorded()), state_dir=tmp_path))
    with TestClient(app) as client:
        body = client.get("/api/learning/lab/experiments").json()
        assert body["experiments"]["total"] == SEEDED and body["sessions"] == 2
        assert body["passes"] == len(pass_cards()) == sum(c["kind"] in ("pass", "merge") for c in body["cards"])
        assert all(c["source"]["ref"] for c in body["cards"])


def test_the_endpoint_serves_the_committed_records_by_default(tmp_path):
    app = FastAPI()
    app.include_router(lab_bench_router(state_dir=tmp_path))
    rows, _ = read_rows(METRICS_FILE)
    with TestClient(app) as client:
        body = client.get("/api/learning/lab/experiments").json()
    kinds = [c["kind"] for c in body["cards"]]
    check_sessions([c for c in body["cards"] if c["kind"] == "bench"], rows)
    assert body["sessions"] == kinds.count("bench") and body["passes"] == kinds.count("pass") + kinds.count("merge")
    assert body["experiments"]["total"] == kinds.count("experiment") == SEEDED and len(kinds) == len(body["cards"])
    assert sum(body["counts"].values()) == len(body["cards"]) and all(c["source"]["ref"] for c in body["cards"])
