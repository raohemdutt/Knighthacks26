"""Pushing the lab's ledger to the box: the file the lab's own verbs wrote is kept whole under the data dir, the
Experiments view reads it there, and a body that is not a clean ledger changes nothing."""

import json
import os

import pytest
from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

from bus import EventBus
from contexts.learning import lab_import
from contexts.learning.application.lab_experiments import imported_state_dir, ledger_source, timeline
from contexts.learning.application.lab_view import LabView
from contexts.learning.lab_bench_api import lab_bench_router
from contexts.learning.lab_import import ImportRefused, check_ledger, lab_import_router, store_ledger
from contexts.learning.tests.test_lab_experiments import E11
from lab.machine import experiments as lab_experiments
from lab.machine import ledger, record
from tests.test_no_secrets import expand

URL = "/api/learning/lab/import?kind=ledger"
PRIVATE = "the pod account is hemdutt-lab on node-7.internal"  # free text a row may carry; never echoed back


@pytest.fixture(autouse=True)
def no_other_ledger(tmp_path, monkeypatch):
    """Only the ledgers a test makes exist: no LAB_STATE_DIR, and an empty folder for the checkout's lab/state."""
    monkeypatch.delenv(ledger.ENV_STATE_DIR, raising=False)
    monkeypatch.setattr(ledger, "DEFAULT_STATE_DIR", tmp_path / "checkout-state")


def lab_ledger(folder, extra_ids=()) -> bytes:
    """A ledger written by the lab's own verbs: the seeds, E11 won, E12 lost, E14 too early, E1 dropped, one note row."""
    lab_experiments.seed(state_dir=folder)
    for eid in ("E11", "E12", "E14", *extra_ids):
        lab_experiments.add({**E11, "id": eid}, state_dir=folder)
    lab_experiments.record_result("E11", {"A": [36.6, 36.5, 36.7, 36.6, 36.7], "B": [3.5, 3.6, 3.56, 3.55, 3.57]}, state_dir=folder)
    lab_experiments.record_result("E12", {"A": [3.5, 3.6, 3.56, 3.55, 3.57], "B": [36.6, 36.5, 36.7, 36.6, 36.7]}, state_dir=folder)
    lab_experiments.record_result("E14", {"A": [3.5, 3.6], "B": [3.1, 3.2]}, state_dir=folder)
    lab_experiments.amend("E1", {"status": "dropped"}, reason="superseded", state_dir=folder)
    record.record("bench", "ok", {"teacher.decode_tok_s": 41.5}, state_dir=folder, node="pod-a", note=PRIVATE)
    return (folder / "ledger.jsonl").read_bytes()


def app_with(data_dir, token=None):
    def guard(authorization: str | None = Header(default=None)) -> None:
        if token and authorization != f"Bearer {token}":
            raise HTTPException(401, "token needed")

    bus = EventBus(log_dir=data_dir / "events")
    app = FastAPI()
    app.include_router(lab_import_router(LabView(), bus.publish, guard=guard, data_dir=data_dir))
    app.include_router(lab_bench_router(imported_dir=imported_state_dir(data_dir)))
    return app


def test_a_ledger_the_labs_verbs_wrote_imports_and_the_experiments_show_its_verdicts(tmp_path):
    body, data = lab_ledger(tmp_path / "lab"), tmp_path / "data"
    rows = len(body.splitlines())
    with TestClient(app_with(data)) as client:
        before = client.get("/api/learning/lab/experiments").json()["experiments"]
        assert before["source"]["kind"] == "seeds" and before["run"] == 0
        first = client.post(URL, content=body)
        assert first.status_code == 200, first.text
        answer = first.json()
        assert answer | {"received_at": None} == {
            "kind": "ledger", "changed": True, "rows": rows, "experiments": 8, "bytes": len(body), "received_at": None,
            "verdicts": {"a_ahead": 1, "b_ahead": 1, "too_early": 1}, "stored": "lab_state/ledger.jsonl"}
        stored = data / "lab_state" / "ledger.jsonl"
        assert stored.read_bytes() == body and os.listdir(stored.parent) == ["ledger.jsonl"]  # whole, and no temp file left
        shown = client.get("/api/learning/lab/experiments").json()
        cards = {c["id"]: c for c in shown["cards"]}
        assert [cards[i]["status"] for i in ("E11", "E12", "E14", "E1", "E2")] == ["won", "lost", "running", "dropped", "queued"]
        assert cards["E11"]["verdict"] == "Arm B beat the control" and cards["E11"]["source"]["ref"].startswith("lab ledger seq ")
        source = shown["experiments"]["source"]
        assert (source["kind"], source["from"], source["rows"], source["ref"]) == ("ledger", "imported", rows, "ledger.jsonl")
        assert source["received_at"] == answer["received_at"] and answer["received_at"].endswith("Z")
        assert PRIVATE not in json.dumps(shown) and str(tmp_path) not in json.dumps(shown)  # a note is stored, never served


def test_posting_the_same_ledger_again_changes_nothing_and_says_so(tmp_path):
    body, data = lab_ledger(tmp_path / "lab"), tmp_path / "data"
    stored = data / "lab_state" / "ledger.jsonl"
    with TestClient(app_with(data)) as client:
        first = client.post(URL, content=body).json()
        os.utime(stored, ns=(1, 1_700_000_000_000_000_000))  # a rewrite would move the file's time
        again = client.post(URL, content=body).json()
        assert again["changed"] is False and "nothing was written" in again["note"] and again["rows"] == first["rows"]
        assert stored.stat().st_mtime_ns == 1_700_000_000_000_000_000 and again["received_at"] == "2023-11-14T22:13:20Z"
        grown = lab_ledger(tmp_path / "lab2", extra_ids=("E15",))  # the lab's ledger, one experiment further on
        assert client.post(URL, content=grown).json() | {"received_at": None, "bytes": 0, "rows": 0} == {
            **first, "experiments": 9, "received_at": None, "bytes": 0, "rows": 0}
        assert stored.read_bytes() == grown


def test_a_bad_line_refuses_the_whole_body_names_every_line_and_leaves_the_previous_copy(tmp_path):
    body, data = lab_ledger(tmp_path / "lab"), tmp_path / "data"
    lines = body.decode().splitlines()
    no_actor = {k: v for k, v in json.loads(lines[1]).items() if k != "actor"}
    bad_metric = {**json.loads(lines[2]), "metrics": {"Not A Metric": PRIVATE}, "note": 7}
    no_seq = {k: v for k, v in json.loads(lines[3]).items() if k != "seq"}
    bad = "\n".join([lines[0], json.dumps(no_actor), json.dumps(bad_metric), json.dumps(no_seq), "{half " + PRIVATE, "[1, 2]"])
    stored = data / "lab_state" / "ledger.jsonl"
    with TestClient(app_with(data)) as client:
        refused = client.post(URL, content=bad)
        assert refused.status_code == 400 and not stored.parent.exists()  # nothing written, not even the folder
        assert refused.json()["detail"] == {"problems": 6, "refused": [
            "line 2: actor is missing", "line 3: metrics is not what the lab's ledger accepts", "line 3: note is not what the lab's ledger accepts",
            "line 4: seq is missing or not an integer", "line 5: not JSON", "line 6: not a JSON object"]}
        assert PRIVATE not in refused.text and "Not A Metric" not in refused.text  # line numbers and field names only
        assert client.post(URL, content="\n\n").json()["detail"]["refused"] == ["the file has no ledger rows"]
        assert client.post(URL, content=b"\xff\xfe").json()["detail"]["refused"] == ["the body is not UTF-8 text"]
        assert client.post(URL, content=body).status_code == 200
        assert client.post(URL, content=bad).status_code == 400
        assert stored.read_bytes() == body and os.listdir(stored.parent) == ["ledger.jsonl"]  # the copy before it is whole
        assert client.get("/api/learning/lab/experiments").json()["experiments"]["run"] == 2


def test_a_line_shaped_like_a_credential_refuses_the_body_by_line_and_kind_only(tmp_path):
    body = lab_ledger(tmp_path / "lab")
    token = expand("{hex32}", "ledger-import")  # generated here: the repo holds no credential-shaped value
    password = expand("{password}", "ledger-import")
    record.record("hub check", "fail", state_dir=tmp_path / "lab", note=f"curl -H 'Authorization: token {token}' failed")
    record.record("hub login", "ok", state_dir=tmp_path / "lab", conditions={"form": f'{{"password": "{password}"}}'})
    leaky = (tmp_path / "lab" / "ledger.jsonl").read_bytes()
    n = len(leaky.splitlines())
    with pytest.raises(ImportRefused) as err:
        check_ledger(leaky.decode())
    assert [p.split(" (")[0] for p in err.value.problems] == [f"line {n - 1}: holds what looks like a credential",
                                                               f"line {n}: holds what looks like a credential"]
    assert "(hub token)" in err.value.problems[0] and "(account password)" in err.value.problems[1]  # found inside the JSON string
    assert token not in str(err.value) and password not in str(err.value)
    with TestClient(app_with(tmp_path / "data")) as client:
        refused = client.post(URL, content=leaky)
        assert refused.status_code == 400 and token not in refused.text and not (tmp_path / "data" / "lab_state").exists()
        assert client.post(URL, content=body).status_code == 200  # the same ledger without those two rows is clean


def test_the_post_is_guarded_and_capped(tmp_path, monkeypatch):
    body, data = lab_ledger(tmp_path / "lab"), tmp_path / "data"
    with TestClient(app_with(data, token="secret")) as client:
        assert client.post(URL, content=body).status_code == 401
        assert client.post(URL, content=body, headers={"Authorization": "Bearer wrong"}).status_code == 401
        monkeypatch.setattr(lab_import, "MAX_BYTES", 100)
        assert client.post(URL, content=body, headers={"Authorization": "Bearer secret"}).status_code == 413
        assert not (data / "lab_state").exists()
        monkeypatch.undo()
        assert client.post(URL, content=body, headers={"Authorization": "Bearer secret"}).status_code == 200
    app = FastAPI()
    app.include_router(lab_import_router(LabView(), EventBus(log_dir=data / "events").publish))  # wired without a data dir
    with TestClient(app) as client:
        assert client.post(URL, content=body).status_code == 400


def test_the_reader_takes_the_named_folder_then_the_pushed_copy_then_the_checkout(tmp_path, monkeypatch):
    named, pushed, checkout = tmp_path / "named", imported_state_dir(tmp_path / "data"), tmp_path / "checkout-state"
    for folder, eid in ((named, "E21"), (pushed, "E22"), (checkout, "E23")):
        lab_experiments.add({**E11, "id": eid}, state_dir=folder)

    def seen(**kwargs):
        source = timeline([], refine_dir=tmp_path / "none", **kwargs)
        return [c["id"] for c in source["cards"]], source["experiments"]["source"]["from"], source["experiments"]["source"]["received_at"]

    monkeypatch.setenv(ledger.ENV_STATE_DIR, str(named))
    assert seen(imported_dir=pushed)[:2] == (["E21"], "LAB_STATE_DIR") and seen(imported_dir=pushed)[2] is None
    monkeypatch.setenv(ledger.ENV_STATE_DIR, str(tmp_path / "empty"))  # set, but no ledger there: the next one is read
    assert seen(imported_dir=pushed)[:2] == (["E22"], "imported") and seen(imported_dir=pushed)[2].endswith("Z")
    monkeypatch.delenv(ledger.ENV_STATE_DIR)
    assert seen(imported_dir=pushed)[:2] == (["E22"], "imported")
    assert seen()[:2] == (["E23"], "checkout") and seen(imported_dir=tmp_path / "nowhere")[:2] == (["E23"], "checkout")
    assert seen(state_dir=named, imported_dir=pushed)[:2] == (["E21"], "LAB_STATE_DIR")
    assert ledger_source(tmp_path / "empty") is None  # a folder named by the caller never falls through to the checkout
    (checkout / "ledger.jsonl").unlink()
    assert timeline([], refine_dir=tmp_path / "none")["experiments"]["source"]["kind"] == "seeds"


def test_the_command_keeps_the_ledger_under_the_data_dir_once(tmp_path, capsys):
    body, data = lab_ledger(tmp_path / "lab"), tmp_path / "data"
    args = ["--ledger", str(tmp_path / "lab" / "ledger.jsonl"), "--data-dir", str(data)]
    assert lab_import.main(args) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True and out["ledger"]["changed"] is True and out["ledger"]["verdicts"] == {"a_ahead": 1, "b_ahead": 1, "too_early": 1}
    assert (data / "lab_state" / "ledger.jsonl").read_bytes() == body and not (data / "events").exists()  # no log opened for it
    assert lab_import.main(args) == 0 and json.loads(capsys.readouterr().out)["ledger"]["changed"] is False
    assert timeline([], imported_dir=imported_state_dir(data), refine_dir=tmp_path / "none")["experiments"]["run"] == 2
    (tmp_path / "bad.jsonl").write_text('{"seq": 1}\n')
    assert lab_import.main(["--ledger", str(tmp_path / "bad.jsonl"), "--data-dir", str(data)]) == 1
    assert "line 1: actor is missing" in capsys.readouterr().err and (data / "lab_state" / "ledger.jsonl").read_bytes() == body
    assert store_ledger(body, data)["changed"] is False
