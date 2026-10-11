"""The lab's "Under the hood" report pack as the Learning lab screen reads it: sentences and numbers are copied, never
recomputed, and a pack older than the lab's rows says so.

Behaviour is tested on a small pack written here. The committed pack (lab/refine/report/under-the-hood.json) is the
lab lane's and is rebuilt as rows land, so only what is true of any pack is asserted on it: no count or value is pinned.
"""

import hashlib
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from contexts.learning.application.lab_bench import METRICS_FILE
from contexts.learning.application.lab_report import FORMAT, KEEP, PACK_FILE, under_the_hood
from contexts.learning.lab_bench_api import lab_bench_router


def n(id_: str, label: str, value: float | None, unit: str, kind: str = "measured", line: int = 1, **more) -> dict:
    return {"id": id_, "label": label, "value": value, "unit": unit, "kind": kind,
            "source": {"file": "lab/refine/metrics.jsonl", "line": line, "ref": "a test"}, "low_confidence": False, "other_chip": False, **more}


def pack(metrics: bytes = b"{}\n{}\n") -> dict:
    best = n("metrics:2", "vLLM, pod-c", 2.88, "tok/s", line=2)
    slow = n("metrics:1", "vLLM, pod-a", 2.1, "tok/s")
    other = n("metrics:9", "vLLM, A100", 64.9, "tok/s", line=9, other_chip=True)
    return {
        "format": FORMAT, "title": "Under the hood", "data_as_of": "2026-10-11T04:13:45Z", "chip": {"current": "gfx1151", "name": "Radeon 8060S"},
        "sources": [{"file": "lab/refine/metrics.jsonl", "role": "rows", "sha256": hashlib.sha256(metrics).hexdigest(), "lines": len(metrics.splitlines())}],
        "sections": [
            {"id": "ceiling", "title": "The ceiling", "reading": "The memory could feed 13 tokens a second; we reach 2.88.", "state": "measured",
             "ceiling": n("derived:ceiling", "ceiling at batch 1", 13.41, "tok/s", kind="derived", formula="bandwidth / bytes per token"),
             "achieved": [slow, best, other], "shares": [n("derived:share:metrics:2", "share of the ceiling", 21.5, "%", kind="derived")],
             "inputs": [n("chipsheet:copy", "memory bandwidth", 207.8, "GB/s")], "awaiting": ["R60", "R1"]},
            {"id": "frontier", "title": "Batching", "reading": "64 at once cuts the energy for each word 18.9 times.", "state": "measured",
             "ratios": [n("derived:ratio", "joules per token, 1 stream over 64", 18.9, "x", kind="derived")], "more_work": [],
             "power": [{"points": [{"batch": 1, "number": n("metrics:5", "watts at 1", 96.0, "W", line=5)},
                                   {"batch": 64, "number": n("metrics:6", "watts at 64", 96.05, "W", line=6)}]}], "awaiting": []},
            {"id": "step", "title": "A decode step", "reading": "Not measured yet: the decode profile has not committed a result.", "state": "empty",
             "why": "no decode profile on main", "batches": [], "awaiting": ["R60"]},
            {"id": "finetune", "title": "The fine-tune", "reading": "Loss went from 0.127 to 0.009.", "state": "measured",
             "loss": [n(f"train:{k}", f"loss at step {k}", 0.2 / k, "loss", line=k) for k in range(1, 12)],
             "median_step": n("derived:median", "median step time", 40.8, "s", kind="derived"), "numbers": [], "awaiting": []},
            {"id": "untitled", "title": "No sentence", "reading": "", "state": "measured"},
            {"id": "new-kind", "title": "A section this reader has never seen", "reading": "It says one thing.", "state": "odd"},
        ],
        "awaited": [{"task": "R60", "what": "the teacher on the new build", "status": "being measured", "landed": False},
                    {"task": "R1", "what": "an old run", "status": "done", "landed": True}],
    }


def write(tmp_path, body: dict, metrics: bytes = b"{}\n{}\n"):
    file = tmp_path / "under-the-hood.json"
    file.write_text(json.dumps(body), encoding="utf-8")
    rows = tmp_path / "metrics.jsonl"
    rows.write_bytes(metrics)
    return file, rows


def test_each_section_gives_its_sentence_its_state_and_numbers_copied_with_file_and_line(tmp_path):
    file, rows = write(tmp_path, pack())
    out = under_the_hood(file, rows)
    assert out["available"] and out["reason"] is None and out["stale"] is None and out["chip"] == "Radeon 8060S"
    by = {s["id"]: s for s in out["sections"]}
    assert "untitled" not in by  # a section with no sentence is never a card with only a title
    assert by["ceiling"]["reading"].startswith("The memory could feed")
    got = {x["id"]: x for x in by["ceiling"]["numbers"]}
    assert list(got) == ["derived:ceiling", "metrics:2", "derived:share:metrics:2", "chipsheet:copy"]  # the best row on our chip, not the A100's
    assert (got["metrics:2"]["value"], got["metrics:2"]["unit"], got["metrics:2"]["file"], got["metrics:2"]["line"]) == (2.88, "tok/s", "lab/refine/metrics.jsonl", 2)
    assert got["derived:ceiling"]["kind"] == "derived" and got["derived:ceiling"]["formula"] == "bandwidth / bytes per token"
    assert [x["value"] for x in by["frontier"]["numbers"]] == [18.9, 96.0, 96.05]
    assert by["step"]["state"] == "empty" and by["step"]["numbers"] == [] and by["step"]["why"] == "no decode profile on main"
    assert len(by["finetune"]["numbers"]) <= KEEP and by["finetune"]["numbers"][0]["id"] == "train:1"
    assert by["new-kind"]["numbers"] == [] and by["new-kind"]["state"] == "partial" and by["new-kind"]["reading"] == "It says one thing."


def test_only_runs_that_have_not_landed_are_named_as_still_coming(tmp_path):
    file, rows = write(tmp_path, pack())
    out = under_the_hood(file, rows)
    assert [w["task"] for w in out["sections"][0]["waiting"]] == ["R60"]
    assert out["still_coming"] == [{"task": "R60", "what": "the teacher on the new build", "status": "being measured"}]


def test_a_pack_older_than_the_rows_on_disk_says_how_far_behind_it_is(tmp_path):
    file, rows = write(tmp_path, pack(), metrics=b"{}\n{}\n{}\n{}\n{}\n")
    file.write_text(json.dumps(pack()), encoding="utf-8")  # built when the file had two lines
    stale = under_the_hood(file, rows)["stale"]
    assert (stale["lines_then"], stale["lines_now"], stale["added"]) == (2, 5, 3)
    assert under_the_hood(file, tmp_path / "gone.jsonl")["stale"] is None  # no file to compare with: not known, not claimed


def test_a_missing_or_foreign_pack_is_not_available_and_says_why(tmp_path):
    missing = under_the_hood(tmp_path / "nothing.json")
    assert not missing["available"] and missing["sections"] == [] and "not on this box" in missing["reason"]
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    assert "not readable" in under_the_hood(tmp_path / "bad.json")["reason"]
    (tmp_path / "other.json").write_text(json.dumps({"format": "something/2", "sections": []}), encoding="utf-8")
    assert FORMAT in under_the_hood(tmp_path / "other.json")["reason"]


def test_the_endpoint_serves_the_pack_under_its_own_path(tmp_path):
    file, rows = write(tmp_path, pack())
    app = FastAPI()
    app.include_router(lab_bench_router(metrics_file=rows, state_dir=tmp_path, report_file=file))
    with TestClient(app) as client:
        body = client.get("/api/learning/lab/report").json()
    assert body["available"] and [s["id"] for s in body["sections"]][:2] == ["ceiling", "frontier"]


def test_the_committed_pack_reads_whole_and_every_number_shown_has_a_file_whatever_the_lab_has_rebuilt():
    out = under_the_hood(PACK_FILE, METRICS_FILE)
    assert out["available"], out["reason"]
    assert out["sections"] and all(s["reading"] and s["title"] and s["state"] in ("measured", "partial", "empty") for s in out["sections"])
    for s in out["sections"]:
        assert len(s["numbers"]) <= KEEP
        for x in s["numbers"]:
            assert x["value"] is not None and x["file"] and x["kind"] and not x["other_chip"], (s["id"], x)
    assert out["stale"] is None or out["stale"]["lines_now"] >= 0
