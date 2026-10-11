"""Tests for the under-the-hood report (lab/report.py, python -m lab report).

The builder reads a small committed tree (fixtures/lab/report/) or an empty folder; no pod, hub, network or
Notion is reached. The real repository is read only to check that the page built from it stays self-contained.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from lab import __main__ as lab_cli
from lab import report

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "fixtures" / "lab" / "report"
LANDED = REPO_ROOT / "fixtures" / "lab" / "report_landed"  # the same tree once every awaited run has landed
EXTERNAL = re.compile(r"https?://|<script|<link|@import|url\(|<iframe|<img|<object|<embed", re.IGNORECASE)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LAB_ENV_FILES", "")


def section(pack: dict, sid: str) -> dict:
    return next(s for s in pack["sections"] if s["id"] == sid)


def links_resolve(page: str, folder: Path) -> None:
    """Every link on the page opens from disk: a file link reaches a file, an in-page link an element's id."""
    ids = set(re.findall(r'\bid="([^"]+)"', page))
    hrefs = re.findall(r'href="([^"]+)"', page)
    assert hrefs
    for h in hrefs:
        if h.startswith("#"):
            assert h[1:] in ids, h
        else:
            assert (folder / h).resolve().is_file(), h


def test_the_fixture_builds_every_section_from_its_rows():
    pack = report.build_pack(FIXTURE)
    assert pack["format"] == report.FORMAT
    assert set(pack) == {"format", "title", "chip", "data_as_of", "sources", "skipped", "sections", "numbers", "awaited"}
    assert [s["id"] for s in pack["sections"]] == ["ceiling", "frontier", "step", "engines", "finetune", "dropped"]
    assert pack["skipped"] == [{"line": 3, "problem": "not JSON"}]

    ceiling = section(pack, "ceiling")
    bw, gb = ceiling["inputs"]
    assert (bw["value"], bw["source"]["line"]) == (200.0, report.line_of(
        FIXTURE / report.DATA_DIR / "chipsheet-gfx1151-2026-10-11.json", '"id": "copy.d2d.size_4gib"'))
    assert bw["id"] == "chipsheet:copy.d2d.size_4gib"  # the memory-sized copy, not the cache-sized 900 GB/s one
    assert (gb["value"], gb["source"]) == (10.0, {"file": "lab/public_prior.py", "line": 4, "ref": gb["source"]["ref"]})
    assert ceiling["ceiling"]["value"] == 20.0 and ceiling["ceiling"]["kind"] == "derived"
    assert [a["value"] for a in ceiling["achieved"]] == [3.0]
    # the R9700 row and the A100 reference are kept apart, each under its own chip, never on the chart
    assert [(a["value"], a["chip_label"]) for a in ceiling["other_chip"]] == [
        (7.5, "R9700 (gfx1201)"), (50.0, "NVIDIA A100-SXM4-40GB")]
    assert ceiling["shares"][0]["value"] == 15.0
    assert ceiling["priors"] == [] and ceiling["prior_gap"]
    assert ceiling["state"] == "measured" and "20 tokens" in ceiling["reading"]

    frontier = section(pack, "frontier")
    assert [p["number"]["value"] for p in frontier["energy"][0]["points"]] == [30.0, 3.0]
    assert frontier["ratios"][0]["value"] == 10.0
    assert "10 times" in frontier["reading"]
    assert frontier["energy"][0]["points"][0]["number"]["reason"] == "3 spawns"


def test_every_number_carries_its_source_line_and_gpu():
    pack = report.build_pack(FIXTURE)
    lines = (FIXTURE / report.METRICS).read_text(encoding="utf-8").splitlines()
    for n in pack["numbers"]:
        assert {"id", "label", "value", "unit", "kind", "source", "gpu", "chip", "low_confidence", "other_chip"} <= set(n)
        if n["kind"] == "measured" and n["source"]["file"] == report.METRICS:
            row = json.loads(lines[n["source"]["line"] - 1])
            assert row["value"] == n["value"] and row["metric"] == n["metric"]
            assert n["gpu"] == row["conditions"].get("gpu") and n["chip"] == row["conditions"].get("chip")
        if n["kind"] == "derived":
            assert n["formula"] and n["inputs"]


def test_an_empty_tree_says_not_measured_everywhere(tmp_path):
    pack = report.build_pack(tmp_path)
    assert {s["id"]: s["state"] for s in pack["sections"]} == {
        "ceiling": "empty", "frontier": "empty", "step": "empty", "engines": "empty", "finetune": "empty",
        "dropped": "empty"}
    assert all(s["reading"].startswith("Not measured yet") for s in pack["sections"][:5])
    assert section(pack, "ceiling")["ceiling"]["value"] is None
    assert [n for n in pack["numbers"] if n["value"] is not None] == []
    page = report.render_html(pack, tmp_path, tmp_path / report.OUT_DIR)
    for sid in ("ceiling", "frontier", "step", "engines", "finetune", "dropped"):
        part = page.split(f'<section id="{sid}">')[1].split("</section>")[0]
        assert report.NOT_MEASURED in part


def test_the_page_has_no_external_reference_and_its_links_resolve(tmp_path):
    result = report.write(FIXTURE, tmp_path / "out")
    page = Path(result["html"]).read_text(encoding="utf-8")
    assert not EXTERNAL.search(page)
    assert page.startswith("<!doctype html>") and 'name="viewport"' in page
    links_resolve(page, tmp_path / "out")
    assert "metrics.jsonl:2" in page and "metrics.jsonl:1" in page  # the old-chip row is shown, apart
    assert page.index("metrics.jsonl:1<") > page.index("Other chip, shown apart: R9700 (gfx1201)")
    assert page.index("metrics.jsonl:8<") > page.index("Other chip, shown apart: NVIDIA A100-SXM4-40GB")
    pack = json.loads(Path(result["pack"]).read_text(encoding="utf-8"))
    assert pack == json.loads(json.dumps(report.build_pack(FIXTURE)))


def test_the_build_is_repeatable(tmp_path):
    report.write(FIXTURE, tmp_path / "a")
    report.write(FIXTURE, tmp_path / "b")
    for name in (report.HTML_NAME, report.PACK_NAME):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_the_page_built_from_this_repository_stays_self_contained(tmp_path):
    result = report.write(REPO_ROOT, tmp_path / report.OUT_DIR)
    page = Path(result["html"]).read_text(encoding="utf-8")
    assert not EXTERNAL.search(page)
    links_resolve(page, Path(result["html"]).parent)
    pack = json.loads(Path(result["pack"]).read_text(encoding="utf-8"))
    for n in pack["numbers"]:
        src = n["source"]
        if src["file"]:
            assert (REPO_ROOT / src["file"]).is_file()
        if src["line"]:
            assert src["line"] <= len((REPO_ROOT / src["file"]).read_text(encoding="utf-8").splitlines())


def test_the_committed_report_is_a_pack_of_this_format():
    folder = REPO_ROOT / report.OUT_DIR
    pack = json.loads((folder / report.PACK_NAME).read_text(encoding="utf-8"))
    assert pack["format"] == report.FORMAT
    assert not EXTERNAL.search((folder / report.HTML_NAME).read_text(encoding="utf-8"))


def test_the_verb_is_registered_in_its_alphabetical_place(tmp_path):
    names = list(lab_cli.VERBS)
    assert lab_cli.VERBS["report"] == "lab.report:main" and names == sorted(names)
    done = subprocess.run([sys.executable, "-m", "lab", "report", "--root", str(FIXTURE), "--out", str(tmp_path)],
                          cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    answer = json.loads(done.stdout.strip().splitlines()[-1])
    assert answer["ok"] and answer["sections"]["ceiling"] == "measured" and answer["sections"]["step"] == "empty"
    assert (tmp_path / report.HTML_NAME).is_file() and (tmp_path / report.PACK_NAME).is_file()


def test_check_says_when_the_files_on_disk_are_stale(tmp_path):
    out = tmp_path / "out"
    first = report.write(FIXTURE, out, check=True)
    assert first["ok"] is False and first["stale"] == [report.PACK_NAME, report.HTML_NAME] and not out.exists()
    report.write(FIXTURE, out)
    current = report.main(["--root", str(FIXTURE), "--out", str(out), "--check"])
    assert current["ok"] is True and current["outcome"] == "current" and current["stale"] == []
    (out / report.HTML_NAME).write_text("an older page", encoding="utf-8")
    stale = report.main(["--root", str(FIXTURE), "--out", str(out), "--check"])
    assert stale["ok"] is False and stale["stale"] == [report.HTML_NAME]
    assert (out / report.HTML_NAME).read_text(encoding="utf-8") == "an older page"  # a check writes nothing


def test_a_line_that_repeats_an_earlier_row_is_counted_once(tmp_path):
    """A merge that keeps both copies of a rewritten row leaves the same measurement on two lines."""
    row = {"metric": "teacher.joules_per_token", "value": 30.0, "unit": "J/token", "node": "pod-a",
           "conditions": {"chip": "gfx1151", "batch_size": 1, "session": "s1"}, "ts": "2026-10-11T00:00:00Z"}
    lines = [
        {**row, "conditions": {**row["conditions"], "sampler": "T6"}, "backfilled": True},  # 1: the rewritten copy
        {**row, "value": 3.0, "conditions": {**row["conditions"], "batch_size": 8}},        # 2: another stream count
        row,                                                                               # 3: line 1 as first written
        {**row, "conditions": {**row["conditions"], "session": "s2"}},                     # 4: another session
        {**row, "ts": "2026-10-11T04:00:00Z"},                                             # 5: measured again later
        {**row, "ts": None},                                                               # 6 and 7: no time recorded,
        {**row, "ts": None},                                                               # so never called a repeat
    ]
    root = tmp_path / "tree"
    (root / "lab" / "refine").mkdir(parents=True)
    (root / report.METRICS).write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
    rows, skipped = report.read_metrics(root / report.METRICS)
    assert [r["_line"] for r in rows] == [1, 2, 4, 5, 6, 7]
    assert skipped == [{"line": 3, "problem": "repeats line 1", "repeats": 1}]
    pack = report.build_pack(root)
    assert pack["skipped"] == skipped
    frontier = section(pack, "frontier")
    assert sorted(len(s["points"]) for s in frontier["energy"]) == [2]  # one sweep, its first point taken once
    # among the rows kept, the last line written at a stream count is still the one a sweep shows
    assert [p["number"]["source"]["line"] for p in frontier["energy"][0]["points"]] == [7, 2]
    page = report.render_html(pack, root, root / report.OUT_DIR)
    assert "Left out of every number above: 1 line of" in page and "repeat an earlier row" in page and "(line 3)" in page


def test_names_tell_builds_and_engines_apart():
    first = {"engine": "vllm@b1388b1+lab/vllm-gfx1201.diff, 0.19.2+gfx1151"}
    assert report.build_label(first) == "vLLM b1388b1 + gfx1201 patch"
    attached = {"engine": "vllm", "engine_commit": "unavailable: attached through the hub proxy",
                "patch_hash": "unavailable: attached through the hub proxy"}
    assert report.build_label(attached) == "vLLM, build not recorded"
    assert report.build_label({"engine": "vllm", "vllm": "0.31.0"}) == "vLLM 0.31.0"
    newer = {"engine": "vllm", "engine_commit": "db9527a46", "patch_hash": {"vllm-main-gfx1151-platform.diff": "00"}}
    assert report.build_label(newer) == "vLLM db9527a + main-gfx1151-platform patch"
    # lab/bench.py spells the engine llamacpp and records llama-server's build line as its commit
    llama = {"engine": "llamacpp", "engine_commit": "6500 (1a2b3c4d)", "backend": "vulkan"}
    assert report.engine_family(llama) == "llama.cpp" and report.build_label(llama) == "llama.cpp Vulkan 1a2b3c4"
    assert report.engine_family({"engine": "llama-server"}) == "llama.cpp"
    assert report.engine_family({}) == "not recorded" and report.build_label({}) == "engine not recorded"
    # a prompt set is named whether the row writes it as an object or as its id (the A100 rows write the id)
    assert report.asked_of({"prompt_set": "bench-v1"}) == "bench-v1 prompt set"
    assert report.asked_of({"prompt_set": {"id": "bench-v1"}}) == "bench-v1 prompt set"
    assert report.asked_of({"prompt": "text"}) == "text prompt" and report.asked_of({}) is None


def test_the_runs_still_coming_are_named_until_their_rows_land(tmp_path):
    pack = report.build_pack(FIXTURE)
    waiting = {a["task"]: a for a in pack["awaited"]}
    assert set(waiting) == {"R60", "R58", "R53", "R52", "R47", "L16"}
    assert not any(a["landed"] or a["evidence"] for a in waiting.values())
    assert all(a["status"] and a["as_of"] == report.STILL_COMING_AS_OF for a in waiting.values())
    page = report.render_html(pack, FIXTURE, tmp_path)
    for sid, tasks in (("ceiling", ("R60", "R58", "R53")), ("frontier", ("R52",)), ("step", ("R47",)),
                       ("engines", ("R60", "R58", "R53")), ("finetune", ("L16",))):
        part = page.split(f'<section id="{sid}">')[1].split("</section>")[0]
        assert "Still coming" in part and report.STILL_COMING_AS_OF in part
        for task in tasks:
            assert f"<b>{task}</b>" in part
    landed = report.build_pack(LANDED)
    assert all(a["landed"] and a["evidence"] for a in landed["awaited"])
    assert {a["task"]: a["file_present"] for a in landed["awaited"]} == {
        "R60": False, "R58": False, "R53": False, "R52": True, "R47": True, "L16": True}
    assert "Still coming" not in report.render_html(landed, LANDED, tmp_path)


def test_a_tree_where_every_run_has_landed_fills_every_section():
    pack = report.build_pack(LANDED)
    assert {s["id"]: s["state"] for s in pack["sections"]} == {
        "ceiling": "measured", "frontier": "measured", "step": "measured", "engines": "measured",
        "finetune": "measured", "dropped": "partial"}

    ceiling = section(pack, "ceiling")  # one bar per build, oldest first, each named from its own row
    assert [(b["build"], b["best"]["value"]) for b in ceiling["builds"]] == [
        ("vLLM b1388b1 + gfx1201 patch", 2.5), ("vLLM db9527a + main-gfx1151-platform patch", 6.0),
        ("vLLM b1388b1 + gfx1151 patch", 4.0), ("llama.cpp Vulkan 1a2b3c4", 9.0), ("engine not recorded", 2.6)]
    assert ceiling["builds"][3]["share"]["value"] == 45.0 and "about 45%" in ceiling["reading"]

    engines = section(pack, "engines")
    assert [(e["engine"], e["best"]["value"]) for e in engines["engines"]] == [("llama.cpp", 9.0), ("vllm", 6.0)]
    assert engines["not_measured"] == [] and engines["compared_on"] == "bench-v1 prompt set"
    assert [d["what"] for d in engines["differences"]] == ["weights"]
    assert [n["value"] for n in engines["unnamed"]] == [2.6]  # a row that names no engine stands for none
    assert "llama.cpp decodes 9.00" in engines["reading"] and "differ in weights" in engines["reading"]

    frontier = section(pack, "frontier")
    # the sweep with the most stream counts leads, although the other pod's metric name sorts first
    assert [(s["node"], len(s["points"])) for s in frontier["energy"]] == [("pod-a", 3), ("pod-b", 2)]
    assert [r["value"] for r in frontier["ratios"]] == [20.0, 3.2]
    assert [r["value"] for r in frontier["more_work"]] == [20.0]
    assert "by 20 times" in frontier["reading"] and "100 watts" in frontier["reading"]
    assert frontier["reading"].count(".") == 1  # one sentence; the second pod's sweep is a note beside the chart
    assert frontier["notes"] == ["Measured again on pod-b, vLLM b1388b1 + gfx1201 patch: 3.2 times less energy for "
                                 "each token at 4 streams than at 1."]
    # the settings sweep's two cells are two slices, never one sweep; its change row and its file are listed
    assert [c["value"] for c in frontier["changes"]] == [12.5]
    assert frontier["pending_files"] == ["sweep-request-gfx1151-2026-10-11.json"]
    assert [s["chip_label"] for s in frontier["other_chip"]] == ["NVIDIA A100-SXM4-40GB"]
    assert frontier["other_ratios"][0]["value"] == 5.0 and frontier["other_ratios"][0]["other_chip"]

    step = section(pack, "step")
    assert [b["step_ms"] for b in step["batches"]] == [300.0, 360.0]
    assert step["top"]["value"] == 60.0 and "60%, is quantized matmul" in step["reading"]
    lines = (LANDED / step["file"]).read_text(encoding="utf-8").splitlines()
    for batch in step["batches"]:
        for part in batch["parts"]:
            assert '"ms_per_step"' in lines[part["source"]["line"] - 1] and part["gpu"] == "AMD Radeon 8060S"
    assert step["batches"][0]["parts"][0]["source"]["line"] < step["batches"][1]["parts"][0]["source"]["line"]

    tune = section(pack, "finetune")
    assert [n["value"] for n in tune["loss"]] == [0.5, 0.3, 0.1]
    assert [(n["source"]["file"], n["source"]["line"]) for n in tune["loss"]] == [
        ("fixtures/demo/l16-telemetry.jsonl", line) for line in (1, 2, 3)]
    assert {n["gpu"] for n in tune["loss"] + tune["step_time"]} == {"Radeon 8060S Graphics (gfx1151)"}
    assert not any(n["other_chip"] for n in tune["loss"]) and tune["other_chip"] == []
    assert tune["median_step"]["value"] == 50.0 and tune["run"]["run_id"] == "train-fixture"
    assert "from 0.5 to 0.1 over 3 logged steps" in tune["reading"] and "50 seconds a step" in tune["reading"]
    facts = {n["label"]: n for n in tune["numbers"]}
    assert facts["trainer.samples_per_s"]["value"] == 0.25
    assert facts["trainer.samples_per_s"]["source"] == {
        "file": "fixtures/demo/l16-run.json", "ref": "metrics.trainer.samples_per_s, run train-fixture",
        "line": report.line_of(LANDED / "fixtures" / "demo" / "l16-run.json", '"trainer.samples_per_s"')}

    dropped = section(pack, "dropped")
    assert [(i["what"], i["kind"], i["source"]["line"]) for i in dropped["items"]] == [
        ("SETTING_A", "ruled out", 4), ("setting b, with a comma", "ruled out", 5)]
    assert "2 settings ruled out" in dropped["reading"]
    assert {x["role"] for x in pack["sources"]} == {
        "rows", "chip sheet", "decode profile", "public prior model", "settings sweep", "search space",
        "fine-tune run"}


def test_the_landed_page_cites_every_number_and_keeps_the_other_chip_apart(tmp_path):
    result = report.write(LANDED, tmp_path / "out")
    page = Path(result["html"]).read_text(encoding="utf-8")
    assert not EXTERNAL.search(page) and result["still_coming"] == []
    links_resolve(page, tmp_path / "out")
    for n in json.loads(Path(result["pack"]).read_text(encoding="utf-8"))["numbers"]:
        if n["source"]["file"] and n["source"]["line"]:  # its file and line, as the text of a link
            assert f'{n["source"]["file"]}:{n["source"]["line"]}</a>' in page, n["id"]
    frontier = page.split('<section id="frontier">')[1].split("</section>")[0]
    ours, apart = frontier.split("Other chip, shown apart: NVIDIA A100-SXM4-40GB")
    assert "metrics.jsonl:21<" not in ours and "metrics.jsonl:21<" in apart  # the A100 rows only in their panel
    assert ours.count("<svg") == 2 and apart.count("<svg") == 1  # and on a chart of their own, never ours
    for sid in ("ceiling", "frontier", "step", "engines", "finetune"):
        part = page.split(f'<section id="{sid}">')[1].split("</section>")[0]
        assert "<svg" in part and 'class="key"' in part and 'class="say"' in part
