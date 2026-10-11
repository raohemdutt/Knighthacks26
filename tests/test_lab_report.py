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
EXTERNAL = re.compile(r"https?://|<script|<link|@import|url\(|<iframe|<img|<object|<embed", re.IGNORECASE)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LAB_ENV_FILES", "")


def section(pack: dict, sid: str) -> dict:
    return next(s for s in pack["sections"] if s["id"] == sid)


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
    hrefs = re.findall(r'href="([^"]+)"', page)
    assert hrefs
    for h in hrefs:
        assert (tmp_path / "out" / h).resolve().is_file(), h
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
    for h in re.findall(r'href="([^"]+)"', page):
        assert (Path(result["html"]).parent / h).resolve().is_file(), h
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
