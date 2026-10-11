"""Guards on the 512-row mixed calibration set the A100 quantization ran on, and on the recipe's dry run with it.

calibration-mixed-512.jsonl was written on the Colab VM by fixtures/lab/quant/make_calibration_mixed.py (it needs the
network and the base model's chat template), so these tests check the committed file's promises instead of
regenerating it: it starts with the Atlas set byte for byte, pads it with the stated counts of general text, holds
no oversized row, and lab/quantize_teacher.py --dry-run takes it without importing any GPU package.
"""

import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from lab import quantize_teacher as qt

REPO_ROOT = Path(__file__).resolve().parents[1]
QUANT = REPO_ROOT / "fixtures" / "lab" / "quant"
_spec = importlib.util.spec_from_file_location("make_calibration_mixed", QUANT / "make_calibration_mixed.py")
mix = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mix)
MIXED = QUANT / "calibration-mixed-512.jsonl"
ATLAS = QUANT / "calibration.jsonl"


def lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def test_the_mixed_set_has_512_rows_and_starts_with_the_atlas_set_byte_for_byte():
    mixed, atlas = lines(MIXED), lines(ATLAS)
    assert len(mixed) == mix.TOTAL == 512
    assert mixed[: len(atlas)] == atlas


def test_the_padding_is_the_stated_general_text_in_the_stated_counts():
    rows = [json.loads(line) for line in lines(MIXED)]
    n_atlas = len(lines(ATLAS))
    counts = Counter(r["source"] for r in rows[n_atlas:])
    assert counts == {"ultrachat-200k": mix.TOTAL - n_atlas - mix.N_WIKI, "wikitext-103-train": mix.N_WIKI}
    # chat rows first, then wiki rows, as build() orders them
    sources = [r["source"] for r in rows[n_atlas:]]
    assert sources == sorted(sources, key=lambda s: s != "ultrachat-200k")


def test_every_row_is_well_formed_and_capped():
    for line in lines(MIXED):
        r = json.loads(line)
        assert set(r) == {"id", "source", "text", "chars"}
        assert r["text"] and r["chars"] == len(r["text"]) <= mix.MAX_CHARS
    general = [json.loads(line) for line in lines(MIXED)[len(lines(ATLAS)):]]
    assert all(r["id"].startswith(r["source"][:2] + "-") for r in general)
    assert all(len(r["text"]) >= mix.MIN_WIKI_CHARS for r in general if r["source"] == "wikitext-103-train")
    assert all(r["text"].startswith("<|im_start|>") for r in general if r["source"] == "ultrachat-200k")


def test_the_recipe_reads_all_512_texts():
    assert len(qt.read_calibration(MIXED)) == 512


def test_dry_run_with_the_mixed_set_prints_the_plan_and_imports_no_gpu_package():
    script = REPO_ROOT / "lab" / "quantize_teacher.py"
    code = "\n".join([
        "import runpy, sys",
        f"sys.argv = [{str(script)!r}, '--dry-run', '--width', '3', '--calibration', {str(MIXED)!r}]",
        "try:",
        "    runpy.run_path(sys.argv[0], run_name='__main__')",
        "except SystemExit as e:",
        "    assert e.code == 0, e.code",
        "bad = [m for m in ('torch', 'transformers', 'llmcompressor', 'datasets') if m in sys.modules]",
        "assert not bad, bad",
    ])
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, cwd=REPO_ROOT)
    assert run.returncode == 0, run.stderr
    plan = json.loads(run.stdout)
    assert plan["calibration"] == str(MIXED)
    assert [v["scheme"] for v in plan["variants"]] == ["W3A16"]
    assert plan["variants"][0]["recipe"]["quant_stage"]["quant_modifiers"]["GPTQModifier"]["scheme"] == "W3A16"
