"""lab/energy_model.py on the committed gfx1151 chip sheet and on a recorded batching sweep of the April build.

The expected numbers are read off the chip sheet's own Markdown summary and the checkpoints' configs, never
recomputed here with the module's formulas."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from lab import bench
from lab import energy_model as em

REPO = Path(__file__).resolve().parents[1]
SWEEP = REPO / "fixtures" / "lab" / "energy_model" / "attach-sweep-pod-c-2026-10-10.jsonl"


@pytest.fixture(scope="module")
def chip() -> em.ChipInputs:
    assert em.SHEET_PATH.is_file(), f"expected the committed chip sheet at {em.SHEET_PATH}"
    return em.chip_inputs(em.load_sheet())


def test_chip_inputs_are_the_sheet_rows(chip):
    # copy.d2d rows above the 32 MiB cache: 214.2, 212.9, 212.1, 210.5, 209.5, 208.8, 207.8 GB/s
    assert chip.cache_bytes == 33_554_432
    assert chip.bandwidth_gbs == pytest.approx(210.5, abs=0.05)
    assert chip.cache_gbs == pytest.approx(943.2, abs=0.05)          # copy.d2d.size_16mib
    assert chip.read_gbs == pytest.approx(219.6, abs=0.1)            # fp16 k5120_n248320 at m1, weight read
    assert chip.compute_tflops["bfloat16"] == pytest.approx(42.3, abs=0.05)   # lora.bf16 ... fwd
    assert chip.compute_tflops["float16"] == pytest.approx(38.3, abs=0.05)    # lora.fp16 ... fwd
    assert chip.watts["dequant"] == pytest.approx(110.0, abs=0.05)
    assert chip.watts["lora"] == pytest.approx(120.8, abs=0.05)     # fwd and dgrad, fp16 and bf16: 79.2 119.2 122.4 126.5
    assert chip.rate_m1["dequant.w4a16.decode27b.k5120_n17408.bn64_bk64"] == pytest.approx(0.58, abs=0.005)
    assert chip.shape_dtype["dequant.w4a16.decode27b.k5120_n17408.bn64_bk64"] == "float16"
    assert chip.idle_sections["copy"] == pytest.approx(24.17, abs=0.01)


def test_every_decode_shape_still_scales_with_batch_at_16(chip):
    # The sheet's matmul and dequant rows run 15 to 21 times faster at M 16 than at M 1: no shape is compute-bound yet.
    assert len(chip.scaling_m16) == 18
    assert min(chip.scaling_m16.values()) >= 15.0
    assert abs(chip.watts_slope) < 0.005    # loaded watts move less than half a percent per added row


def test_teacher_bytes_match_the_checkpoint():
    gb = {c.name: c.bytes / em.GB for c in em.TEACHER.classes}
    # main note section 1.2: 12.60 GB of layer weights (int4 plus bf16 scales, in_proj_a/b bf16), 2.54 GB lm_head
    assert gb["linear"] + gb["gdn_ab"] == pytest.approx(12.60, abs=0.02)
    assert gb["lm_head"] == pytest.approx(2.54, abs=0.01)
    # 48 Gated DeltaNet layers x 48 heads x 128 x 128 x 4 bytes, read and written: about 0.30 GB per sequence
    assert em.TEACHER.state_bytes == 2 * 48 * 48 * 128 * 128 * 4
    assert em.TEACHER.kv_bytes_per_token == 16 * 4 * 256 * 2 * 2      # 64 KiB per token of context
    assert em.STUDENT.state_bytes == 2 * 24 * 32 * 128 * 128 * 4
    assert em.STUDENT.kv_bytes_per_token == 8 * 4 * 256 * 2 * 2
    assert em.STUDENT.weight_bytes / em.GB == pytest.approx(8.4, abs=0.05)   # 4.2 B parameters in bf16


def test_april_build_reproduces_its_measured_batch_one(chip):
    april = em.april_build(chip)
    p = em.predict(em.TEACHER, april, 1, em.DECODE_512)
    assert p["decode_tok_s"] == pytest.approx(2.88, rel=1e-9)
    assert p["watts"] == pytest.approx(120.0, rel=1e-9)
    assert p["joules_per_token"] == pytest.approx(120.0 / 2.88, rel=1e-9)
    assert p["joules_per_token_above_idle"] == pytest.approx((120.0 - 10.2) / 2.88, rel=1e-9)
    # the hardware note's derived 44.6 GB/s for this build's weight path, within the 0.3 GB of state it adds
    assert april.weight_gbs == pytest.approx(44.0, abs=0.5)


def test_ceiling_build_runs_at_the_copy_bandwidth(chip):
    ceiling = em.ceiling_build(chip)
    p = em.predict(em.TEACHER, ceiling, 1, em.DECODE_512)
    # 15.14 GB of weights plus 0.30 GB of state and 34 MB of KV at 210.5 GB/s
    assert p["decode_tok_s"] == pytest.approx(13.6, abs=0.05)
    assert p["watts"] == pytest.approx(110.0, abs=0.05)
    assert p["joules_per_token"] == pytest.approx(8.08, abs=0.02)


def test_knees(chip):
    april = em.knees(em.TEACHER, em.april_build(chip))
    assert (april["knee"], april["binds"]) == (128.0, "m_tile")
    teacher = em.knees(em.TEACHER, em.ceiling_build(chip))
    assert teacher["binds"] == "linear"
    assert teacher["knee"] == pytest.approx(52, abs=1)
    assert teacher["state"] == pytest.approx(45, abs=1)
    student = em.knees(em.STUDENT, em.ceiling_build(chip, section="matmul"))
    assert student["knee"] == pytest.approx(201, abs=1)
    rows = em.sheet_knees(chip)
    # C (fp16 38.3) over the plain Triton W4A16 kernel's batch-1 rate (0.54 to 0.60 TFLOP/s)
    for shape in ("k5120_n6144", "k6144_n5120", "k5120_n17408", "k17408_n5120"):
        assert 60 <= rows[f"dequant.w4a16.decode27b.{shape}.bn64_bk64"] <= 75


def test_joules_per_token_fall_with_batch_until_the_knee_then_flatten(chip):
    ceiling = em.ceiling_build(chip)
    j = {b: em.predict(em.TEACHER, ceiling, b, em.DECODE_512)["joules_per_token"] for b in (1, 2, 4, 8, 16, 32, 52)}
    assert all(j[a] > j[b] for a, b in zip(sorted(j), sorted(j)[1:]))
    assert j[16] < j[1] / 10
    past = em.predict(em.TEACHER, ceiling, 104, em.DECODE_512)["joules_per_token"]
    far = em.predict(em.TEACHER, ceiling, 208, em.DECODE_512)["joules_per_token"]
    assert 0.95 <= far / past <= 1.10
    w1 = em.predict(em.TEACHER, ceiling, 1, em.DECODE_512)["watts"]
    w16 = em.predict(em.TEACHER, ceiling, 16, em.DECODE_512)["watts"]
    assert 1.0 <= w16 / w1 <= 1.03


def test_bench_v1_workload_is_the_recorded_batch_one_requests():
    rows = [json.loads(line) for line in SWEEP.read_text(encoding="utf-8").splitlines() if line.strip()]
    work = em.workload_from_rows(rows)
    assert (work.requests_per_stream, work.max_tokens) == (4, 128)
    assert work.prompt_tokens == pytest.approx(em.BENCH_V1.prompt_tokens, abs=1e-9)
    assert work.prefill_s == pytest.approx(em.BENCH_V1.prefill_s, abs=1e-9)


def test_check_on_the_recorded_april_sweep(chip):
    res = em.check(SWEEP, chip=chip)
    by_batch = {p["batch"]: p for p in res["points"]}
    assert sorted(by_batch) == [1, 4, 8, 16, 64]
    assert by_batch[1]["measured"]["decode_tok_s_total"] == pytest.approx(2.6172)   # median of 2.6295, 2.6172, 2.6172
    assert by_batch[16]["measured"]["joules_per_token"] == pytest.approx(3.56281)
    assert all(p["measured"]["n"] == 3 for p in res["points"])
    # anchored on its own batch 1, the model's batch curve: tokens per second within 2 percent at every batch size
    for p in res["points"]:
        assert abs(p["anchored"]["ratio"]["decode_tok_s_total"] - 1.0) <= 0.02, p["batch"]
        assert p["anchored"]["held"]["joules_per_token"] is True
    # as stated (2.88 tok/s, 120 W), the rate holds and the watts do not: this pod drew a flat 96 W in that sweep
    for p in res["points"]:
        assert p["stated"]["held"]["decode_tok_s_total"] is True
        assert p["stated"]["held"]["watts"] is False


def test_check_reads_a_request_sweep_result(tmp_path):
    def cell(batch: int, tok_s: float, jpt: float) -> dict:
        stats = {m: {"median": None} for m in ("decode_tok_s", "watts_avg", "ttft_s", "joules_per_token")}
        stats.update({"decode_tok_s_total": {"median": tok_s}, "joules_per_token": {"median": jpt},
                      "watts_avg": {"median": 110.0}, "ttft_s": {"median": 0.3}})
        return {"id": f"base-b{batch}", "slice": "base", "batch": batch, "thinking": "off", "max_tokens": 128,
                "prompt": "short", "n": 3, "prompt_tokens": 90, "stats": stats}

    result = {"schema": "atlas.sweep_request/1", "idle_w": {"loaded": 10.0},
              "cells": [cell(1, 2.9, 38.0), cell(4, 11.0, 10.0), {**cell(4, 1.0, 1.0), "slice": "tokens-32"}]}
    path = tmp_path / "sweep-request-gfx1151-test.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    points, work = em.measured_points(path)
    assert sorted(points) == [1, 4]
    assert points[4]["decode_tok_s_total"] == 11.0 and points[4]["joules_per_token"] == 10.0
    assert (work.requests_per_stream, work.max_tokens, work.prompt_tokens, work.prefill_s) == (1, 128, 90.0, 0.3)


def test_objective_names_harness_row_fields():
    assert em.OBJECTIVE["metric"] in bench.ROW_KEYS
    assert em.OBJECTIVE["goal"] == "minimize"
    for lane in em.OBJECTIVE["lanes"].values():
        for constraint in lane["outcome_constraints"]:
            assert constraint.split()[0] in bench.ROW_KEYS, constraint


def test_cli_prints_the_report_and_the_check(capsys):
    assert em.main(["--json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert [p["batch"] for p in rep["predictions"]][:4] == [1, 4, 8, 16]
    assert len(rep["predictions"]) == 20      # teacher April and ceiling with two workloads, student decode only
    assert all(math.isfinite(p["joules_per_token"]) for p in rep["predictions"])
    assert em.main(["check", str(SWEEP)]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0].endswith("tolerance 15%")
    assert len([line for line in out.splitlines() if line.startswith("| ") and line[2].isdigit()]) == 5
