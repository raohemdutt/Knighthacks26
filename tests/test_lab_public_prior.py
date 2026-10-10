"""lab/public_prior.py on the committed public Strix Halo benchmark CSV: the data file's shape and the prior's fit."""

from __future__ import annotations

import csv
import json
import math

import pytest

from lab import public_prior as pp

HEADER = (
    "source_url,source_date,source_title,source_kind,quality,machine,chip,gpu,memory_total_gb,memory_split,os,kernel,"
    "driver,rocm_version,vulkan_driver,engine,engine_version,backend,model,params_b,active_params_b,arch,quant_format,"
    "bits_per_weight,model_size_gb,context_length,batch_size,concurrency,prompt_tokens,gen_tokens,settings,pp_tps,tg_tps,"
    "ttft_s,watts,joules_per_token,notes"
).split(",")
NUMERIC = (
    "memory_total_gb params_b active_params_b bits_per_weight model_size_gb context_length batch_size concurrency "
    "prompt_tokens gen_tokens pp_tps tg_tps ttft_s watts joules_per_token"
).split()
QUALITY = {"vendor", "reproducible-script", "review", "forum-post"}


@pytest.fixture(scope="module")
def records() -> list[dict[str, str]]:
    with open(pp.CSV_PATH, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == HEADER
        return list(reader)


@pytest.fixture(scope="module")
def fitted() -> pp.Fit:
    rows, _ = pp.load_rows()
    return pp.fit(rows)


def test_every_row_names_its_source_quality_and_a_rate(records):
    assert len(records) > 10000
    for rec in records:
        assert rec["source_url"].startswith("http"), rec
        assert rec["quality"] in QUALITY, rec["quality"]
        assert rec["pp_tps"] or rec["tg_tps"] or rec["ttft_s"], rec["source_url"]
        if rec["source_date"]:
            assert len(rec["source_date"]) in (7, 10) and rec["source_date"][:2] == "20", rec["source_date"]


def test_numeric_columns_are_numbers_or_empty(records):
    for rec in records:
        for column in NUMERIC:
            if rec[column]:
                assert math.isfinite(float(rec[column])), (column, rec[column])


def test_the_csv_has_no_em_dashes(records):
    for rec in records:
        for value in rec.values():
            assert "\u2014" not in value and "\u2013" not in value


def test_rows_that_are_not_plain_single_node_decode_stay_out_of_the_fit(tmp_path):
    base = {k: "" for k in HEADER}
    base.update(source_url="https://github.com/a/b", quality="forum-post", engine="llama.cpp", backend="Vulkan",
                model="Llama-3.1-8B-Instruct", quant_format="Q4_K_M", model_size_gb="4.6", tg_tps="40")
    variants = [
        {},  # kept
        {"notes": "RPC across two Framework Desktop nodes"},
        {"settings": "--model-draft small.gguf; speculative"},
        {"notes": "tg_tps column holds TOTAL tokens/s (input+output)"},
        {"engine": "halogen-flash"},
        {"model": "GLM-4.5-Air-UD-Q4_K_XL", "model_size_gb": "70"},  # architecture not stated: unknown, not dense
        {"notes": "single stream, no speculative decoding"},  # kept: a negation is not speculative decoding
    ]
    path = tmp_path / "rows.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        for change in variants:
            writer.writerow({**base, **change})
    rows, records = pp.load_rows(path)
    assert len(records) == 7
    assert len(rows) == 3
    assert [r.gb > 0 for r in rows] == [True, False, True]  # the unknown-architecture row feeds the batch term only


def test_bytes_per_token_follow_the_active_share_and_the_size_unit():
    moe = {"model": "Qwen3-30B-A3B-UD-Q4_K_XL", "params_b": "30", "model_size_gb": "17.0", "notes": "size_unit=GiB"}
    assert pp.bytes_per_token_gb(moe) == pytest.approx(17.0 * pp.GIB * 3 / 30)
    scout = {"model": "Llama-4-Scout-17B-16E-Instruct-Q4_K_M", "params_b": "107.77", "model_size_gb": "60"}
    assert pp.bytes_per_token_gb(scout) == pytest.approx(60 * 17 / 107.77)
    dense = {"model": "Qwen3-0.6B", "quant_format": "FP16"}
    assert pp.bytes_per_token_gb(dense) == pytest.approx(0.6 * 16 / 8)
    awq = {"model": "Qwen3.6-27B-AWQ", "params_b": "27", "arch": "dense", "quant_format": "AWQ-INT4 W4A16 g32"}
    assert pp.bytes_per_token_gb(awq) == pytest.approx(27 * 4.5 / 8)
    assert pp.bytes_per_token_gb({"model": "Gemma 4 E2B", "params_b": "2", "arch": "dense"}) is None


def test_depth_comes_from_the_llama_bench_label_not_a_commit_hash():
    assert pp._depth_k({"notes": "env=rocm-7.2; depth d32768; size_unit=GiB"}) == 32.768
    assert pp._depth_k({"notes": "single stream, depth ctx 2048"}) == 2.048
    assert pp._depth_k({"notes": "first published in commit d674531; env=rocm-7.2"}) == 0.0


def test_the_fit_is_bandwidth_shaped_and_reports_its_quality(fitted):
    assert {"vllm", "llama.cpp-rocm", "llama.cpp-vulkan"} <= set(fitted.a)
    assert 0.6 < fitted.beta < 1.2  # near 1 when decode is bandwidth-bound
    assert fitted.g > 0  # a deeper context decodes slower
    assert fitted.rows > 5000 and fitted.sources >= 20
    assert fitted.r2 > 0.7
    assert fitted.loso_median_abs_pct is not None and fitted.loso_median_abs_pct < 40
    report = fitted.as_dict()
    for key in ("r2_log", "rms_log", "typical_factor_error", "leave_one_source_out_median_abs_pct", "batch_k"):
        assert key in report


def test_predictions_fall_with_bytes_and_batching_trades_per_request_for_aggregate(fitted):
    for backend in pp.PREDICT_BACKENDS:
        small = pp.predict(fitted, 2.4, backend)
        large = pp.predict(fitted, 15.5, backend)
        assert small["per_request_tok_s"] > large["per_request_tok_s"]
        previous = None
        for batch in pp.BATCHES:
            p = pp.predict(fitted, 15.5, backend, batch)
            assert p["prior"] is True
            if previous:
                assert p["aggregate_tok_s"] >= previous["aggregate_tok_s"]
                assert p["per_request_tok_s"] <= previous["per_request_tok_s"]
            previous = p


def test_the_table_covers_teacher_and_student_at_every_batch(capsys):
    assert pp.main([]) == 0
    answer = json.loads(capsys.readouterr().out)
    assert answer["ok"] is True and "prior" in answer["note"]
    seen = {(row["target"], row["backend"], row["batch"]) for row in answer["table"] if "error" not in row}
    for target in pp.TARGETS:
        for backend in ("vllm", "llama.cpp-rocm"):
            for batch in (1, 4, 8, 16):
                assert (target, backend, batch) in seen


def test_one_prediction_from_the_command_line(capsys):
    assert pp.main(["--params-b", "4", "--bits", "16", "--backend", "vllm", "--batch", "8"]) == 0
    answer = json.loads(capsys.readouterr().out)
    assert answer["prediction"]["gb_per_token"] == 8.0 and answer["prediction"]["batch"] == 8
    assert pp.main(["--gb", "8", "--backend", "no-such-backend"]) == 1
