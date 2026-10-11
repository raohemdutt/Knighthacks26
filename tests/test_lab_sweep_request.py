"""lab/sweep_request.py, the request-side settings sweep: the grid parser, the cell-to-row mapping, the run on one
spawn, the summary and the rows it writes, the result block, the pod gate and the pod script.

Nothing here touches a GPU. Runs spawn lab/bench_fake.py as a real server process and read a fake Strix Halo pod
(lab.bench_fake.FakeHost and FakePower), the way tests/test_lab_bench.py drives the harness, and assert structure and
values the fake decides (counts, order, conditions, tokens, the fake's watts), never a rate measured on this machine's
clock. The summary arithmetic runs on constructed rows with chosen numbers, and collect runs on recorded harness output
(fixtures/lab/sweep_request/recorded_full.out, written by make_fixture.py). Every server binds port 0 and names its
port in a file, and a watchdog turns a test that runs past WATCHDOG_S into a crash with every thread's traceback."""
from __future__ import annotations

import base64
import dataclasses
import faulthandler
import hashlib
import importlib.util
import io
import json
import re
import socket
import subprocess
import sys
import tarfile
import gzip
from pathlib import Path

import pytest

from lab import __main__ as lab_cli
from lab import bench, bench_fake
from lab import sweep_request as sr
from lab.machine import ledger, metrics

REPO = Path(__file__).resolve().parents[1]
RECORDED = REPO / "fixtures" / "lab" / "sweep_request" / "recorded_full.out"
PROMPTS = REPO / "fixtures" / "lab" / "bench" / "sweep"
WATCHDOG_S = 120.0
TOKENS = 8
GPU_SHA = hashlib.sha256(b"roscon02").hexdigest()


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("LAB_TASK", raising=False)
    faulthandler.dump_traceback_later(WATCHDOG_S, exit=True, file=sys.__stderr__)
    yield
    faulthandler.cancel_dump_traceback_later()


def committed() -> dict:
    return json.loads(sr.DEFAULT_GRID.read_text(encoding="utf-8"))


def tiny_data(**changes) -> dict:
    """The committed grid cut down for the fake: batches 1 and 2, two slices, 8 tokens, short waits."""
    data = committed()
    data.update({"batches": [1, 2], "repeats": 3, "defaults": {**data["defaults"], "max_tokens": TOKENS},
                 "slices": [{"id": "base", "set": {}}, {"id": "long", "set": {"prompt": "long"}}],
                 "quick": ["base"], "baseline_cell": "base-b1", "warmup": {"max_tokens": 4}})
    data["server"] = {**data["server"], "port": 0, "idle_s": 0.05, "ready_timeout_s": 30, "request_timeout_s": 20,
                      "stream_grace_s": 5}
    data.update(changes)
    return data


def fake_launch(root: Path, name: str = "fake", **options) -> sr.Launch:
    argv = bench_fake.serve_command(root=str(root), set_env=["GPU_MAX_HW_QUEUES=4"], **options)
    return sr.Launch(name, tuple(argv), {}, ())


def broken_launch(code: int = 3) -> sr.Launch:
    return sr.Launch("broken", (bench_fake.spawn_python(), "-c", f"import sys; sys.exit({code})"), {}, ())


def sweep(tmp_path: Path, grid: sr.Grid | None = None, launches=None, **kwargs):
    root = tmp_path / "pod"
    root.mkdir(parents=True, exist_ok=True)
    host = sr.read_once(bench_fake.FakeHost(root))
    power = bench_fake.FakePower(root)
    grid = grid or sr.parse_grid(tiny_data())
    raw = sr.run_sweep(grid, grid.cells(), launches or [fake_launch(root)], host=host, power=power,
                       out_dir=tmp_path / "runs", checkpoint="/home/jovyan/models/Qwen3.8-27B-INT4",
                       engine_dir="/home/jovyan/vllm", patches=[str(REPO / "lab" / "vllm-gfx1201.diff")], port=0,
                       hub="https://sjc.aupcloud.io", account="D", say=lambda message: None, **kwargs)
    return raw, root, host


def synthetic_raw(totals: dict[str, list[float]], *, extra_rows=(), served="pinned", tried=None) -> dict:
    """A run as run_sweep returns it, with only the fields the summary reads: each listed cell's repeats with the
    given aggregate decode rates, the per-stream rate that total over the batch, 100 W and 100 J over the window."""
    grid = sr.parse_grid(committed())
    cells = [c for c in grid.cells() if c.id in totals]
    rows = []
    for cell in cells:
        for repeat, total in enumerate(totals[cell.id]):
            rows.append({"cell": cell.id, "repeat": repeat, "order": len(rows), "run_id": f"run-{cell.id}-r{repeat}",
                         "outcome": "ok", "reason": None, "decode_tok_s": total / cell.batch,
                         "decode_tok_s_total": total, "ttft_s": 0.5, "joules_per_token": 100.0 / total,
                         "joules_per_token_above_idle": 90.0 / total, "watts": {"avg_w": 100.0, "peak_w": 120.0},
                         "tokens_generated": cell.batch * cell.max_tokens, "window_s": 10.0,
                         "in_flight_mean": float(cell.batch), "elapsed_s": 11.0,
                         "requests": [{"prompt_tokens": 60}] * cell.batch,
                         "conditions": {"host": {"chip": "gfx1151"}, "engine_commit": "b1388b1", "checkpoint": "ckpt",
                                        "kv_cache_tokens": 163840, "env": {"GPU_MAX_HW_QUEUES": "4"},
                                        "unavailable": {}}})
    return {"schema": sr.RAW_SCHEMA, "harness": bench.HARNESS, "run_id": "sweep-test", "node": "pod-b",
            "hub": "https://hub.test", "account": "D", "quick": False, "started": "2026-10-11T01:00:00.000Z",
            "ended": "2026-10-11T01:50:00.000Z", "grid": {"file": "grid.json", "sha256": "f" * 64, "data": grid.data},
            "cells": [dataclasses.asdict(c) for c in cells],
            "launch": {"served": served, "tried": tried or [{"launch": served, "error": None}]},
            "spawn": {"ready_s": 300.0, "pid": 7, "stopped": True}, "idle_w": {"no_server": 9.0, "loaded": 10.0},
            "power": {"available": True, "reason": None, "period_s": 0.5}, "warmup": [], "estimate": {},
            "deadline_s": None, "rows": rows + list(extra_rows), "error": None}


TOTALS = {"base-b1": [2.8, 2.9, 3.0], "base-b16": [30.0, 33.0, 36.0], "tokens-32-b1": [2.7, 2.75, 2.8]}


# --- the grid -----------------------------------------------------------------------------------------------------

def test_the_committed_grid_crosses_every_slice_with_batches_1_4_8_16_three_repeats_each():
    grid = sr.load_grid()
    assert grid.batches == (1, 4, 8, 16) and grid.repeats == 3 and grid.order == "serpentine"
    cells = grid.cells()
    assert [c.id for c in cells] == [f"{s}-b{b}" for s in ("base", "tokens-32", "thinking-on-32", "prompt-long")
                                     for b in (1, 4, 8, 16)]
    assert {(c.slice, c.thinking, c.max_tokens, c.prompt, c.sampling) for c in cells} == {
        ("base", "off", 128, "short", "greedy"), ("tokens-32", "off", 32, "short", "greedy"),
        ("thinking-on-32", "on", 32, "short", "greedy"), ("prompt-long", "off", 128, "long", "greedy")}
    assert all(c.requests_per_stream == 1 for c in cells)
    assert [c.id for c in grid.cells(quick=True)] == ["base-b1", "base-b4", "base-b8", "base-b16"]
    baseline = next(c for c in cells if c.id == grid.baseline_cell)
    assert (baseline.batch, baseline.thinking, baseline.max_tokens, baseline.prompt) == (1, "off", 128, "short")
    assert grid.sampling["greedy"]["temperature"] == 0.0 and grid.best_by == "decode_tok_s_total"
    assert grid.file == "lab/refine/research/kernels/data/sweep-request-grid.json"
    assert grid.sha256 == hashlib.sha256(sr.DEFAULT_GRID.read_bytes()).hexdigest()


def test_every_slice_and_held_setting_says_why_and_sets_only_request_settings():
    data = committed()
    for one in data["slices"]:
        assert set(one["set"]) <= set(sr.CELL_KEYS) and one["why"], one["id"]
    assert {h["setting"] for h in data["held"]} >= {"sampling"}
    assert all(h["why"] for h in data["held"])


def test_a_grid_with_problems_is_refused_with_every_problem_named():
    data = committed()
    data["batches"] = [1, 0]
    data["slices"] = [*data["slices"], {"id": "queues", "set": {"gpu_max_hw_queues": 1}},
                      {"id": "base", "set": {"max_tokens": 64}}]
    data["sampling"] = {"greedy": {"temperature": 0.0, "top_p": 2.0}}
    data["quick"] = ["tokens-32"]
    data["best_by"] = "speed"
    with pytest.raises(sr.SweepError) as caught:
        sr.parse_grid(data)
    text = str(caught.value)
    for part in ("batches is a list", "'gpu_max_hw_queues', which is not a request setting", "slice id 'base' is used twice",
                 "top_p is above 0", "baseline_cell 'base-b1' is not a cell", "best_by is one of"):
        assert part in text, part


def test_two_slices_that_give_the_same_request_settings_are_refused():
    data = committed()
    data["slices"].append({"id": "again", "set": {"max_tokens": 128}})
    with pytest.raises(sr.SweepError, match="'base' and 'again' give the same request settings"):
        sr.parse_grid(data)


def test_a_prompt_class_must_be_a_file_in_the_checkout():
    data = committed()
    data["prompts"]["long"]["file"] = "fixtures/lab/bench/sweep/missing.json"
    with pytest.raises(sr.SweepError, match="missing.json is not in this checkout"):
        sr.parse_grid(data)
    assert sr.parse_grid(data, check_files=False).prompts["long"].file.endswith("missing.json")


def test_repeats_run_interleaved_forward_backward_forward():
    grid = sr.parse_grid(tiny_data())
    order = [(r, c.id) for r, c in sr.run_order(grid, grid.cells())]
    forward = ["base-b1", "base-b2", "long-b1", "long-b2"]
    assert order == [(0, c) for c in forward] + [(1, c) for c in reversed(forward)] + [(2, c) for c in forward]
    flat = sr.parse_grid(tiny_data(order="forward"))
    assert [c.id for _, c in sr.run_order(flat, flat.cells())] == forward * 3


def test_a_cell_takes_its_prefill_and_its_tokens_at_the_rate_its_batch_gets():
    data = tiny_data()
    data["timing"] = {**data["timing"], "decode_tok_s_b1": 2.0, "per_stream_slowdown": 0.5, "ttft_s": 0.5,
                      "prefill_tok_s": 100.0, "per_cell_overhead_s": 1.0}
    data["prompts"]["short"]["tokens_est"] = 100
    grid = sr.parse_grid(data)
    one, two = grid.cells()[:2]                                           # base-b1, base-b2 at 8 tokens
    assert sr.cell_seconds(grid, one) == pytest.approx(1.0 + 0.5 + 1 * 100 / 100.0 + 8 / 2.0)
    assert sr.cell_seconds(grid, two) == pytest.approx(1.0 + 0.5 + 2 * 100 / 100.0 + 8 / (2.0 / 1.5))
    assert sr.cell_seconds(grid, one, tokens=4) == pytest.approx(1.0 + 0.5 + 1.0 + 2.0)


def test_the_slot_minutes_of_the_full_and_quick_grids():
    grid = sr.load_grid()
    full, quick = sr.estimate(grid, grid.cells()), sr.estimate(grid, grid.cells(quick=True))
    assert full["slot_minutes"] == 55 and quick["slot_minutes"] == 30
    for e in (full, quick):
        assert e["total_s"] == pytest.approx(e["idle_s"] + e["ready_s"] + e["warmup_s"] + e["measure_s"] + e["stop_s"],
                                             abs=0.2)
        assert e["slot_minutes"] * 60 >= e["total_s"] * (1 + e["margin"])
        assert e["slot_minutes"] % 5 == 0
    assert len(sr.warmup_shapes(grid.cells())) == 8                     # four batch sizes x two prompt classes


# --- the launch ---------------------------------------------------------------------------------------------------

def test_the_pinned_launch_is_the_search_space_baseline_and_the_stock_line_is_the_fallback():
    pinned, stock = sr.grid_launches(sr.load_grid(), run_dir="/home/jovyan/sweep_request/server")
    argv = " ".join(pinned.argv)
    for flags in ("--dtype bfloat16", "--attention-backend ROCM_ATTN", "--enable-prefix-caching",
                  "--mamba-ssm-cache-dtype float32", '--compilation-config {"cudagraph_mode":"FULL_AND_PIECEWISE"}',
                  "--served-model-name teacher", "--max-model-len 16384", "--gpu-memory-utilization 0.90",
                  "--mm-encoder-attn-backend TRITON_ATTN", "--max-num-seqs 256", "--max-num-batched-tokens 2048",
                  "--kv-cache-dtype auto", "--port {port}", "--model {checkpoint}", "--host 127.0.0.1"):
        assert flags in argv, flags
    assert "--speculative-config" not in argv and "--enforce-eager" not in argv
    assert pinned.env == {"VLLM_ROCM_USE_AITER": "0", "AMD_SERIALIZE_KERNEL": "0", "HIP_LAUNCH_BLOCKING": "0",
                          "PYTORCH_TUNABLEOP_ENABLED": "0", "GPU_MAX_HW_QUEUES": "4", "VLLM_ROCM_USE_SKINNY_GEMM": "1",
                          "VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE": "1", "AMDGCN_USE_BUFFER_OPS": "0",
                          "TRITON_CACHE_DIR": "/home/jovyan/sweep_request/server/triton-cache"}
    assert pinned.argv[0] == "env" and pinned.argv[1:3] == ("-u", "VLLM_DISABLED_KERNELS")
    assert "HSA_ENABLE_SDMA" in pinned.unset and "PYTORCH_HIP_ALLOC_CONF" in pinned.unset
    assert pinned.argv.index("python3") == 2 * len(pinned.unset) + 1
    assert list(stock.argv) == bench.default_command("vllm", "gfx1151") and stock.env == {}


def test_a_search_space_with_no_launch_for_its_baseline_is_refused():
    space = json.loads((REPO / committed()["server"]["search_space"]).read_text(encoding="utf-8"))
    del space["mapping"]["attention_backend"]["ROCM_ATTN"]
    with pytest.raises(sr.SweepError, match="no launch for attention_backend = ROCM_ATTN"):
        sr.pinned_launch(space, model="teacher", run_dir="/r")
    space = json.loads((REPO / committed()["server"]["search_space"]).read_text(encoding="utf-8"))
    with pytest.raises(sr.SweepError, match="serves"):
        sr.pinned_launch(space, model="other", run_dir="/r")


# --- a cell on the harness ----------------------------------------------------------------------------------------

def test_a_cell_maps_to_the_harness_config_and_to_its_request_body():
    grid = sr.load_grid()
    cell = next(c for c in grid.cells() if c.id == "thinking-on-32-b4")
    config = sr.cell_config(grid, cell, bench.Config(model="teacher", node="pod-b"))
    assert (config.batches, config.max_tokens, config.thinking, config.requests_per_stream) == ([4], 32, True, 1)
    assert config.prompts == PROMPTS / "short.json" and config.node == "pod-b"
    prompts = bench.PromptSet.load(config.prompts)
    settings = config.settings()
    greedy = sr.SampledPromptSet.wrap(prompts, grid.sampling["greedy"])
    assert greedy.request_body(2, settings, "n-1") == prompts.request_body(2, settings, "n-1")
    body = greedy.request_body(2, settings, "n-1")
    assert body["max_tokens"] == 32 and body["ignore_eos"] is True and body["temperature"] == 0.0
    assert body["chat_template_kwargs"] == {"enable_thinking": True} and "n-1" in json.dumps(body["messages"])
    sampled = sr.SampledPromptSet.wrap(prompts, {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "why": "x"})
    body = sampled.request_body(0, settings, None)
    assert (body["temperature"], body["top_p"], body["top_k"]) == (0.7, 0.8, 20) and "why" not in body


def test_the_prompt_classes_are_text_only_written_by_their_maker_and_long_is_long():
    short, long = (bench.PromptSet.load(PROMPTS / name) for name in ("short.json", "long.json"))
    assert {p["kind"] for p in short.prompts + long.prompts} == {"text"}
    assert (short.id, long.id) == ("sweep-short-v1", "sweep-long-v1")
    assert len(short.prompts) == len(long.prompts) == 4
    assert min(len(p["user"]) for p in long.prompts) > 10 * max(len(p["user"]) for p in short.prompts)
    spec = importlib.util.spec_from_file_location("make_prompts", PROMPTS / "make_prompts.py")
    maker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(maker)
    assert maker.main(["--check"]) == 0


def test_fixed_facts_are_read_once_and_the_watts_every_time(tmp_path):
    host = sr.read_once(bench_fake.FakeHost(tmp_path))
    driver = ["/opt/rocm/bin/amd-smi", "static", "--driver"]
    first, second = host.command(driver), host.command(driver)
    assert first == second and host.calls.count(driver) == 1
    metric = ["/opt/rocm/bin/amd-smi", "metric", "--power"]
    host.run_text(metric)
    host.run_text(metric)
    assert host.calls.count(metric) == 2
    with pytest.raises(bench.HostError):
        host.command(["rocprofv3", "--version"])                          # not a fixed fact: asked each time
    assert sr.read_once(host) is host and isinstance(host.run, sr.ReadOnce)


# --- one run on the fake ------------------------------------------------------------------------------------------

def test_one_spawn_serves_every_cell_each_repeat_in_the_interleaved_order(tmp_path):
    raw, root, host = sweep(tmp_path)
    rows = raw["rows"]
    assert raw["error"] is None and raw["launch"]["served"] == "fake"
    assert len((root / "servers.jsonl").read_text().splitlines()) == 1     # exactly one server spawned
    assert [(r["repeat"], r["cell"]) for r in rows] == [(r, c) for r, c in raw["order"]]
    assert len(rows) == 12 and all(r["outcome"] == "ok" for r in rows), [r["reason"] for r in rows]
    assert {r["spawn"]["pid"] for r in rows} == {raw["spawn"]["pid"]} and raw["spawn"]["stopped"] is True
    assert raw["idle_w"] == {"no_server": bench_fake.IDLE_NO_SERVER_W, "loaded": bench_fake.IDLE_LOADED_W}
    for row in rows:
        batch = row["batch_size"]
        assert row["tokens_generated"] == batch * TOKENS and len(row["requests"]) == batch
        assert all(q["role"] == "measure" and q["tokens"] == TOKENS for q in row["requests"])
        c = row["conditions"]
        assert c["sweep"]["cell"] == row["cell"] and c["sweep"]["run_id"] == raw["run_id"]
        assert c["sweep"]["shared_spawn"] is True and c["sweep"]["launch"] == "fake"
        assert c["batch_size"] == batch and c["max_tokens"] == TOKENS and c["thinking"] is False
        assert c["prompt_set"]["id"] == ("sweep-long-v1" if row["cell"].startswith("long") else "sweep-short-v1")
        assert c["engine_commit"] == bench_fake.FAKE_COMMIT and c["host"]["chip"] == "gfx1151"
        assert c["env"]["GPU_MAX_HW_QUEUES"] == "4" and c["kv_cache_tokens"] == 163_840
        assert c["sampling"] == {"profile": "greedy", "temperature": 0.0, "top_p": 1.0} and c["temperature"] == 0.0
        assert row["warmup"]["requests"] == 0 and row["probe"]["reason"] == "no probe in this run"
        w = row["watts"]
        assert w["available"] is True and bench_fake.IDLE_NO_SERVER_W <= w["avg_w"] <= w["peak_w"]
        assert w["peak_w"] <= bench_fake.IDLE_LOADED_W + bench_fake.WATTS_PER_STREAM * batch
        assert row["run_id"] == f"{raw['run_id']}-{row['cell']}-r{row['repeat']}-b{batch}"
    longs = [q["prompt_tokens"] for r in rows if r["cell"].startswith("long") for q in r["requests"]]
    shorts = [q["prompt_tokens"] for r in rows if r["cell"].startswith("base") for q in r["requests"]]
    assert min(longs) > 5 * max(shorts)
    assert [(w["batch"], w["prompt"], w["requests"], w["max_tokens"]) for w in raw["warmup"]] == [
        (1, "short", 1, 4), (2, "short", 2, 4), (1, "long", 1, 4), (2, "long", 2, 4)]
    assert host.calls.count(["/opt/rocm/bin/amd-smi", "static", "--driver"]) == 1
    saved = json.loads((Path(raw["run_dir"]) / "raw.json").read_text(encoding="utf-8"))
    assert saved["run_id"] == raw["run_id"] and len(saved["rows"]) == 12
    for cell in raw["cells"]:
        lines = (Path(raw["run_dir"]) / "cells" / f"{raw['run_id']}-{cell['id']}" / "rows.jsonl").read_text()
        assert len(lines.splitlines()) == 3


def test_a_launch_that_dies_falls_back_to_the_next_and_the_summary_says_so(tmp_path):
    root = tmp_path / "pod"
    raw, _, _ = sweep(tmp_path, launches=[broken_launch(3), fake_launch(root)])
    assert raw["launch"]["served"] == "fake" and len(raw["rows"]) == 12
    first = raw["launch"]["tried"][0]
    assert first["launch"] == "broken" and "exited with code 3" in first["error"]
    assert raw["launch"]["tried"][1]["error"] is None
    assert any("the broken launch did not come up" in item for item in sr.summarize(raw)["open"])


def test_no_launch_coming_up_is_the_run_error_and_every_cell_fails(tmp_path):
    raw, _, _ = sweep(tmp_path, launches=[broken_launch(3), broken_launch(4)])
    assert raw["rows"] == [] and raw["error"].startswith("no launch came up")
    summary = sr.summarize(raw)
    assert [c["outcome"] for c in summary["cells"]] == ["fail"] * 4
    assert all(c["reasons"] == [f"not measured: {raw['error']}"] for c in summary["cells"])
    assert summary["best"] is None


def test_a_cell_the_deadline_cannot_hold_is_a_skipped_row_with_the_reason(tmp_path):
    data = tiny_data()
    data["prompts"]["long"]["tokens_est"] = 10 ** 9                          # the model says a long cell never ends
    raw, _, _ = sweep(tmp_path, grid=sr.parse_grid(data), deadline_s=600.0)
    by_cell = {}
    for row in raw["rows"]:
        by_cell.setdefault(row["cell"], []).append(row)
    assert [r["outcome"] for c in ("base-b1", "base-b2") for r in by_cell[c]] == ["ok"] * 6
    for c in ("long-b1", "long-b2"):
        assert [r["outcome"] for r in by_cell[c]] == ["skipped"] * 3
        assert all(r["reason"].startswith("not run: past the deadline") for r in by_cell[c])
    assert raw["error"] is None and raw["spawn"]["stopped"] is True
    cells = {c["id"]: c for c in sr.summarize(raw)["cells"]}
    assert cells["long-b1"]["outcome"] == "fail" and cells["base-b2"]["outcome"] == "ok"


def test_stuck_streams_end_the_measuring_and_the_rest_are_not_measured(tmp_path):
    data = tiny_data()
    data["server"] = {**data["server"], "request_timeout_s": 2, "stream_grace_s": 0}
    root = tmp_path / "pod"
    # stall_tok_s=0: the measured streams of that cell send one token and then only keep-alives, so the harness's
    # bound (2 x 2 s) is what ends them whatever the test box is doing; the other cells' eight tokens take milliseconds
    raw, _, _ = sweep(tmp_path, grid=sr.parse_grid(data),
                      launches=[fake_launch(root, stall_match="r1-b2-measure-", stall_tok_s=0.0)])
    rows = raw["rows"]
    assert [r["outcome"] for r in rows[:4]] == ["ok"] * 4
    stuck = rows[4]
    assert (stuck["cell"], stuck["repeat"], stuck["outcome"]) == ("long-b2", 1, "fail")
    assert "had not finished" in stuck["reason"]
    assert all(r["outcome"] == "fail" and r["reason"].startswith("not measured: ") for r in rows[5:])
    assert len(rows) == 12 and "had stuck streams" in raw["error"] and raw["spawn"]["stopped"] is True


# --- the summary (constructed rows with chosen numbers) -------------------------------------------------------------

def test_each_cell_is_the_median_of_its_repeats_with_its_spread_and_its_change_against_the_baseline():
    summary = sr.summarize(synthetic_raw(TOTALS))
    cells = {c["id"]: c for c in summary["cells"]}
    base = cells["base-b1"]["stats"]["decode_tok_s_total"]
    assert (base["n"], base["median"], base["min"], base["max"]) == (3, 2.9, 2.8, 3.0)
    assert base["spread_pct"] == pytest.approx(0.2 / 2.9 * 100, abs=1e-5)
    assert cells["base-b16"]["stats"]["decode_tok_s"]["median"] == pytest.approx(33.0 / 16)
    assert cells["tokens-32-b1"]["vs_baseline_pct"]["decode_tok_s_total"] == pytest.approx((2.75 / 2.9 - 1) * 100,
                                                                                            abs=1e-3)
    assert cells["base-b1"]["vs_baseline_pct"]["decode_tok_s_total"] == 0.0
    assert cells["base-b16"]["vs_baseline_pct"]["ttft_s"] == 0.0 and cells["base-b16"]["prompt_tokens"] == 60
    assert all(c["outcome"] == "ok" and c["n"] == 3 for c in summary["cells"])
    assert summary["baseline"] == "base-b1" and summary["date"] == "2026-10-11" and summary["chip"] == "gfx1151"


def test_the_best_cell_is_named_with_its_gain_over_the_baseline_and_the_range_the_spreads_allow():
    summary = sr.summarize(synthetic_raw(TOTALS))
    best = summary["best"]
    assert best["cell"] == "base-b16" and best["metric"] == "decode_tok_s_total" and best["higher_is_better"]
    assert best["change_pct"] == pytest.approx((33.0 / 2.9 - 1) * 100, abs=1e-3)
    assert best["range_pct"] == [pytest.approx((30.0 / 3.0 - 1) * 100), pytest.approx((36.0 / 2.8 - 1) * 100, abs=1e-3)]
    assert summary["lowest_joules_per_token"]["cell"] == "base-b16"
    assert summary["lowest_joules_per_token"]["value"]["median"] == pytest.approx(100.0 / 33.0, abs=1e-5)
    assert summary["open"] == []


def test_a_cell_short_of_its_repeats_is_partial_and_stays_open():
    skipped = {"cell": "tokens-32-b1", "repeat": 2, "order": 99, "run_id": "run-tokens-32-b1-r2",
               "outcome": "skipped", "reason": "not run: past the deadline", "watts": None, "requests": []}
    raw = synthetic_raw({**TOTALS, "tokens-32-b1": [2.7, 2.8]}, extra_rows=[skipped])
    cells = {c["id"]: c for c in sr.summarize(raw)["cells"]}
    cell = cells["tokens-32-b1"]
    assert (cell["outcome"], cell["n"], cell["reasons"]) == ("partial", 2, ["not run: past the deadline"])
    assert cell["stats"]["decode_tok_s_total"]["median"] == pytest.approx(2.75)
    assert [r["outcome"] for r in cell["repeats"]] == ["ok", "ok", "skipped"]
    assert any(item.startswith("tokens-32-b1: 2 of 3 repeats measured") for item in sr.summarize(raw)["open"])


def test_the_markdown_table_has_a_row_per_cell_against_the_baseline_and_names_the_best():
    text = sr.render_markdown(sr.summarize(synthetic_raw(TOTALS)))
    rows = [line for line in text.splitlines() if line.startswith("| `")]
    assert len(rows) == 3
    assert rows[0].startswith("| `base-b1` (baseline) | 1 | off | 128 | short (60) | 2.90 (2.80-3.00) |")
    assert "| `base-b16` (best) | 16 |" in rows[1] and "| +1037.9% | 3 of 3 |" in rows[1]
    assert "Best cell by decode_tok_s_total: `base-b16` at 33.00 (30.00-36.00) against the baseline's 2.90 " \
           "(2.80-3.00): +1037.9% over the baseline, +900.0% to +1185.7%" in text
    assert "Held: sampling = greedy:" in text and "Lowest joules per token: `base-b16`" in text


def test_ledger_rows_are_one_per_cell_on_pod_b_with_the_medians():
    summary = sr.summarize(synthetic_raw(TOTALS))
    rows = sr.ledger_rows(summary)
    assert [r["conditions"]["cell"] for r in rows] == ["base-b1", "base-b16", "tokens-32-b1"]
    for row in rows:
        assert row["verb"] == "sweep_request.cell" and row["node"] == "pod-b" and row["low_confidence"] is True
        assert set(row["metrics"]) == {f"teacher.{m}" for m in sr.METRICS}
        assert row["units"]["teacher.decode_tok_s_total"] == "tok/s" and row["units"]["teacher.ttft_s"] == "s"
        assert row["refs"] == {"run_id": f"sweep-test/{row['conditions']['cell']}"}
        assert row["conditions"]["statistic"] == "median of 3 repeats on one shared spawn"
        ledger.validate({**row, "actor": "test", "tier": None, "task": "R52"})
    assert rows[1]["metrics"]["teacher.decode_tok_s_total"] == 33.0 and rows[1]["conditions"]["batch_size"] == 16


def test_metric_rows_are_the_baseline_and_best_headlines_and_the_derived_change(tmp_path):
    rows = sr.metric_rows(sr.summarize(synthetic_raw(TOTALS)))
    path = tmp_path / "metrics.jsonl"
    for row in rows:
        metrics.add(row, path=path)
    written = metrics.load(path)
    assert len(written) == 11
    assert [(r["metric"], r["conditions"].get("cell", r["conditions"].get("from"))) for r in written] == [
        *[(f"teacher.{m}", "base-b1") for m in sr.HEADLINE], *[(f"teacher.{m}", "base-b16") for m in sr.HEADLINE],
        ("teacher.decode_tok_s_total_change_pct", "base-b16")]
    assert all(r["node"] == "pod-b" and r["low_confidence"] is True for r in written)
    assert all(r["ts"] == "2026-10-11T01:50:00Z" and r["source"] == "lab/sweep_request.py 1 sweep-test" for r in written)
    assert written[0]["conditions"]["role"] == "baseline" and written[5]["conditions"]["role"] == "best"
    assert "under the 5 that verdict()" in written[0]["conditions"]["low_confidence_reason"]
    change = written[-1]
    assert change["unit"] == "%" and change["value"] == pytest.approx((33.0 / 2.9 - 1) * 100, abs=1e-3)
    assert change["conditions"]["derived"] is True and change["conditions"]["against"] == "base-b1"


def test_when_the_baseline_is_the_best_cell_its_rows_are_written_once_and_no_change_row():
    rows = sr.metric_rows(sr.summarize(synthetic_raw({"base-b1": [5.0, 5.0, 5.0], "base-b4": [1.0, 2.0, 3.0]})))
    assert [r["metric"] for r in rows] == [f"teacher.{m}" for m in sr.HEADLINE]
    assert {r["conditions"]["role"] for r in rows} == {"baseline and best"}


# --- the result block -----------------------------------------------------------------------------------------------

def test_the_result_block_round_trips_through_saved_output_in_utf8_and_utf16(tmp_path):
    raw = synthetic_raw(TOTALS)
    raw["run_dir"] = str(tmp_path)
    (tmp_path / "serve-0-pinned.log").write_text("\n".join(f"line {i}" for i in range(500)), encoding="utf-8")
    text = sr.report(raw)
    assert text.splitlines()[0].startswith("sweep sweep-test: 3 of 3 cells measured in full, launch pinned")
    assert "best base-b16 by decode_tok_s_total: 33.00 (+1037.9% over the baseline" in text
    for blob in (text.encode("utf-8"), text.encode("utf-16"), ("﻿" + text).encode("utf-8"),
                 text.replace("\n", "\r\n").encode("utf-8")):
        files = sr.read_bundle(blob)
        assert json.loads(files["raw.json"])["rows"] == json.loads(json.dumps(raw["rows"]))
        log = files["serve-0-pinned.log"].decode()
        assert log.startswith("line 0\n") and "lines left out" in log and log.rstrip().endswith("line 499")
    lines = text.splitlines()
    k = next(i for i, line in enumerate(lines) if line.startswith(sr.MARK_BEGIN)) + 3
    lines[k] = ("A" if lines[k][0] != "A" else "B") + lines[k][1:]
    with pytest.raises(sr.SweepError, match="damaged in transit"):
        sr.read_bundle("\n".join(lines).encode())
    with pytest.raises(sr.SweepError, match="did not finish"):
        sr.read_bundle(b"the pod printed nothing useful")


def test_the_bundle_is_the_same_bytes_for_the_same_files():
    files = {"b.txt": b"two", "a.txt": b"one"}
    assert sr.make_bundle(files) == sr.make_bundle(dict(reversed(list(files.items()))))
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(sr.make_bundle(files)))) as tar:
        assert [m.name for m in tar.getmembers()] == ["a.txt", "b.txt"]


# --- collect --------------------------------------------------------------------------------------------------------

def test_collect_writes_the_files_and_rows_once_and_again_only_with_force(tmp_path):
    raw = synthetic_raw(TOTALS)
    raw["run_dir"] = str(tmp_path)
    saved = tmp_path / "pod_run.out"
    saved.write_text("[sweep 01:00:00] progress\n" + sr.report(raw) + '{"ok": true}\n', encoding="utf-8")
    out, metrics_path = tmp_path / "data", tmp_path / "metrics.jsonl"
    argv = ["collect", str(saved), "--out-dir", str(out), "--state-dir", str(tmp_path / "kept"), "--metrics-file",
            str(metrics_path)]
    answer = sr.main(argv)
    assert answer["ok"] is True and answer["outcome"] == "collected", answer
    assert answer["best"] == "base-b16" and answer["ledger_rows"] == 3 and answer["metrics_rows"] == 11
    data = json.loads((out / "sweep-request-gfx1151-2026-10-11.json").read_text(encoding="utf-8"))
    assert data["schema"] == sr.RESULT_SCHEMA and data["run_id"] == "sweep-test" and len(data["cells"]) == 3
    assert (out / "sweep-request-gfx1151-2026-10-11.md").read_text(encoding="utf-8").startswith(
        "# Request-side settings sweep on gfx1151 (pod-b), 2026-10-11")
    assert (tmp_path / "kept" / "sweep-test" / "raw.json").exists()
    rows = ledger.read(state_dir=tmp_path / "state")
    cells = [r for r in rows if r["verb"] == "sweep_request.cell"]
    assert len(cells) == 3 and {r["node"] for r in cells} == {"pod-b"} and {r["task"] for r in cells} == {"R52"}
    call = [r for r in rows if r["verb"] == "sweep-request"]
    assert len(call) == 1 and call[0]["metrics"]["teacher.decode_tok_s_total.best"] == 33.0
    again = sr.main(argv)
    assert again["ok"] is False and again["outcome"] == "already_collected"
    assert len(metrics.load(metrics_path)) == 11
    forced = sr.main([*argv, "--force"])
    assert forced["ok"] is True and len(metrics.load(metrics_path)) == 22


def test_a_second_run_on_the_same_date_gets_its_own_file(tmp_path):
    out = tmp_path / "data"
    for run_id in ("sweep-one", "sweep-two"):
        raw = synthetic_raw(TOTALS)
        raw.update(run_id=run_id, run_dir=str(tmp_path))
        saved = tmp_path / f"{run_id}.out"
        saved.write_text(sr.report(raw), encoding="utf-8")
        answer = sr.main(["collect", str(saved), "--out-dir", str(out), "--state-dir", str(tmp_path / "kept"),
                          "--no-ledger"])
        assert answer["ok"] is True and answer["ledger_rows"] == 0
    assert sorted(p.name for p in out.glob("*.json")) == ["sweep-request-gfx1151-2026-10-11-2.json",
                                                          "sweep-request-gfx1151-2026-10-11.json"]


def test_collect_on_the_recorded_run_of_the_committed_grid(tmp_path):
    assert RECORDED.is_file(), "expected the recorded harness output fixtures/lab/sweep_request/recorded_full.out"
    files = sr.read_bundle(RECORDED.read_bytes())
    raw = json.loads(files["raw.json"])
    assert raw["grid"]["sha256"] == hashlib.sha256(sr.DEFAULT_GRID.read_bytes()).hexdigest(), \
        "the grid changed since the fixture was recorded: run fixtures/lab/sweep_request/make_fixture.py"
    assert raw["harness"] == bench.HARNESS and len(raw["rows"]) == 48
    answer = sr.main(["collect", str(RECORDED), "--out-dir", str(tmp_path / "data"), "--state-dir",
                      str(tmp_path / "kept"), "--metrics-file", str(tmp_path / "metrics.jsonl")])
    assert answer["ok"] is True and answer["cells"] == answer["cells_ok"] == 16, answer
    summary = json.loads((tmp_path / "data" / Path(answer["json"]).name).read_text(encoding="utf-8"))
    assert summary["baseline"] == "base-b1" and summary["node"] == "pod-b" and summary["launch"]["served"] == "fake"
    medians = {c["id"]: c["stats"]["decode_tok_s_total"]["median"] for c in summary["cells"]}
    assert summary["best"]["cell"] == max(medians, key=medians.get) and summary["best"]["change_pct"] > 0
    assert next(c for c in summary["cells"] if c["id"] == summary["best"]["cell"])["batch"] > 1
    for cell in summary["cells"]:
        assert cell["n"] == 3 and [r["tokens_generated"] for r in cell["repeats"]] == [
            cell["batch"] * cell["max_tokens"]] * 3
    assert len([r for r in ledger.read(state_dir=tmp_path / "state") if r["verb"] == "sweep_request.cell"]) == 16
    assert answer["metrics_rows"] == len(metrics.load(tmp_path / "metrics.jsonl")) == 11
    md = (tmp_path / "data" / Path(answer["markdown"]).name).read_text(encoding="utf-8")
    assert len([line for line in md.splitlines() if line.startswith("| `")]) == 16


# --- the pod gate and the pod script --------------------------------------------------------------------------------

ROCMINFO_1151 = "  Name:                    gfx1151\n  Marketing Name:          AMD Radeon 8060S\n"


def gate(**changes):
    values = {"node": "pod-b", "expect_user_sha256": GPU_SHA, "environ": {"JUPYTERHUB_USER": "roscon02"},
              "rocminfo": ROCMINFO_1151, "machine": "x86_64", "port": 8100, "port_free": True, "servers": []}
    return sr.gate(**{**values, **changes})


@pytest.mark.parametrize(("changes", "problem"), [
    ({"node": "pod-a"}, "the sweep runs on pod-b, not 'pod-a'"),
    ({"machine": "aarch64"}, "this is an aarch64 machine (the Jetson)"),
    ({"rocminfo": "  Name: gfx1201\n"}, "rocminfo names gfx1201, not gfx1151"),
    ({"rocminfo": None}, "cannot tell what GPU this is"),
    ({"environ": {"JUPYTERHUB_USER": "roscon01"}}, "this pod belongs to another hub account"),
    ({"environ": {}}, "JUPYTERHUB_USER is not set"),
    ({"port_free": False}, "something already listens on port 8100"),
    ({"servers": [4242]}, "a vLLM server already runs here (pid 4242)"),
])
def test_the_pod_gate_refuses_every_other_node(changes, problem):
    problems = gate(**changes)
    assert len(problems) == 1 and problems[0].startswith(problem), problems


def test_the_pod_gate_lets_pod_d_through():
    assert gate() == []


def test_pod_run_on_this_machine_is_refused_before_anything_starts(tmp_path):
    answer = sr.main(["pod-run", "--expect-user-sha256", "0" * 64, "--out", str(tmp_path / "runs")])
    assert answer["ok"] is False and answer["outcome"] == "refused"
    assert any(p.startswith("cannot tell what GPU this is") for p in answer["problems"])
    assert not (tmp_path / "runs").exists()


def test_running_servers_reads_command_lines_under_proc(tmp_path):
    for pid, cmd in ((101, b"python3\0-m\0vllm.entrypoints.openai.api_server\0"), (102, b"bash\0")):
        (tmp_path / str(pid)).mkdir()
        (tmp_path / str(pid) / "cmdline").write_bytes(cmd)
    (tmp_path / "self").mkdir()
    assert sr.running_servers(tmp_path) == [101]
    assert sr.running_servers(tmp_path / "absent") == []


def bundle_of(script: str) -> tuple[bytes, str]:
    lines = script.splitlines()
    start = lines.index("base64 -d > \"$B/bundle.tar.gz\" <<'SWEEPBUNDLE'") + 1
    end = lines.index("SWEEPBUNDLE")
    digest = re.search(r'echo "([0-9a-f]{64})  \$B/bundle.tar.gz"', script).group(1)
    return base64.b64decode("".join(lines[start:end])), digest


def test_the_pod_script_carries_the_lab_files_checked_by_sha256_and_no_tab_or_bang():
    grid = sr.load_grid()
    script, info = sr.pod_script(grid, quick=True, expect_user_sha256=GPU_SHA, deadline_min=30,
                                 hub="https://sjc.aupcloud.io", account="D", node="pod-b")
    assert "\t" not in script and "!" not in script and script.isascii()
    assert script.rstrip().endswith('echo "### END rc=$rc"\nexit $rc')
    payload, digest = bundle_of(script)
    assert hashlib.sha256(payload).hexdigest() == digest == info["bundle_sha256"]
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(payload))) as tar:
        names = {m.name for m in tar.getmembers()}
    assert names == set(info["files"])
    assert {"lab/__init__.py", "lab/sweep_request.py", "lab/bench.py", "lab/pod_power.py", "lab/proxy.py",
            "lab/hub.py", "lab/machine/__init__.py", "lab/machine/record.py", "lab/machine/ledger.py",
            "lab/machine/metrics.py", "lab/machine/verdict.py", "lab/vllm-gfx1201.diff",
            "lab/refine/research/kernels/data/sweep-request-grid.json",
            "lab/refine/research/kernels/data/searchspace-inference-gfx1151.json",
            "fixtures/lab/bench/sweep/short.json", "fixtures/lab/bench/sweep/long.json"} <= names
    run_line = next(line for line in script.splitlines() if line.startswith("python3 -m lab.sweep_request "))
    assert run_line == (f"python3 -m lab.sweep_request pod-run --expect-user-sha256 {GPU_SHA} --deadline-min 30 "
                        "--node pod-b --account D --hub https://sjc.aupcloud.io --quick")
    assert "roscon" not in script


def test_the_bundle_alone_runs_the_dry_run_with_the_standard_library_only(tmp_path):
    script, _ = sr.pod_script(sr.load_grid(), quick=False, expect_user_sha256=GPU_SHA, deadline_min=55, hub=None,
                              account="D", node="pod-b")
    payload, _ = bundle_of(script)
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(payload))) as tar:
        for member in tar.getmembers():
            target = tmp_path / "bundle" / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(tar.extractfile(member).read())
    done = subprocess.run([sys.executable, "-S", "-m", "lab.sweep_request", "--dry-run", "--quick"],
                          cwd=tmp_path / "bundle", capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    answer = json.loads(done.stdout.strip().splitlines()[-1])
    assert answer["ok"] is True and answer["n_cells"] == 4 and answer["slot_minutes"] == 30
    assert answer["grid"] == "lab/refine/research/kernels/data/sweep-request-grid.json"


def test_pod_script_writes_the_script_and_names_the_pod_command(tmp_path):
    out = tmp_path / "pod_run.sh"
    answer = sr.main(["pod-script", "--expect-user-sha256", GPU_SHA, "--hub", "https://sjc.aupcloud.io",
                      "--out", str(out)])
    assert answer["ok"] is True and answer["outcome"] == "written" and out.is_file()
    assert answer["slot_minutes"] == answer["deadline_min"] == 55 and answer["n_cells"] == 16
    assert answer["pod_command"] == f"python tooling/dispatcher/pod_exec.py D term {out} 3600 > {out.with_suffix('.out')}"
    assert out.read_bytes().decode("ascii") == sr.pod_script(
        sr.load_grid(), quick=False, expect_user_sha256=GPU_SHA, deadline_min=55, hub="https://sjc.aupcloud.io",
        account="D", node="pod-b")[0]
    dry = sr.main(["pod-script", "--quick", "--dry-run", "--expect-user-sha256", GPU_SHA, "--out",
                   str(tmp_path / "no.sh")])
    assert dry["ok"] is True and dry["script"] is None and not (tmp_path / "no.sh").exists() and dry["n_cells"] == 4


# --- the command line ------------------------------------------------------------------------------------------------

def test_dry_run_prints_the_grid_and_the_slot_minutes(capsys):
    answer = sr.main(["--dry-run"])
    assert answer["ok"] is True and answer["outcome"] == "dry_run"
    assert answer["n_cells"] == 16 and answer["runs"] == 48 and answer["slot_minutes"] == 55
    assert [c["id"] for c in answer["cells"]][:4] == ["base-b1", "base-b4", "base-b8", "base-b16"]
    assert [launch["name"] for launch in answer["launches"]] == ["pinned", "stock"]
    printed = capsys.readouterr().err
    assert "slot minutes: 55" in printed and all(c["id"] in printed for c in answer["cells"])
    quick = sr.main(["--dry-run", "--quick"])
    assert quick["n_cells"] == 4 and quick["slot_minutes"] == 30
    assert sr.main([])["outcome"] == "usage"


def test_the_verb_is_registered_with_the_lab_command_line():
    assert lab_cli.VERBS["sweep-request"] == "lab.sweep_request:main"
    assert list(lab_cli.VERBS) == sorted(lab_cli.VERBS)


def test_the_server_port_is_refused_when_taken_before_any_spawn(tmp_path):
    with socket.socket() as squatter:
        squatter.bind(("127.0.0.1", 0))
        squatter.listen()
        data = tiny_data()
        data["server"] = {**data["server"], "port": squatter.getsockname()[1]}
        root = tmp_path / "pod"
        root.mkdir()
        grid = sr.parse_grid(data)
        raw = sr.run_sweep(grid, grid.cells(), [fake_launch(root)], host=bench_fake.FakeHost(root),
                           power=bench_fake.FakePower(root), out_dir=tmp_path / "runs", checkpoint="ckpt",
                           say=lambda message: None)
    assert raw["rows"] == [] and "already listens" in raw["launch"]["tried"][0]["error"]
    assert not (root / "servers.jsonl").exists()
