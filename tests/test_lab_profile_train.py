"""lab/profile_train.py: the training step profile's trace parser, result, Markdown and ledger rows, on the synthetic
ROCm trace (fixtures/lab/train_profile/rocm, totals per step that can be checked by hand) and on small traces built
here to break one link at a time. No GPU and no training stack are needed."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lab import profile_train as pt
from lab import trace_profile as tp
from lab.machine import ledger

REPO = Path(__file__).resolve().parents[1]
ROCM = REPO / "fixtures" / "lab" / "train_profile" / "rocm"
EXPECTED = json.loads((ROCM / "expected.json").read_text(encoding="utf-8"))


def rocm_data() -> dict:
    return json.loads((ROCM / "trace.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rocm_analysis() -> dict:
    return pt.analyze_trace(pt.TrainTrace(rocm_data()))


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LAB_ACTOR", "opus-test")
    monkeypatch.delenv("LAB_TASK", raising=False)


# --- the synthetic ROCm trace: every hand-checked total comes back ---------------------------------------------------

def test_step_time_and_spread(rocm_analysis):
    step = rocm_analysis["step_ms"]
    assert rocm_analysis["kind"] == "gpu" and rocm_analysis["steps_measured"] == 3
    assert step["mean"] == pytest.approx(EXPECTED["step_ms_mean"])
    assert (step["min"], step["max"]) == (9.5, 10.5)
    assert step["sd"] == pytest.approx(0.5)
    assert rocm_analysis["step_ms_each"] == [10.0, 10.5, 9.5]


@pytest.mark.parametrize("bucket", ["gdn", "attention", "mlp", "lora", "optimizer", "copies", "data_loading"])
def test_ms_per_step_of_each_bucket(rocm_analysis, bucket):
    assert rocm_analysis["ms_per_step"][bucket] == pytest.approx(EXPECTED["ms_per_step"][bucket], abs=1e-3)


@pytest.mark.parametrize("bucket", ["gdn", "attention", "mlp", "lora", "optimizer", "copies", "data_loading"])
def test_share_of_each_bucket(rocm_analysis, bucket):
    assert rocm_analysis["shares"][bucket] == pytest.approx(EXPECTED["shares"][bucket], abs=1e-4)


def test_nothing_falls_to_other_and_idle_is_the_rest(rocm_analysis):
    shares = rocm_analysis["shares"]
    assert shares["other"] == 0.0 and shares["embed_head"] == 0.0
    busy = sum(EXPECTED["ms_per_step"][k] for k in ("gdn", "attention", "mlp", "lora", "optimizer", "copies"))
    assert shares["gpu_idle"] == pytest.approx(1 - busy / EXPECTED["step_ms_mean"], abs=1e-3)


def test_lora_inside_attention_is_counted_once(rocm_analysis):
    # the LoRA range sits inside the attention range: the innermost wins, so attention keeps exactly its own 0.74 ms
    assert rocm_analysis["ms_per_step"]["attention"] == pytest.approx(0.74, abs=1e-4)
    assert rocm_analysis["ms_per_step"]["lora"] == pytest.approx(0.045, abs=1e-4)


def test_every_backward_op_is_linked_and_every_kernel_attributed(rocm_analysis):
    att = rocm_analysis["attribution"]
    assert att["backward_links"] == 12 and att["unattributed_kernels"] == 0
    assert att["kernels"] == 39


def test_the_top_table_has_launch_counts_and_buckets(rocm_analysis):
    top = rocm_analysis["top"]
    assert 0 < len(top) <= pt.TOP
    by_name = {row["name"]: row for row in top}
    assert by_name["fused_adamw_kernel_impl"]["launches"] == 3
    assert by_name["fused_adamw_kernel_impl"]["bucket"] == "optimizer"
    assert by_name["fused_adamw_kernel_impl"]["ms_per_step"] == pytest.approx(0.3)
    assert by_name["Cijk_Alik_Bljk_BBS_BH_MT16x16x16"]["bucket"] == "lora"
    assert top[0]["rank"] == 1 and top[0]["total_ms"] >= top[-1]["total_ms"]


def test_flops_and_memory(rocm_analysis):
    assert rocm_analysis["bandwidth"]["flops_per_step"] == EXPECTED["flops_per_step"]
    dev = rocm_analysis["memory"]["device"]
    assert (dev["peak_allocated_gib"], dev["peak_reserved_gib"]) == (3.0, 4.0)


def test_no_fast_linear_attention_kernel_in_the_stock_trace(rocm_analysis):
    assert rocm_analysis["fast_linear_attention"] == {"present": False, "evidence": []}


# --- break one link at a time ---------------------------------------------------------------------------------------

def test_without_the_backward_links_backward_time_falls_to_other():
    data = rocm_data()
    data["traceEvents"] = [e for e in data["traceEvents"] if e.get("cat") != "fwdbwd"]
    got = pt.analyze_trace(pt.TrainTrace(data))
    assert got["attribution"]["backward_links"] == 0
    assert got["ms_per_step"]["gdn"] == pytest.approx(1.0, abs=1e-3)          # the forward alone
    assert got["ms_per_step"]["other"] == pytest.approx(1.6 + 0.44 + 0.3 + 0.03, abs=1e-3)
    result = pt.build_result(got, None, None, date="2026-10-11", cpu=False)
    assert any("fwdbwd" in item for item in result["open"])


def test_a_kernel_without_its_launch_call_is_counted_unattributed():
    data = rocm_data()
    gone = next(e for e in data["traceEvents"] if e.get("name") == "fused_adamw_kernel_impl")
    corr = gone["args"]["correlation"]
    data["traceEvents"] = [e for e in data["traceEvents"]
                           if not (e.get("cat") == "cuda_runtime" and e["args"]["correlation"] == corr)]
    got = pt.analyze_trace(pt.TrainTrace(data))
    assert got["attribution"]["unattributed_kernels"] == 1
    assert got["ms_per_step"]["optimizer"] == pytest.approx(0.2, abs=1e-3)


def test_a_fla_kernel_name_marks_the_fast_path_present():
    data = rocm_data()
    first = next(e for e in data["traceEvents"] if e.get("cat") == "kernel")
    first["name"] = "chunk_gated_delta_rule_fwd_kernel_h"
    got = pt.analyze_trace(pt.TrainTrace(data))
    assert got["fast_linear_attention"]["present"] is True
    assert got["fast_linear_attention"]["evidence"] == ["chunk_gated_delta_rule_fwd_kernel_h"]
    result = pt.build_result(got, None, None, date="2026-10-11", cpu=False)
    assert result["fast_linear_attention"]["fallback_share"] == 0.0


def test_warmup_steps_are_left_out_of_the_means():
    got = pt.analyze_trace(pt.TrainTrace(rocm_data()), warmup=1)
    assert got["warmup_steps"] == 1 and got["steps_measured"] == 2
    assert got["step_ms"]["mean"] == pytest.approx(10.0)
    assert got["step_ms_each"] == [10.0, 10.5, 9.5]


def test_a_trace_without_steps_is_refused():
    data = rocm_data()
    data["traceEvents"] = [e for e in data["traceEvents"] if not str(e.get("name", "")).startswith("ProfilerStep")]
    with pytest.raises(pt.ProfileError, match="ProfilerStep"):
        pt.analyze_trace(pt.TrainTrace(data))


def cpu_trace() -> dict:
    """Two CPU steps of 4 ms: a GDN forward op of 1.0 ms self (it calls a 0.5 ms child), its backward of 1.5 ms on
    the autograd thread linked to it, and an optimizer op of 0.25 ms."""
    ev = []

    def x(name, cat, ts, dur, tid, **args):
        ev.append({"ph": "X", "cat": cat, "name": name, "pid": 1, "tid": tid, "ts": ts, "dur": dur,
                   **({"args": args} if args else {})})

    for step in range(2):
        s = step * 4000.0
        x(f"ProfilerStep#{step}", "user_annotation", s, 4000.0, 1)
        x("atlas::gdn::model.layers.0.linear_attn", "user_annotation", s + 100, 1600.0, 1)
        x("aten::linear", "cpu_op", s + 110, 1500.0, 1)
        x("aten::mm", "cpu_op", s + 200, 500.0, 1, **{"Input Dims": [[4, 8], [8, 2]],
                                                     "Input type": ["float", "float"]})
        x("MmBackward0", "cpu_op", s + 2000, 1500.0, 2)
        ev.append({"ph": "s", "cat": "fwdbwd", "id": step + 1, "pid": 1, "tid": 1, "ts": s + 110})
        ev.append({"ph": "f", "cat": "fwdbwd", "id": step + 1, "pid": 1, "tid": 2, "ts": s + 2000, "bp": "e"})
        x("Optimizer.step#AdamW.step", "user_annotation", s + 3600, 300.0, 1)
        x("aten::_foreach_add_", "cpu_op", s + 3610, 250.0, 1)
    return {"traceEvents": ev}


def test_a_cpu_trace_is_read_by_op_self_time():
    got = pt.analyze_trace(pt.TrainTrace(cpu_trace()))
    assert got["kind"] == "cpu" and got["copies"] is None
    assert got["ms_per_step"]["gdn"] == pytest.approx(1.0 + 0.5 + 1.5)
    assert got["ms_per_step"]["optimizer"] == pytest.approx(0.25)
    assert got["shares"]["gpu_idle"] is None and got["shares"]["copies"] is None
    assert got["bandwidth"]["flops_per_step"] == 2 * 4 * 8 * 2
    assert got["bandwidth"]["bytes_per_step"] == (4 * 8 + 8 * 2) * 4 + 0      # leaf mm only; the rest have no shapes
    assert got["top_basis"] == "CPU ops by self time"
    assert {row["name"]: row["bucket"] for row in got["top"]}["MmBackward0"] == "gdn"


# --- small pieces ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name, kind", [
    ("base_model.model.model.layers.0.linear_attn", "gdn"),
    ("base_model.model.model.layers.3.self_attn", "attention"),
    ("base_model.model.model.layers.3.mlp", "mlp"),
    ("base_model.model.model.layers.3.self_attn.q_proj.lora_A.default", "lora"),
    ("base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_B.default", "lora"),
    ("base_model.model.model.embed_tokens", "embed_head"),
    ("base_model.model.lm_head", "embed_head"),
    ("base_model.model.model.layers.0.linear_attn.in_proj_qkv", None),
    ("base_model.model.model.layers.3.self_attn.q_proj.lora_dropout.default", None),
    ("base_model.model.model.norm", None),
])
def test_layer_kind_of_module_names(name, kind):
    assert pt.layer_kind(name) == kind


def test_tag_kind_reads_only_known_kinds():
    assert pt.tag_kind("atlas::gdn::model.layers.0.linear_attn") == "gdn"
    assert pt.tag_kind("atlas::nonsense::x") is None and pt.tag_kind("aten::mm") is None


def test_op_flops_for_each_matmul_form():
    def node(name, dims):
        return pt.Node(name, "cpu_op", 0.0, 1.0, "1", {"Input Dims": dims, "Input type": ["float"] * len(dims)})

    assert pt.op_flops(node("aten::mm", [[2, 3], [3, 5]])) == 2 * 2 * 3 * 5
    assert pt.op_flops(node("aten::addmm", [[5], [2, 3], [3, 5], [], []])) == 2 * 2 * 3 * 5
    assert pt.op_flops(node("aten::bmm", [[4, 2, 3], [4, 3, 5]])) == 2 * 4 * 2 * 3 * 5
    assert pt.op_flops(node("aten::mm", [[2, 3], [4, 5]])) == 0
    assert pt.op_flops(node("aten::add", [[2, 3], [2, 3]])) == 0


def test_op_bytes_uses_each_dtype():
    n = pt.Node("aten::add", "cpu_op", 0.0, 1.0, "1", {"Input Dims": [[2, 3], [2, 3], []],
                                                        "Input type": ["c10::BFloat16", "float", "Scalar"]})
    assert pt.op_bytes(n) == 6 * 2 + 6 * 4


def test_chip_reference_reads_the_measured_sheet_on_main():
    ref = pt.chip_reference()
    assert ref["copy_id"] == "copy.d2d.size_4gib" and 200 < ref["gb_s"] < 220
    assert ref["matmul_id"].startswith("matmul.bf16.") and ref["matmul_tflops"] > 0


def test_a_self_check_chip_sheet_is_never_used(tmp_path):
    sheet = tmp_path / "chipsheet.json"
    sheet.write_text(json.dumps({"mode": "self-check", "measurements": [
        {"id": "copy.d2d.size_1gib", "value": 999.0, "unit": "GB/s", "params": {"bytes": 2**30}}]}), encoding="utf-8")
    ref = pt.chip_reference(sheet)
    assert ref["gb_s"] is None and "self-check" in ref["note"]


def test_analyze_rocprof_cuts_steps_at_a_once_per_step_kernel(tmp_path):
    rows = []
    t = 0
    for step in range(5):
        for name, dur in (("Cijk_gemm", 600_000), ("elementwise_kernel", 100_000), ("fused_adamw_kernel", 300_000)):
            rows.append({"Kernel_Name": name, "Start_Timestamp": t, "End_Timestamp": t + dur, "Workgroup_Size_X": 256,
                         "Grid_Size_X": 256 * 64, "Correlation_Id": len(rows) + 1})
            t += dur
        t += 1_000_000   # 2 ms steps
    folder = tmp_path / "rocprof" / "host" / "1"
    folder.mkdir(parents=True)
    (folder / "1_kernel_trace.csv").write_text(tp.write_rocprof_kernels(rows), encoding="utf-8")
    got = pt.analyze_rocprof(tmp_path / "rocprof", 5)
    assert got["marker"]["name"] in {"Cijk_gemm", "elementwise_kernel", "fused_adamw_kernel"}
    assert got["steps"] == 4 and got["step_ms"]["mean"] == pytest.approx(2.0)
    assert got["category_ms_per_step"]["optimizer"] == pytest.approx(0.3)
    assert got["category_ms_per_step"]["dense_matmul"] == pytest.approx(0.6)


def test_analyze_rocprof_without_csvs_is_none(tmp_path):
    assert pt.analyze_rocprof(tmp_path, 5) is None


# --- the result, its checks, the files and the ledger row -----------------------------------------------------------

def rocm_result(rocm_analysis) -> dict:
    run = json.loads((ROCM / "run.json").read_text(encoding="utf-8"))
    result = pt.build_result(rocm_analysis, run, None, date="2026-10-11", cpu=False)
    pt.apply_reference(result, pt.chip_reference())
    return result


def test_the_result_passes_its_checks_and_carries_the_run(rocm_analysis):
    result = rocm_result(rocm_analysis)
    assert pt.check_result(result) == []
    assert (result["chip"], result["node"], result["task"]) == ("gfx1151", "pod-b", "R34")
    assert result["run"]["samples_per_s"] == 1.6
    assert result["fast_linear_attention"]["run_reported"] == "torch fallback"
    assert result["fast_linear_attention"]["fallback_share"] == pytest.approx(0.26)
    assert "L16" in result["cites"]
    bw = result["bandwidth"]
    assert bw["fraction_of_copy_step"] == pytest.approx(bw["achieved_gb_s_step"] / bw["reference_gb_s"], abs=1e-4)


def test_check_result_catches_broken_results(rocm_analysis):
    good = rocm_result(rocm_analysis)
    for mutate, needle in ((lambda r: r["shares"].pop("gdn"), "share gdn missing"),
                           (lambda r: r["shares"].update(gdn=3.0), "outside"),
                           (lambda r: r.update(top=[]), "top table"),
                           (lambda r: r["step_ms"].update(mean=None), "no step time"),
                           (lambda r: r.update(date="today"), "date"),
                           (lambda r: r["fast_linear_attention"].pop("present"), "fast_linear_attention")):
        bad = copy.deepcopy(good)
        mutate(bad)
        assert any(needle in p for p in pt.check_result(bad)), needle
    assert pt.check_result({"schema": "other"}) == ["not a lab.profile_train/1 result"]


def test_markdown_has_every_section_and_the_numbers(rocm_analysis):
    text = pt.render_markdown(rocm_result(rocm_analysis))
    for section in pt.MD_SECTIONS:
        assert section in text
    assert "| gdn | 26.0% | 2.600 |" in text
    assert any(line.startswith("| lora | ") and line.endswith(" | 0.045 |") for line in text.splitlines())
    assert "Present: no." in text
    assert "—" not in text


def test_ledger_row_of_the_result(rocm_analysis):
    rows = pt.ledger_rows(rocm_result(rocm_analysis), "lab/refine/research/kernels/data/x.json")
    assert len(rows) == 1
    row = rows[0]
    assert (row["verb"], row["node"], row["low_confidence"]) == ("profile_train", "pod-b", True)
    assert row["metrics"]["train_profile.share_gdn"] == 0.26
    assert row["metrics"]["trainer.samples_per_s"] == 1.6
    assert row["units"]["train_profile.step_ms"] == "ms" and row["units"]["trainer.samples_per_s"] == "samples/s"
    assert ledger.validate({"ts": "2026-10-11T04:00:00Z", "actor": "t", "tier": None, "task": "R34", **row})


def test_ingest_writes_both_files_and_the_ledger_row(tmp_path):
    answer = pt.ingest(ROCM, out_dir=tmp_path / "data", warmup=0)
    assert answer["ok"] is True, answer
    assert Path(answer["json"]).name == "train-profile-gfx1151-2026-10-11.json"
    assert Path(answer["markdown"]).is_file()
    rows = ledger.read(state_dir=tmp_path / "state")
    assert [(r["verb"], r["node"], r["task"]) for r in rows] == [("profile_train", "pod-b", "R34")]
    again = pt.main(["ingest", str(ROCM), "--out", str(tmp_path / "data"), "--no-ledger"])
    assert again["ok"] is False and "--force" in again["error"]


def test_ingest_of_a_folder_without_a_trace(tmp_path):
    answer = pt.ingest(tmp_path, out_dir=tmp_path / "data", ledger=False)
    assert answer == {"ok": False, "outcome": "no_trace", "error": f"no torch trace under {tmp_path}"}


def test_the_verb_is_registered_in_its_alphabetical_place():
    from lab import __main__ as lab_cli

    names = list(lab_cli.VERBS)
    assert lab_cli.VERBS["profile-train"] == "lab.profile_train:main"
    assert names == sorted(names)


def test_train_main_hands_its_callbacks_to_the_trainer(monkeypatch, tmp_path):
    from lab import train

    seen = {}
    monkeypatch.setattr(train, "_prepare", lambda args, run_id, refs, node: "plan")

    def fake_train(plan, callbacks=()):
        seen["callbacks"] = list(callbacks)
        return train.Trained({"trainer.steps": 1.0}, {})

    monkeypatch.setattr(train, "train", fake_train)
    marker = object()
    answer = train.main(["--cpu", "--dataset", str(tmp_path), "--rows", str(tmp_path / "rows.jsonl"),
                         "--model", str(tmp_path)], callbacks=[marker])
    assert answer["outcome"] == "trained" and seen["callbacks"] == [marker]


# --- the recorded CPU trace (two steps of the tiny Qwen3.5 under lab/train.py --cpu) ---------------------------------

CPU = REPO / "fixtures" / "lab" / "train_profile" / "cpu"


@pytest.fixture(scope="module")
def cpu_analysis() -> dict:
    return pt.analyze_trace(pt.TrainTrace.load(CPU / "trace.json.gz"))


def test_the_recorded_cpu_trace_has_two_steps_and_its_layers(cpu_analysis):
    assert cpu_analysis["kind"] == "cpu" and cpu_analysis["steps_measured"] == 2
    for bucket in ("gdn", "attention", "lora", "mlp", "optimizer"):
        assert cpu_analysis["shares"][bucket] > 0, bucket
    assert cpu_analysis["shares"]["other"] < 0.05          # almost every op lands in a named bucket


def test_the_recorded_cpu_trace_links_backward_ops_to_their_layers(cpu_analysis):
    assert cpu_analysis["attribution"]["backward_links"] > 1000
    top = {row["name"]: row["bucket"] for row in cpu_analysis["top"]}
    assert top["aten::_scaled_dot_product_flash_attention_for_cpu_backward"] == "attention"
    assert cpu_analysis["fast_linear_attention"]["present"] is False


def test_the_gdn_layer_is_the_biggest_share_on_the_torch_fallback(cpu_analysis):
    named = {k: v for k, v in cpu_analysis["shares"].items() if v is not None and k not in ("other",)}
    assert max(named, key=named.get) == "gdn"


def test_the_quick_self_check_passes_on_both_fixtures(tmp_path):
    answer = pt.main(["self-check", "--quick", "--out", str(tmp_path)])
    assert answer["ok"] is True, answer
    assert answer["checks"] == {"rocm_ingested": True, "rocm_markdown_sections": True, "cpu_ingested": True,
                                "cpu_markdown_sections": True, "rocm_ms_per_step_match": True}
    assert (tmp_path / "rocm" / "train-profile-gfx1151-2026-10-11.json").is_file()
    assert (tmp_path / "cpu" / "train-profile-cpu-2026-10-11.json").is_file()


def test_record_files_a_pod_result_and_writes_its_ledger_row(tmp_path, rocm_analysis):
    source = tmp_path / "from-pod.json"
    source.write_text(json.dumps(rocm_result(rocm_analysis)), encoding="utf-8")
    answer = pt.main(["record", str(source), "--out", str(tmp_path / "data")])
    assert answer["ok"] is True, answer
    assert Path(answer["json"]).name == "train-profile-gfx1151-2026-10-11.json"
    rows = ledger.read(state_dir=tmp_path / "state")
    assert [(r["verb"], r["node"]) for r in rows] == [("profile_train", "pod-b")]


def test_a_pod_run_trains_each_pass_on_its_own_copy_and_deletes_the_originals(tmp_path, monkeypatch):
    """The run's plumbing with the trainer replaced: each pass gets a copy of the data (lab/train.py deletes its own
    data on the pod), the originals go once the passes are done, and the torch pass's trace is ingested."""
    dataset = tmp_path / "datasets" / "tool-calling-fixture"
    dataset.mkdir(parents=True)
    (dataset / "manifest.json").write_text("{}", encoding="utf-8")
    (dataset / "examples.jsonl").write_text("", encoding="utf-8")
    rows = tmp_path / "datasets" / "rows.jsonl"
    rows.write_text("", encoding="utf-8")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        argv = cmd[cmd.index("--", cmd.index("_child") + 1) + 1:]
        data = Path(argv[argv.index("--dataset") + 1])
        assert data.is_dir() and data != dataset and data.parent.name.startswith("data-")
        trace_dir = Path(cmd[cmd.index("--trace-dir") + 1])
        trace_dir.mkdir(parents=True)
        (trace_dir / "trace.json").write_bytes((ROCM / "trace.json").read_bytes())
        out = Path(argv[argv.index("--out") + 1])
        out.mkdir(parents=True)
        (out / "run.json").write_bytes((ROCM / "run.json").read_bytes())
        return type("Done", (), {"returncode": 0, "stdout": "{}", "stderr": ""})()

    monkeypatch.setattr(pt.subprocess, "run", fake_run)
    monkeypatch.setattr(pt.shutil, "which", lambda name: None)
    answer = pt.main(["run", "--dataset", str(dataset), "--rows", str(rows), "--model", "m", "--steps", "3",
                      "--out", str(tmp_path / "run"), "--results", str(tmp_path / "data"), "--no-ledger"])
    assert answer["ok"] is True, answer
    assert answer["tools"]["rocprofv3"] == {"ok": False, "skipped": "not on PATH"}
    assert answer["data_deleted"] == "yes" and not dataset.exists() and not rows.exists()
    assert len(calls) == 1 and "--max-steps" in calls[0] and "3" in calls[0]
    result = json.loads(Path(answer["json"]).read_text(encoding="utf-8"))
    assert result["node"] == "pod-b" and result["shares"]["gdn"] == pytest.approx(0.26)
