"""lab/trace_profile.py: the torch profiler's Chrome trace and rocprofv3's CSVs, read on the synthetic decode run
in fixtures/lab/profile_decode/synthetic (made by fixtures/lab/profile_decode/make_fixture.py, whose docstring gives
every number used here)."""
from __future__ import annotations

import statistics
from pathlib import Path

import pytest

from lab import trace_profile as tp

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "lab" / "profile_decode" / "synthetic"
B1 = FIX / "torch" / "b1.pt.trace.json.gz"
KERNEL_CSV = FIX / "rocprof" / "4243" / "run_kernel_trace.csv"
COPY_CSV = FIX / "rocprof" / "4243" / "run_memory_copy_trace.csv"
GIB = 2**30


def test_torch_trace_reads_kernels_copies_runtime_calls_ranges_and_memory_samples():
    trace = tp.TorchTrace.load(B1)
    # a prefill of 10 kernels, then six decode ranges of 16 forward kernels and one argmax each
    assert len(trace.kernels) == 10 + 6 * 17
    assert [(c.kind, c.bytes, c.dur_us) for c in trace.copies] == [("DtoH", 4, 1000.0)] * 6
    assert len(trace.annotations) == 7          # the gpu_user_annotation copies are not CPU ranges
    assert len(trace.runtime) == 1 + 6 * 3
    assert trace.graph_launches() == 7
    assert len(trace.memory) == 4
    first = trace.kernels[0]
    assert (first.name, first.grid, first.block, first.lds_bytes, first.correlation) == (
        "_gemm_kernel", (80, 1, 1), (256, 1, 1), 49152, 1000)
    assert trace.device["name"] == "AMD Radeon 8060S"


def test_a_kernel_is_tied_to_the_runtime_call_that_launched_it():
    trace = tp.TorchTrace.load(B1)
    assert trace.launch_us(1010) == 1_101_000.0      # the first decode step's hipGraphLaunch
    assert trace.launch_us(1011) == 1_350_000.0      # its argmax, launched after the step's forward range
    assert trace.launch_us(None) is None
    assert trace.launch_us(424242) is None


def test_a_file_that_is_not_a_trace_is_refused(tmp_path):
    path = tmp_path / "x.json"
    path.write_text('{"events": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="no traceEvents"):
        tp.TorchTrace.load(path)


def test_rocprof_kernel_rows_drop_the_kd_suffix_and_give_the_grid_in_workgroups():
    kernels = tp.read_rocprof_kernels(KERNEL_CSV)
    assert len(kernels) == 3 + 2 * 5 * 17
    assert not any(k.name.endswith(".kd") for k in kernels)
    gemm = next(k for k in kernels if k.name == "_gemm_kernel")
    assert (gemm.grid, gemm.block, gemm.lds_bytes, gemm.vgprs, gemm.sgprs, gemm.scratch_bytes, gemm.dur_us) == (
        (80, 1, 1), (256, 1, 1), 49152, 128, 64, 0, 18000.0)
    assert [k.start_us for k in kernels] == sorted(k.start_us for k in kernels)


def test_rocprof_copy_rows_take_their_kind_from_the_direction():
    copies = tp.read_rocprof_copies(COPY_CSV)
    assert [(c.kind, c.dur_us) for c in copies] == [("DtoH", 1000.0)] * 10


def test_older_rocprof_column_names_are_read_as_well(tmp_path):
    path = tmp_path / "k.csv"
    path.write_text('"Kernel_Name","Start_Timestamp","End_Timestamp","Group_Segment_Size","Private_Segment_Size",'
                    '"Arch_VGPR_Count","Workgroup_Size_X","Grid_Size_X"\n"k",1000,3000,512,16,24,64,640\n',
                    encoding="utf-8")
    (kernel,) = tp.read_rocprof_kernels(path)
    assert (kernel.lds_bytes, kernel.scratch_bytes, kernel.vgprs, kernel.grid, kernel.block, kernel.start_us,
            kernel.dur_us) == (512, 16, 24, (10, 1, 1), (64, 1, 1), 1.0, 2.0)


def test_find_rocprof_csvs_finds_every_process_folder(tmp_path):
    for pid in ("11", "22"):
        (tmp_path / pid).mkdir()
        (tmp_path / pid / "run_kernel_trace.csv").write_text("x", encoding="utf-8")
    (tmp_path / "22" / "run_memory_copy_trace.csv").write_text("x", encoding="utf-8")
    found = tp.find_rocprof_csvs(tmp_path)
    assert found == {"kernel": [tmp_path / "11" / "run_kernel_trace.csv", tmp_path / "22" / "run_kernel_trace.csv"],
                     "copy": [tmp_path / "22" / "run_memory_copy_trace.csv"]}


@pytest.mark.parametrize("name,category", [
    ("_gemm_kernel", "w4a16_matmul"),
    ("gemm_half_q_half_gptq_4bit_kernel", "w4a16_matmul"),
    ("kernel_unified_attention_3d", "attention"),
    ("reshape_and_cache_flash_kernel", "attention"),
    ("fused_recurrent_gated_delta_rule_fwd_kernel", "attention_linear"),
    ("causal_conv1d_update_kernel", "attention_linear"),
    ("void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<float> > >",
     "sampling"),
    ("Cijk_Alik_Bljk_BBS_BH_MT64x64x64_SN", "dense_matmul"),
    ("wvSplitK_hf_sml_", "dense_matmul"),
    ("triton_poi_fused_add_rms_norm_0", "norm_activation"),
    ("act_and_mul_kernel", "norm_activation"),
    ("triton_poi_fused_cat_0", "elementwise"),
    ("mystery", "other"),
])
def test_a_kernel_name_falls_in_the_first_category_whose_rule_matches(name, category):
    assert tp.Categorizer()(name) == category


def test_exact_names_come_before_the_rules():
    categorize = tp.Categorizer(exact={"my_attention_kernel": "w4a16_matmul"})
    assert categorize("my_attention_kernel") == "w4a16_matmul"
    assert categorize("other_attention_kernel") == "attention"
    assert categorize.describe()["exact"] == {"my_attention_kernel": "w4a16_matmul"}


def test_the_union_of_intervals_counts_an_overlap_once():
    assert tp.union_us([(0, 10), (5, 15), (20, 25), (30, 30), (24, 26)]) == 21.0
    assert tp.union_us([]) == 0.0


def test_spread_gives_linear_percentiles():
    gaps = [290, 295, 298, 300, 300, 300, 300, 302, 305, 310]
    assert tp.spread([*gaps, None]) == {"n": 10, "mean": 300.0, "sd": round(statistics.stdev(gaps), 3),
                                        "min": 290.0, "p10": 294.5, "p50": 300.0, "p90": 305.5, "p99": 309.55,
                                        "max": 310.0}
    assert tp.spread([None]) == {"n": 0, "mean": None, "sd": None, "min": None, "p10": None, "p50": None,
                                 "p90": None, "p99": None, "max": None}
    assert tp.spread([7.0])["sd"] is None


def test_the_kernel_table_groups_by_name_biggest_first_with_its_dims_and_shares():
    kernels = [tp.Kernel("a", 0, 10, (2, 1, 1), (64, 1, 1)), tp.Kernel("a", 20, 10, (2, 1, 1), (64, 1, 1)),
               tp.Kernel("a", 40, 10, (4, 1, 1), (64, 1, 1)), tp.Kernel("b", 60, 50), tp.Kernel("c", 70, 5)]
    rows = tp.kernel_table(kernels, steps=2, step_us=100.0, categorize=lambda n: "other", top=2)
    assert [(r["rank"], r["name"]) for r in rows] == [(1, "b"), (2, "a")]
    a = rows[1]
    assert (a["launches"], a["launches_per_step"], a["total_ms"], a["ms_per_step"], a["mean_us"], a["max_us"]) == (
        3, 1.5, 0.03, 0.015, 10.0, 10.0)
    assert (a["grid"], a["distinct_grids"], a["block"], a["distinct_blocks"]) == ([2, 1, 1], 2, [64, 1, 1], 1)
    assert (a["share_of_kernel_time"], a["share_of_step"]) == (round(30 / 85, 4), 0.15)
    assert rows[0]["grid"] is None and rows[0]["distinct_grids"] == 0


def test_the_marker_is_the_most_even_kernel_that_runs_once_per_step():
    kernels = ([tp.Kernel("every", t, 1) for t in (0, 100, 200, 300)]
               + [tp.Kernel("twice", t, 1) for t in range(0, 400, 50)]
               + [tp.Kernel("uneven", t, 1) for t in (0, 10, 200, 390)])
    marker = tp.pick_marker(kernels, 4)
    assert (marker["name"], marker["launches"], marker["gaps_us"], marker["cv"]) == ("every", 4,
                                                                                    [100.0, 100.0, 100.0], 0.0)
    assert tp.pick_marker(kernels, 1) is None


def test_memory_peaks_leave_out_cpu_samples_and_keep_to_the_window():
    trace = tp.TorchTrace.load(B1)
    whole = tp.memory_peaks(trace.memory)
    assert whole == {"samples": 3, "peak_allocated_bytes": 20 * GIB + 5_000_000,
                     "peak_reserved_bytes": int(22.5 * GIB), "peak_allocated_gib": 20.005,
                     "peak_reserved_gib": 22.5}
    decode = tp.memory_peaks(trace.memory, start_us=1_100_000, end_us=2_600_000)
    assert (decode["samples"], decode["peak_allocated_bytes"], decode["peak_reserved_bytes"]) == (
        2, 20 * GIB + 3_000_000, 22 * GIB + 2 * 2**20)


def test_copy_summary_per_kind_and_per_step():
    copies = [tp.Copy("DtoH", 0, 1000, 4), tp.Copy("DtoH", 10, 1000, 4), tp.Copy("HtoD", 20, 500, None)]
    assert tp.copy_summary(copies, steps=2) == {
        "count": 3, "time_ms": 2.5, "bytes": 8, "bytes_known_for": 2, "count_per_step": 1.5, "ms_per_step": 1.25,
        "bytes_per_step": 4.0, "by_kind": {"DtoH": {"count": 2, "time_ms": 2.0, "bytes": 8},
                                           "HtoD": {"count": 1, "time_ms": 0.5, "bytes": None}}}


@pytest.mark.parametrize("value,expected", [([80, 1, 1], (80, 1, 1)), ([7], (7, 1, 1)), ("[8, 4, 1]", (8, 4, 1)),
                                            ([1, 2, 3, 4], None), ("x", None), ([-1, 1, 1], None), (None, None)])
def test_dim3_reads_lists_and_text(value, expected):
    assert tp.dim3(value) == expected


@pytest.mark.parametrize("text,kind", [("Memcpy HtoD (Pageable -> Device)", "HtoD"),
                                       ("Memcpy DtoH (Device -> Pinned)", "DtoH"),
                                       ("MEMORY_COPY_DEVICE_TO_DEVICE", "DtoD"), ("Memset (Device)", "memset"),
                                       ("something", "other")])
def test_copy_kind_from_torch_names_and_rocprof_directions(text, kind):
    assert tp.copy_kind(text) == kind


def test_a_trimmed_trace_keeps_the_window_and_the_gpu_work_launched_in_it():
    data = tp.load_json(B1)
    cut = tp.trim_torch_trace(data, 1_100_000, 1_400_000)       # the first decode step
    trace = tp.TorchTrace(cut)
    assert len(trace.kernels) == 17 and len(trace.copies) == 1
    assert [a.start_us for a in trace.annotations] == [1_100_000.0, 1_400_000.0]
    assert cut["deviceProperties"] == data["deviceProperties"]
    assert len(tp.trim_torch_trace(data, 1_100_000, 1_400_000, max_events=5)["traceEvents"]) == 5


def test_rocprof_rows_written_back_read_the_same(tmp_path):
    rows = [{"Kernel_Name": "_gemm_kernel.kd", "Start_Timestamp": 2000, "End_Timestamp": 5000,
             "Workgroup_Size_X": 256, "Workgroup_Size_Y": 1, "Workgroup_Size_Z": 1, "Grid_Size_X": 512,
             "Grid_Size_Y": 1, "Grid_Size_Z": 1}]
    path = tmp_path / "k.csv"
    path.write_text(tp.write_rocprof_kernels(rows), encoding="utf-8")
    (kernel,) = tp.read_rocprof_kernels(path)
    assert (kernel.name, kernel.start_us, kernel.dur_us, kernel.grid) == ("_gemm_kernel", 2.0, 3.0, (2, 1, 1))
