"""Tests for the wavefront microbenchmark harness (lab/kernels/wavefront/): the gfx1151 occupancy arithmetic, the
compile-record and rocprofv3 trace parsers, the sheet format, the self-check, the node gate and the pod command.
Nothing here needs torch, triton or a GPU."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from lab.kernels.chipsheet import chipsheet_format as cf
from lab.kernels.wavefront import conv, gdn, launch, lora, tile
from lab.kernels.wavefront import wavefront as wv
from lab.kernels.wavefront import wavefront_format as wf

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "fixtures" / "lab" / "wavefront"
TRACE = FIXTURES / "rocprof_kernel_trace_synthetic.csv"
MODULES = (gdn, conv, launch, tile, lora)


# --- occupancy arithmetic ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("vgprs, waves", [(1, 16), (64, 16), (96, 16), (97, 12), (120, 12), (144, 10), (168, 9),
                                          (192, 8), (216, 7), (240, 6), (256, 5), (300, 5)])
def test_waves_by_vgprs_follows_the_gfx1151_table(vgprs, waves):
    assert wf.waves_by_vgprs(vgprs) == waves


def test_waves_by_vgprs_unknown_is_none():
    assert wf.waves_by_vgprs(None) is None


@pytest.mark.parametrize("lds, groups", [(None, None), (0, None), (1000, 128), (1024, 128), (20480, 6),
                                         (40960, 3), (65536, 2), (65537, 0)])
def test_workgroups_by_lds_uses_128_kib_per_wgp_in_1_kib_granules(lds, groups):
    assert wf.workgroups_by_lds(lds) == groups


def test_occupancy_names_the_binding_limit_and_whether_the_grid_fills_the_chip():
    stages2 = wf.occupancy(vgprs=88, lds_bytes=40960, threads=128, workgroups=16)
    assert stages2["waves_per_simd_by_vgpr"] == 16
    assert stages2["workgroups_per_wgp_by_lds"] == 3
    assert stages2["waves_per_simd_by_lds"] == 3  # 3 groups x 4 waves over 4 SIMDs
    assert stages2["waves_per_simd"] == 3 and stages2["bound_by"] == "lds"
    assert stages2["fills_wgps"] is False and stages2["assumes_simds_per_wgp"] == 4
    stages1 = wf.occupancy(vgprs=88, lds_bytes=20480, threads=128, workgroups=512)
    assert stages1["waves_per_simd"] == 6 and stages1["fills_wgps"] is True
    no_lds = wf.occupancy(vgprs=200, lds_bytes=0, threads=256)
    assert no_lds["waves_per_simd"] == 7 and no_lds["bound_by"] == "vgpr" and no_lds["assumes_simds_per_wgp"] is None


def test_compile_record_reads_triton_names_and_ours():
    assert wf.compile_record({"n_regs": "88", "n_spills": 0, "shared": 20480, "num_warps": 4, "num_stages": 1}) == \
        {"n_regs": 88, "n_spills": 0, "shared_bytes": 20480, "num_warps": 4, "num_stages": 1}
    assert wf.compile_record({"shared_bytes": 4096, "n_regs": "many"})["shared_bytes"] == 4096
    assert wf.compile_record({"n_regs": "many"})["n_regs"] is None
    occ = wf.compile_occupancy(wf.compile_record({"n_regs": 112, "shared": 40960, "num_warps": 4}), 512)
    assert occ["threads"] == 128 and occ["waves_per_simd_by_vgpr"] == 12 and occ["waves_per_simd"] == 3


def test_remark_occupancy_reads_the_chipsheet_remarks_beside_the_compilers_own_figure():
    remarks = cf.parse_resource_remarks((REPO / "fixtures/lab/chipsheet/remarks_w32_synthetic.txt").read_text())
    found = wf.remark_occupancy(remarks)
    assert found["occ_vgpr_4"]["waves_per_simd_by_vgpr"] == 16
    assert found["occ_vgpr_240"]["vgprs"] == 245
    assert found["occ_vgpr_240"]["waves_per_simd_by_vgpr"] == 5
    assert found["occ_vgpr_240"]["compiler_occupancy"] == 4  # kept as reported, not reconciled


# --- rocprofv3 trace --------------------------------------------------------------------------------------------

def test_trace_kernels_parses_the_rocprofv3_columns():
    kernels = wf.trace_kernels(TRACE.read_text())
    assert len(kernels) == 57
    first = kernels[0]
    assert first["name"] == "Cijk_Alik_Bljk_BBS_BH_MT128x128x32_MI16x16x16_SN"  # .kd dropped
    assert first["dur_us"] == pytest.approx(318.0)
    assert first["threads"] == 256 and first["workgroups"] == 160
    assert first["vgprs"] == 128 and first["lds_bytes"] == 40960
    assert [k["start_ns"] for k in kernels] == sorted(k["start_ns"] for k in kernels)


def test_trace_kernels_skips_rows_without_times_and_reads_older_column_names():
    text = ("Kernel_Name,Start_Timestamp,End_Timestamp,Group_Segment_Size,Arch_VGPR_Count,Workgroup_Size_X,Grid_Size_X\n"
            "a.kd,100,1100,2048,40,64,640\n"
            "b,,,0,0,64,64\n"
            "c,500,400,0,0,64,64\n")
    kernels = wf.trace_kernels(text)
    assert len(kernels) == 1
    assert kernels[0] == {"name": "a", "start_ns": 100, "dur_us": 1.0, "threads": 64, "workgroups": 10,
                          "lds_bytes": 2048, "vgprs": 40}


def test_trace_occupancy_ranks_kernels_by_total_time_with_launch_counts():
    found = wf.trace_occupancy(wf.trace_kernels(TRACE.read_text()), top=10)
    assert found["launches"] == 57 and found["names"] == 4
    names = [row["name"] for row in found["top"]]
    assert names[0].startswith("Cijk_") and names[1].startswith("miopenSp3AsmConv")
    assert sum(row["share"] for row in found["top"]) == pytest.approx(1.0, abs=1e-3)
    gemm = found["top"][0]
    assert gemm["launches"] == 12 and gemm["total_us"] == pytest.approx(12 * 318.0)
    assert gemm["occupancy"]["waves_per_simd_by_vgpr"] == 10  # 128 VGPRs round to 144
    elementwise = next(r for r in found["top"] if "elementwise" in r["name"])
    assert elementwise["launches"] == 36 and elementwise["median_us"] == pytest.approx(9.0)
    assert wf.trace_occupancy(wf.trace_kernels(TRACE.read_text()), top=1)["top"][0]["name"] == names[0]


# --- sheet format -----------------------------------------------------------------------------------------------

def sheet() -> dict:
    return copy.deepcopy(wv.self_check_sheet()["sheet"])


def test_self_check_sheet_conforms_and_covers_every_class():
    s = sheet()
    assert wf.validate(s) == []
    assert s["classes"] == list(wf.CLASSES)
    assert {row["class"] for row in s["rows"]} == set(wf.CLASSES)
    tiles = [row for row in s["rows"] if row["class"] == "tile"]
    assert {row["compile"]["num_stages"] for row in tiles} == {1, 2}
    assert all(row["occupancy"]["waves_per_simd"] for row in tiles)


@pytest.mark.parametrize("breakage, message", [
    (lambda s: s.update(format="atlas.chipsheet/1"), "format"),
    (lambda s: s.update(mode="fast"), "mode"),
    (lambda s: s.update(date="11-10-2026"), "date"),
    (lambda s: s["rows"][0].update(id="GDN.Bad"), "dotted lowercase"),
    (lambda s: s["rows"][1].update(id=s["rows"][0]["id"]), "repeats"),
    (lambda s: s["rows"][0].update(id="conv.impl_torch.b4_t256x"), "does not start with its class"),
    (lambda s: s["rows"][0].update(samples=[], repeats=0), "ok with no samples"),
    (lambda s: s["rows"][0].update(value=1.0), "not the median"),
    (lambda s: s["rows"][0].update(status="skipped", error=None), "with no error"),
    (lambda s: s["rows"][0].update(status="maybe"), "status"),
    (lambda s: s["rows"][0].update(samples=[-1.0, 1.0], repeats=2, value=0.0), "non-negative"),
    (lambda s: next(r for r in s["rows"] if r["compile"]).update(compile={"n_regs": 1}), "compile"),
    (lambda s: s.pop("rows"), "missing key rows"),
])
def test_validate_names_each_kind_of_problem(breakage, message):
    s = sheet()
    breakage(s)
    problems = wf.validate(s)
    assert problems and any(message in p for p in problems), problems


def test_a_skipped_row_with_its_reason_conforms():
    s = sheet()
    s["rows"][0] = wf.row(s["rows"][0]["id"], "gdn", "impl", "torch", gdn.WHAT, command="x", params={},
                          status="skipped", error="flash-linear-attention is not importable")
    assert wf.validate(s) == []
    assert s["rows"][0]["value"] is None and s["rows"][0]["repeats"] == 0


def test_write_and_read_round_trip_and_the_markdown_has_every_row(tmp_path):
    s = sheet()
    json_path, md_path = wf.write(s, tmp_path)
    assert json_path.name.endswith("-selfcheck.json") and json_path.name.startswith("wavefront-gfx1151-")
    assert wf.read(json_path) == json.loads(json.dumps(s))
    text = md_path.read_text()
    for row in s["rows"]:
        assert f"| {row['id']} |" in text
    for klass in wf.CLASSES:
        assert f"## {wf.TITLES[klass]}" in text


def test_write_refuses_a_sheet_that_does_not_conform(tmp_path):
    s = sheet()
    s["rows"][0]["status"] = "maybe"
    with pytest.raises(ValueError, match="does not conform"):
        wf.write(s, tmp_path)
    assert not list(tmp_path.iterdir())


def test_ratio_reads_a_speedup_between_two_ok_rows():
    rows = wf.by_id(sheet())
    torch_id, fla_id = (case["id"] for case in gdn.cases(True))
    assert wf.ratio(rows, fla_id, torch_id) == pytest.approx(2.0)
    assert wf.ratio(rows, fla_id, "gdn.impl_none.b4_t256") is None


# --- kernel classes ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.CLASS)
def test_every_class_names_one_lever_and_unique_ids(module):
    assert module.CLASS in wf.CLASSES
    for quick in (True, False):
        cases = module.cases(quick)
        ids = [case["id"] for case in cases]
        assert len(ids) == len(set(ids)) and len(cases) >= 2
        for case in cases:
            assert wf.ID_RE.fullmatch(case["id"]) and case["id"].startswith(module.CLASS + ".")
            assert f".{module.LEVER}_" in case["id"]
            assert re.sub(r"[^a-z0-9_]", "_", str(case["setting"]).lower()) in case["id"]
        assert {case["setting"] for case in cases} == {case["setting"] for case in module.cases(True)}
    assert {c["id"] for c in module.cases(True)} <= {c["id"] for c in module.cases(False)}


def test_the_conv_floor_is_three_passes_of_a_bf16_tensor_at_the_chip_sheet_bandwidth():
    assert conv.floor_s(4, 256) == pytest.approx(3 * 2 * 4 * 256 * 8192 * 2 / 209.5e9)
    assert conv.floor_s(4, 256) == pytest.approx(0.48e-3, rel=0.01)


def test_the_tile_and_lora_grids_are_the_ones_the_note_reasons_about():
    case = tile.cases(True)[0]["params"]
    assert case["batch"] * case["heads"] * (case["v"] // case["bv"]) == 512
    assert lora.TOKENS // lora.BLOCK_M == 16 and lora.TOKENS // lora.BLOCK_M * 4 == 64


def test_a_missing_package_becomes_a_skipped_row(monkeypatch):
    def measure(case, timer):
        raise wf.Skipped("flash-linear-attention is not importable: No module named 'fla'")

    monkeypatch.setattr(gdn, "measure", measure)
    case = gdn.cases(True)[1]
    row = wv.run_case(gdn, case, timer=None, command="c", left_s=100)
    assert row["status"] == "skipped" and "fla" in row["error"]


def test_a_failure_becomes_a_failed_row_and_the_budget_a_skipped_one(monkeypatch):
    def measure(case, timer):
        raise RuntimeError("boom")

    monkeypatch.setattr(lora, "measure", measure)
    case = lora.cases(True)[0]
    assert wv.run_case(lora, case, timer=None, command="c", left_s=100)["error"] == "RuntimeError: boom"
    skipped = wv.run_case(lora, case, timer=None, command="c", left_s=1)
    assert skipped["status"] == "skipped" and skipped["error"].startswith("budget")


def test_an_ok_measurement_carries_its_compile_record_and_occupancy(monkeypatch):
    def measure(case, timer):
        return {"samples": [3e-5, 2e-5, 4e-5], "first_call": 1.5, "workgroups": 512,
                "compile": wf.compile_record({"n_regs": 88, "shared": 20480, "num_warps": 4, "num_stages": 1}),
                "derived": {"us": 30.0}}

    monkeypatch.setattr(tile, "measure", measure)
    row = wv.run_case(tile, tile.cases(True)[0], timer=None, command="c", left_s=100)
    assert row["status"] == "ok" and row["value"] == pytest.approx(3e-5) and row["first_call"] == 1.5
    assert row["occupancy"]["waves_per_simd"] == 6 and row["occupancy"]["fills_wgps"] is True


# --- command line -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("node, gfx, machine, ok", [
    ("pod-b", "gfx1151", "x86_64", True), ("pod-a", "gfx1151", "x86_64", True),
    ("pod-d", "gfx1151", "x86_64", False), ("pod-b", "gfx1201", "x86_64", False),
    ("pod-b", None, "x86_64", False), ("pod-b", "gfx1151", "aarch64", False)])
def test_the_node_gate(node, gfx, machine, ok):
    assert (wv.gate(node, gfx, machine) is None) is ok


def test_run_refuses_here_before_touching_a_gpu(monkeypatch, capsys):
    monkeypatch.setattr(wv, "rocminfo_gfx", lambda: None)
    assert wv.main(["run", "--node", "pod-b", "--only", "gdn"]) == wv.EXIT_GATE
    answer = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert answer["ok"] is False and answer["outcome"] == "refused" and "gfx1151" in answer["error"]


def test_self_check_writes_and_rereads_a_sheet(tmp_path, capsys):
    assert wv.main(["self-check", "--out", str(tmp_path)]) == 0
    answer = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert answer["ok"] is True and answer["rows"] == sum(len(m.cases(True)) for m in MODULES)
    assert answer["trace_launches"] == 57 and Path(answer["json"]).exists() and Path(answer["md"]).exists()


def test_occupancy_verb_reads_a_trace(capsys):
    assert wv.main(["occupancy", str(TRACE), "--top", "2"]) == 0
    answer = json.loads(capsys.readouterr().out)
    assert answer["launches"] == 57 and len(answer["top"]) == 2


def test_pod_script_is_one_line_with_every_file_the_pod_needs(capsys):
    assert wv.main(["pod-script", "--only", "gdn"]) == 0
    answer = json.loads(capsys.readouterr().out)
    command = answer["command"]
    assert "\n" not in command and answer["slot_minutes"] == 20
    assert command.startswith("python -m lab.controller run-script --account D --timeout 1200 ")
    assert command.endswith(f"{wv.POD_SCRIPT} run --node pod-b --only gdn")
    for path in wv.POD_FILES:
        assert f"--with {path}" in command and (REPO / path).exists()
    assert (REPO / wv.POD_SCRIPT).exists()


def test_pod_command_sums_slot_minutes_and_rejects_unknown_classes():
    found = wv.pod_command(["conv", "launch"], quick=True)
    assert found["slot_minutes"] == 25 and found["command"].endswith("--only conv,launch --quick")
    with pytest.raises(SystemExit):
        wv.selected("gdn,warp")


def test_the_pod_shell_script_copies_exactly_the_uploaded_files():
    text = (REPO / wv.POD_SCRIPT).read_text()
    listed = re.search(r"for f in (.+?); do", text).group(1).split()
    assert sorted(listed) == sorted(Path(p).name for p in wv.POD_FILES)


def test_the_note_names_the_same_pod_commands_the_harness_prints():
    note = (REPO / "lab/refine/research/kernels/wavefront_training_gfx1151.md").read_text()
    for klass in ("gdn", "conv", "launch"):
        assert wv.pod_command([klass])["command"] in note
