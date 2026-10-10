"""lab/kernels/chipsheet: the tool parsers, the sheet format, the node gate, the self-check, the probe section run
against a stand-in probe program, and ingest's files and ledger rows. Never a live GPU: every tool answer is the
recorded output under fixtures/lab/chipsheet/ (synthetic numbers, see its README) or the amd-smi reading recorded
on pod C."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

import lab.__main__ as lab_cli
from lab.kernels.chipsheet import chipsheet as cs
from lab.kernels.chipsheet import chipsheet_format as fmt
from lab.kernels.chipsheet import ingest
from lab.machine import ledger

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "lab" / "chipsheet"


def fixture(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


def full_sheet(**changes):
    sheet = cs.self_check_sheet()
    sheet.update({"mode": "full", "date": "2026-10-11", "notes": []})
    sheet.update(changes)
    return sheet


# --- rocminfo and the gate --------------------------------------------------------------------------------------

def test_rocminfo_names_the_gpu_agent_and_skips_the_cpu():
    agent = fmt.parse_rocminfo(fixture("rocminfo_gfx1151_synthetic.txt"))
    assert agent["gfx"] == "gfx1151"
    assert agent["marketing_name"] == "AMD Radeon 8060S Graphics"
    assert (agent["compute_units"], agent["simds_per_cu"], agent["wavefront_size"]) == (40, 2, 32)
    assert (agent["workgroup_max_size"], agent["max_waves_per_cu"], agent["cacheline_bytes"]) == (1024, 32, 128)
    assert agent["cache_bytes"] == {"l1": 32 * 1024, "l2": 2048 * 1024, "l3": 32768 * 1024}
    assert agent["lds_bytes"] == 64 * 1024
    assert agent["max_clock_mhz"] == 2900


def test_rocminfo_without_a_gpu_agent_gives_none():
    text = fixture("rocminfo_gfx1151_synthetic.txt").split("Agent 2")[0]
    assert fmt.parse_rocminfo(text) is None
    assert fmt.parse_rocminfo("") is None


@pytest.mark.parametrize("node, text, machine, refused", [
    ("pod-b", "ok", "x86_64", None),
    ("pod-a", "ok", "x86_64", None),
    ("pod-c", "ok", "x86_64", "--node"),
    ("jetson", "ok", "aarch64", "--node"),
    ("pod-b", "ok", "aarch64", "aarch64"),
    ("pod-b", "r9700", "x86_64", "gfx1201"),
    ("pod-b", None, "x86_64", "cannot tell"),
    ("pod-b", "cpu-only", "x86_64", "no GPU agent"),
])
def test_the_gate_admits_only_a_strix_halo_pod(node, text, machine, refused):
    full = fixture("rocminfo_gfx1151_synthetic.txt")
    texts = {"ok": full, "r9700": full.replace("gfx1151", "gfx1201"), "cpu-only": full.split("Agent 2")[0],
             None: None}
    answer = cs.gate(node, texts[text], "rocminfo is not installed", machine=machine)
    if refused is None:
        assert answer is None
    else:
        assert refused in answer


def test_run_refuses_on_a_machine_without_rocminfo_and_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cs, "read_rocminfo", lambda: (None, "rocminfo is not installed"))
    code = cs.main(["run", "--node", "pod-b", "--out", str(tmp_path / "out")])
    answer = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == cs.EXIT_GATE and answer["outcome"] == "refused"
    assert not (tmp_path / "out").exists()


# --- versions, probe lines and remarks --------------------------------------------------------------------------

def test_versions_parse_from_hipcc_and_amd_smi():
    assert fmt.parse_hipcc_version(fixture("hipcc_version_synthetic.txt")) == {
        "hip": "7.13.25412-0f1c2a3b4", "clang": "20.0.0git"}
    assert fmt.parse_rocm_version(fixture("amdsmi_version_synthetic.txt")) == "7.13.0"
    assert fmt.parse_rocm_version("7.13.0-71\n") == "7.13.0-71"
    assert fmt.parse_rocm_version("no version here") is None
    assert fmt.parse_hipcc_version("") == {"hip": None, "clang": None}


def test_probe_lines_keep_only_probe_objects():
    text = "warning: something\n" + fixture("probe_w32_synthetic.txt") + '{"not": "a probe"}\n{broken\n'
    lines = fmt.parse_probe_lines(text)
    assert [line["probe"] for line in lines] == ["props", "wave", "lds", "occ_lds", "occ_lds", "occ_vgpr", "occ_vgpr"]
    assert lines[2]["max_ok_bytes"] == 65536 and lines[2]["fail_error"] == "invalid argument"


def test_resource_remarks_per_kernel_without_the_pass_suffix():
    remarks = fmt.parse_resource_remarks(fixture("remarks_w32_synthetic.txt"))
    assert set(remarks) == {"occ_lds", "occ_vgpr_4", "occ_vgpr_240"}
    assert remarks["occ_vgpr_240"]["VGPRs"] == 245
    assert remarks["occ_vgpr_240"]["Occupancy [waves/SIMD]"] == 4
    assert remarks["occ_lds"]["LDS Size [bytes/block]"] == 0
    assert remarks["occ_lds"]["Dynamic Stack"] == "False"


# --- numbers ----------------------------------------------------------------------------------------------------

def test_stats_spread_of_repeats():
    got = fmt.stats([4.0, 1.0, 3.0, 2.0])
    assert (got["n"], got["min"], got["max"], got["median"], got["mean"]) == (4, 1.0, 4.0, 2.5, 2.5)
    assert got["p10"] == 1.0 and got["p90"] == 4.0
    assert got["rel_spread"] == pytest.approx(3.0 / 2.5)
    assert fmt.stats([])["median"] is None and fmt.stats([])["n"] == 0
    assert fmt.stats([5.0])["stdev"] == 0.0


@pytest.mark.parametrize("nbytes, label", [(fmt.MIB, "1mib"), (4096 * fmt.MIB, "4gib"), (1536, "1536b"),
                                           (2048, "2kib"), (64 * fmt.MIB, "64mib")])
def test_size_labels(nbytes, label):
    assert fmt.size_label(nbytes) == label


def test_loads_per_second_from_probe_times():
    found = {"blocks": 1280, "threads": 256, "iters": 512, "ms": [2.0, 4.0, 0.0]}
    assert cs._loads_per_s(found) == pytest.approx([1280 * 256 * 512 / 2e-3 / 1e9, 1280 * 256 * 512 / 4e-3 / 1e9])


# --- the sheet format -------------------------------------------------------------------------------------------

def test_the_self_check_sheet_is_valid():
    assert fmt.validate(cs.self_check_sheet()) == []


def test_validate_names_every_problem():
    sheet = full_sheet(node="pod-c", mode="slow", format="x", date="11/10/2026", gfx="radeon")
    items = sheet["measurements"]
    items.append(dict(items[0]))                                # a repeated id
    items.append({**items[1], "id": "Matmul.Bad"})              # not dotted lowercase
    items.append({**items[1], "id": "matmul.no_value", "value": None})
    items.append({**items[2], "id": "dequant.with_value", "value": 1.0})
    items.append({**items[1], "id": "matmul.repeats", "repeats": 7})
    problems = " | ".join(fmt.validate(sheet))
    for expected in ("node must be", "mode must be", "format must be", "date must be", "gfx must name",
                     "id is used twice", "id must be dotted", "an ok entry needs a finite value",
                     "a skipped entry has no value", "repeats must equal"):
        assert expected in problems


def test_write_and_read_round_trip(tmp_path):
    sheet = full_sheet()
    json_path, md_path = fmt.write(sheet, tmp_path)
    assert json_path.name == "chipsheet-gfx1151-2026-10-11.json" and md_path.suffix == ".md"
    assert fmt.read(json_path) == json.loads(json.dumps(sheet))
    text = md_path.read_text(encoding="utf-8")
    assert text.startswith("# Chip sheet: gfx1151 on pod-b, 2026-10-11")
    assert "## Device-to-device copy bandwidth" in text and "## Occupancy curve" in text
    assert "`python3 chipsheet.py self-check`" in text                      # every number carries its command
    assert "dequant.w4a16.decode27b.k5120_n6144.m1.bn64_bk64: skipped" in text
    assert "| max clock MHz | 2,900 | 2,900 |" in text
    assert "Self-check sheet" not in text


def test_a_self_check_sheet_says_so_in_its_name_and_text(tmp_path):
    json_path, md_path = fmt.write(cs.self_check_sheet(), tmp_path)
    assert json_path.name.endswith("-self-check.json")
    assert "Self-check sheet: parsed fixtures and invented timings" in md_path.read_text(encoding="utf-8")


def test_an_invalid_sheet_writes_nothing(tmp_path):
    with pytest.raises(ValueError, match="node must be"):
        fmt.write(full_sheet(node="pod-c"), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_self_check_command_writes_and_rereads(tmp_path, capsys):
    assert cs.main(["self-check", "--out", str(tmp_path)]) == 0
    answer = json.loads(capsys.readouterr().out)
    assert answer["ok"] and answer["outcome"] == "self_checked" and answer["entries"] >= 8
    assert Path(answer["json"]).exists() and Path(answer["markdown"]).exists()
    assert fmt.read(answer["json"])["mode"] == "self-check"


# --- the run machinery ------------------------------------------------------------------------------------------

def make_run(tmp_path, budget_s=100.0, quick=True):
    args = argparse.Namespace(quick=quick, budget_s=budget_s, out=str(tmp_path), lora_config=str(tmp_path / "x"),
                              only=None, date=None, node="pod-b")
    run = cs.Run(args, "python3 chipsheet.py run --node pod-b --quick")
    run.section = "copy"
    return run


def test_measure_keeps_samples_records_failures_and_skips_past_the_budget(tmp_path):
    run = make_run(tmp_path)
    run.measure("copy.d2d.size_1mib", "copy", "GB/s", "op", {"bytes": 1}, lambda: ([2.0, 1.0, 3.0], {"x": 1}))

    def broken():
        raise RuntimeError("out of memory")

    run.measure("copy.d2d.size_4gib", "copy", "GB/s", "op", {}, broken)
    run.t0 -= 1000                                              # the budget is spent
    run.measure("copy.d2d.size_2gib", "copy", "GB/s", "op", {}, lambda: ([1.0], {}))
    ok, failed, skipped = run.entries
    assert (ok["status"], ok["value"], ok["repeats"], ok["derived"]) == ("ok", 2.0, 3, {"x": 1})
    assert ok["command"] == "python3 chipsheet.py run --node pod-b --quick --only copy"
    assert ok["watts"]["error"] == "no power reader"
    assert failed["status"] == "failed" and "out of memory" in failed["error"] and failed["value"] is None
    assert skipped["status"] == "skipped" and skipped["error"].startswith("budget")


def test_facts_without_a_value_become_failed_entries(tmp_path):
    run = make_run(tmp_path)
    run.fact("cache.rocminfo.l3_bytes", "L3", "bytes", None, source="tool", command="rocminfo", op="L3",
             error="rocminfo printed no L3", section="cache")
    run.fact("cache.rocminfo.l2_bytes", "L2", "bytes", 2048, source="tool", command="rocminfo", op="L2",
             section="cache")
    missing, l2 = run.entries
    assert missing["status"] == "failed" and missing["error"] == "rocminfo printed no L3"
    assert l2["value"] == 2048.0 and l2["samples"] == [2048.0]


def test_lora_shapes_read_the_mlp_width_from_the_checkpoint(tmp_path):
    shapes, why = cs.lora_widths(str(tmp_path / "missing.json"))
    assert shapes == [("k2560_n2560", 2560, 2560)] and "MLP width unknown" in why
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"text_config": {"hidden_size": 2560, "intermediate_size": 9216}}))
    shapes, why = cs.lora_widths(str(config))
    assert shapes[1:] == [("k2560_n9216", 2560, 9216), ("k9216_n2560", 9216, 2560)] and "9216" in why
    config.write_text(json.dumps({"hidden_size": 4096, "intermediate_size": 9216}))
    assert len(cs.lora_widths(str(config))[0]) == 1


def fake_probe(tmp_path) -> Path:
    """A stand-in for the compiled probe: prints the recorded line for the mode (and point) it is asked for."""
    script = tmp_path / "probe_fake"
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        f"lines = [json.loads(l) for l in open({str(FIX / 'probe_w32_synthetic.txt')!r}) if l.startswith('{{')]\n"
        "mode = sys.argv[1]\n"
        "for line in lines:\n"
        "    if line['probe'] == mode and (len(sys.argv) < 3 or line.get('param') == int(sys.argv[2])):\n"
        "        print(json.dumps(line))\n", encoding="utf-8")
    script.chmod(0o755)
    return script


def test_the_probe_sections_from_a_stand_in_probe(tmp_path, monkeypatch):
    binary = fake_probe(tmp_path)
    remarks = fixture("remarks_w32_synthetic.txt")
    monkeypatch.setattr(cs, "compile_probe", lambda run, wave: (binary, f"hipcc wave{wave}", remarks))
    monkeypatch.setattr(cs.time, "sleep", lambda s: None)
    monkeypatch.setattr(cs, "QUICK_OCC_LDS", (0, 65536))
    monkeypatch.setattr(cs, "QUICK_OCC_VGPRS", (4, 240, 64))   # 64 has no recorded line: a failed point
    run = make_run(tmp_path)
    agent = fmt.parse_rocminfo(fixture("rocminfo_gfx1151_synthetic.txt"))
    hip = cs.section_probes(run, agent, "/opt/rocm/bin/rocminfo")
    got = fmt.by_id({"measurements": run.entries})
    assert hip["l2CacheSize"] == 2097152
    assert got["wave.w32.compiled"]["value"] == 32 and got["wave.w64.compiled"]["value"] == 32
    assert got["lds.max_launch_bytes"]["value"] == 65536
    assert got["lds.max_launch_bytes"]["params"]["first_fail_bytes"] == 65664
    assert got["lds.rocminfo.group_segment_bytes"]["value"] == 65536
    assert got["cache.rocminfo.l3_bytes"]["value"] == 32768 * 1024
    assert got["cache.hip.l2_bytes"]["value"] == 2097152
    best = got["occupancy.vgpr.p_4"]
    assert best["derived"]["relative_throughput"] == pytest.approx(1.0)
    assert got["occupancy.vgpr.p_240"]["derived"]["vgprs"] == 245
    assert got["occupancy.vgpr.p_240"]["derived"]["relative_throughput"] < 0.5
    assert got["occupancy.lds.p_65536"]["derived"]["hip_blocks_per_cu"] == 1
    assert got["occupancy.vgpr.p_64"]["status"] == "failed"
    assert all(m["command"] for m in run.entries)


# --- ingest: the data files and the ledger rows -----------------------------------------------------------------

def test_ingest_files_the_sheet_and_writes_rows_on_its_node(tmp_path):
    source = tmp_path / "pulled.json"
    source.write_text(json.dumps(full_sheet()))
    data, state = tmp_path / "data", tmp_path / "state"
    answer = ingest.ingest(source, data, state_dir=state)
    assert answer["ok"] and answer["outcome"] == "recorded"
    assert (data / "chipsheet-gfx1151-2026-10-11.json").exists()
    assert (data / "chipsheet-gfx1151-2026-10-11.md").exists()
    rows = ledger.read(state_dir=state, verb="chipsheet")
    assert [row["conditions"]["section"] for row in rows] == ["copy", "matmul", "lds", "wave", "occupancy", "cache"]
    assert {row["node"] for row in rows} == {"pod-b"}
    copy = rows[0]
    assert copy["metrics"] == {"chip.copy.d2d.size_64mib": 100.0}
    assert copy["units"] == {"chip.copy.d2d.size_64mib": "GB/s"}
    assert copy["conditions"]["spread"]["chip.copy.d2d.size_64mib"]["n"] == 3
    assert copy["refs"]["run_id"] == "chipsheet-gfx1151-2026-10-11" and copy["low_confidence"] is False
    again = ingest.ingest(source, data, state_dir=state)
    assert again["outcome"] == "already_recorded"
    assert len(ledger.read(state_dir=state, verb="chipsheet")) == len(rows)


def test_ingest_refuses_a_self_check_sheet_and_an_invalid_one(tmp_path):
    source = tmp_path / "s.json"
    source.write_text(json.dumps(cs.self_check_sheet()))
    assert ingest.ingest(source, tmp_path / "data", state_dir=tmp_path)["outcome"] == "refused"
    source.write_text(json.dumps(full_sheet(node="jetson")))
    assert ingest.ingest(source, tmp_path / "data", state_dir=tmp_path)["outcome"] == "invalid"
    assert not (tmp_path / "data").exists() and ledger.read(state_dir=tmp_path) == []


def test_a_dry_run_writes_nothing_and_a_quick_sheet_is_low_confidence(tmp_path):
    source = tmp_path / "q.json"
    source.write_text(json.dumps(full_sheet(mode="quick")))
    answer = ingest.ingest(source, tmp_path / "data", dry_run=True, state_dir=tmp_path)
    assert answer["outcome"] == "dry_run" and all(row["low_confidence"] for row in answer["rows"])
    assert not (tmp_path / "data").exists() and ledger.read(state_dir=tmp_path) == []


def test_the_verb_is_registered_in_order():
    names = list(lab_cli.VERBS)
    assert lab_cli.VERBS["chipsheet"] == "lab.kernels.chipsheet.ingest:main" and names == sorted(names)
