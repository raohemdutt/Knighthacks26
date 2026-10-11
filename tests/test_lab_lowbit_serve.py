"""Guards on the low-bit teacher's serving side: the vLLM patch, the pod serve script and the prediction and report.

The patch is checked for what it changes and, where torch is installed, by running its 3-bit zero-point packer
against the kernel's own reader (MatrixView_q3_row in vLLM's csrc, transcribed below as the oracle) and against
compressed-tensors' dense layout. The serve script is run in DRY_RUN and against a fake rocminfo. The report is fed
bench rows and eval files shaped like the real tools' output.
"""

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from lab import lowbit_report as lr

REPO_ROOT = Path(__file__).resolve().parents[1]
DIFF = REPO_ROOT / "lab" / "vllm-lowbit.diff"
SERVE = REPO_ROOT / "lab" / "serve_lowbit.sh"
CHIPSHEET = json.loads(lr.CHIPSHEET.read_text(encoding="utf-8"))


# --- the vLLM patch -------------------------------------------------------------------------------------------------

def diff_text():
    return DIFF.read_text(encoding="utf-8")


def test_the_patch_touches_only_the_exllama_kernel():
    files = re.findall(r"^diff --git a/(\S+) b/", diff_text(), re.M)
    assert files == ["vllm/model_executor/kernels/linear/mixed_precision/exllama.py"]


def test_the_patch_accepts_the_two_low_bit_types():
    added = "\n".join(l[1:] for l in diff_text().splitlines() if l.startswith("+") and not l.startswith("+++"))
    assert "scalar_types.uint2b2" in added and "scalar_types.uint3b4" in added
    assert "_pack_3bit_rows_dense(zeros)" in added


def test_the_header_names_the_target_commit_and_the_b1388b1_answer():
    header = diff_text().split("diff --git", 1)[0]
    assert "276fbcff2717" in header
    assert "b1388b1" in header and "does NOT apply" in header
    assert "--dtype float16" in header


def helper_source() -> str:
    """The two module-level helpers the patch adds, as Python source."""
    lines = [l[1:] for l in diff_text().splitlines() if l.startswith("+") and not l.startswith("+++")]
    start = next(i for i, l in enumerate(lines) if l.startswith("def _zp_pack_columns"))
    end = next(i for i, l in enumerate(lines) if l.startswith("    return words.reshape"))
    return "\n".join(lines[start:end + 1]) + "\n"


def test_the_added_helpers_are_valid_python():
    compile(helper_source(), "exllama_helpers", "exec")


def q3_row_item(data, width, row, column):
    """MatrixView_q3_row::item from vLLM csrc/libtorch_stable/quantization/gptq/matrix_view.cuh (276fbcf)."""
    z_w = column * 3 // 32
    z_mod = column & 0x1F
    base = row * width * 3 // 32
    if z_mod == 10:
        return (data[base + z_w] >> 30) | ((data[base + z_w + 1] << 2) & 0x4)
    if z_mod == 21:
        return (data[base + z_w] >> 31) | ((data[base + z_w + 1] << 1) & 0x6)
    if z_mod < 10:
        return (data[base + z_w] >> (z_mod * 3)) & 0x07
    if z_mod < 21:
        return (data[base + z_w] >> (z_mod * 3 - 32)) & 0x07
    return (data[base + z_w] >> (z_mod * 3 - 64)) & 0x07


def test_the_3bit_packer_round_trips_through_the_kernel_reader():
    torch = pytest.importorskip("torch")
    ns = {"torch": torch}
    exec(helper_source(), ns)
    rows, cols = 3, 64
    values = (torch.arange(rows * cols, dtype=torch.int32).reshape(rows, cols) * 5 + 3) % 8
    packed = ns["_pack_3bit_rows_dense"](values)
    assert packed.dtype == torch.int32 and tuple(packed.shape) == (rows, cols * 3 // 32)
    data = [int(x) & 0xFFFFFFFF for x in packed.flatten().tolist()]
    for r in range(rows):
        for c in range(cols):
            assert q3_row_item(data, cols, r, c) == int(values[r, c]), (r, c)


def test_the_3bit_packer_matches_the_constant_zero_point_the_kernel_expects():
    torch = pytest.importorskip("torch")
    ns = {"torch": torch}
    exec(helper_source(), ns)
    zeros = torch.full((2, 32), 3, dtype=torch.int32)    # uint3b4 bias 4, minus the GPTQ v1 off-by-one
    data = [int(x) & 0xFFFFFFFF for x in ns["_pack_3bit_rows_dense"](zeros).flatten().tolist()]
    assert all(q3_row_item(data, 32, r, c) == 3 for r in range(2) for c in range(32))


def test_zero_point_columns_rule():
    torch = pytest.importorskip("torch")
    ns = {"torch": torch}
    exec(helper_source(), ns)

    class T:
        def __init__(self, bits):
            self.size_bits = bits

    assert ns["_zp_pack_columns"](T(3)) == 32
    assert ns["_zp_pack_columns"](T(2)) == 16
    assert ns["_zp_pack_columns"](T(4)) == 8


# --- the serve script -----------------------------------------------------------------------------------------------

def serve(*args, env=None, step="all"):
    full = {k: v for k, v in os.environ.items() if k not in ("WIDTH", "HUB_REPO", "DRY_RUN")}
    full.update(env or {})
    return subprocess.run(["bash", str(SERVE), step], env=full, capture_output=True, text=True)


def test_dry_run_prints_download_patch_and_a_float16_bench_at_batch_1_and_4():
    out = serve(env={"DRY_RUN": "1", "WIDTH": "3", "HUB_REPO": "someone/Qwen3.8-27B-W3A16-G128-GPTQ-atlas"})
    assert out.returncode == 0, out.stderr
    text = out.stdout
    assert "hf download someone/Qwen3.8-27B-W3A16-G128-GPTQ-atlas" in text
    assert "apply --check" in text and "vllm-lowbit.diff" in text
    assert "lab/bench.py run" in text
    assert "float16" in text and "--batches 1\\,4" in text and "--no-64" in text
    assert "teacher-w3" in text


def test_refuses_widths_other_than_3_and_2():
    out = serve(env={"DRY_RUN": "1", "WIDTH": "4", "HUB_REPO": "a/b"})
    assert out.returncode == 2 and "WIDTH" in out.stderr


def test_refuses_a_missing_or_malformed_hub_repo():
    assert serve(env={"DRY_RUN": "1", "WIDTH": "2"}).returncode == 2
    assert serve(env={"DRY_RUN": "1", "WIDTH": "2", "HUB_REPO": "noslash"}).returncode == 2


def fake_rocminfo(tmp_path, gfx):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    tool = bin_dir / "rocminfo"
    tool.write_text(f"#!/bin/bash\necho '  Name:                    {gfx}'\n", encoding="utf-8")
    tool.chmod(0o755)
    return f"{bin_dir}:{os.environ.get('PATH', '')}"


def test_the_node_gate_refuses_another_chip(tmp_path):
    out = serve(env={"WIDTH": "3", "HUB_REPO": "a/b", "PATH": fake_rocminfo(tmp_path, "gfx1201")}, step="patch")
    assert out.returncode == 2 and "gfx1201" in out.stderr


def test_the_node_gate_refuses_a_machine_without_rocm(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    out = serve(env={"WIDTH": "3", "HUB_REPO": "a/b", "PATH": f"{empty}:/usr/bin:/bin"}, step="patch")
    assert out.returncode == 2 and "expected gfx1151" in out.stderr


def test_on_gfx1151_it_still_needs_the_upgraded_venv(tmp_path):
    out = serve(env={"WIDTH": "3", "HUB_REPO": "a/b", "PATH": fake_rocminfo(tmp_path, "gfx1151"),
                     "VENV": str(tmp_path / "no-venv")}, step="patch")
    assert out.returncode == 2 and "vllm_upgrade.sh" in out.stderr


# --- the prediction and the report ----------------------------------------------------------------------------------

def test_bandwidth_is_the_dram_copy_rate_of_the_chip_sheet():
    assert 205 < lr.bandwidth_gbs(CHIPSHEET) < 215


def test_the_4bit_prediction_matches_the_serving_checkpoints_byte_budget():
    rows = {(r["width"], r["batch"]): r for r in lr.predict(CHIPSHEET)["rows"]}
    assert rows[(4, 1)]["step_gb"] == pytest.approx(15.44, abs=0.01)
    assert rows[(4, 1)]["ceiling_tok_s"] == pytest.approx(13.6, abs=0.1)
    assert rows[(4, 1)]["j_per_token"] == [5.5, 9.4]


def test_fewer_bits_read_fewer_bytes_and_raise_the_ceiling():
    rows = {(r["width"], r["batch"]): r for r in lr.predict(CHIPSHEET)["rows"]}
    for batch in lr.BATCHES:
        assert rows[(4, batch)]["step_gb"] > rows[(3, batch)]["step_gb"] > rows[(2, batch)]["step_gb"]
        assert rows[(4, batch)]["ceiling_tok_s"] < rows[(3, batch)]["ceiling_tok_s"] < rows[(2, batch)]["ceiling_tok_s"]
    assert rows[(3, 1)]["bits_per_weight"] == 3.125 and rows[(2, 1)]["bits_per_weight"] == 2.125


def write_session(folder: Path, rates: dict[int, list[float]], watts: float = 100.0):
    folder.mkdir(parents=True)
    lines = []
    for batch, values in rates.items():
        for v in values:
            lines.append(json.dumps({"outcome": "ok", "batch_size": batch, "decode_tok_s": v,
                                     "decode_tok_s_total": v * batch, "watts": {"avg_w": watts},
                                     "joules_per_token": watts / (v * batch),
                                     "conditions": {"checkpoint": "/home/jovyan/models/x"}}))
    lines.append(json.dumps({"outcome": "failed", "batch_size": 1, "decode_tok_s": None}))
    (folder / "rows.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_eval(path: Path, score: float):
    path.write_text(json.dumps({"score": score, "model": "teacher", "results": []}), encoding="utf-8")


def test_report_reads_sessions_and_evals_and_names_the_lowest_width_within_margin(tmp_path):
    for width, rate in ((4, 7.5), (3, 9.0), (2, 11.0)):
        write_session(tmp_path / f"s{width}", {1: [rate, rate + 0.2, rate - 0.2], 4: [rate * 0.8] * 3})
    write_eval(tmp_path / "e4.json", 0.90)
    write_eval(tmp_path / "e3.json", 0.88)
    write_eval(tmp_path / "e2.json", 0.70)
    out = tmp_path / "lowbit.json"
    rc = lr.main(["report", "--bench", f"4={tmp_path / 's4'}", "--bench", f"3={tmp_path / 's3'}",
                  "--bench", f"2={tmp_path / 's2'}", "--eval", f"4={tmp_path / 'e4.json'}",
                  "--eval", f"3={tmp_path / 'e3.json'}", "--eval", f"2={tmp_path / 'e2.json'}",
                  "--margin", "0.05", "--date", "2026-10-11", "--out", str(out)])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["chosen"]["width"] == 3
    by_width = {w["width"]: w for w in data["widths"]}
    b1 = by_width[3]["batches"][0]
    assert b1["batch"] == 1 and b1["spawns"] == 3 and b1["decode_tok_s"] == pytest.approx(9.0)
    assert b1["fraction_of_ceiling"] == pytest.approx(9.0 / b1["ceiling_tok_s"], abs=0.001)
    assert by_width[2]["eval_drop"] == pytest.approx(0.20)


def test_without_a_margin_no_width_is_named():
    assert lr.choose({4: 0.9, 3: 0.9}, None)["width"] is None


def test_no_width_is_named_when_every_drop_is_too_large():
    assert lr.choose({4: 0.9, 3: 0.7, 2: 0.5}, 0.05)["width"] is None
    assert lr.choose({4: 0.9, 3: 0.88, 2: 0.87}, 0.05)["width"] == 2


def test_a_bad_pair_is_an_error_not_a_crash(capsys):
    assert lr.main(["report", "--bench", "5=somewhere"]) == 2
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_correctness_compare_lines_are_read(tmp_path):
    line = {"action": "compare", "prompts": 4, "matches": 1, "match_rate": 0.25, "normalised_text_matches": 2,
            "condition_drift": True}
    path = tmp_path / "c.json"
    path.write_text(json.dumps(line) + "\n", encoding="utf-8")
    assert lr.read_correctness(path) == {"prompts": 4, "match_rate": 0.25, "normalised_text_matches": 2,
                                         "condition_drift": True}
