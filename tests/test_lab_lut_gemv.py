"""R62: the lookup-table dequant GEMV (lab/kernels/lut_gemv.py).

Three layers: the byte accounting and the metrics rows (standard library only, always run); the packing and the
torch reference (need torch, run on the CPU); the Triton kernel against the reference at the chip sheet's 27B decode
shapes (need a GPU with torch and triton, skipped cleanly otherwise; each case prints its max abs error).
"""
from __future__ import annotations

import importlib.util
import json

import pytest

from lab.kernels import lut_gemv as L

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _gpu() -> bool:
    if not HAS_TORCH or importlib.util.find_spec("triton") is None:
        return False
    import torch

    return bool(torch.cuda.is_available())


needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch is not installed here")
needs_gpu = pytest.mark.skipif(not _gpu(), reason="no GPU with torch and triton here")


# --- no torch -----------------------------------------------------------------------------------------------------

def test_bytes_per_weight_match_the_packing():
    k, n = 5120, 6144
    # codes alone are exactly BITS bits a weight; scale and zero add 4 bytes per group of 128 per column
    for bits in (2, 3, 4):
        arith = L._bytes_of(bits, 0, k, n)
        assert arith == k * n * bits // 8 + (k // 128) * n * 4
        table = L._bytes_of(bits, 1, k, n)
        assert table == k * n * bits // 8 + (k // 128) * n * (1 << bits) * 2
    # the per-group table costs 1 bit a weight at 3 bits (16 bytes per 128 weights), the shared one nothing
    assert 8 * L._bytes_of(3, 1, k, n) / (k * n) == pytest.approx(4.0)
    assert 8 * L._bytes_of(3, 3, k, n) / (k * n) == pytest.approx(3.25, abs=1e-3)


def test_configs_divide_the_decode_shapes():
    for cfg in L.M1_CONFIGS + L.DOT_CONFIGS:
        assert L.GROUP % cfg["BLOCK_K"] == 0 and cfg["BLOCK_K"] % 32 == 0
        for _, k, _ in L.DECODE_SHAPES:
            assert k % cfg["SPLIT_K"] == 0 and (k // cfg["SPLIT_K"]) % cfg["BLOCK_K"] == 0
    assert len(L.M1_CONFIGS) >= 3  # the kill criterion counts three tile configurations


def test_metric_rows_carry_conditions_and_names():
    result = {
        "format": L.FORMAT, "node": "pod-b", "gfx": "gfx1151", "date": "2026-10-11", "host": "h",
        "ended": "2026-10-11T05:40:00Z", "versions": {"torch": "2.9.1", "triton": "3.5.1", "rocm": "7.13.0"},
        "power": {"reader": {"tool": "amd-smi", "period_s": 0.25}},
        "measurements": [
            {"id": "lut_gemv.b3.group_lut.k5120_n6144.m1.v_bn32_bk128_s1_w4", "section": "table", "status": "ok",
             "repeats": 11, "spread": {"rel_spread": 0.02},
             "params": {"bits": 3, "mode": "group_lut", "m": 1, "k": 5120, "n": 6144},
             "derived": {"weight_gbs": 150.0, "tflops": 0.5, "seconds_median": 1e-4, "max_abs_error": 2e-4},
             "watts": {"loaded_mean_w": 100.0, "idle_w": 40.0, "samples": 4}},
            {"id": "lut_gemv.b3.arith.k5120_n6144.m1.x.sweep", "section": "sweep", "status": "ok",
             "params": {}, "derived": {"weight_gbs": 1.0}, "watts": {}},
            {"id": "lut_gemv.b2.group_lut.k5120_n6144.m1.x", "section": "table", "status": "failed",
             "params": {}, "derived": {}, "watts": {}},
        ],
    }
    rows = L.metric_rows(result, "fixtures/lab/lut_gemv/x.json", "D")
    names = {r["metric"]: r for r in rows}
    stem = "lut_gemv.b3.group_lut.k5120_n6144.m1.v_bn32_bk128_s1_w4"
    assert names[f"{stem}.weight_gbs"]["value"] == 150.0
    assert names[f"{stem}.joules_per_gemv"]["value"] == pytest.approx(0.01)
    assert names[f"{stem}.max_abs_error"]["unit"] == "abs"
    row = names[f"{stem}.watts"]
    assert row["node"] == "pod-b" and row["conditions"]["hub"] == "sjc" and row["conditions"]["account"] == "D"
    assert row["conditions"]["sampler"]["tool"] == "amd-smi"
    assert not any(r["metric"].endswith(".sweep.weight_gbs") for r in rows)  # the sweep stays in the file
    assert not any(r["metric"].startswith("lut_gemv.b2.") for r in rows)
    assert names[f"{stem}.watts"]["low_confidence"] is False
    from lab.machine import metrics as ledger_metrics

    for row in rows:
        ledger_metrics.check_row(row)
    json.dumps(rows)


# --- torch on the CPU ---------------------------------------------------------------------------------------------

@needs_torch
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_packing_round_trips_and_table_is_ordinary_quantization(bits):
    import torch

    gen = torch.Generator().manual_seed(bits)
    w = (torch.randn(64, 512, generator=gen) * 0.02).half()
    p = L.quantize(w, bits)
    assert p.main.dtype == torch.int32 and p.main.shape == (512 * L.main_bits(bits) // 32, 64)
    assert (p.hi is not None) == (bits == 3)
    assert torch.equal(L.unpack_codes(p), p.codes.long())
    # the per-group table, the formula and the shared table all give the same weights for the uniform grid
    arith = L.dequant(p, 0)
    assert torch.equal(L.dequant(p, 1), arith) and torch.equal(L.dequant(p, 2), arith)
    assert torch.equal(L.dequant(p, 3), arith)
    # and they are a BITS-bit quantization of w: every error within half a step of its group's scale
    step = p.scale.float().t().repeat_interleave(L.GROUP, dim=1)
    assert float(((arith.float() - w.float()).abs() - step / 2).max()) <= 1e-3


@needs_torch
def test_packed_bytes_are_bits_per_weight():
    import torch

    w = torch.zeros(32, 256).half()
    for bits in (2, 3, 4):
        p = L.quantize(w, bits)
        codes = p.main.numel() * 4 + (p.hi.numel() * 4 if p.hi is not None else 0)
        assert codes * 8 == 32 * 256 * bits
        assert p.nbytes(1) == L._bytes_of(bits, 1, 256, 32)
        assert p.nbytes(0) == L._bytes_of(bits, 0, 256, 32)


# --- the kernel on a GPU ------------------------------------------------------------------------------------------

@needs_gpu
@pytest.mark.parametrize("shape", L.DECODE_SHAPES, ids=[s[0] for s in L.DECODE_SHAPES])
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_kernel_matches_reference_at_decode_shapes(shape, bits):
    import torch

    name, k, n = shape
    gen = torch.Generator(device="cuda").manual_seed(k + n + bits)
    w = (torch.randn(n, k, device="cuda", generator=gen) * 0.02).half()
    p = L.quantize(w, bits)
    for m in (1, 4, 16):
        a = torch.randn(m, k, device="cuda", generator=gen).half()
        for mode in L.MODES:
            out = L.lut_gemv(a, p, mode).float()
            ref = L.reference(a, p, mode)
            err = float((out - ref).abs().max())
            rel = float((out - ref).norm() / ref.norm())
            print(f"{name} bits {bits} m {m} {L.MODES[mode]}: max abs error {err:.3g}, relative L2 {rel:.3g}")
            assert rel < 1e-2, (name, bits, m, L.MODES[mode], err, rel)
