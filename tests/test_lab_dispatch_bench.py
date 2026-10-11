"""R63: the head-to-head of the lookup-table GEMV against vLLM's hybrid W4A16 kernel (lab/kernels/dispatch_bench.py).

The byte accounting, the routing, the decision and the metrics rows need only the standard library; the packing and
the reference need torch on the CPU and are skipped without it. Nothing here touches a GPU.
"""
from __future__ import annotations

import importlib.util

import pytest

from lab.kernels import dispatch_bench as D

HAS_TORCH = importlib.util.find_spec("torch") is not None
needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch is not installed here")


def test_bytes_and_routing():
    assert D.common_bytes(5120, 6144) == 5120 * 6144 // 2 + 40 * 6144 * 2
    assert D.own_bytes("hybrid", 5120, 6144) == D.common_bytes(5120, 6144)
    assert D.own_bytes("lut.group_lut.v_bn32_bk128_s1_w4", 5120, 6144) == 5120 * 6144 // 2 + 40 * 6144 * 32
    assert D.own_bytes("lut.arith.v_bn32_bk128_s1_w4", 5120, 6144) == 5120 * 6144 // 2 + 40 * 6144 * 4
    assert D.SKINNY_LIMIT == 39321
    assert D.hybrid_route(1, 17408) == "hip_wvsplitk_int4"
    assert D.hybrid_route(2, 17408) == "hip_wvsplitk_int4"
    assert D.hybrid_route(4, 17408) == "triton_skinny_fmt"
    assert D.hybrid_route(4, 6144) == "hip_wvsplitk_int4"
    assert D.hybrid_route(16, 5120) == "triton_skinny_fmt"
    assert D.cold_copies(5120, 6144) * D.common_bytes(5120, 6144) >= D.COLD_BYTES


def test_fp16_rounding_rule():
    assert D.at_fp16_rounding(0.01, 5e-4, 10.0)
    assert not D.at_fp16_rounding(0.1, 5e-4, 10.0)      # past 2^-8 of the largest output
    assert not D.at_fp16_rounding(0.01, 2e-3, 10.0)     # relative L2 past 1e-3
    assert not D.at_fp16_rounding(None, 5e-4, 10.0)
    assert not D.at_fp16_rounding(float("nan"), 5e-4, 10.0)


def _entry(shape, m, temp, kernel, us, fp16_ok=True, dtype="float16", section=None):
    k, n = {s[0]: (s[1], s[2]) for s in D.SHAPES}[shape]
    sec = us / 1e6
    return {"id": f"dispatch.{kernel}.{shape}.m{m}.{temp}", "status": "ok", "section": section or f"m{m}",
            "params": {"shape": shape, "k": k, "n": n, "m": m, "temp": temp, "kernel": kernel, "dtype": dtype},
            "derived": {"seconds_median": sec, "common_gbs": D.common_bytes(k, n) / sec / 1e9,
                        "max_abs_error": 0.001, "fp16_ok": fp16_ok, "route": "hip_wvsplitk_int4",
                        "timing": "graph x20"},
            "watts": {"loaded_mean_w": 80.0, "idle_w": 20.0, "samples": 4}, "repeats": 11,
            "spread": {"rel_spread": 0.02}}


def _result(lut_factor: float, fp16_ok: bool = True):
    entries = []
    for temp in ("cold", "hot"):
        for m in D.BATCHES:
            for shape, *_ in D.SHAPES:
                entries.append(_entry(shape, m, temp, "hybrid", 100.0 * m))
                entries.append(_entry(shape, m, temp, "lut.group_lut.v_bn32_bk128_s1_w4", 100.0 * lut_factor * m,
                                      fp16_ok))
                entries.append(_entry(shape, m, temp, "lut.arith.v_bn32_bk128_s1_w4", 130.0 * m, fp16_ok))
    entries.append(_entry("k5120_n6144", 1, "hot", "lut.arith.v_bn16_bk128_s2_w2", 1.0, section="sweep"))
    return {"format": D.FORMAT, "node": "pod-b", "gfx": "gfx1151", "date": "2026-10-11",
            "ended": "2026-10-11T06:30:00Z", "versions": {"vllm": "0.31.2.dev0", "torch": "2.12.0"},
            "power": {"reader": {"tool": "amd-smi", "period_s": 0.25}}, "measurements": entries}


def test_decide_lookup_loses_when_slower():
    out = D.decide(_result(1.5))
    v = out["verdict"]
    assert v["complete"] and v["lookup_wins"] is False and v["threshold_m"] is None
    assert v["lut_mode"] == "arith"            # 130 us beats 150 us
    assert v["ratio"] == pytest.approx(400 / 520, rel=1e-3)


def test_decide_lookup_wins_only_past_ten_percent():
    assert D.decide(_result(0.95))["verdict"]["lookup_wins"] is False     # 5 percent faster is not enough
    v = D.decide(_result(0.8))["verdict"]
    assert v["lookup_wins"] is True and v["lut_mode"] == "group_lut" and v["threshold_m"] == 16
    assert D.decide(_result(0.8, fp16_ok=False))["verdict"]["lookup_wins"] is False


def test_decide_ignores_the_sweep_and_bf16():
    res = _result(1.5)
    res["measurements"].append(_entry("k5120_n6144", 1, "cold", "hybrid", 1.0, dtype="bfloat16"))
    row = next(r for r in D.decide(res)["table"]
               if r["shape"] == "k5120_n6144" and r["m"] == 1 and r["temp"] == "cold")
    assert row["hybrid_us"] == 100.0 and row["lut_us"] == 130.0


def test_decide_incomplete_result():
    res = _result(0.5)
    res["measurements"] = [e for e in res["measurements"] if e["params"]["shape"] != "k17408_n5120"]
    v = D.decide(res)["verdict"]
    assert v["complete"] is False and "lookup_wins" not in v


def test_metric_rows_pass_the_ledger_check():
    from lab.machine import metrics as ledger_metrics

    rows = D.metric_rows(_result(1.5), "fixtures/lab/dispatch/x.json")
    assert rows and not any(".sweep" in r["metric"] or "v_bn16" in r["metric"] for r in rows)
    for row in rows:
        ledger_metrics.check_row(row)
        assert row["metric"].startswith("dispatch.")
        assert row["conditions"]["hub"] == "sjc" and row["conditions"]["account"] == "D"
        assert row["node"] == "pod-b"
    names = {r["metric"].rsplit(".", 1)[1] for r in rows}
    assert {"common_gbs", "us_per_call", "watts", "joules_per_call", "max_abs_error"} <= names


@needs_torch
def test_ct_packing_round_trip_and_vllm_order():
    import torch

    gen = torch.Generator().manual_seed(0)
    codes = torch.randint(0, 16, (16, 256), generator=gen).to(torch.uint8)
    packed = D.pack_ct(codes)
    assert packed.shape == (16, 32) and packed.dtype == torch.int32
    assert torch.equal(D.unpack_ct(packed), codes)
    # vLLM's pack_quantized_values_into_int32(w, uint4b8, packed_dim=1): word j nibble i holds column 8 j + i
    ref = torch.zeros(16, 32, dtype=torch.int64)
    for i in range(8):
        ref |= (codes.to(torch.int64)[:, i::8] & 0xF) << (4 * i)
    ref = torch.where(ref >= 2 ** 31, ref - 2 ** 32, ref).to(torch.int32)
    assert torch.equal(packed, ref)


@needs_torch
def test_lut_packing_matches_the_symmetric_dequant():
    import torch

    from lab.kernels import lut_gemv as L

    gen = torch.Generator().manual_seed(1)
    n, k = 64, 256
    codes = torch.randint(0, 16, (n, k), generator=gen).to(torch.uint8)
    scale = (torch.rand(n, k // 128, generator=gen) * 0.004 + 0.001).to(torch.bfloat16)
    p = D.to_lut(codes, scale)
    w = D.dequant(codes, scale.half())
    for mode in (0, 1):
        assert (L.dequant(p, mode).float() - w).abs().max() <= w.abs().max() * 2 ** -10
    a = torch.randn(2, k)
    ref = a @ w.t()
    assert torch.allclose(L.reference(a, p, 0), ref, rtol=1e-3, atol=1e-3)
