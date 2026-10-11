"""Lookup-table dequant GEMV for 2-, 3- and 4-bit weight-only quantized layers on gfx1151 (R62).

    python3 lut_gemv.py bench --node pod-b [--out ~/r62_lut] [--budget-s 600] [--quick]
    python -m lab.kernels.lut_gemv ingest --result FILE [--metrics lab/refine/metrics.jsonl] [--dry-run]

The idea is LUT-GEMM (arXiv 2206.09557) and FLUTE (arXiv 2407.10960): at 3 bits the unpacking arithmetic, not the
DRAM read, can bound a decode GEMV, so the dequantization becomes a gather from a small table of the 2^BITS grid
values kept on chip. One Triton kernel serves every BITS through a constexpr, with four ways to turn a code into a
weight (MODE):

  0 arith         w = (c - z) * s, the plain per-group scale and zero point (the control, same packing)
  1 group_lut     a per-group, per-column fp16 table of the 2^BITS values, loaded into registers once per K block
                  and read with tl.gather (the compiler lowers it to lane shuffles or LDS: the on-chip lookup)
  2 group_lut_ld  the same table, read with one pointer gather per weight through the vector cache
  3 shared_lut    one table of 2^BITS values for the whole matrix (FLUTE's layout), gathered on chip, then
                  w = (T[c] - z) * s with the per-group scale and zero

The table is filled from a uniform asymmetric grid, so the reference is ordinary min-max quantization; the
mechanism is what is measured, and a non-uniform grid (NF3, a learned codebook) only changes the table's values.

Packing (codes c in [0, 2^BITS), weight matrix W [N, K] as nn.Linear holds it, groups of GROUP = 128 along K):
  main   int32 [K // PW, N]: PW = 32 // MB codes per word, MB = 4 bits for BITS 4 and 2 bits otherwise;
         bits MB*j .. MB*j + MB - 1 of main[kw, n] hold the low MB bits of c[n, kw * PW + j]
  hi     int32 [K // 32, N], BITS 3 only: bit j of hi[kw, n] holds bit 2 of c[n, kw * 32 + j]
  table  fp16 [K // GROUP, N, 2^BITS]: table[g, n, c] = (c - z[g, n]) * s[g, n] in fp32, rounded to fp16
  scale, zero  fp16 [K // GROUP, N]; shared fp16 [2^BITS] = 0, 1, ..., 2^BITS - 1
So 3 bits cost exactly 3 bits per weight in codes, and BITS 4 is the chip sheet's int4 layout's size.

bench runs on the pod (uploaded next to chipsheet.py, chipsheet_format.py and pod_power.py, whose timing helper,
power reader and entry format it reuses) and writes <out>/lut_gemv-<gfx>-<date>.json; ingest turns that file into
lut_gemv.* rows of lab/refine/metrics.jsonl. On a machine with no GPU nothing here touches a device: the packing and
the torch reference run on the CPU.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import shlex
import socket
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GROUP = 128
MODES = {0: "arith", 1: "group_lut", 2: "group_lut_ld", 3: "shared_lut"}
MODE_IDS = {name: mode for mode, name in MODES.items()}
FORMAT = "lut_gemv/1"
DEFAULT_OUT = "~/r62_lut"
FULL_BUDGET_S = 600.0
QUICK_BUDGET_S = 150.0

# The chip sheet's 27B decode shapes as (name, K, N) (lab/kernels/chipsheet/chipsheet.py DECODE_SHAPES).
DECODE_SHAPES = (("k5120_n6144", 5120, 6144), ("k6144_n5120", 6144, 5120),
                 ("k5120_n17408", 5120, 17408), ("k17408_n5120", 17408, 5120))
KILL_SHAPE = DECODE_SHAPES[0]

# Tile configurations. M1 configs reduce over K with a vector multiply (one program per row, BLOCK_M 1); DOT
# configs use tl.dot with BLOCK_M 16, the chip sheet's W4A16 tile being the first.
M1_CONFIGS = (
    {"tag": "v_bn32_bk128_s1_w4", "BLOCK_M": 1, "BLOCK_N": 32, "BLOCK_K": 128, "SPLIT_K": 1, "num_warps": 4,
     "USE_DOT": False},
    {"tag": "v_bn64_bk128_s4_w4", "BLOCK_M": 1, "BLOCK_N": 64, "BLOCK_K": 128, "SPLIT_K": 4, "num_warps": 4,
     "USE_DOT": False},
    {"tag": "v_bn16_bk128_s2_w2", "BLOCK_M": 1, "BLOCK_N": 16, "BLOCK_K": 128, "SPLIT_K": 2, "num_warps": 2,
     "USE_DOT": False},
)
DOT_CONFIGS = (
    {"tag": "d_bm16_bn64_bk64_s1_w4", "BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64, "SPLIT_K": 1, "num_warps": 4,
     "USE_DOT": True},
    {"tag": "d_bm16_bn32_bk128_s2_w4", "BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 128, "SPLIT_K": 2, "num_warps": 4,
     "USE_DOT": True},
    {"tag": "d_bm16_bn64_bk128_s4_w4", "BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 128, "SPLIT_K": 4, "num_warps": 4,
     "USE_DOT": True},
)
ALL_CONFIGS = {cfg["tag"]: cfg for cfg in M1_CONFIGS + DOT_CONFIGS}


# --- packing and the torch reference (CPU or GPU) ----------------------------------------------------------------

@dataclass
class Packed:
    bits: int
    n: int
    k: int
    codes: Any      # uint8 [N, K], kept for the reference
    main: Any       # int32 [K // PW, N]
    hi: Any         # int32 [K // 32, N] or None
    table: Any      # fp16 [K // GROUP, N, 2^BITS]
    scale: Any      # fp16 [K // GROUP, N]
    zero: Any       # fp16 [K // GROUP, N]
    shared: Any     # fp16 [2^BITS]

    @property
    def main_bits(self) -> int:
        return main_bits(self.bits)

    def nbytes(self, mode: int) -> int:
        """Bytes the kernel must read for the weights in this mode: the codes plus the per-group metadata (the
        tables for modes 1 and 2, scale and zero for 0 and 3; the shared table is 2^BITS x 2 bytes, counted)."""
        codes = self.main.numel() * 4 + (self.hi.numel() * 4 if self.hi is not None else 0)
        if mode in (1, 2):
            return codes + self.table.numel() * 2
        return codes + self.scale.numel() * 2 + self.zero.numel() * 2 + (self.shared.numel() * 2 if mode == 3 else 0)


def main_bits(bits: int) -> int:
    if bits not in (2, 3, 4):
        raise ValueError(f"BITS must be 2, 3 or 4, not {bits}")
    return 4 if bits == 4 else 2


def _to_int32(words: Any) -> Any:
    """int64 values in [0, 2^32) to int32 with the same bits."""
    import torch

    return torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)


def _pack_field(field: Any, width: int) -> Any:
    """field int64 [N, K] of width-bit values to int32 [K // (32 // width), N], code j of a word at bits width*j."""
    import torch

    n, k = field.shape
    per = 32 // width
    shifts = torch.arange(per, dtype=torch.int64, device=field.device) * width
    words = (field.t().reshape(k // per, per, n) << shifts[None, :, None]).sum(dim=1)
    return _to_int32(words).contiguous()


def quantize(w: Any, bits: int, group: int = GROUP) -> Packed:
    """Min-max asymmetric quantization of fp16 W [N, K] in groups of `group` along K, packed for the kernel."""
    import torch

    n, k = w.shape
    if k % group or k % 32:
        raise ValueError(f"K {k} must be a multiple of {group} and of 32")
    levels = (1 << bits) - 1
    wg = w.float().reshape(n, k // group, group)
    lo, hi = wg.amin(dim=2), wg.amax(dim=2)
    scale = ((hi - lo) / levels).clamp_min(1e-8).half()
    zero = torch.round(-lo / scale.float()).clamp(0, levels).half()
    codes = torch.round(wg / scale.float()[:, :, None] + zero.float()[:, :, None]).clamp(0, levels)
    codes = codes.reshape(n, k).to(torch.uint8)
    return pack(codes, scale.t().contiguous(), zero.t().contiguous(), bits)


def pack(codes: Any, scale: Any, zero: Any, bits: int) -> Packed:
    """codes uint8 [N, K]; scale and zero fp16 [K // GROUP, N]."""
    import torch

    n, k = codes.shape
    mb = main_bits(bits)
    c = codes.to(torch.int64)
    main = _pack_field(c & ((1 << mb) - 1), mb)
    hi = _pack_field((c >> 2) & 1, 1) if bits == 3 else None
    grid = torch.arange(1 << bits, dtype=torch.float32, device=codes.device)
    table = ((grid[None, None, :] - zero.float()[:, :, None]) * scale.float()[:, :, None]).half().contiguous()
    shared = grid.half().contiguous()
    return Packed(bits, n, k, codes, main, hi, table, scale, zero, shared)


def unpack_codes(p: Packed) -> Any:
    """The codes back from main and hi, [N, K] int64 (the packing's own check)."""
    import torch

    mb = p.main_bits
    per = 32 // mb
    shifts = torch.arange(per, dtype=torch.int64, device=p.main.device) * mb
    words = p.main.to(torch.int64) & 0xFFFFFFFF
    low = ((words[:, None, :] >> shifts[None, :, None]) & ((1 << mb) - 1)).reshape(p.k, p.n)
    if p.hi is not None:
        hw = p.hi.to(torch.int64) & 0xFFFFFFFF
        bit = (hw[:, None, :] >> torch.arange(32, device=hw.device)[None, :, None]) & 1
        low = low | (bit.reshape(p.k, p.n) << 2)
    return low.t().contiguous()


def dequant(p: Packed, mode: int) -> Any:
    """W_hat fp16 [N, K] exactly as the kernel forms each weight in `mode` (gather the table, or the formula)."""
    import torch

    codes = p.codes.to(torch.int64)
    g = torch.arange(p.k, device=codes.device) // GROUP
    if mode in (1, 2):
        cols = torch.arange(p.n, device=codes.device)
        return p.table[g[None, :], cols[:, None], codes]
    s = p.scale.float().t()[:, g]
    z = p.zero.float().t()[:, g]
    c = p.shared.float()[codes] if mode == 3 else codes.float()
    return ((c - z) * s).half()


def reference(a: Any, p: Packed, mode: int) -> Any:
    """fp32 A @ W_hat^T, the answer the kernel's fp16 output is compared with."""
    return a.float() @ dequant(p, mode).float().t()


# --- the Triton kernel --------------------------------------------------------------------------------------------

_KERNEL: Any = None


def kernel() -> Any:
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def lut_gemv_kernel(a_ptr, main_ptr, hi_ptr, tab_ptr, sh_ptr, s_ptr, z_ptr, c_ptr, M, N, K,
                        stride_am, stride_cm,
                        BITS: tl.constexpr, MB: tl.constexpr, PW: tl.constexpr, NV: tl.constexpr,
                        MODE: tl.constexpr, GROUP: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                        BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr, USE_DOT: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_k = tl.program_id(2)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        nmask = rn < N
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        if USE_DOT:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        else:
            acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        k_len = K // SPLIT_K
        k_start = pid_k * k_len
        shifts = tl.arange(0, PW) * MB
        if MODE == 3:
            shared = tl.broadcast_to(tl.load(sh_ptr + tl.arange(0, NV))[None, :], (BLOCK_N, NV))
        for kk in range(0, k_len, BLOCK_K):
            k0 = k_start + kk
            g = k0 // GROUP
            rkw = k0 // PW + tl.arange(0, BLOCK_K // PW)
            mw = tl.load(main_ptr + rkw[None, :] * N + rn[:, None], mask=nmask[:, None], other=0)
            codes = (mw[:, :, None] >> shifts[None, None, :]) & ((1 << MB) - 1)
            codes = tl.reshape(codes, (BLOCK_N, BLOCK_K))
            if BITS == 3:
                rkh = k0 // 32 + tl.arange(0, BLOCK_K // 32)
                hw = tl.load(hi_ptr + rkh[None, :] * N + rn[:, None], mask=nmask[:, None], other=0)
                hb = (hw[:, :, None] >> tl.arange(0, 32)[None, None, :]) & 1
                codes = codes | (tl.reshape(hb, (BLOCK_N, BLOCK_K)) << 2)
            if MODE == 0:
                s = tl.load(s_ptr + g * N + rn, mask=nmask, other=0.0).to(tl.float32)
                z = tl.load(z_ptr + g * N + rn, mask=nmask, other=0.0).to(tl.float32)
                w = ((codes.to(tl.float32) - z[:, None]) * s[:, None]).to(tl.float16)
            elif MODE == 1:
                tab = tl.load(tab_ptr + (g * N + rn[:, None]) * NV + tl.arange(0, NV)[None, :],
                              mask=nmask[:, None], other=0.0)
                w = tl.gather(tab, codes, axis=1)
            elif MODE == 2:
                w = tl.load(tab_ptr + (g * N + rn[:, None]) * NV + codes, mask=nmask[:, None], other=0.0)
            else:
                s = tl.load(s_ptr + g * N + rn, mask=nmask, other=0.0).to(tl.float32)
                z = tl.load(z_ptr + g * N + rn, mask=nmask, other=0.0).to(tl.float32)
                t = tl.gather(shared, codes, axis=1).to(tl.float32)
                w = ((t - z[:, None]) * s[:, None]).to(tl.float16)
            if USE_DOT:
                rk = k0 + tl.arange(0, BLOCK_K)
                a = tl.load(a_ptr + rm[:, None] * stride_am + rk[None, :], mask=rm[:, None] < M, other=0.0)
                acc += tl.dot(a, tl.trans(w))
            else:
                a = tl.load(a_ptr + pid_m * stride_am + k0 + tl.arange(0, BLOCK_K)).to(tl.float32)
                acc += tl.sum(w.to(tl.float32) * a[None, :], axis=1)
        if USE_DOT:
            offs = rm[:, None] * stride_cm + rn[None, :]
            omask = (rm[:, None] < M) & nmask[None, :]
        else:
            offs = pid_m * stride_cm + rn
            omask = nmask
        if SPLIT_K == 1:
            tl.store(c_ptr + offs, acc.to(tl.float16), mask=omask)
        else:
            tl.atomic_add(c_ptr + offs, acc, mask=omask)

    _KERNEL = lut_gemv_kernel
    return _KERNEL


def check_config(p: Packed, cfg: Mapping[str, Any], m: int) -> None:
    bk, split = cfg["BLOCK_K"], cfg["SPLIT_K"]
    if GROUP % bk or bk % 32:
        raise ValueError(f"BLOCK_K {bk} must divide {GROUP} and be a multiple of 32")
    if p.k % split or (p.k // split) % bk:
        raise ValueError(f"K {p.k} does not split {split} ways into whole BLOCK_K {bk} blocks")
    if not cfg["USE_DOT"] and cfg["BLOCK_M"] != 1:
        raise ValueError("a vector (non-dot) configuration has BLOCK_M 1")
    if cfg["USE_DOT"] and cfg["BLOCK_M"] < 16:
        raise ValueError("tl.dot needs BLOCK_M of at least 16")


def launcher(a: Any, p: Packed, mode: int, cfg: Mapping[str, Any]) -> tuple[Callable[[], Any], Any]:
    """(call, out): call() runs one GEMV of a [M, K] fp16 with p and returns the fp16 [M, N] output. A split-K
    configuration accumulates into an fp32 buffer with atomics; zeroing it and the cast to fp16 are inside call."""
    import torch
    import triton

    m = a.shape[0]
    check_config(p, cfg, m)
    kern = kernel()
    split = cfg["SPLIT_K"]
    out16 = torch.empty(m, p.n, dtype=torch.float16, device=a.device)
    ws = torch.zeros(m, p.n, dtype=torch.float32, device=a.device) if split > 1 else None
    grid = (triton.cdiv(m, cfg["BLOCK_M"]), triton.cdiv(p.n, cfg["BLOCK_N"]), split)
    mb = p.main_bits
    hi = p.hi if p.hi is not None else p.main
    target = ws if ws is not None else out16

    def call() -> Any:
        if ws is not None:
            ws.zero_()
        kern[grid](a, p.main, hi, p.table, p.shared, p.scale, p.zero, target, m, p.n, p.k,
                   a.stride(0), target.stride(0),
                   BITS=p.bits, MB=mb, PW=32 // mb, NV=1 << p.bits, MODE=mode, GROUP=GROUP,
                   BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"], SPLIT_K=split,
                   USE_DOT=cfg["USE_DOT"], num_warps=cfg["num_warps"])
        if ws is not None:
            out16.copy_(ws)
        return out16

    return call, out16


def lut_gemv(a: Any, p: Packed, mode: int | str = 1, cfg: Mapping[str, Any] | None = None) -> Any:
    """One GEMV: fp16 A [M, K] times the packed W^T, fp16 [M, N] out."""
    mode = MODE_IDS[mode] if isinstance(mode, str) else mode
    if cfg is None:
        cfg = M1_CONFIGS[0] if a.shape[0] == 1 else DOT_CONFIGS[0]
    call, _ = launcher(a.contiguous(), p, mode, cfg)
    return call().clone()


# --- the pod benchmark --------------------------------------------------------------------------------------------

def _chipsheet() -> Any:
    here = Path(__file__).resolve().parent
    if (here / "chipsheet.py").exists():  # on the pod: uploaded side by side
        sys.path.insert(0, str(here))
        import chipsheet as cs  # type: ignore[import-not-found]
        return cs
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from lab.kernels.chipsheet import chipsheet as cs
    return cs


def _make_layer(bits: int, k: int, n: int, seed: int, device: str = "cuda") -> tuple[Any, Packed]:
    import torch

    gen = torch.Generator(device=device).manual_seed(seed)
    w = (torch.randn(n, k, device=device, generator=gen) * 0.02).half()
    return w, quantize(w, bits)


def _rotating(calls: Sequence[Callable[[], Any]]) -> Callable[[], Any]:
    state = {"i": 0}

    def call() -> Any:
        fn = calls[state["i"] % len(calls)]
        state["i"] += 1
        return fn()
    return call


def bench_main(args: argparse.Namespace, command: str) -> int:
    cs = _chipsheet()
    fmt = cs.fmt
    rocminfo_text, why = cs.read_rocminfo()
    refused = cs.gate(args.node, rocminfo_text, why)
    if refused:
        print(json.dumps({"ok": False, "outcome": "refused", "error": refused}))
        return cs.EXIT_GATE
    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        print(json.dumps({"ok": False, "outcome": "no_gpu"}))
        return 1
    agent = fmt.parse_rocminfo(rocminfo_text or "")
    args.budget_s = args.budget_s or (QUICK_BUDGET_S if args.quick else FULL_BUDGET_S)

    class LutRun(cs.Run):
        def cmd(self, section: str) -> str:
            return f"{self.command}  # section {section}"

    run = LutRun(args, command)
    run.repeats, run.warmup, run.min_load_s = (7, 2, 0.5) if args.quick else (11, 3, 1.0)
    run.idle_s = 1.0 if args.quick else 2.0
    started = cs.utc_now()
    versions = cs.collect_versions(None)
    others = cs.run_tool(["rocm-smi", "--showpids"], timeout_s=20)[1] if cs.tool("rocm-smi") else None
    run.power.start()
    seconds: dict[str, float] = {}

    def lut_point(section: str, bits: int, mode: int, name: str, k: int, n: int, m: int, cfg: Mapping[str, Any],
                  copies: int = 1, quick: bool = False) -> str:
        mid = (f"lut_gemv.b{bits}.{MODES[mode]}.{name}.m{m}.{cfg['tag']}" + (".cold" if copies > 1 else "")
               + (".sweep" if section == "sweep" else ""))

        def body() -> tuple[list[float], dict[str, Any]]:
            layers, calls = [], []
            err = None
            try:
                gen = torch.Generator(device="cuda").manual_seed(m * 7 + bits)
                a = torch.randn(m, k, device="cuda", generator=gen).half()
                for i in range(copies):
                    w, p = _make_layer(bits, k, n, seed=1000 * bits + i + k + n)
                    del w
                    call, _ = launcher(a, p, mode, cfg)
                    if i == 0:
                        out = call().float()
                        ref = reference(a, p, mode)
                        err = float((out - ref).abs().max())
                        rel = float((out - ref).norm() / ref.norm().clamp_min(1e-12))
                        del out, ref
                        if not math.isfinite(err) or rel > 1e-2:
                            raise ValueError(f"wrong answer: max abs error {err:.4g}, relative L2 {rel:.3g}")
                    layers.append(p)
                    calls.append(call)
                fn = _rotating(calls) if copies > 1 else calls[0]
                reps, warm, load = (5, 2, 0.3) if quick else (run.repeats, run.warmup, run.min_load_s)
                secs = cs.time_torch(fn, repeats=reps, warmup=warm, min_load_s=load)
            finally:
                del layers, calls
                cs._free_gpu()
            med = fmt.stats(secs)["median"]
            nbytes = _bytes_of(bits, mode, k, n)
            seconds[mid] = med
            return [nbytes / s / 1e9 for s in secs], {
                "seconds_median": med, "weight_bytes": nbytes, "weight_gbs": nbytes / med / 1e9,
                "tflops": 2 * m * n * k / med / 1e12, "max_abs_error": err, "rel_l2_error": rel,
                "bits_per_weight": 8 * nbytes / (n * k), "copies": copies}

        run.measure(mid, f"{bits}-bit {MODES[mode]} GEMV, batch {m}, K {k} to N {n}, {cfg['tag']}"
                    + (f", {copies} rotated copies" if copies > 1 else ""), "GB/s",
                    f"triton lut_gemv BITS {bits} MODE {mode} ({MODES[mode]}) {dict(cfg)}, value = weight bytes / time",
                    {"bits": bits, "mode": MODES[mode], "m": m, "k": k, "n": n, "group": GROUP, "copies": copies,
                     **{key: val for key, val in cfg.items()}}, body,
                    need_s=(1.0 if quick else run.min_load_s) + 3)
        return mid

    def fp16_point(name: str, k: int, n: int, m: int, copies: int = 1) -> None:
        mid = f"lut_gemv.ref.fp16.{name}.m{m}" + (".cold" if copies > 1 else "")

        def body() -> tuple[list[float], dict[str, Any]]:
            x = torch.randn(m, k, dtype=torch.float16, device="cuda")
            ws = [torch.randn(n, k, dtype=torch.float16, device="cuda") * 0.02 for _ in range(copies)]
            try:
                fn = _rotating([lambda w=w: F.linear(x, w) for w in ws]) if copies > 1 else (lambda: F.linear(x, ws[0]))
                secs = cs.time_torch(fn, repeats=run.repeats, warmup=run.warmup, min_load_s=run.min_load_s)
            finally:
                del x, ws
                cs._free_gpu()
            med = fmt.stats(secs)["median"]
            nbytes = n * k * 2
            return [nbytes / s / 1e9 for s in secs], {"seconds_median": med, "weight_bytes": nbytes,
                                                      "weight_gbs": nbytes / med / 1e9,
                                                      "tflops": 2 * m * n * k / med / 1e12, "copies": copies}

        run.measure(mid, f"fp16 torch GEMV, batch {m}, K {k} to N {n}" + (f", {copies} rotated copies" if copies > 1
                                                                         else ""), "GB/s",
                    f"torch.nn.functional.linear(x[{m},{k}], w[{n},{k}]) float16, value = weight bytes / time",
                    {"m": m, "k": k, "n": n, "copies": copies}, body, need_s=run.min_load_s + 3)

    def w4a16_point(name: str, k: int, n: int, m: int, copies: int = 1) -> None:
        mid = f"lut_gemv.ref.chipsheet_w4a16.{name}.m{m}.bn64_bk64" + (".cold" if copies > 1 else "")
        cfg = cs.DEQUANT_CONFIGS[0]
        kern = cs._w4a16_kernel()

        def body() -> tuple[list[float], dict[str, Any]]:
            import triton

            group = cs.DEQUANT_GROUP
            a = torch.randn(m, k, device="cuda").half()
            c = torch.empty(m, n, dtype=torch.float16, device="cuda")
            sets = []
            for i in range(copies):
                codes = torch.randint(0, 16, (k, n), dtype=torch.uint8, device="cuda")
                qw = (codes[0::2] | (codes[1::2] << 4)).contiguous()
                scales = (torch.rand(k // group, n, device="cuda") * 0.02 + 0.001).half()
                if i == 0:
                    sets_ref = (codes, scales)
                else:
                    del codes
                sets.append((qw, scales))
            grid = (triton.cdiv(m, cfg["BLOCK_M"]), triton.cdiv(n, cfg["BLOCK_N"]))

            def one(qw: Any, scales: Any) -> None:
                kern[grid](a, qw, scales, c, m, n, k, a.stride(0), a.stride(1), qw.stride(0), qw.stride(1),
                           scales.stride(0), scales.stride(1), c.stride(0), c.stride(1), GROUP=group, **cfg)

            try:
                one(*sets[0])
                torch.cuda.synchronize()
                codes, scales = sets_ref
                weights = (codes.float() - 8.0) * scales.float().repeat_interleave(group, 0)
                ref = a.float() @ weights
                err = float((c.float() - ref).abs().max())
                rel = float((c.float() - ref).norm() / ref.norm().clamp_min(1e-12))
                del weights, ref, codes, sets_ref
                fn = _rotating([lambda s=s: one(*s) for s in sets]) if copies > 1 else (lambda: one(*sets[0]))
                secs = cs.time_torch(fn, repeats=run.repeats, warmup=run.warmup, min_load_s=run.min_load_s)
            finally:
                del sets, a, c
                cs._free_gpu()
            med = fmt.stats(secs)["median"]
            nbytes = k * n // 2 + (k // group) * n * 2
            return [nbytes / s / 1e9 for s in secs], {"seconds_median": med, "weight_bytes": nbytes,
                                                      "weight_gbs": nbytes / med / 1e9,
                                                      "tflops": 2 * m * n * k / med / 1e12, "max_abs_error": err,
                                                      "rel_l2_error": rel, "copies": copies}

        run.measure(mid, f"the chip sheet's Triton W4A16 kernel, batch {m}, K {k} to N {n}, bn64_bk64"
                    + (f", {copies} rotated copies" if copies > 1 else ""), "GB/s",
                    f"chipsheet._w4a16_kernel BLOCK_M 16 BLOCK_N 64 BLOCK_K 64 group {cs.DEQUANT_GROUP}, "
                    "value = weight bytes / time", {"m": m, "k": k, "n": n, "copies": copies, **cfg}, body,
                    need_s=run.min_load_s + 4)

    def best(ids: Sequence[str]) -> str | None:
        timed = [(seconds[i], i) for i in ids if i in seconds]
        return min(timed)[1] if timed else None

    def cfg_of(mid: str | None, fallback: Mapping[str, Any]) -> Mapping[str, Any]:
        if mid is None:
            return fallback
        return next((c for tag, c in ALL_CONFIGS.items() if mid.endswith("." + tag) or ("." + tag + ".") in mid),
                    fallback)  # ids end in .<tag>, .<tag>.cold or .<tag>.sweep

    name, k, n = KILL_SHAPE
    choice: dict[str, Any] = {}
    try:
        # 1. Sweep at the kill shape, BITS 3, M 1: every mode in every M1 configuration and the chip sheet's tile.
        run.begin("sweep")
        sweep_ids: dict[int, list[str]] = {mode: [] for mode in MODES}
        for cfg in M1_CONFIGS + DOT_CONFIGS[:1]:
            for mode in MODES:
                sweep_ids[mode].append(lut_point("sweep", 3, mode, name, k, n, 1, cfg, quick=True))
        best_lut = best([i for mode in (1, 2, 3) for i in sweep_ids[mode]])
        lut_mode = next((mode for mode in (1, 2, 3) if best_lut in sweep_ids[mode]), 1)
        m1_cfg = cfg_of(best_lut, M1_CONFIGS[0])
        # 2. Sweep the dot configurations at M 16, BITS 3, for the batched rows.
        dot_ids = {mode: [lut_point("sweep", 3, mode, name, k, n, 16, cfg, quick=True) for cfg in DOT_CONFIGS]
                   for mode in (0, lut_mode)}
        dot_cfg = cfg_of(best(dot_ids[lut_mode]), DOT_CONFIGS[0])
        choice = {"lut_mode": MODES[lut_mode], "best_lut_m1": best_lut, "m1_config": m1_cfg["tag"],
                  "dot_config": dot_cfg["tag"]}
        run.notes.append(f"choice after the sweep: {json.dumps(choice)}")
        # 3. The table: BITS 2, 3, 4; the per-group table (mode 1), the best LUT mode when it differs, and arith.
        run.begin("table")
        modes = [0, 1] + ([lut_mode] if lut_mode != 1 else [])
        points = [(s, 1) for s in DECODE_SHAPES] + [(s, m) for m in (4, 16) for s in (DECODE_SHAPES[0], DECODE_SHAPES[3])]
        for bits in (3, 2, 4):
            for (sname, sk, sn), m in points:
                for mode in modes:
                    lut_point("table", bits, mode, sname, sk, sn, m, m1_cfg if m == 1 else dot_cfg)
        # 4. Same-session references: torch fp16 and the chip sheet's W4A16 kernel at the same points, the copy.
        run.begin("reference")
        for (sname, sk, sn), m in points:
            fp16_point(sname, sk, sn, m)
            w4a16_point(sname, sk, sn, m)
        nbytes = 1024 * fmt.MIB

        def copy_body() -> tuple[list[float], dict[str, Any]]:
            src = torch.empty(nbytes, dtype=torch.uint8, device="cuda").random_(0, 255)
            dst = torch.empty_like(src)
            try:
                secs = cs.time_torch(lambda: dst.copy_(src), repeats=run.repeats, warmup=run.warmup,
                                     min_load_s=run.min_load_s)
            finally:
                del src, dst
                cs._free_gpu()
            return [2 * nbytes / s / 1e9 for s in secs], {"copy_gbs": nbytes / fmt.stats(secs)["median"] / 1e9}

        run.measure("lut_gemv.ref.copy.d2d.size_1gib", "device-to-device copy of 1 GiB (the chip sheet's probe)",
                    "GB/s", "torch: dst.copy_(src), uint8[1073741824], value = 2 x bytes / time", {"bytes": nbytes},
                    copy_body, need_s=run.min_load_s + 3)
        # 5. Cold weights: M 1 with enough rotated copies that the working set passes the 32 MiB MALL several times.
        run.begin("cold")
        for sname, sk, sn in DECODE_SHAPES:
            for bits in (3, 2, 4):
                per = _bytes_of(bits, lut_mode, sk, sn)
                lut_point("cold", bits, lut_mode, sname, sk, sn, 1, m1_cfg, copies=max(2, math.ceil(256e6 / per)))
            fp16_point(sname, sk, sn, 1, copies=max(2, math.ceil(256e6 / (sn * sk * 2))))
            w4a16_point(sname, sk, sn, 1, copies=max(2, math.ceil(256e6 / (sn * sk // 2))))
    except Exception as exc:  # the orchestration itself broke: keep what was measured
        run.notes.append(f"aborted in section {run.section}: {type(exc).__name__}: {exc}"[:600])
    finally:
        run.power.stop()
    result = {
        "format": FORMAT, "gfx": (agent or {}).get("gfx") or cs.POD_GFX, "node": args.node,
        "mode": "quick" if args.quick else "full", "date": args.date or started[:10], "started": started,
        "ended": cs.utc_now(), "gpu_seconds": round(run.elapsed(), 1), "budget_s": run.budget_s,
        "command": command, "host": socket.gethostname(), "versions": versions, "device": {"rocminfo": agent},
        "power": {"reader": run.power.describe(), "idle": run.idle}, "choice": choice,
        "other_gpu_processes": others, "measurements": run.entries, "notes": run.notes,
    }
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"lut_gemv-{result['gfx']}-{result['date']}.json"
    path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    counts = {status: sum(1 for e in run.entries if e["status"] == status) for status in ("ok", "failed", "skipped")}
    print(json.dumps({"ok": True, "outcome": "measured", "json": str(path), "gpu_seconds": result["gpu_seconds"],
                      "counts": counts, "choice": choice}))
    for e in run.entries:
        d = e.get("derived") or {}
        print(json.dumps({"id": e["id"], "status": e["status"], "gbs": d.get("weight_gbs"),
                          "us": round(d["seconds_median"] * 1e6, 1) if d.get("seconds_median") else None,
                          "err": d.get("max_abs_error"), "w": (e.get("watts") or {}).get("loaded_mean_w"),
                          "error": (e.get("error") or "")[:160] or None}))
    return 0


def _bytes_of(bits: int, mode: int, k: int, n: int) -> int:
    groups = k // GROUP
    codes = k * n * bits // 8
    if mode in (1, 2):
        return codes + groups * n * (1 << bits) * 2
    return codes + groups * n * 4 + ((1 << bits) * 2 if mode == 3 else 0)


# --- metrics rows from a pulled result ---------------------------------------------------------------------------

def metric_rows(result: Mapping[str, Any], source: str, account: str = "D") -> list[dict[str, Any]]:
    """lut_gemv.* rows for lab/refine/metrics.jsonl: per measured entry its weight GB/s, TFLOP/s, watts, J per GEMV
    and max abs error, with the node, hub, account and sampler conditions. The tile sweep's entries (5 repeats at
    0.3 s of load, a choice of tile rather than a result) stay in the result file only."""
    rows = []
    reader = (result.get("power") or {}).get("reader") or {}
    base = {"hub": "sjc", "account": account, "gfx": result.get("gfx"), "host": result.get("host"),
            "date": result.get("date"), "torch": (result.get("versions") or {}).get("torch"),
            "triton": (result.get("versions") or {}).get("triton"), "rocm": (result.get("versions") or {}).get("rocm"),
            "sampler": {"tool": reader.get("tool"), "period_s": reader.get("period_s"),
                        "method": "the chip sheet's Power thread (lab/pod_power.py reader), socket watts read every "
                                  "0.25 s during each measurement's window; idle read before each section"},
            "harness": "lab/kernels/chipsheet/chipsheet.py time_torch (CUDA events, back-to-back calls ~2 ms per "
                       "repeat, then load held for min_load_s)", "file": source}
    ts = result.get("ended") or result.get("started")
    for e in result.get("measurements", []):
        if e.get("status") != "ok" or e.get("section") == "sweep":  # the sweep (5 repeats, 0.3 s) stays in the file
            continue
        d = e.get("derived") or {}
        p = e.get("params") or {}
        watts = e.get("watts") or {}
        sec = d.get("seconds_median")
        stem = e["id"]
        cond = {**base, **{k: p[k] for k in ("bits", "mode", "m", "k", "n", "copies", "BLOCK_M", "BLOCK_N", "BLOCK_K",
                                              "SPLIT_K", "num_warps") if k in p},
                "section": e.get("section"), "repeats": e.get("repeats"),
                "rel_spread": (e.get("spread") or {}).get("rel_spread"),
                "idle_w": watts.get("idle_w"), "watt_samples": watts.get("samples")}
        values = []
        if "weight_gbs" in d:
            values.append(("weight_gbs", d["weight_gbs"], "GB/s"))
        if "copy_gbs" in d:
            values.append(("copy_gbs", d["copy_gbs"], "GB/s"))
        if "tflops" in d:
            values.append(("tflops", d["tflops"], "TFLOP/s"))
        if watts.get("loaded_mean_w") is not None:
            values.append(("watts", watts["loaded_mean_w"], "W"))
            if sec:
                values.append(("joules_per_gemv", watts["loaded_mean_w"] * sec, "J"))
        if d.get("max_abs_error") is not None:
            values.append(("max_abs_error", d["max_abs_error"], "abs"))
        for key, value, unit in values:
            rows.append({"metric": f"{stem}.{key}", "value": round(float(value), 6), "unit": unit,
                         "node": result.get("node"), "conditions": cond, "source": f"R62 lut_gemv bench, {source}",
                         "ts": ts, "low_confidence": False})
    return rows


def ingest_main(args: argparse.Namespace) -> int:
    path = Path(args.result)
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("format") != FORMAT:
        print(json.dumps({"ok": False, "outcome": "refused", "error": f"format is {result.get('format')!r}"}))
        return 1
    source = path.as_posix()
    rows = metric_rows(result, source, args.account)
    from lab.machine import metrics as ledger_metrics

    for row in rows:
        ledger_metrics.check_row(row)
    if not args.dry_run:
        for row in rows:
            ledger_metrics.add(row, path=args.metrics)
    print(json.dumps({"ok": True, "outcome": "dry_run" if args.dry_run else "written", "rows": len(rows),
                      "metrics": args.metrics}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lut_gemv.py", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    bench = sub.add_parser("bench", help="measure on the pod (node gate first)")
    bench.add_argument("--node", required=True, help="pod-a or pod-b")
    bench.add_argument("--quick", action="store_true")
    bench.add_argument("--out", default=DEFAULT_OUT)
    bench.add_argument("--budget-s", type=float)
    bench.add_argument("--date")
    ingest = sub.add_parser("ingest", help="append the lut_gemv.* rows of a pulled result to metrics.jsonl")
    ingest.add_argument("--result", required=True)
    ingest.add_argument("--account", default="D")
    ingest.add_argument("--metrics", default=str(Path(__file__).resolve().parents[1] / "refine" / "metrics.jsonl"))
    ingest.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    if args.command == "ingest":
        return ingest_main(args)
    return bench_main(args, shlex.join(["python3", "lut_gemv.py", *argv]))


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())
