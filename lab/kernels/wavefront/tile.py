"""Kernel class tile: a Triton kernel shaped like flash-linear-attention's forward state scan, to read one wavefront
lever at a time from the compiler and the clock together.

Each workgroup owns one (sequence, value head, 32-wide V tile): it carries a float32 state of K 128 x BV 32 across
the chunks of its sequence, and per 64-token chunk reads a query and key tile (64 x 128 bf16) and a value tile
(64 x 32 bf16), writes o = q @ state and updates state = 0.99 state + k^T @ v. Grid: batch 4 x 32 value heads x 4 V
tiles = 512 workgroups. Lever: num_stages, 1 against 2, at num_warps 4 (the stream pipeliner's buffers sit in LDS, so
stages trade resident workgroups for prefetch on a 64 KiB per workgroup, 128 KiB per WGP part). Each row carries the
compiled n_regs, n_spills and shared bytes and the occupancy they imply.
"""
from __future__ import annotations

from typing import Any

try:
    from lab.kernels.wavefront import wavefront_format as wf
except ImportError:  # on the pod
    import wavefront_format as wf  # type: ignore[no-redef]

CLASS = "tile"
LEVER = "num_stages"
WHAT = "chunk scan tile: o = q @ S, S = 0.99 S + k^T v per 64-token chunk"
BATCH, HEADS, K, V, BV, CHUNK, NUM_WARPS = 4, 32, 128, 128, 32, 64, 4
_KERNEL: Any = None


def cases(quick: bool) -> list[dict[str, Any]]:
    lengths = (256,) if quick else (256, 1024)
    return [{"id": wf.row_id(CLASS, LEVER, stages, f"t{t}_w{NUM_WARPS}"), "setting": stages,
             "params": {"batch": BATCH, "heads": HEADS, "tokens": t, "k": K, "v": V, "bv": BV, "chunk": CHUNK,
                        "num_warps": NUM_WARPS, "dtype": "bfloat16"}}
            for t in lengths for stages in (1, 2)]


def _kernel() -> Any:
    global _KERNEL
    if _KERNEL is None:
        import triton
        import triton.language as tl

        @triton.jit
        def scan(q_ptr, k_ptr, v_ptr, o_ptr, T, NC, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                 BV: tl.constexpr, BT: tl.constexpr):
            seq_head = tl.program_id(0)
            vt = tl.program_id(1)
            rows = tl.arange(0, BT)
            ks = tl.arange(0, K)
            vs = vt * BV + tl.arange(0, BV)
            base_qk = seq_head * T * K
            base_v = seq_head * T * V
            state = tl.zeros((K, BV), dtype=tl.float32)
            for c in range(NC):
                t0 = c * BT
                q = tl.load(q_ptr + base_qk + (t0 + rows)[:, None] * K + ks[None, :])
                k = tl.load(k_ptr + base_qk + (t0 + rows)[:, None] * K + ks[None, :])
                v = tl.load(v_ptr + base_v + (t0 + rows)[:, None] * V + vs[None, :])
                o = tl.dot(q, state.to(tl.bfloat16))
                tl.store(o_ptr + base_v + (t0 + rows)[:, None] * V + vs[None, :], o.to(tl.bfloat16))
                state = state * 0.99 + tl.dot(tl.trans(k), v)

        _KERNEL = scan
    return _KERNEL


def measure(case: dict[str, Any], timer: Any) -> dict[str, Any]:
    import torch

    p = case["params"]
    stages = int(case["setting"])
    seqs = p["batch"] * p["heads"]
    gen = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(seqs, p["tokens"], p["k"], device="cuda", dtype=torch.bfloat16, generator=gen) / 16
    k = torch.randn(seqs, p["tokens"], p["k"], device="cuda", dtype=torch.bfloat16, generator=gen) / 16
    v = torch.randn(seqs, p["tokens"], p["v"], device="cuda", dtype=torch.bfloat16, generator=gen)
    o = torch.empty_like(v)
    grid = (seqs, p["v"] // p["bv"])
    kernel = _kernel()
    handle: dict[str, Any] = {}

    def step() -> None:
        handle["k"] = kernel[grid](q, k, v, o, p["tokens"], p["tokens"] // p["chunk"], HV=p["heads"], K=p["k"],
                                   V=p["v"], BV=p["bv"], BT=p["chunk"], num_warps=p["num_warps"], num_stages=stages)

    first = timer.first(step)
    samples = timer.repeat(step)
    compiled = handle.get("k")
    record = wf.compile_record({"n_regs": getattr(compiled, "n_regs", None),
                                "n_spills": getattr(compiled, "n_spills", None),
                                "shared": getattr(getattr(compiled, "metadata", None), "shared", None),
                                "num_warps": p["num_warps"], "num_stages": stages})
    median = sorted(samples)[len(samples) // 2]
    moved = (2 * seqs * p["tokens"] * p["k"] * grid[1] + 2 * seqs * p["tokens"] * p["v"]) * 2
    return {"samples": samples, "first_call": first, "compile": record, "workgroups": grid[0] * grid[1],
            "derived": {"us": median * 1e6, "gbs": moved / median / 1e9}}
