"""Kernel class lora: the LoRA A projection of one adapter, x [tokens, K] @ A [K, 16], the skinny GEMM the chip sheet
measured at 1.58 TFLOP/s.

Lever: split_k. `torch` is torch.nn.functional.linear (the reference row, what PEFT runs); `1` and `4` are a Triton
kernel with BLOCK_M 64, BLOCK_K 64 and num_warps 4 whose grid is tokens / 64 workgroups times split_k: at 1,024 tokens
split_k 1 is 16 workgroups, under the chip's 20 WGPs, and split_k 4 is 64, with the partial sums added into a float32
output with atomics. Shapes: 1,024 tokens (4 x 256), K 2,560 (hidden) and 9,216 (the MLP's down projection input),
rank 16, bf16. derived.floor_s is the time to read x once at 209.5 GB/s.
"""
from __future__ import annotations

from typing import Any

try:
    from lab.kernels.wavefront import wavefront_format as wf
except ImportError:  # on the pod
    import wavefront_format as wf  # type: ignore[no-redef]

CLASS = "lora"
LEVER = "split_k"
WHAT = "LoRA A projection x @ A, rank 16"
TOKENS, RANK, BLOCK_M, BLOCK_K, NUM_WARPS = 1024, 16, 64, 64, 4
_KERNEL: Any = None


def cases(quick: bool) -> list[dict[str, Any]]:
    widths = (2560,) if quick else (2560, 9216)
    return [{"id": wf.row_id(CLASS, LEVER, setting, f"m{TOKENS}_k{k}"), "setting": setting,
             "params": {"tokens": TOKENS, "k": k, "rank": RANK, "dtype": "bfloat16", "block_m": BLOCK_M,
                        "block_k": BLOCK_K, "num_warps": NUM_WARPS}}
            for k in widths for setting in ("torch", 1, 4)]


def _kernel() -> Any:
    global _KERNEL
    if _KERNEL is None:
        import triton
        import triton.language as tl

        @triton.jit
        def lora_a(x_ptr, a_ptr, out_ptr, M, K, stride_xm, stride_ak, stride_om, SPLIT_K: tl.constexpr,
                   STEPS: tl.constexpr, R: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
            pid_m = tl.program_id(0)
            pid_k = tl.program_id(1)
            rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            cols = tl.arange(0, R)
            acc = tl.zeros((BLOCK_M, R), dtype=tl.float32)
            for step in range(STEPS):  # STEPS blocks of BLOCK_K per split, the last ones masked past K
                ks = (pid_k * STEPS + step) * BLOCK_K + tl.arange(0, BLOCK_K)
                x = tl.load(x_ptr + rows[:, None] * stride_xm + ks[None, :],
                            mask=(rows[:, None] < M) & (ks[None, :] < K), other=0.0)
                a = tl.load(a_ptr + ks[:, None] * stride_ak + cols[None, :], mask=ks[:, None] < K, other=0.0)
                acc += tl.dot(x, a)
            ptrs = out_ptr + rows[:, None] * stride_om + cols[None, :]
            if SPLIT_K == 1:
                tl.store(ptrs, acc, mask=rows[:, None] < M)
            else:
                tl.atomic_add(ptrs, acc, mask=rows[:, None] < M)

        _KERNEL = lora_a
    return _KERNEL


def measure(case: dict[str, Any], timer: Any) -> dict[str, Any]:
    import torch

    p = case["params"]
    m, k, r = p["tokens"], p["k"], p["rank"]
    gen = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=gen)
    a = torch.randn(k, r, device="cuda", dtype=torch.bfloat16, generator=gen) / k ** 0.5
    floor = m * k * 2 / (wf.COPY_GBS * 1e9)
    reference = torch.nn.functional.linear(x, a.t()).float()
    if case["setting"] == "torch":
        weight = a.t().contiguous()

        def step() -> None:
            torch.nn.functional.linear(x, weight)

        first = timer.first(step)
        samples = timer.repeat(step)
        record = None
    else:
        split = int(case["setting"])
        kernel = _kernel()
        out = torch.zeros(m, r, device="cuda", dtype=torch.float32)
        grid = (-(-m // p["block_m"]), split)
        steps = -(-(-(-k // p["block_k"])) // split)  # BLOCK_K blocks per split
        handle: dict[str, Any] = {}

        def step() -> None:
            if split > 1:
                out.zero_()
            handle["k"] = kernel[grid](x, a, out, m, k, x.stride(0), a.stride(0), out.stride(0), SPLIT_K=split,
                                       STEPS=steps, R=r, BLOCK_M=p["block_m"], BLOCK_K=p["block_k"], num_warps=p["num_warps"])

        first = timer.first(step)
        error = (out - reference).abs().max().item()
        if error > 0.05 * max(1.0, reference.abs().max().item()):
            raise RuntimeError(f"Triton A projection differs from torch by {error:.3g}")
        samples = timer.repeat(step)
        compiled = handle.get("k")
        record = wf.compile_record({"n_regs": getattr(compiled, "n_regs", None),
                                    "n_spills": getattr(compiled, "n_spills", None),
                                    "shared": getattr(getattr(compiled, "metadata", None), "shared", None),
                                    "num_warps": p["num_warps"], "num_stages": 2})
    median = sorted(samples)[len(samples) // 2]
    result = {"samples": samples, "first_call": first,
              "derived": {"us": median * 1e6, "tflops": 2 * m * k * r / median / 1e12, "floor_s": floor,
                          "ratio_to_floor": median / floor}}
    if record is not None:
        result["compile"] = record
        result["workgroups"] = (-(-m // p["block_m"])) * int(case["setting"])
    return result
