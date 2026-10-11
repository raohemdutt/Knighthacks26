"""Kernel class gdn: one Gated DeltaNet layer's chunk rule at the student's training shapes, forward plus backward.

Lever: impl. `torch` is transformers' torch_chunk_gated_delta_rule (what the trainer runs when flash-linear-attention
is absent: float32 casts, two batched triangular solves, a Python loop over 64-token chunks); `fla` is
flash-linear-attention's chunk_gated_delta_rule (fused Triton kernels). Shapes: batch 4 (the trainer's micro-batch),
256 and 1,024 tokens, 32 value heads with the 16 key heads repeated to 32 (Qwen3.5-4B's linear-attention layout),
head dimension 128, chunk 64. A row is seconds per forward plus backward of one layer; derived.per_micro_batch_s
multiplies by the 24 Gated DeltaNet layers the student has.
"""
from __future__ import annotations

from typing import Any

try:
    from lab.kernels.wavefront import wavefront_format as wf
except ImportError:  # on the pod
    import wavefront_format as wf  # type: ignore[no-redef]

CLASS = "gdn"
LEVER = "impl"
WHAT = "Gated DeltaNet chunk rule, one layer, forward plus backward"
LAYERS = 24
HEADS, HEAD_DIM, BATCH = 32, 128, 4


def cases(quick: bool) -> list[dict[str, Any]]:
    lengths = (256,) if quick else (256, 1024)
    return [{"id": wf.row_id(CLASS, LEVER, impl, f"b{BATCH}_t{t}"), "setting": impl,
             "params": {"batch": BATCH, "tokens": t, "heads": HEADS, "head_dim": HEAD_DIM, "chunk": 64,
                        "dtype": "bfloat16", "use_qk_l2norm_in_kernel": True}}
            for t in lengths for impl in ("torch", "fla")]


def _function(impl: str) -> Any:
    if impl == "torch":
        try:
            from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
        except ImportError as exc:
            raise wf.Skipped(f"transformers has no qwen3_5 torch_chunk_gated_delta_rule: {exc}") from exc
        return torch_chunk_gated_delta_rule
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    except ImportError as exc:
        raise wf.Skipped(f"flash-linear-attention is not importable: {exc}") from exc
    return chunk_gated_delta_rule


def measure(case: dict[str, Any], timer: Any) -> dict[str, Any]:
    import torch

    fn = _function(case["setting"])
    p = case["params"]
    shape = (p["batch"], p["tokens"], p["heads"], p["head_dim"])
    gen = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=gen).requires_grad_() for _ in range(3))
    g = (-torch.rand(shape[:3], device="cuda", generator=gen) * 0.5).requires_grad_()  # log decay, negative
    beta = torch.rand(shape[:3], device="cuda", generator=gen).to(torch.bfloat16).requires_grad_()
    grad = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=gen)

    def step() -> None:
        out = fn(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)[0]
        out.backward(grad)

    first = timer.first(step)
    samples = timer.repeat(step)
    median = sorted(samples)[len(samples) // 2]
    return {"samples": samples, "first_call": first,
            "derived": {"per_layer_ms": median * 1e3, "per_micro_batch_s": median * LAYERS}}
