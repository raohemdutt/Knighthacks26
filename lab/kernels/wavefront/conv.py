"""Kernel class conv: the Gated DeltaNet layer's depthwise causal conv at the student's training shapes, forward plus
backward (input and weight gradients).

Lever: impl. `torch` is F.conv1d with groups = channels and padding kernel - 1, sliced to the causal part, which is
what the layer's nn.Conv1d runs when causal-conv1d is absent (MIOpen picks the kernel on ROCm); `fla` is
flash-linear-attention's Triton causal_conv1d. Shapes: batch 4, 256 and 1,024 tokens, 8,192 channels (2 x 2048 key +
4096 value), kernel 4, bf16. derived.floor_s is the byte floor of the three passes (forward, input gradient, weight
gradient, each reading and writing a [batch, tokens, channels] bf16 tensor) at the chip sheet's 209.5 GB/s, and
derived.ratio_to_floor how far the median sits above it.
"""
from __future__ import annotations

from typing import Any

try:
    from lab.kernels.wavefront import wavefront_format as wf
except ImportError:  # on the pod
    import wavefront_format as wf  # type: ignore[no-redef]

CLASS = "conv"
LEVER = "impl"
WHAT = "depthwise causal conv, one layer, forward plus backward"
CHANNELS, KERNEL, BATCH, LAYERS = 8192, 4, 4, 24


def floor_s(batch: int, tokens: int, channels: int = CHANNELS) -> float:
    """Seconds to move three passes of a bf16 [batch, tokens, channels] tensor in and out at COPY_GBS."""
    return 3 * 2 * batch * tokens * channels * 2 / (wf.COPY_GBS * 1e9)


def cases(quick: bool) -> list[dict[str, Any]]:
    lengths = (256,) if quick else (256, 1024)
    return [{"id": wf.row_id(CLASS, LEVER, impl, f"b{BATCH}_t{t}"), "setting": impl,
             "params": {"batch": BATCH, "tokens": t, "channels": CHANNELS, "kernel": KERNEL, "dtype": "bfloat16"}}
            for t in lengths for impl in ("torch", "fla")]


def measure(case: dict[str, Any], timer: Any) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    p = case["params"]
    b, t, c, w = p["batch"], p["tokens"], p["channels"], p["kernel"]
    gen = torch.Generator(device="cuda").manual_seed(0)
    if case["setting"] == "torch":
        x = torch.randn(b, c, t, device="cuda", dtype=torch.bfloat16, generator=gen).requires_grad_()
        weight = torch.randn(c, 1, w, device="cuda", dtype=torch.bfloat16, generator=gen).requires_grad_()
        grad = torch.randn(b, c, t, device="cuda", dtype=torch.bfloat16, generator=gen)

        def step() -> None:
            F.conv1d(x, weight, padding=w - 1, groups=c)[..., :t].backward(grad)
    else:
        try:
            from fla.modules.conv import causal_conv1d
        except ImportError as exc:
            raise wf.Skipped(f"flash-linear-attention is not importable: {exc}") from exc
        x = torch.randn(b, t, c, device="cuda", dtype=torch.bfloat16, generator=gen).requires_grad_()
        weight = torch.randn(c, w, device="cuda", dtype=torch.bfloat16, generator=gen).requires_grad_()
        grad = torch.randn(b, t, c, device="cuda", dtype=torch.bfloat16, generator=gen)

        def step() -> None:
            out = causal_conv1d(x, weight)
            (out[0] if isinstance(out, tuple) else out).backward(grad)

    first = timer.first(step)
    samples = timer.repeat(step)
    median = sorted(samples)[len(samples) // 2]
    floor = floor_s(b, t, c)
    return {"samples": samples, "first_call": first,
            "derived": {"per_layer_ms": median * 1e3, "per_micro_batch_s": median * LAYERS, "floor_s": floor,
                        "ratio_to_floor": median / floor}}
