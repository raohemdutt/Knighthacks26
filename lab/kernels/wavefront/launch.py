"""Kernel class launch: what one tiny kernel costs per launch from eager PyTorch, against the same chain replayed as
one HIP graph, so a profile's kernel count can be priced as host time.

Lever: mode. `eager` launches a chain of CHAIN in-place adds on a 1,024-element tensor one by one; `graph` captures the
same chain once with torch.cuda.CUDAGraph (HIP graphs on ROCm) and replays it. A row is seconds per op (the chain's
time over its length); first_call is the capture for `graph` and the first eager chain otherwise.
"""
from __future__ import annotations

from typing import Any

try:
    from lab.kernels.wavefront import wavefront_format as wf
except ImportError:  # on the pod
    import wavefront_format as wf  # type: ignore[no-redef]

CLASS = "launch"
LEVER = "mode"
WHAT = "per-op cost of a chain of tiny in-place adds"
ELEMENTS = 1024


def cases(quick: bool) -> list[dict[str, Any]]:
    chain = 2000
    return [{"id": wf.row_id(CLASS, LEVER, mode, f"chain{chain}"), "setting": mode,
             "params": {"chain": chain, "elements": ELEMENTS, "op": "x.add_(1.0)", "dtype": "float32"}}
            for mode in ("eager", "graph")]


def measure(case: dict[str, Any], timer: Any) -> dict[str, Any]:
    import torch

    chain = case["params"]["chain"]
    x = torch.zeros(case["params"]["elements"], device="cuda")

    def eager() -> None:
        for _ in range(chain):
            x.add_(1.0)

    if case["setting"] == "eager":
        first = timer.first(eager)
        samples = timer.repeat(eager, inner=1)
    else:
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())

        def capture() -> None:
            with torch.cuda.stream(stream):
                eager()  # warm the allocator outside the capture
                torch.cuda.current_stream().synchronize()
                with torch.cuda.graph(graph, stream=stream):
                    eager()
            torch.cuda.current_stream().wait_stream(stream)

        first = timer.first(capture)
        samples = timer.repeat(graph.replay, inner=1)
    per_op = [s / chain for s in samples]
    return {"samples": per_op, "first_call": first, "derived": {"chain": chain, "per_op_us": sorted(per_op)[len(per_op) // 2] * 1e6}}
