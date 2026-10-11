"""Head-to-head on gfx1151 (R63): the R62 lookup-table GEMV against vLLM's RDNAHybridW4A16LinearKernel.

    python3 dispatch_bench.py bench --node pod-b [--out ~/r63_dispatch] [--budget-s 600] [--model DIR] [--quick]
    python -m lab.kernels.dispatch_bench ingest --result FILE [--metrics lab/refine/metrics.jsonl] [--dry-run]
    python -m lab.kernels.dispatch_bench decide --result FILE

The question: at batch 1 on the Strix Halo, does the lookup-table GEMV of lab/kernels/lut_gemv.py beat the kernel
vLLM (main 5281e4990, the R60 build) selects for the teacher's compressed-tensors W4A16 layers, on the same packed
4-bit weights? If it does, a batch-size switch inside that kernel class is worth building (R63 step 3).

Both kernels get the same weights: one decoder layer's real codes and group scales from the teacher checkpoint
(RedHatAI/Qwen3.8-27B-INT4: symmetric int4, group 128, uint4b8, no zero points), read with safetensors and, for a
shape no tensor has, the first rows of a larger one, else random codes. The vLLM side is the kernel class itself:
an nn.Module holding `weight_packed` int32 [N, K // 8] (vLLM's pack_quantized_values_into_int32, packed along K, the
checkpoint's own layout) and `weight_scale` [N, K // 128], MPLinearLayerConfig(uint4b8, group 128, no zero points),
RDNAHybridW4A16LinearKernel.process_weights_after_loading, then apply_weights per call (HIP wvSplitK_int4_g when
M <= 5 and K * M <= 39321, its Triton fused-dequant GEMM above). The lookup side is lut_gemv.pack of the same codes
with zero 8 and the same scales, run in mode group_lut (a per-group 16-entry fp16 table) and mode arith (the plain
(c - 8) * s formula, the same kernel without the table), with the tile chosen hot at the first shape.

Timing: each point is captured into a CUDA graph (HIP graph) of enough calls to last about 2 ms, the way vLLM's
decode replays its layers, and the graph is replayed repeats times (chip sheet harness: warmup, repeats, then load
held for min_load_s so the watts are read). A point that cannot be captured falls back to the chip sheet's eager
time_torch and says so. Cold rotates enough copies of the layer that the working set is past 256 MB (eight times the
32 MiB MALL), hot reuses one. The value is the common weight GB/s: the bytes the checkpoint holds for the layer (int4
codes and one fp16 scale per 128 weights, K * N / 2 + K * N / 64) over the time per call, so a higher value means a
shorter call whatever each kernel reads; each kernel's own read bytes are kept beside it.

Error: each kernel's fp16 output against the fp32 product of the activations with the dequantized weights ((c - 8)
* s in fp32), and the two kernels against each other. At fp16 rounding means a relative L2 error at most 1e-3 and a
max abs error at most 2^-8 of the largest reference output (four fp16 ulps at the top of the range).

The rule (written before the run, in lab/refine/research/kernels/dispatch_by_batch_gfx1151.md): the lookup path wins
if, with cold weights at M = 1, its better mode takes at most 1 / 1.10 of the hybrid kernel's time summed over the
four shapes (reads the common weight bytes at least 10 percent faster) and every one of its M = 1 points is at fp16
rounding. The switch threshold is then the largest M in 1, 4, 16 at which it is still faster cold.

bench runs on the pod inside the new build's venv (~/venv-vllm-new, uploaded next to lut_gemv.py, chipsheet.py,
chipsheet_format.py and pod_power.py, by lab/kernels/pod_dispatch_bench.sh) and writes
<out>/dispatch-<gfx>-<date>.json; ingest turns that file into dispatch.* rows of lab/refine/metrics.jsonl. On a
machine with no GPU nothing here touches a device: the packing, the reference and the decision run on the CPU.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import socket
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

FORMAT = "dispatch_bench/1"
GROUP = 128
DEFAULT_OUT = "~/r63_dispatch"
DEFAULT_MODEL = "~/models/Qwen3.8-27B-INT4"
FULL_BUDGET_S = 600.0
QUICK_BUDGET_S = 150.0
BATCHES = (1, 4, 16)
COLD_BYTES = 256e6
GRAPH_TARGET_S = 0.002
WIN_RATIO = 1.10
REL_L2_MAX = 1e-3
MAX_ABS_FRACTION = 2.0 ** -8
LUT_MODES = ("group_lut", "arith")
# The chip sheet's 27B decode shapes as (name, K, N) and the teacher tensor each one is read from (first N rows).
SHAPES = (("k5120_n6144", 5120, 6144, "self_attn.q_proj"), ("k6144_n5120", 6144, 5120, "self_attn.o_proj"),
          ("k5120_n17408", 5120, 17408, "mlp.gate_proj"), ("k17408_n5120", 17408, 5120, "mlp.down_proj"))
LAYER = 3
# vLLM's routing constants at 5281e4990 (rdna_hybrid_w4a16.py): MAX_SKINNY_BATCH_SIZE, int(32768 * 1.2).
SKINNY_MAX_M = 5
SKINNY_LIMIT = int(64 * 1024 // 2 * 1.2)


# --- what each kernel reads, and the routing the hybrid kernel does ------------------------------------------------

def common_bytes(k: int, n: int, group: int = GROUP) -> int:
    """The layer as the checkpoint holds it: int4 codes and one 2-byte scale per group."""
    return k * n // 2 + (k // group) * n * 2


def own_bytes(kernel: str, k: int, n: int, group: int = GROUP) -> int:
    """Bytes the kernel reads for the weights: hybrid reads the common bytes; group_lut a 16-entry fp16 table per
    group and column instead of the scale; arith the scale and an fp16 zero point."""
    codes = k * n // 2
    groups = (k // group) * n
    if kernel.startswith("lut.group_lut"):
        return codes + groups * 16 * 2
    if kernel.startswith("lut.arith"):
        return codes + groups * 4
    return common_bytes(k, n, group)


def hybrid_route(m: int, k: int) -> str:
    """Which path RDNAHybridW4A16LinearKernel takes for M rows of K: the HIP skinny GEMM or its Triton GEMM."""
    return "hip_wvsplitk_int4" if m <= SKINNY_MAX_M and k * m <= SKINNY_LIMIT else "triton_skinny_fmt"


def cold_copies(k: int, n: int) -> int:
    return max(2, math.ceil(COLD_BYTES / common_bytes(k, n)))


def at_fp16_rounding(max_abs: float | None, rel_l2: float | None, ref_max: float | None) -> bool:
    if max_abs is None or rel_l2 is None or ref_max is None:
        return False
    if not (math.isfinite(max_abs) and math.isfinite(rel_l2)):
        return False
    return rel_l2 <= REL_L2_MAX and max_abs <= MAX_ABS_FRACTION * max(ref_max, 1e-12)


# --- packing (CPU or GPU) ------------------------------------------------------------------------------------------

def unpack_ct(packed: Any) -> Any:
    """compressed-tensors pack-quantized int32 [N, K // 8] to uint8 codes [N, K]: code K = 8 j + i is bits 4 i to
    4 i + 3 of word j (vLLM's unpack_quantized_values_into_int32 with packed_dim 1)."""
    import torch

    words = packed.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(8, dtype=torch.int64, device=packed.device) * 4
    codes = (words[:, :, None] >> shifts[None, None, :]) & 0xF
    return codes.reshape(packed.shape[0], packed.shape[1] * 8).to(torch.uint8)


def pack_ct(codes: Any) -> Any:
    """The inverse of unpack_ct: uint8 codes [N, K] to int32 [N, K // 8] (the checkpoint's layout)."""
    import torch

    n, k = codes.shape
    c = codes.to(torch.int64).reshape(n, k // 8, 8)
    shifts = torch.arange(8, dtype=torch.int64, device=codes.device) * 4
    words = (c << shifts[None, None, :]).sum(dim=2)
    return torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32).contiguous()


def dequant(codes: Any, scale: Any, group: int = GROUP) -> Any:
    """fp32 W [N, K] = (c - 8) * s, s [N, K // group] (uint4b8, the symmetric int4 of the checkpoint)."""
    import torch

    return (codes.float() - 8.0) * scale.float().repeat_interleave(group, dim=1).to(torch.float32)


def to_lut(codes: Any, scale: Any) -> Any:
    """The same codes and scales in lut_gemv's packing: zero 8 everywhere, scale [K // 128, N] fp16."""
    import torch

    lut = _lut()
    s = scale.t().contiguous().half()
    z = torch.full_like(s, 8.0)
    return lut.pack(codes, s, z, 4)


def _lut() -> Any:
    here = Path(__file__).resolve().parent
    if (here / "lut_gemv.py").exists() and __package__ in (None, ""):
        sys.path.insert(0, str(here))
        import lut_gemv  # type: ignore[import-not-found]
        return lut_gemv
    from lab.kernels import lut_gemv
    return lut_gemv


def _chipsheet() -> Any:
    here = Path(__file__).resolve().parent
    if (here / "chipsheet.py").exists():
        sys.path.insert(0, str(here))
        import chipsheet as cs  # type: ignore[import-not-found]
        return cs
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from lab.kernels.chipsheet import chipsheet as cs
    return cs


# --- the decision, from a result file ------------------------------------------------------------------------------

def _points(result: Mapping[str, Any]) -> dict[tuple[str, int, str, str], dict[str, Any]]:
    """(shape, m, temp, kernel) -> derived numbers of each ok fp16 entry."""
    out = {}
    for e in result.get("measurements", []):
        p = e.get("params") or {}
        if e.get("status") != "ok" or p.get("dtype", "float16") != "float16" or e.get("section") == "sweep":
            continue
        out[(p["shape"], int(p["m"]), p["temp"], p["kernel"])] = {**(e.get("derived") or {}), "id": e["id"]}
    return out


def decide(result: Mapping[str, Any]) -> dict[str, Any]:
    """The rule of the module docstring applied to a result: per shape and M the hybrid and the best lookup mode,
    cold and hot, the M = 1 cold sums, the verdict and the threshold."""
    pts = _points(result)
    shapes = [s[0] for s in SHAPES]
    table = []
    for temp in ("cold", "hot"):
        for m in BATCHES:
            for shape in shapes:
                hyb = pts.get((shape, m, temp, "hybrid"))
                luts = {kern: d for (s, mm, t, kern), d in pts.items()
                        if s == shape and mm == m and t == temp and kern.startswith("lut.")}
                best = min(luts.items(), key=lambda kv: kv[1]["seconds_median"]) if luts else None
                row = {"shape": shape, "m": m, "temp": temp,
                       "hybrid_us": round(hyb["seconds_median"] * 1e6, 2) if hyb else None,
                       "hybrid_gbs": round(hyb["common_gbs"], 1) if hyb else None,
                       "hybrid_route": hyb.get("route") if hyb else None,
                       "lut_kernel": best[0] if best else None,
                       "lut_us": round(best[1]["seconds_median"] * 1e6, 2) if best else None,
                       "lut_gbs": round(best[1]["common_gbs"], 1) if best else None,
                       "lut_fp16_ok": best[1].get("fp16_ok") if best else None,
                       "hybrid_fp16_ok": hyb.get("fp16_ok") if hyb else None}
                row["speedup"] = (round(row["hybrid_us"] / row["lut_us"], 3)
                                  if row["hybrid_us"] and row["lut_us"] else None)
                table.append(row)
    cold1 = [r for r in table if r["temp"] == "cold" and r["m"] == 1]
    complete = len(cold1) == len(shapes) and all(r["hybrid_us"] and r["lut_us"] for r in cold1)
    verdict: dict[str, Any] = {"rule": f"lookup wins if its cold M=1 time summed over the four shapes is at most "
                                       f"1/{WIN_RATIO:.2f} of the hybrid kernel's and every M=1 lookup point is at "
                                       f"fp16 rounding", "complete": complete}
    if complete:
        # the better lookup mode across all four shapes (one mode for the whole layer, as a patch would ship)
        per_mode = {}
        for mode in LUT_MODES:
            total = 0.0
            for shape in shapes:
                cands = [d for (s, mm, t, kern), d in pts.items()
                         if s == shape and mm == 1 and t == "cold" and kern.startswith(f"lut.{mode}.")]
                if not cands:
                    total = math.inf
                    break
                total += min(d["seconds_median"] for d in cands)
            per_mode[mode] = total
        mode = min(per_mode, key=per_mode.get)
        hyb_sum = sum(r["hybrid_us"] for r in cold1) / 1e6
        lut_sum = per_mode[mode]
        fp16_ok = all(r["lut_fp16_ok"] for r in cold1)
        wins = lut_sum * WIN_RATIO <= hyb_sum and fp16_ok
        threshold = 0
        if wins:
            for m in BATCHES:
                rows = [r for r in table if r["temp"] == "cold" and r["m"] == m]
                if rows and all(r["speedup"] and r["speedup"] > 1.0 for r in rows):
                    threshold = m
                else:
                    break
        verdict.update({"lut_mode": mode, "hybrid_sum_us": round(hyb_sum * 1e6, 1),
                        "lut_sum_us": round(lut_sum * 1e6, 1) if math.isfinite(lut_sum) else None,
                        "ratio": round(hyb_sum / lut_sum, 3) if lut_sum and math.isfinite(lut_sum) else None,
                        "lut_fp16_ok": fp16_ok, "lookup_wins": wins, "threshold_m": threshold if wins else None})
    return {"verdict": verdict, "table": table}


# --- the pod benchmark ---------------------------------------------------------------------------------------------

def _real_layer(model_dir: Path, tensor: str, k: int, n: int) -> tuple[Any, Any, str] | None:
    """codes uint8 [n, k] and scale bf16 [n, k // 128] of layers.<LAYER>.<tensor>, the first n rows; None when the
    checkpoint has no such tensor or it is too small."""
    from safetensors import safe_open

    index = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    key = next((name for name in index if name.endswith(f".layers.{LAYER}.{tensor}.weight_packed")
                and ".visual." not in name and "mtp" not in name), None)
    if key is None:
        return None
    skey = key.replace("weight_packed", "weight_scale")
    with safe_open(str(model_dir / index[key]), "pt") as f:
        shape = f.get_slice(key).get_shape()
        if shape[0] < n or shape[1] * 8 != k:
            return None
        packed = f.get_slice(key)[:n]
        scale = f.get_slice(skey)[:n]
    return unpack_ct(packed), scale, f"{key}[:{n}]"


def _random_layer(k: int, n: int, seed: int) -> tuple[Any, Any, str]:
    import torch

    gen = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (n, k), generator=gen, dtype=torch.int64).to(torch.uint8)
    scale = (torch.rand(n, k // GROUP, generator=gen) * 0.004 + 0.001).to(torch.bfloat16)
    return codes, scale, f"random seed {seed}"


def hybrid_layer(codes: Any, scale: Any, dtype: Any) -> tuple[Callable[[Any], Any], Any]:
    """(apply, layer): vLLM's RDNAHybridW4A16LinearKernel built the way compressed-tensors builds it, on the GPU."""
    import torch
    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import MPLinearLayerConfig
    from vllm.model_executor.kernels.linear.mixed_precision.rdna_hybrid_w4a16 import RDNAHybridW4A16LinearKernel
    from vllm.model_executor.layers.quantization.utils.quant_utils import pack_quantized_values_into_int32
    from vllm.scalar_type import scalar_types

    n, k = codes.shape
    cfg = MPLinearLayerConfig(full_weight_shape=(k, n), partition_weight_shape=(k, n),
                              weight_type=scalar_types.uint4b8, act_type=dtype, group_size=GROUP,
                              zero_points=False, out_type=dtype)
    ok, why = RDNAHybridW4A16LinearKernel.can_implement(cfg)
    if not ok:
        raise RuntimeError(f"RDNAHybridW4A16LinearKernel.can_implement refused: {why}")
    layer = torch.nn.Module()
    packed = pack_quantized_values_into_int32(codes.to("cuda", torch.int32), scalar_types.uint4b8, packed_dim=1)
    wq = torch.nn.Parameter(packed.contiguous(), requires_grad=False)
    wq.input_dim, wq.output_dim, wq.packed_dim = 1, 0, 1
    ws = torch.nn.Parameter(scale.to("cuda", dtype).contiguous(), requires_grad=False)
    ws.input_dim, ws.output_dim = 1, 0
    layer.register_parameter("weight_packed", wq)
    layer.register_parameter("weight_scale", ws)
    kern = RDNAHybridW4A16LinearKernel(cfg, "weight_packed", "weight_scale")
    kern.process_weights_after_loading(layer)
    return (lambda x: kern.apply_weights(layer, x)), layer


def time_graph(cs: Any, fn: Callable[[], Any], *, copies: int, repeats: int, warmup: int,
               min_load_s: float) -> tuple[list[float], str]:
    """Seconds per call from a CUDA graph of `calls` calls (a multiple of copies, about GRAPH_TARGET_S long); falls
    back to the chip sheet's eager time_torch when capture fails."""
    import time

    import torch

    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(max(warmup, copies)):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(copies):
            fn()
        end.record()
        end.synchronize()
        one = max(start.elapsed_time(end) / 1e3 / copies, 1e-7)
        calls = max(copies, min(400, int(GRAPH_TARGET_S / one)))
        calls = math.ceil(calls / copies) * copies
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(calls):
                fn()
        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize()
        began = time.monotonic()
        samples = []
        for _ in range(repeats):
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / 1e3 / calls)
        while time.monotonic() - began < min_load_s:
            graph.replay()
            torch.cuda.synchronize()
        del graph
        return samples, f"graph x{calls}"
    except Exception as exc:  # capture refused: time it eagerly, and say so
        torch.cuda.synchronize()
        samples = cs.time_torch(fn, repeats=repeats, warmup=warmup, min_load_s=min_load_s)
        return samples, f"eager ({type(exc).__name__}: {str(exc)[:120]})"


def bench_main(args: argparse.Namespace, command: str) -> int:
    cs = _chipsheet()
    lut = _lut()
    fmt = cs.fmt
    rocminfo_text, why = cs.read_rocminfo()
    refused = cs.gate(args.node, rocminfo_text, why)
    if refused:
        print(json.dumps({"ok": False, "outcome": "refused", "error": refused}))
        return cs.EXIT_GATE
    import vllm  # noqa: F401  (first, as the server does: vllm sets up its libraries before torch loads)
    import torch

    if not torch.cuda.is_available():
        print(json.dumps({"ok": False, "outcome": "no_gpu"}))
        return 1
    agent = fmt.parse_rocminfo(rocminfo_text or "")
    args.budget_s = args.budget_s or (QUICK_BUDGET_S if args.quick else FULL_BUDGET_S)

    class DispatchRun(cs.Run):
        def cmd(self, section: str) -> str:
            return f"{self.command}  # section {section}"

    run = DispatchRun(args, command)
    run.repeats, run.warmup, run.min_load_s = (7, 2, 0.5) if args.quick else (11, 3, 1.0)
    run.idle_s = 1.0 if args.quick else 2.0
    started = cs.utc_now()
    versions = cs.collect_versions(None)
    versions["vllm"] = getattr(vllm, "__version__", None)
    others = cs.run_tool(["rocm-smi", "--showpids"], timeout_s=20)[1] if cs.tool("rocm-smi") else None
    model_dir = Path(args.model).expanduser()
    layers: dict[str, tuple[Any, Any, str]] = {}
    for name, k, n, tensor in SHAPES:
        got = None
        try:
            got = _real_layer(model_dir, tensor, k, n)
        except Exception as exc:
            run.notes.append(f"{name}: real weights unreadable ({type(exc).__name__}: {exc})"[:300])
        layers[name] = got or _random_layer(k, n, seed=k + n)
    run.notes.append("weights: " + "; ".join(f"{name} from {layers[name][2]}" for name, *_ in SHAPES))
    run.power.start()
    seconds: dict[str, float] = {}
    tiles: dict[tuple[str, int], Mapping[str, Any]] = {}

    def point(section: str, shape: tuple[str, int, int, str], m: int, temp: str, kernel: str,
              dtype: Any = torch.float16, cfg: Mapping[str, Any] | None = None, quick: bool = False) -> str:
        name, k, n, _ = shape
        dname = "bfloat16" if dtype == torch.bfloat16 else "float16"
        copies = cold_copies(k, n) if temp == "cold" else 1
        mid = (f"dispatch.{kernel}.{name}.m{m}.{temp}" + ("" if dname == "float16" else ".bf16")
               + (".sweep" if section == "sweep" else ""))
        route = hybrid_route(m, k) if kernel == "hybrid" else None

        def body() -> tuple[list[float], dict[str, Any]]:
            calls: list[Callable[[], Any]] = []
            keep: list[Any] = []
            err: dict[str, Any] = {}
            try:
                codes, scale, _src = layers[name]
                gen = torch.Generator(device="cuda").manual_seed(m * 7 + k)
                a = torch.randn(m, k, device="cuda", generator=gen).to(dtype)
                for i in range(copies):
                    if kernel == "hybrid":
                        apply, layer = hybrid_layer(codes, scale, dtype)
                        keep.append(layer)
                        calls.append(lambda apply=apply: apply(a))
                    else:
                        mode = kernel.split(".")[1]
                        p = to_lut(codes.cuda(), scale.cuda())
                        call, _ = lut.launcher(a, p, lut.MODE_IDS[mode], cfg)
                        keep.append(p)
                        calls.append(call)
                    if i == 0:
                        out = calls[0]().float()
                        ref = a.float() @ dequant(codes.cuda(), scale.cuda().half()).t()
                        max_abs = float((out - ref).abs().max())
                        rel = float((out - ref).norm() / ref.norm().clamp_min(1e-12))
                        ref_max = float(ref.abs().max())
                        err = {"max_abs_error": max_abs, "rel_l2_error": rel, "ref_max": ref_max,
                               "fp16_ok": at_fp16_rounding(max_abs, rel, ref_max)}
                        if kernel != "hybrid" and dname == "float16":  # against the hybrid kernel's own output
                            apply_h, layer_h = hybrid_layer(codes, scale, torch.float16)
                            out_h = apply_h(a).float()
                            err["max_abs_vs_hybrid"] = float((out - out_h).abs().max())
                            del apply_h, layer_h, out_h
                        del out, ref
                        if not math.isfinite(max_abs) or rel > 1e-1:
                            raise ValueError(f"wrong answer: max abs error {max_abs:.4g}, relative L2 {rel:.3g}")
                state = {"i": 0}

                def fn() -> Any:
                    c = calls[state["i"] % len(calls)]
                    state["i"] += 1
                    return c()
                reps, warm, load = (5, 2, 0.3) if quick else (run.repeats, run.warmup, run.min_load_s)
                secs, how = time_graph(cs, fn, copies=copies, repeats=reps, warmup=warm, min_load_s=load)
            finally:
                del calls, keep
                cs._free_gpu()
            med = fmt.stats(secs)["median"]
            seconds[mid] = med
            nb = common_bytes(k, n)
            return [nb / s / 1e9 for s in secs], {
                "seconds_median": med, "common_bytes": nb, "common_gbs": nb / med / 1e9,
                "own_bytes": own_bytes(kernel, k, n), "own_gbs": own_bytes(kernel, k, n) / med / 1e9,
                "tflops": 2 * m * n * k / med / 1e12, "copies": copies, "timing": how, "route": route, **err}

        op = ("vllm RDNAHybridW4A16LinearKernel.apply_weights (uint4b8, group 128, no zero points), "
              f"route {route}" if kernel == "hybrid" else f"triton lut_gemv BITS 4 MODE {kernel.split('.')[1]} "
              f"{dict(cfg or {})}") + ", value = common weight bytes / time per call"
        run.measure(mid, f"{kernel}, batch {m}, K {k} to N {n}, {temp}, {dname}", "GB/s", op,
                    {"shape": name, "k": k, "n": n, "m": m, "temp": temp, "kernel": kernel, "dtype": dname,
                     "copies": copies, **({key: val for key, val in cfg.items()} if cfg else {})}, body,
                    need_s=(1.0 if quick else run.min_load_s) + 4)
        return mid

    def lut_kernel(mode: str, cfg: Mapping[str, Any]) -> str:
        return f"lut.{mode}.{cfg['tag']}"

    def candidates(m: int) -> list[Mapping[str, Any]]:
        if m == 1:
            return list(lut.M1_CONFIGS)
        if m == 4:
            return [lut.M1_CONFIGS[0], lut.DOT_CONFIGS[0]]
        return list(lut.DOT_CONFIGS[:2])

    try:
        # 1. The lookup tile per mode and M, hot at the first shape (short timings, kept in the file only).
        run.begin("sweep")
        first = SHAPES[0]
        for m in BATCHES:
            for mode in LUT_MODES:
                ids = [(point("sweep", first, m, "hot", lut_kernel(mode, cfg), cfg=cfg, quick=True), cfg)
                       for cfg in candidates(m)]
                timed = [(seconds[i], j, cfg) for j, (i, cfg) in enumerate(ids) if i in seconds]
                tiles[(mode, m)] = min(timed, key=lambda t: (t[0], t[1]))[2] if timed else candidates(m)[0]
        run.notes.append("tiles: " + json.dumps({f"{mode}.m{m}": cfg["tag"] for (mode, m), cfg in tiles.items()}))
        # 2. The table, M = 1 first (the decision), cold before hot.
        for m in BATCHES:
            run.begin(f"m{m}")
            for temp in ("cold", "hot"):
                for shape in SHAPES:
                    point(f"m{m}", shape, m, temp, "hybrid")
                    for mode in LUT_MODES:
                        cfg = tiles[(mode, m)]
                        point(f"m{m}", shape, m, temp, lut_kernel(mode, cfg), cfg=cfg)
        # 3. The teacher's own activation dtype on the hybrid kernel (bf16), cold, for reference.
        run.begin("bf16")
        for m in BATCHES:
            for shape in SHAPES:
                point("bf16", shape, m, "cold", "hybrid", dtype=torch.bfloat16)
    except Exception as exc:
        run.notes.append(f"aborted in section {run.section}: {type(exc).__name__}: {exc}"[:600])
    finally:
        run.power.stop()
    result = {
        "format": FORMAT, "gfx": (agent or {}).get("gfx") or cs.POD_GFX, "node": args.node,
        "mode": "quick" if args.quick else "full", "date": args.date or started[:10], "started": started,
        "ended": cs.utc_now(), "gpu_seconds": round(run.elapsed(), 1), "budget_s": run.budget_s,
        "command": command, "host": socket.gethostname(), "versions": versions, "device": {"rocminfo": agent},
        "power": {"reader": run.power.describe(), "idle": run.idle},
        "tiles": {f"{mode}.m{m}": cfg["tag"] for (mode, m), cfg in tiles.items()},
        "other_gpu_processes": others, "measurements": run.entries, "notes": run.notes,
    }
    result["decision"] = decide(result)
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"dispatch-{result['gfx']}-{result['date']}.json"
    path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    counts = {status: sum(1 for e in run.entries if e["status"] == status) for status in ("ok", "failed", "skipped")}
    print(json.dumps({"ok": True, "outcome": "measured", "json": str(path), "gpu_seconds": result["gpu_seconds"],
                      "counts": counts, "verdict": result["decision"]["verdict"]}))
    for row in result["decision"]["table"]:
        print(json.dumps(row))
    for e in run.entries:
        if e["status"] != "ok":
            print(json.dumps({"id": e["id"], "status": e["status"], "error": (e.get("error") or "")[:200]}))
    return 0


# --- metrics rows from a pulled result -----------------------------------------------------------------------------

def metric_rows(result: Mapping[str, Any], source: str, account: str = "D") -> list[dict[str, Any]]:
    """dispatch.* rows for lab/refine/metrics.jsonl: per measured entry its common weight GB/s, microseconds per
    call, watts, joules per call and max abs error, with the node, hub, account and sampler conditions. The tile
    sweep stays in the result file only."""
    rows = []
    reader = (result.get("power") or {}).get("reader") or {}
    versions = result.get("versions") or {}
    base = {"hub": "sjc", "account": account, "gfx": result.get("gfx"), "host": result.get("host"),
            "date": result.get("date"), "vllm": versions.get("vllm"), "torch": versions.get("torch"),
            "triton": versions.get("triton"), "rocm": versions.get("rocm"),
            "sampler": {"tool": reader.get("tool"), "period_s": reader.get("period_s"),
                        "method": "the chip sheet's Power thread (lab/pod_power.py reader), socket watts read every "
                                  "0.25 s during each measurement's window; idle read before each section"},
            "harness": "lab/kernels/dispatch_bench.py time_graph (CUDA graph of about 2 ms of calls, replayed "
                       "repeats times, then load held for min_load_s)", "file": source}
    ts = result.get("ended") or result.get("started")
    for e in result.get("measurements", []):
        if e.get("status") != "ok" or e.get("section") == "sweep":
            continue
        d = e.get("derived") or {}
        p = e.get("params") or {}
        watts = e.get("watts") or {}
        sec = d.get("seconds_median")
        cond = {**base, **{key: p[key] for key in ("shape", "k", "n", "m", "temp", "kernel", "dtype", "copies",
                                                   "BLOCK_M", "BLOCK_N", "BLOCK_K", "SPLIT_K", "num_warps")
                           if key in p},
                "section": e.get("section"), "repeats": e.get("repeats"), "timing": d.get("timing"),
                "route": d.get("route"), "rel_spread": (e.get("spread") or {}).get("rel_spread"),
                "idle_w": watts.get("idle_w"), "watt_samples": watts.get("samples")}
        values = [("common_gbs", d.get("common_gbs"), "GB/s"),
                  ("us_per_call", sec * 1e6 if sec else None, "us")]
        if watts.get("loaded_mean_w") is not None:
            values.append(("watts", watts["loaded_mean_w"], "W"))
            if sec:
                values.append(("joules_per_call", watts["loaded_mean_w"] * sec, "J"))
        values.append(("max_abs_error", d.get("max_abs_error"), "abs"))
        for key, value, unit in values:
            if value is None:
                continue
            rows.append({"metric": f"{e['id']}.{key}", "value": round(float(value), 6), "unit": unit,
                         "node": result.get("node"), "conditions": cond,
                         "source": f"R63 dispatch head-to-head, {source}", "ts": ts, "low_confidence": False})
    return rows


def ingest_main(args: argparse.Namespace) -> int:
    path = Path(args.result)
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("format") != FORMAT:
        print(json.dumps({"ok": False, "outcome": "refused", "error": f"format is {result.get('format')!r}"}))
        return 1
    rows = metric_rows(result, path.as_posix(), args.account)
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
    parser = argparse.ArgumentParser(prog="dispatch_bench.py", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    bench = sub.add_parser("bench", help="measure on the pod (node gate first)")
    bench.add_argument("--node", required=True, help="pod-a or pod-b")
    bench.add_argument("--quick", action="store_true")
    bench.add_argument("--out", default=DEFAULT_OUT)
    bench.add_argument("--model", default=DEFAULT_MODEL)
    bench.add_argument("--budget-s", type=float)
    bench.add_argument("--date")
    ingest = sub.add_parser("ingest", help="append the dispatch.* rows of a pulled result to metrics.jsonl")
    ingest.add_argument("--result", required=True)
    ingest.add_argument("--account", default="D")
    ingest.add_argument("--metrics", default=str(Path(__file__).resolve().parents[1] / "refine" / "metrics.jsonl"))
    ingest.add_argument("--dry-run", action="store_true")
    dec = sub.add_parser("decide", help="print the rule's verdict and table for a pulled result")
    dec.add_argument("--result", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    if args.command == "ingest":
        return ingest_main(args)
    if args.command == "decide":
        print(json.dumps(decide(json.loads(Path(args.result).read_text(encoding="utf-8"))), indent=1))
        return 0
    return bench_main(args, shlex.join(["python3", "dispatch_bench.py", *argv]))


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())
