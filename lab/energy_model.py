"""Decode energy model for the Strix Halo (Radeon 8060S, gfx1151): joules per generated token from the chip sheet.

    python lab/energy_model.py                                  # predictions for the teacher and the student
    python lab/energy_model.py --json                           # the same as one JSON object
    python lab/energy_model.py --idle-w 24 --batches 1,4,8,16,64 --sheet <chipsheet.json>
    python lab/energy_model.py check <rows.jsonl | sweep.json>  # a measured batch curve against the prediction
    python -m lab.energy_model ...                              # the same

lab/refine/research/kernels/energy_model_gfx1151.md derives and cites every formula and default below; this file is
the arithmetic, so each prediction in that note is reproducible from the committed chip sheet. Standard library only.

The model. At B concurrent streams, with P the socket watts while decoding:

    J/token(B) = (P_dynamic(B) + P_idle) / tok_s(B),    tok_s(B) = B / t_step(B)

One decode step runs its kernels one after another, so its time is a sum, each kernel class at its own roofline:

    t_step(B) = sum over weight classes c of max(tiles(B) * bytes_c / BW_w, B * flops_c / C) + B * seq_bytes / BW

- bytes_c: the class's weights, read once per step (int4 values plus one bf16 scale per group of 128 for W4A16, two
  bytes a parameter for bf16); flops_c = 2 x parameters per generated token.
- seq_bytes: what every sequence adds per step: its Gated DeltaNet recurrent state, read and written (fp32 in both
  checkpoints), and the KV cache of the full-attention layers at the current context.
- BW: the chip sheet's device copy bandwidth beyond the 32 MiB last-level cache. C: the chip sheet's dense matmul rate
  (the 4B LoRA forward GEMM) in the activation dtype. BW_w: BW for a build at the bandwidth ceiling; for a measured
  build, the effective weight bandwidth its batch-1 rate implies.
- tiles(B): 1 for a kernel that serves the whole batch from one weight read; ceil(B / m_tile) for one whose weight
  pass serves m_tile rows (vLLM's Conch W4A16 kernel at the April commit launches BLOCK_M 128).

Below the knee the max() is the memory term, so tok_s rises almost in proportion to B while P stays nearly flat (the
chip sheet's matmul and dequant rows draw within a few watts of each other from M 1 to 16), and J/token falls as 1/B
until the per-sequence bytes catch up with the shared weight bytes (the state knee). Past the compute knee, where
B * flops_c / C overtakes the weight read, tok_s stops rising and J/token flattens. From the chip sheet directly, a
weight class turns compute-bound at B = C / rate(M=1), its batch-1 TFLOP/s while bandwidth-bound.

P(B) = (1 - u) * P_bw(1) * (1 + s * (B - 1)) + u * P_compute, where u is the share of the step spent in
compute-bound GEMMs, s the median relative slope of loaded watts against M over the chip sheet's matmul and dequant
shapes, P_bw(1) the batch-1 watts (measured for a measured build; the chip sheet's median for the kernel class at
the bandwidth ceiling) and P_compute the sheet's median over its compute-bound LoRA GEMM rows. P_idle is the loaded
idle the benchmark harness subtracts for joules_per_token_above_idle: 10.2 W measured with the teacher loaded; the
chip sheet's per-section idles (24 to 55 W, read 3 s after the section before) are not settled and are not used.

The harness workload. lab/bench.py integrates power over the decode window (first token of the first request to the
last token of the last), which also holds the prefills of every request but the first. With R requests per stream
of T tokens each and a mean prefill time T_pf per request:

    window(B) = R * T * t_step(B) + (R * B - 1) * T_pf,    tokens = R * T * B,    J/token = P(B) * window / tokens

so a prediction for a harness row names its workload; Workload(prefill_s=0) is decode alone.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SHEET_PATH = REPO / "lab" / "refine" / "research" / "kernels" / "data" / "chipsheet-gfx1151-2026-10-11.json"
GB = 1e9
TERA = 1e12
W4A16_GROUP = 128
W4A16_BYTES = 0.5 + 2.0 / W4A16_GROUP   # int4 value plus a bf16 scale per group of 128 (symmetric: no zero point)
BF16_BYTES = 2.0
APRIL_TOK_S_B1 = 2.88       # the April vLLM build's batch-1 decode on pod C, lab/README.md (teacher section)
APRIL_WATTS_B1 = 120.0      # socket watts averaged while it decoded, same run
APRIL_M_TILE = 128          # Conch 1.2.1's BLOCK_M at the April commit
LOADED_IDLE_W = 10.2        # pod C, teacher loaded, nothing running, 60 s (lab/README.md, pod power sampler run)
DEFAULT_BATCHES = (1, 4, 8, 16)
DEFAULT_TOLERANCE = 0.15


# --- what the chip sheet says ----------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ChipInputs:
    """The chip sheet numbers the model reads, with the rows they came from."""

    bandwidth_gbs: float                 # median device copy above the last-level cache (2 x bytes / time)
    cache_gbs: float                     # the best copy rate with the working set inside the cache
    read_gbs: float                      # the best weight-read rate of a bandwidth-bound matmul at M = 1
    compute_tflops: dict[str, float]     # dtype -> the 4B LoRA forward GEMM rate (the dense compute ceiling)
    rate_m1: dict[str, float]            # shape -> TFLOP/s at M = 1 (matmul and dequant rows)
    scaling_m16: dict[str, float]        # shape -> rate(M = 16) / rate(M = 1)
    shape_dtype: dict[str, str]          # shape -> the dtype its MMA runs in
    watts: dict[str, float]              # section -> median loaded watts over its rows (lora: fwd and dgrad only)
    watts_slope: float                   # median relative change of loaded watts per unit of M, matmul and dequant
    idle_sections: dict[str, float]      # section -> the idle the sheet read before it (not settled; recorded only)
    cache_bytes: int
    source: str = ""


def load_sheet(path: str | Path = SHEET_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        sheet = json.load(handle)
    if sheet.get("format") != "atlas.chipsheet/1":
        raise ValueError(f"{path}: format {sheet.get('format')!r}, not atlas.chipsheet/1")
    return sheet


def _shape(entry: Mapping[str, Any]) -> str:
    """The row id without its batch part: matmul.fp16.decode27b.k5120_n6144.m4 -> matmul.fp16.decode27b.k5120_n6144."""
    return ".".join(part for part in entry["id"].split(".") if not (part[:1] == "m" and part[1:].isdigit()))


def _slope(points: Sequence[tuple[float, float]]) -> float:
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return 0.0 if den == 0 else sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def chip_inputs(sheet: Mapping[str, Any], source: str = "") -> ChipInputs:
    """Read the model's inputs from a chip sheet (format atlas.chipsheet/1); rows whose status is not ok are left out."""
    rows = [m for m in sheet["measurements"] if m.get("status") == "ok" and m.get("value") is not None]
    cache = int(sheet["device"]["rocminfo"]["cache_bytes"]["l3"])
    copy = [m for m in rows if m["section"] == "copy"]
    beyond = [m for m in copy if m["params"]["bytes"] > cache]
    inside = [m for m in copy if m["params"]["bytes"] <= cache]
    if not beyond:
        raise ValueError("the chip sheet has no copy row beyond the last-level cache")
    lora = [m for m in rows if m["section"] == "lora" and m["params"].get("step") in ("fwd", "dgrad")]
    compute = {m["params"]["dtype"]: float(m["value"]) for m in lora if m["params"]["step"] == "fwd"}
    gemm = [m for m in rows if m["section"] in ("matmul", "dequant")]
    by_shape: dict[str, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for m in gemm:
        by_shape[_shape(m)][int(m["params"]["m"])] = m
    rate_m1, scaling, dtypes, slopes = {}, {}, {}, []
    for shape, per_m in by_shape.items():
        if 1 in per_m:
            rate_m1[shape] = float(per_m[1]["value"])
            dtypes[shape] = per_m[1]["params"].get("dtype", "float16")   # the dequant kernel computes in fp16
        if 1 in per_m and 16 in per_m:
            scaling[shape] = float(per_m[16]["value"]) / float(per_m[1]["value"])
        pts = [(float(k), float(v["watts"]["loaded_mean_w"])) for k, v in sorted(per_m.items())
               if (v.get("watts") or {}).get("loaded_mean_w") is not None]
        if len(pts) >= 2:
            slopes.append(_slope(pts) / statistics.fmean(w for _, w in pts))
    watts = {}
    for section, members in (("copy", beyond), ("matmul", [m for m in gemm if m["section"] == "matmul"]),
                             ("dequant", [m for m in gemm if m["section"] == "dequant"]), ("lora", lora)):
        loaded = [m["watts"]["loaded_mean_w"] for m in members if (m.get("watts") or {}).get("loaded_mean_w")]
        if loaded:
            watts[section] = statistics.median(loaded)
    reads = [m["derived"]["weight_gbs"] for m in gemm
             if m["section"] == "matmul" and m["params"]["m"] == 1 and (m.get("derived") or {}).get("weight_gbs")]
    idle = {k: float(v["idle_w"]) for k, v in (sheet.get("power") or {}).get("idle", {}).items()
            if v.get("idle_w") is not None}
    return ChipInputs(
        bandwidth_gbs=statistics.median(float(m["value"]) for m in beyond),
        cache_gbs=max((float(m["value"]) for m in inside), default=float("nan")),
        read_gbs=max(reads) if reads else float("nan"), compute_tflops=compute, rate_m1=rate_m1,
        scaling_m16=scaling, shape_dtype=dtypes, watts=watts,
        watts_slope=statistics.median(slopes) if slopes else 0.0, idle_sections=idle, cache_bytes=cache,
        source=source)


def sheet_knees(chip: ChipInputs) -> dict[str, float]:
    """Per measured shape, the batch where it would turn compute-bound: C (its dtype) / its rate at M = 1."""
    return {shape: chip.compute_tflops[chip.shape_dtype[shape]] / rate for shape, rate in chip.rate_m1.items()
            if chip.shape_dtype.get(shape) in chip.compute_tflops}


# --- the models -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class WeightClass:
    name: str
    params: float
    bytes_per_param: float

    @property
    def bytes(self) -> float:
        return self.params * self.bytes_per_param

    @property
    def flops(self) -> float:
        return 2.0 * self.params


@dataclass(frozen=True)
class Model:
    name: str
    classes: tuple[WeightClass, ...]
    state_bytes: float          # recurrent state read and written per sequence per step
    kv_bytes_per_token: float   # KV cache read per sequence per step, per token of context
    attn_layers: int
    gdn_layers: int

    @property
    def weight_bytes(self) -> float:
        return sum(c.bytes for c in self.classes)

    @property
    def flops(self) -> float:
        return sum(c.flops for c in self.classes)

    def seq_bytes(self, context: float) -> float:
        return self.state_bytes + self.kv_bytes_per_token * context


def qwen35_hybrid(name: str, *, hidden: int, layers: int, full_attention_interval: int, heads: int, kv_heads: int,
                  head_dim: int, linear_k_heads: int, linear_v_heads: int, linear_k_dim: int, linear_v_dim: int,
                  intermediate: int, vocab: int, linear_bytes: float, head_bytes: float = BF16_BYTES,
                  ssm_state_bytes: float = 4.0, kv_bytes: float = 2.0) -> Model:
    """A Qwen3.5-family hybrid decoder from its config: one full-attention layer in every `full_attention_interval`
    (gated: q_proj also emits the output gate), Gated DeltaNet in the rest (in_proj_qkvz and out_proj in the
    checkpoint's linear format, in_proj_a and in_proj_b in bf16), a SwiGLU MLP in every layer, and the lm_head read
    once per step. Embedding lookups, norms and the short convolution are left out (a row or a few KB per step)."""
    attn_layers = layers // full_attention_interval
    gdn_layers = layers - attn_layers
    q_dim, kv_dim = heads * head_dim, kv_heads * head_dim
    attn = hidden * 2 * q_dim + 2 * hidden * kv_dim + q_dim * hidden
    k_dim, v_dim = linear_k_heads * linear_k_dim, linear_v_heads * linear_v_dim
    gdn = hidden * (2 * k_dim + 2 * v_dim) + v_dim * hidden
    gdn_ab = hidden * 2 * linear_v_heads
    mlp = 3 * hidden * intermediate
    classes = (
        WeightClass("linear", attn_layers * attn + gdn_layers * gdn + layers * mlp, linear_bytes),
        WeightClass("gdn_ab", gdn_layers * gdn_ab, BF16_BYTES),
        WeightClass("lm_head", vocab * hidden, head_bytes),
    )
    state = 2.0 * gdn_layers * linear_v_heads * linear_k_dim * linear_v_dim * ssm_state_bytes
    kv = attn_layers * kv_heads * head_dim * 2 * kv_bytes
    return Model(name, classes, state, kv, attn_layers, gdn_layers)


# RedHatAI/Qwen3.8-27B-INT4 config.json: compressed-tensors int4, symmetric, group 128, lm_head and in_proj_a/b in bf16.
TEACHER = qwen35_hybrid(
    "teacher Qwen3.8-27B W4A16", hidden=5120, layers=64, full_attention_interval=4, heads=24, kv_heads=4,
    head_dim=256, linear_k_heads=16, linear_v_heads=48, linear_k_dim=128, linear_v_dim=128, intermediate=17408,
    vocab=248320, linear_bytes=W4A16_BYTES)
# Qwen/Qwen3.5-4B config.json: bf16, tied embeddings (the lm_head is the embedding matrix, still read every step).
STUDENT = qwen35_hybrid(
    "student Qwen3.5-4B bf16", hidden=2560, layers=32, full_attention_interval=4, heads=16, kv_heads=4,
    head_dim=256, linear_k_heads=16, linear_v_heads=32, linear_k_dim=128, linear_v_dim=128, intermediate=9216,
    vocab=248320, linear_bytes=BF16_BYTES)
STUDENT_W4A16 = replace(STUDENT, name="student Qwen3.5-4B W4A16", classes=(
    replace(STUDENT.classes[0], bytes_per_param=W4A16_BYTES), *STUDENT.classes[1:]))
MODELS = {"teacher": TEACHER, "student": STUDENT, "student-w4a16": STUDENT_W4A16}


# --- builds, workloads, predictions ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class Build:
    name: str
    weight_gbs: float           # effective rate of the GEMM weight reads
    state_gbs: float            # rate of the per-sequence state and KV traffic
    compute_tflops: float       # dense matmul ceiling in the activation dtype
    m_tile: int | None          # rows one weight pass serves; None: the whole batch
    watts_b1: float             # socket watts while decoding at batch 1
    watts_slope: float          # relative change of those watts per added sequence while bandwidth-bound
    watts_compute: float        # socket watts with the GEMMs compute-bound
    basis: str = ""


@dataclass(frozen=True)
class Workload:
    """What one stream sends: requests one after another, each prefilled then decoded for max_tokens."""

    requests_per_stream: int = 1
    max_tokens: int = 128
    prompt_tokens: float = 0.0
    prefill_s: float = 0.0      # mean prefill time per request; 0 means decode alone
    name: str = "decode only"

    @property
    def context(self) -> float:
        """The mean context a decode step reads: the prompt plus half the generated tokens."""
        return self.prompt_tokens + self.max_tokens / 2.0


DECODE_512 = Workload(prompt_tokens=448.0, name="decode only, 512-token mean context")
# lab/bench.py's defaults with prompt set bench-v1: 4 requests a stream of 128 tokens; the prompt tokens and mean
# prefill time are the batch-1 requests of the April build's attach sweep (workload_from_rows on the fixture).
BENCH_V1 = Workload(requests_per_stream=4, max_tokens=128, prompt_tokens=401.75, prefill_s=1.62105,
                    name="harness bench-v1 (4 x 128 tokens a stream, 2 text and 2 image prompts)")


def ceiling_build(chip: ChipInputs, *, section: str = "dequant", dtype: str = "bfloat16",
                  name: str = "ceiling") -> Build:
    """A build whose every kernel runs at the chip sheet's roofline: weights and state at the copy bandwidth, GEMMs up
    to the dense matmul rate, one weight read for the whole batch, and the watts the sheet drew for that kernel
    class while bandwidth-bound."""
    return Build(name=name, weight_gbs=chip.bandwidth_gbs, state_gbs=chip.bandwidth_gbs,
                 compute_tflops=chip.compute_tflops[dtype], m_tile=None, watts_b1=chip.watts[section],
                 watts_slope=chip.watts_slope, watts_compute=chip.watts["lora"],
                 basis=f"chip sheet: copy beyond cache, {dtype} LoRA forward GEMM, {section} watts")


def measured_build(chip: ChipInputs, model: Model, *, tok_s_b1: float, watts_b1: float, m_tile: int | None,
                   context: float, dtype: str = "bfloat16", name: str = "measured") -> Build:
    """A build known by its measured batch-1 decode rate: its weight path's effective bandwidth is what that rate leaves
    after the per-sequence traffic at the sheet's bandwidth; everything else is the chip sheet's."""
    weight_s = 1.0 / tok_s_b1 - model.seq_bytes(context) / (chip.bandwidth_gbs * GB)
    if weight_s <= 0:
        raise ValueError(f"{tok_s_b1} tok/s is faster than the per-sequence traffic alone allows")
    return Build(name=name, weight_gbs=model.weight_bytes / weight_s / GB, state_gbs=chip.bandwidth_gbs,
                 compute_tflops=chip.compute_tflops[dtype], m_tile=m_tile, watts_b1=watts_b1,
                 watts_slope=chip.watts_slope, watts_compute=chip.watts["lora"],
                 basis=f"measured {tok_s_b1} tok/s at {watts_b1} W at batch 1; tile {m_tile}")


def april_build(chip: ChipInputs, model: Model = TEACHER, *, context: float = DECODE_512.context) -> Build:
    return measured_build(chip, model, tok_s_b1=APRIL_TOK_S_B1, watts_b1=APRIL_WATTS_B1, m_tile=APRIL_M_TILE,
                          context=context, name="April build (vLLM b1388b1, Conch W4A16)")


def step(model: Model, build: Build, batch: int, context: float) -> dict[str, float]:
    """One decode step at `batch` streams: its time and the share of it spent in compute-bound GEMMs."""
    tiles = 1 if build.m_tile is None else math.ceil(batch / build.m_tile)
    total = bound = 0.0
    for c in model.classes:
        memory = tiles * c.bytes / (build.weight_gbs * GB)
        compute = batch * c.flops / (build.compute_tflops * TERA)
        total += max(memory, compute)
        bound += compute if compute > memory else 0.0
    seq = batch * model.seq_bytes(context) / (build.state_gbs * GB)
    return {"step_s": total + seq, "compute_share": bound / (total + seq), "seq_share": seq / (total + seq)}


def watts(build: Build, batch: int, compute_share: float) -> float:
    bandwidth_bound = build.watts_b1 * (1.0 + build.watts_slope * (batch - 1))
    return (1.0 - compute_share) * bandwidth_bound + compute_share * build.watts_compute


def predict(model: Model, build: Build, batch: int, workload: Workload = DECODE_512,
            idle_w: float = LOADED_IDLE_W) -> dict[str, Any]:
    """Decode rate, watts and joules per token at `batch` streams under `workload` (see the module docstring)."""
    s = step(model, build, batch, workload.context)
    p = watts(build, batch, s["compute_share"])
    tokens = workload.requests_per_stream * workload.max_tokens * batch
    window = (workload.requests_per_stream * workload.max_tokens * s["step_s"]
              + max(workload.requests_per_stream * batch - 1, 0) * workload.prefill_s)
    return {"model": model.name, "build": build.name, "batch": batch, "workload": workload.name,
            "step_s": s["step_s"], "compute_share": s["compute_share"], "seq_share": s["seq_share"],
            "decode_tok_s": 1.0 / s["step_s"], "decode_only_tok_s_total": batch / s["step_s"],
            "decode_tok_s_total": tokens / window, "window_s": window, "watts": p, "idle_w": idle_w,
            "joules_per_token": p * window / tokens,
            "joules_per_token_above_idle": (p - idle_w) * window / tokens,
            "ttft_first_round_s": workload.prefill_s * (batch + 1) / 2.0}


def knees(model: Model, build: Build, context: float = DECODE_512.context) -> dict[str, Any]:
    """Where the J/token curve bends: the batch where a weight class turns compute-bound (and which class), the
    kernel's tile boundary when its weight pass serves a fixed number of rows, and the state knee, the batch where
    the per-sequence traffic takes as long as the shared weight read. The knee is the smallest of the first two."""
    compute = {}
    for c in model.classes:
        memory = c.bytes / (build.weight_gbs * GB)
        compute[c.name] = memory * build.compute_tflops * TERA / c.flops
    weight_s = sum(c.bytes for c in model.classes) / (build.weight_gbs * GB)
    state = weight_s / (model.seq_bytes(context) / (build.state_gbs * GB))
    first = min(compute, key=compute.get)
    knee = compute[first] if build.m_tile is None else min(compute[first], float(build.m_tile))
    return {"model": model.name, "build": build.name, "knee": knee,
            "binds": first if build.m_tile is None or compute[first] < build.m_tile else "m_tile",
            "compute": compute, "tile": build.m_tile, "state": state, "context": context}


# --- the optimizer's objective -----------------------------------------------------------------------------------

OBJECTIVE: dict[str, Any] = {
    "metric": "joules_per_token",
    "goal": "minimize",
    "definition": "lab/bench.py row field joules_per_token: socket energy (amd-smi SOCKET_POWER through "
                  "lab/pod_power.py, trapezoid over the decode window, edges interpolated) over the tokens generated "
                  "in that window; gross, nothing subtracted; median over the arm's spawns",
    "beside": ["joules_per_token_above_idle", "idle_w.loaded", "watts.avg_w", "decode_tok_s", "decode_tok_s_total",
               "ttft_s", "in_flight_mean"],
    "fixed": "one prompt set, max_tokens, requests per stream and thinking setting for every arm of a comparison",
    "lanes": {
        "interactive": {
            "batch": 1,
            "outcome_constraints": ["decode_tok_s >= 2.88", "ttft_s <= 2.0"],
            "why": "a live answer on the box is one stream: never slower per token than the April build's 2.88 "
                   "tok/s, and its first token within 2 s (the April build's batch-1 harness median is 1.04 to "
                   "1.37 s through the proxy)",
        },
        "labeling": {
            "batch": 8,
            "outcome_constraints": ["decode_tok_s_total >= incumbent", "ttft_s <= 60",
                                    "ttft_s + (max_tokens - 1) / decode_tok_s <= 120"],
            "why": "bulk teacher labeling: a lower J/token must not come from fewer tokens per second (kernels "
                   "rubric K6), and every request must finish inside the runtime's 120 s timeout "
                   "(config/models.yaml timeout_s)",
        },
    },
}


# --- a measured curve against the prediction -------------------------------------------------------------------------

def _median(values: Iterable[float | None]) -> float | None:
    xs = [float(v) for v in values if v is not None]
    return statistics.median(xs) if xs else None


def measured_points(path: str | Path) -> tuple[dict[int, dict[str, Any]], Workload]:
    """Per batch size, the medians of a measured file, and the workload it ran. Reads lab/bench.py rows (rows.jsonl,
    schema lab.bench.row/1; ok rows only) or a request-side sweep result (schema atlas.sweep_request/1; the base
    slice, one request per stream unless its cells say otherwise)."""
    text = Path(path).read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("{") and '"atlas.sweep_request/1"' in stripped[:400]:
        return _sweep_points(json.loads(text))
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    rows = [r for r in rows if r.get("schema") == "lab.bench.row/1" and r.get("outcome") == "ok"]
    if not rows:
        raise ValueError(f"{path}: no ok lab.bench.row/1 rows")
    by_batch: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        by_batch[int(r["batch_size"])].append(r)
    points = {b: {"n": len(rs), "decode_tok_s": _median(r.get("decode_tok_s") for r in rs),
                  "decode_tok_s_total": _median(r.get("decode_tok_s_total") for r in rs),
                  "watts": _median((r.get("watts") or {}).get("avg_w") for r in rs),
                  "joules_per_token": _median(r.get("joules_per_token") for r in rs),
                  "ttft_s": _median(r.get("ttft_s") for r in rs),
                  "idle_w": _median((r.get("idle_w") or {}).get("loaded") for r in rs)}
              for b, rs in sorted(by_batch.items())}
    return points, workload_from_rows(rows)


def workload_from_rows(rows: Sequence[Mapping[str, Any]]) -> Workload:
    """The workload the rows ran: requests per stream and max tokens from their conditions, and the mean prompt
    tokens and mean time to first token of the batch-1 requests (one request in flight, so the prefill alone)."""
    cond = rows[0].get("conditions") or {}
    single = [q for r in rows if int(r["batch_size"]) == 1 for q in r.get("requests") or []
              if q.get("ttft_s") is not None and not q.get("error")]
    if not single:
        raise ValueError("no batch-1 request to read the prefill time from")
    return Workload(requests_per_stream=int(cond.get("requests_per_stream") or 1),
                    max_tokens=int(cond.get("max_tokens") or 128),
                    prompt_tokens=statistics.fmean(q.get("prompt_tokens") or 0 for q in single),
                    prefill_s=statistics.fmean(q["ttft_s"] for q in single),
                    name=f"as measured ({(cond.get('prompt_set') or {}).get('id', 'unknown prompt set')})")


def _sweep_points(result: Mapping[str, Any]) -> tuple[dict[int, dict[str, Any]], Workload]:
    cells = [c for c in result.get("cells") or [] if c.get("slice") == "base" and c.get("n")]
    if not cells:
        raise ValueError("the sweep result has no measured base cell")

    def med(cell: Mapping[str, Any], metric: str) -> float | None:
        return ((cell.get("stats") or {}).get(metric) or {}).get("median")

    points = {int(c["batch"]): {"n": c["n"], "decode_tok_s": med(c, "decode_tok_s"),
                                "decode_tok_s_total": med(c, "decode_tok_s_total"), "watts": med(c, "watts_avg"),
                                "joules_per_token": med(c, "joules_per_token"), "ttft_s": med(c, "ttft_s"),
                                "idle_w": ((result.get("idle_w") or {}).get("loaded"))}
              for c in sorted(cells, key=lambda c: c["batch"])}
    one = next((c for c in cells if int(c["batch"]) == 1), None)
    if one is None:
        raise ValueError("the sweep result has no batch-1 base cell")
    work = Workload(requests_per_stream=int(one.get("requests_per_stream") or 1),
                    max_tokens=int(one.get("max_tokens") or 128), prompt_tokens=float(one.get("prompt_tokens") or 0),
                    prefill_s=float(med(one, "ttft_s") or 0.0), name=f"as measured (sweep, {one.get('prompt')})")
    return points, work


def anchored_build(chip: ChipInputs, model: Model, points: Mapping[int, Mapping[str, Any]], work: Workload,
                   *, m_tile: int | None = APRIL_M_TILE) -> Build:
    """The measured build calibrated on the file's own batch-1 point: the decode rate that, with the workload's
    prefills, gives the measured batch-1 tokens per second, and the measured batch-1 watts."""
    one = points.get(1)
    if not one or not one.get("decode_tok_s_total") or not one.get("watts"):
        raise ValueError("anchoring needs a batch-1 point with a decode rate and watts")
    per_stream_tokens = work.requests_per_stream * work.max_tokens
    window = per_stream_tokens / one["decode_tok_s_total"]
    step_s = (window - (work.requests_per_stream - 1) * work.prefill_s) / per_stream_tokens
    return measured_build(chip, model, tok_s_b1=1.0 / step_s, watts_b1=one["watts"], m_tile=m_tile,
                          context=work.context, name="anchored on the file's batch 1")


def check(path: str | Path, *, chip: ChipInputs, model: Model = TEACHER, idle_w: float = LOADED_IDLE_W,
          tolerance: float = DEFAULT_TOLERANCE) -> dict[str, Any]:
    """A measured batch curve against two predictions under the file's own workload: the April build as the model
    states it (2.88 tok/s and 120 W at batch 1), and the same model anchored on the file's batch-1 point, which
    tests the batch dependence alone. A metric held at a batch size when measured / predicted is within tolerance."""
    points, work = measured_points(path)
    stated = april_build(chip, model, context=work.context)
    anchored = anchored_build(chip, model, points, work)
    out = []
    for batch, m in points.items():
        row: dict[str, Any] = {"batch": batch, "measured": m}
        for label, build in (("stated", stated), ("anchored", anchored)):
            p = predict(model, build, batch, work, idle_w)
            ratios = {k: (m[k] / p[k] if m.get(k) is not None and p[k] else None)
                      for k in ("decode_tok_s_total", "watts", "joules_per_token")}
            row[label] = {"decode_tok_s_total": p["decode_tok_s_total"], "watts": p["watts"],
                          "joules_per_token": p["joules_per_token"], "ttft_first_round_s": p["ttft_first_round_s"],
                          "ratio": ratios,
                          "held": {k: (None if v is None else abs(v - 1.0) <= tolerance) for k, v in ratios.items()}}
        out.append(row)
    return {"file": str(path), "model": model.name, "workload": asdict(work), "tolerance": tolerance,
            "builds": {"stated": asdict(stated), "anchored": asdict(anchored)}, "points": out}


# --- the report ------------------------------------------------------------------------------------------------------

def report(chip: ChipInputs, *, batches: Sequence[int] = DEFAULT_BATCHES, idle_w: float = LOADED_IDLE_W) -> dict:
    teacher_april = april_build(chip, TEACHER)
    teacher_ceiling = ceiling_build(chip, section="dequant", name="ceiling (teacher W4A16)")
    student_ceiling = ceiling_build(chip, section="matmul", name="ceiling (student bf16)")
    student_w4_ceiling = ceiling_build(chip, section="dequant", name="ceiling (student W4A16)")
    rows = []
    for model, build, works in ((TEACHER, teacher_april, (DECODE_512, BENCH_V1)),
                                (TEACHER, teacher_ceiling, (DECODE_512, BENCH_V1)),
                                (STUDENT, student_ceiling, (DECODE_512,))):
        for work in works:   # BENCH_V1's prefill time is the teacher's, so the student is shown decoding alone
            for b in batches:
                rows.append(predict(model, build, b, work, idle_w))
    return {
        "sheet": chip.source,
        "inputs": asdict(chip),
        "idle_w": idle_w,
        "builds": {b.name: asdict(b) for b in (teacher_april, teacher_ceiling, student_ceiling, student_w4_ceiling)},
        "models": {k: {"weight_gb": m.weight_bytes / GB, "flops_per_token_g": m.flops / 1e9,
                       "state_gb": m.state_bytes / GB, "kv_bytes_per_token": m.kv_bytes_per_token,
                       "classes": [asdict(c) | {"gb": c.bytes / GB} for c in m.classes]}
                   for k, m in MODELS.items()},
        "knees": [knees(TEACHER, teacher_april), knees(TEACHER, teacher_ceiling),
                  knees(STUDENT, student_ceiling), knees(STUDENT_W4A16, student_w4_ceiling)],
        "sheet_knees": sheet_knees(chip),
        "predictions": rows,
        "objective": OBJECTIVE,
    }


def _table(rep: Mapping[str, Any]) -> str:
    lines = ["| Model | Build | Workload | Batch | tok/s per stream | tok/s total | W | J/token | J/token above idle |",
             "|---|---|---|---|---|---|---|---|---|"]
    for p in rep["predictions"]:
        lines.append(f"| {p['model']} | {p['build']} | {p['workload']} | {p['batch']} | {p['decode_tok_s']:.2f} | "
                     f"{p['decode_tok_s_total']:.1f} | {p['watts']:.1f} | {p['joules_per_token']:.2f} | "
                     f"{p['joules_per_token_above_idle']:.2f} |")
    lines += ["", "| Model | Build | Knee (batch) | Binds | Compute knee per class | State knee |", "|---|---|---|---|---|---|"]
    for k in rep["knees"]:
        per = ", ".join(f"{n} {v:.0f}" for n, v in k["compute"].items())
        lines.append(f"| {k['model']} | {k['build']} | {k['knee']:.0f} | {k['binds']} | {per} | {k['state']:.0f} |")
    c = rep["inputs"]
    lines += ["", f"Chip sheet: {rep['sheet']}; copy beyond cache {c['bandwidth_gbs']:.1f} GB/s, in cache "
              f"{c['cache_gbs']:.1f} GB/s, best matmul weight read {c['read_gbs']:.1f} GB/s; dense rate "
              + ", ".join(f"{k} {v:.1f} TFLOP/s" for k, v in c["compute_tflops"].items())
              + "; watts by section " + ", ".join(f"{k} {v:.1f}" for k, v in c["watts"].items())
              + f"; watts slope {c['watts_slope'] * 100:.3f} % per added row; loaded idle {rep['idle_w']} W."]
    return "\n".join(lines)


def _check_table(res: Mapping[str, Any]) -> str:
    lines = [f"{res['file']}: {res['workload']['name']}, tolerance {res['tolerance']:.0%}",
             "| Batch | n | tok/s measured | stated | anchored | W measured | stated | anchored | J/token measured | "
             "stated | anchored |", "|---|---|---|---|---|---|---|---|---|---|---|"]

    def cell(p: Mapping[str, Any], key: str, digits: int) -> str:
        held = p["held"][key]
        mark = "" if held is None else (" held" if held else " missed")
        return f"{p[key]:.{digits}f} ({p['ratio'][key]:.2f}{mark})" if p["ratio"][key] is not None else "-"

    for row in res["points"]:
        m = row["measured"]
        lines.append(f"| {row['batch']} | {m['n']} | {m['decode_tok_s_total']} | "
                     f"{cell(row['stated'], 'decode_tok_s_total', 1)} | {cell(row['anchored'], 'decode_tok_s_total', 1)} | "
                     f"{m['watts']} | {cell(row['stated'], 'watts', 1)} | {cell(row['anchored'], 'watts', 1)} | "
                     f"{m['joules_per_token']} | {cell(row['stated'], 'joules_per_token', 2)} | "
                     f"{cell(row['anchored'], 'joules_per_token', 2)} |")
    return "\n".join(lines)


def _batches(text: str) -> list[int]:
    out = [int(x) for x in text.split(",") if x.strip()]
    if not out or any(b < 1 for b in out):
        raise argparse.ArgumentTypeError("batch sizes are positive integers, comma separated")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lab.energy_model", description=__doc__.splitlines()[0])
    parser.add_argument("command", nargs="?", choices=("report", "check"), default="report")
    parser.add_argument("path", nargs="?", help="check: a lab/bench.py rows.jsonl or a sweep_request result JSON")
    parser.add_argument("--sheet", default=str(SHEET_PATH))
    parser.add_argument("--batches", type=_batches, default=list(DEFAULT_BATCHES))
    parser.add_argument("--idle-w", type=float, default=LOADED_IDLE_W)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--model", choices=sorted(MODELS), default="teacher", help="check: the model the file ran")
    parser.add_argument("--json", action="store_true", help="one JSON object instead of Markdown tables")
    args = parser.parse_args(argv)
    sheet_path = Path(args.sheet)
    try:
        sheet_name = sheet_path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        sheet_name = str(sheet_path)
    chip = chip_inputs(load_sheet(sheet_path), source=sheet_name)
    if args.command == "check":
        if not args.path:
            parser.error("check needs the path of a measured file")
        res = check(args.path, chip=chip, model=MODELS[args.model], idle_w=args.idle_w, tolerance=args.tolerance)
        print(json.dumps(res) if args.json else _check_table(res))
        return 0
    rep = report(chip, batches=args.batches, idle_w=args.idle_w)
    print(json.dumps(rep) if args.json else _table(rep))
    return 0


if __name__ == "__main__":
    if not __package__:
        sys.path.insert(0, str(REPO))
    sys.exit(main())
