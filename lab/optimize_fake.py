"""Stand-ins for the settings optimizer's two trial runners: a settings-aware fake vLLM server and a fake trainer.

    python lab/optimize_fake.py serve <lab/bench_fake.py serve flags> --engine-args [engine args]
    python lab/optimize_fake.py train --settings S.json --out DIR --steps 20 [--time-scale 0.001]

    from lab.optimize_fake import serve_command, train_command, decode_rate, train_rate

Nothing here touches a GPU or the network beyond 127.0.0.1. Each fake reads its settings the way the real process
does, under the parameter names of the search spaces on main (R35 inference, R36 training):

The server reads its engine args (everything after --engine-args, where the inference runner appends a trial's
mapped args) and its own environment, exactly what lab/refine/research/kernels/data/searchspace-inference-gfx1151.json
maps each value to: --dtype float16 with VLLM_DISABLED_KERNELS naming ConchLinearKernel is the exllama_fp16 linear
kernel, --speculative-config {"num_speculative_tokens": n}, --compilation-config {"cudagraph_mode": ...} or
--enforce-eager, --mamba-ssm-cache-dtype, --attention-backend, GPU_MAX_HW_QUEUES, VLLM_ROCM_USE_SKINNY_GEMM,
VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE, AMDGCN_USE_BUFFER_OPS. decode_rate is a fixed response surface over them
(exllama_fp16 doubles the rate, one or two speculative tokens add a third, the packed recurrent decode adds 5 percent
only with no speculation in flight, eager mode costs 5 percent, prefix caching and buffer ops change nothing). A spawn
lands in a slow band (SLOW_FACTOR) with probability SLOW_SHARE unless GPU_MAX_HW_QUEUES is 1, which is how the old
chip's bimodal spawns behaved and why the R35 prior P3 exists. `serve` runs lab/bench_fake.py's server at that rate,
so lab/bench.py measures it end to end; watts come from bench_fake's power model, so joules per token falls as the
rate rises. The rates are scaled up (BASE_TOK_S) so a test finishes quickly; they are the fake's, never a measurement.

The trainer reads --settings (the JSON file the training runner writes: every R36 parameter, the derived
gradient_accumulation_steps and the pinned block) and its environment (TORCH_BLAS_PREFER_HIPBLASLT,
PYTORCH_TUNABLEOP_ENABLED, TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL), and follows the trainer contract the training
runner expects of lab/train.py later: one JSON line per optimizer step appended to <out>/steps.jsonl
({"step", "samples_per_s", "loss", "mem_gb", "t"}, flushed as it happens, so the runner can stop a hopeless run after
a few steps) and <out>/summary.json at the end ({"steps", "samples_per_s", "peak_mem_gb", "first_step_s",
"last_loss", "settings", "fake"}). Peak memory over MEM_LIMIT_GB is an out-of-memory exit (code 3, a HIP OOM line on
stderr). --time-scale shrinks every sleep. The noise is seeded from --noise-seed and the trial, so a run is
reproducible. Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if not __package__:  # run by path (the optimizer spawns it so): the repo root, one folder up, holds lab/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab import bench_fake  # noqa: E402

SCRIPT = Path(__file__).resolve()
SETTINGS_MARK = "--engine-args"   # everything after it is a trial's engine args
OOM_EXIT = 3

# inference
BASE_TOK_S = 300.0
SLOW_SHARE = 0.25
SLOW_FACTOR = 0.75
RATE_NOISE = 0.02
EXLLAMA_FACTOR = 2.0
SPEC_FACTOR = {0: 1.0, 1: 1.33, 2: 1.38, 3: 1.30}
PACKED_FACTOR = 1.05
EAGER_FACTOR = 0.95
SKINNY_FACTOR = 1.03
TRITON_ATTN_FACTOR = 0.98
FP16_SSM_FACTOR = 1.02
BASE_TTFT_S = 0.01
SPEC_TTFT_S = 0.004

# training
BASE_SAMPLES_S = 6.0
BATCH_FACTOR = {1: 0.45, 2: 0.75, 4: 1.0, 8: 1.18, 16: 1.25}
CHECKPOINTING_SPEED = 0.78
CHECKPOINTING_MEMORY = 0.45
FLA_FACTOR = 1.6
HIPBLASLT_FACTOR = 1.15
TUNABLEOP_FACTOR = 1.05
AOTRITON_FACTOR = 1.04
CONV1D_FACTOR = 1.03
COMPILE_FACTOR = 1.10
GROUP_FACTOR = 1.02
BASE_MEM_GB = 9.0
MEM_PER_SAMPLE_GB = 2.5
MEM_LIMIT_GB = 24.0
SAMPLES_NOISE = 0.03
WARMUP_STEPS = 2
WARMUP_FACTOR = 0.4
COMPILE_FIRST_STEP_S = 30.0


# --- reading settings back ------------------------------------------------------------------------------------------

def parse_args_list(tokens: Sequence[str]) -> dict[str, Any]:
    """Engine args as vLLM's parser sees them: {"--name": "value"}, a flag with no value is True."""
    out: dict[str, Any] = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("--"):
            raise ValueError(f"expected a flag, got {token!r}")
        name, eq, value = token.partition("=")
        if eq:
            out[name] = value
            i += 1
        elif i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            out[name] = tokens[i + 1]
            i += 2
        else:
            out[name] = True
            i += 1
    return out


def split_settings(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    """(the fake's own arguments, the trial's engine args after SETTINGS_MARK)."""
    argv = list(argv)
    if SETTINGS_MARK in argv:
        i = argv.index(SETTINGS_MARK)
        return argv[:i], argv[i + 1:]
    return argv, []


def _on(value: Any, default: bool) -> bool:
    if value is None:
        return default
    return value is True or str(value).strip().lower() in ("1", "true", "yes", "on")


def _json(value: Any) -> dict[str, Any]:
    try:
        out = json.loads(value) if isinstance(value, str) else {}
    except ValueError:
        return {}
    return out if isinstance(out, dict) else {}


def engine_settings(args: Mapping[str, Any], environ: Mapping[str, str]) -> dict[str, Any]:
    """The R35 parameter values a vLLM spawn would run with, read from its engine args and its environment."""
    disabled = str(environ.get("VLLM_DISABLED_KERNELS", ""))
    spec = _json(args.get("--speculative-config"))
    if args.get("--enforce-eager"):
        graphs = "NONE"
    else:
        graphs = _json(args.get("--compilation-config")).get("cudagraph_mode", "FULL_AND_PIECEWISE")
    return {
        "linear_kernel": "exllama_fp16" if (args.get("--dtype") == "float16" and "ConchLinearKernel" in disabled)
        else "conch_bf16",
        "num_speculative_tokens": int(spec.get("num_speculative_tokens", 0) or 0),
        "gpu_max_hw_queues": int(environ.get("GPU_MAX_HW_QUEUES", 4)),
        "cudagraph_mode": graphs,
        "mamba_ssm_cache_dtype": str(args.get("--mamba-ssm-cache-dtype", "float32")),
        "vllm_rocm_use_skinny_gemm": _on(environ.get("VLLM_ROCM_USE_SKINNY_GEMM"), True),
        "fla_packed_recurrent_decode": _on(environ.get("VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE"), True),
        "attention_backend": str(args.get("--attention-backend", "ROCM_ATTN")),
        "enable_prefix_caching": not bool(args.get("--no-enable-prefix-caching")),
        "amdgcn_use_buffer_ops": _on(environ.get("AMDGCN_USE_BUFFER_OPS"), False),
    }


def _rng(seed: int, trial: int, salt: str) -> random.Random:
    return random.Random(f"{salt}:{seed}:{trial}")


# --- the response surfaces ------------------------------------------------------------------------------------------

def decode_rate(s: Mapping[str, Any]) -> float:
    """Single-stream decode tok/s of a fake spawn with these R35 settings, before its band and noise."""
    rate = BASE_TOK_S
    rate *= EXLLAMA_FACTOR if s.get("linear_kernel") == "exllama_fp16" else 1.0
    spec = int(s.get("num_speculative_tokens", 0))
    rate *= SPEC_FACTOR.get(spec, 1.0)
    rate *= PACKED_FACTOR if (spec == 0 and s.get("fla_packed_recurrent_decode", True)) else 1.0
    rate *= EAGER_FACTOR if s.get("cudagraph_mode") == "NONE" else 1.0
    rate *= SKINNY_FACTOR if s.get("vllm_rocm_use_skinny_gemm", True) else 1.0
    rate *= TRITON_ATTN_FACTOR if s.get("attention_backend") == "TRITON_ATTN" else 1.0
    rate *= FP16_SSM_FACTOR if s.get("mamba_ssm_cache_dtype") == "float16" else 1.0
    return rate


def spawn_factor(s: Mapping[str, Any], seed: int, trial: int) -> tuple[float, str]:
    """This spawn's speed factor and band; GPU_MAX_HW_QUEUES=1 never lands slow."""
    rng = _rng(seed, trial, "spawn")
    slow = rng.random() < SLOW_SHARE and int(s.get("gpu_max_hw_queues", 4)) != 1
    noise = max(0.5, rng.gauss(1.0, RATE_NOISE))
    return (SLOW_FACTOR if slow else 1.0) * noise, "slow" if slow else "fast"


def train_rate(s: Mapping[str, Any], environ: Mapping[str, str]) -> tuple[float, float]:
    """(samples/s after warm-up, peak memory in GB) of the fake trainer with these R36 settings."""
    batch = int(s.get("per_device_train_batch_size", 4))
    ckpt = bool(s.get("gradient_checkpointing", True))
    peak = BASE_MEM_GB + MEM_PER_SAMPLE_GB * batch * (CHECKPOINTING_MEMORY if ckpt else 1.0)
    rate = BASE_SAMPLES_S * BATCH_FACTOR.get(batch, 0.4) * (CHECKPOINTING_SPEED if ckpt else 1.0)
    rate *= FLA_FACTOR if s.get("gdn_kernel") == "fla" else 1.0
    rate *= HIPBLASLT_FACTOR if environ.get("TORCH_BLAS_PREFER_HIPBLASLT") == "1" else 1.0
    rate *= TUNABLEOP_FACTOR if environ.get("PYTORCH_TUNABLEOP_ENABLED") == "1" else 1.0
    rate *= AOTRITON_FACTOR if environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL") == "1" else 1.0
    rate *= CONV1D_FACTOR if s.get("causal_conv1d_kernel") == "causal_conv1d" else 1.0
    rate *= COMPILE_FACTOR if (s.get("torch_compile") and s.get("gdn_kernel") == "fla") else 1.0
    rate *= GROUP_FACTOR if s.get("group_by_length") else 1.0
    return rate, peak


# --- the processes --------------------------------------------------------------------------------------------------

def serve(argv: Sequence[str]) -> int:
    """Run lab/bench_fake.py's server at the decode rate this spawn's engine args and environment give."""
    own_argv, engine = split_settings(argv)
    own = argparse.ArgumentParser(prog="lab/optimize_fake.py serve", add_help=False)
    own.add_argument("--noise-seed", type=int, default=0)
    own.add_argument("--trial", type=int, default=0)
    mine, rest = own.parse_known_args(own_argv)
    args = bench_fake.build_parser().parse_args(["serve", *rest])
    settings = engine_settings(parse_args_list(engine), os.environ)
    factor, band = spawn_factor(settings, mine.noise_seed, mine.trial)
    args.stream_tok_s = decode_rate(settings) * factor
    args.ttft_s = BASE_TTFT_S + SPEC_TTFT_S * settings["num_speculative_tokens"]
    args.prefix_caching = "on" if settings["enable_prefix_caching"] else "off"
    print(json.dumps({"fake_settings": settings, "band": band, "stream_tok_s": round(args.stream_tok_s, 3)}),
          file=sys.stderr, flush=True)
    return bench_fake.serve(args)


def train(argv: Sequence[str]) -> int:
    """Train nothing, at the rate the settings give: steps.jsonl as it goes, summary.json at the end."""
    p = argparse.ArgumentParser(prog="lab/optimize_fake.py train")
    p.add_argument("--settings", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--noise-seed", type=int, default=0)
    p.add_argument("--trial", type=int, default=0)
    p.add_argument("--time-scale", type=float, default=0.001)
    args = p.parse_args(argv)
    settings = json.loads(Path(args.settings).read_text(encoding="utf-8"))
    rate, peak = train_rate(settings, os.environ)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if peak > MEM_LIMIT_GB:
        print(f"torch.OutOfMemoryError: HIP out of memory: peak {peak:.1f} GB over the {MEM_LIMIT_GB} GB the fake "
              "pod has", file=sys.stderr, flush=True)
        return OOM_EXIT
    rng = _rng(args.noise_seed, args.trial, "train")
    effective = int(settings.get("per_device_train_batch_size", 4)) * int(settings.get("gradient_accumulation_steps", 1))
    first_step_s = None
    loss = 0.0
    t = 0.0
    with open(out / "steps.jsonl", "a", encoding="utf-8") as fh:
        for step in range(1, args.steps + 1):
            r = rate * max(0.5, rng.gauss(1.0, SAMPLES_NOISE)) * (WARMUP_FACTOR if step <= WARMUP_STEPS else 1.0)
            seconds = effective / r
            if step == 1 and settings.get("torch_compile"):
                seconds += COMPILE_FIRST_STEP_S
                r = effective / seconds
            first_step_s = seconds if step == 1 else first_step_s
            t += seconds
            time.sleep(seconds * args.time_scale)
            loss = round(1.1 + 1.2 * math.exp(-step / 6.0) + rng.gauss(0.0, 0.01), 4)
            fh.write(json.dumps({"step": step, "samples_per_s": round(r, 4), "loss": loss,
                                 "mem_gb": round(peak * (0.9 + 0.1 * min(step, 3) / 3), 3), "t": round(t, 3)}) + "\n")
            fh.flush()
    summary = {"steps": args.steps, "samples_per_s": round(rate, 4), "peak_mem_gb": round(peak, 3),
               "first_step_s": round(first_step_s or 0.0, 3), "last_loss": loss, "settings": settings, "fake": True}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


def serve_command(root: str | Path, *, noise_seed: int = 0, trial: int = 0) -> list[str]:
    """The launch template lab/bench.py spawns for the fake ({port} {port_file} {log} are bench's to fill); a trial's
    engine args are appended after SETTINGS_MARK."""
    return [bench_fake.spawn_python(), str(SCRIPT), "serve", "--port", "{port}", "--port-file", "{port_file}",
            "--log", "{log}", "--root", str(root), "--noise-seed", str(noise_seed), "--trial", str(trial),
            SETTINGS_MARK]


def train_command(*, noise_seed: int = 0, time_scale: float = 0.001) -> list[str]:
    """The trainer template for the fake; {settings} {out} {steps} {trial} are the training runner's to fill."""
    return [sys.executable, str(SCRIPT), "train", "--settings", "{settings}", "--out", "{out}", "--steps", "{steps}",
            "--noise-seed", str(noise_seed), "--trial", "{trial}", "--time-scale", str(time_scale)]


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("serve", "train"):
        print("usage: lab/optimize_fake.py serve|train ...", file=sys.stderr)
        return 2
    return serve(argv[1:]) if argv[0] == "serve" else train(argv[1:])


if __name__ == "__main__":
    sys.exit(main())
