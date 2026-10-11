"""Stand-ins for the settings optimizer's two trial runners: a settings-aware fake model server and a fake trainer.

    python lab/optimize_fake.py serve <lab/bench_fake.py serve flags> --engine-args [engine flags]
    python lab/optimize_fake.py train --steps 20 --out DIR --engine-args [trainer flags]

    from lab.optimize_fake import serve_command, train_command, SurfaceRunner

Nothing here touches a GPU or the network beyond 127.0.0.1. Each fake answers from a fixed response surface over the
settings it was started with, read the way the real process reads them: environment variables from its own
environment, engine or trainer flags from its own command line (everything after --engine-args, which is where
lab/optimize_settings.py appends a trial's flags, so a flag can never be mistaken for one of the fake's own). The
numbers are the fake's, never a measurement of this machine: tests assert what the driver does with them (the median
of the repeats, the files, the resume), not the numbers themselves.

The inference surface (decode_rate) is over the knobs of the sample search space
(lab/refine/research/kernels/data/searchspace-sample.json): --max-num-seqs peaks at 24, GPU_MAX_HW_QUEUES is best at 2,
--kv-cache-dtype fp8 is 8 % faster than auto, and --gpu-memory-utilization changes nothing (an irrelevant knob the
optimizer has to learn to ignore). A spawn lands in a slow band (SLOW_FACTOR of the rate) with probability SLOW_SHARE,
as the same server spawn decoded at two different speeds on the old chip; that is why a trial reports the median of
its repeats. `serve` turns the rate into lab/bench_fake.py's --stream-tok-s and runs that fake server, so
lab/bench.py measures it end to end: a real spawn, a real token stream, watts from bench_fake's power model.

The training surface (train_numbers) is over the knobs of the sample training space: --per-device-train-batch-size
(1, 2, 4, 8), --gradient-checkpointing (slower, much less memory), PYTORCH_TUNABLEOP_ENABLED (10 % faster) and
--dataloader-num-workers (best at 4). Peak memory above MEM_LIMIT_GB is an out-of-memory exit, as the real trainer
dies, so a search space has configurations that fail. The loss curve does not depend on the settings (loss parity).
`train` writes <out>/summary.json in the shape lab/smoke_lora.py writes (steps, examples_seen, elapsed_s,
samples_per_s, first_loss, last_loss, train_loss, peak_mem_gb) plus the settings it read and "fake": true.

The noise is seeded from --noise-seed, the trial and the repeat, so a run is reproducible. SurfaceRunner answers
from the same surfaces in-process, without a spawn, for tests that drive many trials quickly. Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if not __package__:  # run by path (the optimizer spawns it so): the repo root, one folder up, holds lab/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab import bench_fake  # noqa: E402

SCRIPT = Path(__file__).resolve()
SETTINGS_MARK = "--engine-args"   # everything after it is a trial's engine or trainer flags

# inference
BASE_TOK_S = 200.0
SLOW_SHARE = 0.2
SLOW_FACTOR = 0.8
RATE_NOISE = 0.03
BEST_SEQS = 24
QUEUE_FACTOR = {1: 0.85, 2: 1.0, 4: 0.95, 8: 0.88}
FP8_FACTOR = 1.08
DEFAULT_SEQS = 16
DEFAULT_QUEUES = 4

# training
BASE_SAMPLES_S = 6.0
BATCH_FACTOR = {1: 0.45, 2: 0.75, 4: 1.0, 8: 1.18}
CHECKPOINTING_SPEED = 0.78
CHECKPOINTING_MEMORY = 0.45
TUNABLEOP_FACTOR = 1.10
BEST_WORKERS = 4
BASE_MEM_GB = 9.0
MEM_PER_SAMPLE_GB = 2.5
MEM_LIMIT_GB = 24.0
SAMPLES_NOISE = 0.03
DEFAULT_BATCH = 4
DEFAULT_WORKERS = 2


# --- reading settings back --------------------------------------------------------------------------------------

def parse_flags(tokens: Sequence[str]) -> dict[str, Any]:
    """Flags as a process sees them: {"--name": "value"}, a flag with no value is True, --name=value is split."""
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
    """(the fake's own arguments, the trial's flags after SETTINGS_MARK)."""
    argv = list(argv)
    if SETTINGS_MARK in argv:
        i = argv.index(SETTINGS_MARK)
        return argv[:i], argv[i + 1:]
    return argv, []


def _number(value: Any, default: float) -> float:
    if value is None or value is True or value is False:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _truthy(value: Any) -> bool:
    return value is True or str(value).strip().lower() in ("1", "true", "yes", "on")


def _rng(seed: int, trial: int, repeat: int, salt: str) -> random.Random:
    return random.Random(f"{salt}:{seed}:{trial}:{repeat}")


# --- the response surfaces --------------------------------------------------------------------------------------

def decode_rate(settings: Mapping[str, Any]) -> float:
    """Single-stream decode tok/s of a fake spawn with these settings, before its band and noise."""
    seqs = _number(settings.get("--max-num-seqs"), DEFAULT_SEQS)
    queues = int(_number(settings.get("GPU_MAX_HW_QUEUES"), DEFAULT_QUEUES))
    kv = str(settings.get("--kv-cache-dtype", "auto"))
    rate = BASE_TOK_S * (1.0 - 0.5 * ((seqs - BEST_SEQS) / 40.0) ** 2)
    rate *= QUEUE_FACTOR.get(queues, 0.8) * (FP8_FACTOR if kv == "fp8" else 1.0)
    return max(rate, 1.0)


def spawn_factor(seed: int, trial: int, repeat: int) -> tuple[float, str]:
    """This spawn's speed factor and band: slow with probability SLOW_SHARE, then a few percent of noise."""
    rng = _rng(seed, trial, repeat, "spawn")
    slow = rng.random() < SLOW_SHARE
    noise = max(0.5, rng.gauss(1.0, RATE_NOISE))
    return (SLOW_FACTOR if slow else 1.0) * noise, "slow" if slow else "fast"


def train_numbers(settings: Mapping[str, Any], *, steps: int, seed: int = 0, trial: int = 0,
                  repeat: int = 0) -> dict[str, Any]:
    """What the fake trainer reports for these settings, or {"oom": True, ...} when it runs out of memory."""
    batch = int(_number(settings.get("--per-device-train-batch-size"), DEFAULT_BATCH))
    checkpointing = _truthy(settings.get("--gradient-checkpointing", False))
    tunable = _truthy(settings.get("PYTORCH_TUNABLEOP_ENABLED", "0"))
    workers = _number(settings.get("--dataloader-num-workers"), DEFAULT_WORKERS)
    peak = BASE_MEM_GB + MEM_PER_SAMPLE_GB * batch * (CHECKPOINTING_MEMORY if checkpointing else 1.0)
    if peak > MEM_LIMIT_GB:
        return {"oom": True, "peak_mem_gb": round(peak, 2), "limit_gb": MEM_LIMIT_GB}
    rng = _rng(seed, trial, repeat, "train")
    rate = BASE_SAMPLES_S * BATCH_FACTOR.get(batch, 0.4) * (CHECKPOINTING_SPEED if checkpointing else 1.0)
    rate *= (TUNABLEOP_FACTOR if tunable else 1.0) * max(0.5, 1.0 - 0.02 * (workers - BEST_WORKERS) ** 2)
    rate *= max(0.5, rng.gauss(1.0, SAMPLES_NOISE))
    losses = [round(1.1 + 1.2 * math.exp(-i / 6.0) + rng.gauss(0.0, 0.01), 4) for i in range(steps)]
    seen = steps * batch
    return {"oom": False, "steps": steps, "examples_seen": seen, "elapsed_s": round(seen / rate, 3),
            "samples_per_s": round(rate, 4), "first_loss": losses[0] if losses else None,
            "last_loss": losses[-1] if losses else None,
            "train_loss": round(sum(losses) / len(losses), 4) if losses else None, "peak_mem_gb": round(peak, 2),
            "losses": losses}


# --- the processes ----------------------------------------------------------------------------------------------

def _settings_from(environ: Mapping[str, str], flags: Sequence[str]) -> dict[str, Any]:
    return {**{k: v for k, v in environ.items() if k in ("GPU_MAX_HW_QUEUES", "PYTORCH_TUNABLEOP_ENABLED")},
            **parse_flags(flags)}


def serve(argv: Sequence[str]) -> int:
    """Run lab/bench_fake.py's server at the decode rate this spawn's settings give."""
    own_argv, flags = split_settings(argv)
    own = argparse.ArgumentParser(prog="lab/optimize_fake.py serve", add_help=False)
    own.add_argument("--noise-seed", type=int, default=0)
    own.add_argument("--trial", type=int, default=0)
    own.add_argument("--repeat", type=int, default=0)
    mine, rest = own.parse_known_args(own_argv)
    args = bench_fake.build_parser().parse_args(["serve", *rest])
    settings = _settings_from(os.environ, flags)
    factor, band = spawn_factor(mine.noise_seed, mine.trial, mine.repeat)
    args.stream_tok_s = decode_rate(settings) * factor
    print(json.dumps({"fake_settings": settings, "band": band, "stream_tok_s": round(args.stream_tok_s, 3)}),
          file=sys.stderr, flush=True)
    return bench_fake.serve(args)


def train(argv: Sequence[str]) -> int:
    """Write the fake trainer's summary.json for these settings, or exit 1 the way an out-of-memory run does."""
    own_argv, flags = split_settings(argv)
    p = argparse.ArgumentParser(prog="lab/optimize_fake.py train")
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--noise-seed", type=int, default=0)
    p.add_argument("--trial", type=int, default=0)
    p.add_argument("--repeat", type=int, default=0)
    args = p.parse_args(own_argv)
    settings = _settings_from(os.environ, flags)
    numbers = train_numbers(settings, steps=args.steps, seed=args.noise_seed, trial=args.trial, repeat=args.repeat)
    if numbers.pop("oom"):
        print(f"torch.OutOfMemoryError: HIP out of memory: peak {numbers['peak_mem_gb']} GB over the "
              f"{numbers['limit_gb']} GB the fake pod has", file=sys.stderr, flush=True)
        return 1
    summary = {**numbers, "settings": settings, "fake": True}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("FAKE_TRAIN " + json.dumps({k: v for k, v in summary.items() if k != "losses"}), flush=True)
    return 0


def serve_command(root: str | Path, *, noise_seed: int = 0) -> list[str]:
    """The launch template lab/bench.py spawns for the fake: {port} {port_file} {log} are bench's to fill, {trial}
    and {repeat} the optimizer's; a trial's flags are appended after SETTINGS_MARK."""
    return [bench_fake.spawn_python(), str(SCRIPT), "serve", "--port", "{port}", "--port-file", "{port_file}",
            "--log", "{log}", "--root", str(root), "--noise-seed", str(noise_seed), "--trial", "{trial}",
            "--repeat", "{repeat}", SETTINGS_MARK]


def train_command(*, noise_seed: int = 0) -> list[str]:
    """The trainer template for the fake: {steps} {out} {trial} {repeat} are the optimizer's to fill."""
    return [sys.executable, str(SCRIPT), "train", "--steps", "{steps}", "--out", "{out}", "--noise-seed",
            str(noise_seed), "--trial", "{trial}", "--repeat", "{repeat}", SETTINGS_MARK]


# --- in-process --------------------------------------------------------------------------------------------------

class SurfaceRunner:
    """A trial runner that answers from the response surfaces without spawning anything (see the module docstring).

    kind "inference" reports decode_tok_s, decode_tok_s_total and joules_per_token per batch size (watts from
    lab/bench_fake.py's power model); kind "training" reports what train_numbers gives, and fails a repeat that runs
    out of memory. `calls` lists every (trial, repeat) it ran, so a test can tell a resumed run re-ran nothing."""

    fake = True

    def __init__(self, kind: str, *, noise_seed: int = 0, batches: Sequence[int] = (1,), steps: int = 20) -> None:
        if kind not in ("inference", "training"):
            raise ValueError(f"kind is inference or training, not {kind!r}")
        self.kind = kind
        self.noise_seed = noise_seed
        self.batches = list(batches)
        self.steps = steps
        self.calls: list[tuple[int, int]] = []

    def describe(self) -> dict[str, Any]:
        return {"kind": self.kind, "fake": True, "runner": "lab/optimize_fake.py SurfaceRunner",
                "noise_seed": self.noise_seed, "batches": self.batches, "steps": self.steps}

    def run_once(self, applied: Any, *, trial: int, repeat: int, workdir: Path) -> Any:
        from lab.optimize_settings import RepeatResult

        self.calls.append((trial, repeat))
        settings = {**applied.env, **parse_flags(applied.flags)}
        conditions = {"settings": settings, "runner": "SurfaceRunner"}
        if self.kind == "training":
            numbers = train_numbers(settings, steps=self.steps, seed=self.noise_seed, trial=trial, repeat=repeat)
            if numbers.pop("oom"):
                return RepeatResult("fail", {}, conditions, f"out of memory: peak {numbers['peak_mem_gb']} GB")
            metrics = {k: float(v) for k, v in numbers.items() if isinstance(v, (int, float)) and k != "steps"}
            return RepeatResult("ok", metrics, conditions)
        factor, band = spawn_factor(self.noise_seed, trial, repeat)
        rate = decode_rate(settings) * factor
        metrics: dict[str, float] = {}
        for b in self.batches:
            watts = bench_fake.IDLE_NO_SERVER_W + bench_fake.WATTS_PER_STREAM * b
            metrics[f"decode_tok_s.b{b}"] = rate
            metrics[f"decode_tok_s_total.b{b}"] = rate * b
            metrics[f"joules_per_token.b{b}"] = watts / (rate * b)
        return RepeatResult("ok", metrics, {**conditions, "band": band})


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("serve", "train"):
        print("usage: lab/optimize_fake.py serve|train ...", file=sys.stderr)
        return 2
    return serve(argv[1:]) if argv[0] == "serve" else train(argv[1:])


if __name__ == "__main__":
    sys.exit(main())
