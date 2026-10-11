"""interference: what a teacher-style decode load and a LoRA run cost each other on one Strix Halo chip.

    python -m lab interference --dry-run                     # the plan and the slot minutes, nothing written
    python -m lab interference pod-script [--out FILE]       # this machine: the one script the pod runs
    python tooling/dispatcher/pod_exec.py D run <script> r43_pod_run.sh 2700 > <output file>   # the pod run, in a slot
    python -m lab interference collect <output file> [--out DIR] [--no-ledger]       # the data file and ledger rows
    python -m lab interference pod-run --expect-user-sha256 HEX [...]                # on the pod: what the script runs

The question. The placement plan (lab/placement.py) says a 27B INT4 teacher and a 4B LoRA job fit the 64 GB of one
pod together on paper once the teacher is served at a smaller memory share. Whether they should share depends on what
each slows the other: decode is memory-bound and a LoRA step is compute-bound, but both run on one GPU and one memory
system. This measures it once, on pod D (account D, node pod-b), inside one GPU Queue slot, in three arms on the same
chip and the same server:

  decode   the teacher alone: lab/bench.py's fixed prompt set (fixtures/lab/bench/prompts.json) at a fixed batch
           (default 4 streams at once, every stream on its own prompt, greedy, max_tokens with ignore_eos), round
           after round for --decode-s seconds;
  train    the LoRA alone, with the teacher loaded and idle: lab/train.py for --steps optimizer steps (default 20)
           on the tool-calling rows, a fresh copy per arm (the trainer deletes its copy of the data when it ends);
  both     the same LoRA run with the same decode rounds sent for as long as it trains.

Each arm records decode tokens per second (all streams' tokens over the window), answers per second, the trainer's
samples per second (steps x batch x gradient accumulation over the window of its steps, read from the trainer's own
telemetry.jsonl) and its steps per second, the watts over the window (amd-smi, read in this process by lab/bench.py's
LocalPower, the reader lab/profile_decode_pod.py uses) and the joules: per answer for the decode arm, per step for the
train arm. In the both arm the window is the trainer's steps; only decode rounds wholly inside it count, the energy
is the window's, and the comparison is against doing the same work one after the other: the decode arm's joules per
answer times the answers plus the train arm's joules per step times the steps, and the same for time.

The server. The run starts its own teacher (lab/bench.py's vLLM serve line, lab/pod_a_teacher.sh's flags) on port
8100 with --gpu-memory-utilization at --share (default and ceiling MAX_SHARE): at the teacher's usual 0.90 it reserves
57.6 of the 58 GiB the pod has free and no LoRA job fits beside it. It refuses to run where it is not pod D's Strix
Halo, where the port is taken or where a vLLM server already runs (stop the pod's own teacher first: this run stops
nothing it did not start), and it stops its server however it ends.

The script is about 260 KB (the trainer, the bench and their imports, base64), so pod_exec.py's run mode, which
uploads it through the Jupyter contents API and types only its bash line, is the way to send it; term mode types it
all. 2700 s is the 30-minute slot plus its one 15-minute extension.

What comes back. The pod prints one marked base64 block (gzip'd tar of raw.json, result.json and the pod's summary);
`collect` reads it from the saved terminal output, checks its sha256, and writes
lab/refine/research/kernels/data/interference-gfx1151-<date>.json and its .md table, and the ledger rows (verbs
interference.arm and interference.summary, node pod-b, task R43, low confidence: one run per arm). The placement plan
reads the newest such file for its "teacher + lora" sharing line.
"""
from __future__ import annotations

import argparse
import ast
import base64
import datetime as dt
import gzip
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

if not __package__:  # run by path: the repo root, one folder up, holds lab/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SCHEMA = "lab.interference/1"
RAW_SCHEMA = "lab.interference.raw/1"
TASK = "R43"
NODE = "pod-b"                     # pod D fills pod B's role
CHIP = "gfx1151"
ARMS = ("decode", "train", "both")
REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "lab" / "refine" / "research" / "kernels" / "data"
RESULT_PREFIX = f"interference-{CHIP}"
STATE_DIR = REPO / "lab" / "state" / "interference"
POD_SCRIPT = STATE_DIR / "pod_run.sh"
SCRIPT_ROOT = "r43_interference"
DEFAULT_ACCOUNT = "D"
DEFAULT_PORT = 8100                # not the teacher's 8000
DEFAULT_CHECKPOINT = "~/models/Qwen3.8-27B-INT4"
DEFAULT_MODEL = "~/models/Qwen3.5-4B"
DEFAULT_DATASET = "fixtures/learning/train-tiny/tool-calling-tiny"
DEFAULT_ROWS = "fixtures/learning/train-tiny/rows.jsonl"
DEFAULT_OUT = f"~/{SCRIPT_ROOT}"
DEFAULT_BATCH = 4
DEFAULT_MAX_TOKENS = 128
DEFAULT_DECODE_S = 120.0
DEFAULT_STEPS = 20
DEFAULT_TRAIN_BATCH = 4
DEFAULT_GRAD_ACCUM = 1             # 20 steps of 4 examples: 80 examples, not L16's 320
MAX_SHARE = 0.63                   # the largest teacher share that leaves room for the LoRA job (see SLOT_FACTS)
DEFAULT_SHARE = MAX_SHARE
MAX_LEN = 16384
MODEL_NAME = "teacher"
WARMUP_TOKENS = 16
POWER_PERIOD_S = 0.5
READY_TIMEOUT_S = 1500.0
TRAIN_TIMEOUT_S = 1500.0
DEFAULT_DEADLINE_MIN = 40.0        # the slot (30) plus the one extension of 15, less the paste and the stop
MARK_BEGIN = "=== R43-RESULT BEGIN"
MARK_END = "=== R43-RESULT END ==="
B64_LINE = 76
BOX_ONLY = ("lab.placement", "lab.kernels")

#: The facts the slot estimate and the share ceiling rest on, with where each was read.
SLOT_FACTS: dict[str, dict[str, Any]] = {
    "engine_cold_s": {"value": 523.0, "source": "lab/README.md: pod D's engine ready 523 s after a cold start"},
    "engine_warm_s": {"value": 297.0, "source": "lab/refine/metrics.jsonl teacher.engine_init_s on pod-c (R29)"},
    "train_examples_per_s": {"value": 0.32, "source": "L16 on pod D, 2026-10-11 (dispatcher's brief): "
                                                      "0.32 examples per second"},
    "train_peak_gib": {"value": 11.1, "source": "L16 on pod D, 2026-10-11 (dispatcher's brief): peak 11.1 GiB of 64"},
    "teacher_hold_gib": {"value": 57.6, "source": "the teacher at --gpu-memory-utilization 0.90 holds 57.6 of the "
                                                  "pod's 58 GiB free (dispatcher's brief, 2026-10-11)"},
    "train_load_s": {"value": 120.0, "source": "estimate: the trainer's imports, the 4B model's load and the "
                                               "requirements check before its first step"},
}


class InterferenceError(Exception):
    """An input or a result that cannot be used: bad arguments, a broken bundle, a result that fails its checks."""


def _r(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _div(a: float | None, b: float | None) -> float | None:
    return a / b if a is not None and b not in (None, 0) else None


def _pct(new: float | None, old: float | None) -> float | None:
    """new against old, in percent of old (negative: new is smaller)."""
    return _r((new - old) / old * 100.0, 2) if new is not None and old not in (None, 0) else None


# --- the plan and the slot ----------------------------------------------------------------------------------------

def slot_estimate(*, decode_s: float = DEFAULT_DECODE_S, steps: int = DEFAULT_STEPS,
                  train_batch: int = DEFAULT_TRAIN_BATCH, grad_accum: int = DEFAULT_GRAD_ACCUM,
                  cold: bool = True) -> dict[str, Any]:
    """The slot's minutes, part by part, from SLOT_FACTS. The both arm is costed as the train arm with its steps
    taking half as long again; a slower shared run uses the one 15-minute extension."""
    engine = SLOT_FACTS["engine_cold_s" if cold else "engine_warm_s"]["value"]
    rate = SLOT_FACTS["train_examples_per_s"]["value"]
    load = SLOT_FACTS["train_load_s"]["value"]
    train_s = steps * train_batch * grad_accum / rate
    parts = {"paste_and_gate_s": 60.0, "server_start_s": engine, "warmup_s": 60.0, "decode_arm_s": decode_s,
             "train_arm_s": load + train_s, "both_arm_s": load + 1.5 * train_s, "stop_and_bundle_s": 60.0}
    total = sum(parts.values())
    minutes = int(-(-total // 60))
    slot = 15 if minutes <= 15 else 30 if minutes <= 30 else 60
    return {"parts_s": {k: round(v) for k, v in parts.items()}, "total_s": round(total), "minutes": minutes,
            "slot_minutes": slot, "extension_minutes": 15, "train_examples": steps * train_batch * grad_accum,
            "facts": SLOT_FACTS}


def memory_check(share: float) -> dict[str, Any]:
    """Whether the teacher at this share leaves room for the LoRA job, from the pod's recorded facts: the teacher
    holds share x visible, where visible is what 0.90 x visible = teacher_hold_gib makes it."""
    visible = SLOT_FACTS["teacher_hold_gib"]["value"] / 0.90
    free = 58.0
    hold = share * visible
    peak = SLOT_FACTS["train_peak_gib"]["value"]
    left = free - hold - peak
    return {"share": share, "visible_gib": _r(visible, 2), "free_gib": free, "teacher_gib": _r(hold, 2),
            "lora_peak_gib": peak, "left_gib": _r(left, 2), "fits": left > 0 and share <= MAX_SHARE}


def plan(args: argparse.Namespace) -> dict[str, Any]:
    """What the pod run will do, in order, and what slot it needs; nothing is run or written."""
    slot = slot_estimate(decode_s=args.decode_s, steps=args.steps, train_batch=args.train_batch,
                         grad_accum=args.grad_accum)
    mem = memory_check(args.share)
    order = [
        f"gate: Strix Halo ({CHIP}), the account the script was made for (pod D), port {args.port} free, no vLLM "
        "server running (stop the pod's own teacher first; this run stops nothing it did not start)",
        f"start the teacher ({args.checkpoint}) on port {args.port} at --gpu-memory-utilization {args.share:g}: it "
        f"holds about {mem['teacher_gib']} GiB, leaving {mem['left_gib']} GiB beside the LoRA's "
        f"{mem['lora_peak_gib']} GiB peak",
        f"warm up: every prompt once at batch 1 and once at batch {args.batch}, {WARMUP_TOKENS} tokens",
        f"decode arm: rounds of {args.batch} streams x {args.max_tokens} tokens for {args.decode_s:g} s",
        f"train arm: lab/train.py --max-steps {args.steps} --batch {args.train_batch} --grad-accum "
        f"{args.grad_accum} on {args.model}, rows {args.rows}, with the teacher idle",
        "both arm: the same train run with decode rounds sent until it ends",
        "stop the server, print the result block",
    ]
    return {"ok": mem["fits"], "outcome": "plan", "task": TASK, "node": NODE, "account": args.account,
            "order": order, "memory": mem, "slot": slot,
            "slot_request": f'python -m lab slot request --task {TASK} --what "interference: teacher decode at batch '
                             f'{args.batch} and a {args.steps}-step LoRA, alone and together on one chip" --minutes '
                             f'{slot["slot_minutes"]}',
            **({} if mem["fits"] else {"error": f"--share {args.share:g} leaves no room for the LoRA job "
                                                f"(ceiling {MAX_SHARE})"})}


# --- the analysis: raw -> result ----------------------------------------------------------------------------------

def _t(value: Any) -> float | None:
    """Seconds since the epoch from a number or an ISO time."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str) and value:
        text = value.replace("Z", "+00:00")
        try:
            moment = dt.datetime.fromisoformat(text)
        except ValueError:
            return None
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=dt.timezone.utc)
        return moment.timestamp()
    return None


def train_window(steps: Sequence[Mapping[str, Any]]) -> tuple[float, float] | None:
    """[first step's start, last step's end] from telemetry.train.step events (ts is the step's end)."""
    ends = [(_t(s.get("ts")), _num(s.get("step_ms"))) for s in steps]
    ends = [(t, ms) for t, ms in ends if t is not None]
    if not ends:
        return None
    ends.sort()
    first_t, first_ms = ends[0]
    start = first_t - (first_ms or 0.0) / 1000.0
    return (start, ends[-1][0]) if ends[-1][0] > start else None


def decode_summary(rounds: Sequence[Mapping[str, Any]], window: tuple[float, float] | None = None) -> dict[str, Any]:
    """Tokens, answers and their rates over the rounds (only those wholly inside window, when one is given)."""
    kept = [r for r in rounds if r.get("start") is not None and r.get("end") is not None
            and (window is None or (window[0] <= r["start"] and r["end"] <= window[1]))]
    if not kept:
        return {"rounds": 0, "answers": 0, "tokens": 0, "errors": 0, "window": None, "decode_tok_s": None,
                "answers_per_s": None}
    start = min(r["start"] for r in kept)
    end = max(r["end"] for r in kept)
    busy = sum(r["end"] - r["start"] for r in kept)
    requests = [q for r in kept for q in r.get("requests", [])]
    good = [q for q in requests if not q.get("error")]
    tokens = sum(int(q.get("tokens") or 0) for q in good)
    per_request = [q["decode_tok_s"] for q in good if _num(q.get("decode_tok_s"))]
    return {"rounds": len(kept), "answers": len(good), "tokens": tokens, "errors": len(requests) - len(good),
            "window": [start, end], "busy_s": _r(busy, 3), "decode_tok_s": _r(_div(tokens, busy)),
            "answers_per_s": _r(_div(len(good), busy)),
            "decode_tok_s_per_stream_median": _r(sorted(per_request)[len(per_request) // 2]) if per_request else None}


def train_summary(train: Mapping[str, Any]) -> dict[str, Any]:
    steps = train.get("steps") or []
    window = train_window(steps)
    n = len(steps)
    per_step = int(train.get("batch") or 0) * int(train.get("grad_accum") or 0)
    seconds = window[1] - window[0] if window else None
    run = train.get("run") or {}
    return {"steps": n, "window": list(window) if window else None, "seconds": _r(seconds, 3),
            "samples": n * per_step, "samples_per_s": _r(_div(n * per_step, seconds)),
            "steps_per_s": _r(_div(n, seconds)), "outcome": run.get("outcome"), "exit_code": train.get("exit_code"),
            "trainer_samples_per_s": _num((run.get("metrics") or {}).get("trainer.samples_per_s")),
            "trainer_peak_mem_gb": _num((run.get("metrics") or {}).get("trainer.peak_mem_gb"))}


def _watts(arm: Mapping[str, Any]) -> dict[str, Any]:
    w = arm.get("watts") or {}
    return {"available": bool(w.get("available")), "avg_w": _num(w.get("avg_w")), "peak_w": _num(w.get("peak_w")),
            "energy_j": _num(w.get("energy_j")), "reason": w.get("reason")}


def build_result(raw: Mapping[str, Any]) -> dict[str, Any]:
    """The result from the pod's raw record: per arm the rates, watts and joules, and the shared-against-serial
    comparison. Pure."""
    arms_raw = raw.get("arms") or {}
    arms: dict[str, Any] = {}
    d_raw = arms_raw.get("decode") or {}
    decode = decode_summary(d_raw.get("rounds") or [])
    dw = _watts(d_raw)
    arms["decode"] = {**decode, "watts": dw, "joules_per_answer": _r(_div(dw["energy_j"], decode["answers"] or None)),
                      "error": d_raw.get("error")}
    t_raw = arms_raw.get("train") or {}
    train = train_summary(t_raw)
    tw = _watts(t_raw)
    arms["train"] = {**train, "watts": tw, "joules_per_step": _r(_div(tw["energy_j"], train["steps"] or None)),
                     "error": t_raw.get("error")}
    b_raw = arms_raw.get("both") or {}
    b_train = train_summary(b_raw)
    window = tuple(b_train["window"]) if b_train["window"] else None
    b_decode = decode_summary(b_raw.get("rounds") or [], window) if window else decode_summary([])
    bw = _watts(b_raw)
    arms["both"] = {"train": b_train, "decode": b_decode, "watts": bw,
                    "joules_per_answer_all": _r(_div(bw["energy_j"], b_decode["answers"] or None)),
                    "joules_per_step_all": _r(_div(bw["energy_j"], b_train["steps"] or None)),
                    "error": b_raw.get("error")}

    compare: dict[str, Any] = {
        "decode_tok_s_change_pct": _pct(b_decode["decode_tok_s"], decode["decode_tok_s"]),
        "answers_per_s_change_pct": _pct(b_decode["answers_per_s"], decode["answers_per_s"]),
        "samples_per_s_change_pct": _pct(b_train["samples_per_s"], train["samples_per_s"]),
    }
    jpa = arms["decode"]["joules_per_answer"]
    jps = arms["train"]["joules_per_step"]
    serial_j = (b_decode["answers"] * jpa + b_train["steps"] * jps) if jpa is not None and jps is not None else None
    serial_s = None
    if decode["answers_per_s"] and train["steps_per_s"]:
        serial_s = b_decode["answers"] / decode["answers_per_s"] + b_train["steps"] / train["steps_per_s"]
    compare.update({"serial_energy_j": _r(serial_j, 1), "shared_energy_j": _r(bw["energy_j"], 1),
                    "energy_change_pct": _pct(bw["energy_j"], serial_j),
                    "serial_seconds": _r(serial_s, 1), "shared_seconds": b_train["seconds"],
                    "time_change_pct": _pct(b_train["seconds"], serial_s)})
    compare["verdict"] = verdict(compare)

    started = str(raw.get("started") or "")
    date = started[:10] if re.match(r"\d{4}-\d{2}-\d{2}", started) else dt.date.today().isoformat()
    open_items = []
    for name in ARMS:
        if arms[name].get("error"):
            open_items.append(f"{name} arm: {arms[name]['error']}")
    if not dw["available"] or not tw["available"] or not bw["available"]:
        open_items.append("watts missing in an arm: " + "; ".join(
            f"{n}: {w['reason']}" for n, w in (("decode", dw), ("train", tw), ("both", bw)) if not w["available"]))
    if arms["both"]["decode"]["rounds"] == 0:
        open_items.append("no decode round fell wholly inside the shared train window")
    serve = raw.get("serve") or {}
    workload = raw.get("workload") or {}
    return {"schema": SCHEMA, "task": TASK, "run_id": raw.get("run_id"), "date": date, "chip": CHIP,
            "node": raw.get("node") or NODE, "hub": raw.get("hub"), "account": raw.get("account"),
            "started": raw.get("started"), "ended": raw.get("ended"),
            "serve": {"share": serve.get("share"), "port": serve.get("port"), "ready_s": serve.get("ready_s"),
                      "checkpoint": serve.get("checkpoint"), "argv": serve.get("argv")},
            "workload": workload, "power_reader": raw.get("power_reader"), "facts": raw.get("facts"),
            "arms": arms, "compare": compare, "open": open_items}


def verdict(compare: Mapping[str, Any]) -> str:
    """share: together they did the same work in less time and no more energy; serial: they did not; unknown: a
    number is missing."""
    t, e = compare.get("time_change_pct"), compare.get("energy_change_pct")
    if t is None or e is None:
        return "unknown"
    return "share" if t < 0 and e <= 0 else "serial"


def check_result(result: Any) -> list[str]:
    problems = []
    if not isinstance(result, Mapping) or result.get("schema") != SCHEMA:
        return ["not an interference result (schema)"]
    arms = result.get("arms") or {}
    for name in ARMS:
        if name not in arms:
            problems.append(f"no {name} arm")
    if (arms.get("decode") or {}).get("answers", 0) == 0:
        problems.append("the decode arm has no answer")
    if (arms.get("train") or {}).get("steps", 0) == 0:
        problems.append("the train arm has no step")
    if ((arms.get("both") or {}).get("train") or {}).get("steps", 0) == 0:
        problems.append("the both arm has no train step")
    return problems


# --- the Markdown table -------------------------------------------------------------------------------------------

def _f(value: Any, digits: int = 2, suffix: str = "") -> str:
    v = _num(value)
    return "n/a" if v is None else f"{v:.{digits}f}{suffix}"


def render_markdown(result: Mapping[str, Any], source: str | None = None) -> str:
    arms = result["arms"]
    d, t, b = arms["decode"], arms["train"], arms["both"]
    c = result["compare"]
    serve = result.get("serve") or {}
    lines = [
        f"# Interference on one {CHIP} chip: teacher decode and a LoRA run ({result.get('date')})",
        "",
        f"Run {result.get('run_id')} on {result.get('node')} (hub {result.get('hub')}, account "
        f"{result.get('account')}), teacher served at --gpu-memory-utilization {serve.get('share')}. Made by "
        "`python -m lab interference`; the JSON beside this file holds every number"
        + (f" ({source})." if source else "."),
        "",
        "| arm | decode tok/s | answers/s | samples/s | steps/s | watts avg | joules/answer | joules/step |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        f"| decode alone | {_f(d['decode_tok_s'])} | {_f(d['answers_per_s'], 3)} | - | - | "
        f"{_f(d['watts']['avg_w'], 1)} | {_f(d['joules_per_answer'], 1)} | - |",
        f"| train alone | - | - | {_f(t['samples_per_s'], 3)} | {_f(t['steps_per_s'], 4)} | "
        f"{_f(t['watts']['avg_w'], 1)} | - | {_f(t['joules_per_step'], 1)} |",
        f"| both | {_f(b['decode']['decode_tok_s'])} | {_f(b['decode']['answers_per_s'], 3)} | "
        f"{_f(b['train']['samples_per_s'], 3)} | {_f(b['train']['steps_per_s'], 4)} | {_f(b['watts']['avg_w'], 1)} | "
        f"{_f(b['joules_per_answer_all'], 1)} (all) | {_f(b['joules_per_step_all'], 1)} (all) |",
        "",
        "In the both row the joules are the whole window's, charged to all answers or to all steps.",
        "",
        "## Shared against one after the other",
        "",
        "| | value |",
        "| --- | --- |",
        f"| decode tok/s, shared against alone | {_f(c['decode_tok_s_change_pct'], 1, ' %')} |",
        f"| LoRA samples/s, shared against alone | {_f(c['samples_per_s_change_pct'], 1, ' %')} |",
        f"| time for the shared work, one after the other | {_f(c['serial_seconds'], 1, ' s')} |",
        f"| time for it shared | {_f(c['shared_seconds'], 1, ' s')} ({_f(c['time_change_pct'], 1, ' %')}) |",
        f"| energy one after the other | {_f(c['serial_energy_j'], 0, ' J')} |",
        f"| energy shared | {_f(c['shared_energy_j'], 0, ' J')} ({_f(c['energy_change_pct'], 1, ' %')}) |",
        f"| verdict | {c['verdict']} |",
        "",
        "## Open items",
        "",
    ]
    lines += [f"- {item}" for item in result.get("open") or []] or ["- none"]
    return "\n".join(lines) + "\n"


# --- ledger rows --------------------------------------------------------------------------------------------------

def ledger_rows(result: Mapping[str, Any], source: str) -> list[dict[str, Any]]:
    """One interference.arm row per arm and one interference.summary row; node pod-b, low confidence."""
    arms = result["arms"]
    base = {"chip": CHIP, "hub": result.get("hub"), "account": result.get("account"),
            "share": (result.get("serve") or {}).get("share"), "run_id": result.get("run_id"),
            "batch": (result.get("workload") or {}).get("batch"),
            "max_tokens": (result.get("workload") or {}).get("max_tokens"),
            "steps": (result.get("workload") or {}).get("steps"),
            "train_model": (result.get("workload") or {}).get("train_model")}
    d, t, b = arms["decode"], arms["train"], arms["both"]
    per_arm = {
        "decode": {"interference.decode_tok_s": d["decode_tok_s"], "interference.answers_per_s": d["answers_per_s"],
                   "interference.watts_avg": d["watts"]["avg_w"],
                   "interference.joules_per_answer": d["joules_per_answer"]},
        "train": {"interference.samples_per_s": t["samples_per_s"], "interference.steps_per_s": t["steps_per_s"],
                  "interference.watts_avg": t["watts"]["avg_w"], "interference.joules_per_step": t["joules_per_step"]},
        "both": {"interference.decode_tok_s": b["decode"]["decode_tok_s"],
                 "interference.answers_per_s": b["decode"]["answers_per_s"],
                 "interference.samples_per_s": b["train"]["samples_per_s"],
                 "interference.steps_per_s": b["train"]["steps_per_s"],
                 "interference.watts_avg": b["watts"]["avg_w"],
                 "interference.joules_per_answer_all": b["joules_per_answer_all"],
                 "interference.joules_per_step_all": b["joules_per_step_all"]},
    }
    units = {"decode_tok_s": "tok/s", "answers_per_s": "answers/s", "samples_per_s": "samples/s",
             "steps_per_s": "steps/s", "watts_avg": "W", "joules_per_answer": "J", "joules_per_step": "J",
             "joules_per_answer_all": "J", "joules_per_step_all": "J"}
    rows = []
    for name in ARMS:
        metrics = {k: v for k, v in per_arm[name].items() if _num(v) is not None}
        rows.append({"verb": "interference.arm", "outcome": "ok" if metrics and not arms[name].get("error") else "fail",
                     "metrics": metrics, "units": {k: units[k.split(".", 1)[1]] for k in metrics} or None,
                     "conditions": {**base, "arm": name}, "note": arms[name].get("error")})
    c = result["compare"]
    metrics = {f"interference.{k}": c[k] for k in ("decode_tok_s_change_pct", "samples_per_s_change_pct",
                                                   "time_change_pct", "energy_change_pct") if _num(c.get(k)) is not None}
    rows.append({"verb": "interference.summary", "outcome": "ok" if c["verdict"] != "unknown" else "fail",
                 "metrics": metrics, "units": {k: "%" for k in metrics} or None,
                 "conditions": {**base, "verdict": c["verdict"]},
                 "note": "; ".join(result.get("open") or [])[:480] or None})
    for row in rows:
        row.update({"node": result.get("node") or NODE, "source": source, "low_confidence": True,
                    "refs": {"run_id": result.get("run_id")}})
    return rows


# --- files --------------------------------------------------------------------------------------------------------

def result_paths(result: Mapping[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    base = Path(out_dir) / f"{RESULT_PREFIX}-{result['date']}"
    return base.with_suffix(".json"), base.with_suffix(".md")


def write_outputs(result: Mapping[str, Any], out_dir: str | Path, *, force: bool = False) -> tuple[Path, Path]:
    json_path, md_path = result_paths(result, out_dir)
    if not force and (json_path.exists() or md_path.exists()):
        raise InterferenceError(f"{json_path.name} already exists; pass --force to replace it")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    md_path.write_text(render_markdown(result, json_path.name), encoding="utf-8", newline="\n")
    return json_path, md_path


def newest_result(folder: Path = DATA_DIR) -> Path | None:
    """The newest interference-gfx1151-<date>.json in folder, or None."""
    found = sorted(p for p in folder.glob(f"{RESULT_PREFIX}-*.json")
                   if re.fullmatch(rf"{RESULT_PREFIX}-\d{{4}}-\d{{2}}-\d{{2}}\.json", p.name))
    return found[-1] if found else None


# --- the bundle that crosses the terminal -------------------------------------------------------------------------

def make_bundle(files: Mapping[str, bytes]) -> bytes:
    """A gzip'd tar of name -> bytes, the same bytes for the same files."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name in sorted(files):
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(files[name]), 0, 0o644
            tar.addfile(info, io.BytesIO(files[name]))
    return gzip.compress(raw.getvalue(), compresslevel=9, mtime=0)


def _b64_lines(payload: bytes) -> list[str]:
    body = base64.b64encode(payload).decode("ascii")
    return [body[i:i + B64_LINE] for i in range(0, len(body), B64_LINE)]


def bundle_text(payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    return (f"{MARK_BEGIN} sha256={digest} bytes={len(payload)} ===\n" + "\n".join(_b64_lines(payload))
            + f"\n{MARK_END}\n")


def read_bundle(data: bytes) -> dict[str, bytes]:
    """name -> bytes from the saved terminal output (the last marked block); its sha256 and length must match."""
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = data.decode("utf-16")
    else:
        text = data.decode("utf-8", "replace").lstrip("﻿")
    text = text.replace("\r", "")
    begin = text.rfind(MARK_BEGIN)
    if begin < 0:
        raise InterferenceError(f"no '{MARK_BEGIN}' line in the input: the pod run did not finish, or the output was "
                                "not saved; the block is also on the pod as <run dir>/result_bundle.b64")
    header_end = text.find("\n", begin)
    end = text.find(MARK_END, header_end)
    if end < 0:
        raise InterferenceError("the result block has no end line: the output was cut short")
    m = re.search(r"sha256=([0-9a-f]{64}) bytes=(\d+)", text[begin:header_end])
    if not m:
        raise InterferenceError("the result block's first line carries no sha256 and length")
    payload = base64.b64decode(re.sub(r"[^A-Za-z0-9+/=]", "", text[header_end:end]))
    if len(payload) != int(m.group(2)) or hashlib.sha256(payload).hexdigest() != m.group(1):
        raise InterferenceError("the result block does not match its sha256 and length: it was damaged in transit")
    out: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(payload)), mode="r") as tar:
        for member in tar.getmembers():
            if member.isfile() and not member.name.startswith(("/", "..")) and "/../" not in member.name:
                handle = tar.extractfile(member)
                if handle is not None:
                    out[member.name] = handle.read()
    return out


# --- the script the pod runs --------------------------------------------------------------------------------------

def _module_file(name: str, root: Path) -> Path | None:
    base = root.joinpath(*name.split("."))
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    return None


def module_closure(starts: Sequence[str] = ("lab.interference", "lab.profile_decode_pod", "lab.train"),
                   root: Path = REPO) -> list[Path]:
    """Every lab and contracts module the start modules import, directly or not (function-level imports too), with
    the package __init__ files; BOX_ONLY modules stay behind."""
    seen: dict[str, Path] = {}
    todo = list(starts)
    while todo:
        name = todo.pop()
        if name in seen or not any(name == top or name.startswith(top + ".") for top in ("lab", "contracts")) or any(
                name == box or name.startswith(box + ".") for box in BOX_ONLY):
            continue
        path = _module_file(name, root)
        if path is None:
            continue
        seen[name] = path
        parts = name.split(".")
        todo.extend(".".join(parts[:i]) for i in range(1, len(parts)))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                todo.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                todo.append(node.module)
                todo.extend(f"{node.module}.{a.name}" for a in node.names)
    return sorted(set(seen.values()))


def data_files(root: Path = REPO, *, dataset: str = DEFAULT_DATASET, rows: str = DEFAULT_ROWS) -> list[Path]:
    """The prompt set lab/bench.py loads and the images it names, the trainer's requirements file, and the bundled
    tool-calling rows with their dataset folder (when they are repo paths)."""
    from lab import bench

    prompts = Path(bench.DEFAULT_PROMPTS)
    data = json.loads(prompts.read_text(encoding="utf-8"))
    out = [prompts, *(prompts.parent / p["image"] for p in data.get("prompts", []) if p.get("kind") == "image"),
           root / "lab" / "requirements-train.txt"]
    for rel in (dataset, rows):
        path = root / rel
        if path.is_file():
            out.append(path)
        elif path.is_dir():
            out.extend(sorted(p for p in path.rglob("*") if p.is_file()))
    return out


def pod_script(*, expect_user_sha256: str, args: Sequence[str], root: Path = REPO,
               dataset: str = DEFAULT_DATASET, rows: str = DEFAULT_ROWS) -> tuple[str, dict[str, Any]]:
    """The bash script the pod runs: the lab files it needs as one base64 block checked against its sha256,
    unpacked under ~/r43_interference/bundle-<sha>, then pod-run there. Its last line is `### END rc=<code>`.
    No tab and no exclamation mark: it may be typed into an interactive shell."""
    files = {str(p.relative_to(root)).replace("\\", "/"): p.read_bytes()
             for p in (*module_closure(root=root), *data_files(root, dataset=dataset, rows=rows))}
    payload = make_bundle(files)
    digest = hashlib.sha256(payload).hexdigest()
    run_args = " ".join(shlex.quote(a) for a in ["--expect-user-sha256", expect_user_sha256, *args])
    script = "\n".join([
        "# Interference of teacher decode and a LoRA run on one chip (R43), made by python -m lab interference "
        "pod-script; run with bash.",
        "# It unpacks the lab files below, then runs the three arms; the result comes back as one marked base64 block.",
        "set -u",
        f'B="$HOME/{SCRIPT_ROOT}/bundle-{digest[:12]}"',
        'mkdir -p "$B"',
        "base64 -d > \"$B/bundle.tar.gz\" <<'R43BUNDLE'",
        *_b64_lines(payload),
        "R43BUNDLE",
        f'echo "{digest}  $B/bundle.tar.gz" | sha256sum -c --quiet - '
        '|| { echo "R43 bundle damaged in transit"; exit 3; }',
        'tar -xzf "$B/bundle.tar.gz" -C "$B" || exit 3',
        'python3 -c "import httpx" 2>/dev/null || pip install --user --quiet httpx',
        'cd "$B" || exit 3',
        f"python3 -m lab.interference pod-run {run_args}",
        "rc=$?",
        'echo "### R43 pod-run rc=$rc"',
        'echo "### END rc=$rc"',
        "exit $rc",
        ""])
    info = {"files": len(files), "bundle_bytes": len(payload), "bundle_sha256": digest,
            "script_bytes": len(script.encode()), "lines": script.count("\n")}
    return script, info


# --- the pod run --------------------------------------------------------------------------------------------------

def say(message: str) -> None:
    print(f"[r43 {dt.datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def serve_argv(command: Sequence[str], share: float) -> list[str]:
    """The serve line with --gpu-memory-utilization set to share (added when the line has none)."""
    argv = list(command)
    if "--gpu-memory-utilization" in argv:
        argv[argv.index("--gpu-memory-utilization") + 1] = f"{share:g}"
    else:
        argv += ["--gpu-memory-utilization", f"{share:g}"]
    return argv


def read_steps(path: Path) -> list[dict[str, Any]]:
    """The telemetry.train.step events a trainer wrote to its telemetry.jsonl."""
    out = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("type") == "telemetry.train.step":
            out.append({k: event.get(k) for k in ("ts", "step", "step_ms", "loss", "tokens")})
    return out


class PodRun:
    """The three arms on the pod, in order; see the module docstring. Every outside piece (the server command, the
    trainer command, the power reader) can be handed in, so the tests run it against lab/bench_fake.py and a fake
    trainer."""

    def __init__(self, args: Any, *, power: Any = None) -> None:
        from lab import bench

        self.bench = bench
        self.args = args
        self.started = time.time()
        self.deadline = self.started + float(args.deadline_min) * 60.0
        self.run_id = "r43-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:4]
        self.dir = Path(args.out).expanduser() / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.prompts = bench.PromptSet.load(args.prompts) if args.prompts else bench.PromptSet.load()
        self.settings = bench.Settings(model=MODEL_NAME, max_tokens=int(args.max_tokens))
        self.warm = bench.Settings(model=MODEL_NAME, max_tokens=WARMUP_TOKENS)
        self.clock = bench.WallClock()
        if power is None:
            power = bench.LocalPower.for_host(bench.Host(), node=NODE, period_s=POWER_PERIOD_S)
        self.power = power
        self.raw: dict[str, Any] = {
            "schema": RAW_SCHEMA, "run_id": self.run_id, "task": TASK, "node": NODE, "hub": args.hub,
            "account": args.account, "started": now_iso(), "ended": None,
            "args": {k: v for k, v in vars(args).items() if k not in ("func", "expect_user_sha256")},
            "power_reader": "lab/bench.py LocalPower (amd-smi in this process, "
                            f"{POWER_PERIOD_S:g} s)" if getattr(power, "available", False) else
                            f"none: {getattr(power, 'reason', None)}",
            "workload": {"prompt_set": self.prompts.describe(), "batch": int(args.batch),
                         "max_tokens": int(args.max_tokens), "decode_s": float(args.decode_s),
                         "steps": int(args.steps), "train_batch": int(args.train_batch),
                         "grad_accum": int(args.grad_accum), "train_model": args.model,
                         "dataset": args.dataset, "rows": args.rows},
            "arms": {}}

    def left_s(self) -> float:
        return self.deadline - time.time()

    def save(self) -> Path:
        path = self.dir / "raw.json"
        path.write_text(json.dumps(self.raw, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
        return path

    # the server

    def start_server(self) -> tuple[Any, Any]:
        bench, args = self.bench, self.args
        log_path = self.dir / "serve.log"
        command = args.serve_cmd or bench.default_command("vllm", CHIP)
        argv = bench.fill_command(serve_argv(command, args.share) if not args.serve_cmd else command,
                                  {"checkpoint": str(Path(args.checkpoint).expanduser()), "model": MODEL_NAME,
                                   "port": args.port, "max_len": MAX_LEN, "log": log_path,
                                   "port_file": log_path.with_suffix(".port"), "share": args.share})
        cwd = Path(args.serve_dir or "~/serve").expanduser()
        cwd.mkdir(parents=True, exist_ok=True)
        spawned = bench.spawn_server(argv, port=args.port, log_path=log_path, env={}, cwd=cwd,
                                     own_log=bool(args.serve_cmd))
        say(f"server pid {spawned.pid} starting at share {args.share:g} (log {log_path.name})")
        self.raw["serve"] = {"argv": argv, "share": args.share, "port": args.port,
                             "checkpoint": str(args.checkpoint), "pid": spawned.pid}
        try:
            endpoint = bench.DirectEndpoint(f"http://127.0.0.1:{args.port}", timeout_s=600.0)
            began = time.time()
            bench.wait_ready(spawned, endpoint, min(READY_TIMEOUT_S, max(60.0, self.left_s())))
            self.raw["serve"]["ready_s"] = round(time.time() - began, 1)
        except BaseException:
            spawned.stop(timeout_s=60.0)
            raise
        say(f"server ready after {self.raw['serve']['ready_s']} s")
        return spawned, endpoint

    # decode rounds

    def round(self, endpoint: Any, tag: str) -> dict[str, Any]:
        from lab import profile_decode_pod as pod

        start = time.time()
        requests = pod.run_requests(endpoint, self.prompts, self.settings, batch=int(self.args.batch),
                                    clock=self.clock, tag=f"{self.run_id}-{tag}", role="measure")
        end = time.time()
        return {"start": start, "end": end,
                "requests": [{k: v for k, v in r.items() if k not in ("token_times", "sent_t", "first_t", "last_t")}
                             for r in requests]}

    def decode_until(self, endpoint: Any, tag: str, stop: Callable[[], bool]) -> list[dict[str, Any]]:
        rounds = []
        while not stop() and self.left_s() > 30.0:
            rounds.append(self.round(endpoint, f"{tag}{len(rounds)}"))
        return rounds

    # the trainer

    def train_argv(self, arm_dir: Path, dataset: Path, rows: Path) -> list[str]:
        args = self.args
        values = {"dataset": dataset, "rows": rows, "model": args.model, "steps": args.steps, "out": arm_dir / "run",
                  "run_id": f"{self.run_id}-{arm_dir.name}", "batch": args.train_batch,
                  "grad_accum": args.grad_accum, "python": args.python}
        command = args.train_cmd or ["{python}", "lab/train.py", "--dataset", "{dataset}", "--rows", "{rows}",
                                     "--model", "{model}", "--max-steps", "{steps}", "--batch", "{batch}",
                                     "--grad-accum", "{grad_accum}", "--out", "{out}", "--run-id", "{run_id}"]
        return [re.sub(r"\{(\w+)\}", lambda m: str(values[m.group(1)]) if m.group(1) in values else m.group(0), str(a))
                for a in command]

    def start_train(self, name: str) -> tuple[subprocess.Popen, Path, Any]:
        """A fresh copy of the dataset and rows (the trainer deletes its copy), then the trainer, logging to a file."""
        arm_dir = self.dir / name
        data = arm_dir / "data"
        source = Path(self.args.dataset).expanduser()
        dataset = data / source.name
        shutil.copytree(source, dataset)
        rows = data / Path(self.args.rows).name
        shutil.copyfile(Path(self.args.rows).expanduser(), rows)
        argv = self.train_argv(arm_dir, dataset, rows)
        log = open(arm_dir / "train.log", "wb")
        say(f"{name}: trainer starting ({' '.join(map(str, argv[:4]))} ...)")
        proc = subprocess.Popen([str(a) for a in argv], stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                cwd=str(Path.cwd()), start_new_session=os.name == "posix")
        return proc, arm_dir, log

    def finish_train(self, proc: subprocess.Popen, arm_dir: Path, log: Any, timeout_s: float) -> dict[str, Any]:
        try:
            code = proc.wait(timeout=max(1.0, timeout_s))
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGTERM)
            else:
                proc.terminate()
            code = proc.wait(timeout=60)
        finally:
            log.close()
        run_dir = arm_dir / "run"
        try:
            run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            run = {}
        steps = read_steps(run_dir / "telemetry.jsonl")
        tail = (arm_dir / "train.log").read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
        return {"exit_code": code, "steps": steps, "batch": int(self.args.train_batch),
                "grad_accum": int(self.args.grad_accum),
                "run": {"outcome": run.get("outcome"), "ok": run.get("ok"), "metrics": run.get("metrics"),
                        "error": run.get("error")}, "log_tail": tail}

    def watts(self, start: float | None, end: float | None) -> dict[str, Any]:
        if start is None or end is None:
            return {"available": False, "reason": "no window"}
        return self.bench.window_watts(self.power, start, end)

    # the arms

    def decode_arm(self, endpoint: Any) -> dict[str, Any]:
        until = time.time() + float(self.args.decode_s)
        rounds = self.decode_until(endpoint, "d", lambda: time.time() >= until)
        summary = decode_summary(rounds)
        window = summary["window"] or [None, None]
        say(f"decode: {summary['answers']} answers, {summary['decode_tok_s']} tok/s")
        return {"rounds": rounds, "watts": self.watts(*window)}

    def train_arm(self) -> dict[str, Any]:
        proc, arm_dir, log = self.start_train("train")
        out = self.finish_train(proc, arm_dir, log, min(TRAIN_TIMEOUT_S, self.left_s() - 60.0))
        window = train_window(out["steps"])
        out["watts"] = self.watts(*(window or (None, None)))
        say(f"train: {len(out['steps'])} steps, exit {out['exit_code']}")
        return out

    def both_arm(self, endpoint: Any) -> dict[str, Any]:
        proc, arm_dir, log = self.start_train("both")
        rounds = self.decode_until(endpoint, "b", lambda: proc.poll() is not None)
        out = self.finish_train(proc, arm_dir, log, min(TRAIN_TIMEOUT_S, self.left_s() - 60.0))
        window = train_window(out["steps"])
        out["rounds"] = rounds
        out["watts"] = self.watts(*(window or (None, None)))
        say(f"both: {len(out['steps'])} steps and {len(rounds)} decode rounds")
        return out

    def run(self) -> dict[str, Any]:
        spawned = None
        self.power.start()
        try:
            spawned, endpoint = self.start_server()
            from lab import profile_decode_pod as pod

            for batch in (1, int(self.args.batch)):
                pod.run_requests(endpoint, self.prompts, self.warm, batch=batch, clock=self.clock,
                                 tag=f"{self.run_id}-warm", role="warmup")
            for name, step in (("decode", lambda: self.decode_arm(endpoint)), ("train", self.train_arm),
                               ("both", lambda: self.both_arm(endpoint))):
                try:
                    self.raw["arms"][name] = step()
                except Exception as exc:  # one arm failing must not lose the others
                    self.raw["arms"][name] = {"error": f"{type(exc).__name__}: {exc}"[:400]}
                self.save()
        except Exception as exc:
            self.raw["error"] = f"{type(exc).__name__}: {exc}"[:400]
        finally:
            if spawned is not None:
                spawned.stop(timeout_s=120.0)
            self.power.stop()
            self.raw["ended"] = now_iso()
            self.save()
        return self.raw


def run_and_report(run: PodRun) -> dict[str, Any]:
    """Run the arms, build the result, keep it in the run folder and print it as the marked block."""
    say(f"run {run.run_id} in {run.dir}")
    raw = run.run()
    files = {"raw.json": json.dumps(raw, default=str).encode()}
    error = None
    result = None
    try:
        result = build_result(raw)
        files["result.json"] = json.dumps(result, indent=1).encode()
        files["summary_pod.md"] = render_markdown(result).encode()
        (run.dir / "result.json").write_bytes(files["result.json"])
        (run.dir / "summary_pod.md").write_bytes(files["summary_pod.md"])
    except Exception as exc:  # the raw data must still come back for a second look
        error = f"{type(exc).__name__}: {exc}"
        files["analysis_error.txt"] = error.encode()
    text = bundle_text(make_bundle(files))
    (run.dir / "result_bundle.b64").write_text(text, encoding="utf-8", newline="\n")
    print(text, flush=True)
    return {"ok": result is not None and not check_result(result), "outcome": "measured", "run_dir": str(run.dir),
            "analysis_error": error, "error": raw.get("error"),
            "problems": check_result(result) if result is not None else None}


# --- verbs --------------------------------------------------------------------------------------------------------

def pod_run_main(args: argparse.Namespace) -> dict[str, Any]:
    from lab import bench
    from lab import profile_decode_pod as pod

    if not args.serve_cmd:
        problems = pod.gate(expect_user_sha256=args.expect_user_sha256, environ=os.environ,
                            rocminfo=pod.read_rocminfo, port=args.port,
                            servers=lambda: pod.vllm_pids(pod.proc_table(bench.Host())))
        if args.share > MAX_SHARE:
            problems.append(f"--share {args.share:g} is above {MAX_SHARE}: the LoRA job would not fit beside the "
                            "teacher")
        if problems:
            return {"ok": False, "outcome": "refused", "problems": problems}

    def stop_here(signum: int, frame: Any) -> None:
        raise SystemExit(f"signal {signum}")

    for name in ("SIGHUP", "SIGTERM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), stop_here)
    return run_and_report(PodRun(args))


def pod_script_main(args: argparse.Namespace) -> dict[str, Any]:
    from lab import profile_decode_pod as pod

    if args.share > MAX_SHARE:
        raise InterferenceError(f"--share {args.share:g} is above {MAX_SHARE}")
    if args.expect_user_sha256:
        expect = args.expect_user_sha256
    else:
        from lab import hub

        try:
            account = hub.load_accounts([args.account])[args.account]
        except hub.MissingCredentials as exc:
            raise InterferenceError(f"{exc}: pod-script reads account {args.account}'s user name from the workspace "
                                    ".env (or pass --expect-user-sha256)") from exc
        expect = pod.user_sha256(account.user)
    run_args = ["--share", f"{args.share:g}", "--batch", str(args.batch), "--max-tokens", str(args.max_tokens),
                "--decode-s", f"{args.decode_s:g}", "--steps", str(args.steps), "--train-batch", str(args.train_batch),
                "--grad-accum", str(args.grad_accum), "--model", args.model, "--dataset", args.dataset,
                "--rows", args.rows, "--checkpoint", args.checkpoint, "--port", str(args.port),
                "--deadline-min", f"{args.deadline_min:g}", "--hub", args.hub, "--account", args.account]
    script, info = pod_script(expect_user_sha256=expect, args=run_args, dataset=args.dataset, rows=args.rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(script.encode("ascii"))
    slot = slot_estimate(decode_s=args.decode_s, steps=args.steps, train_batch=args.train_batch,
                         grad_accum=args.grad_accum)
    wait = (slot["slot_minutes"] + slot["extension_minutes"]) * 60
    return {"ok": True, "outcome": "written", "script": str(out), **info, "slot_minutes": slot["slot_minutes"],
            "next": f"python tooling/dispatcher/pod_exec.py {args.account} run <this script> r43_pod_run.sh {wait} "
                    "> <output file>",
            "or": f"python tooling/dispatcher/pod_exec.py {args.account} term <this script> {wait} > <output file>",
            "then": "python -m lab interference collect <output file>"}


def collect_main(args: argparse.Namespace) -> dict[str, Any]:
    from lab.machine.record import record

    files = read_bundle(Path(args.input).read_bytes())
    if "result.json" not in files:
        error = files.get("analysis_error.txt", b"no result.json in the bundle").decode("utf-8", "replace")
        return {"ok": False, "outcome": "no_result", "error": error, "files": sorted(files)}
    result = json.loads(files["result.json"].decode("utf-8"))
    problems = check_result(result)
    if problems:
        return {"ok": False, "outcome": "invalid", "problems": problems}
    json_path, md_path = write_outputs(result, Path(args.out) if args.out else DATA_DIR, force=args.force)
    try:
        source = str(json_path.resolve().relative_to(REPO)).replace("\\", "/")
    except ValueError:
        source = str(json_path)
    seqs = []
    if not args.no_ledger:
        for row in ledger_rows(result, source):
            seqs.append(record(row["verb"], row["outcome"], row["metrics"], row["refs"], task=TASK, node=row["node"],
                               units=row["units"], conditions=row["conditions"], source=row["source"],
                               low_confidence=row["low_confidence"], note=row["note"]))
    return {"ok": True, "outcome": "collected", "json": str(json_path), "markdown": str(md_path),
            "ledger_seqs": seqs, "verdict": result["compare"]["verdict"], "open": result["open"]}


def _share(value: str) -> float:
    try:
        share = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {value!r}") from None
    if not 0.05 <= share <= 0.95:
        raise argparse.ArgumentTypeError("a share is between 0.05 and 0.95")
    return share


def _workload_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--share", type=_share, default=DEFAULT_SHARE,
                   help=f"the teacher's --gpu-memory-utilization (default and ceiling {MAX_SHARE})")
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="decode streams at once")
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--decode-s", type=float, default=DEFAULT_DECODE_S, help="the decode arm's length")
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="the LoRA run's optimizer steps")
    p.add_argument("--train-batch", type=int, default=DEFAULT_TRAIN_BATCH)
    p.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    p.add_argument("--model", default=DEFAULT_MODEL, help="the student the LoRA trains (a folder on the pod)")
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="the tool-calling dataset folder (bundled when a repo "
                                                              "path)")
    p.add_argument("--rows", default=DEFAULT_ROWS, help="its training rows (bundled when a repo path)")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="the teacher checkpoint on the pod")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--deadline-min", type=float, default=DEFAULT_DEADLINE_MIN)
    p.add_argument("--hub", default="sjc")
    p.add_argument("--account", default=DEFAULT_ACCOUNT)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m lab interference",
                                description="Teacher decode and a LoRA run on one chip, alone and together.")
    p.add_argument("--dry-run", action="store_true", help="print the plan and the slot minutes; nothing runs")
    _workload_args(p)
    sub = p.add_subparsers(dest="command")
    script = sub.add_parser("pod-script", help="write the script the pod runs")
    _workload_args(script)
    script.add_argument("--out", default=str(POD_SCRIPT))
    script.add_argument("--expect-user-sha256", default=None, help="the pod account's hash, instead of the .env's")
    run = sub.add_parser("pod-run", help="on the pod: run the three arms (what the script runs)")
    _workload_args(run)
    run.add_argument("--expect-user-sha256", required=True)
    run.add_argument("--out", default=DEFAULT_OUT)
    run.add_argument("--prompts", default=None)
    run.add_argument("--python", default="python3")
    run.add_argument("--serve-dir", default=None)
    run.add_argument("--serve-cmd", type=json.loads, default=None, help=argparse.SUPPRESS)   # tests: a JSON argv
    run.add_argument("--train-cmd", type=json.loads, default=None, help=argparse.SUPPRESS)
    collect = sub.add_parser("collect", help="the data file, its table and the ledger rows from the pod's output")
    collect.add_argument("input", help="the saved terminal output (or the result_bundle.b64)")
    collect.add_argument("--out", default=None, help=f"default {DATA_DIR.relative_to(REPO)}")
    collect.add_argument("--force", action="store_true")
    collect.add_argument("--no-ledger", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Teacher decode and a LoRA run on one chip: alone and together (plan, pod script, collect)."""
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        if args.command == "pod-script":
            return pod_script_main(args)
        if args.command == "pod-run":
            return pod_run_main(args)
        if args.command == "collect":
            return collect_main(args)
        return plan(args)
    except InterferenceError as exc:
        return {"ok": False, "outcome": "error", "error": str(exc)}


if __name__ == "__main__":
    answer = main()
    print(json.dumps(answer, default=str))
    sys.exit(0 if answer.get("ok", True) else 1)
