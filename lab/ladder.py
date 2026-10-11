"""ladder: the largest student whose LoRA fine-tune fits the chips on offer, with its predicted run time and why.

    python -m lab.ladder --capacity CAP.json --samples 2000 --epochs 2                    # one chip
    python -m lab.ladder --capacity A.json --capacity B.json --samples 2000 --epochs 2 \\
        --teacher-share 0 [--within-hours 3] [--reference-tflops X] [--json]               # two chips

The ladder (LADDER) is committed: today's student, Qwen/Qwen3.5-4B, and the next dense sizes up the Qwen3.5 family
offers, 9B and 27B. Each rung carries its LoRA step memory (peak allocated GiB at R36's pinned baseline: bf16, LoRA
r 16 on all linear layers, batch 4 by accumulation 4, max_length 256, gradient checkpointing) and its training samples
per second, each marked measured or predicted with its source. The 4B numbers are the one measured LoRA step
(lab/refine/metrics.jsonl lines 13 and 14, the smoke run on one R9700, another chip); the larger rungs are predicted
from it by parameter count, since bf16 weights dominate the step at that batch and length
(settings_training_gfx1151.md section 1: the 4B step fits under 10 GB). R34's step profile and L3's real runs
replace them with measurements.

A rung fits a chip when its step memory plus the placement plan's margin and host reserve (placement.Rules: 50
percent and 4 GiB, the same budget the plan gives a LoRA job) is no more than the chip's free memory in its
capacity record (lab/capacity.py). Its rate on that chip is the rung's rate scaled by the chip's measured bf16
matmul rate over the reference chip's, when both are known, and the rung's own rate otherwise (said so). Its run
time is samples x epochs / rate.

One chip: the pick is the largest rung that fits (and, with --within-hours, finishes in time). More than one chip
(the Circle's opt-in sharing; pods C and D stand in for two friends' boxes): the chips must be the placement plan's
pods (pod-a, pod-b). The plan (lab/placement.py build_plan) says which of them take a LoRA job now and that two
LoRA jobs never share a chip, so the ladder runs at most one job per such chip, all at once: rungs largest first,
each on the free chip where it finishes soonest. The pick is the largest rung scheduled (the best predicted
quality: a larger student of the same family on the same rows; the L-chain eval measures it), and among equals the
one that finishes soonest. The smaller rung beside it is the comparison the demo evidence needs. Training one model
across chips (data parallel) is never assumed: it is the experiment in lab/R44_REAL_RUN.md, with its kill
criterion on the measured pod-to-pod link.

Nothing here calls a pod, the hub or a network address; it reads capacity records, the metrics file, the search
space and the teacher's serve script.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lab import capacity, placement
from lab.placement import Num

SEARCH_SPACE = placement.CHIPSHEET_DIR / "searchspace-training-gfx1151.json"
SETTINGS_NOTE = "lab/refine/research/kernels/settings_training_gfx1151.md"
REAL_RUN = "lab/R44_REAL_RUN.md"
STUDENT = "Qwen/Qwen3.5-4B"


@dataclasses.dataclass(frozen=True)
class Candidate:
    """One rung: the model, its size in billions of parameters and where that size is named, its LoRA step memory
    (GiB) and its training samples per second, each a Num of kind measured or predicted."""

    model: str
    params_b: float
    size_source: str
    step_mem: Num
    samples_per_s: Num

    def view(self) -> dict[str, Any]:
        return {"model": self.model, "params_b": self.params_b, "size_source": self.size_source,
                "step_mem": self.step_mem.view(), "samples_per_s": self.samples_per_s.view()}


# The measured rung: the 20-step smoke LoRA of the student (lab/smoke_lora.py) on one R9700, gfx1201.
_PEAK_GIB = 8.7
_RATE = 0.51
_MEASURED = ("lab/refine/metrics.jsonl:{line} ({metric}, the 4B smoke LoRA at batch 4, max_length 256, r 16, "
             "on one R9700 (gfx1201), another chip: PR #37)")


def _predicted(params_b: float) -> tuple[Num, Num]:
    ratio = params_b / 4.0
    why = (f"the measured 4B rung scaled by parameter count ({params_b:g}B / 4B): bf16 weights dominate a LoRA step at "
           f"batch 4 and max_length 256 ({SETTINGS_NOTE} section 1); R34 and L3 replace it")
    return (Num(round(_PEAK_GIB * ratio, 3), "GiB", f"predicted: {_PEAK_GIB:g} GiB x {ratio:g}, {why}", "predicted"),
            Num(round(_RATE / ratio, 4), "samples/s", f"predicted: {_RATE:g} samples/s / {ratio:g}, {why}",
                "predicted"))


LADDER: tuple[Candidate, ...] = (
    Candidate(STUDENT, 4.0, "the student the Jetson runs (lab/README.md, Trainer; lab/train.py)",
              Num(_PEAK_GIB, "GiB", _MEASURED.format(line=14, metric="trainer.peak_mem_gb"), "measured"),
              Num(_RATE, "samples/s", _MEASURED.format(line=13, metric="trainer.samples_per_s"), "measured")),
    Candidate("Qwen/Qwen3.5-9B", 9.0, "lab/refine/research/kernels/SOURCES.md S27 and w4a16_decode_gfx1201.md "
              "(Qwen3.5-9B runs on the same RDNA4 silicon)", *_predicted(9.0)),
    Candidate("Qwen/Qwen3.5-27B", 27.0, "lab/refine/research/kernels/w4a16_decode_gfx1201.md and "
              "graphics_pipeline_for_ml.md (dense Qwen3.5-27B runs on one R9700); the teacher, Qwen/Qwen3.8-27B, is "
              "the same size", *_predicted(27.0)),
)


def candidate(model: str) -> Candidate:
    for rung in LADDER:
        if rung.model == model:
            return rung
    raise KeyError(model)


def settings(path: Path = SEARCH_SPACE) -> dict[str, Any]:
    """The settings every rung's numbers assume: R36's pinned baseline and the pinned LoRA and length."""
    space = json.loads(path.read_text(encoding="utf-8"))
    pinned = space["pinned"]
    return {**space["baseline"], "max_length": pinned["max_length"], "lora_r": pinned["lora"]["r"],
            "lora_alpha": pinned["lora"]["lora_alpha"], "source": placement.relative(path) + ": baseline, pinned"}


# --- arithmetic -------------------------------------------------------------------------------------------------

def run_seconds(samples: int, epochs: float, rate: float) -> float:
    """samples x epochs / samples per second."""
    if samples <= 0 or epochs <= 0:
        raise ValueError("samples and epochs must be positive")
    if not rate or rate <= 0:
        raise ValueError("the rate must be positive")
    return samples * epochs / rate


def need_gib(rung: Candidate, rules: placement.Rules) -> float:
    """What a rung's fine-tune takes from a chip: step peak x (1 + margin) + host reserve, as the plan budgets."""
    return round(rung.step_mem.value * (1 + rules.margin) + rules.host_reserve_gib, 3)


def _value(record: Mapping[str, Any], field: str) -> float | None:
    part = record.get(field)
    return placement._number(part)


def rate_on(rung: Candidate, record: Mapping[str, Any], reference_tflops: float | None = None) -> tuple[float, str]:
    """A rung's samples per second on a chip, and how it was found."""
    chip = _value(record, "bf16_tflops")
    base = rung.samples_per_s.value
    if chip is not None and reference_tflops:
        return base * chip / reference_tflops, (f"{base:g} samples/s x the chip's {chip:g} TFLOPS bf16 / the "
                                                f"reference's {reference_tflops:g}")
    missing = "the chip's bf16 rate is unknown" if chip is None else "the reference chip's bf16 rate is unknown"
    return base, f"{base:g} samples/s unscaled: {missing}"


def _h(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h"


def _gib(value: float) -> str:
    return f"{round(value, 2):g} GiB"


# --- the pick ---------------------------------------------------------------------------------------------------

def options(records: Sequence[Mapping[str, Any]], *, samples: int, epochs: float, within_s: float | None,
            rules: placement.Rules, reference_tflops: float | None,
            free_of: Mapping[str, float | None]) -> list[dict[str, Any]]:
    """Every rung on every chip: need, free, fits, rate, run time, in time."""
    rows = []
    for rec in records:
        node = rec["node"]
        free = free_of[node]
        for rung in LADDER:
            need = need_gib(rung, rules)
            rate, how = rate_on(rung, rec, reference_tflops)
            run_s = run_seconds(samples, epochs, rate)
            fits = free is not None and need <= free
            rows.append({"model": rung.model, "params_b": rung.params_b, "node": node, "need_gib": need,
                         "free_gib": free, "fits": fits, "samples_per_s": round(rate, 4), "rate_from": how,
                         "run_s": round(run_s, 3), "in_time": within_s is None or run_s <= within_s,
                         "kind": rung.step_mem.kind})
    return rows


def _schedule(rows: list[dict[str, Any]], nodes: Sequence[str]) -> list[dict[str, Any]]:
    """At most one job per chip, all at once: rungs largest first, each on the free chip where it finishes soonest."""
    jobs: list[dict[str, Any]] = []
    taken: set[str] = set()
    for rung in sorted(LADDER, key=lambda c: -c.params_b):
        choices = [row for row in rows if row["model"] == rung.model and row["node"] in nodes
                   and row["node"] not in taken and row["fits"] and row["in_time"]]
        if not choices:
            continue
        best = min(choices, key=lambda row: (row["run_s"], row["node"]))
        jobs.append(best)
        taken.add(best["node"])
        if len(taken) == len(nodes):
            break
    return jobs


def _plan(records: Sequence[Mapping[str, Any]], rules: placement.Rules, teacher_on: str | None) -> placement.Plan:
    pods = {node: placement.Pod(node) for node in placement.NODES}
    for rec in records:
        pod = pods[rec["node"]]
        pod.up, pod.up_source = True, f"capacity record of {rec['node']} ({rec.get('probed_at')})"
        free, visible = _value(rec, "free_gib"), _value(rec, "visible_gib")
        if free is not None:
            pod.available = Num(free, "GiB", f"capacity record: free_gib ({rec['free_gib'].get('source')})")
        if visible is not None:
            pod.visible = Num(visible, "GiB", f"capacity record: visible_gib ({rec['visible_gib'].get('source')})")
        pod.gfx = (rec.get("chip") or {}).get("gfx")
        pod.gpu = (rec.get("chip") or {}).get("name")
        if rec["node"] == teacher_on:
            pod.serving = (placement.TEACHER_MODEL,)
            pod.serving_source = "--teacher-on"
    return placement.build_plan(pods, rows=None, rules=rules)


def pick(records: Sequence[Mapping[str, Any]], *, samples: int, epochs: float, within_s: float | None = None,
         rules: placement.Rules | None = None, reference_tflops: float | None = None,
         teacher_share: float | None = None, teacher_on: str | None = None) -> dict[str, Any]:
    """The rung to fine-tune on the chips these capacity records describe, its predicted run time and why."""
    if not records:
        raise ValueError("no capacity record given")
    rules = rules or placement.Rules()
    if teacher_share is not None:
        rules = dataclasses.replace(rules, teacher_share=teacher_share)
    nodes = [rec["node"] for rec in records]
    if len(set(nodes)) != len(nodes):
        raise ValueError(f"two records for one chip: {nodes}")
    free_of = {rec["node"]: _value(rec, "free_gib") for rec in records}
    answer: dict[str, Any] = {"samples": samples, "epochs": epochs, "within_s": within_s, "settings": settings(),
                              "ladder": [rung.view() for rung in LADDER]}

    if len(records) == 1:
        lora_nodes = nodes
        placement_view: dict[str, Any] | None = None
    else:
        unknown_nodes = [node for node in nodes if node not in placement.NODES]
        if unknown_nodes:
            raise ValueError(f"the placement plan places pods {', '.join(placement.NODES)} only, not "
                             f"{', '.join(unknown_nodes)}")
        plan = _plan(records, rules, teacher_on)
        lora_nodes = [node for node in plan.where.get("lora", []) if node in free_of]
        lora_line = plan.line("lora")
        share = next(s for s in plan.sharing if s.jobs == ("lora", "lora"))
        if share.allowed:   # the plan has never allowed it; if it ever does, this ladder still runs one per chip
            raise ValueError("the placement plan allows two LoRA jobs on one chip; the ladder needs updating")
        if teacher_on in free_of and plan.line("teacher") and teacher_on in plan.line("teacher").pods:
            share_value = rules.teacher_share if rules.teacher_share is not None else \
                placement.script_setting("GPU_UTIL").value
            visible = _value(records[nodes.index(teacher_on)], "visible_gib")
            if free_of[teacher_on] is not None and visible is not None and share_value is not None:
                free_of[teacher_on] = max(0.0, free_of[teacher_on] - share_value * visible)
        left_out = [f"{node}: {placement.plain(lora_line.reason) if lora_line else 'no LoRA line in the plan'}"
                    for node in nodes if node not in lora_nodes]
        placement_view = {"lora": lora_nodes, "one_lora_per_chip": True, "lora_lora": placement.plain(share.reason),
                          "lora_line": placement.plain(lora_line.reason) if lora_line else None,
                          "left_out": left_out, "teacher_on": teacher_on,
                          "teacher_share": rules.teacher_share}

    rows = options(records, samples=samples, epochs=epochs, within_s=within_s, rules=rules,
                   reference_tflops=reference_tflops, free_of=free_of)
    jobs = _schedule(rows, lora_nodes)
    answer.update({"options": rows, "jobs": jobs, "placement": placement_view,
                   "data_parallel": {"assumed": False, "see": REAL_RUN,
                                     "why": "one model across chips waits on the measured pod-to-pod link"}})
    if not jobs:
        smallest = min(LADDER, key=lambda c: c.params_b)
        need = need_gib(smallest, rules)
        frees = ", ".join(f"{node} {_gib(free) if free is not None else 'unknown'}" for node, free in free_of.items()
                          if node in lora_nodes) or "no chip the plan gives a LoRA job"
        late = any(row["fits"] and not row["in_time"] for row in rows)
        reason = (f"No rung fits: even {smallest.model} needs {_gib(need)} ({_gib(smallest.step_mem.value)} step peak "
                  f"plus a {rules.margin * 100:g} percent margin and {_gib(rules.host_reserve_gib)} host reserve), and "
                  f"the chips offer {frees} free"
                  + (", or what fits does not finish in time." if late else "."))
        return {"ok": False, "outcome": "none_fits", "model": None, "node": None, "run_s": None, "reason": reason,
                **answer}

    chosen = min(jobs, key=lambda job: (-job["params_b"], job["run_s"]))
    rung = candidate(chosen["model"])
    bigger = [c for c in LADDER if c.params_b > rung.params_b]
    why_not = ""
    if bigger:
        nxt = bigger[0]
        best_free = max((free for node, free in free_of.items() if node in lora_nodes and free is not None),
                        default=None)
        row = min((r for r in rows if r["model"] == nxt.model and r["node"] in lora_nodes),
                  key=lambda r: r["run_s"], default=None)
        if row is not None and not any(r["fits"] for r in rows if r["model"] == nxt.model and r["node"] in lora_nodes):
            why_not = (f"; the next rung up, {nxt.model}, needs {_gib(need_gib(nxt, rules))}, more than the "
                       f"{_gib(best_free) if best_free is not None else 'unknown'} free")
        elif row is not None:
            why_not = f"; the next rung up, {nxt.model}, would take {_h(row['run_s'])}, past the deadline"
    reason = (f"{rung.model} on {chosen['node']} is the largest student whose LoRA fits "
              f"({_gib(chosen['need_gib'])} of {_gib(chosen['free_gib'])} free, step memory {rung.step_mem.kind}) and "
              f"is predicted to take {_h(chosen['run_s'])} for {samples} samples x {epochs:g} epochs at "
              f"{chosen['samples_per_s']:g} samples/s{why_not}")
    others = [job for job in jobs if job is not chosen]
    if others:
        reason += "; " + ", ".join(f"{job['node']} trains {job['model']} at the same time ({_h(job['run_s'])}) as "
                                   "the comparison" for job in others)
        reason += ", one LoRA job per chip as the placement plan allows, with no data-parallel speedup assumed"
    reason += "."
    return {"ok": True, "outcome": "picked", "model": rung.model, "node": chosen["node"], "run_s": chosen["run_s"],
            "reason": reason, **answer}


# --- the command line -------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m lab.ladder", description=main.__doc__, allow_abbrev=False)
    parser.add_argument("--capacity", action="append", required=True, metavar="FILE",
                        help="a capacity record from python -m lab capacity (repeat for each chip offered)")
    parser.add_argument("--samples", type=int, required=True, help="training samples per epoch")
    parser.add_argument("--epochs", type=float, default=1.0, help="epochs (default 1)")
    parser.add_argument("--within-hours", type=float, help="drop rungs that would take longer than this")
    parser.add_argument("--reference-tflops", type=float,
                        help="bf16 TFLOPS of the chip the ladder's rates were measured on, to scale them per chip")
    parser.add_argument("--teacher-share", type=float, help="the teacher's --gpu-memory-utilization (0: no teacher "
                        "on these chips, as on a friend's box)")
    parser.add_argument("--teacher-on", choices=placement.NODES, help="the pod that serves the teacher now")
    parser.add_argument("--json", action="store_true", help="answer with one JSON object")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Pick the largest student whose fine-tune fits the chips offered, with its predicted run time and why."""
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    records = [capacity.read_record(path) for path in args.capacity]
    answer = pick(records, samples=args.samples, epochs=args.epochs,
                  within_s=args.within_hours * 3600 if args.within_hours else None,
                  reference_tflops=args.reference_tflops, teacher_share=args.teacher_share, teacher_on=args.teacher_on)
    print(json.dumps(answer) if args.json else answer["reason"])
    return 0 if answer["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
