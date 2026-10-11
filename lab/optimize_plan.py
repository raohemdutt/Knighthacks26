"""The night configuration's rules as pure functions: restart classes, grouping, fidelity, early stop, observations,
the confirmation verdict and the minute budget. No I/O; lab/optimize_settings.py calls them.

Restart classes (lanes.<lane>.grouping.class_keys). A configuration's compile class is its values of the class keys.
Inference: every trial is one server start; it is warm when the worker's previous server had the same compile class
(the compile and graph caches are reused) and cold otherwise; the first server of a worker is cold. Training: every
trial is one process start (the trainer cannot yet change in-process parameters without reloading), labelled
"process", with the class recorded so the grouping shows in the files.

Grouping (lanes.<lane>.grouping.warm_class_every). With N set, every N-th trial a worker asks for is asked with
fixed_parameters = that worker's last compile class, so each class is started cold once and reused warm once (N = 2
in the configuration). None means no fixed parameters.

Fidelity. Inference trials run the "screen" bench arguments, or "finalist" (batch 1 and 8 in the same spawn) when the
strategy's model ranks the new arm among the top 3 of the arms so far; confirmation spawns run "confirm". Training
trials run "screen" (steps and the steps excluded from the median).

Early stop (lanes.<lane>.early_stop). Inference: a spawn stops after its first measured request when that request
decodes under 0.5 x the baseline's per-request median; the trial is completed with that one request and an SEM of
2 x the baseline's per-request sd. Training: a run stops when its mean samples/s over steps 2 to 5 is under
0.5 x the baseline's; it is completed with those steps. A trial that fails (the server was not ready within its
class's ready timeout, the trainer ran out of memory) is marked failed; max_consecutive_failures in a row stops the
worker. Training's Ax percentile early stopping (ax_percentile_strategy) is asked after every progression once
enabled_after_completed_trials trials have completed.

Observation (lanes.<lane>.repeat_policy). One spawn (or one process) per trial; the observation is the median over
its requests (or steps) with SEM = MEDIAN_SEM_FACTOR x sd / sqrt(n). Inference: joules_per_token.b1 takes the
decode rate's relative SEM, as the configuration says (decode power is near constant); ttft_s and the watts carry no
SEM, so Ax infers their noise.

Confirmation (lanes.<lane>.confirmation). spawns_per_arm spawns each of the best feasible arm and the baseline in the
order best, baseline, best, ...; read with lab/machine/verdict.py's Welch test, one-sided 95 percent (alpha 0.10
two-sided read in the direction of the gain), low_confidence until verdict.MIN_RUNS_PER_ARM spawns per arm.

Budget. A trial does not start when the minutes used plus its estimated seconds (the fidelity's est_trial_s for its
restart class, or the median seconds of this run's own trials of that class once there are some) pass the minute
budget; --budget caps the number of trials. plan_minutes() lays the trials out over the workers the same way, so
the dry run can say how many trials and restarts fit.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from lab.machine import verdict as verdict_math
from lab.optimize_space import Lane, same

MEDIAN_SEM_FACTOR = math.sqrt(math.pi / 2)   # about 1.2533: a median's standard error over a mean's, normal draws
EARLY_FACTOR = 0.5
EARLY_SEM_SD = 2.0
FINALIST_TOP = 3
CONFIRM_ALPHA_TWO_SIDED = 0.10               # one-sided 95 percent, read in the direction of the gain


def compile_class(lane: Lane, params: Mapping[str, Any]) -> dict[str, Any]:
    return {k: params[k] for k in lane.grouping.get("class_keys") or [] if k in params}


def same_class(a: Mapping[str, Any] | None, b: Mapping[str, Any]) -> bool:
    return a is not None and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)


def restart_class(lane: Lane, last_class: Mapping[str, Any] | None, params: Mapping[str, Any]) -> str:
    if lane.name == "training":
        return "process"
    return "warm" if same_class(last_class, compile_class(lane, params)) else "cold"


def fixed_for_ask(lane: Lane, asks: int, last_class: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The fixed parameters of a worker's ask number `asks` (0-based, counted over its own asks)."""
    every = lane.grouping.get("warm_class_every")
    if not every or last_class is None or (asks + 1) % int(every) != 0:
        return None
    return dict(last_class)


def ready_timeouts(lane: Lane) -> dict[str, float]:
    restart = lane.restart
    out = {k: float(v) for k, v in (restart.get("ready_timeout_s") or {}).items()}
    factor = float(restart.get("ready_timeout_factor") or 1.5)
    for key, cls in (("cold_s", "cold"), ("warm_s", "warm"), ("process_s", "process")):
        if cls not in out and isinstance(restart.get(key), (int, float)):
            out[cls] = restart[key] * factor
    return out


def bench_fidelities(lane: Lane) -> dict[str, list[str]]:
    return {name: list(f["bench_args"]) for name, f in lane.fidelity.items()
            if isinstance(f, Mapping) and isinstance(f.get("bench_args"), list)}


def est_seconds(lane: Lane, fidelity: str, restart: str) -> float | None:
    f = lane.fidelity.get(fidelity) or {}
    est = f.get("est_trial_s", f.get("est_spawn_s", f.get("est_run_s")))
    if isinstance(est, Mapping):
        est = est.get(restart, est.get("grouped_mean"))
    return float(est) if isinstance(est, (int, float)) else None


def is_finalist(lane: Lane, predicted_new: float | None, predicted_done: Sequence[float]) -> bool:
    """Is the new arm's predicted objective among the top FINALIST_TOP of the arms so far (and it)?"""
    if "finalist" not in lane.fidelity or predicted_new is None:
        return False
    ranked = sorted([*predicted_done, predicted_new], reverse=not lane.minimize)
    return ranked.index(predicted_new) < FINALIST_TOP


# --- early stop -----------------------------------------------------------------------------------------------------

def sample_key(lane: Lane) -> str:
    """The per-request (or per-step) samples a lane's observations and early stop are read from."""
    return "decode_tok_s.b1" if lane.name == "inference" else "samples_per_s"


def early_threshold(lane: Lane, baseline_samples: Sequence[float] | None) -> float | None:
    """The rate under which a trial is stopped early, from the baseline trial's samples; None without them."""
    if not baseline_samples:
        return None
    if lane.name == "inference":
        if not (lane.early_stop.get("after_first_request") or {}):
            return None
        return EARLY_FACTOR * statistics.median(baseline_samples)
    after = lane.early_stop.get("after_steps") or {}
    if not after:
        return None
    n = int(after.get("steps") or 5)
    window = list(baseline_samples)[1:n] or list(baseline_samples)
    return EARLY_FACTOR * statistics.fmean(window)


def early_after_steps(lane: Lane) -> int:
    return int((lane.early_stop.get("after_steps") or {}).get("steps") or 5)


def max_failures(lane: Lane) -> int:
    return int(lane.early_stop.get("max_consecutive_failures") or 3)


def percentile_after(lane: Lane) -> int | None:
    rule = lane.early_stop.get("ax_percentile_strategy")
    if isinstance(rule, Mapping) and rule.get("enabled_after_completed_trials") is not None:
        return int(rule["enabled_after_completed_trials"])
    return None


# --- observations ---------------------------------------------------------------------------------------------------

def median_sem(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    return MEDIAN_SEM_FACTOR * statistics.stdev(values) / math.sqrt(len(values))


def observation(lane: Lane, metrics: Mapping[str, float], samples: Mapping[str, Sequence[float]], *,
                early: bool = False, baseline_sd: float | None = None) -> dict[str, tuple[float, float | None]]:
    """{metric: (mean, sem)} for the lane's metrics this trial measured (repeat_policy, early stop's SEM rule)."""
    key = sample_key(lane)
    values = [float(v) for v in samples.get(key) or []]
    if early and baseline_sd is not None:
        sem = EARLY_SEM_SD * baseline_sd
    else:
        sem = median_sem(values)
    out: dict[str, tuple[float, float | None]] = {}
    center = statistics.median(values) if values else metrics.get(key)
    for m in lane.metrics:
        if m not in metrics:
            continue
        v = float(metrics[m])
        if m == key:
            out[m] = (float(center) if center is not None else v, sem)
        elif lane.name == "inference" and m.startswith("joules_per_token.") and sem is not None and center:
            out[m] = (v, v * sem / float(center))
        else:
            out[m] = (v, None)
    return out


def feasible(lane: Lane, means: Mapping[str, float]) -> bool:
    return all(c.holds(means.get(c.metric)) is not False for c in lane.constraints)


def best_observed(lane: Lane, trials: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The completed, feasible trial with the best observed objective (early-stopped and failed trials never win)."""
    best = None
    for t in trials:
        if t.get("status") != "completed" or t.get("phase") != "search":
            continue
        means = {k: v[0] for k, v in (t.get("observation") or {}).items()}
        if lane.metric not in means or not feasible(lane, means):
            continue
        if best is None or lane.better(means[lane.metric], best["observation"][lane.metric][0]):
            best = t
    return best


# --- confirmation ---------------------------------------------------------------------------------------------------

def confirmation_order(lane: Lane) -> list[str]:
    c = lane.confirmation
    n = int(c.get("spawns_per_arm") or c.get("runs_per_arm") or 0)
    arms = list(c.get("arms") or ["best", "baseline"])
    return [arm for _ in range(n) for arm in arms]


def confirm_verdict(lane: Lane, baseline_runs: Sequence[float], best_runs: Sequence[float]) -> dict[str, Any]:
    """Best (B) against the baseline (A) on the objective: Welch, one-sided 95 percent in the direction of a gain."""
    n = min(len(baseline_runs), len(best_runs))
    out: dict[str, Any] = {"metric": lane.metric, "runs_per_arm": [len(baseline_runs), len(best_runs)],
                           "test": "welch, one-sided 95 percent", "low_confidence": n < verdict_math.MIN_RUNS_PER_ARM}
    if n < 2:
        return {**out, "label": verdict_math.TOO_EARLY, "statistic": None, "p_one_sided": None}
    v = verdict_math.verdict(list(baseline_runs), list(best_runs), higher_is_better=not lane.minimize,
                             alpha=CONFIRM_ALPHA_TWO_SIDED, min_per_arm=2)
    gain = v.statistic is not None and ((v.statistic > 0) != lane.minimize)
    p_one = None if v.p is None else (v.p / 2 if gain else 1 - v.p / 2)
    return {**out, "label": v.label, "statistic": v.statistic, "p_one_sided": p_one,
            "best_ahead": v.label == verdict_math.B_AHEAD}


# --- the minute budget ----------------------------------------------------------------------------------------------

def plan_minutes(lane: Lane, *, workers: int, minutes: float, budget: int | None, initial: Sequence[Mapping[str, Any]],
                 done: int = 0) -> dict[str, Any]:
    """Lay the remaining trials out over the workers: the initial arms (with their declared restart) first, then
    trials alternating cold and warm when grouping is on (all cold when it is off; process starts for training),
    each worker in turn taking the next trial while its own clock stays inside the minutes."""
    workers = max(1, workers)
    clocks = [0.0] * workers
    out: list[dict[str, Any]] = []
    every = lane.grouping.get("warm_class_every")
    n = done
    queue = [dict(a) for a in initial[done:]] if done < len(initial) else []
    later = 0
    while budget is None or n < budget:
        w = min(range(workers), key=lambda i: clocks[i])
        if queue:
            arm = queue[0]
            restart = arm.get("restart") or ("process" if lane.name == "training" else "cold")
            restart = {"in_process": "process"}.get(restart, restart)
            name = arm["name"]
        else:
            if lane.name == "training":
                restart = "process"
            else:
                restart = "warm" if every and later % int(every) == int(every) - 1 else "cold"
            name = f"model {n}"
        seconds = est_seconds(lane, "screen", restart) or est_seconds(lane, "screen", "grouped_mean") or 0.0
        if clocks[w] + seconds > minutes * 60.0:
            if all(c + seconds > minutes * 60.0 for c in clocks):
                break
            clocks[w] = minutes * 60.0
            continue
        if queue:
            queue.pop(0)
        else:
            later += 1
        out.append({"trial": n, "arm": name, "worker": w, "restart": restart, "start_min": round(clocks[w] / 60, 1),
                    "est_s": seconds})
        clocks[w] += seconds
        n += 1
    restarts = {k: sum(1 for t in out if t["restart"] == k) for k in ("cold", "warm", "process")}
    return {"trials": out, "fit": len(out), "restarts": {k: v for k, v in restarts.items() if v},
            "minutes_used": round(max(clocks) / 60.0, 1)}
