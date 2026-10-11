"""E18, the hardware-queue check: the teacher build served with GPU_MAX_HW_QUEUES=1 against the stock spawn, on pod-b.

    python -m lab.sweep_e18 --dry-run [--arms stock,queue1]    # this machine: the cells, the arms, the slot minutes
    python -m lab.sweep_e18 pod-script [--arms ...]            # this machine: writes lab/state/sweep_request/e18_run.sh
    python tooling/dispatcher/pod_exec.py D term <script> <timeout_s> > <output>   # the pod run (workspace tool)
    python -m lab.sweep_e18 collect <output> [--r52 <sweep-request json>] [--metrics-file]   # this machine
    python -m lab.sweep_e18 pod-run --expect-user-sha256 HEX [...]  # on the pod: what the script runs

`python -m lab sweep-e18 ...` is the same. Every call prints one JSON line and exits 0 when it worked.

It is built from R52's serve and bench steps (lab/sweep_request.py): the same grid file, the same pinned launch of the
inference settings sheet's baseline (vLLM b1388b1 plus lab/vllm-gfx1201.diff, the sheet's flags and variables), the same
harness session per cell (TTFT, decode rate, amd-smi watts and joules per token, the conditions with the environment
read back from the server process). The only difference between the two arms is one variable:

    stock   the pinned launch as R52 runs it (GPU_MAX_HW_QUEUES=4, the HIP default written out)
    queue1  the same launch with GPU_MAX_HW_QUEUES=1

Per arm, one cold spawn: idle watts, the wait for /v1/models, one discarded warm-up round per batch size, one probe
request alone (its decode rate and mem_busy_percent, the harness's band inputs), the grid's base cells at batch 1 and 4
with the grid's repeats (serpentine), then the correctness harness's prompt set (lab/correctness.py: greedy, 128 tokens,
token ids and logprobs, no nonce) once or twice, then the stop. The stock arm sends the prompt set twice, so the server's
own repeat-to-repeat difference is measured and a divergence of queue1 can be told from it. The report then holds, per cell,
the median of the repeats with the lowest and highest and the change of queue1 against stock; the token-for-token match
of queue1's outputs with stock's (lab/correctness.py's compare_outputs) beside the stock-against-stock match; and the
band classification of lab/bench.py's classify_bands over the spawns' probe rates (it needs 4 spawns to split, and this
run has 2, so it says so instead of inventing a band).

--arms stock,queue1 (the default) is one slot. --arms stock and --arms queue1 are the same run in two slots: the first
leaves its outputs in ~/sweep_request/e18/stock-outputs.json on the pod, and the second compares with them.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import os
import platform
import shlex
import signal
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

if not __package__:  # run by path: the repo root, one folder up, holds lab/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab import bench, correctness  # noqa: E402
from lab import sweep_request as sr  # noqa: E402
from lab.machine import metrics as metrics_file  # noqa: E402
from lab.machine.record import record, recorded  # noqa: E402

E18 = "lab/sweep_e18.py 1"
RAW_SCHEMA = "atlas.e18_hwqueues.raw/1"
RESULT_SCHEMA = "atlas.e18_hwqueues/1"
REPO = sr.REPO
STATE_DIR = sr.STATE_DIR
POD_SCRIPT = STATE_DIR / "e18_run.sh"
POD_DIR = sr.POD_DIR + "/e18"           # under the pod's home: the runs, the arms' outputs and the last result
ARMS = ("stock", "queue1")
ARM_VARIABLE = {"stock": "4", "queue1": "1"}    # GPU_MAX_HW_QUEUES per arm (stock: the pinned launch's own value)
BATCHES = (1, 4)
SLICE = "base"
PASSES = {"stock": 2, "queue1": 1}
WARM_READY_S = 300.0       # pod C's warm start (297 s, R30); a second spawn finds the compile cache the first wrote
PROMPTS = REPO / "fixtures" / "lab" / "bench" / "prompts.json"
IMAGES = REPO / "fixtures" / "lab" / "bench" / "images"
RATE_SEPARATION_PCT = 5.0  # a median change under this is read as no difference, whatever the ranges say
COMPARED = ("decode_tok_s", "decode_tok_s_total", "ttft_s", "watts_avg", "joules_per_token")
DATA_DIR = sr.DATA_DIR
SWEEP_NAME = "sweep_e18"


def log(message: str) -> None:
    print(f"[e18 {time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


# --- the plan ------------------------------------------------------------------------------------------------------

def e18_cells(grid: sr.Grid) -> list[sr.Cell]:
    """The grid's base cells at batch 1 and 4."""
    cells = [c for c in grid.cells() if c.slice == SLICE and c.batch in BATCHES]
    if [c.batch for c in cells] != list(BATCHES):
        raise sr.SweepError(f"the grid has no {SLICE} cells at batch {', '.join(map(str, BATCHES))}")
    return cells


def pass_seconds(grid: sr.Grid, prompts: int = 4) -> float:
    """One pass of the correctness prompt set by the grid's timing model, every prompt priced as the 1280x720 image."""
    t = grid.timing
    return prompts * (t.ttft_s + 900.0 / t.prefill_tok_s + correctness.DEFAULT_MAX_TOKENS / t.decode_tok_s_b1)


def arm_seconds(grid: sr.Grid, cells: Sequence[sr.Cell], arm: str, *, warm: bool) -> dict[str, float]:
    t = grid.timing
    idle = float(grid.server["idle_s"])
    warm_s = sum(sr.cell_seconds(grid, c, tokens=min(grid.warmup_tokens, c.max_tokens))
                 for c in sr.warmup_shapes(cells))
    probe = sr.cell_seconds(grid, cells[0], tokens=cells[0].max_tokens)
    measure = grid.repeats * sum(sr.cell_seconds(grid, c) for c in cells)
    out = {"ready_s": WARM_READY_S if warm else t.ready_s, "idle_loaded_s": idle, "warmup_s": warm_s,
           "probe_s": probe, "measure_s": measure, "outputs_s": PASSES[arm] * pass_seconds(grid),
           "stop_s": t.stop_s}
    out["total_s"] = sum(out.values())
    return {k: round(v, 1) for k, v in out.items()}


def estimate(grid: sr.Grid, cells: Sequence[sr.Cell], arms: Sequence[str], *, warm_first: bool = False
             ) -> dict[str, Any]:
    """Seconds by arm and the slot minutes, rounded up with the grid's margin. The first spawn of a pod that has not
    served since the vLLM build is cold (the grid's ready_s); a second one in the same run is warm."""
    per_arm = {arm: arm_seconds(grid, cells, arm, warm=warm_first or k > 0) for k, arm in enumerate(arms)}
    total = float(grid.server["idle_s"]) + sum(a["total_s"] for a in per_arm.values())
    minutes = math.ceil(total * (1.0 + grid.timing.margin) / 60.0 / grid.timing.round_to_min) * grid.timing.round_to_min
    return {"arms": per_arm, "idle_s": float(grid.server["idle_s"]), "total_s": round(total, 1),
            "margin": grid.timing.margin, "slot_minutes": int(minutes),
            "warm_total_s": round(float(grid.server["idle_s"]) + sum(
                arm_seconds(grid, cells, arm, warm=True)["total_s"] for arm in arms), 1)}


def arm_launches(grid: sr.Grid, arm: str, *, run_dir: str) -> list[sr.Launch]:
    """R52's launches for the grid with the arm's variable written into each environment."""
    out = []
    for launch in sr.grid_launches(grid, run_dir=run_dir):
        env = dict(launch.env)
        if arm == "queue1" or "GPU_MAX_HW_QUEUES" in env:
            env["GPU_MAX_HW_QUEUES"] = ARM_VARIABLE[arm]
        out.append(dataclasses.replace(launch, env=env))
    return out


def plan(grid: sr.Grid, arms: Sequence[str], *, deadline_min: float | None = None) -> dict[str, Any]:
    cells = e18_cells(grid)
    est = estimate(grid, cells, arms)
    return {"grid": grid.file, "grid_sha256": grid.sha256, "arms": list(arms), "repeats": grid.repeats,
            "order": grid.order, "cells": [dataclasses.asdict(c) for c in cells],
            "passes": {arm: PASSES[arm] for arm in arms}, "estimate": est, "slot_minutes": est["slot_minutes"],
            "deadline_min": deadline_min if deadline_min is not None else est["slot_minutes"],
            "launches": {arm: [launch.describe() for launch in arm_launches(grid, arm, run_dir=grid.server["run_dir"])]
                         for arm in arms}}


def plan_text(p: Mapping[str, Any]) -> str:
    e = p["estimate"]
    lines = [f"E18 on {', '.join(p['arms'])}: cells {', '.join(c['id'] for c in p['cells'])}, {p['repeats']} repeats "
             f"({p['order']}), the correctness prompt set {', '.join(f'{a} x{n}' for a, n in p['passes'].items())}"]
    for arm, s in e["arms"].items():
        lines.append(f"  {arm:<7} ready {s['ready_s']:g} s, warm-up {s['warmup_s']:g} s, probe {s['probe_s']:g} s, "
                     f"cells {s['measure_s']:g} s, outputs {s['outputs_s']:g} s, stop {s['stop_s']:g} s: "
                     f"{s['total_s']:g} s")
    lines.append(f"slot minutes: {p['slot_minutes']} ({e['total_s']:g} s plus {e['margin']:.0%}; "
                 f"{e['warm_total_s']:g} s if every spawn starts warm); deadline {p['deadline_min']:g} min")
    for arm, launches in p["launches"].items():
        for launch in launches:
            lines.append(f"launch {arm}/{launch['name']}: env {json.dumps(launch['env'], sort_keys=True)}")
    return "\n".join(lines)


# --- one arm -------------------------------------------------------------------------------------------------------

def wait_card_free(port: int, seconds: float = 90.0) -> bool:
    """After a stop: no vLLM process and nothing on the port, so the next spawn finds the card as the first did."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if not sr.running_servers() and not bench.port_open(port):
            time.sleep(3.0)
            return True
        time.sleep(2.0)
    return False


def run_arm(arm: str, grid: sr.Grid, cells: Sequence[sr.Cell], *, host: bench.Host, power: Any, out_dir: Path,
            checkpoint: str, engine_dir: str | None, patches: Sequence[str], node: str, port: int, hub: str | None,
            account: str | None, python: str, run_id: str, deadline: float | None,
            launches: Sequence[sr.Launch], idle_no_server: float | None,
            say: Callable[[str], None] = log) -> dict[str, Any]:
    """One cold spawn with the arm's variable: warm-up, probe, the cells, the correctness passes, the stop. Returns a
    raw result in R52's shape (so sweep_request.summarize reads it) plus `probe` and `outputs`; never raises."""
    server = grid.server
    arm_dir = out_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    order = sr.run_order(grid, cells)
    raw: dict[str, Any] = {
        "schema": sr.RAW_SCHEMA, "sweep": E18, "harness": bench.HARNESS, "run_id": f"{run_id}-{arm}", "arm": arm,
        "node": node, "hub": hub, "account": account, "quick": True, "started": bench.utc(), "ended": None,
        "grid": {"file": grid.file, "sha256": grid.sha256, "data": grid.data},
        "cells": [dataclasses.asdict(c) for c in cells], "order": [[r, c.id] for r, c in order],
        "deadline_s": None, "estimate": {},
        "launch": {"served": None, "tried": [], "launches": [launch.describe() for launch in launches]},
        "spawn": None, "idle_w": {"no_server": idle_no_server, "loaded": None},
        "power": {"available": bool(power.available), "reason": power.reason, "period_s": power.period_s},
        "warmup": [], "rows": [], "probe": None, "outputs": [], "output_errors": [], "error": None,
        "run_dir": str(arm_dir), "skipped": None}
    spawned = endpoint = None
    spawn: dict[str, Any] = {}
    try:
        values = {"port": port, "batch": max(c.batch for c in cells), "max_len": server["max_len"],
                  "model": server["model"], "checkpoint": checkpoint}
        spawned, endpoint, launch, argv, log_path = sr.start_server(
            launches, values=values, run_dir=arm_dir, port=port, server=server, deadline=deadline,
            tried=raw["launch"]["tried"], say=say)
        raw["launch"]["served"] = launch.name
        spawn = {"index": 0, "pid": spawned.pid, "cold": True, "attached": False, "spawned": bench.utc(spawned.spawned),
                 "spawned_t": spawned.spawned, "ready_s": sr._r(spawned.ready - spawned.spawned, 3), "stopped": False,
                 "log": str(log_path), "shared": True, "command": argv, "launch": launch.name}
        raw["spawn"] = {k: v for k, v in spawn.items() if k != "spawned_t"}
        log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else None
        raw["idle_w"]["loaded"] = bench.idle_watts(power, float(server["idle_s"]))
        base = bench.Config(
            engine="vllm", checkpoint=checkpoint, model=server["model"], max_len=server["max_len"],
            engine_dir=engine_dir, patches=list(patches), launch_env=dict(launch.env), node=node, port=port,
            add_64=False, repeats=1, share_spawn=True, warmup_rounds=0, probe=False, idle_s=0.0,
            power_period_s=power.period_s, request_timeout_s=float(server["request_timeout_s"]),
            stream_grace_s=float(server["stream_grace_s"]), ready_timeout_s=float(server["ready_timeout_s"]),
            out_dir=arm_dir / "cells", python=python, hub=hub, account=account, record=False)
        sweep = {"run_id": raw["run_id"], "launch": launch.name, "grid_sha256": grid.sha256, "shared_spawn": True,
                 "baseline_cell": grid.baseline_cell, "experiment": "E18", "arm": arm,
                 "gpu_max_hw_queues_meant": ARM_VARIABLE[arm] if arm == "queue1" or "GPU_MAX_HW_QUEUES" in launch.env
                 else None}
        runs = {c.id: sr.prepare_cell(grid, c, base, host=host, power=power, endpoint=endpoint, pid=spawned.pid,
                                      log_text=log_text, sweep=sweep) for c in cells}
        raw["warmup"] = sr.warm_up(endpoint, [runs[c.id] for c in cells], tokens=grid.warmup_tokens, server=server,
                                   say=say)
        first = runs[cells[0].id]
        raw["probe"] = first.session.run_probe(endpoint, f"{raw['run_id']}-probe")
        say(f"{arm}: probe decode {raw['probe'].get('decode_tok_s')} tok/s, mem_busy "
            f"{(raw['probe'].get('mem_busy_pct') or {}).get('median')}")
        idle = dict(raw["idle_w"])
        stuck = None
        for k, (repeat, cell) in enumerate(order):
            run = runs[cell.id]
            if stuck is not None:
                row = run.session.blank_row(repeat=repeat, batch=cell.batch, spawn=spawn, conditions=run.conditions)
                row["outcome"], row["reason"] = "fail", f"not measured: {stuck}"
            else:
                need = run.seconds[-1] if run.seconds else sr.cell_seconds(grid, cell)
                left = None if deadline is None else deadline - grid.timing.stop_s - time.monotonic()
                if left is not None and need > left:
                    row = run.session.blank_row(repeat=repeat, batch=cell.batch, spawn=spawn,
                                                conditions=run.conditions)
                    row["outcome"] = "skipped"
                    row["reason"] = (f"not run: past the deadline (the cell takes about {need:.0f} s and "
                                     f"{max(left, 0.0):.0f} s were left)")
                else:
                    say(f"{arm} {k + 1}/{len(order)}: {cell.id}, repeat {repeat}")
                    began = time.monotonic()
                    try:
                        row = run.session.measure(endpoint, bench.PlannedBatch(cell.batch), repeat=repeat,
                                                  spawn=spawn, conditions=run.conditions, idle=idle, probe_now=False)
                    except bench.StuckRun as exc:
                        row = exc.row
                        stuck = f"{row['run_id']} ({cell.id}, repeat {repeat}) had stuck streams: {row['reason']}"
                    run.seconds.append(time.monotonic() - began)
                    row["elapsed_s"] = sr._r(run.seconds[-1], 3)
            row["cell"], row["order"] = cell.id, k
            run.session.keep(row)
            raw["rows"].append(row)
        if stuck is not None:
            raw["error"] = stuck
        else:
            prompts = bench.PromptSet.load(PROMPTS)
            settings = bench.Settings(model=server["model"], max_tokens=correctness.DEFAULT_MAX_TOKENS,
                                      ignore_eos=True, thinking=False, prefix_caching=False)
            need = pass_seconds(grid, len(prompts.prompts))
            for n in range(PASSES[arm]):
                left = None if deadline is None else deadline - grid.timing.stop_s - time.monotonic()
                if left is not None and need > left:
                    raw["output_errors"].append(f"pass {n + 1} not run: past the deadline ({left:.0f} s left)")
                    break
                say(f"{arm}: correctness pass {n + 1} of {PASSES[arm]}")
                outputs = correctness.run_prompts(endpoint, prompts, settings)
                raw["outputs"].append({"pass": n + 1, "prompt_set": prompts.describe(), "outputs": outputs})
                bad = [f"{o['prompt']}: {o['error']}" for o in outputs if o["error"]]
                if bad:
                    raw["output_errors"].append(f"pass {n + 1}: " + "; ".join(bad))
    except (sr.SweepError, bench.BenchError, correctness.CorrectnessError) as exc:
        raw["error"] = str(exc)
    except (KeyboardInterrupt, SystemExit) as exc:
        raw["error"] = f"interrupted ({type(exc).__name__}: {exc})"
        raise
    finally:
        if endpoint is not None:
            endpoint.close()
        if spawned is not None:
            stopped = spawned.stop()
            if raw["spawn"] is not None:
                raw["spawn"]["stopped"] = stopped
        raw["ended"] = bench.utc()
        for row in raw["rows"]:
            row["spawn"] = dict(raw["spawn"] or {})
        (arm_dir / "raw.json").write_text(json.dumps(raw, default=str), encoding="utf-8")
    return raw


def run_e18(grid: sr.Grid, arms: Sequence[str], *, host: bench.Host, power: Any, out_dir: str | Path, checkpoint: str,
            engine_dir: str | None = None, patches: Sequence[str] = (), node: str | None = None,
            port: int | None = None, hub: str | None = None, account: str | None = None, python: str = "python3",
            deadline_s: float | None = None, idle_s: float | None = None, reference: Path | None = None,
            outputs_dir: Path | None = None, launches: Mapping[str, Sequence[sr.Launch]] | None = None,
            say: Callable[[str], None] = log) -> dict[str, Any]:
    """The arms one after the other, each on its own cold spawn, then the raw result (also <run dir>/raw.json).
    reference is where an earlier run left the stock arm's outputs, for a run of queue1 alone; launches replaces the
    grid's launches per arm (the tests' fake servers)."""
    server = grid.server
    node = node or server["node"]
    port = server["port"] if port is None else port
    cells = e18_cells(grid)
    deadline = None if deadline_s is None else time.monotonic() + deadline_s
    run_id = "e18-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + os.urandom(2).hex()
    run_dir = Path(out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    raw: dict[str, Any] = {
        "schema": RAW_SCHEMA, "harness": E18, "run_id": run_id, "node": node, "hub": hub, "account": account,
        "started": bench.utc(), "ended": None, "arms_asked": list(arms),
        "grid": {"file": grid.file, "sha256": grid.sha256, "data": grid.data}, "cells": [c.id for c in cells],
        "estimate": estimate(grid, cells, arms), "deadline_s": deadline_s, "arms": {}, "reference": None,
        "error": None, "run_dir": str(run_dir)}
    idle_s = float(server["idle_s"]) if idle_s is None else idle_s
    power.start()
    try:
        idle_no_server = bench.idle_watts(power, idle_s)
        raw["idle_w"] = {"no_server": idle_no_server}
        for k, arm in enumerate(arms):
            if k > 0:
                if not wait_card_free(port):
                    raw["error"] = f"the {arms[k - 1]} server did not leave the card; {arm} was not started"
                    break
                need = arm_seconds(grid, cells, arm, warm=True)["total_s"]
                left = None if deadline is None else deadline - time.monotonic()
                if left is not None and need > left:
                    raw["arms"][arm] = {"skipped": f"not started: it takes about {need:.0f} s and {max(left, 0):.0f} s "
                                                   "were left"}
                    break
            result = run_arm(arm, grid, cells, host=host, power=power, out_dir=run_dir, checkpoint=checkpoint,
                             engine_dir=engine_dir, patches=patches, node=node, port=port, hub=hub, account=account,
                             python=python, run_id=run_id, deadline=deadline, idle_no_server=idle_no_server, say=say,
                             launches=(launches or {}).get(arm) or arm_launches(
                                 grid, arm, run_dir=os.path.expanduser(server["run_dir"])))
            raw["arms"][arm] = result
            if outputs_dir is not None and result["outputs"]:
                outputs_dir.mkdir(parents=True, exist_ok=True)
                (outputs_dir / f"{arm}-outputs.json").write_text(
                    json.dumps({"run_id": result["run_id"], "arm": arm, "passes": result["outputs"]}, default=str),
                    encoding="utf-8")
            if result["error"]:
                raw["error"] = f"{arm}: {result['error']}"
                break
        if "stock" not in raw["arms"] and reference is not None:
            try:
                ref = json.loads(reference.read_text(encoding="utf-8"))
                raw["reference"] = {"file": str(reference), "run_id": ref.get("run_id"), "arm": ref.get("arm"),
                                    "passes": ref.get("passes")}
            except (OSError, ValueError) as exc:
                raw["reference"] = {"file": str(reference), "error": f"{type(exc).__name__}: {exc}"}
    finally:
        power.stop()
        raw["ended"] = bench.utc()
        (run_dir / "raw.json").write_text(json.dumps(raw, default=str), encoding="utf-8")
    return raw


# --- the summary ---------------------------------------------------------------------------------------------------

def env_read_back(arm_summary: Mapping[str, Any]) -> str | None:
    """GPU_MAX_HW_QUEUES as the harness read it from the running server's /proc/<pid>/environ (None: absent)."""
    env = (arm_summary.get("server") or {}).get("env") or {}
    return env.get("GPU_MAX_HW_QUEUES")


def compare_cell(stock: Mapping[str, Any], queue1: Mapping[str, Any]) -> dict[str, Any]:
    """One cell, queue1 against stock: per metric both medians with ranges, the change and whether the two ranges are
    apart."""
    out: dict[str, Any] = {"cell": stock["id"], "batch": stock["batch"], "n": {"stock": stock["n"],
                                                                                "queue1": queue1["n"]}}
    for m in COMPARED:
        a, b = stock["stats"][m], queue1["stats"][m]
        entry = {"stock": a, "queue1": b, "change_pct": sr.change_pct(b["median"], a["median"]),
                 "range_pct": None, "ranges_apart": None}
        if None not in (a["median"], b["median"], a["min"], a["max"], b["min"], b["max"]) and a["median"]:
            entry["range_pct"] = [sr.change_pct(b["min"], a["max"]), sr.change_pct(b["max"], a["min"])]
            entry["ranges_apart"] = b["min"] > a["max"] or b["max"] < a["min"]
        out[m] = entry
    return out


def reading(compared: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The E18 reading from the batch-1 cell's per-stream decode rate: b_ahead (queue1 faster), a_ahead (stock
    faster) or no_difference, with the change and why. One spawn per arm cannot show a band split; it can show that
    the two spawns differ by more than their repeats do."""
    b1 = next((c for c in compared if c["batch"] == 1), None)
    if b1 is None or b1["decode_tok_s"]["change_pct"] is None:
        return {"reading": "not_measured", "why": "a batch 1 cell has no value in one of the arms"}
    change, apart = b1["decode_tok_s"]["change_pct"], b1["decode_tok_s"]["ranges_apart"]
    if abs(change) < RATE_SEPARATION_PCT:
        word = "no_difference"
    else:
        word = ("b_ahead" if change > 0 else "a_ahead") if apart else "no_difference"
    return {"reading": word, "decode_change_pct": change, "ranges_apart": apart,
            "joules_per_token_change_pct": b1["joules_per_token"]["change_pct"],
            "why": f"batch 1 decode per stream {change:+.1f}% for queue1; "
                   f"repeats' ranges {'apart' if apart else 'overlap'}; a change under {RATE_SEPARATION_PCT:g}% "
                   "reads as no difference. Two spawns are low confidence (no band can be drawn from them)."}


def spawn_bands(arms: Mapping[str, Any], extra: Mapping[str, float] | None = None) -> dict[str, Any]:
    """lab/bench.py's band classification over the spawns' probe decode rates (and mem_busy_percent). It splits only
    from 4 spawns; with fewer it returns one group and says why."""
    rates: dict[str, float] = {}
    busy: dict[str, float | None] = {}
    for arm, raw in arms.items():
        probe = raw.get("probe") or {}
        if probe.get("decode_tok_s") is not None:
            rates[arm] = float(probe["decode_tok_s"])
            busy[arm] = (probe.get("mem_busy_pct") or {}).get("median")
    for name, rate in (extra or {}).items():
        rates[name] = rate
        busy[name] = None
    bands = bench.classify_bands(rates)
    rule = bench.band_rule(bands, rates, busy) if bands else {"clusters": [], "boundaries": []}
    return {"spawns": rates, "mem_busy_pct": busy, "bands": bands, "rule": rule,
            "classified": len(rates) >= bench.MIN_SPAWNS_TO_BAND,
            "note": (f"{len(rates)} spawns with a probe rate; the harness splits bands from {bench.MIN_SPAWNS_TO_BAND}, "
                     "so this is one group by construction, not a finding that there is one band"
                     if len(rates) < bench.MIN_SPAWNS_TO_BAND else "classified by the harness's rule")}


def output_comparison(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Queue1's first pass against each stock pass (token for token), and stock's second pass against its first."""
    stock_passes = None
    source = None
    if "stock" in (raw.get("arms") or {}) and raw["arms"]["stock"].get("outputs"):
        stock_passes = raw["arms"]["stock"]["outputs"]
        source = {"from": "this run", "run_id": raw["arms"]["stock"]["run_id"]}
    elif (raw.get("reference") or {}).get("passes"):
        stock_passes = raw["reference"]["passes"]
        source = {"from": "an earlier run's file", "run_id": raw["reference"].get("run_id")}
    q1 = ((raw.get("arms") or {}).get("queue1") or {}).get("outputs") or []
    out: dict[str, Any] = {"stock_source": source, "stock_passes": len(stock_passes or []),
                           "queue1_passes": len(q1), "stock_vs_stock": None, "queue1_vs_stock": []}
    if stock_passes and len(stock_passes) >= 2:
        rows = correctness.compare_outputs(stock_passes[0]["outputs"], stock_passes[1]["outputs"])
        out["stock_vs_stock"] = {"summary": correctness.summarize(rows), "prompts": _trim(rows)}
    if stock_passes and q1:
        for k, one in enumerate(stock_passes):
            rows = correctness.compare_outputs(one["outputs"], q1[0]["outputs"])
            out["queue1_vs_stock"].append({"against_stock_pass": k + 1, "summary": correctness.summarize(rows),
                                           "prompts": _trim(rows)})
    floor = (out["stock_vs_stock"] or {}).get("summary", {}).get("match_rate")
    best = max((c["summary"]["match_rate"] for c in out["queue1_vs_stock"]), default=None)
    first = out["queue1_vs_stock"][0]["summary"]["match_rate"] if out["queue1_vs_stock"] else None
    out["unchanged"] = None if first is None else bool(first == 1.0 or (floor is not None and first >= floor))
    out["identical"] = None if first is None else first == 1.0
    out["why"] = ("queue1 was not compared: " + ("no queue1 outputs" if not q1 else "no stock outputs")
                  if first is None else
                  f"queue1 matched stock's first pass on {out['queue1_vs_stock'][0]['summary']['matches']} of "
                  f"{out['queue1_vs_stock'][0]['summary']['prompts']} prompts"
                  + (f"; stock matched itself on {out['stock_vs_stock']['summary']['matches']} of "
                     f"{out['stock_vs_stock']['summary']['prompts']}" if out["stock_vs_stock"] else
                     "; there is no stock-against-stock pass to set a floor"))
    out["best_match_rate"] = best
    return out


def _trim(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    keys = ("prompt", "tokens_match", "first_divergence", "baseline_tokens", "run_tokens", "logprob_difference",
            "text_normalised_equal")
    return [{k: r[k] for k in keys} for r in rows]


def summarize(raw: Mapping[str, Any], r52: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The E18 result, pure: each arm through R52's summarize, queue1 against stock per cell, the reading, the output
    comparison, the bands, and what stays open. r52 is a collected R52 result: its base cells at the same batches
    stand beside the arms as one more stock spawn."""
    if raw.get("schema") != RAW_SCHEMA:
        raise sr.SweepError(f"not an E18 result: schema {raw.get('schema')!r}")
    arms = raw.get("arms") or {}
    per_arm = {arm: sr.summarize(a) for arm, a in arms.items() if "rows" in a}
    skipped = {arm: a["skipped"] for arm, a in arms.items() if "rows" not in a}
    compared = []
    if "stock" in per_arm and "queue1" in per_arm:
        by_id = {c["id"]: c for c in per_arm["queue1"]["cells"]}
        compared = [compare_cell(c, by_id[c["id"]]) for c in per_arm["stock"]["cells"] if c["id"] in by_id]
    outputs = output_comparison(raw)
    extra = {}
    r52_cells = {}
    if r52:
        r52_cells = {c["id"]: c for c in r52.get("cells", []) if c["id"] in raw.get("cells", [])}
        one = r52_cells.get("base-b1")
        if one and one["stats"]["decode_tok_s"]["median"] is not None:
            extra["r52_stock"] = float(one["stats"]["decode_tok_s"]["median"])
    bands = spawn_bands(arms, extra)
    open_items = []
    if raw.get("error"):
        open_items.append(f"the run ended early: {raw['error']}")
    for arm, why in skipped.items():
        open_items.append(f"{arm}: {why}")
    for arm, s in per_arm.items():
        open_items += [f"{arm}: {item}" for item in s["open"]]
        want = ARM_VARIABLE[arm] if arm == "queue1" else None
        got = env_read_back(s)
        if arm == "queue1" and got != "1":
            open_items.append(f"queue1: the server process environment reads GPU_MAX_HW_QUEUES={got!r}, not '1'; "
                              "the arm is not valid")
        if arm == "stock" and got not in ("4", None):
            open_items.append(f"stock: the server process environment reads GPU_MAX_HW_QUEUES={got!r}, not the "
                              "default 4")
        raw_arm = arms[arm]
        open_items += [f"{arm}: {e}" for e in raw_arm.get("output_errors") or []]
    read = reading(compared) if compared else {"reading": "not_measured", "why": "both arms are needed"}
    if not outputs["queue1_vs_stock"] and "queue1" in per_arm:
        open_items.append("the outputs of queue1 were not compared: " + outputs["why"])
    open_items.append("two spawns (one per arm) are low confidence and show no band split: the harness classifies "
                      f"bands from {bench.MIN_SPAWNS_TO_BAND} spawns, and the owner's ruling 6 wants E18 to read "
                      "b_ahead before the teacher adopts the variable")
    started = str(raw.get("started") or "")
    sample = per_arm.get("queue1") or per_arm.get("stock") or {}
    return {
        "schema": RESULT_SCHEMA, "sweep": E18, "run_id": raw.get("run_id"), "node": raw.get("node"),
        "hub": raw.get("hub"), "account": raw.get("account"), "chip": sample.get("chip") or sr.CHIP,
        "date": started[:10], "started": raw.get("started"), "ended": raw.get("ended"),
        "arms_asked": raw.get("arms_asked"), "grid": {"file": raw["grid"].get("file"), "sha256": raw["grid"].get("sha256")},
        "estimate": raw.get("estimate"), "error": raw.get("error"), "idle_w": raw.get("idle_w"),
        "arms": {arm: {"summary": s, "env_gpu_max_hw_queues": env_read_back(s),
                       "probe": {k: (arms[arm].get("probe") or {}).get(k) for k in ("decode_tok_s", "ttft_s",
                                                                                    "mem_busy_pct", "reason")},
                       "spawn": s.get("spawn"), "launch": s["launch"]["served"], "idle_w": s.get("idle_w")}
                 for arm, s in per_arm.items()},
        "compared": compared, "reading": read, "outputs": outputs, "bands": bands,
        "r52_stock": {k: {m: v["stats"][m] for m in COMPARED} for k, v in r52_cells.items()} or None,
        "open": open_items}


def _f(value: float | None, digits: int = 2) -> str:
    return sr._fmt(value, digits)


def headline_lines(summary: Mapping[str, Any]) -> list[str]:
    lines = [f"e18 {summary['run_id']}: arms {', '.join(summary['arms']) or 'none'}"
             + (f", error: {summary['error']}" if summary.get("error") else "")]
    for c in summary["compared"]:
        d, j = c["decode_tok_s"], c["joules_per_token"]
        lines.append(f"{c['cell']}: decode {_f(d['stock']['median'])} -> {_f(d['queue1']['median'])} tok/s per stream "
                     f"({sr._pct(d['change_pct'])}), {_f(j['stock']['median'])} -> {_f(j['queue1']['median'])} J/token "
                     f"({sr._pct(j['change_pct'])})")
    lines.append(f"reading: {summary['reading']['reading']}; outputs: {summary['outputs']['why']}")
    return lines


def render_markdown(summary: Mapping[str, Any]) -> str:
    arms = summary["arms"]
    head = [f"# E18, GPU_MAX_HW_QUEUES=1 against stock on {summary['chip']} ({summary['node']}), {summary['date']}", "",
            f"Run `{summary['run_id']}` from {summary['started']} to {summary['ended']}: one cold spawn per arm of the "
            "same build and launch (R52's pinned launch), the stock arm with GPU_MAX_HW_QUEUES=4 as the pinned launch "
            "writes it, the queue1 arm with 1; the base cells at batch 1 and 4, three repeats each in serpentine order, "
            "then the correctness prompt set (greedy, 128 tokens). Values are medians of the repeats with the lowest and "
            f"highest in brackets. Grid `{summary['grid']['file']}` (sha256 {str(summary['grid']['sha256'])[:12]}).", ""]
    for arm, a in arms.items():
        probe = a["probe"]
        head.append(f"- {arm}: launch {a['launch']}, ready after {(a['spawn'] or {}).get('ready_s')} s, GPU_MAX_HW_QUEUES "
                    f"read back from the server process: {a['env_gpu_max_hw_queues']!r}; probe (one request alone) "
                    f"{_f(probe['decode_tok_s'])} tok/s, mem_busy_percent median "
                    f"{(probe.get('mem_busy_pct') or {}).get('median')}; idle {(a['idle_w'] or {}).get('no_server')} W "
                    f"with no server, {(a['idle_w'] or {}).get('loaded')} W loaded.")
    table = ["", "| Cell | Metric | Stock | Queue1 | Change | Repeat ranges |", "|---|---|---|---|---|---|"]
    names = {"decode_tok_s": ("Decode tok/s per stream", 2), "decode_tok_s_total": ("Decode tok/s total", 2),
             "ttft_s": ("TTFT s", 2), "watts_avg": ("Watts avg", 1), "joules_per_token": ("J/token", 2)}
    for c in summary["compared"]:
        for m in COMPARED:
            e = c[m]
            apart = "-" if e["ranges_apart"] is None else ("apart" if e["ranges_apart"] else "overlap")
            table.append(f"| `{c['cell']}` | {names[m][0]} | {sr._cell_value(e['stock'], names[m][1])} | "
                         f"{sr._cell_value(e['queue1'], names[m][1])} | {sr._pct(e['change_pct'])} | {apart} |")
    r = summary["reading"]
    out = summary["outputs"]
    tail = ["", f"Reading: **{r['reading']}**. {r.get('why', '')}".replace("**", ""), ""]
    tail.append("Outputs, token for token (lab/correctness.py compare_outputs, greedy, 128 tokens, 4 prompts):")
    if out["stock_vs_stock"]:
        s = out["stock_vs_stock"]["summary"]
        tail.append(f"- stock pass 2 against stock pass 1 (the server's own floor): {s['matches']} of {s['prompts']} match"
                    + (f", first divergence {s['first_divergence_prompt']} at token {s['first_divergence_index']}"
                       if s["divergences"] else ""))
    for c in out["queue1_vs_stock"]:
        s = c["summary"]
        lp = [p["logprob_difference"] for p in c["prompts"] if not p["tokens_match"]]
        tail.append(f"- queue1 against stock pass {c['against_stock_pass']}: {s['matches']} of {s['prompts']} match"
                    + (f", first divergence {s['first_divergence_prompt']} at token {s['first_divergence_index']}, "
                       f"logprob differences {lp}" if s["divergences"] else ""))
    if not out["queue1_vs_stock"]:
        tail.append(f"- not compared: {out['why']}")
    tail += ["", "Bands (lab/bench.py classify_bands over the spawns' probe decode rates): "
             + ", ".join(f"{k} {v:.2f} tok/s" for k, v in summary["bands"]["spawns"].items()) + f". {summary['bands']['note']}."]
    if summary.get("r52_stock"):
        tail.append("R52's stock spawn, same cells: " + "; ".join(
            f"`{k}` decode {_f(v['decode_tok_s']['median'])} tok/s, {_f(v['joules_per_token']['median'])} J/token"
            for k, v in summary["r52_stock"].items()) + ".")
    if summary["open"]:
        tail += ["", "Open:"] + [f"- {item}" for item in summary["open"]]
    return "\n".join(head + table + tail) + "\n"


# --- rows ----------------------------------------------------------------------------------------------------------

def ledger_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One `sweep_e18.cell` row per arm and cell and one `sweep_e18.outputs` row, all on the run's node, low_confidence."""
    rows = []
    for arm, a in summary["arms"].items():
        s = a["summary"]
        for cell in s["cells"]:
            metrics = {f"teacher.{m}": sr._r(cell["stats"][m]["median"], 6) for m in sr.METRICS
                       if cell["stats"][m]["median"] is not None}
            conditions = {**sr._row_conditions(s, cell), "experiment": "E18", "arm": arm,
                          "gpu_max_hw_queues": a["env_gpu_max_hw_queues"], "probe": a["probe"]}
            rows.append({"verb": "sweep_e18.cell", "outcome": cell["outcome"], "metrics": metrics,
                         "refs": {"run_id": f"{summary['run_id']}/{arm}/{cell['id']}"}, "node": summary["node"],
                         "units": {k: sr.METRICS[k.split(".", 1)[1]][0] for k in metrics} or None,
                         "conditions": conditions, "low_confidence": True, "source": f"{E18} {summary['run_id']}",
                         "note": "; ".join(cell["reasons"])[:480] if cell["outcome"] != "ok" else None})
    out = summary["outputs"]
    if out["queue1_vs_stock"]:
        s = out["queue1_vs_stock"][0]["summary"]
        metrics = {"teacher.output_matches": s["matches"], "teacher.output_divergences": s["divergences"],
                   "teacher.output_prompts": s["prompts"], "teacher.output_match_rate": s["match_rate"]}
        floor = (out["stock_vs_stock"] or {}).get("summary")
        rows.append({"verb": "sweep_e18.outputs", "outcome": "ok" if s["divergences"] == 0 else "diverged",
                     "metrics": metrics, "refs": {"run_id": summary["run_id"]}, "node": summary["node"],
                     "units": None, "low_confidence": True, "source": f"{E18} {summary['run_id']}",
                     "conditions": {"experiment": "E18", "compared": "queue1 against stock, greedy, 128 tokens",
                                    "stock_floor_match_rate": floor["match_rate"] if floor else None,
                                    "reading": summary["reading"]["reading"], "grid_sha256": summary["grid"]["sha256"]},
                     "note": out["why"][:480]})
    return rows


def metric_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Both arms' batch 1 headline medians (arm and the read-back variable in the conditions), the change of queue1
    over stock at batch 1 as derived rows, and the output match rate."""
    rows: list[dict[str, Any]] = []
    ts = sr._metrics_ts(summary.get("ended"))
    source = f"{E18} {summary['run_id']}"
    why = "one cold spawn per arm: fewer than the 5 spawns per arm that verdict() needs to call a difference"
    for arm, a in summary["arms"].items():
        cell = next((c for c in a["summary"]["cells"] if c["batch"] == 1), None)
        if cell is None:
            continue
        for m in sr.HEADLINE:
            s = cell["stats"][m]
            if s["median"] is None:
                continue
            rows.append({"metric": f"teacher.{m}", "value": round(s["median"], 4), "unit": sr.METRICS[m][0],
                         "node": summary["node"], "source": source, "ts": ts, "low_confidence": True,
                         "conditions": {**sr._row_conditions(a["summary"], cell), "experiment": "E18", "arm": arm,
                                        "gpu_max_hw_queues": a["env_gpu_max_hw_queues"],
                                        "low_confidence_reason": why}})
    b1 = next((c for c in summary["compared"] if c["batch"] == 1), None)
    if b1:
        for m in ("decode_tok_s", "joules_per_token"):
            if b1[m]["change_pct"] is not None:
                rows.append({"metric": f"teacher.{m}_change_pct", "value": b1[m]["change_pct"], "unit": "%",
                             "node": summary["node"], "source": source, "ts": ts, "low_confidence": True,
                             "conditions": {"derived": True, "experiment": "E18", "run_id": summary["run_id"],
                                            "from": "queue1 (GPU_MAX_HW_QUEUES=1)", "against": "stock",
                                            "cell": b1["cell"], "range_pct": b1[m]["range_pct"],
                                            "reading": summary["reading"]["reading"], "low_confidence_reason": why}})
    out = summary["outputs"]
    if out["queue1_vs_stock"]:
        s = out["queue1_vs_stock"][0]["summary"]
        rows.append({"metric": "teacher.output_match_rate", "value": s["match_rate"], "unit": "fraction",
                     "node": summary["node"], "source": source, "ts": ts, "low_confidence": True,
                     "conditions": {"experiment": "E18", "run_id": summary["run_id"], "compared": "queue1 against stock",
                                    "prompts": s["prompts"], "stock_floor_match_rate":
                                    ((out["stock_vs_stock"] or {}).get("summary") or {}).get("match_rate"),
                                    "low_confidence_reason": "one compare of 4 prompts"}})
    return rows


# --- the pod script ------------------------------------------------------------------------------------------------

def bundle_files(grid: sr.Grid, root: Path = REPO) -> dict[str, bytes]:
    """R52's bundle plus this module's closure (correctness included), the correctness prompt set and its images."""
    files = dict(sr.bundle_files(grid, root))
    paths = list(sr.module_closure("lab.sweep_e18", root))
    paths.append(root / "fixtures" / "lab" / "bench" / "prompts.json")
    paths += sorted((root / "fixtures" / "lab" / "bench" / "images").glob("*.png"))
    for path in paths:
        if not path.is_file():
            raise sr.SweepError(f"{path.relative_to(root).as_posix()} is not in this checkout")
        files[path.relative_to(root).as_posix()] = path.read_bytes()
    return dict(sorted(files.items()))


def pod_script(grid: sr.Grid, *, arms: Sequence[str], expect_user_sha256: str, deadline_min: float, hub: str | None,
               account: str, node: str, root: Path = REPO) -> tuple[str, dict[str, Any]]:
    """The bash script the pod runs: R52's shape (the bundle as one base64 block checked by sha256, unpacked under
    ~/sweep_request/bundle-<sha>, then pod-run), with this module. No tab and no exclamation mark."""
    import hashlib

    files = bundle_files(grid, root)
    payload = sr.make_bundle(files)
    digest = hashlib.sha256(payload).hexdigest()
    args = ["pod-run", "--expect-user-sha256", expect_user_sha256, "--deadline-min", f"{deadline_min:g}",
            "--node", node, "--account", account, "--arms", ",".join(arms), *(["--hub", hub] if hub else [])]
    script = "\n".join([
        "# E18, the hardware-queue check, written by python -m lab.sweep_e18 pod-script; run with bash.",
        "# It unpacks the lab files below, runs the arms, and prints the result as one marked base64 block.",
        "set -u",
        f'B="$HOME/{sr.POD_DIR}/bundle-{digest[:12]}"',
        'mkdir -p "$B"',
        "base64 -d > \"$B/bundle.tar.gz\" <<'SWEEPBUNDLE'",
        *sr.b64_lines(payload),
        "SWEEPBUNDLE",
        f'echo "{digest}  $B/bundle.tar.gz" | sha256sum -c --quiet - '
        '|| { echo "the bundle was damaged in transit"; echo "### END rc=3"; exit 3; }',
        'tar -xzf "$B/bundle.tar.gz" -C "$B" || { echo "### END rc=3"; exit 3; }',
        'python3 -c "import httpx" 2>/dev/null || python3 -m pip install --user --quiet httpx',
        'cd "$B" || { echo "### END rc=3"; exit 3; }',
        "python3 -m lab.sweep_e18 " + " ".join(shlex.quote(a) for a in args),
        "rc=$?",
        'echo "### END rc=$rc"',
        "exit $rc",
        ""])
    if "\t" in script or "!" in script:
        raise sr.SweepError("the pod script would carry a tab or an exclamation mark")
    return script, {"files": sorted(files), "bundle_bytes": len(payload), "bundle_sha256": digest,
                    "script_bytes": len(script.encode("ascii")), "pod_args": args}


def result_files(raw: Mapping[str, Any]) -> dict[str, bytes]:
    """raw.json (the arms' outputs included) and the head and tail of each arm's serve logs."""
    files = {"raw.json": json.dumps(raw, separators=(",", ":"), default=str).encode("utf-8")}
    for path in sorted(Path(raw.get("run_dir") or ".").glob("*/serve-*.log")):
        files[f"{path.parent.name}-{path.name}"] = sr._head_tail(
            path.read_text(encoding="utf-8", errors="replace")).encode("utf-8")
    return files


def report(raw: Mapping[str, Any]) -> str:
    try:
        lines = headline_lines(summarize(raw))
    except Exception as exc:  # the block must come back even when the summary cannot be made on the pod
        lines = [f"no summary on the pod: {type(exc).__name__}: {exc}"]
    return "\n".join(lines) + "\n" + sr.bundle_text(sr.make_bundle(result_files(raw)))


# --- verbs ---------------------------------------------------------------------------------------------------------

def parse_arms(text: str) -> list[str]:
    arms = [a.strip() for a in text.split(",") if a.strip()]
    if not arms or any(a not in ARMS for a in arms) or len(set(arms)) != len(arms):
        raise sr.SweepError(f"--arms is a comma list of {', '.join(ARMS)}, each at most once, not {text!r}")
    return [a for a in ARMS if a in arms]    # stock first: its outputs are the reference


def dry_run_main(args: argparse.Namespace) -> dict[str, Any]:
    p = plan(sr.load_grid(args.grid), parse_arms(args.arms), deadline_min=args.deadline_min)
    print(plan_text(p), file=sys.stderr, flush=True)
    return {"ok": True, "outcome": "dry_run", **p}


def pod_script_main(args: argparse.Namespace) -> dict[str, Any]:
    grid = sr.load_grid(args.grid)
    arms = parse_arms(args.arms)
    p = plan(grid, arms, deadline_min=args.deadline_min)
    letter = str(args.account or grid.server["account"]).upper()
    if args.expect_user_sha256:
        expect, hub_url = args.expect_user_sha256, args.hub
    else:
        from lab import hub as hub_module
        try:
            account = hub_module.load_accounts([letter])[letter]
        except hub_module.MissingCredentials as exc:
            raise sr.SweepError(f"account {letter} has no user name in the environment or the .env files: {exc}") from exc
        expect, hub_url = sr.user_sha256(account.user), args.hub or account.url
    script, info = pod_script(grid, arms=arms, expect_user_sha256=expect, deadline_min=p["deadline_min"],
                              hub=hub_url, account=letter, node=grid.server["node"])
    out = Path(args.out)
    timeout = int(p["deadline_min"] * 60 + 300)
    answer = {**info, "slot_minutes": p["slot_minutes"], "deadline_min": p["deadline_min"], "arms": arms,
              "pod_command": f"python tooling/dispatcher/pod_exec.py {letter} term {out} {timeout} > "
                             f"{out.with_suffix('.out')}",
              "collect": f"python -m lab sweep-e18 collect {out.with_suffix('.out')} --metrics-file"}
    if args.dry_run:
        print(plan_text(p), file=sys.stderr, flush=True)
        return {"ok": True, "outcome": "dry_run", "script": None, **answer}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(script.encode("ascii"))
    return {"ok": True, "outcome": "written", "script": str(out), **answer}


def pod_run_main(args: argparse.Namespace, *, host: bench.Host | None = None, power: Any = None) -> dict[str, Any]:
    grid = sr.load_grid(args.grid)
    server = grid.server
    arms = parse_arms(args.arms)
    node = args.node or server["node"]
    port = server["port"] if args.port is None else args.port
    problems = sr.gate(node=node, expect_user_sha256=args.expect_user_sha256, environ=os.environ,
                       rocminfo=sr.read_rocminfo(), machine=platform.machine(), port=port,
                       port_free=not bench.port_open(port), servers=sr.running_servers(), want_node=server["node"])
    if problems:
        return {"ok": False, "outcome": "refused", "problems": problems, "error": "; ".join(problems)}
    p = plan(grid, arms, deadline_min=args.deadline_min)
    if args.dry_run:
        print(plan_text(p), file=sys.stderr, flush=True)
        return {"ok": True, "outcome": "dry_run", "gate": "passed", **p}

    def stop_here(signum: int, frame: Any) -> None:
        raise SystemExit(f"signal {signum}")

    for name in ("SIGHUP", "SIGTERM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), stop_here)
    home = Path.home() / POD_DIR
    host = sr.read_once(host or bench.Host())
    power = power or bench.LocalPower.for_host(host, node=node, period_s=float(server["power_period_s"]))
    engine_dir = os.path.expanduser(args.engine_dir or server["engine_dir"])
    patch = REPO / server["patch"]
    reference = home / "stock-outputs.json" if "stock" not in arms else None
    raw = run_e18(grid, arms, host=host, power=power, out_dir=Path(args.out) if args.out else home / "runs",
                  checkpoint=os.path.expanduser(args.checkpoint or server["checkpoint"]),
                  engine_dir=engine_dir if Path(engine_dir).is_dir() else None,
                  patches=[str(patch)] if patch.is_file() else [], node=node, port=port, hub=args.hub,
                  account=args.account or server["account"], deadline_s=p["deadline_min"] * 60.0,
                  reference=reference, outputs_dir=home)
    text = report(raw)
    home.mkdir(parents=True, exist_ok=True)
    (Path(raw["run_dir"]) / "result_bundle.b64").write_text(text, encoding="utf-8", newline="\n")
    (home / "last_result.b64").write_text(text, encoding="utf-8", newline="\n")
    print(text, flush=True)
    done = [a for a in raw["arms"].values() if "rows" in a]
    rows = [r for a in done for r in a["rows"]]
    measured = sum(1 for r in rows if r["outcome"] == "ok")
    ok = raw["error"] is None and len(done) == len(arms) and bool(rows) and measured == len(rows)
    return {"ok": ok, "outcome": "measured" if ok else "incomplete", "run_id": raw["run_id"], "run_dir": raw["run_dir"],
            "arms": list(raw["arms"]), "rows": len(rows), "rows_ok": measured, "refs": {"run_id": raw["run_id"]},
            **({} if ok else {"error": raw["error"] or f"{len(rows) - measured} of {len(rows)} runs not measured"})}


def result_paths(summary: Mapping[str, Any], out_dir: Path) -> tuple[Path, Path, bool]:
    stem = f"e18-hwqueues-{sr.CHIP}-{summary['date']}"
    k = 1
    while True:
        name = stem if k == 1 else f"{stem}-{k}"
        json_path, md_path = out_dir / f"{name}.json", out_dir / f"{name}.md"
        if not json_path.exists():
            return json_path, md_path, False
        try:
            if json.loads(json_path.read_text(encoding="utf-8")).get("run_id") == summary["run_id"]:
                return json_path, md_path, True
        except (OSError, ValueError):
            pass
        k += 1


def collect_main(args: argparse.Namespace) -> dict[str, Any]:
    try:
        files = sr.read_bundle(Path(args.input).read_bytes())
    except OSError as exc:
        raise sr.SweepError(f"cannot read {args.input}: {exc}") from exc
    if "raw.json" not in files:
        raise sr.SweepError("the result block holds no raw.json")
    r52 = json.loads(Path(args.r52).read_text(encoding="utf-8")) if args.r52 else None
    summary = summarize(json.loads(files["raw.json"]), r52)
    out_dir = Path(args.out_dir)
    json_path, md_path, again = result_paths(summary, out_dir)
    if again and not args.force:
        return {"ok": False, "outcome": "already_collected", "run_id": summary["run_id"],
                "error": f"{sr._rel(json_path)} already holds run {summary['run_id']}; --force writes the files and "
                         "the rows again"}
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n")
    md_path.write_text(render_markdown(summary), encoding="utf-8", newline="\n")
    keep = Path(args.state_dir) / summary["run_id"]
    keep.mkdir(parents=True, exist_ok=True)
    for name, blob in files.items():
        (keep / name).write_bytes(blob)
    ledger_n = 0
    if not args.no_ledger:
        for row in ledger_rows(summary):
            record(row.pop("verb"), row.pop("outcome"), row.pop("metrics"), row.pop("refs"), task="R52", **row)
            ledger_n += 1
    metrics_n = 0
    if args.metrics_file:
        for row in metric_rows(summary):
            metrics_file.add(row, path=args.metrics_file)
            metrics_n += 1
    return {"ok": True, "outcome": "collected", "run_id": summary["run_id"], "json": sr._rel(json_path),
            "markdown": sr._rel(md_path), "arms": list(summary["arms"]), "reading": summary["reading"],
            "outputs_unchanged": summary["outputs"]["unchanged"], "outputs_identical": summary["outputs"]["identical"],
            "ledger_rows": ledger_n, "metrics_rows": metrics_n, "open": summary["open"],
            "refs": {"run_id": summary["run_id"]}}


def build_parser() -> argparse.ArgumentParser:
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--grid", default=argparse.SUPPRESS)
    shared.add_argument("--arms", default=argparse.SUPPRESS)
    shared.add_argument("--dry-run", action="store_true", default=argparse.SUPPRESS)
    shared.add_argument("--deadline-min", type=float, default=argparse.SUPPRESS)
    p = argparse.ArgumentParser(prog="lab/sweep_e18.py", description=__doc__.splitlines()[0])
    p.add_argument("--grid", default=str(sr.DEFAULT_GRID), help="R52's grid file (default: the committed one)")
    p.add_argument("--arms", default=",".join(ARMS), help="stock,queue1 (one slot) or one of them (two slots)")
    p.add_argument("--dry-run", action="store_true", help="print the plan and the slot minutes; run nothing")
    p.add_argument("--deadline-min", type=float, default=None,
                   help="start nothing that cannot finish in this many minutes (default: the slot minutes)")
    sub = p.add_subparsers(dest="cmd")
    script = sub.add_parser("pod-script", parents=[shared], help="write the script the pod runs")
    script.add_argument("--out", default=str(POD_SCRIPT))
    script.add_argument("--account", default=None, help="the hub account letter (default: the grid's)")
    script.add_argument("--expect-user-sha256", default=None)
    script.add_argument("--hub", default=None)
    run = sub.add_parser("pod-run", parents=[shared], help="on the pod: gate, the arms, the result")
    run.add_argument("--expect-user-sha256", required=True)
    run.add_argument("--node", default=None)
    run.add_argument("--account", default=None)
    run.add_argument("--hub", default=None)
    run.add_argument("--port", type=int, default=None)
    run.add_argument("--checkpoint", default=None)
    run.add_argument("--engine-dir", default=None)
    run.add_argument("--out", default=None, help="where the run folders go (default: ~/sweep_request/e18/runs)")
    collect = sub.add_parser("collect", help="read the pod's output and write the result files and rows")
    collect.add_argument("input", help="the saved output of the pod run (or ~/sweep_request/e18/last_result.b64)")
    collect.add_argument("--out-dir", default=str(DATA_DIR))
    collect.add_argument("--state-dir", default=str(STATE_DIR))
    collect.add_argument("--r52", default=None, help="a collected R52 result (JSON): its base cells stand beside the arms")
    collect.add_argument("--metrics-file", nargs="?", const=str(metrics_file.DEFAULT_PATH), default=None)
    collect.add_argument("--no-ledger", action="store_true")
    collect.add_argument("--force", action="store_true")
    return p


@recorded("sweep-e18")
def main(argv: Sequence[str] | None = None, *, host: bench.Host | None = None, power: Any = None) -> dict[str, Any]:
    """E18, the hardware-queue check: the teacher with GPU_MAX_HW_QUEUES=1 against stock; the plan, the pod script, the pod run, collect."""
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        if args.cmd is None:
            if not args.dry_run:
                return {"ok": False, "outcome": "usage",
                        "error": "say --dry-run, or one of pod-script, pod-run and collect"}
            return dry_run_main(args)
        if args.cmd == "pod-script":
            return pod_script_main(args)
        if args.cmd == "pod-run":
            return pod_run_main(args, host=host, power=power)
        return collect_main(args)
    except (sr.SweepError, correctness.CorrectnessError) as exc:
        return {"ok": False, "outcome": "fail", "error": str(exc)}


if __name__ == "__main__":
    answer = main()
    print(json.dumps(answer, default=str))
    sys.exit(0 if answer.get("ok") else 1)
