"""The pod sampler's own cost, measured on pod B with and without it (T9, telemetry rubric test 5).

    python -m lab sampler-overhead run --account D [--row ID] [--repeats 3] [--minutes 15] [--out DIR] [--dry-run]
    python -m lab sampler-overhead analyze --rows FILE [--rows FILE ...]

The question (RUBRIC.md TT5): do decode tokens per second and time to first token with the pod sampler running stay
within the run-to-run spread of the same runs without it? The experiment is the same workload twice, interleaved:
off, on, off, on, off, on (three repeats each, after one discarded warm-up with the sampler off). Interleaving is the
rubric's rule: slow drift on the pod (thermal state, the clock, a neighbour on the shared memory) falls on both arms.

One repeat is one pass of lab/bench.py's fixed prompt set (fixtures/lab/bench/prompts.json: two text prompts and two
drawn images, greedy, a fixed max_tokens with ignore_eos) at batch 4, so every prompt is sent once and four streams
decode together. The harness is the real one: lab.bench.attach_session against the model server already up on the pod,
reached through the hub proxy, with the harness's own power reading left off (the T6 sampler is the thing being
switched, so the harness must not read it). The tokens per second are the harness's, from the client's own token
stream: `decode_tok_s` (the median per-request rate), `decode_tok_s_total` (tokens over the decode window) and `ttft_s`.

CPU percent is read on the pod, in the middle of the repeat, through the controller's run-script: two snapshots of
/proc/stat a window apart (the whole pod's busy share) and two of the sampler's own /proc/<pid>/stat (its share of
one core, the children it waited for included, which is where its amd-smi calls land). The reading is marked
`cpu_overlap` only when the repeat was still decoding when the reading came back; a reading taken after the decode ended
is kept in the row and left out of the CPU numbers.

Who does what. The slot is taken through the GPU queue's own rules (lab/machine/slots.py, `QueueSlots` calls
slots.run, so every request, start and release is a ledger row and a Notion row exactly as `python -m lab slot` makes
them), polled every `--poll-s` (120) until `--max-wait-min` (90); a slot that never starts is cancelled and nothing
runs on the pod. Inside the slot: the sampler's state is read (lab/pod_power.sh status), a model server is served with
lab/pod_a_teacher.sh serve when none answers on the proxy, the sampler is started and stopped through
lab/pod_power.sh, and at the end the sampler, the server and the slot are put back as found, each step guarded so one
failure never skips the next. The slot is extended once, with a note, if the remaining repeats will not fit.

Reading the result. For each metric the arms' mean, sd (n - 1), min and max; the difference on minus off, as a percent
of off; the spread, the larger of the two arms' sd; whether the difference is inside it (the rubric's test); and Welch's
t test from lab/machine/verdict.py with its floor lowered to the three repeats taken, shown beside it because three runs
cannot rule out a small effect. Verdict per metric: `within_spread`, `worse_outside_spread`, `better_outside_spread`
(drift, not a gain from the sampler) or `inconclusive`. The overall verdict reads decode tok/s and TTFT: `interference`
when either is worse outside the spread, `inconclusive` when decode tok/s has no spread to compare with, else
`no_interference`. The result is low confidence under verdict.MIN_RUNS_PER_ARM runs an arm (telemetry ruling 7).

Files. <out>/rows.jsonl (one row per repeat, written as each ends, the warm-up included and marked), <out>/analysis.json,
the slot row's Result line, and ledger rows `sampler-overhead.repeat` (one per repeat) and `sampler-overhead.summary`
on node pod-b (metrics `sampler_overhead.*` and `telemetry.sampler_interference_pct`). Nothing is written to
lab/refine/. No credential is read here, printed or stored: the accounts, the hub and the queue are reached through
lab/hub.py, lab/controller.py and lab/machine/slots.py, which scrub them.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import statistics
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from lab.machine import verdict as verdict_module
from lab.machine.record import record, recorded

CLI_VERB = "sampler-overhead"   # registers python -m lab sampler-overhead
TASK = "T9"
NODE = "pod-b"
ROW_SCHEMA = "lab.sampler_overhead.row/1"
CPU_SCHEMA = "lab.sampler_overhead.cpu/1"
CPU_MARKER = "SAMPLER_OVERHEAD_CPU "
CPU_SCRIPT_NAME = "sampler_overhead_cpu.sh"
REPO = Path(__file__).resolve().parents[1]
POD_POWER_SH = REPO / "lab" / "pod_power.sh"
POD_POWER_PY = REPO / "lab" / "pod_power.py"
TEACHER_SH = REPO / "lab" / "pod_a_teacher.sh"
DEFAULT_OUT = REPO / "lab" / "state" / "sampler_overhead"
SLOT_WHAT = "sampler overhead benchmark: the same prompt set with and without the pod sampler, three repeats each"
RETRYABLE_STARTS = ("not_next", "busy", "lost_race")
TRANSIENT_KINDS = ("rate_limited", "transport", "server_error", "timeout")
WARMUP = "warmup"
ARMS = ("off", "on")
#: (metric, higher is better, shown in the verdict)
METRICS = (("decode_tok_s", True), ("decode_tok_s_total", True), ("ttft_s", False), ("cpu_busy_pct", False))
UNITS = {"decode_tok_s": "tok/s", "decode_tok_s_total": "tok/s", "ttft_s": "s", "cpu_busy_pct": "%"}
RULE = ("decode tok/s and time to first token with the sampler running stay within the run-to-run spread of the "
        "same runs without it (telemetry RUBRIC.md TT5)")

#: What the pod runs for the CPU reading. Stdlib only. PROC_ROOT is for tests; on a pod it is /proc.
CPU_PY = '''import json, os, sys, time
root = os.environ.get("PROC_ROOT", "/proc")
window = float(sys.argv[1])
pidfile = sys.argv[2]


def snapshot():
    with open(root + "/stat") as fh:
        lines = fh.read().splitlines()
    total = [int(x) for x in lines[0].split()[1:9]]
    cores = sum(1 for ln in lines if ln.startswith("cpu") and ln[3:4].isdigit())
    return total, cores


def sampler_pid():
    try:
        with open(pidfile) as fh:
            return int(fh.read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def ticks(pid):
    if pid is None:
        return None
    try:
        with open("%s/%d/stat" % (root, pid)) as fh:
            rest = fh.read().rsplit(")", 1)[1].split()
        return sum(int(rest[i]) for i in (11, 12, 13, 14))
    except (OSError, ValueError, IndexError):
        return None


try:
    tck = os.sysconf("SC_CLK_TCK")
except (AttributeError, ValueError, OSError):
    tck = 100
pid = sampler_pid()
t0 = time.time()
s0, ncpu = snapshot()
k0 = ticks(pid)
time.sleep(window)
t1 = time.time()
s1, _ = snapshot()
k1 = ticks(pid)
print("SAMPLER_OVERHEAD_CPU " + json.dumps({"schema": "lab.sampler_overhead.cpu/1", "t0": t0, "t1": t1, "tck": tck,
      "ncpu": ncpu, "stat0": s0, "stat1": s1, "pid": pid, "ticks0": k0, "ticks1": k1}))
'''


class OverheadError(Exception):
    """A step of the experiment failed in a way that ends the run."""


class SlotUnavailable(OverheadError):
    """The slot was refused or never started."""

    def __init__(self, message: str, row: str | None = None) -> None:
        super().__init__(message)
        self.row = row


class OutOfTime(OverheadError):
    """The slot cannot hold the next repeat and its extension was refused or already used."""


# --- the arithmetic ----------------------------------------------------------------------------------------------

def schedule(repeats: int = 3) -> list[str]:
    """off, on, off, on, ...: the arms interleaved, `repeats` of each."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    return [arm for _ in range(repeats) for arm in ARMS]


def cpu_script_text() -> str:
    """The shell script run-script uploads: CPU_PY on the pod, window seconds as its first argument."""
    return ("#!/bin/bash\n"
            "# T9: two /proc snapshots a window apart (the pod's busy share, and the pod sampler's own share of one core).\n"
            "# Prints one line, SAMPLER_OVERHEAD_CPU {json}. Usage: bash sampler_overhead_cpu.sh [window_s]\n"
            "set -u\n"
            "WINDOW=\"${1:-12}\"\n"
            "PIDFILE=\"${POD_POWER_DIR:-$HOME/atlas-power}/sampler.pid\"\n"
            "exec python3 - \"$WINDOW\" \"$PIDFILE\" <<'PY'\n" + CPU_PY + "PY\n")


def parse_cpu(output: str) -> dict[str, Any]:
    """The CPU reading in a script's output: the pod's busy percent, in cores, the sampler's share of one core."""
    line = next((ln for ln in reversed(str(output).splitlines()) if ln.startswith(CPU_MARKER)), None)
    if line is None:
        raise OverheadError("the CPU script printed no reading")
    try:
        data = json.loads(line[len(CPU_MARKER):])
        s0, s1 = [int(x) for x in data["stat0"][:8]], [int(x) for x in data["stat1"][:8]]
        window = float(data["t1"]) - float(data["t0"])
        tck, ncpu = int(data["tck"]), int(data["ncpu"])
    except (ValueError, KeyError, TypeError) as exc:
        raise OverheadError(f"the CPU reading is not readable: {type(exc).__name__}: {exc}") from None
    total = sum(s1) - sum(s0)
    idle = (s1[3] + s1[4]) - (s0[3] + s0[4])
    if total <= 0 or window <= 0:
        raise OverheadError("the CPU counters did not move between the two snapshots")
    busy = 100.0 * (1.0 - idle / total)
    share = None
    if data.get("pid") is not None and data.get("ticks0") is not None and data.get("ticks1") is not None:
        share = 100.0 * (int(data["ticks1"]) - int(data["ticks0"])) / tck / window
    return {"busy_pct": round(busy, 4), "cores_busy": round(busy / 100.0 * ncpu, 4), "ncpu": ncpu,
            "window_s": round(window, 3), "sampler_pid": data.get("pid"),
            "sampler_cpu_pct": None if share is None else round(share, 4)}


def stats(values: Iterable[float | None]) -> dict[str, Any]:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None}
    return {"n": len(xs), "mean": round(statistics.fmean(xs), 4),
            "sd": round(statistics.stdev(xs), 4) if len(xs) > 1 else None, "min": round(min(xs), 4),
            "max": round(max(xs), 4), "values": [round(x, 4) for x in xs]}


def compare(off: Sequence[float | None], on: Sequence[float | None], *, higher_is_better: bool) -> dict[str, Any]:
    """On against off for one metric: means, difference, spread, the rubric's test and Welch's."""
    a, b = stats(off), stats(on)
    out: dict[str, Any] = {"off": a, "on": b, "diff": None, "pct": None, "spread": None, "within_spread": None,
                           "welch": None, "verdict": "inconclusive"}
    if a["mean"] is None or b["mean"] is None:
        return out
    diff = b["mean"] - a["mean"]
    out["diff"] = round(diff, 4)
    out["pct"] = round(100.0 * diff / a["mean"], 4) if a["mean"] else None
    if a["sd"] is None or b["sd"] is None:
        return out
    spread = max(a["sd"], b["sd"])
    out["spread"] = round(spread, 4)
    out["spread_pct"] = round(100.0 * spread / a["mean"], 4) if a["mean"] else None
    out["within_spread"] = abs(diff) <= spread
    worse = diff < 0 if higher_is_better else diff > 0
    out["verdict"] = "within_spread" if out["within_spread"] else (
        "worse_outside_spread" if worse else "better_outside_spread")
    welch = verdict_module.verdict([float(v) for v in off if v is not None], [float(v) for v in on if v is not None],
                                   higher_is_better=higher_is_better, min_per_arm=3)
    out["welch"] = {"label": welch.label, "p": None if welch.p is None else round(welch.p, 4),
                    "t": None if welch.statistic is None else round(welch.statistic, 4),
                    "needed_per_arm": welch.needed_per_arm, "note": welch.note}
    return out


def analyze(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Turn the repeat rows into the numbers, the verdict and the rubric bound's confirmation or correction."""
    arms: dict[str, list[Mapping[str, Any]]] = {"off": [], "on": []}
    for row in rows:
        if row.get("arm") in arms and row.get("outcome", "ok") == "ok":
            arms[row["arm"]].append(row)

    def column(arm: str, key: str, *, cpu: bool = False) -> list[float | None]:
        return [r.get(key) for r in arms[arm] if not cpu or r.get("cpu_overlap", True)]

    metrics = {name: compare(column("off", name, cpu=name == "cpu_busy_pct"),
                             column("on", name, cpu=name == "cpu_busy_pct"), higher_is_better=higher)
               for name, higher in METRICS}
    share = stats(r.get("sampler_cpu_pct") for r in arms["on"] if r.get("cpu_overlap", True))
    n = {arm: sum(1 for r in arms[arm] if r.get("decode_tok_s") is not None) for arm in ARMS}
    tok, ttft = metrics["decode_tok_s"], metrics["ttft_s"]
    notes: list[str] = []
    if tok["verdict"] == "inconclusive":
        verdict = "inconclusive"
    elif "worse_outside_spread" in (tok["verdict"], ttft["verdict"]):
        verdict = "interference"
    else:
        verdict = "no_interference"
    for name in ("decode_tok_s", "ttft_s"):
        if metrics[name]["verdict"] == "better_outside_spread":
            notes.append(f"{name} was better with the sampler on, beyond the spread: drift or warm-up in the run, not "
                         "a gain from the sampler")
    if ttft["verdict"] == "inconclusive" and tok["verdict"] != "inconclusive":
        notes.append("time to first token has no spread to compare with; the verdict rests on decode tok/s")
    holds = {"no_interference": True, "interference": False}.get(verdict)
    action = {True: "confirmed", False: "corrected", None: "undecided"}[holds]
    return {"schema": "lab.sampler_overhead.analysis/1", "rule": RULE, "n": n, "metrics": metrics,
            "sampler_cpu_pct": share if share["n"] else None, "verdict": verdict,
            "interference_pct": tok["pct"],
            "bound": {"rule": RULE, "holds": holds, "action": action, "measured_pct": tok["pct"],
                      "spread_pct": tok.get("spread_pct")},
            "low_confidence": min(n.values()) < verdict_module.MIN_RUNS_PER_ARM, "notes": notes}


def result_line(analysis: Mapping[str, Any]) -> str:
    """The slot row's Result: one line with the numbers."""
    tok = analysis["metrics"]["decode_tok_s"]
    if tok["diff"] is None:
        return f"T9: sampler overhead {analysis['verdict']}; too few repeats to compare ({analysis['n']['off']} off, {analysis['n']['on']} on)"
    spread = "n/a" if tok["spread"] is None else f"{tok['spread']:.3g}"
    return (f"T9: decode tok/s off {tok['off']['mean']:.3g} (sd {tok['off']['sd'] or 0:.2g}), on {tok['on']['mean']:.3g} "
            f"(sd {tok['on']['sd'] or 0:.2g}), n {analysis['n']['off']}+{analysis['n']['on']}; diff {tok['diff']:+.3g} "
            f"({tok['pct']:+.2f} percent), spread {spread}: {analysis['verdict']}")


# --- clients (injectable) ---------------------------------------------------------------------------------------------

class SlotClient(Protocol):
    def request(self, *, task: str, what: str, minutes: int) -> Mapping[str, Any]: ...
    def start(self, row: str) -> Mapping[str, Any]: ...
    def extend(self, row: str, note: str) -> Mapping[str, Any]: ...
    def done(self, row: str, result: str | None, cancel: bool = False) -> Mapping[str, Any]: ...


class QueueSlots:
    """The GPU queue through its own command, slots.run: the same rules, Notion rows and ledger rows as
    `python -m lab slot`. client and now replace the Notion client and the clock, for tests."""

    def __init__(self, *, client: Any = None, now: Callable[[], dt.datetime] | None = None, agent: str | None = None,
                 environ: Mapping[str, str] | None = None, state_dir: str | os.PathLike[str] | None = None) -> None:
        self.client, self.now, self.agent, self.environ, self.state_dir = client, now, agent, environ, state_dir

    def _call(self, *argv: str) -> dict[str, Any]:
        from lab.machine import slots
        extra = ["--agent", self.agent] if self.agent else []
        answer, _ = slots.run([argv[0], *extra, *argv[1:]], client=self.client, now=self.now, environ=self.environ,
                              state_dir=self.state_dir)
        return answer

    def request(self, *, task: str, what: str, minutes: int) -> Mapping[str, Any]:
        return self._call("request", "--task", task, "--what", what, "--minutes", str(minutes))

    def start(self, row: str) -> Mapping[str, Any]:
        return self._call("start", "--row", row)

    def extend(self, row: str, note: str) -> Mapping[str, Any]:
        return self._call("extend", "--row", row, "--note", note)

    def done(self, row: str, result: str | None, cancel: bool = False) -> Mapping[str, Any]:
        if cancel:
            return self._call("done", "--row", row, "--cancel")
        return self._call("done", "--row", row, "--result", result or "")


class ControllerPod:
    """Run a script on the pod through lab/controller.py's run-script; the answer is its dict."""

    def __init__(self, controller: Any, letter: str) -> None:
        self.controller, self.letter = controller, letter

    def run_script(self, script: Path | str, *, args: Sequence[str] = (), files: Sequence[Path | str] = (),
                   timeout_s: float = 900.0) -> Mapping[str, Any]:
        return self.controller.run_script(self.letter, script, args=list(args), files=list(files), timeout_s=timeout_s)


class AttachBench:
    """One repeat of lab/bench.py's fixed prompt set against the server already up, through the hub proxy (or any
    endpoint the factory makes). The harness's own power reading and ledger rows are off: the sampler is the variable."""

    def __init__(self, endpoint_factory: Callable[[], Any], *, out_dir: Path | str, batch: int = 4, max_tokens: int = 96,
                 hub: str | None = None, account: str | None = None, node: str = NODE,
                 request_timeout_s: float = 600.0) -> None:
        self.endpoint_factory, self.out_dir = endpoint_factory, Path(out_dir)
        self.batch, self.max_tokens, self.hub, self.account = batch, max_tokens, hub, account
        self.node, self.request_timeout_s = node, request_timeout_s

    def __call__(self, label: str) -> dict[str, Any]:
        from lab import bench
        config = bench.Config(engine="vllm", model="teacher", node=self.node, batches=[self.batch], add_64=False,
                              repeats=1, max_tokens=self.max_tokens, requests_per_stream=1, warmup_rounds=0, probe=False,
                              idle_s=0.0, out_dir=self.out_dir / label, hub=self.hub, account=self.account,
                              request_timeout_s=self.request_timeout_s, record=False)
        endpoint = self.endpoint_factory()
        try:
            result = bench.attach_session(config, endpoint=endpoint)
        finally:
            endpoint.close()
        return result["rows"][0]


# --- the experiment ---------------------------------------------------------------------------------------------------

@dataclasses.dataclass
class Settings:
    account: str = "D"
    node: str = NODE
    sampler_node: str = "gpu-b"        # lab/pod_power.sh's name for the trainer pod
    period_ms: int = 1000
    repeats: int = 3
    slot_minutes: int = 15
    poll_s: float = 120.0
    max_wait_s: float = 5400.0
    row: str | None = None             # a slot already Running: use it (and still release it)
    cpu_delay_s: float = 8.0           # into the repeat before the CPU window opens
    cpu_window_s: float = 12.0
    port: int = 8000
    serve_timeout_s: float = 900.0
    exec_timeout_s: float = 240.0
    repeat_estimate_s: float = 90.0    # until a repeat has been timed
    switch_s: float = 60.0             # a sampler start or stop and the CPU call around a repeat
    reserve_s: float = 120.0           # kept for the clean-up at the end of the slot


class Experiment:
    def __init__(self, settings: Settings, *, slots: SlotClient, pod: Any, bench: Callable[[str], Mapping[str, Any]],
                 endpoint: Any, out_dir: Path | str, clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep, recorder: Callable[..., Any] = record) -> None:
        self.s, self.slots, self.pod, self.bench, self.endpoint = settings, slots, pod, bench, endpoint
        self.out_dir = Path(out_dir)
        self.clock, self.sleep, self.recorder = clock, sleep, recorder
        self.session = "so-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:4]
        self.rows: list[dict[str, Any]] = []
        self.row_id: str | None = None
        self.sampler_state: str = "unknown"          # on, off or unknown
        self.sampler_at_start: bool | None = None
        self.started_server = False
        self.slot_started_at = 0.0
        self.cap_s = settings.slot_minutes * 60.0
        self.extended = False
        self.longest_repeat_s = 0.0
        self.errors: list[str] = []

    # -- the pod ---

    def script(self, script: Path, *args: str, files: Sequence[Path] = (), timeout_s: float | None = None,
               ok_codes: Sequence[int] = (0,)) -> Mapping[str, Any]:
        answer = self.pod.run_script(script, args=args, files=files, timeout_s=timeout_s or self.s.exec_timeout_s)
        code = answer.get("exit_code")
        if code is None or code not in ok_codes:
            tail = " ".join(str(answer.get("output_tail") or answer.get("error") or "").split())[-300:]
            raise OverheadError(f"{script.name} {' '.join(args)} ended {answer.get('outcome')} (exit {code}): {tail}")
        return answer

    def sampler_running(self) -> bool:
        answer = self.script(POD_POWER_SH, "status", files=[POD_POWER_PY], ok_codes=(0, 1))
        running = answer.get("exit_code") == 0
        if not running and "not running" not in str(answer.get("output_tail")):
            raise OverheadError(f"cannot tell whether the sampler runs: {str(answer.get('output_tail'))[-200:]}")
        return running

    def set_sampler(self, on: bool) -> None:
        if self.sampler_state == ("on" if on else "off"):
            return
        self.sampler_state = "unknown"
        if on:
            self.script(POD_POWER_SH, "start", self.s.sampler_node, str(self.s.period_ms), files=[POD_POWER_PY],
                        timeout_s=300)
        else:
            self.script(POD_POWER_SH, "stop", files=[POD_POWER_PY])
        self.sampler_state = "on" if on else "off"

    def server_up(self) -> bool:
        try:
            return bool(self.endpoint.models())
        except Exception:
            return False

    def ensure_server(self) -> None:
        if self.server_up():
            return
        self.started_server = True
        try:
            self.script(TEACHER_SH, "serve", timeout_s=self.s.serve_timeout_s)
        except OverheadError as exc:
            raise OverheadError(f"the model server did not start: {exc}") from None
        if not self.server_up():
            raise OverheadError("the model server was started but does not answer through the hub proxy")

    # -- the slot ---

    def acquire(self) -> str:
        if self.s.row:
            self.row_id, self.slot_started_at = self.s.row, self.clock()
            return self.s.row
        asked = self.slots.request(task=TASK, what=SLOT_WHAT, minutes=self.s.slot_minutes)
        if not asked.get("ok"):
            raise SlotUnavailable(f"the slot request was refused: {asked.get('outcome')}: {asked.get('reason') or asked.get('error')}")
        self.row_id = str(asked["row"])
        deadline = self.clock() + self.s.max_wait_s
        while True:
            started = self.slots.start(self.row_id)
            if started.get("ok") and started.get("outcome") == "started":
                self.slot_started_at = self.clock()
                return self.row_id
            outcome = str(started.get("outcome"))
            transient = outcome in RETRYABLE_STARTS or (outcome == "error" and started.get("kind") in TRANSIENT_KINDS)
            if not transient:
                raise SlotUnavailable(f"the slot start was refused: {outcome}: {started.get('reason') or started.get('error')}",
                                      self.row_id)
            if self.clock() + self.s.poll_s > deadline:
                raise SlotUnavailable(f"the slot never started within {self.s.max_wait_s / 60:g} minutes (last answer: {outcome})",
                                      self.row_id)
            self.sleep(self.s.poll_s)

    def make_room(self, done: int, total: int) -> None:
        """Before a repeat: extend the slot once if the repeat and the clean-up will not fit; else stop."""
        need = max(self.longest_repeat_s, self.s.repeat_estimate_s) + self.s.switch_s + self.s.reserve_s
        if self.slot_started_at + self.cap_s - self.clock() >= need:
            return
        if not self.extended and self.row_id:
            answer = self.slots.extend(self.row_id, f"sampler overhead: {done} of {total} repeats measured, the rest need "
                                                    "the slot's second quarter hour")
            self.extended = True
            if answer.get("ok"):
                self.cap_s += 15 * 60.0
                return
        raise OutOfTime(f"the slot cannot hold the next repeat ({done} of {total} measured) and was not extended")

    def release(self, result: str) -> None:
        if self.row_id:
            self.slots.done(self.row_id, result)

    # -- rows ---

    def keep(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.out_dir / "rows.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")
        metrics = {f"sampler_overhead.{k}": row[k] for k in ("decode_tok_s", "decode_tok_s_total", "ttft_s", "cpu_busy_pct")
                   if row.get(k) is not None}
        if row.get("sampler_cpu_pct") is not None:
            metrics["telemetry.sampler_cpu_pct"] = row["sampler_cpu_pct"]
        units = {k: ("%" if k.endswith("_pct") else UNITS.get(k.split(".", 1)[1], "")) for k in metrics}
        conditions = {k: row[k] for k in ("sampler", "arm", "repeat", "order", "cpu_overlap", "batch_size", "max_tokens",
                                          "period_ms") if k in row}
        conditions["sampler"] = row["sampler"] if row["arm"] != WARMUP else "warmup-off"
        try:
            self.recorder("sampler-overhead.repeat", row["outcome"], metrics, {"run_id": row["run_id"], "slot": self.row_id},
                          node=self.s.node, units=units, conditions=conditions, low_confidence=True,
                          source="lab/sampler_overhead.py", note=row.get("reason"), task=TASK)
        except Exception as exc:   # the rows file is the record; a ledger that refuses is said, not fatal
            self.errors.append(f"no ledger row for {row['run_id']}: {type(exc).__name__}: {exc}")

    def measure(self, arm: str, repeat: int, order: int) -> None:
        label = f"{arm}-{repeat}" if arm != WARMUP else WARMUP
        started = self.clock()
        row: dict[str, Any] = {"schema": ROW_SCHEMA, "run_id": f"{self.session}-{order:02d}", "session": self.session,
                               "task": TASK, "node": self.s.node, "arm": arm, "repeat": repeat, "order": order,
                               "sampler": "on" if arm == "on" else "off", "outcome": "ok", "reason": None,
                               "period_ms": self.s.period_ms, "batch_size": None, "max_tokens": None,
                               "decode_tok_s": None, "decode_tok_s_total": None, "ttft_s": None, "cpu_busy_pct": None,
                               "sampler_cpu_pct": None, "cpu_overlap": None, "cpu": None,
                               "started": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
        box: dict[str, Any] = {}

        def work() -> None:
            try:
                box["row"] = self.bench(label)
            except BaseException as exc:     # carried to the main thread
                box["error"] = exc

        thread = threading.Thread(target=work, name=f"bench-{label}", daemon=True)
        thread.start()
        self.sleep(self.s.cpu_delay_s)
        cpu_error = None
        try:
            answer = self.script(self.out_dir / CPU_SCRIPT_NAME, f"{self.s.cpu_window_s:g}", timeout_s=self.s.exec_timeout_s)
            reading = parse_cpu(str(answer.get("output_tail") or ""))
        except OverheadError as exc:
            reading, cpu_error = None, str(exc)
        overlapped = thread.is_alive()
        thread.join()
        self.longest_repeat_s = max(self.longest_repeat_s, self.clock() - started)
        if reading is not None:
            row.update(cpu_busy_pct=reading["busy_pct"], sampler_cpu_pct=reading["sampler_cpu_pct"], cpu=reading,
                       cpu_overlap=overlapped)
        else:
            row.update(cpu_overlap=False, cpu={"error": cpu_error})
        if "error" in box or box["row"].get("outcome") != "ok":
            reason = str(box["error"]) if "error" in box else str(box["row"].get("reason") or "the bench row failed")
            row.update(outcome="fail", reason=reason)
            self.keep(row)
            raise OverheadError(f"{label}: {reason}")
        got = box["row"]
        row.update(decode_tok_s=got.get("decode_tok_s"), decode_tok_s_total=got.get("decode_tok_s_total"),
                   ttft_s=got.get("ttft_s"), batch_size=got.get("batch_size"),
                   bench_run_id=got.get("run_id"), tokens_generated=got.get("tokens_generated"),
                   window_s=got.get("window_s"), in_flight_mean=got.get("in_flight_mean"),
                   requests=len(got.get("requests") or []))
        self.keep(row)

    # -- the run ---

    def run(self) -> dict[str, Any]:
        base = {"session": self.session, "task": TASK, "node": self.s.node, "rows_path": str(self.out_dir / "rows.jsonl")}
        try:
            self.acquire()
        except SlotUnavailable as exc:
            if exc.row and not self.s.row:
                try:
                    self.slots.done(exc.row, None, cancel=True)
                except Exception as cancel_error:
                    self.errors.append(f"could not cancel the row: {cancel_error}")
            return {**base, "ok": False, "outcome": "no_slot", "error": str(exc), "harness_ready": True,
                    "slot": {"row": exc.row}}
        failure: Exception | None = None
        partial = False
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            (self.out_dir / CPU_SCRIPT_NAME).write_text(cpu_script_text(), encoding="utf-8", newline="\n")
            self.sampler_at_start = self.sampler_running()
            self.sampler_state = "on" if self.sampler_at_start else "off"
            self.ensure_server()
            self.set_sampler(False)
            plan = [(WARMUP, 0)] + [(arm, k // 2 + 1) for k, arm in enumerate(schedule(self.s.repeats))]
            for order, (arm, repeat) in enumerate(plan):
                try:
                    self.make_room(order, len(plan))
                except OutOfTime as exc:
                    partial, failure = True, exc
                    break
                self.set_sampler(arm == "on")
                self.measure(arm, repeat, order)
        except Exception as exc:
            failure = exc
        analysis = analyze(self.rows)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
        self.cleanup()
        result = (f"T9: stopped early ({failure}); " + result_line(analysis)) if failure else result_line(analysis)
        if failure and not partial:
            result = f"T9: failed before it finished ({failure})"
        try:
            self.release(result)
        except Exception as exc:
            self.errors.append(f"the slot was not released: {exc}")
        self.summarize(analysis)
        outcome = "partial" if partial else ("fail" if failure else "ok")
        answer = {**base, "ok": outcome == "ok", "outcome": outcome, "analysis": analysis, "slot": {"row": self.row_id,
                  "extended": self.extended}, "started_server": self.started_server, "cleanup_errors": self.errors,
                  "harness_ready": True}
        if failure:
            answer["error"] = str(failure)
        return answer

    def cleanup(self) -> None:
        """Put the sampler and the server back as found; every step guarded."""
        try:
            if self.sampler_state == "unknown" and self.sampler_at_start is not None:
                self.sampler_state = "on" if self.sampler_running() else "off"
            if self.sampler_at_start is not None and (self.sampler_state == "on") != self.sampler_at_start:
                self.set_sampler(self.sampler_at_start)
        except Exception as exc:
            self.errors.append(f"sampler not restored: {exc}")
        if self.started_server:
            try:
                self.script(TEACHER_SH, "stop")
            except Exception as exc:
                self.errors.append(f"server not stopped: {exc}")

    def summarize(self, analysis: Mapping[str, Any]) -> None:
        tok = analysis["metrics"]["decode_tok_s"]
        cpu = analysis["metrics"]["cpu_busy_pct"]
        metrics: dict[str, float] = {}
        units: dict[str, str] = {}

        def put(name: str, value: Any, unit: str) -> None:
            if value is not None:
                metrics[name], units[name] = value, unit

        put("telemetry.sampler_interference_pct", analysis.get("interference_pct"), "%")
        put("sampler_overhead.decode_tok_s_off", tok["off"]["mean"], "tok/s")
        put("sampler_overhead.decode_tok_s_on", tok["on"]["mean"], "tok/s")
        put("sampler_overhead.decode_tok_s_spread", tok["spread"], "tok/s")
        put("sampler_overhead.cpu_busy_pct_off", cpu["off"]["mean"], "%")
        put("sampler_overhead.cpu_busy_pct_on", cpu["on"]["mean"], "%")
        share = analysis.get("sampler_cpu_pct")
        put("telemetry.sampler_cpu_pct", share["mean"] if share else None, "%")
        conditions = {"verdict": analysis["verdict"], "bound": analysis["bound"], "n": analysis["n"],
                      "period_ms": self.s.period_ms, "repeats": self.s.repeats, "session": self.session,
                      "low_confidence_reason": "three repeats an arm, under the five a verdict wants"}
        try:
            self.recorder("sampler-overhead.summary", "ok" if analysis["verdict"] != "inconclusive" else "inconclusive",
                          metrics, {"slot": self.row_id, "run_id": self.session}, node=self.s.node, units=units,
                          conditions=conditions, low_confidence=bool(analysis["low_confidence"]),
                          source="lab/sampler_overhead.py", task=TASK)
        except Exception as exc:
            self.errors.append(f"no summary ledger row: {type(exc).__name__}: {exc}")


# --- the command line ------------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m lab sampler-overhead", allow_abbrev=False,
                                     description="Measure the pod sampler's own cost on pod B, with and without it.")
    subs = parser.add_subparsers(dest="sub", required=True)
    run = subs.add_parser("run", help="take a GPU queue slot and run the experiment on the pod")
    run.add_argument("--account", default="D", help="the trainer pod's account letter (default D, pod-b)")
    run.add_argument("--row", help="a slot row that is already Running: use it instead of requesting one")
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--minutes", type=int, default=15, choices=(15, 30), help="the slot to ask for")
    run.add_argument("--poll-s", type=float, default=120.0, help="seconds between tries to start the slot")
    run.add_argument("--max-wait-min", type=float, default=90.0, help="give up on the slot after this long")
    run.add_argument("--batch", type=int, default=4, help="concurrent streams; 4 sends every fixed prompt once")
    run.add_argument("--max-tokens", type=int, default=96)
    run.add_argument("--period-ms", type=int, default=1000, help="the sampler's period while it is on")
    run.add_argument("--port", type=int, default=8000, help="the model server's port on the pod")
    run.add_argument("--cpu-delay-s", type=float, default=8.0)
    run.add_argument("--cpu-window-s", type=float, default=12.0)
    run.add_argument("--out", default=None, help=f"where rows.jsonl and analysis.json go (default {DEFAULT_OUT}/<session>)")
    run.add_argument("--dry-run", action="store_true", help="show the schedule and the steps; touch nothing")
    analyze_cmd = subs.add_parser("analyze", help="the numbers and the verdict from rows.jsonl files")
    analyze_cmd.add_argument("--rows", action="append", required=True, help="a rows.jsonl (repeatable)")
    return parser


def settings_of(args: argparse.Namespace) -> Settings:
    return Settings(account=args.account.upper(), repeats=args.repeats, slot_minutes=args.minutes, poll_s=args.poll_s,
                    max_wait_s=args.max_wait_min * 60.0, row=args.row, period_ms=args.period_ms, port=args.port,
                    cpu_delay_s=args.cpu_delay_s, cpu_window_s=args.cpu_window_s)


def dry_run(args: argparse.Namespace) -> dict[str, Any]:
    s = settings_of(args)
    steps = [f"python -m lab slot request --task {TASK} --minutes {s.slot_minutes}, then slot start every {s.poll_s:g} s up to "
             f"{s.max_wait_s / 60:g} minutes (cancelled if it never starts)",
             "pod_power.sh status (the sampler's state at the start is restored at the end)",
             "pod_a_teacher.sh serve, only when no model server answers on the proxy; pod_a_teacher.sh stop at the end then",
             "pod_power.sh stop if the sampler runs, then one warm-up repeat with the sampler off (discarded)"]
    steps += [f"{arm} repeat {k // 2 + 1}: " + ("pod_power.sh start " + f"{s.sampler_node} {s.period_ms}, " if arm == "on" else
              "pod_power.sh stop (if on), ") + f"bench batch {args.batch} x {args.max_tokens} tokens, CPU read at +{s.cpu_delay_s:g} s for {s.cpu_window_s:g} s"
              for k, arm in enumerate(schedule(s.repeats))]
    steps += ["slot extend once if the remaining repeats will not fit", "put the sampler and the server back, slot done with the numbers"]
    return {"ok": True, "outcome": "dry_run", "schedule": schedule(s.repeats), "steps": steps,
            "slot": {"minutes": s.slot_minutes, "poll_s": s.poll_s, "max_wait_s": s.max_wait_s}}


def run_real(args: argparse.Namespace) -> dict[str, Any]:
    from lab import bench, controller, hub, proxy
    letter = args.account.upper()
    try:
        accounts = hub.load_accounts([letter], require_role=True)
    except hub.MissingCredentials as exc:
        return {"ok": False, "outcome": "no_credentials", "missing": list(exc.missing),
                "error": "keys missing from the environment and the .env files (names only; values are never printed)"}
    except hub.PodRoleError as exc:
        return {"ok": False, "outcome": "no_pod_role", "error": str(exc)}
    settings = settings_of(args)
    session_out = Path(args.out) if args.out else DEFAULT_OUT / dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with controller.Controller(accounts) as ctl, proxy.PodLink.for_account(accounts[letter]) as link:
        def endpoint() -> Any:
            return bench.ProxyEndpoint(link, settings.port)

        runner = AttachBench(endpoint, out_dir=session_out / "bench", batch=args.batch, max_tokens=args.max_tokens,
                             hub=link.base_url, account=accounts[letter].user, node=settings.node)
        experiment = Experiment(settings, slots=QueueSlots(), pod=ControllerPod(ctl, letter), bench=runner,
                                endpoint=endpoint(), out_dir=session_out)
        return experiment.run()


@recorded("sampler-overhead")
def run_command(args: argparse.Namespace) -> dict[str, Any]:
    if args.dry_run:
        return dry_run(args)
    answer = run_real(args)
    analysis = answer.get("analysis") or {}
    return {**{k: v for k, v in answer.items() if k != "analysis"}, "verdict": analysis.get("verdict"),
            "interference_pct": analysis.get("interference_pct"), "analysis": analysis,
            "refs": {"slot": (answer.get("slot") or {}).get("row")}}


def analyze_command(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in args.rows:
        try:
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        except (OSError, ValueError) as exc:
            return {"ok": False, "outcome": "error", "error": f"cannot read {path}: {type(exc).__name__}: {exc}"}
    return {"ok": True, "outcome": "ok", **analyze(rows), "line": result_line(analyze(rows))}


def main(argv: Sequence[str]) -> dict[str, Any]:
    """The pod sampler's own cost on pod B, measured with and without it (T9)."""
    args = build_parser().parse_args(list(argv))
    return run_command(args) if args.sub == "run" else analyze_command(args)


if __name__ == "__main__":
    answer = main(sys.argv[1:])
    print(json.dumps(answer, default=str))
    sys.exit(0 if answer.get("ok", True) else 1)
