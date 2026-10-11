"""Attribution check (T8): do the answers' windows hold the energy the teacher's pod spends on them?

    python -m lab attribution-check run [--account C] [--requests 100] [--idle-s 60] [--gap-s 10] [--max-tokens 128]
                                        [--out fixtures/telemetry] [--tag DATE] [--wait-min 30] [--no-ledger]
    python -m lab attribution-check replay --tag DATE [--dir fixtures/telemetry] [--record]

What it does. On pod A (the teacher, account C), with the pod's power sampler (lab/pod_power.py) already running:
check that the teacher answers and that nothing else is in flight on it (vLLM's /metrics through the hub proxy),
wait `settle_s`, read the idle watts for `idle_s` with the teacher loaded and no request, then send `requests` text
requests one after another, each followed by a pause of `gap_s` with nothing in flight. The requests are bench's
(lab/bench.py): the text prompts of fixtures/lab/bench/prompts.json in turn, greedy, thinking off, a fixed
max_tokens with ignore_eos, a nonce in front so no prefix-cache hit. Nothing is started, stopped or changed on the pod.

What it computes, all gross (telemetry ruling 1: nothing is subtracted from any value):
- each answer's joules from T4's attribution (contexts/telemetry/application/joules.py), fed the pod's samples, the
  request windows and the pod's clock offset as the box feeds it, and their sum over the run;
- the energy in the gaps between answers (and after the last one), integrated the same way
  (contexts/telemetry/domain/energy.py: the straight line between samples, the edges interpolated);
- the whole run's energy, first request sent to the end of the last gap, and the conservation check: answers plus
  gaps against the whole run;
- the idle watts (time-weighted mean over the idle window);
- what the windows missed: the gap energy above the idle watts times the gap time. Its share of the answers' energy
  (missed / (answers + missed)) is the relative error of the attribution; `attribution_error_pct` is the same
  number signed as the metric set defines it, (window energy - reference) / reference, so it is negative when the
  windows miss energy;
- the samples per window (median, min, max) and a verdict (lab/machine/verdict.py, Welch's t test) of arm B, each
  answer's joules plus the gap after it above idle, against arm A, its joules alone. no_difference means what the
  windows miss is inside the answer-to-answer spread, and the attribution is called trustworthy at this sampling
  rate when that holds, every window has at least 10 samples (TT1), the energy is conserved, the idle window saw no
  load and no other client's request ran during the run.

One clock. The requests are stamped on this machine's clock (bench.WallClock, perf_counter anchored once, so a wall
clock step mid-run moves nothing); the samples on the pod's. Every read of the sampler is one four-timestamp
exchange, and the one with the smallest round trip gives the offset (contexts/telemetry/infrastructure/clock.py),
the same way the box's reader measures it.

The raw rows are kept as three files in --out, so a replay recomputes every number: attribution_check_<tag>_samples
(the pod's telemetry.gpu.power rows as served), _requests (one answer a line) and _run (settings, the idle and run
windows, the clock exchanges, vLLM's counters before and after). `run` writes one ledger row, verb
attribution-check, task T8, node pod-a; it stays low_confidence because the sensor's own error is not measured on
our node against a reference (TT1), which this check does not do.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from contexts.telemetry.application.joules import MIN_SAMPLES, Calibration, JoulesPerAnswerProjection
from contexts.telemetry.domain.energy import integrate
from contexts.telemetry.infrastructure.clock import Exchange, best_offset, exchange_of
from contracts.events import parse_event
from contracts.telemetry import ClockOffset, GpuPower, RequestWindow
from lab import bench, pod_power
from lab.machine import ledger
from lab.machine.record import record
from lab.machine.verdict import A_AHEAD, B_AHEAD, NO_DIFFERENCE, TOO_EARLY, verdict

NODE = "pod-a"
CLOCK = "local:client"            # the clock the request windows are stamped on: the machine that sends them
TASK = "T8"
VERB = "attribution-check"
DEFAULT_REQUESTS = 100
DEFAULT_IDLE_S = 60.0
DEFAULT_GAP_S = 10.0              # a request's tail settles in 3 to 5 s on this pod (lab/README.md, T10)
DEFAULT_SETTLE_S = 5.0            # before the idle window, so the tail of anything before it has settled
DEFAULT_MAX_TOKENS = 128
DEFAULT_OUT = Path("fixtures") / "telemetry"
WAIT_PAUSE_S = 120.0              # between checks while the teacher is busy or not answering
CONSERVATION_TOLERANCE = 1e-6     # relative: answers plus gaps against the whole run
GPU_POWER = "telemetry.gpu.power"


@dataclass(frozen=True)
class Answer:
    """One request: its window on the client's clock (epoch seconds) and what came back."""

    index: int
    request_id: str
    prompt_id: str
    started: float
    ended: float
    tokens_in: int | None
    tokens_out: int | None
    first_token: float | None = None
    finish_reason: str | None = None
    error: str | None = None


def _dt(s: float) -> datetime:
    return datetime.fromtimestamp(s, UTC)


def _iso(s: float) -> str:
    return _dt(s).isoformat().replace("+00:00", "Z")


# --- the clients --------------------------------------------------------------------------------------------------

class SamplerReader(bench.RemotePower):
    """bench's reader of the pod sampler through the hub proxy (GET /status, /samples?after=<seq>), keeping the four
    timestamps of every answer so the pod's clock offset can be measured from the best one."""

    def __init__(self, link: Any, port: int = bench.SAMPLER_PORT, *, now: Callable[[], float] = time.time) -> None:
        super().__init__(link, port)
        self.now = now
        self.exchanges: list[Exchange] = []

    def _get(self, path: str, **params: Any) -> dict[str, Any]:
        t1 = self.now()
        body = super()._get(path, **params)
        t4 = self.now()
        exchange = exchange_of(_dt(t1), body, _dt(t4))
        if exchange is not None:
            self.exchanges.append(exchange)
        return body


def parse_load(text: str) -> dict[str, int] | None:
    """vLLM's request counters from its Prometheus text: requests running, waiting, and finished (all reasons)."""
    found: dict[str, float] = {}
    for line in text.splitlines():
        match = re.match(r"^vllm:(num_requests_running|num_requests_waiting|request_success_total)(\{[^}]*\})?\s+"
                         r"([-+0-9.eE]+)\s*$", line)
        if match:
            found[match.group(1)] = found.get(match.group(1), 0.0) + float(match.group(3))
    if "num_requests_running" not in found or "num_requests_waiting" not in found:
        return None
    return {"running": int(found["num_requests_running"]), "waiting": int(found["num_requests_waiting"]),
            "finished": int(found.get("request_success_total", 0))}


class ProxyTeacher:
    """The teacher through the hub proxy, asked bench's way: bench's endpoint, prompt set, settings and request
    runner, text prompts only, one at a time."""

    def __init__(self, link: Any, *, port: int = 8000, model: str = "teacher", max_tokens: int = DEFAULT_MAX_TOKENS,
                 prompts: bench.PromptSet | None = None, clock: bench.WallClock | None = None) -> None:
        self.link = link
        self.port = port
        self.endpoint = bench.ProxyEndpoint(link, port)
        self.prompts = prompts or bench.PromptSet.load()
        self.text = [i for i, p in enumerate(self.prompts.prompts) if p["kind"] == "text"]
        self.settings = bench.Settings(model=model, max_tokens=max_tokens)   # greedy, ignore_eos, thinking off
        self.clock = clock or bench.WallClock()
        self.tag = uuid.uuid4().hex[:8]

    def now(self) -> float:
        return self.clock.wall(time.perf_counter())

    def ready(self) -> bool:
        try:
            return any(m.get("id") == self.settings.model for m in self.endpoint.models())
        except Exception:
            return False

    def load(self) -> dict[str, int] | None:
        try:
            reply = self.link.request("GET", self.link.proxy_rel(self.port, "metrics"))
            return parse_load(reply.text)
        except Exception:
            return None

    def ask(self, index: int) -> Answer:
        i = self.text[index % len(self.text)]
        request_id = f"t8-{self.tag}-{index:03d}"
        body = self.prompts.request_body(i, self.settings, request_id)
        r = bench.run_request(self.endpoint, body, role="measure", stream=0, index=index,
                              prompt_id=self.prompts.prompt_id(i), clock=self.clock)
        ended = self.now()
        return Answer(index=index, request_id=request_id, prompt_id=r.prompt_id, started=r.sent, ended=ended,
                      tokens_in=r.prompt_tokens, tokens_out=r.tokens, first_token=r.first,
                      finish_reason=r.finish_reason, error=r.error)

    def describe(self) -> dict[str, Any]:
        s = self.settings
        return {"model": s.model, "max_tokens": s.max_tokens, "temperature": s.temperature, "top_p": 1.0, "seed": 0,
                "ignore_eos": s.ignore_eos, "thinking": s.thinking, "nonce": s.nonce_on,
                "prompts": self.prompts.describe(), "text_prompts": [self.prompts.prompt_id(i) for i in self.text]}


# --- the computation ----------------------------------------------------------------------------------------------

def _gpu_samples(samples: Iterable[Mapping[str, Any]]) -> list[GpuPower]:
    out = []
    for e in samples:
        if e.get("type") != GPU_POWER:
            continue
        event = parse_event({k: v for k, v in e.items() if k != "t_local"})
        if isinstance(event, GpuPower):
            out.append(event)
    return out


def _stats(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None, "sd": None}
    return {"mean": statistics.fmean(values), "median": statistics.median(values), "min": min(values),
            "max": max(values), "sd": statistics.stdev(values) if len(values) > 1 else None}


def compute(samples: Iterable[Mapping[str, Any]], answers: Sequence[Answer], offset: ClockOffset | None, *,
            idle: tuple[float, float], run_end: float, node: str = NODE, clock: str = CLOCK,
            min_samples: int = MIN_SAMPLES) -> dict[str, Any]:
    """Every number of the check from the raw rows. Times are the client's; `offset` is the pod's clock minus it."""
    gpu = _gpu_samples(samples)
    shift = offset.offset_s if offset is not None else 0.0
    trace = sorted((e.ts.timestamp() - shift, float(e.socket_w)) for e in gpu if e.socket_w is not None)
    order = sorted(answers, key=lambda a: a.started)
    if not order:
        raise ValueError("no answers to check")
    run_start = order[0].started

    # T4's attribution, fed as the box feeds it: the samples on the pod's clock, the windows on ours, the offset
    projection = JoulesPerAnswerProjection(Calibration(min_samples=min_samples, settle_s=0.0, reference=clock))
    if offset is not None:
        projection.on_event(offset)
    for event in gpu:
        projection.on_event(event)
    for a in order:
        projection.on_event(RequestWindow(
            node=node, ts=_dt(a.ended), request_id=a.request_id, started=_dt(a.started), ended=_dt(a.ended),
            tokens_in=a.tokens_in, tokens_out=a.tokens_out, target="teacher", model="teacher", kind="chat",
            first_token=_dt(a.first_token) if a.first_token is not None and a.started <= a.first_token <= a.ended
            else None, clock=clock if offset is not None else None))
    values = {v.request_id: v for v in projection.ready()}
    missing = [a.request_id for a in order if a.request_id not in values]
    if missing:
        raise ValueError(f"{len(missing)} answers got no value from the attribution ({projection.waiting})")
    joules = [values[a.request_id].joules for a in order]

    gaps = []
    for i, a in enumerate(order):
        end = order[i + 1].started if i + 1 < len(order) else run_end
        if end < a.ended:
            raise ValueError(f"answer {a.request_id} overlaps the next one: the run was not sequential")
        gaps.append((a.ended, end, integrate(trace, a.ended, end)))
    if any(j is None for _, _, j in gaps):
        raise ValueError("the samples do not cover every gap")
    whole = integrate(trace, run_start, run_end)
    idle_j = integrate(trace, idle[0], idle[1])
    if whole is None or idle_j is None:
        raise ValueError("the samples do not cover the run and the idle window")
    idle_s = idle[1] - idle[0]
    idle_w = idle_j / idle_s
    answers_j = sum(joules)
    gap_j = sum(j for _, _, j in gaps)
    gap_s = sum(b - a for a, b, _ in gaps)
    residual = whole - answers_j - gap_j
    conserved = abs(residual) <= CONSERVATION_TOLERANCE * max(abs(whole), 1.0)
    missed = gap_j - idle_w * gap_s
    # which side of the gaps the miss sits on: the first half (the tail after an answer, the sensor's lag) or the
    # second (the ramp up to the next answer, an edge of the next window between two samples)
    halves = [(a, (a + b) / 2, b) for a, b, _ in gaps]
    after_end = sum(integrate(trace, a, m) - idle_w * (m - a) for a, m, _ in halves)
    before_start = sum(integrate(trace, m, b) - idle_w * (b - m) for _, m, b in halves)
    share = missed / (answers_j + missed) if answers_j + missed else None

    arm_a = joules
    arm_b = [j + (g - idle_w * (b - a)) for j, (a, b, g) in zip(joules, gaps)]
    reading = verdict(arm_a, arm_b)
    counts = [values[a.request_id].samples for a in order]
    under = sum(1 for c in counts if c < min_samples)

    idle_events = [e for e in gpu if idle[0] <= e.ts.timestamp() - shift <= idle[1]]
    idle_watts = [float(e.socket_w) for e in idle_events if e.socket_w is not None]
    run_events = [e for e in gpu if run_start <= e.ts.timestamp() - shift <= run_end]
    summary = pod_power.summarize([e.model_dump(mode="json") for e in run_events]) if run_events else {}
    tokens = [a.tokens_out for a in order if a.tokens_out]
    reasons_lc = Counter(r for v in values.values() for r in v.reasons)

    reasons = []
    if not conserved:
        reasons.append(f"the answers and gaps do not add up to the run's energy (residual {residual:.3f} J)")
    if under:
        reasons.append(f"{under} of {len(order)} windows hold fewer than {min_samples} samples")
    if reading.label == B_AHEAD:
        reasons.append(f"the windows miss a measurable {100 * (share or 0):.2f} percent of the answers' energy "
                       f"(the gaps above idle, {reading.note})")
    elif reading.label == A_AHEAD:
        reasons.append("the gaps read below the idle watts measured before the run: that idle did not hold")
    elif reading.label == TOO_EARLY:
        reasons.append(f"too early: {reading.note}")
    loaded = sum(1 for e in idle_events if e.under_load)
    if loaded:
        reasons.append(f"{loaded} samples of the idle window were under load")
    failed = sum(1 for a in order if a.error)
    if failed:
        reasons.append(f"{failed} requests failed")
    trustworthy = not reasons and reading.label == NO_DIFFERENCE
    period = gpu[0].period_ms if gpu else None
    return {
        "node": node, "clock": clock, "basis": "gross", "answers": len(order), "failed": failed,
        "idle": {"start": _iso(idle[0]), "end": _iso(idle[1]), "seconds": idle_s, "watts": idle_w,
                 "energy_j": idle_j, "samples": len(idle_events), "under_load_samples": loaded,
                 "min_w": min(idle_watts) if idle_watts else None, "max_w": max(idle_watts) if idle_watts else None},
        "run": {"start": _iso(run_start), "end": _iso(run_end), "seconds": run_end - run_start, "energy_j": whole,
                "samples": len(run_events)},
        "answers_energy_j": answers_j, "gap_energy_j": gap_j, "gap_seconds": gap_s, "gap_count": len(gaps),
        "answer_seconds": sum(a.ended - a.started for a in order),
        "conservation": {"residual_j": residual, "relative": residual / whole if whole else None, "ok": conserved,
                         "tolerance": CONSERVATION_TOLERANCE},
        "missed_j": missed, "missed_share": share,
        "missed_split": {"after_end_j": after_end, "before_start_j": before_start},
        "attribution_error_pct": None if share is None else -100.0 * share,
        "samples_per_window": {"median": float(statistics.median(counts)), "min": min(counts), "max": max(counts),
                               "under_min": under, "min_required": min_samples},
        "joules_per_answer": _stats(joules),
        "joules_per_token_mean": statistics.fmean(j / a.tokens_out for j, a in zip(joules, order) if a.tokens_out)
        if tokens else None,
        "tokens_out": _stats([float(t) for t in tokens]),
        "seconds_per_answer": _stats([a.ended - a.started for a in order]),
        "period_ms": period, "measured_period_ms": summary.get("measured_period_ms"),
        "max_gap_ms": summary.get("max_gap_ms"), "sampler": summary.get("sampler"), "source": summary.get("source"),
        "power_field": summary.get("power_field"), "restarts": summary.get("restarts"),
        "temps_c": {"start": summary.get("start_temps_c"), "end": summary.get("end_temps_c")},
        "clock_offset": None if offset is None else {"offset_s": offset.offset_s, "error_s": offset.error_s,
                                                     "round_trip_s": offset.round_trip_s,
                                                     "exchanges": offset.exchanges},
        "low_confidence_reasons": dict(reasons_lc),
        "verdict": reading.as_dict(),
        "trustworthy": trustworthy, "reasons": reasons,
    }


# --- the run ------------------------------------------------------------------------------------------------------

def _busy(load: Mapping[str, int] | None) -> bool:
    return load is not None and (load["running"] > 0 or load["waiting"] > 0)


def _jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=False) + "\n" for r in rows), encoding="utf-8")


def _files(out_dir: Path, tag: str) -> dict[str, Path]:
    return {k: out_dir / f"attribution_check_{tag}_{k}.jsonl" for k in ("samples", "requests", "run")}


def _answer_row(a: Answer) -> dict[str, Any]:
    return {**asdict(a), "started_iso": _iso(a.started), "ended_iso": _iso(a.ended)}


def _write(out_dir: Path, tag: str, events: Sequence[Mapping[str, Any]], answers: Sequence[Answer],
           meta: Sequence[Mapping[str, Any]]) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = _files(out_dir, tag)
    _jsonl(files["samples"], ({k: v for k, v in e.items() if k != "t_local"} for e in events))
    _jsonl(files["requests"], (_answer_row(a) for a in answers))
    _jsonl(files["run"], meta)
    return [str(p) for p in files.values()]


def _exchange_row(e: Exchange) -> dict[str, Any]:
    return {"kind": "exchange", "t1": e.t1.timestamp(), "t2": e.t2.isoformat(), "t3": e.t3.isoformat(),
            "t4": e.t4.timestamp()}


def run_experiment(teacher: Any, power: Any, *, requests: int = DEFAULT_REQUESTS, idle_s: float = DEFAULT_IDLE_S,
                   gap_s: float = DEFAULT_GAP_S, settle_s: float = DEFAULT_SETTLE_S, now: Callable[[], float],
                   sleep: Callable[[float], None], out_dir: str | Path | None = None, tag: str | None = None,
                   log: Callable[[str], None] = lambda message: None) -> dict[str, Any]:
    """Idle, then the requests one at a time with a gap after each, then the computation. Refuses (outcome busy)
    when the teacher has requests in flight before it starts; sends nothing then."""
    before = teacher.load()
    if _busy(before):
        return {"ok": False, "outcome": "busy", "error": f"the teacher has requests in flight: {before}"}
    power.start()
    if not power.available:
        return {"ok": False, "outcome": "no_sampler", "error": power.reason}
    sleep(settle_s)
    idle_start = now()
    sleep(idle_s)
    idle_end = now()
    after_idle = teacher.load()
    sleep(2 * power.period_s)   # the sample after the idle window's end is read before the first request starts
    log(f"idle window {idle_end - idle_start:.1f} s, teacher {after_idle}")
    answers: list[Answer] = []
    meta: list[dict[str, Any]] = [
        {"kind": "settings", "node": NODE, "clock": CLOCK, "requests": requests, "idle_s": idle_s, "gap_s": gap_s,
         "settle_s": settle_s, "teacher": teacher.describe() if hasattr(teacher, "describe") else None,
         "sampler_port": getattr(power, "port", None)},
        {"kind": "window", "name": "idle", "start": idle_start, "end": idle_end},
        {"kind": "load", "when": "before", "load": before}, {"kind": "load", "when": "after_idle", "load": after_idle}]
    for i in range(requests):
        answer = teacher.ask(i)
        answers.append(answer)
        log(f"{i + 1}/{requests} {answer.prompt_id} {answer.ended - answer.started:.1f} s "
            f"{answer.tokens_out} tokens{' ERROR ' + answer.error if answer.error else ''}")
        sleep(gap_s)
        events = power.events()   # keeps the cursor fresh and adds a clock exchange
        if out_dir is not None and tag and (i + 1) % 10 == 0:
            _write(Path(out_dir), tag, events, answers, meta)
    run_end = now()
    sleep(2 * power.period_s + 1.0)   # a sample at or after the end of the last gap
    events = power.events()
    after = teacher.load()
    offset = best_offset(NODE, power.exchanges, reference=CLOCK) if getattr(power, "exchanges", None) else None
    meta += [{"kind": "window", "name": "run", "start": answers[0].started if answers else None, "end": run_end},
             {"kind": "load", "when": "after", "load": after}, *(_exchange_row(e) for e in power.exchanges)]
    files = _write(Path(out_dir), tag, events, answers, meta) if out_dir is not None and tag else []
    result = compute(events, answers, offset, idle=(idle_start, idle_end), run_end=run_end)
    _others(result, before, after_idle, after, len(answers))
    return {"ok": True, "outcome": "ok", "result": result, "files": files}


def _others(result: dict[str, Any], before: Mapping[str, int] | None, after_idle: Mapping[str, int] | None,
            after: Mapping[str, int] | None, sent: int) -> None:
    """Requests other clients finished on the teacher between the start and the end, from vLLM's counters."""
    others = None
    if before is not None and after is not None:
        others = after["finished"] - before["finished"] - sent
    result["other_requests"] = others
    result["teacher_load"] = {"before": before, "after_idle": after_idle, "after": after}
    if others is None:
        result["reasons"].append("vLLM's counters could not be read: other clients' requests are not ruled out")
    elif others != 0:
        result["reasons"].append(f"{others} requests from other clients finished on the teacher during the run")
    if _busy(after_idle):
        result["reasons"].append(f"the teacher had requests in flight right after the idle window: {after_idle}")
    result["trustworthy"] = result["trustworthy"] and not result["reasons"]


def replay(directory: str | Path, tag: str) -> dict[str, Any]:
    """The result again, from the three files a run wrote."""
    files = _files(Path(directory), tag)
    samples = pod_power.read_samples([files["samples"]])
    answers = [Answer(**{k: v for k, v in row.items() if k in Answer.__dataclass_fields__})
               for row in pod_power.read_samples([files["requests"]])]
    meta = pod_power.read_samples([files["run"]])
    windows = {m["name"]: m for m in meta if m.get("kind") == "window"}
    loads = {m["when"]: m["load"] for m in meta if m.get("kind") == "load"}
    exchanges = [Exchange(_dt(m["t1"]), datetime.fromisoformat(m["t2"]), datetime.fromisoformat(m["t3"]), _dt(m["t4"]))
                 for m in meta if m.get("kind") == "exchange"]
    offset = best_offset(NODE, exchanges, reference=CLOCK) if exchanges else None
    result = compute(samples, answers, offset, idle=(windows["idle"]["start"], windows["idle"]["end"]),
                     run_end=windows["run"]["end"])
    _others(result, loads.get("before"), loads.get("after_idle"), loads.get("after"), len(answers))
    return result


# --- the ledger row -----------------------------------------------------------------------------------------------

def _r(value: Any, digits: int = 3) -> Any:
    return round(value, digits) if isinstance(value, float) else value


def record_result(result: Mapping[str, Any], *, source: str, state_dir: str | Path | None = None) -> int:
    """One ledger row with the run's numbers, the idle watts beside them, and the verdict."""
    metrics = {
        "telemetry.attribution_error_pct": _r(result["attribution_error_pct"], 4),
        "telemetry.samples_per_window": result["samples_per_window"]["median"],
        "pod.idle_watts": _r(result["idle"]["watts"]),
        "telemetry.joules_per_answer": _r(result["joules_per_answer"]["mean"]),
        "telemetry.answers_energy_j": _r(result["answers_energy_j"]),
        "telemetry.gap_energy_j": _r(result["gap_energy_j"]),
        "telemetry.run_energy_j": _r(result["run"]["energy_j"]),
        "telemetry.missed_energy_j": _r(result["missed_j"]),
    }
    units = {"telemetry.attribution_error_pct": "%", "telemetry.samples_per_window": "count", "pod.idle_watts": "W",
             **{k: "J" for k in metrics if k.endswith("_j") or k == "telemetry.joules_per_answer"}}
    conditions = {k: v for k, v in result.items() if k not in ("reasons",)}
    conditions["error"] = "unknown: the sensor's own error is not measured on our node against a reference (TT1)"
    verdict_label = "trustworthy" if result["trustworthy"] else "not trustworthy as is"
    note = f"{verdict_label}: " + ("; ".join(result["reasons"]) or result["verdict"]["note"])
    return record(VERB, "ok", metrics={k: v for k, v in metrics.items() if v is not None}, task=TASK, node=NODE,
                  units=units, conditions=json.loads(json.dumps(conditions, default=str)), low_confidence=True,
                  source=source, note=note, state_dir=state_dir)


# --- the command line ---------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m lab attribution-check", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="idle, the requests and the computation, on the live teacher")
    r.add_argument("--account", default="C", help="hub account letter of the teacher's pod (C)")
    r.add_argument("--port", type=int, default=8000)
    r.add_argument("--power-port", type=int, default=bench.SAMPLER_PORT)
    r.add_argument("--requests", type=int, default=DEFAULT_REQUESTS)
    r.add_argument("--idle-s", type=float, default=DEFAULT_IDLE_S)
    r.add_argument("--gap-s", type=float, default=DEFAULT_GAP_S)
    r.add_argument("--settle-s", type=float, default=DEFAULT_SETTLE_S)
    r.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    r.add_argument("--wait-min", type=float, default=0.0, help="wait this long for a busy or silent teacher")
    r.add_argument("--no-ledger", action="store_true")
    for p in (r, sub.add_parser("replay", help="the numbers again from a run's files")):
        p.add_argument("--out", "--dir", dest="out", default=str(DEFAULT_OUT))
        p.add_argument("--tag", default=None, help="the files' tag (default: the UTC date)")
        p.add_argument("--state-dir", default=None)
    sub.choices["replay"].add_argument("--record", action="store_true", help="write the ledger row from the files")
    return parser


def _log(message: str) -> None:
    print(f"[attribution-check {time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def main(argv: Sequence[str] | None = None, *, teacher: Any = None, power: Any = None,
         now: Callable[[], float] | None = None, sleep: Callable[[float], None] | None = None) -> dict[str, Any]:
    """Validate joules per answer on pod A: answers, gaps and whole-run energy, idle watts, the missed share."""
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    tag = args.tag or datetime.now(UTC).date().isoformat()
    source = f"lab/attribution_check.py from {Path(args.out).as_posix()}/attribution_check_{tag}_*.jsonl"
    if args.cmd == "replay":
        result = replay(args.out, tag)
        out: dict[str, Any] = {"ok": True, "outcome": "ok", "result": result}
        if args.record:
            out["ledger_seq"] = record_result(result, source=source, state_dir=args.state_dir)
        return out
    link = None
    if teacher is None or power is None:
        from lab import hub
        from lab.proxy import PodLink

        link = PodLink.for_account(hub.load_accounts([args.account.upper()])[args.account.upper()])
        clock = bench.WallClock()
        teacher = ProxyTeacher(link, port=args.port, max_tokens=args.max_tokens, clock=clock)
        power = SamplerReader(link, args.power_port, now=teacher.now)
        now = teacher.now
    now = now or time.time
    sleep = sleep or time.sleep
    try:
        waited = 0.0
        while True:
            ready, load = teacher.ready(), teacher.load()
            if ready and not _busy(load):
                break
            if waited + WAIT_PAUSE_S > args.wait_min * 60.0 + 1e-9:
                why = "busy" if ready else "not_ready"
                return {"ok": False, "outcome": "busy" if ready else "not_ready", "waited_s": waited,
                        "error": f"the teacher {'has requests in flight' if ready else 'does not answer'}: {load}",
                        "why": why}
            _log(f"teacher {'busy ' + str(load) if ready else 'not answering'}; waiting {WAIT_PAUSE_S:.0f} s")
            sleep(WAIT_PAUSE_S)
            waited += WAIT_PAUSE_S
        out = run_experiment(teacher, power, requests=args.requests, idle_s=args.idle_s, gap_s=args.gap_s,
                             settle_s=args.settle_s, now=now, sleep=sleep, out_dir=args.out, tag=tag, log=_log)
    finally:
        if link is not None:
            link.close()
    if out["ok"] and not args.no_ledger:
        out["ledger_seq"] = record_result(out["result"], source=source, state_dir=args.state_dir)
    return out


if __name__ == "__main__":
    answer = main()
    print(json.dumps(answer, default=str))
    sys.exit(0 if answer.get("ok") else 1)
