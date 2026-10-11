"""The routing hook's closed-loop test (T11): the same lab requests routed under policy one and policy two, with the
effect on joules per answer, latency and quality measured and read with the verdict math.

    python -m lab routing-closed-loop --fake                                  # the teacher and student fakes
    python -m lab routing-closed-loop --account C --student-url http://127.0.0.1:8080 \\
        --power-log data/events.jsonl --window-clock jetson                    # the real run (lab/T11_REAL_RUN.md)

The requests. Labeling requests drawn from the L11 fixture dataset (fixtures/learning/tool-calling-fixture, built by
lab/fixture_dataset.py), at least MIN_REQUESTS of them, shuffled with a fixed seed. Golden questions are never used:
each one is about the user's own space and answers on the box under both policies. Each request goes to the hook as
RouteRequest(kind "label", source "learning", own_space false, student_can --student-can): work the Learning context
already sends to the lab, so the hook's privacy line lets it through to the policy.

The two arms, interleaved request by request (A, B, A, B, ...):
    A   policy one, capability (contexts/telemetry/application/routing.py), the control
    B   policy two, energy: the target whose latest joules per answer for "label" in the telemetry snapshot is lower
Both go through runtime/routing.py's route(), which is T5's hook, with the snapshot of the moment.

The closed loop. The snapshot is T4's: a JoulesPerAnswerProjection (contexts/telemetry/application/joules.py) fed the
power samples of both nodes and one telemetry.request window per answer, and read with its routing_snapshot(). Every
answer of either arm feeds it, so policy two decides on what the run has just measured. Before the arms, --probes
requests per node (drawn after the arm requests, so never one of them) put a first value for each node into the
snapshot; without them policy two can only fall back to capability, which the report says. After each answer the
meter polls the power feed until the projection reports that answer's joules (it waits for the samples to run
--settle-s past the window's end, at most --wait-s); an answer it cannot measure keeps joules null and is left out of
the energy arm, and the report counts it.

Each answer is asked with lab/label.py's own labeler (the tool-trace task: the agent's TOOLS run against the golden
fixture memory, the same checks before a label is kept), against the target the hook chose: the teacher (vLLM on pod
A, through the hub proxy) or the student (llama-server on the Jetson). Both speak the OpenAI chat API, so both are a
lab.label.Teacher client; the student's under its own model name.

What is recorded, per request: the arm, the policy, the policy that decided (energy, or energy>capability when the
snapshot held no value for a node), the target and node, the snapshot's joules for both nodes at decision time, the
joules per answer and whether they are low confidence, the latency (the labeler's seconds over every round of the
request) and the quality score. Quality is the learning track's label agreement (learning.label_agreement): 1 when
the label was kept and its tool calls, names and arguments in order, are the fixture reference's (expected.jsonl),
else 0. A label the labeler rejected or that failed scores 0.

The verdict. Per arm, the mean and sample standard deviation of each of the three. lab/machine/verdict.py reads B
against A: joules per answer and latency as measurements (Welch's t, lower is better), label agreement as a rate
(two-proportion z, higher is better). B changed energy, or quality, when that reading names an arm ahead (p under
0.05): a difference beyond the spread. The telemetry rubric's keep rule (TT6) is kept beside it: policy two is kept
when joules per answer drop beyond the spread while quality and latency stay inside theirs.

What it writes, under --out/<run id>/: requests.jsonl (one line per answer, the probes marked arm "probe"),
report.json (the summary, the verdicts and the finding), and t0_result.txt, the experiment result text the
dispatcher posts on T0 with its citation. Into the ledger (--state-dir): one row per arm (verb routing-closed-loop,
task T11) and two T0 experiments with their results (lab/machine/experiments.py): the energy one on
telemetry.joules_per_answer and the quality one on learning.label_agreement. A --fake run writes its ledger rows
into --out/<run id>/state unless --state-dir names another, so fake numbers never reach the lab's own ledger.

Rubric test 6. Whether this run meets telemetry rubric test 6 goes into the report and the T0 text as a finding for
the merge pass. Nothing under lab/refine/ is written (plumbing ruling 5).
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import random
import secrets
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if not __package__:  # run by path: the repo root, one folder up, is what holds contracts/ and lab/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contexts.telemetry.application.joules import Calibration, JoulesPerAnswerProjection
from contexts.telemetry.application.routing import CAPABILITY, ENERGY, STUDENT_NODE, TEACHER_NODE, capability, \
    energy, joules_for
from contracts.events import Event, parse_event
from contracts.learning import DatasetExample
from contracts.telemetry import JoulesPerAnswer, RequestWindow
from lab.dataset import Dataset, load
from lab.fixture_dataset import Expected, load_expected
from lab.label import Decode, Teacher, Toolbox, _Labeler, golden_toolbox
from lab.machine import experiments, stop
from lab.machine import verdict as verdict_math
from lab.machine.record import record
from runtime.routing import Policy, RouteRequest, route

__all__ = ["VERB", "TASK", "JOULES", "LATENCY", "QUALITY", "MIN_REQUESTS", "KIND", "REQUESTS_FILE", "REPORT_FILE",
           "T0_FILE", "LoopError", "Request", "Record", "Meter", "JsonlFeed", "RealClock", "Target", "FakeSetup",
           "Result", "draw_requests", "agreement", "run_arms", "summarize", "run", "run_fake", "main"]

VERB = "routing-closed-loop"
TASK = "T11"
KIND = "label"                              # the kind of work, as telemetry and the hook name it
JOULES = "telemetry.joules_per_answer"
LATENCY = "label.seconds"                   # the labeler's own name for the seconds one example took (lab/label.py)
QUALITY = "learning.label_agreement"
UNITS = {JOULES: "J", LATENCY: "s", QUALITY: "fraction"}
MIN_REQUESTS = 100
DEFAULT_SEED = 11
DEFAULT_PROBES = 1
SETTLE_S = 2.0                              # samples past a window's end before its joules are read
WAIT_S = 30.0                               # longest wait for an answer's joules
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "learning" / "tool-calling-fixture"
DEFAULT_OUT = Path(__file__).resolve().parent / "state" / VERB
REQUESTS_FILE = "requests.jsonl"
REPORT_FILE = "report.json"
T0_FILE = "t0_result.txt"
ARMS = (("A", CAPABILITY, capability), ("B", ENERGY, energy))
AHEAD = (verdict_math.B_AHEAD, verdict_math.A_AHEAD)
SOURCES = ("contexts/telemetry/application/routing.py (T5), contexts/telemetry/application/joules.py (T4), "
           "lab/machine/verdict.py, telemetry rubric TT6")


class LoopError(ValueError):
    """A run that cannot start: too few requests, a dataset that does not hold up, a missing target."""


# --- the requests -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Request:
    example: DatasetExample
    expected: Expected
    route: RouteRequest


def _pool(root: Path) -> tuple[Dataset, list[DatasetExample], dict[str, Expected]]:
    data = load(root, require_images=False)
    expected = load_expected(root)
    missing = [e.id for e in data.examples if e.id not in expected]
    if missing:
        raise LoopError(f"{len(missing)} examples have no reference in expected.jsonl, e.g. {missing[0]}")
    return data, [e for e in data.examples if expected[e.id].origin != "golden"], expected


def _shuffled(pool: list[DatasetExample], seed: int) -> list[DatasetExample]:
    order = list(pool)
    random.Random(seed).shuffle(order)
    return order


def draw_requests(root: str | os.PathLike[str] = FIXTURE, *, n: int = MIN_REQUESTS, seed: int = DEFAULT_SEED,
                  skip: int = 0, student_can: bool = True) -> list[Request]:
    """n labeling requests from the fixture, never a golden question, in the order a seeded shuffle gives; skip
    passes over the first ones (the probes are drawn after the arm requests this way)."""
    _, pool, expected = _pool(Path(root))
    if n + skip > len(pool):
        raise LoopError(f"{n + skip} requests asked, but the fixture holds only {len(pool)} examples that are not "
                        "golden questions")
    hook = RouteRequest(kind=KIND, source="learning", own_space=False, student_can=student_can)
    return [Request(e, expected[e.id], hook) for e in _shuffled(pool, seed)[skip: skip + n]]


def agreement(label: str | None, expected: Expected) -> float:
    """learning.label_agreement for one label: 1.0 when its tool calls (names and arguments, in order) are the
    reference's, else 0.0. No label (rejected or failed) is 0.0."""
    if not label:
        return 0.0
    try:
        value = json.loads(label)
    except ValueError:
        return 0.0
    calls = value.get("tool_calls") if isinstance(value, dict) else None
    if not isinstance(calls, list):
        return 0.0
    made = [{"name": c.get("name"), "arguments": c.get("arguments")} for c in calls if isinstance(c, dict)]
    gold = [{"name": c.name, "arguments": c.arguments} for c in expected.label.tool_calls]
    return 1.0 if made == gold and len(made) == len(calls) else 0.0


# --- the clock and the power feed ---------------------------------------------------------------------------------

class RealClock:
    """Wall-clock seconds since the epoch, and a real sleep."""

    def time(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        time.sleep(max(0.0, seconds))


class JsonlFeed:
    """Telemetry events appended to JSONL files (the Jetson's event log, a pod sampler's day file): poll() returns
    the events written since the last poll. A line that is not an event is skipped; a line still being written is
    read on the next poll."""

    def __init__(self, paths: Sequence[str | os.PathLike[str]]) -> None:
        self.paths = [Path(p) for p in paths]
        self._offsets = {p: 0 for p in self.paths}

    def poll(self) -> list[Event]:
        out: list[Event] = []
        for path in self.paths:
            try:
                with path.open("rb") as fh:
                    fh.seek(self._offsets[path])
                    data = fh.read()
            except FileNotFoundError:
                continue
            end = data.rfind(b"\n") + 1
            self._offsets[path] += end
            for line in data[:end].splitlines():
                if not line.strip():
                    continue
                try:
                    out.append(parse_event(line))
                except ValueError:
                    continue
        return out


def _dt(t: float) -> dt.datetime:
    return dt.datetime.fromtimestamp(t, tz=dt.UTC)


class Meter:
    """T4's joules per answer and the snapshot the energy policy reads, fed by a power feed and the run's windows.

    window_clock names the node whose clock the run's own clock is ("jetson" when the run is on the Jetson): a window
    on another node is stamped with it, and the projection moves it onto that node's clock with the latest
    telemetry.clock.offset in the feed. None when every node shares the run's clock (the fakes)."""

    def __init__(self, feed: Any, *, clock: Any, settle_s: float = SETTLE_S, wait_s: float = WAIT_S,
                 window_clock: str | None = None, step_s: float = 0.5) -> None:
        self.feed = feed
        self.clock = clock
        self.wait_s = wait_s
        self.step_s = step_s
        self.window_clock = window_clock
        self.projection = JoulesPerAnswerProjection(Calibration(settle_s=settle_s))
        self._values: dict[tuple[str, str], JoulesPerAnswer] = {}

    def _fold(self) -> None:
        for event in self.feed.poll():
            self.projection.on_event(event)
        for value in self.projection.ready():
            self._values[(value.node, value.request_id)] = value

    def measure(self, node: str, request_id: str, started: float, ended: float, *, target: str, kind: str,
                policy: str, tokens_out: int | None = None, model: str | None = None) -> JoulesPerAnswer | None:
        """The answer's joules, once the projection can report them, or None after wait_s."""
        if hasattr(self.feed, "note"):
            self.feed.note(node, started, ended, request_id)
        clock = self.window_clock if self.window_clock and self.window_clock != node else None
        self.projection.on_event(RequestWindow(
            ts=_dt(ended), node=node, request_id=request_id, started=_dt(started), ended=_dt(ended), tokens_in=None,
            tokens_out=tokens_out, target=target, model=model, kind=kind, policy=policy, clock=clock))
        deadline = self.clock.time() + self.wait_s
        while True:
            self._fold()
            value = self._values.get((node, request_id))
            if value is not None or self.clock.time() >= deadline:
                return value
            self.clock.sleep(self.step_s)

    def snapshot(self) -> dict[str, Any]:
        self._fold()
        return self.projection.routing_snapshot()


# --- the targets and one answer -------------------------------------------------------------------------------------

@dataclass
class Target:
    """One end of the hook: "student" (the Jetson) or "teacher" (pod A), the node telemetry names it by, and the
    chat client that reaches it."""

    name: str
    node: str
    client: Teacher
    model: str | None = None


@dataclass
class Record:
    index: int
    arm: str                      # A, B, or probe
    policy: str                   # the policy the arm runs
    decided_by: str               # the policy name the decision carries (energy>capability when it fell back)
    target: str
    node: str
    example_id: str
    request_id: str
    snapshot_student_j: float | None
    snapshot_teacher_j: float | None
    joules: float | None
    low_confidence: bool | None
    latency_s: float
    quality: float
    status: str
    reason: str | None
    why: str

    def row(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _ask(index: int, arm: str, policy_name: str, policy: Policy, request: Request, targets: Mapping[str, Target],
         labelers: Mapping[str, _Labeler], meter: Meter, run_id: str, force: str | None = None) -> Record:
    snapshot = meter.snapshot()
    if force is None:
        decision = route(request.route, snapshot, policy)
        target, decided_by, why = decision.target, decision.policy, decision.reason
    else:
        target, decided_by, why = force, "probe", f"probe of the {force}"
    chosen = targets[target]
    request_id = f"{run_id}-{index:03d}-{arm}" + (f"-{target}" if force else "")
    started = meter.clock.time()
    outcome = labelers[target].label(request.example)
    ended = meter.clock.time()
    value = meter.measure(chosen.node, request_id, started, ended, target=target, kind=request.route.kind,
                          policy=decided_by, tokens_out=outcome.completion_tokens, model=chosen.model)
    label = outcome.target if outcome.status == "accepted" else None
    return Record(index=index, arm=arm, policy=policy_name, decided_by=decided_by, target=target, node=chosen.node,
                  example_id=request.example.id, request_id=request_id,
                  snapshot_student_j=joules_for(snapshot, STUDENT_NODE, request.route.kind),
                  snapshot_teacher_j=joules_for(snapshot, TEACHER_NODE, request.route.kind),
                  joules=value.joules if value else None, low_confidence=value.low_confidence if value else None,
                  latency_s=outcome.seconds, quality=agreement(label, request.expected), status=outcome.status,
                  reason=outcome.reason, why=why)


def run_arms(requests: Sequence[Request], probes: Sequence[Request], targets: Mapping[str, Target], meter: Meter, *,
             data: Dataset, toolbox: Toolbox, decode: Decode | None = None,
             run_id: str) -> tuple[list[Record], list[Record]]:
    """The probes (each on every target), then every request under A and then B: (arm records, probe records)."""
    decode = decode or Decode()
    labelers = {name: _Labeler(data, t.client, decode, toolbox, None, meter.clock.time) for name, t in targets.items()}
    probed = [_ask(i, "probe", "probe", capability, request, targets, labelers, meter, run_id, force=name)
              for i, request in enumerate(probes) for name in targets]
    records = []
    for i, request in enumerate(requests):
        stop.guard()
        for arm, name, policy in ARMS:
            records.append(_ask(i, arm, name, policy, request, targets, labelers, meter, run_id))
    return records, probed


# --- the reading ----------------------------------------------------------------------------------------------------

def _stats(values: Sequence[float]) -> dict[str, Any]:
    s = verdict_math.Samples.from_values(values)
    return {"n": s.n, "mean": s.mean, "sd": s.sd}


def _changed(reading: Mapping[str, Any]) -> bool:
    return reading["label"] in AHEAD


def summarize(records: Sequence[Record], probes: Sequence[Record] = ()) -> dict[str, Any]:
    """Per arm: mean and spread of joules, latency and agreement; the verdicts of B against A; the TT6 finding."""
    arms: dict[str, Any] = {}
    values: dict[str, dict[str, list[float]]] = {}
    for arm, policy, _ in ARMS:
        mine = [r for r in records if r.arm == arm]
        joules = [r.joules for r in mine if r.joules is not None]
        values[arm] = {JOULES: joules, LATENCY: [r.latency_s for r in mine], QUALITY: [r.quality for r in mine]}
        targets: dict[str, int] = {}
        decided: dict[str, int] = {}
        for r in mine:
            targets[r.target] = targets.get(r.target, 0) + 1
            decided[r.decided_by] = decided.get(r.decided_by, 0) + 1
        arms[arm] = {"policy": policy, "requests": len(mine), "targets": targets, "decided_by": decided,
                     "unmeasured": len(mine) - len(joules),
                     "low_confidence": sum(1 for r in mine if r.low_confidence),
                     **{metric: _stats(v) for metric, v in values[arm].items()},
                     "agreed": int(sum(values[arm][QUALITY]))}
    a, b = values["A"], values["B"]
    verdicts = {
        JOULES: verdict_math.verdict(a[JOULES], b[JOULES], higher_is_better=False).as_dict(),
        LATENCY: verdict_math.verdict(a[LATENCY], b[LATENCY], higher_is_better=False).as_dict(),
        QUALITY: verdict_math.verdict(verdict_math.Counts(arms["A"]["agreed"], len(a[QUALITY])),
                                      verdict_math.Counts(arms["B"]["agreed"], len(b[QUALITY]))).as_dict(),
    }
    changed = {"energy": _changed(verdicts[JOULES]), "quality": _changed(verdicts[QUALITY])}
    keep = (verdicts[JOULES]["label"] == verdict_math.B_AHEAD
            and verdicts[QUALITY]["label"] != verdict_math.A_AHEAD
            and verdicts[LATENCY]["label"] != verdict_math.A_AHEAD)
    summary = {"arms": arms, "verdicts": verdicts, "changed": changed, "keep_policy_two": keep,
               "probes": len(probes)}
    summary["rubric_tt6"] = _tt6(summary)
    return summary


def _tt6(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Whether this run meets telemetry rubric test 6, as a finding for the merge pass (never an edit)."""
    b = summary["arms"]["B"]
    read = b["decided_by"].get(ENERGY, 0)
    early = [m for m, v in summary["verdicts"].items() if v["label"] == verdict_math.TOO_EARLY]
    spread = all(summary["arms"][arm][m]["sd"] is not None for arm in ("A", "B") for m in (JOULES, LATENCY, QUALITY))
    met = read > 0 and not early and spread
    a_targets = summary["arms"]["A"]["targets"]
    baseline = (f"the control sent every request to the {next(iter(a_targets))}, the rubric's fixed all-"
                f"{next(iter(a_targets))} baseline" if len(a_targets) == 1
                else "the control split the requests between the nodes, so it is not the rubric's fixed baseline")
    reasons = []
    if read == 0:
        reasons.append("policy two never read a joules value from the snapshot (it fell back to capability on every "
                       "request)")
    if early:
        reasons.append(f"the verdict is too early for {', '.join(early)}")
    if not spread:
        reasons.append("an arm has no spread for a metric")
    v = summary["verdicts"]
    reading = (f"joules per answer {v[JOULES]['label']}, label agreement {v[QUALITY]['label']}, latency "
               f"{v[LATENCY]['label']}; by the TT6 keep rule policy two is "
               f"{'kept' if summary['keep_policy_two'] else 'not kept (revert or rule on it)'}")
    split = ""
    if len(b["targets"]) > 1:
        split = (f"; policy two split the requests ({', '.join(f'{t} {k}' for t, k in sorted(b['targets'].items()))}): "
                 "the snapshot holds only the latest joules per answer per node, so one short answer moves the next "
                 "decision, and with the arms interleaved that answer is often arm A's on the same request")
    text = (f"TT6 finding for the merge pass (not an edit to the rubric): {'met' if met else 'not met'}. "
            f"Policy two read the snapshot on {read} of {b['requests']} requests; {baseline}; {reading}{split}"
            + (f". Not met because {'; '.join(reasons)}" if reasons else "") + ".")
    return {"met": met, "read_snapshot": read, "reasons": reasons, "text": text}


# --- what is written ----------------------------------------------------------------------------------------------

def _ledger_rows(summary: Mapping[str, Any], *, run_id: str, source: str, fake: bool, state_dir: Any,
                 conditions: Mapping[str, Any]) -> dict[str, int]:
    seqs = {}
    for arm, policy, _ in ARMS:
        a = summary["arms"][arm]
        metrics = {m: a[m]["mean"] for m in (JOULES, LATENCY, QUALITY) if a[m]["mean"] is not None}
        seqs[arm] = record(
            VERB, "ok", metrics=metrics, refs={"run_id": run_id}, task=TASK, state_dir=state_dir, units=dict(UNITS),
            conditions={"arm": arm, "policy": policy, "n": a["requests"], "targets": a["targets"],
                        "decided_by": a["decided_by"], "unmeasured": a["unmeasured"],
                        **{m: a[m] for m in (JOULES, LATENCY, QUALITY)}, **conditions},
            low_confidence=fake or a["low_confidence"] > 0, source=source,
            note=f"arm {arm} of {run_id}: policy {policy}")
    return seqs


def _experiment(state_dir: Any, title: str, fields: Mapping[str, Any]) -> str:
    """The id of the T0 experiment with this title, added the first time."""
    for eid, e in experiments.load(state_dir).items():
        if e.get("title") == title and e.get("task") == experiments.DEFAULT_TASK:
            return eid
    return experiments.add({"title": title, **fields}, state_dir=state_dir)["id"]


def _t0(summary: Mapping[str, Any], records: Sequence[Record], *, run_id: str, source: str, citation: str,
        state_dir: Any) -> tuple[str, list[str]]:
    arms_text = {"A": "policy one, capability (the control)", "B": "policy two, energy: reads the telemetry snapshot"}
    common = {"track": "telemetry", "lever": "the routing hook's policy (routing.policy in config/models.yaml)",
              "arms": arms_text, "citation": citation, "node": None,
              "note": "T11 closed loop: arms interleaved request by request over the same labeling requests"}
    energy_id = _experiment(state_dir, "T11 routing hook: energy policy, joules per answer", {
        **common, "hypothesis": "routing by the snapshot's joules per answer lowers the joules each label costs",
        "kind": "measure", "metric": JOULES, "better": "lower", "secondary": [LATENCY]})
    quality_id = _experiment(state_dir, "T11 routing hook: energy policy, label agreement", {
        **common, "hypothesis": "routing by joules per answer leaves the label agreement where it was",
        "kind": "rate", "metric": QUALITY, "better": "higher"})
    by_arm = {arm: [r for r in records if r.arm == arm] for arm in ("A", "B")}
    note = f"run {run_id} ({source}); per-request records in {REQUESTS_FILE} of the run"
    experiments.record_result(energy_id, {arm: [r.joules for r in rs if r.joules is not None]
                                          for arm, rs in by_arm.items()},
                              final=True, note=note, refs={"run_id": run_id}, state_dir=state_dir)
    experiments.record_result(quality_id, {arm: verdict_math.Counts(int(sum(r.quality for r in rs)), len(rs))
                                           for arm, rs in by_arm.items()},
                              final=True, note=note, refs={"run_id": run_id}, state_dir=state_dir)
    folded = experiments.load(state_dir)
    lat = summary["arms"]
    v = summary["verdicts"][LATENCY]
    lines = [f"T11 closed-loop result, run {run_id} ({source})",
             experiments.checklist_line(folded[energy_id]),
             experiments.checklist_line(folded[quality_id]),
             (f"Latency ({LATENCY}, lower is better, not judged): A mean {lat['A'][LATENCY]['mean']:.4g} s sd "
              f"{lat['A'][LATENCY]['sd']:.4g}, B mean {lat['B'][LATENCY]['mean']:.4g} s sd "
              f"{lat['B'][LATENCY]['sd']:.4g}; {v['label']}."),
             (f"Changed beyond the spread: energy {'yes' if summary['changed']['energy'] else 'no'}, quality "
              f"{'yes' if summary['changed']['quality'] else 'no'}."),
             summary["rubric_tt6"]["text"],
             f"Citation: {citation}"]
    return "\n".join(lines) + "\n", [energy_id, quality_id]


@dataclass
class Result:
    run_id: str
    records: list[Record]
    probes: list[Record]
    summary: dict[str, Any]
    folder: Path
    experiments: list[str] = field(default_factory=list)
    seqs: dict[str, int] = field(default_factory=dict)


def run(requests: Sequence[Request], probes: Sequence[Request], targets: Mapping[str, Target], meter: Meter, *,
        data: Dataset, toolbox: Toolbox, out_dir: str | os.PathLike[str], state_dir: Any, source: str, fake: bool,
        run_id: str | None = None, conditions: Mapping[str, Any] | None = None) -> Result:
    """Route, answer and measure every request under both arms, then write the records, the report, the ledger
    rows and the T0 text."""
    run_id = run_id or f"t11-{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%SZ}-{secrets.token_hex(2)}"
    records, probed = run_arms(requests, probes, targets, meter, data=data, toolbox=toolbox, run_id=run_id)
    summary = summarize(records, probed)
    folder = Path(out_dir) / run_id
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / REQUESTS_FILE).open("w", encoding="utf-8", newline="\n") as fh:
        for r in records + probed:
            fh.write(json.dumps(r.row()) + "\n")
    seqs = _ledger_rows(summary, run_id=run_id, source=source, fake=fake, state_dir=state_dir,
                        conditions=dict(conditions or {}))
    citation = (f"T11 run {run_id}, python -m lab routing-closed-loop ({source}) in lab/routing_closed_loop.py; "
                f"{len(requests)} labeling requests from {data.dataset_id}, no golden question; per-request records "
                f"{REQUESTS_FILE} and {REPORT_FILE} of the run; ledger rows seq {seqs['A']} (arm A) and {seqs['B']} "
                f"(arm B), verb {VERB}, task {TASK}; {SOURCES}")
    text, eids = _t0(summary, records, run_id=run_id, source=source, citation=citation, state_dir=state_dir)
    (folder / T0_FILE).write_text(text, encoding="utf-8", newline="\n")
    report = {"run_id": run_id, "source": source, "requests": len(requests), "dataset_id": data.dataset_id,
              "experiments": eids, "ledger_seqs": seqs, "citation": citation, **summary}
    (folder / REPORT_FILE).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    return Result(run_id, records, probed, summary, folder, eids, seqs)


# --- the fakes ------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class FakeSetup:
    """The fakes' canned numbers (lab/student_fake.py says where the defaults come from)."""

    teacher_watts: float | None = None
    student_watts: float | None = None
    teacher_ttft_s: float = 0.52
    teacher_decode_tok_s: float = 2.83
    student_ttft_s: float | None = None
    student_decode_tok_s: float | None = None
    student_miss_share: float = 0.3
    jitter: float = 0.05


def run_fake(*, n: int = MIN_REQUESTS, seed: int = DEFAULT_SEED, probes: int = DEFAULT_PROBES,
             out_dir: str | os.PathLike[str], state_dir: Any = None, fake: FakeSetup | None = None,
             root: str | os.PathLike[str] = FIXTURE, student_can: bool = True, run_id: str | None = None) -> Result:
    """The whole test on the teacher and student fakes, on a fake clock."""
    fake = fake or FakeSetup()
    from lab import student_fake as sf
    from lab.hub_fake import FakeClock
    from lab.teacher_fake import FakeTeacher, reference_policy

    if n < MIN_REQUESTS:
        raise LoopError(f"the closed-loop test needs at least {MIN_REQUESTS} requests, not {n}")
    clock = FakeClock()
    expected = load_expected(root)
    teacher = FakeTeacher(reference_policy(expected), ttft_s=fake.teacher_ttft_s,
                          decode_tok_s=fake.teacher_decode_tok_s, clock=clock)
    student = sf.fake_student(expected, miss_share=fake.student_miss_share, seed=seed, clock=clock,
                              ttft_s=fake.student_ttft_s or sf.STUDENT_TTFT_S,
                              decode_tok_s=fake.student_decode_tok_s or sf.STUDENT_DECODE_TOK_S)
    feed = sf.FakePowerFeed({STUDENT_NODE: fake.student_watts or sf.STUDENT_BUSY_W,
                             TEACHER_NODE: fake.teacher_watts or sf.TEACHER_BUSY_W},
                            jitter=fake.jitter, seed=seed, clock=clock)
    targets = {
        "teacher": Target("teacher", TEACHER_NODE, Teacher(teacher.base_url, transport=teacher.transport,
                                                           clock=clock.time, sleep=clock.sleep), teacher.model),
        "student": Target("student", STUDENT_NODE, Teacher(student.base_url, transport=student.transport,
                                                           model=sf.STUDENT_MODEL, clock=clock.time,
                                                           sleep=clock.sleep), sf.STUDENT_MODEL),
    }
    meter = Meter(feed, clock=clock, settle_s=1.0, step_s=feed.period_s)
    run_id = run_id or f"t11-fake-{seed}-{secrets.token_hex(2)}"
    state = state_dir if state_dir is not None else Path(out_dir) / run_id / "state"
    data = load(root, require_images=False)
    conditions = {"seed": seed, "probes_per_node": probes, "student_can": student_can, "fake": dataclasses.asdict(fake)}
    try:
        return run(draw_requests(root, n=n, seed=seed, student_can=student_can),
                   draw_requests(root, n=probes, seed=seed, skip=n, student_can=student_can) if probes else [],
                   targets, meter, data=data, toolbox=golden_toolbox(), out_dir=out_dir, state_dir=state,
                   source="fake", fake=True, run_id=run_id, conditions=conditions)
    finally:
        for t in targets.values():
            t.client.close()


# --- the real run ---------------------------------------------------------------------------------------------------

def _real(args: argparse.Namespace) -> Result:
    from lab import hub

    if not args.power_log:
        raise LoopError("the real run reads joules from the power samples: give --power-log (the Jetson's event "
                        "log, and the pod sampler's file when its samples are not copied into it)")
    if args.teacher_url:
        teacher = Teacher(args.teacher_url, node=TEACHER_NODE)
    elif args.account:
        accounts = hub.load_accounts([args.account], require_role=True)
        teacher = Teacher.from_hub(accounts[args.account])
    else:
        raise LoopError("name the teacher: --account C (through the hub proxy) or --teacher-url")
    student = Teacher(args.student_url, model=args.student_model, node=STUDENT_NODE)
    targets = {"teacher": Target("teacher", TEACHER_NODE, teacher, "teacher"),
               "student": Target("student", STUDENT_NODE, student, args.student_model)}
    clock = RealClock()
    meter = Meter(JsonlFeed(args.power_log), clock=clock, settle_s=args.settle_s, wait_s=args.wait_s,
                  window_clock=args.window_clock)
    conditions = {"seed": args.seed, "probes_per_node": args.probes, "student_can": args.student_can == "yes",
                  "window_clock": args.window_clock, "settle_s": args.settle_s}
    try:
        return run(draw_requests(args.dataset, n=args.n, seed=args.seed, student_can=args.student_can == "yes"),
                   draw_requests(args.dataset, n=args.probes, seed=args.seed, skip=args.n,
                                 student_can=args.student_can == "yes") if args.probes else [],
                   targets, meter, data=load(args.dataset, require_images=False), toolbox=golden_toolbox(),
                   out_dir=args.out, state_dir=args.state_dir, source="real", fake=False, conditions=conditions)
    finally:
        teacher.close()
        student.close()


@stop.guarded
def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """The routing hook's closed-loop test (T11): policy one against policy two on the same lab requests."""
    p = argparse.ArgumentParser(prog=f"python -m lab {VERB}", description=main.__doc__, allow_abbrev=False)
    p.add_argument("--fake", action="store_true", help="the teacher and student fakes on a fake clock")
    p.add_argument("--dataset", default=str(FIXTURE), help="the L11 fixture folder (default: %(default)s)")
    p.add_argument("--n", type=int, default=MIN_REQUESTS, help=f"requests (at least {MIN_REQUESTS})")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--probes", type=int, default=DEFAULT_PROBES, help="probe requests per node before the arms")
    p.add_argument("--student-can", choices=("yes", "no"), default="yes",
                   help="RouteRequest.student_can for labeling (default: yes)")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="where <run id>/ is written")
    p.add_argument("--state-dir", default=None, help="the ledger's state dir (a --fake run: <out>/<run id>/state)")
    p.add_argument("--account", help="the teacher's hub account letter (C), reached through the hub proxy")
    p.add_argument("--teacher-url", help="the teacher's server address, reached directly")
    p.add_argument("--student-url", default="http://127.0.0.1:8080", help="the student's llama-server")
    p.add_argument("--student-model", default="atlas-base")
    p.add_argument("--power-log", nargs="+", default=[], help="JSONL files of telemetry events to read power from")
    p.add_argument("--window-clock", default=None, help="the node whose clock the run uses (jetson on the Jetson)")
    p.add_argument("--settle-s", type=float, default=SETTLE_S)
    p.add_argument("--wait-s", type=float, default=WAIT_S)
    args = p.parse_args(list(argv) if argv is not None else None)
    try:
        if args.n < MIN_REQUESTS:
            raise LoopError(f"the closed-loop test needs at least {MIN_REQUESTS} requests, not {args.n}")
        if args.fake:
            result = run_fake(n=args.n, seed=args.seed, probes=args.probes, out_dir=args.out,
                              state_dir=args.state_dir, root=args.dataset, student_can=args.student_can == "yes")
        else:
            result = _real(args)
    except LoopError as err:
        return {"ok": False, "outcome": "refused", "error": str(err)}
    s = result.summary
    return {"ok": True, "outcome": "done", "run_id": result.run_id, "requests": len(result.records) // 2,
            "changed": s["changed"], "keep_policy_two": s["keep_policy_two"], "rubric_tt6_met": s["rubric_tt6"]["met"],
            "verdicts": {m: v["label"] for m, v in s["verdicts"].items()}, "experiments": result.experiments,
            "folder": str(result.folder), "t0_text": str(result.folder / T0_FILE),
            "refs": {"run_id": result.run_id}}


if __name__ == "__main__":
    answer = main()
    print(json.dumps(answer))
    sys.exit(0 if answer.get("ok", True) else 1)

CLI_VERB = "routing-closed-loop"
