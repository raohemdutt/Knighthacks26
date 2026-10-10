"""Experiments: one lever, two or more arms, a metric to move, an estimate written before the run and a verdict
read after it.

An experiment lives in the lab ledger (lab/machine/ledger.py) as rows, never in a file of its own:

    experiment add      its definition
    experiment amend    a change to some of its fields, or to its status (queued, running, dropped)
    experiment result   the readings of its arms and the verdict lab/machine/verdict.py gives them

Each row carries the experiment under the key "experiment" and the Task Board ID it belongs to in the row's task
field (T0, the standing ticket whose experiment queue lists them, unless the experiment names another). What an
experiment is now is those rows folded in ledger order (fold, load): the first add defines it, each amend changes
it, the latest result is its result. A second add of the same id is ignored and reported as a problem, so two
writers that raced cannot replace a definition. Writers hold a lock of their own (experiments.lock in the state
dir) while they read the ledger and append, so an id is never handed out twice.

The definition (DEFINITION_FIELDS):

    id              E1, E2, ... (the next free E number when none is given), or a research id such as KE2
    title           a short name for the queue line, or None
    track           telemetry, learning, plumbing or kernels: the refinement track whose metric it moves
    task            the Task Board ID the experiment belongs to; T0 by default
    hypothesis      what is expected to happen, and why
    lever           the one thing the arms change; two levers make two experiments
    arms            {name: description}, at least two; the first is the control (A) every other arm is read
                    against, as in the A/B method: each arm against A
    kind            measure (runs of a number, read with Welch's t test) or rate (x successes in n trials, read
                    with the pooled two-proportion z test)
    metric          the dotted metric name the verdict reads, such as teacher.joules_per_token
    better          higher or lower: which way the metric improves
    secondary       other metrics to record beside it (not judged), or []
    expected        the expected movement in words, or None
    expected_delta  the expected change of the metric from the control to the best arm, as a signed fraction of the
                    control's value (-0.85 is 85 percent lower), or None when not estimated
    confidence      the probability, 0 to 1, that the experiment delivers at least half of that change, or None
    min_n           runs (or trials) each arm needs before the verdict reads it: below it the verdict is too_early.
                    Defaults to the verdict floor (5 runs, 50 trials); `--spread SD --baseline MEAN` sizes it with
                    verdict.runs_per_arm_relative (or trials_per_arm for a rate) and never goes below the floor
    citation        where the hypothesis and the estimate come from
    node            where it runs (pod-a, pod-b, jetson), or None
    note            anything else a runner must know, or None

Folding adds status (queued, running, done, dropped), result (None until a result row), gain and score.

Ranking. gain is the expected fractional cut in what the metric costs: for a metric where lower is better, the
fraction it falls (-expected_delta); for one where higher is better, the fraction its reciprocal falls,
expected_delta / (1 + expected_delta), so doubling tokens per second counts 0.5, the same as halving joules per
token at constant power. Every gain is on one scale whichever way its metric points, and a negative gain expects a
loss. score is gain times confidence. rank orders the open experiments (queued and running) by score, higher
first, then by confidence, then by the order they were added; an experiment without an expected_delta or a
confidence has no score and comes after every scored one, with what it needs.

A result reads each arm given against the control with verdict.verdict, the experiment's min_n as the floor and its
better as the direction. With more than one arm against the control, alpha is split between the comparisons
(Bonferroni), so reading three arms is not three chances for luck to look like a win. The overall label is
too_early when any comparison is, b_ahead when an arm beats the control (winner names the best such arm), a_ahead
when the control beats every arm, else no_difference. The status becomes done, or stays running while the label is
too_early (pass --final to close it anyway: a slower or unreadable result is still a result). Pass every run of an
arm each time, not only the new ones: the latest result row is the experiment's result.

The verb, `python -m lab experiment <sub-verb>`:

    add [ID] --track T --hypothesis TEXT --lever TEXT --arm A=TEXT --arm B=TEXT --metric NAME --better higher|lower
        --citation TEXT [--kind measure|rate] [--expected TEXT] [--expected-delta X] [--confidence C]
        [--min-n N | --spread SD --baseline MEAN] [--title TEXT] [--task ID] [--node NAME] [--secondary NAME ...]
        [--note TEXT]
    add ID --amend [any field above] [--status queued|running|dropped] [--reason WHY]
    list [--id ID ...] [--status S ...] [--track T] [--print]
    result ID --arm A=V1,V2,... --arm B=V1,V2,... [--control NAME] [--final] [--note TEXT] [--slot S] [--run-id R]
        an arm is run values (7.7,7.9,7.6), a rate (12/60), or run:ID[,ID...] for the values of the metric in the
        ledger rows whose refs.run_id is one of those ids, one value per row in ledger order
    rank [--track T] [--all] [--print]
    seed                                  add the experiments of T0's queue (SEEDS) that the ledger lacks

Each call answers with one JSON line, except --print on list and rank: that prints the queue as checklist lines, one
per experiment, in the form the standing ticket T0 keeps them (`- [ ] E3 (kernels) ...`, checked once done or
dropped), so a sitting can mirror the ledger onto T0 with its Notion tools. The text is plain ASCII apart from what
the experiments' own fields hold.
"""
from __future__ import annotations

import argparse
import logging
import math
import numbers
import os
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from lab.machine import ledger, record
from lab.machine import refine_formats
from lab.machine import verdict as verdict_math
from lab.machine.lock import FileLock

log = logging.getLogger("lab.experiments")

__all__ = [
    "VERB_ADD", "VERB_AMEND", "VERB_RESULT", "VERBS", "TRACKS", "KINDS", "BETTER", "STATUSES", "SETTABLE_STATUSES",
    "QUEUED", "RUNNING", "DONE", "DROPPED", "MEASURE", "RATE", "HIGHER", "LOWER", "DEFAULT_TASK",
    "DEFINITION_FIELDS", "LOCK_FILE", "SEEDS", "ExperimentError",
    "validate_definition", "default_min_n", "planned_min_n", "expected_gain", "score_of", "fold", "load", "next_id",
    "add", "amend", "record_result", "runs_from_ledger", "rank", "checklist", "checklist_line", "seed", "main",
]

VERB_ADD = "experiment add"
VERB_AMEND = "experiment amend"
VERB_RESULT = "experiment result"
VERBS = (VERB_ADD, VERB_AMEND, VERB_RESULT)

TRACKS = refine_formats.TRACKS
MEASURE, RATE = "measure", "rate"
KINDS = (MEASURE, RATE)
HIGHER, LOWER = "higher", "lower"
BETTER = (HIGHER, LOWER)
QUEUED, RUNNING, DONE, DROPPED = "queued", "running", "done", "dropped"
STATUSES = (QUEUED, RUNNING, DONE, DROPPED)
#: The statuses an amend may set; done comes only from a result.
SETTABLE_STATUSES = (QUEUED, RUNNING, DROPPED)
OPEN_STATUSES = (QUEUED, RUNNING)

DEFAULT_TASK = "T0"
LOCK_FILE = "experiments.lock"
LOCK_TIMEOUT = 30.0

DEFINITION_FIELDS = ("id", "title", "track", "task", "hypothesis", "lever", "arms", "kind", "metric", "better",
                     "secondary", "expected", "expected_delta", "confidence", "min_n", "citation", "node", "note")
_REQUIRED_TEXT = ("hypothesis", "lever", "citation")
_OPTIONAL_TEXT = ("title", "expected", "node", "note")

_ID = re.compile(r"[A-Za-z][A-Za-z0-9._-]{0,31}")
_ARM = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,15}")
_METRIC = refine_formats.METRIC_NAME
_TASK = re.compile(r"[A-Z]{1,3}[0-9]{1,3}")
_E_NUMBER = re.compile(r"E([0-9]+)")


class ExperimentError(ValueError):
    """An experiment, a change or a result the ledger is not given. problems lists every reason."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


# --- the definition -------------------------------------------------------------------------------------------

def _one_line(value: Any) -> Any:
    """Text with every run of whitespace, newlines included, made one space: a queue line is one line."""
    return " ".join(value.split()) if isinstance(value, str) else value


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def default_min_n(kind: str) -> int:
    """The verdict's floor for a kind: 5 runs for a measurement, 50 trials for a rate."""
    return verdict_math.MIN_TRIALS_PER_ARM if kind == RATE else verdict_math.MIN_RUNS_PER_ARM


def planned_min_n(kind: str, expected_delta: float, baseline: float, spread: float | None = None) -> int:
    """The runs (or trials) an arm needs to see expected_delta from a control at baseline, never below the floor.

    A measurement is sized with verdict.runs_per_arm_relative(spread, baseline, expected_delta): spread 5 on a
    baseline of 100, to see -0.05, takes 17 runs an arm. A rate is sized with verdict.trials_per_arm(baseline,
    baseline * (1 + expected_delta)): 5% to 10% takes 435 trials an arm. ValueError for a plan that cannot be sized."""
    if kind == RATE:
        target = baseline * (1.0 + expected_delta)
        sized = verdict_math.trials_per_arm(baseline, target)
    else:
        if spread is None:
            raise ValueError("a measurement is sized from the spread of one run: give --spread with --baseline")
        sized = verdict_math.runs_per_arm_relative(spread, baseline, expected_delta)
    return max(default_min_n(kind), sized)


def _check_text(fields: Mapping[str, Any], out: dict, problems: list[str]) -> None:
    for key in _REQUIRED_TEXT:
        value = _one_line(fields.get(key))
        if isinstance(value, str) and value:
            out[key] = value
        else:
            problems.append(f"{key} is missing" if fields.get(key) is None else f"{key} must be non-empty text")
    for key in _OPTIONAL_TEXT:
        value = _one_line(fields.get(key))
        if value is None or value == "":
            out[key] = None
        elif isinstance(value, str):
            out[key] = value
        else:
            problems.append(f"{key} must be text or None, not {value!r}")


def _check_choice(fields: Mapping[str, Any], key: str, choices: Sequence[str], default: str | None,
                  out: dict, problems: list[str]) -> None:
    value = fields.get(key)
    if value is None and default is not None:
        value = default
    if value in choices:
        out[key] = value
    elif value is None:
        problems.append(f"{key} is missing: one of {', '.join(choices)}")
    else:
        problems.append(f"{key} must be one of {', '.join(choices)}, not {value!r}")


def _check_arms(value: Any, out: dict, problems: list[str]) -> None:
    if not isinstance(value, Mapping) or len(value) < 2:
        problems.append("arms must name at least two arms, the control first (--arm A=... --arm B=...)")
        return
    arms: dict[str, str] = {}
    for name, description in value.items():
        text = _one_line(description)
        if not isinstance(name, str) or not _ARM.fullmatch(name):
            problems.append(f"arm name {name!r} must be a letter or a short word (letters, digits, . _ -)")
        elif not isinstance(text, str) or not text:
            problems.append(f"arm {name} needs a description of what it runs")
        else:
            arms[name] = text
    out["arms"] = arms


def _check_metrics(fields: Mapping[str, Any], out: dict, problems: list[str]) -> None:
    metric = fields.get("metric")
    if metric is None:
        problems.append("metric is missing")
    elif not isinstance(metric, str) or not _METRIC.fullmatch(metric):
        problems.append(f"metric must be a dotted lowercase name such as teacher.joules_per_token, not {metric!r}")
    else:
        out["metric"] = metric
    secondary = fields.get("secondary") or []
    if isinstance(secondary, str) or not isinstance(secondary, Iterable):
        problems.append(f"secondary must be a list of metric names, not {secondary!r}")
        return
    clean: list[str] = []
    for name in secondary:
        if not isinstance(name, str) or not _METRIC.fullmatch(name):
            problems.append(f"secondary metric {name!r} must be a dotted lowercase name")
        elif name != metric and name not in clean:
            clean.append(name)
    out["secondary"] = clean


def _check_estimate(fields: Mapping[str, Any], out: dict, problems: list[str]) -> None:
    delta = fields.get("expected_delta")
    if delta is None:
        out["expected_delta"] = None
    elif _number(delta) is None or _number(delta) <= -1.0:
        problems.append(f"expected_delta must be a finite fraction above -1 (-0.85 is 85% lower), not {delta!r}")
    else:
        out["expected_delta"] = _number(delta)
    confidence = fields.get("confidence")
    if confidence is None:
        out["confidence"] = None
    elif _number(confidence) is None or not 0.0 <= _number(confidence) <= 1.0:
        problems.append(f"confidence must be a probability from 0 to 1, not {confidence!r}")
    else:
        out["confidence"] = _number(confidence)


def _check_min_n(fields: Mapping[str, Any], out: dict, problems: list[str]) -> None:
    kind = out.get("kind")
    value = fields.get("min_n")
    if value is None:
        if kind is not None:
            out["min_n"] = default_min_n(kind)
        return
    lowest = 2 if kind == MEASURE else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < lowest:
        problems.append(f"min_n must be a whole number of at least {lowest} for a {kind or 'measure'}, not {value!r}")
    else:
        out["min_n"] = value


def validate_definition(fields: Mapping[str, Any]) -> dict[str, Any]:
    """The experiment as the ledger stores it: every field of DEFINITION_FIELDS, in that order, text on one line,
    numbers as floats, min_n filled in from the kind when missing, task T0 and kind measure by default.
    ExperimentError lists every problem; a field that is not a definition field is a problem too."""
    if not isinstance(fields, Mapping):
        raise ExperimentError([f"an experiment is a dict of fields, not {type(fields).__name__}"])
    problems: list[str] = []
    out: dict[str, Any] = {}
    unknown = [key for key in fields if key not in DEFINITION_FIELDS]
    if unknown:
        problems.append(f"not experiment fields: {', '.join(map(str, unknown))}")
    eid = fields.get("id")
    if not isinstance(eid, str) or not _ID.fullmatch(eid):
        problems.append(f"id must start with a letter and hold up to 32 letters, digits, '.', '_' or '-', not {eid!r}")
    else:
        out["id"] = eid
    _check_choice(fields, "track", TRACKS, None, out, problems)
    task = fields.get("task") or DEFAULT_TASK
    if isinstance(task, str) and _TASK.fullmatch(task):
        out["task"] = task
    else:
        problems.append(f"task must be a Task Board ID such as T0 or R12, not {task!r}")
    _check_text(fields, out, problems)
    _check_arms(fields.get("arms"), out, problems)
    _check_choice(fields, "kind", KINDS, MEASURE, out, problems)
    _check_metrics(fields, out, problems)
    _check_choice(fields, "better", BETTER, None, out, problems)
    _check_estimate(fields, out, problems)
    _check_min_n(fields, out, problems)
    if problems:
        raise ExperimentError(problems)
    return {key: out[key] for key in DEFINITION_FIELDS}


# --- gain and score -------------------------------------------------------------------------------------------

def expected_gain(expected_delta: float | None, better: str) -> float | None:
    """The expected fractional cut in what the metric costs (see the module docstring); None when not estimated."""
    if expected_delta is None:
        return None
    if better == LOWER:
        return -expected_delta
    return expected_delta / (1.0 + expected_delta)


def score_of(experiment: Mapping[str, Any]) -> float | None:
    """gain times confidence; None when either is missing."""
    gain = expected_gain(experiment.get("expected_delta"), experiment.get("better"))
    confidence = experiment.get("confidence")
    if gain is None or confidence is None:
        return None
    return gain * confidence


def _needs(experiment: Mapping[str, Any]) -> list[str]:
    return [key for key in ("expected_delta", "confidence") if experiment.get(key) is None]


# --- the fold -------------------------------------------------------------------------------------------------

def _definition(experiment: Mapping[str, Any]) -> dict[str, Any]:
    return {key: experiment[key] for key in DEFINITION_FIELDS}


def _derive(experiment: dict[str, Any]) -> None:
    """Fill in gain and score, rounded to nine places so float noise neither shows nor breaks a tie."""
    gain, score = expected_gain(experiment["expected_delta"], experiment["better"]), score_of(experiment)
    experiment["gain"] = None if gain is None else round(gain, 9)
    experiment["score"] = None if score is None else round(score, 9)


def fold(rows: Iterable[Mapping[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """The experiments the rows describe, by id in the order they were added, and the problems met on the way: a
    row with no experiment id, a second add of one id, a change or result for an id never added, a change that
    would break the definition. A problem row is skipped; the rest still count."""
    experiments: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    for row in rows:
        verb, seq, payload = row.get("verb"), row.get("seq"), row.get("experiment")
        if verb not in VERBS:
            continue
        eid = payload.get("id") if isinstance(payload, Mapping) else None
        if not isinstance(eid, str) or not eid:
            problems.append(f"seq {seq}: {verb} row carries no experiment id; skipped")
            continue
        stamp = {"seq": seq, "ts": row.get("ts")}
        if verb == VERB_ADD:
            if eid in experiments:
                problems.append(f"seq {seq}: {eid} was already added at seq {experiments[eid]['added']['seq']}; "
                                "this second add is ignored (use add --amend to change it)")
                continue
            try:
                definition = validate_definition(payload)
            except ExperimentError as exc:
                problems.append(f"seq {seq}: {eid} add is not a valid experiment ({exc}); skipped")
                continue
            experiments[eid] = {**definition, "status": QUEUED, "status_note": None, "result": None,
                                "added": {**stamp, "actor": row.get("actor")}, "updated": stamp, "seqs": [seq]}
            continue
        experiment = experiments.get(eid)
        if experiment is None:
            problems.append(f"seq {seq}: {verb} for {eid}, which was never added; skipped")
            continue
        if verb == VERB_AMEND:
            changes = payload.get("changes")
            changes = dict(changes) if isinstance(changes, Mapping) else {}
            status = changes.pop("status", None)
            if status is not None and status not in SETTABLE_STATUSES:
                problems.append(f"seq {seq}: {eid} amend sets status {status!r}; skipped")
                continue
            try:
                definition = validate_definition({**_definition(experiment), **changes, "id": eid})
            except ExperimentError as exc:
                problems.append(f"seq {seq}: {eid} amend would break the experiment ({exc}); skipped")
                continue
            experiment.update(definition)
            if status is not None:
                experiment["status"] = status
                experiment["status_note"] = _one_line(payload.get("reason")) or None
        else:
            result = {key: value for key, value in payload.items() if key != "id"}
            result.update(stamp)
            experiment["result"] = result
            experiment["status"] = payload.get("status") if payload.get("status") in (RUNNING, DONE) else DONE
            experiment["status_note"] = None
        experiment["updated"] = stamp
        experiment["seqs"].append(seq)
    for experiment in experiments.values():
        _derive(experiment)
    return experiments, problems


def _rows(state_dir: str | os.PathLike[str] | None) -> list[dict[str, Any]]:
    return ledger.read(state_dir=state_dir, verb=lambda verb: verb in VERBS)


def load(state_dir: str | os.PathLike[str] | None = None) -> dict[str, dict[str, Any]]:
    """The experiments in the ledger as they stand now, by id in the order they were added. Rows the fold had to
    skip are logged as warnings."""
    experiments, problems = fold(_rows(state_dir))
    for problem in problems:
        log.warning("experiments: %s", problem)
    return experiments


def next_id(existing: Iterable[str]) -> str:
    """One past the highest E number in existing and in SEEDS, so a new experiment never takes the id of a line
    T0 already has, seeded or not."""
    taken = [*existing, *(seed_def["id"] for seed_def in SEEDS)]
    numbers_taken = [int(m.group(1)) for eid in taken if (m := _E_NUMBER.fullmatch(eid))]
    return f"E{max(numbers_taken, default=0) + 1}"


# --- writing --------------------------------------------------------------------------------------------------

def _lock(state_dir: str | os.PathLike[str] | None) -> FileLock:
    path = ledger.resolve_state_dir(state_dir) / LOCK_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(path, timeout=LOCK_TIMEOUT)


def _append(verb: str, task: str, payload: dict[str, Any], outcome: str, *, state_dir: Any,
            refs: Mapping[str, Any] | None = None, source: str | None = None) -> int:
    """Write one experiment row through record.record, which names the writer (actor and tier) the way every
    ledger row is named."""
    extra = {"source": source} if source else {}
    return record.record(verb, outcome, refs=dict(refs or {}), task=task, state_dir=state_dir, experiment=payload,
                         **extra)


def add(fields: Mapping[str, Any], *, state_dir: str | os.PathLike[str] | None = None,
        source: str | None = None) -> dict[str, Any]:
    """Add a new experiment and return it as folded, with its "seq". The id is the next free E number when fields
    give none. ExperimentError, writing nothing, when the id is taken or the definition has problems."""
    with _lock(state_dir):
        experiments = load(state_dir)
        eid = fields.get("id") or next_id(experiments)
        if eid in experiments:
            raise ExperimentError([f"{eid} already exists (status {experiments[eid]['status']}); change it with "
                                   f"add {eid} --amend, or pick another id"])
        definition = validate_definition({**fields, "id": eid})
        seq = _append(VERB_ADD, definition["task"], definition, "ok", state_dir=state_dir, source=source)
        return {**load(state_dir)[eid], "seq": seq}


def amend(eid: str, changes: Mapping[str, Any], *, reason: str | None = None,
          state_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Change some fields of an experiment, its status among them (queued, running or dropped), and return it as
    folded, with its "seq". reason says why; it is shown beside a dropped experiment. A field given as None is left
    as it is. ExperimentError, writing nothing, for an unknown id, no change, or a change that would break the
    definition."""
    with _lock(state_dir):
        experiments = load(state_dir)
        experiment = experiments.get(eid)
        if experiment is None:
            raise ExperimentError([f"no experiment {eid}; known: {', '.join(experiments) or 'none'}"])
        clean = {key: value for key, value in changes.items() if value is not None}
        status = clean.get("status")
        problems = []
        if "id" in clean and clean["id"] != eid:
            problems.append("an amend cannot change the id; add a new experiment and drop this one")
        if status is not None and status not in SETTABLE_STATUSES:
            problems.append(f"status may be set to {', '.join(SETTABLE_STATUSES)}, not {status!r} "
                            "(done comes from a result)")
        if not clean:
            problems.append("nothing to change")
        if problems:
            raise ExperimentError(problems)
        definition_changes = {key: value for key, value in clean.items() if key not in ("status", "id")}
        validate_definition({**_definition(experiment), **definition_changes})
        payload = {"id": eid, "changes": clean, "reason": _one_line(reason) or None}
        seq = _append(VERB_AMEND, experiment["task"], payload, "ok", state_dir=state_dir)
        return {**load(state_dir)[eid], "seq": seq}


def runs_from_ledger(metric: str, run_ids: Sequence[str], *,
                     state_dir: str | os.PathLike[str] | None = None) -> list[float]:
    """The values of metric in the ledger rows whose refs.run_id is one of run_ids, one per row, in ledger order."""
    wanted = set(run_ids)
    rows = ledger.read(state_dir=state_dir, has_metric=metric,
                       refs=lambda refs: isinstance(refs, dict) and refs.get("run_id") in wanted)
    return [float(row["metrics"][metric]) for row in rows]


def _as_arm(value: Any, kind: str, name: str) -> verdict_math.Counts | verdict_math.Samples:
    if kind == RATE:
        if not isinstance(value, verdict_math.Counts):
            raise ExperimentError([f"arm {name}: a rate experiment reads counts such as 12/60"])
        return value
    if isinstance(value, verdict_math.Counts):
        raise ExperimentError([f"arm {name}: a measure experiment reads run values, not counts"])
    if isinstance(value, verdict_math.Samples):
        return value
    try:
        return verdict_math.Samples.from_values(value)
    except (TypeError, ValueError) as exc:
        raise ExperimentError([f"arm {name}: {exc}"]) from None


def _summary(arm: verdict_math.Counts | verdict_math.Samples, values: Any) -> dict[str, Any]:
    if isinstance(arm, verdict_math.Counts):
        return {"x": arm.x, "n": arm.n, "rate": arm.x / arm.n if arm.n else None}
    summary: dict[str, Any] = {"n": arm.n, "mean": arm.mean, "sd": arm.sd}
    if not isinstance(values, verdict_math.Samples):
        summary["values"] = [float(v) for v in values]
    return summary


def _overall(comparisons: Mapping[str, Mapping[str, Any]], control: str,
             higher_is_better: bool) -> tuple[str, str | None]:
    labels = {name: comparison["label"] for name, comparison in comparisons.items()}
    if verdict_math.TOO_EARLY in labels.values():
        return verdict_math.TOO_EARLY, None
    ahead = [name for name, label in labels.items() if label == verdict_math.B_AHEAD]
    if ahead:
        sign = 1.0 if higher_is_better else -1.0
        return verdict_math.B_AHEAD, max(ahead, key=lambda name: sign * comparisons[name]["diff"])
    if all(label == verdict_math.A_AHEAD for label in labels.values()):
        return verdict_math.A_AHEAD, control
    return verdict_math.NO_DIFFERENCE, None


def record_result(eid: str, arms: Mapping[str, Any], *, control: str | None = None, final: bool = False,
                  note: str | None = None, refs: Mapping[str, Any] | None = None,
                  state_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Read the arms with verdict.verdict and append the result; return the experiment as folded, with its "seq".

    arms maps arm names to readings: verdict.Counts for a rate, a list of run values or verdict.Samples for a
    measurement. The control is the experiment's first arm unless control names another, and must be among the
    arms given; every other arm given is read against it. ExperimentError, writing nothing, for an unknown
    experiment or arm, a reading of the wrong kind, or fewer than two arms."""
    with _lock(state_dir):
        experiments = load(state_dir)
        experiment = experiments.get(eid)
        if experiment is None:
            raise ExperimentError([f"no experiment {eid}; known: {', '.join(experiments) or 'none'}"])
        declared = list(experiment["arms"])
        control = control or declared[0]
        problems = [f"arm {name!r} is not an arm of {eid} ({', '.join(declared)})"
                    for name in arms if name not in declared]
        if control not in declared:
            problems.append(f"control {control!r} is not an arm of {eid} ({', '.join(declared)})")
        elif control not in arms:
            problems.append(f"the control {control} has no reading; give --arm {control}=...")
        if len([name for name in arms if name != control]) < 1:
            problems.append("give the control and at least one other arm")
        if problems:
            raise ExperimentError(problems)
        readings = {name: _as_arm(arms[name], experiment["kind"], name) for name in declared if name in arms}
        others = [name for name in readings if name != control]
        higher = experiment["better"] == HIGHER
        alpha = verdict_math.ALPHA / len(others)
        comparisons = {name: verdict_math.verdict(readings[control], readings[name], higher_is_better=higher,
                                                  alpha=alpha, min_per_arm=experiment["min_n"]).as_dict()
                       for name in others}
        label, winner = _overall(comparisons, control, higher)
        status = DONE if final or label != verdict_math.TOO_EARLY else RUNNING
        payload = {"id": eid, "control": control,
                   "arms": {name: _summary(readings[name], arms[name]) for name in readings},
                   "comparisons": comparisons, "label": label, "winner": winner, "alpha": alpha, "final": final,
                   "status": status, "note": _one_line(note) or None}
        seq = _append(VERB_RESULT, experiment["task"], payload, label, state_dir=state_dir, refs=refs)
        return {**load(state_dir)[eid], "seq": seq}


def seed(*, state_dir: str | os.PathLike[str] | None = None) -> dict[str, list[str]]:
    """Add every experiment of SEEDS the ledger does not hold yet; {"added": [...], "present": [...]}. Running it
    again adds nothing."""
    added: list[str] = []
    present: list[str] = []
    for fields in SEEDS:
        try:
            add(fields, state_dir=state_dir, source=SEED_SOURCE)
        except ExperimentError:
            if fields["id"] not in load(state_dir):
                raise
            present.append(fields["id"])
        else:
            added.append(fields["id"])
    return {"added": added, "present": present}


# --- ranking and the T0 checklist -----------------------------------------------------------------------------

def rank(experiments: Iterable[Mapping[str, Any]], *, include_finished: bool = False) -> list[dict[str, Any]]:
    """The experiments in queue order, each with "rank" (1, 2, ... for the scored ones, None for the rest) and
    "needs" (what an unscored one lacks). Open ones (queued, running) first: scored by score, then confidence, then
    the order they were added; then the unscored ones in the order they were added. With include_finished the
    done and dropped ones follow, in the order they were added."""
    pool = list(experiments)
    open_ones = [e for e in pool if e["status"] in OPEN_STATUSES]
    scored = sorted((e for e in open_ones if e["score"] is not None),
                    key=lambda e: (-e["score"], -e["confidence"], e["added"]["seq"]))
    unscored = sorted((e for e in open_ones if e["score"] is None), key=lambda e: e["added"]["seq"])
    finished = sorted((e for e in pool if e["status"] not in OPEN_STATUSES), key=lambda e: e["added"]["seq"]) \
        if include_finished else []
    ordered = [{**e, "rank": i, "needs": []} for i, e in enumerate(scored, 1)]
    ordered += [{**e, "rank": None, "needs": _needs(e)} for e in unscored + finished]
    return ordered


def _sentence(text: str) -> str:
    text = text.strip()
    return text if text.endswith((".", "!", "?")) else text + "."


def _pct(fraction: float) -> str:
    return f"{100.0 * fraction:+.3g}%"


def _p_text(p: float | None) -> str:
    if p is None:
        return ""
    return "p < 0.0001" if p < 0.0001 else f"p = {round(p, 4):g}"


def _stat_text(comparison: Mapping[str, Any]) -> str:
    if comparison.get("statistic") is None:
        return ""
    name = "z" if comparison.get("test") == verdict_math.TEST_TWO_PROPORTION else "t"
    parts = [f"{name} = {comparison['statistic']:.3f}"]
    if comparison.get("df") is not None:
        parts.append(f"df = {comparison['df']:.1f}")
    parts.append(_p_text(comparison.get("p")))
    return ", ".join(parts)


def _reading(summary: Mapping[str, Any] | None) -> str:
    if not summary:
        return "no reading"
    if "rate" in summary:
        rate = summary["rate"]
        share = "no trials" if rate is None else f"{100.0 * rate:.1f}%"
        return f"{share} ({summary['x']}/{summary['n']})"
    runs = "run" if summary["n"] == 1 else "runs"
    mean = "no mean" if summary["mean"] is None else f"{summary['mean']:.4g}"
    return f"{mean} ({summary['n']} {runs})"


def _comparison_text(name: str, control: str, comparison: Mapping[str, Any], arms: Mapping[str, Any]) -> str:
    """One arm against the control in words, with the arms' own names (verdict's notes say A and B)."""
    arm_reading, control_reading = _reading(arms.get(name)), _reading(arms.get(control))
    stats = _stat_text(comparison)
    tail = f" ({stats})" if stats else ""
    label = comparison["label"]
    if label == verdict_math.B_AHEAD:
        return f"{name} ahead of {control}: {arm_reading} against {control_reading}{tail}"
    if label == verdict_math.A_AHEAD:
        return f"{control} ahead of {name}: {control_reading} against {arm_reading}{tail}"
    if label == verdict_math.NO_DIFFERENCE:
        return f"no difference between {name} and {control}: {arm_reading} against {control_reading}{tail}"
    more_a, more_b = comparison.get("more_a"), comparison.get("more_b")
    if more_a is None or more_b is None:
        need = "the run-to-run noise cannot be estimated from repeated values"
    else:
        need = f"{more_a} more for {control}, {more_b} more for {name}"
    return f"too early for {name} against {control}: {arm_reading} against {control_reading}; {need}"


def _result_text(result: Mapping[str, Any]) -> str:
    control = result.get("control")
    comparisons = result.get("comparisons") or {}
    arms = result.get("arms") or {}
    texts = [_comparison_text(name, control, comparison, arms) for name, comparison in comparisons.items()]
    head = result.get("label", "")
    if result.get("winner"):
        head += f", winner {result['winner']}"
    note = f" Note: {_sentence(result['note'])}" if result.get("note") else ""
    return f"{head}: {'; '.join(texts)}.{note}"


def checklist_line(experiment: Mapping[str, Any]) -> str:
    """The experiment as one checklist line of T0's queue: checked once done or dropped."""
    e = experiment
    box = "[x]" if e["status"] in (DONE, DROPPED) else "[ ]"
    title = f" {_sentence(e['title'])}" if e.get("title") else ""
    control = next(iter(e["arms"]))
    arms = "; ".join(f"{name}{' (control)' if name == control else ''}: {text}" for name, text in e["arms"].items())
    metric = f"{e['metric']}, {e['better']} is better"
    if e.get("secondary"):
        metric += f" (also {', '.join(e['secondary'])})"
    estimate = [e["expected"].rstrip(".")] if e.get("expected") else []
    if e.get("expected_delta") is not None:
        estimate.append(f"delta {_pct(e['expected_delta'])}")
    if e.get("confidence") is not None:
        estimate.append(f"confidence {e['confidence']:g}")
    if e.get("score") is not None:
        estimate.append(f"score {e['score']:.2g}")
    else:
        estimate.append("unscored, needs " + " and ".join(_needs(e)))
    unit = "trials" if e["kind"] == RATE else "runs"
    parts = [f"- {box} {e['id']} ({e['track']}){title}",
             f"Hypothesis: {_sentence(e['hypothesis'])}",
             f"Lever: {_sentence(e['lever'])}",
             f"Arms: {arms}.",
             f"Metric: {metric}.",
             f"Expected: {'; '.join(estimate)}.",
             f"Minimum {e['min_n']} {unit} per arm.",
             f"Source: {_sentence(e['citation'])}"]
    if e.get("node"):
        parts.append(f"Node: {e['node']}.")
    if e.get("note"):
        parts.append(f"Note: {_sentence(e['note'])}")
    result = e.get("result")
    if e["status"] == DONE and result:
        day = (result.get("ts") or "")[:10]
        parts.append(f"Result{' ' + day if day else ''}: {_result_text(result)}")
    elif e["status"] == DROPPED:
        parts.append(f"Dropped{': ' + _sentence(e['status_note']) if e.get('status_note') else '.'}")
    else:
        parts.append(f"Status: {e['status']}.")
        if result:
            parts.append(f"Latest reading: {_result_text(result)}")
    return " ".join(parts)


def checklist(experiments: Iterable[Mapping[str, Any]]) -> str:
    """Checklist lines, one per experiment, in the order given."""
    return "\n".join(checklist_line(e) for e in experiments)


def _print_text(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.flush()
        sys.stdout.buffer.write((text + "\n").encode("utf-8"))
        sys.stdout.buffer.flush()


# --- the verb -------------------------------------------------------------------------------------------------

def _pair(text: str, flag: str) -> tuple[str, str]:
    name, sep, value = text.partition("=")
    if not sep or not name.strip() or not value.strip():
        raise argparse.ArgumentTypeError(f"{flag} takes NAME=VALUE, not {text!r}")
    return name.strip(), value.strip()


def _arm(text: str) -> tuple[str, str]:
    return _pair(text, "--arm")


def _parse_reading(name: str, spec: str, experiment: Mapping[str, Any], state_dir: Any) -> Any:
    """One --arm value of result: run:ID[,ID...], x/n, or comma-separated run values."""
    if spec.startswith("run:"):
        run_ids = [part.strip() for part in spec[len("run:"):].split(",") if part.strip()]
        if not run_ids:
            raise ExperimentError([f"arm {name}: run: needs at least one run id"])
        values = runs_from_ledger(experiment["metric"], run_ids, state_dir=state_dir)
        if not values:
            raise ExperimentError([f"arm {name}: no ledger row carries {experiment['metric']} under run id "
                                   f"{', '.join(run_ids)}"])
        return values
    if "/" in spec:
        x, _, n = spec.partition("/")
        try:
            return verdict_math.Counts(int(x), int(n))
        except (TypeError, ValueError) as exc:
            raise ExperimentError([f"arm {name}: {spec!r} is not x/n counts ({exc})"]) from None
    try:
        values = [float(part) for part in spec.split(",") if part.strip()]
    except ValueError:
        raise ExperimentError([f"arm {name}: {spec!r} is not a list of numbers, x/n counts or run:ID"]) from None
    if not values:
        raise ExperimentError([f"arm {name}: no run values in {spec!r}"])
    return values


def _definition_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--track", choices=TRACKS)
    p.add_argument("--hypothesis")
    p.add_argument("--lever")
    p.add_argument("--arm", dest="arms", action="append", type=_arm, metavar="NAME=TEXT",
                   help="an arm and what it runs; repeat it, the control first")
    p.add_argument("--kind", choices=KINDS, help="measure (default) or rate")
    p.add_argument("--metric")
    p.add_argument("--better", choices=BETTER)
    p.add_argument("--secondary", nargs="+", metavar="METRIC")
    p.add_argument("--expected", help="the expected movement in words")
    p.add_argument("--expected-delta", type=float, metavar="X",
                   help="the expected change as a signed fraction of the control's value (-0.85 is 85%% lower)")
    p.add_argument("--confidence", type=float, metavar="C", help="0 to 1")
    p.add_argument("--min-n", type=int, metavar="N", help="runs or trials an arm needs before a verdict")
    p.add_argument("--spread", type=float, metavar="SD", help="the spread of one run, to size min_n")
    p.add_argument("--baseline", type=float, metavar="VALUE",
                   help="the control's current value (a rate for a rate experiment), to size min_n")
    p.add_argument("--citation")
    p.add_argument("--title")
    p.add_argument("--task", help=f"the Task Board ID it belongs to (default {DEFAULT_TASK})")
    p.add_argument("--node")
    p.add_argument("--note")


def _fields_from(args: argparse.Namespace, current: Mapping[str, Any] | None) -> dict[str, Any]:
    """The definition fields the arguments give; min_n sized from --spread and --baseline when asked."""
    fields: dict[str, Any] = {}
    for key in ("track", "hypothesis", "lever", "kind", "metric", "better", "secondary", "expected",
                "expected_delta", "confidence", "min_n", "citation", "title", "task", "node", "note"):
        value = getattr(args, key)
        if value is not None:
            fields[key] = value
    if args.arms:
        names = [name for name, _ in args.arms]
        if len(set(names)) != len(names):
            raise ExperimentError([f"an arm is named twice: {', '.join(names)}"])
        fields["arms"] = dict(args.arms)
    if args.baseline is not None or args.spread is not None:
        if args.min_n is not None:
            raise ExperimentError(["give --min-n or --spread with --baseline, not both"])
        base = {**(current or {}), **fields}
        kind = base.get("kind") or MEASURE
        delta = base.get("expected_delta")
        if args.baseline is None or delta is None:
            raise ExperimentError(["sizing min_n needs --baseline and --expected-delta (and --spread for a measure)"])
        try:
            fields["min_n"] = planned_min_n(kind, delta, args.baseline, args.spread)
        except ValueError as exc:
            raise ExperimentError([f"cannot size min_n: {exc}"]) from None
    return fields


def _refused(exc: ExperimentError, **more: Any) -> dict[str, Any]:
    return {"ok": False, "outcome": "refused", "error": str(exc), "problems": exc.problems, **more}


def _run_add(args: argparse.Namespace) -> dict[str, Any]:
    if args.amend:
        if not args.id:
            return _refused(ExperimentError(["--amend needs the id of the experiment to change"]))
        current = load().get(args.id)
        try:
            changes = _fields_from(args, current)
            if args.status is not None:
                changes["status"] = args.status
            experiment = amend(args.id, changes, reason=args.reason)
        except ExperimentError as exc:
            return _refused(exc, id=args.id)
        return {"ok": True, "outcome": "amended", "id": args.id, "seq": experiment["seq"], "experiment": experiment}
    if args.status is not None or args.reason is not None:
        return _refused(ExperimentError(["--status and --reason change an experiment that exists: use them with "
                                         "--amend"]))
    try:
        fields = _fields_from(args, None)
        if args.id:
            fields["id"] = args.id
        experiment = add(fields)
    except ExperimentError as exc:
        return _refused(exc, id=args.id)
    warnings = [f"no {key}: the experiment cannot be ranked until it has one" for key in _needs(experiment)]
    return {"ok": True, "outcome": "added", "id": experiment["id"], "seq": experiment["seq"],
            "experiment": experiment, "warnings": warnings}


def _select(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    experiments, problems = fold(_rows(None))
    chosen = list(experiments.values())
    if getattr(args, "ids", None):
        chosen = [e for e in chosen if e["id"] in args.ids]
    if getattr(args, "statuses", None):
        chosen = [e for e in chosen if e["status"] in args.statuses]
    if args.track:
        chosen = [e for e in chosen if e["track"] == args.track]
    return chosen, problems


def _warn(problems: Sequence[str]) -> None:
    for problem in problems:
        print(f"experiments: {problem}", file=sys.stderr)


def _run_list(args: argparse.Namespace) -> dict[str, Any] | int:
    chosen, problems = _select(args)
    if args.print_text:
        _warn(problems)
        _print_text(checklist(chosen) if chosen else "No experiments match.")
        return 0
    return {"ok": True, "outcome": "listed", "count": len(chosen), "experiments": chosen, "problems": problems}


def _run_rank(args: argparse.Namespace) -> dict[str, Any] | int:
    chosen, problems = _select(args)
    ordered = rank(chosen, include_finished=args.all)
    if args.print_text:
        _warn(problems)
        _print_text(checklist(ordered) if ordered else "No experiments match.")
        return 0
    entries = [{"rank": e["rank"], "id": e["id"], "track": e["track"], "status": e["status"], "score": e["score"],
                "gain": e["gain"], "confidence": e["confidence"], "expected_delta": e["expected_delta"],
                "metric": e["metric"], "better": e["better"], "min_n": e["min_n"], "lever": e["lever"],
                "needs": e["needs"]} for e in ordered]
    return {"ok": True, "outcome": "ranked", "count": len(entries), "experiments": entries, "problems": problems}


def _run_result(args: argparse.Namespace) -> dict[str, Any]:
    experiment = load().get(args.id)
    if experiment is None:
        return _refused(ExperimentError([f"no experiment {args.id}"]), id=args.id)
    names = [name for name, _ in args.arms]
    if len(set(names)) != len(names):
        return _refused(ExperimentError([f"an arm is read twice: {', '.join(names)}"]), id=args.id)
    refs = {key: value for key, value in (("slot", args.slot), ("run_id", args.run_id)) if value is not None}
    try:
        arms = {name: _parse_reading(name, spec, experiment, None) for name, spec in args.arms}
        updated = record_result(args.id, arms, control=args.control, final=args.final, note=args.note, refs=refs)
    except ExperimentError as exc:
        return _refused(exc, id=args.id)
    result = updated["result"]
    return {"ok": True, "outcome": "recorded", "id": args.id, "seq": updated["seq"], "label": result["label"],
            "winner": result["winner"], "status": updated["status"], "summary": _result_text(result),
            "result": result}


def _run_seed(args: argparse.Namespace) -> dict[str, Any]:
    try:
        outcome = seed()
    except ExperimentError as exc:
        return _refused(exc)
    return {"ok": True, "outcome": "seeded", **outcome}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m lab experiment", description=main.__doc__)
    subs = parser.add_subparsers(dest="sub", required=True, metavar="{add,list,result,rank,seed}")

    p = subs.add_parser("add", help="add an experiment, or change one with --amend")
    p.add_argument("id", nargs="?", help="E<n> when left out")
    _definition_args(p)
    p.add_argument("--amend", action="store_true", help="change an existing experiment")
    p.add_argument("--status", choices=SETTABLE_STATUSES, help="with --amend: queued, running or dropped")
    p.add_argument("--reason", help="with --amend: why it changed (shown beside a dropped experiment)")
    p.set_defaults(run=_run_add)

    p = subs.add_parser("list", help="the experiments as they stand, in the order they were added")
    p.add_argument("--id", dest="ids", action="append", metavar="ID")
    p.add_argument("--status", dest="statuses", action="append", choices=STATUSES)
    p.add_argument("--track", choices=TRACKS)
    p.add_argument("--print", dest="print_text", action="store_true", help="print T0 checklist lines, not JSON")
    p.set_defaults(run=_run_list)

    p = subs.add_parser("result", help="read the arms with the verdict math and record the result")
    p.add_argument("id")
    p.add_argument("--arm", dest="arms", action="append", required=True, type=_arm, metavar="NAME=READING",
                   help="run values (7.7,7.9), a rate (12/60) or run:ID[,ID...]; the control and at least one more")
    p.add_argument("--control", help="the arm the others are read against (default: the first arm)")
    p.add_argument("--final", action="store_true", help="close the experiment even if the reading is too early")
    p.add_argument("--note")
    p.add_argument("--slot", help="the GPU queue slot the runs used")
    p.add_argument("--run-id", help="the run the readings came from")
    p.set_defaults(run=_run_result)

    p = subs.add_parser("rank", help="the open experiments by expected gain times confidence")
    p.add_argument("--track", choices=TRACKS)
    p.add_argument("--all", action="store_true", help="also list the done and dropped ones, after the open ones")
    p.add_argument("--print", dest="print_text", action="store_true", help="print T0 checklist lines, not JSON")
    p.set_defaults(run=_run_rank)

    p = subs.add_parser("seed", help="add the experiments of T0's queue that the ledger lacks")
    p.set_defaults(run=_run_seed)
    return parser


def main(argv: Sequence[str]) -> dict[str, Any] | int:
    """Experiments as ledger rows: add, list, result (read by the verdict math), rank, and T0's checklist (--print)."""
    args = _parser().parse_args(list(argv))
    return args.run(args)


# --- the seeds: the experiment queue on T0 --------------------------------------------------------------------

SEED_SOURCE = ("seed: T0 experiment queue (https://app.notion.com/p/3f596ef5fec581819588ec7a702a4bb1) and the "
               "research notes under lab/refine/research/")
_T0 = "T0 experiment queue (https://app.notion.com/p/3f596ef5fec581819588ec7a702a4bb1)"
_KERNELS_NOTE = "lab/refine/research/kernels/w4a16_decode_gfx1201.md"

#: The five experiments on T0 when the ledger took over the queue. The arms and metrics are T0's; the estimates
#: come from the research notes named in each citation, and stay None where no note gives one.
SEEDS: tuple[dict[str, Any], ...] = (
    {"id": "E1", "title": "Senses contention", "track": "plumbing", "task": DEFAULT_TASK,
     "hypothesis": "A fine-tune and a serving replica sized to share GPU B can run together; the trainer's "
                   "samples per second show what coexisting costs against stopping the replica.",
     "lever": "whether the serving replica on GPU B is stopped while a fine-tune runs",
     "arms": {"A": "preempt: stop the serving replica for the fine-tune", "B": "coexist: replica and fine-tune on "
              "the card together, both sized to fit"},
     "kind": MEASURE, "metric": "trainer.samples_per_s", "better": HIGHER,
     "secondary": ["teacher.decode_tok_s", "teacher.joules_per_token"],
     "expected": None, "expected_delta": None, "confidence": None, "min_n": None,
     "citation": f"{_T0}, E1", "node": "pod-b",
     "note": "Also record watts and any out-of-memory error, with telemetry on. The result decides the "
             "controller's stop and start policy."},
    {"id": "E2", "title": "vLLM vs llama.cpp on the R9700", "track": "kernels", "task": DEFAULT_TASK,
     "hypothesis": "On the same card and prompts llama.cpp decodes the 4-bit teacher about 4x faster than vLLM at "
                   "b1388b1, whose W4A16 layers run on the Conch Triton fallback.",
     "lever": "the serving engine",
     "arms": {"A": "vLLM b1388b1 with lab/vllm-gfx1201.diff serving RedHatAI/Qwen3.8-27B-INT4",
              "B": "llama.cpp built for gfx1201 serving a 4-bit GGUF of the same model"},
     "kind": MEASURE, "metric": "teacher.decode_tok_s", "better": HIGHER,
     "secondary": ["teacher.joules_per_token", "teacher.ttft_s"],
     "expected": "about 4x tokens per second: 7.7 tok/s today against the 29 to 31 tok/s llama.cpp measured for a "
                 "4-bit 27B on one R9700",
     "expected_delta": 2.9, "confidence": None, "min_n": 5,
     "citation": f"{_T0}, E2; {_KERNELS_NOTE} section 4 (S26)", "node": "pod-b", "note": None},
    {"id": "E3", "title": "Batching vs joules per token", "track": "kernels", "task": DEFAULT_TASK,
     "hypothesis": "Serving several requests at once spreads each step's weight reads over more tokens, so joules "
                   "per token fall steeply with batch size.",
     "lever": "batch size (concurrent requests) on the vLLM teacher",
     "arms": {"A": "batch 1", "B": "batch 4", "C": "batch 16", "D": "batch 64"},
     "kind": MEASURE, "metric": "teacher.joules_per_token", "better": LOWER, "secondary": ["teacher.decode_tok_s"],
     "expected": "26 to 2-4 J/token at concurrency 16, aggregate 7.7 to 60-110 tok/s, per-request time per token "
                 "up 1.1x to 1.5x",
     "expected_delta": -0.85, "confidence": 0.75, "min_n": 3,
     "citation": f"{_T0}, E3; {_KERNELS_NOTE} section 7, KE1", "node": "pod-b",
     "note": "3 spawns per level with 10 requests each (KE1); KV room may cap the largest batch."},
    {"id": "E4", "title": "Jetson power modes", "track": "telemetry", "task": DEFAULT_TASK,
     "hypothesis": "The student's joules per token are lowest at 25 W rather than at MAXN SUPER, because the top "
                   "clocks buy speed with energy.",
     "lever": "the Jetson's nvpmodel power mode",
     "arms": {"A": "MAXN SUPER (mode 2, what the box runs today)", "B": "25 W", "C": "15 W", "D": "7 W"},
     "kind": MEASURE, "metric": "student.joules_per_token", "better": LOWER, "secondary": ["student.decode_tok_s"],
     "expected": "J/token lowest at 25 W and within 10% of the other modes; tok/s about 30% lower at 15 W than at "
                 "25 W and about 10% higher at MAXN SUPER",
     "expected_delta": None, "confidence": None, "min_n": 32,
     "citation": f"{_T0}, E4; lab/refine/research/telemetry/power-and-clock-levers.md section 2 (LV-05, LV-07)",
     "node": "jetson",
     "note": "Needs Ilan's go and a time outside the demo window: a mode change can require a reboot, and the box "
             "goes back to mode 2 after. The Super configuration lists 15 W, 25 W and MAXN SUPER; check that 7 W "
             "exists before the run."},
    {"id": "E5", "title": "Teacher checkpoint shootout", "track": "kernels", "task": DEFAULT_TASK,
     "hypothesis": "The official FP8 checkpoint cuts single-stream joules per token against RedHat INT4, while "
                   "AMD's Quark AWQ-INT4 matches RedHat INT4 within noise because both run the same W4A16 kernel.",
     "lever": "the teacher checkpoint",
     "arms": {"A": "RedHatAI/Qwen3.8-27B-INT4 (served today)", "B": "amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16",
              "C": "Qwen/Qwen3.8-27B-FP8"},
     "kind": MEASURE, "metric": "teacher.joules_per_token", "better": LOWER, "secondary": ["teacher.decode_tok_s"],
     "expected": "C -40% to -50% J/token at single-stream decode with almost no KV room left (KE14, confidence "
                 "0.2); B within the spawn band of A (confidence 0.6)",
     "expected_delta": -0.40, "confidence": 0.20, "min_n": 5,
     "citation": f"{_T0}, E5; {_KERNELS_NOTE} sections 3.4 and 7 (KE14)", "node": "pod-b",
     "note": "B loads only after the Quark weight mapper is backported or vLLM moves on (lab/README.md). Check "
             "that each arm loads, and score caption quality on 20 fixture frames."},
)
