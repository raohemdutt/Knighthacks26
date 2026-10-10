"""Summarize pod GPU power samples, or store a window of them as a ledger row: summary | record.

    python -m lab gpu-power summary --samples FILE [FILE ...] [--node gpu-a] [--since ISO] [--until ISO]
    python -m lab gpu-power record --samples FILE [FILE ...] --metric pod.idle_watts --node pod-a
                                   [--value mean|peak|min] [--since ISO] [--until ISO] [--state idle]
                                   [--metrics-file [PATH]] [--dry-run]

The samples are telemetry.gpu.power events written by lab/pod_power.py: the pod's own daily files
(~/atlas-power/samples-<day>.jsonl, fetched from the pod) or the Jetson's event log, whose copies also carry
received_ts. Other lines in the files are ignored. Nodes may be given by either name: the telemetry one (gpu-a,
gpu-b) or the ledger one (pod-a, pod-b).

summary prints what lab.pod_power.summarize says about the window and writes nothing.

record writes one ledger row: verb "gpu-power record", node pod-a or pod-b, one statistic of the window's watts under
the metric (unit W), and the whole summary as its conditions. --value picks the statistic: mean (the default, the
summary's mean_w), peak (its max_w) or min (its min_w), so pod.idle_watts stores the mean and pod.peak_watts can store
the peak. Any other choice than mean is kept in the conditions as value (value: peak), so a mean row reads exactly as
rows written before --value existed, and a window with no power reading is refused with outcome no_readings whichever
statistic was asked for. The conditions carry the sampler, the tool, the power field,
period_ms next to the period actually measured between samples, the longest gap and the restarts inside the window
(and, for samples read from the Jetson log, the longest gap between their receive times and the latest arrival),
the samples and how many were distinct updates, the starting temperatures, the energy and where it came from, and
error, which stays "unknown" until a reference run measures the sampler's error. That makes the row low_confidence.
--metrics-file also appends the reading to lab/refine/metrics.jsonl, or to the file named. --dry-run prints the row
and writes nothing.

The two stores never disagree about a reading. --metric must be a name the metrics file accepts (dotted lowercase
words, e.g. pod.idle_watts), checked before anything is read; with --metrics-file the whole metrics row is checked
before either file is written, and the metrics file, the stricter store, is appended to before the ledger, so a row it
refuses or a lock it cannot take leaves both files untouched and answers outcome error with the reason.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from lab import pod_power
from lab.machine import ledger, metrics
from lab.machine.record import record

VERB = "gpu-power record"
VALUE_STATISTIC = {"mean": "mean_w", "peak": "max_w", "min": "min_w"}  # --value choice -> summary key it stores
DEFAULT_VALUE = "mean"
TELEMETRY_NODE = {ledger: node for node, ledger in pod_power.LEDGER_NODE.items()}


def _node(text: str) -> str:
    """The telemetry name of a pod given by either of its names."""
    name = text.strip().lower()
    if name in pod_power.LEDGER_NODE:
        return name
    if name in TELEMETRY_NODE:
        return TELEMETRY_NODE[name]
    raise argparse.ArgumentTypeError(f"{text!r} is not a pod: use gpu-a, gpu-b, pod-a or pod-b")


def _metric(text: str) -> str:
    """A metric name both the ledger and the metrics file accept, so a reading never lands in only one of them."""
    try:
        return metrics.check_metric_name(text)
    except metrics.MetricsFormatError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _when(text: str) -> datetime:
    try:
        return pod_power.parse_ts(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an ISO 8601 time such as 2026-10-10T18:00:00Z") from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m lab gpu-power", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("summary", "record"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--samples", nargs="+", required=True, metavar="FILE", help="JSONL files of samples")
        cmd.add_argument("--node", type=_node, required=name == "record", help="gpu-a/pod-a or gpu-b/pod-b")
        cmd.add_argument("--since", type=_when, help="first sample time to include (ISO 8601)")
        cmd.add_argument("--until", type=_when, help="last sample time to include (ISO 8601)")
        if name == "record":
            cmd.add_argument("--metric", type=_metric, required=True, help="dotted metric name, e.g. pod.idle_watts")
            cmd.add_argument("--value", choices=tuple(VALUE_STATISTIC), default=DEFAULT_VALUE,
                             help="statistic of the window's watts to store: mean (default), peak or min")
            cmd.add_argument("--state", help="what the card was doing, kept in the conditions (idle, teacher-loaded ...)")
            cmd.add_argument("--metrics-file", nargs="?", const=str(metrics.DEFAULT_PATH), metavar="PATH",
                             help="also append the reading to the metrics file (default lab/refine/metrics.jsonl)")
            cmd.add_argument("--dry-run", action="store_true", help="print the row, write nothing")
    return parser


def summary_of(args: argparse.Namespace) -> dict[str, Any]:
    events = pod_power.read_samples(args.samples)
    return pod_power.summarize(events, node=args.node, start=args.since, end=args.until)


def _metrics_ts(iso_text: str) -> str:
    """The summary's end time in the metrics file's format (UTC, at most six decimals, a Z)."""
    return pod_power.parse_ts(iso_text).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _error(reason: str) -> dict[str, Any]:
    """The dispatcher's error answer: not ok, outcome error, and the reason."""
    return {"ok": False, "outcome": "error", "error": reason}


def main(argv: Sequence[str]) -> dict[str, Any] | int:
    """Summarize pod GPU power samples, or store a window of them as a ledger row: summary | record."""
    args = build_parser().parse_args(list(argv))
    try:
        summary = summary_of(args)
    except (OSError, ValueError) as exc:
        return _error(str(exc))
    if args.command == "summary":
        return {"ok": True, "summary": summary}
    value: float | None = summary[VALUE_STATISTIC[args.value]]
    if value is None:
        return {"ok": False, "outcome": "no_readings",
                "error": f"no sample in the window has a power reading, so there is no {args.value} to store",
                "summary": summary}
    ledger_node = pod_power.LEDGER_NODE[args.node]
    conditions = {**summary, "state": args.state}
    if args.value != DEFAULT_VALUE:  # a mean row stays the shape every row before --value had
        conditions["value"] = args.value
    source = f"{summary['sampler']} samples from {', '.join(Path(p).name for p in args.samples)}"
    row = {"verb": VERB, "outcome": "ok", "node": ledger_node, "metrics": {args.metric: value},
           "units": {args.metric: "W"}, "conditions": conditions, "low_confidence": True, "source": source}
    metrics_row = {"metric": args.metric, "value": value, "unit": "W", "node": ledger_node,
                   "conditions": conditions, "source": source, "ts": _metrics_ts(summary["end"]),
                   "low_confidence": True}
    if args.metrics_file:
        try:
            metrics.check_row(metrics_row)
        except metrics.MetricsFormatError as exc:
            return _error(f"the metrics file would refuse the reading, so nothing was written: {exc}")
    if args.dry_run:
        return {"ok": True, "outcome": "dry_run", "row": row,
                "metrics_row": metrics_row if args.metrics_file else None}
    written = None
    if args.metrics_file:
        try:
            written = metrics.add(metrics_row, path=args.metrics_file)
        except (metrics.MetricsFormatError, metrics.MetricsLocked) as exc:
            return _error(f"could not append the reading to {args.metrics_file}, "
                          f"so neither it nor the ledger was written: {exc}")
    try:
        seq = record(VERB, "ok", metrics=row["metrics"], node=ledger_node, units=row["units"], conditions=conditions,
                     low_confidence=True, source=source)
    except (ledger.LockTimeout, ledger.RowError) as exc:
        if written is None:
            return _error(f"the ledger refused the reading, so nothing was written: {exc}")
        return {**_error(f"{args.metrics_file} holds the reading but the ledger refused it: {exc}"),
                "metrics_file": args.metrics_file, "metrics_row": written}
    return {"ok": True, "outcome": "recorded", "seq": seq, "row": row, "metrics_file": args.metrics_file,
            "metrics_row": written}
