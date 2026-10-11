"""What GET /runtime/stats serves: the box's and the lab's numbers, assembled from what is already recorded.

    stats = RuntimeStats(evals, projection, LabRecords())   # contexts/telemetry/runtime_stats.py builds it
    await stats.snapshot()

    {"detector_fps":          Figure,   Jetson detector, frames per second
     "student_tok_s":         Figure,   on-device model decode, tokens per second (the Jetson)
     "caption_latency_s":     Figure,   on-device caption, seconds per caption (the Jetson)
     "trainer_samples_per_s": Figure,   trainer card (GPU B), training samples per second
     "train_minutes":         Figure,   trainer card (GPU B), minutes per training run
     "teacher_tok_s":         Figure,   teacher server (GPU A), decode tokens per second
     "eval_scores":           {"models": [EvalRow, ...], "reason": ...}}   before and after per deployed model

Each Figure is {value, unit, node, ts, source, low_confidence, reason} (domain/stats.py). It measures nothing and
calls nothing: no model server, pod, hub or Jetson tool is asked. A figure takes the newest of its candidates:
- the lab's metrics file and ledger (infrastructure/lab_records.py), under the metric names in FIGURES;
- the telemetry projection, when the gateway runs telemetry: decode rates over the newest RECENT_WINDOWS
  `telemetry.request` windows of the node that answered (jetson for the on-device model, pod-a for the teacher), and
  the minutes of the newest training
  run from its `telemetry.train.step` events on pod-b. A run counts once a `learning.run.evaluated` names it (its
  training has ended); while none does, the newest run is served with low_confidence, since it may still be training.
On a tie the metrics file is cited first, then the ledger (a ledger backfill copies metrics rows with their times).
When no candidate exists the figure is null and its reason names the metric, the node and the events looked for.

Most figures have no producer in the repo yet, so they can only be null until one is built or a number is recorded by
hand (`python -m lab ledger add ...`, or a row in lab/refine/metrics.jsonl). Nothing writes a detector.fps,
student.decode_tok_s, student.caption_s or trainer.run_minutes row; nothing publishes `telemetry.request` or
`telemetry.train.step` (the Jetson's only telemetry source, tegrastats, publishes power readings); and nothing publishes
`learning.run.evaluated`, so a deployed model's before and after scores are null with their reason. Today only
teacher_tok_s and trainer_samples_per_s carry values, both from the committed metrics file. This endpoint does not
measure them itself: the producers belong to the contexts that run the detector, the on-device model and the trainer.

The projection and the eval fold are read on the event loop, where their events land; the files are read in a
worker thread, so a slow disk or a held lock never stalls the loop.
"""

import asyncio
from dataclasses import dataclass
from typing import Any

from contexts.telemetry.application.evals import EvalScores
from contexts.telemetry.application.projection import TelemetryProjection
from contexts.telemetry.domain.nodes import JETSON, POD_A, POD_B
from contexts.telemetry.domain.stats import Figure, decode_rate, missing, newest, run_minutes
from contexts.telemetry.infrastructure.lab_records import LabRecords

TELEMETRY_OFF = "telemetry is off on this box (ATLAS_TELEMETRY_SOURCE is not set)"
RECENT_WINDOWS = 50  # a decode rate is taken over this many of the node's newest request windows


@dataclass(frozen=True)
class Spec:
    key: str  # the field in the response
    unit: str
    metric: str  # its name in the lab's metrics file and ledger
    node: str | None  # the node its lab rows must name; None for any
    telemetry: str | None = None  # what the projection gives: "decode" (a request-window rate) or "run" (minutes)
    telemetry_node: str | None = None  # the node whose telemetry events it is computed from
    no_telemetry: str = ""  # why telemetry cannot give it, for a figure it cannot


FIGURES = (
    Spec("detector_fps", "frames/s", "detector.fps", JETSON, no_telemetry=(
        "the gateway keeps no detector timing (GET /capture/status counts detections without timing them)")),
    Spec("student_tok_s", "tok/s", "student.decode_tok_s", JETSON, telemetry="decode", telemetry_node=JETSON),
    Spec("caption_latency_s", "s", "student.caption_s", JETSON,
         no_telemetry="telemetry.request windows do not say which requests were captions"),
    Spec("trainer_samples_per_s", "samples/s", "trainer.samples_per_s", None,
         no_telemetry="telemetry.train.step carries no sample count"),
    Spec("train_minutes", "min", "trainer.run_minutes", None, telemetry="run", telemetry_node=POD_B),
    Spec("teacher_tok_s", "tok/s", "teacher.decode_tok_s", None, telemetry="decode", telemetry_node=POD_A),
)


class RuntimeStats:
    def __init__(self, evals: EvalScores, projection: TelemetryProjection | None = None,
                 lab: LabRecords | None = None) -> None:
        self.evals = evals
        self.projection = projection  # None when the gateway runs no telemetry
        self.lab = lab or LabRecords()

    async def snapshot(self) -> dict[str, Any]:
        telemetry = {s.key: self._from_telemetry(s) for s in FIGURES}  # (figure or None, why not)
        evals = self.evals.snapshot()
        found, problems = await asyncio.to_thread(self.lab.find, [(s.metric, s.node) for s in FIGURES])
        out: dict[str, Any] = {}
        for s in FIGURES:
            figure, why_not = telemetry[s.key]
            best = newest([*found[(s.metric, s.node)], figure])
            if best is None:
                node = f" for node {s.node}" if s.node else ""
                best = missing(s.unit, "; ".join([
                    f"not recorded yet: no {s.metric} row{node} in the lab metrics file or the lab ledger", why_not,
                    *problems]))
            out[s.key] = best.model_dump(mode="json")
        out["eval_scores"] = evals
        return out

    def _from_telemetry(self, s: Spec) -> tuple[Figure | None, str]:
        """The figure computed from the projection, or None and why not."""
        if s.telemetry is None:
            return None, s.no_telemetry
        if self.projection is None:
            return None, TELEMETRY_OFF
        if s.telemetry == "run":
            return self._run_minutes(s.telemetry_node)
        node = s.telemetry_node
        rate = decode_rate(self.projection.requests(node)[-RECENT_WINDOWS:])
        if rate is None:
            return None, (f"no telemetry.request window from node {node} with a first-token time and two or more "
                          "output tokens")
        value, used, last = rate
        return Figure(value=value, unit=s.unit, node=node, ts=last,
                      source=f"telemetry.request: {used} window{'s' if used != 1 else ''} from node {node}"), ""

    def _run_minutes(self, node: str) -> tuple[Figure | None, str]:
        runs = {k: r for k, r in self.projection.snapshot()["train_runs"].items() if r["node"] == node}
        if not runs:
            return None, f"no telemetry.train.step from node {node}"
        ended = {k: r for k, r in runs.items() if k in self.evals.evaluated_runs}
        run_id, run = max((ended or runs).items(), key=lambda kv: kv[1]["last_ts"])
        source = f"telemetry.train.step: run {run_id}, {run['steps']} step{'s' if run['steps'] != 1 else ''}"
        if not ended:
            source += " (no learning.run.evaluated for it yet, so it may still be training)"
        return Figure(value=run_minutes(run["first_ts"], run["last_ts"], run["step_ms"]), unit="min", node=node,
                      ts=run["last_ts"], source=source, low_confidence=not ended), ""
