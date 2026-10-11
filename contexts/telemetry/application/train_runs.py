"""Training runs as their `telemetry.train.step` events describe them, folded and rebuilt by replaying the log.

    runs = TrainRuns()
    runs.subscribe(bus)         # replay=True: telemetry.train.step, whichever way it reached the log
    runs.on_node("pod-b")       # {run_id: {node, steps, first_ts, last_ts, step_ms, loss, loss_step, loss_ts,
                                #           first_loss}}

This fold is on whether or not the box runs a power sampler (ATLAS_TELEMETRY_SOURCE): the trainer's steps reach the
box's log through contexts/learning/lab_import.py (the command, or POST /api/learning/lab/import from the lab), not
through a sampler, so GET /runtime/stats reads the runs here rather than from the telemetry projection.

A step is known by (run_id, step): the same step arriving twice (a file imported again, a replay) counts once, and
the later copy replaces it. `loss` is the newest step that has a finite loss (a step with none is still counted);
`first_loss` the oldest such step, so a reader can say where the run started from.
"""

import math
from datetime import datetime
from typing import Any

from bus import EventBus
from contexts.telemetry.domain.nodes import canonical
from contracts.telemetry import TrainStep


class TrainRuns:
    def __init__(self) -> None:
        self._steps: dict[str, dict[int, TrainStep]] = {}

    def subscribe(self, bus: EventBus) -> None:
        bus.subscribe(TrainStep.TYPE, self.on_step, replay=True)

    def on_step(self, e: TrainStep) -> None:
        self._steps.setdefault(e.run_id, {})[e.step] = e

    def runs(self) -> dict[str, dict[str, Any]]:
        return {run_id: _summary(steps) for run_id, steps in sorted(self._steps.items()) if steps}

    def on_node(self, node: str) -> dict[str, dict[str, Any]]:
        """The runs whose newest step names `node` (an old gpu-b counts as pod-b)."""
        return {k: r for k, r in self.runs().items() if r["node"] == node}


def _finite(loss: float | None) -> bool:
    return loss is not None and math.isfinite(loss)


def _summary(steps: dict[int, TrainStep]) -> dict[str, Any]:
    by_time = sorted(steps.values(), key=lambda s: (s.ts, s.step))
    last = by_time[-1]
    with_loss = [s for s in by_time if _finite(s.loss)]
    first_ts: datetime = by_time[0].ts
    out: dict[str, Any] = {"node": canonical(last.node), "steps": len(by_time), "first_ts": first_ts,
                           "last_ts": last.ts, "step_ms": last.step_ms, "loss": None, "loss_step": None,
                           "loss_ts": None, "first_loss": None}
    if with_loss:
        newest, oldest = with_loss[-1], with_loss[0]
        out.update(loss=newest.loss, loss_step=newest.step, loss_ts=newest.ts, first_loss=oldest.loss)
    return out
