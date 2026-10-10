"""What the telemetry log says right now, per node: folded from `telemetry.*` and rebuilt by replaying the log.

    projection = TelemetryProjection()
    projection.subscribe(bus)                       # replay=True: the log rebuilds it on start
    projection.snapshot()                           # nodes, recent requests, training runs, counts
    projection.readings("gpu-a", start, end)        # power readings in a window, oldest first

Per node and power source it keeps the latest watts, how many samples arrived (and how many had no
reading), the configured period next to the period actually measured between samples, the longest
gap, and samples that arrived out of order. Request windows and power readings are kept for a
bounded recent stretch; energy over a window is computed from `readings()` by its consumers.

Event types it does not know yet (a new source) are counted per node and otherwise left alone,
so adding a source never breaks the projection.
"""

from collections import Counter, deque
from datetime import datetime
from typing import Any

from bus import EventBus
from contexts.telemetry.domain.power import PowerReading, reading_of
from contracts.events import Event
from contracts.telemetry import GpuPower, JetsonPower, RequestWindow, TrainStep


class TelemetryProjection:
    def __init__(self, keep: int = 20_000, keep_requests: int = 1_000, recent: int = 20) -> None:
        self.keep = keep  # power readings kept per node and source
        self.keep_requests = keep_requests  # request windows kept per node
        self.recent = recent  # request windows per node in the snapshot
        self.counts: Counter[str] = Counter()
        self._nodes: dict[str, dict[str, Any]] = {}
        self._power: dict[tuple[str, str], dict[str, Any]] = {}
        self._readings: dict[tuple[str, str], deque[PowerReading]] = {}
        self._requests: dict[str, deque[RequestWindow]] = {}
        self._runs: dict[str, dict[str, Any]] = {}

    def subscribe(self, bus: EventBus) -> None:
        bus.subscribe("telemetry.", self.on_event, replay=True)

    def on_event(self, event: Event) -> None:
        node = getattr(event, "node", None)
        if not isinstance(node, str) or not node:
            return  # every telemetry event names its node; anything else is not ours to fold
        self.counts[event.type] += 1
        status = self._node(node)
        status["events"][event.type] = status["events"].get(event.type, 0) + 1
        if status["last_ts"] is None or event.ts > status["last_ts"]:
            status["last_ts"] = event.ts
        reading = reading_of(event)
        if reading is not None:
            self._on_reading(reading, event)
        elif isinstance(event, RequestWindow):
            self._requests.setdefault(node, deque(maxlen=self.keep_requests)).append(event)
        elif isinstance(event, TrainStep):
            self._on_step(event)

    def _node(self, node: str) -> dict[str, Any]:
        return self._nodes.setdefault(node, {"last_ts": None, "events": {}, "nvpmodel": None, "throttled": None,
                                             "temps_c": {}})

    def _on_reading(self, r: PowerReading, event: Event) -> None:
        key = (r.node, r.source)
        p = self._power.setdefault(key, {"boundary": r.boundary, "samples": 0, "missing": 0, "latest_w": None,
                                         "latest_ts": None, "first_ts": None, "period_ms": None,
                                         "measured_period_ms": None, "max_gap_ms": None, "out_of_order": 0})
        p["samples"] += 1
        p["missing"] += int(r.watts is None)
        p["period_ms"] = event.period_ms
        if p["latest_ts"] is not None and r.ts <= p["latest_ts"]:
            p["out_of_order"] += 1  # a late sample: kept in the readings, not the latest
        else:
            if p["latest_ts"] is not None:
                gap_ms = (r.ts - p["latest_ts"]).total_seconds() * 1000
                p["max_gap_ms"] = gap_ms if p["max_gap_ms"] is None else max(p["max_gap_ms"], gap_ms)
            p["latest_ts"], p["latest_w"] = r.ts, r.watts
            status = self._node(r.node)
            if isinstance(event, JetsonPower):
                status["nvpmodel"], status["temps_c"] = event.nvpmodel, dict(event.temps_c)
            elif isinstance(event, GpuPower):
                status["throttled"], status["temps_c"] = event.throttled, dict(event.temps_c)
        if p["first_ts"] is None or r.ts < p["first_ts"]:
            p["first_ts"] = r.ts
        if p["samples"] > 1:
            span_ms = (p["latest_ts"] - p["first_ts"]).total_seconds() * 1000
            p["measured_period_ms"] = span_ms / (p["samples"] - 1)
        self._readings.setdefault(key, deque(maxlen=self.keep)).append(r)

    def _on_step(self, e: TrainStep) -> None:
        run = self._runs.setdefault(e.run_id, {"node": e.node, "steps": 0, "first_ts": e.ts, "last_ts": e.ts,
                                               "step": None, "loss": None, "step_ms": None, "socket_w": None})
        run["steps"] += 1
        run["first_ts"] = min(run["first_ts"], e.ts)
        if e.ts >= run["last_ts"] or run["step"] is None:
            run.update(node=e.node, last_ts=e.ts, step=e.step, loss=e.loss, step_ms=e.step_ms, socket_w=e.socket_w)

    # --- reads ---

    def readings(self, node: str, start: datetime | None = None, end: datetime | None = None,
                 source: str | None = None) -> list[PowerReading]:
        """Power readings of `node` with start <= ts <= end, oldest first (every source unless one is named)."""
        out = [r for (n, s), rs in self._readings.items() if n == node and source in (None, s) for r in rs
               if (start is None or r.ts >= start) and (end is None or r.ts <= end)]
        return sorted(out, key=lambda r: r.ts)

    def requests(self, node: str, start: datetime | None = None, end: datetime | None = None) -> list[RequestWindow]:
        """Request windows answered on `node` that overlap [start, end], in the order they started."""
        out = [w for w in self._requests.get(node, ())
               if (start is None or w.ended >= start) and (end is None or w.started <= end)]
        return sorted(out, key=lambda w: w.started)

    def snapshot(self) -> dict[str, Any]:
        nodes = {}
        for node, status in sorted(self._nodes.items()):
            power = {s: dict(p) for (n, s), p in sorted(self._power.items()) if n == node}
            nodes[node] = {**status, "events": dict(sorted(status["events"].items())), "temps_c": dict(status["temps_c"]),
                           "power": power}
        requests = {node: [_window(w) for w in self.requests(node)[-self.recent:]] for node in sorted(self._requests)}
        return {"nodes": nodes, "requests": requests, "train_runs": {k: dict(v) for k, v in sorted(self._runs.items())},
                "counts": dict(sorted(self.counts.items()))}


def _window(w: RequestWindow) -> dict[str, Any]:
    prefill_s = (w.first_token - w.started).total_seconds() if w.first_token else None
    return {"request_id": w.request_id, "target": w.target, "started": w.started, "ended": w.ended,
            "duration_s": (w.ended - w.started).total_seconds(), "prefill_s": prefill_s, "tokens_in": w.tokens_in,
            "tokens_out": w.tokens_out, "model": w.model, "engine": w.engine, "since_load": w.since_load,
            "clock": w.clock}
