"""Joules per answer: each answered request's energy, gross, with what it was measured under (telemetry.joules_per_answer).

    joules = JoulesPerAnswerProjection()
    joules.subscribe(bus)              # replay=True: samples, windows, offsets and stored values rebuild it
    for event in joules.ready(): ...   # values that can be computed now and are not in the log yet
    joules.stats()                     # watts now, joules per answer, and energy per day, per node

What a value is. The answering node's power integrated over the request window (telemetry.request: node, started,
ended, tokens_out), with nothing subtracted: gross, as Context: Telemetry defines it and telemetry ruling 1 keeps it.
The node's latest idle reading and its time are stored beside it, so a marginal figure can be derived later without
measuring again. The Jetson integrates VDD_IN (module input); the pods integrate socket_w, which on the Strix Halo is
the whole APU socket (CPU cores and graphics together); every value names its reading, boundary and accelerator.
The integral and the split between overlapping requests are domain/energy.py's: bracketing samples, linear
interpolation at both edges, every repeated reading kept, and each stretch split by output tokens.

One clock. A window is stamped on the clock its `clock` field names (the node's own when unset); a pod's samples
are on the pod's clock. Where the two differ, times move through the reference clock (the Jetson's) by the pod's
latest telemetry.clock.offset (pod minus Jetson, measured through the hub proxy from four timestamps), and the
offset and its error are stored with the value. A window that needs an offset nobody has measured yet waits.

When a value is computed. A request is reported once the node's samples have run `settle_s` past the end of every
request that overlaps it (so a longer request that started inside it and ends later is known first: the runtime's
request timeout is 120 s, the default settle 150 s), and once a sample at or after its end exists. All the requests
of one overlapping group are reported together, so their shares are consistent. A request that turns up later and
overlaps one already reported gets nothing from the stretches already given out, and says so (low_confidence).

low_confidence is true, with each reason named, when the window holds fewer than `min_samples` (10) samples, when
a pod window is shorter than the sensor's measured averaging lag, while the node's error has not been measured
(Calibration.error: none is, so every value is low confidence today; the Jetson's stays so by telemetry ruling 7),
when the samples carry two power modes, a pod sample was throttled, a Jetson overcurrent counter rose across the
window, a training step on the same node overlapped it, or it overlaps an answer reported before it was known.
"""

from bisect import bisect_left, bisect_right, insort
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from bus import EventBus
from contexts.telemetry.domain.energy import Window, attribute, brackets, integrate
from contexts.telemetry.domain.nodes import JETSON, canonical
from contexts.telemetry.domain.power import ACCELERATOR, BOUNDARY_TEXT, POWER_FIELD, reading_of
from contracts.events import Event
from contracts.telemetry import ClockOffset, GpuPower, JetsonPower, JoulesPerAnswer, RequestWindow, TrainStep, \
    WindowConditions

EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
SETTLE_S = 150.0  # past the runtime's 120 s request timeout: no request still running can overlap a reported one
MIN_SAMPLES = 10


@dataclass
class Calibration:
    """What has been measured about each node's sensor. Empty today: no error and no averaging lag is measured."""

    error: dict[str, float] = field(default_factory=dict)  # node -> relative error measured on our node
    averaging_lag_s: dict[str, float] = field(default_factory=dict)  # pod -> the sensor's measured averaging lag
    min_samples: int = MIN_SAMPLES
    settle_s: float = SETTLE_S
    reference: str = JETSON  # the clock offsets are measured against


def _s(t: datetime) -> float:
    return (t - EPOCH).total_seconds()


def _dt(s: float) -> datetime:
    return EPOCH + timedelta(seconds=s)


@dataclass
class _Sample:
    t: float  # on the node's clock
    watts: float | None
    event: JetsonPower | GpuPower

    def __lt__(self, other: "_Sample") -> bool:  # insort keeps arrival order among equal times
        return self.t < other.t


class JoulesPerAnswerProjection:
    def __init__(self, calibration: Calibration | None = None, keep: int = 200_000) -> None:
        self.calibration = calibration or Calibration()
        self.keep = keep  # power samples kept per node
        self._samples: dict[str, list[_Sample]] = defaultdict(list)
        self._requests: dict[str, dict[tuple, RequestWindow]] = defaultdict(dict)
        self._steps: dict[str, list[TrainStep]] = defaultdict(list)
        self._offsets: dict[str, ClockOffset] = {}
        self._values: dict[tuple, JoulesPerAnswer] = {}
        self.waiting: dict[str, int] = {}  # reason -> requests waiting for it, after the last ready()

    def subscribe(self, bus: EventBus) -> None:
        bus.subscribe("telemetry.", self.on_event, replay=True)

    # --- folding ---

    def on_event(self, event: Event) -> None:
        node = getattr(event, "node", None)
        if not isinstance(node, str) or not node:
            return
        if canonical(node) != node:
            node = canonical(node)
            event = event.model_copy(update={"node": node})
        reading = reading_of(event)
        if reading is not None:
            samples = self._samples[node]
            insort(samples, _Sample(_s(event.ts), reading.watts, event))
            if len(samples) > self.keep:
                del samples[: len(samples) - self.keep]
        elif isinstance(event, RequestWindow):
            self._requests[node][_key(event)] = event
        elif isinstance(event, TrainStep):
            self._steps[node].append(event)
        elif isinstance(event, ClockOffset):
            held = self._offsets.get(node)
            if held is None or event.ts >= held.ts:
                self._offsets[node] = event
        elif isinstance(event, JoulesPerAnswer):
            self._values[(node, event.request_id, event.started)] = event

    # --- clocks ---

    def _offset(self, clock: str) -> ClockOffset | None | bool:
        """The offset of `clock` against the reference: True for the reference itself, None when unmeasured."""
        if clock == self.calibration.reference:
            return True
        return self._offsets.get(clock)

    def _to_node(self, w: RequestWindow) -> tuple[float, float, float | None, float | None] | None:
        """The window on its node's clock, and the offset and error used; None while an offset is missing."""
        clock = canonical(w.clock or w.node)
        if clock == w.node:
            return _s(w.started), _s(w.ended), None, None
        a, b = self._offset(clock), self._offset(w.node)
        if a is None or b is None:
            return None
        # window clock -> reference -> node clock: subtract the window clock's offset, add the node's
        shift = (0.0 if a is True else -a.offset_s) + (0.0 if b is True else b.offset_s)
        error = (0.0 if a is True else a.error_s) + (0.0 if b is True else b.error_s)
        pod = b if b is not True else a
        return _s(w.started) + shift, _s(w.ended) + shift, pod.offset_s, error

    # --- computing ---

    def ready(self) -> list[JoulesPerAnswer]:
        """Values for the requests that can be reported now and are not in the log yet, oldest window first."""
        out: list[JoulesPerAnswer] = []
        waiting: dict[str, int] = defaultdict(int)
        for node, requests in self._requests.items():
            out += self._ready_on(node, requests, waiting)
        self.waiting = dict(waiting)
        return sorted(out, key=lambda v: v.ended)

    def _ready_on(self, node: str, requests: dict[tuple, RequestWindow], waiting: dict[str, int]) -> list[JoulesPerAnswer]:
        placed: list[tuple[float, float, tuple, RequestWindow, float | None, float | None]] = []
        for key, w in requests.items():
            on_node = self._to_node(w)
            if on_node is None:
                if key not in self._values:
                    waiting["no clock offset for the window's clock yet"] += 1
                continue
            placed.append((on_node[0], on_node[1], key, w, on_node[2], on_node[3]))
        if not any(p[2] not in self._values for p in placed):
            return []
        samples = [s for s in self._samples.get(node, []) if s.watts is not None]
        latest = samples[-1].t if samples else None
        out = []
        for group in _groups(placed):
            pending = [p for p in group if p[2] not in self._values]
            if not pending:
                continue
            end = max(p[1] for p in group)
            if latest is None or latest < end + self.calibration.settle_s:
                waiting["samples have not run past the window and its settle time"] += len(pending)
                continue
            out += self._report(node, group, samples, placed, waiting)
        return out

    def _report(self, node: str, group: list, samples: list[_Sample], placed: list,
                waiting: dict[str, int]) -> list[JoulesPerAnswer]:
        lo_t, hi_t = min(p[0] for p in group), max(p[1] for p in group)
        times = [s.t for s in samples]
        i, j = bisect_right(times, lo_t) - 1, bisect_left(times, hi_t)
        trace = [(s.t, s.watts) for s in samples[max(i, 0): j + 1]]
        windows = [Window(repr(p[2]), p[0], p[1], p[3].tokens_out, fixed=p[2] in self._values) for p in group]
        shares = attribute(trace, windows)
        out = []
        for p, win in zip(group, windows):
            if win.fixed:
                continue
            if shares[win.key] is None:
                waiting["no sample at or before the window start"] += 1
                continue
            others = [(q, v) for q, v in zip(group, windows) if q is not p and q[0] < p[1] and q[1] > p[0]]
            late = any(v.fixed for _, v in others)  # reported before this request was known
            value = self._value(node, p, shares[win.key], trace, samples, times, placed, len(others), late)
            self._values[p[2]] = value
            out.append(value)
        return out

    def _value(self, node: str, p, joules: float, trace, samples: list[_Sample], times: list[float], placed,
               overlapping: int, late: bool) -> JoulesPerAnswer:
        start, end, w = p[0], p[1], p[3]
        cal = self.calibration
        lo, hi = brackets(times, start, end)
        every = self._samples[node]  # every sample, a missing reading included, for the conditions
        span = every[bisect_left(every, _Sample(times[lo], None, None)): bisect_right(every, _Sample(times[hi], None, None))]
        first = samples[lo].event
        in_window = sum(1 for t in times[lo: hi + 1] if start <= t <= end)
        conditions = _conditions(first, samples[hi].event, span, self._steps.get(node, []), start, end)
        error = cal.error.get(node)
        reasons = []
        if in_window < cal.min_samples:
            reasons.append(f"fewer than {cal.min_samples} samples in the window")
        lag = cal.averaging_lag_s.get(node)
        if isinstance(first, GpuPower) and lag is not None and end - start < lag:
            reasons.append("window shorter than the sensor's averaging lag")
        if error is None:
            reasons.append("error not measured on this node")
        if len(conditions.power_modes) > 1:
            reasons.append("two power modes in the window")
        if conditions.throttled:
            reasons.append("throttled")
        if conditions.overcurrent:
            reasons.append("overcurrent event")
        if conditions.train_overlap:
            reasons.append("overlapped a training step")
        if late:
            reasons.append("overlaps an answer reported before this request was known")
        idle = self._idle(node, start, placed)
        back = start - _s(w.started)  # window clock = node clock - back
        sampler = first.sampler if isinstance(first, GpuPower) and first.sampler else first.source
        return JoulesPerAnswer(
            event_id=f"jpa:{node}:{w.request_id}:{w.started.isoformat()}", ts=w.ended, node=node,
            request_id=w.request_id, target=w.target, model=w.model, kind=w.kind, started=w.started, ended=w.ended,
            clock=canonical(w.clock or w.node), joules=joules, node_energy_j=integrate(trace, start, end),
            overlapping=overlapping, tokens_out=w.tokens_out,
            idle_w=idle.watts if idle else None, idle_ts=_dt(idle.t - back) if idle else None,
            power_field=POWER_FIELD[first.type], boundary=BOUNDARY_TEXT[first.type], accelerator=ACCELERATOR[first.type],
            sampler=sampler, period_ms=first.period_ms, samples=in_window, error=error,
            clock_offset_s=p[4], clock_error_s=p[5], low_confidence=bool(reasons), reasons=reasons,
            conditions=conditions)

    def _idle(self, node: str, before: float, placed) -> _Sample | None:
        """The node's latest reading before `before` taken while it ran no request and no training step and, on a
        pod, while the sampler did not call the card busy."""
        busy = [(p[0], p[1]) for p in placed]
        busy += [(_s(s.ts) - s.step_ms / 1000.0, _s(s.ts)) for s in self._steps.get(node, [])]
        samples = self._samples.get(node, [])
        for s in reversed(samples[: bisect_left(samples, _Sample(before, None, None))]):
            if s.watts is None or (isinstance(s.event, GpuPower) and s.event.under_load):
                continue
            if any(a <= s.t <= b for a, b in busy):
                continue
            return s
        return None

    # --- reads ---

    def values(self, node: str | None = None) -> list[JoulesPerAnswer]:
        out = [v for v in self._values.values() if node in (None, v.node)]
        return sorted(out, key=lambda v: v.ended)

    def routing_snapshot(self) -> dict[str, Any]:
        """What the energy routing policy reads (T5): the latest joules per answer per node and kind of work."""
        from contexts.telemetry.application.routing import routing_snapshot

        return routing_snapshot(self.values())

    def stats(self, recent: int = 20) -> dict[str, Any]:
        """For the stats page, per node: watts now, joules per answer (recent values and their mean), and per UTC day
        the node's energy, its answers and their joules. Times are each node's own clock for watts and energy."""
        nodes: dict[str, Any] = {}
        for node in sorted(set(self._samples) | {v.node for v in self._values.values()}):
            samples = [s for s in self._samples.get(node, []) if s.watts is not None]
            values = self.values(node)
            days: dict[str, dict[str, Any]] = defaultdict(lambda: {"node_j": None, "answers": 0, "answer_j": 0.0})
            trace = [(s.t, s.watts) for s in samples]
            if len(trace) > 1:  # the node's energy over the part of each UTC day its samples cover
                day = _dt(trace[0][0]).date()
                while (day_start := _s(datetime(day.year, day.month, day.day, tzinfo=UTC))) <= trace[-1][0]:
                    a, b = max(trace[0][0], day_start), min(trace[-1][0], day_start + 86400)
                    days[day.isoformat()]["node_j"] = integrate(trace, a, b)
                    day += timedelta(days=1)
            for v in values:
                d = days[v.ended.astimezone(UTC).date().isoformat()]
                d["answers"] += 1
                d["answer_j"] += v.joules
            first = samples[-1].event if samples else None
            nodes[node] = {
                "watts_now": samples[-1].watts if samples else None,
                "watts_ts": samples[-1].event.ts if samples else None,
                "power_field": POWER_FIELD[first.type] if first else None,
                "boundary": BOUNDARY_TEXT[first.type] if first else None,
                "answers": len(values),
                "joules_per_answer_mean": sum(v.joules for v in values) / len(values) if values else None,
                "low_confidence": any(v.low_confidence for v in values) if values else None,
                "recent": [{"request_id": v.request_id, "ended": v.ended, "joules": v.joules,
                            "low_confidence": v.low_confidence} for v in values[-recent:]],
                "per_day": {k: dict(days[k]) for k in sorted(days)},
                "clock_offset": (self._offsets[node].model_dump(include={"offset_s", "error_s", "ts"})
                                 if node in self._offsets else None),
            }
        return {"nodes": nodes, "waiting": dict(self.waiting)}


def _key(w: RequestWindow) -> tuple:
    return (canonical(w.node), w.request_id, w.started)


def _groups(placed: list) -> list[list]:
    """Requests that overlap, directly or through others, in one group each."""
    groups: list[list] = []
    end = None
    for p in sorted(placed, key=lambda p: (p[0], p[1])):
        if end is None or p[0] >= end:
            groups.append([p])
            end = p[1]
        else:
            groups[-1].append(p)
            end = max(end, p[1])
    return groups


def _conditions(first: Event, last: Event, span: list[_Sample], steps: list[TrainStep], start: float,
                end: float) -> WindowConditions:
    modes: list[str] = []
    throttles: list[bool] = []
    for s in span:
        if isinstance(s.event, JetsonPower) and s.event.nvpmodel and s.event.nvpmodel not in modes:
            modes.append(s.event.nvpmodel)
        if isinstance(s.event, GpuPower) and s.event.throttled is not None:
            throttles.append(s.event.throttled)
    overcurrent = None
    if isinstance(first, JetsonPower) and isinstance(last, JetsonPower) and first.oc_events and last.oc_events:
        overcurrent = any(last.oc_events.get(k, 0) > v for k, v in first.oc_events.items())
    train = any(_s(s.ts) - s.step_ms / 1000.0 < end and _s(s.ts) > start for s in steps)
    return WindowConditions(power_modes=modes, temps_c=dict(first.temps_c),
                            throttled=any(throttles) if throttles else None, overcurrent=overcurrent,
                            train_overlap=train)
