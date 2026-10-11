"""Telemetry events: what Atlas costs to run, per node, one event type per source.

Each source publishes at the cadence it supports and carries its own fields. The only
fields every telemetry event has are `node` (`jetson`, `pod-a`, `pod-b`) and `ts`; there is
no shared metric list, and a new source is a new event type rather than a change here.

- `telemetry.jetson.power`  one sample per sampler period (tegrastats: 1000 ms by default)
- `telemetry.gpu.power`     one sample per poll of the pod's card (amd-smi, rocm-smi or sysfs)
- `telemetry.request`       one per answered request: its window on the answering node
- `telemetry.train.step`    one per training step on the trainer's card
- `telemetry.clock.offset`  one per measurement of a pod's clock against the Jetson's, through the hub proxy
- `telemetry.joules_per_answer`  one per answered request: the energy of the answer (J, gross), how it was
                            attributed, and what the window ran under

`ts` is when the sample was taken or the step or request finished, on the clock of the node
that produced it. Units are in the field names, except the Jetson `rails`, which are mW as
the sensor reports them. A reading the source could not take is null, never a sentinel
value such as 0 or UINT32_MAX.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.events import TypedEvent, register


class _TelemetryEvent(TypedEvent):
    node: str = Field(min_length=1)  # "jetson", "pod-a", "pod-b": the node the numbers describe


@register
class JetsonPower(_TelemetryEvent):
    """One Jetson power sample: every INA3221 rail at one instant, and the power mode it was taken in."""

    type: Literal["telemetry.jetson.power"] = "telemetry.jetson.power"
    source: str  # the tool that read it: "tegrastats", "ina3221-sysfs", "fake"
    period_ms: float  # the sampler's configured period
    rails: dict[str, float]  # rail -> instantaneous mW, e.g. VDD_IN (module input), VDD_CPU_GPU_CV, VDD_SOC
    nvpmodel: str | None  # power mode name as `nvpmodel -q` prints it, e.g. "MAXN_SUPER"; null if unread
    temps_c: dict[str, float] = {}  # sensor -> degrees C, e.g. {"gpu": 47.5, "cpu": 48.0}
    oc_events: dict[str, int] = {}  # overcurrent counter -> cumulative count since boot (oc1_event_cnt ...)
    sensor_avg_samples: int | None = None  # INA3221 conversions averaged into each reading
    sensor_update_ms: float | None = None  # INA3221 update interval; faster polling re-reads the same value


class EnergyCheck(BaseModel):
    """A pod sampler's check of the card's energy counter against power integrated over the same samples.

    It runs at start-up and once under load; the sample that closes the window carries it, and `ts` of that
    sample is the window's end."""

    model_config = ConfigDict(extra="forbid")

    phase: Literal["start-up", "load"]
    window_s: float  # seconds between the window's first and last sample
    samples: int
    advanced: bool | None  # the counter's delta was positive and it never went backwards; null if unreadable
    counter_delta_j: float | None
    integrated_j: float | None  # socket_w integrated over the same samples (trapezoid rule)
    ratio: float | None  # counter_delta_j / integrated_j
    tolerance: float  # 0.10: agrees when the ratio is within this of 1
    agrees: bool


@register
class GpuPower(_TelemetryEvent):
    """One sample of a pod's power, clocks and throttle state, read on its pod (no root: reads only)."""

    type: Literal["telemetry.gpu.power"] = "telemetry.gpu.power"
    source: str  # the tool that read it: "amd-smi", "amdsmi" (library), "rocm-smi", "hwmon", "fake"
    period_ms: float  # the sampler's configured period
    socket_w: float | None  # socket power in W, whole watts; the whole APU (CPU and graphics) on the Strix Halo; null when N/A
    gfx_mhz: float | None
    mem_mhz: float | None
    throttled: bool | None  # the firmware reports any throttling (power, thermal, current) at this sample
    power_field: str | None = None  # the field socket_w came from: "average_socket_power", "power1_average" ...
    temps_c: dict[str, float] = {}  # sensor -> degrees C, e.g. {"edge": 41.0, "hotspot": 47.0, "mem": 52.0}
    energy_counter_j: float | None = None  # the card's cumulative energy counter in J, when the driver exposes one
    energy_counter_ns: int | None = None  # the counter's own timestamp in ns, when reported with it
    received_ts: datetime | None = None  # when the Jetson's reader got the sample, on the Jetson's clock
    sampler: str | None = None  # the program that took the sample and its version, e.g. "lab/pod_power.py 1"
    sampler_started: datetime | None = None  # when that sampler started; a new value means it restarted
    polls: int | None = None  # reads the sampler has made since it started, this one included
    updates: int | None = None  # of those, reads that differed from the one before: distinct updates, not polls
    gfx_activity_pct: float | None = None  # GFX engine busy percent, when the tool reads it
    under_load: bool | None = None  # the card was busy at this sample (GFX activity, else power, past a threshold)
    stale: bool | None = None  # under load, yet no reading has changed for the sampler's stale window
    counter_ok: bool | None = None  # every energy-counter check so far agreed; null until the first one ends
    energy_check: EnergyCheck | None = None  # set on the sample that closed a counter check


@register
class RequestWindow(_TelemetryEvent):
    """One answered request: when it ran on the answering node, for energy per answer.

    `node` is the node that did the work (whose power is integrated over the window), `target` is
    what the router picked. `ts` is when the window closed."""

    type: Literal["telemetry.request"] = "telemetry.request"
    request_id: str
    started: datetime
    ended: datetime
    tokens_in: int | None
    tokens_out: int | None
    target: str  # the routing choice, e.g. "student" or "teacher"
    first_token: datetime | None = None  # end of prefill, when the engine reports it
    model: str | None = None  # the model or version that answered, e.g. "atlas-base"
    engine: str | None = None  # engine and build, e.g. "llama.cpp b6123", "vllm b1388b1"
    since_load: int | None = None  # 1 for the first request after the model loaded, 2 for the next ...
    clock: str | None = None  # node whose clock stamped started/first_token/ended when it is not `node`
    kind: str | None = None  # the kind of work, as the routing hook names it: "label", "caption", "chat", "eval", ...
    policy: str | None = None  # the routing policy that chose `target` (runtime/routing.py), e.g. "capability"

    @model_validator(mode="after")
    def _window_in_order(self) -> "RequestWindow":
        if self.ended < self.started:
            raise ValueError("ended is before started")
        if self.first_token is not None and not self.started <= self.first_token <= self.ended:
            raise ValueError("first_token is outside the request window")
        return self


@register
class TrainStep(_TelemetryEvent):
    """One optimizer step of a training run, with the card's power over it."""

    type: Literal["telemetry.train.step"] = "telemetry.train.step"
    run_id: str  # the run in learning.run.progress
    step: int
    loss: float | None  # null when the step produced no finite loss
    step_ms: float
    socket_w: float | None  # mean graphics package power over the step in W; null if not read
    energy_j: float | None = None  # energy the card used during this step, from its energy counter
    tokens: int | None = None  # tokens the step trained on
    received_ts: datetime | None = None  # when the Jetson's reader got it, on the Jetson's clock


@register
class ClockOffset(_TelemetryEvent):
    """A pod's clock against the reference clock (the Jetson's), from four timestamps of one exchange through the hub
    proxy, as NTP computes it (RFC 5905, section 8).

    t1 and t4 are on the reference clock (request sent, reply received), t2 and t3 on the pod's (request received,
    reply sent). offset_s = ((t2 - t1) + (t3 - t4)) / 2 is the pod's clock minus the reference clock, so a pod time
    moves onto the reference clock as pod_ts - offset_s; round_trip_s = (t4 - t1) - (t3 - t2), and the offset is
    within error_s = round_trip_s / 2. `node` is the pod; `ts` is t4."""

    type: Literal["telemetry.clock.offset"] = "telemetry.clock.offset"
    reference: str  # the clock the offset is measured against: "jetson"
    via: str  # the path the exchange took, e.g. "hub proxy /samples"
    t1: datetime
    t2: datetime
    t3: datetime
    t4: datetime
    offset_s: float
    round_trip_s: float
    error_s: float
    exchanges: int = 1  # exchanges this one was chosen from, by the smallest round trip


class WindowConditions(BaseModel):
    """What a request window ran under, read from the power samples that bracket and fall in it."""

    model_config = ConfigDict(extra="forbid")

    power_modes: list[str] = []  # Jetson nvpmodel names seen in the window, in order; two means it changed
    temps_c: dict[str, float] = {}  # the sample at or before the window start
    throttled: bool | None = None  # any pod sample in the window says throttled; null when none could say
    overcurrent: bool | None = None  # a Jetson overcurrent counter rose across the window; null without counters
    train_overlap: bool = False  # a training step (telemetry.train.step) on the same node overlapped the window


@register
class JoulesPerAnswer(_TelemetryEvent):
    """The energy of one answer: the answering node's power integrated over the request window (J), gross.

    Gross means nothing is subtracted (telemetry ruling 1); the node's idle watts are stored beside the value so a
    marginal figure can be derived without measuring again. The idle is the node's calibrated figure (the deep-idle
    reading taken by the sampler alone, named by `idle_metric` and `idle_measured`) and `idle_source` says
    "calibrated"; where no figure is calibrated it is the latest quiet reading before the window, "latest_quiet". The integral takes the last
    sample at or before the window start and the first at or after its end, clipped to the window by linear
    interpolation; repeated identical readings are all kept. Where requests overlap on the node, each stretch
    between samples and window edges is split among the requests running in it in proportion to their output
    tokens (`attribution`), so the answers of a node never sum past the node's energy. `node_energy_j` is the
    node's whole energy over the same window. `started` and `ended` are on the clock named by `clock`; pod
    samples were moved onto it by `clock_offset_s` (pod minus that clock), known within `clock_error_s`.
    `ts` is the window's end on that clock."""

    type: Literal["telemetry.joules_per_answer"] = "telemetry.joules_per_answer"
    request_id: str
    target: str | None = None
    model: str | None = None
    kind: str | None = None  # the request's kind of work, copied from telemetry.request (the energy policy reads it)
    started: datetime
    ended: datetime
    clock: str  # the clock started, ended and idle_ts are on: "jetson", or the node's own
    joules: float  # J, gross, this answer's share of the node's energy over its window
    node_energy_j: float  # J, the node's whole energy over the window, every request's share included
    basis: Literal["gross"] = "gross"
    attribution: Literal["token_proportional"] = "token_proportional"
    overlapping: int = 0  # other requests that ran on the node during part of this window
    tokens_out: int | None = None
    idle_w: float | None  # the node's idle, W: its calibrated figure, else the latest quiet reading before the window
    idle_ts: datetime | None  # when the latest quiet reading was taken (on `clock`); null for a calibrated figure
    idle_source: Literal["calibrated", "latest_quiet"] | None = None  # which idle_w is; null in rows from before it
    idle_metric: str | None = None  # a calibrated figure: the ledger metric it was measured as, e.g. pod.idle_watts
    idle_measured: datetime | None = None  # a calibrated figure: the timestamp of that ledger row (UTC)
    power_field: str  # the reading integrated: "VDD_IN" on the Jetson, "socket_w" on the pods
    boundary: str  # what that reading covers, e.g. "module input" or "APU socket (CPU cores and graphics)"
    accelerator: str  # the chip that answered, e.g. "Radeon 8060S (gfx1151, Strix Halo)"
    sampler: str  # the program or tool that took the samples
    period_ms: float | None  # the sampler's configured period
    samples: int  # samples with started <= ts <= ended
    error: float | None  # relative error of this node's energy, measured on our node; null while unmeasured
    clock_offset_s: float | None = None  # null when the samples and the window share a clock
    clock_error_s: float | None = None
    low_confidence: bool
    reasons: list[str] = []  # why low_confidence is set, one short phrase each
    conditions: WindowConditions = WindowConditions()
