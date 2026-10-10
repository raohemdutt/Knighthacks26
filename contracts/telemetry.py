"""Telemetry events: what Atlas costs to run, per node, one event type per source.

Each source publishes at the cadence it supports and carries its own fields. The only
fields every telemetry event has are `node` (`jetson`, `gpu-a`, `gpu-b`) and `ts`; there is
no shared metric list, and a new source is a new event type rather than a change here.

- `telemetry.jetson.power`  one sample per sampler period (tegrastats: 1000 ms by default)
- `telemetry.gpu.power`     one sample per poll of the pod's card (amd-smi, rocm-smi or sysfs)
- `telemetry.request`       one per answered request: its window on the answering node
- `telemetry.train.step`    one per training step on the trainer's card

`ts` is when the sample was taken or the step or request finished, on the clock of the node
that produced it. Units are in the field names, except the Jetson `rails`, which are mW as
the sensor reports them. A reading the source could not take is null, never a sentinel
value such as 0 or UINT32_MAX.
"""

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from contracts.events import TypedEvent, register


class _TelemetryEvent(TypedEvent):
    node: str = Field(min_length=1)  # "jetson", "gpu-a", "gpu-b": the node the numbers describe


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


@register
class GpuPower(_TelemetryEvent):
    """One sample of an R9700's power, clocks and throttle state, read on its pod (no root: reads only)."""

    type: Literal["telemetry.gpu.power"] = "telemetry.gpu.power"
    source: str  # the tool that read it: "amd-smi", "amdsmi" (library), "rocm-smi", "hwmon", "fake"
    period_ms: float  # the sampler's configured period
    socket_w: float | None  # graphics package power in W (whole watts on RDNA4); null when N/A
    gfx_mhz: float | None
    mem_mhz: float | None
    throttled: bool | None  # the firmware reports any throttling (power, thermal, current) at this sample
    power_field: str | None = None  # the field socket_w came from: "average_socket_power", "power1_average" ...
    temps_c: dict[str, float] = {}  # sensor -> degrees C, e.g. {"edge": 41.0, "hotspot": 47.0, "mem": 52.0}
    energy_counter_j: float | None = None  # the card's cumulative energy counter in J, when the driver exposes one
    energy_counter_ns: int | None = None  # the counter's own timestamp in ns, when reported with it
    received_ts: datetime | None = None  # when the Jetson's reader got the sample, on the Jetson's clock


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
