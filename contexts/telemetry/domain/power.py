"""Power readings in one unit: the single place where each source's sample becomes watts.

A Jetson sample is module input power (the VDD_IN rail, reported in mW); a pod sample is the
R9700's graphics package (socket) power in W. The two boundaries differ (the Jetson's carrier
board and the pod's host are outside them), so every reading names its boundary and any number
that compares nodes can say what it compares.
"""

from dataclasses import dataclass
from datetime import datetime

from contracts.events import Event
from contracts.telemetry import GpuPower, JetsonPower

MODULE_RAIL = "VDD_IN"  # Jetson Orin Nano: "Total Module Power" on the INA3221
BOUNDARY = {
    JetsonPower.TYPE: "module",  # Jetson module input, VDD_IN
    GpuPower.TYPE: "graphics package",  # R9700 socket power
}


@dataclass(frozen=True)
class PowerReading:
    node: str
    ts: datetime
    watts: float | None  # None when the source could not read power at this sample
    source: str
    boundary: str


def reading_of(event: Event) -> PowerReading | None:
    """The power reading in a power sample event, in watts; None for events that are not power samples."""
    if isinstance(event, JetsonPower):
        mw = event.rails.get(MODULE_RAIL)
        watts = None if mw is None else mw / 1000.0
    elif isinstance(event, GpuPower):
        watts = event.socket_w
    else:
        return None
    return PowerReading(node=event.node, ts=event.ts, watts=watts, source=event.source, boundary=BOUNDARY[event.type])
