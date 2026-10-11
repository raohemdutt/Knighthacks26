"""Power readings in one unit: the single place where each source's sample becomes watts.

A Jetson sample is module input power (the VDD_IN rail, reported in mW). A pod sample is socket_w, the firmware's
socket power in W: on the Strix Halo pods the lab runs (Radeon 8060S, gfx1151, an APU) the socket is the whole APU
package, CPU cores and graphics together, not the graphics alone as it was on the R9700. The boundaries differ (the
Jetson's carrier board and the pod's memory, fans and supply are outside them), so every reading names its boundary
and accelerator, and any number that compares nodes can say what it compares.
"""

from dataclasses import dataclass
from datetime import datetime

from contracts.events import Event
from contracts.telemetry import GpuPower, JetsonPower

MODULE_RAIL = "VDD_IN"  # Jetson Orin Nano: "Total Module Power" on the INA3221
BOUNDARY = {
    JetsonPower.TYPE: "module",  # Jetson module input, VDD_IN
    GpuPower.TYPE: "APU socket",  # Strix Halo socket power: CPU cores and graphics together
}
POWER_FIELD = {JetsonPower.TYPE: MODULE_RAIL, GpuPower.TYPE: "socket_w"}  # the reading integrated for energy
BOUNDARY_TEXT = {  # what each boundary covers, as a stored value names it
    JetsonPower.TYPE: "module input (VDD_IN: CPU, GPU, SoC and memory on the module; carrier board outside)",
    GpuPower.TYPE: "APU socket (CPU cores and graphics together; memory, fans and supply outside)",
}
ACCELERATOR = {
    JetsonPower.TYPE: "Jetson Orin Nano (Ampere GPU)",
    GpuPower.TYPE: "Radeon 8060S (gfx1151, Strix Halo)",
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
