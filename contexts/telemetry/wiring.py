"""Building the telemetry context from one setting, for the composition root.

    telemetry = build_telemetry("fake", bus)        # what ATLAS_TELEMETRY_SOURCE=fake does
    task = asyncio.create_task(telemetry.run())     # once the event loop is up, after replay

The projection is subscribed to the bus here (replay=True); the sampler publishes when `run()` runs.

Sources, one or several joined by commas ("tegrastats,pods" on the Jetson runs both samplers into one projection
and one joules-per-answer computation; "fake" runs alone):
- "fake" replays recorded samples (fixtures/telemetry) as live ones, at the recorded pace and on a
  loop, for laptops and demos without hardware. It refuses to run on a Jetson, where its samples
  would land in the box's own event log as if they had been measured.
- "tegrastats" samples the Jetson's power with tegrastats (application/jetson_sampler.py) and runs
  only on a Jetson. On the box it runs inside the gateway, so the atlas-gateway systemd unit starts
  it at boot and again after every restart; a tegrastats that dies is started again by the runner.
  Settings, all optional:
    ATLAS_TELEMETRY_JETSON_INTERVAL_MS  sampling period in ms (default 350, for the 1 percent CPU
                                        budget; never shorter than the INA3221 update interval)
  jetson/atlas-gateway.service sets ATLAS_TELEMETRY_SOURCE=tegrastats and the 350 ms period, so the
  box samples from boot (telemetry owner ruling 8). The unit's settings win over .env; change them on
  a box with `systemctl edit atlas-gateway` (e.g. "tegrastats,pods", or an empty source to turn it off).
    ATLAS_TEGRASTATS                    the tegrastats command (default "tegrastats")
    ATLAS_NVPMODEL                      the nvpmodel command (default "/usr/sbin/nvpmodel")
- "pods" reads the pod power samplers (lab/pod_power.py) through the hub proxy and publishes their
  telemetry.gpu.power samples for pod-a and pod-b, each stamped with the time it arrived. It runs off the pods (the
  Jetson or a laptop) and reads the accounts from the environment: AUP_HUB_USER_<X> / AUP_HUB_TOKEN_<X> for each
  pod's account (either pod is enough), with ATLAS_POD_ACCOUNTS (default pod-a=A,pod-b=B; the SJC pods are
  pod-a=C,pod-b=D), AUP_HUB_URL_<X> or LAB_HUB_URL (the hub settings lab/hub.py reads; default the DUB hub),
  LAB_HUB_USER_AGENT (default lab/hub.py's browser-like agent), ATLAS_POD_POWER_PORT (default 8790) and
  ATLAS_POD_POWER_POLL_S (default 2) to override. A hub host other than dub.aupcloud.io must also be admitted by
  the Jetson firewall (the hub group of jetson/egress.py), or the reader's polls never leave the box. Samples
  already in the log are not published again after a restart. Every poll's answer also carries the pod's
  receive and send times, so the reader measures the pod's clock against its own and publishes it
  (telemetry.clock.offset) on the first poll and about once a minute after, the best of its last ten exchanges.
  The old node names in ATLAS_POD_ACCOUNTS (gpu-a, gpu-b) still read as pod-a and pod-b.

ATLAS_TELEMETRY_IDLE_W (pod-a=10.27,pod-b=10.27,jetson=15.76) sets the calibrated idle per node over
config/telemetry_idle.yaml, the figure stored beside each joules-per-answer value (application/idle.py).

Every source also gets joules per answer (application/joules.py): the service publishes each answered request's
telemetry.joules_per_answer once its window has settled, and `service.stats()` gives the stats page watts now,
joules per answer and energy per day per node. Nothing is measured about the sensors yet (Calibration), so every
value is low confidence.
"""

import os
import shlex
from collections.abc import Callable, Mapping
from pathlib import Path

import httpx

from bus import EventBus
from contexts.telemetry.application.idle import idle_figures
from contexts.telemetry.application.jetson_sampler import DEFAULT_INTERVAL_MS, JetsonPowerSampler
from contexts.telemetry.application.joules import Calibration, JoulesPerAnswerProjection
from contexts.telemetry.application.projection import TelemetryProjection
from contexts.telemetry.application.sampling import SamplerRunner
from contexts.telemetry.application.service import TelemetryService
from contexts.telemetry.domain.nodes import JETSON, KNOWN_NODES
from contexts.telemetry.fakes.sampler import FakeSampler
from contexts.telemetry.infrastructure import host, pods, recorded
from contexts.telemetry.infrastructure.jetson_sysfs import JetsonSysfs
from contexts.telemetry.infrastructure.jetson_tools import NVPMODEL, TEGRASTATS, NvpmodelQuery, TegrastatsProcess

SOURCES = ("fake", "tegrastats", "pods")


def build_telemetry(source: str, bus: EventBus, *, fixture: str | Path | None = None, speed: float = 1.0,
                    loop: bool = True, on_jetson: Callable[[], bool] | None = None,
                    env: Mapping[str, str] | None = None,
                    transport: httpx.AsyncBaseTransport | None = None,
                    calibration: Calibration | None = None) -> TelemetryService:
    """The telemetry service for `source`: one source, or several joined by commas ("tegrastats,pods" on the
    Jetson, both samplers into one projection). "fake" runs alone."""
    names = [n.strip() for n in source.split(",") if n.strip()]
    for name in names:
        if name not in SOURCES:
            raise ValueError(f"unknown telemetry source {name!r}; known sources: {', '.join(SOURCES)}")
    if not names:
        raise ValueError(f"no telemetry source named; known sources: {', '.join(SOURCES)}")
    if len(set(names)) != len(names):
        raise ValueError(f"telemetry source {source!r} names a source twice")
    if "fake" in names and len(names) > 1:
        raise ValueError("the fake telemetry source runs alone: its replayed samples would mix with measured ones")
    env = os.environ if env is None else env
    projection = TelemetryProjection()
    runners = [_runner(name, bus, projection, fixture=fixture, speed=speed, loop=loop, on_jetson=on_jetson, env=env,
                       transport=transport) for name in names]
    calibration = calibration or Calibration(idle=idle_figures(env))  # the nodes' calibrated idle, stored beside values
    service = TelemetryService(projection, runners, source=",".join(names),
                               joules=JoulesPerAnswerProjection(calibration))
    service.subscribe(bus)
    return service


def _runner(source: str, bus: EventBus, projection: TelemetryProjection, *, fixture: str | Path | None, speed: float,
            loop: bool, on_jetson: Callable[[], bool] | None, env: Mapping[str, str],
            transport: httpx.AsyncBaseTransport | None) -> SamplerRunner:
    if source == "pods":  # read over the hub proxy, so it runs on any machine that holds the account keys
        endpoints = pods.pod_endpoints(env)
        reader = pods.PodReader(endpoints, poll_s=float(env.get("ATLAS_POD_POWER_POLL_S") or pods.POLL_S),
                                transport=transport, skip_until=projection.latest_power_ts,
                                user_agent=pods.hub_user_agent(env))
        return SamplerRunner(reader, bus.publish, nodes=[p.node for p in endpoints])
    is_jetson = (on_jetson or host.on_jetson)()
    if source == "tegrastats":
        if not is_jetson:
            raise RuntimeError("the tegrastats telemetry source runs only on a Jetson: anywhere else its samples "
                               "would speak for a box they were not taken on")
        return SamplerRunner(jetson_power_sampler(env), bus.publish, nodes=[JETSON])
    if is_jetson:
        raise RuntimeError("the fake telemetry source does not run on a Jetson: its samples would go into the "
                           "box's event log as if they had been measured")
    sampler = FakeSampler(recorded.load(fixture or recorded.DEFAULT), speed=speed, loop=loop)
    return SamplerRunner(sampler, bus.publish, nodes=KNOWN_NODES)


def jetson_power_sampler(env: Mapping[str, str], interval_ms: int | None = None) -> JetsonPowerSampler:
    """The Jetson power sampler as the settings above describe it; `interval_ms` overrides the setting."""
    if interval_ms is None:
        raw = env.get("ATLAS_TELEMETRY_JETSON_INTERVAL_MS", "").strip() or str(DEFAULT_INTERVAL_MS)
        try:
            interval_ms = int(raw)
        except ValueError:
            raise ValueError(f"ATLAS_TELEMETRY_JETSON_INTERVAL_MS must be a whole number of ms, not {raw!r}") from None
    return JetsonPowerSampler(TegrastatsProcess(_command(env, "ATLAS_TEGRASTATS", TEGRASTATS)), JetsonSysfs(),
                              NvpmodelQuery(_command(env, "ATLAS_NVPMODEL", NVPMODEL)), interval_ms=interval_ms)


def _command(env: Mapping[str, str], name: str, default: tuple[str, ...]) -> list[str]:
    raw = env.get(name, "").strip()
    return shlex.split(raw) if raw else list(default)
