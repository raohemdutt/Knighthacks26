"""Building the telemetry context from one setting, for the composition root.

    telemetry = build_telemetry("fake", bus)        # what ATLAS_TELEMETRY_SOURCE=fake does
    task = asyncio.create_task(telemetry.run())     # once the event loop is up, after replay

The projection is subscribed to the bus here (replay=True); the sampler publishes when `run()` runs.

Sources:
- "fake" replays recorded samples (fixtures/telemetry) as live ones, at the recorded pace and on a
  loop, for laptops and demos without hardware. It refuses to run on a Jetson, where its samples
  would land in the box's own event log as if they had been measured.
- "tegrastats" samples the Jetson's power with tegrastats (application/jetson_sampler.py) and runs
  only on a Jetson. On the box it runs inside the gateway, so the atlas-gateway systemd unit starts
  it at boot and again after every restart; a tegrastats that dies is started again by the runner.
  Settings, all optional:
    ATLAS_TELEMETRY_JETSON_INTERVAL_MS  sampling period in ms (default 100; never shorter than the
                                        INA3221 update interval)
    ATLAS_TEGRASTATS                    the tegrastats command (default "tegrastats")
    ATLAS_NVPMODEL                      the nvpmodel command (default "/usr/sbin/nvpmodel")
- "pods" reads the pod power samplers (lab/pod_power.py) through the hub proxy and publishes their
  telemetry.gpu.power samples for gpu-a and gpu-b, each stamped with the time it arrived. It runs off the pods (the
  Jetson or a laptop) and reads the accounts from the environment: AUP_HUB_USER_<X> / AUP_HUB_TOKEN_<X> for each
  pod's account (either pod is enough), with ATLAS_POD_ACCOUNTS (default gpu-a=A,gpu-b=B; the SJC pods are
  gpu-a=C,gpu-b=D), AUP_HUB_URL_<X> or LAB_HUB_URL (the hub settings lab/hub.py reads; default the DUB hub),
  LAB_HUB_USER_AGENT (default lab/hub.py's browser-like agent), ATLAS_POD_POWER_PORT (default 8790) and
  ATLAS_POD_POWER_POLL_S (default 2) to override. A hub host other than dub.aupcloud.io must also be admitted by
  the Jetson firewall (the hub group of jetson/egress.py), or the reader's polls never leave the box. Samples
  already in the log are not published again after a restart.
"""

import os
import shlex
from collections.abc import Callable, Mapping
from pathlib import Path

import httpx

from bus import EventBus
from contexts.telemetry.application.jetson_sampler import DEFAULT_INTERVAL_MS, JetsonPowerSampler
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
                    transport: httpx.AsyncBaseTransport | None = None) -> TelemetryService:
    if source not in SOURCES:
        raise ValueError(f"unknown telemetry source {source!r}; known sources: {', '.join(SOURCES)}")
    env = os.environ if env is None else env
    projection = TelemetryProjection()
    if source == "pods":  # read over the hub proxy, so it runs on any machine that holds the account keys
        endpoints = pods.pod_endpoints(env)
        reader = pods.PodReader(endpoints, poll_s=float(env.get("ATLAS_POD_POWER_POLL_S") or pods.POLL_S),
                                transport=transport, skip_until=projection.latest_power_ts,
                                user_agent=pods.hub_user_agent(env))
        service = TelemetryService(projection, SamplerRunner(reader, bus.publish, nodes=[p.node for p in endpoints]),
                                   source=source)
        service.subscribe(bus)
        return service
    is_jetson = (on_jetson or host.on_jetson)()
    if source == "tegrastats":
        if not is_jetson:
            raise RuntimeError("the tegrastats telemetry source runs only on a Jetson: anywhere else its samples "
                               "would speak for a box they were not taken on")
        runner = SamplerRunner(jetson_power_sampler(env), bus.publish, nodes=[JETSON])
    else:
        if is_jetson:
            raise RuntimeError("the fake telemetry source does not run on a Jetson: its samples would go into the "
                               "box's event log as if they had been measured")
        sampler = FakeSampler(recorded.load(fixture or recorded.DEFAULT), speed=speed, loop=loop)
        runner = SamplerRunner(sampler, bus.publish, nodes=KNOWN_NODES)
    service = TelemetryService(projection, runner, source=source)
    service.subscribe(bus)
    return service


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
